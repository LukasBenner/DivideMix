from __future__ import print_function

import sys
import os
import json
import logging
import argparse
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from sklearn.mixture import GaussianMixture

import optuna

from MobileNetSmall import mobilenet_small
import dataloader_imagefolder as dataloader

# --- TorchMetrics (mirrors metrics used in BaseRobustModule / PyTorch Lightning) ---
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


# -------------------------
# CLI / configuration
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DivideMix ImageFolder Training")

    # core training hparams
    parser.add_argument("--batch_size", default=32, type=int, help="train batchsize")
    parser.add_argument("--lr", "--learning_rate", dest="lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="SGD weight decay")

    # DivideMix knobs
    parser.add_argument("--noise_mode", default="sym", type=str, help="sym or asym (for warmup penalty)")
    parser.add_argument("--alpha", default=0.5, type=float, help="parameter for Beta")
    parser.add_argument("--lambda_u", default=0.1, type=float, help="weight for unsupervised loss")
    parser.add_argument("--p_threshold", default=0.5, type=float, help="clean probability threshold")
    parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--warm_up", default=10, type=int, help="warmup epochs")
    parser.add_argument("--early_stop_patience", default=20, type=int, help="early stopping patience on primary metric")

    # data / runtime
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--num_class", required=True, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--val_dir", default="", type=str)
    parser.add_argument("--test_dir", default="", type=str)
    parser.add_argument("--id", default="imagefolder", type=str)

    # metric selection
    parser.add_argument(
        "--primary_metric",
        default="f1_macro",
        choices=["f1_macro", "acc", "precision_macro", "recall_macro", "f1_weighted"],
        help="metric used for early-stopping / checkpoint selection / optuna objective",
    )
    parser.add_argument(
        "--log_per_class",
        action="store_true",
        help="log per-class precision/recall/f1 for val/test (can be verbose)",
    )

    # search mode
    parser.add_argument(
        "--search",
        default="none",
        choices=["none", "grid", "optuna"],
        help="hyperparameter search mode",
    )

    # --- Grid search (simple fallback) ---
    parser.add_argument("--grid_lr", default="0.01,0.005", type=str, help="comma list, e.g. 0.01,0.005")
    parser.add_argument("--grid_batch_size", default="16,32", type=str, help="comma list, e.g. 16,32")
    parser.add_argument("--grid_alpha", default="0.5,1.0", type=str, help="comma list")
    parser.add_argument("--grid_lambda_u", default="0.1,1.0", type=str, help="comma list")
    parser.add_argument("--grid_p_threshold", default="0.8", type=str, help="comma list")
    parser.add_argument("--grid_T", default="0.5,0.8", type=str, help="comma list")

    # --- Optuna ---
    parser.add_argument("--optuna_trials", default=25, type=int, help="number of optuna trials")
    parser.add_argument("--optuna_timeout", default=0, type=int, help="timeout seconds (0 = no timeout)")
    parser.add_argument("--optuna_study_name", default="", type=str, help="study name (default derived from --id)")
    parser.add_argument("--optuna_storage", default="", type=str, help="storage URL (e.g. sqlite:///checkpoint/study.db)")
    parser.add_argument(
        "--optuna_sampler",
        default="tpe",
        choices=["tpe", "random"],
        type=str,
        help="optuna sampler",
    )
    parser.add_argument(
        "--optuna_pruner",
        default="median",
        choices=["median", "hyperband", "sha", "nop"],
        type=str,
        help="optuna pruner",
    )
    parser.add_argument(
        "--optuna_prune_warmup_epochs",
        default=-1,
        type=int,
        help="do not prune before this epoch (default: warm_up). Use 0 to allow immediate pruning.",
    )
    parser.add_argument(
        "--optuna_n_jobs",
        default=1,
        type=int,
        help="parallel jobs (GPU training is typically 1).",
    )

    return parser


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def setup_device_and_seeds(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def make_base_logger(run_dir: str, logger_name: str = "run") -> logging.Logger:
    """Create a logger that writes both to stdout and to <run_dir>/run.log.

    Note: logger_name must be unique per run to avoid handler reuse across Optuna trials.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "run.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # clear handlers (important for repeated runs / optuna trials)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger



# -------------------------
# Transforms
# -------------------------
gpu_train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# -------------------------
# DivideMix core
# -------------------------
def train(
    epoch: int,
    net: torch.nn.Module,
    net2: torch.nn.Module,
    optimizer: optim.Optimizer,
    labeled_trainloader,
    unlabeled_trainloader,
    criterion,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    net.train()
    net2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    running_loss = 0.0
    running_loss_x = 0.0
    running_loss_u = 0.0

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        inputs_x, inputs_x2 = gpu_train_transforms(inputs_x), gpu_train_transforms(inputs_x2)
        inputs_u, inputs_u2 = gpu_train_transforms(inputs_u), gpu_train_transforms(inputs_u2)

        with torch.no_grad():
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            args.warm_up,
        )

        prior = torch.ones(args.num_class, device=logits.device) / args.num_class
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu # + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        running_loss_x += float(Lx.item())
        running_loss_u += float(Lu.item())

    logger.info(
        f"Epoch {epoch} - Loss: {running_loss / num_iter:.4f} - "
        f"Lx: {running_loss_x / num_iter:.4f} - Lu: {running_loss_u / num_iter:.4f}"
    )


def warmup(
    epoch: int,
    net: torch.nn.Module,
    optimizer: optim.Optimizer,
    loader,
    ce_loss: nn.Module,
    conf_penalty,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    net.train()
    num_iter = (len(loader.dataset) // loader.batch_size) + 1
    running_loss = 0.0

    for batch_idx, (inputs, labels, index) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = gpu_train_transforms(inputs)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = ce_loss(outputs, labels)

        if args.noise_mode == "asym" and conf_penalty is not None:
            penalty = conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss

        L.backward()
        optimizer.step()
        running_loss += float(loss.item())

    logger.info(f"Epoch {epoch} - Warmup Loss: {running_loss / num_iter:.4f}")


def eval_train(model: torch.nn.Module, eval_loader, ce_per_sample: nn.Module, all_loss: List[torch.Tensor]) -> Tuple[np.ndarray, List[torch.Tensor]]:
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset), device="cpu")

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = ce_per_sample(outputs, targets)  # shape [B]
            loss_cpu = loss.detach().cpu()
            for b in range(inputs.size(0)):
                losses[index[b]] = loss_cpu[b]

    # normalize losses to [0, 1]
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    all_loss.append(losses)

    input_loss = losses.numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


# -------------------------
# Loss components
# -------------------------
class SemiLoss:
    def __init__(self, lambda_u: float):
        self.lambda_u = float(lambda_u)

    @staticmethod
    def _linear_rampup(current: float, warm_up: int, rampup_length: int = 16) -> float:
        current = float(np.clip((current - warm_up) / rampup_length, 0.0, 1.0))
        return current

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        w = self.lambda_u * self._linear_rampup(epoch, warm_up)
        return Lx, Lu, w


class NegEntropy:
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model(args: argparse.Namespace) -> torch.nn.Module:
    model = mobilenet_small(num_classes=args.num_class)
    return model.cuda()


# -------------------------
# Metrics (BaseRobustModule-style)
# -------------------------
def _sanitize_class_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _get_class_names_from_loader(loader, num_classes: int) -> Optional[List[str]]:
    # ImageFolder usually provides dataset.classes
    ds = getattr(loader, "dataset", None)
    classes = getattr(ds, "classes", None)
    if classes is None:
        return None
    classes = list(classes)
    if len(classes) < num_classes:
        return None
    return classes


def _build_metrics(num_classes: int, device: torch.device, prefix: str) -> MetricCollection:
    macro_metrics = {
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
    }
    context_metrics = {
        "acc": MulticlassAccuracy(num_classes=num_classes),
        "precision_weighted": MulticlassPrecision(num_classes=num_classes, average="weighted"),
        "recall_weighted": MulticlassRecall(num_classes=num_classes, average="weighted"),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
    }
    combined = MetricCollection({**macro_metrics, **context_metrics}).clone(prefix=prefix)
    return combined.to(device)


def _build_per_class(num_classes: int, device: torch.device, prefix: str) -> MetricCollection:
    per_class = MetricCollection(
        {
            "precision": MulticlassPrecision(num_classes=num_classes, average=None),
            "recall": MulticlassRecall(num_classes=num_classes, average=None),
            "f1": MulticlassF1Score(num_classes=num_classes, average=None),
        }
    ).clone(prefix=prefix)
    return per_class.to(device)


@torch.no_grad()
def evaluate(
    net1: torch.nn.Module,
    net2: torch.nn.Module,
    loader,
    split_prefix: str,  # "val/" or "test/"
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    net1.eval()
    net2.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _build_metrics(args.num_class, device=device, prefix=split_prefix)

    per_class = None
    class_names = None
    if args.log_per_class:
        per_class = _build_per_class(args.num_class, device=device, prefix=f"{split_prefix}per_class/")
        class_names = _get_class_names_from_loader(loader, args.num_class)

    for batch_idx, batch in enumerate(loader):
        # loader is expected to yield (inputs, targets)
        inputs, targets = batch
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net1(inputs) + net2(inputs)
        metrics.update(outputs, targets)
        if per_class is not None:
            per_class.update(outputs, targets)

    out: Dict[str, float] = {}
    computed = metrics.compute()
    for k, v in computed.items():
        out[k] = float(v.detach().cpu().item())

    # per-class logging (optional)
    if per_class is not None:
        pc = per_class.compute()
        # pc keys like "val/per_class/precision" with tensor shape [C]
        for base_key, tensor in pc.items():
            # base_key includes prefix already
            values = tensor.detach().cpu().tolist()
            for i, val in enumerate(values):
                cname = f"c{i}"
                if class_names is not None:
                    cname = _sanitize_class_name(class_names[i])
                out[f"{base_key}_{cname}"] = float(val)

    # compact log line (primary signal)
    primary_key = f"{split_prefix}{args.primary_metric}"
    primary_val = out.get(primary_key, None)

    acc_key = f"{split_prefix}acc"
    acc_val = out.get(acc_key, None)

    f1_key = f"{split_prefix}f1_macro"
    f1_val = out.get(f1_key, None)

    msg_bits = []
    if primary_val is not None:
        msg_bits.append(f"{args.primary_metric}={primary_val:.4f}")

    if acc_val is not None and args.primary_metric != "acc":
        msg_bits.append(f"acc={acc_val:.4f}")

    if f1_val is not None and args.primary_metric != "f1_macro":
        msg_bits.append(f"f1_macro={f1_val:.4f}")
    logger.info(f"{split_prefix.rstrip('/').upper()} metrics: " + " | ".join(msg_bits))

    metrics.reset()
    if per_class is not None:
        per_class.reset()

    return out


# -------------------------
# Train run (single config) / Optuna hook
# -------------------------
def train_one_run(hparams: Dict[str, Any], base_args: argparse.Namespace, run_root: str, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
    import copy

    args = copy.deepcopy(base_args)
    for k, v in hparams.items():
        setattr(args, k, v)

    # Trial-specific seed helps explore stochasticity while staying reproducible
    if trial is not None:
        args.seed = int(args.seed) + int(trial.number)

    setup_device_and_seeds(args)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    # Directory layout (no hyperparameters in filenames):
    #   <run_root>/single/...
    #   <run_root>/trial_<n>/...
    if trial is None:
        run_dir = os.path.join(run_root, "single")
        logger_name = f"{args.id}.single.{timestamp}"
    else:
        run_dir = os.path.join(run_root, f"trial_{trial.number}")
        logger_name = f"{args.id}.trial{trial.number}.{timestamp}"

    os.makedirs(run_dir, exist_ok=True)

    # Log hyperparameters separately (Optuna-style) rather than encoding them in file names.
    hparams_path = os.path.join(run_dir, "hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_number": (None if trial is None else int(trial.number)),
                "seed": int(args.seed),
                "hparams": dict(hparams),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    logger = make_base_logger(run_dir, logger_name=logger_name)
    logger.info(f"Starting run in {run_dir}")
    for arg, value in vars(args).items():
        logger.info(f"ARG {arg}: {value}")

    if trial is not None and args.val_dir == "":
        raise ValueError("Optuna requires --val_dir to be set (objective needs a validation metric).")

    stats_log = open(os.path.join(run_dir, "stats.txt"), "w", encoding="utf-8")

    net1 = create_model(args)
    net2 = create_model(args)
    cudnn.benchmark = True

    criterion = SemiLoss(lambda_u=args.lambda_u)
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ce_per_sample = nn.CrossEntropyLoss(reduction="none")
    ce_loss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy() if args.noise_mode == "asym" else None

    loader = dataloader.imagefolder_dataloader(
        train_dir=args.train_dir,
        val_dir=args.val_dir if args.val_dir else None,
        test_dir=args.test_dir if args.test_dir else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log=stats_log,
    )

    train_targets = datasets.ImageFolder(args.train_dir).targets  # type: ignore[attr-defined]

    all_loss = [[], []]

    best_score = -1.0
    best_epoch = 0

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # pruning warmup
    prune_warmup = args.warm_up if args.optuna_prune_warmup_epochs < 0 else args.optuna_prune_warmup_epochs

    for epoch in range(args.num_epochs + 1):
        # step LR schedule (DivideMix default)
        lr = args.lr
        if epoch >= int(args.num_epochs * 0.5):
            lr /= 10
        if epoch >= int(args.num_epochs * 0.75):
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        eval_loader = loader.run("eval_train")

        if epoch < args.warm_up:
            warmup_trainloader = loader.run("warmup")
            warmup(epoch, net1, optimizer1, warmup_trainloader, ce_loss, conf_penalty, args, logger)
            warmup(epoch, net2, optimizer2, warmup_trainloader, ce_loss, conf_penalty, args, logger)
        else:
            prob1, all_loss[0] = eval_train(net1, eval_loader, ce_per_sample, all_loss[0])
            prob2, all_loss[1] = eval_train(net2, eval_loader, ce_per_sample, all_loss[1])

            pred1 = prob1 > args.p_threshold

            # overall labeled fraction
            print("labeled%:", pred1.mean())

            # per-class labeled fraction (using current dataset labels from ImageFolder)
            # you can access labels via datasets.ImageFolder(train_dir).targets once
            # e.g. precompute train_targets (list of ints) in main
            import numpy as np
            train_targets_np = np.array(train_targets)
            for c in range(args.num_class):
                idx = (train_targets_np == c)
                if idx.sum() > 0:
                    print(c, "count", idx.sum(), "labeled%", pred1[idx].mean())


            pred2 = prob2 > args.p_threshold

            labeled_trainloader, unlabeled_trainloader = loader.run("train", pred2, prob2)
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion, args, logger)

            labeled_trainloader, unlabeled_trainloader = loader.run("train", pred1, prob1)
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion, args, logger)

        # --- validation / early stop / pruning ---
        if args.val_dir:
            val_loader = loader.run("val")
            val_metrics = evaluate(net1, net2, val_loader, split_prefix="val/", args=args, logger=logger)

            score_key = f"val/{args.primary_metric}"
            current_score = float(val_metrics.get(score_key, 0.0))

            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save(net1.state_dict(), os.path.join(ckpt_dir, "net1_best.pth"))
                torch.save(net2.state_dict(), os.path.join(ckpt_dir, "net2_best.pth"))
                logger.info(f"New best {args.primary_metric}: {best_score:.4f} (epoch {best_epoch}), model saved.")
            else:
                epochs_since_improvement = epoch - best_epoch
                if epochs_since_improvement >= args.early_stop_patience:
                    logger.info(
                        f"No improvement in {args.primary_metric} for {epochs_since_improvement} epochs. Stopping early."
                    )
                    break

            # optuna pruning
            if trial is not None:
                trial.report(current_score, step=epoch)
                if epoch >= prune_warmup and trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at epoch {epoch} with {args.primary_metric}={current_score:.4f}")

    logger.info(f"Run finished. Best {args.primary_metric}={best_score:.4f} @ epoch {best_epoch}")

    # --- test evaluation (load best checkpoint if available) ---
    test_metrics: Optional[Dict[str, float]] = None
    if args.test_dir:
        best_net1_path = os.path.join(ckpt_dir, "net1_best.pth")
        best_net2_path = os.path.join(ckpt_dir, "net2_best.pth")
        if os.path.isfile(best_net1_path) and os.path.isfile(best_net2_path):
            logger.info("Loading best checkpoints for test evaluation...")
            net1.load_state_dict(torch.load(best_net1_path, weights_only=True))
            net2.load_state_dict(torch.load(best_net2_path, weights_only=True))
        else:
            logger.warning("No best checkpoint found, testing with final model weights.")

        test_loader = loader.run("test")
        test_metrics = evaluate(net1, net2, test_loader, split_prefix="test/", args=args, logger=logger)
        logger.info(f"Test {args.primary_metric}={test_metrics.get(f'test/{args.primary_metric}', 0.0):.4f}")

    stats_log.close()
    return {
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "run_dir": run_dir,
        "test_metrics": test_metrics,
    }


# -------------------------
# Optuna study runner
# -------------------------
def _make_sampler(args: argparse.Namespace) -> optuna.samplers.BaseSampler:
    if args.optuna_sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed)


def _make_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.optuna_pruner == "nop":
        return optuna.pruners.NopPruner()
    if args.optuna_pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if args.optuna_pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner()
    # default
    return optuna.pruners.MedianPruner()


def run_optuna(args: argparse.Namespace, base_logger: logging.Logger, run_root: str) -> None:
    if args.val_dir == "":
        raise ValueError("--search optuna requires --val_dir (need validation metric).")

    if args.optuna_n_jobs != 1:
        base_logger.warning("GPU training is typically single-process. For safety, forcing --optuna_n_jobs=1.")
        args.optuna_n_jobs = 1

    sampler = _make_sampler(args)
    pruner = _make_pruner(args)

    study_name = args.optuna_study_name.strip() or f"{args.id}_study"
    storage = args.optuna_storage.strip()
    if storage == "":
        storage = f"sqlite:///logs/{study_name}.db"

    base_logger.info(f"Optuna study: name={study_name} storage={storage} sampler={args.optuna_sampler} pruner={args.optuna_pruner}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        # Search space focused on the knobs that usually matter for DivideMix
        hparams: Dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "alpha": trial.suggest_float("alpha", 0.2, 2.0),
            "lambda_u": trial.suggest_float("lambda_u", 1e-2, 5.0, log=True),
            "p_threshold": trial.suggest_float("p_threshold", 0.3, 0.95),
            "T": trial.suggest_float("T", 0.2, 1.0),
            "warm_up": trial.suggest_int("warm_up", 1, max(2, min(20, args.num_epochs // 2))),
            "momentum": trial.suggest_float("momentum", 0.8, 0.95),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        }
        return train_one_run(hparams, args, run_root, trial=trial)["best_score"]

    timeout = None if args.optuna_timeout <= 0 else int(args.optuna_timeout)
    study.optimize(objective, n_trials=int(args.optuna_trials), timeout=timeout, n_jobs=int(args.optuna_n_jobs))

    best = study.best_trial
    base_logger.info(f"Optuna finished. Best value={best.value:.6f} trial={best.number}")
    base_logger.info("Best params:")
    for k, v in best.params.items():
        base_logger.info(f"  {k}: {v}")


# -------------------------
# Grid search runner
# -------------------------
def run_grid(args: argparse.Namespace, base_logger: logging.Logger, run_root: str) -> None:
    if args.val_dir == "":
        base_logger.warning("--search grid without --val_dir: grid search will still run, but best selection uses val metrics.")
    search_space = {
        "lr": _parse_csv_floats(args.grid_lr),
        "batch_size": _parse_csv_ints(args.grid_batch_size),
        "alpha": _parse_csv_floats(args.grid_alpha),
        "lambda_u": _parse_csv_floats(args.grid_lambda_u),
        "p_threshold": _parse_csv_floats(args.grid_p_threshold),
        "T": _parse_csv_floats(args.grid_T),
    }

    keys, values = zip(*search_space.items())
    base_logger.info("Running grid search over:")
    for k in keys:
        base_logger.info(f"  {k}: {search_space[k]}")

    best_score = -1.0
    best_hparams = None

    for hparam_values in product(*values):
        hparams = dict(zip(keys, hparam_values))
        score = train_one_run(hparams, args, run_root, trial=None)["best_score"]
        if score > best_score:
            best_score = score
            best_hparams = hparams

    base_logger.info(f"Grid search finished. Best {args.primary_metric}={best_score:.4f}")
    if best_hparams is not None:
        for k, v in best_hparams.items():
            base_logger.info(f"  {k}: {v}")


# -------------------------
# Entrypoint
# -------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    run_root = os.path.join("logs", f"{args.id}_{timestamp}")
    base_logger = make_base_logger(run_root, logger_name=f"{args.id}.base.{timestamp}")

    base_logger.info("Starting DivideMix")
    for arg, value in vars(args).items():
        base_logger.info(f"ARG {arg}: {value}")

    # Ensure device + seeds are set for any mode
    setup_device_and_seeds(args)

    if args.search == "none":
        # single run with CLI hparams
        train_one_run({}, args, run_root, trial=None)
    elif args.search == "grid":
        run_grid(args, base_logger, run_root)
    elif args.search == "optuna":
        run_optuna(args, base_logger, run_root)
    else:
        raise ValueError(f"Unknown search mode: {args.search}")


if __name__ == "__main__":
    main()
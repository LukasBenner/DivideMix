from __future__ import print_function
import sys
import os
import logging
import argparse
import random
import torchvision.transforms.v2 as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from MobileNetSmall import mobilenet_small
from sklearn.mixture import GaussianMixture
import dataloader_imagefolder as dataloader
from datetime import datetime, timezone


parser = argparse.ArgumentParser(description='DivideMix ImageFolder Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', type=str, help='sym or asym (for warmup penalty)')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0.1, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--warm_up', default=10, type=int, help='warmup epochs')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', required=True, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--train_dir', required=True, type=str)
parser.add_argument('--val_dir', default='', type=str)
parser.add_argument('--test_dir', default='', type=str)
parser.add_argument('--id', default='imagefolder', type=str)
args = parser.parse_args()


timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
run_id = f'{args.id}_{timestamp}'

# Setup logging
os.makedirs('checkpoint', exist_ok=True)
log_file = os.path.join('checkpoint', f'{run_id}_run.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('DivideMix')
logger.info('Starting DivideMix training')

# Log all arguments
for arg, value in vars(args).items():
    logger.info(f'ARG {arg}: {value}')


torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

gpu_train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
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

        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_x += Lx.item()
        running_loss_u += Lu.item()
    
    logger.info(f'Epoch {epoch} - Loss: {running_loss / num_iter:.4f} - Lx: {running_loss_x / num_iter:.4f} - Lu: {running_loss_u / num_iter:.4f}')


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    running_loss = 0.0
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = gpu_train_transforms(inputs)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if args.noise_mode == 'asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss
        L.backward()
        optimizer.step()

        running_loss += loss.item()

    logger.info(f'Epoch {epoch} - Warmup Loss: {running_loss / num_iter:.4f}')


def evaluate(net1, net2, loader, tag):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.0 * correct / total
    logger.info('%s Accuracy: %.2f%%' % (tag, acc))
    return acc


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = mobilenet_small(num_classes=args.num_class)
    model = model.cuda()
    return model


stats_log = open('./checkpoint/%s_stats.txt' % run_id, 'w')

net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()

loader = dataloader.imagefolder_dataloader(
    train_dir=args.train_dir,
    val_dir=args.val_dir if args.val_dir else None,
    test_dir=args.test_dir if args.test_dir else None,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    log=stats_log,
)

all_loss = [[], []]

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= int(args.num_epochs * 0.5):
        lr /= 10
    if epoch >= int(args.num_epochs * 0.75):
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    eval_loader = loader.run('eval_train')

    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = prob1 > args.p_threshold
        pred2 = prob2 > args.p_threshold

        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)

        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)

    if args.val_dir:
        val_loader = loader.run('val')
        evaluate(net1, net2, val_loader, 'Val')
    if args.test_dir:
        test_loader = loader.run('test')
        evaluate(net1, net2, test_loader, 'Test')

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image, ImageReadMode
import torch


class ImagefolderDataset(Dataset):
    def __init__(self, train_dir, transform, mode, val_dir=None, test_dir=None, pred=[], probability=[], log=''):
        self.transform = transform
        self.mode = mode
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.log = log

        if self.mode == 'test':
            if not self.test_dir:
                raise ValueError('test_dir is required for mode="test"')
            dataset = datasets.ImageFolder(self.test_dir)
            self.test_imgs = [path for path, _ in dataset.samples]
            self.test_labels = [label for _, label in dataset.samples]
        elif self.mode == 'val':
            if not self.val_dir:
                raise ValueError('val_dir is required for mode="val"')
            dataset = datasets.ImageFolder(self.val_dir)
            self.val_imgs = [path for path, _ in dataset.samples]
            self.val_labels = [label for _, label in dataset.samples]
        else:
            dataset = datasets.ImageFolder(self.train_dir)
            self.train_imgs = [path for path, _ in dataset.samples]
            self.noise_label = [label for _, label in dataset.samples]

            if self.mode == 'all':
                pass
            elif self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                self.train_imgs = [self.train_imgs[i] for i in pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]
                self.probability = [probability[i] for i in pred_idx]
                if self.log:
                    self.log.write('Numer of labeled samples:%d\n' % (pred.sum()))
                    self.log.flush()
            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]
                self.train_imgs = [self.train_imgs[i] for i in pred_idx]
                self.noise_label = [self.noise_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            prob = self.probability[index]
            image = read_image(img_path, mode=ImageReadMode.RGB)
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = read_image(img_path, mode=ImageReadMode.RGB)
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.noise_label[index]
            image = read_image(img_path, mode=ImageReadMode.RGB)
            img = self.transform(image)
            return img, target, index
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[index]
            image = read_image(img_path, mode=ImageReadMode.RGB)
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.val_labels[index]
            image = read_image(img_path, mode=ImageReadMode.RGB)
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        return len(self.train_imgs)


class imagefolder_dataloader:
    def __init__(self, train_dir, val_dir, test_dir, batch_size, num_workers, log=''):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log = log

        self.cpu_train_transform = transforms.Compose([
            transforms.Resize(480, antialias=True),
            transforms.CenterCrop(480)
        ])

        self.cpu_test_transform = transforms.Compose([
            transforms.Resize(480, antialias=True),
            transforms.CenterCrop(480),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_train_transform,
                mode='all',
                log=self.log
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )
            return trainloader
        elif mode == 'train':
            labeled_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_train_transform,
                mode='labeled',
                pred=pred,
                probability=prob,
                log=self.log
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )

            unlabeled_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_train_transform,
                mode='unlabeled',
                pred=pred,
                log=self.log
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )
            return labeled_trainloader, unlabeled_trainloader
        elif mode == 'eval_train':
            eval_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_test_transform,
                mode='all',
                log=self.log
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )
            return eval_loader
        elif mode == 'test':
            test_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_test_transform,
                mode='test',
                test_dir=self.test_dir
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )
            return test_loader
        elif mode == 'val':
            val_dataset = ImagefolderDataset(
                train_dir=self.train_dir,
                transform=self.cpu_test_transform,
                mode='val',
                val_dir=self.val_dir
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=4
            )
            return val_loader

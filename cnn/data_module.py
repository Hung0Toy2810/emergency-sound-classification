import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SoundDataModule:
    """Quản lý transform, split và DataLoader cho dữ liệu mel-spectrogram."""

    def __init__(self, data_path, batch_size=16, val_ratio=0.15, num_workers=0, seed=42, img_size=224, device=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.img_size = img_size
        self.device = device or torch.device("cpu")
        self.class_names = []
        self.train_targets = []
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self):
        train_set, val_set, test_set = self._build_datasets()
        self.train_loader = self._loader(train_set, shuffle=True)
        self.val_loader = self._loader(val_set, shuffle=False) if val_set else None
        self.test_loader = self._loader(test_set, shuffle=False) if test_set else None
        return self

    def loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def _transforms(self):
        train_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.1, 3.3)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return train_tf, eval_tf

    def _build_datasets(self):
        train_tf, eval_tf = self._transforms()
        train_dir = os.path.join(self.data_path, "train")
        val_dir = os.path.join(self.data_path, "val")
        test_dir = os.path.join(self.data_path, "test")

        if os.path.isdir(train_dir):
            train_full = datasets.ImageFolder(train_dir, transform=train_tf)
            self.class_names = train_full.classes
            self.train_targets = train_full.targets
            if os.path.isdir(val_dir):
                val_set = datasets.ImageFolder(val_dir, transform=eval_tf)
            else:
                val_set = None
            test_set = datasets.ImageFolder(test_dir, transform=eval_tf) if os.path.isdir(test_dir) else None
            return train_full, val_set, test_set

        train_full = datasets.ImageFolder(self.data_path, transform=train_tf)
        self.class_names = train_full.classes
        self.train_targets = train_full.targets
        return train_full, None, None

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

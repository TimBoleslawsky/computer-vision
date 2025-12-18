from pathlib import Path

import pytorch_lightning as pl
import rasterio
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset


class GeoTiffDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with rasterio.open(path) as src:
            image = src.read().astype("float32") / 255.0
            target = image[-1]  # last band
            image = image[:-1]  # remaining bands
            image = torch.tensor(image)
            target = torch.tensor(target, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, target


class GeoTiffDataModule(pl.LightningDataModule):
    def __init__(self, raw_dir, batch_size=16, num_workers=4, transform=None, n_splits=5, fold_idx=0, random_state=42):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.random_state = random_state

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None  # optional

    def setup(self, stage=None):
        # Load all file paths
        all_files = sorted(list(self.raw_dir.glob("*.tif")))
        self.full_dataset = GeoTiffDataset(all_files, transform=self.transform)

        # Prepare k-fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(all_files))
        train_idx, val_idx = folds[self.fold_idx]

        # Create train/val subsets
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)

        # Optional: leave some separate test set if desired
        # self.test_dataset = Subset(self.full_dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageFolderDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping torchvision.datasets.ImageFolder."""

    def __init__(
            self,
            train_dir: Optional[str],
            val_dir: Optional[str],
            test_dir: Optional[str],
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: Tuple[int, int] = (128, 128),
            mean: Tuple[float, ...] = (0.5,),
            std: Tuple[float, ...] = (0.5,),
            augment: bool = True,
    ):
        super().__init__()
        self.train_dir = Path(train_dir) if train_dir is not None else None
        self.val_dir = Path(val_dir) if val_dir is not None else None
        self.test_dir = Path(test_dir) if test_dir is not None else None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment

        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None

        # default transforms
        resize_and_to_tensor = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        if self.augment:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    *resize_and_to_tensor,
                ]
            )
        else:
            self.train_transform = transforms.Compose(resize_and_to_tensor)

        self.eval_transform = transforms.Compose(resize_and_to_tensor)

    def prepare_data(self):
        # Intentionally left blank: ImageFolder expects data already present on disk.
        pass

    def setup(self, stage: Optional[str] = None):
        # create datasets if dirs provided
        if stage in (None, "fit"):
            if self.train_dir is not None:
                self.train_dataset = ImageFolder(str(self.train_dir), transform=self.train_transform)
            if self.val_dir is not None:
                self.val_dataset = ImageFolder(str(self.val_dir), transform=self.eval_transform)

        if stage in (None, "test"):
            if self.test_dir is not None:
                self.test_dataset = ImageFolder(str(self.test_dir), transform=self.eval_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

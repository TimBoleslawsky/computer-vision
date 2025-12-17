from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CustomDataLoader:
    @staticmethod
    def load_data(train_path, val_path, test_path) -> tuple[DataLoader, DataLoader, DataLoader]:
        # Defining Transformation
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Added for data augmentation
            transforms.RandomRotation(15),  # Added for data augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # Added for data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # transform_vgg_train = pretrained_transforms # This is for transfer learning
        # transform_vgg_val = pretrained_transforms # This is for transfer learning

        # ImageFolder locates and formats the data
        train_set = ImageFolder(train_path, transform=transform_train)
        val_set = ImageFolder(val_path, transform=transform_val)
        test_set = ImageFolder(test_path, transform=transform_val)

        # DataLoader delivers it efficiently to the model
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

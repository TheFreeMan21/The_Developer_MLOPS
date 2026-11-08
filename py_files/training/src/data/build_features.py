import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.config import IMAGE_SIZE, TRANSFORM_MEAN, TRANSFORM_STD, BATCH_SIZE

class GlassVialDataset(Dataset):
    """Loads vial images and their labels and applies the transform."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img.copy()

        if self.transform:
            image = self.transform(image)

        return image, label

def define_transforms():
    """Defines training (with augmentation) and validation transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])
    return train_transforms, val_transforms

def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Stratified split so both classes keep the same balance in every set."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]

    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        return data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1

    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, train_transforms, val_transforms):
    """Wraps datasets in PyTorch DataLoaders."""
    train_paths, train_labels = zip(*train_dataset) if train_dataset else ([], [])
    val_paths, val_labels = zip(*val_dataset) if val_dataset else ([], [])
    test_paths, test_labels = zip(*test_dataset) if test_dataset else ([], [])

    train_data = GlassVialDataset(list(train_paths), list(train_labels), train_transforms)
    val_data = GlassVialDataset(list(val_paths), list(val_labels), val_transforms)
    test_data = GlassVialDataset(list(test_paths), list(test_labels), val_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

def calculate_image_statistics(data_loader):
    """Calculate mean and standard deviation per channel for a dataset."""
    total_mean, total_std, total_samples = np.zeros(3), np.zeros(3), 0

    for images, _ in data_loader:
        batch_images = images.numpy()
        batch_size = batch_images.shape[0]
        batch_images = batch_images.transpose(0, 2, 3, 1).reshape(batch_size, -1, 3)

        total_mean += batch_images.mean(axis=1).sum(axis=0)
        total_std += batch_images.std(axis=1).sum(axis=0)
        total_samples += batch_size

    return {
        "mean": (total_mean / total_samples).tolist(),
        "std": (total_std / total_samples).tolist(),
        "total_samples": total_samples,
    }
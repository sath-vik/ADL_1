import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from cutmix.cutmix import CutMix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, samples_per_class=None, return_class_names=False):
        self.image_dir = image_dir
        self.transform = transform
        self.return_class_names = return_class_names
        self.image_paths = []
        self.labels = []

        # Load class definitions
        classes_csv = os.path.join(image_dir, '_classes.csv')
        classes_df = pd.read_csv(classes_csv)
        
        # Extract class names and mapping
        self.classes = classes_df.columns[1:].tolist()
        file_to_label = {
            row['filename']: row.drop('filename').values.argmax() 
            for _, row in classes_df.iterrows()
        }

        # Collect valid samples
        all_paths = []
        all_labels = []
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.jpg', '.png')) and img_name in file_to_label:
                img_path = os.path.join(image_dir, img_name)
                all_paths.append(img_path)
                all_labels.append(file_to_label[img_name])

        # Apply few-shot sampling if requested
        if samples_per_class is not None:
            class_counts = defaultdict(int)
            filtered_paths = []
            filtered_labels = []
            
            for path, label in zip(all_paths, all_labels):
                if class_counts[label] < samples_per_class:
                    filtered_paths.append(path)
                    filtered_labels.append(label)
                    class_counts[label] += 1
            self.image_paths = filtered_paths
            self.labels = filtered_labels
        else:
            self.image_paths = all_paths
            self.labels = all_labels

        print(f"\nDataset: {image_dir}")
        print(f"Total samples: {len(self.image_paths)}")
        print(f"Class distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.return_class_names:
            return image, self.classes[label]  # Return class name for zero-shot
        return image, torch.tensor(label, dtype=torch.long)  # Return index for training

    def get_class_list(self):
        return self.classes

def get_data_loaders(train_dir, val_dir, test_dir, num_classes, 
                    batch_size=32, img_size=224, augmentation='standard',
                    samples_per_class=None):
    
    # Base transforms for all datasets
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Training augmentations
    if augmentation == 'advanced':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            base_transform
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(40),
            base_transform
        ])

    # Create datasets
    train_dataset = ImageDataset(
        train_dir, 
        train_transform, 
        samples_per_class=samples_per_class
    )
    val_dataset = ImageDataset(val_dir, base_transform)
    test_dataset = ImageDataset(test_dir, base_transform)

    # Apply CutMix for advanced augmentation
    if augmentation == 'advanced':
        train_dataset = CutMix(
            train_dataset,
            num_class=num_classes,
            beta=1.0,
            prob=0.5,
            num_mix=1
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
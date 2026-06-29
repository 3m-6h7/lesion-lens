import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LesionDataset(Dataset):
    def __init__(self, img_dir, csv_file, augment=False):
        """
        Custom dataset processor for parsing image frames and corresponding categorical matrices.
        """
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.augment = augment
        
        # Extract target names and translate one-hot vectors into integer classification targets
        self.img_ids = self.df.iloc[:, 0].values
        self.labels = self.df.iloc[:, 1:].values.argmax(axis=1)
        
        # Consistent structural normalization values across experimental backbones
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Task 1e: Conditional augmentation logic
        if self.augment:
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),      # Mandatory Requirement
                transforms.RandomVerticalFlip(p=0.5),        # Non-Deterministic Option 1
                transforms.RandomRotation(degrees=20),       # Non-Deterministic Option 2
                transforms.ColorJitter(brightness=0.15, contrast=0.15)
            ]
            self.transform = transforms.Compose(augment_transforms + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        # Append extension dynamically if missing within underlying source labels
        if not str(img_name).lower().endswith('.jpg'):
            img_name = f"{img_name}.jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import os

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_path, img_dir, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.df['dx'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row['dx']]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def get_weighted_sampler(dataset):
    targets = [dataset.class_to_idx[row] for row in dataset.df['dx']]
    class_counts = pd.Series(targets).value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    return sampler
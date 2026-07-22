import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dataset import SkinLesionDataset
from model import LesionCNN

def run_hybrid_pipeline():
    #1. Setup device and load trained model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    model = LesionCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load("models/lesion_model.pth", map_location=device))
    model.eval()

    #2. Set up data loader (without the weighted sampler, for sequential feature extraction)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SkinLesionDataset(
        metadata_path='data/subset_3800/metadata_subset.csv',
        img_dir='data/subset_3800/',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    #3. Extract 2048-dimensional deep latent features from the CNN
    print("Extracting CNN latent representations...")
    all_features = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _, features = model(images)
            all_features.append(features.cpu().numpy())
            
    cnn_features = np.vstack(all_features)

    #4. Process clinical metadata (one-hot encode categorical features like sex and localization)
    print("Processing clinical metadata...")
    df = dataset.df.copy()
    
    #Fill missing values if any exist in age
    df['age'] = df['age'].fillna(df['age'].median())
    
    #One-hot encode categorical features
    clinical_features = pd.get_dummies(df[['age', 'sex', 'localization']])
    
    #Scale age or numerical attributes if desired, then combine
    X_tabular = clinical_features.values
    
    #5. Concatenate deep features + tabular clinical features
    X_hybrid = np.hstack([cnn_features, X_tabular])
    y = df['dx'].values

    #6. Train/Test Split for the hybrid classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.2, random_state=42, stratify=y
    )

    #7. Train a scikit-learn classifier on the hybrid feature space
    print("Training scikit-learn Random Forest hybrid model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    #8. Evaluate results
    y_pred = clf.predict(X_test)
    print("\n--- Hybrid Model Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("Hybrid pipeline execution complete!")

if __name__ == "__main__":
    run_hybrid_pipeline()
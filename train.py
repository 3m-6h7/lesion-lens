import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import wandb

from dataset import SkinLesionDataset, get_weighted_sampler
from model import LesionCNN

def train_model():
    # 1. Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="lesion-lens",
        config={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "subset_size": 3800
        }
    )
    config = wandb.config

    # 2. Set up device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 3. Define data transforms (Augmentation + ImageNet Normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load Dataset and Weighted Sampler to handle class imbalance
    dataset = SkinLesionDataset(
        metadata_path='data/subset_3800/metadata_subset.csv',
        img_dir='data/subset_3800/',
        transform=transform
    )
    
    sampler = get_weighted_sampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=2
    )

    # 5. Initialize Model, Loss, and Optimizer
    model = LesionCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Watch model gradients with W&B
    wandb.watch(model, log="all", log_freq=10)

    # 6. Training Loop
    print("Starting training loop...")
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits, features = model(images)
            loss = criterion(logits, labels)

            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log batch metrics to W&B
            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.epochs}] | Loss: {epoch_loss:.4f} | Approx Accuracy: {epoch_acc:.2f}%")

        # Log epoch metrics to W&B
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc})

    # 7. Save model checkpoint
    os.makedirs("models", exist_ok=True)
    model_path = "models/lesion_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save artifact to W&B
    artifact = wandb.Artifact("lesion_cnn", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print("Training complete and logged to Weights & Biases!")

if __name__ == "__main__":
    train_model()
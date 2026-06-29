import torch
import torch.nn as nn
import torchmetrics
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, class_names, debug_single_batch=False):
    """
    Core training and validation orchestration framework. Logs information directly to W&B.
    """
    model.to(device)
    
    # Initialize standard benchmarking indicators
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
    uar_metric = torchmetrics.Recall(task="multiclass", num_classes=len(class_names), average="macro").to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        accuracy_metric.reset()
        uar_metric.reset()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            accuracy_metric.update(preds, labels)
            uar_metric.update(preds, labels)
            
            # Task 1b Q2: Debugging hook enabling quick single-batch runtime evaluation
            if debug_single_batch:
                print("⚡ Debugging Mode: Single-batch processing block completed.")
                break

        epoch_train_loss = running_loss / (len(train_loader.dataset) if not debug_single_batch else images.size(0))
        epoch_train_acc = accuracy_metric.compute().item()
        epoch_train_uar = uar_metric.compute().item()
        
        # Validation Checkpoint
        val_loss, val_acc, val_uar, all_preds, all_labels = validate_model(
            model, val_loader, criterion, device, accuracy_metric, uar_metric
        )
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} [Acc: {epoch_train_acc:.3f}, UAR: {epoch_train_uar:.3f}] ── "
              f"Val Loss: {val_loss:.4f} [Acc: {val_acc:.3f}, Val UAR: {val_uar:.3f}]")
        
        # Route pipeline data metrics directly to Weights & Biases if a workspace is active
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_train_loss,
                "train/accuracy": epoch_train_acc,
                "train/uar": epoch_train_uar,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
                "validation/uar": val_uar
            })
            
    # Task 1c: Export and plot evaluation matrices after final convergence sequence
    plot_confusion_matrix(all_labels, all_preds, class_names)


def validate_model(model, val_loader, criterion, device, accuracy_metric, uar_metric):
    model.eval()
    running_loss = 0.0
    
    accuracy_metric.reset()
    uar_metric.reset()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            accuracy_metric.update(preds, labels)
            uar_metric.update(preds, labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = accuracy_metric.compute().item()
    val_uar = uar_metric.compute().item()
    
    return val_loss, val_acc, val_uar, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates and saves a Normalized Confusion Matrix as requested in Task 1c.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Row-wise normalization mapping (accounts for sample prevalence shifts)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix (Validation Distribution Check)")
    plt.ylabel("Ground Truth Class")
    plt.xlabel("Predicted Class Mapping")
    plt.tight_layout()
    
    # Save the output visualization directly to the configured file engine path
    plt.savefig("logs/normalized_confusion_matrix.png", dpi=150)
    plt.show()
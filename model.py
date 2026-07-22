import torch
import torch.nn as nn
from torchvision import models

class LesionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(LesionCNN, self).__init__()
        #Load pre-trained ResNet50 with modern weights syntax
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        #Remove the original final fully connected layer (-1) 
        #so the output becomes the 2048-dimensional latent vector
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        #New classification head for our 7 HAM10000 diagnostic classes
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # Extract latent representations
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        #Get class prediction logits
        logits = self.classifier(features)
        
        #Return both so we can use them for deep learning or the hybrid pipeline
        return logits, features
import torch
import torch.nn as nn
import torchvision.models as models

# Task 1c: Baseline Convolutional Neural Network Implementation
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleBNConv, self).__init__()
        
        # 5 Convolutional sequences with scaling output maps: 8 -> 16 -> 32 -> 64 -> 128
        self.features = nn.Sequential(
            # Sequence 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Sequence 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Sequence 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Sequence 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Sequence 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling ensures dimensionality safety across diverse image dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Task 1f: Pretrained Target Backbone Implementation Factory
def get_pretrained_resnet18(num_classes=7, freeze_features=True):
    """
    Instantiates a ResNet18 backbone from the Torchvision Model Zoo.
    Enables structural testing of baseline feature extraction vs absolute fine-tuning.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
        
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
            
    # Swap out terminal linear configuration with appropriate diagnostic dimensions
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
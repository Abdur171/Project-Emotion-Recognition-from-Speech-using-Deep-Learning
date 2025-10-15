# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4, config=None):
        super(EmotionCNN, self).__init__()
        
        # Input: (1, 64, 94) for 3s audio at 16kHz with hop_length=512
        # n_mels=64, time_frames = (16000*3)//512 = 93.75 -> 94
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions and pooling
        # After each pool, the time and frequency dimensions are halved
        # Initial: (1, 64, 94)
        # After conv1 -> (32, 64, 94), pool -> (32, 32, 47)
        # After conv2 -> (64, 32, 47), pool -> (64, 16, 23)
        # After conv3 -> (128, 16, 23), pool -> (128, 8, 11)
        self.linear_input_size = 128 * 8 * 11
        
        self.fc1 = nn.Linear(self.linear_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vision encoder for the VLM portion.
# For simplicity a small-scale convolutional network is used
class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, d_model=128):
        # Declare layers in constructor
        super().__init__()
        # 3 Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Flatten / fully-connect output layer
        self.proj = nn.Linear(128, d_model)
        # Layer Normalization 
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # Forward pass
        # Activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Avg pooling
        x = x.mean(dim=[2, 3])  # GAP
        # Fully-connect + Layer Norm
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)




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



# Encoding text using gated-recurrent unit
# Encodes input sequence e.g. "push the object to goal" into a fixed-length embedding
# in the same space as the CNN (output dims of forward need to be the same dimension i.e. B x d_model)
class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        # Initialize embedding table
        self.embed = nn.Embedding(vocab_size, d_word)
        # Gated-recurrent unit
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        # Layer Normalization
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        # Forward pass
        # Embedding lookup
        x = self.embed(token_ids)
        # Gated-recurrent unit
        _, h_last = self.gru(x)
        # Last hidden state
        x = h_last[0]
        # Layer Normalization
        x = self.ln(x)
        return x  # (B, d_model)


# Encoding robot state using a simple Multi-Layer Perceptron
# State vector gets encoded into the same [B x d_model] dimension as the CNN and text encoder
# This step is typically the bottleneck in terms of representational power / efficiency for downstream tasks
# e.g. attention to query for what parts of the state vector are relevant for the assigned instruction 
class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.ln(x)
        return x
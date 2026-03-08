
"""Fusion module: Combines the different
encodings from our various modalities (image, text, state).
More generally this module also decides HOW information is blended
across the different modalities but the implementation here
is trivial (concatenation   )"""

import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token):
        # Concatenate the tokens along the last dimension
        x = torch.cat([img_token, txt_token, state_token], dim=-1)
        # Pass through MLP (Multi-Layer Perceptron)
        x = self.net(x)
        # Layer Normalization
        x = self.ln(x)
        return x
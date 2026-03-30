import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)   # (B,1,28,28) → (B,784)
        x = self.fc(x)        # → (B,10)
        return x
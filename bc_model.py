"""BCNet：图像 + qpos -> 未来 chunk 步动作。"""

import torch
import torch.nn as nn


class BCNet(nn.Module):
    def __init__(self, chunk=16, action_dim=8, qpos_dim=36):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.q = nn.Sequential(nn.Linear(qpos_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 64, 256), nn.ReLU(),
            nn.Linear(256, chunk * action_dim),
        )

    def forward(self, img, qpos):
        f = self.cnn(img).flatten(1)
        q = self.q(qpos)
        out = self.head(torch.cat([f, q], dim=1))
        return out.view(-1, self.chunk, self.action_dim)

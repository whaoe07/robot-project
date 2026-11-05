"""ACT-lite：图像历史 + qpos 历史 -> 未来 chunk 步动作。

结构：
  CNN 编码每一帧 -> 拼 qpos -> Transformer Encoder（时序）
  -> 可学习的 chunk 个 query -> Transformer Decoder -> 动作
"""

import torch
import torch.nn as nn


class ACTNet(nn.Module):
    def __init__(
        self,
        chunk=16,
        action_dim=8,
        qpos_dim=9,
        history=4,
        d_model=128,
        nhead=4,
        enc_layers=2,
        dec_layers=2,
    ):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.history = history

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(128 * 4 * 4 + qpos_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, history, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.head = nn.Linear(d_model, action_dim)

    def forward(self, imgs, qpos):
        # imgs: (B, H, 3, S, S)   qpos: (B, H*qpos_dim)
        B, H = imgs.shape[0], imgs.shape[1]
        f = self.cnn(imgs.reshape(B * H, *imgs.shape[2:])).reshape(B, H, -1)
        q = qpos.reshape(B, H, -1)
        x = self.proj(torch.cat([f, q], dim=-1)) + self.pos
        memory = self.encoder(x)
        queries = self.queries.expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return self.head(out)

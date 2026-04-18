from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder3D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 16, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(base_ch * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.proj(x)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return torch.bmm(adj, h)


class GraphResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gc1 = GraphConv(dim, dim)
        self.gc2 = GraphConv(dim, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, adj))
        h = self.gc2(h, adj)
        return F.relu(x + h)


class MeshDeformNetLite(nn.Module):
    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        delta_scale: float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder3D(in_ch=1, base_ch=16, feat_dim=feat_dim)
        self.input_proj = nn.Linear(3 + feat_dim, hidden_dim)
        self.blocks = nn.ModuleList([GraphResBlock(hidden_dim) for _ in range(num_blocks)])
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.delta_scale = float(delta_scale)

    def forward(
        self,
        image: torch.Tensor,
        init_vertices: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        # image: [B,1,D,H,W], init_vertices: [B,N,3], adj: [B,N,N]
        g = self.encoder(image)  # [B,F]
        g = g.unsqueeze(1).expand(-1, init_vertices.shape[1], -1)  # [B,N,F]
        x = torch.cat([init_vertices, g], dim=-1)
        x = F.relu(self.input_proj(x))
        for blk in self.blocks:
            x = blk(x, adj)
        delta = self.delta_head(x) * self.delta_scale
        return init_vertices + delta

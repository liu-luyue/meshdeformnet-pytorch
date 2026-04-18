from __future__ import annotations

from typing import Dict

import torch


def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred, gt: [B,N,3]
    dist = torch.cdist(pred, gt, p=2) ** 2
    loss_1 = dist.min(dim=2)[0].mean()
    loss_2 = dist.min(dim=1)[0].mean()
    return loss_1 + loss_2


def edge_length_loss(pred: torch.Tensor, gt: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    # edges: [E,2]
    p1 = pred[:, edges[:, 0], :]
    p2 = pred[:, edges[:, 1], :]
    g1 = gt[:, edges[:, 0], :]
    g2 = gt[:, edges[:, 1], :]
    lp = torch.norm(p1 - p2, dim=-1)
    lg = torch.norm(g1 - g2, dim=-1)
    return torch.mean(torch.abs(lp - lg))


def laplacian_smooth_loss(pred: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    # Row-normalized adjacency, including self-loop
    lap = pred - torch.bmm(adj, pred)
    return (lap ** 2).mean()


def geometric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    edges: torch.Tensor,
    adj: torch.Tensor,
    w_chamfer: float = 1.0,
    w_edge: float = 0.1,
    w_lap: float = 0.1,
) -> Dict[str, torch.Tensor]:
    l_ch = chamfer_loss(pred, gt)
    l_ed = edge_length_loss(pred, gt, edges)
    l_lp = laplacian_smooth_loss(pred, adj)
    total = w_chamfer * l_ch + w_edge * l_ed + w_lap * l_lp
    return {
        "total": total,
        "chamfer": l_ch.detach(),
        "edge": l_ed.detach(),
        "lap": l_lp.detach(),
    }


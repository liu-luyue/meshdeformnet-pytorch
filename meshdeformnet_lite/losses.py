from __future__ import annotations

from typing import Dict

import torch


def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(weights.sum(), min=1.0)
    return (values * weights).sum() / denom


def chamfer_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # pred, gt: [B,S,N,3], valid: [B,S]
    bsz, num_struct, num_points, _ = pred.shape
    pred_f = pred.reshape(bsz * num_struct, num_points, 3)
    gt_f = gt.reshape(bsz * num_struct, num_points, 3)
    dist = torch.cdist(pred_f, gt_f, p=2) ** 2
    ch = dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)  # [B*S]
    return _safe_weighted_mean(ch, valid.reshape(-1))


def edge_length_loss(pred: torch.Tensor, gt: torch.Tensor, edges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # pred, gt: [B,S,N,3], edges: [E,2], valid: [B,S]
    p1 = pred[:, :, edges[:, 0], :]
    p2 = pred[:, :, edges[:, 1], :]
    g1 = gt[:, :, edges[:, 0], :]
    g2 = gt[:, :, edges[:, 1], :]
    lp = torch.norm(p1 - p2, dim=-1)
    lg = torch.norm(g1 - g2, dim=-1)
    ed = torch.abs(lp - lg).mean(dim=-1)  # [B,S]
    return _safe_weighted_mean(ed.reshape(-1), valid.reshape(-1))


def laplacian_smooth_loss(pred: torch.Tensor, adj: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # pred: [B,S,N,3], adj: [B,N,N], valid: [B,S]
    bsz, num_struct, num_points, _ = pred.shape
    pred_f = pred.reshape(bsz * num_struct, num_points, 3)
    adj_f = adj.unsqueeze(1).expand(-1, num_struct, -1, -1).reshape(bsz * num_struct, num_points, num_points)
    lap = pred_f - torch.bmm(adj_f, pred_f)
    lp = (lap**2).mean(dim=(1, 2))  # [B*S]
    return _safe_weighted_mean(lp, valid.reshape(-1))


def _vertex_normals(mesh: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # mesh: [B,S,N,3], faces: [F,3]
    bsz, num_struct, num_points, _ = mesh.shape
    m = mesh.reshape(bsz * num_struct, num_points, 3)
    i0 = faces[:, 0].long()
    i1 = faces[:, 1].long()
    i2 = faces[:, 2].long()
    v0 = m[:, i0, :]
    v1 = m[:, i1, :]
    v2 = m[:, i2, :]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [BS,F,3]
    normals = torch.zeros_like(m)
    normals.index_add_(1, i0, fn)
    normals.index_add_(1, i1, fn)
    normals.index_add_(1, i2, fn)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    return normals.reshape(bsz, num_struct, num_points, 3)


def normal_loss(pred: torch.Tensor, gt: torch.Tensor, faces: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # pred, gt: [B,S,N,3], faces [F,3], valid [B,S]
    if faces.numel() == 0:
        return pred.new_tensor(0.0)
    n_pred = _vertex_normals(pred, faces)
    n_gt = _vertex_normals(gt, faces)
    cos = torch.sum(n_pred * n_gt, dim=-1).clamp(min=-1.0, max=1.0)
    nl = (1.0 - cos).mean(dim=-1)  # [B,S]
    return _safe_weighted_mean(nl.reshape(-1), valid.reshape(-1))


def geometric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    edges: torch.Tensor,
    adj: torch.Tensor,
    faces: torch.Tensor,
    valid: torch.Tensor,
    w_chamfer: float = 1.0,
    w_edge: float = 0.1,
    w_lap: float = 0.05,
    w_normal: float = 0.2,
) -> Dict[str, torch.Tensor]:
    l_ch = chamfer_loss(pred, gt, valid)
    l_ed = edge_length_loss(pred, gt, edges, valid)
    l_lp = laplacian_smooth_loss(pred, adj, valid)
    l_nm = normal_loss(pred, gt, faces, valid)
    total = w_chamfer * l_ch + w_edge * l_ed + w_lap * l_lp + w_normal * l_nm
    return {
        "total": total,
        "chamfer": l_ch.detach(),
        "edge": l_ed.detach(),
        "lap": l_lp.detach(),
        "normal": l_nm.detach(),
    }

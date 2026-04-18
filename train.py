from __future__ import annotations

import argparse
import csv
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from meshdeformnet_lite.dataset import build_datasets
from meshdeformnet_lite.losses import geometric_loss
from meshdeformnet_lite.model import MeshDeformNetLite
from meshdeformnet_lite.mesh_utils import create_template
from meshdeformnet_lite.mesh_utils import create_template_from_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MeshDeformNet Lite (PyTorch)")
    p.add_argument("--data_mode", type=str, default="dummy", choices=["dummy", "npz"])
    p.add_argument("--train_dir", type=str, default="")
    p.add_argument("--val_dir", type=str, default="")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_vertices", type=int, default=512)
    p.add_argument("--k_neighbors", type=int, default=6)
    p.add_argument("--template_mesh", type=str, default="data/template/sphere_coarse.vtp")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--delta_scale", type=float, default=0.2)
    p.add_argument("--num_structures", type=int, default=7)
    p.add_argument("--w_chamfer", type=float, default=1.0)
    p.add_argument("--w_edge", type=float, default=0.1)
    p.add_argument("--w_lap", type=float, default=0.05)
    p.add_argument("--w_normal", type=float, default=0.2)
    p.add_argument("--save_dir", type=str, default="runs/lite")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate(
    model: MeshDeformNetLite,
    loader: DataLoader,
    init_struct: torch.Tensor,
    edges: torch.Tensor,
    faces: torch.Tensor,
    adj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            b = batch["image"].shape[0]
            valid = batch["valid"]
            init_b = init_struct.unsqueeze(0).expand(b, -1, -1, -1)
            adj_b = adj.unsqueeze(0).expand(b, -1, -1)
            pred = model(batch["image"], init_b, adj_b)
            losses = geometric_loss(
                pred,
                batch["points"],
                edges=edges,
                adj=adj_b,
                faces=faces,
                valid=valid,
                w_chamfer=args.w_chamfer,
                w_edge=args.w_edge,
                w_lap=args.w_lap,
                w_normal=args.w_normal,
            )
            total += float(losses["total"].item()) * b
            n += b
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    metrics_csv = os.path.join(args.save_dir, "metrics.csv")

    if args.template_mesh:
        template_v, template_f, template_e, template_adj = create_template_from_mesh(args.template_mesh)
        args.num_vertices = int(template_v.shape[0])
    else:
        template_v, template_e, template_adj = create_template(args.num_vertices, k=args.k_neighbors)
        template_f = np.zeros((0, 3), dtype=np.int64)
    init_vertices = torch.from_numpy(template_v).float().to(device)
    edges = torch.from_numpy(template_e).long().to(device)
    faces = torch.from_numpy(template_f).long().to(device)
    adj = torch.from_numpy(template_adj).float().to(device)
    init_struct = init_vertices.unsqueeze(0).repeat(args.num_structures, 1, 1)

    datasets = build_datasets(
        data_mode=args.data_mode,
        image_size=args.image_size,
        num_vertices=args.num_vertices,
        template_vertices=template_v,
        num_structures=args.num_structures,
        train_dir=args.train_dir if args.train_dir else None,
        val_dir=args.val_dir if args.val_dir else None,
    )
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MeshDeformNetLite(feat_dim=128, hidden_dim=128, num_blocks=4, delta_scale=args.delta_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["epoch", "train_total", "val_total", "best_val", "is_best"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        train_total_sum = 0.0
        train_n = 0
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            b = batch["image"].shape[0]
            valid = batch["valid"]
            init_b = init_struct.unsqueeze(0).expand(b, -1, -1, -1)
            adj_b = adj.unsqueeze(0).expand(b, -1, -1)

            pred = model(batch["image"], init_b, adj_b)
            losses = geometric_loss(
                pred,
                batch["points"],
                edges=edges,
                adj=adj_b,
                faces=faces,
                valid=valid,
                w_chamfer=args.w_chamfer,
                w_edge=args.w_edge,
                w_lap=args.w_lap,
                w_normal=args.w_normal,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            train_total_sum += float(losses["total"].item()) * b
            train_n += int(b)

            pbar.set_postfix(
                total=f"{losses['total'].item():.4f}",
                ch=f"{losses['chamfer'].item():.4f}",
                edge=f"{losses['edge'].item():.4f}",
                lap=f"{losses['lap'].item():.4f}",
                normal=f"{losses['normal'].item():.4f}",
            )

        train_total = train_total_sum / max(1, train_n)
        val_loss = evaluate(model, val_loader, init_struct, edges, faces, adj, args, device)
        print(f"[Epoch {epoch}] train_total={train_total:.6f} val_total={val_loss:.6f}")

        ckpt_last = os.path.join(args.save_dir, "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "template_vertices": template_v.astype(np.float32),
                "template_edges": template_e.astype(np.int64),
                "template_faces": template_f.astype(np.int64),
            },
            ckpt_last,
        )
        is_best = 0
        if val_loss < best_val:
            best_val = val_loss
            is_best = 1
            ckpt_best = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "template_vertices": template_v.astype(np.float32),
                    "template_edges": template_e.astype(np.int64),
                    "template_faces": template_f.astype(np.int64),
                },
                ckpt_best,
            )
            print(f"Saved new best model to {ckpt_best}")

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([epoch, f"{train_total:.6f}", f"{val_loss:.6f}", f"{best_val:.6f}", is_best])


if __name__ == "__main__":
    main()

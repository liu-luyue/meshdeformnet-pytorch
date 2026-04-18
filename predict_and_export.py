from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch

from meshdeformnet_lite.dataset import build_datasets
from meshdeformnet_lite.dataset import NPZMeshDataset
from meshdeformnet_lite.model import MeshDeformNetLite
from meshdeformnet_lite.mesh_utils import adjacency_from_edges
from meshdeformnet_lite.mesh_utils import create_template
from meshdeformnet_lite.mesh_utils import create_template_from_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Predict and export MeshDeformNet Lite outputs")
    p.add_argument("--ckpt", type=str, default="runs/lite/best.pt")
    p.add_argument("--data_mode", type=str, default="dummy", choices=["dummy", "npz"])
    p.add_argument("--train_dir", type=str, default="")
    p.add_argument("--val_dir", type=str, default="")
    p.add_argument("--npz_dir", type=str, default="", help="Optional direct NPZ folder for inference")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_vertices", type=int, default=512)
    p.add_argument("--k_neighbors", type=int, default=6)
    p.add_argument("--template_mesh", type=str, default="")
    p.add_argument("--faces_npy", type=str, default="")
    p.add_argument("--out_dir", type=str, default="runs/lite/exports")
    p.add_argument("--prefix", type=str, default="sample")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def write_pointcloud_ply(path: str, vertices: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def write_obj_wireframe(path: str, vertices: np.ndarray, edges: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for e in edges:
            # OBJ uses 1-based indices
            f.write(f"l {int(e[0]) + 1} {int(e[1]) + 1}\n")


def write_obj_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def _points_ascii(vertices: np.ndarray) -> str:
    return " ".join(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices)


def _connectivity_ascii(cells: np.ndarray) -> str:
    return " ".join(str(int(i)) for i in cells.reshape(-1))


def _offsets_ascii(num_cells: int, verts_per_cell: int) -> str:
    offsets = np.arange(1, num_cells + 1, dtype=np.int64) * int(verts_per_cell)
    return " ".join(str(int(x)) for x in offsets)


def write_vtp_wireframe(path: str, vertices: np.ndarray, edges: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(
            f'    <Piece NumberOfPoints="{vertices.shape[0]}" NumberOfVerts="0" '
            f'NumberOfLines="{edges.shape[0]}" NumberOfStrips="0" NumberOfPolys="0">\n'
        )
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {_points_ascii(vertices)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Lines>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(f"          {_connectivity_ascii(edges)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(f"          {_offsets_ascii(edges.shape[0], 2)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Lines>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def write_vtp_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(
            f'    <Piece NumberOfPoints="{vertices.shape[0]}" NumberOfVerts="0" '
            f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{faces.shape[0]}">\n'
        )
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {_points_ascii(vertices)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Polys>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(f"          {_connectivity_ascii(faces)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(f"          {_offsets_ascii(faces.shape[0], 3)}\n")
        f.write("        </DataArray>\n")
        f.write("      </Polys>\n")
        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def _prepare_sample(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.unsqueeze(0).to(device) for k, v in batch.items()}


def _merge_meshes(vertices_s: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # vertices_s: [S,N,3], faces: [F,3]
    s, n, _ = vertices_s.shape
    verts = vertices_s.reshape(s * n, 3)
    faces_all = []
    for i in range(s):
        faces_all.append(faces + i * n)
    return verts, np.concatenate(faces_all, axis=0)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(args.ckpt, map_location=device)

    template_v = None
    template_e = None
    template_f = None
    if "template_vertices" in ckpt and "template_edges" in ckpt:
        template_v = np.asarray(ckpt["template_vertices"], dtype=np.float32)
        template_e = np.asarray(ckpt["template_edges"], dtype=np.int64)
        template_f = np.asarray(ckpt.get("template_faces", np.zeros((0, 3))), dtype=np.int64)
        template_adj = adjacency_from_edges(template_v.shape[0], template_e)
        args.num_vertices = int(template_v.shape[0])
    elif args.template_mesh:
        template_v, template_f, template_e, template_adj = create_template_from_mesh(args.template_mesh)
        args.num_vertices = int(template_v.shape[0])
    else:
        template_v, template_e, template_adj = create_template(args.num_vertices, k=args.k_neighbors)
        template_f = np.zeros((0, 3), dtype=np.int64)

    if args.data_mode == "npz" and args.npz_dir:
        ds = NPZMeshDataset(args.npz_dir, num_vertices=args.num_vertices)
    else:
        datasets = build_datasets(
            data_mode=args.data_mode,
            image_size=args.image_size,
            num_vertices=args.num_vertices,
            template_vertices=template_v,
            train_dir=args.train_dir if args.train_dir else None,
            val_dir=args.val_dir if args.val_dir else None,
        )
        ds = datasets[args.split]
    if len(ds) == 0:
        raise RuntimeError(f"Dataset split '{args.split}' is empty.")
    idx = args.sample_idx % len(ds)
    batch = _prepare_sample(ds[idx], device=device)

    ckpt_args = ckpt.get("args", {})
    num_structures = int(ckpt_args.get("num_structures", 1))
    model = MeshDeformNetLite(
        feat_dim=128,
        hidden_dim=128,
        num_blocks=4,
        delta_scale=float(ckpt_args.get("delta_scale", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    init_vertices = torch.from_numpy(template_v).float().to(device)
    init_struct = init_vertices.unsqueeze(0).repeat(num_structures, 1, 1).unsqueeze(0)  # [1,S,N,3]
    adj = torch.from_numpy(template_adj).float().to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(batch["image"], init_struct, adj)[0].cpu().numpy()  # [S,N,3]
    gt = batch["points"][0].detach().cpu().numpy()
    if gt.ndim == 2:
        gt = gt[None, ...]

    base = f"{args.prefix}_{args.split}_{idx:03d}"
    pred_ply = os.path.join(args.out_dir, f"{base}_pred_points.ply")
    gt_ply = os.path.join(args.out_dir, f"{base}_gt_points.ply")
    pred_wire_obj = os.path.join(args.out_dir, f"{base}_pred_wire.obj")
    gt_wire_obj = os.path.join(args.out_dir, f"{base}_gt_wire.obj")
    pred_wire_vtp = os.path.join(args.out_dir, f"{base}_pred_wire.vtp")
    gt_wire_vtp = os.path.join(args.out_dir, f"{base}_gt_wire.vtp")
    pred_npy = os.path.join(args.out_dir, f"{base}_pred_vertices.npy")
    gt_npy = os.path.join(args.out_dir, f"{base}_gt_vertices.npy")

    pred_points = pred.reshape(-1, 3)
    gt_points = gt.reshape(-1, 3)
    write_pointcloud_ply(pred_ply, pred_points)
    write_pointcloud_ply(gt_ply, gt_points)
    if pred.shape[0] == 1:
        pred_wire_points = pred[0]
        gt_wire_points = gt[0]
    else:
        pred_wire_points, pred_wire_faces = _merge_meshes(pred, np.zeros((0, 3), dtype=np.int64))
        gt_wire_points, _ = _merge_meshes(gt, np.zeros((0, 3), dtype=np.int64))
    if pred.shape[0] == 1:
        write_obj_wireframe(pred_wire_obj, pred_wire_points, template_e)
        write_obj_wireframe(gt_wire_obj, gt_wire_points, template_e)
        write_vtp_wireframe(pred_wire_vtp, pred_wire_points, template_e)
        write_vtp_wireframe(gt_wire_vtp, gt_wire_points, template_e)
    else:
        e_list = []
        n = pred.shape[1]
        for i in range(pred.shape[0]):
            e_list.append(template_e + i * n)
        edges_cat = np.concatenate(e_list, axis=0)
        write_obj_wireframe(pred_wire_obj, pred_points, edges_cat)
        write_obj_wireframe(gt_wire_obj, gt_points, edges_cat)
        write_vtp_wireframe(pred_wire_vtp, pred_points, edges_cat)
        write_vtp_wireframe(gt_wire_vtp, gt_points, edges_cat)
    np.save(pred_npy, pred)
    np.save(gt_npy, gt)

    print(f"Exported point cloud: {pred_ply}")
    print(f"Exported wireframe:   {pred_wire_obj}")
    print(f"Exported wireframe:   {pred_wire_vtp}")
    print(f"Exported vertices:    {pred_npy}")

    faces = None
    if args.faces_npy:
        faces = np.load(args.faces_npy)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces_npy must be [M,3], got {faces.shape}")
    elif template_f is not None and template_f.ndim == 2 and template_f.shape[1] == 3 and template_f.shape[0] > 0:
        faces = template_f

    if faces is not None:
        pred_mesh_obj = os.path.join(args.out_dir, f"{base}_pred_mesh.obj")
        gt_mesh_obj = os.path.join(args.out_dir, f"{base}_gt_mesh.obj")
        pred_mesh_vtp = os.path.join(args.out_dir, f"{base}_pred_mesh.vtp")
        gt_mesh_vtp = os.path.join(args.out_dir, f"{base}_gt_mesh.vtp")
        if pred.shape[0] == 1:
            pred_mesh_pts, pred_mesh_faces = pred[0], faces
            gt_mesh_pts, gt_mesh_faces = gt[0], faces
        else:
            pred_mesh_pts, pred_mesh_faces = _merge_meshes(pred, faces)
            gt_mesh_pts, gt_mesh_faces = _merge_meshes(gt, faces)
        write_obj_mesh(pred_mesh_obj, pred_mesh_pts, pred_mesh_faces)
        write_obj_mesh(gt_mesh_obj, gt_mesh_pts, gt_mesh_faces)
        write_vtp_mesh(pred_mesh_vtp, pred_mesh_pts, pred_mesh_faces)
        write_vtp_mesh(gt_mesh_vtp, gt_mesh_pts, gt_mesh_faces)
        print(f"Exported triangle mesh: {pred_mesh_obj}")
        print(f"Exported triangle mesh: {pred_mesh_vtp}")


if __name__ == "__main__":
    main()

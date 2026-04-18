from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from meshdeformnet_lite.mesh_utils import load_template_mesh


CASE_RE = re.compile(r"(Case\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Convert CT/Seg NIfTI folders to MeshDeformNet-Lite NPZ format")
    p.add_argument("--data_root", type=str, default="data_needed")
    p.add_argument("--out_root", type=str, default="pytorch_lite/data_npz")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_points", type=int, default=0, help="0 means infer from template_mesh vertex count")
    p.add_argument("--template_mesh", type=str, default="data/template/sphere_coarse.vtp")
    p.add_argument("--seg_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--label_positive_threshold", type=float, default=0.5)
    p.add_argument("--max_cases_per_split", type=int, default=0, help="0 means use all")
    return p.parse_args()


def _case_id(path: str) -> str:
    m = CASE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse case id from filename: {path}")
    return m.group(1)


def _scan_nii(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.nii"))) + sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI file found in {folder}")
    return files


def _resize_3d(arr: np.ndarray, target_size: int, mode: str) -> np.ndarray:
    # arr: [D,H,W]
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    if mode == "image":
        y = F.interpolate(x, size=(target_size, target_size, target_size), mode="trilinear", align_corners=False)
    else:
        y = F.interpolate(x, size=(target_size, target_size, target_size), mode="nearest")
    return y[0, 0].cpu().numpy()


def _normalize_ct(ct: np.ndarray) -> np.ndarray:
    # robust CT normalization for quick baseline
    lo, hi = np.percentile(ct, 0.5), np.percentile(ct, 99.5)
    ct = np.clip(ct, lo, hi)
    mean, std = float(ct.mean()), float(ct.std() + 1e-6)
    ct = (ct - mean) / std
    return ct.astype(np.float32)


def _remap_seg_to_dense_labels(seg: np.ndarray) -> np.ndarray:
    # Some datasets store labels as encoded intensities (e.g., 205, 420, ...).
    # Remap non-zero unique values to dense IDs 1..K.
    out = np.zeros_like(seg, dtype=np.int32)
    vals = np.unique(seg)
    vals = vals[vals > 0]
    vals = np.sort(vals)
    for i, v in enumerate(vals):
        out[seg == v] = i + 1
    return out


def _extract_surface_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if m.size == 0:
        return m
    p = np.pad(m, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)
    c = p[1:-1, 1:-1, 1:-1]
    n_xm = p[:-2, 1:-1, 1:-1]
    n_xp = p[2:, 1:-1, 1:-1]
    n_ym = p[1:-1, :-2, 1:-1]
    n_yp = p[1:-1, 2:, 1:-1]
    n_zm = p[1:-1, 1:-1, :-2]
    n_zp = p[1:-1, 1:-1, 2:]
    boundary = c & ~(n_xm & n_xp & n_ym & n_yp & n_zm & n_zp)
    return boundary


def _sample_surface_points(mask: np.ndarray, num_points: int) -> np.ndarray:
    surf = _extract_surface_mask(mask)
    coords = np.argwhere(surf > 0)  # [K,3], order z,y,x
    if coords.shape[0] < 32:
        coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        # fallback: tiny centered cloud
        pts = np.zeros((num_points, 3), dtype=np.float32)
        pts += np.random.randn(num_points, 3).astype(np.float32) * 0.01
        return pts
    if coords.shape[0] >= num_points:
        idx = np.random.choice(coords.shape[0], size=num_points, replace=False)
    else:
        idx = np.random.choice(coords.shape[0], size=num_points, replace=True)
    coords = coords[idx].astype(np.float32)
    center = np.mean(coords, axis=0, keepdims=True)
    coords = coords - center
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
    coords = coords / scale
    return coords.astype(np.float32)


def _normalize_vectors(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def _radial_correspondence_points(mask: np.ndarray, template_dirs: np.ndarray, chunk_size: int = 256) -> tuple[np.ndarray, float]:
    coords = np.argwhere(_extract_surface_mask(mask) > 0).astype(np.float32)
    if coords.shape[0] < 64:
        coords = np.argwhere(mask > 0).astype(np.float32)
    if coords.shape[0] == 0:
        pts = np.zeros((template_dirs.shape[0], 3), dtype=np.float32)
        pts += np.random.randn(template_dirs.shape[0], 3).astype(np.float32) * 0.01
        return pts, 0.0

    center = np.mean(coords, axis=0, keepdims=True)
    vec = coords - center
    radii = np.linalg.norm(vec, axis=1, keepdims=True)
    dirs = _normalize_vectors(vec)
    r_norm = radii / np.clip(np.max(radii), 1e-6, None)

    selected = np.zeros((template_dirs.shape[0], 3), dtype=np.float32)
    conf = np.zeros((template_dirs.shape[0],), dtype=np.float32)
    for i in range(0, template_dirs.shape[0], chunk_size):
        q = template_dirs[i : i + chunk_size]  # [Q,3]
        cosine = q @ dirs.T  # [Q,K]
        score = 0.85 * cosine + 0.15 * r_norm.T
        idx = np.argmax(score, axis=1)
        selected[i : i + chunk_size] = vec[idx]
        conf[i : i + chunk_size] = cosine[np.arange(cosine.shape[0]), idx]
    scale = np.max(np.linalg.norm(selected, axis=1)) + 1e-6
    selected = selected / scale
    return selected.astype(np.float32), float(np.mean(conf))


def _pair_cases(image_dir: str, seg_dir: str) -> List[Tuple[str, str, str]]:
    image_files = _scan_nii(image_dir)
    seg_files = _scan_nii(seg_dir)
    seg_map: Dict[str, str] = {_case_id(p): p for p in seg_files}
    pairs = []
    for img in image_files:
        cid = _case_id(img)
        if cid in seg_map:
            pairs.append((cid, img, seg_map[cid]))
    if not pairs:
        raise RuntimeError(f"No paired cases found between {image_dir} and {seg_dir}")
    return pairs


def _convert_split(
    split: str,
    image_dir: str,
    seg_dir: str,
    out_dir: str,
    image_size: int,
    num_points: int,
    seg_ids: List[int],
    template_dirs: np.ndarray,
    threshold: float,
    max_cases: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pairs = _pair_cases(image_dir, seg_dir)
    if max_cases > 0:
        pairs = pairs[:max_cases]

    for cid, img_path, seg_path in tqdm(pairs, desc=f"Converting {split}", ncols=100):
        ct = np.asarray(nib.load(img_path).get_fdata(), dtype=np.float32)
        seg = np.asarray(nib.load(seg_path).get_fdata(), dtype=np.float32)

        ct = _resize_3d(ct, image_size, mode="image")
        seg = _resize_3d(seg, image_size, mode="label")
        ct = _normalize_ct(ct)
        seg_dense = _remap_seg_to_dense_labels(seg)
        point_list = []
        valid_list = []
        quality_list = []
        for sid in seg_ids:
            mask = (seg_dense == int(sid)).astype(np.uint8)
            points_i, quality_i = _radial_correspondence_points(mask, template_dirs)
            valid_i = 1.0 if np.sum(mask) > 0 else 0.0
            point_list.append(points_i)
            valid_list.append(valid_i)
            quality_list.append(quality_i)
        points = np.stack(point_list, axis=0).astype(np.float32)  # [S,N,3]
        valid = np.asarray(valid_list, dtype=np.float32)  # [S]

        out_file = os.path.join(out_dir, f"{cid}.npz")
        np.savez_compressed(
            out_file,
            image=ct[None, ...].astype(np.float32),
            points=points,
            valid=valid,
            seg_ids=np.asarray(seg_ids, dtype=np.int32),
            quality=np.asarray(quality_list, dtype=np.float32),
        )


def main() -> None:
    args = parse_args()
    if args.num_points <= 0:
        if not args.template_mesh:
            raise ValueError("num_points<=0 requires --template_mesh")
        v_tmplt, _ = load_template_mesh(args.template_mesh)
        args.num_points = int(v_tmplt.shape[0])
        print(f"Auto-set num_points from template '{args.template_mesh}': {args.num_points}")
    v_tmplt, _ = load_template_mesh(args.template_mesh)
    template_dirs = _normalize_vectors(v_tmplt.astype(np.float32))
    if template_dirs.shape[0] != args.num_points:
        raise ValueError(
            f"Template vertex count ({template_dirs.shape[0]}) must equal num_points ({args.num_points})."
        )
    split_cfg = [
        ("train", "ct_train", "ct_train_seg"),
        ("val", "ct_val", "ct_val_seg"),
        ("test", "ct_test", "ct_test_seg"),
    ]
    for split, im_sub, seg_sub in split_cfg:
        _convert_split(
            split=split,
            image_dir=os.path.join(args.data_root, im_sub),
            seg_dir=os.path.join(args.data_root, seg_sub),
            out_dir=os.path.join(args.out_root, split),
            image_size=args.image_size,
            num_points=args.num_points,
            seg_ids=args.seg_ids,
            template_dirs=template_dirs,
            threshold=args.label_positive_threshold,
            max_cases=args.max_cases_per_split,
        )
    print(f"Done. NPZ dataset written to: {os.path.abspath(args.out_root)}")


if __name__ == "__main__":
    main()

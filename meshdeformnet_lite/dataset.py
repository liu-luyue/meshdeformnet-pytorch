from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _resample_points(points: np.ndarray, target_n: int) -> np.ndarray:
    n = points.shape[0]
    if n == target_n:
        return points.astype(np.float32)
    if n > target_n:
        idx = np.random.choice(n, size=target_n, replace=False)
        return points[idx].astype(np.float32)
    idx = np.random.choice(n, size=target_n - n, replace=True)
    return np.concatenate([points, points[idx]], axis=0).astype(np.float32)


class DummyHeartDataset(Dataset):
    def __init__(self, length: int, image_size: int, template_vertices: np.ndarray):
        self.length = length
        self.image_size = image_size
        self.template_vertices = template_vertices.astype(np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.image_size
        image = np.random.randn(1, d, d, d).astype(np.float32) * 0.05
        scale = 1.0 + 0.05 * np.random.randn(1).astype(np.float32)
        shift = 0.03 * np.random.randn(1, 3).astype(np.float32)
        gt = self.template_vertices * scale + shift
        image += np.random.randn(1, d, d, d).astype(np.float32) * 0.01
        return {
            "image": torch.from_numpy(image),
            "points": torch.from_numpy(gt),
        }


class NPZMeshDataset(Dataset):
    def __init__(self, root_dir: str, num_vertices: int):
        self.files: List[str] = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {root_dir}")
        self.num_vertices = num_vertices

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        image = data["image"].astype(np.float32)  # [D,H,W] or [1,D,H,W]
        points = data["points"].astype(np.float32)  # [N,3]
        if image.ndim == 3:
            image = image[None, ...]
        if image.ndim != 4:
            raise ValueError(f"image must have shape [D,H,W] or [1,D,H,W], got {image.shape}")
        points = _resample_points(points, self.num_vertices)
        return {
            "image": torch.from_numpy(image),
            "points": torch.from_numpy(points),
        }


def build_datasets(
    data_mode: str,
    image_size: int,
    num_vertices: int,
    template_vertices: np.ndarray,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
) -> Dict[str, Dataset]:
    if data_mode == "dummy":
        return {
            "train": DummyHeartDataset(length=64, image_size=image_size, template_vertices=template_vertices),
            "val": DummyHeartDataset(length=16, image_size=image_size, template_vertices=template_vertices),
        }
    if data_mode == "npz":
        if not train_dir or not val_dir:
            raise ValueError("train_dir and val_dir are required when data_mode=npz")
        return {
            "train": NPZMeshDataset(train_dir, num_vertices=num_vertices),
            "val": NPZMeshDataset(val_dir, num_vertices=num_vertices),
        }
    raise ValueError(f"Unsupported data_mode: {data_mode}")


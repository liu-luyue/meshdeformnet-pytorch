from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def fibonacci_sphere(num_points: int, radius: float = 1.0) -> np.ndarray:
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        y = 1 - (i / float(max(1, num_points - 1))) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.asarray(points, dtype=np.float32)


def build_knn_edges(vertices: np.ndarray, k: int = 6) -> np.ndarray:
    d = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    nn = np.argsort(d, axis=1)[:, 1 : k + 1]
    edges = set()
    for i in range(vertices.shape[0]):
        for j in nn[i]:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def adjacency_from_edges(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    adj = np.zeros((num_vertices, num_vertices), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj += np.eye(num_vertices, dtype=np.float32)
    deg = np.sum(adj, axis=1, keepdims=True).clip(min=1.0)
    return adj / deg


def create_template(num_vertices: int, k: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = fibonacci_sphere(num_vertices, radius=1.0)
    edges = build_knn_edges(verts, k=k)
    adj = adjacency_from_edges(num_vertices, edges)
    return verts, edges, adj


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            i, j = (u, v) if u < v else (v, u)
            edges.add((i, j))
    return np.asarray(sorted(edges), dtype=np.int64)


def load_template_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported template mesh extension: {ext}")
    reader.SetFileName(mesh_path)
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
    polys = vtk_to_numpy(poly.GetPolys().GetData())
    if polys.size == 0:
        raise RuntimeError(f"No faces found in template mesh: {mesh_path}")
    faces = polys.reshape(-1, 4)[:, 1:].astype(np.int64)
    return verts, faces


def create_template_from_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts, faces = load_template_mesh(mesh_path)
    edges = edges_from_faces(faces)
    adj = adjacency_from_edges(verts.shape[0], edges)
    return verts, faces, edges, adj


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)

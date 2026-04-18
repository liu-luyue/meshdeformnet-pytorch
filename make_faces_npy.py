from __future__ import annotations

import argparse
import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from meshdeformnet_lite.mesh_utils import fibonacci_sphere


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate faces.npy for template vertices using VTK surface triangulation")
    p.add_argument("--num_vertices", type=int, default=128)
    p.add_argument("--out_faces", type=str, default="pytorch_lite/assets/faces_128.npy")
    p.add_argument("--out_vertices", type=str, default="pytorch_lite/assets/template_vertices_128.npy")
    p.add_argument("--tol", type=float, default=1e-4, help="Max nearest-neighbor distance for index mapping")
    return p.parse_args()


def _build_polydata(points: np.ndarray) -> vtk.vtkPolyData:
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points.astype(np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    return poly


def _triangulate_surface(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poly = _build_polydata(points)

    del3d = vtk.vtkDelaunay3D()
    del3d.SetInputData(poly)
    del3d.SetTolerance(0.001)
    del3d.Update()

    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputConnection(del3d.GetOutputPort())
    surf.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surf.GetOutputPort())
    tri.Update()
    out = tri.GetOutput()

    out_pts = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float32)
    polys_raw = vtk_to_numpy(out.GetPolys().GetData()).astype(np.int64)
    if polys_raw.size == 0:
        raise RuntimeError("No triangular faces produced by triangulation.")
    faces_out = polys_raw.reshape(-1, 4)[:, 1:]
    return out_pts, faces_out


def _map_surface_points_to_original(surface_pts: np.ndarray, original_pts: np.ndarray, tol: float) -> np.ndarray:
    # Map triangulated points back to the original vertex indices used by the model template.
    d = np.linalg.norm(surface_pts[:, None, :] - original_pts[None, :, :], axis=-1)
    idx = np.argmin(d, axis=1)
    min_dist = d[np.arange(d.shape[0]), idx]
    if np.any(min_dist > tol):
        raise RuntimeError(
            f"Point mapping failed: max min-distance={float(min_dist.max()):.6f} > tol={tol}. "
            "Try a larger tol."
        )
    if len(np.unique(idx)) != original_pts.shape[0]:
        # For sphere sampling all points should be on surface and all should map uniquely.
        # Keep running, but warn by raising clear error to avoid wrong topology.
        raise RuntimeError(
            f"Mapped unique points={len(np.unique(idx))}, expected={original_pts.shape[0]}. "
            "Triangulation did not preserve one-to-one mapping."
        )
    return idx.astype(np.int32)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_faces), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_vertices), exist_ok=True)

    v = fibonacci_sphere(args.num_vertices, radius=1.0).astype(np.float32)
    surf_pts, surf_faces = _triangulate_surface(v)
    point_map = _map_surface_points_to_original(surf_pts, v, tol=args.tol)
    faces = point_map[surf_faces]

    np.save(args.out_vertices, v.astype(np.float32))
    np.save(args.out_faces, faces.astype(np.int32))

    print(f"Saved vertices: {os.path.abspath(args.out_vertices)} shape={v.shape}")
    print(f"Saved faces:    {os.path.abspath(args.out_faces)} shape={faces.shape}")


if __name__ == "__main__":
    main()


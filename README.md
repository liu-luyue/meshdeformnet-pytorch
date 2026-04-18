# MeshDeformNet Lite (PyTorch)

This is a minimal, thesis-friendly rewrite scaffold inspired by MeshDeformNet:
- 3D image encoder
- graph-based mesh deformation decoder
- geometric losses (chamfer + edge + laplacian)

It is intentionally simplified to be easy to read, modify, and run.

## 1) Install

```bash
pip install -r pytorch_lite/requirements.txt
```

## 2) Quick run (no real dataset needed)

```bash
python pytorch_lite/train.py --data_mode dummy --epochs 2 --batch_size 2
```

This uses synthetic data and should produce:
- training logs
- `runs/lite/last.pt`
- `runs/lite/best.pt`

## 3) Use your own data (`npz`)

Prepare:
- `train_dir/*.npz`
- `val_dir/*.npz`

Each `.npz` file must contain:
- `image`: `[D,H,W]` or `[1,D,H,W]` float32 volume
- `points`: `[N,3]` float32 mesh points (N can vary; loader will resample)

Run:

```bash
python pytorch_lite/train.py \
  --data_mode npz \
  --train_dir /path/to/train_npz \
  --val_dir /path/to/val_npz \
  --image_size 64 \
  --template_mesh data/template/sphere_coarse.vtp \
  --epochs 50 \
  --batch_size 2
```

## 4) End-to-end CT (.nii) -> Mesh (.vtp)

1) Convert `data_needed` to `npz` with **surface-point** supervision:

```bash
python pytorch_lite/prepare_npz_from_nii.py \
  --data_root data_needed \
  --out_root pytorch_lite/data_npz_surface \
  --image_size 32 \
  --template_mesh data/template/sphere_coarse.vtp \
  --num_points 0
```

2) Train with fixed template topology:

```bash
python pytorch_lite/train.py \
  --data_mode npz \
  --train_dir pytorch_lite/data_npz_surface/train \
  --val_dir pytorch_lite/data_npz_surface/val \
  --image_size 32 \
  --template_mesh data/template/sphere_coarse.vtp \
  --delta_scale 0.3 \
  --epochs 20 \
  --batch_size 1 \
  --save_dir runs/lite_ct_surface
```

3) Predict and export mesh (`.vtp`) directly:

```bash
python pytorch_lite/predict_and_export.py \
  --ckpt runs/lite_ct_surface/best.pt \
  --data_mode npz \
  --npz_dir pytorch_lite/data_npz_surface/test \
  --sample_idx 0 \
  --image_size 32 \
  --out_dir runs/lite_ct_surface/exports \
  --prefix ct_surface
```

`predict_and_export.py` now reads template topology from checkpoint, so it can export
`*_pred_mesh.vtp` without manually providing `faces.npy`.

## 5) Code layout

- `pytorch_lite/train.py`: training entry
- `pytorch_lite/predict_and_export.py`: run inference and export 3D files
- `pytorch_lite/prepare_npz_from_nii.py`: convert CT/seg NIfTI to NPZ with surface points
- `pytorch_lite/meshdeformnet_lite/dataset.py`: dummy + npz dataset
- `pytorch_lite/meshdeformnet_lite/model.py`: encoder + graph deform net
- `pytorch_lite/meshdeformnet_lite/losses.py`: geometric losses
- `pytorch_lite/meshdeformnet_lite/mesh_utils.py`: template mesh + adjacency utils

## 6) Export predictions for visualization

```bash
python pytorch_lite/predict_and_export.py \
  --ckpt runs/lite/best.pt \
  --data_mode dummy \
  --split val \
  --sample_idx 0 \
  --out_dir runs/lite/exports
```

Exports include:
- `*_pred_points.ply`: predicted point cloud
- `*_pred_wire.obj`: predicted wireframe from template edges
- `*_pred_vertices.npy`: predicted vertices array

If you also have fixed template faces (`faces.npy`, shape `[M,3]`), export triangle mesh OBJ:

```bash
python pytorch_lite/predict_and_export.py \
  --ckpt runs/lite/best.pt \
  --data_mode npz \
  --train_dir /path/to/train_npz \
  --val_dir /path/to/val_npz \
  --split val \
  --sample_idx 0 \
  --faces_npy /path/to/faces.npy \
  --out_dir runs/lite/exports
```

## 7) How this maps to original MeshDeformNet

- Original `UNet encoder`: kept as a smaller `Encoder3D`
- Original graph decoder with projection blocks: reduced to residual GCN blocks
- Original TF custom chamfer op: replaced by `torch.cdist` chamfer
- Original multi-output / multi-stage mesh heads: merged to single-stage output
- Original TFRecords pipeline: replaced with simple `npz` pipeline

## 8) Recommended incremental upgrades

1. Add segmentation branch (multi-task, optional).
2. Add multi-stage deformation heads (`mesh1, mesh2, mesh3`).
3. Replace KNN template edges with real template connectivity from your mesh.
4. Add validation metrics (ASSD, HD95, Dice via voxelized mesh if needed).
5. Add experiment config (yaml) + reproducibility seed controls.

# U-HYDRA

**Uncertainty-Aware Multi-Hypothesis Depth Reasoning for Multi-Person Perspective Distortion Mesh Recovery**
(ICASSP 2027 submission)

U-HYDRA recovers 3D human meshes from close-range monocular images. Instead of
regressing a single root depth, a lightweight head models the log-depth of each
person as a Gaussian, **MHDR** places *K* depth hypotheses at the quantiles of
that distribution, **DCME** regresses one SMPL-X candidate per hypothesis, and
the candidate whose full-perspective rendering best matches the person mask is
selected. The predicted spread also gives a calibrated per-person depth
uncertainty at no extra cost.

The code is a drop-in package for the [BLADE](https://github.com/NVlabs/BLADE)
repository: it reuses BLADE's data pipelines, body models, SMPL-X → SMPL
evaluation protocol and mmcv train/test scripts, and adds the U-HYDRA model,
configs, evaluation diagnostics, ablation and cost tools. **No BLADE file is
modified.**

---

## 2. Installation

Tested with Python 3.9, PyTorch 2.0.1 + CUDA 11.8, PyTorch3D 0.7, mmcv 1.x.

```bash
# 1) BLADE environment (identical to BLADE's README)
git clone https://github.com/NVlabs/BLADE blade-main && cd blade-main
conda create -n uhydra python=3.9.19 -y && conda activate uhydra

pip install tqdm torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath numpy==1.24.4 wandb
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
pip install matplotlib==3.8.4 smplx==0.1.28 mediapipe chumpy scipy opencv-python \
    omegaconf trimesh pytest einops timm tensorboard

# mmcv (with CUDA ops) and the vendored packages
cd mmcv && MMCV_WITH_OPS=1 pip install -e . -v && cd ..
pip install -e .
cd smplx_repo && pip install -e . && cd ..
cd torch-trust-ncg && python setup.py install && cd ..
pip install numpy==1.23.1          # BLADE pins this last; ignore version warnings

# 2) U-HYDRA
unzip uhydra_code.zip
cp -r uhydra_code/uhydra ./uhydra   # must sit next to blade/ and scripts/
```

Headless servers (no display) need, before any training or testing:

```bash
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
```

U-HYDRA does **not** use BLADE's Depth-Anything-V2, AiOS or Sapiens models,
so those checkpoints are not required.

---

## 3. Pretrained weights and body models

| What                         | Where to put it                                   | Source |
|------------------------------|---------------------------------------------------|--------|
| SMPL-X (neutral, male, female)| `body_models/smplx/`                             | https://smpl-x.is.tue.mpg.de |
| SMPL (neutral + `J_regressor_h36m.npy`, `J_regressor_extra.npy`) | `body_models/smpl/` | https://smpl.is.tue.mpg.de and MMHuman3D |
| SMPL-X → SMPL transfer data  | `pretrained/transfer_data/`                       | https://github.com/vchoutas/smplx/tree/main/transfer_model |
| HRNet-W48 ImageNet weights   | `pretrained/hrnet_w48_imagenet.pth`               | mmpose / mmhuman3d model zoo (or set `HRNET_PRETRAINED`) |
| HRNet-W32 ImageNet weights   | `pretrained/hrnet_w32_imagenet.pth`               | idem |

The SMPL → SMPL-X conversion and the transfer-data files are needed only for
evaluation on the SMPL-labelled benchmarks (same protocol as BLADE).

Expected tree:

```
body_models/
├── smpl/   SMPL_NEUTRAL.pkl  J_regressor_h36m.npy  J_regressor_extra.npy  smpl_uv_decomr.npz
└── smplx/  SMPLX_NEUTRAL.npz SMPLX_MALE.npz SMPLX_FEMALE.npz
pretrained/
├── hrnet_w48_imagenet.pth  hrnet_w32_imagenet.pth
└── transfer_data/  smplx2smpl_deftrafo_setup.pkl  smplx_mask_ids.npy  smplx_to_smpl.pkl ...
```

---

## 4. Datasets

Follow BLADE's `docs/DATASETS.md` to obtain images and labels; U-HYDRA reads
exactly the same `.npz` files.

```
mmhuman_data/
├── datasets/
│   ├── pdhuman/imgs/           PDHuman (from ZOLLY)
│   ├── humman/{train,test}_images/
│   ├── h36m/S1 S5 S6 S7 S8 S9 S11/
│   ├── spec_mtp/imgs/
│   ├── pw3d/imageFiles/        3DPW
│   └── agora/                  AGORA validation images
└── preprocessed_datasets/
    ├── pdhuman_train_smplx.npz  humman_train_smplx.npz  h36m_mosh_train_transl_smplx.npz   # training (SMPL-X)
    ├── pdhuman_test_p5.npz      spec_mtp_p3.npz         humman_test_p3.npz                   # test (SMPL)
    └── pw3d_test.npz            agora_val.npz                                                # test (SMPL)
```

*Training* uses the SMPL-X-converted labels of PDHuman, HuMMan and Human3.6M
(mixing ratios 0.435 / 0.436 / 0.129 as in BLADE). BEDLAM-CC renders can be
added as a fourth source once converted to the `HumanImageDataset_SMPLX`
format (see the commented block in `configs/uhydra_base.py`).

*3DPW and AGORA* are not shipped with BLADE. Create `pw3d_test.npz` and
`agora_val.npz` with the MMHuman3D converters
(`python tools/convert_datasets.py --datasets pw3d agora ...` in MMHuman3D) and
keep the `K` or `focal_length` field; the test pipeline reads the camera
intrinsics from it.

The first epoch builds cache files under `cache/`; later runs start faster.

---

## 5. Quick check

```bash
# pure-tensor unit tests (CPU, ~2 s): quantile levels, hypothesis placement, coverage,
# selection rules, Eq. 6 minimum, Eq. 5 shapes, projection, soft IoU, gradient flow to (mu, sigma)
python -m pytest uhydra/tests -q

# build the model and run one random forward pass on the GPU (no data needed)
python - << 'EOF'
import os, torch
os.environ.update(UHYDRA_WIDTH='48', UHYDRA_MASK_BACKEND='none', MINI_BATCHSIZE='1')
from mmcv import Config
from blade.models.architectures.builder import build_architecture
cfg = Config.fromfile('uhydra/configs/uhydra_hrnet.py')
cfg.model.backbone.init_cfg = None; cfg.model.convert_to_smpl = False; cfg.model.compute_iou = False
m = build_architecture(cfg.model).cuda().eval()
img = torch.randn(2, 3, 224, 224).cuda()
K = torch.tensor([[[1000., 0, 112.], [0, 1000., 112.], [0, 0, 1.]]]).expand(2, 3, 3).cuda()
c = m.forward_candidates(img, K)
print('hypotheses (m):', c['tz'][0].tolist(), ' sigma:', c['sigma'][0].item())
print('silhouettes:', m.render_candidates(c, soft=False).shape)   # (2, 5, 224, 224)
EOF
```

---

## 6. Training

Paper setting (Sec. 3, *Implementation*): HRNet-W48, *K* = 5, σ_min = 0.05,
λ_z = 1, λ_σ = 0.01, λ_mesh = 1, λ_align = 0.1, Adam 1e-4 decayed ×0.1 at
epochs 90 and 120, 150 epochs, batch 64 on three RTX 3090.

```bash
export HRNET_PRETRAINED=pretrained/hrnet_w48_imagenet.pth
bash uhydra/tools/train.sh 48 work_dirs/uhydra_w48        # 3 GPUs x 22 images ~ batch 64
bash uhydra/tools/train.sh 32 work_dirs/uhydra_w32
```

Equivalent explicit command:

```bash
UHYDRA_WIDTH=48 MINI_BATCHSIZE=22 \
python -m torch.distributed.launch --nproc_per_node=3 scripts/train.py \
    uhydra/configs/uhydra_hrnet.py --work-dir work_dirs/uhydra_w48 --launcher pytorch --seed 0
```

Single GPU: `MINI_BATCHSIZE=64 python scripts/train.py uhydra/configs/uhydra_hrnet.py --work-dir ... --launcher none`
(reduce `MINI_BATCHSIZE` if memory is short; the schedule is by epoch so it is unaffected).

What to watch in the log (`work_dirs/.../*.log`):

| key                      | meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `loss_depth`             | Gaussian NLL on log-depth (Eq. 6), every instance                       |
| `loss_sigma`             | spread regulariser                                                      |
| `loss_mesh`, `loss_align`| Eq. 5 and 1 − sIoU, selected candidate only                             |
| `metric:tz_err_selected` | |T_z,k* − T_z,gt| in metres (should drop below `metric:tz_err_point`)  |
| `metric:sigma_mean`      | mean predicted spread in log-depth (paper ≈ 0.2)                        |
| `metric:coverage`        | fraction of GT depths inside [T_z,1, T_z,K]; nominal (K−1)/K = 0.8     |
| `metric:oracle_gap_mesh` | L_mesh(selected) − min_k L_mesh,k                                       |
| `metric:mpjpe_mm`        | pelvis-relative MPJPE of the selected candidate on SMPL-X body joints   |

Checkpoints are saved every 5 epochs; validation on SPEC-MTP runs every 5 epochs.
Resume with `--resume-from work_dirs/uhydra_w48/epoch_50.pth`.

---

## 7. Testing / evaluation

```bash
bash uhydra/tools/test.sh work_dirs/uhydra_w48/epoch_150.pth 48 work_dirs/uhydra_w48/eval
```

This evaluates SPEC-MTP, PDHuman, 3DPW and AGORA and writes, per dataset,
`<name>.log`, `<name>.pkl` (raw predictions), `results.json` (PA-MPJPE, MPJPE,
PVE, mIoU, z-error) and `<name>_diag.json` (σ̄, coverage, oracle gap, k*
histogram).

One dataset by hand:

```bash
UHYDRA_WIDTH=48 MINI_BATCHSIZE=16 UHYDRA_DIAG_DIR=work_dirs/uhydra_w48/eval/diag_pdhuman \
python scripts/test.py uhydra/configs/uhydra_hrnet.py \
    --checkpoint work_dirs/uhydra_w48/epoch_150.pth --data-name pdhuman_p5 \
    --work-dir work_dirs/uhydra_w48/eval --out work_dirs/uhydra_w48/eval/pdhuman.pkl
python uhydra/evaluation/summarize_diagnostics.py work_dirs/uhydra_w48/eval/diag_pdhuman --K 5
```

Valid `--data-name` values: `spec_mtp_p3`, `pdhuman_p5`, `humman_p3`, `pw3d`, `agora_val`.

**Inference protocol.** The selected candidate is the output; there is no
test-time optimisation. The person mask used by the selection rule is a
pseudo-label from an off-the-shelf segmenter (`UHYDRA_MASK_BACKEND=mediapipe`
by default, or `external` with pre-computed PNGs named `<sample_idx>.png` in
`UHYDRA_MASK_DIR`). Ground truth never enters the inference path; if a mask is
missing for an instance, the median hypothesis is used. The metrics follow
BLADE: predictions are converted SMPL-X → SMPL and compared with the SMPL
labels (PA-MPJPE / MPJPE on H3.6M joints, PVE on vertices, mIoU on silhouettes).

---

## 8. Ablation experiments

Fig. 3 of the paper (PDHuman, HRNet-W48). Each variant is an ordinary train or
test run driven by environment switches.

| block | variants                                   | cost                                  |
|-------|--------------------------------------------|---------------------------------------|
| (a) placement | single (K=1), fixed sweep {1.0,1.2,1.5,1.8,2.1}, learned quantiles | 2 extra training runs |
| (b) number of hypotheses | K = 1, 3, 5, 7                   | 2 extra training runs (K=1 shared with (a)) |
| (c) selection rule | random, median, image (ours), oracle | evaluation only, same checkpoint      |
| (d) depth supervision | L1, NLL without L_σ, NLL + L_σ     | 2 extra training runs                 |

```bash
python uhydra/tools/ablation.py --stage train --block a b d    # launches the training runs
python uhydra/tools/ablation.py --stage eval  --block a b c d  # evaluates every run on PDHuman
python uhydra/tools/ablation.py --collect                      # prints the table
```

Minimum viable set that still supports the paper's claims: blocks (a) and (c),
which need a single extra training run (the fixed sweep).

Manual example of one variant:

```bash
UHYDRA_PLACEMENT=fixed bash uhydra/tools/train.sh 48 work_dirs/ablation/fixed_sweep
UHYDRA_SELECTION=oracle MINI_BATCHSIZE=16 python scripts/test.py uhydra/configs/uhydra_hrnet.py \
    --checkpoint work_dirs/uhydra_w48/epoch_150.pth --data-name pdhuman_p5 \
    --work-dir work_dirs/uhydra_w48/eval_oracle --out work_dirs/uhydra_w48/eval_oracle/res.pkl
```

Coverage rises with *K* by construction (nominal (K−1)/K); report it as a
calibration check, not as a gain.

---

## 9. Computational cost

Table 3: parameters, GFLOPs, peak GPU memory and median latency over 100 runs
at batch size 1 with a single person on one GPU (20 warm-up iterations).

```bash
python uhydra/tools/cost.py --width 48 --K 5 --checkpoint work_dirs/uhydra_w48/epoch_150.pth
```

Rows: *Baseline* (single depth, no rendering), *+ DCME (K=1)* (one rendered
silhouette and the selection path), *Full U-HYDRA (K=5)*. The parameter count
is independent of *K* because the mesh head is shared; FLOPs, memory and
latency grow with *K*. The PyTorch3D rasteriser is a custom CUDA op and is not
counted by fvcore; the external segmentation model is excluded from latency.

---

## 10. In-the-wild multi-person demo

```bash
python uhydra/tools/demo_multiperson.py --checkpoint work_dirs/uhydra_w48/epoch_150.pth \
    --images demo_images/ --out results/demo [--focal 1500] [--width 48]
```

People are detected with torchvision Faster R-CNN; each person gets a square
224×224 crop whose intrinsics carry the image principal point, U-HYDRA runs
per instance, and the selected candidate is placed in the common camera frame
with its full translation T_k*. The script prints, per person, the selected
depth, σ and k*, and saves vertices, shape and translation to
`results/demo/<image>.npy`. If the focal length is unknown, f = 1.2·max(H, W) is
assumed.

---

## 11. Configuration and environment switches

Main config: `uhydra/configs/uhydra_hrnet.py` (model) and
`uhydra/configs/uhydra_base.py` (data, schedule). Any value can also be
overridden on the command line with `--cfg-options model.K=7`.

| variable              | values                                 | default    | effect                                      |
|-----------------------|----------------------------------------|------------|---------------------------------------------|
| `UHYDRA_WIDTH`        | 32 / 48                                | 48         | HRNet width                                 |
| `UHYDRA_K`            | 1, 3, 5, 7, …                          | 5          | number of hypotheses                        |
| `UHYDRA_PLACEMENT`    | learned / fixed / single               | learned    | hypothesis placement (Fig. 3a)              |
| `UHYDRA_DEPTH_SUP`    | nll / nll_nosigma / l1                 | nll        | depth supervision variant                   |
| `UHYDRA_SELECTION`    | image / oracle / random / median       | image      | inference selection rule (Fig. 3c)          |
| `UHYDRA_MASK_BACKEND` | mediapipe / external / none            | mediapipe  | pseudo-label mask source at inference       |
| `UHYDRA_MASK_DIR`     | path                                   | –          | PNG masks for the `external` backend        |
| `UHYDRA_DIAG_DIR`     | path                                   | –          | dump per-batch σ / coverage / k* diagnostics|
| `HRNET_PRETRAINED`    | path                                   | `pretrained/hrnet_w{W}_imagenet.pth` | backbone init          |
| `MINI_BATCHSIZE`      | int                                    | 22         | images per GPU                              |
| `NGPU`                | int                                    | 3          | GPUs used by `train.sh`                     |

Key model arguments (`model=dict(...)`): `sigma_min=0.05`,
`fixed_scales=(1.0,1.2,1.5,1.8,2.1)`, `lambda_z/lambda_sigma/lambda_mesh/lambda_align = 1/0.01/1/0.1`,
`mesh_loss_weights=dict(beta=1, theta=1, J=5, V=5)`, `render_size=224`,
`soft_sigma=1e-4`, `faces_per_pixel=50`, `convert_to_smpl=True`, `compute_iou=True`.

---

## 12. Paper → code map

| Paper                                                        | Code                                                             |
|--------------------------------------------------------------|------------------------------------------------------------------|
| Eq. 1 (μ, ρ) = F_depth(F), σ = σ_min + softplus(ρ)             | `models/mhdr.py: DistributionalDepthHead`                        |
| Eq. 2 T_z,k = exp(μ + z_k σ), z_k = Φ⁻¹((k−½)/K)              | `standard_normal_quantiles`, `HypothesisProposal`                |
| L_align,k = 1 − sIoU(Ŝ_k, S)                                  | `geometry.soft_iou` / `hard_iou`, `UHYDRA.render_candidates`     |
| Eq. 3 selection (inference / training)                        | `selection.select_candidates(rule='image' / 'train')`            |
| Eq. 4 F_mesh and SMPL-X M                                     | `models/dcme.py`, `UHYDRA.smplx_forward`                         |
| full-perspective projection u = f(x+T_x)/(z+T_z)+c_x           | `geometry.perspective_project`, `SilhouetteRenderer`             |
| Eq. 5 L_mesh,k                                                | `losses.mesh_loss_per_candidate`                                 |
| Eq. 6 L_depth, L_σ                                            | `losses.depth_nll`, `losses.spread_regularizer`                  |
| Eq. 7 full objective                                          | `UHYDRA.forward_train`                                           |
| mask S: GT at training, pseudo-label at inference             | `UHYDRA.render_gt_mask`, `models/masks.py`                       |
| coverage, oracle gap                                          | `HypothesisProposal.coverage`, `evaluation/metrics.py`           |
| Table 3                                                       | `tools/cost.py`                                                  |
| Fig. 3                                                        | `tools/ablation.py`                                              |

Implementation notes: candidates are produced by replicating the pooled
descriptor *K* times through the shared mesh head; the head regresses
(T_x, T_y) in a crop-centred virtual camera and `UHYDRA.to_true_frame` converts
them to the true camera using the crop intrinsics; hands and face are kept at
rest, as in BLADE's protocol; the network consumes BLADE's person-centred crop
(`depthnet_img`, 224×224) with its intrinsics `K`, and the un-augmented
`*_orig` SMPL-X labels, so horizontal flipping is disabled in the training
pipeline.

---

## 13. Troubleshooting

* `KeyError: 'depthnet_img'` or `'SMPLX_betas'` — the dataset is not a
  `HumanImageDataset_SMPLX` with SMPL-X labels; check `ann_file` and the
  `*_smplx.npz` conversion.
* `AssertionError: set img_res_train = resolution` — keep `img_res_train` in
  `uhydra_base.py` equal to `model.resolution` (224).
* `no seg mask` / all `k_star` equal to the median — MediaPipe found no person
  in the crop; use `UHYDRA_MASK_BACKEND=external` with masks from a stronger
  segmenter (Sapiens-Seg, SAM).
* EGL / XRender errors on headless servers — export `PYGLET_HEADLESS=True
  PYOPENGL_PLATFORM=egl` (see BLADE's README for the `libEGL.so` symlink).
* Evaluation is slow — the SMPL-X → SMPL conversion is an optimisation
  (BLADE's protocol); set `--cfg-options model.convert_to_smpl=False` for quick
  SMPL-X-only sanity runs (metrics are then not comparable to the paper).
* CUDA out of memory in training — lower `MINI_BATCHSIZE`, or set
  `model.faces_per_pixel=20` to reduce the soft rasteriser's memory.

---

## 14. Citation

```bibtex
@inproceedings{ullah2027uhydra,
  title     = {U-HYDRA: Uncertainty-Aware Multi-Hypothesis Depth Reasoning for
               Multi-Person Perspective Distortion Mesh Recovery},
  author    = {Ullah, Saif and Ahmad, Niaz and Wang, Guanghui and Lee, Youngmoon},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

U-HYDRA builds on BLADE (Wang et al., CVPR 2025), MMHuman3D, ZOLLY/PDHuman,
SMPL/SMPL-X and PyTorch3D; please cite them as well when using this code.

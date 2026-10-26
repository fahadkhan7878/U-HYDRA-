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

This repository contains the complete U-HYDRA implementation: the model
(`uhydra/`), the data pipelines, body models and mmcv train/test scripts it
runs on, the evaluation protocol, and the ablation, cost and demo tools.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Pretrained weights and body models](#3-pretrained-weights-and-body-models)
4. [Datasets](#4-datasets)
5. [Quick check](#5-quick-check)
6. [Training](#6-training)
7. [Testing / evaluation](#7-testing--evaluation)
8. [Ablation experiments](#8-ablation-experiments)
9. [Computational cost](#9-computational-cost)
10. [In-the-wild multi-person demo](#10-in-the-wild-multi-person-demo)
11. [Configuration and environment switches](#11-configuration-and-environment-switches)
12. [Paper → code map](#12-paper--code-map)
13. [Troubleshooting](#13-troubleshooting)
14. [Citation](#14-citation)
15. [Acknowledgements](#15-acknowledgements)

---

## 1. Repository layout

```
U-HYDRA/
├── scripts/train.py       mmcv training entry point, driven by the U-HYDRA configs
├── scripts/test.py        mmcv evaluation entry point
├── body_models/           SMPL / SMPL-X (manual download, Sec. 3)
├── pretrained/            HRNet ImageNet weights, SMPL-X transfer data
├── mmhuman_data/          datasets and preprocessed labels (Sec. 4)
├── (support libraries)    data pipelines, body models, renderers, mmcv / mmhuman3d
└── uhydra/                <-- the U-HYDRA model and experiment tools
    ├── models/
    │   ├── mhdr.py            MHDR: distributional depth head + quantile hypotheses (Eqs. 1-2)
    │   ├── dcme.py            DCME: depth-conditioned SMPL-X regressor (Eq. 4)
    │   ├── geometry.py        full-perspective projection, soft silhouettes, soft IoU
    │   ├── selection.py       candidate selection (Eq. 3) + ablation rules
    │   ├── losses.py          training objective (Eqs. 5-7)
    │   ├── masks.py           person mask: GT at training, pseudo-label at inference
    │   └── architecture.py    UHYDRA (mmcv architecture): train_step / forward_test
    ├── configs/
    │   ├── uhydra_base.py     pipelines, datasets, schedule
    │   └── uhydra_hrnet.py    HRNet-W32 / W48 model config
    ├── evaluation/
    │   ├── metrics.py         PA-MPJPE, MPJPE, PVE, mIoU, coverage, oracle gap
    │   └── summarize_diagnostics.py
    ├── tools/
    │   ├── train.sh           train (3 GPUs)
    │   ├── test.sh            evaluate on all benchmarks
    │   ├── ablation.py        Fig. 3 ablation runner
    │   ├── cost.py            Table 3: params / GFLOPs / memory / latency
    │   └── demo_multiperson.py
    ├── api.py                 build_model / load_checkpoint helpers
    ├── tests/test_core.py     unit tests (no GPU needed)
    └── README.md
```

---

## 2. Installation

Tested with Python 3.9, PyTorch 2.0.1 + CUDA 11.8, PyTorch3D 0.7, mmcv 1.x.

```bash
# 1) environment
git clone <U-HYDRA repository URL> U-HYDRA && cd U-HYDRA
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
pip install numpy==1.23.1          # pinned last on purpose; ignore version warnings

# 2) U-HYDRA package (skip if it is already in the repository)
unzip uhydra_code.zip
cp -r uhydra_code/uhydra ./uhydra   # must sit in the repository root next to scripts/
```

Headless servers (no display) need, before any training or testing:

```bash
export PYGLET_HEADLESS=True
export PYOPENGL_PLATFORM=egl
```

U-HYDRA does **not** use Depth-Anything-V2, AiOS or Sapiens, so none of those
checkpoints are required.

---

## 3. Pretrained weights and body models

| What                         | Where to put it                                   | Source |
|------------------------------|---------------------------------------------------|--------|
| SMPL-X (neutral, male, female)| `body_models/smplx/`                             | https://smpl-x.is.tue.mpg.de |
| SMPL (neutral + `J_regressor_h36m.npy`, `J_regressor_extra.npy`) | `body_models/smpl/` | https://smpl.is.tue.mpg.de and MMHuman3D |
| SMPL-X → SMPL transfer data  | `pretrained/transfer_data/`                       | https://github.com/vchoutas/smplx/tree/main/transfer_model |
| HRNet-W48 ImageNet weights   | `pretrained/hrnet_w48_imagenet.pth`               | mmpose / mmhuman3d model zoo (or set `HRNET_PRETRAINED`) |
| HRNet-W32 ImageNet weights   | `pretrained/hrnet_w32_imagenet.pth`               | idem |

The SMPL-X → SMPL transfer-data files are needed only for evaluation on the
SMPL-labelled benchmarks.

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

Follow `docs/DATASETS.md` to obtain the images and the preprocessed `.npz`
labels.

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
(mixing ratios 0.435 / 0.436 / 0.129). BEDLAM-CC renders can be
added as a fourth source once converted to the `HumanImageDataset_SMPLX`
format (see the commented block in `configs/uhydra_base.py`).

*3DPW and AGORA*: create `pw3d_test.npz` and
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
import torch
from uhydra.api import build_model
m = build_model('uhydra/configs/uhydra_hrnet.py', width=48)      # random init, GPU
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
missing for an instance, the median hypothesis is used. For the metrics,
predictions are converted SMPL-X → SMPL and compared with the SMPL
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

<img src="Multi-view.png" alt="Project Logo" width="800" />

<img src="Examples from 3DPW.png" alt="Project Logo" width="800" />

<img src="Multi-person human mesh recovery.png" alt="Project Logo" width="800" />

<img src="distorted Images 2.png" alt="Project Logo" width="800" />

<img src="distorted Images.png" alt="Project Logo" width="800" />





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
rest; the network consumes the person-centred crop (`depthnet_img`, 224×224)
with its intrinsics `K`, and the un-augmented `*_orig` SMPL-X labels, so
horizontal flipping is disabled in the training pipeline.

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
  PYOPENGL_PLATFORM=egl`; if `libEGL.so` is missing, link it:
  `ln -s $CONDA_PREFIX/lib/libEGL.so.1 $CONDA_PREFIX/lib/libEGL.so`.
* Evaluation is slow — the SMPL-X → SMPL conversion is an optimisation;
  set `--cfg-options model.convert_to_smpl=False` for quick
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

## 15. Acknowledgements

The data pipelines, body-model wrappers, renderer utilities and the SMPL-X →
SMPL evaluation protocol are adapted from
[BLADE](https://github.com/NVlabs/BLADE) (Wang et al., CVPR 2025), which in
turn builds on [MMHuman3D](https://github.com/open-mmlab/mmhuman3d). We also
thank the authors of [ZOLLY / PDHuman](https://github.com/SMPLCap/Zolly),
[SPEC](https://spec.is.tue.mpg.de/), [HuMMan](https://caizhongang.com/projects/HuMMan/),
[BEDLAM](https://bedlam.is.tue.mpg.de/), [SMPL / SMPL-X](https://smpl-x.is.tue.mpg.de/)
and [PyTorch3D](https://pytorch3d.org/). Please cite these works as well when
using this code.

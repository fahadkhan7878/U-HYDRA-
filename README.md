# U-HYDRA on top of BLADE

Code for **U-HYDRA: Uncertainty-Aware Multi-Hypothesis Depth Reasoning for
Multi-Person Perspective Distortion Mesh Recovery** (ICASSP 2027 submission),
implemented as a drop-in package for the BLADE repository.

```
blade-main/
├── blade/            (unchanged)
├── mmhuman3d/        (unchanged)
├── scripts/          train.py / test.py (unchanged, driven by uhydra configs)
└── uhydra/           <-- this package
    ├── models/
    │   ├── mhdr.py           Sec. 2.1  distributional depth head + quantile hypotheses (Eqs. 1-2)
    │   ├── dcme.py           Sec. 2.2  depth-conditioned SMPL-X regressor (Eq. 4)
    │   ├── geometry.py       Sec. 2.2  full-perspective projection, soft silhouettes, sIoU
    │   ├── selection.py      Sec. 2.1  candidate selection (Eq. 3) + ablation rules
    │   ├── losses.py         Sec. 2.3  Eqs. 5-7
    │   ├── masks.py          person mask S^(i): GT at training, pseudo-label at inference
    │   └── architecture.py   the UHYDRA mmcv architecture (train_step / forward_test)
    ├── configs/
    │   ├── uhydra_base.py    pipelines, datasets, schedule (Sec. 3 "Training and Evaluation Setup")
    │   └── uhydra_hrnet.py   HRNet-W32/W48 model config ("Implementation")
    ├── evaluation/
    │   ├── metrics.py        PA-MPJPE, MPJPE, PVE, mIoU, coverage, oracle gap
    │   └── summarize_diagnostics.py
    ├── tools/
    │   ├── train.sh / test.sh
    │   ├── ablation.py       Sec. 3.3 / Fig. 3 (a)-(d)
    │   ├── cost.py           Table 3
    │   └── demo_multiperson.py
    └── tests/test_core.py
```

## What changes relative to BLADE

| BLADE                                                     | U-HYDRA                                                                 |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
| DepthNet (DAv2-ViT-L + transformer decoder) -> one `pred_z` | HRNet -> GAP -> 2-layer MLP -> Gaussian over log-depth (mu, sigma)     |
| PoseNet = AiOS DETR + ControlNet conditioned on `pred_z`  | 2-layer MLP conditioned on `log T_z,k`, run K times with shared weights |
| Camera solved per image by test-time optimisation (fTxTy) | `T_k = [T_x,k, T_y,k, T_z,k]` regressed per candidate; no optimisation  |
| One mesh per image                                         | K candidates, rendered under full perspective, selected by silhouette IoU |
| Two-stage training, L1 on `T_z`                            | End-to-end, Eq. 7: Gaussian NLL + spread reg. + winner-takes-all mesh/align |

Everything else — dataset classes, preprocessing, SMPL-X -> SMPL conversion for
evaluation, `scripts/train.py` / `scripts/test.py`, metrics in
`HumanImageDataset_SMPLX.evaluate` — is reused untouched.

## Paper -> code map

| Paper                                        | Code                                                                   |
|----------------------------------------------|------------------------------------------------------------------------|
| Eq. 1  `(mu, rho) = F_depth(F)`, `sigma = sigma_min + softplus(rho)` | `DistributionalDepthHead.forward`                     |
| Eq. 2  `T_z,k = exp(mu + z_k sigma)`, `z_k = Phi^-1((k-1/2)/K)`       | `standard_normal_quantiles`, `HypothesisProposal.forward` |
| `L_align,k = 1 - sIoU(S_hat_k, S)`           | `geometry.soft_iou` (train) / `hard_iou` (inference), `architecture.render_candidates` |
| Eq. 3  selection (inference / training)       | `selection.select_candidates(rule='image' / 'train')`                  |
| Eq. 4  `F_mesh`, SMPL-X `M`                   | `DepthConditionedMeshHead`, `UHYDRA.smplx_forward`                     |
| full-perspective projection                  | `geometry.perspective_project`, `SilhouetteRenderer` (OpenCV-convention cameras) |
| Eq. 5  `L_mesh,k`                             | `losses.mesh_loss_per_candidate`                                       |
| Eq. 6  `L_depth`, `L_sigma`                   | `losses.depth_nll`, `losses.spread_regularizer`                        |
| Eq. 7                                         | `UHYDRA.forward_train` (`lambda_z=1, lambda_sigma=0.01, lambda_mesh=1, lambda_align=0.1`) |
| `sigma_min = 0.05`, `K = 5`, quantiles `{-1.28,-0.52,0,0.52,1.28}` | `configs/uhydra_hrnet.py`                               |
| Adam 1e-4, x0.1 @ 90/120, 150 ep, batch 64   | `configs/uhydra_base.py`                                               |
| coverage, oracle gap                          | `HypothesisProposal.coverage`, `evaluation/metrics.py`, `forward_test` diagnostics |

### Mask `S^(i)` (Sec. 2.1)
* training: rendered from the ground-truth SMPL-X body of the instance (`render_gt_mask`);
* inference: **always** a pseudo-label from an off-the-shelf segmenter
  (`UHYDRA_MASK_BACKEND=mediapipe`, or `external` with pre-computed PNGs in
  `UHYDRA_MASK_DIR`). No ground truth enters the inference path. If a mask is
  missing for an instance, the median hypothesis is used.

### Instance features for multi-person images
Training/benchmark data are per-person crops produced by BLADE's dataset
class (`depthnet_img`, 224x224, with crop intrinsics `K`). For in-the-wild
multi-person images `tools/demo_multiperson.py` detects people, builds one crop
per person with the principal point carried into `K`, and runs the model per
instance; selected candidates are placed in the common camera frame via `T_k*`.
The mesh head regresses `(T_x, T_y)` in a crop-centred virtual camera and
`UHYDRA.to_true_frame` converts them to the true camera using `K`.

## Setup

1. Install BLADE as in its README (PyTorch 2.0.1 + PyTorch3D + mmcv + smplx).
2. Copy `uhydra/` into the repo root.
3. Download an ImageNet-pretrained HRNet (`hrnet_w32_imagenet.pth` /
   `hrnet_w48_imagenet.pth`, e.g. from mmpose/mmhuman3d model zoos) to
   `pretrained/`, or set `HRNET_PRETRAINED=/path/to.pth`.
4. Prepare the SMPL-X training labels and the SMPL test labels exactly as for
   BLADE (`mmhuman_data/preprocessed_datasets/*.npz`). For 3DPW and AGORA,
   create `pw3d_test.npz` / `agora_val.npz` with the MMHuman3D converters and
   keep the `K`/`focal_length` fields.
5. `python -m pytest uhydra/tests -q` (pure-tensor checks).

## Running the experiments (Sec. 3)

```bash
# Table 1-2: train W48 and W32, evaluate on SPEC-MTP, PDHuman, 3DPW, AGORA
bash uhydra/tools/train.sh 48 work_dirs/uhydra_w48
bash uhydra/tools/train.sh 32 work_dirs/uhydra_w32
bash uhydra/tools/test.sh work_dirs/uhydra_w48/epoch_150.pth 48

# Table 3: parameters, GFLOPs, memory, latency (batch 1, one RTX 3090)
python uhydra/tools/cost.py --width 48 --K 5 --checkpoint work_dirs/uhydra_w48/epoch_150.pth

# Fig. 3 ablation on PDHuman (W48)
python uhydra/tools/ablation.py --stage train --block a b d     # 6 training runs
python uhydra/tools/ablation.py --stage eval  --block a b c d   # (c) re-uses the ours_K5 checkpoint
python uhydra/tools/ablation.py --collect
```

Minimum-cost ablation that still supports the paper's claims: blocks (a)
and (c) need only **one extra training run** (the fixed sweep); the single-
hypothesis row is the `single` placement, and the four selection rules are
evaluation-only on the full model.

### Environment switches
| variable               | values                                     | effect                              |
|------------------------|--------------------------------------------|-------------------------------------|
| `UHYDRA_WIDTH`         | 32 / 48                                    | HRNet width                         |
| `UHYDRA_K`             | 1,3,5,7,...                                | number of hypotheses                |
| `UHYDRA_PLACEMENT`     | learned / fixed / single                   | Fig. 3(a)                           |
| `UHYDRA_DEPTH_SUP`     | nll / nll_nosigma / l1                     | depth-supervision ablation          |
| `UHYDRA_SELECTION`     | image / oracle / random / median           | selection-rule ablation (eval only) |
| `UHYDRA_MASK_BACKEND`  | mediapipe / external / none                | pseudo-label source at inference    |
| `UHYDRA_DIAG_DIR`      | path                                       | dump coverage / sigma / k* per batch|

## Notes and assumptions
* `K` rows are produced by replicating the pooled descriptor K times, so the
  parameter count is independent of K and only FLOPs / memory / latency grow
  (Table 3). The soft rasteriser is a custom CUDA op and is not counted by
  fvcore; report it as such.
* `mesh_loss_weights` (lambda_beta, lambda_theta, lambda_J, lambda_V) default to
  (1, 1, 5, 5), matching BLADE's joint/vertex weighting.
* The `oracle` rule and the `coverage` diagnostic use ground truth and are for
  analysis only; they are never used by the inference rule.
* Hands and face are kept at rest (SMPL-X hand/expression parameters are not
  regressed), as in BLADE's evaluation protocol.

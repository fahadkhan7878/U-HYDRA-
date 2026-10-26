"""Evaluation metrics (paper Sec. 3): PA-MPJPE, MPJPE, PVE, mIoU, and the two
multi-hypothesis diagnostics, coverage and oracle gap.

BLADE's dataset ``evaluate()`` already computes PA-MPJPE / MPJPE / PVE from the
``keypoints_3d`` / ``vertices`` the model returns (through the SMPL-X -> SMPL
conversion).  The functions here are the reference implementations used by the
ablation / diagnostics scripts, written in numpy so that they can be run on
saved predictions without the training environment.
"""
import numpy as np


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity Procrustes alignment of pred (J,3) onto gt (J,3)."""
    mu_p, mu_g = pred.mean(0, keepdims=True), gt.mean(0, keepdims=True)
    p, g = pred - mu_p, gt - mu_g
    var_p = (p ** 2).sum()
    H = p.T @ g
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        S[-1] *= -1
        R = Vt.T @ U.T
    s = S.sum() / var_p
    return s * (p @ R.T) + mu_g


def mpjpe(pred: np.ndarray, gt: np.ndarray, root: int = 0) -> np.ndarray:
    """Mean per-joint position error after pelvis alignment, (N,) in mm."""
    pred = pred - pred[:, root:root + 1]
    gt = gt - gt[:, root:root + 1]
    return np.linalg.norm(pred - gt, axis=-1).mean(-1) * 1000.0


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.array([np.linalg.norm(procrustes_align(p, g) - g, axis=-1).mean()
                    for p, g in zip(pred, gt)])
    return out * 1000.0


def pve(pred_verts: np.ndarray, gt_verts: np.ndarray, pred_pelvis, gt_pelvis) -> np.ndarray:
    pred = pred_verts - pred_pelvis[:, None]
    gt = gt_verts - gt_pelvis[:, None]
    return np.linalg.norm(pred - gt, axis=-1).mean(-1) * 1000.0


def miou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    p, g = pred_mask.reshape(len(pred_mask), -1) > 0.5, gt_mask.reshape(len(gt_mask), -1) > 0.5
    return (p & g).sum(1) / np.maximum((p | g).sum(1), 1) * 100.0


def coverage(tz_hyp: np.ndarray, tz_gt: np.ndarray) -> float:
    """Fraction of instances whose GT root depth lies inside [min_k T_z,k, max_k T_z,k].
    Nominal value for learned quantile placement: (K-1)/K  (80 % at K = 5)."""
    lo, hi = tz_hyp.min(1), tz_hyp.max(1)
    return float(((tz_gt >= lo) & (tz_gt <= hi)).mean() * 100.0)


def oracle_gap(err_selected: np.ndarray, err_per_candidate: np.ndarray) -> float:
    """Mean (error of the selected candidate - error of the best-of-K candidate).
    Separates hypothesis *coverage* (best-of-K) from hypothesis *selection*."""
    return float((err_selected - err_per_candidate.min(1)).mean())


def nominal_coverage(K: int) -> float:
    return (K - 1) / K * 100.0

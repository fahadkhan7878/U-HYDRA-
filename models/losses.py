"""Training objective of U-HYDRA  --  paper Sec. 2.3.

    L = lambda_z L_depth + lambda_sigma L_sigma + lambda_mesh L_mesh + lambda_align L_align   (Eq. 7)

    L_depth : Gaussian NLL of log T_z,gt under N(mu, sigma^2)              (Eq. 6)
    L_sigma : spread regulariser  mean_i sigma^(i)
    L_mesh  : per-candidate reconstruction loss (Eq. 5), evaluated at k*
    L_align : per-candidate silhouette alignment 1 - sIoU, evaluated at k*

Ablation variants of the depth supervision (Sec. 3.3 / Fig. 3d of the
long version): 'nll' (paper), 'nll_nosigma' (no L_sigma), 'l1' (plain L1 on
log-depth, no learning signal for sigma).
"""
import torch
import torch.nn.functional as F

from .geometry import geodesic_angle


def depth_nll(mu: torch.Tensor, sigma: torch.Tensor, tz_gt: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    """Eq. 6, averaged over valid instances.  mu, sigma, tz_gt, valid: (B,)."""
    log_gt = torch.log(tz_gt.clamp(min=1e-3))
    nll = (log_gt - mu) ** 2 / (2.0 * sigma ** 2) + torch.log(sigma)
    return (nll * valid).sum() / valid.sum().clamp(min=1.0)


def depth_l1(mu: torch.Tensor, tz_gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Ablation: L1 on log-depth (sigma receives no gradient)."""
    log_gt = torch.log(tz_gt.clamp(min=1e-3))
    return ((log_gt - mu).abs() * valid).sum() / valid.sum().clamp(min=1.0)


def spread_regularizer(sigma: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L_sigma = (1/m) sum_i sigma^(i)."""
    return (sigma * valid).sum() / valid.sum().clamp(min=1.0)


def mesh_loss_per_candidate(pred: dict, gt: dict, weights: dict) -> torch.Tensor:
    """Eq. 5 for every (instance, candidate).  Returns (B, K).

    pred: betas (B,K,10), global_orient (B,K,3,3), body_pose (B,K,21,3,3),
          joints (B,K,J,3) pelvis-relative, verts (B,K,V,3) pelvis-relative
    gt:   betas (B,10), global_orient (B,3,3), body_pose (B,21,3,3),
          joints (B,J,3), verts (B,V,3)
    """
    B, K = pred['betas'].shape[:2]
    l_beta = (pred['betas'] - gt['betas'][:, None]).abs().mean(-1)                      # (B,K)

    R_pred = torch.cat([pred['global_orient'][:, :, None], pred['body_pose']], dim=2)   # (B,K,22,3,3)
    R_gt = torch.cat([gt['global_orient'][:, None], gt['body_pose']], dim=1)            # (B,22,3,3)
    R_gt = R_gt[:, None].expand(B, K, -1, -1, -1)
    l_theta = geodesic_angle(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3)).view(B, K, -1).mean(-1)

    l_J = (pred['joints'] - gt['joints'][:, None]).abs().mean((-1, -2))                 # (B,K)
    l_V = (pred['verts'] - gt['verts'][:, None]).abs().mean((-1, -2))                   # (B,K)

    return (weights['beta'] * l_beta + weights['theta'] * l_theta
            + weights['J'] * l_J + weights['V'] * l_V)


def gather_winner(x: torch.Tensor, k_star: torch.Tensor) -> torch.Tensor:
    """x: (B, K, ...) -> (B, ...) selecting index k_star (B,)."""
    idx = k_star.view(-1, *([1] * (x.dim() - 1))).expand(-1, 1, *x.shape[2:])
    return x.gather(1, idx).squeeze(1)

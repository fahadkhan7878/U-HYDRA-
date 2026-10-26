"""Multi-Hypothesis Depth Reasoning (MHDR)  --  paper Sec. 2.1.

Distributional root-depth prediction (Eq. 1):
    log T_z^(i) ~ N(mu^(i), sigma^(i)^2),   (mu, rho) = F_depth(F^(i)),
    sigma = sigma_min + softplus(rho).

Uncertainty-aware hypothesis proposal (Eq. 2):
    q_k = (k - 1/2) / K,   z_k = Phi^{-1}(q_k),
    T_{z,k}^(i) = exp(mu^(i) + z_k sigma^(i)),   k = 1..K.

Two ablation variants of the proposal are provided (Sec. 3.3, Fig. 3a):
    * 'single' : K = 1, the point estimate exp(mu) only.
    * 'fixed'  : a hand-chosen metric scale set {s_k} applied to exp(mu),
                 T_{z,k} = s_k * exp(mu)  (the "fixed sweep" baseline).
"""
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def standard_normal_quantiles(K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """z_k = Phi^{-1}((k - 1/2) / K), k = 1..K  (Eq. 2).

    Phi^{-1} is the inverse of the standard-normal CDF, evaluated with
    ``erfinv``:  Phi^{-1}(q) = sqrt(2) * erfinv(2q - 1).
    For K = 5 this gives {-1.2816, -0.5244, 0, 0.5244, 1.2816}.
    """
    k = torch.arange(1, K + 1, device=device, dtype=dtype)
    q = (k - 0.5) / K
    return math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


class DistributionalDepthHead(nn.Module):
    """F_depth in Eq. 1: GAP -> 2-layer MLP -> (mu, rho); sigma = sigma_min + softplus(rho)."""

    def __init__(self, in_channels: int = 2048, hidden: int = 256,
                 sigma_min: float = 0.05, mu_init: float = math.log(2.0)):
        super().__init__()
        self.sigma_min = sigma_min
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        # initialise mu around log(2 m) and sigma around 0.3 so that the first
        # hypotheses are sensible before any training signal arrives.
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor([mu_init, math.log(math.exp(0.25) - 1.0)]))

    def forward(self, f: torch.Tensor):
        """f: (B, C) pooled descriptor -> mu (B,), sigma (B,)."""
        h = F.relu(self.fc1(f))
        out = self.fc2(h)
        mu, rho = out[:, 0], out[:, 1]
        sigma = self.sigma_min + F.softplus(rho)
        return mu, sigma


class HypothesisProposal(nn.Module):
    """Places K depth hypotheses from (mu, sigma)  (Eq. 2) or its ablation variants."""

    def __init__(self, K: int = 5, placement: str = 'learned',
                 fixed_scales: Optional[Sequence[float]] = (1.0, 1.2, 1.5, 1.8, 2.1)):
        super().__init__()
        assert placement in ('learned', 'fixed', 'single'), placement
        self.placement = placement
        if placement == 'single':
            K = 1
        if placement == 'fixed':
            K = len(fixed_scales)
            self.register_buffer('fixed_scales', torch.tensor(fixed_scales, dtype=torch.float32))
        self.K = K
        self.register_buffer('z', standard_normal_quantiles(K))  # (K,)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Returns T_z of shape (B, K)  (metric depth, strictly positive)."""
        if self.placement == 'learned':
            log_tz = mu[:, None] + self.z[None, :] * sigma[:, None]        # (B, K)
            return torch.exp(log_tz)
        point = torch.exp(mu)[:, None]                                       # (B, 1)
        if self.placement == 'single':
            return point
        return point * self.fixed_scales[None, :]                            # fixed sweep

    @torch.no_grad()
    def coverage(self, tz: torch.Tensor, tz_gt: torch.Tensor) -> torch.Tensor:
        """Coverage diagnostic: 1 if T_z,gt lies within [min_k T_z,k, max_k T_z,k]."""
        lo, hi = tz.min(dim=1).values, tz.max(dim=1).values
        return ((tz_gt >= lo) & (tz_gt <= hi)).float()


class MHDR(nn.Module):
    """Depth head + hypothesis proposal, bundled for the architecture."""

    def __init__(self, in_channels=2048, hidden=256, sigma_min=0.05, K=5,
                 placement='learned', fixed_scales=(1.0, 1.2, 1.5, 1.8, 2.1)):
        super().__init__()
        self.depth_head = DistributionalDepthHead(in_channels, hidden, sigma_min)
        self.proposal = HypothesisProposal(K, placement, fixed_scales)

    @property
    def K(self):
        return self.proposal.K

    def forward(self, f: torch.Tensor):
        mu, sigma = self.depth_head(f)
        tz = self.proposal(mu, sigma)
        return dict(mu=mu, sigma=sigma, tz_hat=torch.exp(mu), tz=tz)

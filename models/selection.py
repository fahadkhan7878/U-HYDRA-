"""Candidate selection  --  paper Sec. 2.1 ("Candidate selection"), Eq. 3.

    k* = argmin_k  L_align,k                                   (inference)
    k* = argmin_k  L_mesh,k + lambda_align L_align,k           (training, winner-takes-all)

Ablation rules for Sec. 3.3 (selection-rule study, evaluation only):
    'image'  : the paper's inference rule  (argmin L_align)
    'oracle' : argmin_k L_mesh,k using ground truth  -> upper bound / oracle gap
    'random' : a uniformly random candidate
    'median' : the middle hypothesis (k = ceil(K/2)), i.e. no selection
The argmin is discrete, but the per-candidate losses are differentiable, so
backpropagation through the selected candidate is well-posed and needs no
Gumbel-softmax or straight-through relaxation.
"""
import math
import torch


@torch.no_grad()
def select_candidates(l_align: torch.Tensor, l_mesh: torch.Tensor = None,
                      lambda_align: float = 0.1, rule: str = 'image',
                      has_mask: torch.Tensor = None) -> torch.Tensor:
    """Returns k* of shape (B,).

    l_align : (B, K) alignment loss per candidate (may be NaN where no mask)
    l_mesh  : (B, K) reconstruction loss per candidate (training / oracle only)
    has_mask: (B,) bool; instances without a mask fall back to the median hypothesis.
    """
    B, K = l_align.shape
    median = torch.full((B,), (K - 1) // 2, device=l_align.device, dtype=torch.long)

    if rule == 'median':
        return median
    if rule == 'random':
        return torch.randint(0, K, (B,), device=l_align.device)
    if rule == 'oracle':
        assert l_mesh is not None, 'oracle selection needs L_mesh (ground truth)'
        return l_mesh.argmin(dim=1)
    if rule == 'train':
        assert l_mesh is not None
        score = l_mesh + lambda_align * torch.nan_to_num(l_align, nan=0.0)
        return score.argmin(dim=1)
    if rule == 'image':
        k = torch.nan_to_num(l_align, nan=float('inf')).argmin(dim=1)
        if has_mask is not None:
            k = torch.where(has_mask, k, median)
        return k
    raise ValueError(f'unknown selection rule {rule}')

"""Light-weight checks that need only PyTorch (no PyTorch3D / SMPL-X):
quantile levels, hypothesis placement, coverage, selection rules, losses.

    python -m pytest uhydra/tests/test_core.py -q
"""
import math
import sys
import types

import torch

# geometry.py imports pytorch3d at module level; stub it when absent so that
# the pure-tensor modules can still be tested.
try:
    import pytorch3d  # noqa: F401
except ImportError:  # pragma: no cover
    for name in ['pytorch3d', 'pytorch3d.structures', 'pytorch3d.renderer', 'pytorch3d.utils',
                 'pytorch3d.transforms']:
        sys.modules[name] = types.ModuleType(name)
    for attr in ['Meshes']:
        setattr(sys.modules['pytorch3d.structures'], attr, object)
    for attr in ['RasterizationSettings', 'MeshRasterizer', 'MeshRenderer', 'SoftSilhouetteShader',
                 'BlendParams']:
        setattr(sys.modules['pytorch3d.renderer'], attr, object)
    sys.modules['pytorch3d.utils'].cameras_from_opencv_projection = None
    sys.modules['pytorch3d.transforms'].matrix_to_axis_angle = None
    sys.modules['pytorch3d.transforms'].axis_angle_to_matrix = None

from uhydra.models.mhdr import standard_normal_quantiles, HypothesisProposal, DistributionalDepthHead
from uhydra.models.selection import select_candidates
from uhydra.models.losses import depth_nll, gather_winner, mesh_loss_per_candidate
from uhydra.models.geometry import rot6d_to_rotmat, geodesic_angle, perspective_project, soft_iou


def test_quantiles_K5():
    z = standard_normal_quantiles(5)
    assert torch.allclose(z, torch.tensor([-1.2816, -0.5244, 0.0, 0.5244, 1.2816]), atol=1e-3)


def test_hypotheses_order_and_coverage():
    prop = HypothesisProposal(K=5)
    mu = torch.log(torch.tensor([2.0, 0.8]))
    sigma = torch.tensor([0.2, 0.05])
    tz = prop(mu, sigma)
    assert tz.shape == (2, 5)
    assert (tz[:, 1:] > tz[:, :-1]).all()                      # monotone in k
    assert torch.allclose(tz[:, 2], torch.exp(mu))             # median = point estimate
    cov = prop.coverage(tz, torch.tensor([2.1, 2.0]))
    assert cov.tolist() == [1.0, 0.0]


def test_fixed_and_single_variants():
    assert HypothesisProposal(placement='single').K == 1
    fixed = HypothesisProposal(placement='fixed', fixed_scales=(1.0, 1.5))
    tz = fixed(torch.zeros(1), torch.ones(1))
    assert torch.allclose(tz, torch.tensor([[1.0, 1.5]]))


def test_depth_head_sigma_floor():
    head = DistributionalDepthHead(16, 8, sigma_min=0.05)
    mu, sigma = head(torch.randn(4, 16))
    assert (sigma >= 0.05).all()


def test_selection_rules():
    l_align = torch.tensor([[0.5, 0.1, 0.3], [0.2, 0.9, 0.4]])
    l_mesh = torch.tensor([[1.0, 2.0, 0.1], [0.3, 0.2, 0.9]])
    assert select_candidates(l_align, rule='image').tolist() == [1, 0]
    assert select_candidates(l_align, l_mesh, rule='oracle').tolist() == [2, 1]
    assert select_candidates(l_align, rule='median').tolist() == [1, 1]
    k = select_candidates(l_align, l_mesh, lambda_align=0.1, rule='train')
    assert k.tolist() == [2, 1]
    has = torch.tensor([True, False])
    assert select_candidates(l_align, rule='image', has_mask=has).tolist() == [1, 1]


def test_depth_nll_minimum_at_gt():
    tz_gt = torch.tensor([2.0]); valid = torch.ones(1)
    sigma = torch.tensor([0.1])
    at_gt = depth_nll(torch.log(tz_gt), sigma, tz_gt, valid)
    off = depth_nll(torch.log(tz_gt) + 0.3, sigma, tz_gt, valid)
    assert at_gt < off


def test_gather_and_mesh_loss_shapes():
    B, K = 3, 4
    R = rot6d_to_rotmat(torch.randn(B * K * 22, 6)).view(B, K, 22, 3, 3)
    pred = dict(betas=torch.randn(B, K, 10), global_orient=R[:, :, 0], body_pose=R[:, :, 1:],
                joints=torch.randn(B, K, 22, 3), verts=torch.randn(B, K, 100, 3))
    Rg = rot6d_to_rotmat(torch.randn(B * 22, 6)).view(B, 22, 3, 3)
    gt = dict(betas=torch.randn(B, 10), global_orient=Rg[:, 0], body_pose=Rg[:, 1:],
              joints=torch.randn(B, 22, 3), verts=torch.randn(B, 100, 3))
    l = mesh_loss_per_candidate(pred, gt, dict(beta=1, theta=1, J=5, V=5))
    assert l.shape == (B, K) and torch.isfinite(l).all()
    k = l.argmin(1)
    assert gather_winner(l, k).shape == (B,)
    assert gather_winner(pred['verts'], k).shape == (B, 100, 3)


def test_rotation_and_projection():
    R = rot6d_to_rotmat(torch.randn(5, 6))
    I = R.transpose(1, 2) @ R
    assert torch.allclose(I, torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(geodesic_angle(R, R), torch.zeros(5), atol=1e-3)
    pts = torch.tensor([[[0.0, 0.0, 0.0]]])
    K = torch.tensor([[[500.0, 0, 112.0], [0, 500.0, 112.0], [0, 0, 1]]])
    uv = perspective_project(pts, torch.tensor([[0.1, 0.0, 2.0]]), K)
    assert torch.allclose(uv[0, 0], torch.tensor([112.0 + 500 * 0.05, 112.0]))


def test_soft_iou():
    a = torch.zeros(1, 8, 8); a[:, :4] = 1
    b = torch.zeros(1, 8, 8); b[:, 2:6] = 1
    assert abs(soft_iou(a, b).item() - 16 / 48) < 1e-5


def test_dcme_shapes_and_gradient_reaches_depth_distribution():
    """The selected candidate's loss must back-propagate into (mu, sigma) through Eq. 2."""
    from uhydra.models.mhdr import MHDR
    from uhydra.models.dcme import DepthConditionedMeshHead
    B, C, K = 2, 32, 5
    mhdr = MHDR(C, 16, 0.05, K)
    head = DepthConditionedMeshHead(C, 64)
    f = torch.randn(B, C)
    d = mhdr(f)
    p = head(f, d['tz'])
    assert p['betas'].shape == (B, K, 10)
    assert p['global_orient'].shape == (B, K, 3, 3)
    assert p['body_pose'].shape == (B, K, 21, 3, 3)
    assert p['transl_xy'].shape == (B, K, 2)
    k_star = torch.tensor([3, 1])
    loss = gather_winner(p['betas'], k_star).pow(2).sum() + gather_winner(d['tz'], k_star).sum()
    loss.backward()
    g1 = mhdr.depth_head.fc2.weight.grad
    assert g1 is not None and g1.abs().sum() > 0          # mu / rho receive gradient

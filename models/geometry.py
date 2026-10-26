"""Geometry utilities for U-HYDRA.

* 6D rotation representation <-> rotation matrices
* Full-perspective projection (paper Sec. 2.2):
      u = f (x_m + T_x) / (z_m + T_z) + c_x,   v analogously,
  with the depth term retained in the denominator.
* Differentiable silhouette rendering (PyTorch3D soft rasteriser) and the
  soft IoU used by the alignment loss  L_align,k = 1 - sIoU(S_hat_k, S).
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer,
                                MeshRenderer, SoftSilhouetteShader, BlendParams)
from pytorch3d.utils import cameras_from_opencv_projection


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(N, 6) -> (N, 3, 3), Zhou et al. CVPR'19 continuous representation."""
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> (N, 6): first two columns."""
    return R[:, :, :2].transpose(1, 2).reshape(-1, 6)


def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    from pytorch3d.transforms import matrix_to_axis_angle
    return matrix_to_axis_angle(R)


def axis_angle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    from pytorch3d.transforms import axis_angle_to_matrix
    return axis_angle_to_matrix(aa)


def geodesic_angle(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """E_ang of Eq. 5: geodesic distance on SO(3), (N,) radians."""
    cos = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.acos(cos.clamp(-1.0 + eps, 1.0 - eps))


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def perspective_project(points: torch.Tensor, transl: torch.Tensor,
                        K: torch.Tensor) -> torch.Tensor:
    """Full-perspective projection (Sec. 2.2).

    points: (N, P, 3) pelvis-relative 3D points
    transl: (N, 3)    camera translation T = [T_x, T_y, T_z]
    K:      (N, 3, 3) pixel intrinsics of the crop
    returns (N, P, 2) pixel coordinates
    """
    p = points + transl[:, None, :]
    z = p[..., 2:3].clamp(min=1e-3)
    fx, fy = K[:, 0, 0][:, None, None], K[:, 1, 1][:, None, None]
    cx, cy = K[:, 0, 2][:, None, None], K[:, 1, 2][:, None, None]
    u = fx * p[..., 0:1] / z + cx
    v = fy * p[..., 1:2] / z + cy
    return torch.cat([u, v], dim=-1)


def scale_intrinsics(K: torch.Tensor, src_hw: Tuple[int, int], dst: int) -> torch.Tensor:
    """Rescale pixel intrinsics from a src_hw crop to a dst x dst render canvas."""
    K = K.clone()
    s = dst / float(max(src_hw))
    K[:, 0] *= s
    K[:, 1] *= s
    return K


# --------------------------------------------------------------------------- #
# silhouettes
# --------------------------------------------------------------------------- #
class SilhouetteRenderer(torch.nn.Module):
    """Renders soft (training) or hard (inference) silhouettes with PyTorch3D.

    Soft rasterisation follows Liu et al. ICCV'19 (SoftRas): sigma = 1e-4,
    gamma = 1e-4, faces_per_pixel = 50.  At inference, blur_radius = 0 and
    faces_per_pixel = 1 give a binary silhouette.
    """

    def __init__(self, image_size: int = 224, sigma: float = 1e-4,
                 gamma: float = 1e-4, faces_per_pixel: int = 50):
        super().__init__()
        self.image_size = image_size
        self.sigma, self.gamma = sigma, gamma
        self.faces_per_pixel = faces_per_pixel
        import math
        blur = math.log(1.0 / 1e-4 - 1.0) * sigma
        self.soft_settings = RasterizationSettings(image_size=image_size, blur_radius=blur,
                                                   faces_per_pixel=faces_per_pixel,
                                                   perspective_correct=False)
        self.hard_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                                   faces_per_pixel=1)

    def _cameras(self, K: torch.Tensor, device):
        N = K.shape[0]
        R = torch.eye(3, device=device)[None].expand(N, 3, 3)
        t = torch.zeros(N, 3, device=device)
        size = torch.tensor([[self.image_size, self.image_size]], device=device).expand(N, 2)
        return cameras_from_opencv_projection(R, t, K, size)

    def forward(self, verts_cam: torch.Tensor, faces: torch.Tensor,
                K: torch.Tensor, soft: bool = True) -> torch.Tensor:
        """verts_cam (N, V, 3) already in camera coordinates (translation applied),
        faces (F, 3), K (N, 3, 3) scaled to image_size.  Returns (N, H, W) in [0, 1]."""
        device = verts_cam.device
        N = verts_cam.shape[0]
        mesh = Meshes(verts=verts_cam, faces=faces[None].expand(N, -1, -1))
        cams = self._cameras(K, device)
        if soft:
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cams, raster_settings=self.soft_settings),
                shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=self.sigma, gamma=self.gamma)))
            return renderer(mesh)[..., 3]
        rasterizer = MeshRasterizer(cameras=cams, raster_settings=self.hard_settings)
        frags = rasterizer(mesh)
        return (frags.pix_to_face[..., 0] >= 0).float()


def soft_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """sIoU with set intersection/union replaced by pixel-wise products and sums.
    pred, target: (N, H, W) in [0, 1]  ->  (N,)"""
    p, t = pred.flatten(1), target.flatten(1)
    inter = (p * t).sum(1)
    union = (p + t - p * t).sum(1)
    return inter / (union + eps)


def hard_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p, t = (pred.flatten(1) > 0.5), (target.flatten(1) > 0.5)
    inter = (p & t).sum(1).float()
    union = (p | t).sum(1).float()
    return inter / (union + eps)

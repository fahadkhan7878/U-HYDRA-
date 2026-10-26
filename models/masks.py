"""Person mask S^(i) for candidate selection (paper Sec. 2.1).

Training : the ground-truth silhouette rendered from the ground-truth SMPL-X
           body when the dataset provides 3D labels (all training sets do).
Inference: ALWAYS a pseudo-label from an off-the-shelf segmentation model;
           no ground-truth annotation enters the inference path.

Backends
    'mediapipe' : MediaPipe Pose segmentation (the segmenter BLADE ships with)
    'external'  : masks pre-computed by any segmenter (e.g. Sapiens-Seg, SAM)
                  and stored as <mask_dir>/<image_idx>.png
    'none'      : no mask available -> selection falls back to the median
                  hypothesis (used only for the 'median' ablation / debugging)
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class MaskProvider:
    def __init__(self, backend: str = 'mediapipe', mask_dir: Optional[str] = None,
                 threshold: float = 0.2):
        assert backend in ('mediapipe', 'external', 'none'), backend
        self.backend = backend
        self.mask_dir = mask_dir
        self.threshold = threshold
        self._mp = None

    def _lazy_mediapipe(self):
        if self._mp is None:
            import mediapipe as mp
            self._mp = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                              enable_segmentation=True)
        return self._mp

    @torch.no_grad()
    def __call__(self, images_uint8: torch.Tensor, out_size: int,
                 sample_idx: Optional[torch.Tensor] = None):
        """images_uint8: (B, 3, H, W) RGB in [0, 255] (the un-normalised crop).
        Returns masks (B, out_size, out_size) in {0,1} and has_mask (B,) bool."""
        B, _, H, W = images_uint8.shape
        device = images_uint8.device
        masks = torch.zeros(B, out_size, out_size, device=device)
        has = torch.zeros(B, dtype=torch.bool, device=device)

        if self.backend == 'none':
            return masks, has

        for i in range(B):
            m = None
            if self.backend == 'mediapipe':
                seg = self._lazy_mediapipe()
                res = seg.process(images_uint8[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                if res.segmentation_mask is not None:
                    m = torch.from_numpy((res.segmentation_mask > self.threshold).astype(np.float32))
            elif self.backend == 'external':
                import cv2
                fn = os.path.join(self.mask_dir, f'{int(sample_idx[i])}.png')
                if os.path.exists(fn):
                    m = torch.from_numpy((cv2.imread(fn, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32))
            if m is not None:
                m = F.interpolate(m[None, None].to(device), size=(out_size, out_size),
                                  mode='bilinear', align_corners=False)[0, 0]
                masks[i] = (m > 0.5).float()
                has[i] = True
        return masks, has


def denormalize_to_uint8(img: torch.Tensor, mean=(123.675, 116.28, 103.53),
                         std=(58.395, 57.12, 57.375)) -> torch.Tensor:
    """Undo mmcv Normalize (RGB) -> uint8 RGB tensor (B,3,H,W)."""
    mean = torch.tensor(mean, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, 3, 1, 1)
    return (img * std + mean).clamp(0, 255)

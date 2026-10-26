"""U-HYDRA architecture (mmcv ``BaseArchitecture``)  --  the complete methodology.

Pipeline for every person crop (paper Fig. 2):
    HRNet -> F^(i) -> GAP -> f^(i)
    MHDR : (mu, sigma) -> K depth hypotheses T_{z,k}            (Sec. 2.1, Eqs. 1-2)
    DCME : (f, log T_{z,k}) -> (beta_k, theta_k) -> SMPL-X -> (V_k, J_k)   (Sec. 2.2, Eq. 4)
    full-perspective rendering of every candidate -> silhouette S_hat_k
    selection k* = argmin_k L_align,k (inference) / argmin_k L_mesh,k + lambda L_align,k (training)
    training objective Eq. 7

The module is registered in BLADE's ARCHITECTURES registry, so BLADE's
``scripts/train.py`` / ``scripts/test.py`` run it unchanged through the
configs in ``uhydra/configs``.
"""
import os
import sys
import math
from os.path import abspath, dirname
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmhuman3d.models.architectures.base_architecture import BaseArchitecture
from blade.models.architectures.builder import ARCHITECTURES
from blade.models.backbones.builder import build_backbone
from blade.models.body_models.builder import build_body_model
from blade.models.body_models.mappings import get_keypoint_idx
from blade.utils.torch_utils import dict2numpy

from .mhdr import MHDR
from .dcme import DepthConditionedMeshHead
from .geometry import (SilhouetteRenderer, soft_iou, hard_iou, scale_intrinsics,
                       axis_angle_to_rotmat, rotmat_to_axis_angle)
from .losses import (depth_nll, depth_l1, spread_regularizer,
                     mesh_loss_per_candidate, gather_winner)
from .selection import select_candidates
from .masks import MaskProvider, denormalize_to_uint8

ROOT = dirname(dirname(dirname(abspath(__file__))))      # BLADE repo root
sys.path.insert(0, os.path.join(ROOT, 'smplx_repo'))


@ARCHITECTURES.register_module(name='UHYDRA')
class UHYDRA(BaseArchitecture):

    def __init__(self,
                 backbone: dict,
                 feat_channels: int = 2048,
                 resolution: int = 224,
                 # MHDR
                 K: int = 5,
                 sigma_min: float = 0.05,
                 placement: str = 'learned',             # 'learned' | 'fixed' | 'single'
                 fixed_scales=(1.0, 1.2, 1.5, 1.8, 2.1),
                 depth_hidden: int = 256,
                 # DCME
                 mesh_hidden: int = 1024,
                 # selection
                 selection_rule: str = 'image',          # inference rule (ablation: oracle/random/median)
                 mask_backend: str = 'mediapipe',
                 mask_dir: Optional[str] = None,
                 # losses (Eq. 7 and Eq. 5)
                 lambda_z: float = 1.0,
                 lambda_sigma: float = 0.01,
                 lambda_mesh: float = 1.0,
                 lambda_align: float = 0.1,
                 mesh_loss_weights=dict(beta=1.0, theta=1.0, J=5.0, V=5.0),
                 depth_supervision: str = 'nll',          # 'nll' | 'nll_nosigma' | 'l1'
                 # rendering
                 render_size: int = 224,
                 soft_sigma: float = 1e-4,
                 faces_per_pixel: int = 50,
                 # evaluation
                 body_model_train: Optional[dict] = None,
                 body_model_test: Optional[dict] = None,
                 convert_to_smpl: bool = True,
                 compute_iou: bool = True,
                 diag_dir: Optional[str] = None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.resolution = resolution
        self.lambda_z, self.lambda_sigma = lambda_z, lambda_sigma
        self.lambda_mesh, self.lambda_align = lambda_mesh, lambda_align
        self.mesh_loss_weights = mesh_loss_weights
        self.depth_supervision = depth_supervision
        self.selection_rule = selection_rule
        self.convert_to_smpl = convert_to_smpl
        self.compute_iou = compute_iou
        self.diag_dir = diag_dir
        self.render_size = render_size

        # ---- shared backbone  F^(i) ----
        self.backbone = build_backbone(backbone)
        # ---- MHDR  (Sec. 2.1) ----
        self.mhdr = MHDR(feat_channels, depth_hidden, sigma_min, K, placement, fixed_scales)
        # ---- DCME  (Sec. 2.2) ----
        self.mesh_head = DepthConditionedMeshHead(feat_channels, mesh_hidden)
        # ---- SMPL-X body model M (neutral, hands/face at rest) ----
        import smplx
        self.smplx = smplx.create(os.path.join(ROOT, 'body_models'), model_type='smplx',
                                  gender='neutral', use_pca=False, flat_hand_mean=False,
                                  num_betas=10)
        for p in self.smplx.parameters():
            p.requires_grad = False
        self.register_buffer('smplx_faces', torch.tensor(self.smplx.faces.astype(np.int64)))
        # ---- renderer for S_hat_k ----
        self.renderer = SilhouetteRenderer(render_size, sigma=soft_sigma,
                                           faces_per_pixel=faces_per_pixel)
        self.mask_provider = MaskProvider(mask_backend, mask_dir)

        # ---- evaluation helpers (BLADE protocol: SMPL-X -> SMPL, h36m joints) ----
        self.body_model_train = build_body_model(body_model_train) if body_model_train else None
        self.body_model_test = build_body_model(body_model_test) if body_model_test else None
        self._conversion_ready = False
        self.num_body_joints = 22   # pelvis + 21 body joints of SMPL-X

    # ------------------------------------------------------------------ #
    # core forward shared by train / test
    # ------------------------------------------------------------------ #
    def extract_descriptor(self, img: torch.Tensor) -> torch.Tensor:
        """img (B,3,H,W) -> f^(i) (B, C): HRNet feature map + global average pooling."""
        feat = self.backbone(img)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        return feat

    def smplx_forward(self, betas, global_orient, body_pose):
        """Decode K candidates with the shared SMPL-X model.  Inputs are (N, ...)
        with N = B*K; rotations as matrices.  Returns pelvis-relative verts/joints."""
        N = betas.shape[0]
        zeros3 = torch.zeros(N, 3, device=betas.device)
        out = self.smplx(betas=betas,
                         global_orient=rotmat_to_axis_angle(global_orient),
                         body_pose=rotmat_to_axis_angle(body_pose.reshape(-1, 3, 3)).view(N, -1),
                         left_hand_pose=torch.zeros(N, 45, device=betas.device),
                         right_hand_pose=torch.zeros(N, 45, device=betas.device),
                         jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                         expression=torch.zeros(N, 10, device=betas.device),
                         transl=zeros3, return_verts=True)
        pelvis = out.joints[:, :1]
        return out.vertices - pelvis, out.joints - pelvis, pelvis

    def to_true_frame(self, xy_crop: torch.Tensor, tz: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """The mesh head regresses (T_x, T_y) in a virtual crop camera whose
        principal point is the crop centre.  Convert to the true camera frame
        (K carries the true principal point of the crop):
            T_x = T_x' - (c_x - W/2) * T_z / f_x   (exact at the pelvis)."""
        W = self.resolution
        cx = K[:, 0, 2][:, None]; cy = K[:, 1, 2][:, None]
        fx = K[:, 0, 0][:, None]; fy = K[:, 1, 1][:, None]
        tx = xy_crop[..., 0] - (cx - W / 2.0) * tz / fx
        ty = xy_crop[..., 1] - (cy - W / 2.0) * tz / fy
        return torch.stack([tx, ty], dim=-1)

    def forward_candidates(self, img: torch.Tensor, K: torch.Tensor):
        """Runs MHDR + DCME + SMPL-X for all K hypotheses.  Returns a dict with
        per-candidate quantities of shape (B, K, ...)."""
        B = img.shape[0]
        f = self.extract_descriptor(img)                                   # (B, C)
        d = self.mhdr(f)                                                   # mu, sigma, tz (B,K)
        tz = d['tz']
        Kh = tz.shape[1]
        p = self.mesh_head(f, tz)                                          # betas, rotations, xy
        verts, joints, _ = self.smplx_forward(p['betas'].reshape(B * Kh, -1),
                                              p['global_orient'].reshape(B * Kh, 3, 3),
                                              p['body_pose'].reshape(B * Kh, -1, 3, 3))
        V, J = verts.shape[1], joints.shape[1]
        xy = self.to_true_frame(p['transl_xy'], tz, K)
        transl = torch.cat([xy, tz[..., None]], dim=-1)                    # (B, K, 3)
        return dict(mu=d['mu'], sigma=d['sigma'], tz_hat=d['tz_hat'], tz=tz,
                    betas=p['betas'], global_orient=p['global_orient'], body_pose=p['body_pose'],
                    verts=verts.view(B, Kh, V, 3), joints=joints.view(B, Kh, J, 3),
                    transl=transl, K=K)

    def render_candidates(self, cand: dict, soft: bool) -> torch.Tensor:
        """Silhouette S_hat_k of every candidate under full perspective -> (B, K, H, W)."""
        B, Kh = cand['tz'].shape
        verts_cam = cand['verts'] + cand['transl'][:, :, None, :]          # (B,K,V,3)
        Kr = scale_intrinsics(cand['K'], (self.resolution, self.resolution), self.render_size)
        Kr = Kr[:, None].expand(B, Kh, 3, 3).reshape(B * Kh, 3, 3)
        sil = self.renderer(verts_cam.reshape(B * Kh, -1, 3), self.smplx_faces, Kr, soft=soft)
        return sil.view(B, Kh, self.render_size, self.render_size)

    @torch.no_grad()
    def render_gt_mask(self, verts_rel: torch.Tensor, transl: torch.Tensor, K: torch.Tensor,
                       faces: torch.Tensor) -> torch.Tensor:
        Kr = scale_intrinsics(K, (self.resolution, self.resolution), self.render_size)
        return self.renderer(verts_rel + transl[:, None, :], faces, Kr, soft=False)

    # ------------------------------------------------------------------ #
    # targets
    # ------------------------------------------------------------------ #
    def prepare_targets_smplx(self, data_batch: dict) -> dict:
        """Ground truth from SMPL-X labels (training sets).  Uses the un-augmented
        ``*_orig`` parameters, which correspond to the un-augmented person crop
        ``depthnet_img`` and its intrinsics ``K`` (see configs: flips disabled)."""
        betas = data_batch['SMPLX_betas']
        go = data_batch['SMPLX_global_orient_orig'].view(-1, 3)
        bp = data_batch['SMPLX_body_pose_orig'].view(-1, 21, 3)
        lh = data_batch['SMPLX_left_hand_pose_orig'].view(betas.shape[0], -1)
        rh = data_batch['SMPLX_right_hand_pose_orig'].view(betas.shape[0], -1)
        N = betas.shape[0]
        zeros3 = torch.zeros(N, 3, device=betas.device)
        with torch.no_grad():
            out = self.smplx(betas=betas, global_orient=go, body_pose=bp.reshape(N, -1),
                             left_hand_pose=lh, right_hand_pose=rh,
                             jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                             expression=torch.zeros(N, 10, device=betas.device),
                             transl=zeros3, return_verts=True)
        pelvis = out.joints[:, :1]
        transl_pelvis = data_batch['SMPLX_transl_orig'].view(N, 3) + pelvis[:, 0]   # pelvis in camera frame
        return dict(betas=betas,
                    global_orient=axis_angle_to_rotmat(go),
                    body_pose=axis_angle_to_rotmat(bp),
                    verts=out.vertices - pelvis,
                    joints=(out.joints - pelvis)[:, :self.num_body_joints],
                    transl=transl_pelvis,
                    tz=transl_pelvis[:, 2],
                    valid=torch.ones(N, device=betas.device))

    # ------------------------------------------------------------------ #
    # training  (Sec. 2.3)
    # ------------------------------------------------------------------ #
    def forward_train(self, **data_batch):
        img = data_batch['depthnet_img']
        K = data_batch['K'].float()
        assert img.shape[-1] == self.resolution, 'set img_res_train = resolution in the config'
        assert bool(data_batch['has_SMPLX'].view(-1).all()), 'U-HYDRA trains on SMPL-X labelled data'

        gt = self.prepare_targets_smplx(data_batch)
        cand = self.forward_candidates(img, K)
        B, Kh = cand['tz'].shape

        # ---- L_depth (Eq. 6) and L_sigma, applied to EVERY instance ----
        if self.depth_supervision == 'l1':
            l_depth = depth_l1(cand['mu'], gt['tz'], gt['valid'])
        else:
            l_depth = depth_nll(cand['mu'], cand['sigma'], gt['tz'], gt['valid'])
        l_sigma = spread_regularizer(cand['sigma'], gt['valid'])
        if self.depth_supervision == 'nll_nosigma':
            l_sigma = l_sigma * 0.0

        # ---- per-candidate reconstruction loss (Eq. 5) ----
        pred = dict(betas=cand['betas'], global_orient=cand['global_orient'],
                    body_pose=cand['body_pose'], verts=cand['verts'],
                    joints=cand['joints'][:, :, :self.num_body_joints])
        l_mesh_k = mesh_loss_per_candidate(pred, gt, self.mesh_loss_weights)     # (B,K)

        # ---- per-candidate alignment loss  1 - sIoU(S_hat_k, S) ----
        S = self.render_gt_mask(gt['verts'], gt['transl'], K, self.smplx_faces)   # GT silhouette (B,H,W)
        S_hat = self.render_candidates(cand, soft=True)                            # (B,K,H,W)
        l_align_k = 1.0 - soft_iou(S_hat.reshape(B * Kh, *S_hat.shape[2:]),
                                   S[:, None].expand(B, Kh, -1, -1).reshape(B * Kh, *S.shape[1:])).view(B, Kh)

        # ---- winner-takes-all selection (Eq. 3, training branch) ----
        k_star = select_candidates(l_align_k.detach(), l_mesh_k.detach(),
                                   self.lambda_align, rule='train')
        l_mesh = gather_winner(l_mesh_k, k_star).mean()
        l_align = gather_winner(l_align_k, k_star).mean()

        losses = dict(
            loss_depth=self.lambda_z * l_depth,
            loss_sigma=self.lambda_sigma * l_sigma,
            loss_mesh=self.lambda_mesh * l_mesh,
            loss_align=self.lambda_align * l_align,
        )
        # diagnostics (not back-propagated; mmcv logs anything without 'loss' in the key)
        with torch.no_grad():
            tz_sel = gather_winner(cand['tz'], k_star)
            losses['metric:tz_err_selected'] = (tz_sel - gt['tz']).abs().mean()
            losses['metric:tz_err_point'] = (cand['tz_hat'] - gt['tz']).abs().mean()
            losses['metric:sigma_mean'] = cand['sigma'].mean()
            losses['metric:coverage'] = self.mhdr.proposal.coverage(cand['tz'], gt['tz']).mean()
            losses['metric:oracle_gap_mesh'] = (gather_winner(l_mesh_k, k_star)
                                                - l_mesh_k.min(dim=1).values).mean()
            js = gather_winner(cand['joints'], k_star)[:, :self.num_body_joints]
            losses['metric:mpjpe_mm'] = ((js - gt['joints']) ** 2).sum(-1).sqrt().mean() * 1000.0
        return losses

    # ------------------------------------------------------------------ #
    # inference  (Eq. 3, inference branch)  +  BLADE evaluation protocol
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_test(self, **data_batch):
        img = data_batch['depthnet_img']
        K = data_batch['K'].float()
        B = img.shape[0]
        device = img.device
        all_preds = {'image_idx': data_batch['sample_idx']}

        cand = self.forward_candidates(img, K)
        Kh = cand['tz'].shape[1]

        # pseudo-label mask S^(i): never ground truth at inference
        S, has_mask = self.mask_provider(denormalize_to_uint8(img), self.render_size,
                                         data_batch['sample_idx'])
        S_hat = self.render_candidates(cand, soft=False)                               # hard silhouettes
        l_align_k = 1.0 - hard_iou(S_hat.reshape(B * Kh, *S_hat.shape[2:]),
                                   S[:, None].expand(B, Kh, -1, -1).reshape(B * Kh, *S.shape[1:])).view(B, Kh)
        l_align_k[~has_mask] = float('nan')

        # oracle / diagnostics need L_mesh with ground truth (evaluation only)
        l_mesh_k = None
        gt = None
        if 'has_SMPLX' in data_batch and bool(data_batch['has_SMPLX'].view(-1).all()):
            gt = self.prepare_targets_smplx(data_batch)
            pred = dict(betas=cand['betas'], global_orient=cand['global_orient'],
                        body_pose=cand['body_pose'], verts=cand['verts'],
                        joints=cand['joints'][:, :, :self.num_body_joints])
            l_mesh_k = mesh_loss_per_candidate(pred, gt, self.mesh_loss_weights)

        k_star = select_candidates(l_align_k, l_mesh_k, self.lambda_align,
                                   rule=self.selection_rule, has_mask=has_mask)

        # ---- selected reconstruction ----
        sel = {k: gather_winner(cand[k], k_star) for k in
               ('betas', 'global_orient', 'body_pose', 'verts', 'joints', 'transl', 'tz')}
        all_preds['k_star'] = k_star
        all_preds['sigma'] = cand['sigma']
        all_preds['tz_hypotheses'] = cand['tz']
        all_preds['pred_transl'] = sel['transl']
        all_preds['smplx_betas'] = sel['betas']
        all_preds['smplx_global_orient'] = rotmat_to_axis_angle(sel['global_orient'])
        all_preds['smplx_body_pose'] = rotmat_to_axis_angle(sel['body_pose'].reshape(-1, 3, 3)).view(B, -1)
        all_preds['vertices_smplx'] = sel['verts']

        # ---- SMPL-X -> SMPL conversion for the SMPL-labelled benchmarks (BLADE protocol) ----
        if self.convert_to_smpl:
            self._convert_to_smpl(sel, all_preds, device)
        else:
            all_preds['vertices'] = sel['verts']
            all_preds['keypoints_3d'] = sel['joints'][:, :self.num_body_joints]

        # ---- depth / coverage / oracle diagnostics against the benchmark GT ----
        if 'smpl_transl' in data_batch and self.body_model_train is not None:
            tz_gt = self._gt_pelvis_depth_smpl(data_batch)
            all_preds['z_error'] = (tz_gt - sel['tz']).abs()
            all_preds['inv_z_error'] = (1.0 / tz_gt - 1.0 / sel['tz']).abs()
            all_preds['coverage'] = self.mhdr.proposal.coverage(cand['tz'], tz_gt)
            if self.compute_iou:
                self._silhouette_iou(data_batch, all_preds, device)
            self._per_candidate_errors(data_batch, cand, k_star, all_preds)
        if self.diag_dir:
            self._dump_diagnostics(all_preds)
        return dict2numpy(all_preds)

    # ------------------------------------------------------------------ #
    # evaluation helpers
    # ------------------------------------------------------------------ #
    def _gt_pelvis_depth_smpl(self, data_batch):
        out = self.body_model_train(betas=data_batch['smpl_betas'].float(),
                                    body_pose=data_batch['smpl_body_pose'].float(),
                                    global_orient=data_batch['smpl_global_orient'].float())
        pelvis = out['joints'][:, 0]
        return (pelvis + data_batch['smpl_transl'].float())[:, 2]

    def _init_conversion(self, device):
        if self._conversion_ready:
            return
        from blade.datasets.utils import init_conversion
        cfg, dst_model, def_matrix, mask_ids = init_conversion(
            os.path.join(ROOT, 'smplx_repo'), device, 'smplx', 'smpl')
        self.conv_cfg, self.conv_model, self.conv_def, self.conv_mask = cfg, dst_model, def_matrix, mask_ids
        kd = self.body_model_test.keypoint_dst
        self.right_hip_idx = get_keypoint_idx('right_hip_extra', kd)
        self.left_hip_idx = get_keypoint_idx('left_hip_extra', kd)
        self._conversion_ready = True

    def _convert_to_smpl(self, sel, all_preds, device):
        from transfer_model.transfer_model import run_fitting
        self._init_conversion(device)
        B = sel['verts'].shape[0]
        # pelvis-relative SMPL-X mesh (BLADE converts the un-translated body as well)
        mesh = {'vertices': sel['verts'],
                'faces': self.smplx_faces[None].expand(B, -1, -1)}
        with torch.enable_grad():
            out = run_fitting(self.conv_cfg, mesh, self.conv_model.to(device),
                              self.conv_def.to(device), self.conv_mask)
        verts_smpl = out['vertices'].detach()
        joints_smpl = self.body_model_test.forward_joints(dict(vertices=verts_smpl))[0]
        all_preds['vertices'] = verts_smpl
        all_preds['keypoints_3d'] = joints_smpl
        all_preds['smpl_beta'] = out['betas'].detach()
        all_preds['smpl_pose'] = torch.cat([rotmat_to_axis_angle(out['global_orient']).reshape(B, -1),
                                            rotmat_to_axis_angle(out['body_pose']).reshape(B, -1)], 1).detach()

    def _silhouette_iou(self, data_batch, all_preds, device):
        """mIoU of the selected candidate against the GT silhouette (benchmark GT body)."""
        out = self.body_model_train(betas=data_batch['smpl_betas'].float(),
                                    body_pose=data_batch['smpl_body_pose'].float(),
                                    global_orient=data_batch['smpl_global_orient'].float())
        faces = self.body_model_train.faces_tensor.long().to(device)
        K = data_batch['K'].float()
        gt_sil = self.render_gt_mask(out['vertices'], data_batch['smpl_transl'].float(), K, faces)
        pred_sil = self.render_gt_mask(all_preds['vertices_smplx'], all_preds['pred_transl'], K, self.smplx_faces)
        all_preds['raw_batch_miou'] = hard_iou(pred_sil, gt_sil) * 100.0

    def _per_candidate_errors(self, data_batch, cand, k_star, all_preds):
        """Oracle gap: MPJPE(selected) - min_k MPJPE(candidate k), on SMPL-X body joints
        vs. the benchmark's SMPL joints mapped through the h36m regressor is not
        possible without conversion of all K candidates; we therefore report the
        oracle gap on pelvis-relative SMPL-X body joints against SMPL-X GT when
        available, and on root depth otherwise."""
        tz_gt = self._gt_pelvis_depth_smpl(data_batch)
        err_k = (cand['tz'] - tz_gt[:, None]).abs()                      # (B,K)
        all_preds['oracle_gap_tz'] = gather_winner(err_k, k_star) - err_k.min(dim=1).values

    def _dump_diagnostics(self, all_preds):
        os.makedirs(self.diag_dir, exist_ok=True)
        keys = ['image_idx', 'k_star', 'sigma', 'tz_hypotheses', 'z_error', 'coverage',
                'oracle_gap_tz', 'raw_batch_miou']
        d = {k: all_preds[k].detach().cpu().numpy() for k in keys if k in all_preds}
        idx = int(all_preds['image_idx'].view(-1)[0])
        np.savez(os.path.join(self.diag_dir, f'batch_{idx:08d}.npz'), **d)

    # mmcv entry points -------------------------------------------------- #
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def train_step(self, data, optimizer):
        """Standard mmcv step: the OptimizerHook performs backward + step
        (``use_adversarial_train = False`` in the configs)."""
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        return dict(loss=loss, log_vars=log_vars, num_samples=data['depthnet_img'].shape[0])

    def val_step(self, data, optimizer=None):
        return self.train_step(data, optimizer)

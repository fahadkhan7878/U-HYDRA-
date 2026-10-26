"""Depth-Conditioned Mesh Estimation (DCME)  --  paper Sec. 2.2, Eq. 4.

    (beta_k, theta_k) = F_mesh(F^(i), T_{z,k}),   (V_k, J_k) = M(beta_k, theta_k)

F_mesh concatenates the pooled descriptor f^(i) = GAP(F^(i)) in R^C with
log T_{z,k} and regresses the SMPL-X parameters with a two-layer MLP.  The
same weights are shared across the K hypotheses; the K candidates are
produced by running the head on the batch replicated K times.

Outputs per candidate
    betas        (B, K, 10)
    global_orient(B, K, 3, 3)  rotation matrix  (from 6D continuous rep.)
    body_pose    (B, K, 21, 3, 3)
    transl_xy    (B, K, 2)     in-plane camera translation (metres)
The rotation outputs are produced as residuals on a mean-pose initialisation
(HMR-style), which keeps the two-layer regressor well-conditioned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import rot6d_to_rotmat, rotmat_to_rot6d

NUM_BODY_JOINTS = 21   # SMPL-X body joints (without pelvis / global orient)
NUM_BETAS = 10


class DepthConditionedMeshHead(nn.Module):
    """F_mesh of Eq. 4."""

    def __init__(self, in_channels: int = 2048, hidden: int = 1024,
                 num_betas: int = NUM_BETAS, dropout: float = 0.0):
        super().__init__()
        self.num_betas = num_betas
        n_rot = 6 * (1 + NUM_BODY_JOINTS)             # global orient + 21 joints, 6D each
        self.n_out = num_betas + n_rot + 2            # + (T_x, T_y)
        self.fc1 = nn.Linear(in_channels + 1, hidden)  # +1 for log T_z,k
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, self.n_out)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

        # mean parameters used as the residual base (identity rotations, zero shape)
        eye6d = rotmat_to_rot6d(torch.eye(3)[None]).view(-1)       # (6,)
        init_pose = eye6d.repeat(1 + NUM_BODY_JOINTS)                # (132,)
        self.register_buffer('init_betas', torch.zeros(num_betas))
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_xy', torch.zeros(2))

    def forward(self, f: torch.Tensor, tz: torch.Tensor):
        """f: (B, C) pooled descriptor, tz: (B, K) depth hypotheses (metres)."""
        B, K = tz.shape
        f_rep = f[:, None, :].expand(B, K, f.shape[-1])              # share features across K
        x = torch.cat([f_rep, torch.log(tz)[..., None]], dim=-1)    # (B, K, C+1)
        x = x.reshape(B * K, -1)
        h = self.drop(F.relu(self.fc1(x)))
        out = self.fc2(h)                                            # (B*K, n_out)

        nb = self.num_betas
        betas = out[:, :nb] + self.init_betas
        pose6d = out[:, nb:nb + 6 * (1 + NUM_BODY_JOINTS)] + self.init_pose
        xy = out[:, -2:] + self.init_xy

        rotmats = rot6d_to_rotmat(pose6d.reshape(-1, 6)).view(B * K, 1 + NUM_BODY_JOINTS, 3, 3)
        return dict(
            betas=betas.view(B, K, nb),
            global_orient=rotmats[:, 0].view(B, K, 3, 3),
            body_pose=rotmats[:, 1:].view(B, K, NUM_BODY_JOINTS, 3, 3),
            transl_xy=xy.view(B, K, 2),
        )

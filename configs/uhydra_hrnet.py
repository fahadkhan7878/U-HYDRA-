"""U-HYDRA with an HRNet-W32 / W48 backbone (Table 1-3 of the paper).

    UHYDRA_WIDTH=48 MINI_BATCHSIZE=22 python scripts/train.py uhydra/configs/uhydra_hrnet.py \
        --work-dir work_dirs/uhydra_w48 --launcher pytorch

    MINI_BATCHSIZE=16 python scripts/test.py uhydra/configs/uhydra_hrnet.py \
        --checkpoint work_dirs/uhydra_w48/epoch_150.pth --data-name pdhuman_p5 \
        --work-dir work_dirs/uhydra_w48/eval --out work_dirs/uhydra_w48/eval/pdhuman.pkl
"""
import os
from uhydra.configs.uhydra_base import *   # noqa: F401,F403
from blade.configs.base import body_model_train, body_model_test

_base_ = ['uhydra_base.py']

width = int(os.environ.get('UHYDRA_WIDTH', 48))
hrnet_extra = dict(
    stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
    stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), num_channels=(width, width * 2)),
    stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4),
                num_channels=(width, width * 2, width * 4)),
    stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4),
                num_channels=(width, width * 2, width * 4, width * 8)),
    downsample=False, use_conv=True,
    pretrained_layers=['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2',
                       'transition2', 'stage3', 'transition3', 'stage4'],
    final_conv_kernel=1, return_list=False)

hrnet_ckpt = os.environ.get('HRNET_PRETRAINED', f'{root}/pretrained/hrnet_w{width}_imagenet.pth')

model = dict(
    type='UHYDRA',
    backbone=dict(type='PoseHighResolutionNetExpose', extra=hrnet_extra, num_joints=24,
                  norm_eval=False,      # BN statistics are updated (end-to-end training)
                  init_cfg=dict(type='Pretrained', checkpoint=hrnet_ckpt)),
    feat_channels=2048,
    resolution=img_res_train,
    # MHDR (Sec. 2.1)
    K=int(os.environ.get('UHYDRA_K', 5)),
    sigma_min=0.05,
    placement=os.environ.get('UHYDRA_PLACEMENT', 'learned'),
    fixed_scales=(1.0, 1.2, 1.5, 1.8, 2.1),
    depth_hidden=256,
    # DCME (Sec. 2.2)
    mesh_hidden=1024,
    # selection (Eq. 3)
    selection_rule=os.environ.get('UHYDRA_SELECTION', 'image'),
    mask_backend=os.environ.get('UHYDRA_MASK_BACKEND', 'mediapipe'),
    mask_dir=os.environ.get('UHYDRA_MASK_DIR', None),
    # objective (Eq. 7)
    lambda_z=1.0, lambda_sigma=0.01, lambda_mesh=1.0, lambda_align=0.1,
    mesh_loss_weights=dict(beta=1.0, theta=1.0, J=5.0, V=5.0),
    depth_supervision=os.environ.get('UHYDRA_DEPTH_SUP', 'nll'),
    # rendering (PyTorch3D soft rasteriser, 224x224)
    render_size=224, soft_sigma=1e-4, faces_per_pixel=50,
    # evaluation (BLADE protocol)
    body_model_train=body_model_train,
    body_model_test=body_model_test,
    convert_to_smpl=True,
    compute_iou=True,
    diag_dir=os.environ.get('UHYDRA_DIAG_DIR', None),
)

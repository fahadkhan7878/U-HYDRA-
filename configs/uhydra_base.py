"""U-HYDRA base config: data pipelines, datasets, schedule  (paper Sec. 3).

Differences from BLADE's base config
  * the network consumes the person-centred crop ``depthnet_img`` at 224x224
    with its intrinsics ``K`` (full perspective, principal point carried in K);
  * horizontal flipping is disabled so that the un-augmented ``*_orig`` SMPL-X
    labels match the crop (BLADE couples the flip flag of the pose stage to K);
  * single end-to-end optimiser: Adam 1e-4, x0.1 at epochs 90/120, 150 epochs,
    batch 64 (= 3 GPUs x MINI_BATCHSIZE 22 by default).
Ablation switches are read from environment variables (see uhydra/tools/ablation.py):
  UHYDRA_WIDTH=32|48   UHYDRA_K=5   UHYDRA_PLACEMENT=learned|fixed|single
  UHYDRA_DEPTH_SUP=nll|nll_nosigma|l1   UHYDRA_SELECTION=image|oracle|random|median
"""
import os
from blade.configs.base import (root, convention, convention_test, img_norm_cfg, data_keys,
                                data_keys_test, cache_files, body_model_train, body_model_test)

custom_imports = dict(imports=['uhydra.models.architecture', 'blade.custom_hooks.wandb_hook'],
                      allow_failed_imports=False)

# ----------------------------------------------------------------------------- #
# resolution and pipelines
# ----------------------------------------------------------------------------- #
img_res_train = 224          # person crop fed to HRNet  (depthnet_img in BLADE's dataset)
img_res_posenet = 224        # second crop produced by BLADE's dataset class (unused)

_depth_stage_train = [
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
    dict(type='GetRandomScaleRotation', rot_factor=0., scale_factor=0., rot_prob=0., scale_add=0.),
    dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
    dict(type='MeshAffine', img_res=dict(img=img_res_train), require_origin_kp2d=True, img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=['image_path', 'center', 'scale', 'rotation']),
]
_pose_stage_train = [
    dict(type='RandomChannelNoise', noise_factor=0., img_fields=['img']),
    dict(type='RandomHorizontalFlip', flip_prob=0., convention=convention, img_fields=['img']),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0., rot_prob=0., scale_add=0.),
    dict(type='GetBboxInfo', rand_shift=0., rand_shift_prob=0.),
    dict(type='MeshAffine', img_res=dict(img=img_res_posenet), require_origin_kp2d=True, img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=['image_path', 'center', 'scale', 'rotation']),
]
train_pipeline_uhydra = [dict(type='LoadImageFromFile')] + _depth_stage_train + _pose_stage_train

_depth_stage_test = [
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res_train),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect', keys=['img', *data_keys_test], meta_keys=['image_path', 'center', 'scale', 'rotation']),
]
_pose_stage_test = [
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res_posenet),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect', keys=['img', *data_keys_test], meta_keys=['image_path', 'center', 'scale', 'rotation']),
]
test_pipeline_uhydra = [dict(type='LoadImageFromFile')] + _depth_stage_test + _pose_stage_test

# ----------------------------------------------------------------------------- #
# datasets  (Sec. 3, "Training and Evaluation Setup")
# ----------------------------------------------------------------------------- #
def _smplx_train(name, ann, cache):
    return dict(type='HumanImageDataset_SMPLX', data_prefix=f'{root}/mmhuman_data/',
                convention=convention, ann_file=ann, dataset_name=name, is_distorted=True,
                do_black_pad_aug=False, cache_data_path=cache, pipeline=train_pipeline_uhydra)

pdhuman_train = _smplx_train('pdhuman', 'pdhuman_train_smplx.npz', cache_files['pdhuman_train_smplx'])
humman_train = _smplx_train('humman', 'humman_train_smplx.npz', cache_files['humman_train_smplx'])
h36m_train = _smplx_train('h36m', 'h36m_mosh_train_transl_smplx.npz', cache_files['h36m_transl_smplx'])
h36m_train.pop('is_distorted')
# BEDLAM-CC (SMPL-X renders, see BLADE docs/DATASETS.md) and MPI-INF-3DHP can be added
# here once converted to the HumanImageDataset_SMPLX format:
# bedlamcc_train = _smplx_train('bedlamcc', 'bedlamcc_smplx.npz', None)

train_mix = dict(type='MixedDataset',
                 configs=[pdhuman_train, humman_train, h36m_train],
                 partition=[0.435, 0.436, 0.129])

def _smplx_test(name, ann, conv=convention):
    return dict(type='HumanImageDataset_SMPLX', data_prefix=f'{root}/mmhuman_data/',
                convention=conv, ann_file=ann, dataset_name=name, test_mode=True,
                is_distorted=True, body_model=body_model_test, pipeline=test_pipeline_uhydra)

test_dict = dict(
    spec_mtp_p3=_smplx_test('spec_mtp', 'spec_mtp_p3.npz'),
    pdhuman_p5=_smplx_test('pdhuman', 'pdhuman_test_p5.npz', convention_test),
    humman_p3=_smplx_test('humman', 'humman_test_p3.npz'),
    # 3DPW / AGORA: prepare npz files with the MMHuman3D converters (see uhydra/README.md)
    pw3d=dict(type='HumanImageDataset_SMPLX', data_prefix=f'{root}/mmhuman_data/',
              convention=convention_test, ann_file='pw3d_test.npz', dataset_name='pw3d',
              test_mode=True, body_model=body_model_test, pipeline=test_pipeline_uhydra),
    agora_val=dict(type='HumanImageDataset_SMPLX', data_prefix=f'{root}/mmhuman_data/',
                   convention=convention_test, ann_file='agora_val.npz', dataset_name='agora',
                   test_mode=True, body_model=body_model_test, pipeline=test_pipeline_uhydra),
)
val_dict = test_dict['spec_mtp_p3']

# ----------------------------------------------------------------------------- #
# schedule  (Sec. 3, "Implementation")
# ----------------------------------------------------------------------------- #
use_adversarial_train = False          # standard mmcv OptimizerHook does backward/step
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
lr_config = dict(policy='step', step=[90, 120], gamma=0.1, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric=['pa-mpjpe', 'mpjpe', 'pve', 'raw_miou', 'z_error', 'inv_z_error'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

data = dict(samples_per_gpu=int(os.environ.get('MINI_BATCHSIZE', 22)),   # 3 x 22 ~ 64
            workers_per_gpu=8, train=train_mix, test=test_dict, val=val_dict)

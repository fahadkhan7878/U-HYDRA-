#!/usr/bin/env bash
# Train U-HYDRA (paper Sec. 3, Implementation): Adam 1e-4, decay x0.1 at epochs 90/120,
# 150 epochs, batch 64 on three RTX 3090 (3 x 22 = 66 ~ 64), HRNet-W48 by default.
#
#   bash uhydra/tools/train.sh 48 work_dirs/uhydra_w48
#   bash uhydra/tools/train.sh 32 work_dirs/uhydra_w32
set -e
WIDTH=${1:-48}
WORKDIR=${2:-work_dirs/uhydra_w${WIDTH}}
NGPU=${NGPU:-3}
export UHYDRA_WIDTH=$WIDTH
export MINI_BATCHSIZE=${MINI_BATCHSIZE:-22}
export PYGLET_HEADLESS=True PYOPENGL_PLATFORM=egl

python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=${PORT:-29511} \
    scripts/train.py uhydra/configs/uhydra_hrnet.py --work-dir "$WORKDIR" --launcher pytorch --seed 0

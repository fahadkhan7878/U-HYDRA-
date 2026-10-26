#!/usr/bin/env bash
# Evaluate a U-HYDRA checkpoint on the four benchmarks of Tables 1-2.
#   bash uhydra/tools/test.sh work_dirs/uhydra_w48/epoch_150.pth 48 work_dirs/uhydra_w48/eval
set -e
CKPT=$1
WIDTH=${2:-48}
OUT=${3:-$(dirname "$CKPT")/eval}
export UHYDRA_WIDTH=$WIDTH
export MINI_BATCHSIZE=${MINI_BATCHSIZE:-16}
export PYGLET_HEADLESS=True PYOPENGL_PLATFORM=egl
mkdir -p "$OUT"

for DS in spec_mtp_p3 pdhuman_p5 pw3d agora_val; do
  export UHYDRA_DIAG_DIR="$OUT/diag_$DS"
  python scripts/test.py uhydra/configs/uhydra_hrnet.py \
      --checkpoint "$CKPT" --data-name $DS --work-dir "$OUT" --out "$OUT/${DS}.pkl" \
      2>&1 | tee "$OUT/${DS}.log"
  python uhydra/evaluation/summarize_diagnostics.py "$UHYDRA_DIAG_DIR" \
      --K ${UHYDRA_K:-5} --out "$OUT/${DS}_diag.json" || true
done

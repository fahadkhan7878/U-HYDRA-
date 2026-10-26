"""Computational cost (Table 3): parameters, GFLOPs, peak GPU memory and latency.

Rows
  Baseline            : HRNet + depth head + mesh head run ONCE with the point
                        estimate exp(mu) and no rendering/selection (K = 1, placement 'single',
                        selection skipped)  -- a single-depth regressor.
  + DCME (K=1)        : same forward with one rendered silhouette and the selection code path.
  Full U-HYDRA (K=5)  : K hypotheses, K shared mesh-head passes, K silhouettes, selection.

Latency is the median over 100 runs with batch size 1 and a single person on
one GPU, after 20 warm-up iterations, including the renderer but excluding the
external segmentation model (reported separately when a backend is given).

    python uhydra/tools/cost.py --width 48 --K 5 [--checkpoint ckpt.pth]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def build(width, K, placement, ckpt=None):
    from mmcv import Config
    from blade.models.architectures.builder import build_architecture
    os.environ['UHYDRA_WIDTH'] = str(width)
    os.environ['UHYDRA_K'] = str(K)
    os.environ['UHYDRA_PLACEMENT'] = placement
    os.environ['UHYDRA_MASK_BACKEND'] = 'none'
    os.environ.setdefault('MINI_BATCHSIZE', '1')
    cfg = Config.fromfile(os.path.join(ROOT, 'uhydra/configs/uhydra_hrnet.py'))
    cfg.model.backbone.init_cfg = None
    cfg.model.convert_to_smpl = False
    cfg.model.compute_iou = False
    model = build_architecture(cfg.model)
    if ckpt:
        sd = torch.load(ckpt, map_location='cpu')['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.cuda().eval()


def count_params(model):
    n = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith('smplx'))
    return n / 1e6


@torch.no_grad()
def forward_once(model, img, K_intr, with_render=True):
    cand = model.forward_candidates(img, K_intr)
    if with_render:
        sil = model.render_candidates(cand, soft=False)
        from uhydra.models.geometry import hard_iou
        B, Kh = cand['tz'].shape
        S = torch.ones_like(sil[:, 0])
        l = 1 - hard_iou(sil.reshape(B * Kh, *sil.shape[2:]),
                         S[:, None].expand(B, Kh, -1, -1).reshape(B * Kh, *S.shape[1:])).view(B, Kh)
        _ = l.argmin(1)
    return cand


def gflops(model, img, K_intr, with_render):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return float('nan')

    class W(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, img, K_intr):
            return forward_once(s.m, img, K_intr, with_render)['verts']
    fca = FlopCountAnalysis(W(model), (img, K_intr))
    fca.unsupported_ops_warnings(False); fca.uncalled_modules_warnings(False)
    return fca.total() / 1e9   # NOTE: the rasteriser is a custom CUDA op and is not counted


def latency_and_memory(model, img, K_intr, with_render, n=100, warm=20):
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warm):
        forward_once(model, img, K_intr, with_render)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        forward_once(model, img, K_intr, with_render)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return float(np.median(ts)), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    img = torch.randn(1, 3, 224, 224).cuda()
    K_intr = torch.tensor([[[1000., 0, 112.], [0, 1000., 112.], [0, 0, 1.]]]).cuda()

    rows = [('Baseline (single depth, no rendering)', 'single', 1, False),
            ('+ DCME (K=1)', 'single', 1, True),
            (f'Full U-HYDRA (K={args.K})', 'learned', args.K, True)]
    print(f"{'Variant':40s} {'Params(M)':>10s} {'GFLOPs':>8s} {'Mem(GB)':>8s} {'Latency(ms)':>12s}")
    for name, placement, K, render in rows:
        model = build(args.width, K, placement, args.checkpoint)
        p = count_params(model)
        g = gflops(model, img, K_intr, render)
        lat, mem = latency_and_memory(model, img, K_intr, render)
        print(f'{name:40s} {p:10.1f} {g:8.1f} {mem:8.2f} {lat:12.1f}')
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

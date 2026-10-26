"""Multi-person in-the-wild inference (Fig. 1 right).

Every detected person is cropped to a square 224x224 crop around its box; the
crop intrinsics K are derived from the image focal length (or a default
f = 1.2 * max(H, W) if unknown) so that the principal point is carried into
the crop.  Each crop is an independent instance i: U-HYDRA reasons over K
depth hypotheses per person and the selected candidate is placed back into
the common camera frame with its full camera translation T_{k*}.

    python uhydra/tools/demo_multiperson.py --checkpoint ckpt.pth --images demo_images/ \
        --out results/demo [--focal 1500]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def detect_people(img_rgb_uint8: np.ndarray, thr=0.7):
    """torchvision Faster R-CNN person boxes (x1, y1, x2, y2)."""
    import torchvision
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT').eval()
    x = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = det([x])[0]
    keep = (out['labels'] == 1) & (out['scores'] > thr)
    return out['boxes'][keep].numpy()


def crop_with_intrinsics(img: np.ndarray, box, K_img: np.ndarray, size=224, pad=1.2):
    """Square crop around box; returns the normalised crop tensor and the crop intrinsics."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * pad
    x0, y0 = cx - s / 2, cy - s / 2
    import cv2
    M = np.array([[size / s, 0, -x0 * size / s], [0, size / s, -y0 * size / s]], dtype=np.float32)
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
    K = K_img.copy()
    K[0, 0] *= size / s; K[1, 1] *= size / s
    K[0, 2] = (K[0, 2] - x0) * size / s
    K[1, 2] = (K[1, 2] - y0) * size / s
    t = torch.from_numpy(crop).permute(2, 0, 1).float()[None]
    t = (t - MEAN) / STD
    return t, torch.from_numpy(K)[None].float(), crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', default='results/demo')
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--focal', type=float, default=None, help='pixel focal length; default 1.2*max(H,W)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from mmcv import Config
    from blade.models.architectures.builder import build_architecture
    import cv2
    os.environ['UHYDRA_WIDTH'] = str(args.width)
    os.environ.setdefault('MINI_BATCHSIZE', '1')
    cfg = Config.fromfile(os.path.join(ROOT, 'uhydra/configs/uhydra_hrnet.py'))
    cfg.model.backbone.init_cfg = None
    cfg.model.convert_to_smpl = False
    cfg.model.compute_iou = False
    model = build_architecture(cfg.model)
    sd = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    model = model.cuda().eval()

    files = sorted(glob.glob(os.path.join(args.images, '*')))
    for fn in files:
        bgr = cv2.imread(fn)
        if bgr is None:
            continue
        img = bgr[..., ::-1].copy()
        H, W = img.shape[:2]
        f = args.focal or 1.2 * max(H, W)
        K_img = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
        boxes = detect_people(img)
        results = []
        for b in boxes:
            crop_t, K_t, crop_u8 = crop_with_intrinsics(img, b, K_img)
            with torch.no_grad():
                cand = model.forward_candidates(crop_t.cuda(), K_t.cuda())
                sil = model.render_candidates(cand, soft=False)
                S, has = model.mask_provider(torch.from_numpy(crop_u8).permute(2, 0, 1)[None].float().cuda(),
                                             model.render_size)
                from uhydra.models.geometry import hard_iou
                from uhydra.models.selection import select_candidates
                from uhydra.models.losses import gather_winner
                Kh = cand['tz'].shape[1]
                l_al = 1 - hard_iou(sil.reshape(Kh, *sil.shape[2:]), S.expand(Kh, -1, -1)).view(1, Kh)
                k = select_candidates(l_al, rule='image', has_mask=has)
            results.append(dict(box=b.tolist(), k_star=int(k), sigma=float(cand['sigma']),
                                tz_hypotheses=cand['tz'][0].tolist(),
                                transl=gather_winner(cand['transl'], k)[0].tolist(),
                                betas=gather_winner(cand['betas'], k)[0].tolist(),
                                verts=gather_winner(cand['verts'], k)[0].cpu().numpy()))
        np.save(os.path.join(args.out, os.path.splitext(os.path.basename(fn))[0] + '.npy'),
                np.array(results, dtype=object), allow_pickle=True)
        print(f'{os.path.basename(fn)}: {len(results)} persons, '
              + ', '.join(f"Tz={r['transl'][2]:.2f}m (sigma={r['sigma']:.2f}, k*={r['k_star']})" for r in results))


if __name__ == '__main__':
    main()

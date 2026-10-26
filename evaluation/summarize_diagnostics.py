"""Aggregate the per-batch diagnostics written by UHYDRA.forward_test
(``diag_dir`` / env UHYDRA_DIAG_DIR) into the numbers reported in the paper:
mean sigma, coverage (%), oracle gap on root depth, z-error, mIoU, and the
histogram of selected hypothesis indices.

    python uhydra/evaluation/summarize_diagnostics.py work_dirs/uhydra_w48/eval/diag --K 5
"""
import argparse
import glob
import json
import os

import numpy as np

from uhydra.evaluation.metrics import coverage, nominal_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('diag_dir')
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.diag_dir, 'batch_*.npz')))
    assert files, f'no diagnostics in {args.diag_dir}'
    acc = {}
    for fn in files:
        d = np.load(fn)
        for k in d.files:
            acc.setdefault(k, []).append(d[k])
    acc = {k: np.concatenate(v, 0) for k, v in acc.items()}

    tz_gt = None
    summary = dict(n=int(len(acc['image_idx'])), K=args.K, nominal_coverage=nominal_coverage(args.K))
    if 'sigma' in acc:
        summary['sigma_mean'] = float(acc['sigma'].mean())
    if 'coverage' in acc:
        summary['coverage'] = float(acc['coverage'].mean() * 100.0)
    if 'z_error' in acc:
        summary['z_error_m'] = float(acc['z_error'].mean())
    if 'oracle_gap_tz' in acc:
        summary['oracle_gap_tz_m'] = float(acc['oracle_gap_tz'].mean())
    if 'raw_batch_miou' in acc:
        summary['miou'] = float(acc['raw_batch_miou'].mean())
    if 'k_star' in acc:
        hist = np.bincount(acc['k_star'].astype(int), minlength=args.K)
        summary['k_star_hist'] = (hist / hist.sum()).round(3).tolist()
    if 'tz_hypotheses' in acc:
        spread = np.log(acc['tz_hypotheses'][:, -1]) - np.log(acc['tz_hypotheses'][:, 0])
        summary['log_depth_spread_mean'] = float(spread.mean())

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()

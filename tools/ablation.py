"""Ablation runner for Sec. 3.3 / Fig. 3 (PDHuman, HRNet-W48).

Blocks and cost
  (a) hypothesis placement  : single (K=1) [train], fixed sweep [train], learned (ours)
  (b) number of hypotheses  : K in {1, 3, 5, 7}           [train each]
  (c) selection rule        : random / median / image (ours) / oracle   [evaluation only, same checkpoint]
  (d) depth supervision     : L1 / NLL w/o L_sigma / NLL + L_sigma (ours)  [train each]

Every variant is a plain BLADE train/test run with environment switches, so
the results land in the same results.json / diagnostics format and can be
collected with ``--collect``.

    python uhydra/tools/ablation.py --stage train --block a b d        # launches training jobs
    python uhydra/tools/ablation.py --stage eval  --block a b c d      # evaluates on PDHuman
    python uhydra/tools/ablation.py --collect                          # prints the table
"""
import argparse
import json
import os
import subprocess
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WD = os.path.join(ROOT, 'work_dirs', 'ablation')
CFG = 'uhydra/configs/uhydra_hrnet.py'
DATA = 'pdhuman_p5'

# name -> environment overrides (training variants)
TRAIN_VARIANTS = {
    'ours_K5':        dict(UHYDRA_K='5', UHYDRA_PLACEMENT='learned', UHYDRA_DEPTH_SUP='nll'),
    'single_K1':      dict(UHYDRA_PLACEMENT='single'),
    'fixed_sweep':    dict(UHYDRA_PLACEMENT='fixed'),
    'K3':             dict(UHYDRA_K='3'),
    'K7':             dict(UHYDRA_K='7'),
    'depth_l1':       dict(UHYDRA_DEPTH_SUP='l1'),
    'depth_nll_nosig': dict(UHYDRA_DEPTH_SUP='nll_nosigma'),
}
BLOCKS = {
    'a': ['single_K1', 'fixed_sweep', 'ours_K5'],
    'b': ['single_K1', 'K3', 'ours_K5', 'K7'],
    'c': ['ours_K5'],                                   # selection rules evaluated on ours_K5
    'd': ['depth_l1', 'depth_nll_nosig', 'ours_K5'],
}
SELECTION_RULES = ['random', 'median', 'image', 'oracle']


def run(cmd, env):
    e = os.environ.copy()
    e.update(env)
    e.setdefault('UHYDRA_WIDTH', '48')
    e.setdefault('MINI_BATCHSIZE', '22')
    print('>>', ' '.join(cmd), {k: v for k, v in env.items()})
    subprocess.run(cmd, cwd=ROOT, env=e, check=True)


def train(name):
    env = TRAIN_VARIANTS[name]
    out = os.path.join(WD, name)
    run(['python', '-m', 'torch.distributed.launch', f'--nproc_per_node={os.environ.get("NGPU", 3)}',
         'scripts/train.py', CFG, '--work-dir', out, '--launcher', 'pytorch', '--seed', '0'], env)


def evaluate(name, rule='image'):
    env = dict(TRAIN_VARIANTS[name])
    env['UHYDRA_SELECTION'] = rule
    env['MINI_BATCHSIZE'] = '16'
    ckpt = sorted(glob.glob(os.path.join(WD, name, 'epoch_*.pth')),
                  key=lambda p: int(p.split('_')[-1].split('.')[0]))[-1]
    out = os.path.join(WD, name, f'eval_{rule}')
    env['UHYDRA_DIAG_DIR'] = os.path.join(out, 'diag')
    run(['python', 'scripts/test.py', CFG, '--checkpoint', ckpt, '--data-name', DATA,
         '--work-dir', out, '--out', os.path.join(out, 'res.pkl')], env)
    K = env.get('UHYDRA_K', '5') if env.get('UHYDRA_PLACEMENT', 'learned') == 'learned' else \
        ('1' if env.get('UHYDRA_PLACEMENT') == 'single' else '5')
    run(['python', 'uhydra/evaluation/summarize_diagnostics.py', env['UHYDRA_DIAG_DIR'],
         '--K', K, '--out', os.path.join(out, 'diag.json')], env)


def collect():
    rows = []
    for name in TRAIN_VARIANTS:
        for d in sorted(glob.glob(os.path.join(WD, name, 'eval_*'))):
            rule = os.path.basename(d).replace('eval_', '')
            res = glob.glob(os.path.join(d, '*results.json')) + glob.glob(os.path.join(d, 'results.json'))
            diag = os.path.join(d, 'diag.json')
            r = json.load(open(res[0])) if res else {}
            g = json.load(open(diag)) if os.path.exists(diag) else {}
            rows.append((name, rule, r.get('pa-mpjpe', float('nan')), r.get('mpjpe', float('nan')),
                         r.get('pve', float('nan')), g.get('coverage', float('nan')),
                         g.get('sigma_mean', float('nan'))))
    print(f"{'variant':18s} {'rule':8s} {'PA-MPJPE':>9s} {'MPJPE':>7s} {'PVE':>7s} {'Cov%':>6s} {'sigma':>6s}")
    for r in rows:
        print(f'{r[0]:18s} {r[1]:8s} {r[2]:9.1f} {r[3]:7.1f} {r[4]:7.1f} {r[5]:6.1f} {r[6]:6.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'eval'], default=None)
    ap.add_argument('--block', nargs='+', default=['a', 'b', 'c', 'd'])
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    if args.collect:
        collect()
        return
    names = []
    for b in args.block:
        names += [n for n in BLOCKS[b] if n not in names]
    if args.stage == 'train':
        for n in names:
            train(n)
    elif args.stage == 'eval':
        for n in names:
            evaluate(n, 'image')
        if 'c' in args.block:
            for rule in SELECTION_RULES:
                if rule != 'image':
                    evaluate('ours_K5', rule)


if __name__ == '__main__':
    main()

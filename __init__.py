"""U-HYDRA: Uncertainty-Aware Multi-Hypothesis Depth Reasoning for
Multi-Person Perspective Distortion Mesh Recovery.

This package is a drop-in extension of the BLADE repository.  It re-uses
BLADE's data pipelines, body models, SMPL-X -> SMPL evaluation conversion and
the mmcv training / testing scripts, and replaces the two-stage DepthNet +
ControlNet-PoseNet with the single end-to-end U-HYDRA model described in the
ICASSP 2027 paper:

    Sec. 2.1  Multi-Hypothesis Depth Reasoning (MHDR)      -> models/mhdr.py
    Sec. 2.2  Depth-Conditioned Mesh Estimation (DCME)     -> models/dcme.py
              full-perspective projection + silhouettes    -> models/geometry.py
              candidate selection (Eq. 3)                   -> models/selection.py
    Sec. 2.3  Training objective (Eqs. 5-7)                 -> models/losses.py
    Sec. 3    Experiments (metrics, ablation, cost)         -> evaluation/, tools/

Install:  copy this folder next to ``blade/`` in the BLADE repo root, then
          ``pip install -e .`` is NOT needed -- the configs use
          ``custom_imports`` to register the modules with mmcv.
"""

__version__ = '1.0.0'

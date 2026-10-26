from .mhdr import MHDR, DistributionalDepthHead, HypothesisProposal, standard_normal_quantiles
from .dcme import DepthConditionedMeshHead
from .selection import select_candidates
from .geometry import SilhouetteRenderer, soft_iou, hard_iou, perspective_project

# The mmcv architecture needs the full BLADE environment (mmhuman3d, smplx,
# pytorch3d).  Import it lazily so that the pure-tensor modules above can be
# used and unit-tested on their own; the configs import
# ``uhydra.models.architecture`` explicitly through ``custom_imports``.
try:
    from .architecture import UHYDRA                      # registers 'UHYDRA' with mmcv
except ImportError as _e:                                 # pragma: no cover
    UHYDRA = None
    _ARCHITECTURE_IMPORT_ERROR = _e

__all__ = ['UHYDRA', 'MHDR', 'DistributionalDepthHead', 'HypothesisProposal',
           'standard_normal_quantiles', 'DepthConditionedMeshHead', 'select_candidates',
           'SilhouetteRenderer', 'soft_iou', 'hard_iou', 'perspective_project']

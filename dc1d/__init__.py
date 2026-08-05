"""
dc1d -- 1D deformable convolution in pure PyTorch.

Author: William Ravenscroft, August 2022
Copyright William Ravenscroft 2022
"""

from dc1d.nn import DeformConv1d, PackedDeformConv1d, cLN, gLN
from dc1d.ops import (
    efficient_linterpolate,
    full_seq_linterpolate,
    kernel_width_linterpolate,
)

__version__ = "0.2.0"

__all__ = [
    "DeformConv1d",
    "PackedDeformConv1d",
    "cLN",
    "gLN",
    "efficient_linterpolate",
    "full_seq_linterpolate",
    "kernel_width_linterpolate",
    "__version__",
]

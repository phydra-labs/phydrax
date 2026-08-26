#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._cayley_dickson import (
    CayleyDicksonAlgebraSpec,
    ComplexAlgebraSpec,
    OctonionAlgebraSpec,
    QuaternionAlgebraSpec,
    RealAlgebraSpec,
)
from ._multicomplex import MulticomplexAlgebraSpec


__all__ = [
    "CayleyDicksonAlgebraSpec",
    "ComplexAlgebraSpec",
    "MulticomplexAlgebraSpec",
    "OctonionAlgebraSpec",
    "QuaternionAlgebraSpec",
    "RealAlgebraSpec",
]

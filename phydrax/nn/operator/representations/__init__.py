"""Static physical representations used by neural operators."""

from ._clifford import CliffordGradeFeatures, CliffordGradeRepresentation
from ._groups import FiniteOrthogonalGroup
from ._irreps import (
    o3_irrep_action,
    o3_real_coupling,
    O3IrrepBlock,
    O3IrrepLayout,
    O3Parity,
    O3RealCoupling,
)
from ._o3 import O3Features, O3Representation
from ._tensor import (
    TensorFieldBlock,
    TensorFieldLayout,
    TensorParity,
    TensorType,
    TensorVariance,
)


__all__ = [
    "CliffordGradeFeatures",
    "CliffordGradeRepresentation",
    "FiniteOrthogonalGroup",
    "O3Features",
    "O3IrrepBlock",
    "O3IrrepLayout",
    "O3Parity",
    "O3RealCoupling",
    "O3Representation",
    "TensorFieldBlock",
    "TensorFieldLayout",
    "TensorParity",
    "TensorType",
    "TensorVariance",
    "o3_irrep_action",
    "o3_real_coupling",
]

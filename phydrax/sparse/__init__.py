#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""JAX-native sparse relations, derivative plans, routing kernels, and actions."""

from ._coloring import (
    SparseColoring,
    SparseDerivativeCompiler,
    SparseDerivativeKind,
    SparseDerivativeMode,
    SparseHessianMode,
    SparseJacobianMode,
)
from ._derivative import (
    compile_sparse_hessian,
    compile_sparse_jacobian,
    prepare_sparse_linearization,
    PreparedSparseDerivative,
    SparseDerivativePlan,
    SparseDerivativeVerification,
    verify_sparse_derivative,
)
from ._linear import LinearAction, SparseCoordinateOperator, SparseLinearMap
from ._ops import (
    gather_routes,
    linear_adjoint_apply,
    linear_apply,
    linear_transpose_apply,
    mask_routes,
    route_reduce,
    RouteReduction,
)
from ._pattern import SparsePattern, SparsePatternOrigin
from ._relation import EdgeRelation, RowRelation, SparseRelation


__all__ = [
    "LinearAction",
    "PreparedSparseDerivative",
    "EdgeRelation",
    "RouteReduction",
    "RowRelation",
    "SparseColoring",
    "SparseCoordinateOperator",
    "SparseDerivativeCompiler",
    "SparseDerivativeKind",
    "SparseDerivativeMode",
    "SparseDerivativePlan",
    "SparseDerivativeVerification",
    "SparseHessianMode",
    "SparseJacobianMode",
    "SparseLinearMap",
    "SparsePattern",
    "SparsePatternOrigin",
    "SparseRelation",
    "compile_sparse_hessian",
    "compile_sparse_jacobian",
    "gather_routes",
    "linear_adjoint_apply",
    "linear_apply",
    "linear_transpose_apply",
    "mask_routes",
    "prepare_sparse_linearization",
    "route_reduce",
    "verify_sparse_derivative",
]

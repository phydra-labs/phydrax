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
    SparseDerivativePrecisionPolicy,
    SparseDerivativeVerification,
    SparseHessianContract,
    verify_sparse_derivative,
)
from ._linear import LinearAction, SparseCoordinateOperator, SparseLinearMap
from ._local_tensor import ElementTensorOperator, scatter_local
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
    "ElementTensorOperator",
    "SparseDerivativeCompiler",
    "SparseDerivativeKind",
    "SparseDerivativeMode",
    "SparseDerivativePlan",
    "SparseDerivativeVerification",
    "SparseDerivativePrecisionPolicy",
    "SparseHessianMode",
    "SparseHessianContract",
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
    "scatter_local",
    "route_reduce",
    "verify_sparse_derivative",
]

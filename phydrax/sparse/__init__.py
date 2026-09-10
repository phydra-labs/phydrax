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
from ._execution import (
    canonical_row_route_ids,
    RelationAccumulation,
    RelationExecutionPlan,
    RelationExecutionState,
    RelationOutput,
    RelationReduction,
    RelationReductionEvidence,
)
from ._key_groups import (
    align_key_groups,
    KeyGroupEvidence,
    KeyGroupLookup,
    KeyGroupPlan,
    KeyGroupState,
    KeyGroupTransition,
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
    "KeyGroupEvidence",
    "KeyGroupLookup",
    "KeyGroupPlan",
    "KeyGroupState",
    "KeyGroupTransition",
    "canonical_row_route_ids",
    "RelationAccumulation",
    "RelationExecutionPlan",
    "RelationExecutionState",
    "RelationOutput",
    "RelationReduction",
    "RelationReductionEvidence",
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
    "align_key_groups",
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

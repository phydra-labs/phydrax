#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded tensor-train, quantics, structured-operator, and solver workflows."""

from ._completion import (
    TensorCompletionEvidence,
    TensorCompletionPlan,
    TensorCompletionResult,
    weighted_tensor_completion,
)
from ._core import (
    round_tensor_train,
    TensorTrain,
    TensorTrainCompressionResult,
    TensorTrainOperator,
    TensorTrainOperatorCompressionResult,
    tt_svd,
    TTRoundingEvidence,
)
from ._cross import tensor_train_cross, TTCrossEvidence, TTCrossPlan, TTCrossResult
from ._eigen import (
    BlockTensorTrainEigenEvidence,
    BlockTensorTrainEigenPlan,
    BlockTensorTrainEigenResult,
    smallest_eigenpairs,
)
from ._linear import TensorTrainLinear, TensorTrainLinearCompressionEvidence
from ._quantics import (
    DigitOrdering,
    GridRule,
    qtt_digitize,
    qtt_evaluate,
    qtt_quadrature,
    qtt_sample,
    QuanticsLayout,
    TensorFunction,
    TensorizedGrid,
)
from ._solvers import (
    plan_als,
    plan_amen,
    plan_tensor_train_solve,
    prepare_tensor_train_solve,
    PreparedTensorTrainSolve,
    refresh_tensor_train_solve,
    solve_tensor_train,
    TensorTrainSolveEvidence,
    TensorTrainSolveMethod,
    TensorTrainSolvePlan,
    TensorTrainSolveResult,
)
from ._structured import (
    BoundaryKind,
    BoundaryPolicy,
    cartesian_identity,
    diagonal_operator,
    identity_operator,
    kronecker_operator,
    laplacian_operator,
    shift_operator,
)


__all__ = [
    "BlockTensorTrainEigenEvidence",
    "BlockTensorTrainEigenPlan",
    "BlockTensorTrainEigenResult",
    "BoundaryKind",
    "BoundaryPolicy",
    "DigitOrdering",
    "GridRule",
    "PreparedTensorTrainSolve",
    "QuanticsLayout",
    "TTCrossEvidence",
    "TTCrossPlan",
    "TTCrossResult",
    "TTRoundingEvidence",
    "TensorCompletionEvidence",
    "TensorCompletionPlan",
    "TensorCompletionResult",
    "TensorFunction",
    "TensorTrain",
    "TensorTrainCompressionResult",
    "TensorTrainLinear",
    "TensorTrainLinearCompressionEvidence",
    "TensorTrainOperator",
    "TensorTrainOperatorCompressionResult",
    "TensorTrainSolveEvidence",
    "TensorTrainSolveMethod",
    "TensorTrainSolvePlan",
    "TensorTrainSolveResult",
    "TensorizedGrid",
    "cartesian_identity",
    "diagonal_operator",
    "identity_operator",
    "kronecker_operator",
    "laplacian_operator",
    "plan_als",
    "plan_amen",
    "plan_tensor_train_solve",
    "prepare_tensor_train_solve",
    "qtt_digitize",
    "qtt_evaluate",
    "qtt_quadrature",
    "qtt_sample",
    "refresh_tensor_train_solve",
    "round_tensor_train",
    "shift_operator",
    "smallest_eigenpairs",
    "solve_tensor_train",
    "tensor_train_cross",
    "tt_svd",
    "weighted_tensor_completion",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._preconditioning import PreparedPreconditioner
from .._spaces import _coordinate_dtype
from ._plans import EigenSolvePlan
from ._policies import DenseEigh
from ._problems import Eigenproblem, EigenproblemLike, GeneralizedEigenproblem


class DenseEigenState(StrictModule):
    """Prepared paired-coordinate reduction for a dense Hermitian problem."""

    reduced_operator: Array
    metric_factor: Array
    operator_matvec_count: int = eqx.field(static=True)
    metric_matvec_count: int = eqx.field(static=True)

    def __init__(
        self,
        reduced_operator: ArrayLike,
        metric_factor: ArrayLike,
        /,
        *,
        operator_matvec_count: int,
        metric_matvec_count: int,
    ):
        reduced = jnp.asarray(reduced_operator)
        factor = jnp.asarray(metric_factor)
        if (
            reduced.ndim != 2
            or reduced.shape[0] != reduced.shape[1]
            or factor.shape != reduced.shape
            or factor.dtype != reduced.dtype
        ):
            raise ValueError(
                "Dense eigen reduction and metric factor must be matching square arrays."
            )
        counts = int(operator_matvec_count), int(metric_matvec_count)
        if any(value < 0 for value in counts):
            raise ValueError("Dense eigen action counts must be non-negative.")
        self.reduced_operator = reduced
        self.metric_factor = factor
        self.operator_matvec_count, self.metric_matvec_count = counts


class PreparedEigenSolve(StrictModule):
    """Fixed-shape coordinate state bound to one problem and symbolic plan."""

    problem: EigenproblemLike
    plan: EigenSolvePlan
    initial_basis: Array
    constraint_basis: Array
    metric_constraint_basis: Array
    preconditioning_state: PreparedPreconditioner | None
    dense_state: DenseEigenState | None
    initial_rank: Array
    symbolic_version: int = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        problem: EigenproblemLike,
        plan: EigenSolvePlan,
        initial_basis: ArrayLike,
        constraint_basis: ArrayLike,
        metric_constraint_basis: ArrayLike,
        /,
        *,
        preconditioning_state: PreparedPreconditioner | None = None,
        dense_state: DenseEigenState | None = None,
        initial_rank: ArrayLike | None = None,
        symbolic_version: int = 1,
        numeric_version: int = 0,
    ):
        if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
            raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")
        if not isinstance(plan, EigenSolvePlan):
            raise TypeError("plan must be an EigenSolvePlan.")
        if plan.problem_id != problem.problem_id:
            raise ValueError("Prepared problem and plan IDs must match.")
        symbolic = int(symbolic_version)
        numeric = int(numeric_version)
        if symbolic != 1:
            raise ValueError(
                "symbolic_version must match the eigen plan schema version 1."
            )
        if numeric < 0:
            raise ValueError("numeric_version must be non-negative.")
        n = problem.dimension
        capacity = 0 if problem.constraints is None else problem.constraints.capacity
        dtype = _coordinate_dtype(problem.operator.source)
        initial = _coordinate_matrix(
            initial_basis,
            (n, plan.block_dimension),
            dtype,
            "initial_basis",
        )
        constraints = _coordinate_matrix(
            constraint_basis,
            (n, capacity),
            dtype,
            "constraint_basis",
        )
        metric_constraints = _coordinate_matrix(
            metric_constraint_basis,
            (n, capacity),
            dtype,
            "metric_constraint_basis",
        )
        rank = jnp.asarray(
            plan.block_dimension if initial_rank is None else initial_rank,
            dtype=jnp.int32,
        )
        if rank.shape != ():
            raise ValueError("initial_rank must be scalar.")
        rank = eqx.error_if(
            rank,
            (rank < 0) | (rank > plan.block_dimension),
            "initial_rank must lie between zero and block_dimension.",
        )
        expected_preconditioner = plan.preconditioner_plan
        if expected_preconditioner is None:
            if preconditioning_state is not None:
                raise ValueError(
                    "A plan without preconditioning cannot own preconditioning_state."
                )
        else:
            if not isinstance(preconditioning_state, PreparedPreconditioner):
                raise TypeError(
                    "A preconditioned plan requires PreparedPreconditioner state."
                )
            if preconditioning_state.plan.plan_id != expected_preconditioner.plan_id:
                raise ValueError("Prepared preconditioner and eigen plan IDs must match.")
            if preconditioning_state.numeric_version != numeric:
                raise ValueError(
                    "Prepared preconditioner and eigen numeric versions must match."
                )
        if isinstance(plan.selected_method, DenseEigh):
            if not isinstance(dense_state, DenseEigenState):
                raise TypeError("DenseEigh preparation requires DenseEigenState.")
        elif dense_state is not None:
            raise ValueError("Iterative eigen preparation cannot own dense state.")
        self.problem = problem
        self.plan = plan
        self.initial_basis = initial
        self.constraint_basis = constraints
        self.metric_constraint_basis = metric_constraints
        self.preconditioning_state = preconditioning_state
        self.dense_state = dense_state
        self.initial_rank = rank
        self.symbolic_version = symbolic
        self.numeric_version = numeric


def _coordinate_matrix(
    value: ArrayLike,
    shape: tuple[int, int],
    dtype: np.dtype,
    name: str,
    /,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if np.dtype(array.dtype) != dtype:
        raise TypeError(f"{name} must have coordinate dtype {dtype}.")
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)),
        f"{name} entries must be finite.",
    )


__all__ = ["DenseEigenState", "PreparedEigenSolve"]

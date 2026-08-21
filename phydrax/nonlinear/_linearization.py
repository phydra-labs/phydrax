#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
    prepare_linearization,
    PyTreeSpace,
)
from ..sparse import (
    prepare_sparse_linearization,
    PreparedSparseDerivative,
    SparseDerivativePlan,
)
from ._types import NonlinearSystemProblem


JacobianMode: TypeAlias = Literal[
    "autodiff", "sparse", "directional-finite-difference", "explicit"
]


class JacobianPolicy(StrictModule):
    """Explicit source and numerical controls for nonlinear Jacobian actions."""

    mode: JacobianMode = eqx.field(static=True)
    sparse_plan: SparseDerivativePlan | None
    operator_function: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None
    finite_difference_step: float = eqx.field(static=True)

    def __init__(
        self,
        mode: JacobianMode = "autodiff",
        /,
        *,
        sparse_plan: SparseDerivativePlan | None = None,
        operator: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None = None,
        finite_difference_step: float = 1e-6,
    ):
        if mode not in (
            "autodiff",
            "sparse",
            "directional-finite-difference",
            "explicit",
        ):
            raise ValueError("Unknown Jacobian mode.")
        if mode == "sparse" and not isinstance(sparse_plan, SparseDerivativePlan):
            raise TypeError("Sparse Jacobian mode requires a SparseDerivativePlan.")
        if mode != "sparse" and sparse_plan is not None:
            raise ValueError("sparse_plan is only valid for sparse Jacobian mode.")
        if mode == "explicit" and not callable(operator):
            raise TypeError("Explicit Jacobian mode requires an operator callable.")
        if mode != "explicit" and operator is not None:
            raise ValueError("operator is only valid for explicit Jacobian mode.")
        step = float(finite_difference_step)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive.")
        self.mode = mode
        self.sparse_plan = sparse_plan
        self.operator_function = operator
        self.finite_difference_step = step

    @property
    def policy_id(self) -> str:
        return self.mode


class PreparedJacobian(StrictModule):
    """Residual value and one reusable Jacobian operator at an accepted state."""

    residual: PyTree
    auxiliary: Any
    operator: AbstractLinearOperator
    sparse_derivative: PreparedSparseDerivative | None
    derivative_id: str = eqx.field(static=True)
    residual_evaluations: int = eqx.field(static=True)

    def __init__(
        self,
        residual: PyTree,
        operator: AbstractLinearOperator,
        /,
        *,
        auxiliary: Any = None,
        sparse_derivative: PreparedSparseDerivative | None = None,
        derivative_id: str,
        residual_evaluations: int = 1,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if sparse_derivative is not None and not isinstance(
            sparse_derivative, PreparedSparseDerivative
        ):
            raise TypeError("sparse_derivative must be PreparedSparseDerivative or None.")
        identifier = str(derivative_id)
        if not identifier:
            raise ValueError("derivative_id must be non-empty.")
        self.residual = residual
        evaluations = int(residual_evaluations)
        if evaluations < 1:
            raise ValueError("residual_evaluations must be positive.")
        self.operator = operator
        self.sparse_derivative = sparse_derivative
        self.auxiliary = auxiliary
        self.derivative_id = identifier
        self.residual_evaluations = evaluations


class _CoordinateRebasedLinearOperator(AbstractLinearOperator):
    """Canonical coordinate endomorphism for an equal-size structured map."""

    operator: AbstractLinearOperator
    coordinate_space: ArraySpace

    def __init__(self, operator: AbstractLinearOperator, /):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if operator.batch_shape:
            raise ValueError("Coordinate rebasing requires an unbatched operator.")
        if operator.source.size != operator.target.size:
            raise ValueError(
                "Coordinate rebasing requires equal source and target dimensions."
            )
        source_coordinates = operator.source.flatten(operator.source.zeros())
        target_coordinates = operator.target.flatten(operator.target.zeros())
        if source_coordinates.dtype != target_coordinates.dtype:
            raise TypeError(
                "Coordinate rebasing requires matching source and target dtypes."
            )
        coordinate_space = ArraySpace(
            (operator.source.size,),
            dtype=source_coordinates.dtype,
        )
        self.operator = operator
        self.coordinate_space = coordinate_space
        self.source = coordinate_space
        self.target = coordinate_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.adjoint,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = f"{operator.operator_id}/canonical-coordinate-rebase"

    def flatten_target(self, vector: PyTree[Any], /) -> Array:
        return self.coordinate_space.validate(self.operator.target.flatten(vector))

    def unflatten_source(self, coordinates: Array, /) -> PyTree[Array]:
        value = self.coordinate_space.validate(coordinates)
        return self.operator.source.unflatten(value)

    def mv(self, vector: PyTree[Any], /) -> Array:
        state_direction = self.unflatten_source(vector)
        return self.flatten_target(self.operator.mv(state_direction))

    def transpose_mv(self, vector: PyTree[Any], /) -> Array:
        coordinates = self.coordinate_space.validate(vector)
        residual_direction = self.operator.target.unflatten(coordinates)
        transposed = self.operator.transpose_mv(residual_direction)
        return self.coordinate_space.validate(self.operator.source.flatten(transposed))

    def adjoint_mv(self, vector: PyTree[Any], /) -> Array:
        coordinates = self.coordinate_space.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(coordinates)))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "A coordinate-rebased Jacobian is matrix-free and cannot materialize."
        )


def _rebase_jacobian_coordinates(
    operator: AbstractLinearOperator,
    /,
) -> _CoordinateRebasedLinearOperator:
    return _CoordinateRebasedLinearOperator(operator)


def _jacobian_solve_operator(
    operator: AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    if operator.source.compatible(operator.target):
        return operator
    return _rebase_jacobian_coordinates(operator)


def _jacobian_solve_right_hand_side(
    operator: AbstractLinearOperator,
    residual: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return operator.flatten_target(jax.tree.map(jnp.negative, residual))
    return operator.target.validate(jax.tree.map(jnp.negative, residual))


def _jacobian_solve_direction(
    operator: AbstractLinearOperator,
    value: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return operator.unflatten_source(value)
    return operator.source.validate(value)


def prepare_jacobian(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    policy: JacobianPolicy,
    args: Any = None,
    /,
) -> PreparedJacobian:
    """Prepare one Jacobian action without silently changing derivative source."""
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    if not isinstance(policy, JacobianPolicy):
        raise TypeError("policy must be a JacobianPolicy.")
    source = PyTreeSpace(state) if problem.state_space is None else problem.state_space

    if policy.mode == "autodiff":
        function = (
            (lambda candidate: problem.evaluate(candidate, args))
            if problem.has_aux
            else (lambda candidate: problem.residual(candidate, args))
        )
        linearization = prepare_linearization(
            function,
            state,
            source=source,
            has_aux=problem.has_aux,
            target=problem.residual_space,
            linearization_id=f"{problem.problem_id}/jacobian/autodiff",
        )
        operator = JacobianLinearOperator(
            linearization,
            operator_id=f"{problem.problem_id}/jacobian",
        )
        return PreparedJacobian(
            linearization.primal,
            operator,
            auxiliary=linearization.auxiliary,
            derivative_id="autodiff",
        )

    if policy.mode == "sparse":
        sparse = prepare_sparse_linearization(policy.sparse_plan, state, args)
        if not source.compatible(sparse.operator.source):
            raise ValueError("Sparse Jacobian source must match the nonlinear state space.")
        if (
            problem.residual_space is not None
            and not problem.residual_space.compatible(sparse.operator.target)
        ):
            raise ValueError(
                "Sparse Jacobian target must match the nonlinear residual space."
            )
        if problem.has_aux:
            _, auxiliary = problem.evaluate(state, args)
        else:
            auxiliary = None
        return PreparedJacobian(
            sparse.linearization.primal,
            sparse.operator,
            auxiliary=auxiliary,
            sparse_derivative=sparse,
            derivative_id=f"sparse:{policy.sparse_plan.plan_id}",
            residual_evaluations=1 + int(problem.has_aux),
        )

    residual, auxiliary = problem.evaluate(state, args)
    target = (
        PyTreeSpace(residual)
        if problem.residual_space is None
        else problem.residual_space
    )
    residual = target.validate(residual)
    if policy.mode == "explicit":
        operator = policy.operator_function(state, args)
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError(
                "The explicit Jacobian callable must return a linear operator."
            )
        if not source.compatible(operator.source) or not target.compatible(
            operator.target
        ):
            raise ValueError(
                "Explicit Jacobian source and target spaces must match the residual."
            )
        return PreparedJacobian(
            residual,
            operator,
            auxiliary=auxiliary,
            derivative_id="explicit",
        )

    step = jnp.asarray(
        policy.finite_difference_step, dtype=source.flatten(state).real.dtype
    )

    def action(tangent):
        candidate = jax.tree.map(
            lambda value, delta: value + step * delta, state, tangent
        )
        shifted = problem.residual(candidate, args)
        return jax.tree.map(lambda new, old: (new - old) / step, shifted, residual)

    operator = FunctionLinearOperator(
        action,
        source=source,
        target=target,
        operator_id=f"{problem.problem_id}/jacobian/directional-finite-difference",
    )
    return PreparedJacobian(
        residual,
        operator,
        auxiliary=auxiliary,
        derivative_id="directional-finite-difference",
    )


__all__ = ["JacobianMode", "JacobianPolicy", "PreparedJacobian", "prepare_jacobian"]

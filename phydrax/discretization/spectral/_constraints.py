#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._orthogonal import (
    standard_derivative_matrix,
    standard_vandermonde,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLinearTransform,
    FactorizationPolicy,
    factorize,
)
from .._spectral import ModalTransform
from ._basis import (
    _analysis_from_synthesis,
    AbstractSpectralBasisPlan,
    PreparedSpectralAxis,
    SpectralModeLayout,
)
from ._precision import SpectralPrecisionPolicy


class EndpointConstraint(StrictModule, NonTrainableState):
    """One homogeneous endpoint derivative constraint."""

    side: Literal["lower", "upper"] = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        side: Literal["lower", "upper"],
        derivative_order: int = 0,
        /,
    ):
        order = int(derivative_order)
        if side not in ("lower", "upper") or order < 0:
            raise ValueError(
                "Endpoint constraints require a side and non-negative order."
            )
        self.side = side
        self.derivative_order = order
        self.constraint_id = canonical_fingerprint(
            {"kind": "endpoint-constraint", "side": side, "order": order}
        )


class SpectralBoundaryConditionPlan(StrictModule, NonTrainableState):
    """Independent homogeneous endpoint constraints for one polynomial basis."""

    constraints: tuple[EndpointConstraint, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, constraints: Sequence[EndpointConstraint], /):
        values = tuple(constraints)
        if not values or not all(
            isinstance(value, EndpointConstraint) for value in values
        ):
            raise TypeError("constraints must contain EndpointConstraint values.")
        keys = tuple((value.side, value.derivative_order) for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("Endpoint constraints must be unique.")
        self.constraints = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-boundary-condition-plan",
                "constraints": [value.constraint_id for value in values],
            }
        )

    @classmethod
    def dirichlet(cls) -> "SpectralBoundaryConditionPlan":
        return cls((EndpointConstraint("lower"), EndpointConstraint("upper")))

    @classmethod
    def neumann(cls) -> "SpectralBoundaryConditionPlan":
        return cls(
            (
                EndpointConstraint("lower", 1),
                EndpointConstraint("upper", 1),
            )
        )


def _basis_normalizers(
    prepared: PreparedSpectralAxis,
    /,
) -> np.ndarray:
    count = prepared.mode_count
    if prepared.family == "chebyshev":
        return np.ones((count,), dtype=float)
    if prepared.family == "legendre":
        length = float(np.asarray(prepared.length))
        return np.sqrt((2.0 * np.arange(count) + 1.0) / length)
    raise ValueError("Endpoint constraints require Chebyshev or Legendre bases.")


def _constraint_matrix(
    prepared: PreparedSpectralAxis,
    conditions: SpectralBoundaryConditionPlan,
    /,
) -> np.ndarray:
    count = prepared.mode_count
    normalizers = _basis_normalizers(prepared)
    length = float(np.asarray(prepared.length))
    matrix = np.zeros((len(conditions.constraints), count), dtype=float)
    for row, condition in enumerate(conditions.constraints):
        point = -1.0 if condition.side == "lower" else 1.0
        derivative = standard_derivative_matrix(
            prepared.family,
            count,
            condition.derivative_order,
            scale=2.0 / length,
            dtype=prepared.precision.physical_dtype,
        )
        evaluation = standard_vandermonde(
            prepared.family,
            jnp.asarray((point,)),
            count - 1,
        )[0]
        matrix[row] = np.asarray(evaluation @ derivative) * normalizers
    return matrix


def _canonical_columns(values: np.ndarray, /) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


class ConstrainedBasisPlan(AbstractSpectralBasisPlan):
    """Polynomial basis restricted to the nullspace of endpoint traces."""

    base: AbstractSpectralBasisPlan
    conditions: SpectralBoundaryConditionPlan
    base_mode_count: int = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractSpectralBasisPlan,
        conditions: SpectralBoundaryConditionPlan,
        /,
    ):
        if not isinstance(base, AbstractSpectralBasisPlan):
            raise TypeError("base must be an AbstractSpectralBasisPlan.")
        if base.family not in ("chebyshev", "legendre"):
            raise ValueError("Constrained bases initially require Chebyshev or Legendre.")
        if not isinstance(conditions, SpectralBoundaryConditionPlan):
            raise TypeError("conditions must be a SpectralBoundaryConditionPlan.")
        free = base.mode_count - len(conditions.constraints)
        if free <= 0:
            raise ValueError("Boundary constraints leave no spectral degrees of freedom.")
        orders = tuple(value.derivative_order for value in conditions.constraints)
        boundary = (
            "homogeneous_dirichlet"
            if orders == (0, 0)
            else "homogeneous_neumann"
            if orders == (1, 1)
            else "unconstrained"
        )
        self.base = base
        self.conditions = conditions
        self.base_mode_count = base.mode_count
        self.mode_count = free
        self.family = base.family
        self.periodic = False
        self.boundary = boundary
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constrained-spectral-basis-plan",
                "base": base.plan_id,
                "conditions": conditions.plan_id,
                "free_modes": free,
            }
        )

    def resized(self, mode_count: int, /) -> "ConstrainedBasisPlan":
        free = int(mode_count)
        return ConstrainedBasisPlan(
            self.base.resized(free + len(self.conditions.constraints)),
            self.conditions,
        )

    def prepare(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        base = self.base.prepare(lower, upper, precision=precision)
        if base.modal_transform is None:
            raise RuntimeError("Constrained polynomial base lacks dense modal metadata.")
        constraints = _constraint_matrix(base, self.conditions)
        constraint_operator = DenseLinearOperator(jnp.asarray(constraints))
        factorization = factorize(
            constraint_operator,
            FactorizationPolicy("svd"),
        )
        nullspace = factorization.right_nullspace()
        dimension = int(np.asarray(nullspace.dimension))
        stencil = _canonical_columns(np.asarray(nullspace.basis[:, :dimension]))
        if stencil.shape != (self.base_mode_count, self.mode_count):
            raise RuntimeError(
                "Boundary constraint matrix has unexpected numerical rank."
            )
        synthesis = np.asarray(base.modal_transform.synthesis) @ stencil
        analysis = _analysis_from_synthesis(
            synthesis,
            precision.coefficient_dtype,
        )
        weights = np.asarray(base.quadrature_weights)
        mode_ids = tuple(f"constrained:{index}" for index in range(self.mode_count))
        modal = ModalTransform(
            analysis,
            synthesis,
            weights,
            mode_ids=mode_ids,
        )
        execution = DenseLinearTransform(
            np.asarray(analysis, dtype=precision.coefficient_dtype),
            np.asarray(synthesis, dtype=precision.coefficient_dtype),
            transform_id=modal.transform_id,
        )
        return PreparedSpectralAxis(
            self,
            base.nodes,
            weights,
            base.bounds,
            SpectralModeLayout(
                self.family,
                np.arange(self.mode_count),
                mode_ids=mode_ids,
            ),
            execution,
            precision,
            modal_transform=modal,
        )


class BoundaryLiftPlan(StrictModule, NonTrainableState):
    """Minimum-norm polynomial lift for inhomogeneous endpoint data."""

    conditions: SpectralBoundaryConditionPlan
    values: Array
    lift_id: str = eqx.field(static=True)

    def __init__(
        self,
        conditions: SpectralBoundaryConditionPlan,
        values: ArrayLike,
        /,
    ):
        if not isinstance(conditions, SpectralBoundaryConditionPlan):
            raise TypeError("conditions must be a SpectralBoundaryConditionPlan.")
        values_ = jnp.asarray(values).reshape((-1,))
        if values_.shape != (len(conditions.constraints),):
            raise ValueError("Boundary lift values must match endpoint constraints.")
        if not jnp.issubdtype(values_.dtype, jnp.inexact):
            values_ = values_.astype(float)
        values_ = eqx.error_if(
            values_,
            jnp.any(~jnp.isfinite(values_)),
            "Boundary lift values must be finite.",
        )
        self.conditions = conditions
        self.values = values_
        self.lift_id = canonical_fingerprint(
            {
                "kind": "spectral-boundary-lift-plan",
                "conditions": conditions.plan_id,
                "values": array_tree_fingerprint(values_),
            }
        )

    def prepare(self, base: PreparedSpectralAxis, /) -> "PreparedBoundaryLift":
        matrix = _constraint_matrix(base, self.conditions)
        dtype = jnp.result_type(
            jnp.dtype(base.precision.coefficient_dtype),
            self.values.dtype,
        )
        operator = DenseLinearOperator(jnp.asarray(matrix, dtype=dtype))
        factorization = factorize(operator, FactorizationPolicy("svd"))
        result = factorization.solve(jnp.asarray(self.values, dtype=dtype))
        if not bool(result.successful):
            raise RuntimeError("Boundary lift minimum-norm solve did not converge.")
        coefficients = result.value
        values = base.synthesize(
            jnp.asarray(coefficients, dtype=base.precision.coefficient_dtype)
        )
        return PreparedBoundaryLift(
            self,
            base,
            coefficients=jnp.asarray(
                coefficients,
                dtype=base.precision.coefficient_dtype,
            ),
            values=values,
        )


class PreparedBoundaryLift(StrictModule, NonTrainableState):
    plan: BoundaryLiftPlan
    base: PreparedSpectralAxis
    coefficients: Array
    values: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BoundaryLiftPlan,
        base: PreparedSpectralAxis,
        /,
        *,
        coefficients: ArrayLike,
        values: ArrayLike,
    ):
        self.plan = plan
        self.base = base
        self.coefficients = jnp.asarray(coefficients)
        self.values = jnp.asarray(values)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-boundary-lift",
                "plan": plan.lift_id,
                "base": base.axis_id,
            }
        )


__all__ = [
    "BoundaryLiftPlan",
    "ConstrainedBasisPlan",
    "EndpointConstraint",
    "PreparedBoundaryLift",
    "SpectralBoundaryConditionPlan",
]

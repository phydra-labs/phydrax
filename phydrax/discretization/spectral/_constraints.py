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
from .._axis_domain import AxisDomain
from .._spectral import ModalTransform
from ._basis import (
    _analysis_from_synthesis,
    AbstractSpectralBasisPlan,
    PreparedSpectralAxis,
    SpectralBoundaryKind,
    SpectralModeLayout,
)
from ._precision import SpectralPrecisionPolicy


class SpectralTraceTerm(StrictModule, NonTrainableState):
    """One coefficient multiplying an endpoint derivative trace."""

    derivative_order: int = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(self, derivative_order: int, coefficient: float = 1.0, /):
        order = int(derivative_order)
        coefficient_ = float(coefficient)
        if order < 0 or not np.isfinite(coefficient_) or coefficient_ == 0.0:
            raise ValueError(
                "Trace terms require a non-negative order and finite nonzero coefficient."
            )
        self.derivative_order = order
        self.coefficient = coefficient_
        self.term_id = canonical_fingerprint(
            {
                "kind": "spectral-trace-term",
                "order": order,
                "coefficient": coefficient_,
            }
        )


class SpectralTraceConstraint(StrictModule, NonTrainableState):
    """One homogeneous linear trace on a finite or compactified endpoint."""

    terms: tuple[SpectralTraceTerm, ...]
    side: Literal["lower", "upper"] = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        side: Literal["lower", "upper"],
        terms: Sequence[SpectralTraceTerm],
        /,
    ):
        terms_input = tuple(terms)
        if not terms_input or not all(
            isinstance(term, SpectralTraceTerm) for term in terms_input
        ):
            raise TypeError("terms must contain SpectralTraceTerm values.")
        terms_ = tuple(sorted(terms_input, key=lambda term: term.derivative_order))
        if side not in ("lower", "upper"):
            raise ValueError("Trace constraints require a lower or upper side.")
        orders = tuple(term.derivative_order for term in terms_)
        if len(set(orders)) != len(orders):
            raise ValueError("Trace terms must use unique derivative orders.")
        self.terms = terms_
        self.side = side
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "spectral-trace-constraint",
                "side": side,
                "terms": [term.term_id for term in terms_],
            }
        )

    @classmethod
    def derivative(
        cls,
        side: Literal["lower", "upper"],
        derivative_order: int = 0,
        /,
    ) -> "SpectralTraceConstraint":
        return cls(side, (SpectralTraceTerm(derivative_order),))

    @property
    def derivative_orders(self) -> tuple[int, ...]:
        return tuple(term.derivative_order for term in self.terms)


class SpectralBoundaryConditionPlan(StrictModule, NonTrainableState):
    """Independent homogeneous linear traces for one spectral basis."""

    constraints: tuple[SpectralTraceConstraint, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, constraints: Sequence[SpectralTraceConstraint], /):
        values = tuple(constraints)
        if not values or not all(
            isinstance(value, SpectralTraceConstraint) for value in values
        ):
            raise TypeError("constraints must contain SpectralTraceConstraint values.")
        identifiers = tuple(value.constraint_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Trace constraints must be unique.")
        self.constraints = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-boundary-condition-plan",
                "constraints": list(identifiers),
            }
        )

    @classmethod
    def dirichlet(cls) -> "SpectralBoundaryConditionPlan":
        return cls(
            (
                SpectralTraceConstraint.derivative("lower"),
                SpectralTraceConstraint.derivative("upper"),
            )
        )

    @classmethod
    def neumann(cls) -> "SpectralBoundaryConditionPlan":
        return cls(
            (
                SpectralTraceConstraint.derivative("lower", 1),
                SpectralTraceConstraint.derivative("upper", 1),
            )
        )

    @classmethod
    def robin(
        cls,
        /,
        *,
        lower: tuple[float, float],
        upper: tuple[float, float],
    ) -> "SpectralBoundaryConditionPlan":
        def constraint(side, coefficients):
            terms = tuple(
                SpectralTraceTerm(order, coefficient)
                for order, coefficient in enumerate(coefficients)
                if float(coefficient) != 0.0
            )
            return SpectralTraceConstraint(side, terms)

        return cls((constraint("lower", lower), constraint("upper", upper)))

    @classmethod
    def decay(
        cls,
        sides: Sequence[Literal["lower", "upper"]] = ("lower", "upper"),
        /,
    ) -> "SpectralBoundaryConditionPlan":
        return cls(tuple(SpectralTraceConstraint.derivative(side) for side in sides))


def _basis_normalizers(
    prepared: PreparedSpectralAxis,
    /,
) -> np.ndarray:
    count = prepared.mode_count
    if prepared.family in (
        "chebyshev",
        "rational_chebyshev_line",
        "rational_chebyshev_half_line",
    ):
        return np.ones((count,), dtype=float)
    if prepared.family == "legendre":
        length = float(np.asarray(prepared.length))
        return np.sqrt((2.0 * np.arange(count) + 1.0) / length)
    raise ValueError("The prepared basis does not expose polynomial trace rows.")


def _trace_row(
    prepared: PreparedSpectralAxis,
    constraint: SpectralTraceConstraint,
    /,
) -> np.ndarray:
    count = prepared.mode_count
    point = -1.0 if constraint.side == "lower" else 1.0
    normalizers = _basis_normalizers(prepared)
    if prepared.family in ("chebyshev", "legendre"):
        length = float(np.asarray(prepared.length))
        evaluation = standard_vandermonde(
            prepared.family,
            jnp.asarray((point,)),
            count - 1,
        )[0]
        row = np.zeros((count,), dtype=float)
        for term in constraint.terms:
            derivative = standard_derivative_matrix(
                prepared.family,
                count,
                term.derivative_order,
                scale=2.0 / length,
                dtype=prepared.precision.physical_dtype,
            )
            row += term.coefficient * np.asarray(evaluation @ derivative) * normalizers
        return row
    if any(term.derivative_order != 0 for term in constraint.terms):
        raise ValueError(
            "Rational bases initially support only value traces at infinity."
        )
    values = point ** np.arange(count)
    coefficient = sum(term.coefficient for term in constraint.terms)
    return coefficient * values


def _constraint_matrix(
    prepared: PreparedSpectralAxis,
    conditions: SpectralBoundaryConditionPlan,
    /,
) -> np.ndarray:
    return np.stack(
        tuple(_trace_row(prepared, constraint) for constraint in conditions.constraints),
        axis=0,
    )


def _canonical_columns(values: np.ndarray, /) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


class ConstrainedBasisPlan(AbstractSpectralBasisPlan):
    """Spectral basis restricted to the nullspace of declared linear traces."""

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
        if base.family not in (
            "chebyshev",
            "legendre",
            "rational_chebyshev_line",
            "rational_chebyshev_half_line",
        ):
            raise ValueError(
                "Constrained bases require polynomial or rational Chebyshev modes."
            )
        if not isinstance(conditions, SpectralBoundaryConditionPlan):
            raise TypeError("conditions must be a SpectralBoundaryConditionPlan.")
        free = base.mode_count - len(conditions.constraints)
        if free <= 0:
            raise ValueError("Boundary constraints leave no spectral degrees of freedom.")
        simple = tuple(
            (
                constraint.side,
                constraint.derivative_orders,
                tuple(term.coefficient for term in constraint.terms),
            )
            for constraint in conditions.constraints
        )
        boundary: SpectralBoundaryKind = (
            "homogeneous_dirichlet"
            if set(simple)
            == {
                ("lower", (0,), (1.0,)),
                ("upper", (0,), (1.0,)),
            }
            and base.family in ("chebyshev", "legendre")
            else "homogeneous_neumann"
            if set(simple)
            == {
                ("lower", (1,), (1.0,)),
                ("upper", (1,), (1.0,)),
            }
            and base.family in ("chebyshev", "legendre")
            else "decay"
            if base.family in ("rational_chebyshev_line", "rational_chebyshev_half_line")
            and all(orders == (0,) for _, orders, _ in simple)
            else "homogeneous_trace"
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
        domain: AxisDomain,
        /,
        *,
        precision: SpectralPrecisionPolicy,
    ) -> PreparedSpectralAxis:
        base = self.base.prepare(domain, precision=precision)
        if base.modal_transform is None:
            raise RuntimeError("Constrained base lacks dense modal metadata.")
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
        derivative = None
        derivative_exact = True
        derivative_residual = 0.0
        if (
            base.family == "rational_chebyshev_line"
            and base.derivative_matrix is not None
        ):
            derivative = (
                analysis
                @ np.asarray(base.modal_transform.synthesis)
                @ np.asarray(base.derivative_matrix)
                @ stencil
            )
            derivative_exact = False
            derivative_residual = base.derivative_residual
        return PreparedSpectralAxis(
            self,
            base.nodes,
            base.reference_nodes,
            weights,
            base.domain,
            SpectralModeLayout(
                self.family,
                np.arange(self.mode_count),
                mode_ids=mode_ids,
            ),
            execution,
            precision,
            lower_endpoint_included=base.lower_endpoint_included,
            upper_endpoint_included=base.upper_endpoint_included,
            derivative_matrix=derivative,
            derivative_exact=derivative_exact,
            derivative_residual=derivative_residual,
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
    "PreparedBoundaryLift",
    "SpectralBoundaryConditionPlan",
    "SpectralTraceConstraint",
    "SpectralTraceTerm",
]

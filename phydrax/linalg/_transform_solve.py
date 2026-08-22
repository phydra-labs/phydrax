#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._matrix_functions import TransformDiagonalRepresentation
from ._spaces import ArraySpace


CompatibilityPolicy: TypeAlias = Literal["error", "project_rhs"]
GaugePolicy: TypeAlias = Literal["zero_mean", "minimum_norm"]


class TransformDiagonalSolvePlan(StrictModule, NonTrainableState):
    """Compatibility- and gauge-aware direct solve in modal coordinates."""

    representation: TransformDiagonalRepresentation
    diagonal_shift: Array
    compatibility: CompatibilityPolicy = eqx.field(static=True)
    gauge: GaugePolicy = eqx.field(static=True)
    zero_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: TransformDiagonalRepresentation,
        /,
        *,
        diagonal_shift: ArrayLike = 0.0,
        compatibility: CompatibilityPolicy = "error",
        gauge: GaugePolicy = "minimum_norm",
        zero_tolerance: float = 1e-10,
        plan_id: str | None = None,
    ):
        if not isinstance(representation, TransformDiagonalRepresentation):
            raise TypeError("representation must be TransformDiagonalRepresentation.")
        if not isinstance(representation.operator.source, ArraySpace) or not isinstance(
            representation.operator.target, ArraySpace
        ):
            raise TypeError(
                "Transform-diagonal direct solves currently require ArraySpace."
            )
        shift = jnp.asarray(diagonal_shift)
        if shift.shape != () or not bool(np.isfinite(np.asarray(shift))):
            raise ValueError("diagonal_shift must be one finite scalar.")
        if compatibility not in ("error", "project_rhs"):
            raise ValueError("Unknown compatibility policy.")
        if gauge not in ("zero_mean", "minimum_norm"):
            raise ValueError("Unknown gauge policy.")
        tolerance = float(zero_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("zero_tolerance must be finite and positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "transform-diagonal-solve-plan",
                    "representation": representation.representation_id,
                    "diagonal_shift": repr(np.asarray(shift).item()),
                    "compatibility": compatibility,
                    "gauge": gauge,
                    "zero_tolerance": tolerance,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.representation = representation
        self.diagonal_shift = shift
        self.compatibility = compatibility
        self.gauge = gauge
        self.zero_tolerance = tolerance
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedTransformDiagonalSolve":
        return PreparedTransformDiagonalSolve(self)


class TransformDiagonalSolveResult(StrictModule):
    """Direct modal solve and explicit compatibility diagnostics."""

    value: Array
    residual_norm: Array
    compatibility_residual: Array
    removed_component_norm: Array
    converged: Array
    plan_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)


class PreparedTransformDiagonalSolve(StrictModule, NonTrainableState):
    """Prepared diagonal denominator, nullspace, and explicit gauge basis."""

    plan: TransformDiagonalSolvePlan
    diagonal: Array
    nullspace_mask: Array
    nullspace_basis: Array
    zero_mean_denominator: Array
    nullspace_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: TransformDiagonalSolvePlan, /):
        if not isinstance(plan, TransformDiagonalSolvePlan):
            raise TypeError("plan must be a TransformDiagonalSolvePlan.")
        diagonal = (plan.representation.modal_values + plan.diagonal_shift).reshape((-1,))
        nullspace = jnp.abs(diagonal) <= plan.zero_tolerance
        null_indices = np.flatnonzero(np.asarray(nullspace))
        if null_indices.size == diagonal.size:
            basis = jnp.zeros((diagonal.size, 0), dtype=diagonal.dtype)
        elif null_indices.size:
            unit_modes = jnp.eye(diagonal.size, dtype=diagonal.dtype)[null_indices]
            basis = jax.vmap(plan.representation.synthesize_coordinates)(unit_modes).T
        else:
            basis = jnp.zeros((diagonal.size, 0), dtype=diagonal.dtype)
        if plan.gauge == "zero_mean" and null_indices.size not in (0, 1):
            raise ValueError("zero_mean gauge requires a one-dimensional nullspace.")
        denominator = (
            jnp.asarray(1.0, dtype=basis.dtype)
            if basis.shape[1] == 0
            else jnp.mean(basis[:, 0])
        )
        if (
            basis.shape[1]
            and abs(complex(np.asarray(denominator))) <= plan.zero_tolerance
        ):
            raise ValueError("zero_mean gauge requires a nonzero-mean null vector.")
        self.plan = plan
        self.diagonal = diagonal
        self.nullspace_mask = nullspace
        self.nullspace_basis = basis
        self.zero_mean_denominator = denominator
        self.nullspace_dimension = int(null_indices.size)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-transform-diagonal-solve",
                "plan": plan.plan_id,
                "representation": plan.representation.representation_id,
                "nullspace_dimension": int(null_indices.size),
            }
        )

    def solve(self, right_hand_side: ArrayLike, /) -> TransformDiagonalSolveResult:
        representation = self.plan.representation
        operator = representation.operator
        source_space = operator.source
        target_space = operator.target
        if not isinstance(source_space, ArraySpace) or not isinstance(
            target_space,
            ArraySpace,
        ):
            raise RuntimeError("Prepared transform solve lost its ArraySpace contract.")
        rhs = target_space.validate(jnp.asarray(right_hand_side))
        coordinates = target_space.flatten(rhs)
        modal_rhs = representation.analyze_coordinates(coordinates)
        incompatible = jnp.where(self.nullspace_mask, modal_rhs, 0.0)
        compatibility_residual = jnp.linalg.norm(incompatible)
        if self.plan.compatibility == "error":
            modal_rhs = eqx.error_if(
                modal_rhs,
                compatibility_residual > self.plan.zero_tolerance,
                "Transform-diagonal RHS is incompatible with the operator nullspace.",
            )
        projected_rhs = jnp.where(self.nullspace_mask, 0.0, modal_rhs)
        effective_rhs_coordinates = representation.synthesize_coordinates(projected_rhs)
        if not jnp.issubdtype(target_space.dtype, jnp.complexfloating):
            effective_rhs_coordinates = jnp.real(effective_rhs_coordinates)
        effective_rhs = target_space.unflatten(effective_rhs_coordinates)
        coefficients = jnp.where(
            self.nullspace_mask,
            jnp.zeros((), dtype=projected_rhs.dtype),
            projected_rhs / self.diagonal,
        )
        solution_coordinates = representation.synthesize_coordinates(coefficients)
        if self.nullspace_basis.shape[1]:
            if self.plan.gauge == "minimum_norm":
                basis = self.nullspace_basis
                gram = jnp.conj(basis.T) @ basis
                correction = jnp.linalg.solve(
                    gram,
                    jnp.conj(basis.T) @ solution_coordinates,
                )
                solution_coordinates = solution_coordinates - basis @ correction
            else:
                solution_coordinates = solution_coordinates - self.nullspace_basis[
                    :, 0
                ] * (jnp.mean(solution_coordinates) / self.zero_mean_denominator)
        if not jnp.issubdtype(source_space.dtype, jnp.complexfloating):
            solution_coordinates = jnp.real(solution_coordinates)
        value = source_space.unflatten(solution_coordinates)
        residual = operator.mv(value) + self.plan.diagonal_shift * value - effective_rhs
        residual_norm = jnp.linalg.norm(target_space.flatten(residual))
        converged = jnp.isfinite(residual_norm) & (
            residual_norm
            <= self.plan.zero_tolerance * (1.0 + jnp.linalg.norm(coordinates))
        )
        return TransformDiagonalSolveResult(
            value=value,
            residual_norm=residual_norm,
            compatibility_residual=compatibility_residual,
            removed_component_norm=jnp.linalg.norm(incompatible),
            converged=converged,
            plan_id=self.plan.plan_id,
            representation_id=representation.representation_id,
        )


__all__ = [
    "CompatibilityPolicy",
    "GaugePolicy",
    "PreparedTransformDiagonalSolve",
    "TransformDiagonalSolvePlan",
    "TransformDiagonalSolveResult",
]

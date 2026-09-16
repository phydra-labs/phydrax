#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded stationary one-dimensional double-well kink solve and evidence."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._phase_field import DoubleWellFreeEnergy
from ..._strict import StrictModule
from ...linalg import inverse


class DoubleWellKinkPlan(StrictModule):
    coordinates: Array
    free_energy: DoubleWellFreeEnergy
    gradient_coefficient: float = eqx.field(static=True)
    maximum_newton_steps: int = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        free_energy: DoubleWellFreeEnergy,
        /,
        *,
        gradient_coefficient: float,
        maximum_newton_steps: int = 32,
        maximum_backtracks: int = 12,
        residual_tolerance: float = 1e-10,
        maximum_matrix_elements: int = 10_000_000,
    ):
        points = np.asarray(coordinates, dtype=float)
        if not isinstance(free_energy, DoubleWellFreeEnergy):
            raise TypeError("free_energy must be DoubleWellFreeEnergy.")
        coefficient = float(gradient_coefficient)
        newton = int(maximum_newton_steps)
        backtracks = int(maximum_backtracks)
        tolerance = float(residual_tolerance)
        maximum = int(maximum_matrix_elements)
        if points.ndim != 1 or points.size < 5 or not np.all(np.isfinite(points)):
            raise ValueError(
                "Kink coordinates must be one finite vector with at least five nodes."
            )
        differences = np.diff(points)
        if not np.all(differences > 0.0) or not np.allclose(differences, differences[0]):
            raise ValueError("Kink coordinates must be uniformly increasing.")
        if points[0] >= 0.0 or points[-1] <= 0.0:
            raise ValueError("Kink interval must straddle zero.")
        if (
            not np.isfinite(coefficient)
            or coefficient <= 0.0
            or newton < 1
            or backtracks < 1
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or maximum < points.size**2
        ):
            raise ValueError(
                "Kink coefficients, work, tolerance, or matrix capacity are invalid."
            )
        self.coordinates = jnp.asarray(points)
        self.free_energy = free_energy
        self.gradient_coefficient = coefficient
        self.maximum_newton_steps = newton
        self.maximum_backtracks = backtracks
        self.residual_tolerance = tolerance
        self.maximum_matrix_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stationary-double-well-kink-plan",
                "coordinates": array_tree_fingerprint(points),
                "free_energy": free_energy.free_energy_id,
                "gradient_coefficient": coefficient,
                "maximum_newton_steps": newton,
                "maximum_backtracks": backtracks,
                "residual_tolerance": tolerance,
                "maximum_matrix_elements": maximum,
                "boundary_values": (-1.0, 1.0),
            }
        )

    @property
    def spacing(self) -> float:
        return float(self.coordinates[1] - self.coordinates[0])

    @property
    def interface_width(self) -> float:
        return float(
            np.sqrt(2.0 * self.gradient_coefficient / float(self.free_energy.scale))
        )


class DoubleWellKinkEvidence(StrictModule):
    euler_lagrange_residual: Array
    maximum_residual: Array
    energy: Array
    gradient_energy: Array
    bulk_energy: Array
    boundary_residual: Array
    topological_sector: Array
    center_residual: Array
    translational_mode_residual: Array
    minimum_stability_eigenvalue: Array
    negative_mode_count: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class DoubleWellKinkResult(StrictModule):
    field: Array
    residual_history: Array
    accepted_steps: Array
    evidence: DoubleWellKinkEvidence
    converged: Array
    plan_id: str = eqx.field(static=True)


def analytic_double_well_kink(plan: DoubleWellKinkPlan, /) -> Array:
    if not isinstance(plan, DoubleWellKinkPlan):
        raise TypeError("plan must be DoubleWellKinkPlan.")
    return jnp.tanh(plan.coordinates / plan.interface_width)


def _residual(plan: DoubleWellKinkPlan, field: Array, /) -> Array:
    spacing = plan.spacing
    interior = -plan.gradient_coefficient * (
        field[2:] - 2.0 * field[1:-1] + field[:-2]
    ) / spacing**2 + plan.free_energy.derivative(field[1:-1])
    return interior


def _jacobian(plan: DoubleWellKinkPlan, field: Array, /) -> Array:
    count = field.size - 2
    spacing = plan.spacing
    off_diagonal = -plan.gradient_coefficient / spacing**2
    diagonal = 2.0 * plan.gradient_coefficient / spacing**2 + plan.free_energy.scale * (
        3.0 * field[1:-1] ** 2 - 1.0
    )
    matrix = jnp.diag(diagonal)
    if count > 1:
        matrix = (
            matrix
            + jnp.diag(jnp.full((count - 1,), off_diagonal, dtype=field.dtype), k=1)
            + jnp.diag(jnp.full((count - 1,), off_diagonal, dtype=field.dtype), k=-1)
        )
    return matrix


def _evidence(plan: DoubleWellKinkPlan, field: Array, /) -> DoubleWellKinkEvidence:
    residual = _residual(plan, field)
    spacing = plan.spacing
    gradient = (field[1:] - field[:-1]) / spacing
    gradient_energy = 0.5 * plan.gradient_coefficient * spacing * jnp.sum(gradient**2)
    bulk_density = plan.free_energy.density(field)
    bulk_energy = spacing * (
        0.5 * bulk_density[0] + jnp.sum(bulk_density[1:-1]) + 0.5 * bulk_density[-1]
    )
    energy = gradient_energy + bulk_energy
    boundary = jnp.maximum(jnp.abs(field[0] + 1.0), jnp.abs(field[-1] - 1.0))
    sector = 0.5 * (field[-1] - field[0])
    center_index = int(np.argmin(np.abs(np.asarray(plan.coordinates))))
    center = jnp.abs(field[center_index])
    stability = _jacobian(plan, field)
    eigenvalues = jnp.linalg.eigvalsh(stability)
    translation = (field[2:] - field[:-2]) / (2.0 * spacing)
    translation_residual = jnp.linalg.norm(stability @ translation) / jnp.maximum(
        1.0, jnp.linalg.norm(translation)
    )
    maximum = jnp.max(jnp.abs(residual))
    finite = (
        jnp.all(jnp.isfinite(field))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(eigenvalues))
    )
    accepted = (
        finite
        & (maximum <= plan.residual_tolerance)
        & (boundary <= plan.residual_tolerance)
        & (jnp.abs(sector - 1.0) <= plan.residual_tolerance)
        & (jnp.min(eigenvalues) >= -10.0 * plan.residual_tolerance)
    )
    return DoubleWellKinkEvidence(
        euler_lagrange_residual=residual,
        maximum_residual=maximum,
        energy=energy,
        gradient_energy=gradient_energy,
        bulk_energy=bulk_energy,
        boundary_residual=boundary,
        topological_sector=sector,
        center_residual=center,
        translational_mode_residual=translation_residual,
        minimum_stability_eigenvalue=jnp.min(eigenvalues),
        negative_mode_count=jnp.sum(
            (eigenvalues < -10.0 * plan.residual_tolerance).astype(jnp.int32)
        ),
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        claim="finite-interval-stationary-double-well-kink-and-linear-stability-evidence",
    )


def solve_double_well_kink(
    plan: DoubleWellKinkPlan,
    initial_field: ArrayLike | None = None,
    /,
) -> DoubleWellKinkResult:
    if not isinstance(plan, DoubleWellKinkPlan):
        raise TypeError("plan must be DoubleWellKinkPlan.")
    field = (
        analytic_double_well_kink(plan)
        if initial_field is None
        else jnp.asarray(initial_field, dtype=plan.coordinates.dtype)
    )
    if field.shape != plan.coordinates.shape:
        raise ValueError("initial_field must match the kink coordinate grid.")
    field = field.at[0].set(-1.0).at[-1].set(1.0)
    histories = []
    accepted = []
    converged = False
    for _ in range(plan.maximum_newton_steps):
        residual = _residual(plan, field)
        norm = jnp.max(jnp.abs(residual))
        histories.append(norm)
        if float(norm) <= plan.residual_tolerance:
            converged = True
            break
        jacobian = _jacobian(plan, field)
        inverse_result = inverse(jacobian)
        if not bool(jnp.all(inverse_result.successful)):
            accepted.append(False)
            break
        direction = -(inverse_result.value @ residual)
        step = 1.0
        did_accept = False
        for _ in range(plan.maximum_backtracks):
            candidate = field.at[1:-1].add(step * direction)
            candidate = candidate.at[0].set(-1.0).at[-1].set(1.0)
            candidate_norm = jnp.max(jnp.abs(_residual(plan, candidate)))
            if bool(jnp.isfinite(candidate_norm) & (candidate_norm < norm)):
                field = candidate
                did_accept = True
                break
            step *= 0.5
        accepted.append(did_accept)
        if not did_accept:
            break
    evidence = _evidence(plan, field)
    converged = converged or bool(evidence.accepted)
    return DoubleWellKinkResult(
        field=field,
        residual_history=jnp.asarray(histories),
        accepted_steps=jnp.asarray(accepted, dtype=bool),
        evidence=evidence,
        converged=jnp.asarray(converged),
        plan_id=plan.plan_id,
    )


__all__ = [
    "DoubleWellKinkEvidence",
    "DoubleWellKinkPlan",
    "DoubleWellKinkResult",
    "analytic_double_well_kink",
    "solve_double_well_kink",
]

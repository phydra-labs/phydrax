#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mapped-infinite multifield defects, radial defects, continuation, and scattering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


@dataclass(frozen=True, slots=True)
class PolynomialPotentialTerm:
    exponents: tuple[int, ...]
    coefficient: float
    term_id: str

    def __init__(self, exponents: Sequence[int], coefficient: float, /):
        powers = tuple(int(value) for value in exponents)
        coefficient_ = float(coefficient)
        if (
            not powers
            or any(value < 0 for value in powers)
            or not math.isfinite(coefficient_)
        ):
            raise ValueError(
                "Polynomial potential terms require finite coefficients and nonnegative powers."
            )
        content = {
            "kind": "polynomial-potential-term",
            "exponents": powers,
            "coefficient": coefficient_,
        }
        object.__setattr__(self, "exponents", powers)
        object.__setattr__(self, "coefficient", coefficient_)
        object.__setattr__(self, "term_id", canonical_fingerprint(content))


class PolynomialDefectPotential(StrictModule):
    """Source-identified finite polynomial potential for one or more real fields."""

    terms: tuple[PolynomialPotentialTerm, ...] = eqx.field(static=True)
    field_labels: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_labels: Sequence[str],
        terms: Sequence[PolynomialPotentialTerm],
        source_id: str,
        /,
    ):
        labels = tuple(str(value).strip() for value in field_labels)
        terms_ = tuple(terms)
        source = str(source_id).strip()
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Defect field labels must be unique and non-empty.")
        if not terms_ or any(
            not isinstance(value, PolynomialPotentialTerm) for value in terms_
        ):
            raise TypeError("terms must contain PolynomialPotentialTerm values.")
        if any(len(value.exponents) != len(labels) for value in terms_) or not source:
            raise ValueError(
                "Potential terms and source identity do not match the field roster."
            )
        self.terms = terms_
        self.field_labels = labels
        self.source_id = source
        self.potential_id = canonical_fingerprint(
            {
                "kind": "polynomial-defect-potential",
                "field_labels": labels,
                "terms": [value.term_id for value in terms_],
                "source_id": source,
            }
        )

    @property
    def field_count(self) -> int:
        return len(self.field_labels)

    def value(self, fields: ArrayLike, /) -> Array:
        values = jnp.asarray(fields)
        if values.shape[-1:] != (self.field_count,):
            raise ValueError("Potential fields have the wrong trailing dimension.")
        result = jnp.zeros(values.shape[:-1], dtype=values.dtype)
        for term in self.terms:
            monomial = jnp.asarray(term.coefficient, dtype=values.dtype)
            for index, exponent in enumerate(term.exponents):
                monomial = monomial * values[..., index] ** exponent
            result = result + monomial
        return result

    def gradient(self, fields: ArrayLike, /) -> Array:
        values = jnp.asarray(fields)
        return jax.grad(lambda value: self.value(value))(values)

    def hessian(self, fields: ArrayLike, /) -> Array:
        values = jnp.asarray(fields)
        return jax.hessian(lambda value: self.value(value))(values)

    def with_coefficient(
        self, term_index: int, coefficient: float, /
    ) -> PolynomialDefectPotential:
        index = int(term_index)
        if index < 0 or index >= len(self.terms):
            raise ValueError("Potential term index is out of range.")
        terms = list(self.terms)
        terms[index] = PolynomialPotentialTerm(terms[index].exponents, coefficient)
        return PolynomialDefectPotential(self.field_labels, terms, self.source_id)


class MappedInfiniteDefectPlan(StrictModule):
    potential: PolynomialDefectPotential
    gradient_matrix: Array
    left_vacuum: Array
    right_vacuum: Array
    collocation_count: int = eqx.field(static=True)
    map_scale: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    backtracking_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: PolynomialDefectPotential,
        gradient_matrix: ArrayLike,
        left_vacuum: ArrayLike,
        right_vacuum: ArrayLike,
        /,
        *,
        collocation_count: int = 129,
        map_scale: float = 1.0,
        residual_tolerance: float = 1e-10,
        maximum_iterations: int = 40,
        backtracking_steps: int = 12,
    ):
        if not isinstance(potential, PolynomialDefectPotential):
            raise TypeError("potential must be PolynomialDefectPotential.")
        gradient = np.asarray(gradient_matrix, dtype=float)
        left = np.asarray(left_vacuum, dtype=float)
        right = np.asarray(right_vacuum, dtype=float)
        count = int(collocation_count)
        scale = float(map_scale)
        tolerance = float(residual_tolerance)
        iterations = int(maximum_iterations)
        backtracking = int(backtracking_steps)
        dimension = potential.field_count
        if (
            gradient.shape != (dimension, dimension)
            or left.shape != (dimension,)
            or right.shape != (dimension,)
        ):
            raise ValueError("Defect gradient matrix or vacua have the wrong shape.")
        if np.linalg.eigvalsh(0.5 * (gradient + gradient.T))[0] <= 0.0:
            raise ValueError("Defect gradient matrix must be positive definite.")
        if (
            count < 17
            or count % 2 == 0
            or scale <= 0.0
            or tolerance <= 0.0
            or iterations < 1
            or backtracking < 1
        ):
            raise ValueError("Mapped-infinite defect controls are invalid.")
        for vacuum in (left, right):
            if (
                np.linalg.norm(np.asarray(potential.gradient(jnp.asarray(vacuum))))
                > 100.0 * tolerance
            ):
                raise ValueError(
                    "Declared defect vacua are not stationary points of the potential."
                )
        content = {
            "kind": "mapped-infinite-defect-plan",
            "potential": potential.potential_id,
            "gradient_matrix": array_tree_fingerprint(gradient),
            "left_vacuum": array_tree_fingerprint(left),
            "right_vacuum": array_tree_fingerprint(right),
            "collocation_count": count,
            "map_scale": scale,
            "residual_tolerance": tolerance,
            "maximum_iterations": iterations,
            "backtracking_steps": backtracking,
        }
        self.potential = potential
        self.gradient_matrix = jnp.asarray(gradient)
        self.left_vacuum = jnp.asarray(left)
        self.right_vacuum = jnp.asarray(right)
        self.collocation_count = count
        self.map_scale = scale
        self.residual_tolerance = tolerance
        self.maximum_iterations = iterations
        self.backtracking_steps = backtracking
        self.plan_id = canonical_fingerprint(content)


class MappedInfiniteDefectEvidence(StrictModule):
    residual_norm: Array
    energy: Array
    topological_charge: Array
    translation_mode_residual: Array
    stability_eigenvalues: Array
    negative_mode_count: Array
    converged: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MappedInfiniteDefectResult(StrictModule):
    compact_coordinates: Array
    physical_coordinates: Array
    fields: Array
    first_derivative: Array
    second_derivative: Array
    evidence: MappedInfiniteDefectEvidence
    plan_id: str = eqx.field(static=True)


def _chebyshev_operators(count: int, scale: float, /):
    order = count - 1
    descending = np.cos(np.pi * np.arange(count) / order)
    coefficients = np.ones(count)
    coefficients[[0, -1]] = 2.0
    coefficients *= (-1.0) ** np.arange(count)
    difference = descending[:, None] - descending[None, :]
    matrix = (coefficients[:, None] / coefficients[None, :]) / (
        difference + np.eye(count)
    )
    matrix = matrix - np.diag(np.sum(matrix, axis=1))
    permutation = np.arange(count - 1, -1, -1)
    compact = descending[permutation]
    derivative_compact = -matrix[np.ix_(permutation, permutation)]
    one_minus = 1.0 - compact**2
    compact_per_physical = one_minus**2 / (scale * (1.0 + compact**2))
    derivative = np.diag(compact_per_physical) @ derivative_compact
    second = derivative @ derivative
    physical = np.empty_like(compact)
    physical[1:-1] = scale * compact[1:-1] / (1.0 - compact[1:-1] ** 2)
    physical[0], physical[-1] = -np.inf, np.inf
    return compact, physical, derivative, second


def solve_mapped_infinite_defect(
    plan: MappedInfiniteDefectPlan,
    /,
    *,
    initial_fields: ArrayLike | None = None,
) -> MappedInfiniteDefectResult:
    if not isinstance(plan, MappedInfiniteDefectPlan):
        raise TypeError("plan must be MappedInfiniteDefectPlan.")
    compact, physical, derivative, second = _chebyshev_operators(
        plan.collocation_count,
        plan.map_scale,
    )
    if initial_fields is None:
        interpolation = 0.5 * (1.0 + compact[:, None])
        fields = np.asarray(plan.left_vacuum) + interpolation * (
            np.asarray(plan.right_vacuum) - np.asarray(plan.left_vacuum)
        )
    else:
        fields = np.asarray(initial_fields, dtype=float).copy()
    expected = (plan.collocation_count, plan.potential.field_count)
    if fields.shape != expected:
        raise ValueError("Initial defect fields have the wrong shape.")
    fields[0] = np.asarray(plan.left_vacuum)
    fields[-1] = np.asarray(plan.right_vacuum)
    interior_shape = fields[1:-1].shape

    def residual(interior: Array) -> Array:
        full = jnp.concatenate(
            (
                plan.left_vacuum[None, :],
                interior.reshape(interior_shape),
                plan.right_vacuum[None, :],
            ),
            axis=0,
        )
        second_derivative = jnp.asarray(second) @ full
        potential_gradient = jax.vmap(plan.potential.gradient)(full)
        equation = -second_derivative @ plan.gradient_matrix.T + potential_gradient
        return equation[1:-1].reshape(-1)

    interior = jnp.asarray(fields[1:-1]).reshape(-1)
    residual_norm = math.inf
    converged = False
    for _ in range(plan.maximum_iterations):
        value = residual(interior)
        residual_norm = float(jnp.linalg.norm(value))
        if residual_norm <= plan.residual_tolerance:
            converged = True
            break
        jacobian = jax.jacfwd(residual)(interior)
        update = jnp.linalg.solve(jacobian, -value)
        accepted = False
        for backtrack in range(plan.backtracking_steps):
            candidate = interior + 0.5**backtrack * update
            candidate_norm = float(jnp.linalg.norm(residual(candidate)))
            if candidate_norm < residual_norm:
                interior = candidate
                accepted = True
                break
        if not accepted:
            break
    final_fields = np.concatenate(
        (
            np.asarray(plan.left_vacuum)[None, :],
            np.asarray(interior).reshape(interior_shape),
            np.asarray(plan.right_vacuum)[None, :],
        ),
        axis=0,
    )
    first = derivative @ final_fields
    second_values = second @ final_fields
    interior_x = physical[1:-1]
    density = 0.5 * np.asarray(
        ein.contract(
            "ni,ij,nj->n",
            first[1:-1],
            np.asarray(plan.gradient_matrix),
            first[1:-1],
        )
    ) + np.asarray(plan.potential.value(jnp.asarray(final_fields[1:-1])))
    energy = float(np.trapezoid(density, interior_x))
    topological = final_fields[-1] - final_fields[0]
    hessian_blocks = np.asarray(
        jax.vmap(plan.potential.hessian)(jnp.asarray(final_fields[1:-1]))
    )
    interior_count = plan.collocation_count - 2
    stability = -np.kron(second[1:-1, 1:-1], np.asarray(plan.gradient_matrix))
    for index in range(interior_count):
        block = slice(
            index * plan.potential.field_count, (index + 1) * plan.potential.field_count
        )
        stability[block, block] += hessian_blocks[index]
    stability = 0.5 * (stability + stability.T)
    eigenvalues = np.linalg.eigvalsh(stability)
    translation = first[1:-1].reshape(-1)
    translation_residual = float(
        np.linalg.norm(stability @ translation) / max(1.0, np.linalg.norm(translation))
    )
    negative = int(np.sum(eigenvalues < -math.sqrt(plan.residual_tolerance)))
    evidence_id = canonical_fingerprint(
        {
            "kind": "mapped-infinite-defect-evidence",
            "plan": plan.plan_id,
            "fields": array_tree_fingerprint(final_fields),
            "residual_norm": residual_norm,
            "energy": energy,
            "topological_charge": topological.tolist(),
        }
    )
    evidence = MappedInfiniteDefectEvidence(
        residual_norm=jnp.asarray(residual_norm),
        energy=jnp.asarray(energy),
        topological_charge=jnp.asarray(topological),
        translation_mode_residual=jnp.asarray(translation_residual),
        stability_eigenvalues=jnp.asarray(eigenvalues),
        negative_mode_count=jnp.asarray(negative, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )
    return MappedInfiniteDefectResult(
        compact_coordinates=jnp.asarray(compact),
        physical_coordinates=jnp.asarray(physical),
        fields=jnp.asarray(final_fields),
        first_derivative=jnp.asarray(first),
        second_derivative=jnp.asarray(second_values),
        evidence=evidence,
        plan_id=plan.plan_id,
    )


class RadialDefectPlan(StrictModule):
    base: MappedInfiniteDefectPlan
    spatial_dimension: int = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    radial_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: MappedInfiniteDefectPlan,
        spatial_dimension: int,
        maximum_radius: float,
        radial_points: int,
        /,
    ):
        if not isinstance(base, MappedInfiniteDefectPlan):
            raise TypeError("base must be MappedInfiniteDefectPlan.")
        dimension = int(spatial_dimension)
        radius = float(maximum_radius)
        points = int(radial_points)
        if dimension < 2 or radius <= 0.0 or points < 9:
            raise ValueError("Radial defect geometry is invalid.")
        self.base = base
        self.spatial_dimension = dimension
        self.maximum_radius = radius
        self.radial_points = points
        self.plan_id = canonical_fingerprint(
            {
                "kind": "radial-defect-plan",
                "base": base.plan_id,
                "spatial_dimension": dimension,
                "maximum_radius": radius,
                "radial_points": points,
            }
        )


class RadialDefectResult(StrictModule):
    radii: Array
    fields: Array
    residual_norm: Array
    energy: Array
    converged: Array
    plan_id: str = eqx.field(static=True)


def solve_radial_defect(
    plan: RadialDefectPlan,
    center_values: ArrayLike,
    boundary_values: ArrayLike,
    /,
) -> RadialDefectResult:
    """Solve a bounded radially symmetric multifield Euler–Lagrange problem."""

    if not isinstance(plan, RadialDefectPlan):
        raise TypeError("plan must be RadialDefectPlan.")
    center = jnp.asarray(center_values, dtype=plan.base.left_vacuum.dtype)
    boundary = jnp.asarray(boundary_values, dtype=center.dtype)
    field_count = plan.base.potential.field_count
    if center.shape != (field_count,) or boundary.shape != (field_count,):
        raise ValueError("Radial defect boundary values have the wrong shape.")
    radii = jnp.linspace(0.0, plan.maximum_radius, plan.radial_points)
    spacing = float(radii[1] - radii[0])
    interpolation = radii[:, None] / plan.maximum_radius
    fields = center + interpolation * (boundary - center)
    interior_shape = (plan.radial_points - 2, field_count)

    def residual(interior: Array) -> Array:
        full = jnp.concatenate(
            (center[None], interior.reshape(interior_shape), boundary[None])
        )
        first = (full[2:] - full[:-2]) / (2.0 * spacing)
        second = (full[2:] - 2.0 * full[1:-1] + full[:-2]) / spacing**2
        radial = first * (plan.spatial_dimension - 1) / radii[1:-1, None]
        gradient = jax.vmap(plan.base.potential.gradient)(full[1:-1])
        equation = -(second + radial) @ plan.base.gradient_matrix.T + gradient
        return equation.reshape(-1)

    interior = fields[1:-1].reshape(-1)
    converged = False
    residual_norm = math.inf
    for _ in range(plan.base.maximum_iterations):
        value = residual(interior)
        residual_norm = float(jnp.linalg.norm(value))
        if residual_norm <= plan.base.residual_tolerance:
            converged = True
            break
        jacobian = jax.jacfwd(residual)(interior)
        interior = interior + jnp.linalg.solve(jacobian, -value)
    final = jnp.concatenate(
        (center[None], interior.reshape(interior_shape), boundary[None])
    )
    derivative = jnp.gradient(final, spacing, axis=0)
    density = 0.5 * ein.contract(
        "ni,ij,nj->n",
        derivative,
        plan.base.gradient_matrix,
        derivative,
    ) + plan.base.potential.value(final)
    sphere_area = (
        2.0
        * np.pi ** (0.5 * plan.spatial_dimension)
        / math.gamma(0.5 * plan.spatial_dimension)
    )
    energy = sphere_area * jnp.trapezoid(
        density * radii ** (plan.spatial_dimension - 1), radii
    )
    return RadialDefectResult(
        radii=radii,
        fields=final,
        residual_norm=jnp.asarray(residual_norm),
        energy=energy,
        converged=jnp.asarray(converged),
        plan_id=plan.plan_id,
    )


class DefectContinuationResult(StrictModule):
    parameter_values: Array
    fields: Array
    energies: Array
    residual_norms: Array
    converged: Array
    branch_id: str = eqx.field(static=True)


def continue_polynomial_defect(
    plan: MappedInfiniteDefectPlan,
    term_index: int,
    parameter_values: Sequence[float],
    /,
) -> DefectContinuationResult:
    """Track one stationary branch by warm-started coefficient continuation."""

    values = tuple(float(value) for value in parameter_values)
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Continuation requires at least two finite parameter values.")
    fields = None
    results = []
    for value in values:
        potential = plan.potential.with_coefficient(term_index, value)
        candidate_plan = MappedInfiniteDefectPlan(
            potential,
            plan.gradient_matrix,
            plan.left_vacuum,
            plan.right_vacuum,
            collocation_count=plan.collocation_count,
            map_scale=plan.map_scale,
            residual_tolerance=plan.residual_tolerance,
            maximum_iterations=plan.maximum_iterations,
            backtracking_steps=plan.backtracking_steps,
        )
        result = solve_mapped_infinite_defect(candidate_plan, initial_fields=fields)
        results.append(result)
        fields = result.fields
    branch_id = canonical_fingerprint(
        {
            "kind": "polynomial-defect-continuation",
            "base": plan.plan_id,
            "term_index": int(term_index),
            "parameter_values": values,
            "plans": [value.plan_id for value in results],
        }
    )
    return DefectContinuationResult(
        parameter_values=jnp.asarray(values),
        fields=jnp.stack(tuple(value.fields for value in results)),
        energies=jnp.stack(tuple(value.evidence.energy for value in results)),
        residual_norms=jnp.stack(
            tuple(value.evidence.residual_norm for value in results)
        ),
        converged=jnp.stack(tuple(value.evidence.converged for value in results)),
        branch_id=branch_id,
    )


class DefectScatteringPlan(StrictModule):
    potential: PolynomialDefectPotential
    gradient_matrix: Array
    spatial_points: Array
    time_step: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    damping: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: PolynomialDefectPotential,
        gradient_matrix: ArrayLike,
        spatial_points: ArrayLike,
        /,
        *,
        time_step: float,
        steps: int,
        boundary_damping: float = 1.0,
        damping_fraction: float = 0.1,
    ):
        if not isinstance(potential, PolynomialDefectPotential):
            raise TypeError("potential must be PolynomialDefectPotential.")
        gradient = np.asarray(gradient_matrix, dtype=float)
        points = np.asarray(spatial_points, dtype=float)
        step = float(time_step)
        count = int(steps)
        if points.ndim != 1 or points.size < 9 or np.any(np.diff(points) <= 0.0):
            raise ValueError(
                "Scattering spatial points must be increasing and nontrivial."
            )
        spacing = np.diff(points)
        if (
            not np.allclose(spacing, spacing[0])
            or step <= 0.0
            or step > 0.5 * spacing[0]
            or count < 1
        ):
            raise ValueError("Scattering grid or time step is invalid.")
        fraction = float(damping_fraction)
        damping_value = float(boundary_damping)
        if not 0.0 <= fraction < 0.5 or damping_value < 0.0:
            raise ValueError("Scattering damping controls are invalid.")
        edge = max(1, int(fraction * points.size))
        damping = np.zeros(points.size)
        ramp = np.linspace(1.0, 0.0, edge, endpoint=False)
        damping[:edge] = damping_value * ramp**2
        damping[-edge:] = damping_value * ramp[::-1] ** 2
        self.potential = potential
        self.gradient_matrix = jnp.asarray(gradient)
        self.spatial_points = jnp.asarray(points)
        self.time_step = step
        self.steps = count
        self.damping = jnp.asarray(damping)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "defect-scattering-plan",
                "potential": potential.potential_id,
                "gradient_matrix": array_tree_fingerprint(gradient),
                "spatial_points": array_tree_fingerprint(points),
                "time_step": step,
                "steps": count,
                "damping": array_tree_fingerprint(damping),
            }
        )


class DefectScatteringRun(StrictModule):
    fields: Array
    momenta: Array
    energies: Array
    topological_charge: Array
    relative_energy_change: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def run_defect_scattering(
    plan: DefectScatteringPlan,
    initial_fields: ArrayLike,
    initial_momenta: ArrayLike,
    /,
) -> DefectScatteringRun:
    """Evolve nonlinear multifield Klein–Gordon defects with damped boundaries."""

    fields = jnp.asarray(initial_fields, dtype=plan.spatial_points.dtype)
    momenta = jnp.asarray(initial_momenta, dtype=fields.dtype)
    expected = (plan.spatial_points.shape[0], plan.potential.field_count)
    if fields.shape != expected or momenta.shape != expected:
        raise ValueError("Defect scattering fields have the wrong shape.")
    spacing = float(plan.spatial_points[1] - plan.spatial_points[0])
    inverse_gradient = jnp.linalg.inv(plan.gradient_matrix)

    def acceleration(field: Array, momentum: Array) -> Array:
        laplacian = jnp.zeros_like(field)
        laplacian = laplacian.at[1:-1].set(
            (field[2:] - 2.0 * field[1:-1] + field[:-2]) / spacing**2
        )
        gradient = jax.vmap(plan.potential.gradient)(field)
        return (
            laplacian - gradient @ inverse_gradient.T - plan.damping[:, None] * momentum
        )

    def energy(field: Array, momentum: Array) -> Array:
        derivative = jnp.gradient(field, spacing, axis=0)
        kinetic = 0.5 * ein.contract(
            "ni,ij,nj->n", momentum, plan.gradient_matrix, momentum
        )
        spatial = 0.5 * ein.contract(
            "ni,ij,nj->n", derivative, plan.gradient_matrix, derivative
        )
        return jnp.trapezoid(
            kinetic + spatial + plan.potential.value(field), plan.spatial_points
        )

    field_history = [fields]
    momentum_history = [momenta]
    energies = [energy(fields, momenta)]
    momentum = momenta + 0.5 * plan.time_step * acceleration(fields, momenta)
    for _ in range(plan.steps):
        fields = fields + plan.time_step * momentum
        acceleration_value = acceleration(fields, momentum)
        momentum = momentum + plan.time_step * acceleration_value
        physical_momentum = momentum - 0.5 * plan.time_step * acceleration_value
        field_history.append(fields)
        momentum_history.append(physical_momentum)
        energies.append(energy(fields, physical_momentum))
    energy_values = jnp.stack(tuple(energies))
    relative = jnp.abs(energy_values[-1] - energy_values[0]) / jnp.maximum(
        1.0, jnp.abs(energy_values[0])
    )
    return DefectScatteringRun(
        fields=jnp.stack(tuple(field_history)),
        momenta=jnp.stack(tuple(momentum_history)),
        energies=energy_values,
        topological_charge=fields[-1] - fields[0],
        relative_energy_change=relative,
        finite=jnp.all(jnp.isfinite(energy_values)),
        plan_id=plan.plan_id,
    )


__all__ = [
    "DefectContinuationResult",
    "DefectScatteringPlan",
    "DefectScatteringRun",
    "MappedInfiniteDefectEvidence",
    "MappedInfiniteDefectPlan",
    "MappedInfiniteDefectResult",
    "PolynomialDefectPotential",
    "PolynomialPotentialTerm",
    "RadialDefectPlan",
    "RadialDefectResult",
    "continue_polynomial_defect",
    "run_defect_scattering",
    "solve_mapped_infinite_defect",
    "solve_radial_defect",
]

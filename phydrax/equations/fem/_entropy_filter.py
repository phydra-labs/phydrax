#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike
from scipy.special import eval_legendre

from ..._fingerprint import canonical_fingerprint
from ..._numerics._ssp_runge_kutta import (
    AbstractSSPRKStageTransform,
    StageTransformResult,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._conservation_boundary import PrescribedNormalFluxBoundary
from ...discretization.fem._boundary import tensor_local_face
from .._hyperbolic_systems import EulerSystem
from ._conservation import PreparedDGSEMConservationDynamics


class EntropyFilterEvidence(StrictModule, NonTrainableState):
    minimum_density: Array
    minimum_pressure: Array
    minimum_specific_entropy: Array
    maximum_filter_strength: Array
    minimum_linear_factor: Array
    mean_defect: Array
    applied: Array
    successful: Array
    filter_id: str = eqx.field(static=True)


class EntropyFilterPlan(StrictModule, NonTrainableState):
    density_floor: float | None = eqx.field(static=True)
    pressure_floor: float | None = eqx.field(static=True)
    entropy_tolerance: float = eqx.field(static=True)
    modal_exponent: int = eqx.field(static=True)
    maximum_strength: float = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    mean_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        density_floor: float | None = None,
        pressure_floor: float | None = None,
        entropy_tolerance: float = 1.0e-10,
        modal_exponent: int = 8,
        maximum_strength: float = 36.0,
        bisection_iterations: int = 24,
        mean_tolerance: float = 1.0e-10,
    ):
        density = None if density_floor is None else float(density_floor)
        pressure = None if pressure_floor is None else float(pressure_floor)
        entropy = float(entropy_tolerance)
        exponent = int(modal_exponent)
        strength = float(maximum_strength)
        iterations = int(bisection_iterations)
        mean = float(mean_tolerance)
        if density is not None and (not math.isfinite(density) or density <= 0.0):
            raise ValueError("density_floor must be finite and positive when supplied.")
        if pressure is not None and (not math.isfinite(pressure) or pressure <= 0.0):
            raise ValueError("pressure_floor must be finite and positive when supplied.")
        if (
            not math.isfinite(entropy)
            or entropy < 0.0
            or exponent <= 0
            or not math.isfinite(strength)
            or strength <= 0.0
            or iterations <= 0
            or not math.isfinite(mean)
            or mean < 0.0
        ):
            raise ValueError("Entropy-filter parameters are invalid.")
        self.density_floor = density
        self.pressure_floor = pressure
        self.entropy_tolerance = entropy
        self.modal_exponent = exponent
        self.maximum_strength = strength
        self.bisection_iterations = iterations
        self.mean_tolerance = mean
        self.plan_id = canonical_fingerprint(
            {
                "kind": "entropy-filter-plan",
                "density_floor": density,
                "pressure_floor": pressure,
                "entropy_tolerance": entropy,
                "modal_exponent": exponent,
                "maximum_strength": strength,
                "bisection_iterations": iterations,
                "mean_tolerance": mean,
                "differentiability": "branchwise-stopped-decisions",
            }
        )

    def prepare(
        self, dynamics: PreparedDGSEMConservationDynamics, /
    ) -> PreparedEntropyFilter:
        return PreparedEntropyFilter(self, dynamics)


class PreparedEntropyFilter(AbstractSSPRKStageTransform):
    plan: EntropyFilterPlan
    dynamics: PreparedDGSEMConservationDynamics
    nodal_to_modal: Array
    modal_to_nodal: Array
    modal_scale: Array
    local_mass_weights: Array
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: EntropyFilterPlan,
        dynamics: PreparedDGSEMConservationDynamics,
        /,
    ):
        if not isinstance(plan, EntropyFilterPlan):
            raise TypeError("plan must be EntropyFilterPlan.")
        if not isinstance(dynamics, PreparedDGSEMConservationDynamics):
            raise TypeError("dynamics must be PreparedDGSEMConservationDynamics.")
        if not isinstance(dynamics.system, EulerSystem):
            raise TypeError("The entropy filter currently requires EulerSystem.")
        if dynamics.entropy_pair is None:
            raise ValueError("The entropy filter requires prepared entropy diagnostics.")
        nodes = np.asarray(dynamics.sbp.nodes, dtype=float)
        degree = dynamics.sbp.order
        vandermonde = np.stack(
            tuple(eval_legendre(mode, 2.0 * nodes - 1.0) for mode in range(degree + 1)),
            axis=-1,
        )
        inverse = np.linalg.solve(vandermonde, np.eye(degree + 1))
        mode_axes = np.meshgrid(
            *(np.arange(degree + 1, dtype=float),) * dynamics.metrics.dimension,
            indexing="ij",
        )
        total_degree = sum(mode_axes)
        normalized = total_degree / max(float(degree), 1.0)
        modal_scale = normalized**plan.modal_exponent
        routes = dynamics.discretization.dof_maps[0].cell_dofs[0]
        cell_count = routes.shape[0]
        local_mass = dynamics.scalar_mass_weights[routes].reshape(
            (cell_count,) + (degree + 1,) * dynamics.metrics.dimension
        )
        density_floor = (
            dynamics.system.density_floor
            if plan.density_floor is None
            else plan.density_floor
        )
        pressure_floor = (
            dynamics.system.pressure_floor
            if plan.pressure_floor is None
            else plan.pressure_floor
        )
        self.plan = plan
        self.dynamics = dynamics
        self.nodal_to_modal = jnp.asarray(inverse)
        self.modal_to_nodal = jnp.asarray(vandermonde)
        self.modal_scale = jnp.asarray(modal_scale)
        self.local_mass_weights = local_mass
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.transform_id = canonical_fingerprint(
            {
                "kind": "prepared-entropy-filter",
                "plan": plan.plan_id,
                "dynamics": dynamics.dynamics_id,
                "nodes": dynamics.sbp.data_id,
            }
        )

    @property
    def _axes(self) -> tuple[int, ...]:
        return tuple(range(1, self.dynamics.metrics.dimension + 1))

    def _apply_axes(self, matrix: Array, value: Array, /) -> Array:
        result = value
        for axis in range(self.dynamics.metrics.dimension):
            moved = jnp.moveaxis(result, axis + 1, 1)
            moved = oe.contract("ij,cj...->ci...", matrix, moved, backend="jax")
            result = jnp.moveaxis(moved, 1, axis + 1)
        return result

    def _weighted_mean(self, local: Array, /) -> Array:
        weights = self.local_mass_weights
        numerator = jnp.sum(weights[..., None] * local, axis=self._axes)
        denominator = jnp.sum(weights, axis=self._axes)
        return numerator / denominator[:, None]

    def _broadcast_mean(self, mean: Array, /) -> Array:
        shape = (
            (mean.shape[0],) + (1,) * self.dynamics.metrics.dimension + (mean.shape[-1],)
        )
        return mean.reshape(shape)

    def _restore_mean(self, local: Array, target_mean: Array, /) -> Array:
        correction = target_mean - self._weighted_mean(local)
        return local + self._broadcast_mean(correction)

    def _specific_entropy(self, local: Array, /) -> Array:
        density = local[..., 0]
        pressure = self.dynamics.system.pressure(local)
        gamma = self.dynamics.system.material.gamma
        safe_density = jnp.maximum(density, self.density_floor)
        safe_pressure = jnp.maximum(pressure, self.pressure_floor)
        return jnp.log(safe_pressure) - gamma * jnp.log(safe_density)

    def _cell_minimum(self, value: Array, /) -> Array:
        return jnp.min(value, axis=self._axes)

    def _entropy_bounds(self, time: Array, local: Array, args: Any, /) -> Array:
        bounds = self._cell_minimum(self._specific_entropy(local))
        for pair in self.dynamics.face_pairs:
            owner = bounds[pair.owner_cell]
            neighbour = bounds[pair.neighbour_cell]
            bounds = bounds.at[pair.owner_cell].min(neighbour)
            bounds = bounds.at[pair.neighbour_cell].min(owner)
        if self.dynamics.boundaries is None:
            return bounds
        context = self.dynamics._context(time, args)
        cell_kind = self.dynamics.discretization.mesh.blocks[0].cell_kind
        for patch in self.dynamics.boundaries.patches:
            if isinstance(patch.boundary, PrescribedNormalFluxBoundary):
                continue
            owners = np.asarray(patch.domain.owner_cells, dtype=np.int32)
            local_facets = np.asarray(patch.domain.owner_local_entities, dtype=np.int32)
            for owner_cell, local_facet in zip(owners, local_facets, strict=True):
                axis, side = tensor_local_face(cell_kind, int(local_facet))
                plus = self.dynamics._face_value(local, int(owner_cell), axis, side)
                points = self.dynamics.metrics.face_coordinates[axis][
                    int(owner_cell), side
                ].reshape((-1, self.dynamics.metrics.dimension))
                scaled_normal = self.dynamics.metrics.face_scaled_normals[axis][
                    int(owner_cell), side
                ].reshape((-1, self.dynamics.metrics.dimension))
                measure = jnp.sqrt(
                    oe.contract("qd,qd->q", scaled_normal, scaled_normal, backend="jax")
                )
                normal = scaled_normal / measure[:, None]
                exterior = patch.boundary.exterior_state(
                    self.dynamics.system,
                    time,
                    plus,
                    points,
                    normal,
                    axis,
                    context.user_args,
                )
                exterior_minimum = jnp.min(self._specific_entropy(exterior))
                bounds = bounds.at[int(owner_cell)].min(exterior_minimum)
        return bounds

    def _admissibility(
        self, local: Array, entropy_bounds: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        density = local[..., 0]
        pressure = self.dynamics.system.pressure(local)
        entropy = self._specific_entropy(local)
        density_minimum = self._cell_minimum(density)
        pressure_minimum = self._cell_minimum(pressure)
        entropy_minimum = self._cell_minimum(entropy)
        finite = jnp.all(jnp.isfinite(local), axis=self._axes + (local.ndim - 1,))
        valid = (
            finite
            & (density_minimum >= self.density_floor)
            & (pressure_minimum >= self.pressure_floor)
            & (entropy_minimum >= entropy_bounds - self.plan.entropy_tolerance)
        )
        return valid, density_minimum, pressure_minimum, entropy_minimum

    def _modal_filter(
        self, modal: Array, strength: Array, target_mean: Array, /
    ) -> Array:
        shape = (strength.shape[0],) + (1,) * self.dynamics.metrics.dimension + (1,)
        scale = self.modal_scale.reshape((1,) + self.modal_scale.shape + (1,))
        factor = jnp.exp(-strength.reshape(shape) * scale)
        filtered = self._apply_axes(self.modal_to_nodal, modal * factor)
        return self._restore_mean(filtered, target_mean)

    def _safe_mean_state(self, mean: Array, /) -> Array:
        density = jnp.maximum(jnp.abs(mean[..., :1]), 2.0 * self.density_floor)
        momentum = jnp.zeros_like(mean[..., 1:-1])
        energy = jnp.full_like(
            mean[..., -1:],
            2.0 * self.pressure_floor / (self.dynamics.system.material.gamma - 1.0),
        )
        return jnp.concatenate((density, momentum, energy), axis=-1)

    def filter(
        self, time: Array, state: ArrayLike, args: Any = None, /
    ) -> tuple[Array, EntropyFilterEvidence]:
        value = self.dynamics._state(state)
        local = self.dynamics._local_state(value)
        target_mean = self._weighted_mean(local)
        entropy_bounds = self._entropy_bounds(jnp.asarray(time), local, args)
        modal = self._apply_axes(self.nodal_to_modal, local)
        cell_count = local.shape[0]
        zero = jnp.zeros((cell_count,), dtype=value.real.dtype)
        maximum = jnp.full_like(zero, self.plan.maximum_strength)
        unfiltered = self._modal_filter(modal, zero, target_mean)
        unfiltered_valid = self._admissibility(unfiltered, entropy_bounds)[0]
        low = zero
        high = maximum
        for _iteration in range(self.plan.bisection_iterations):
            midpoint = 0.5 * (low + high)
            candidate = self._modal_filter(modal, midpoint, target_mean)
            valid = self._admissibility(candidate, entropy_bounds)[0]
            high = jnp.where(valid, midpoint, high)
            low = jnp.where(valid, low, midpoint)
        strength = jnp.where(unfiltered_valid, zero, high)
        filtered = self._modal_filter(modal, strength, target_mean)
        filtered_valid = self._admissibility(filtered, entropy_bounds)[0]

        mean_local = jnp.broadcast_to(self._broadcast_mean(target_mean), local.shape)
        mean_valid = self._admissibility(mean_local, entropy_bounds)[0]
        theta_low = jnp.zeros_like(zero)
        theta_high = jnp.ones_like(zero)
        for _iteration in range(self.plan.bisection_iterations):
            midpoint = 0.5 * (theta_low + theta_high)
            candidate = mean_local + midpoint.reshape(
                (cell_count,) + (1,) * self.dynamics.metrics.dimension + (1,)
            ) * (filtered - mean_local)
            valid = self._admissibility(candidate, entropy_bounds)[0]
            theta_low = jnp.where(valid, midpoint, theta_low)
            theta_high = jnp.where(valid, theta_high, midpoint)
        theta = jnp.where(filtered_valid, jnp.ones_like(theta_low), theta_low)
        final = mean_local + theta.reshape(
            (cell_count,) + (1,) * self.dynamics.metrics.dimension + (1,)
        ) * (filtered - mean_local)
        final_valid, density_minimum, pressure_minimum, entropy_minimum = (
            self._admissibility(final, entropy_bounds)
        )
        mean_defect = jnp.max(jnp.abs(self._weighted_mean(final) - target_mean))
        cell_success = final_valid & mean_valid
        successful = jnp.all(cell_success) & (mean_defect <= self.plan.mean_tolerance)
        safe_mean = self._safe_mean_state(target_mean)
        safe_local = jnp.broadcast_to(self._broadcast_mean(safe_mean), local.shape)
        accepted_local = jnp.where(
            cell_success.reshape(
                (cell_count,) + (1,) * self.dynamics.metrics.dimension + (1,)
            ),
            final,
            safe_local,
        )
        routes = self.dynamics.discretization.dof_maps[0].cell_dofs[0]
        result = (
            jnp.zeros_like(value)
            .at[routes]
            .set(accepted_local.reshape(value[routes].shape))
        )
        applied = jnp.any(strength > self.plan.entropy_tolerance) | jnp.any(
            theta < 1.0 - self.plan.entropy_tolerance
        )
        evidence = EntropyFilterEvidence(
            jnp.min(density_minimum),
            jnp.min(pressure_minimum),
            jnp.min(entropy_minimum),
            jnp.max(strength),
            jnp.min(theta),
            mean_defect,
            applied,
            successful,
            self.transform_id,
        )
        return result, evidence

    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        del stage_index
        filtered, evidence = self.filter(time, candidate_state, args)
        return StageTransformResult(
            filtered,
            evidence.applied,
            evidence.successful,
            jnp.sqrt(jnp.sum(jnp.abs(filtered - candidate_state) ** 2)),
        )


__all__ = [
    "EntropyFilterEvidence",
    "EntropyFilterPlan",
    "PreparedEntropyFilter",
]

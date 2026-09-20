#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import ArraySpace, DenseLinearOperator, OperatorProperties
from ._hydrodynamic_mobility import (
    _active_slots,
    AbstractHydrodynamicMobilityPlan,
    AbstractPreparedHydrodynamicMobility,
)
from ._system import PreparedAtomisticSystem


def _mode_indices(extent: int) -> np.ndarray:
    return np.asarray(
        [
            mode
            for mode in product(range(-extent, extent + 1), repeat=3)
            if mode != (0, 0, 0)
        ],
        dtype=np.int32,
    )


def _spectral_blocks(
    positions: Array,
    modes: Array,
    reciprocal_vectors: Array,
    radius: float,
    viscosity: float,
    volume: float,
) -> tuple[Array, Array]:
    wavevector = contract("ki,ij->kj", modes, reciprocal_vectors)
    squared_wave = jnp.sum(wavevector * wavevector, axis=-1)
    projector = (
        jnp.eye(3, dtype=positions.dtype)
        - (wavevector[..., :, None] * wavevector[..., None, :])
        / squared_wave[..., None, None]
    )
    form = jnp.sinc(
        jnp.sqrt(squared_wave) * jnp.asarray(radius, dtype=positions.dtype) / jnp.pi
    )
    coefficient = form * form / squared_wave
    displacement = positions[:, None, :] - positions[None, :, :]
    phase = contract("ka,ija->kij", wavevector, displacement)
    blocks = contract(
        "k,kij,kab->ijab",
        coefficient,
        jnp.cos(phase),
        projector,
    ) / (jnp.asarray(viscosity, dtype=positions.dtype) * volume)
    self_block = contract("k,kab->ab", coefficient, projector) / (
        jnp.asarray(viscosity, dtype=positions.dtype) * volume
    )
    return blocks, self_block


def _dense_matrix(blocks: Array) -> Array:
    count = blocks.shape[0]
    return jnp.transpose(blocks, (0, 2, 1, 3)).reshape((3 * count, 3 * count))


def _positive_properties(*, definite: bool) -> OperatorProperties:
    evidence = {
        "self_adjoint": "construction",
        "positive_semidefinite": "construction",
    }
    if definite:
        evidence["positive_definite"] = "construction"
    return OperatorProperties(
        self_adjoint=True,
        positive_definite=definite,
        positive_semidefinite=True,
        evidence=evidence,
    )


class PeriodicRPYOracleEvidence(StrictModule):
    matrix: Array
    minimum_eigenvalue: Array
    symmetry_residual: Array
    shell_relative_residual: Array
    self_trace_residual: Array
    unresolved_self_mobility: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class DirectPeriodicRPYMobilityPlan(AbstractHydrodynamicMobilityPlan):
    hydrodynamic_radius: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    reciprocal_extent: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_radius: float,
        dynamic_viscosity: float,
        reciprocal_extent: int,
        /,
        *,
        maximum_particles: int,
        maximum_modes: int = 100_000,
        convergence_tolerance: float = 5.0e-3,
    ):
        radius = float(hydrodynamic_radius)
        viscosity = float(dynamic_viscosity)
        extent = int(reciprocal_extent)
        maximum = int(maximum_particles)
        mode_limit = int(maximum_modes)
        tolerance = float(convergence_tolerance)
        mode_count = (2 * extent + 1) ** 3 - 1
        if (
            not math.isfinite(radius)
            or radius <= 0.0
            or not math.isfinite(viscosity)
            or viscosity <= 0.0
            or extent < 2
            or maximum <= 0
            or mode_limit <= 0
            or mode_count > mode_limit
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Periodic RPY parameters or capacities are invalid.")
        self.hydrodynamic_radius = radius
        self.dynamic_viscosity = viscosity
        self.reciprocal_extent = extent
        self.maximum_particles = maximum
        self.maximum_modes = mode_limit
        self.convergence_tolerance = tolerance
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "direct-periodic-spectral-rpy",
                "radius": radius,
                "viscosity": viscosity,
                "reciprocal_extent": extent,
                "maximum_particles": maximum,
                "maximum_modes": mode_limit,
                "convergence_tolerance": tolerance,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> PreparedDirectPeriodicRPYMobility:
        return PreparedDirectPeriodicRPYMobility(self, system, active_slots)


class PreparedDirectPeriodicRPYMobility(AbstractPreparedHydrodynamicMobility):
    plan: DirectPeriodicRPYMobilityPlan
    system: PreparedAtomisticSystem
    active_slots: Array
    coordinate_space: ArraySpace
    modes: Array
    previous_modes: Array
    unresolved_self_mobility: Array
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DirectPeriodicRPYMobilityPlan,
        system: PreparedAtomisticSystem,
        active_slots: ArrayLike,
        /,
    ):
        if system.cell is None or not system.cell.fully_periodic or system.cell.rank != 3:
            raise ValueError("Direct periodic RPY requires a fully periodic 3-D cell.")
        slots = _active_slots(plan.maximum_particles, system, active_slots)
        modes = _mode_indices(plan.reciprocal_extent)
        previous = _mode_indices(plan.reciprocal_extent - 1)
        dtype = system.plan.coordinate_dtype
        _, self_block = _spectral_blocks(
            jnp.zeros((1, 3), dtype=dtype),
            jnp.asarray(modes, dtype=dtype),
            system.cell.reciprocal_vectors.astype(dtype),
            plan.hydrodynamic_radius,
            plan.dynamic_viscosity,
            system.cell.volume,
        )
        self_mobility = 1.0 / (
            6.0 * jnp.pi * plan.dynamic_viscosity * plan.hydrodynamic_radius
        )
        unresolved = self_mobility - jnp.trace(self_block) / 3.0
        if float(unresolved) <= 0.0:
            raise ValueError(
                "The reciprocal extent exhausts the positive unresolved self tail; "
                "reduce the extent or increase the cell size."
            )
        space = ArraySpace(
            (slots.size, 3),
            dtype=dtype,
            space_id=f"periodic-rpy:{plan.mobility_id}:coordinates",
        )
        self.plan = plan
        self.system = system
        self.active_slots = slots
        self.coordinate_space = space
        self.modes = jnp.asarray(modes, dtype=dtype)
        self.previous_modes = jnp.asarray(previous, dtype=dtype)
        self.unresolved_self_mobility = unresolved
        self.route_id = canonical_fingerprint(
            {
                "kind": "direct-periodic-spectral-rpy-route",
                "cell": system.cell.cell_id,
                "active_slots": np.asarray(slots).tolist(),
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-direct-periodic-rpy",
                "plan": plan.mobility_id,
                "system": system.prepared_id,
                "route": self.route_id,
            }
        )

    def _matrix(self, positions: Array, modes: Array, *, restore_self: bool) -> Array:
        cell = self.system.cell
        if cell is None:
            raise RuntimeError("Validated periodic RPY cell is absent.")
        blocks, self_block = _spectral_blocks(
            positions,
            modes,
            cell.reciprocal_vectors.astype(positions.dtype),
            self.plan.hydrodynamic_radius,
            self.plan.dynamic_viscosity,
            cell.volume,
        )
        if restore_self:
            self_mobility = 1.0 / (
                6.0 * jnp.pi * self.plan.dynamic_viscosity * self.plan.hydrodynamic_radius
            )
            tail = self_mobility - jnp.trace(self_block) / 3.0
            blocks = blocks + (
                jnp.eye(positions.shape[0], dtype=positions.dtype)[:, :, None, None]
                * tail
                * jnp.eye(3, dtype=positions.dtype)
            )
        return _dense_matrix(blocks)

    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        return jnp.all(jnp.isfinite(value))

    def operator(self, positions: ArrayLike, /) -> DenseLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        matrix = self._matrix(value, self.modes, restore_self=True)
        return DenseLinearOperator(
            matrix,
            source=self.coordinate_space,
            target=self.coordinate_space,
            properties=_positive_properties(definite=True),
            operator_id=f"{self.prepared_id}:operator",
        )

    def evaluate(self, positions: ArrayLike, /) -> PeriodicRPYOracleEvidence:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        matrix = self._matrix(value, self.modes, restore_self=True)
        previous = self._matrix(value, self.previous_modes, restore_self=True)
        scale = jnp.maximum(
            jnp.sqrt(jnp.sum(matrix * matrix)), jnp.finfo(matrix.dtype).tiny
        )
        shell_residual = jnp.sqrt(jnp.sum((matrix - previous) ** 2)) / scale
        symmetry = jnp.max(jnp.abs(matrix - matrix.T))
        minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T)))
        diagonal_blocks = jnp.diagonal(
            matrix.reshape((value.shape[0], 3, value.shape[0], 3)),
            axis1=0,
            axis2=2,
        )
        self_mobility = 1.0 / (
            6.0 * jnp.pi * self.plan.dynamic_viscosity * self.plan.hydrodynamic_radius
        )
        self_residual = jnp.max(
            jnp.abs(jnp.trace(diagonal_blocks, axis1=0, axis2=1) / 3.0 - self_mobility)
        )
        finite = jnp.all(jnp.isfinite(matrix)) & jnp.isfinite(shell_residual)
        successful = (
            finite
            & (symmetry <= 64.0 * jnp.finfo(matrix.dtype).eps)
            & (minimum > 0.0)
            & (shell_residual <= self.plan.convergence_tolerance)
        )
        return PeriodicRPYOracleEvidence(
            matrix,
            minimum,
            symmetry,
            shell_residual,
            self_residual,
            self.unresolved_self_mobility,
            finite,
            successful,
            self.prepared_id,
        )


class PositiveSplitPeriodicRPYMobilityPlan(AbstractHydrodynamicMobilityPlan):
    hydrodynamic_radius: float = eqx.field(static=True)
    dynamic_viscosity: float = eqx.field(static=True)
    reciprocal_extent: int = eqx.field(static=True)
    wave_extent: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)

    def __init__(
        self,
        hydrodynamic_radius: float,
        dynamic_viscosity: float,
        reciprocal_extent: int,
        wave_extent: int,
        /,
        *,
        maximum_particles: int,
        maximum_modes: int = 100_000,
        convergence_tolerance: float = 5.0e-3,
    ):
        direct = DirectPeriodicRPYMobilityPlan(
            hydrodynamic_radius,
            dynamic_viscosity,
            reciprocal_extent,
            maximum_particles=maximum_particles,
            maximum_modes=maximum_modes,
            convergence_tolerance=convergence_tolerance,
        )
        wave = int(wave_extent)
        if wave <= 0 or wave >= direct.reciprocal_extent:
            raise ValueError("wave_extent must lie strictly inside reciprocal_extent.")
        self.hydrodynamic_radius = direct.hydrodynamic_radius
        self.dynamic_viscosity = direct.dynamic_viscosity
        self.reciprocal_extent = direct.reciprocal_extent
        self.wave_extent = wave
        self.maximum_particles = direct.maximum_particles
        self.maximum_modes = direct.maximum_modes
        self.convergence_tolerance = direct.convergence_tolerance
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "positive-split-periodic-spectral-rpy",
                "direct": direct.mobility_id,
                "wave_extent": wave,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> PreparedPositiveSplitPeriodicRPYMobility:
        return PreparedPositiveSplitPeriodicRPYMobility(self, system, active_slots)


class PositiveSplitPeriodicRPYEvidence(StrictModule):
    total_matrix: Array
    wave_matrix: Array
    local_matrix: Array
    split_residual: Array
    minimum_total_eigenvalue: Array
    minimum_wave_eigenvalue: Array
    minimum_local_eigenvalue: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedPositiveSplitPeriodicRPYMobility(AbstractPreparedHydrodynamicMobility):
    plan: PositiveSplitPeriodicRPYMobilityPlan
    system: PreparedAtomisticSystem
    active_slots: Array
    coordinate_space: ArraySpace
    direct: PreparedDirectPeriodicRPYMobility
    wave_modes: Array
    local_modes: Array
    unresolved_self_mobility: Array
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PositiveSplitPeriodicRPYMobilityPlan,
        system: PreparedAtomisticSystem,
        active_slots: ArrayLike,
        /,
    ):
        direct_plan = DirectPeriodicRPYMobilityPlan(
            plan.hydrodynamic_radius,
            plan.dynamic_viscosity,
            plan.reciprocal_extent,
            maximum_particles=plan.maximum_particles,
            maximum_modes=plan.maximum_modes,
            convergence_tolerance=plan.convergence_tolerance,
        )
        direct = direct_plan.prepare(system, active_slots)
        host_modes = np.asarray(direct.modes)
        wave_mask = np.max(np.abs(host_modes), axis=1) <= plan.wave_extent
        self.plan = plan
        self.system = system
        self.active_slots = direct.active_slots
        self.coordinate_space = ArraySpace(
            direct.coordinate_space.shape,
            dtype=direct.coordinate_space.dtype,
            space_id=f"positive-split-rpy:{plan.mobility_id}:coordinates",
        )
        self.direct = direct
        self.wave_modes = direct.modes[jnp.asarray(wave_mask)]
        self.local_modes = direct.modes[jnp.asarray(~wave_mask)]
        self.unresolved_self_mobility = direct.unresolved_self_mobility
        self.route_id = canonical_fingerprint(
            {
                "kind": "positive-split-periodic-rpy-route",
                "cell": system.cell.cell_id if system.cell is not None else "missing",
                "active_slots": np.asarray(self.active_slots).tolist(),
                "wave_extent": plan.wave_extent,
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-positive-split-periodic-rpy",
                "plan": plan.mobility_id,
                "system": system.prepared_id,
                "route": self.route_id,
            }
        )

    def _matrix(self, positions: Array, modes: Array) -> Array:
        return self.direct._matrix(positions, modes, restore_self=False)

    def wave_operator(self, positions: ArrayLike, /) -> DenseLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        matrix = self._matrix(value, self.wave_modes)
        return DenseLinearOperator(
            matrix,
            source=self.coordinate_space,
            target=self.coordinate_space,
            properties=_positive_properties(definite=False),
            operator_id=f"{self.prepared_id}:wave",
        )

    def local_operator(self, positions: ArrayLike, /) -> DenseLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        matrix = self._matrix(value, self.local_modes)
        matrix = matrix + self.unresolved_self_mobility * jnp.eye(
            matrix.shape[0], dtype=matrix.dtype
        )
        return DenseLinearOperator(
            matrix,
            source=self.coordinate_space,
            target=self.coordinate_space,
            properties=_positive_properties(definite=True),
            operator_id=f"{self.prepared_id}:local",
        )

    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        return jnp.all(jnp.isfinite(value))

    def operator(self, positions: ArrayLike, /) -> DenseLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        matrix = self.wave_operator(value).matrix + self.local_operator(value).matrix
        return DenseLinearOperator(
            matrix,
            source=self.coordinate_space,
            target=self.coordinate_space,
            properties=_positive_properties(definite=True),
            operator_id=f"{self.prepared_id}:operator",
        )

    def evaluate(self, positions: ArrayLike, /) -> PositiveSplitPeriodicRPYEvidence:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        wave = self.wave_operator(value).matrix
        local = self.local_operator(value).matrix
        total = self.operator(value).matrix
        scale = jnp.maximum(jnp.sqrt(jnp.sum(total * total)), jnp.finfo(total.dtype).tiny)
        split_residual = jnp.sqrt(jnp.sum((total - wave - local) ** 2)) / scale
        total_minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (total + total.T)))
        wave_minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (wave + wave.T)))
        local_minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (local + local.T)))
        finite = (
            jnp.all(jnp.isfinite(total))
            & jnp.all(jnp.isfinite(wave))
            & jnp.all(jnp.isfinite(local))
        )
        successful = (
            finite
            & (split_residual <= 64.0 * jnp.finfo(total.dtype).eps)
            & (total_minimum > 0.0)
            & (wave_minimum >= -128.0 * jnp.finfo(total.dtype).eps)
            & (local_minimum > 0.0)
        )
        return PositiveSplitPeriodicRPYEvidence(
            total,
            wave,
            local,
            split_residual,
            total_minimum,
            wave_minimum,
            local_minimum,
            finite,
            successful,
            self.prepared_id,
        )


__all__ = [
    "DirectPeriodicRPYMobilityPlan",
    "PeriodicRPYOracleEvidence",
    "PositiveSplitPeriodicRPYEvidence",
    "PositiveSplitPeriodicRPYMobilityPlan",
    "PreparedDirectPeriodicRPYMobility",
    "PreparedPositiveSplitPeriodicRPYMobility",
]

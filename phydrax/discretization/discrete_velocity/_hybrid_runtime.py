#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from itertools import product
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...closure_data._kinetic_equilibrium import (
    EnergyEquilibriumSupportEvidence,
    PreparedLearnedEnergyEquilibriumBinding,
)
from ...solver._finite_volume import PreparedFiniteVolumeDynamics
from ...solver._finite_volume_runtime import (
    FiniteVolumeAdvanceResult,
    FiniteVolumeRuntimeState,
    FiniteVolumeScheduledAdvanceResult,
    FiniteVolumeStageFlux,
    FiniteVolumeStageFluxProvider,
    FiniteVolumeStageFluxTrace,
    PreparedFiniteVolumeRuntime,
)
from ._hybrid import (
    CommonFVKineticFluxEvidence,
    FixedConformingFVKineticInterfacePlan,
)
from ._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
    SmoothCompressibleLearnedCollisionResult,
    SmoothCompressibleLearnedEquilibriumEvidence,
    SmoothCompressibleRealizabilityEvidence,
)
from ._spatial import PreparedSmoothCompressibleD2V17SpatialDynamics
from ._spatial_boundary import SpecularAdiabaticD2VBoundaryPlan


_SSPRK3_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)


class FixedPartitionHybridStatus(IntEnum):
    SUCCESS = 0
    FINITE_VOLUME_REJECTED = 1
    KINETIC_COLLISION_REJECTED = 2
    LEARNED_LIFT_REJECTED = 3
    FINITE_VOLUME_POSITIVITY_REJECTED = 4
    KINETIC_REALIZABILITY_REJECTED = 5
    CONSERVATION_REJECTED = 6


class FixedPartitionHybridState(StrictModule):
    """Only the jointly committed FV and kinetic states are checkpointable."""

    finite_volume: FiniteVolumeRuntimeState
    kinetic: SmoothCompressibleKineticState
    checkpoint_eligible: Array

    def __init__(
        self,
        finite_volume: FiniteVolumeRuntimeState,
        kinetic: SmoothCompressibleKineticState,
        /,
        *,
        checkpoint_eligible: ArrayLike = True,
    ):
        if not isinstance(finite_volume, FiniteVolumeRuntimeState):
            raise TypeError("finite_volume must be FiniteVolumeRuntimeState.")
        if not isinstance(kinetic, SmoothCompressibleKineticState):
            raise TypeError("kinetic must be SmoothCompressibleKineticState.")
        eligible = jnp.asarray(checkpoint_eligible, dtype=jnp.bool_)
        if eligible.shape != ():
            raise ValueError("checkpoint_eligible must be scalar.")
        self.finite_volume = finite_volume
        self.kinetic = kinetic
        self.checkpoint_eligible = eligible

    @property
    def time(self) -> Array:
        return self.finite_volume.time

    @property
    def accepted_step(self) -> Array:
        return self.finite_volume.accepted_step


class FixedPartitionHybridCheckpoint(StrictModule, NonTrainableState):
    """Accepted-state-only restart payload bound to one frozen model artifact."""

    state: FixedPartitionHybridState
    checkpoint_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    learned_energy_artifact_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class FixedHybridStageEvidence(StrictModule):
    """The exact population fluxes and learned-lift evidence for one FV stage."""

    common_fluxes: tuple[CommonFVKineticFluxEvidence, ...]
    learned_support: tuple[EnergyEquilibriumSupportEvidence, ...]
    successful: Array


class FixedPartitionHybridAudit(StrictModule):
    pre_step_content: Array
    candidate_content: Array
    finite_volume_content_change: Array
    kinetic_content_change: Array
    interface_conservative_integrals: Array
    kinetic_outer_boundary_exchange: Array
    kinetic_moment_exchange_residual: Array
    finite_volume_interface_flux_residual: Array
    global_conservation_residual: Array
    maximum_absolute_residual: Array


class FixedPartitionHybridAdvanceEvidence(StrictModule):
    accepted: Array
    rollback_applied: Array
    exact_step: Array
    finite_volume_accepted: Array
    kinetic_collision_accepted: Array
    kinetic_interface_route_accepted: Array
    learned_lifts_accepted: Array
    finite_volume_positive: Array
    kinetic_realizability: SmoothCompressibleRealizabilityEvidence
    audit: FixedPartitionHybridAudit
    status: Array
    shock_owner: str = eqx.field(static=True)
    ownership_differentiability: str = eqx.field(static=True)


class FixedPartitionHybridAdvanceResult(StrictModule):
    previous: FixedPartitionHybridState
    candidate: FixedPartitionHybridState
    runtime_state: FixedPartitionHybridState
    finite_volume: FiniteVolumeScheduledAdvanceResult
    kinetic_collision: SmoothCompressibleLearnedCollisionResult
    stage_flux_trace: FiniteVolumeStageFluxTrace
    evidence: FixedPartitionHybridAdvanceEvidence


class _FixedHybridStageFluxCallback(StrictModule, NonTrainableState):
    finite_volume: PreparedFiniteVolumeRuntime
    learned_energy: PreparedLearnedEnergyEquilibriumBinding
    interfaces: tuple[FixedConformingFVKineticInterfacePlan, ...]
    finite_volume_face_axes: tuple[int, ...] = eqx.field(static=True)
    finite_volume_face_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    finite_volume_cell_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    kinetic_cell_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    normal_signs: tuple[float, ...] = eqx.field(static=True)
    kinetic_trace: SmoothCompressibleKineticState

    def _kinetic_face_trace(
        self,
        kinetic_cell: tuple[int, int],
        normal_sign: float,
        /,
    ) -> SmoothCompressibleKineticState:
        """Average every outbound D2V link layer into one FV face trace."""

        particles = self.kinetic_trace.particle_populations[kinetic_cell]
        energy = self.kinetic_trace.total_energy_populations[kinetic_cell]
        velocities = np.asarray(self.interfaces[0].method.quadrature.velocities)
        nx = self.kinetic_trace.particle_populations.shape[0]
        y_index = kinetic_cell[1]
        for direction, velocity in enumerate(velocities):
            normal_velocity = normal_sign * float(velocity[0])
            if normal_velocity >= 0.0:
                continue
            reach = int(abs(velocity[0]))
            x_indices = jnp.asarray(
                tuple(range(reach))
                if normal_sign > 0.0
                else tuple(range(nx - reach, nx)),
                dtype=jnp.int32,
            )
            particles = particles.at[direction].set(
                jnp.mean(
                    self.kinetic_trace.particle_populations[x_indices, y_index, direction]
                )
            )
            energy = energy.at[direction].set(
                jnp.mean(
                    self.kinetic_trace.total_energy_populations[
                        x_indices, y_index, direction
                    ]
                )
            )
        return SmoothCompressibleKineticState(particles, energy)

    def __call__(
        self,
        stage_index: int,
        time: Array,
        state: Array,
        /,
    ) -> FiniteVolumeStageFlux:
        del stage_index, time
        discretization = self.finite_volume.dynamics.discretization
        replacements = tuple(
            jnp.zeros(
                layout.shape + (state.shape[-1],),
                dtype=state.dtype,
            )
            for layout in discretization.face_layouts
        )
        masks = tuple(
            jnp.zeros(layout.shape, dtype=jnp.bool_)
            for layout in discretization.face_layouts
        )
        replacement_values = list(replacements)
        mask_values = list(masks)
        common_fluxes = []
        support_evidence = []
        successful = []
        energy_plan = self.learned_energy.plan.equilibrium_plan
        for (
            interface,
            axis,
            face_index,
            finite_volume_cell,
            kinetic_cell,
            normal_sign,
        ) in zip(
            self.interfaces,
            self.finite_volume_face_axes,
            self.finite_volume_face_indices,
            self.finite_volume_cell_indices,
            self.kinetic_cell_indices,
            self.normal_signs,
            strict=True,
        ):
            conserved = state[finite_volume_cell]
            kinetic = self._kinetic_face_trace(kinetic_cell, normal_sign)
            dual, support = self.learned_energy.predict_dual_with_evidence(conserved)
            common = interface.common_flux_from_energy_dual(
                conserved,
                kinetic,
                dual,
                energy_plan,
            )
            learned = common.learned_lift_evidence
            if not isinstance(learned, SmoothCompressibleLearnedEquilibriumEvidence):
                raise RuntimeError("Explicit learned common flux has no lift evidence.")
            coordinate_flux = normal_sign * common.common_conservative_flux
            replacement_values[axis] = (
                replacement_values[axis].at[face_index].set(coordinate_flux)
            )
            mask_values[axis] = mask_values[axis].at[face_index].set(True)
            common_fluxes.append(common)
            support_evidence.append(support)
            successful.append(jnp.all(support.successful) & learned.successful)
        stacked_flux = jnp.stack(
            tuple(value.common_conservative_flux for value in common_fluxes), axis=0
        )
        stage_evidence = FixedHybridStageEvidence(
            common_fluxes=tuple(common_fluxes),
            learned_support=tuple(support_evidence),
            successful=jnp.all(jnp.stack(tuple(successful))),
        )
        return FiniteVolumeStageFlux(
            tuple(replacement_values),
            tuple(mask_values),
            stacked_flux,
            stage_evidence,
        )


class PreparedFixedPartitionHybridRuntime(StrictModule, NonTrainableState):
    """Atomic fixed-step FV/D2V17 runtime with one common stage flux per face."""

    finite_volume: PreparedFiniteVolumeRuntime
    spatial: PreparedSmoothCompressibleD2V17SpatialDynamics
    learned_energy: PreparedLearnedEnergyEquilibriumBinding
    interfaces: tuple[FixedConformingFVKineticInterfacePlan, ...]
    finite_volume_face_axes: tuple[int, ...] = eqx.field(static=True)
    finite_volume_face_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    finite_volume_cell_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    kinetic_cell_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    normal_signs: tuple[float, ...] = eqx.field(static=True)
    kinetic_outer_boundary: SpecularAdiabaticD2VBoundaryPlan
    face_measures: Array
    conservation_tolerance: float = eqx.field(static=True)
    shock_owner: str = eqx.field(static=True)
    ownership_differentiability: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        finite_volume: PreparedFiniteVolumeRuntime,
        spatial: PreparedSmoothCompressibleD2V17SpatialDynamics,
        learned_energy: PreparedLearnedEnergyEquilibriumBinding,
        interfaces: tuple[FixedConformingFVKineticInterfacePlan, ...],
        finite_volume_face_axes: tuple[int, ...],
        finite_volume_face_indices: tuple[tuple[int, ...], ...],
        finite_volume_cell_indices: tuple[tuple[int, ...], ...],
        kinetic_cell_indices: tuple[tuple[int, int], ...],
        /,
        *,
        conservation_tolerance: float = 1.0e-10,
    ):
        if not isinstance(finite_volume, PreparedFiniteVolumeRuntime):
            raise TypeError("finite_volume must be PreparedFiniteVolumeRuntime.")
        if not isinstance(finite_volume.dynamics, PreparedFiniteVolumeDynamics):
            raise ValueError("Fixed hybrid coupling requires stationary structured FV.")
        if finite_volume.stage_flux_provider is not None:
            raise ValueError("Hybrid runtime requires an unbound FV stage-flux provider.")
        if finite_volume.policy.maximum_retries != 0:
            raise ValueError("Fixed hybrid coupling forbids finite-volume retries.")
        if not isinstance(spatial, PreparedSmoothCompressibleD2V17SpatialDynamics):
            raise TypeError(
                "spatial must be PreparedSmoothCompressibleD2V17SpatialDynamics."
            )
        if spatial.boundary is not None or spatial.forcing is not None:
            raise ValueError(
                "Fixed hybrid runtime owns its interface routing and currently "
                "requires an unforced spatial collision plan."
            )
        if not isinstance(learned_energy, PreparedLearnedEnergyEquilibriumBinding):
            raise TypeError(
                "learned_energy must be PreparedLearnedEnergyEquilibriumBinding."
            )
        interfaces_ = tuple(interfaces)
        axes = tuple(finite_volume_face_axes)
        face_indices = tuple(tuple(value) for value in finite_volume_face_indices)
        fv_cells = tuple(tuple(value) for value in finite_volume_cell_indices)
        kinetic_cells = tuple(tuple(value) for value in kinetic_cell_indices)
        count = len(interfaces_)
        if (
            count == 0
            or any(
                not isinstance(value, FixedConformingFVKineticInterfacePlan)
                for value in interfaces_
            )
            or any(
                len(value) != count
                for value in (axes, face_indices, fv_cells, kinetic_cells)
            )
        ):
            raise ValueError(
                "Fixed hybrid interface routes must be aligned and nonempty."
            )
        if len(set(zip(axes, face_indices, strict=True))) != count:
            raise ValueError("Each fixed hybrid FV face may be coupled only once.")
        if set(axes) != {0}:
            raise ValueError(
                "Fixed D2V17 hybrid routing currently supports x-normal interfaces only."
            )
        tolerance = float(conservation_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")

        dynamics = finite_volume.dynamics
        discretization = dynamics.discretization
        cell_shape = tuple(discretization.state_shape[:-1])
        dimension = len(cell_shape)
        method = spatial.method
        if (
            dimension != 2
            or dynamics.system.dimension != 2
            or dynamics.system.component_count != 4
            or tuple(spatial.transport.spatial_shape) != cell_shape
        ):
            raise ValueError(
                "Fixed hybrid runtime requires matching two-dimensional FV and D2V17 grids."
            )
        if (
            learned_energy.plan.equilibrium_plan.plan_id != spatial.energy_plan.plan_id
            or learned_energy.plan.material.material_id != method.material.material_id
            or learned_energy.plan.equilibrium_plan.quadrature.quadrature_id
            != method.quadrature.quadrature_id
        ):
            raise ValueError(
                "Frozen learned-energy binding, spatial dynamics, and material must match."
            )
        covered_kinetic_rows = {-1.0: set(), 1.0: set()}

        normal_signs = []
        measures = []
        for interface, axis, face_index, fv_cell, kinetic_cell in zip(
            interfaces_, axes, face_indices, fv_cells, kinetic_cells, strict=True
        ):
            if (
                interface.method.method_id != method.method_id
                or interface.finite_volume_system.system_id != dynamics.system.system_id
            ):
                raise ValueError("Every interface primitive must match both solvers.")
            if axis < 0 or axis >= dimension:
                raise ValueError("Finite-volume face axis is outside the grid dimension.")
            face_shape = tuple(discretization.face_layouts[axis].shape)
            if (
                len(face_index) != dimension
                or any(
                    index < 0 or index >= face_shape[i]
                    for i, index in enumerate(face_index)
                )
                or len(fv_cell) != dimension
                or any(
                    index < 0 or index >= cell_shape[i] for i, index in enumerate(fv_cell)
                )
                or len(kinetic_cell) != 2
                or any(
                    index < 0 or index >= spatial.transport.spatial_shape[i]
                    for i, index in enumerate(kinetic_cell)
                )
            ):
                raise ValueError("A fixed hybrid interface route lies outside its grid.")
            if discretization.grid.structured_axes[axis].periodic:
                raise ValueError(
                    "Fixed hybrid interfaces must replace physical FV faces."
                )
            if face_index[axis] == 0:
                sign = -1.0
                expected_cell = tuple(
                    0 if i == axis else face_index[i] for i in range(dimension)
                )
            elif face_index[axis] == face_shape[axis] - 1:
                sign = 1.0
                expected_cell = tuple(
                    cell_shape[i] - 1 if i == axis else face_index[i]
                    for i in range(dimension)
                )
            else:
                raise ValueError("Fixed hybrid FV routes must select boundary faces.")
            expected_normal = np.zeros((dimension,), dtype=np.float64)
            expected_normal[axis] = sign
            if fv_cell != expected_cell or not np.allclose(
                np.asarray(interface.normal), expected_normal, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    "FV cell, boundary face, and outward interface normal are inconsistent."
                )
            normal_signs.append(sign)
            expected_kinetic_x = (
                0 if sign > 0.0 else spatial.transport.spatial_shape[0] - 1
            )
            if kinetic_cell[0] != expected_kinetic_x:
                raise ValueError(
                    "A kinetic interface cell must lie on its matching x boundary."
                )
            covered_kinetic_rows[sign].add(kinetic_cell[1])
            measure = float(np.asarray(discretization.face_measures[axis][face_index]))
            volume = float(np.asarray(discretization.cell_volumes[fv_cell]))
            expected_measure = float(
                np.prod(
                    tuple(
                        spatial.transport.cell_spacing[index]
                        for index in range(dimension)
                        if index != axis
                    )
                )
            )
            expected_volume = spatial.transport.cell_volume
            if not (
                np.isclose(measure, expected_measure, rtol=1.0e-12, atol=1.0e-14)
                and np.isclose(volume, expected_volume, rtol=1.0e-12, atol=1.0e-14)
            ):
                raise ValueError(
                    "Fixed hybrid FV faces and cells must match the D2V spatial metric."
                )
            measures.append(measure)
        expected_rows = set(range(spatial.transport.spatial_shape[1]))
        if any(covered_kinetic_rows[sign] != expected_rows for sign in (-1.0, 1.0)):
            raise ValueError(
                "Fixed hybrid routing requires every y row on both kinetic x interfaces."
            )

        self.finite_volume = finite_volume
        self.spatial = spatial
        self.learned_energy = learned_energy
        self.interfaces = interfaces_
        self.finite_volume_face_axes = axes
        self.finite_volume_face_indices = face_indices
        self.finite_volume_cell_indices = fv_cells
        self.kinetic_cell_indices = kinetic_cells
        self.normal_signs = tuple(normal_signs)
        outer_boundary = SpecularAdiabaticD2VBoundaryPlan(
            method.quadrature,
            spatial.transport.spatial_shape,
            spatial.transport.cell_spacing,
            spatial.required_step_size,
            periodic_axes=(False, True),
        )
        self.face_measures = jnp.asarray(
            measures, dtype=method.quadrature.velocities.dtype
        )
        self.conservation_tolerance = tolerance
        self.shock_owner = "finite_volume"
        self.ownership_differentiability = "none"
        self.kinetic_outer_boundary = outer_boundary
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-partition-fv-d2v17-hybrid-runtime",
                "finite_volume": finite_volume.runtime_id,
                "spatial": spatial.prepared_id,
                "learned_energy_artifact": learned_energy.prepared_id,
                "kinetic_outer_boundary": outer_boundary.plan_id,
                "interfaces": [value.plan_id for value in interfaces_],
                "finite_volume_face_axes": axes,
                "finite_volume_face_indices": face_indices,
                "finite_volume_cell_indices": fv_cells,
                "kinetic_cell_indices": kinetic_cells,
                "required_step_size": spatial.required_step_size,
                "conservation_tolerance": tolerance,
                "stage_weights": _SSPRK3_WEIGHTS,
                "shock_owner": self.shock_owner,
                "ownership_differentiability": self.ownership_differentiability,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.spatial.required_step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def initialize_state(
        self,
        finite_volume: FiniteVolumeRuntimeState,
        kinetic: SmoothCompressibleKineticState,
        /,
    ) -> FixedPartitionHybridState:
        state = FixedPartitionHybridState(finite_volume, kinetic)
        self._validate_state(state)
        return state

    def _validate_state(self, state: FixedPartitionHybridState, /) -> None:
        if not isinstance(state, FixedPartitionHybridState):
            raise TypeError("state must be FixedPartitionHybridState.")
        if (
            state.finite_volume.content_state.geometry_layout_id
            != self.finite_volume.geometry_layout_id
        ):
            raise ValueError("Hybrid FV state does not match the prepared runtime.")
        if tuple(
            self.finite_volume._dynamics_cell_average(
                state.finite_volume.content_state
            ).shape[:-1]
        ) != tuple(self.spatial.transport.spatial_shape):
            raise ValueError("Hybrid FV state has the wrong spatial shape.")
        self.spatial.validate_state(state.kinetic)

    def _require_exact_step(
        self, state: FixedPartitionHybridState, step_size: ArrayLike, /
    ) -> Array:
        requested = np.asarray(step_size)
        required = float(self.required_step_size)
        current = np.asarray(state.finite_volume.step_size)
        if (
            requested.shape != ()
            or not np.issubdtype(requested.dtype, np.inexact)
            or not np.isfinite(requested)
            or float(requested) != required
            or current.shape != ()
            or not np.isfinite(current)
            or float(current) != required
        ):
            raise ValueError(
                "Hybrid advancement requires the exact fixed lattice step and refuses reduction."
            )
        return jnp.asarray(required, dtype=state.kinetic.particle_populations.dtype)

    def _stage_provider(
        self, kinetic_trace: SmoothCompressibleKineticState, /
    ) -> FiniteVolumeStageFluxProvider:
        callback = _FixedHybridStageFluxCallback(
            self.finite_volume,
            self.learned_energy,
            self.interfaces,
            self.finite_volume_face_axes,
            self.finite_volume_face_indices,
            self.finite_volume_cell_indices,
            self.kinetic_cell_indices,
            self.normal_signs,
            kinetic_trace,
        )
        provider_id = canonical_fingerprint(
            {
                "kind": "fixed-partition-hybrid-stage-flux-provider",
                "runtime": self.runtime_id,
                "kinetic_trace": array_tree_fingerprint(kinetic_trace),
            }
        )
        return FiniteVolumeStageFluxProvider(callback, provider_id=provider_id)

    def _kinetic_content(self, state: SmoothCompressibleKineticState, /) -> Array:
        volume = jnp.asarray(
            self.spatial.transport.cell_volume,
            dtype=state.particle_populations.dtype,
        )
        mass = jnp.sum(state.particle_populations) * volume
        momentum = (
            jnp.sum(
                ein.contract(
                    "xyq,qd->xyd",
                    state.particle_populations,
                    self.spatial.method.quadrature.velocities,
                ),
                axis=(0, 1),
            )
            * volume
        )
        energy = jnp.sum(state.total_energy_populations) * volume
        return jnp.concatenate((mass[None], momentum, energy[None]), axis=0)

    @staticmethod
    def _finite_volume_content(state: FiniteVolumeRuntimeState, /) -> Array:
        return jnp.sum(state.content_state.conservative_content, axis=0)

    def _exchange_candidate(
        self,
        collision_state: SmoothCompressibleKineticState,
        trace: FiniteVolumeStageFluxTrace,
        step_size: Array,
        /,
    ) -> tuple[SmoothCompressibleKineticState, Array, Array, Array, Array]:
        """Route FV-owned incoming links instead of superposing a periodic flux."""

        outer_route = self.kinetic_outer_boundary.route(collision_state)
        particles = outer_route.candidate_state.particle_populations
        energy = outer_route.candidate_state.total_energy_populations
        conservative_integrals = []
        moment_residuals = []
        velocity = self.spatial.method.quadrature.velocities
        velocity_host = np.asarray(velocity)
        nx, _ = self.spatial.transport.spatial_shape
        stage_evidence = tuple(stage.evidence for stage in trace.stages)
        if any(
            not isinstance(value, FixedHybridStageEvidence) for value in stage_evidence
        ):
            raise TypeError("Hybrid FV trace contains foreign stage evidence.")
        for interface_index, kinetic_cell in enumerate(self.kinetic_cell_indices):
            common = tuple(
                value.common_fluxes[interface_index] for value in stage_evidence
            )
            weighted_particles = sum(
                weight * value.particle_population_flux
                for weight, value in zip(_SSPRK3_WEIGHTS, common, strict=True)
            )
            weighted_energy = sum(
                weight * value.total_energy_population_flux
                for weight, value in zip(_SSPRK3_WEIGHTS, common, strict=True)
            )
            weighted_conservative = sum(
                weight * value.common_conservative_flux
                for weight, value in zip(_SSPRK3_WEIGHTS, common, strict=True)
            )
            normal_sign = self.normal_signs[interface_index]
            y_index = kinetic_cell[1]
            for direction, offset in enumerate(self.spatial.transport.pull_offsets):
                normal_velocity = normal_sign * float(velocity_host[direction, 0])
                if normal_velocity <= 0.0:
                    continue
                reach = abs(offset[0])
                if reach == 0:
                    raise RuntimeError(
                        "Incoming hybrid links must cross the x-normal interface."
                    )
                x_indices = (
                    tuple(range(reach))
                    if normal_sign > 0.0
                    else tuple(range(nx - reach, nx))
                )
                incoming_particles = weighted_particles[direction] / normal_velocity
                incoming_energy = weighted_energy[direction] / normal_velocity
                for x_index in x_indices:
                    particles = particles.at[x_index, y_index, direction].set(
                        incoming_particles
                    )
                    energy = energy.at[x_index, y_index, direction].set(incoming_energy)
            measure = jnp.asarray(
                self.face_measures[interface_index], dtype=particles.dtype
            )
            particle_integral = step_size * measure * weighted_particles
            energy_integral = step_size * measure * weighted_energy
            conservative_integral = step_size * measure * weighted_conservative
            recovered_integral = jnp.concatenate(
                (
                    jnp.sum(particle_integral)[None],
                    ein.contract("q,qd->d", particle_integral, velocity),
                    jnp.sum(energy_integral)[None],
                ),
                axis=0,
            )
            conservative_integrals.append(conservative_integral)
            moment_residuals.append(recovered_integral - conservative_integral)
        return (
            SmoothCompressibleKineticState(particles, energy),
            jnp.stack(tuple(conservative_integrals), axis=0),
            jnp.stack(tuple(moment_residuals), axis=0),
            jnp.zeros((4,), dtype=particles.dtype),
            outer_route.successful,
        )

    def _finite_volume_interface_residual(
        self,
        result: FiniteVolumeAdvanceResult,
        conservative_integrals: Array,
        /,
    ) -> Array:
        residuals = []
        discretization = self.finite_volume.dynamics.discretization
        for interface_index, (axis, face_index) in enumerate(
            zip(
                self.finite_volume_face_axes,
                self.finite_volume_face_indices,
                strict=True,
            )
        ):
            face_shape = tuple(discretization.face_layouts[axis].shape)
            flat_index = int(np.ravel_multi_index(face_index, face_shape))
            recorded = result.accepted_flux_integrals.blocks[axis].flux_integral[
                flat_index
            ]
            residuals.append(recorded - conservative_integrals[interface_index])
        return jnp.stack(tuple(residuals), axis=0)

    def advance(
        self,
        state: FixedPartitionHybridState,
        step_size: ArrayLike,
        finite_volume_args: Any = None,
        kinetic_args: Any = None,
        /,
    ) -> FixedPartitionHybridAdvanceResult:
        self._validate_state(state)
        dt = self._require_exact_step(state, step_size)
        initial_average = self.finite_volume._provide_stage_state(
            state.time,
            self.finite_volume._dynamics_cell_average(state.finite_volume.content_state),
        )
        stable = np.asarray(
            self.finite_volume.dynamics.stable_step(
                initial_average,
                finite_volume_args,
                cfl=self.finite_volume.policy.cfl,
            )
        )
        if stable.shape != () or not np.isfinite(stable) or float(stable) < float(dt):
            raise ValueError(
                "The fixed lattice step exceeds the FV stability bound; reduction is forbidden."
            )

        kinetic_conserved = self.spatial.method.moments(state.kinetic).conserved
        kinetic_dual, kinetic_support = self.learned_energy.predict_dual_with_evidence(
            kinetic_conserved
        )
        kinetic_collision = self.spatial.method.collide_with_energy_dual_with_evidence(
            state.kinetic,
            dt,
            kinetic_dual,
            self.spatial.energy_plan,
            kinetic_args,
        )
        provider = self._stage_provider(kinetic_collision.candidate_state)
        coupled_finite_volume = self.finite_volume.with_stage_flux_provider(provider)
        finite_volume = coupled_finite_volume.advance_prescribed(
            state.finite_volume,
            dt,
            finite_volume_args,
        )
        trace = finite_volume.attempted.stage_flux_trace
        if not isinstance(trace, FiniteVolumeStageFluxTrace):
            raise RuntimeError(
                "Coupled finite-volume advance did not retain stage fluxes."
            )

        (
            kinetic_candidate,
            conservative_integrals,
            moment_residual,
            kinetic_outer_boundary_exchange,
            kinetic_interface_route_accepted,
        ) = self._exchange_candidate(
            kinetic_collision.candidate_state,
            trace,
            dt,
        )
        finite_volume_candidate = finite_volume.attempted.runtime_state
        candidate = FixedPartitionHybridState(
            finite_volume_candidate,
            kinetic_candidate,
            checkpoint_eligible=False,
        )
        kinetic_realizability = self.spatial.method.realizability(
            kinetic_candidate,
            population_floor=self.spatial.population_floor,
        )
        stage_evidence = tuple(stage.evidence for stage in trace.stages)
        if any(
            not isinstance(value, FixedHybridStageEvidence) for value in stage_evidence
        ):
            raise TypeError("Coupled finite-volume trace contains foreign evidence.")
        learned_lifts_accepted = jnp.all(
            jnp.stack(tuple(value.successful for value in stage_evidence))
        ) & jnp.all(kinetic_support.successful)
        finite_volume_positive = (
            finite_volume.accepted
            & finite_volume.attempted.positivity.limited_state_valid
            & jnp.all(
                self.finite_volume.dynamics.system.admissible(
                    finite_volume_candidate.cell_average()
                )
            )
        )
        fv_interface_residual = self._finite_volume_interface_residual(
            finite_volume.attempted,
            conservative_integrals,
        )
        pre_fv = self._finite_volume_content(state.finite_volume)
        candidate_fv = self._finite_volume_content(finite_volume_candidate)
        pre_kinetic = self._kinetic_content(state.kinetic)
        candidate_kinetic = self._kinetic_content(kinetic_candidate)
        fv_change = candidate_fv - pre_fv
        kinetic_change = candidate_kinetic - pre_kinetic
        total_interface = jnp.sum(conservative_integrals, axis=0)
        kinetic_external = kinetic_outer_boundary_exchange
        _, _, finite_volume_expected_change = (
            finite_volume.attempted.accepted_flux_integrals.conservation_sums()
        )
        global_residual = (
            fv_change
            + kinetic_change
            - (finite_volume_expected_change + total_interface + kinetic_external)
        )
        residuals = jnp.concatenate(
            (
                moment_residual.reshape((-1,)),
                fv_interface_residual.reshape((-1,)),
                global_residual.reshape((-1,)),
                kinetic_collision.collision_evidence.conservation_residual.reshape((-1,)),
            ),
            axis=0,
        )
        maximum_residual = jnp.max(jnp.abs(residuals))
        scale = jnp.maximum(
            jnp.max(jnp.abs(jnp.concatenate((pre_fv, pre_kinetic), axis=0))),
            1.0,
        )
        tolerance = jnp.asarray(self.conservation_tolerance, dtype=maximum_residual.dtype)
        tolerance = tolerance + 1024.0 * jnp.finfo(maximum_residual.dtype).eps * scale
        conservation_accepted = maximum_residual <= tolerance
        exact_step = (
            finite_volume.accepted
            & (finite_volume.attempted.retries == 0)
            & (finite_volume.attempted.accepted_step_size == dt)
        )
        kinetic_route_accepted = (
            kinetic_collision.successful & kinetic_interface_route_accepted
        )
        accepted = (
            exact_step
            & finite_volume_positive
            & kinetic_route_accepted
            & learned_lifts_accepted
            & kinetic_realizability.realizable
            & conservation_accepted
        )
        status = jnp.asarray(int(FixedPartitionHybridStatus.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            ~finite_volume.accepted,
            int(FixedPartitionHybridStatus.FINITE_VOLUME_REJECTED),
            status,
        )
        status = jnp.where(
            finite_volume.accepted & ~kinetic_route_accepted,
            int(FixedPartitionHybridStatus.KINETIC_COLLISION_REJECTED),
            status,
        )
        status = jnp.where(
            finite_volume.accepted & kinetic_route_accepted & ~learned_lifts_accepted,
            int(FixedPartitionHybridStatus.LEARNED_LIFT_REJECTED),
            status,
        )
        status = jnp.where(
            finite_volume.accepted
            & kinetic_route_accepted
            & learned_lifts_accepted
            & ~finite_volume_positive,
            int(FixedPartitionHybridStatus.FINITE_VOLUME_POSITIVITY_REJECTED),
            status,
        )
        status = jnp.where(
            finite_volume.accepted
            & kinetic_route_accepted
            & learned_lifts_accepted
            & finite_volume_positive
            & ~kinetic_realizability.realizable,
            int(FixedPartitionHybridStatus.KINETIC_REALIZABILITY_REJECTED),
            status,
        )
        status = jnp.where(
            finite_volume.accepted
            & kinetic_route_accepted
            & learned_lifts_accepted
            & finite_volume_positive
            & kinetic_realizability.realizable
            & ~conservation_accepted,
            int(FixedPartitionHybridStatus.CONSERVATION_REJECTED),
            status,
        ).astype(jnp.int32)
        candidate = eqx.tree_at(
            lambda value: value.checkpoint_eligible,
            candidate,
            accepted,
        )
        committed = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old),
            candidate,
            state,
        )
        audit = FixedPartitionHybridAudit(
            pre_step_content=pre_fv + pre_kinetic,
            candidate_content=candidate_fv + candidate_kinetic,
            finite_volume_content_change=fv_change,
            kinetic_content_change=kinetic_change,
            interface_conservative_integrals=conservative_integrals,
            kinetic_outer_boundary_exchange=kinetic_outer_boundary_exchange,
            kinetic_moment_exchange_residual=moment_residual,
            finite_volume_interface_flux_residual=fv_interface_residual,
            global_conservation_residual=global_residual,
            maximum_absolute_residual=maximum_residual,
        )
        return FixedPartitionHybridAdvanceResult(
            previous=state,
            candidate=candidate,
            runtime_state=committed,
            finite_volume=finite_volume,
            kinetic_collision=kinetic_collision,
            stage_flux_trace=trace,
            evidence=FixedPartitionHybridAdvanceEvidence(
                accepted=accepted,
                rollback_applied=~accepted,
                exact_step=exact_step,
                finite_volume_accepted=finite_volume.accepted,
                kinetic_collision_accepted=kinetic_collision.successful,
                kinetic_interface_route_accepted=kinetic_interface_route_accepted,
                learned_lifts_accepted=learned_lifts_accepted,
                finite_volume_positive=finite_volume_positive,
                kinetic_realizability=kinetic_realizability,
                audit=audit,
                status=status,
                shock_owner=self.shock_owner,
                ownership_differentiability=self.ownership_differentiability,
            ),
        )

    def checkpoint(
        self,
        state: FixedPartitionHybridState,
        checkpoint_id: str,
        /,
    ) -> FixedPartitionHybridCheckpoint:
        self._validate_state(state)
        if not bool(np.asarray(state.checkpoint_eligible)):
            raise ValueError("Only jointly accepted hybrid states may be checkpointed.")
        identifier = str(checkpoint_id)
        if not identifier:
            raise ValueError("checkpoint_id must be non-empty.")
        payload_id = canonical_fingerprint(
            {
                "kind": "fixed-partition-hybrid-accepted-checkpoint",
                "checkpoint": identifier,
                "runtime": self.runtime_id,
                "learned_energy_artifact": self.learned_energy.prepared_id,
                "accepted_step": array_tree_fingerprint(state.accepted_step),
                "state": array_tree_fingerprint(state),
            }
        )
        return FixedPartitionHybridCheckpoint(
            state,
            identifier,
            self.runtime_id,
            self.learned_energy.prepared_id,
            payload_id,
        )

    def restore(
        self, checkpoint: FixedPartitionHybridCheckpoint, /
    ) -> FixedPartitionHybridState:
        if not isinstance(checkpoint, FixedPartitionHybridCheckpoint):
            raise TypeError("checkpoint must be FixedPartitionHybridCheckpoint.")
        if (
            checkpoint.runtime_id != self.runtime_id
            or checkpoint.learned_energy_artifact_id != self.learned_energy.prepared_id
        ):
            raise ValueError(
                "Hybrid checkpoint runtime or frozen learned-energy artifact is incompatible."
            )
        expected = self.checkpoint(
            checkpoint.state,
            checkpoint.checkpoint_id,
        )
        if checkpoint.payload_id != expected.payload_id:
            raise ValueError("Hybrid checkpoint payload identity is invalid.")
        return checkpoint.state


class DynamicHybridOwnershipState(StrictModule):
    finite_volume_owned: Array
    dwell_steps: Array
    last_change_step: Array
    transition_count: Array
    accepted_step: Array

    def __init__(
        self,
        finite_volume_owned: ArrayLike,
        dwell_steps: ArrayLike,
        last_change_step: ArrayLike,
        transition_count: ArrayLike,
        accepted_step: ArrayLike,
        /,
    ):
        owned = jnp.asarray(finite_volume_owned, dtype=jnp.bool_)
        dwell = jnp.asarray(dwell_steps, dtype=jnp.int32)
        last_change = jnp.asarray(last_change_step, dtype=jnp.int32)
        transitions = jnp.asarray(transition_count, dtype=jnp.int32)
        step = jnp.asarray(accepted_step, dtype=jnp.int32)
        if (
            owned.ndim != 2
            or dwell.shape != owned.shape
            or last_change.shape != owned.shape
            or transitions.shape != owned.shape
            or step.shape != ()
        ):
            raise ValueError("Dynamic ownership history shapes are inconsistent.")
        self.finite_volume_owned = owned
        self.dwell_steps = dwell
        self.last_change_step = last_change
        self.transition_count = transitions
        self.accepted_step = step


class DynamicHybridOwnershipDecision(StrictModule):
    finite_volume_owned: Array
    entered_finite_volume: Array
    exited_finite_volume: Array
    dilation_added: Array
    score: Array
    shock_mask: Array
    decision_step: Array
    plan_id: str = eqx.field(static=True)


class DynamicHybridCompositeState(StrictModule):
    finite_volume_conserved: Array
    kinetic: SmoothCompressibleKineticState
    ownership: DynamicHybridOwnershipState
    checkpoint_eligible: Array

    def __init__(
        self,
        finite_volume_conserved: ArrayLike,
        kinetic: SmoothCompressibleKineticState,
        ownership: DynamicHybridOwnershipState,
        /,
        *,
        checkpoint_eligible: ArrayLike = True,
    ):
        conserved = jnp.asarray(finite_volume_conserved)
        if not isinstance(kinetic, SmoothCompressibleKineticState):
            raise TypeError("kinetic must be SmoothCompressibleKineticState.")
        if not isinstance(ownership, DynamicHybridOwnershipState):
            raise TypeError("ownership must be DynamicHybridOwnershipState.")
        eligible = jnp.asarray(checkpoint_eligible, dtype=jnp.bool_)
        if eligible.shape != ():
            raise ValueError("checkpoint_eligible must be scalar.")
        if (
            conserved.shape != ownership.finite_volume_owned.shape + (4,)
            or kinetic.particle_populations.shape[:-1]
            != ownership.finite_volume_owned.shape
            or kinetic.total_energy_populations.shape
            != kinetic.particle_populations.shape
        ):
            raise ValueError("Dynamic FV, kinetic, and ownership fields must align.")
        self.finite_volume_conserved = conserved
        self.kinetic = kinetic
        self.ownership = ownership
        self.checkpoint_eligible = eligible


class DynamicHybridMigrationEvidence(StrictModule):
    accepted: Array
    rollback_applied: Array
    accepted_boundary: Array
    entered_finite_volume: Array
    exited_finite_volume: Array
    kinetic_to_finite_volume_residual: Array
    learned_support: EnergyEquilibriumSupportEvidence
    learned_lift: SmoothCompressibleLearnedEquilibriumEvidence
    kinetic_realizability: SmoothCompressibleRealizabilityEvidence
    ownership_differentiability: str = eqx.field(static=True)


class DynamicHybridMigrationResult(StrictModule):
    previous: DynamicHybridCompositeState
    candidate: DynamicHybridCompositeState
    runtime_state: DynamicHybridCompositeState
    decision: DynamicHybridOwnershipDecision
    evidence: DynamicHybridMigrationEvidence


class DynamicHybridCheckpoint(StrictModule, NonTrainableState):
    state: DynamicHybridCompositeState
    checkpoint_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    learned_energy_artifact_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class DynamicHybridOwnershipPlan(StrictModule, NonTrainableState):
    """Accepted-boundary ownership migration with non-differentiable hysteresis."""

    method: SmoothCompressibleD2VKineticMethod
    learned_energy: PreparedLearnedEnergyEquilibriumBinding
    spatial_shape: tuple[int, int] = eqx.field(static=True)
    enter_threshold: float = eqx.field(static=True)
    exit_threshold: float = eqx.field(static=True)
    minimum_dwell_steps: int = eqx.field(static=True)
    finite_volume_stencil_radius: tuple[int, int] = eqx.field(static=True)
    kinetic_reach: tuple[int, int] = eqx.field(static=True)
    dilation_radius: tuple[int, int] = eqx.field(static=True)
    dilation_shifts: tuple[tuple[int, int], ...] = eqx.field(static=True)
    population_floor: float = eqx.field(static=True)
    shock_owner: str = eqx.field(static=True)
    ownership_differentiability: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: SmoothCompressibleD2VKineticMethod,
        learned_energy: PreparedLearnedEnergyEquilibriumBinding,
        spatial_shape: tuple[int, int],
        /,
        *,
        enter_threshold: float,
        exit_threshold: float,
        minimum_dwell_steps: int,
        finite_volume_stencil_radius: tuple[int, int],
        kinetic_reach: tuple[int, int],
        population_floor: float = 0.0,
    ):
        if not isinstance(method, SmoothCompressibleD2VKineticMethod):
            raise TypeError("method must be SmoothCompressibleD2VKineticMethod.")
        if not isinstance(learned_energy, PreparedLearnedEnergyEquilibriumBinding):
            raise TypeError(
                "learned_energy must be PreparedLearnedEnergyEquilibriumBinding."
            )
        shape = tuple(spatial_shape)
        stencil = tuple(finite_volume_stencil_radius)
        reach = tuple(kinetic_reach)
        enter = float(enter_threshold)
        exit_ = float(exit_threshold)
        dwell = int(minimum_dwell_steps)
        floor = float(population_floor)
        if (
            len(shape) != 2
            or any(value <= 0 for value in shape)
            or len(stencil) != 2
            or len(reach) != 2
            or any(value < 0 for value in (*stencil, *reach))
            or not isfinite(enter)
            or not isfinite(exit_)
            or enter <= exit_
            or dwell < 0
            or not isfinite(floor)
            or floor < 0.0
        ):
            raise ValueError("Dynamic ownership thresholds, dwell, or radii are invalid.")
        if (
            learned_energy.plan.equilibrium_plan.quadrature.quadrature_id
            != method.quadrature.quadrature_id
            or learned_energy.plan.material.material_id != method.material.material_id
        ):
            raise ValueError("Dynamic ownership method and learned lift must match.")
        radius = tuple(stencil[axis] + reach[axis] for axis in range(2))
        shifts = tuple(
            (first, second)
            for first, second in product(
                range(-radius[0], radius[0] + 1),
                range(-radius[1], radius[1] + 1),
            )
        )
        self.method = method
        self.learned_energy = learned_energy
        self.spatial_shape = shape
        self.enter_threshold = enter
        self.exit_threshold = exit_
        self.minimum_dwell_steps = dwell
        self.finite_volume_stencil_radius = stencil
        self.kinetic_reach = reach
        self.dilation_radius = radius
        self.dilation_shifts = shifts
        self.population_floor = floor
        self.shock_owner = "finite_volume"
        self.ownership_differentiability = "none"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dynamic-fv-kinetic-ownership-plan",
                "method": method.method_id,
                "learned_energy_artifact": learned_energy.prepared_id,
                "spatial_shape": shape,
                "enter_threshold": enter,
                "exit_threshold": exit_,
                "minimum_dwell_steps": dwell,
                "finite_volume_stencil_radius": stencil,
                "kinetic_reach": reach,
                "dilation_radius": radius,
                "population_floor": floor,
                "shock_owner": self.shock_owner,
                "ownership_differentiability": self.ownership_differentiability,
            }
        )

    def initialize(
        self,
        finite_volume_owned: ArrayLike,
        /,
        *,
        accepted_step: ArrayLike = 0,
    ) -> DynamicHybridOwnershipState:
        owned = jnp.asarray(finite_volume_owned, dtype=jnp.bool_)
        if owned.shape != self.spatial_shape:
            raise ValueError("Initial ownership must match spatial_shape.")
        step = jnp.asarray(accepted_step, dtype=jnp.int32).reshape(())
        dwell = jnp.full(
            self.spatial_shape,
            self.minimum_dwell_steps,
            dtype=jnp.int32,
        )
        return DynamicHybridOwnershipState(
            owned,
            dwell,
            jnp.full(self.spatial_shape, -1, dtype=jnp.int32),
            jnp.zeros(self.spatial_shape, dtype=jnp.int32),
            step,
        )

    def _dilate(self, mask: Array, /) -> Array:
        dilated = jnp.zeros_like(mask)
        for shift in self.dilation_shifts:
            dilated = dilated | jnp.roll(mask, shift=shift, axis=(0, 1))
        return dilated

    def propose(
        self,
        state: DynamicHybridOwnershipState,
        score: ArrayLike,
        shock_mask: ArrayLike,
        /,
    ) -> DynamicHybridOwnershipDecision:
        if not isinstance(state, DynamicHybridOwnershipState):
            raise TypeError("state must be DynamicHybridOwnershipState.")
        if state.finite_volume_owned.shape != self.spatial_shape:
            raise ValueError("Ownership state does not match this plan.")
        scores = jax.lax.stop_gradient(jnp.asarray(score))
        shocks = jax.lax.stop_gradient(jnp.asarray(shock_mask, dtype=jnp.bool_))
        if scores.shape != self.spatial_shape or shocks.shape != self.spatial_shape:
            raise ValueError("Ownership score and shock mask must match spatial_shape.")
        eligible = state.dwell_steps >= self.minimum_dwell_steps
        enter = (~state.finite_volume_owned) & eligible & (scores >= self.enter_threshold)
        exit_ = (
            state.finite_volume_owned
            & eligible
            & (scores <= self.exit_threshold)
            & ~shocks
        )
        raw = (state.finite_volume_owned | enter) & ~exit_
        raw = raw | shocks
        owned = self._dilate(raw)
        entered = owned & ~state.finite_volume_owned
        exited = ~owned & state.finite_volume_owned
        return DynamicHybridOwnershipDecision(
            finite_volume_owned=owned,
            entered_finite_volume=entered,
            exited_finite_volume=exited,
            dilation_added=owned & ~raw,
            score=scores,
            shock_mask=shocks,
            decision_step=state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            plan_id=self.plan_id,
        )

    def migrate(
        self,
        state: DynamicHybridCompositeState,
        decision: DynamicHybridOwnershipDecision,
        accepted_boundary: ArrayLike,
        /,
    ) -> DynamicHybridMigrationResult:
        if not isinstance(state, DynamicHybridCompositeState):
            raise TypeError("state must be DynamicHybridCompositeState.")
        if not isinstance(decision, DynamicHybridOwnershipDecision):
            raise TypeError("decision must be DynamicHybridOwnershipDecision.")
        if decision.plan_id != self.plan_id:
            raise ValueError("Dynamic ownership decision belongs to another plan.")
        if state.ownership.finite_volume_owned.shape != self.spatial_shape:
            raise ValueError("Dynamic composite state does not match this plan.")
        if decision.finite_volume_owned.shape != self.spatial_shape:
            raise ValueError("Dynamic ownership decision does not match this plan.")
        boundary = jnp.asarray(accepted_boundary, dtype=jnp.bool_).reshape(())
        decision_aligned = decision.decision_step == (
            state.ownership.accepted_step + jnp.asarray(1, dtype=jnp.int32)
        )
        entered = decision.finite_volume_owned & ~state.ownership.finite_volume_owned
        exited = ~decision.finite_volume_owned & state.ownership.finite_volume_owned

        kinetic_moments = self.method.moments(state.kinetic).conserved
        candidate_fv = jnp.where(
            entered[..., None],
            kinetic_moments,
            state.finite_volume_conserved,
        )
        dual, support = self.learned_energy.predict_dual_with_evidence(
            state.finite_volume_conserved
        )
        lifted, lift_evidence = self.method.equilibrium_from_energy_dual_with_evidence(
            state.finite_volume_conserved,
            dual,
            self.learned_energy.plan.equilibrium_plan,
        )
        candidate_kinetic = SmoothCompressibleKineticState(
            jnp.where(
                exited[..., None],
                lifted.particle_populations,
                state.kinetic.particle_populations,
            ),
            jnp.where(
                exited[..., None],
                lifted.total_energy_populations,
                state.kinetic.total_energy_populations,
            ),
        )
        realizability = self.method.realizability(
            candidate_kinetic,
            population_floor=self.population_floor,
        )
        transfer_residual = jnp.where(
            entered[..., None],
            candidate_fv - kinetic_moments,
            jnp.zeros_like(candidate_fv),
        )
        lift_supported = jnp.all((~exited) | support.successful)
        lift_successful = jnp.all((~exited) | lift_evidence.successful)
        exited_realizable = jnp.all((~exited) | realizability.local_realizable)
        exact_kinetic_to_fv = jnp.all(transfer_residual == 0.0)
        successful = (
            boundary
            & decision_aligned
            & lift_supported
            & lift_successful
            & exited_realizable
            & exact_kinetic_to_fv
            & jnp.all(decision.finite_volume_owned | ~decision.shock_mask)
        )
        changed = entered | exited
        next_ownership = DynamicHybridOwnershipState(
            decision.finite_volume_owned,
            jnp.where(
                changed,
                jnp.zeros_like(state.ownership.dwell_steps),
                state.ownership.dwell_steps + jnp.asarray(1, dtype=jnp.int32),
            ),
            jnp.where(
                changed,
                decision.decision_step,
                state.ownership.last_change_step,
            ),
            state.ownership.transition_count + changed.astype(jnp.int32),
            decision.decision_step,
        )
        candidate = DynamicHybridCompositeState(
            candidate_fv,
            candidate_kinetic,
            next_ownership,
            checkpoint_eligible=successful,
        )
        committed = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old),
            candidate,
            state,
        )
        return DynamicHybridMigrationResult(
            previous=state,
            candidate=candidate,
            runtime_state=committed,
            decision=decision,
            evidence=DynamicHybridMigrationEvidence(
                accepted=successful,
                rollback_applied=~successful,
                accepted_boundary=boundary,
                entered_finite_volume=entered,
                exited_finite_volume=exited,
                kinetic_to_finite_volume_residual=transfer_residual,
                learned_support=support,
                learned_lift=lift_evidence,
                kinetic_realizability=realizability,
                ownership_differentiability=self.ownership_differentiability,
            ),
        )

    def checkpoint(
        self,
        state: DynamicHybridCompositeState,
        checkpoint_id: str,
        /,
    ) -> DynamicHybridCheckpoint:
        if not isinstance(state, DynamicHybridCompositeState):
            raise TypeError("state must be DynamicHybridCompositeState.")
        if not bool(np.asarray(state.checkpoint_eligible)):
            raise ValueError(
                "Only accepted dynamic ownership states may be checkpointed."
            )
        identifier = str(checkpoint_id)
        if not identifier:
            raise ValueError("checkpoint_id must be non-empty.")
        payload_id = canonical_fingerprint(
            {
                "kind": "dynamic-hybrid-accepted-checkpoint",
                "checkpoint": identifier,
                "plan": self.plan_id,
                "learned_energy_artifact": self.learned_energy.prepared_id,
                "state": array_tree_fingerprint(state),
                "ownership_history": array_tree_fingerprint(state.ownership),
            }
        )
        return DynamicHybridCheckpoint(
            state,
            identifier,
            self.plan_id,
            self.learned_energy.prepared_id,
            payload_id,
        )

    def restore(
        self, checkpoint: DynamicHybridCheckpoint, /
    ) -> DynamicHybridCompositeState:
        if not isinstance(checkpoint, DynamicHybridCheckpoint):
            raise TypeError("checkpoint must be DynamicHybridCheckpoint.")
        if (
            checkpoint.plan_id != self.plan_id
            or checkpoint.learned_energy_artifact_id != self.learned_energy.prepared_id
        ):
            raise ValueError(
                "Dynamic checkpoint ownership plan or frozen learned artifact is incompatible."
            )
        expected = self.checkpoint(checkpoint.state, checkpoint.checkpoint_id)
        if checkpoint.payload_id != expected.payload_id:
            raise ValueError("Dynamic checkpoint payload identity is invalid.")
        return checkpoint.state


__all__ = [
    "DynamicHybridCheckpoint",
    "DynamicHybridCompositeState",
    "DynamicHybridMigrationEvidence",
    "DynamicHybridMigrationResult",
    "DynamicHybridOwnershipDecision",
    "DynamicHybridOwnershipPlan",
    "DynamicHybridOwnershipState",
    "FixedHybridStageEvidence",
    "FixedPartitionHybridAdvanceEvidence",
    "FixedPartitionHybridAdvanceResult",
    "FixedPartitionHybridAudit",
    "FixedPartitionHybridCheckpoint",
    "FixedPartitionHybridState",
    "FixedPartitionHybridStatus",
    "PreparedFixedPartitionHybridRuntime",
]

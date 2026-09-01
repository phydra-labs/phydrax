#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._boundary import FreeSurfaceBoundaryPlan
from ._capillarity import GraphCapillarityPlan
from ._free_surface_ale import (
    _tuple_add,
    FaceTuple,
    FreeSurfaceALEState,
    FreeSurfaceALEStateView,
    GraphSurfaceALEPlan,
    PreparedGraphSurfaceALE,
)
from ._projection import FreeSurfaceProjectionResult, MappedFreeSurfaceProjectionPlan
from ._waves import ActiveAbsorptionState, WaveForcingPlan


def _cell_net_flux(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    difference = (
        jnp.roll(moved, -1, axis=0) - moved if periodic else moved[1:] - moved[:-1]
    )
    return jnp.moveaxis(difference, 0, axis)


def _face_upwind(value: Array, flux: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    moved_flux = jnp.moveaxis(flux, axis, 0)
    if periodic:
        lower = jnp.roll(moved, 1, axis=0)
        upper = moved
    else:
        lower = jnp.concatenate((moved[:1], moved), axis=0)
        upper = jnp.concatenate((moved, moved[-1:]), axis=0)
    return jnp.moveaxis(jnp.where(moved_flux >= 0.0, lower, upper), 0, axis)


class FreeSurfaceALELedger(StrictModule):
    volume_change: Array
    scalar_change: dict[str, Array]
    kinetic_energy_change: Array
    gravitational_energy_change: Array
    surface_energy_change: Array
    pressure_work: Array
    capillary_work: Array
    gas_pressure_work: Array
    wave_work: Array
    relaxation_work: Array
    sponge_dissipation: Array
    body_work: Array
    body_energy_change: Array
    remap_energy_defect: Array
    shoreline_work: Array
    controller_work: Array
    gcl_residual: Array
    divergence_residual: Array
    kinematic_residual: Array
    dynamic_residual: Array
    capillary_dual_residual: Array
    nonlinear_stage_residual: Array
    total_energy_residual: Array

    @classmethod
    def zeros(cls, scalar_names: tuple[str, ...], dtype, /) -> "FreeSurfaceALELedger":
        zero = jnp.zeros((), dtype=dtype)
        return cls(
            volume_change=zero,
            scalar_change={name: zero for name in scalar_names},
            kinetic_energy_change=zero,
            gravitational_energy_change=zero,
            surface_energy_change=zero,
            pressure_work=zero,
            capillary_work=zero,
            gas_pressure_work=zero,
            wave_work=zero,
            relaxation_work=zero,
            sponge_dissipation=zero,
            body_work=zero,
            body_energy_change=zero,
            remap_energy_defect=zero,
            shoreline_work=zero,
            controller_work=zero,
            gcl_residual=zero,
            divergence_residual=zero,
            kinematic_residual=zero,
            dynamic_residual=zero,
            capillary_dual_residual=zero,
            nonlinear_stage_residual=zero,
            total_energy_residual=zero,
        )


class FreeSurfaceALEContinuationState(StrictModule):
    state: FreeSurfaceALEState
    eta_rate: Array
    pressure_head: Array
    ledger: FreeSurfaceALELedger
    mesh_epoch: Array
    wave_controller: ActiveAbsorptionState | None

    @classmethod
    def initialize(
        cls,
        state: FreeSurfaceALEState,
        /,
    ) -> "FreeSurfaceALEContinuationState":
        zero_eta = jnp.zeros_like(state.eta)
        pressure_shape = (
            next(iter(state.scalar_content.values())).shape
            if state.scalar_content
            else None
        )
        if pressure_shape is None:
            raise ValueError(
                "Free-surface state requires at least one extensive scalar content field."
            )
        pressure = jnp.zeros(pressure_shape, dtype=state.eta.dtype)
        return cls(
            state,
            zero_eta,
            pressure,
            FreeSurfaceALELedger.zeros(
                tuple(sorted(state.scalar_content)), state.eta.dtype
            ),
            jnp.asarray(0, dtype=jnp.int32),
            None,
        )


class FreeSurfaceALEStageEvidence(StrictModule):
    geometry_valid: Array
    projection_successful: Array
    kinematic_successful: Array
    volume_residual: Array
    capillary_dual_residual: Array
    capillary_work_rate: Array
    surface_energy: Array
    wave_work_rate: Array
    relaxation_work_rate: Array
    sponge_dissipation_rate: Array
    controller_work_rate: Array
    gcl_residual: Array
    divergence_residual: Array
    kinematic_residual: Array
    dynamic_residual: Array
    pressure_work_rate: Array
    nonlinear_residual: Array
    finite: Array
    successful: Array


class OnePhaseFreeSurfaceALEPlan(StrictModule, NonTrainableState):
    """Compile fixed-topology inviscid one-phase graph-surface ALE hydrodynamics."""

    surface_plan: GraphSurfaceALEPlan
    boundary: FreeSurfaceBoundaryPlan
    density: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    surface_tension: float = eqx.field(static=True)
    wave: WaveForcingPlan | None
    coupling_iterations: int = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_plan: GraphSurfaceALEPlan,
        /,
        *,
        boundary: FreeSurfaceBoundaryPlan | None = None,
        density: float = 1000.0,
        gravity: float = 9.81,
        surface_tension: float = 0.0,
        wave: WaveForcingPlan | None = None,
        coupling_iterations: int = 6,
        coupling_tolerance: float = 1.0e-8,
    ):
        if not isinstance(surface_plan, GraphSurfaceALEPlan):
            raise TypeError("surface_plan must be GraphSurfaceALEPlan.")
        boundary_ = FreeSurfaceBoundaryPlan() if boundary is None else boundary
        density_ = float(density)
        gravity_ = float(gravity)
        surface_tension_ = float(surface_tension)
        iterations = int(coupling_iterations)
        tolerance = float(coupling_tolerance)
        if (
            density_ <= 0.0
            or gravity_ <= 0.0
            or surface_tension_ < 0.0
            or iterations <= 0
            or tolerance <= 0.0
            or any(
                not np.isfinite(v)
                for v in (
                    density_,
                    gravity_,
                    surface_tension_,
                    tolerance,
                )
            )
        ):
            raise ValueError(
                "Invalid free-surface density, gravity, tension, or coupling policy."
            )
        if wave is not None and not isinstance(wave, WaveForcingPlan):
            raise TypeError("wave must be WaveForcingPlan or None.")
        self.surface_plan = surface_plan
        self.boundary = boundary_
        self.density = density_
        self.gravity = gravity_
        self.coupling_iterations = iterations
        self.surface_tension = surface_tension_
        self.wave = wave
        self.coupling_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "one-phase-free-surface-ale-plan",
                "surface": surface_plan.plan_id,
                "boundary": boundary_.layout_id,
                "density": density_,
                "gravity": gravity_,
                "coupling_iterations": iterations,
                "surface_tension": surface_tension_,
                "wave": "none" if wave is None else wave.plan_id,
                "coupling_tolerance": tolerance,
            }
        )

    def prepare(self) -> "PreparedOnePhaseFreeSurfaceALE":
        surface = self.surface_plan.prepare()
        projection = MappedFreeSurfaceProjectionPlan(
            surface,
            tolerance=self.coupling_tolerance,
            maximum_iterations=self.surface_plan.maximum_iterations,
        )
        capillarity = GraphCapillarityPlan(
            surface,
            self.surface_tension,
            tolerance=self.coupling_tolerance,
            maximum_iterations=self.surface_plan.maximum_iterations,
        )
        return PreparedOnePhaseFreeSurfaceALE(
            self, surface, projection, capillarity, self.wave
        )


class PreparedOnePhaseFreeSurfaceALE(StrictModule):
    plan: OnePhaseFreeSurfaceALEPlan
    surface: PreparedGraphSurfaceALE
    projection: MappedFreeSurfaceProjectionPlan
    capillarity: GraphCapillarityPlan
    wave: WaveForcingPlan | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: OnePhaseFreeSurfaceALEPlan,
        surface: PreparedGraphSurfaceALE,
        projection: MappedFreeSurfaceProjectionPlan,
        capillarity: GraphCapillarityPlan,
        wave: WaveForcingPlan | None,
        /,
    ):
        self.plan = plan
        self.surface = surface
        self.projection = projection
        self.capillarity = capillarity
        self.wave = wave
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-one-phase-free-surface-ale",
                "plan": plan.plan_id,
                "surface": surface.surface_id,
                "projection": projection.plan_id,
                "capillarity": capillarity.plan_id,
                "wave": "none" if wave is None else wave.plan_id,
            }
        )

    @property
    def reference(self):
        return self.surface.plan.reference

    def initial_state(
        self,
        eta: ArrayLike,
        /,
        *,
        velocity: FaceTuple | None = None,
        scalars: dict[str, ArrayLike] | None = None,
    ) -> FreeSurfaceALEState:
        eta_ = jnp.asarray(eta, dtype=self.reference.cell_volumes.dtype)
        if eta_.shape != self.surface.eta_shape:
            raise ValueError("Initial free-surface eta shape is invalid.")
        zero_rate = jnp.zeros_like(eta_)
        geometry = self.surface.geometry(jnp.asarray(0.0), eta_, zero_rate)
        velocity_ = (
            tuple(jnp.zeros_like(value) for value in geometry.face_measures)
            if velocity is None
            else geometry.validate_velocity(velocity)
        )
        momentum = self.surface.apply_hodge(geometry, velocity_)
        supplied = {"volume_marker": 1.0} if scalars is None else dict(scalars)
        content = {}
        for name, value in supplied.items():
            concentration = jnp.asarray(value, dtype=eta_.dtype)
            if concentration.shape == ():
                concentration = jnp.broadcast_to(
                    concentration, geometry.cell_volumes.shape
                )
            if concentration.shape != geometry.cell_volumes.shape:
                raise ValueError(f"Scalar {name!r} shape is invalid.")
            content[name] = geometry.cell_volumes * concentration
        return FreeSurfaceALEState(eta_, momentum, content)

    def view(
        self,
        state: FreeSurfaceALEState,
        eta_rate: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        /,
    ) -> FreeSurfaceALEStateView:
        rate = jnp.zeros_like(state.eta) if eta_rate is None else jnp.asarray(eta_rate)
        geometry = self.surface.geometry(time, state.eta, rate)
        hodge = self.surface.inverse_hodge(geometry, state.momentum)
        scalars = {
            name: jnp.where(
                geometry.cell_volumes > 0.0,
                content / geometry.cell_volumes,
                0.0,
            )
            for name, content in state.scalar_content.items()
        }
        return FreeSurfaceALEStateView(
            eta=state.eta,
            velocity=hodge.velocity,
            scalars=scalars,
            geometry=geometry,
            kinetic_energy=self.plan.density
            * self.surface.kinetic_energy(geometry, hodge.velocity),
            volume=jnp.sum(geometry.cell_volumes),
            view_id=self.prepared_id,
        )

    def _scalar_rate(
        self,
        geometry,
        velocity: FaceTuple,
        scalar_content: dict[str, Array],
        /,
    ) -> dict[str, Array]:
        relative_flux = self.surface.ale.relative_flux(geometry, velocity)
        periodic = tuple(axis.periodic for axis in self.reference.grid.structured_axes)
        rates = {}
        for name, content in scalar_content.items():
            concentration = jnp.where(
                geometry.cell_volumes > 0.0,
                content / geometry.cell_volumes,
                0.0,
            )
            net = jnp.zeros_like(content)
            for axis, flux in enumerate(relative_flux):
                face_value = _face_upwind(concentration, flux, axis, periodic[axis])
                net = net + _cell_net_flux(flux * face_value, axis, periodic[axis])
            rates[name] = -net
        return rates


class OnePhaseFreeSurfaceALEMethod(AbstractFixedStepMethod):
    """Strongly coupled explicit-midpoint graph-surface ALE step."""

    hydrodynamics: PreparedOnePhaseFreeSurfaceALE
    method_id: str = eqx.field(static=True)

    def __init__(self, hydrodynamics: PreparedOnePhaseFreeSurfaceALE, /):
        if not isinstance(hydrodynamics, PreparedOnePhaseFreeSurfaceALE):
            raise TypeError("hydrodynamics must be PreparedOnePhaseFreeSurfaceALE.")
        self.hydrodynamics = hydrodynamics
        self.method_id = canonical_fingerprint(
            {
                "kind": "one-phase-free-surface-ale-midpoint",
                "hydrodynamics": hydrodynamics.prepared_id,
                "coupling_iterations": hydrodynamics.plan.coupling_iterations,
            }
        )

    def _coupled_advance(
        self,
        base: FreeSurfaceALEState,
        evaluation: FreeSurfaceALEState,
        eta_rate_guess: Array,
        pressure_guess: Array,
        base_time: Array,
        evaluation_time: Array,
        target_time: Array,
        dt: Array,
        wave_controller: ActiveAbsorptionState | None,
        args: Any,
        /,
    ) -> tuple[
        FreeSurfaceALEState,
        Array,
        Array,
        FreeSurfaceProjectionResult,
        FreeSurfaceALEStageEvidence,
    ]:
        hydro = self.hydrodynamics
        eta_rate = eta_rate_guess
        pressure = pressure_guess
        projection = None
        geometry = None
        stage_residual = jnp.asarray(jnp.inf, dtype=base.eta.dtype)
        wave_result = None

        for iteration in range(hydro.plan.coupling_iterations):
            geometry = hydro.surface.geometry(
                evaluation_time, evaluation.eta, eta_rate, args
            )
            evaluation_velocity = hydro.surface.inverse_hodge(
                geometry, evaluation.momentum
            )
            momentum_rate_velocity = hydro.surface.ale.momentum_rate(
                geometry,
                evaluation_velocity.velocity,
                viscosity=0.0,
                forcing=None,
            )
            wave_result = (
                None
                if hydro.wave is None
                else hydro.wave.evaluate(
                    hydro.surface,
                    geometry,
                    evaluation_time,
                    evaluation_velocity.velocity,
                    evaluation.eta,
                    wave_controller,
                )
            )
            base_geometry = hydro.surface.geometry(base_time, base.eta, eta_rate, args)
            base_velocity = hydro.surface.inverse_hodge(base_geometry, base.momentum)
            candidate_velocity = tuple(
                value + dt * rate
                for value, rate in zip(
                    base_velocity.velocity,
                    momentum_rate_velocity,
                    strict=True,
                )
            )
            eta_candidate = base.eta + dt * eta_rate
            end_geometry = hydro.surface.geometry(
                target_time, eta_candidate, eta_rate, args
            )
            capillary = hydro.capillarity.evaluate(eta_candidate, hydro.plan.density)
            boundary_stage = hydro.plan.boundary.stage(
                hydro.surface,
                end_geometry,
                eta_candidate,
                gravity=hydro.plan.gravity,
                density=hydro.plan.density,
                capillary_head=capillary.pressure_head,
                wave_pressure_head=(
                    None if wave_result is None else wave_result.surface_pressure_head
                ),
                prescribed_velocity=(
                    None if wave_result is None else wave_result.prescribed_velocity
                ),
                stage_tag=(target_time, iteration),
            )
            tentative_momentum = hydro.surface.apply_hodge(
                end_geometry, candidate_velocity
            )
            if wave_result is not None:
                tentative_momentum = _tuple_add(
                    tentative_momentum,
                    dt,
                    wave_result.momentum_rate,
                )
            surface_force = hydro.projection.surface_pressure_force(
                end_geometry,
                boundary_stage.surface_pressure_head,
            )
            tentative_momentum = _tuple_add(tentative_momentum, dt, surface_force)
            projection = hydro.projection.project(
                end_geometry,
                tentative_momentum,
                boundary_stage,
                dt,
                pressure,
            )
            target_flux = hydro.surface.top_volume_flux(end_geometry, projection.velocity)
            wave_eta_source = (
                jnp.zeros_like(eta_candidate)
                if wave_result is None
                else wave_result.eta_rate_source
            )
            wave_volume_source = jax.jvp(
                hydro.surface._column_volumes,
                (eta_candidate,),
                (wave_eta_source,),
            )[1]
            kinematic = hydro.surface.solve_eta_rate(
                eta_candidate, target_flux + wave_volume_source
            )
            next_rate = kinematic.eta_rate
            stage_residual = jnp.max(jnp.abs(next_rate - eta_rate))
            eta_rate = next_rate
            pressure = projection.pressure_head
            geometry = end_geometry

        eta_new = base.eta + dt * eta_rate
        final_geometry = hydro.surface.geometry(target_time, eta_new, eta_rate, args)
        final_capillary = hydro.capillarity.evaluate(eta_new, hydro.plan.density)
        final_boundary = hydro.plan.boundary.stage(
            hydro.surface,
            final_geometry,
            eta_new,
            gravity=hydro.plan.gravity,
            density=hydro.plan.density,
            capillary_head=final_capillary.pressure_head,
            wave_pressure_head=(
                None if wave_result is None else wave_result.surface_pressure_head
            ),
            prescribed_velocity=(
                None if wave_result is None else wave_result.prescribed_velocity
            ),
            stage_tag=(target_time, "final"),
        )
        final_projection = hydro.projection.project(
            final_geometry,
            projection.momentum,
            final_boundary,
            dt,
            pressure,
        )
        scalar_rate = hydro._scalar_rate(
            geometry,
            projection.velocity,
            evaluation.scalar_content,
        )
        scalar_content = {
            name: base.scalar_content[name] + dt * scalar_rate[name]
            for name in base.scalar_content
        }
        candidate = FreeSurfaceALEState(
            eta_new,
            final_projection.momentum,
            scalar_content,
        )
        target_flux = hydro.surface.top_volume_flux(
            final_geometry, final_projection.velocity
        )
        final_wave_source = (
            jnp.zeros_like(eta_new)
            if wave_result is None
            else wave_result.eta_rate_source
        )
        final_wave_volume = jax.jvp(
            hydro.surface._column_volumes,
            (eta_new,),
            (final_wave_source,),
        )[1]
        kinematic = hydro.surface.solve_eta_rate(eta_new, target_flux + final_wave_volume)
        geometry_evidence = hydro.surface.geometry_evidence(
            eta_new, eta_rate, target_time
        )
        volume_before = jnp.sum(base_geometry.cell_volumes)
        volume_after = jnp.sum(final_geometry.cell_volumes)
        top_flux = jnp.sum(target_flux)
        volume_residual = volume_after - volume_before - dt * top_flux
        divergence_residual = jnp.sqrt(
            jnp.sum(final_geometry.cell_volumes * final_projection.divergence_after**2)
        )
        applied_force = hydro.projection.surface_pressure_force(
            final_geometry, final_boundary.surface_pressure_head
        )
        top_area = jnp.take(final_geometry.face_measures[2], -1, axis=2)
        applied_head = -jnp.take(applied_force[2], -1, axis=2) / top_area
        dynamic_residual = jnp.max(
            jnp.abs(applied_head - final_boundary.surface_pressure_head)
        )
        base_capillary = hydro.capillarity.evaluate(base.eta, hydro.plan.density)
        base_wave_result = (
            None
            if hydro.wave is None
            else hydro.wave.evaluate(
                hydro.surface,
                base_geometry,
                base_time,
                base_velocity.velocity,
                base.eta,
                wave_controller,
            )
        )
        base_boundary = hydro.plan.boundary.stage(
            hydro.surface,
            base_geometry,
            base.eta,
            gravity=hydro.plan.gravity,
            density=hydro.plan.density,
            capillary_head=base_capillary.pressure_head,
            wave_pressure_head=(
                None
                if base_wave_result is None
                else base_wave_result.surface_pressure_head
            ),
            prescribed_velocity=(
                None if base_wave_result is None else base_wave_result.prescribed_velocity
            ),
            stage_tag=(base_time, "base"),
        )
        base_top_flux = hydro.surface.top_volume_flux(
            base_geometry, base_velocity.velocity
        )
        total_pressure_work_rate = (
            -0.5
            * hydro.plan.density
            * (
                jnp.sum(base_boundary.surface_pressure_head * base_top_flux)
                + jnp.sum(final_boundary.surface_pressure_head * target_flux)
            )
        )
        capillary_work_rate = (
            -0.5
            * hydro.plan.density
            * (
                jnp.sum(base_capillary.pressure_head * base_top_flux)
                + jnp.sum(final_capillary.pressure_head * target_flux)
            )
        )
        pressure_work_rate = total_pressure_work_rate - capillary_work_rate
        wave_valid = jnp.asarray(True) if wave_result is None else wave_result.valid
        wave_finite = jnp.asarray(True) if wave_result is None else wave_result.finite
        finite = (
            geometry_evidence.finite
            & final_boundary.finite
            & final_projection.finite
            & kinematic.finite
            & final_capillary.finite
            & base_capillary.finite
            & wave_finite
            & jnp.isfinite(pressure_work_rate)
            & jnp.all(jnp.isfinite(eta_new))
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value)) for value in scalar_content.values()
                    )
                )
            )
        )
        kinematic_success = kinematic.converged | (
            (wave_result is not None)
            & (kinematic.residual_norm <= 4.0 * hydro.plan.coupling_tolerance)
        )
        successful = (
            finite
            & geometry_evidence.valid
            & final_boundary.valid
            & final_projection.successful
            & kinematic_success
            & final_capillary.successful
            & wave_valid
            & (dt <= final_capillary.timestep_limit)
            & (
                stage_residual
                <= hydro.plan.coupling_tolerance
                * (4.0 if hydro.wave is not None else 1.0)
            )
        )
        zero = jnp.asarray(0.0, dtype=eta_new.dtype)
        evidence = FreeSurfaceALEStageEvidence(
            geometry_valid=geometry_evidence.valid,
            projection_successful=final_projection.successful,
            kinematic_successful=kinematic_success,
            volume_residual=volume_residual,
            capillary_dual_residual=final_capillary.dual_residual_norm,
            capillary_work_rate=capillary_work_rate,
            surface_energy=final_capillary.surface_energy,
            wave_work_rate=(zero if wave_result is None else wave_result.wave_work_rate),
            relaxation_work_rate=(
                zero if wave_result is None else wave_result.relaxation_work_rate
            ),
            sponge_dissipation_rate=(
                zero if wave_result is None else wave_result.sponge_dissipation_rate
            ),
            controller_work_rate=(
                zero if wave_result is None else wave_result.controller_work_rate
            ),
            gcl_residual=geometry_evidence.volume_gcl_residual,
            divergence_residual=divergence_residual,
            kinematic_residual=kinematic.residual_norm,
            dynamic_residual=dynamic_residual,
            pressure_work_rate=pressure_work_rate,
            nonlinear_residual=stage_residual,
            finite=finite,
            successful=successful,
        )
        return candidate, eta_rate, pressure, final_projection, evidence

    @staticmethod
    def _energy(
        hydro: PreparedOnePhaseFreeSurfaceALE,
        state: FreeSurfaceALEState,
        eta_rate: Array,
        time: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        view = hydro.view(state, eta_rate, time)
        gravitational = (
            0.5
            * hydro.plan.density
            * hydro.plan.gravity
            * jnp.sum(hydro.surface.horizontal_area * state.eta**2)
        )
        surface_energy = hydro.capillarity.evaluate(
            state.eta, hydro.plan.density
        ).surface_energy
        return (
            view.volume,
            view.kinetic_energy,
            gravitational,
            surface_energy,
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: FreeSurfaceALEContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        controller = state.wave_controller
        if self.hydrodynamics.wave is not None and controller is None:
            controller = self.hydrodynamics.wave.initial_controller_state(
                state.state.eta.shape, state.state.eta.dtype
            )
        dt = jnp.asarray(step_size, dtype=state.state.eta.dtype)
        half, half_rate, half_pressure, _, half_evidence = self._coupled_advance(
            state.state,
            state.state,
            state.eta_rate,
            state.pressure_head,
            time,
            time,
            time + 0.5 * dt,
            0.5 * dt,
            controller,
            args,
        )
        candidate, eta_rate, pressure, projection, evidence = self._coupled_advance(
            state.state,
            half,
            half_rate,
            half_pressure,
            time,
            time + 0.5 * dt,
            time + dt,
            dt,
            controller,
            args,
        )
        successful = half_evidence.successful & evidence.successful
        (
            volume_before,
            kinetic_before,
            gravity_before,
            surface_before,
        ) = self._energy(self.hydrodynamics, state.state, state.eta_rate, time)
        (
            volume_after,
            kinetic_after,
            gravity_after,
            surface_after,
        ) = self._energy(self.hydrodynamics, candidate, eta_rate, time + dt)
        scalar_change = {
            name: jnp.sum(
                candidate.scalar_content[name] - state.state.scalar_content[name]
            )
            for name in candidate.scalar_content
        }
        pressure_work = dt * evidence.pressure_work_rate
        capillary_work = dt * evidence.capillary_work_rate
        volume_change = volume_after - volume_before
        gas_head = (
            self.hydrodynamics.plan.boundary.gas_pressure
            - self.hydrodynamics.plan.boundary.reference_pressure
        ) / self.hydrodynamics.plan.density
        gas_pressure_work = -self.hydrodynamics.plan.density * gas_head * volume_change
        wave_work = dt * evidence.wave_work_rate
        relaxation_work = dt * evidence.relaxation_work_rate
        sponge_dissipation = dt * evidence.sponge_dissipation_rate
        controller_work = dt * evidence.controller_work_rate
        energy_residual = (
            kinetic_after
            - kinetic_before
            + gravity_after
            - gravity_before
            + surface_after
            - surface_before
            - gas_pressure_work
        )
        zero = jnp.asarray(0.0, dtype=dt.dtype)
        total_energy_residual = (
            energy_residual
            - wave_work
            - relaxation_work
            + sponge_dissipation
            - controller_work
        )
        ledger_increment = FreeSurfaceALELedger(
            volume_change=volume_change,
            scalar_change=scalar_change,
            kinetic_energy_change=kinetic_after - kinetic_before,
            gravitational_energy_change=gravity_after - gravity_before,
            surface_energy_change=surface_after - surface_before,
            pressure_work=pressure_work,
            capillary_work=capillary_work,
            gas_pressure_work=gas_pressure_work,
            wave_work=wave_work,
            relaxation_work=relaxation_work,
            sponge_dissipation=sponge_dissipation,
            body_work=zero,
            body_energy_change=zero,
            remap_energy_defect=zero,
            shoreline_work=zero,
            controller_work=controller_work,
            gcl_residual=evidence.gcl_residual,
            divergence_residual=evidence.divergence_residual,
            kinematic_residual=evidence.kinematic_residual,
            dynamic_residual=evidence.dynamic_residual,
            capillary_dual_residual=evidence.capillary_dual_residual,
            nonlinear_stage_residual=evidence.nonlinear_residual,
            total_energy_residual=total_energy_residual,
        )
        new_controller = controller
        if self.hydrodynamics.wave is not None and controller is not None:
            final_view = self.hydrodynamics.view(candidate, eta_rate, time + dt)
            top_coordinates = jnp.concatenate(
                (
                    final_view.geometry.cell_centers[..., -1, :2],
                    candidate.eta[..., None],
                ),
                axis=-1,
            )
            target_eta = self.hydrodynamics.wave.provider.sample(
                time + dt, top_coordinates
            ).eta
            new_controller = self.hydrodynamics.wave.update_controller(
                controller, time + dt, candidate.eta, target_eta
            )
        ledger = jax.tree.map(
            lambda total, increment: total + increment,
            state.ledger,
            ledger_increment,
        )
        proposed = FreeSurfaceALEContinuationState(
            candidate,
            eta_rate,
            pressure,
            ledger,
            state.mesh_epoch,
            new_controller,
        )
        selection_state = state
        if state.wave_controller is None and controller is not None:
            selection_state = FreeSurfaceALEContinuationState(
                state.state,
                state.eta_rate,
                state.pressure_head,
                state.ledger,
                state.mesh_epoch,
                controller,
            )
        accepted = jax.tree.map(
            lambda proposal, current: jnp.where(successful, proposal, current),
            proposed,
            selection_state,
        )
        residual = jnp.max(
            jnp.stack(
                (
                    jnp.abs(evidence.volume_residual),
                    evidence.divergence_residual,
                    evidence.kinematic_residual,
                    evidence.nonlinear_residual,
                    projection.pressure_residual_norm,
                )
            )
        )
        return FixedStepResult(
            candidate_state=proposed,
            accepted_state=accepted,
            successful=successful,
            residual=residual,
            iterations=jnp.asarray(
                2 * self.hydrodynamics.plan.coupling_iterations,
                dtype=jnp.int32,
            ),
            work=jnp.asarray(
                2 * self.hydrodynamics.plan.coupling_iterations,
                dtype=jnp.int32,
            ),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0, dtype=dt.dtype),
        )


def write_free_surface_checkpoint(
    path: str | Path,
    hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
    method: OnePhaseFreeSurfaceALEMethod,
    time: ArrayLike,
    accepted_step: ArrayLike,
    state: FreeSurfaceALEContinuationState,
    /,
) -> Path:
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "accepted_step": jnp.asarray(accepted_step),
    }
    specification = pack_array_tree("state", state, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "one-phase-free-surface-ale-checkpoint-v2",
            "schema": "pressure-boundary-work-capillary-mesh-epoch",
            "hydrodynamics_id": hydrodynamics.prepared_id,
            "method_id": method.method_id,
            "state": specification,
        },
        arrays=arrays,
    )


def read_free_surface_checkpoint(
    path: str | Path,
    hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
    method: OnePhaseFreeSurfaceALEMethod,
    template: FreeSurfaceALEContinuationState,
    /,
) -> tuple[Array, Array, FreeSurfaceALEContinuationState]:
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "one-phase-free-surface-ale-checkpoint-v2":
        raise ValueError("Archive is not a one-phase free-surface checkpoint.")
    if manifest.get("hydrodynamics_id") != hydrodynamics.prepared_id:
        raise ValueError("Free-surface checkpoint model identity mismatch.")
    if manifest.get("method_id") != method.method_id:
        raise ValueError("Free-surface checkpoint method identity mismatch.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    return jnp.asarray(arrays["time"]), jnp.asarray(arrays["accepted_step"]), restored


__all__ = [
    "FreeSurfaceALEContinuationState",
    "FreeSurfaceALELedger",
    "FreeSurfaceALEStageEvidence",
    "FreeSurfaceBoundaryPlan",
    "OnePhaseFreeSurfaceALEMethod",
    "OnePhaseFreeSurfaceALEPlan",
    "PreparedOnePhaseFreeSurfaceALE",
    "read_free_surface_checkpoint",
    "write_free_surface_checkpoint",
]

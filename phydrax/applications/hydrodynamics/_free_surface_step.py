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
from ...solver._free_surface_ale_projection import (
    FreeSurfaceProjectionResult,
    MappedFreeSurfaceProjectionPlan,
)
from ._free_surface_ale import (
    _tuple_add,
    FaceTuple,
    FreeSurfaceALEState,
    FreeSurfaceALEStateView,
    GraphSurfaceALEPlan,
    PreparedGraphSurfaceALE,
)


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


class FreeSurfaceBoundaryPlan(StrictModule, NonTrainableState):
    """Atmospheric pressure and closed/periodic boundary semantics for G0-G4."""

    atmospheric_pressure: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(self, atmospheric_pressure: float = 0.0, /):
        pressure = float(atmospheric_pressure)
        if not np.isfinite(pressure):
            raise ValueError("Atmospheric pressure must be finite.")
        self.atmospheric_pressure = pressure
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "free-surface-boundary-plan",
                "atmospheric_pressure": pressure,
                "lateral": "periodic-or-closed",
                "bottom": "impermeable",
                "top": "pressure-dirichlet",
            }
        )


class FreeSurfaceALELedger(StrictModule):
    volume_change: Array
    scalar_change: dict[str, Array]
    kinetic_energy_change: Array
    gravitational_energy_change: Array
    pressure_work: Array
    gcl_residual: Array
    divergence_residual: Array
    kinematic_residual: Array
    dynamic_residual: Array
    nonlinear_stage_residual: Array
    total_energy_residual: Array

    @classmethod
    def zeros(cls, scalar_names: tuple[str, ...], dtype, /) -> "FreeSurfaceALELedger":
        zero = jnp.zeros((), dtype=dtype)
        return cls(
            zero,
            {name: zero for name in scalar_names},
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )


class FreeSurfaceALEContinuationState(StrictModule):
    state: FreeSurfaceALEState
    eta_rate: Array
    pressure_head: Array
    ledger: FreeSurfaceALELedger

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
        )


class FreeSurfaceALEStageEvidence(StrictModule):
    geometry_valid: Array
    projection_successful: Array
    kinematic_successful: Array
    volume_residual: Array
    gcl_residual: Array
    divergence_residual: Array
    kinematic_residual: Array
    dynamic_residual: Array
    nonlinear_residual: Array
    finite: Array
    successful: Array


class OnePhaseFreeSurfaceALEPlan(StrictModule, NonTrainableState):
    """Compile fixed-topology inviscid one-phase graph-surface ALE hydrodynamics."""

    surface_plan: GraphSurfaceALEPlan
    boundary: FreeSurfaceBoundaryPlan
    density: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
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
        coupling_iterations: int = 6,
        coupling_tolerance: float = 1.0e-8,
    ):
        if not isinstance(surface_plan, GraphSurfaceALEPlan):
            raise TypeError("surface_plan must be GraphSurfaceALEPlan.")
        boundary_ = FreeSurfaceBoundaryPlan() if boundary is None else boundary
        density_ = float(density)
        gravity_ = float(gravity)
        iterations = int(coupling_iterations)
        tolerance = float(coupling_tolerance)
        if (
            density_ <= 0.0
            or gravity_ <= 0.0
            or iterations <= 0
            or tolerance <= 0.0
            or any(not np.isfinite(v) for v in (density_, gravity_, tolerance))
        ):
            raise ValueError("Invalid free-surface density, gravity, or coupling policy.")
        self.surface_plan = surface_plan
        self.boundary = boundary_
        self.density = density_
        self.gravity = gravity_
        self.coupling_iterations = iterations
        self.coupling_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "one-phase-free-surface-ale-plan",
                "surface": surface_plan.plan_id,
                "boundary": boundary_.boundary_id,
                "density": density_,
                "gravity": gravity_,
                "coupling_iterations": iterations,
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
        return PreparedOnePhaseFreeSurfaceALE(self, surface, projection)


class PreparedOnePhaseFreeSurfaceALE(StrictModule):
    plan: OnePhaseFreeSurfaceALEPlan
    surface: PreparedGraphSurfaceALE
    projection: MappedFreeSurfaceProjectionPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: OnePhaseFreeSurfaceALEPlan,
        surface: PreparedGraphSurfaceALE,
        projection: MappedFreeSurfaceProjectionPlan,
        /,
    ):
        self.plan = plan
        self.surface = surface
        self.projection = projection
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-one-phase-free-surface-ale",
                "plan": plan.plan_id,
                "surface": surface.surface_id,
                "projection": projection.plan_id,
            }
        )

    @property
    def reference(self):
        return self.surface.plan.reference

    def initial_state(
        self,
        eta: ArrayLike,
        velocity: FaceTuple | None = None,
        scalars: dict[str, ArrayLike] | None = None,
        /,
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
            kinetic_energy=self.plan.density * geometry.kinetic_energy(hodge.velocity),
            volume=jnp.sum(geometry.cell_volumes),
            view_id=self.prepared_id,
        )

    def surface_pressure_head(self, eta: ArrayLike, /) -> Array:
        eta_ = jnp.asarray(eta)
        return (
            self.plan.gravity * eta_
            - self.plan.boundary.atmospheric_pressure / self.plan.density
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
        time: Array,
        dt: Array,
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

        for _ in range(hydro.plan.coupling_iterations):
            geometry = hydro.surface.geometry(time, evaluation.eta, eta_rate, args)
            evaluation_velocity = hydro.surface.inverse_hodge(
                geometry, evaluation.momentum
            )
            momentum_rate_velocity = hydro.surface.ale.momentum_rate(
                geometry,
                evaluation_velocity.velocity,
                viscosity=0.0,
                forcing=None,
            )
            base_geometry = hydro.surface.geometry(time, base.eta, eta_rate, args)
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
                time + dt, eta_candidate, eta_rate, args
            )
            tentative_momentum = hydro.surface.apply_hodge(
                end_geometry, candidate_velocity
            )
            surface_force = hydro.projection.surface_pressure_force(
                end_geometry,
                hydro.surface_pressure_head(eta_candidate),
            )
            tentative_momentum = _tuple_add(tentative_momentum, dt, surface_force)
            projection = hydro.projection.project(
                end_geometry,
                tentative_momentum,
                dt,
                pressure,
            )
            target_flux = hydro.surface.top_volume_flux(end_geometry, projection.velocity)
            kinematic = hydro.surface.solve_eta_rate(eta_candidate, target_flux)
            next_rate = kinematic.eta_rate
            stage_residual = jnp.max(jnp.abs(next_rate - eta_rate))
            eta_rate = next_rate
            pressure = projection.pressure_head
            geometry = end_geometry

        eta_new = base.eta + dt * eta_rate
        final_geometry = hydro.surface.geometry(time + dt, eta_new, eta_rate, args)
        final_projection = hydro.projection.project(
            final_geometry,
            projection.momentum,
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
        kinematic = hydro.surface.solve_eta_rate(eta_new, target_flux)
        geometry_evidence = hydro.surface.geometry_evidence(eta_new, eta_rate, time + dt)
        volume_before = jnp.sum(base_geometry.cell_volumes)
        volume_after = jnp.sum(final_geometry.cell_volumes)
        top_flux = jnp.sum(target_flux)
        volume_residual = volume_after - volume_before - dt * top_flux
        divergence_residual = jnp.sqrt(
            jnp.sum(final_geometry.cell_volumes * final_projection.divergence_after**2)
        )
        dynamic_residual = jnp.max(
            jnp.abs(
                hydro.surface_pressure_head(eta_new)
                - hydro.surface_pressure_head(eta_new)
            )
        )
        finite = (
            geometry_evidence.finite
            & final_projection.finite
            & kinematic.finite
            & jnp.all(jnp.isfinite(eta_new))
            & jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value)) for value in scalar_content.values()
                    )
                )
            )
        )
        successful = (
            finite
            & geometry_evidence.valid
            & final_projection.successful
            & kinematic.converged
            & (stage_residual <= hydro.plan.coupling_tolerance)
        )
        evidence = FreeSurfaceALEStageEvidence(
            geometry_valid=geometry_evidence.valid,
            projection_successful=final_projection.successful,
            kinematic_successful=kinematic.converged,
            volume_residual=volume_residual,
            gcl_residual=geometry_evidence.volume_gcl_residual,
            divergence_residual=divergence_residual,
            kinematic_residual=kinematic.residual_norm,
            dynamic_residual=dynamic_residual,
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
    ) -> tuple[Array, Array, Array]:
        view = hydro.view(state, eta_rate, time)
        gravitational = (
            0.5
            * hydro.plan.density
            * hydro.plan.gravity
            * jnp.sum(hydro.surface.horizontal_area * state.eta**2)
        )
        return view.volume, view.kinetic_energy, gravitational

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
        dt = jnp.asarray(step_size, dtype=state.state.eta.dtype)
        half, half_rate, half_pressure, _, half_evidence = self._coupled_advance(
            state.state,
            state.state,
            state.eta_rate,
            state.pressure_head,
            time,
            0.5 * dt,
            args,
        )
        candidate, eta_rate, pressure, projection, evidence = self._coupled_advance(
            state.state,
            half,
            half_rate,
            half_pressure,
            time + 0.5 * dt,
            dt,
            args,
        )
        successful = half_evidence.successful & evidence.successful
        volume_before, kinetic_before, gravity_before = self._energy(
            self.hydrodynamics, state.state, state.eta_rate, time
        )
        volume_after, kinetic_after, gravity_after = self._energy(
            self.hydrodynamics, candidate, eta_rate, time + dt
        )
        scalar_change = {
            name: jnp.sum(
                candidate.scalar_content[name] - state.state.scalar_content[name]
            )
            for name in candidate.scalar_content
        }
        pressure_work = (kinetic_after - kinetic_before) + (
            gravity_after - gravity_before
        )
        energy_residual = (
            kinetic_after
            - kinetic_before
            + gravity_after
            - gravity_before
            - pressure_work
        )
        ledger_increment = FreeSurfaceALELedger(
            volume_change=volume_after - volume_before,
            scalar_change=scalar_change,
            kinetic_energy_change=kinetic_after - kinetic_before,
            gravitational_energy_change=gravity_after - gravity_before,
            pressure_work=pressure_work,
            gcl_residual=evidence.gcl_residual,
            divergence_residual=evidence.divergence_residual,
            kinematic_residual=evidence.kinematic_residual,
            dynamic_residual=evidence.dynamic_residual,
            nonlinear_stage_residual=evidence.nonlinear_residual,
            total_energy_residual=energy_residual,
        )
        ledger = jax.tree.map(
            lambda total, increment: total + increment,
            state.ledger,
            ledger_increment,
        )
        proposed = FreeSurfaceALEContinuationState(candidate, eta_rate, pressure, ledger)
        accepted = jax.tree.map(
            lambda proposal, current: jnp.where(successful, proposal, current),
            proposed,
            state,
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
            "kind": "one-phase-free-surface-ale-checkpoint",
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
    if manifest.get("kind") != "one-phase-free-surface-ale-checkpoint":
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

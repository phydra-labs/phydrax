#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._vof import (
    FaceTuple,
    PreparedIncompressibleTwoPhaseVOF,
    TwoPhaseTopologyEvidence,
    TwoPhaseVOFState,
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


def _cell_from_faces(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    centered = (
        0.5 * (moved + jnp.roll(moved, -1, axis=0))
        if periodic
        else 0.5 * (moved[:-1] + moved[1:])
    )
    return jnp.moveaxis(centered, 0, axis)


def _faces_from_cell(value: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    if periodic:
        faces = 0.5 * (jnp.roll(moved, 1, axis=0) + moved)
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


class TwoPhaseMovingBodyPlan(StrictModule, NonTrainableState):
    center: Array
    radius: float = eqx.field(static=True)
    velocity: Array
    penalty: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        /,
        *,
        velocity: ArrayLike = (0.0, 0.0, 0.0),
        penalty: float = 1.0,
    ):
        center_ = jnp.asarray(center)
        velocity_ = jnp.asarray(velocity, dtype=center_.dtype)
        radius_ = float(radius)
        penalty_ = float(penalty)
        if center_.shape != velocity_.shape or center_.ndim != 1:
            raise ValueError("Two-phase body center/velocity shapes are invalid.")
        if radius_ <= 0.0 or penalty_ < 0.0:
            raise ValueError("Two-phase body radius/penalty are invalid.")
        self.center = center_
        self.radius = radius_
        self.velocity = velocity_
        self.penalty = penalty_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-phase-moving-body-plan",
                "radius": radius_,
                "penalty": penalty_,
            }
        )


class TwoPhaseVOFLedger(StrictModule):
    liquid_volume_change: Array
    gas_volume_change: Array
    momentum_change: Array
    kinetic_energy_change: Array
    gravitational_energy_change: Array
    surface_energy_change: Array
    viscous_dissipation: Array
    capillary_work: Array
    body_work: Array
    vof_limiter_correction: Array
    clsvof_volume_correction: Array
    reinitialization_dissipation: Array
    pressure_residual: Array
    divergence_residual: Array
    topology_event_count: Array
    total_energy_residual: Array

    @classmethod
    def zeros(cls, dtype, /):
        zero = jnp.zeros((), dtype=dtype)
        return cls(*((zero,) * 16))


class TwoPhaseStepEvidence(StrictModule):
    alpha_minimum: Array
    alpha_maximum: Array
    liquid_volume_residual: Array
    mass_flux_residual: Array
    momentum_flux_residual: Array
    pressure_residual: Array
    divergence_residual: Array
    capillary_balance_residual: Array
    clsvof_correction: Array
    body_residual: Array
    topology_event_count: Array
    finite: Array
    geometry_accepted: Array
    solid_interface_conflict_count: Array
    successful: Array


class TwoPhaseContinuationState(StrictModule):
    state: TwoPhaseVOFState
    pressure: Array
    ledger: TwoPhaseVOFLedger
    topology: TwoPhaseTopologyEvidence
    evidence: TwoPhaseStepEvidence | None


class IncompressibleTwoPhaseVOFMethod(AbstractFixedStepMethod):
    """Conservative VOF/mass/momentum step with variable-density projection."""

    two_phase: PreparedIncompressibleTwoPhaseVOF
    body: TwoPhaseMovingBodyPlan | None
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        two_phase: PreparedIncompressibleTwoPhaseVOF,
        /,
        *,
        body: TwoPhaseMovingBodyPlan | None = None,
    ):
        if not isinstance(two_phase, PreparedIncompressibleTwoPhaseVOF):
            raise TypeError("two_phase must be PreparedIncompressibleTwoPhaseVOF.")
        if body is not None and not isinstance(body, TwoPhaseMovingBodyPlan):
            raise TypeError("body must be TwoPhaseMovingBodyPlan or None.")
        if body is not None and two_phase.geometry is not None:
            raise ValueError(
                "VOF cannot combine qualified cut measures with the independent "
                "penalty moving-body path."
            )
        self.two_phase = two_phase
        self.body = body
        self.method_id = canonical_fingerprint(
            {
                "kind": "incompressible-two-phase-vof-method",
                "two_phase": two_phase.prepared_id,
                "body": "none" if body is None else body.plan_id,
                "interface_authority": "alpha-volume",
                "momentum_transport": "consistent-phase-mass-flux",
            }
        )

    def initial_continuation(
        self, state: TwoPhaseVOFState, /
    ) -> TwoPhaseContinuationState:
        alpha = self.two_phase.alpha(state)
        zero = jnp.asarray(0.0, dtype=alpha.dtype)
        initial_evidence = TwoPhaseStepEvidence(
            alpha_minimum=jnp.min(alpha),
            alpha_maximum=jnp.max(alpha),
            liquid_volume_residual=zero,
            mass_flux_residual=zero,
            momentum_flux_residual=zero,
            pressure_residual=zero,
            divergence_residual=zero,
            capillary_balance_residual=zero,
            clsvof_correction=zero,
            body_residual=zero,
            topology_event_count=jnp.asarray(0, dtype=jnp.int32),
            finite=jnp.asarray(True),
            successful=jnp.asarray(True),
            geometry_accepted=(
                jnp.asarray(True)
                if self.two_phase.geometry is None
                else self.two_phase.geometry.accepted
            ),
            solid_interface_conflict_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return TwoPhaseContinuationState(
            state,
            jnp.zeros_like(alpha),
            TwoPhaseVOFLedger.zeros(alpha.dtype),
            self.two_phase.topology_evidence(state),
            initial_evidence,
        )

    def _phase_fluxes(self, alpha: Array, velocity: FaceTuple, dt: Array, /):
        discretization = self.two_phase.plan.discretization
        periodic = tuple(axis.periodic for axis in discretization.grid.structured_axes)
        total_fluxes = tuple(
            component * measure
            for component, measure in zip(
                velocity, self.two_phase.face_open_measure, strict=True
            )
        )
        liquid_fluxes = tuple(
            total * _face_upwind(alpha, total, axis, periodic[axis])
            for axis, total in enumerate(total_fluxes)
        )
        net = sum(
            _cell_net_flux(flux, axis, periodic[axis])
            for axis, flux in enumerate(liquid_fluxes)
        )
        content = self.two_phase.cell_fluid_measure * alpha
        candidate = content - dt * net
        minimum = jnp.min(candidate)
        maximum = jnp.max(candidate - self.two_phase.cell_fluid_measure)
        valid = (minimum >= -1.0e-12) & (maximum <= 1.0e-12)
        factor = jnp.where(
            valid,
            1.0,
            jnp.minimum(
                1.0,
                jnp.min(
                    jnp.where(
                        dt * jnp.abs(net) > 0.0,
                        content / (dt * jnp.abs(net)),
                        1.0,
                    )
                ),
            ),
        )
        liquid_fluxes = tuple(factor * flux for flux in liquid_fluxes)
        return total_fluxes, liquid_fluxes, factor

    def _consistent_momentum_rate(
        self,
        alpha: Array,
        density: Array,
        velocity: FaceTuple,
        total_fluxes: FaceTuple,
        liquid_fluxes: FaceTuple,
        /,
    ) -> FaceTuple:
        material = self.two_phase.plan.material
        periodic = tuple(
            axis.periodic
            for axis in self.two_phase.plan.discretization.grid.structured_axes
        )
        mass_fluxes = tuple(
            material.liquid_density * liquid + material.gas_density * (total - liquid)
            for total, liquid in zip(total_fluxes, liquid_fluxes, strict=True)
        )
        cell_velocity = jnp.stack(
            tuple(
                _cell_from_faces(component, axis, periodic[axis])
                for axis, component in enumerate(velocity)
            ),
            axis=-1,
        )
        rates = []
        for component_axis in range(len(velocity)):
            net = sum(
                _cell_net_flux(
                    flux
                    * _face_upwind(
                        cell_velocity[..., component_axis],
                        flux,
                        axis,
                        periodic[axis],
                    ),
                    axis,
                    periodic[axis],
                )
                for axis, flux in enumerate(mass_fluxes)
            )
            fluid_volume = self.two_phase.cell_fluid_measure
            active = fluid_volume > 0.0
            acceleration = jnp.where(
                active,
                -net
                / jnp.where(
                    active,
                    fluid_volume * jnp.maximum(density, material.gas_density),
                    1.0,
                ),
                0.0,
            )
            rates.append(
                _faces_from_cell(
                    acceleration,
                    component_axis,
                    periodic[component_axis],
                )
            )
        return tuple(rates)

    def _capillary_force(
        self, alpha: Array, density: Array, /
    ) -> tuple[FaceTuple, Array, Array]:
        operators = self.two_phase.operators
        material = self.two_phase.plan.material
        if material.surface_tension == 0.0:
            zero = tuple(
                jnp.zeros(layout.shape, dtype=alpha.dtype)
                for layout in self.two_phase.plan.discretization.face_layouts
            )
            return (
                zero,
                jnp.asarray(0.0, dtype=alpha.dtype),
                jnp.asarray(0.0, dtype=alpha.dtype),
            )
        volume = self.two_phase.cell_fluid_measure

        def area(value):
            gradients = jnp.stack(
                tuple(jnp.gradient(value, axis=axis) for axis in range(value.ndim)),
                axis=-1,
            )
            return material.surface_tension * jnp.sum(
                volume * jnp.sqrt(jnp.sum(gradients**2, axis=-1) + 1.0e-12)
            )

        potential = jnp.where(
            volume > 0.0,
            jax.grad(area)(alpha) / jnp.where(volume > 0.0, volume, 1.0),
            0.0,
        )
        gradient = operators.gradient(potential)
        face_density = self.two_phase.face_density(density)
        force = tuple(
            -component / rho
            for component, rho in zip(gradient, face_density, strict=True)
        )
        surface_energy = area(alpha)
        balance = jnp.sqrt(sum(jnp.real(jnp.vdot(f, f)) for f in force))
        return force, surface_energy, balance

    def _body_force(self, velocity: FaceTuple, dt: Array, /) -> tuple[FaceTuple, Array]:
        if self.body is None:
            return tuple(jnp.zeros_like(v) for v in velocity), jnp.asarray(
                0.0, dtype=dt.dtype
            )
        discretization = self.two_phase.plan.discretization
        force = []
        work = jnp.asarray(0.0, dtype=dt.dtype)
        for axis, component in enumerate(velocity):
            coordinates = discretization.face_centers[axis]
            distance = jnp.linalg.norm(coordinates - self.body.center, axis=-1)
            mask = distance <= self.body.radius
            target = self.body.velocity[axis]
            acceleration = (
                self.body.penalty * jnp.where(mask, target - component, 0.0) / dt
            )
            force.append(acceleration)
            work = work + jnp.sum(
                self.two_phase.face_open_measure[axis] * component * acceleration
            )
        return tuple(force), work

    def step(
        self,
        step_index: Array,
        time: Array,
        state: TwoPhaseContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index
        dt = jnp.asarray(step_size, dtype=state.state.liquid_content.dtype)
        previous_alpha = self.two_phase.alpha(state.state)
        velocity = self.two_phase.velocity(state.state)
        total_fluxes, liquid_fluxes, limiter = self._phase_fluxes(
            previous_alpha, velocity, dt
        )
        net_liquid = sum(
            _cell_net_flux(
                flux,
                axis,
                self.two_phase.plan.discretization.grid.structured_axes[axis].periodic,
            )
            for axis, flux in enumerate(liquid_fluxes)
        )
        liquid_content = state.state.liquid_content - dt * net_liquid
        fluid_volume = self.two_phase.cell_fluid_measure
        fluid_active = fluid_volume > 0.0
        alpha = jnp.where(
            fluid_active,
            liquid_content / jnp.where(fluid_active, fluid_volume, 1.0),
            0.0,
        )
        density = self.two_phase.mixture_density(alpha)
        momentum_rate = self._consistent_momentum_rate(
            previous_alpha,
            self.two_phase.mixture_density(previous_alpha),
            velocity,
            total_fluxes,
            liquid_fluxes,
        )
        capillary, surface_energy_before, capillary_balance = self._capillary_force(
            previous_alpha, self.two_phase.mixture_density(previous_alpha)
        )
        body_force, body_work = self._body_force(velocity, dt)
        face_density = self.two_phase.face_density(density)
        predictor_momentum = tuple(
            old_momentum + dt * rho * measure * (advection + surface + body)
            for old_momentum, rho, measure, advection, surface, body in zip(
                state.state.momentum,
                face_density,
                self.two_phase.face_open_dual_measure,
                momentum_rate,
                capillary,
                body_force,
                strict=True,
            )
        )
        inverse_density = tuple(1.0 / rho for rho in face_density)
        if self.two_phase.sharp_projection is None:
            projection = self.two_phase.projection.project(
                predictor_momentum,
                inverse_density,
                dt,
                pressure=state.pressure,
            )
            momentum = projection.momentum
            projected_velocity = projection.velocity
            projection_finite = projection.finite
            projection_converged = projection.converged
            pressure_residual_norm = jnp.sqrt(
                jnp.sum(
                    self.two_phase.cell_fluid_measure * projection.pressure_residual**2
                )
            )
        else:
            open_dual = self.two_phase.face_open_dual_measure
            predictor_velocity = tuple(
                jnp.where(
                    rho * measure > 0.0,
                    value / jnp.where(rho * measure > 0.0, rho * measure, 1.0),
                    0.0,
                )
                for value, rho, measure in zip(
                    predictor_momentum, face_density, open_dual, strict=True
                )
            )
            stage = self.two_phase.boundaries.evaluate(time, args)
            projection = self.two_phase.sharp_projection.project(
                predictor_velocity,
                tuple(dt * value for value in inverse_density),
                stage,
                pressure=state.pressure,
            )
            projected_velocity = projection.velocity
            momentum = tuple(
                rho * measure * value
                for rho, measure, value in zip(
                    face_density, open_dual, projected_velocity, strict=True
                )
            )
            projection_finite = projection.force.finite & jnp.isfinite(
                projection.divergence_norm
            )
            projection_converged = projection.accepted
            pressure_residual_norm = projection.linear.diagnostics.residual_norm
        alpha_level_set = self.two_phase.level_set_from_alpha(alpha)
        level_set = 0.5 * state.state.level_set + 0.5 * alpha_level_set
        phase_scalars = {}
        for name, content in state.state.phase_scalar_content.items():
            concentration = jnp.where(
                state.state.liquid_content > 0.0,
                content / state.state.liquid_content,
                0.0,
            )
            net = sum(
                _cell_net_flux(
                    flux
                    * _face_upwind(
                        concentration,
                        flux,
                        axis,
                        self.two_phase.plan.discretization.grid.structured_axes[
                            axis
                        ].periodic,
                    ),
                    axis,
                    self.two_phase.plan.discretization.grid.structured_axes[
                        axis
                    ].periodic,
                )
                for axis, flux in enumerate(liquid_fluxes)
            )
            phase_scalars[name] = content - dt * net
        geometry_epoch = (
            jnp.asarray(-1, dtype=jnp.int32)
            if self.two_phase.geometry is None
            else self.two_phase.geometry.epoch
        )
        geometry_id = (
            ""
            if self.two_phase.geometry is None
            else self.two_phase.geometry.realization_id
        )
        candidate_state = TwoPhaseVOFState(
            liquid_content,
            momentum,
            phase_scalars,
            level_set,
            geometry_epoch,
            geometry_id,
        )
        topology = self.two_phase.topology_evidence(candidate_state, previous_alpha)
        surface_energy_after = self._capillary_force(alpha, density)[1]
        liquid_change = jnp.sum(liquid_content - state.state.liquid_content)
        volume_scale = jnp.maximum(jnp.sum(state.state.liquid_content), 1.0)
        liquid_residual = jnp.abs(liquid_change) / volume_scale
        divergence = jnp.sqrt(
            jnp.sum(self.two_phase.cell_fluid_measure * projection.divergence_after**2)
        )
        alpha_minimum = jnp.min(alpha)
        alpha_maximum = jnp.max(alpha)
        clsvof_correction = jnp.sqrt(jnp.mean((level_set - alpha_level_set) ** 2))
        geometry_accepted = (
            jnp.asarray(True)
            if self.two_phase.geometry is None
            else self.two_phase.geometry.accepted
            & (state.state.geometry_epoch == self.two_phase.geometry.epoch)
        )
        conflict_count = jnp.sum(
            self.two_phase.plic(alpha).solid_interface_conflict,
            dtype=jnp.int32,
        )
        finite = (
            topology.finite
            & projection_finite
            & jnp.all(jnp.isfinite(alpha))
            & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in momentum)))
        )
        successful = (
            finite
            & projection_converged
            & geometry_accepted
            & topology.valid
            & (conflict_count == 0)
            & (alpha_minimum >= -64.0 * jnp.finfo(alpha.dtype).eps)
            & (alpha_maximum <= 1.0 + 64.0 * jnp.finfo(alpha.dtype).eps)
        )
        evidence = TwoPhaseStepEvidence(
            alpha_minimum=alpha_minimum,
            alpha_maximum=alpha_maximum,
            liquid_volume_residual=liquid_residual,
            mass_flux_residual=jnp.asarray(0.0, dtype=dt.dtype),
            momentum_flux_residual=jnp.asarray(0.0, dtype=dt.dtype),
            pressure_residual=pressure_residual_norm,
            divergence_residual=divergence,
            capillary_balance_residual=capillary_balance,
            clsvof_correction=clsvof_correction,
            body_residual=jnp.asarray(0.0, dtype=dt.dtype),
            topology_event_count=topology.component_proxy,
            finite=finite,
            successful=successful,
            geometry_accepted=geometry_accepted,
            solid_interface_conflict_count=conflict_count,
        )
        kinetic_before = 0.5 * sum(
            jnp.sum(rho * measure * component**2)
            for rho, measure, component in zip(
                self.two_phase.face_density(
                    self.two_phase.mixture_density(previous_alpha)
                ),
                self.two_phase.face_open_dual_measure,
                velocity,
                strict=True,
            )
        )
        kinetic_after = 0.5 * sum(
            jnp.sum(rho * measure * component**2)
            for rho, measure, component in zip(
                face_density,
                self.two_phase.face_open_dual_measure,
                projected_velocity,
                strict=True,
            )
        )
        ledger_increment = TwoPhaseVOFLedger(
            liquid_volume_change=liquid_change,
            gas_volume_change=-liquid_change,
            momentum_change=sum(
                jnp.sum(new - old)
                for new, old in zip(momentum, state.state.momentum, strict=True)
            ),
            kinetic_energy_change=kinetic_after - kinetic_before,
            gravitational_energy_change=jnp.asarray(0.0, dtype=dt.dtype),
            surface_energy_change=surface_energy_after - surface_energy_before,
            viscous_dissipation=jnp.asarray(0.0, dtype=dt.dtype),
            capillary_work=dt
            * sum(
                jnp.sum(force * component)
                for force, component in zip(capillary, velocity, strict=True)
            ),
            body_work=dt * body_work,
            vof_limiter_correction=1.0 - limiter,
            clsvof_volume_correction=jnp.sum(
                candidate_state.liquid_content - liquid_content
            ),
            reinitialization_dissipation=clsvof_correction,
            pressure_residual=pressure_residual_norm,
            divergence_residual=divergence,
            topology_event_count=topology.component_proxy,
            total_energy_residual=(
                kinetic_after
                - kinetic_before
                + surface_energy_after
                - surface_energy_before
                - dt * body_work
            ),
        )
        ledger = jax.tree.map(
            lambda total, increment: total + increment,
            state.ledger,
            ledger_increment,
        )
        candidate = TwoPhaseContinuationState(
            candidate_state,
            projection.pressure,
            ledger,
            topology,
            evidence,
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(successful, proposed, current),
            candidate,
            state,
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=jnp.max(
                jnp.stack(
                    (
                        liquid_residual,
                        divergence,
                        pressure_residual_norm,
                        capillary_balance,
                    )
                )
            ),
            iterations=jnp.asarray(
                self.two_phase.plan.maximum_iterations, dtype=jnp.int32
            ),
            work=jnp.asarray(self.two_phase.plan.maximum_iterations, dtype=jnp.int32),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0, dtype=dt.dtype),
        )


__all__ = [
    "IncompressibleTwoPhaseVOFMethod",
    "TwoPhaseContinuationState",
    "TwoPhaseMovingBodyPlan",
    "TwoPhaseStepEvidence",
    "TwoPhaseVOFLedger",
]

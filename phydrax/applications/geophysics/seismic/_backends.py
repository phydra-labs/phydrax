#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import TopologyEpochTransition


class SecondOrderWaveState(StrictModule):
    displacement: object
    velocity_half_step: object
    step_index: Array
    plan_id: str = eqx.field(static=True)


class SpectralElementWavePlan(StrictModule, NonTrainableState):
    """Explicit second-order wave backend over prepared hp mass/stiffness actions."""

    mass: la.AbstractLinearOperator
    stiffness: la.AbstractLinearOperator
    mass_inverse: la.AbstractPreconditioner
    time_step_s: float = eqx.field(static=True)
    maximum_angular_frequency: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass: la.AbstractLinearOperator,
        stiffness: la.AbstractLinearOperator,
        mass_inverse: la.AbstractPreconditioner,
        time_step_s: float,
        maximum_angular_frequency: float,
        /,
    ):
        if not isinstance(mass, la.AbstractLinearOperator) or not isinstance(
            stiffness, la.AbstractLinearOperator
        ):
            raise TypeError("Spectral wave mass/stiffness must be native operators.")
        if not (
            mass.source.compatible(mass.target)
            and stiffness.source.compatible(stiffness.target)
            and mass.source.compatible(stiffness.source)
            and mass.source.compatible(mass_inverse.space)
        ):
            raise ValueError("Spectral wave operator and inverse spaces must agree.")
        dt, maximum = float(time_step_s), float(maximum_angular_frequency)
        if (
            not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(maximum)
            or maximum <= 0
            or dt * maximum >= 2.0
        ):
            raise ValueError(
                "Central-difference spectral wave stability requires dt*omega_max < 2."
            )
        self.mass, self.stiffness, self.mass_inverse = mass, stiffness, mass_inverse
        self.time_step_s, self.maximum_angular_frequency = dt, maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-element-wave-backend",
                "mass": mass.operator_id,
                "stiffness": stiffness.operator_id,
                "mass_inverse": mass_inverse.preconditioner_id,
                "time_step_s": dt,
                "maximum_angular_frequency": maximum,
            }
        )

    def initial_state(self, displacement: object | None = None) -> SecondOrderWaveState:
        value = (
            self.mass.source.zeros()
            if displacement is None
            else self.mass.source.validate(displacement)
        )
        return SecondOrderWaveState(
            value,
            self.mass.source.zeros(),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def step(self, state: SecondOrderWaveState, force: object, /) -> SecondOrderWaveState:
        if state.plan_id != self.plan_id:
            raise ValueError("Spectral wave state belongs to another plan.")
        force_ = self.mass.target.validate(force)
        elastic = self.stiffness.mv(state.displacement)
        residual = jax.tree.map(lambda applied, load: load - applied, elastic, force_)
        acceleration = self.mass_inverse.apply(residual)
        velocity = jax.tree.map(
            lambda old, value: old + self.time_step_s * value,
            state.velocity_half_step,
            acceleration,
        )
        displacement = jax.tree.map(
            lambda old, value: old + self.time_step_s * value,
            state.displacement,
            velocity,
        )
        return SecondOrderWaveState(
            displacement, velocity, state.step_index + 1, self.plan_id
        )

    def discrete_energy(self, state: SecondOrderWaveState, /) -> Array:
        mass_velocity = self.mass.mv(state.velocity_half_step)
        stiffness_displacement = self.stiffness.mv(state.displacement)
        kinetic = jnp.vdot(
            self.mass.source.flatten(state.velocity_half_step),
            self.mass.target.flatten(mass_velocity),
        )
        potential = jnp.vdot(
            self.stiffness.source.flatten(state.displacement),
            self.stiffness.target.flatten(stiffness_displacement),
        )
        return 0.5 * jnp.real(kinetic + potential)


class SeismicAMRTransitionResult(StrictModule):
    state: SecondOrderWaveState
    displacement_conservation: Array
    velocity_conservation: Array
    successful: Array
    differentiation_available: Array


class SeismicAMRTransition(StrictModule, NonTrainableState):
    displacement: TopologyEpochTransition
    velocity: TopologyEpochTransition
    source_plan_id: str = eqx.field(static=True)
    target_plan_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        displacement: TopologyEpochTransition,
        velocity: TopologyEpochTransition,
        source_plan_id: str,
        target_plan_id: str,
        /,
    ):
        if not isinstance(displacement, TopologyEpochTransition) or not isinstance(
            velocity, TopologyEpochTransition
        ):
            raise TypeError(
                "Seismic AMR needs displacement and velocity epoch transfers."
            )
        if (
            displacement.source.epoch_id != velocity.source.epoch_id
            or displacement.target.epoch_id != velocity.target.epoch_id
        ):
            raise ValueError("Seismic AMR field transfers must connect identical epochs.")
        source_identifier = str(source_plan_id).strip()
        target_identifier = str(target_plan_id).strip()
        if not source_identifier or not target_identifier:
            raise ValueError("Source and target wave plan identities are required.")
        self.displacement, self.velocity = displacement, velocity
        self.source_plan_id, self.target_plan_id = (
            source_identifier,
            target_identifier,
        )
        self.transition_id = canonical_fingerprint(
            {
                "kind": "seismic-amr-transition",
                "displacement": displacement.transition_id,
                "velocity": velocity.transition_id,
                "source_plan": source_identifier,
                "target_plan": target_identifier,
            }
        )

    def apply(self, state: SecondOrderWaveState, /) -> SeismicAMRTransitionResult:
        if (
            not isinstance(state, SecondOrderWaveState)
            or state.plan_id != self.source_plan_id
        ):
            raise ValueError("Seismic AMR state belongs to a different source plan.")
        displacement = self.displacement.apply(
            self.displacement.transfer.primal_operator.source.flatten(state.displacement)
        )
        velocity = self.velocity.apply(
            self.velocity.transfer.primal_operator.source.flatten(
                state.velocity_half_step
            )
        )
        target_displacement = self.displacement.transfer.primal_operator.target.unflatten(
            displacement.values
        )
        target_velocity = self.velocity.transfer.primal_operator.target.unflatten(
            velocity.values
        )
        transferred = SecondOrderWaveState(
            target_displacement,
            target_velocity,
            state.step_index,
            self.target_plan_id,
        )
        successful = displacement.successful & velocity.successful
        return SeismicAMRTransitionResult(
            transferred,
            displacement.conservation_residual,
            velocity.conservation_residual,
            successful,
            jnp.asarray(False),
        )


__all__ = [
    "SecondOrderWaveState",
    "SeismicAMRTransition",
    "SeismicAMRTransitionResult",
    "SpectralElementWavePlan",
]

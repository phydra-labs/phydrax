#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....nonlinear import (
    implicit_root_result,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._faults import RateStateFaultLaw, RateStateFaultState


class MaxwellViscoelasticState(StrictModule):
    viscous_strain: Array


class MaxwellViscoelasticRelaxation(StrictModule):
    shear_modulus_Pa: Array
    viscosity_Pa_s: Array

    def __init__(self, shear_modulus_Pa: ArrayLike, viscosity_Pa_s: ArrayLike, /):
        shear, viscosity = jnp.broadcast_arrays(
            jnp.asarray(shear_modulus_Pa), jnp.asarray(viscosity_Pa_s)
        )
        self.shear_modulus_Pa = eqx.error_if(
            shear,
            jnp.any(~jnp.isfinite(shear))
            | jnp.any(shear <= 0)
            | jnp.any(~jnp.isfinite(viscosity))
            | jnp.any(viscosity <= 0),
            "Maxwell modulus and viscosity must be positive and finite.",
        )
        self.viscosity_Pa_s = viscosity

    def initialize(self, shape: tuple[int, ...]) -> MaxwellViscoelasticState:
        return MaxwellViscoelasticState(jnp.zeros(shape))

    def step(
        self,
        state: MaxwellViscoelasticState,
        total_strain: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> tuple[MaxwellViscoelasticState, Array]:
        strain, dt = jnp.asarray(total_strain), jnp.asarray(dt_s)
        if strain.shape != state.viscous_strain.shape or dt.shape != ():
            raise ValueError("Maxwell strain state or timestep shape is invalid.")
        relaxation = self.viscosity_Pa_s / self.shear_modulus_Pa
        decay = jnp.exp(-dt / relaxation)
        viscous = decay * state.viscous_strain + (1.0 - decay) * strain
        stress = 2.0 * self.shear_modulus_Pa * (strain - viscous)
        return MaxwellViscoelasticState(viscous), stress


class EarthquakeCycleState(StrictModule):
    slip_m: Array
    shear_stress_Pa: Array
    fault_state: RateStateFaultState
    previous_slip_rate_m_s: Array
    time_s: Array
    plan_id: str = eqx.field(static=True)


class EarthquakeCycleStepResult(StrictModule):
    state: EarthquakeCycleState
    slip_rate_m_s: Array
    friction_traction_Pa: Array
    residual: Array
    successful: Array
    derivative_available: Array


class EarthquakeCyclePlan(StrictModule, NonTrainableState):
    """Quasi-dynamic rate-state fault network with elastic stress transfer."""

    stiffness_Pa_m: Array
    loading_rate_Pa_s: Array
    effective_normal_stress_Pa: Array
    radiation_damping_Pa_s_m: Array
    friction: RateStateFaultLaw
    termination: NonlinearTermination
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness_Pa_m: ArrayLike,
        loading_rate_Pa_s: ArrayLike,
        effective_normal_stress_Pa: ArrayLike,
        radiation_damping_Pa_s_m: ArrayLike,
        friction: RateStateFaultLaw,
        /,
        *,
        termination: NonlinearTermination | None = None,
    ):
        stiffness = np.asarray(stiffness_Pa_m, dtype=float)
        count = stiffness.shape[0] if stiffness.ndim == 2 else 0
        loading = np.broadcast_to(np.asarray(loading_rate_Pa_s, dtype=float), (count,))
        normal = np.broadcast_to(
            np.asarray(effective_normal_stress_Pa, dtype=float), (count,)
        )
        damping = np.broadcast_to(
            np.asarray(radiation_damping_Pa_s_m, dtype=float), (count,)
        )
        if (
            count == 0
            or stiffness.shape != (count, count)
            or np.any(~np.isfinite(stiffness))
            or not np.allclose(stiffness, stiffness.T)
            or np.min(np.linalg.eigvalsh(stiffness)) < -1e-10
            or np.any(~np.isfinite(loading))
            or np.any(~np.isfinite(normal))
            or np.any(normal <= 0)
            or np.any(~np.isfinite(damping))
            or np.any(damping < 0)
            or not isinstance(friction, RateStateFaultLaw)
        ):
            raise ValueError(
                "Earthquake-cycle stiffness/loading/stress/damping are invalid."
            )
        self.stiffness_Pa_m, self.loading_rate_Pa_s = (
            jnp.asarray(stiffness),
            jnp.asarray(loading),
        )
        self.effective_normal_stress_Pa, self.radiation_damping_Pa_s_m = (
            jnp.asarray(normal),
            jnp.asarray(damping),
        )
        self.friction = friction
        self.termination = (
            NonlinearTermination(
                absolute_residual=1e-7,
                relative_residual=1e-9,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=80,
            )
            if termination is None
            else termination
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-dynamic-earthquake-cycle",
                "stiffness_Pa_m": stiffness,
                "loading_rate_Pa_s": loading,
                "effective_normal_stress_Pa": normal,
                "radiation_damping_Pa_s_m": damping,
                "friction": {
                    "reference_friction": friction.reference_friction,
                    "direct_effect": friction.direct_effect,
                    "evolution_effect": friction.evolution_effect,
                    "critical_slip_distance_m": friction.critical_slip_distance_m,
                    "reference_velocity_m_s": friction.reference_velocity_m_s,
                    "regularization_velocity_m_s": friction.regularization_velocity_m_s,
                },
            }
        )

    def initialize(
        self,
        shear_stress_Pa: ArrayLike,
        /,
        *,
        slip_m: ArrayLike = 0.0,
        slip_rate_m_s: ArrayLike = 1e-12,
    ) -> EarthquakeCycleState:
        count = self.stiffness_Pa_m.shape[0]
        stress = jnp.broadcast_to(jnp.asarray(shear_stress_Pa), (count,))
        slip = jnp.broadcast_to(jnp.asarray(slip_m), (count,))
        velocity = jnp.broadcast_to(jnp.asarray(slip_rate_m_s), (count,))
        stress = eqx.error_if(
            stress,
            jnp.any(~jnp.isfinite(stress))
            | jnp.any(~jnp.isfinite(slip))
            | jnp.any(~jnp.isfinite(velocity)),
            "Earthquake-cycle initial stress, slip, and rate must be finite.",
        )
        state = self.friction.initialize()
        state = RateStateFaultState(
            jnp.broadcast_to(state.state_time_s, (count,)),
            jnp.broadcast_to(state.accumulated_slip_m, (count,)),
        )
        return EarthquakeCycleState(
            slip, stress, state, velocity, jnp.asarray(0.0), self.plan_id
        )

    def step(
        self,
        previous: EarthquakeCycleState,
        dt_s: ArrayLike,
        /,
    ) -> EarthquakeCycleStepResult:
        if (
            not isinstance(previous, EarthquakeCycleState)
            or previous.plan_id != self.plan_id
        ):
            raise ValueError("Earthquake-cycle state belongs to a different plan.")
        dt = jnp.asarray(dt_s)
        if dt.shape != ():
            raise ValueError("Earthquake-cycle timestep must be scalar.")
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0),
            "Earthquake-cycle timestep must be positive.",
        )

        def residual(velocity, args):
            old, step = args
            slip_increment = step * velocity
            stress = (
                old.shear_stress_Pa
                + step * self.loading_rate_Pa_s
                - self.stiffness_Pa_m @ slip_increment
            )
            friction = self.friction.step(
                old.fault_state,
                velocity,
                self.effective_normal_stress_Pa,
                jnp.zeros_like(velocity),
                step,
            )
            return (
                stress
                - friction.shear_traction_Pa
                - self.radiation_damping_Pa_s_m * velocity
            )

        problem = NonlinearSystemProblem(
            residual, problem_id="quasi-dynamic-earthquake-cycle"
        )
        root = implicit_root_result(
            problem,
            previous.previous_slip_rate_m_s,
            termination=self.termination,
            args=(previous, dt),
        )
        velocity = root.state
        slip_increment = dt * velocity
        stress = (
            previous.shear_stress_Pa
            + dt * self.loading_rate_Pa_s
            - self.stiffness_Pa_m @ slip_increment
        )
        friction = self.friction.step(
            previous.fault_state,
            velocity,
            self.effective_normal_stress_Pa,
            jnp.zeros_like(velocity),
            dt,
        )
        candidate = EarthquakeCycleState(
            previous.slip_m + slip_increment,
            stress,
            friction.state,
            velocity,
            previous.time_s + dt,
            self.plan_id,
        )
        successful = root.successful & jnp.all(jnp.isfinite(stress))
        state = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, previous
        )
        return EarthquakeCycleStepResult(
            state,
            velocity,
            friction.shear_traction_Pa,
            residual(velocity, (previous, dt)),
            successful,
            successful & friction.derivative_available,
        )


__all__ = [
    "EarthquakeCyclePlan",
    "EarthquakeCycleState",
    "EarthquakeCycleStepResult",
    "MaxwellViscoelasticRelaxation",
    "MaxwellViscoelasticState",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule


class PhaseFieldDamageMaterial(StrictModule):
    fracture_energy_J_m2: Array
    length_scale_m: Array
    residual_stiffness: Array

    def __init__(
        self,
        fracture_energy_J_m2: ArrayLike,
        length_scale_m: ArrayLike,
        residual_stiffness: ArrayLike = 1e-8,
        /,
    ):
        energy, length, residual = jnp.broadcast_arrays(
            jnp.asarray(fracture_energy_J_m2),
            jnp.asarray(length_scale_m),
            jnp.asarray(residual_stiffness),
        )
        invalid = (
            jnp.any(~jnp.isfinite(energy))
            | jnp.any(energy <= 0)
            | jnp.any(~jnp.isfinite(length))
            | jnp.any(length <= 0)
            | jnp.any(~jnp.isfinite(residual))
            | jnp.any((residual <= 0) | (residual >= 1))
        )
        self.fracture_energy_J_m2 = eqx.error_if(
            energy, invalid, "Phase-field damage parameters must be finite and physical."
        )
        self.length_scale_m, self.residual_stiffness = length, residual

    def degradation(self, damage: ArrayLike, /) -> Array:
        value = jnp.asarray(damage)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any((value < 0) | (value > 1)),
            "Damage must lie in [0,1].",
        )
        return (1.0 - value) ** 2 + self.residual_stiffness

    def local_driving_residual(
        self,
        damage: ArrayLike,
        tensile_energy_J_m3: ArrayLike,
        history_energy_J_m3: ArrayLike,
        laplacian_damage_m2_inverse: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        damage_, energy, history, laplacian = jnp.broadcast_arrays(
            jnp.asarray(damage),
            jnp.asarray(tensile_energy_J_m3),
            jnp.asarray(history_energy_J_m3),
            jnp.asarray(laplacian_damage_m2_inverse),
        )
        damage_ = eqx.error_if(
            damage_,
            jnp.any(~jnp.isfinite(damage_))
            | jnp.any((damage_ < 0) | (damage_ > 1))
            | jnp.any(~jnp.isfinite(energy))
            | jnp.any(energy < 0)
            | jnp.any(~jnp.isfinite(history))
            | jnp.any(history < 0)
            | jnp.any(~jnp.isfinite(laplacian)),
            "Phase-field damage, energy history, and Laplacian must be finite and physical.",
        )
        history_next = jnp.maximum(history, energy)
        residual = (
            self.fracture_energy_J_m2
            * (damage_ / self.length_scale_m - self.length_scale_m * laplacian)
            - 2.0 * (1.0 - damage_) * history_next
        )
        return residual, history_next


class RateStateFaultState(StrictModule):
    state_time_s: Array
    accumulated_slip_m: Array


class RateStateFaultResult(StrictModule):
    shear_traction_Pa: Array
    friction_coefficient: Array
    state: RateStateFaultState
    effective_normal_stress_Pa: Array
    dissipated_power_W_m2: Array
    derivative_available: Array


class RateStateFaultLaw(StrictModule):
    reference_friction: Array
    direct_effect: Array
    evolution_effect: Array
    critical_slip_distance_m: Array
    reference_velocity_m_s: Array
    regularization_velocity_m_s: Array

    def __init__(
        self,
        reference_friction: ArrayLike,
        direct_effect: ArrayLike,
        evolution_effect: ArrayLike,
        critical_slip_distance_m: ArrayLike,
        reference_velocity_m_s: ArrayLike,
        /,
        *,
        regularization_velocity_m_s: ArrayLike = 1e-12,
    ):
        values = jnp.broadcast_arrays(
            *(
                jnp.asarray(value)
                for value in (
                    reference_friction,
                    direct_effect,
                    evolution_effect,
                    critical_slip_distance_m,
                    reference_velocity_m_s,
                    regularization_velocity_m_s,
                )
            )
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | jnp.any(values[0] < 0)
            | jnp.any(values[1] < 0)
            | jnp.any(values[2] < 0)
            | jnp.any(values[3] <= 0)
            | jnp.any(values[4] <= 0)
            | jnp.any(values[5] <= 0)
        )
        self.reference_friction = eqx.error_if(
            values[0], invalid, "Rate-state parameters must be finite and physical."
        )
        (
            self.direct_effect,
            self.evolution_effect,
            self.critical_slip_distance_m,
            self.reference_velocity_m_s,
            self.regularization_velocity_m_s,
        ) = values[1:]

    def initialize(self) -> RateStateFaultState:
        return RateStateFaultState(
            self.critical_slip_distance_m / self.reference_velocity_m_s,
            jnp.zeros_like(self.reference_velocity_m_s),
        )

    def step(
        self,
        state: RateStateFaultState,
        slip_rate_m_s: ArrayLike,
        normal_compression_Pa: ArrayLike,
        pore_pressure_Pa: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> RateStateFaultResult:
        if not isinstance(state, RateStateFaultState):
            raise TypeError("Rate-state update requires RateStateFaultState.")
        velocity, normal, pressure, dt = jnp.broadcast_arrays(
            jnp.asarray(slip_rate_m_s),
            jnp.asarray(normal_compression_Pa),
            jnp.asarray(pore_pressure_Pa),
            jnp.asarray(dt_s),
        )
        speed = jnp.sqrt(velocity**2 + self.regularization_velocity_m_s**2)
        effective = normal - pressure
        velocity = eqx.error_if(
            velocity,
            jnp.any(~jnp.isfinite(velocity))
            | jnp.any(~jnp.isfinite(normal))
            | jnp.any(~jnp.isfinite(pressure))
            | jnp.any(effective < 0)
            | jnp.any(~jnp.isfinite(dt))
            | jnp.any(dt <= 0),
            "Fault velocity/effective stress/timestep must be finite and physical.",
        )
        velocity = eqx.error_if(
            velocity,
            jnp.any(~jnp.isfinite(state.state_time_s))
            | jnp.any(state.state_time_s <= 0)
            | jnp.any(~jnp.isfinite(state.accumulated_slip_m)),
            "Rate-state history must be finite with positive state time.",
        )
        decay = jnp.exp(-speed * dt / self.critical_slip_distance_m)
        steady = self.critical_slip_distance_m / speed
        theta = decay * state.state_time_s + (1.0 - decay) * steady
        friction = (
            self.reference_friction
            + self.direct_effect * jnp.log(speed / self.reference_velocity_m_s)
            + self.evolution_effect
            * jnp.log(theta * self.reference_velocity_m_s / self.critical_slip_distance_m)
        )
        traction = friction * effective * jnp.sign(velocity)
        power = traction * velocity
        return RateStateFaultResult(
            traction,
            friction,
            RateStateFaultState(theta, state.accumulated_slip_m + dt * velocity),
            effective,
            power,
            jnp.all(jnp.abs(velocity) > 10 * self.regularization_velocity_m_s),
        )


class CoulombContactResult(StrictModule):
    normal_traction_Pa: Array
    shear_traction_Pa: Array
    slip_increment_m: Array
    active_contact: Array
    sliding: Array
    derivative_available: Array


class CoulombContactLaw(StrictModule):
    normal_penalty_Pa_m: Array
    tangential_penalty_Pa_m: Array
    friction_coefficient: Array

    def __init__(
        self,
        normal_penalty_Pa_m: ArrayLike,
        tangential_penalty_Pa_m: ArrayLike,
        friction_coefficient: ArrayLike,
        /,
    ):
        normal, tangent, friction = jnp.broadcast_arrays(
            jnp.asarray(normal_penalty_Pa_m),
            jnp.asarray(tangential_penalty_Pa_m),
            jnp.asarray(friction_coefficient),
        )
        invalid = (
            jnp.any(~jnp.isfinite(normal))
            | jnp.any(normal <= 0)
            | jnp.any(~jnp.isfinite(tangent))
            | jnp.any(tangent <= 0)
            | jnp.any(~jnp.isfinite(friction))
            | jnp.any(friction < 0)
        )
        self.normal_penalty_Pa_m = eqx.error_if(
            normal, invalid, "Coulomb contact parameters must be finite and physical."
        )
        self.tangential_penalty_Pa_m, self.friction_coefficient = tangent, friction

    def evaluate(
        self,
        normal_gap_m: ArrayLike,
        tangential_trial_displacement_m: ArrayLike,
        /,
    ) -> CoulombContactResult:
        gap = jnp.asarray(normal_gap_m)
        tangent = jnp.asarray(tangential_trial_displacement_m)
        if tangent.ndim == 0 or tangent.shape[-1] not in (1, 2):
            raise ValueError(
                "Coulomb tangential displacement must have one or two components."
            )
        gap = eqx.error_if(
            gap,
            jnp.any(~jnp.isfinite(gap)) | jnp.any(~jnp.isfinite(tangent)),
            "Coulomb gap and tangential trial displacement must be finite.",
        )
        normal = self.normal_penalty_Pa_m * jnp.maximum(-gap, 0.0)
        trial = self.tangential_penalty_Pa_m[..., None] * tangent
        magnitude = jnp.sqrt(jnp.sum(trial**2, axis=-1))
        limit = self.friction_coefficient * normal
        scale = jnp.minimum(1.0, limit / jnp.where(magnitude > 0, magnitude, 1.0))
        traction = scale[..., None] * trial
        slip = tangent - traction / self.tangential_penalty_Pa_m[..., None]
        contact = gap < 0
        sliding = contact & (magnitude > limit)
        return CoulombContactResult(
            normal,
            traction,
            slip,
            contact,
            sliding,
            jnp.all((jnp.abs(gap) > 1e-10) & (jnp.abs(magnitude - limit) > 1e-10)),
        )


__all__ = [
    "CoulombContactLaw",
    "CoulombContactResult",
    "PhaseFieldDamageMaterial",
    "RateStateFaultLaw",
    "RateStateFaultResult",
    "RateStateFaultState",
]

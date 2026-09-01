#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization.contact._kinematics import ContactKinematicsBatch
from ._closure import (
    AbstractTangentialContactLaw,
    ContactClosureCapability,
    NormalContactResponse,
    TangentialContactResponse,
)
from ._materials import ContactPairParameters
from ._route_state import ContactRouteState


def _regularized_speed(value, regularization, /):
    squared = jnp.sum(value * value, axis=-1)
    speed = jnp.sqrt(squared + regularization * regularization)
    return speed, squared


def _smooth_slip_potential(speed, epsilon, /):
    interior = (
        epsilon / 3.0
        + speed * speed / epsilon
        - speed * speed * speed / (3.0 * epsilon * epsilon)
    )
    return jnp.where(speed < epsilon, interior, speed)


def _smooth_slip_first_derivative(speed, epsilon, /):
    interior = 2.0 * speed / epsilon - speed * speed / (epsilon * epsilon)
    return jnp.where(speed < epsilon, interior, 1.0)


class RegularizedCoulombContactLaw(AbstractTangentialContactLaw):
    velocity_threshold: float = eqx.field(static=True)
    stribeck_velocity: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_threshold: float,
        /,
        *,
        stribeck_velocity: float | None = None,
    ):
        epsilon = float(velocity_threshold)
        stribeck = (
            10.0 * epsilon if stribeck_velocity is None else float(stribeck_velocity)
        )
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("velocity_threshold must be finite and positive.")
        if not np.isfinite(stribeck) or stribeck <= 0.0:
            raise ValueError("stribeck_velocity must be finite and positive.")
        self.velocity_threshold = epsilon
        self.stribeck_velocity = stribeck
        self._law_id = canonical_fingerprint(
            {
                "kind": "regularized-coulomb-contact-law",
                "velocity_threshold": epsilon.hex(),
                "stribeck_velocity": stribeck.hex(),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.POTENTIAL
            | ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.STATEFUL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        state: ContactRouteState,
        /,
    ) -> TangentialContactResponse:
        del state
        dtype = kinematics.gap.dtype
        epsilon = jnp.asarray(self.velocity_threshold, dtype=dtype)
        regularization = jnp.finfo(dtype).eps * epsilon
        velocity = kinematics.tangential_velocity
        speed, _ = _regularized_speed(velocity, regularization)
        stribeck = jnp.asarray(self.stribeck_velocity, dtype=dtype)
        static_mu = parameters.static_friction.astype(dtype)
        dynamic_mu = parameters.dynamic_friction.astype(dtype)
        coefficient = dynamic_mu + (static_mu - dynamic_mu) * jnp.exp(
            -((speed / stribeck) ** 2)
        )
        lagged_normal = jax.lax.stop_gradient(jnp.maximum(normal.traction, 0.0))
        potential = coefficient * lagged_normal * _smooth_slip_potential(speed, epsilon)
        derivative = _smooth_slip_first_derivative(speed, epsilon)
        direction = velocity / speed[..., None]
        traction = -(coefficient * lagged_normal * derivative)[..., None] * direction
        active = kinematics.valid & normal.active
        traction = jnp.where(active[..., None], traction, 0.0)
        potential = jnp.where(active, potential, 0.0)
        dissipated = -jnp.sum(traction * velocity, axis=-1)
        traction_norm = jnp.sqrt(jnp.sum(traction * traction, axis=-1))
        cone_limit = static_mu * lagged_normal
        cone_defect = jnp.maximum(traction_norm - cone_limit, 0.0)
        stick = active & (speed < epsilon) & (traction_norm < cone_limit)
        slip = active & ~stick
        finite = (
            jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(dissipated))
            & jnp.all(jnp.isfinite(cone_defect))
            & jnp.all(dissipated >= -64.0 * jnp.finfo(dtype).eps)
        )
        return TangentialContactResponse(
            traction,
            potential,
            jnp.maximum(dissipated, 0.0),
            stick,
            slip,
            cone_defect,
            finite,
        )


class AnisotropicCoulombContactLaw(AbstractTangentialContactLaw):
    velocity_threshold: float = eqx.field(static=True)
    tangent_scale: tuple[float, ...] = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_threshold: float,
        tangent_scale: tuple[float, ...],
        /,
    ):
        epsilon = float(velocity_threshold)
        scales = tuple(float(value) for value in tangent_scale)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("velocity_threshold must be finite and positive.")
        if len(scales) not in (1, 2) or any(
            not np.isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError(
                "Anisotropic tangent scales must be one or two positive values."
            )
        self.velocity_threshold = epsilon
        self.tangent_scale = scales
        self._law_id = canonical_fingerprint(
            {
                "kind": "anisotropic-coulomb-contact-law",
                "velocity_threshold": epsilon.hex(),
                "tangent_scale": scales,
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.POTENTIAL
            | ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.STATEFUL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        state: ContactRouteState,
        /,
    ) -> TangentialContactResponse:
        del state
        velocity = kinematics.tangential_velocity
        if velocity.shape[-1] != len(self.tangent_scale):
            raise ValueError(
                "Anisotropic friction scale does not match tangent dimension."
            )
        dtype = velocity.dtype
        scale = jnp.asarray(self.tangent_scale, dtype=dtype)
        transformed = velocity / scale
        epsilon = jnp.asarray(self.velocity_threshold, dtype=dtype)
        regularization = jnp.finfo(dtype).eps * epsilon
        speed, _ = _regularized_speed(transformed, regularization)
        normal_force = jax.lax.stop_gradient(jnp.maximum(normal.traction, 0.0))
        coefficient = parameters.dynamic_friction.astype(dtype)
        potential = coefficient * normal_force * _smooth_slip_potential(speed, epsilon)
        derivative = _smooth_slip_first_derivative(speed, epsilon)
        traction = (
            -(coefficient * normal_force * derivative / speed)[..., None]
            * velocity
            / (scale * scale)
        )
        active = kinematics.valid & normal.active
        traction = jnp.where(active[..., None], traction, 0.0)
        potential = jnp.where(active, potential, 0.0)
        dissipated = -jnp.sum(traction * velocity, axis=-1)
        dual_norm = jnp.sqrt(jnp.sum((traction * scale) ** 2, axis=-1))
        cone_limit = parameters.static_friction.astype(dtype) * normal_force
        cone_defect = jnp.maximum(dual_norm - cone_limit, 0.0)
        stick = active & (speed < epsilon) & (dual_norm < cone_limit)
        slip = active & ~stick
        finite = (
            jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(dissipated))
            & jnp.all(dissipated >= -64.0 * jnp.finfo(dtype).eps)
        )
        return TangentialContactResponse(
            traction,
            potential,
            jnp.maximum(dissipated, 0.0),
            stick,
            slip,
            cone_defect,
            finite,
        )


class RateStateFrictionContactLaw(AbstractTangentialContactLaw):
    reference_friction: float = eqx.field(static=True)
    direct_effect: float = eqx.field(static=True)
    evolution_effect: float = eqx.field(static=True)
    reference_velocity: float = eqx.field(static=True)
    critical_slip_distance: float = eqx.field(static=True)
    velocity_threshold: float = eqx.field(static=True)
    _law_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_friction: float,
        direct_effect: float,
        evolution_effect: float,
        reference_velocity: float,
        critical_slip_distance: float,
        velocity_threshold: float,
    ):
        values = tuple(
            float(value)
            for value in (
                reference_friction,
                direct_effect,
                evolution_effect,
                reference_velocity,
                critical_slip_distance,
                velocity_threshold,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Rate-state friction parameters must be finite and positive."
            )
        (
            self.reference_friction,
            self.direct_effect,
            self.evolution_effect,
            self.reference_velocity,
            self.critical_slip_distance,
            self.velocity_threshold,
        ) = values
        self._law_id = canonical_fingerprint(
            {
                "kind": "rate-state-friction-contact-law",
                "parameters": tuple(value.hex() for value in values),
            }
        )

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.STATEFUL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        state: ContactRouteState,
        /,
    ) -> TangentialContactResponse:
        del parameters
        velocity = kinematics.tangential_velocity
        dtype = velocity.dtype
        epsilon = jnp.asarray(self.velocity_threshold, dtype=dtype)
        speed, _ = _regularized_speed(velocity, epsilon)
        reference_velocity = jnp.asarray(self.reference_velocity, dtype=dtype)
        critical = jnp.asarray(self.critical_slip_distance, dtype=dtype)
        theta = jnp.maximum(state.rate_state, jnp.finfo(dtype).eps)
        coefficient = (
            self.reference_friction
            + self.direct_effect * jnp.log(speed / reference_velocity)
            + self.evolution_effect * jnp.log(theta * reference_velocity / critical)
        )
        coefficient = jnp.maximum(coefficient, 0.0)
        normal_force = jnp.maximum(normal.traction, 0.0)
        direction = velocity / speed[..., None]
        traction = -(coefficient * normal_force)[..., None] * direction
        active = kinematics.valid & normal.active
        traction = jnp.where(active[..., None], traction, 0.0)
        dissipated = -jnp.sum(traction * velocity, axis=-1)
        cone_limit = coefficient * normal_force
        traction_norm = jnp.sqrt(jnp.sum(traction * traction, axis=-1))
        cone_defect = jnp.maximum(traction_norm - cone_limit, 0.0)
        zero = jnp.zeros_like(speed)
        finite = (
            jnp.all(jnp.isfinite(traction))
            & jnp.all(jnp.isfinite(dissipated))
            & jnp.all(dissipated >= -64.0 * jnp.finfo(dtype).eps)
        )
        return TangentialContactResponse(
            traction,
            zero,
            jnp.maximum(dissipated, 0.0),
            active & (speed <= epsilon),
            active & (speed > epsilon),
            cone_defect,
            finite,
        )


__all__ = [
    "AnisotropicCoulombContactLaw",
    "RateStateFrictionContactLaw",
    "RegularizedCoulombContactLaw",
]

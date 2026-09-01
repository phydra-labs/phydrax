#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import DiscretizationKey, DiscretizationRole, PreparationReport
from ..particle import ParticleDiscretization, ParticlePrecisionPolicy
from ._interfaces import (
    AbstractPreparedVortexDiffusion,
    AbstractPreparedVortexVelocity,
    AbstractVortexDiffusionPlan,
    AbstractVortexVelocityPlan,
    VortexDiffusionDiagnostics,
    VortexDiffusionEvaluation,
    VortexFieldRequest,
)
from ._particle import VortexParticleProperties, VortexParticleStateLayout


BackgroundVortexVelocity = Callable[[Array, Array, Any], ArrayLike]


class InviscidVortexDiffusionPlan(AbstractVortexDiffusionPlan):
    """Exactly zero molecular diffusion."""

    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Inviscid vortex diffusion requires dimension 2 or 3.")
        self.dimension = dimension_
        self.plan_id = canonical_fingerprint(
            {"kind": "inviscid-vortex-diffusion", "dimension": dimension_}
        )

    def prepare(
        self,
        /,
        *,
        capacity: int,
        dimension: int,
    ) -> PreparedInviscidVortexDiffusion:
        if int(dimension) != self.dimension:
            raise ValueError("Diffusion plan and particle dimensions differ.")
        return PreparedInviscidVortexDiffusion(self, int(capacity))


class PreparedInviscidVortexDiffusion(AbstractPreparedVortexDiffusion):
    plan: InviscidVortexDiffusionPlan
    dimension: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: InviscidVortexDiffusionPlan, capacity: int, /):
        capacity_ = int(capacity)
        if capacity_ <= 0:
            raise ValueError("Vortex diffusion capacity must be positive.")
        self.plan = plan
        self.dimension = plan.dimension
        self.capacity = capacity_
        self.backend_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-inviscid-vortex-diffusion",
                "plan": plan.plan_id,
                "capacity": capacity_,
            }
        )

    def evaluate(
        self,
        position: ArrayLike,
        strength: ArrayLike,
        volume: ArrayLike,
        viscosity: ArrayLike,
        /,
    ) -> VortexDiffusionEvaluation:
        positions = jnp.asarray(position)
        strengths = jnp.asarray(strength)
        volumes = jnp.asarray(volume)
        viscosity_ = jnp.asarray(viscosity, dtype=positions.dtype)
        expected_position = (self.capacity, self.dimension)
        expected_strength = (self.capacity,) if self.dimension == 2 else expected_position
        if positions.shape != expected_position or strengths.shape != expected_strength:
            raise ValueError("Inviscid vortex arrays do not match prepared capacity.")
        if volumes.shape != (self.capacity,):
            raise ValueError("Vortex volumes must have particle-capacity shape.")
        if viscosity_.shape != ():
            raise ValueError("Vortex viscosity must be scalar.")
        rate = jnp.zeros_like(strengths)
        total = jnp.sum(rate, axis=0)
        finite = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(strengths))
            & jnp.all(jnp.isfinite(volumes))
            & jnp.isfinite(viscosity_)
            & (viscosity_ == 0.0)
        )
        diagnostics = VortexDiffusionDiagnostics(
            jnp.asarray(self.capacity, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            total,
            finite,
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(True),
            finite,
            None,
        )
        return VortexDiffusionEvaluation(
            rate,
            finite,
            self.backend_id,
            canonical_fingerprint(
                {"kind": "inviscid-vortex-evaluation", "prepared": self.prepared_id}
            ),
            diagnostics,
        )


class VortexParticleMethodPlan(StrictModule, NonTrainableState):
    """Composition of one induced-velocity and one diffusion realization."""

    velocity: AbstractVortexVelocityPlan
    diffusion: AbstractVortexDiffusionPlan
    advective_cfl: float = eqx.field(static=True)
    diffusive_cfl: float = eqx.field(static=True)
    key: DiscretizationKey
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity: AbstractVortexVelocityPlan,
        /,
        *,
        diffusion: AbstractVortexDiffusionPlan | None = None,
        advective_cfl: float = 0.25,
        diffusive_cfl: float = 0.125,
        name: str = "vortex-particle-method",
    ):
        if not isinstance(velocity, AbstractVortexVelocityPlan):
            raise TypeError("velocity must be an AbstractVortexVelocityPlan.")
        diffusion_ = (
            InviscidVortexDiffusionPlan(velocity.dimension)
            if diffusion is None
            else diffusion
        )
        if not isinstance(diffusion_, AbstractVortexDiffusionPlan):
            raise TypeError("diffusion must be an AbstractVortexDiffusionPlan or None.")
        if diffusion_.dimension != velocity.dimension:
            raise ValueError("Velocity and diffusion dimensions differ.")
        advective = float(advective_cfl)
        diffusive = float(diffusive_cfl)
        if advective <= 0.0 or diffusive <= 0.0:
            raise ValueError("Vortex CFL coefficients must be positive.")
        key = DiscretizationKey(
            str(name),
            DiscretizationRole.RESIDUAL,
            domain_labels=("material_point", "vorticity"),
        )
        self.velocity = velocity
        self.diffusion = diffusion_
        self.advective_cfl = advective
        self.diffusive_cfl = diffusive
        self.key = key
        self.method_id = canonical_fingerprint(
            {
                "kind": "vortex-particle-method",
                "dimension": velocity.dimension,
                "velocity": velocity.plan_id,
                "diffusion": diffusion_.plan_id,
                "advective_cfl": advective,
                "diffusive_cfl": diffusive,
                "key": key.key_id,
            }
        )


class VortexParticleStepRestriction(StrictModule):
    advective: Array
    diffusive: Array
    selected: Array


class VortexParticleDiagnostics(StrictModule):
    total_strength: Array
    strength_rate_defect: Array
    linear_impulse: Array
    angular_impulse: Array
    maximum_speed: Array
    minimum_core_radius: Array
    velocity_successful: Array
    diffusion_successful: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class PreparedVortexParticleDynamics(StrictModule, NonTrainableState):
    """Fixed-population first-order vortex-particle drift in two or three dimensions."""

    particles: ParticleDiscretization
    properties: VortexParticleProperties
    velocity: AbstractPreparedVortexVelocity
    diffusion: AbstractPreparedVortexDiffusion
    method: VortexParticleMethodPlan
    viscosity: Array
    background_velocity: BackgroundVortexVelocity | None
    background_velocity_id: str | None = eqx.field(static=True)
    precision: ParticlePrecisionPolicy
    state_layout: VortexParticleStateLayout
    key: DiscretizationKey
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        properties: VortexParticleProperties,
        velocity: AbstractPreparedVortexVelocity,
        diffusion: AbstractPreparedVortexDiffusion,
        method: VortexParticleMethodPlan,
        viscosity: ArrayLike,
        /,
        *,
        precision: ParticlePrecisionPolicy | None = None,
        background_velocity: BackgroundVortexVelocity | None = None,
        background_velocity_id: str | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(properties, VortexParticleProperties):
            raise TypeError("properties must be VortexParticleProperties.")
        if not isinstance(velocity, AbstractPreparedVortexVelocity):
            raise TypeError("velocity must be a prepared vortex velocity.")
        if not isinstance(diffusion, AbstractPreparedVortexDiffusion):
            raise TypeError("diffusion must be a prepared vortex diffusion.")
        if not isinstance(method, VortexParticleMethodPlan):
            raise TypeError("method must be VortexParticleMethodPlan.")
        dimension = particles.ambient_dimension
        if (
            dimension not in (2, 3)
            or velocity.dimension != dimension
            or diffusion.dimension != dimension
        ):
            raise ValueError("Prepared vortex dimensions are incompatible.")
        if (
            velocity.source_capacity != particles.capacity
            or velocity.target_capacity != particles.capacity
        ):
            raise ValueError("Prepared velocity must bind the complete particle support.")
        if diffusion.capacity != particles.capacity:
            raise ValueError(
                "Prepared diffusion must bind the complete particle support."
            )
        properties.validate(
            particles.capacity, require_core_radius=True, require_volume=True
        )
        viscosity_ = jnp.asarray(viscosity, dtype=particles.safe_masses.dtype)
        if viscosity_.shape != ():
            raise ValueError("Vortex viscosity must be scalar.")
        viscosity_ = eqx.error_if(
            viscosity_,
            ~jnp.isfinite(viscosity_) | (viscosity_ < 0.0),
            "Vortex viscosity must be finite and nonnegative.",
        )
        if background_velocity is not None and not callable(background_velocity):
            raise TypeError("background_velocity must be callable or None.")
        if background_velocity is None and background_velocity_id is not None:
            raise ValueError("background_velocity_id requires a callback.")
        if background_velocity is not None and not background_velocity_id:
            raise ValueError("Background velocity requires a stable nonempty ID.")
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, ParticlePrecisionPolicy):
            raise TypeError("precision must be ParticlePrecisionPolicy or None.")
        layout = VortexParticleStateLayout(particles.capacity, dimension)
        preparation = PreparationReport(
            capabilities=particles.capabilities,
            diagnostics=(
                "fixed vortex-particle population",
                "circulation/vector strength remains independent of particle mass",
                "topology decisions are outside the smooth drift",
                "free-space and periodic velocity semantics remain backend-specific",
            ),
            resource_counts={
                "particle_capacity": particles.capacity,
                "ambient_dimension": dimension,
                "state_size": layout.state_size,
            },
        )
        self.particles = particles
        self.properties = properties
        self.velocity = velocity
        self.diffusion = diffusion
        self.method = method
        self.viscosity = viscosity_
        self.background_velocity = background_velocity
        self.background_velocity_id = background_velocity_id
        self.precision = precision_
        self.state_layout = layout
        self.key = method.key
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vortex-particle-dynamics",
                "particles": particles.prepared_id,
                "properties": properties.properties_id,
                "velocity": velocity.prepared_id,
                "diffusion": diffusion.prepared_id,
                "method": method.method_id,
                "background_velocity": background_velocity_id,
                "precision": precision_.policy_id,
            }
        )

    def initialize_state(self, position: ArrayLike, strength: ArrayLike, /) -> Array:
        state = self.state_layout.pack(position, strength)
        unpacked = self.state_layout.unpack(state)
        active = self.particles.active_mask
        position_ = jnp.where(active[:, None], unpacked.position, 0.0)
        strength_mask = active if self.state_layout.dimension == 2 else active[:, None]
        strength_ = jnp.where(strength_mask, unpacked.strength, 0.0)
        position_ = eqx.error_if(
            position_,
            jnp.any(jnp.where(active[:, None], ~jnp.isfinite(position_), False)),
            "Active vortex positions must be finite.",
        )
        strength_ = eqx.error_if(
            strength_,
            jnp.any(jnp.where(strength_mask, ~jnp.isfinite(strength_), False)),
            "Active vortex strengths must be finite.",
        )
        return self.state_layout.pack(position_, strength_)

    def _background(self, time: Array, position: Array, args: Any, /) -> Array:
        if self.background_velocity is None:
            return jnp.zeros_like(position)
        value = jnp.asarray(
            self.background_velocity(time, position, args), dtype=position.dtype
        )
        if value.shape != position.shape:
            raise ValueError("Background vortex velocity must match positions.")
        active = self.particles.active_mask[:, None]
        value = eqx.error_if(
            value,
            jnp.any(jnp.where(active, ~jnp.isfinite(value), False)),
            "Active background vortex velocity must be finite.",
        )
        return jnp.where(active, value, 0.0)

    def evaluate(self, time: ArrayLike, state: ArrayLike, args: Any = None, /):
        unpacked = self.state_layout.unpack(state)
        active = self.particles.active_mask
        position = jnp.where(active[:, None], unpacked.position, 0.0)
        strength_mask = active if self.state_layout.dimension == 2 else active[:, None]
        strength = jnp.where(strength_mask, unpacked.strength, 0.0)
        core = self.properties.safe_core_radius(active, dtype=position.dtype)
        volume = self.properties.safe_volume(active, dtype=position.dtype)
        request = VortexFieldRequest(
            velocity=True,
            velocity_gradient=self.state_layout.dimension == 3,
        )
        velocity_evaluation = self.velocity.evaluate(
            position,
            strength,
            core,
            request=request,
        )
        diffusion_evaluation = self.diffusion.evaluate(
            position,
            strength,
            volume,
            self.viscosity,
        )
        if velocity_evaluation.velocity is None:
            raise ValueError("Vortex velocity backend returned no velocity.")
        induced = jnp.where(active[:, None], velocity_evaluation.velocity, 0.0)
        total_velocity = induced + self._background(jnp.asarray(time), position, args)
        strength_rate = diffusion_evaluation.rate
        if self.state_layout.dimension == 3:
            gradient = velocity_evaluation.velocity_gradient
            if gradient is None:
                raise ValueError(
                    "Three-dimensional vortex dynamics require velocity gradients."
                )
            strength_rate = strength_rate + contract(
                "...ij,...j->...i", gradient, strength
            )
        strength_rate = jnp.where(strength_mask, strength_rate, 0.0)
        return (
            position,
            strength,
            total_velocity,
            strength_rate,
            velocity_evaluation,
            diffusion_evaluation,
        )

    def __call__(self, time: ArrayLike, state: ArrayLike, args: Any = None, /) -> Array:
        _, _, velocity, strength_rate, _, _ = self.evaluate(time, state, args)
        return self.state_layout.pack(velocity, strength_rate)

    def diagnostics(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> VortexParticleDiagnostics:
        (
            position,
            strength,
            velocity,
            strength_rate,
            velocity_evaluation,
            diffusion_evaluation,
        ) = self.evaluate(time, state, args)
        active = self.particles.active_mask
        if self.state_layout.dimension == 2:
            total_strength = jnp.sum(jnp.where(active, strength, 0.0))
            rate_defect = jnp.sum(jnp.where(active, strength_rate, 0.0))
            impulse = jnp.sum(
                jnp.where(
                    active[:, None],
                    strength[:, None]
                    * jnp.stack((-position[:, 1], position[:, 0]), axis=-1),
                    0.0,
                ),
                axis=0,
            )
            angular = 0.5 * jnp.sum(
                jnp.where(active, strength * jnp.sum(position * position, axis=-1), 0.0)
            )
        else:
            total_strength = jnp.sum(jnp.where(active[:, None], strength, 0.0), axis=0)
            rate_defect = jnp.sum(jnp.where(active[:, None], strength_rate, 0.0), axis=0)
            impulse = 0.5 * jnp.sum(
                jnp.where(active[:, None], jnp.cross(position, strength), 0.0), axis=0
            )
            angular = 0.5 * jnp.sum(
                jnp.where(
                    active[:, None],
                    jnp.cross(position, jnp.cross(position, strength)),
                    0.0,
                ),
                axis=0,
            )
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        core = self.properties.safe_core_radius(active, dtype=position.dtype)
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(strength))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(strength_rate))
        )
        return VortexParticleDiagnostics(
            total_strength,
            rate_defect,
            impulse,
            angular,
            jnp.max(jnp.where(active, speed, 0.0)),
            jnp.min(jnp.where(active, core, jnp.inf)),
            velocity_evaluation.successful,
            diffusion_evaluation.successful,
            finite,
            self.prepared_id,
        )

    def stable_step(
        self, time: ArrayLike, state: ArrayLike, args: Any = None, /
    ) -> VortexParticleStepRestriction:
        position, _, velocity, _, _, _ = self.evaluate(time, state, args)
        active = self.particles.active_mask
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        maximum_speed = jnp.max(jnp.where(active, speed, 0.0))
        core = self.properties.safe_core_radius(active, dtype=position.dtype)
        minimum_scale = jnp.min(jnp.where(active, core, jnp.inf))
        tiny = jnp.finfo(position.dtype).tiny
        advective = jnp.where(
            maximum_speed > 0.0,
            self.method.advective_cfl * minimum_scale / jnp.maximum(maximum_speed, tiny),
            jnp.asarray(jnp.inf, dtype=position.dtype),
        )
        diffusive = jnp.where(
            self.viscosity > 0.0,
            self.method.diffusive_cfl
            * minimum_scale**2
            / jnp.maximum(self.viscosity, tiny),
            jnp.asarray(jnp.inf, dtype=position.dtype),
        )
        return VortexParticleStepRestriction(
            advective, diffusive, jnp.minimum(advective, diffusive)
        )

    def linearize(self, time: Array, state: Array, args: Any = None, /):
        value, jvp = jax.linearize(lambda current: self(time, current, args), state)
        _, vjp = jax.vjp(lambda current: self(time, current, args), state)
        return value, jvp, vjp


__all__ = [
    "BackgroundVortexVelocity",
    "InviscidVortexDiffusionPlan",
    "PreparedInviscidVortexDiffusion",
    "PreparedVortexParticleDynamics",
    "VortexParticleDiagnostics",
    "VortexParticleMethodPlan",
    "VortexParticleStepRestriction",
]

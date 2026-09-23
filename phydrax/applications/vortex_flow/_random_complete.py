#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ...discretization.vortex._interfaces import AbstractPreparedVortexVelocity
from ...discretization.vortex._source import VortexSourceState, VortexTargetState


class RandomVortexEnsembleState(StrictModule):
    positions: Array
    strength: Array
    core_radius: Array
    volume: Array
    weights: Array
    active_mask: Array
    realization_ids: Array
    time: Array


class RandomVortexStepEvidence(StrictModule):
    mean_displacement: Array
    displacement_variance: Array
    absorbed_count: Array
    reflected_count: Array
    total_weight_before: Array
    total_weight_after: Array
    weak_moment_residual: Array
    antithetic: bool = eqx.field(static=True)
    finite: Array
    backend_successful: Array


class RandomVortexStepResult(StrictModule):
    state: RandomVortexEnsembleState
    evidence: RandomVortexStepEvidence
    successful: Array
    solver_id: str = eqx.field(static=True)


class RandomVortexSolverPlan(StrictModule, NonTrainableState):
    velocity: AbstractPreparedVortexVelocity
    viscosity: float = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    lower: Array | None
    upper: Array | None
    antithetic: bool = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity: AbstractPreparedVortexVelocity,
        viscosity: float,
        ensemble_size: int,
        /,
        *,
        boundary: str = "free",
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        antithetic: bool = True,
    ):
        if (
            not isinstance(velocity, AbstractPreparedVortexVelocity)
            or not isfinite(viscosity)
            or float(viscosity) <= 0.0
            or int(ensemble_size) <= 0
        ):
            raise ValueError(
                "Random-vortex velocity/viscosity/ensemble controls are invalid."
            )
        if boundary not in ("free", "periodic", "reflect", "absorb"):
            raise ValueError("Random-vortex boundary policy is unsupported.")
        if bool(antithetic) and int(ensemble_size) % 2:
            raise ValueError("Antithetic random-vortex ensembles require even size.")
        lower_ = None if lower is None else jnp.asarray(lower, dtype=jnp.float64)
        upper_ = None if upper is None else jnp.asarray(upper, dtype=jnp.float64)
        if boundary != "free":
            if (
                lower_ is None
                or upper_ is None
                or lower_.shape != (velocity.dimension,)
                or upper_.shape != lower_.shape
                or jnp.any(~jnp.isfinite(lower_))
                or jnp.any(~jnp.isfinite(upper_))
                or jnp.any(upper_ <= lower_)
            ):
                raise ValueError(
                    "Bounded random-vortex policies require increasing bounds."
                )
        self.velocity, self.viscosity, self.ensemble_size = (
            velocity,
            float(viscosity),
            int(ensemble_size),
        )
        self.boundary, self.lower, self.upper, self.antithetic = (
            boundary,
            lower_,
            upper_,
            bool(antithetic),
        )
        self.solver_id = canonical_fingerprint(
            {
                "kind": "random-vortex-solver",
                "velocity": velocity.prepared_id,
                "viscosity": self.viscosity,
                "ensemble_size": self.ensemble_size,
                "boundary": boundary,
                "lower": lower_,
                "upper": upper_,
                "antithetic": self.antithetic,
            }
        )

    def initialize(self, source: VortexSourceState, /) -> RandomVortexEnsembleState:
        if (
            source.dimension != self.velocity.dimension
            or source.capacity != self.velocity.source_capacity
            or source.core_radius is None
            or source.volume is None
        ):
            raise ValueError(
                "Random-vortex source must match the backend and provide core/volume."
            )
        positions = jnp.broadcast_to(
            source.positions,
            (self.ensemble_size,) + source.positions.shape,
        )
        strength = jnp.broadcast_to(
            source.strength,
            (self.ensemble_size,) + source.strength.shape,
        )
        core = jnp.broadcast_to(
            source.core_radius,
            (self.ensemble_size, source.capacity),
        )
        volume = jnp.broadcast_to(
            source.volume,
            (self.ensemble_size, source.capacity),
        )
        active = jnp.broadcast_to(
            source.active_mask,
            (self.ensemble_size, source.capacity),
        )
        weights = jnp.full(
            (self.ensemble_size, source.capacity),
            1.0 / self.ensemble_size,
            dtype=source.positions.dtype,
        )
        return RandomVortexEnsembleState(
            positions,
            strength,
            core,
            volume,
            weights,
            active,
            jnp.arange(self.ensemble_size, dtype=jnp.int64),
            jnp.asarray(0.0, dtype=source.positions.dtype),
        )

    def step(
        self,
        state: RandomVortexEnsembleState,
        key: Array,
        time_step: ArrayLike,
        /,
        *,
        forcing_rate: ArrayLike | None = None,
    ) -> RandomVortexStepResult:
        dt = jnp.asarray(time_step, dtype=state.positions.dtype)
        typed_key = jax.dtypes.issubdtype(
            key.dtype,
            jax.dtypes.prng_key,
        )
        if dt.shape != () or (not typed_key and key.shape != (2,)):
            raise ValueError("Random-vortex step requires scalar dt and a JAX key.")

        def velocity_one(position, strength, core, volume, active):
            source = VortexSourceState(
                position,
                strength,
                core_radius=core,
                volume=volume,
                active_mask=active,
            )
            target = VortexTargetState(
                position,
                source_indices=jnp.arange(
                    position.shape[0],
                    dtype=jnp.int32,
                ),
            )
            result = self.velocity.evaluate(source, target)
            if result.velocity is None:
                return jnp.zeros_like(position), jnp.asarray(False)
            return result.velocity, result.successful

        drift, backend_successful = jax.vmap(velocity_one)(
            state.positions,
            state.strength,
            state.core_radius,
            state.volume,
            state.active_mask,
        )
        half = self.ensemble_size // 2
        if self.antithetic:
            noise_half = jax.random.normal(
                key, (half,) + state.positions.shape[1:], dtype=state.positions.dtype
            )
            noise = jnp.concatenate((noise_half, -noise_half), axis=0)
        else:
            noise = jax.random.normal(
                key, state.positions.shape, dtype=state.positions.dtype
            )
        displacement = dt * drift + jnp.sqrt(2.0 * self.viscosity * dt) * noise
        candidate = state.positions + displacement
        absorbed = jnp.zeros(state.active_mask.shape, dtype=jnp.bool_)
        reflected = jnp.zeros(state.active_mask.shape, dtype=jnp.bool_)
        if self.boundary == "periodic":
            width = self.upper - self.lower
            candidate = self.lower + jnp.mod(candidate - self.lower, width)
        elif self.boundary == "reflect":
            width = self.upper - self.lower
            folded = jnp.mod(candidate - self.lower, 2.0 * width)
            reflected = jnp.any(
                (candidate < self.lower) | (candidate > self.upper), axis=-1
            )
            candidate = self.lower + jnp.where(
                folded <= width,
                folded,
                2.0 * width - folded,
            )
        elif self.boundary == "absorb":
            absorbed = jnp.any(
                (candidate < self.lower) | (candidate > self.upper), axis=-1
            )
        active = state.active_mask & ~absorbed
        forcing = (
            jnp.zeros_like(state.weights)
            if forcing_rate is None
            else jnp.asarray(forcing_rate, dtype=state.weights.dtype)
        )
        if forcing.shape != state.weights.shape:
            raise ValueError("Random-vortex forcing_rate must match ensemble weights.")
        weights = jnp.where(active, state.weights + dt * forcing, 0.0)
        total_before, total_after = jnp.sum(state.weights), jnp.sum(weights)
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(weights))
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        successful = finite & jnp.all(backend_successful)
        candidate_state = RandomVortexEnsembleState(
            candidate,
            state.strength,
            state.core_radius,
            state.volume,
            weights,
            active,
            state.realization_ids,
            state.time + dt,
        )
        next_state = tree_where(successful, candidate_state, state)
        evidence = RandomVortexStepEvidence(
            jnp.mean(displacement, axis=(0, 1)),
            jnp.var(displacement, axis=(0, 1)),
            jnp.sum(absorbed, dtype=jnp.int32),
            jnp.sum(reflected, dtype=jnp.int32),
            total_before,
            total_after,
            jnp.linalg.norm(jnp.mean(noise, axis=0)),
            self.antithetic,
            finite,
            backend_successful,
        )
        return RandomVortexStepResult(next_state, evidence, successful, self.solver_id)


__all__ = [
    "RandomVortexEnsembleState",
    "RandomVortexSolverPlan",
    "RandomVortexStepEvidence",
    "RandomVortexStepResult",
]

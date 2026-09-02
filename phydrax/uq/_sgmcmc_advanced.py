#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ._minibatch_posterior import MinibatchPosteriorProblem, MinibatchSource


class SGMCMCStepSchedule(StrictModule):
    """Absolute-update fixed or polynomial SG-MCMC step schedule."""

    initial: float = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    kind: Literal["constant", "polynomial"] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["constant", "polynomial"],
        initial: float,
        /,
        *,
        offset: float = 0.0,
        exponent: float = 0.0,
    ):
        rate = float(initial)
        offset_ = float(offset)
        exponent_ = float(exponent)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("SG-MCMC initial step must be finite and positive.")
        if kind == "constant":
            if offset_ != 0.0 or exponent_ != 0.0:
                raise ValueError("Constant schedules do not accept offset/exponent.")
        elif kind == "polynomial":
            if not math.isfinite(offset_) or offset_ <= 0.0:
                raise ValueError(
                    "Polynomial schedule offset must be finite and positive."
                )
            if not 0.5 < exponent_ <= 1.0:
                raise ValueError(
                    "Robbins-Monro polynomial exponent must satisfy 1/2 < exponent <= 1."
                )
        else:
            raise ValueError("Unknown SG-MCMC step schedule.")
        self.initial = rate
        self.offset = offset_
        self.exponent = exponent_
        self.kind = kind
        self.schedule_id = (
            f"constant:{rate:.17g}"
            if kind == "constant"
            else f"polynomial:{rate:.17g}:{offset_:.17g}:{exponent_:.17g}"
        )

    @classmethod
    def constant(cls, step_size: float, /) -> SGMCMCStepSchedule:
        return cls("constant", step_size)

    @classmethod
    def polynomial(
        cls,
        initial: float,
        offset: float,
        exponent: float,
        /,
    ) -> SGMCMCStepSchedule:
        return cls("polynomial", initial, offset=offset, exponent=exponent)

    def __call__(self, update: int | Array, /) -> Array:
        index = jnp.asarray(update, dtype=float)
        if self.kind == "constant":
            return jnp.asarray(self.initial, dtype=float)
        return self.initial * (self.offset + index) ** (-self.exponent)


class SGMCMCAdaptationConfig(StrictModule):
    """Finite pilot controls frozen before retained production draws."""

    num_pilot_steps: int = eqx.field(static=True)
    target_normalized_update_rms: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_pilot_steps: int,
        target_normalized_update_rms: float,
    ):
        steps = int(num_pilot_steps)
        target = float(target_normalized_update_rms)
        if steps <= 0 or not math.isfinite(target) or target <= 0.0:
            raise ValueError("Pilot steps and normalized RMS target must be positive.")
        self.num_pilot_steps = steps
        self.target_normalized_update_rms = target


class GradientNoiseCovarianceConfig(StrictModule):
    """Online fixed-layout gradient-noise covariance declaration."""

    kind: Literal["diagonal", "blocks", "diagonal_low_rank"] = eqx.field(static=True)
    phase: Literal["pilot", "all"] = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["diagonal", "blocks", "diagonal_low_rank"] = "diagonal",
        /,
        *,
        phase: Literal["pilot", "all"] = "pilot",
        rank: int = 0,
    ):
        if kind not in ("diagonal", "blocks", "diagonal_low_rank"):
            raise ValueError("Unknown gradient-noise covariance kind.")
        if phase not in ("pilot", "all"):
            raise ValueError("Gradient-noise phase must be 'pilot' or 'all'.")
        if kind == "diagonal_low_rank" and int(rank) <= 0:
            raise ValueError("Diagonal-low-rank covariance requires rank > 0.")
        if kind != "diagonal_low_rank" and int(rank) != 0:
            raise ValueError("rank is valid only for diagonal-low-rank covariance.")
        self.kind = kind
        self.phase = phase
        self.rank = int(rank)


class SGMCMCNoiseCovarianceState(StrictModule):
    """Welford online diagonal covariance with explicit frozen status."""

    mean: Array
    m2: Array
    count: Array
    frozen: Array
    kind: str = eqx.field(static=True)
    phase: str = eqx.field(static=True)

    @classmethod
    def initialize(
        cls,
        dimension: int,
        /,
        *,
        config: GradientNoiseCovarianceConfig,
        dtype: Any,
    ) -> SGMCMCNoiseCovarianceState:
        return cls(
            mean=jnp.zeros((dimension,), dtype=dtype),
            m2=jnp.zeros((dimension,), dtype=dtype),
            count=jnp.asarray(0, dtype=jnp.int32),
            frozen=jnp.asarray(False),
            kind=config.kind,
            phase=config.phase,
        )

    @property
    def diagonal(self) -> Array:
        return jnp.where(self.count > 1, self.m2 / (self.count - 1), 0.0)

    def update(self, gradient: ArrayLike, /) -> SGMCMCNoiseCovarianceState:
        value = jnp.asarray(gradient, dtype=self.mean.dtype)
        if value.shape != self.mean.shape:
            raise ValueError("Gradient-noise layout changed.")
        count = self.count + 1
        delta = value - self.mean
        mean = self.mean + delta / count.astype(value.dtype)
        m2 = self.m2 + delta * (value - mean)
        return jax.lax.cond(
            self.frozen,
            lambda _: self,
            lambda _: eqx.tree_at(
                lambda state: (state.mean, state.m2, state.count),
                self,
                (mean, m2, count),
            ),
            operand=None,
        )

    def freeze(self) -> SGMCMCNoiseCovarianceState:
        return eqx.tree_at(lambda state: state.frozen, self, jnp.asarray(True))


class RMSPropGeometryConfig(StrictModule):
    """pSGLD RMSProp geometry and honest correction declaration."""

    decay: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    correction: Literal["frozen", "smooth_position"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        decay: float = 0.99,
        regularization: float = 1e-8,
        correction: Literal["frozen", "smooth_position"] = "frozen",
    ):
        decay_ = float(decay)
        regularization_ = float(regularization)
        if not 0.0 <= decay_ < 1.0:
            raise ValueError("RMSProp decay must lie in [0, 1).")
        if not math.isfinite(regularization_) or regularization_ <= 0.0:
            raise ValueError("RMSProp regularization must be finite and positive.")
        if correction not in ("frozen", "smooth_position"):
            raise ValueError("Unknown pSGLD correction mode.")
        self.decay = decay_
        self.regularization = regularization_
        self.correction = correction


class SGHMCState(StrictModule):
    position: Array
    momentum: Array
    noise: SGMCMCNoiseCovarianceState


class PSGLDState(StrictModule):
    position: Array
    square_average: Array


class AdvancedSGMCMCResult(StrictModule):
    """Deterministic absolute-update SGHMC/pSGLD state and retained draws."""

    problem: MinibatchPosteriorProblem
    unconstrained_samples: PyTree[Array]
    samples: PyTree[Array]
    final_states: Any
    step_size_trace: Array
    root_key: Array
    final_update: int = eqx.field(static=True)
    algorithm: Literal["sghmc", "psgld"] = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    source_fingerprint: str = eqx.field(static=True)


def sample_sghmc(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    schedule: SGMCMCStepSchedule,
    friction: ArrayLike,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    steps_per_sample: int = 1,
    initial_position: PyTree[Any] | None = None,
    noise: GradientNoiseCovarianceConfig | None = None,
    continuation: AdvancedSGMCMCResult | None = None,
) -> AdvancedSGMCMCResult:
    """Run SGHMC with a certified nonnegative diagonal diffusion."""
    return _sample_advanced(
        problem,
        source,
        key=key,
        schedule=schedule,
        algorithm="sghmc",
        friction=jnp.asarray(friction, dtype=float),
        geometry=None,
        num_chains=num_chains,
        num_burnin=num_burnin,
        num_samples=num_samples,
        steps_per_sample=steps_per_sample,
        initial_position=initial_position,
        noise_config=(GradientNoiseCovarianceConfig() if noise is None else noise),
        continuation=continuation,
    )


def sample_psgld(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    schedule: SGMCMCStepSchedule,
    geometry: RMSPropGeometryConfig | None = None,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    steps_per_sample: int = 1,
    initial_position: PyTree[Any] | None = None,
    continuation: AdvancedSGMCMCResult | None = None,
) -> AdvancedSGMCMCResult:
    """Run pSGLD with history-based frozen/stopped RMSProp correction semantics."""
    geometry_ = RMSPropGeometryConfig() if geometry is None else geometry
    if geometry_.correction == "smooth_position":
        raise ValueError(
            "smooth_position correction requires an explicit position-geometry JVP; "
            "history-based RMSProp cannot supply it."
        )
    return _sample_advanced(
        problem,
        source,
        key=key,
        schedule=schedule,
        algorithm="psgld",
        friction=None,
        geometry=geometry_,
        num_chains=num_chains,
        num_burnin=num_burnin,
        num_samples=num_samples,
        steps_per_sample=steps_per_sample,
        initial_position=initial_position,
        noise_config=None,
        continuation=continuation,
    )


def _sample_advanced(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    schedule: SGMCMCStepSchedule,
    algorithm: Literal["sghmc", "psgld"],
    friction: Array | None,
    geometry: RMSPropGeometryConfig | None,
    num_chains: int,
    num_burnin: int,
    num_samples: int,
    steps_per_sample: int,
    initial_position: PyTree[Any] | None,
    noise_config: GradientNoiseCovarianceConfig | None,
    continuation: AdvancedSGMCMCResult | None,
) -> AdvancedSGMCMCResult:
    if not isinstance(problem, MinibatchPosteriorProblem):
        raise TypeError("problem must be MinibatchPosteriorProblem.")
    if not isinstance(schedule, SGMCMCStepSchedule):
        raise TypeError("schedule must be SGMCMCStepSchedule.")
    chains, burnin, draws, thinning = map(
        int, (num_chains, num_burnin, num_samples, steps_per_sample)
    )
    if chains <= 0 or burnin < 0 or draws <= 0 or thinning <= 0:
        raise ValueError(
            "Chain/draw/thinning counts must be positive; burnin nonnegative."
        )
    if (
        noise_config is not None
        and noise_config.phase == "all"
        and schedule.kind == "constant"
    ):
        raise ValueError("Online-all covariance requires a decreasing schedule.")
    reference = problem.initial_position if initial_position is None else initial_position
    flat_reference, unravel = ravel_pytree(reference)
    dimension = int(flat_reference.size)
    if dimension == 0 or not jnp.issubdtype(flat_reference.dtype, jnp.floating):
        raise TypeError("Advanced SG-MCMC requires a nonempty real floating PyTree.")
    if continuation is not None:
        if continuation.algorithm != algorithm:
            raise ValueError("Continuation algorithm changed.")
        if continuation.source_fingerprint != source.fingerprint:
            raise ValueError("Continuation minibatch source changed.")
        if continuation.schedule_id != schedule.schedule_id:
            raise ValueError("Continuation step schedule changed.")
        if not bool(jnp.array_equal(continuation.root_key, key)):
            raise ValueError("Continuation requires the original root key.")
        states = continuation.final_states
        start_update = continuation.final_update
    else:
        starts = jnp.broadcast_to(flat_reference, (chains, dimension))
        start_update = 0
        if algorithm == "sghmc":
            assert noise_config is not None
            states = tuple(
                SGHMCState(
                    starts[index],
                    jnp.zeros((dimension,), dtype=flat_reference.dtype),
                    SGMCMCNoiseCovarianceState.initialize(
                        dimension, config=noise_config, dtype=flat_reference.dtype
                    ),
                )
                for index in range(chains)
            )
        else:
            states = tuple(
                PSGLDState(
                    starts[index],
                    jnp.zeros((dimension,), dtype=flat_reference.dtype),
                )
                for index in range(chains)
            )
    if len(states) != chains:
        raise ValueError("Continuation chain count changed.")
    address = SampleAddress(
        "uq.sgmcmc", algorithm, target=source.fingerprint, role="transition"
    )
    total_updates = burnin + draws * thinning
    retained: list[Array] = []
    step_trace: list[Array] = []
    gradient_fn = jax.grad(problem.log_density_estimate)
    epoch_cache: dict[int, tuple[Any, ...]] = {}
    for local_update in range(total_updates):
        update = start_update + local_update
        epoch = update // source.batches_per_epoch
        batch_index = update % source.batches_per_epoch
        if epoch not in epoch_cache:
            epoch_cache[epoch] = tuple(source.epoch(epoch))
        batch = epoch_cache[epoch][batch_index]
        epsilon = schedule(update)
        step_trace.append(epsilon)
        next_states = []
        for chain, state in enumerate(states):
            transition_key = derive_key(key, address, chain, update)
            position_tree = unravel(state.position)
            gradient_tree = gradient_fn(position_tree, batch)
            gradient, _ = ravel_pytree(gradient_tree)
            if bool(jnp.any(~jnp.isfinite(gradient))):
                raise FloatingPointError("SG-MCMC gradient is nonfinite.")
            if algorithm == "sghmc":
                assert isinstance(state, SGHMCState)
                assert friction is not None
                friction_vector = jnp.broadcast_to(friction, (dimension,))
                noise_state = state.noise.update(gradient)
                if noise_state.phase == "pilot" and local_update >= burnin:
                    noise_state = noise_state.freeze()
                b_hat = 0.5 * epsilon * noise_state.diagonal
                diffusion = friction_vector - b_hat
                if bool(jnp.any(~jnp.isfinite(diffusion))) or bool(
                    jnp.any(diffusion < 0.0)
                ):
                    raise ValueError(
                        "SGHMC friction minus estimated gradient noise is not PSD."
                    )
                random = jr.normal(
                    transition_key, (dimension,), dtype=state.position.dtype
                )
                momentum = (
                    state.momentum
                    + epsilon * gradient
                    - epsilon * friction_vector * state.momentum
                    + jnp.sqrt(2.0 * epsilon * diffusion) * random
                )
                next_state = SGHMCState(
                    position=state.position + epsilon * momentum,
                    momentum=momentum,
                    noise=noise_state,
                )
            else:
                assert isinstance(state, PSGLDState)
                assert geometry is not None
                square_average = (
                    geometry.decay * state.square_average
                    + (1.0 - geometry.decay) * gradient**2
                )
                metric = 1.0 / (geometry.regularization + jnp.sqrt(square_average))
                random = jr.normal(
                    transition_key, (dimension,), dtype=state.position.dtype
                )
                position = (
                    state.position
                    + 0.5 * epsilon * metric * gradient
                    + jnp.sqrt(epsilon * metric) * random
                )
                next_state = PSGLDState(position=position, square_average=square_average)
            if bool(jnp.any(~jnp.isfinite(next_state.position))):
                raise FloatingPointError("SG-MCMC transition produced nonfinite state.")
            next_states.append(next_state)
        states = tuple(next_states)
        if local_update >= burnin and (local_update - burnin + 1) % thinning == 0:
            retained.append(jnp.stack(tuple(state.position for state in states)))
    flat_samples = jnp.stack(retained, axis=1)
    unconstrained = jax.vmap(jax.vmap(unravel))(flat_samples)
    constrained = problem.parameter_space.constrain(unconstrained)
    approximation = (
        "decreasing_step_stochastic_approximation"
        if schedule.kind == "polynomial"
        else "unadjusted_fixed_step"
        if algorithm == "sghmc"
        else "rmsprop_psgld_frozen_geometry_correction"
    )
    return AdvancedSGMCMCResult(
        problem=problem,
        unconstrained_samples=unconstrained,
        samples=constrained,
        final_states=states,
        step_size_trace=jnp.stack(step_trace),
        root_key=jnp.asarray(key),
        final_update=start_update + total_updates,
        algorithm=algorithm,
        approximation=approximation,
        schedule_id=schedule.schedule_id,
        source_fingerprint=source.fingerprint,
    )


__all__ = [
    "AdvancedSGMCMCResult",
    "GradientNoiseCovarianceConfig",
    "PSGLDState",
    "RMSPropGeometryConfig",
    "SGHMCState",
    "SGMCMCAdaptationConfig",
    "SGMCMCNoiseCovarianceState",
    "SGMCMCStepSchedule",
    "sample_psgld",
    "sample_sghmc",
]

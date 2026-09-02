#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic import (
    CompositeStochasticRealization,
    EmpiricalMeanField,
    MeanFieldSnapshot,
    StochasticTrajectory,
    WienerRealization,
)
from ..stochastic._trajectory import _TrajectoryRecord
from ._memory import _solution_valid, _time_grid, _wiener_increments


ParticleVectorField: TypeAlias = Callable[
    [Array, Array, MeanFieldSnapshot, Any], ArrayLike
]


class InteractingParticleProblem(StrictModule):
    """Finite weighted particle approximation of a McKean--Vlasov SDE.

    ``diffusion`` drives each particle with its own Wiener component, while
    ``common_diffusion`` drives the complete population with one shared Wiener path.
    Either source is optional. Both vector fields receive the current empirical law.
    """

    drift: ParticleVectorField
    diffusion: ParticleVectorField | None
    common_diffusion: ParticleVectorField | None
    initial_particles: Array
    weights: Array
    t0: Array
    t1: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    common_noise_shape: tuple[int, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    common_noise_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    mean_field_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: ParticleVectorField,
        initial_particles: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        diffusion: ParticleVectorField | None = None,
        noise_shape: Sequence[int] | None = None,
        noise_id: str | None = None,
        common_diffusion: ParticleVectorField | None = None,
        common_noise_shape: Sequence[int] | None = None,
        common_noise_id: str | None = None,
        weights: ArrayLike | None = None,
        args: Any = None,
        problem_id: str = "interacting-particle-problem",
        mean_field_id: str | None = None,
    ):
        if not callable(drift):
            raise TypeError("drift must be callable.")
        if diffusion is not None and not callable(diffusion):
            raise TypeError("diffusion must be callable or None.")
        if common_diffusion is not None and not callable(common_diffusion):
            raise TypeError("common_diffusion must be callable or None.")
        particles = jnp.asarray(initial_particles)
        if particles.ndim < 2:
            raise ValueError(
                "initial_particles must have particle axis followed by a state shape."
            )
        count = int(particles.shape[0])
        state_shape = tuple(int(size) for size in particles.shape[1:])
        if count < 2 or any(size <= 0 for size in state_shape):
            raise ValueError(
                "At least two particles with a positive state shape are required."
            )
        if weights is None:
            weight_values = jnp.full((count,), 1.0 / count, dtype=particles.real.dtype)
        else:
            raw_weights = jnp.asarray(weights, dtype=float)
            if raw_weights.shape != (count,):
                raise ValueError("weights must have exact shape (num_particles,).")
            if bool(jnp.any(~jnp.isfinite(raw_weights)) | jnp.any(raw_weights < 0.0)):
                raise ValueError("weights must be finite and nonnegative.")
            mass = jnp.sum(raw_weights)
            if not bool(mass > 0.0):
                raise ValueError("weights must have positive total mass.")
            weight_values = raw_weights / mass
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if (
            start.shape != ()
            or end.shape != ()
            or not bool(jnp.isfinite(start) & jnp.isfinite(end) & (end > start))
        ):
            raise ValueError("InteractingParticleProblem requires finite scalar t1 > t0.")
        resolved_noise_shape = _noise_contract(
            diffusion,
            noise_shape,
            noise_id,
            owner="idiosyncratic",
        )
        resolved_common_shape = _noise_contract(
            common_diffusion,
            common_noise_shape,
            common_noise_id,
            owner="common",
        )
        identifier = _identifier(problem_id, owner="problem_id")
        field_identifier = _identifier(
            f"{identifier}:empirical-law" if mean_field_id is None else mean_field_id,
            owner="mean_field_id",
        )
        snapshot = _mean_field_snapshot(
            start,
            particles,
            weight_values,
            state_shape=state_shape,
            mean_field_id=field_identifier,
        )
        drift_values = jax.vmap(
            lambda state: jnp.asarray(drift(start, state, snapshot, args))
        )(particles)
        if drift_values.shape != particles.shape:
            raise ValueError("drift must preserve each particle state shape.")
        if diffusion is not None:
            values = jax.vmap(
                lambda state: jnp.asarray(diffusion(start, state, snapshot, args))
            )(particles)
            expected = (count,) + state_shape + resolved_noise_shape
            if values.shape != expected:
                raise ValueError(
                    f"diffusion must stack to shape {expected}; got {values.shape}."
                )
        if common_diffusion is not None:
            values = jax.vmap(
                lambda state: jnp.asarray(common_diffusion(start, state, snapshot, args))
            )(particles)
            expected = (count,) + state_shape + resolved_common_shape
            if values.shape != expected:
                raise ValueError(
                    f"common_diffusion must stack to shape {expected}; got {values.shape}."
                )
        self.drift = drift
        self.diffusion = diffusion
        self.common_diffusion = common_diffusion
        self.initial_particles = particles
        self.weights = weight_values
        self.t0 = start
        self.t1 = end
        self.args = args
        self.state_shape = state_shape
        self.noise_shape = resolved_noise_shape
        self.common_noise_shape = resolved_common_shape
        self.num_particles = count
        self.noise_id = noise_id
        self.common_noise_id = common_noise_id
        self.problem_id = identifier
        self.mean_field_id = field_identifier

    @property
    def stochastic(self) -> bool:
        return self.diffusion is not None or self.common_diffusion is not None


class InteractingParticleSolution(StrictModule):
    """Time-resolved interacting population and empirical moment diagnostics."""

    times: Array
    particles: Array
    valid: Array
    weights: Array
    means: Array
    covariances: Array
    realization: WienerRealization | None
    common_realization: WienerRealization | None
    metadata: frozendict[str, Any]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    mean_field_id: str = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        particles: ArrayLike,
        valid: ArrayLike,
        weights: ArrayLike,
        means: ArrayLike,
        covariances: ArrayLike,
        realization: WienerRealization | None,
        common_realization: WienerRealization | None,
        sample_shape: Sequence[int],
        state_shape: Sequence[int],
        num_particles: int,
        mean_field_id: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        grid = jnp.asarray(times, dtype=float)
        values = jnp.asarray(particles)
        validity = jnp.asarray(valid, dtype=bool)
        weight_values = jnp.asarray(weights, dtype=float)
        mean_values = jnp.asarray(means)
        covariance_values = jnp.asarray(covariances)
        samples = tuple(int(size) for size in sample_shape)
        state = tuple(int(size) for size in state_shape)
        count = int(num_particles)
        num_times = int(grid.size)
        flat_state = prod(state)
        if values.shape != samples + (num_times, count) + state:
            raise ValueError(
                "particles do not align with declared sample/time/particle axes."
            )
        if validity.shape != samples + (num_times, count):
            raise ValueError("valid must align with sample/time/particle axes.")
        if weight_values.shape != (count,):
            raise ValueError("weights must have shape (num_particles,).")
        if mean_values.shape != samples + (num_times,) + state:
            raise ValueError("means do not align with sample/time/state axes.")
        if covariance_values.shape != samples + (num_times, flat_state, flat_state):
            raise ValueError("covariances do not align with flattened state axes.")
        for value in (realization, common_realization):
            if value is not None and not isinstance(value, WienerRealization):
                raise TypeError(
                    "Particle drivers must be WienerRealization objects or None."
                )
        self.times = grid
        self.particles = values
        self.valid = validity
        self.weights = weight_values
        self.means = mean_values
        self.covariances = covariance_values
        self.realization = realization
        self.common_realization = common_realization
        self.metadata = frozendict({} if metadata is None else metadata)
        self.sample_shape = samples
        self.state_shape = state
        self.num_particles = count
        self.mean_field_id = _identifier(mean_field_id, owner="mean_field_id")
        self.solver_name = "InteractingParticleEulerMaruyama"

    @property
    def successful(self) -> Array:
        axes = tuple(range(len(self.sample_shape), self.valid.ndim))
        return jnp.all(self.valid, axis=axes)

    def empirical_mean_field(
        self,
        system_index: Sequence[int] = (),
        /,
    ) -> EmpiricalMeanField:
        """Select one independently driven system as an empirical measure flow."""
        index = tuple(int(value) for value in system_index)
        if len(index) != len(self.sample_shape):
            raise ValueError("system_index must select every sample_shape axis.")
        if any(
            value < 0 or value >= size for value, size in zip(index, self.sample_shape)
        ):
            raise ValueError("system_index is outside sample_shape.")
        particles = self.particles[index] if index else self.particles
        valid = self.valid[index] if index else self.valid
        particle_major = jnp.moveaxis(particles, 1, 0)
        valid_major = jnp.moveaxis(valid, 1, 0)
        weight_history = jnp.broadcast_to(
            self.weights[:, None],
            (self.num_particles, int(self.times.size)),
        )
        source = (
            self.realization if self.realization is not None else self.common_realization
        )
        return EmpiricalMeanField(
            self.times,
            particle_major,
            sample_shape=(self.num_particles,),
            state_shape=self.state_shape,
            mean_field_id=self.mean_field_id,
            weights=weight_history,
            valid=valid_major,
            source_path_id=None if source is None else source.realization_id,
        )

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        """Expose each coupled population as one vector-valued stochastic path."""
        axes = (
            tuple(f"system_{index}" for index in range(len(self.sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        physical_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        components = {
            name: value
            for name, value in (
                ("idiosyncratic", self.realization),
                ("common", self.common_realization),
            )
            if value is not None
        }
        if not components:
            source = None
        elif len(components) == 1:
            source = next(iter(components.values()))
        else:
            source = CompositeStochasticRealization(components)
        record = _TrajectoryRecord(
            self.times,
            self.particles,
            state_shape=(self.num_particles,) + self.state_shape,
            realization_shape=self.sample_shape,
            valid=jnp.all(self.valid, axis=-1),
            realizations=(source,),
            solver_name=self.solver_name,
            uncertainty_source="process",
            metadata={
                **dict(self.metadata),
                "particle_coupling": "empirical-mean-field",
            },
        )
        return record.to_stochastic_trajectory(
            realization_axes=axes,
            state_axes=("particle",) + physical_state_axes,
        )


def _identifier(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _noise_contract(
    diffusion: ParticleVectorField | None,
    noise_shape: Sequence[int] | None,
    noise_id: str | None,
    /,
    *,
    owner: str,
) -> tuple[int, ...]:
    if diffusion is None:
        if noise_shape is not None or noise_id is not None:
            raise ValueError(f"{owner} noise metadata requires its diffusion field.")
        return ()
    if noise_shape is None:
        raise ValueError(f"{owner}_noise_shape is required with its diffusion field.")
    shape = tuple(int(size) for size in noise_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner}_noise_shape must contain positive dimensions.")
    if noise_id is not None and (not isinstance(noise_id, str) or not noise_id):
        raise ValueError(f"{owner}_noise_id must be non-empty or None.")
    return shape


def _mean_field_snapshot(
    time: Array,
    particles: Array,
    weights: Array,
    /,
    *,
    state_shape: tuple[int, ...],
    mean_field_id: str,
) -> MeanFieldSnapshot:
    event_suffix = (1,) * len(state_shape)
    mean = jnp.sum(particles * weights.reshape(weights.shape + event_suffix), axis=0)
    flat = particles.reshape((particles.shape[0], -1))
    centered = flat - mean.reshape((-1,))
    covariance = ein.contract("p,pi,pj->ij", weights, centered, centered)
    effective = 1.0 / jnp.sum(weights**2)
    state_axes = tuple(range(1, particles.ndim))
    valid = jnp.all(jnp.isfinite(particles), axis=state_axes)
    return MeanFieldSnapshot(
        time=time,
        particles=particles,
        weights=weights,
        mean=mean,
        covariance=covariance,
        effective_sample_size=effective,
        valid=jnp.all(valid),
        state_shape=state_shape,
        mean_field_id=mean_field_id,
    )


def _contract_noise(coefficients: Array, increment: Array, state_rank: int, /) -> Array:
    coefficient_axes = tuple(range(state_rank, coefficients.ndim))
    increment_axes = tuple(range(increment.ndim))
    return jnp.tensordot(
        coefficients,
        increment,
        axes=(coefficient_axes, increment_axes),
    )


def solve_interacting_particles(
    problem: InteractingParticleProblem,
    /,
    *,
    times: ArrayLike,
    realization: WienerRealization | None = None,
    common_realization: WienerRealization | None = None,
) -> InteractingParticleSolution:
    """Integrate an interacting McKean--Vlasov particle system by Euler--Maruyama."""
    if not isinstance(problem, InteractingParticleProblem):
        raise TypeError("problem must be an InteractingParticleProblem.")
    grid = _time_grid(problem.t0, problem.t1, times)
    idiosyncratic_increments, idiosyncratic_samples = _wiener_increments(
        stochastic=problem.diffusion is not None,
        noise_shape=(problem.num_particles,) + problem.noise_shape,
        noise_id=problem.noise_id,
        t0=problem.t0,
        t1=problem.t1,
        times=grid,
        realization=realization,
        dtype=problem.initial_particles.real.dtype,
    )
    common_increments, common_samples = _wiener_increments(
        stochastic=problem.common_diffusion is not None,
        noise_shape=problem.common_noise_shape,
        noise_id=problem.common_noise_id,
        t0=problem.t0,
        t1=problem.t1,
        times=grid,
        realization=common_realization,
        dtype=problem.initial_particles.real.dtype,
    )
    if (
        idiosyncratic_samples
        and common_samples
        and idiosyncratic_samples != common_samples
    ):
        raise ValueError("Idiosyncratic and common realizations must share sample_shape.")
    sample_shape = idiosyncratic_samples or common_samples
    if problem.diffusion is not None and idiosyncratic_samples != sample_shape:
        raise ValueError("Idiosyncratic realization sample_shape does not align.")
    if problem.common_diffusion is not None and common_samples != sample_shape:
        raise ValueError("Common realization sample_shape does not align.")
    diffusion_fn = problem.diffusion
    common_diffusion_fn = problem.common_diffusion
    num_times = int(grid.size)
    num_steps = num_times - 1
    step_sizes = jnp.diff(grid)

    def one_system(idiosyncratic_path, common_path):
        particles = jnp.zeros(
            (num_times, problem.num_particles) + problem.state_shape,
            dtype=problem.initial_particles.dtype,
        )
        particles = particles.at[0].set(problem.initial_particles)

        def step(index, buffer):
            time = grid[index]
            current = buffer[index]
            snapshot = _mean_field_snapshot(
                time,
                current,
                problem.weights,
                state_shape=problem.state_shape,
                mean_field_id=problem.mean_field_id,
            )
            drift = jax.vmap(
                lambda state: jnp.asarray(
                    problem.drift(time, state, snapshot, problem.args)
                )
            )(current)
            if drift.shape != current.shape:
                raise ValueError("drift must preserve each particle state shape.")
            update = step_sizes[index] * drift
            if diffusion_fn is not None:
                diffusion = jax.vmap(
                    lambda state: jnp.asarray(
                        diffusion_fn(time, state, snapshot, problem.args)
                    )
                )(current)
                expected = (
                    (problem.num_particles,) + problem.state_shape + problem.noise_shape
                )
                if diffusion.shape != expected:
                    raise ValueError(
                        f"diffusion must stack to shape {expected}; got {diffusion.shape}."
                    )
                update = update + jax.vmap(
                    lambda coefficient, increment: _contract_noise(
                        coefficient,
                        increment,
                        len(problem.state_shape),
                    )
                )(diffusion, idiosyncratic_path[index])
            if common_diffusion_fn is not None:
                common_diffusion = jax.vmap(
                    lambda state: jnp.asarray(
                        common_diffusion_fn(time, state, snapshot, problem.args)
                    )
                )(current)
                expected = (
                    (problem.num_particles,)
                    + problem.state_shape
                    + problem.common_noise_shape
                )
                if common_diffusion.shape != expected:
                    raise ValueError(
                        "common_diffusion returned an incompatible stacked shape."
                    )
                update = update + jax.vmap(
                    lambda coefficient: _contract_noise(
                        coefficient,
                        common_path[index],
                        len(problem.state_shape),
                    )
                )(common_diffusion)
            return buffer.at[index + 1].set(current + update)

        return jax.lax.fori_loop(0, num_steps, step, particles)

    if sample_shape:
        count = prod(sample_shape)
        if problem.diffusion is None:
            flat_idiosyncratic = jnp.zeros((count, num_steps, 0))
        else:
            flat_idiosyncratic = idiosyncratic_increments.reshape(
                (count, num_steps, problem.num_particles) + problem.noise_shape
            )
        if problem.common_diffusion is None:
            flat_common = jnp.zeros((count, num_steps, 0))
        else:
            flat_common = common_increments.reshape(
                (count, num_steps) + problem.common_noise_shape
            )
        particles = jax.vmap(one_system)(flat_idiosyncratic, flat_common).reshape(
            sample_shape + (num_times, problem.num_particles) + problem.state_shape
        )
    else:
        particles = one_system(idiosyncratic_increments, common_increments)
    valid = _solution_valid(
        particles,
        sample_shape + (num_times,),
    )
    event_suffix = (1,) * len(problem.state_shape)
    weight_shape = (1,) * len(sample_shape) + (1, problem.num_particles) + event_suffix
    means = jnp.sum(
        particles * problem.weights.reshape(weight_shape),
        axis=len(sample_shape) + 1,
    )
    flat_particles = particles.reshape(
        sample_shape + (num_times, problem.num_particles, prod(problem.state_shape))
    )
    flat_means = means.reshape(sample_shape + (num_times, prod(problem.state_shape)))
    centered = flat_particles - jnp.expand_dims(flat_means, axis=len(sample_shape) + 1)
    covariances = ein.contract(
        "...tpi,p,...tpj->...tij",
        centered,
        problem.weights,
        centered,
    )
    return InteractingParticleSolution(
        times=grid,
        particles=particles,
        valid=valid,
        weights=problem.weights,
        means=means,
        covariances=covariances,
        realization=realization,
        common_realization=common_realization,
        sample_shape=sample_shape,
        state_shape=problem.state_shape,
        num_particles=problem.num_particles,
        mean_field_id=problem.mean_field_id,
        metadata={
            "problem_id": problem.problem_id,
            "num_steps": num_steps,
            "has_idiosyncratic_noise": problem.diffusion is not None,
            "has_common_noise": problem.common_diffusion is not None,
        },
    )


__all__ = [
    "InteractingParticleProblem",
    "InteractingParticleSolution",
    "ParticleVectorField",
    "solve_interacting_particles",
]

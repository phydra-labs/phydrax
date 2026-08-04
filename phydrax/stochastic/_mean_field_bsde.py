#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ._bsde import BSDEPathBatch, BSDEProblem


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _event_finite(values: Array, event_rank: int, /) -> Array:
    if event_rank == 0:
        return jnp.isfinite(values)
    return jnp.all(
        jnp.isfinite(values),
        axis=tuple(range(values.ndim - event_rank, values.ndim)),
    )


class MeanFieldSnapshot(StrictModule):
    """One weighted empirical law and its first two moments."""

    time: Array
    particles: Array
    weights: Array
    mean: Array
    covariance: Array
    effective_sample_size: Array
    valid: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    mean_field_id: str = eqx.field(static=True)

    def expectation(self, observable: Callable[[Array], Array], /) -> Array:
        """Evaluate a weighted particle expectation without hiding its finite support."""
        if not callable(observable):
            raise TypeError("observable must be callable.")
        values = jax.vmap(observable)(self.particles)
        weight_shape = self.weights.shape + (1,) * (values.ndim - 1)
        return jnp.sum(values * self.weights.reshape(weight_shape), axis=0)


class EmpiricalMeanField(StrictModule):
    """Time-aligned weighted particle representation of a measure flow."""

    times: Array
    particles: Array
    weights: Array
    valid: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    mean_field_id: str = eqx.field(static=True)
    source_path_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        particles: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int],
        state_shape: Sequence[int],
        mean_field_id: str,
        weights: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        source_path_id: str | None = None,
    ):
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        state_event = _shape(state_shape, owner="state_shape")
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] < 2:
            raise ValueError(
                "times must be a one-dimensional grid with at least two nodes."
            )
        if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
            jnp.any(jnp.diff(time_values) <= 0.0)
        ):
            raise ValueError("times must be finite and strictly increasing.")
        particle_values = jnp.asarray(particles)
        expected_particles = samples + (time_values.shape[0],) + state_event
        if particle_values.shape != expected_particles:
            raise ValueError(
                f"particles must have shape {expected_particles}; got {particle_values.shape}."
            )
        measure_shape = samples + (time_values.shape[0],)
        if valid is None:
            validity = _event_finite(particle_values, len(state_event))
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != measure_shape:
                raise ValueError("valid must have sample_shape + (num_nodes,) shape.")
            validity = validity & _event_finite(particle_values, len(state_event))
        if weights is None:
            weight_values = jnp.ones(measure_shape, dtype=particle_values.dtype)
        else:
            weight_values = jnp.asarray(weights, dtype=float)
            if weight_values.shape != measure_shape:
                raise ValueError("weights must have sample_shape + (num_nodes,) shape.")
            if bool(jnp.any(~jnp.isfinite(weight_values))) or bool(
                jnp.any(weight_values < 0.0)
            ):
                raise ValueError("weights must be finite and nonnegative.")
        flat_weights = weight_values.reshape((-1, time_values.shape[0]))
        flat_valid = validity.reshape((-1, time_values.shape[0]))
        node_mass = jnp.sum(jnp.where(flat_valid, flat_weights, 0.0), axis=0)
        if bool(jnp.any(node_mass <= 0.0)):
            raise ValueError("Every mean-field node must retain positive valid mass.")
        if source_path_id is not None:
            _name(source_path_id, owner="source_path_id")
        self.times = time_values
        self.particles = particle_values
        self.weights = weight_values
        self.valid = validity
        self.sample_shape = samples
        self.state_shape = state_event
        self.num_particles = prod(samples) if samples else 1
        self.mean_field_id = _name(mean_field_id, owner="mean_field_id")
        self.source_path_id = source_path_id

    @classmethod
    def from_paths(
        cls,
        paths: BSDEPathBatch,
        /,
        *,
        weights: ArrayLike | None = None,
        mean_field_id: str | None = None,
    ) -> EmpiricalMeanField:
        if not isinstance(paths, BSDEPathBatch):
            raise TypeError("paths must be a BSDEPathBatch.")
        return cls(
            paths.times,
            paths.states,
            sample_shape=paths.sample_shape,
            state_shape=paths.state_shape,
            mean_field_id=(
                f"{paths.path_id}:empirical-mean-field"
                if mean_field_id is None
                else mean_field_id
            ),
            weights=weights,
            valid=paths.valid,
            source_path_id=paths.path_id,
        )

    @property
    def support(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    def snapshot(self, time: ArrayLike, /) -> MeanFieldSnapshot:
        """Interpolate the Lagrangian particle law at one scalar query time."""
        query = jnp.asarray(time, dtype=self.times.dtype)
        if query.shape != ():
            raise ValueError("time must be scalar.")
        upper = jnp.clip(
            jnp.searchsorted(self.times, query, side="right"),
            1,
            self.times.shape[0] - 1,
        )
        lower = upper - 1
        left_time = self.times[lower]
        right_time = self.times[upper]
        alpha = jnp.clip((query - left_time) / (right_time - left_time), 0.0, 1.0)
        flat_particles = self.particles.reshape(
            (self.num_particles, self.times.shape[0]) + self.state_shape
        )
        flat_weights = self.weights.reshape((self.num_particles, self.times.shape[0]))
        flat_valid = self.valid.reshape((self.num_particles, self.times.shape[0]))
        left_valid = flat_valid[:, lower]
        right_valid = flat_valid[:, upper]
        interpolation_valid = jnp.where(
            alpha == 0.0,
            left_valid,
            jnp.where(alpha == 1.0, right_valid, left_valid & right_valid),
        )
        event_suffix = (1,) * len(self.state_shape)
        left_particles = jnp.where(
            left_valid.reshape(left_valid.shape + event_suffix),
            flat_particles[:, lower],
            0.0,
        )
        right_particles = jnp.where(
            right_valid.reshape(right_valid.shape + event_suffix),
            flat_particles[:, upper],
            0.0,
        )
        particles = (1.0 - alpha) * left_particles + alpha * right_particles
        raw_weights = (1.0 - alpha) * flat_weights[:, lower] + alpha * flat_weights[
            :, upper
        ]
        raw_weights = jnp.where(interpolation_valid, raw_weights, 0.0)
        mass = jnp.sum(raw_weights)
        normalized = raw_weights / jnp.maximum(mass, jnp.finfo(raw_weights.dtype).tiny)
        normalized_shape = normalized.shape + event_suffix
        mean = jnp.sum(particles * normalized.reshape(normalized_shape), axis=0)
        flat = particles.reshape((self.num_particles, -1))
        flat_mean = mean.reshape((-1,))
        centered = flat - flat_mean
        covariance = jnp.einsum("p,pi,pj->ij", normalized, centered, centered)
        effective = 1.0 / jnp.maximum(
            jnp.sum(normalized**2), jnp.finfo(normalized.dtype).tiny
        )
        in_support = (query >= self.times[0]) & (query <= self.times[-1])
        return MeanFieldSnapshot(
            time=query,
            particles=particles,
            weights=normalized,
            mean=mean,
            covariance=covariance,
            effective_sample_size=effective,
            valid=in_support & (mass > 0.0),
            state_shape=self.state_shape,
            mean_field_id=self.mean_field_id,
        )


class MeanFieldBSDEControlAdapter(StrictModule):
    """Hamiltonian adapter for controls represented as Wiener drift shifts.

    The controlled drift must satisfy ``b_controlled = b_reference + sigma @ shift``.
    Given a policy and running cost, the BSDE generator is ``cost + Z · shift``.
    A policy that minimizes this expression implements the stochastic-control HJB
    Hamiltonian without requiring an inverse of ``sigma``.
    """

    policy: Callable[[Array, Array, MeanFieldSnapshot, Array, Array, Any], Array]
    running_cost: Callable[[Array, Array, MeanFieldSnapshot, Array, Any], Array]
    wiener_drift_shift: Callable[[Array, Array, MeanFieldSnapshot, Array, Any], Array]
    control_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: Callable[[Array, Array, MeanFieldSnapshot, Array, Array, Any], Array],
        running_cost: Callable[[Array, Array, MeanFieldSnapshot, Array, Any], Array],
        wiener_drift_shift: Callable[
            [Array, Array, MeanFieldSnapshot, Array, Any], Array
        ],
        /,
        *,
        control_shape: Sequence[int],
        output_shape: Sequence[int],
        noise_shape: Sequence[int],
        adapter_id: str,
    ):
        for owner, value in (
            ("policy", policy),
            ("running_cost", running_cost),
            ("wiener_drift_shift", wiener_drift_shift),
        ):
            if not callable(value):
                raise TypeError(f"{owner} must be callable.")
        self.policy = policy
        self.running_cost = running_cost
        self.wiener_drift_shift = wiener_drift_shift
        self.control_shape = _shape(control_shape, owner="control_shape")
        self.output_shape = _shape(output_shape, owner="output_shape")
        self.noise_shape = _shape(noise_shape, owner="noise_shape")
        self.adapter_id = _name(adapter_id, owner="adapter_id")

    def control(
        self,
        time: Array,
        state: Array,
        mean_field: MeanFieldSnapshot,
        value: Array,
        bsde_control: Array,
        args: Any,
        /,
    ) -> Array:
        action = jnp.asarray(
            self.policy(time, state, mean_field, value, bsde_control, args)
        )
        if action.shape != self.control_shape:
            raise ValueError("policy returned an incompatible control shape.")
        return action

    def generator(
        self,
        time: Array,
        state: Array,
        mean_field: MeanFieldSnapshot,
        value: Array,
        bsde_control: Array,
        args: Any,
        /,
    ) -> Array:
        if bsde_control.shape != self.output_shape + self.noise_shape:
            raise ValueError("bsde_control has an incompatible output/noise shape.")
        action = self.control(time, state, mean_field, value, bsde_control, args)
        cost = jnp.asarray(self.running_cost(time, state, mean_field, action, args))
        shift = jnp.asarray(
            self.wiener_drift_shift(time, state, mean_field, action, args)
        )
        if cost.shape != self.output_shape:
            raise ValueError("running_cost returned an incompatible output shape.")
        if shift.shape != self.noise_shape:
            raise ValueError("wiener_drift_shift returned an incompatible noise shape.")
        contraction = bsde_control.reshape(
            (prod(self.output_shape), prod(self.noise_shape))
        ) @ shift.reshape((prod(self.noise_shape),))
        return cost + contraction.reshape(self.output_shape)


class MeanFieldBSDEProblem(StrictModule):
    """McKean--Vlasov BSDE frozen against one explicit empirical measure flow."""

    forward_sampler: Callable[[Array], BSDEPathBatch]
    mean_field: EmpiricalMeanField
    drift: Callable[[Array, Array, MeanFieldSnapshot, Any], Array]
    diffusion: Callable[[Array, Array, MeanFieldSnapshot, Any], Array]
    generator: Callable[[Array, Array, MeanFieldSnapshot, Array, Array, Any], Array]
    terminal: Callable[[Array, MeanFieldSnapshot, Any], Array]
    control_adapter: MeanFieldBSDEControlAdapter | None
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward_sampler: Callable[[Array], BSDEPathBatch],
        mean_field: EmpiricalMeanField,
        drift: Callable[[Array, Array, MeanFieldSnapshot, Any], Array],
        diffusion: Callable[[Array, Array, MeanFieldSnapshot, Any], Array],
        generator: Callable[[Array, Array, MeanFieldSnapshot, Array, Array, Any], Array],
        terminal: Callable[[Array, MeanFieldSnapshot, Any], Array],
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        output_shape: Sequence[int],
        problem_id: str,
        process_id: str,
        args: Any = None,
        control_adapter: MeanFieldBSDEControlAdapter | None = None,
    ):
        for owner, value in (
            ("forward_sampler", forward_sampler),
            ("drift", drift),
            ("diffusion", diffusion),
            ("generator", generator),
            ("terminal", terminal),
        ):
            if not callable(value):
                raise TypeError(f"{owner} must be callable.")
        if not isinstance(mean_field, EmpiricalMeanField):
            raise TypeError("mean_field must be an EmpiricalMeanField.")
        state_event = _shape(state_shape, owner="state_shape")
        noise_event = _shape(noise_shape, owner="noise_shape")
        output_event = _shape(output_shape, owner="output_shape")
        if mean_field.state_shape != state_event:
            raise ValueError("mean_field state_shape must match state_shape.")
        if control_adapter is not None:
            if not isinstance(control_adapter, MeanFieldBSDEControlAdapter):
                raise TypeError(
                    "control_adapter must be a MeanFieldBSDEControlAdapter or None."
                )
            if (
                control_adapter.noise_shape != noise_event
                or control_adapter.output_shape != output_event
            ):
                raise ValueError("control_adapter output/noise shapes do not match.")
        self.forward_sampler = forward_sampler
        self.mean_field = mean_field
        self.drift = drift
        self.diffusion = diffusion
        self.generator = generator
        self.terminal = terminal
        self.control_adapter = control_adapter
        self.args = args
        self.state_shape = state_event
        self.noise_shape = noise_event
        self.output_shape = output_event
        self.problem_id = _name(problem_id, owner="problem_id")
        self.process_id = _name(process_id, owner="process_id")

    def sample(self, key: Key[Array, ""], /) -> BSDEPathBatch:
        paths = self.forward_sampler(key)
        if not isinstance(paths, BSDEPathBatch):
            raise TypeError("forward_sampler must return a BSDEPathBatch.")
        if paths.state_shape != self.state_shape or paths.noise_shape != self.noise_shape:
            raise ValueError("Forward path state/noise shapes do not match the problem.")
        if paths.process_id != self.process_id:
            raise ValueError("Forward path and mean-field BSDE process IDs do not match.")
        if (
            float(paths.times[0]) != self.mean_field.support[0]
            or float(paths.times[-1]) != self.mean_field.support[1]
        ):
            raise ValueError("Forward paths and mean-field flow must share time support.")
        return paths

    def as_bsde_problem(self) -> BSDEProblem:
        """Freeze the empirical law into the canonical Phydrax BSDE contract."""

        def forward_sampler(key):
            return self.sample(key)

        def drift(time, state, args):
            snapshot = self.mean_field.snapshot(time)
            value = jnp.asarray(self.drift(time, state, snapshot, args))
            if value.shape != self.state_shape:
                raise ValueError("mean-field drift returned an incompatible shape.")
            return value

        def diffusion(time, state, args):
            snapshot = self.mean_field.snapshot(time)
            value = jnp.asarray(self.diffusion(time, state, snapshot, args))
            if value.shape != self.state_shape + self.noise_shape:
                raise ValueError("mean-field diffusion returned an incompatible shape.")
            return value

        def generator(time, state, value, control, args):
            snapshot = self.mean_field.snapshot(time)
            output = jnp.asarray(
                self.generator(time, state, snapshot, value, control, args)
            )
            if output.shape != self.output_shape:
                raise ValueError("mean-field generator returned an incompatible shape.")
            return output

        def terminal(state, args):
            snapshot = self.mean_field.snapshot(self.mean_field.times[-1])
            value = jnp.asarray(self.terminal(state, snapshot, args))
            if value.shape != self.output_shape:
                raise ValueError("mean-field terminal returned an incompatible shape.")
            return value

        return BSDEProblem(
            forward_sampler,
            drift,
            diffusion,
            generator,
            terminal,
            state_shape=self.state_shape,
            noise_shape=self.noise_shape,
            output_shape=self.output_shape,
            problem_id=self.problem_id,
            process_id=self.process_id,
            args=self.args,
        )


def adapt_mean_field_control_bsde(
    forward_sampler: Callable[[Array], BSDEPathBatch],
    mean_field: EmpiricalMeanField,
    reference_drift: Callable[[Array, Array, MeanFieldSnapshot, Any], Array],
    diffusion: Callable[[Array, Array, MeanFieldSnapshot, Any], Array],
    terminal_cost: Callable[[Array, MeanFieldSnapshot, Any], Array],
    control_adapter: MeanFieldBSDEControlAdapter,
    /,
    *,
    state_shape: Sequence[int],
    problem_id: str,
    process_id: str,
    args: Any = None,
) -> MeanFieldBSDEProblem:
    """Adapt a mean-field stochastic control Hamiltonian to a canonical BSDE."""
    if not isinstance(control_adapter, MeanFieldBSDEControlAdapter):
        raise TypeError("control_adapter must be a MeanFieldBSDEControlAdapter.")

    def generator(time, state, snapshot, value, bsde_control, problem_args):
        return control_adapter.generator(
            time,
            state,
            snapshot,
            value,
            bsde_control,
            problem_args,
        )

    return MeanFieldBSDEProblem(
        forward_sampler,
        mean_field,
        reference_drift,
        diffusion,
        generator,
        terminal_cost,
        state_shape=state_shape,
        noise_shape=control_adapter.noise_shape,
        output_shape=control_adapter.output_shape,
        problem_id=problem_id,
        process_id=process_id,
        args=args,
        control_adapter=control_adapter,
    )


def evaluate_mean_field_bsde_control(
    problem: MeanFieldBSDEProblem,
    time: ArrayLike,
    state: ArrayLike,
    value: ArrayLike,
    bsde_control: ArrayLike,
    /,
) -> Array:
    """Recover the physical control selected by a control-adapted BSDE."""
    if not isinstance(problem, MeanFieldBSDEProblem):
        raise TypeError("problem must be a MeanFieldBSDEProblem.")
    if problem.control_adapter is None:
        raise ValueError("Mean-field BSDE problem has no control adapter.")
    time_value = jnp.asarray(time)
    state_value = jnp.asarray(state)
    value_array = jnp.asarray(value)
    control_array = jnp.asarray(bsde_control)
    if time_value.shape != () or state_value.shape != problem.state_shape:
        raise ValueError("time/state shapes do not match the mean-field BSDE.")
    if value_array.shape != problem.output_shape:
        raise ValueError("value has an incompatible output shape.")
    if control_array.shape != problem.output_shape + problem.noise_shape:
        raise ValueError("bsde_control has an incompatible output/noise shape.")
    snapshot = problem.mean_field.snapshot(time_value)
    return problem.control_adapter.control(
        time_value,
        state_value,
        snapshot,
        value_array,
        control_array,
        problem.args,
    )


__all__ = [
    "adapt_mean_field_control_bsde",
    "EmpiricalMeanField",
    "evaluate_mean_field_bsde_control",
    "MeanFieldBSDEControlAdapter",
    "MeanFieldBSDEProblem",
    "MeanFieldSnapshot",
]

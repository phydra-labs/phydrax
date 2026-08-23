#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ...dynamics._evolution import (
    EVOLUTION_BACKEND_FAILED,
    EVOLUTION_NONFINITE,
    EVOLUTION_SUCCESS,
)
from ...operators.differential._stochastic_estimators import (
    exact_state_divergence,
    stochastic_divergence_samples,
    StochasticTracePolicy,
)
from ._transport import ContinuousTransport


_DIFFRAX_SUCCESS = jax.tree.leaves(dfx.RESULTS.successful)[0]


def _event_shape(value: Array, event_shape: tuple[int, ...], /) -> tuple[int, ...]:
    rank = len(event_shape)
    if rank and (value.ndim < rank or tuple(value.shape[-rank:]) != event_shape):
        raise ValueError(
            f"Value must end in event shape {event_shape}; got {value.shape}."
        )
    return tuple(value.shape[:-rank]) if rank else tuple(value.shape)


def _validate_density_transport(transport: ContinuousTransport, /) -> Any:
    from ...solver._dynamics_evolution import DiffraxEvolution

    if not isinstance(transport, ContinuousTransport):
        raise TypeError("transport must be ContinuousTransport.")
    evolution = transport.evolution
    if not isinstance(evolution, DiffraxEvolution):
        raise TypeError("Continuous flow density currently requires DiffraxEvolution.")
    if transport.source_law.density_measure_kind != "lebesgue":
        raise ValueError("Continuous flow density requires a Lebesgue source law.")
    if not evolution.state_layout.geometry.trivial:
        raise ValueError("Continuous flow density initially requires trivial geometry.")
    if evolution.system.input_layout is not None or evolution.input_policy is not None:
        raise ValueError(
            "Continuous flow density initially requires an autonomous system."
        )
    if evolution.event is not None:
        raise ValueError(
            "Continuous flow density does not accept state-shaped events yet."
        )
    return evolution


class _ExactAugmentedField(eqx.Module):
    transport: ContinuousTransport
    reverse: bool = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    event_size: int = eqx.field(static=True)

    def __init__(self, transport: ContinuousTransport, /, *, reverse: bool):
        self.transport = transport
        self.reverse = bool(reverse)
        self.event_shape = transport.event_shape
        self.event_size = max(prod(self.event_shape), 1)

    def __call__(self, coordinate: Array, augmented: Array, args: Any, /) -> Array:
        del args
        state = augmented[: self.event_size].reshape(self.event_shape)
        physical_coordinate = (
            self.transport.source_coordinate
            + self.transport.target_coordinate
            - coordinate
            if self.reverse
            else coordinate
        )

        def vector_field(current):
            return self.transport.evolution.system.evaluate(
                physical_coordinate,
                current,
                self.transport.args,
            )

        velocity = vector_field(state)
        divergence = exact_state_divergence(vector_field, state)
        sign = -1.0 if self.reverse else 1.0
        return jnp.concatenate(
            (sign * velocity.reshape((-1,)), (sign * divergence).reshape((1,)))
        )


class _StochasticAugmentedField(eqx.Module):
    transport: ContinuousTransport
    probe_key: Array
    policy: StochasticTracePolicy
    event_shape: tuple[int, ...] = eqx.field(static=True)
    event_size: int = eqx.field(static=True)

    def __init__(
        self,
        transport: ContinuousTransport,
        probe_key: Key[Array, ""],
        policy: StochasticTracePolicy,
        /,
    ):
        self.transport = transport
        self.probe_key = jnp.asarray(probe_key)
        self.policy = policy
        self.event_shape = transport.event_shape
        self.event_size = max(prod(self.event_shape), 1)

    def __call__(self, coordinate: Array, augmented: Array, args: Any, /) -> Array:
        del args
        state = augmented[: self.event_size].reshape(self.event_shape)
        physical_coordinate = (
            self.transport.source_coordinate
            + self.transport.target_coordinate
            - coordinate
        )

        def vector_field(current):
            return self.transport.evolution.system.evaluate(
                physical_coordinate,
                current,
                self.transport.args,
            )

        velocity = vector_field(state)
        divergence = stochastic_divergence_samples(
            vector_field,
            state,
            self.probe_key,
            policy=self.policy,
        )
        return jnp.concatenate(
            (-velocity.reshape((-1,)), -divergence.values.reshape((-1,)))
        )


def _solve_augmented(
    transport: ContinuousTransport,
    initial_state: Array,
    field: _ExactAugmentedField | _StochasticAugmentedField,
    accumulator_count: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    from ...solver._differential import DifferentialProblem
    from ...solver._diffrax_backend import solve_diffrax

    evolution = _validate_density_transport(transport)
    flat = jnp.asarray(initial_state).reshape((-1,))
    if jnp.issubdtype(flat.dtype, jnp.complexfloating):
        raise TypeError("Continuous flow density requires explicit real coordinates.")
    augmented = jnp.concatenate((flat, jnp.zeros((accumulator_count,), dtype=flat.dtype)))
    problem = DifferentialProblem(
        field,
        augmented,
        t0=transport.source_coordinate,
        t1=transport.target_coordinate,
    )
    solution = solve_diffrax(
        problem,
        save_times=jnp.asarray([transport.target_coordinate]),
        solver=evolution.solver,
        stepsize_controller=evolution.stepsize_controller,
        adjoint=evolution.adjoint,
        dt0=evolution.dt0,
        event=None,
        rtol=evolution.rtol,
        atol=evolution.atol,
        dense=False,
        max_steps=evolution.max_steps,
        throw=False,
    )
    final = solution.states[-1]
    event_size = max(prod(transport.event_shape), 1)
    transformed = final[:event_size].reshape(transport.event_shape)
    accumulators = final[event_size:]
    backend = jax.tree.leaves(solution.backend_result)[0]
    finite = solution.successful & jnp.all(jnp.isfinite(final))
    backend_valid = backend == _DIFFRAX_SUCCESS
    valid = finite & backend_valid
    status = jnp.where(
        ~backend_valid,
        EVOLUTION_BACKEND_FAILED,
        jnp.where(finite, EVOLUTION_SUCCESS, EVOLUTION_NONFINITE),
    ).astype(jnp.int32)
    return (
        transformed,
        accumulators,
        valid,
        status,
        backend,
        jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32),
        jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32),
    )


class ContinuousFlowDensityResult(StrictModule):
    """Density value with numerical and optional probe evidence."""

    data_state: Array
    base_state: Array
    base_log_prob: Array
    log_volume: Array
    log_prob: Array
    probe_log_volumes: Array
    standard_error: Array
    valid: Array
    status: Array
    backend_status: Array
    accepted_steps: Array
    rejected_steps: Array
    leading_shape: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    direction: str = eqx.field(static=True)
    divergence_method: str = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    probe_distribution: str = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        data_state: ArrayLike,
        base_state: ArrayLike,
        base_log_prob: ArrayLike,
        log_volume: ArrayLike,
        log_prob: ArrayLike,
        probe_log_volumes: ArrayLike,
        standard_error: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        backend_status: ArrayLike,
        accepted_steps: ArrayLike,
        rejected_steps: ArrayLike,
        event_shape,
        direction: str,
        divergence_method: str,
        num_probes: int,
        probe_distribution: str,
        flow_id: str,
    ):
        events = tuple(int(size) for size in event_shape)
        data = jnp.asarray(data_state)
        base = jnp.asarray(base_state, dtype=data.dtype)
        if data.shape != base.shape:
            raise ValueError("Continuous-flow data and base states must have one shape.")
        leading = _event_shape(data, events)
        base_density = jnp.asarray(base_log_prob)
        volume = jnp.asarray(log_volume)
        density = jnp.asarray(log_prob)
        error = jnp.asarray(standard_error)
        validity = jnp.asarray(valid, dtype=bool)
        statuses = jnp.asarray(status, dtype=jnp.int32)
        backend = jnp.asarray(backend_status)
        accepted = jnp.asarray(accepted_steps, dtype=jnp.int32)
        rejected = jnp.asarray(rejected_steps, dtype=jnp.int32)
        if not (
            base_density.shape
            == volume.shape
            == density.shape
            == error.shape
            == validity.shape
            == statuses.shape
            == backend.shape
            == accepted.shape
            == rejected.shape
            == leading
        ):
            raise ValueError("Continuous-flow scalar evidence must match leading shape.")
        probes = jnp.asarray(probe_log_volumes)
        count = int(num_probes)
        if probes.shape != leading + (max(count, 1),):
            raise ValueError("Probe log volumes have incompatible leading/probe shape.")
        for name, identifier in (
            ("direction", direction),
            ("divergence_method", divergence_method),
            ("probe_distribution", probe_distribution),
            ("flow_id", flow_id),
        ):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{name} must be a non-empty string.")
        self.data_state = data
        self.base_state = base
        self.base_log_prob = base_density
        self.log_volume = volume
        self.log_prob = density
        self.probe_log_volumes = probes
        self.standard_error = error
        self.valid = validity
        self.status = statuses
        self.backend_status = backend
        self.accepted_steps = accepted
        self.rejected_steps = rejected
        self.leading_shape = leading
        self.event_shape = events
        self.direction = direction
        self.divergence_method = divergence_method
        self.num_probes = count
        self.probe_distribution = probe_distribution
        self.flow_id = flow_id

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid) & jnp.all(self.status == EVOLUTION_SUCCESS)


def _base_log_prob(law: AbstractProbabilityLaw, states: Array, event_shape, /) -> Array:
    leading = _event_shape(states, event_shape)
    count = prod(leading) if leading else 1
    flat = states.reshape((count,) + tuple(event_shape))
    values = jax.vmap(law.log_prob)(flat)
    return values.reshape(leading)


def _base_contains(law: AbstractProbabilityLaw, states: Array, event_shape, /) -> Array:
    leading = _event_shape(states, event_shape)
    count = prod(leading) if leading else 1
    flat = states.reshape((count,) + tuple(event_shape))
    values = jax.vmap(law.contains)(flat)
    return jnp.asarray(values, dtype=bool).reshape(leading)


def _exact_density_batch(
    transport: ContinuousTransport,
    states: Array,
    /,
    *,
    reverse: bool,
    flow_id: str,
) -> ContinuousFlowDensityResult:
    event_shape = transport.event_shape
    leading = _event_shape(states, event_shape)
    count = prod(leading) if leading else 1
    flat = states.reshape((count,) + event_shape)
    field = _ExactAugmentedField(transport, reverse=reverse)

    def one(state):
        return _solve_augmented(transport, state, field, 1)

    transformed, raw_volume, valid, status, backend, accepted, rejected = jax.vmap(one)(
        flat
    )
    transformed = transformed.reshape(leading + event_shape)
    raw_volume = raw_volume.reshape(leading + (1,))
    correction = raw_volume[..., 0]
    if reverse:
        data = states
        base = transformed
        base_density = _base_log_prob(transport.source_law, base, event_shape)
        density = base_density + correction
        direction = "data-to-base"
    else:
        data = transformed
        base = states
        base_density = _base_log_prob(transport.source_law, base, event_shape)
        density = base_density - correction
        direction = "base-to-data"
    return ContinuousFlowDensityResult(
        data_state=data,
        base_state=base,
        base_log_prob=base_density,
        log_volume=correction,
        log_prob=density,
        probe_log_volumes=raw_volume,
        standard_error=jnp.zeros(leading, dtype=density.dtype),
        valid=valid.reshape(leading),
        status=status.reshape(leading),
        backend_status=backend.reshape(leading),
        accepted_steps=accepted.reshape(leading),
        rejected_steps=rejected.reshape(leading),
        event_shape=event_shape,
        direction=direction,
        divergence_method="exact-state-jacobian",
        num_probes=0,
        probe_distribution="none",
        flow_id=flow_id,
    )


class ContinuousFlowLaw(AbstractProbabilityLaw):
    """Exact finite-dimensional Lebesgue law induced by continuous transport."""

    transport: ContinuousTransport
    max_exact_dimension: int = eqx.field(static=True)
    flow_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: ContinuousTransport,
        /,
        *,
        max_exact_dimension: int = 32,
        flow_id: str | None = None,
    ):
        _validate_density_transport(transport)
        limit = int(max_exact_dimension)
        if limit <= 0:
            raise ValueError("max_exact_dimension must be positive.")
        dimension = max(prod(transport.event_shape), 1)
        if dimension > limit:
            raise ValueError(
                f"Exact continuous-flow dimension {dimension} exceeds cap {limit}."
            )
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "exact-continuous-flow-law-v1",
                    "transport_id": transport.transport_id,
                    "max_exact_dimension": limit,
                }
            )
            if flow_id is None
            else str(flow_id)
        )
        if not resolved_id:
            raise ValueError("flow_id must be non-empty.")
        self.transport = transport
        self.max_exact_dimension = limit
        self.flow_id = resolved_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.transport.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self) -> str:
        return "lebesgue"

    def sample_with_diagnostics(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ):
        return self.transport.sample_with_diagnostics(key, sample_shape)

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.transport.sample(key, sample_shape)

    def sample_and_log_prob(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Array, Array]:
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        base = jnp.asarray(self.transport.source_law.sample(key, samples))
        result = _exact_density_batch(
            self.transport,
            base,
            reverse=False,
            flow_id=self.flow_id,
        )
        data = eqx.error_if(
            result.data_state,
            ~result.successful,
            "Continuous-flow forward density solve failed.",
        )
        density = eqx.error_if(
            result.log_prob,
            ~result.successful,
            "Continuous-flow forward density solve failed.",
        )
        return data, density

    def log_prob_with_diagnostics(
        self,
        value: ArrayLike,
        /,
    ) -> ContinuousFlowDensityResult:
        values = jnp.asarray(value)
        _event_shape(values, self.event_shape)
        return _exact_density_batch(
            self.transport,
            values,
            reverse=True,
            flow_id=self.flow_id,
        )

    def log_prob(self, value: ArrayLike, /) -> Array:
        result = self.log_prob_with_diagnostics(value)
        return eqx.error_if(
            result.log_prob,
            ~result.successful,
            "Continuous-flow inverse density solve failed.",
        )

    def contains(self, value: ArrayLike, /) -> Array:
        result = self.log_prob_with_diagnostics(value)
        support = _base_contains(
            self.transport.source_law,
            result.base_state,
            self.event_shape,
        )
        return result.valid & support & jnp.isfinite(result.log_prob)


def estimate_continuous_flow_log_prob(
    transport: ContinuousTransport,
    value: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    policy: StochasticTracePolicy | None = None,
) -> ContinuousFlowDensityResult:
    """Estimate inverse-flow log density with fixed Hutchinson probes and error."""
    _validate_density_transport(transport)
    resolved = StochasticTracePolicy() if policy is None else policy
    if not isinstance(resolved, StochasticTracePolicy):
        raise TypeError("policy must be StochasticTracePolicy or None.")
    states = jnp.asarray(value)
    event_shape = transport.event_shape
    leading = _event_shape(states, event_shape)
    count = prod(leading) if leading else 1
    flat = states.reshape((count,) + event_shape)
    keys = jr.split(key, count)

    def one(state, probe_key):
        field = _StochasticAugmentedField(transport, probe_key, resolved)
        return _solve_augmented(
            transport,
            state,
            field,
            resolved.num_probes,
        )

    base, raw, valid, status, backend, accepted, rejected = jax.vmap(one)(flat, keys)
    base = base.reshape(leading + event_shape)
    raw = raw.reshape(leading + (resolved.num_probes,))
    correction = jnp.mean(raw, axis=-1)
    centered = raw - correction[..., None]
    sample_variance = jnp.sum(centered**2, axis=-1) / float(resolved.num_probes - 1)
    standard_error = jnp.sqrt(sample_variance / float(resolved.num_probes))
    base_density = _base_log_prob(transport.source_law, base, event_shape)
    density = base_density + correction
    flow_id = canonical_fingerprint(
        {
            "kind": "stochastic-continuous-flow-density-v1",
            "transport_id": transport.transport_id,
            "num_probes": resolved.num_probes,
            "distribution": resolved.distribution,
        }
    )
    return ContinuousFlowDensityResult(
        data_state=states,
        base_state=base,
        base_log_prob=base_density,
        log_volume=correction,
        log_prob=density,
        probe_log_volumes=raw,
        standard_error=standard_error,
        valid=valid.reshape(leading),
        status=status.reshape(leading),
        backend_status=backend.reshape(leading),
        accepted_steps=accepted.reshape(leading),
        rejected_steps=rejected.reshape(leading),
        event_shape=event_shape,
        direction="data-to-base",
        divergence_method="hutchinson-state-jvp",
        num_probes=resolved.num_probes,
        probe_distribution=resolved.distribution,
        flow_id=flow_id,
    )


__all__ = [
    "ContinuousFlowDensityResult",
    "ContinuousFlowLaw",
    "estimate_continuous_flow_log_prob",
]

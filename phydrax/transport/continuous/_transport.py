#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ...dynamics._evolution import AbstractEvolution, EVOLUTION_SUCCESS


def _sample_shape(value, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    return shape


class ContinuousTransportSample(StrictModule):
    """Samples and per-solve evidence from deterministic continuous transport."""

    source_states: Array
    final_states: Array
    valid: Array
    status: Array
    backend_status: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    density_measure_kind: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    evolution_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_states: ArrayLike,
        final_states: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        backend_status: ArrayLike,
        sample_shape,
        event_shape,
        density_measure_kind: str,
        system_id: str,
        evolution_id: str,
        approximation_id: str,
        transport_id: str,
    ):
        samples = _sample_shape(sample_shape)
        events = tuple(int(size) for size in event_shape)
        source = jnp.asarray(source_states)
        final = jnp.asarray(final_states, dtype=source.dtype)
        expected = samples + events
        if source.shape != expected or final.shape != expected:
            raise ValueError(
                f"Continuous transport states must have shape {expected}; "
                f"got {source.shape} and {final.shape}."
            )
        validity = jnp.asarray(valid, dtype=bool)
        statuses = jnp.asarray(status, dtype=jnp.int32)
        backend = jnp.asarray(backend_status)
        if (
            validity.shape != samples
            or statuses.shape != samples
            or backend.shape != samples
        ):
            raise ValueError(
                "Continuous transport status arrays must match sample_shape."
            )
        for name, identifier in (
            ("density_measure_kind", density_measure_kind),
            ("system_id", system_id),
            ("evolution_id", evolution_id),
            ("approximation_id", approximation_id),
            ("transport_id", transport_id),
        ):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{name} must be a non-empty string.")
        self.source_states = source
        self.final_states = final
        self.valid = validity
        self.status = statuses
        self.backend_status = backend
        self.sample_shape = samples
        self.event_shape = events
        self.density_measure_kind = density_measure_kind
        self.system_id = system_id
        self.evolution_id = evolution_id
        self.approximation_id = approximation_id
        self.transport_id = transport_id

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid) & jnp.all(self.status == EVOLUTION_SUCCESS)

    @property
    def num_samples(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1


class ContinuousTransport(StrictModule):
    """Push a normalized source law through one declared deterministic evolution."""

    source_law: AbstractProbabilityLaw
    evolution: AbstractEvolution
    source_coordinate: Array
    target_coordinate: Array
    args: Any
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_law: AbstractProbabilityLaw,
        evolution: AbstractEvolution,
        /,
        *,
        source_coordinate: ArrayLike = 0.0,
        target_coordinate: ArrayLike = 1.0,
        args: Any = None,
        transport_id: str | None = None,
    ):
        if not isinstance(source_law, AbstractProbabilityLaw):
            raise TypeError("source_law must implement AbstractProbabilityLaw.")
        if not isinstance(evolution, AbstractEvolution):
            raise TypeError("evolution must implement AbstractEvolution.")
        if tuple(source_law.batch_shape):
            raise ValueError(
                "ContinuousTransport initially requires an unbatched source law."
            )
        if tuple(source_law.event_shape) != tuple(evolution.state_layout.shape):
            raise ValueError(
                "Source-law event shape must match the evolution state layout exactly."
            )
        source = jnp.asarray(source_coordinate, dtype=float).reshape(())
        target = jnp.asarray(target_coordinate, dtype=float).reshape(())
        if not bool(jnp.isfinite(source) & jnp.isfinite(target)):
            raise ValueError("Continuous transport coordinates must be finite.")
        if not bool(target > source):
            raise ValueError("Continuous transport target coordinate must exceed source.")
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "continuous-transport-v1",
                    "source_law": f"{type(source_law).__module__}.{type(source_law).__name__}",
                    "event_shape": list(source_law.event_shape),
                    "evolution_id": evolution.evolution_id,
                    "source_coordinate": float(source),
                    "target_coordinate": float(target),
                }
            )
            if transport_id is None
            else str(transport_id)
        )
        if not resolved_id:
            raise ValueError("transport_id must be non-empty.")
        self.source_law = source_law
        self.evolution = evolution
        self.source_coordinate = source
        self.target_coordinate = target
        self.args = args
        self.transport_id = resolved_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self.source_law.event_shape)

    def sample_with_diagnostics(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> ContinuousTransportSample:
        samples = _sample_shape(sample_shape)
        source = jnp.asarray(self.source_law.sample(key, samples))
        expected = samples + self.event_shape
        if source.shape != expected:
            raise ValueError(
                f"Source law returned shape {source.shape}; expected {expected}."
            )
        count = prod(samples) if samples else 1
        flat_source = source.reshape((count,) + self.event_shape)

        def advance(state):
            return self.evolution.advance(
                state,
                self.source_coordinate,
                self.target_coordinate,
                self.args,
            )

        steps = jax.vmap(advance)(flat_source)
        final = steps.final_state.reshape(expected)
        valid = steps.valid.reshape(samples)
        status = steps.status.reshape(samples)
        backend_status = steps.backend_status.reshape(samples)
        return ContinuousTransportSample(
            source_states=source,
            final_states=final,
            valid=valid,
            status=status,
            backend_status=backend_status,
            sample_shape=samples,
            event_shape=self.event_shape,
            density_measure_kind=self.source_law.density_measure_kind,
            system_id=self.evolution.system.system_id,
            evolution_id=self.evolution.evolution_id,
            approximation_id=self.evolution.approximation_id,
            transport_id=self.transport_id,
        )

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        result = self.sample_with_diagnostics(key, sample_shape)
        return eqx.error_if(
            result.final_states,
            ~result.successful,
            "Continuous transport evolution failed for at least one source sample.",
        )


__all__ = ["ContinuousTransport", "ContinuousTransportSample"]

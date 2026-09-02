#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Key

from .._sampling import get_sampler
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._measure import BaseMeasure, ExactMass
from ._scalar import AbstractScalarDomain
from ._selection import Fixed, Interior, Selection


ReferenceMeasure = Literal["uniform", "standard-normal"]


def open_unit_interval(values: Any, /) -> Array:
    """Map probabilities into the representable open unit interval."""
    unit = jnp.asarray(values, dtype=float)
    epsilon = jnp.finfo(unit.dtype).eps
    return jnp.clip(unit, epsilon, 1.0 - epsilon)


class ReferenceTransportEvidence(StrictModule, NonTrainableState):
    """Preparation evidence for one declared exact reference transport."""

    provider: str = eqx.field(static=True)
    reference_measure: ReferenceMeasure = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_round_trip_residual: float = eqx.field(static=True)
    orientation_preserving: bool = eqx.field(static=True)
    tail_open: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider: str,
        reference_measure: ReferenceMeasure,
        event_shape: tuple[int, ...],
        maximum_round_trip_residual: float = 0.0,
        orientation_preserving: bool = True,
        tail_open: bool = False,
    ):
        if not provider:
            raise ValueError("Reference transport provider identity must be nonempty.")
        if reference_measure not in ("uniform", "standard-normal"):
            raise ValueError("Unsupported reference measure.")
        residual = float(maximum_round_trip_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError(
                "Round-trip residual evidence must be finite and nonnegative."
            )
        self.provider = provider
        self.reference_measure = reference_measure
        self.event_shape = tuple(int(size) for size in event_shape)
        self.maximum_round_trip_residual = residual
        self.orientation_preserving = bool(orientation_preserving)
        self.tail_open = bool(tail_open)


class ReferenceTransport(StrictModule, NonTrainableState):
    """Typed bijection between a canonical reference law and physical events."""

    reference_measure: ReferenceMeasure = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    forward_map: Callable = eqx.field(static=True)
    inverse_map: Callable = eqx.field(static=True)
    log_abs_det_jacobian: Callable | None = eqx.field(static=True)
    evidence: ReferenceTransportEvidence

    def __init__(
        self,
        *,
        reference_measure: ReferenceMeasure,
        forward: Callable,
        inverse: Callable,
        event_shape: tuple[int, ...] = (),
        log_abs_det_jacobian: Callable | None = None,
        evidence: ReferenceTransportEvidence | None = None,
    ):
        if reference_measure not in ("uniform", "standard-normal"):
            raise ValueError("reference_measure must be 'uniform' or 'standard-normal'.")
        if not callable(forward) or not callable(inverse):
            raise TypeError(
                "Reference transport forward and inverse maps must be callable."
            )
        if log_abs_det_jacobian is not None and not callable(log_abs_det_jacobian):
            raise TypeError("log_abs_det_jacobian must be callable when supplied.")
        shape = tuple(int(size) for size in event_shape)
        if any(size < 1 for size in shape):
            raise ValueError("Reference transport event dimensions must be positive.")
        resolved_evidence = evidence or ReferenceTransportEvidence(
            provider="explicit",
            reference_measure=reference_measure,
            event_shape=shape,
        )
        if (
            resolved_evidence.reference_measure != reference_measure
            or resolved_evidence.event_shape != shape
        ):
            raise ValueError("Reference transport evidence does not match the transport.")
        self.reference_measure = reference_measure
        self.event_shape = shape
        self.forward_map = forward
        self.inverse_map = inverse
        self.log_abs_det_jacobian = log_abs_det_jacobian
        self.evidence = resolved_evidence

    def from_reference(self, value: Any, /) -> Array:
        return jnp.asarray(self.forward_map(value), dtype=float)

    def to_reference(self, value: Any, /) -> Array:
        return jnp.asarray(self.inverse_map(value), dtype=float)


@runtime_checkable
class ReferenceTransportProvider(Protocol):
    """Law that explicitly supplies its canonical transport and hypotheses."""

    def reference_transport(self) -> ReferenceTransport: ...


@runtime_checkable
class ContinuousScalarQuantileLaw(Protocol):
    """Declared continuous strictly increasing scalar CDF/quantile pair."""

    @property
    def continuous_strictly_increasing_cdf(self) -> bool: ...

    def cdf(self, value: Any, /) -> Array: ...

    def icdf(self, probability: Any, /) -> Array: ...


@runtime_checkable
class _ProbabilityLaw(Protocol):
    @property
    def support(self) -> tuple[Array, Array] | None: ...

    def sample(
        self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()
    ) -> Array: ...

    def icdf(self, probability: Any, /) -> Array: ...

    def log_prob(self, value: Any, /) -> Array: ...

    def contains(self, value: Any, /) -> Array: ...

    def equivalent(self, other: object, /) -> bool: ...


def _validate_transport(transport: ReferenceTransport, /) -> ReferenceTransport:
    if not isinstance(transport, ReferenceTransport):
        raise TypeError("reference_transport() must return ReferenceTransport.")
    if transport.event_shape != ():
        raise ValueError("ProbabilityDomain currently requires a scalar-event transport.")
    reference = (
        jnp.asarray([-0.75, 0.0, 0.75])
        if transport.reference_measure == "uniform"
        else jnp.asarray([-1.0, 0.0, 1.0])
    )
    physical = transport.from_reference(reference)
    recovered = transport.to_reference(physical)
    physical_host = np.asarray(physical)
    recovered_host = np.asarray(recovered)
    if physical.shape != reference.shape or recovered.shape != reference.shape:
        raise ValueError("Reference transport must preserve scalar event batch shape.")
    if not np.all(np.isfinite(physical_host)) or not np.all(np.isfinite(recovered_host)):
        raise ValueError("Reference transport produced nonfinite preparation probes.")
    residual = float(np.max(np.abs(recovered_host - np.asarray(reference))))
    tolerance = 128.0 * np.finfo(np.asarray(reference).dtype).eps
    if residual > tolerance:
        raise ValueError("Reference transport failed its preparation round-trip check.")
    evidence = ReferenceTransportEvidence(
        provider=transport.evidence.provider,
        reference_measure=transport.reference_measure,
        event_shape=transport.event_shape,
        maximum_round_trip_residual=residual,
        orientation_preserving=transport.evidence.orientation_preserving,
        tail_open=transport.evidence.tail_open,
    )
    return eqx.tree_at(lambda current: current.evidence, transport, evidence)


def construct_reference_transport(distribution: Any, /) -> ReferenceTransport:
    """Construct the bounded canonical transport declared by a supported law."""
    if isinstance(distribution, ReferenceTransportProvider):
        return _validate_transport(distribution.reference_transport())
    if not isinstance(distribution, ContinuousScalarQuantileLaw):
        raise TypeError(
            "The law does not declare an exact reference transport or a continuous "
            "strictly increasing scalar CDF/quantile pair."
        )
    if distribution.continuous_strictly_increasing_cdf is not True:
        raise ValueError(
            "Quantile transport construction requires the declared CDF hypothesis."
        )

    def forward(reference):
        probability = open_unit_interval(0.5 * (jnp.asarray(reference) + 1.0))
        return distribution.icdf(probability)

    def inverse(value):
        return 2.0 * jnp.asarray(distribution.cdf(value)) - 1.0

    return _validate_transport(
        ReferenceTransport(
            reference_measure="uniform",
            forward=forward,
            inverse=inverse,
            evidence=ReferenceTransportEvidence(
                provider=type(distribution).__qualname__,
                reference_measure="uniform",
                event_shape=(),
                tail_open=True,
            ),
        )
    )


class ProbabilityDomain(AbstractScalarDomain):
    """A labeled scalar random variable carrying unit probability measure."""

    distribution: _ProbabilityLaw
    _label: str
    _reference_transport: ReferenceTransport | None

    def __init__(
        self,
        distribution: _ProbabilityLaw,
        /,
        *,
        label: str,
        transport: ReferenceTransport | None = None,
    ):
        if not isinstance(distribution, _ProbabilityLaw):
            raise TypeError(
                "distribution must provide sample, icdf, log_prob, contains, support, "
                "and equivalent."
            )
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        resolved_transport = (
            _validate_transport(transport)
            if transport is not None
            else construct_reference_transport(distribution)
            if isinstance(
                distribution,
                (ReferenceTransportProvider, ContinuousScalarQuantileLaw),
            )
            else None
        )
        self.distribution = distribution
        self._label = label
        self._reference_transport = resolved_transport

    @property
    def label(self) -> str:
        return self._label

    @property
    def measure(self) -> Array:
        return jnp.asarray(1.0, dtype=float)

    @property
    def reference_transport(self) -> ReferenceTransport:
        if self._reference_transport is None:
            raise ValueError(
                "This probability law has no declared exact reference transport; "
                "use a discrete/weighted target or supply a certified transport."
            )
        return self._reference_transport

    def _component_base_measure(self, selection: Selection, /) -> BaseMeasure:
        if isinstance(selection, Interior):
            return BaseMeasure("probability", ExactMass(1.0), normalized=True)
        if isinstance(selection, Fixed):
            if selection.value.ndim != 0:
                raise ValueError("A probability Fixed selection requires a scalar value.")
            if not bool(jnp.asarray(self._contains(selection.value))):
                raise ValueError(
                    f"Fixed value {selection.value} lies outside probability support."
                )
            return BaseMeasure("dirac", ExactMass(1.0), normalized=True)
        raise TypeError(
            "Probability factors support only Interior or explicit Fixed selections."
        )

    @property
    def bounds(self) -> Iterator[Array]:
        support = self.distribution.support
        if support is None:
            raise ValueError("This probability distribution has unbounded support.")
        return iter(support)

    def fixed(self, which: str, /) -> Array:
        support = self.distribution.support
        if support is None:
            raise ValueError(
                "Endpoint-dependent components are unavailable for unbounded distributions."
            )
        if which == "start":
            return jnp.asarray(support[0], dtype=float).reshape(())
        if which == "end":
            return jnp.asarray(support[1], dtype=float).reshape(())
        raise ValueError("fixed(which) must be 'start' or 'end'.")

    def sample(
        self,
        num_points: int,
        *,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""],
    ) -> Array:
        count = int(num_points)
        if count < 0:
            raise ValueError("num_points must be non-negative.")
        if sampler == "uniform":
            return jnp.asarray(
                self.distribution.sample(key, sample_shape=(count,)), dtype=float
            )
        unit = open_unit_interval(get_sampler(sampler)(count, 1, key)).reshape((count,))
        return jnp.asarray(self.distribution.icdf(unit), dtype=float)

    def _same_factor_support(self, other: object, /) -> bool:
        return isinstance(other, ProbabilityDomain) and self.distribution.equivalent(
            other.distribution
        )

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        return jnp.asarray(self.distribution.contains(points), dtype=bool)


__all__ = [
    "ContinuousScalarQuantileLaw",
    "ProbabilityDomain",
    "ReferenceMeasure",
    "ReferenceTransport",
    "ReferenceTransportEvidence",
    "ReferenceTransportProvider",
    "construct_reference_transport",
    "open_unit_interval",
]

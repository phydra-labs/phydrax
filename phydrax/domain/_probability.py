#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Key

from .._sampling import get_sampler
from ._measure import BaseMeasure, ExactMass
from ._scalar import AbstractScalarDomain
from ._selection import Fixed, Interior, Selection


def open_unit_interval(values: Any, /) -> Array:
    unit = jnp.asarray(values, dtype=float)
    epsilon = jnp.finfo(unit.dtype).eps
    return jnp.clip(unit, epsilon, 1.0 - epsilon)


@runtime_checkable
class ReferenceDistribution(Protocol):
    """Distribution with an explicit canonical-coordinate bijection."""

    @property
    def reference_measure(self) -> Literal["uniform", "standard-normal"]: ...

    def to_reference(self, value: Any, /) -> Array: ...

    def from_reference(self, value: Any, /) -> Array: ...


class ProbabilityDomain(AbstractScalarDomain):
    """A labeled scalar random variable carrying unit probability measure."""

    distribution: Any
    _label: str

    def __init__(self, distribution: Any, /, *, label: str):
        required = ("sample", "icdf", "log_prob", "contains")
        if any(not callable(getattr(distribution, name, None)) for name in required):
            raise TypeError(
                "distribution must provide sample, icdf, log_prob, and contains methods."
            )
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        self.distribution = distribution
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def measure(self) -> Array:
        return jnp.asarray(1.0, dtype=float)
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
        support = getattr(self.distribution, "support", None)
        if support is None:
            raise ValueError("This probability distribution has unbounded support.")
        return iter(support)

    def fixed(self, which: str, /) -> Array:
        support = getattr(self.distribution, "support", None)
        if support is None:
            raise ValueError(
                "Endpoint-dependent components are unavailable for unbounded distributions."
            )
        if which == "start":
            return jnp.asarray(support[0], dtype=float).reshape(())
        if which == "end":
            return jnp.asarray(support[1], dtype=float).reshape(())
        raise ValueError("fixed(which) must be 'start' or 'end'.")

    @property
    def supports_reference_transform(self) -> bool:
        return isinstance(self.distribution, ReferenceDistribution)

    @property
    def reference_measure(self) -> Literal["uniform", "standard-normal"]:
        if not isinstance(self.distribution, ReferenceDistribution):
            raise ValueError(
                "This probability distribution has no canonical reference transform."
            )
        measure = self.distribution.reference_measure
        if measure not in ("uniform", "standard-normal"):
            raise ValueError("reference_measure must be 'uniform' or 'standard-normal'.")
        return measure

    def to_reference(self, value: Any, /) -> Array:
        if not isinstance(self.distribution, ReferenceDistribution):
            raise ValueError(
                "This probability distribution has no canonical reference transform."
            )
        return jnp.asarray(self.distribution.to_reference(value), dtype=float)

    def from_reference(self, value: Any, /) -> Array:
        if not isinstance(self.distribution, ReferenceDistribution):
            raise ValueError(
                "This probability distribution has no canonical reference transform."
            )
        return jnp.asarray(self.distribution.from_reference(value), dtype=float)

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
        if not isinstance(other, ProbabilityDomain):
            return False
        equivalent = getattr(self.distribution, "equivalent", None)
        if callable(equivalent):
            return bool(equivalent(other.distribution))
        return self.distribution == other.distribution

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        return jnp.asarray(self.distribution.contains(points), dtype=bool)


__all__ = ["open_unit_interval", "ProbabilityDomain", "ReferenceDistribution"]

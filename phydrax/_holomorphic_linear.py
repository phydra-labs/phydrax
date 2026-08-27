#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from operator import index
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


MultiIndex = tuple[int, ...]


def _canonical_multi_index(value: Sequence[int], dimension: int, /) -> MultiIndex:
    index = tuple(int(item) for item in value)
    if len(index) != dimension or any(item < 0 for item in index):
        raise ValueError(
            f"Holomorphic multi-indices must contain {dimension} nonnegative entries."
        )
    return index


class HolomorphicMultiIndexSet(StrictModule, NonTrainableState):
    """Canonical finite derivative or monomial multi-index set."""

    indices: tuple[MultiIndex, ...] = eqx.field(static=True)
    complex_dimension: int = eqx.field(static=True)
    maximum_total_order: int = eqx.field(static=True)
    maximum_count: int = eqx.field(static=True)
    downward_closed: bool = eqx.field(static=True)
    index_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_dimension: int,
        indices: Sequence[Sequence[int]],
        /,
        *,
        require_downward_closed: bool = False,
        maximum_count: int = 10_000,
    ):
        dimension = int(complex_dimension)
        if isinstance(maximum_count, bool):
            raise TypeError("maximum_count must be an integer.")
        maximum_count_ = index(maximum_count)
        if dimension <= 0 or maximum_count_ <= 0:
            raise ValueError("Holomorphic multi-index resource limits are invalid.")
        resolved = tuple(
            sorted(
                {_canonical_multi_index(value, dimension) for value in tuple(indices)},
                key=lambda value: (sum(value), value),
            )
        )
        zero = (0,) * dimension
        if not resolved or zero not in resolved:
            raise ValueError("Holomorphic multi-index sets must contain the zero index.")
        if len(resolved) > maximum_count_:
            raise ValueError(
                f"Holomorphic multi-index count {len(resolved)} exceeds "
                f"the configured maximum {maximum_count_}."
            )
        available = set(resolved)
        downward = all(
            item[axis] == 0
            or tuple(
                value - 1 if current == axis else value
                for current, value in enumerate(item)
            )
            in available
            for item in resolved
            for axis in range(dimension)
        )
        if require_downward_closed and not downward:
            raise ValueError("Holomorphic multi-index set must be downward closed.")
        maximum = max(sum(value) for value in resolved)
        self.indices = resolved
        self.complex_dimension = dimension
        self.maximum_total_order = maximum
        self.maximum_count = maximum_count_
        self.downward_closed = downward
        self.index_set_id = canonical_fingerprint(
            {
                "kind": "holomorphic-multi-index-set",
                "complex_dimension": dimension,
                "indices": [list(value) for value in resolved],
                "maximum_count": maximum_count_,
                "downward_closed": downward,
            }
        )

    @classmethod
    def total_degree(
        cls,
        complex_dimension: int,
        maximum_total_order: int,
        /,
        *,
        maximum_count: int = 10_000,
    ) -> HolomorphicMultiIndexSet:
        dimension = int(complex_dimension)
        maximum = int(maximum_total_order)
        if dimension <= 0 or maximum < 0:
            raise ValueError("Total-degree index dimensions and order are invalid.")
        indices = tuple(
            value
            for value in product(range(maximum + 1), repeat=dimension)
            if sum(value) <= maximum
        )
        return cls(
            dimension,
            indices,
            require_downward_closed=True,
            maximum_count=maximum_count,
        )

    @property
    def count(self) -> int:
        return len(self.indices)

    @property
    def nonzero_indices(self) -> tuple[MultiIndex, ...]:
        zero = (0,) * self.complex_dimension
        return tuple(value for value in self.indices if value != zero)

    def contains(self, multi_index: Sequence[int], /) -> bool:
        return (
            _canonical_multi_index(
                multi_index,
                self.complex_dimension,
            )
            in self.indices
        )


class HolomorphicMultiJet(StrictModule):
    """Complex value and explicitly indexed multivariable holomorphic derivatives."""

    value: Array
    derivatives: tuple[Array, ...]
    index_set: HolomorphicMultiIndexSet

    def __init__(
        self,
        value: ArrayLike,
        derivatives: Sequence[ArrayLike],
        index_set: HolomorphicMultiIndexSet,
        /,
    ):
        if not isinstance(index_set, HolomorphicMultiIndexSet):
            raise TypeError("index_set must be HolomorphicMultiIndexSet.")
        value_ = jnp.asarray(value)
        derivatives_ = tuple(jnp.asarray(item) for item in derivatives)
        if len(derivatives_) != len(index_set.nonzero_indices):
            raise ValueError("Holomorphic multijet derivatives must match the index set.")
        if any(item.shape != value_.shape for item in derivatives_):
            raise ValueError(
                "Holomorphic multijet derivatives must match the value shape."
            )
        self.value = value_
        self.derivatives = derivatives_
        self.index_set = index_set

    def derivative(self, multi_index: Sequence[int], /) -> Array:
        index = _canonical_multi_index(
            multi_index,
            self.index_set.complex_dimension,
        )
        zero = (0,) * self.index_set.complex_dimension
        if index == zero:
            return self.value
        if index not in self.index_set.nonzero_indices:
            raise ValueError("Requested derivative is unavailable in this multijet.")
        return self.derivatives[self.index_set.nonzero_indices.index(index)]


class HolomorphicLinearFrameCertificate(StrictModule, NonTrainableState):
    """Construction evidence for a finite real-coordinate holomorphic frame."""

    complex_input_size: int = eqx.field(static=True)
    complex_output_size: int = eqx.field(static=True)
    real_coefficient_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)
    basis_construction: str = eqx.field(static=True)
    coefficient_mode: str = eqx.field(static=True)
    construction_dependencies: tuple[str, ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        complex_input_size: int,
        complex_output_size: int,
        real_coefficient_count: int,
        maximum_derivative_order: int,
        normalization_id: str,
        basis_construction: str,
        construction_dependencies: Sequence[str] = (),
        coefficient_mode: str = "real-cartesian-linear-frame",
    ):
        input_size = int(complex_input_size)
        output_size = int(complex_output_size)
        coefficient_count = int(real_coefficient_count)
        derivative_order = int(maximum_derivative_order)
        identifiers = (
            str(normalization_id),
            str(basis_construction),
            str(coefficient_mode),
        )
        dependencies = tuple(str(value) for value in construction_dependencies)
        if min(input_size, output_size, coefficient_count) <= 0:
            raise ValueError("Holomorphic frame dimensions must be positive.")
        if derivative_order < 0:
            raise ValueError("Holomorphic frame derivative order must be nonnegative.")
        if any(not value for value in identifiers) or any(
            not value for value in dependencies
        ):
            raise ValueError("Holomorphic frame identifiers must be nonempty.")
        self.complex_input_size = input_size
        self.complex_output_size = output_size
        self.real_coefficient_count = coefficient_count
        self.maximum_derivative_order = derivative_order
        self.normalization_id = identifiers[0]
        self.basis_construction = identifiers[1]
        self.coefficient_mode = identifiers[2]
        self.construction_dependencies = dependencies
        self.frame_id = canonical_fingerprint(
            {
                "kind": "holomorphic-linear-frame-certificate",
                "complex_input_size": input_size,
                "complex_output_size": output_size,
                "real_coefficient_count": coefficient_count,
                "maximum_derivative_order": derivative_order,
                "normalization_id": identifiers[0],
                "basis_construction": identifiers[1],
                "coefficient_mode": identifiers[2],
                "construction_dependencies": list(dependencies),
            }
        )


@runtime_checkable
class HolomorphicLinearFrame(Protocol):
    def linear_frame_certificate(self) -> HolomorphicLinearFrameCertificate: ...

    def basis_derivative(
        self,
        coordinates: ArrayLike,
        multi_index: Sequence[int],
        /,
    ) -> Array: ...


@runtime_checkable
class MultivariableHolomorphicPotentialProvider(Protocol):
    def __call__(self, coordinates: ArrayLike, /) -> Array: ...

    def holomorphic_certificate(self): ...

    def multi_jet(
        self,
        coordinates: ArrayLike,
        index_set: HolomorphicMultiIndexSet,
        /,
    ) -> HolomorphicMultiJet: ...


__all__ = [
    "HolomorphicLinearFrame",
    "HolomorphicLinearFrameCertificate",
    "HolomorphicMultiIndexSet",
    "HolomorphicMultiJet",
    "MultiIndex",
    "MultivariableHolomorphicPotentialProvider",
]

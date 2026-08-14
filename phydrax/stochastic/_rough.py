#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._fractional import FractionalGaussianRealization


def _digest_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _rough_control_id(
    times: Array, first: Array, second: Array, driver_id: str | None
) -> str:
    digest = hashlib.sha256(b"phydrax-geometric-rough-control\0")
    _digest_array(digest, times)
    _digest_array(digest, first)
    _digest_array(digest, second)
    digest.update(repr(driver_id).encode("utf-8"))
    return digest.hexdigest()


class AbstractRoughControl(StrictModule):
    """Finite-depth geometric control on one explicit integration partition."""

    times: AbstractAttribute[Array]
    realization: AbstractAttribute[FractionalGaussianRealization | None]
    sample_shape: AbstractAttribute[tuple[int, ...]]
    dimension: AbstractAttribute[int]
    num_steps: AbstractAttribute[int]
    depth: AbstractAttribute[int]
    control_id: AbstractAttribute[str]

    @property
    @abstractmethod
    def levels(self) -> tuple[Array, ...]:
        """Per-interval tensor-signature levels, ordered from degree one."""
        raise NotImplementedError

    @abstractmethod
    def signature(self, start_index: int, end_index: int, /) -> tuple[Array, ...]:
        """Aggregate a non-empty partition slice using Chen multiplication."""
        raise NotImplementedError

    @property
    def terminal_signature(self) -> tuple[Array, ...]:
        return self.signature(0, self.num_steps)


def compose_rough_path_segments(
    first_left: ArrayLike,
    second_left: ArrayLike,
    first_right: ArrayLike,
    second_right: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Compose adjacent step-2 signatures using Chen's identity."""
    left = jnp.asarray(first_left)
    right = jnp.asarray(first_right)
    left_second = jnp.asarray(second_left)
    right_second = jnp.asarray(second_right)
    if left.shape != right.shape or not left.shape:
        raise ValueError("First-level segments must have one matching driver axis.")
    expected = left.shape + (left.shape[-1],)
    if left_second.shape != expected or right_second.shape != expected:
        raise ValueError("Second-level segments must append two driver axes.")
    return (
        left + right,
        left_second + right_second + jnp.einsum("...i,...j->...ij", left, right),
    )


class GeometricRoughPath(AbstractRoughControl):
    """Optimized depth-2 geometric rough control on one explicit partition."""

    times: Array
    first_level: Array
    second_level: Array
    realization: FractionalGaussianRealization | None
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    driver_id: str | None = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        first_level: ArrayLike,
        second_level: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int] = (),
        realization: FractionalGaussianRealization | None = None,
        driver_id: str | None = None,
    ):
        nodes = jnp.asarray(times, dtype=float)
        if nodes.ndim != 1 or int(nodes.size) < 2:
            raise ValueError("times must contain at least two partition nodes.")
        if bool(jnp.any(~jnp.isfinite(nodes))) or bool(jnp.any(jnp.diff(nodes) <= 0.0)):
            raise ValueError("times must be finite and strictly increasing.")
        samples = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("sample_shape dimensions must be positive.")
        first = jnp.asarray(first_level)
        second = jnp.asarray(second_level)
        step_count = int(nodes.size) - 1
        if first.ndim != len(samples) + 2:
            raise ValueError(
                "first_level must have shape sample_shape + (num_steps, dimension)."
            )
        dimension = int(first.shape[-1])
        expected_first = samples + (step_count, dimension)
        expected_second = samples + (step_count, dimension, dimension)
        if first.shape != expected_first or second.shape != expected_second:
            raise ValueError(
                f"rough levels must have shapes {expected_first} and {expected_second}."
            )
        if (
            dimension <= 0
            or bool(jnp.any(~jnp.isfinite(first)))
            or bool(jnp.any(~jnp.isfinite(second)))
        ):
            raise ValueError("rough path levels must be non-empty and finite.")
        symmetric = 0.5 * (second + jnp.swapaxes(second, -1, -2))
        geometric = 0.5 * jnp.einsum("...i,...j->...ij", first, first)
        tolerance = 1000.0 * jnp.finfo(second.dtype).eps
        if not bool(jnp.allclose(symmetric, geometric, rtol=1e-7, atol=tolerance)):
            raise ValueError(
                "second_level must satisfy the step-2 geometric symmetry identity."
            )
        if realization is not None:
            if not isinstance(realization, FractionalGaussianRealization):
                raise TypeError(
                    "realization must be a FractionalGaussianRealization or None."
                )
            if realization.sample_shape != samples:
                raise ValueError(
                    "realization sample_shape must match rough path samples."
                )
            if realization.process.dimension != dimension:
                raise ValueError("realization dimension must match the rough path.")
            if not jnp.array_equal(realization.grid, nodes):
                raise ValueError("realization grid must match rough path times.")
            resolved_driver_id = realization.realization_id
        else:
            resolved_driver_id = None if driver_id is None else str(driver_id)
            if resolved_driver_id == "":
                raise ValueError("driver_id must be non-empty or None.")
        self.times = nodes
        self.first_level = first
        self.second_level = second
        self.realization = realization
        self.sample_shape = samples
        self.dimension = dimension
        self.num_steps = step_count
        self.depth = 2
        self.driver_id = resolved_driver_id
        self.control_id = _rough_control_id(nodes, first, second, resolved_driver_id)

    @property
    def levels(self) -> tuple[Array, Array]:
        return self.first_level, self.second_level

    @classmethod
    def from_values(
        cls,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int] = (),
        realization: FractionalGaussianRealization | None = None,
        driver_id: str | None = None,
    ) -> GeometricRoughPath:
        """Lift one piecewise-linear path to its canonical geometric second level."""
        samples = tuple(int(size) for size in sample_shape)
        path_values = jnp.asarray(values)
        nodes = jnp.asarray(times, dtype=float)
        if path_values.ndim != len(samples) + 2:
            raise ValueError(
                "values must have shape sample_shape + (num_times, dimension)."
            )
        if (
            path_values.shape[: len(samples)] != samples
            or path_values.shape[len(samples)] != nodes.size
        ):
            raise ValueError("values must align with sample_shape and times.")
        increments = jnp.diff(path_values, axis=len(samples))
        second = 0.5 * jnp.einsum("...i,...j->...ij", increments, increments)
        return cls(
            nodes,
            increments,
            second,
            sample_shape=samples,
            realization=realization,
            driver_id=driver_id,
        )

    @classmethod
    def from_fractional_gaussian(
        cls,
        realization: FractionalGaussianRealization,
        /,
    ) -> GeometricRoughPath:
        if not isinstance(realization, FractionalGaussianRealization):
            raise TypeError("realization must be a FractionalGaussianRealization.")
        return cls.from_values(
            realization.grid,
            realization.values,
            sample_shape=realization.sample_shape,
            realization=realization,
        )

    def signature(self, start_index: int, end_index: int, /) -> tuple[Array, Array]:
        """Aggregate a partition slice into one depth-2 signature."""
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end > self.num_steps or end <= start:
            raise ValueError(
                "signature indices must satisfy 0 <= start < end <= num_steps."
            )
        first = self.first_level[..., start:end, :]
        second = self.second_level[..., start:end, :, :]
        flat_first = first.reshape((-1, end - start, self.dimension))
        flat_second = second.reshape((-1, end - start, self.dimension, self.dimension))

        def one_path(path_first, path_second):
            def combine(carry, item):
                return compose_rough_path_segments(*carry, *item), None

            initial = (
                jnp.zeros((self.dimension,), dtype=path_first.dtype),
                jnp.zeros((self.dimension, self.dimension), dtype=path_second.dtype),
            )
            return jax.lax.scan(combine, initial, (path_first, path_second))[0]

        aggregated_first, aggregated_second = jax.vmap(one_path)(flat_first, flat_second)
        return (
            aggregated_first.reshape(self.sample_shape + (self.dimension,)),
            aggregated_second.reshape(
                self.sample_shape + (self.dimension, self.dimension)
            ),
        )

    def coarsen(self, node_indices: Sequence[int], /) -> GeometricRoughPath:
        """Coarsen the partition exactly while preserving Chen-consistent levels."""
        indices = tuple(int(index) for index in node_indices)
        if (
            len(indices) < 2
            or indices[0] != 0
            or indices[-1] != self.num_steps
            or any(right <= left for left, right in pairwise(indices))
        ):
            raise ValueError(
                "node_indices must increase from zero through the final node."
            )
        signatures = tuple(
            self.signature(left, right) for left, right in pairwise(indices)
        )
        step_axis = len(self.sample_shape)
        first = jnp.stack(tuple(value[0] for value in signatures), axis=step_axis)
        second = jnp.stack(tuple(value[1] for value in signatures), axis=step_axis)
        return GeometricRoughPath(
            self.times[jnp.asarray(indices)],
            first,
            second,
            sample_shape=self.sample_shape,
            driver_id=self.driver_id,
        )


__all__ = [
    "AbstractRoughControl",
    "GeometricRoughPath",
    "compose_rough_path_segments",
]

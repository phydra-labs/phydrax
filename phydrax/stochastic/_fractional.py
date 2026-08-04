#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule


FractionalGaussianInterpolation: TypeAlias = Literal["grid", "linear"]


def _digest_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _digest(prefix: bytes, *parts) -> str:
    digest = hashlib.sha256(prefix)
    for part in parts:
        if isinstance(part, (jax.Array, np.ndarray)):
            _digest_array(digest, part)
        else:
            digest.update(repr(part).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _scalar_key(value: Key[Array, ""], /) -> Array:
    if jr.key_data(value).shape != (2,):
        raise ValueError("FractionalGaussianRealization requires one scalar PRNG key.")
    return value


def _samples(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    return shape


class FractionalGaussianProcess(StrictModule):
    """Vector fractional Brownian motion with independent scaled components.

    The process is anchored at ``reference_time``. Component ``j`` has covariance
    ``scale[j]² / 2 * (|t-a|²ᴴ + |s-a|²ᴴ - |t-s|²ᴴ)`` and an optional linear drift.
    """

    scale: Array
    drift: Array
    hurst: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    reference_time: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        hurst: float,
        scale: ArrayLike = 1.0,
        /,
        *,
        dimension: int | None = None,
        drift: ArrayLike = 0.0,
        reference_time: float = 0.0,
        process_id: str | None = None,
    ):
        exponent = float(hurst)
        if not isfinite(exponent) or not 0.0 < exponent < 1.0:
            raise ValueError("hurst must be finite and lie strictly between 0 and 1.")
        scale_value = jnp.asarray(scale, dtype=float)
        if scale_value.ndim > 1:
            raise ValueError("scale must be scalar or a rank-1 component vector.")
        if scale_value.ndim == 0:
            resolved_dimension = 1 if dimension is None else int(dimension)
            scale_value = jnp.broadcast_to(scale_value, (resolved_dimension,))
        else:
            resolved_dimension = int(scale_value.size)
            if dimension is not None and int(dimension) != resolved_dimension:
                raise ValueError("dimension must match the scale vector length.")
        if resolved_dimension <= 0:
            raise ValueError("dimension must be positive.")
        if bool(jnp.any(~jnp.isfinite(scale_value))) or bool(jnp.any(scale_value <= 0.0)):
            raise ValueError("scale values must be finite and positive.")
        drift_value = jnp.asarray(drift, dtype=float)
        if drift_value.ndim == 0:
            drift_value = jnp.broadcast_to(drift_value, (resolved_dimension,))
        if drift_value.shape != (resolved_dimension,):
            raise ValueError("drift must be scalar or have shape (dimension,).")
        if bool(jnp.any(~jnp.isfinite(drift_value))):
            raise ValueError("drift values must be finite.")
        anchor = float(reference_time)
        if not isfinite(anchor):
            raise ValueError("reference_time must be finite.")
        identifier = process_id or _digest(
            b"phydrax-fractional-gaussian-process\0",
            exponent,
            scale_value,
            drift_value,
            anchor,
        )
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("process_id must be a non-empty string.")
        self.scale = scale_value
        self.drift = drift_value
        self.hurst = exponent
        self.dimension = resolved_dimension
        self.reference_time = anchor
        self.process_id = identifier

    def time_covariance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Unscaled scalar fractional-Brownian covariance under broadcasting."""
        first = jnp.asarray(left, dtype=float)
        second = jnp.asarray(right, dtype=float)
        power = 2.0 * self.hurst
        return 0.5 * (
            jnp.abs(first - self.reference_time) ** power
            + jnp.abs(second - self.reference_time) ** power
            - jnp.abs(first - second) ** power
        )

    def covariance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Component covariance with trailing ``(dimension, dimension)`` axes."""
        base = self.time_covariance(left, right)
        return base[..., None, None] * jnp.diag(self.scale**2)

    def mean(self, times: ArrayLike, /) -> Array:
        values = jnp.asarray(times, dtype=float)
        return (values - self.reference_time)[..., None] * self.drift

    def increment_covariance(
        self,
        first_start: ArrayLike,
        first_end: ArrayLike,
        second_start: ArrayLike,
        second_end: ArrayLike,
        /,
    ) -> Array:
        base = (
            self.time_covariance(first_end, second_end)
            - self.time_covariance(first_end, second_start)
            - self.time_covariance(first_start, second_end)
            + self.time_covariance(first_start, second_start)
        )
        return base[..., None, None] * jnp.diag(self.scale**2)


class FractionalGaussianRealization(StrictModule):
    """Exact finite-grid realization of a fractional Gaussian process.

    Values on ``grid`` are sampled from the exact covariance matrix. Queries either
    require grid nodes or use one declared global linear interpolant; repeated and
    overlapping interval queries therefore share the same path and add exactly.
    """

    process: FractionalGaussianProcess
    root_key: Array
    path_indices: Array
    grid: Array
    covariance_factor: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    support: tuple[float, float] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: FractionalGaussianProcess,
        root_key: Key[Array, ""],
        grid: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int] = (),
        label: str | None = None,
        coupling_id: str | None = None,
        _path_indices: Array | None = None,
    ):
        if not isinstance(process, FractionalGaussianProcess):
            raise TypeError("process must be a FractionalGaussianProcess.")
        key = _scalar_key(root_key)
        nodes = jnp.asarray(grid, dtype=float)
        if nodes.ndim != 1 or int(nodes.size) < 2:
            raise ValueError("grid must be a rank-1 array with at least two nodes.")
        nodes_host = np.asarray(jax.device_get(nodes))
        if np.any(~np.isfinite(nodes_host)) or np.any(np.diff(nodes_host) <= 0.0):
            raise ValueError("grid nodes must be finite and strictly increasing.")
        if not np.isclose(nodes_host[0], process.reference_time):
            raise ValueError("grid must begin at the process reference_time.")
        samples = _samples(sample_shape)
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be a non-empty string or None.")
        if coupling_id is not None and (
            not isinstance(coupling_id, str) or not coupling_id
        ):
            raise ValueError("coupling_id must be a non-empty string or None.")
        count = prod(samples) if samples else 1
        if _path_indices is None:
            indices = jnp.arange(count, dtype=jnp.uint32).reshape(samples)
        else:
            indices = jnp.asarray(_path_indices, dtype=jnp.uint32)
            if tuple(indices.shape) != samples:
                raise ValueError("path indices must match sample_shape.")
        covariance = np.asarray(
            jax.device_get(process.time_covariance(nodes[:, None], nodes[None, :]))
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = (
            1000.0
            * np.finfo(covariance.dtype).eps
            * max(
                1.0,
                float(np.linalg.norm(covariance, ord=2)),
            )
        )
        if np.any(eigenvalues < -tolerance):
            raise ValueError("fractional Gaussian grid covariance is not semidefinite.")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        factor = jnp.asarray(eigenvectors * np.sqrt(eigenvalues)[None, :])
        support = (float(nodes_host[0]), float(nodes_host[-1]))
        resolved_coupling = coupling_id or _digest(
            b"phydrax-fractional-gaussian-coupling\0",
            jr.key_data(key),
            process.process_id,
            nodes,
        )
        identifier = _digest(
            b"phydrax-fractional-gaussian-realization\0",
            jr.key_data(key),
            process.process_id,
            nodes,
            samples,
            indices,
        )
        self.process = process
        self.root_key = key
        self.path_indices = indices
        self.grid = nodes
        self.covariance_factor = factor
        self.sample_shape = samples
        self.support = support
        self.label = label
        self.realization_id = identifier
        self.coupling_id = resolved_coupling

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def path_keys(self) -> Array:
        flat = self.path_indices.reshape((-1,))
        keys = jax.vmap(lambda index: jr.fold_in(self.root_key, index))(flat)
        return keys.reshape(self.sample_shape + tuple(self.root_key.shape))

    @property
    def values(self) -> Array:
        """Path values with shape ``sample_shape + (num_times, dimension)``."""
        keys = self.path_keys.reshape((-1,) + tuple(self.root_key.shape))
        num_times = int(self.grid.size)
        normals = jax.vmap(
            lambda key: jr.normal(
                key,
                (num_times, self.process.dimension),
                dtype=self.grid.dtype,
            )
        )(keys)
        centered = (
            jnp.einsum(
                "ij,pjd->pid",
                self.covariance_factor,
                normals,
            )
            * self.process.scale
        )
        means = self.process.mean(self.grid)
        return (centered + means).reshape(
            self.sample_shape + (num_times, self.process.dimension)
        )

    def evaluate(
        self,
        times: ArrayLike,
        /,
        *,
        interpolation: FractionalGaussianInterpolation = "grid",
    ) -> Array:
        query = jnp.asarray(times, dtype=float)
        if query.size == 0 or bool(jnp.any(~jnp.isfinite(query))):
            raise ValueError("query times must be non-empty and finite.")
        if bool(jnp.any(query < self.support[0]) | jnp.any(query > self.support[1])):
            raise ValueError("query times must lie inside realization support.")
        if interpolation not in ("grid", "linear"):
            raise ValueError("interpolation must be 'grid' or 'linear'.")
        if interpolation == "grid":
            indices = jnp.searchsorted(self.grid, query)
            indices = jnp.clip(indices, 0, int(self.grid.size) - 1)
            if not bool(jnp.all(self.grid[indices] == query)):
                raise ValueError(
                    "Grid interpolation requires every query time to be a grid node."
                )
            return jnp.take(self.values, indices, axis=len(self.sample_shape))
        flat_query = query.reshape((-1,))
        paths = self.values.reshape(
            (self.num_paths, int(self.grid.size), self.process.dimension)
        )

        def interpolate_path(path):
            return jax.vmap(
                lambda component: jnp.interp(flat_query, self.grid, component),
                in_axes=1,
                out_axes=1,
            )(path)

        values = jax.vmap(interpolate_path)(paths)
        return values.reshape(self.sample_shape + query.shape + (self.process.dimension,))

    def increments(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        /,
        *,
        interpolation: FractionalGaussianInterpolation = "grid",
    ) -> Array:
        start = jnp.asarray(starts, dtype=float)
        end = jnp.asarray(ends, dtype=float)
        if start.shape != end.shape:
            raise ValueError("increment bounds must have matching shapes.")
        if bool(jnp.any(end < start)):
            raise ValueError("increment ends must not precede starts.")
        return self.evaluate(end, interpolation=interpolation) - self.evaluate(
            start,
            interpolation=interpolation,
        )

    @property
    def fractional_gaussian_noise(self) -> Array:
        """Successive increments on the declared grid."""
        return jnp.diff(self.values, axis=len(self.sample_shape))

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axis: str = "component",
    ):
        from ._trajectory import StochasticTrajectory

        axes = (
            tuple(f"path_{index}" for index in range(len(self.sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        values = self.values
        valid = jnp.all(jnp.isfinite(values), axis=-1)
        return StochasticTrajectory(
            self.grid,
            values,
            valid=valid,
            realization_axes=axes,
            realization_shape=self.sample_shape,
            state_axes=(state_axis,),
            realizations=(self,),
            metadata={
                "process_id": self.process.process_id,
                "hurst": self.process.hurst,
                "uncertainty_source": "process",
            },
        )


__all__ = [
    "FractionalGaussianInterpolation",
    "FractionalGaussianProcess",
    "FractionalGaussianRealization",
]

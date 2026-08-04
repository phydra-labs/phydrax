#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Sequence
from math import gamma, isfinite, pi, prod, sqrt
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import AbstractAttribute, StrictModule
from ._wiener import WienerRealization


def _hash_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.ascontiguousarray(np.asarray(jax.device_get(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _hash_parts(prefix: bytes, *parts: Any) -> str:
    digest = hashlib.sha256(prefix)
    for part in parts:
        if isinstance(part, (jax.Array, np.ndarray)):
            _hash_array(digest, part)
        else:
            digest.update(repr(part).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _scalar_key(value: Key[Array, ""], /, *, owner: str) -> Array:
    if jr.key_data(value).shape != (2,):
        raise ValueError(f"{owner} requires one scalar JAX PRNG key.")
    return value


def _support(value: tuple[float, float], /, *, owner: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{owner} support must contain exactly two bounds.")
    start, end = (float(bound) for bound in value)
    if not isfinite(start) or not isfinite(end) or not end > start:
        raise ValueError(f"{owner} support requires finite bounds with end > start.")
    return start, end


def _sample_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    return shape


class AbstractLevyProcess(StrictModule):
    """Lévy law represented by a decreasing Poisson-series construction.

    ``series_terms`` maps unit-rate Poisson arrival levels and uniform marks to
    vector jumps. The returned radial envelope must decrease with the arrival
    index; it is the certificate that all jumps above a requested cutoff were
    represented. This separates process law from reusable path randomness.
    """

    dimension: AbstractAttribute[int]
    mark_dimension: AbstractAttribute[int]
    drift: AbstractAttribute[Array]
    process_id: AbstractAttribute[str]

    @abstractmethod
    def series_terms(
        self,
        arrival_levels: ArrayLike,
        uniform_marks: ArrayLike,
        duration: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Return ``(..., term, dimension)`` jumps and ``(..., term)`` envelopes."""
        raise NotImplementedError

    @abstractmethod
    def small_jump_covariance(self, cutoff: ArrayLike, /) -> Array:
        """Return covariance rate of jumps with Euclidean norm at most cutoff."""
        raise NotImplementedError

    @abstractmethod
    def truncation_drift(self, cutoff: ArrayLike, /) -> Array:
        """Return canonical drift correction for removing jumps below cutoff."""
        raise NotImplementedError

    @abstractmethod
    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        """Return Ψ(ξ) such that E exp(i ξ·Lₜ) = exp(t Ψ(ξ))."""
        raise NotImplementedError


class SymmetricStableLevyProcess(AbstractLevyProcess):
    """Independent symmetric alpha-stable components with exact Lévy scaling.

    Component ``j`` has characteristic exponent
    ``-scale[j] ** alpha * abs(frequency[j]) ** alpha``. The Poisson series is
    ordered globally by jump magnitude, so extending its capacity preserves every
    existing term and decreasing a cutoff reuses the same path.
    """

    scale: Array
    drift: Array
    alpha: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    mark_dimension: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    levy_density_coefficients: Array
    total_tail_coefficient: Array

    def __init__(
        self,
        alpha: float,
        scale: ArrayLike = 1.0,
        /,
        *,
        dimension: int | None = None,
        drift: ArrayLike = 0.0,
        process_id: str | None = None,
    ):
        stability = float(alpha)
        if not isfinite(stability) or not 0.0 < stability < 2.0:
            raise ValueError("alpha must be finite and lie strictly between 0 and 2.")
        scale_value = jnp.asarray(scale, dtype=float)
        if scale_value.ndim > 1:
            raise ValueError("scale must be scalar or a rank-1 component vector.")
        if scale_value.ndim == 0:
            resolved_dimension = 1 if dimension is None else int(dimension)
            scale_value = jnp.broadcast_to(scale_value, (resolved_dimension,))
        else:
            resolved_dimension = int(scale_value.size)
            if dimension is not None and int(dimension) != resolved_dimension:
                raise ValueError("dimension must match the vector scale length.")
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
        stable_constant = (
            stability
            * 2.0 ** (stability - 1.0)
            * gamma((1.0 + stability) / 2.0)
            / (sqrt(pi) * gamma(1.0 - stability / 2.0))
        )
        coefficients = stable_constant * scale_value**stability
        total_tail = (2.0 / stability) * jnp.sum(coefficients)
        resolved_id = process_id or _hash_parts(
            b"phydrax-symmetric-stable-process\0",
            stability,
            scale_value,
            drift_value,
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("process_id must be a non-empty string.")
        self.scale = scale_value
        self.drift = drift_value
        self.alpha = stability
        self.dimension = resolved_dimension
        self.mark_dimension = 2
        self.process_id = resolved_id
        self.levy_density_coefficients = coefficients
        self.total_tail_coefficient = total_tail

    def series_terms(
        self,
        arrival_levels: ArrayLike,
        uniform_marks: ArrayLike,
        duration: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        arrivals = jnp.asarray(arrival_levels, dtype=float)
        marks = jnp.asarray(uniform_marks, dtype=float)
        if marks.shape != arrivals.shape + (self.mark_dimension,):
            raise ValueError(
                "uniform_marks must have arrival_levels.shape + (mark_dimension,)."
            )
        time = jnp.asarray(duration, dtype=float)
        if time.shape != ():
            raise ValueError("duration must be scalar.")
        time = eqx.error_if(
            time,
            ~(jnp.isfinite(time) & (time > 0.0)),
            "duration must be finite and positive.",
        )
        arrivals = eqx.error_if(
            arrivals,
            jnp.any(~jnp.isfinite(arrivals) | (arrivals <= 0.0)),
            "arrival levels must be finite and positive.",
        )
        safe_arrivals = jnp.maximum(arrivals, jnp.finfo(arrivals.dtype).tiny)
        radii = (time * self.total_tail_coefficient / safe_arrivals) ** (1.0 / self.alpha)
        probabilities = self.levy_density_coefficients / jnp.sum(
            self.levy_density_coefficients
        )
        cumulative = jnp.cumsum(probabilities)
        components = jnp.sum(marks[..., 0, None] > cumulative, axis=-1)
        components = jnp.minimum(components, self.dimension - 1)
        signs = jnp.where(marks[..., 1] < 0.5, -1.0, 1.0)
        directions = jax.nn.one_hot(
            components,
            self.dimension,
            dtype=radii.dtype,
        )
        return signs[..., None] * radii[..., None] * directions, radii

    def small_jump_covariance(self, cutoff: ArrayLike, /) -> Array:
        threshold = jnp.asarray(cutoff, dtype=float)
        if threshold.shape != ():
            raise ValueError("cutoff must be scalar.")
        threshold = eqx.error_if(
            threshold,
            ~(jnp.isfinite(threshold) & (threshold > 0.0)),
            "cutoff must be finite and positive.",
        )
        variances = (
            2.0
            * self.levy_density_coefficients
            * threshold ** (2.0 - self.alpha)
            / (2.0 - self.alpha)
        )
        return jnp.diag(variances)

    def truncation_drift(self, cutoff: ArrayLike, /) -> Array:
        threshold = jnp.asarray(cutoff, dtype=float)
        if threshold.shape != ():
            raise ValueError("cutoff must be scalar.")
        threshold = eqx.error_if(
            threshold,
            ~(jnp.isfinite(threshold) & (threshold > 0.0)),
            "cutoff must be finite and positive.",
        )
        return jnp.zeros((self.dimension,), dtype=self.drift.dtype)

    def characteristic_exponent(self, frequency: ArrayLike, /) -> Array:
        values = jnp.asarray(frequency)
        if values.shape[-1:] != (self.dimension,):
            raise ValueError("frequency must have a trailing dimension axis.")
        return 1j * jnp.sum(values * self.drift, axis=-1) - jnp.sum(
            self.scale**self.alpha * jnp.abs(values) ** self.alpha,
            axis=-1,
        )


class LevyJumpSeries(StrictModule):
    """One fixed-capacity decreasing jump series with query-consistent increments."""

    times: Array
    jumps: Array
    arrival_levels: Array
    truncation_radii: Array
    support: tuple[float, float] = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    num_terms: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        jumps: ArrayLike,
        arrival_levels: ArrayLike,
        truncation_radii: ArrayLike,
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int],
        dimension: int,
        process_id: str,
        realization_id: str,
        coupling_id: str,
    ):
        samples = _sample_shape(sample_shape)
        dimension_value = int(dimension)
        time_values = jnp.asarray(times, dtype=float)
        jump_values = jnp.asarray(jumps)
        arrival_values = jnp.asarray(arrival_levels, dtype=float)
        radius_values = jnp.asarray(truncation_radii, dtype=float)
        expected_prefix = samples + (int(time_values.shape[-1]),)
        if time_values.ndim != len(samples) + 1:
            raise ValueError("times must have shape sample_shape + (num_terms,).")
        if (
            arrival_values.shape != expected_prefix
            or radius_values.shape != expected_prefix
        ):
            raise ValueError("arrival levels and radii must align with series times.")
        if jump_values.shape != expected_prefix + (dimension_value,):
            raise ValueError("jumps must append the declared process dimension.")
        term_count = int(time_values.shape[-1])
        if term_count <= 0 or dimension_value <= 0:
            raise ValueError("num_terms and dimension must be positive.")
        if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
            jnp.any(~jnp.isfinite(jump_values))
        ):
            raise ValueError("series times and jumps must be finite.")
        if bool(jnp.any(jnp.diff(arrival_values, axis=-1) <= 0.0)):
            raise ValueError("arrival levels must be strictly increasing.")
        tolerance = 100.0 * jnp.finfo(radius_values.dtype).eps
        if bool(jnp.any(jnp.diff(radius_values, axis=-1) > tolerance)):
            raise ValueError("truncation radii must be non-increasing.")
        support_value = _support(support, owner="LevyJumpSeries")
        if bool(
            jnp.any(time_values < support_value[0])
            | jnp.any(time_values > support_value[1])
        ):
            raise ValueError("series times must lie inside support.")
        for name, value in (
            ("process_id", process_id),
            ("realization_id", realization_id),
            ("coupling_id", coupling_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        self.times = time_values
        self.jumps = jump_values
        self.arrival_levels = arrival_values
        self.truncation_radii = radius_values
        self.support = support_value
        self.sample_shape = samples
        self.dimension = dimension_value
        self.num_terms = term_count
        self.process_id = process_id
        self.realization_id = realization_id
        self.coupling_id = coupling_id

    @property
    def smallest_radius(self) -> Array:
        """Radial envelope after the final represented Poisson point."""
        return self.truncation_radii[..., -1]

    def complete_above(self, cutoff: ArrayLike, /) -> Array:
        """Whether every proposal above cutoff is represented on each path."""
        threshold = jnp.asarray(cutoff, dtype=float)
        if threshold.shape != () or not bool(jnp.isfinite(threshold) & (threshold > 0.0)):
            raise ValueError("cutoff must be a finite positive scalar.")
        return self.smallest_radius <= threshold

    def num_jumps_above(self, cutoff: ArrayLike, /) -> Array:
        threshold = jnp.asarray(cutoff, dtype=float)
        if threshold.shape != () or not bool(jnp.isfinite(threshold) & (threshold > 0.0)):
            raise ValueError("cutoff must be a finite positive scalar.")
        norms = jnp.linalg.norm(self.jumps, axis=-1)
        return jnp.sum(norms > threshold, axis=-1)

    def increments(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        /,
        *,
        cutoff: ArrayLike = 0.0,
    ) -> Array:
        """Sum represented jumps in half-open path intervals ``(start, end]``."""
        start = jnp.asarray(starts, dtype=float)
        end = jnp.asarray(ends, dtype=float)
        if start.shape != end.shape:
            raise ValueError("increment bounds must have matching shapes.")
        threshold = jnp.asarray(cutoff, dtype=float)
        if threshold.shape != () or not bool(
            jnp.isfinite(threshold) & (threshold >= 0.0)
        ):
            raise ValueError("cutoff must be a finite non-negative scalar.")
        support_start, support_end = self.support
        if bool(
            jnp.any(~jnp.isfinite(start))
            | jnp.any(~jnp.isfinite(end))
            | jnp.any(start < support_start)
            | jnp.any(end > support_end)
            | jnp.any(end < start)
        ):
            raise ValueError(
                "increment intervals must lie in support with end greater than start."
            )
        interval_shape = start.shape
        sample_ndim = len(self.sample_shape)
        interval_ndim = start.ndim
        times = self.times.reshape(
            self.sample_shape + (1,) * interval_ndim + (self.num_terms,)
        )
        jumps = self.jumps.reshape(
            self.sample_shape + (1,) * interval_ndim + (self.num_terms, self.dimension)
        )
        bound_shape = (1,) * sample_ndim + interval_shape + (1,)
        lower = start.reshape(bound_shape)
        upper = end.reshape(bound_shape)
        norms = jnp.linalg.norm(jumps, axis=-1)
        selected = (times > lower) & (times <= upper) & (norms > threshold)
        return jnp.sum(jnp.where(selected[..., None], jumps, 0.0), axis=-2)


class LevyProcessRealization(StrictModule):
    """Prefix-stable Poisson-series randomness for one vector Lévy process."""

    root_key: Array
    path_indices: Array
    support: tuple[float, float] = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    max_terms: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    gaussian_tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_key: Key[Array, ""],
        dimension: int,
        /,
        *,
        support: tuple[float, float],
        max_terms: int,
        sample_shape: Sequence[int] = (),
        gaussian_tolerance: float = 1e-4,
        process_id: str,
        label: str | None = None,
        coupling_id: str | None = None,
        _path_indices: Array | None = None,
    ):
        key = _scalar_key(root_key, owner="LevyProcessRealization")
        dimension_value = int(dimension)
        capacity = int(max_terms)
        if dimension_value <= 0 or capacity <= 0:
            raise ValueError("dimension and max_terms must be positive.")
        samples = _sample_shape(sample_shape)
        tolerance = float(gaussian_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("gaussian_tolerance must be finite and positive.")
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be a non-empty string.")
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
        support_value = _support(support, owner="LevyProcessRealization")
        resolved_coupling = coupling_id or _hash_parts(
            b"phydrax-levy-series-coupling\0",
            jr.key_data(key),
            support_value,
            dimension_value,
            tolerance,
            process_id,
        )
        realization_id = _hash_parts(
            b"phydrax-levy-series-realization\0",
            jr.key_data(key),
            support_value,
            samples,
            indices,
            dimension_value,
            capacity,
            tolerance,
            process_id,
        )
        self.root_key = key
        self.path_indices = indices
        self.support = support_value
        self.sample_shape = samples
        self.dimension = dimension_value
        self.gaussian_tolerance = tolerance
        self.max_terms = capacity
        self.process_id = process_id
        self.label = label
        self.realization_id = realization_id
        self.coupling_id = resolved_coupling

    @classmethod
    def from_process(
        cls,
        process: AbstractLevyProcess,
        root_key: Key[Array, ""],
        /,
        *,
        support: tuple[float, float],
        max_terms: int,
        gaussian_tolerance: float = 1e-4,
        sample_shape: Sequence[int] = (),
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> LevyProcessRealization:
        if not isinstance(process, AbstractLevyProcess):
            raise TypeError("process must implement AbstractLevyProcess.")
        return cls(
            root_key,
            process.dimension,
            support=support,
            max_terms=max_terms,
            gaussian_tolerance=gaussian_tolerance,
            sample_shape=sample_shape,
            process_id=process.process_id,
            label=label,
            coupling_id=coupling_id,
        )

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def path_keys(self) -> Array:
        flat = self.path_indices.reshape((-1,))
        keys = jax.vmap(lambda index: jr.fold_in(self.root_key, index))(flat)
        return keys.reshape(self.sample_shape + tuple(self.root_key.shape))

    def _term_keys(self, namespace: int, /) -> Array:
        terms = jnp.arange(self.max_terms, dtype=jnp.uint32)
        flat_paths = self.path_keys.reshape((-1,) + tuple(self.root_key.shape))

        def one_path(path_key):
            namespaced = jr.fold_in(path_key, namespace)
            return jax.vmap(lambda term: jr.fold_in(namespaced, term))(terms)

        keys = jax.vmap(one_path)(flat_paths)
        return keys.reshape(
            self.sample_shape + (self.max_terms,) + tuple(self.root_key.shape)
        )

    @property
    def arrival_levels(self) -> Array:
        keys = self._term_keys(0)
        flat = keys.reshape((-1,) + tuple(self.root_key.shape))
        increments = jax.vmap(lambda key: jr.exponential(key, dtype=float))(flat)
        return jnp.cumsum(
            increments.reshape(self.sample_shape + (self.max_terms,)),
            axis=-1,
        )

    @property
    def event_times(self) -> Array:
        keys = self._term_keys(1)
        flat = keys.reshape((-1,) + tuple(self.root_key.shape))
        start, end = self.support
        values = jax.vmap(
            lambda key: jr.uniform(key, dtype=float, minval=start, maxval=end)
        )(flat)
        return values.reshape(self.sample_shape + (self.max_terms,))

    def uniform_marks(self, mark_dimension: int, /) -> Array:
        size = int(mark_dimension)
        if size <= 0:
            raise ValueError("mark_dimension must be positive.")
        keys = self._term_keys(2)
        flat = keys.reshape((-1,) + tuple(self.root_key.shape))
        values = jax.vmap(lambda key: jr.uniform(key, (size,), dtype=float))(flat)
        return values.reshape(self.sample_shape + (self.max_terms, size))

    def series(self, process: AbstractLevyProcess, /) -> LevyJumpSeries:
        if not isinstance(process, AbstractLevyProcess):
            raise TypeError("process must implement AbstractLevyProcess.")
        if process.process_id != self.process_id:
            raise ValueError("process and realization process_id values must match.")
        if process.dimension != self.dimension:
            raise ValueError("process and realization dimensions must match.")
        arrivals = self.arrival_levels
        duration = self.support[1] - self.support[0]
        jumps, radii = process.series_terms(
            arrivals,
            self.uniform_marks(process.mark_dimension),
            duration,
        )
        return LevyJumpSeries(
            self.event_times,
            jumps,
            arrivals,
            radii,
            support=self.support,
            sample_shape=self.sample_shape,
            dimension=self.dimension,
            process_id=self.process_id,
            realization_id=self.realization_id,
            coupling_id=self.coupling_id,
        )

    def truncated_increments(
        self,
        process: AbstractLevyProcess,
        starts: ArrayLike,
        ends: ArrayLike,
        /,
        *,
        cutoff: ArrayLike,
    ) -> Array:
        """Evaluate drift-plus-jump increments for one explicit cutoff."""
        start = jnp.asarray(starts, dtype=float)
        end = jnp.asarray(ends, dtype=float)
        jumps = self.series(process).increments(start, end, cutoff=cutoff)
        durations = end - start
        deterministic_rate = process.drift + process.truncation_drift(cutoff)
        drift_shape = (1,) * (len(self.sample_shape) + start.ndim) + (self.dimension,)
        duration_shape = (1,) * len(self.sample_shape) + start.shape + (1,)
        return jumps + durations.reshape(duration_shape) * deterministic_rate.reshape(
            drift_shape
        )

    def gaussian_realization(self, /) -> WienerRealization:
        """Return the coupled global Wiener path reserved for small-jump closure."""
        gaussian_key = jr.fold_in(self.root_key, 3)
        coupling_id = _hash_parts(
            b"phydrax-levy-gaussian-coupling\0",
            self.coupling_id,
        )
        noise_id = f"{self.process_id}:small-jump-gaussian"
        return WienerRealization(
            gaussian_key,
            (self.dimension,),
            support=self.support,
            sample_shape=self.sample_shape,
            tolerance=self.gaussian_tolerance,
            noise_id=noise_id,
            label=(None if self.label is None else f"{self.label}:small-jump-gaussian"),
            coupling_id=coupling_id,
            _path_indices=self.path_indices,
        )

    def extend(self, max_terms: int, /) -> LevyProcessRealization:
        """Increase series capacity while preserving every existing random term."""
        capacity = int(max_terms)
        if capacity < self.max_terms:
            raise ValueError("Extended capacity cannot be smaller than current capacity.")
        return LevyProcessRealization(
            self.root_key,
            self.dimension,
            support=self.support,
            gaussian_tolerance=self.gaussian_tolerance,
            max_terms=capacity,
            sample_shape=self.sample_shape,
            process_id=self.process_id,
            label=self.label,
            coupling_id=self.coupling_id,
            _path_indices=self.path_indices,
        )


__all__ = [
    "AbstractLevyProcess",
    "LevyJumpSeries",
    "LevyProcessRealization",
    "SymmetricStableLevyProcess",
]

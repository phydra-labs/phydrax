#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._wiener import WienerRealization


def _log_expm1_positive(value: Array, /) -> Array:
    """Evaluate log(expm1(value)) without overflowing for large positive values."""
    direct = jnp.log(jnp.expm1(value))
    asymptotic = value + jnp.log1p(-jnp.exp(-value))
    return jnp.where(value < 20.0, direct, asymptotic)


class OrnsteinUhlenbeckRealization(StrictModule):
    """A global exact-transition OU innovation path on physical time.

    A unit-stationary OU transition over ``[s, t]`` is represented as
    ``x_t = decay * x_s + innovation(s, t)``. The realization maps physical
    time through the exact OU Brownian clock and queries one global virtual
    Brownian tree. Consequently, interval innovations obey the OU semigroup
    under arbitrary subdivision of the same path.
    """

    driver: WienerRealization
    support: tuple[float, float] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
        _path_indices: Array | None = None,
        _path_signs: Array | None = None,
    ):
        if len(support) != 2:
            raise ValueError(
                "OrnsteinUhlenbeckRealization support must contain two bounds."
            )
        start, end = (float(value) for value in support)
        if not isfinite(start) or not isfinite(end) or not end > start:
            raise ValueError(
                "OrnsteinUhlenbeckRealization support must be finite and increasing."
            )
        noise = tuple(int(size) for size in noise_shape)
        samples = tuple(int(size) for size in sample_shape)
        if not noise or any(size <= 0 for size in noise):
            raise ValueError("OU noise_shape must contain positive dimensions.")
        if any(size <= 0 for size in samples):
            raise ValueError("OU sample dimensions must be positive.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("OU tolerance must be finite and positive.")
        if noise_id is not None and (not isinstance(noise_id, str) or not noise_id):
            raise ValueError("OU noise_id must be non-empty or None.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("OU label must be non-empty or None.")
        if coupling_id is not None and (
            not isinstance(coupling_id, str) or not coupling_id
        ):
            raise ValueError("OU coupling_id must be non-empty or None.")

        driver = WienerRealization(
            root_key,
            noise,
            support=(0.0, 1.0),
            sample_shape=samples,
            tolerance=tolerance_,
            noise_id=noise_id,
            label=label,
            _path_indices=_path_indices,
            _path_signs=_path_signs,
        )
        support_ = (start, end)
        resolved_coupling = coupling_id or canonical_fingerprint(
            {
                "kind": "ornstein-uhlenbeck-coupling",
                "driver": driver.coupling_id,
                "support": support_,
                "noise_shape": noise,
                "noise_id": noise_id,
            }
        )
        self.driver = driver
        self.support = support_
        self.noise_shape = noise
        self.sample_shape = samples
        self.tolerance = tolerance_
        self.noise_id = noise_id
        self.label = label
        self.realization_id = canonical_fingerprint(
            {
                "kind": "ornstein-uhlenbeck-realization",
                "driver": driver.realization_id,
                "support": support_,
                "noise_shape": noise,
                "sample_shape": samples,
                "noise_id": noise_id,
            }
        )
        self.coupling_id = resolved_coupling

    @classmethod
    def independent(
        cls,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> OrnsteinUhlenbeckRealization:
        return cls(
            root_key,
            noise_shape,
            support=support,
            sample_shape=sample_shape,
            tolerance=tolerance,
            noise_id=noise_id,
            label=label,
            coupling_id=coupling_id,
        )

    @classmethod
    def antithetic(
        cls,
        root_key: Key[Array, ""],
        noise_shape: Sequence[int],
        /,
        *,
        support: tuple[float, float],
        num_pairs: int,
        tolerance: float = 1e-3,
        noise_id: str | None = None,
        label: str | None = None,
        coupling_id: str | None = None,
    ) -> OrnsteinUhlenbeckRealization:
        pairs = int(num_pairs)
        if pairs <= 0:
            raise ValueError("num_pairs must be positive.")
        indices = jnp.repeat(jnp.arange(pairs, dtype=jnp.uint32), 2)
        signs = jnp.tile(jnp.asarray([1.0, -1.0]), pairs)
        return cls(
            root_key,
            noise_shape,
            support=support,
            sample_shape=(2 * pairs,),
            tolerance=tolerance,
            noise_id=noise_id,
            label=label,
            coupling_id=coupling_id,
            _path_indices=indices,
            _path_signs=signs,
        )

    @property
    def root_key(self) -> Array:
        return self.driver.root_key

    @property
    def path_indices(self) -> Array:
        return self.driver.path_indices

    @property
    def path_signs(self) -> Array:
        return self.driver.path_signs

    @property
    def num_paths(self) -> int:
        return prod(self.sample_shape) if self.sample_shape else 1

    @property
    def path_keys(self) -> Array:
        return self.driver.path_keys

    def decay(
        self,
        starts: Array,
        ends: Array,
        correlation_time: Array,
        /,
    ) -> Array:
        start, end, correlation = self._validated_intervals(
            starts, ends, correlation_time
        )
        return jnp.exp(-(end - start) / correlation)

    def innovations(
        self,
        starts: Array,
        ends: Array,
        correlation_time: Array,
        /,
        *,
        dtype: jnp.dtype | type = float,
    ) -> Array:
        """Return exact unit-stationary OU transition innovations.

        The result has shape ``sample_shape + interval_shape + noise_shape`` and
        variance ``1 - exp(-2 * (end - start) / correlation_time)``.
        """
        start, end, correlation = self._validated_intervals(
            starts, ends, correlation_time
        )
        resolved_dtype = jnp.dtype(dtype)
        origin, support_end = self.support
        duration = jnp.asarray(support_end - origin, dtype=start.dtype)
        inverse_correlation = 1.0 / correlation
        total_argument = 2.0 * inverse_correlation * duration
        log_clock_extent = _log_expm1_positive(total_argument)

        shifted_start = start - origin
        shifted_end = end - origin
        start_argument = 2.0 * inverse_correlation * shifted_start
        end_argument = 2.0 * inverse_correlation * shifted_end
        transformed_start = jnp.exp(
            _log_expm1_positive(start_argument) - log_clock_extent
        )
        transformed_end = jnp.exp(_log_expm1_positive(end_argument) - log_clock_extent)
        transformed_start = jnp.where(shifted_start == 0.0, 0.0, transformed_start)
        transformed_end = jnp.where(shifted_end == 0.0, 0.0, transformed_end)
        transformed_start = jnp.clip(transformed_start, 0.0, 1.0)
        transformed_end = jnp.clip(transformed_end, transformed_start, 1.0)

        brownian = self.driver.increments(
            transformed_start,
            transformed_end,
            dtype=resolved_dtype,
        )
        log_scale = -inverse_correlation * shifted_end + 0.5 * log_clock_extent
        scale = jnp.exp(log_scale).astype(resolved_dtype)
        broadcast_shape = (
            (1,) * len(self.sample_shape) + scale.shape + (1,) * len(self.noise_shape)
        )
        return brownian * scale.reshape(broadcast_shape)

    def transition(
        self,
        previous: Array,
        start: Array,
        end: Array,
        correlation_time: Array,
        /,
    ) -> Array:
        """Apply one exact OU transition to a state aligned with this realization."""
        start_ = jnp.asarray(start)
        end_ = jnp.asarray(end, dtype=start_.dtype)
        if start_.shape or end_.shape:
            raise ValueError("OU transition requires scalar interval bounds.")
        expected = self.sample_shape + self.noise_shape
        state = jnp.asarray(previous)
        if state.shape != expected:
            raise ValueError(f"OU state must have shape {expected}; got {state.shape}.")
        decay = self.decay(start_, end_, correlation_time)
        innovation = self.innovations(start_, end_, correlation_time, dtype=state.dtype)
        return decay * state + innovation

    def _validated_intervals(
        self,
        starts: Array,
        ends: Array,
        correlation_time: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        start = jnp.asarray(starts)
        end = jnp.asarray(ends, dtype=start.dtype)
        if start.shape != end.shape:
            raise ValueError("OU interval bounds must have matching shapes.")
        correlation = jnp.asarray(correlation_time, dtype=start.dtype).reshape(())
        support_start, support_end = self.support
        start = eqx.error_if(
            start,
            jnp.any(~jnp.isfinite(start))
            | jnp.any(~jnp.isfinite(end))
            | ~jnp.isfinite(correlation)
            | (correlation <= 0.0)
            | jnp.any(start < support_start)
            | jnp.any(end > support_end)
            | jnp.any(end < start),
            "OU intervals must increase inside support with finite positive correlation time.",
        )
        return start, end, correlation


__all__ = ["OrnsteinUhlenbeckRealization"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class AbstractTimeSamplingPolicy(StrictModule):
    """A finite-interval continuous-time sampling law."""

    minimum_time: Array
    maximum_time: Array
    policy_id: str = eqx.field(static=True)

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        shape: Sequence[int],
        /,
        *,
        dtype,
    ) -> Array:
        raise NotImplementedError


class UniformTimeSamplingPolicy(AbstractTimeSamplingPolicy):
    """Uniform sampling on one finite continuous-time interval."""

    minimum_time: Array
    maximum_time: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_time: ArrayLike = 0.0,
        maximum_time: ArrayLike = 1.0,
        /,
        *,
        policy_id: str | None = None,
    ):
        lower = jnp.asarray(minimum_time, dtype=float).reshape(())
        upper = jnp.asarray(maximum_time, dtype=float).reshape(())
        if not bool(jnp.isfinite(lower) & jnp.isfinite(upper)):
            raise ValueError("Time-sampling bounds must be finite.")
        if not bool(upper > lower):
            raise ValueError("maximum_time must exceed minimum_time.")
        resolved_id = policy_id or canonical_fingerprint(
            {
                "kind": "uniform-continuous-time-sampling",
                "minimum_time": float(lower),
                "maximum_time": float(upper),
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("policy_id must be a non-empty string or None.")
        self.minimum_time = lower
        self.maximum_time = upper
        self.policy_id = resolved_id

    def sample(
        self,
        key: Key[Array, ""],
        shape: Sequence[int],
        /,
        *,
        dtype,
    ) -> Array:
        sample_shape = tuple(int(size) for size in shape)
        if any(size <= 0 for size in sample_shape):
            raise ValueError("Time sample dimensions must be positive.")
        return jr.uniform(
            key,
            sample_shape,
            minval=self.minimum_time,
            maxval=self.maximum_time,
            dtype=dtype,
        )


class LogitNormalTimeSamplingPolicy(AbstractTimeSamplingPolicy):
    """Sigmoid-normal sampling rescaled onto one finite time interval."""

    location: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        minimum_time: ArrayLike = 0.0,
        maximum_time: ArrayLike = 1.0,
        /,
        *,
        location: float = 0.0,
        scale: float = 1.0,
        policy_id: str | None = None,
    ):
        lower = jnp.asarray(minimum_time, dtype=float).reshape(())
        upper = jnp.asarray(maximum_time, dtype=float).reshape(())
        location_ = float(location)
        scale_ = float(scale)
        if not bool(jnp.isfinite(lower) & jnp.isfinite(upper)) or not bool(upper > lower):
            raise ValueError("Time-sampling bounds must be finite and increasing.")
        if not jnp.isfinite(location_) or not jnp.isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("Logit-normal location/scale are invalid.")
        resolved_id = policy_id or canonical_fingerprint(
            {
                "kind": "logit-normal-continuous-time-sampling",
                "minimum_time": float(lower),
                "maximum_time": float(upper),
                "location": location_,
                "scale": scale_,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("policy_id must be a non-empty string or None.")
        self.minimum_time = lower
        self.maximum_time = upper
        self.location = location_
        self.scale = scale_
        self.policy_id = resolved_id

    def sample(
        self,
        key: Key[Array, ""],
        shape: Sequence[int],
        /,
        *,
        dtype,
    ) -> Array:
        sample_shape = tuple(int(size) for size in shape)
        if any(size <= 0 for size in sample_shape):
            raise ValueError("Time sample dimensions must be positive.")
        normal = self.location + self.scale * jr.normal(
            key,
            sample_shape,
            dtype=dtype,
        )
        unit = jax.nn.sigmoid(normal)
        return self.minimum_time + (self.maximum_time - self.minimum_time) * unit


__all__ = [
    "AbstractTimeSamplingPolicy",
    "LogitNormalTimeSamplingPolicy",
    "UniformTimeSamplingPolicy",
]

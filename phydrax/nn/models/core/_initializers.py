#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Key


_FanMode = Literal["fan_in", "fan_avg"]


def _fan_denominator(in_size: int, out_size: int, /, *, mode: _FanMode) -> float:
    if mode == "fan_in":
        return float(in_size)
    if mode == "fan_avg":
        return (in_size + out_size) / 2.0
    raise ValueError(f"Unknown mode {mode!r}. Expected 'fan_in' or 'fan_avg'.")


def _variance(in_size: int, out_size: int, /, *, scale: float, mode: _FanMode) -> float:
    return scale / _fan_denominator(in_size, out_size, mode=mode)


def _truncated_normal(
    shape: tuple[int, int],
    /,
    *,
    key: Key[Array, ""],
    stddev: Float[Array, ""] | float,
) -> Float[Array, "rows cols"]:
    return stddev * jr.truncated_normal(key, shape=shape, lower=-2, upper=2)


def _uniform(
    shape: tuple[int, int],
    /,
    *,
    key: Key[Array, ""],
    limit: Float[Array, ""] | float,
) -> Float[Array, "rows cols"]:
    return jr.uniform(key, shape=shape, minval=-limit, maxval=limit)


def _variance_scaling_truncated_normal_init(
    in_size: int,
    out_size: int,
    /,
    *,
    key: Key[Array, ""],
    scale: float,
    mode: _FanMode,
) -> Float[Array, "out_size in_size"]:
    shape = (out_size, in_size)
    stddev = jnp.sqrt(_variance(in_size, out_size, scale=scale, mode=mode))
    return _truncated_normal(shape, key=key, stddev=stddev)


def _variance_scaling_uniform_init(
    in_size: int,
    out_size: int,
    /,
    *,
    key: Key[Array, ""],
    scale: float,
    mode: _FanMode,
) -> Float[Array, "out_size in_size"]:
    shape = (out_size, in_size)
    limit = jnp.sqrt(3.0 * _variance(in_size, out_size, scale=scale, mode=mode))
    return _uniform(shape, key=key, limit=limit)


def _lecun_normal_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_truncated_normal_init(
        in_size, out_size, key=key, scale=1.0, mode="fan_in"
    )


def _lecun_uniform_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_uniform_init(
        in_size, out_size, key=key, scale=1.0, mode="fan_in"
    )


def _he_normal_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_truncated_normal_init(
        in_size, out_size, key=key, scale=2.0, mode="fan_in"
    )


def _he_uniform_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_uniform_init(
        in_size, out_size, key=key, scale=2.0, mode="fan_in"
    )


def _glorot_normal_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_truncated_normal_init(
        in_size, out_size, key=key, scale=1.0, mode="fan_avg"
    )


def _glorot_uniform_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    return _variance_scaling_uniform_init(
        in_size, out_size, key=key, scale=1.0, mode="fan_avg"
    )


def _orthogonal_init(
    in_size: int, out_size: int, /, *, key: Key[Array, ""]
) -> Float[Array, "out_size in_size"]:
    bigger_dim = max(out_size, in_size)
    a = jr.normal(key, shape=(bigger_dim, bigger_dim))
    q, r = jnp.linalg.qr(a)
    d = jnp.diag(r)
    ph = jnp.sign(d)
    q *= ph
    return q[:out_size, :in_size]


_initializer_dict = {
    "lecun_normal": _lecun_normal_init,
    "lecun_uniform": _lecun_uniform_init,
    "he_normal": _he_normal_init,
    "he_uniform": _he_uniform_init,
    "glorot_normal": _glorot_normal_init,
    "glorot_uniform": _glorot_uniform_init,
    "orthogonal": _orthogonal_init,
}

_initializer_dict.update(
    {
        "kaiming_normal": _initializer_dict["he_normal"],
        "kaiming_uniform": _initializer_dict["he_uniform"],
        "xavier_normal": _initializer_dict["glorot_normal"],
        "xavier_uniform": _initializer_dict["glorot_uniform"],
    }
)

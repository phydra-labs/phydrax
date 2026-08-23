#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


StageTransform = Callable[[Array], Array]


def _stage(value: Array, transform: StageTransform | None, /) -> Array:
    result = jnp.asarray(value if transform is None else transform(value))
    if result.shape != value.shape:
        raise ValueError("SSP stage transform must preserve the state shape.")
    return result


def ssprk33_step(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
) -> Array:
    """Advance one Shu--Osher SSPRK(3,3) step."""
    t = jnp.asarray(time)
    h = jnp.asarray(step_size)
    y0 = jnp.asarray(state)
    y1 = _stage(y0 + h * jnp.asarray(vector_field(t, y0, args)), stage_transform)
    y2 = _stage(
        0.75 * y0 + 0.25 * (y1 + h * jnp.asarray(vector_field(t + h, y1, args))),
        stage_transform,
    )
    return _stage(
        (1.0 / 3.0) * y0
        + (2.0 / 3.0) * (y2 + h * jnp.asarray(vector_field(t + 0.5 * h, y2, args))),
        stage_transform,
    )


def ssprk54_step(
    vector_field: Callable[[Array, Array, Any], ArrayLike],
    time: Array,
    state: Array,
    step_size: Array,
    args: Any = None,
    /,
    *,
    stage_transform: StageTransform | None = None,
) -> Array:
    """Advance one five-stage, fourth-order optimal SSP Runge--Kutta step."""
    t = jnp.asarray(time)
    h = jnp.asarray(step_size)
    y0 = jnp.asarray(state)
    y1 = _stage(
        y0 + 0.391752226571890 * h * jnp.asarray(vector_field(t, y0, args)),
        stage_transform,
    )
    y2 = _stage(
        0.444370493651235 * y0
        + 0.555629506348765 * y1
        + 0.368410593050371
        * h
        * jnp.asarray(vector_field(t + 0.391752226571890 * h, y1, args)),
        stage_transform,
    )
    y3 = _stage(
        0.620101851488403 * y0
        + 0.379898148511597 * y2
        + 0.251891774271694
        * h
        * jnp.asarray(vector_field(t + 0.586079689311540 * h, y2, args)),
        stage_transform,
    )
    y4 = _stage(
        0.178079954393132 * y0
        + 0.821920045606868 * y3
        + 0.544974750228521
        * h
        * jnp.asarray(vector_field(t + 0.474542363026870 * h, y3, args)),
        stage_transform,
    )
    return _stage(
        0.517231671970585 * y2
        + 0.096059710526147 * y3
        + 0.063692468666290
        * h
        * jnp.asarray(vector_field(t + 0.474542363026870 * h, y3, args))
        + 0.386708617503269 * y4
        + 0.226007483236906
        * h
        * jnp.asarray(vector_field(t + 0.935010631009240 * h, y4, args)),
        stage_transform,
    )


__all__ = ["StageTransform", "ssprk33_step", "ssprk54_step"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


KineticVolumeProjection = Literal["slice", "maximum", "mean"]


class KineticVolumeRenderEvidence(StrictModule):
    lower_bound: Array
    upper_bound: Array
    finite_fraction: Array
    saturated_fraction: Array
    successful: Array
    render_id: str = eqx.field(static=True)


class KineticVolumeRenderResult(StrictModule):
    rgba: Array
    evidence: KineticVolumeRenderEvidence


class KineticVolumeRenderPlan(StrictModule, NonTrainableState):
    projection: KineticVolumeProjection = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    slice_index: int = eqx.field(static=True)
    lower_bound: float | None = eqx.field(static=True)
    upper_bound: float | None = eqx.field(static=True)
    render_id: str = eqx.field(static=True)

    def __init__(
        self,
        projection: KineticVolumeProjection = "slice",
        /,
        *,
        axis: int = 2,
        slice_index: int = 0,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        if projection not in ("slice", "maximum", "mean"):
            raise ValueError(f"Unknown kinetic volume projection {projection!r}.")
        axis_value = int(axis)
        index = int(slice_index)
        lower = None if lower_bound is None else float(lower_bound)
        upper = None if upper_bound is None else float(upper_bound)
        if axis_value not in (0, 1, 2) or index < 0:
            raise ValueError("axis or slice_index is invalid.")
        if lower is not None and not np.isfinite(lower):
            raise ValueError("lower_bound must be finite.")
        if upper is not None and not np.isfinite(upper):
            raise ValueError("upper_bound must be finite.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower_bound must be below upper_bound.")
        self.projection = projection
        self.axis = axis_value
        self.slice_index = index
        self.lower_bound = lower
        self.upper_bound = upper
        self.render_id = canonical_fingerprint(
            {
                "kind": "kinetic-volume-render",
                "projection": projection,
                "axis": axis_value,
                "slice_index": index,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )

    def render(self, field: ArrayLike, /) -> KineticVolumeRenderResult:
        values = jnp.asarray(field)
        if values.ndim != 3 or jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise ValueError("Kinetic volume rendering requires one real 3D field.")
        finite = jnp.isfinite(values)
        safe = jnp.where(finite, values, 0.0)
        lower = (
            jnp.min(jnp.where(finite, values, jnp.inf))
            if self.lower_bound is None
            else jnp.asarray(self.lower_bound, dtype=values.dtype)
        )
        upper = (
            jnp.max(jnp.where(finite, values, -jnp.inf))
            if self.upper_bound is None
            else jnp.asarray(self.upper_bound, dtype=values.dtype)
        )
        if self.projection == "slice":
            if self.slice_index >= values.shape[self.axis]:
                raise ValueError("slice_index lies outside the selected axis.")
            image = jnp.take(safe, self.slice_index, axis=self.axis)
        elif self.projection == "maximum":
            image = jnp.max(safe, axis=self.axis)
        else:
            image = jnp.mean(safe, axis=self.axis)
        scale = jnp.where(upper > lower, upper - lower, 1.0)
        normalized = jnp.clip((image - lower) / scale, 0.0, 1.0)
        red = jnp.clip(1.5 * normalized, 0.0, 1.0)
        blue = jnp.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
        green = 1.0 - jnp.abs(2.0 * normalized - 1.0)
        alpha = jnp.ones_like(normalized)
        rgba = jnp.rint(255.0 * jnp.stack((red, green, blue, alpha), axis=-1)).astype(
            jnp.uint8
        )
        saturated = (image <= lower) | (image >= upper)
        success = (
            jnp.any(finite) & jnp.isfinite(lower) & jnp.isfinite(upper) & (upper >= lower)
        )
        return KineticVolumeRenderResult(
            rgba=rgba,
            evidence=KineticVolumeRenderEvidence(
                lower_bound=lower,
                upper_bound=upper,
                finite_fraction=jnp.mean(finite),
                saturated_fraction=jnp.mean(saturated),
                successful=success,
                render_id=self.render_id,
            ),
        )


__all__ = [
    "KineticVolumeProjection",
    "KineticVolumeRenderEvidence",
    "KineticVolumeRenderPlan",
    "KineticVolumeRenderResult",
]

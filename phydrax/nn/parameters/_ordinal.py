# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule


OrdinalCutpointAnchor = Literal["mean", "fixed_first"]


def _inverse_softplus(value: Array, /) -> Array:
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))


class OrderedOrdinalCutpoints(StrictModule):
    """Strictly ordered trainable ordinal cutpoints with a location anchor."""

    raw_gaps: Array
    class_count: int = eqx.field(static=True)
    minimum_gap: float = eqx.field(static=True)
    anchor: OrdinalCutpointAnchor = eqx.field(static=True)
    anchor_value: float = eqx.field(static=True)

    def __init__(
        self,
        class_count: int,
        /,
        *,
        key: Key | None = None,
        initial: ArrayLike | None = None,
        minimum_gap: float = 1.0e-3,
        anchor: OrdinalCutpointAnchor = "mean",
        anchor_value: float = 0.0,
    ):
        count = int(class_count)
        if count < 3:
            raise ValueError("Ordered ordinal cutpoints require at least three classes.")
        gap_floor = float(minimum_gap)
        if not jnp.isfinite(gap_floor) or gap_floor <= 0.0:
            raise ValueError("minimum_gap must be finite and strictly positive.")
        if anchor not in ("mean", "fixed_first"):
            raise ValueError("anchor must be 'mean' or 'fixed_first'.")
        if not jnp.isfinite(float(anchor_value)):
            raise ValueError("anchor_value must be finite.")
        cutpoint_count = count - 1
        if initial is None:
            if key is None:
                gaps = jnp.ones((cutpoint_count - 1,), dtype=float)
            else:
                gaps = 0.75 + 0.5 * jax.random.uniform(
                    key, (cutpoint_count - 1,), dtype=float
                )
        else:
            values = jnp.asarray(initial, dtype=float)
            if values.shape != (cutpoint_count,):
                raise ValueError("initial must contain class_count - 1 cutpoints.")
            if bool(jnp.any(~jnp.isfinite(values))) or bool(
                jnp.any(jnp.diff(values) <= gap_floor)
            ):
                raise ValueError(
                    "initial cutpoints must be finite and separated by minimum_gap."
                )
            gaps = jnp.diff(values)
        self.raw_gaps = _inverse_softplus(gaps - gap_floor)
        self.class_count = count
        self.minimum_gap = gap_floor
        self.anchor = anchor
        self.anchor_value = float(anchor_value)

    def __call__(self) -> Array:
        gaps = jax.nn.softplus(self.raw_gaps) + self.minimum_gap
        values = jnp.concatenate(
            (jnp.zeros((1,), dtype=gaps.dtype), jnp.cumsum(gaps)), axis=0
        )
        if self.anchor == "mean":
            values = values - jnp.mean(values) + self.anchor_value
        else:
            values = values + self.anchor_value
        return values


__all__ = ["OrderedOrdinalCutpoints", "OrdinalCutpointAnchor"]

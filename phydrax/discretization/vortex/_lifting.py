#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PreparedLiftingSurface(StrictModule, NonTrainableState):
    leading_edge: Array
    trailing_edge: Array
    bound_start: Array
    bound_end: Array
    control_point: Array
    normal: Array
    chord: Array
    span_width: Array
    trailing_start: Array
    trailing_end: Array
    panel_count: int = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)


class LiftingSurfacePlan(StrictModule, NonTrainableState):
    """Spanwise lifting panels from ordered leading/trailing section points."""

    leading_edge: Array
    trailing_edge: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, leading_edge: ArrayLike, trailing_edge: ArrayLike, /):
        leading = np.asarray(leading_edge, dtype=float)
        trailing = np.asarray(trailing_edge, dtype=float)
        if leading.ndim != 2 or leading.shape[1] != 3 or trailing.shape != leading.shape:
            raise ValueError("Leading/trailing edges must share shape (sections, 3).")
        if (
            leading.shape[0] < 2
            or np.any(~np.isfinite(leading))
            or np.any(~np.isfinite(trailing))
        ):
            raise ValueError("A lifting surface requires at least two finite sections.")
        if np.any(np.linalg.norm(trailing - leading, axis=-1) <= 0.0):
            raise ValueError("Every lifting section must have positive chord.")
        if np.any(
            np.linalg.norm(np.diff(0.5 * (leading + trailing), axis=0), axis=-1) <= 0.0
        ):
            raise ValueError(
                "Adjacent lifting sections must have positive span separation."
            )
        self.leading_edge = jnp.asarray(leading)
        self.trailing_edge = jnp.asarray(trailing)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lifting-surface-plan",
                "leading_edge": array_tree_fingerprint(leading),
                "trailing_edge": array_tree_fingerprint(trailing),
            }
        )

    def prepare(self, /) -> PreparedLiftingSurface:
        leading = self.leading_edge
        trailing = self.trailing_edge
        quarter = leading + 0.25 * (trailing - leading)
        three_quarter = leading + 0.75 * (trailing - leading)
        bound_start = quarter[:-1]
        bound_end = quarter[1:]
        control = 0.5 * (three_quarter[:-1] + three_quarter[1:])
        chord_vector = 0.5 * (
            (trailing[:-1] - leading[:-1]) + (trailing[1:] - leading[1:])
        )
        span_vector = bound_end - bound_start
        normal_raw = jnp.cross(chord_vector, span_vector)
        normal_norm = jnp.linalg.norm(normal_raw, axis=-1)
        normal = normal_raw / normal_norm[:, None]
        chord = jnp.linalg.norm(chord_vector, axis=-1)
        width = jnp.linalg.norm(span_vector, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(normal))
            & jnp.all(normal_norm > 0.0)
            & jnp.all(chord > 0.0)
            & jnp.all(width > 0.0)
        )
        normal = eqx.error_if(normal, ~finite, "Prepared lifting panels are degenerate.")
        panel_count = int(bound_start.shape[0])
        surface_id = canonical_fingerprint(
            {
                "kind": "prepared-lifting-surface",
                "plan": self.plan_id,
                "panel_count": panel_count,
            }
        )
        return PreparedLiftingSurface(
            leading,
            trailing,
            bound_start,
            bound_end,
            control,
            normal,
            chord,
            width,
            trailing[:-1],
            trailing[1:],
            panel_count,
            surface_id,
        )


__all__ = ["LiftingSurfacePlan", "PreparedLiftingSurface"]

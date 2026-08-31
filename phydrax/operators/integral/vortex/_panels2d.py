#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule


class FlowPanelGeometry2D(StrictModule):
    start: Array
    end: Array
    control: Array
    tangent: Array
    normal: Array
    length: Array
    finite: Array
    geometry_id: str = eqx.field(static=True)

    @classmethod
    def from_vertices(cls, vertices: ArrayLike, /, *, geometry_id: str | None = None):
        points_host = np.asarray(vertices, dtype=float).copy()
        if points_host.ndim != 2 or points_host.shape[1] != 2 or points_host.shape[0] < 4:
            raise ValueError("Closed flow-panel vertices require shape (count >= 4, 2).")
        if not np.allclose(points_host[0], points_host[-1], rtol=0.0, atol=1.0e-12):
            raise ValueError("Flow-panel vertices must be explicitly closed.")
        points_host[-1] = points_host[0]
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "flow-panel-geometry-2d",
                    "panel_count": int(points_host.shape[0] - 1),
                }
            )
            if geometry_id is None
            else str(geometry_id)
        )
        if not identifier:
            raise ValueError("geometry_id must be nonempty.")
        return _realize_panel_geometry(jnp.asarray(points_host), identifier)


def _realize_panel_geometry(points: Array, geometry_id: str, /) -> FlowPanelGeometry2D:
    start = points[:-1]
    end = points[1:]
    delta = end - start
    length = jnp.linalg.norm(delta, axis=-1)
    tangent = delta / jnp.maximum(length, jnp.finfo(points.dtype).tiny)[:, None]
    signed_area = 0.5 * jnp.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1])
    orientation = jnp.where(signed_area >= 0.0, 1.0, -1.0)
    normal = orientation * jnp.stack(
        (tangent[:, 1], -tangent[:, 0]),
        axis=-1,
    )
    finite = (
        jnp.all(jnp.isfinite(points))
        & jnp.all(length > 0.0)
        & (jnp.abs(signed_area) > 0.0)
    )
    normal = eqx.error_if(
        normal,
        ~finite,
        "Flow-panel contour is nonfinite or degenerate.",
    )
    return FlowPanelGeometry2D(
        start,
        end,
        0.5 * (start + end),
        tangent,
        normal,
        length,
        finite,
        geometry_id,
    )


class RigidPanelMotion2D(StrictModule):
    angle: Array
    translation: Array
    linear_velocity: Array
    angular_velocity: Array

    def realize(
        self, reference: FlowPanelGeometry2D, /
    ) -> tuple[FlowPanelGeometry2D, Array]:
        angle = jnp.asarray(self.angle, dtype=reference.start.dtype)
        translation = jnp.asarray(self.translation, dtype=reference.start.dtype)
        linear = jnp.asarray(self.linear_velocity, dtype=reference.start.dtype)
        omega = jnp.asarray(self.angular_velocity, dtype=reference.start.dtype)
        if (
            angle.shape != ()
            or omega.shape != ()
            or translation.shape != (2,)
            or linear.shape != (2,)
        ):
            raise ValueError("RigidPanelMotion2D shapes are invalid.")
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        rotation = jnp.asarray(((cosine, -sine), (sine, cosine)))
        vertices = jnp.concatenate((reference.start[:1], reference.end), axis=0)
        transformed = vertices @ rotation.T + translation
        geometry = _realize_panel_geometry(transformed, reference.geometry_id)
        relative = geometry.control - translation
        surface_velocity = linear + omega * jnp.stack(
            (-relative[:, 1], relative[:, 0]), axis=-1
        )
        return geometry, surface_velocity


def constant_panel_velocity_2d(
    targets: ArrayLike,
    geometry: FlowPanelGeometry2D,
    strength: ArrayLike,
    /,
    *,
    kind: str,
) -> Array:
    """Analytic constant source- or vortex-sheet velocity off panel endpoints."""

    target = jnp.asarray(targets, dtype=geometry.start.dtype)
    values = jnp.asarray(strength, dtype=target.dtype)
    if target.ndim != 2 or target.shape[1] != 2 or values.shape != geometry.length.shape:
        raise ValueError("Panel targets/strengths have invalid shapes.")
    if kind not in ("source", "vortex"):
        raise ValueError("kind must be 'source' or 'vortex'.")
    relative = target[:, None, :] - geometry.start[None, :, :]
    x = jnp.sum(relative * geometry.tangent[None, :, :], axis=-1)
    y = jnp.sum(relative * geometry.normal[None, :, :], axis=-1)
    x2 = x - geometry.length[None, :]
    tiny = jnp.finfo(target.dtype).tiny
    r1 = jnp.maximum(x * x + y * y, tiny)
    r2 = jnp.maximum(x2 * x2 + y * y, tiny)
    theta = jnp.arctan2(y, x2) - jnp.arctan2(y, x)
    tangent_component = 0.25 / math.pi * jnp.log(r1 / r2)
    normal_component = 0.5 / math.pi * theta
    source_velocity = (
        tangent_component[..., None] * geometry.tangent[None, :, :]
        + normal_component[..., None] * geometry.normal[None, :, :]
    )
    panel_velocity = (
        source_velocity
        if kind == "source"
        else jnp.stack((-source_velocity[..., 1], source_velocity[..., 0]), axis=-1)
    )
    return jnp.sum(values[None, :, None] * panel_velocity, axis=1)


def panel_influence_matrix_2d(
    geometry: FlowPanelGeometry2D, /, *, kind: str = "vortex"
) -> tuple[Array, Array]:
    unit_columns = []
    for panel in range(int(geometry.length.size)):
        strength = jnp.zeros_like(geometry.length).at[panel].set(1.0)
        unit_columns.append(
            constant_panel_velocity_2d(geometry.control, geometry, strength, kind=kind)
        )
    velocity = jnp.stack(tuple(unit_columns), axis=1)
    normal = jnp.sum(velocity * geometry.normal[:, None, :], axis=-1)
    tangential = jnp.sum(velocity * geometry.tangent[:, None, :], axis=-1)
    if kind == "vortex":
        normal = normal.at[jnp.diag_indices(normal.shape[0])].set(0.0)
        tangential = tangential.at[jnp.diag_indices(tangential.shape[0])].set(0.5)
    return normal, tangential


__all__ = [
    "FlowPanelGeometry2D",
    "RigidPanelMotion2D",
    "constant_panel_velocity_2d",
    "panel_influence_matrix_2d",
]

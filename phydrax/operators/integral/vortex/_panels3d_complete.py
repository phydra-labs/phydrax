#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...integral.layer_potential import (
    evaluate_laplace_layer_3d,
    LaplaceLayerPotential3D,
    SurfacePanelization3D,
)


class NativePanelGeometry3D(StrictModule):
    panelization: SurfacePanelization3D
    control_point: Array
    normal: Array
    area: Array
    panel_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @classmethod
    def from_panelization(cls, panelization: SurfacePanelization3D, /):
        if not isinstance(panelization, SurfacePanelization3D):
            raise TypeError("panelization must be SurfacePanelization3D.")
        count = panelization.panel_count
        area = (
            jnp.zeros((count,), dtype=panelization.weights.dtype)
            .at[panelization.panel_ids]
            .add(panelization.weights)
        )
        weighted_points = (
            jnp.zeros((count, 3), dtype=panelization.points.dtype)
            .at[panelization.panel_ids]
            .add(panelization.weights[:, None] * panelization.points)
        )
        weighted_normals = (
            jnp.zeros((count, 3), dtype=panelization.normals.dtype)
            .at[panelization.panel_ids]
            .add(panelization.weights[:, None] * panelization.normals)
        )
        control = weighted_points / area[:, None]
        normal = (
            weighted_normals
            / jnp.maximum(
                jnp.linalg.norm(weighted_normals, axis=-1), jnp.finfo(area.dtype).tiny
            )[:, None]
        )
        return cls(
            panelization,
            control,
            normal,
            area,
            count,
            canonical_fingerprint(
                {
                    "kind": "native-panel-geometry-3d",
                    "panelization": panelization.panelization_id,
                }
            ),
        )


class PanelFieldEvaluation3D(StrictModule):
    potential: Array
    velocity: Array
    target_report: object
    finite: Array
    successful: Array
    field_id: str = eqx.field(static=True)


class NativePanelFieldPlan3D(StrictModule):
    geometry: NativePanelGeometry3D
    field_id: str = eqx.field(static=True)

    def __init__(self, geometry: NativePanelGeometry3D, /):
        if not isinstance(geometry, NativePanelGeometry3D):
            raise TypeError("geometry must be NativePanelGeometry3D.")
        self.geometry = geometry
        self.field_id = canonical_fingerprint(
            {"kind": "native-panel-field-3d", "geometry": geometry.geometry_id}
        )

    def _node_density(self, panel_density: ArrayLike, /) -> Array:
        density = jnp.asarray(panel_density, dtype=self.geometry.control_point.dtype)
        if density.shape == (self.geometry.panel_count,):
            return density[self.geometry.panelization.panel_ids]
        if density.shape == (self.geometry.panelization.node_count,):
            return density
        raise ValueError("3-D panel density must be panel- or node-valued.")

    def evaluate(
        self,
        targets: ArrayLike,
        density: ArrayLike,
        /,
        *,
        kind: str,
        target_side: str = "exterior",
        accuracy_clearance: float = 0.0,
    ) -> PanelFieldEvaluation3D:
        if kind not in ("source", "doublet"):
            raise ValueError("3-D panel kind must be source or doublet.")
        target = jnp.asarray(targets, dtype=self.geometry.control_point.dtype)
        nodes = self._node_density(density)
        potential_model = LaplaceLayerPotential3D(
            self.geometry.panelization,
            density=nodes,
            kind="single" if kind == "source" else "double",
        )
        potential, report = evaluate_laplace_layer_3d(
            potential_model,
            target,
            target_side=target_side,
            accuracy_clearance=accuracy_clearance,
        )
        velocity = jax.vmap(jax.grad(potential_model))(target)
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(velocity))
        successful = finite & report.accuracy_supported
        return PanelFieldEvaluation3D(
            potential,
            velocity,
            report,
            finite,
            successful,
            canonical_fingerprint(
                {
                    "kind": "native-panel-field-evaluation-3d",
                    "field": self.field_id,
                    "field_kind": kind,
                    "target_count": int(target.shape[0]),
                }
            ),
        )

    def influence(
        self, /, *, kind: str, offset_fraction: float = 1.0e-4
    ) -> tuple[Array, Array]:
        offset = offset_fraction * jnp.sqrt(self.geometry.area)
        targets = self.geometry.control_point + offset[:, None] * self.geometry.normal
        columns_velocity, columns_potential = [], []
        for panel in range(self.geometry.panel_count):
            density = (
                jnp.zeros((self.geometry.panel_count,), dtype=targets.dtype)
                .at[panel]
                .set(1.0)
            )
            evaluation = self.evaluate(
                targets,
                density,
                kind=kind,
                target_side="exterior",
                accuracy_clearance=0.0,
            )
            columns_velocity.append(evaluation.velocity)
            columns_potential.append(evaluation.potential)
        velocity = jnp.stack(tuple(columns_velocity), axis=1)
        potential = jnp.stack(tuple(columns_potential), axis=1)
        return velocity, potential


__all__ = ["NativePanelFieldPlan3D", "NativePanelGeometry3D", "PanelFieldEvaluation3D"]

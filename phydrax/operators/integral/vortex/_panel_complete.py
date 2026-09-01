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
    BoundaryPanelization2D,
    LaplaceLayerPotential2D,
)
from ._panels2d import (
    constant_panel_velocity_2d,
    FlowPanelGeometry2D,
    panel_influence_matrix_2d,
)


class NativePanelGeometry2D(StrictModule):
    panelization: BoundaryPanelization2D
    straight: FlowPanelGeometry2D
    geometry_id: str = eqx.field(static=True)

    @classmethod
    def from_panelization(cls, panelization: BoundaryPanelization2D, /):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        chart = jnp.repeat(panelization.panel_chart_indices, 2)
        reference = panelization.panel_reference_bounds.reshape((-1, 1))
        frame = panelization.atlas.frame(chart, reference)
        endpoints = frame.origin.reshape((panelization.panel_count, 2, 2))
        start, end = endpoints[:, 0], endpoints[:, 1]
        delta = end - start
        length = jnp.linalg.norm(delta, axis=-1)
        tangent = delta / jnp.maximum(length, jnp.finfo(length.dtype).tiny)[:, None]
        midpoint_reference = jnp.mean(
            panelization.panel_reference_bounds, axis=-1, keepdims=True
        )
        midpoint_frame = panelization.atlas.frame(
            panelization.panel_chart_indices, midpoint_reference
        )
        straight = FlowPanelGeometry2D(
            start,
            end,
            midpoint_frame.origin,
            tangent,
            midpoint_frame.normal,
            length,
            jnp.all(jnp.isfinite(endpoints)),
            panelization.panelization_id,
        )
        return cls(
            panelization,
            straight,
            canonical_fingerprint(
                {
                    "kind": "native-flow-panel-geometry-2d",
                    "panelization": panelization.panelization_id,
                }
            ),
        )


class PanelFieldEvaluation2D(StrictModule):
    velocity: Array
    potential: Array | None
    minimum_clearance: Array
    accuracy_supported: Array
    finite: Array
    evaluation_id: str = eqx.field(static=True)


class NativePanelFieldPlan2D(StrictModule):
    geometry: NativePanelGeometry2D
    basis_degree: int = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(self, geometry: NativePanelGeometry2D, /, *, basis_degree: int = 0):
        if not isinstance(geometry, NativePanelGeometry2D) or int(basis_degree) not in (
            0,
            1,
        ):
            raise ValueError("Native panel field requires geometry and degree 0 or 1.")
        self.geometry, self.basis_degree = geometry, int(basis_degree)
        self.field_id = canonical_fingerprint(
            {
                "kind": "native-panel-field-2d",
                "geometry": geometry.geometry_id,
                "basis_degree": int(basis_degree),
            }
        )

    def evaluate(
        self,
        targets: ArrayLike,
        density: ArrayLike,
        /,
        *,
        kind: str,
        target_side: str = "exterior",
        accuracy_clearance: float = 0.0,
    ) -> PanelFieldEvaluation2D:
        target = jnp.asarray(targets, dtype=self.geometry.straight.control.dtype)
        values = jnp.asarray(density, dtype=target.dtype)
        if (
            target.ndim != 2
            or target.shape[1] != 2
            or kind not in ("source", "vortex", "doublet")
        ):
            raise ValueError("Panel target shape or field kind is invalid.")
        if values.shape == (self.geometry.straight.length.size,):
            panel_density = values
        elif self.basis_degree == 1 and values.shape == (
            self.geometry.straight.length.size,
            2,
        ):
            panel_density = jnp.mean(values, axis=-1)
        else:
            raise ValueError("Panel density shape is incompatible with basis degree.")
        if kind in ("source", "vortex"):
            velocity = constant_panel_velocity_2d(
                target, self.geometry.straight, panel_density, kind=kind
            )
            potential = None
            report_distance = jnp.min(
                jnp.linalg.norm(
                    target[:, None, :] - self.geometry.panelization.points[None, :, :],
                    axis=-1,
                )
            )
            supported = report_distance >= accuracy_clearance
        else:
            node_density = panel_density[self.geometry.panelization.panel_ids]
            potential_model = LaplaceLayerPotential2D(
                self.geometry.panelization, node_density, kind="double"
            )
            del target_side
            potential = jax.vmap(potential_model)(target)
            velocity = jax.vmap(jax.grad(potential_model))(target)
            report_distance = jnp.min(
                jnp.linalg.norm(
                    target[:, None, :] - self.geometry.panelization.points[None, :, :],
                    axis=-1,
                )
            )
            supported = report_distance >= accuracy_clearance
        finite = jnp.all(jnp.isfinite(velocity)) & (
            potential is None or jnp.all(jnp.isfinite(potential))
        )
        return PanelFieldEvaluation2D(
            velocity,
            potential,
            report_distance,
            supported,
            finite,
            canonical_fingerprint(
                {
                    "kind": "native-panel-field-evaluation-2d",
                    "field": self.field_id,
                    "target_count": int(target.shape[0]),
                    "kind_name": kind,
                }
            ),
        )

    def influence(self, /, *, kind: str) -> tuple[Array, Array]:
        if kind not in ("source", "vortex"):
            raise ValueError("Influence matrix supports source or vortex basis.")
        return panel_influence_matrix_2d(self.geometry.straight, kind=kind)


__all__ = ["NativePanelFieldPlan2D", "NativePanelGeometry2D", "PanelFieldEvaluation2D"]

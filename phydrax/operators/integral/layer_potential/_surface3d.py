#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....equations.trefftz._core import (
    AbstractTrialSpaceAdmissibility,
    trial_target_fingerprint,
)
from ....geometry import (
    BoundaryAtlas,
    CompiledGeometry,
    GeometryCapability,
    SignReliability,
    ZeroSetAccuracy,
)
from ....integration import GaussLegendreRule, reference_rule_data, ReferenceTriangleRule


def _surface_support_id(atlas: BoundaryAtlas, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "surface-singular-support-3d-v1",
            "source_id": atlas.source_id,
            "atlas_type": f"{type(atlas).__module__}.{type(atlas).__qualname__}",
            "atlas_arrays": array_tree_fingerprint(atlas),
        }
    )


class SurfacePanelization3D(StrictModule, NonTrainableState):
    """Fixed reference-triangle panelization of oriented 3D surface charts."""

    atlas: BoundaryAtlas
    geometry: CompiledGeometry | None
    chart_indices: Array
    references: Array
    points: Array
    normals: Array
    weights: Array
    panel_ids: Array
    quadrature_order: int = eqx.field(static=True)
    nodes_per_panel: int = eqx.field(static=True)
    quadrature_rule_id: str = eqx.field(static=True)
    source_support_id: str = eqx.field(static=True)
    panelization_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: BoundaryAtlas,
        /,
        *,
        quadrature_order: int = 8,
        geometry: CompiledGeometry | None = None,
    ):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("SurfacePanelization3D requires a BoundaryAtlas.")
        if atlas.ambient_dimension != 3 or atlas.reference_dimension != 2:
            raise ValueError("SurfacePanelization3D requires 2D charts in 3D.")
        order = int(quadrature_order)
        if order < 2:
            raise ValueError("quadrature_order must be at least two.")
        if geometry is not None:
            if not isinstance(geometry, CompiledGeometry):
                raise TypeError("Surface geometry must be a CompiledGeometry.")
            if geometry.ambient_dimension != 3:
                raise ValueError("Surface geometry must be three-dimensional.")
            if not geometry.has_capability(GeometryCapability.BOUNDARY_ATLAS):
                raise TypeError("Surface geometry must provide a boundary atlas.")
            if _surface_support_id(geometry.boundary_atlas) != _surface_support_id(atlas):
                raise ValueError("Surface geometry and atlas must describe the same support.")
        rule = ReferenceTriangleRule(GaussLegendreRule(order))
        data = reference_rule_data(rule)
        references = []
        chart_indices = []
        panel_ids = []
        weights = []
        panel_id = 0
        for chart in range(atlas.num_charts):
            if not bool(atlas.seam_owner[chart]):
                continue
            count = int(data.points.shape[0])
            references.extend(np.asarray(data.points))
            chart_indices.extend([chart] * count)
            panel_ids.extend([panel_id] * count)
            weights.extend(np.asarray(data.weights))
            panel_id += 1
        if not references:
            raise ValueError("Surface panelization has no owned charts.")
        chart_array = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_array = jnp.asarray(np.asarray(references), dtype=float)
        frame = atlas.frame(chart_array, reference_array)
        physical_weights = jnp.asarray(weights) * frame.jacobian
        if not bool(jnp.all(jnp.isfinite(frame.origin))) or not bool(
            jnp.all(jnp.isfinite(physical_weights))
        ):
            raise ValueError("Surface panelization geometry must be finite.")
        if bool(jnp.any(physical_weights <= 0.0)):
            raise ValueError("Surface panelization weights must be positive.")
        support_id = _surface_support_id(atlas)
        self.atlas = atlas
        self.geometry = geometry
        self.chart_indices = chart_array
        self.references = reference_array
        self.points = frame.origin
        self.normals = frame.normal
        self.weights = physical_weights
        self.panel_ids = jnp.asarray(panel_ids, dtype=jnp.int32)
        self.quadrature_order = order
        self.nodes_per_panel = int(data.points.shape[0])
        rule_id = f"reference-triangle:{type(rule.rule).__name__}"
        self.quadrature_rule_id = rule_id
        self.source_support_id = support_id
        self.panelization_id = canonical_fingerprint(
            {
                "kind": "surface-panelization-3d-v1",
                "source_support_id": support_id,
                "quadrature_rule_id": rule_id,
                "chart_indices": array_tree_fingerprint(chart_array),
                "references": array_tree_fingerprint(reference_array),
            }
        )

    @property
    def node_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def panel_count(self) -> int:
        return int(jnp.max(self.panel_ids)) + 1

    @property
    def boundary_measure(self) -> Array:
        return jnp.sum(self.weights)


class SurfaceTargetReport3D(AbstractTrialSpaceAdmissibility):
    """Continuous geometry admissibility evidence for 3D surface targets."""

    minimum_distance: Array
    intersects_singular_support: Array
    pde_membership_valid: Array
    requested_accuracy_clearance: Array
    accuracy_supported: Array
    target_count: int = eqx.field(static=True)
    singular_support_id: str = eqx.field(static=True)
    target_side: Literal["interior", "exterior", "boundary"] = eqx.field(static=True)
    target_fingerprint: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        targets: ArrayLike,
        panelization: SurfacePanelization3D,
        /,
        *,
        target_side: Literal["interior", "exterior", "boundary"],
        accuracy_clearance: float = 0.0,
    ):
        values = jnp.asarray(targets, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("3D surface targets must have shape (target_count, 3).")
        if target_side not in ("interior", "exterior", "boundary"):
            raise ValueError("Unknown 3D surface target side.")
        if panelization.geometry is None:
            raise TypeError("3D target admissibility requires compiled geometry.")
        geometry = panelization.geometry
        for capability in (GeometryCapability.REGION_QUERY, GeometryCapability.SIGNED_DISTANCE):
            if not geometry.has_capability(capability):
                raise TypeError("3D target admissibility requires region and distance queries.")
        certificate = geometry.field_certificate
        if (
            certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
            or certificate.sign_reliability is not SignReliability.RELIABLE
            or not certificate.is_signed_distance
        ):
            raise TypeError("3D target admissibility requires exact signed-distance evidence.")
        clearance = float(accuracy_clearance)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("accuracy_clearance must be finite and nonnegative.")
        signed_distance = jnp.asarray(geometry.signed_distance(values))
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
        tolerance = 64.0 * jnp.finfo(values.dtype).eps * scale
        on_boundary = jnp.abs(signed_distance) <= tolerance
        inside = jnp.asarray(geometry.contains(values), dtype=bool)
        if target_side == "interior":
            side_matches = jnp.all(inside & (signed_distance < -tolerance))
        elif target_side == "exterior":
            side_matches = jnp.all((~inside) & (signed_distance > tolerance))
        else:
            side_matches = jnp.all(on_boundary)
        intersects = jnp.any(on_boundary)
        membership = (~intersects) & side_matches
        minimum = jnp.min(jnp.abs(signed_distance))
        self.minimum_distance = minimum
        self.intersects_singular_support = intersects
        self.pde_membership_valid = membership
        self.requested_accuracy_clearance = jnp.asarray(clearance)
        self.accuracy_supported = minimum >= clearance
        self.target_count = int(values.shape[0])
        self.singular_support_id = panelization.source_support_id
        self.target_side = target_side
        self.target_fingerprint = trial_target_fingerprint(values, 3)
        self.report_id = canonical_fingerprint(
            {
                "kind": "surface-target-report-3d-v1",
                "support_id": panelization.source_support_id,
                "target_fingerprint": self.target_fingerprint,
                "target_side": target_side,
                "accuracy_clearance": clearance,
            }
        )


__all__ = ["SurfacePanelization3D", "SurfaceTargetReport3D"]

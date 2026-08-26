#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._interpolation import barycentric_basis
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
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


def _surface_support_id(atlas: BoundaryAtlas, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "surface-singular-support-3d-v1",
            "source_id": atlas.source_id,
            "atlas_type": f"{type(atlas).__module__}.{type(atlas).__qualname__}",
            "atlas_arrays": array_tree_fingerprint(atlas),
        }
    )


def _reference_inverse(affine: Array, panel_id: int, /) -> Array:
    return solve(
        LinearSystem(
            DenseLinearOperator(affine),
            problem_id=f"surface-reference-panel-{panel_id}",
        ),
        jnp.eye(2, dtype=affine.dtype),
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    ).value


def interpolate_surface_panel_density(
    panelization: "SurfacePanelization3D",
    density: Array,
    panel_id: int,
    reference: Array,
    /,
) -> Array:
    """Evaluate the panel polynomial in its declared affine reference triangle."""
    order = panelization.quadrature_order
    data = reference_rule_data(ReferenceTriangleRule(GaussLegendreRule(order)))
    standard_grid = data.points.reshape((order, order, 2))
    first_nodes = standard_grid[:, 0, 0]
    second_nodes = standard_grid[0, :, 1] / (1.0 - first_nodes[0])
    first_difference = first_nodes[:, None] - first_nodes[None, :]
    second_difference = second_nodes[:, None] - second_nodes[None, :]
    first_weights = jnp.reciprocal(jnp.prod(first_difference + jnp.eye(order), axis=1))
    second_weights = jnp.reciprocal(jnp.prod(second_difference + jnp.eye(order), axis=1))
    vertices = panelization.panel_reference_vertices[panel_id]
    affine = jnp.stack((vertices[1] - vertices[0], vertices[2] - vertices[0]), axis=-1)
    local = (reference - vertices[0]) @ panelization.panel_reference_inverses[panel_id].T
    first_basis = jax.vmap(
        lambda value: barycentric_basis(value, first_nodes, first_weights)
    )(local[:, 0])
    second_coordinate = local[:, 1] / jnp.maximum(1.0 - local[:, 0], 1e-15)
    second_basis = jax.vmap(
        lambda value: barycentric_basis(value, second_nodes, second_weights)
    )(second_coordinate)
    start = panel_id * panelization.nodes_per_panel
    stop = start + panelization.nodes_per_panel
    density_grid = density[start:stop].reshape((order, order))
    return oe.contract(
        "ni,nj,ij->n",
        first_basis,
        second_basis,
        density_grid,
        backend="jax",
    )


class SurfacePanelization3D(StrictModule, NonTrainableState):
    """Fixed reference-triangle panelization of oriented 3D surface charts."""

    atlas: BoundaryAtlas
    geometry: CompiledGeometry | None
    chart_indices: Array
    references: Array
    panel_reference_vertices: Array
    points: Array
    normals: Array
    weights: Array
    panel_ids: Array
    quadrature_order: int = eqx.field(static=True)
    nodes_per_panel: int = eqx.field(static=True)
    quadrature_rule_id: str = eqx.field(static=True)
    panel_reference_inverses: Array
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
                raise ValueError(
                    "Surface geometry and atlas must describe the same support."
                )
        rule = ReferenceTriangleRule(GaussLegendreRule(order))
        data = reference_rule_data(rule)
        references = []
        chart_indices = []
        panel_ids = []
        weights = []
        reference_vertices = []
        reference_inverses = []
        panel_id = 0
        standard_vertices = np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=float,
        )
        standard_points = np.asarray(data.points)
        for chart in range(atlas.num_charts):
            if not bool(atlas.seam_owner[chart]):
                continue
            trim = atlas.trim_domains[chart]
            if trim is None:
                vertices = standard_vertices
            else:
                vertices = np.asarray(trim.outer)
                if vertices.shape != (3, 2) or trim.holes:
                    raise ValueError(
                        "Surface panelization supports affine triangular trim cells only."
                    )
            affine = np.stack(
                (vertices[1] - vertices[0], vertices[2] - vertices[0]),
                axis=-1,
            )
            determinant = abs(
                float(affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0])
            )
            if not np.isfinite(determinant) or determinant <= 0.0:
                raise ValueError("Surface reference triangles must be nondegenerate.")
            reference_inverses.append(
                np.asarray(_reference_inverse(jnp.asarray(affine), panel_id))
            )
            mapped = vertices[0] + standard_points @ affine.T
            count = int(data.points.shape[0])
            references.extend(mapped)
            chart_indices.extend([chart] * count)
            panel_ids.extend([panel_id] * count)
            weights.extend(np.asarray(data.weights) * determinant)
            reference_vertices.append(vertices)
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
        self.panel_reference_vertices = jnp.asarray(reference_vertices, dtype=float)
        self.points = frame.origin
        self.normals = frame.normal
        self.weights = physical_weights
        self.panel_ids = jnp.asarray(panel_ids, dtype=jnp.int32)
        self.quadrature_order = order
        self.nodes_per_panel = int(data.points.shape[0])
        rule_id = f"reference-triangle:{type(rule.rule).__name__}"
        self.quadrature_rule_id = rule_id
        self.source_support_id = support_id
        self.panel_reference_inverses = jnp.asarray(reference_inverses, dtype=float)
        self.panelization_id = canonical_fingerprint(
            {
                "kind": "surface-panelization-3d-v2",
                "source_support_id": support_id,
                "quadrature_rule_id": rule_id,
                "chart_indices": array_tree_fingerprint(chart_array),
                "references": array_tree_fingerprint(reference_array),
                "panel_reference_vertices": array_tree_fingerprint(
                    self.panel_reference_vertices
                ),
                "panel_reference_inverses": array_tree_fingerprint(
                    self.panel_reference_inverses
                ),
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
        for capability in (
            GeometryCapability.REGION_QUERY,
            GeometryCapability.SIGNED_DISTANCE,
        ):
            if not geometry.has_capability(capability):
                raise TypeError(
                    "3D target admissibility requires region and distance queries."
                )
        certificate = geometry.field_certificate
        if (
            certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
            or certificate.sign_reliability is not SignReliability.RELIABLE
            or not certificate.is_signed_distance
        ):
            raise TypeError(
                "3D target admissibility requires exact signed-distance evidence."
            )
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


__all__ = [
    "interpolate_surface_panel_density",
    "SurfacePanelization3D",
    "SurfaceTargetReport3D",
]

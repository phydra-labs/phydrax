#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ....integration import (
    adaptive_interval_callable,
    AdaptiveQuadraturePlan,
    GaussLegendreRule,
    IntegrationEstimate,
    IntegrationProvenance,
    interval_rule_data,
)
from ._surface3d import interpolate_surface_panel_density, SurfacePanelization3D


def _target_reference_cell(
    panelization: SurfacePanelization3D,
    panel_id: int,
    target_reference: ArrayLike,
    /,
) -> tuple[Array, Array]:
    if panel_id < 0 or panel_id >= panelization.panel_count:
        raise ValueError("panel_id is outside the surface panelization.")
    target = jnp.asarray(target_reference, dtype=float)
    if target.shape != (2,) or not bool(jnp.all(jnp.isfinite(target))):
        raise ValueError("target_reference must be one finite reference point.")
    vertices = panelization.panel_reference_vertices[panel_id]
    local = panelization.panel_reference_inverses[panel_id] @ (target - vertices[0])
    tolerance = 64.0 * jnp.finfo(target.dtype).eps
    if not bool(
        (local[0] >= -tolerance)
        & (local[1] >= -tolerance)
        & (jnp.sum(local) <= 1.0 + tolerance)
    ):
        raise ValueError("target_reference lies outside its source reference triangle.")
    return target, vertices


def evaluate_single_layer_self_triangle_3d(
    potential,
    panel_id: int,
    target_reference: ArrayLike,
    interval_plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Integrate a 3D single layer with a target-centered Duffy map."""
    if potential.kind != "single":
        raise ValueError("Duffy self quadrature currently supports single layers.")
    panelization = potential.panelization
    if not isinstance(panelization, SurfacePanelization3D):
        raise TypeError("Duffy self quadrature requires SurfacePanelization3D.")
    if not isinstance(interval_plan, AdaptiveQuadraturePlan):
        raise TypeError("interval_plan must be an AdaptiveQuadraturePlan.")
    target_reference_, reference_vertices = _target_reference_cell(
        panelization,
        panel_id,
        target_reference,
    )
    chart = panelization.chart_indices[panel_id * panelization.nodes_per_panel]
    target_frame = panelization.atlas.frame(
        jnp.asarray([chart], dtype=jnp.int32),
        target_reference_[None, :],
    )
    target = target_frame.origin[0]
    order = panelization.quadrature_order
    subtriangles = jnp.stack(
        (
            jnp.stack((target_reference_, reference_vertices[0], reference_vertices[1])),
            jnp.stack((target_reference_, reference_vertices[1], reference_vertices[2])),
            jnp.stack((target_reference_, reference_vertices[2], reference_vertices[0])),
        )
    )
    interval_data = interval_rule_data(GaussLegendreRule(order))
    t_nodes = 0.5 * (interval_data.nodes + 1.0)
    t_weights = 0.5 * interval_data.weights

    def density_at(reference: Array) -> Array:
        return interpolate_surface_panel_density(
            panelization,
            potential.density,
            panel_id,
            reference,
        )

    estimates = []
    for triangle in subtriangles:
        target_vertex, first_vertex, second_vertex = triangle
        edge_first = first_vertex - target_vertex
        edge_second = second_vertex - target_vertex
        reference_jacobian = jnp.abs(
            jnp.linalg.det(jnp.stack((edge_first, edge_second), axis=-1))
        )
        if bool(reference_jacobian <= 64.0 * jnp.finfo(target_reference_.dtype).eps):
            continue

        def duffy_integrand(scale: Array) -> Array:
            reference = target_vertex[None, None, :] + scale[:, None, None] * (
                (1.0 - t_nodes)[None, :, None] * edge_first[None, None, :]
                + t_nodes[None, :, None] * edge_second[None, None, :]
            )
            flat_reference = reference.reshape((-1, 2))
            chart_indices = jnp.full((flat_reference.shape[0],), chart, dtype=jnp.int32)
            frame = panelization.atlas.frame(chart_indices, flat_reference)
            kernel = jax.vmap(
                potential.kernel.value,
                in_axes=(None, 0),
            )(target, frame.origin)
            density = density_at(flat_reference)
            values = (
                kernel
                * density
                * frame.jacobian
                * scale[:, None].repeat(t_nodes.size, axis=1).reshape((-1,))
                * reference_jacobian
            )
            return jnp.sum(
                values.reshape((scale.shape[0], t_nodes.size)) * t_weights, axis=1
            )

        estimates.append(
            adaptive_interval_callable(
                duffy_integrand,
                jnp.asarray((0.0, 1.0)),
                interval_plan,
            )
        )
    if not estimates:
        raise ValueError("Duffy decomposition contains no nondegenerate subtriangle.")
    value = jnp.sum(jnp.stack([estimate.value for estimate in estimates]))
    errors = jnp.stack(
        [
            jnp.inf if estimate.error_estimate is None else estimate.error_estimate
            for estimate in estimates
        ]
    )
    status = jnp.max(jnp.stack([estimate.status for estimate in estimates]))
    evaluations = jnp.sum(jnp.stack([estimate.num_evaluations for estimate in estimates]))
    return IntegrationEstimate(
        value,
        status=status,
        num_evaluations=evaluations,
        error_estimate=jnp.sum(errors),
        error_kind="duffy-target-centered-interval-rule",
        diagnostics=None,
        provenance=IntegrationProvenance(
            "adaptive-duffy-self", "surface-layer", type(interval_plan.rule).__name__
        ),
    )


def evaluate_double_layer_self_triangle_3d(
    potential,
    panel_id: int,
    target_reference: ArrayLike,
    interval_plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Integrate a 3D principal-value double layer by combined Duffy cancellation."""
    if potential.kind != "double":
        raise ValueError("Double-layer Duffy quadrature requires a double layer.")
    panelization = potential.panelization
    if not isinstance(panelization, SurfacePanelization3D):
        raise TypeError("Duffy self quadrature requires SurfacePanelization3D.")
    target_reference_, reference_vertices = _target_reference_cell(
        panelization,
        panel_id,
        target_reference,
    )
    chart = panelization.chart_indices[panel_id * panelization.nodes_per_panel]
    target_frame = panelization.atlas.frame(
        jnp.asarray([chart], dtype=jnp.int32),
        target_reference_[None, :],
    )
    target = target_frame.origin[0]
    order = panelization.quadrature_order
    subtriangles = jnp.stack(
        (
            jnp.stack((target_reference_, reference_vertices[0], reference_vertices[1])),
            jnp.stack((target_reference_, reference_vertices[1], reference_vertices[2])),
            jnp.stack((target_reference_, reference_vertices[2], reference_vertices[0])),
        )
    )
    interval_data = interval_rule_data(GaussLegendreRule(order))
    t_nodes = 0.5 * (interval_data.nodes + 1.0)
    t_weights = 0.5 * interval_data.weights

    def density_at(reference: Array) -> Array:
        return interpolate_surface_panel_density(
            panelization,
            potential.density,
            panel_id,
            reference,
        )

    def combined(scale: Array) -> Array:
        total = jnp.zeros_like(scale)
        for triangle in subtriangles:
            target_vertex, first_vertex, second_vertex = triangle
            edge_first = first_vertex - target_vertex
            edge_second = second_vertex - target_vertex
            reference_jacobian = jnp.abs(
                jnp.linalg.det(jnp.stack((edge_first, edge_second), axis=-1))
            )
            active_triangle = (
                reference_jacobian > 64.0 * jnp.finfo(target_reference_.dtype).eps
            )
            reference = target_vertex[None, None, :] + scale[:, None, None] * (
                (1.0 - t_nodes)[None, :, None] * edge_first[None, None, :]
                + t_nodes[None, :, None] * edge_second[None, None, :]
            )
            flat_reference = reference.reshape((-1, 2))
            chart_indices = jnp.full(flat_reference.shape[:-1], chart, dtype=jnp.int32)
            frame = panelization.atlas.frame(chart_indices, flat_reference)
            kernels = jax.vmap(
                potential.kernel.source_normal_derivative,
                in_axes=(None, 0, 0),
            )(target, frame.origin, frame.normal)
            density = density_at(flat_reference)
            values = (
                kernels
                * density
                * frame.jacobian
                * scale[:, None].repeat(t_nodes.size, axis=1).reshape((-1,))
                * reference_jacobian
            )
            contribution = jnp.sum(
                jnp.where(
                    scale[:, None].repeat(t_nodes.size, axis=1).reshape((-1,)) == 0.0,
                    0.0,
                    values,
                ).reshape((scale.shape[0], t_nodes.size))
                * t_weights,
                axis=1,
            )
            total = total + jnp.where(active_triangle, contribution, 0.0)
        return total

    estimate = adaptive_interval_callable(
        combined,
        jnp.asarray((0.0, 1.0)),
        interval_plan,
    )
    return IntegrationEstimate(
        estimate.value,
        status=estimate.status,
        num_evaluations=estimate.num_evaluations,
        error_estimate=estimate.error_estimate,
        error_kind="duffy-principal-value-double-layer",
        diagnostics=None,
        provenance=IntegrationProvenance(
            "adaptive-duffy-double-self",
            "surface-layer",
            type(interval_plan.rule).__name__,
        ),
    )


__all__ = [
    "evaluate_double_layer_self_triangle_3d",
    "evaluate_single_layer_self_triangle_3d",
]

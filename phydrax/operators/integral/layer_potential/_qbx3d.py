#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....geometry import GeometryCapability, SignReliability, ZeroSetAccuracy
from ....integration import (
    adaptive_triangle_callable,
    AdaptiveTrianglePlan,
    IntegrationStatus,
)
from ....operators.differential._jet import jet_terms
from ._qbx2d import _bounded_expansion_tail
from ._surface3d import interpolate_surface_panel_density


class QBXEvaluation3D(StrictModule):
    """Certified target-associated 3D local expansion evidence."""

    values: Array
    coefficient_quadrature_error: Array
    truncation_error: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    clearance: Array
    association_id: str = eqx.field(static=True)
    expansion_order: int = eqx.field(static=True)


def _directional_terms(
    potential,
    center: Array,
    source: Array,
    normal: Array,
    direction: Array,
    order: int,
) -> Array:
    def function(distance: Array) -> Array:
        target = center + distance * direction
        if potential.kind == "single":
            return potential.kernel.value(target, source)
        return potential.kernel.source_normal_derivative(target, source, normal)

    primal, terms = jet_terms(
        function,
        jnp.asarray(0.0, dtype=center.dtype),
        jnp.asarray(1.0, dtype=center.dtype),
        order=order,
    )
    return jnp.stack((primal, *terms))


def _expand(coefficients: Array, displacement: Array, order: int) -> Array:
    distance = jnp.linalg.norm(displacement)
    value = coefficients[0]
    factorial = 1.0
    for degree in range(1, order + 1):
        factorial *= degree
        value = value + coefficients[degree] * distance**degree / factorial
    return value


def evaluate_qbx_3d(
    potential,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    order: int,
    radius_factor: float,
    triangle_plan: AdaptiveTrianglePlan,
) -> QBXEvaluation3D:
    """Evaluate a 3D Laplace field with surface coefficient quadrature."""
    values = jnp.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("3D QBX targets must have shape (target_count, 3).")
    if not isinstance(triangle_plan, AdaptiveTrianglePlan):
        raise TypeError("triangle_plan must be an AdaptiveTrianglePlan.")
    expansion_order = int(order)
    factor = float(radius_factor)
    if expansion_order < 1 or not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("3D QBX order and radius_factor must be positive and finite.")
    panelization = potential.panelization
    geometry = panelization.geometry
    if (
        geometry is None
        or not geometry.has_capability(GeometryCapability.SIGNED_DISTANCE)
        or not geometry.has_capability(GeometryCapability.BOUNDARY_NORMAL)
    ):
        raise TypeError(
            "3D QBX requires compiled signed-distance geometry and target normals."
        )
    certificate = geometry.field_certificate
    if (
        certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
        or certificate.sign_reliability is not SignReliability.RELIABLE
        or not certificate.is_signed_distance
    ):
        raise TypeError("3D QBX requires exact signed-distance evidence.")
    panel_measures = jnp.sqrt(
        jnp.sum(
            panelization.weights.reshape(
                (panelization.panel_count, panelization.nodes_per_panel)
            ),
            axis=1,
        )
    )
    outputs = []
    coefficient_errors = []
    truncation_errors = []
    clearances = []
    statuses = []
    evaluations = []
    associations = []
    for target in values:
        distances = jnp.linalg.norm(target[None, :] - panelization.points, axis=-1)
        node_index = int(jnp.argmin(distances))
        panel_id = node_index // panelization.nodes_per_panel
        radius = factor * panel_measures[panel_id]
        candidate_normal = geometry.boundary_normal(target[None, :])[0]
        candidate_norm = jnp.sqrt(jnp.sum(candidate_normal * candidate_normal))
        normal = jnp.where(
            jnp.all(jnp.isfinite(candidate_normal)) & (candidate_norm > 0.0),
            candidate_normal / candidate_norm,
            panelization.normals[node_index],
        )
        centers = []
        if target_side in ("interior", "boundary"):
            centers.append(target - radius * normal)
        if target_side in ("exterior", "boundary"):
            centers.append(target + radius * normal)
        center_values = []
        center_errors = []
        center_status = []
        center_evaluations = []
        target_clearance = jnp.inf
        for center in centers:
            clearance = jnp.abs(geometry.signed_distance(center[None, :])[0]) - radius
            target_clearance = jnp.minimum(target_clearance, clearance)
            tolerance = 64.0 * jnp.finfo(values.dtype).eps
            allowed = (
                clearance >= -tolerance
                if target_side == "boundary"
                else clearance > tolerance
            )
            if not bool(allowed):
                center_values.append(jnp.asarray(jnp.nan))
                center_errors.append(jnp.asarray(jnp.inf))
                center_status.append(jnp.asarray(int(IntegrationStatus.INVALID_BOUNDS)))
                center_evaluations.append(jnp.asarray(0, dtype=jnp.int32))
                coefficient_errors.append(jnp.asarray(jnp.inf))
                continue
            coefficients = None
            coefficient_error = jnp.asarray(0.0)
            status = jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32)
            evaluation_count = jnp.asarray(0, dtype=jnp.int32)
            displacement = target - center
            direction = displacement / jnp.linalg.norm(displacement)
            for source_panel in range(panelization.panel_count):
                start = source_panel * panelization.nodes_per_panel
                stop = start + panelization.nodes_per_panel
                source_chart = panelization.chart_indices[start]

                def density_at(reference: Array) -> Array:
                    return interpolate_surface_panel_density(
                        panelization,
                        potential.density,
                        source_panel,
                        reference,
                    )

                def coefficient_integrand(reference: Array) -> Array:
                    chart_indices = jnp.full(
                        reference.shape[:-1],
                        source_chart,
                        dtype=jnp.int32,
                    )
                    frame = panelization.atlas.frame(chart_indices, reference)
                    densities = density_at(reference)

                    def one(source, source_normal, jacobian, density):
                        return (
                            _directional_terms(
                                potential,
                                center,
                                source,
                                source_normal,
                                direction,
                                expansion_order + 1,
                            )
                            * jacobian
                            * density
                        )

                    return jax.vmap(one)(
                        frame.origin,
                        frame.normal,
                        frame.jacobian,
                        densities,
                    )

                estimate = adaptive_triangle_callable(
                    coefficient_integrand,
                    panelization.panel_reference_vertices[source_panel][None, ...],
                    triangle_plan,
                )
                coefficients = (
                    estimate.value
                    if coefficients is None
                    else coefficients + estimate.value
                )
                coefficient_error = (
                    jnp.asarray(jnp.inf)
                    if estimate.error_estimate is None
                    else coefficient_error + estimate.error_estimate
                )
                status = jnp.maximum(status, estimate.status)
                evaluation_count = evaluation_count + estimate.num_evaluations
            if coefficients is None:
                raise ValueError("3D QBX coefficient quadrature has no panels.")
            high = _expand(coefficients, target - center, expansion_order)
            low = _expand(coefficients, target - center, max(expansion_order - 1, 0))
            extra = _expand(coefficients, target - center, expansion_order + 1)
            center_values.append(high)
            center_errors.append(
                _bounded_expansion_tail(
                    high - low,
                    extra - high,
                )
            )
            center_status.append(status)
            center_evaluations.append(evaluation_count)
            coefficient_errors.append(coefficient_error)
        outputs.append(jnp.mean(jnp.stack(center_values)))
        truncation_errors.append(jnp.max(jnp.stack(center_errors)))
        statuses.append(jnp.max(jnp.stack(center_status)))
        evaluations.append(jnp.sum(jnp.stack(center_evaluations)))
        clearances.append(target_clearance)
        associations.append((node_index, panel_id))
    values_ = jnp.stack(outputs)
    coefficient_error = jnp.max(jnp.stack(coefficient_errors))
    truncation_error = jnp.max(jnp.stack(truncation_errors))
    error_estimate = coefficient_error + truncation_error
    status = jnp.max(jnp.stack(statuses))
    finite = jnp.all(jnp.isfinite(values_)) & jnp.isfinite(error_estimate)
    decision_dtype = jnp.real(values_).dtype
    absolute = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if triangle_plan.absolute_tolerance is None
        else jnp.asarray(triangle_plan.absolute_tolerance, dtype=decision_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if triangle_plan.relative_tolerance is None
        else jnp.asarray(triangle_plan.relative_tolerance, dtype=decision_dtype)
    )
    clearance_values = jnp.stack(clearances)
    clearance_tolerance = 64.0 * jnp.finfo(values.dtype).eps
    clearance_supported = (
        jnp.all(clearance_values >= -clearance_tolerance)
        if target_side == "boundary"
        else jnp.all(clearance_values > clearance_tolerance)
    )
    accuracy_supported = (
        finite
        & (status == 0)
        & clearance_supported
        & (error_estimate <= absolute + relative * jnp.max(jnp.abs(values_)))
    )
    if triangle_plan.throw:
        values_ = eqx.error_if(
            values_, ~accuracy_supported, "3D QBX failed its contract."
        )
    return QBXEvaluation3D(
        values=values_,
        coefficient_quadrature_error=coefficient_error,
        truncation_error=truncation_error,
        error_estimate=error_estimate,
        status=status,
        num_evaluations=jnp.sum(jnp.stack(evaluations)),
        accuracy_supported=accuracy_supported,
        clearance=clearance_values,
        association_id=canonical_fingerprint(
            {
                "kind": "qbx-target-association-3d-v1",
                "panelization_id": panelization.panelization_id,
                "target_count": int(values.shape[0]),
                "order": expansion_order,
                "radius_factor": factor,
                "associations": associations,
            }
        ),
        expansion_order=expansion_order,
    )


__all__ = ["QBXEvaluation3D", "evaluate_qbx_3d"]

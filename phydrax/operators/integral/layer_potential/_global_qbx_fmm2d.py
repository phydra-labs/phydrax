#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry import GeometryCapability, SignReliability, ZeroSetAccuracy
from ....integration import AdaptiveQuadraturePlan, IntegrationStatus
from ._fmm2d import LaplaceFMMBackend2D
from ._laplace2d import LaplaceLayerPotential2D
from ._qbx2d import (
    _center_clearance,
    _expand_coefficients,
    _integrate_center_coefficients,
)
from ._quadrature2d import classify_panel_interactions_2d


class GlobalQBXFMMEvaluation2D(StrictModule, NonTrainableState):
    """FMM far coefficients plus direct panel coefficient corrections."""

    values: Array
    coefficient_quadrature_error: Array
    fmm_truncation_error: Array
    expansion_truncation_error: Array
    error_estimate: Array
    status: Array
    accuracy_supported: Array
    clearance: Array
    association_id: str = eqx.field(static=True)
    m2m_translations: int = eqx.field(static=True)
    m2l_translations: int = eqx.field(static=True)
    l2l_translations: int = eqx.field(static=True)
    near_panel_count: int = eqx.field(static=True)


def _complex_power_to_directional(
    coefficients: Array,
    direction: Array,
    order: int,
    /,
) -> Array:
    """Convert analytic complex powers to directional derivative terms."""
    direction_complex = direction[0] + 1j * direction[1]
    values = [jnp.real(coefficients[0])]
    for degree in range(1, order + 1):
        values.append(
            jnp.real(
                coefficients[degree]
                * math.factorial(degree)
                * direction_complex**degree
            )
        )
    return jnp.stack(values)


def evaluate_global_qbx_fmm_2d(
    potential: LaplaceLayerPotential2D,
    backend: LaplaceFMMBackend2D,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    expansion_order: int,
    radius_factor: float,
    adaptive_plan: AdaptiveQuadraturePlan,
    near_ratio: float = 4.0,
) -> GlobalQBXFMMEvaluation2D:
    """Evaluate associated QBX expansions using FMM far fields and panel near fields."""
    if not isinstance(potential, LaplaceLayerPotential2D) or potential.kind != "single":
        raise TypeError("Global QBX/FMM requires a single Laplace layer.")
    if not isinstance(backend, LaplaceFMMBackend2D):
        raise TypeError("backend must be LaplaceFMMBackend2D.")
    values = jnp.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("Global QBX/FMM targets must have shape (target_count, 2).")
    if not isinstance(adaptive_plan, AdaptiveQuadraturePlan):
        raise TypeError("adaptive_plan must be an AdaptiveQuadraturePlan.")
    order = int(expansion_order)
    if backend.expansion_order < order:
        raise ValueError("FMM expansion_order must cover the requested QBX order.")
    factor = float(radius_factor)
    ratio = float(near_ratio)
    if order < 1 or not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Invalid global QBX/FMM expansion policy.")
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("near_ratio must be finite and positive.")
    geometry = potential.panelization.geometry
    if geometry is None or not geometry.has_capability(GeometryCapability.SIGNED_DISTANCE):
        raise TypeError("Global QBX/FMM requires compiled signed-distance geometry.")
    certificate = geometry.field_certificate
    if (
        certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
        or certificate.sign_reliability is not SignReliability.RELIABLE
        or not certificate.is_signed_distance
    ):
        raise TypeError("Global QBX/FMM requires exact signed-distance evidence.")
    panelization = potential.panelization
    panel_scales = jnp.sum(
        panelization.weights.reshape((panelization.panel_count, panelization.quadrature_order)),
        axis=1,
    )
    centers = []
    target_indices = []
    associated_panels = []
    clearances = []
    for target_index, target in enumerate(values):
        node_index = int(
            jnp.argmin(jnp.linalg.norm(target[None, :] - panelization.points, axis=-1))
        )
        panel_id = node_index // panelization.quadrature_order
        radius = factor * panel_scales[panel_id]
        normal = panelization.normals[node_index]
        signs = (-1.0, 1.0) if target_side == "boundary" else (
            (-1.0,) if target_side == "interior" else (1.0,)
        )
        for sign in signs:
            center = target + sign * radius * normal
            clearance = _center_clearance(panelization, center, panel_id, radius)
            tolerance = 64.0 * jnp.finfo(values.dtype).eps
            allowed = clearance >= -tolerance if target_side == "boundary" else clearance > tolerance
            if not bool(allowed):
                raise ValueError("Global QBX/FMM expansion disk lacks continuous clearance.")
            centers.append(center)
            target_indices.append(target_index)
            associated_panels.append(panel_id)
            clearances.append(clearance)
    centers_ = jnp.stack(centers)
    interactions = classify_panel_interactions_2d(
        panelization,
        centers_,
        near_ratio=ratio,
    )
    near_panels = {
        int(panel_id)
        for panel_id in range(panelization.panel_count)
        if bool(jnp.any(interactions.near_mask[:, panel_id]))
    }
    center_data = None
    leaf_map = None
    local = None
    translations = None
    for _ in range(panelization.panel_count + 1):
        excluded_indices = tuple(
            index
            for panel_id in sorted(near_panels)
            for index in range(
                panel_id * panelization.quadrature_order,
                (panel_id + 1) * panelization.quadrature_order,
            )
        )
        center_data, leaf_map, local, near_sources, translations = (
            backend.local_expansions(
                potential,
                centers_,
                excluded_source_indices=excluded_indices,
            )
        )
        observed_panels = set(near_panels)
        for source_blocks in near_sources:
            for source_block in source_blocks:
                observed_panels.update(
                    int(index) // panelization.quadrature_order
                    for index in source_block
                )
        if observed_panels == near_panels:
            break
        near_panels = observed_panels
    else:
        raise ValueError("FMM near-correction closure did not stabilize.")
    if center_data is None or leaf_map is None or local is None or translations is None:
        raise RuntimeError("FMM local expansion preparation produced no state.")
    outputs = [[] for _ in range(values.shape[0])]
    coefficient_errors = []
    fmm_errors = []
    expansion_errors = []
    statuses = []
    evaluations = []
    for center_index, center in enumerate(centers_):
        leaf = leaf_map[center_index]
        target_index = target_indices[center_index]
        displacement = values[target_index] - center
        direction = displacement / jnp.linalg.norm(displacement)
        (
            near_coefficients,
            coefficient_error,
            status,
            evaluation_count,
        ) = _integrate_center_coefficients(
            potential,
            center,
            direction,
            order,
            adaptive_plan,
            selected_panels=tuple(sorted(near_panels)),
        )
        fmm_coefficients = _complex_power_to_directional(
            local[leaf],
            direction,
            order,
        )
        coefficients = fmm_coefficients + near_coefficients
        high = _expand_coefficients(coefficients, displacement, order)
        low = _expand_coefficients(coefficients, displacement, max(order - 1, 0))
        far_high = _expand_coefficients(fmm_coefficients, displacement, order)
        far_low = _expand_coefficients(
            fmm_coefficients,
            displacement,
            max(order - 1, 0),
        )
        outputs[target_index].append(high)
        coefficient_errors.append(coefficient_error)
        fmm_errors.append(jnp.abs(far_high - far_low))
        expansion_errors.append(jnp.abs(high - low))
        statuses.append(status)
        evaluations.append(evaluation_count)
    values_ = jnp.stack([jnp.real(jnp.mean(jnp.stack(row))) for row in outputs])
    coefficient_error = jnp.max(jnp.stack(coefficient_errors))
    fmm_error = jnp.max(jnp.stack(fmm_errors))
    expansion_error = jnp.max(jnp.stack(expansion_errors))
    error_estimate = coefficient_error + fmm_error + expansion_error
    status = jnp.max(jnp.stack(statuses))
    finite = jnp.all(jnp.isfinite(values_)) & jnp.isfinite(error_estimate)
    absolute = (
        jnp.sqrt(jnp.finfo(values_.dtype).eps)
        if adaptive_plan.absolute_tolerance is None
        else jnp.asarray(adaptive_plan.absolute_tolerance, dtype=values_.dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(values_.dtype).eps)
        if adaptive_plan.relative_tolerance is None
        else jnp.asarray(adaptive_plan.relative_tolerance, dtype=values_.dtype)
    )
    supported = finite & (status == int(IntegrationStatus.CONVERGED)) & (
        error_estimate <= absolute + relative * jnp.max(jnp.abs(values_))
    )
    return GlobalQBXFMMEvaluation2D(
        values=values_,
        coefficient_quadrature_error=coefficient_error,
        fmm_truncation_error=fmm_error,
        expansion_truncation_error=expansion_error,
        error_estimate=error_estimate,
        status=status,
        accuracy_supported=supported,
        clearance=jnp.stack(clearances),
        association_id=canonical_fingerprint(
            {
                "kind": "global-qbx-fmm-association-2d-v1",
                "panelization_id": panelization.panelization_id,
                "target_count": int(values.shape[0]),
                "target_side": target_side,
                "order": order,
                "radius_factor": factor,
                "near_ratio": ratio,
                "associated_panels": associated_panels,
            }
        ),
        m2m_translations=max(len(backend.source_indices) - 1, 0),
        m2l_translations=translations[0],
        l2l_translations=translations[1],
        near_panel_count=len(near_panels),
    )


__all__ = ["GlobalQBXFMMEvaluation2D", "evaluate_global_qbx_fmm_2d"]

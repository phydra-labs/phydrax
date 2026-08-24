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

from ...._strict import StrictModule
from ....integration import IntegrationStatus


class QBXEvaluation2D(StrictModule):
    """Target-associated local Taylor/QBX expansion evidence."""

    values: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    expansion_order: int = eqx.field(static=True)
    radius: Array


def _taylor_value(function, center: Array, target: Array, order: int) -> Array:
    displacement = target - center
    value = function(center)
    derivative = function
    factorial = 1.0
    for degree in range(1, order + 1):
        derivative = jax.jacfwd(derivative)
        tensor = derivative(center)
        for _ in range(degree):
            tensor = jnp.tensordot(tensor, displacement, axes=(0, 0))
        factorial *= degree
        value = value + tensor / factorial
    return value


def evaluate_qbx_2d(
    potential,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    order: int,
    radius_factor: float,
    absolute_tolerance: float | None,
    relative_tolerance: float | None,
    throw: bool,
) -> QBXEvaluation2D:
    """Evaluate analytic layer fields from target-associated local expansions."""
    values = jnp.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("QBX targets must have shape (target_count, 2).")
    expansion_order = int(order)
    if expansion_order < 1:
        raise ValueError("QBX expansion order must be positive.")
    factor = float(radius_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("QBX radius_factor must be finite and positive.")
    panelization = potential.panelization
    panel_scales = jnp.sum(
        panelization.weights.reshape(
            (panelization.panel_count, panelization.quadrature_order)
        ),
        axis=1,
    )
    output = []
    errors = []
    radii = []
    evaluations = 0
    for target in values:
        distances = jnp.linalg.norm(target[None, :] - panelization.points, axis=-1)
        node_index = int(jnp.argmin(distances))
        panel_id = node_index // panelization.quadrature_order
        radius = factor * panel_scales[panel_id]
        normal = panelization.normals[node_index]
        centers = []
        if target_side in ("interior", "boundary"):
            centers.append(target - radius * normal)
        if target_side in ("exterior", "boundary"):
            centers.append(target + radius * normal)
        high_values = []
        low_values = []
        for center in centers:
            high_values.append(_taylor_value(potential, center, target, expansion_order))
            low_values.append(
                _taylor_value(
                    potential,
                    center,
                    target,
                    max(expansion_order - 1, 0),
                )
            )
        high = jnp.mean(jnp.stack(high_values))
        low = jnp.mean(jnp.stack(low_values))
        output.append(high)
        errors.append(jnp.abs(high - low))
        radii.append(radius)
        evaluations += len(centers) * (expansion_order + max(expansion_order - 1, 0) + 1)
    output_ = jnp.stack(output)
    error_estimate = jnp.max(jnp.stack(errors))
    radius_ = jnp.stack(radii)
    finite = jnp.all(jnp.isfinite(output_)) & jnp.isfinite(error_estimate)
    decision_dtype = jnp.real(output_).dtype
    absolute = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if absolute_tolerance is None
        else jnp.asarray(absolute_tolerance, dtype=decision_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if relative_tolerance is None
        else jnp.asarray(relative_tolerance, dtype=decision_dtype)
    )
    accuracy_supported = finite & (
        error_estimate <= absolute + relative * jnp.max(jnp.abs(output_))
    )
    status = jnp.where(
        finite,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    ).astype(jnp.int32)
    if throw:
        output_ = eqx.error_if(
            output_,
            ~accuracy_supported,
            "QBX local expansion failed its truncation contract.",
        )
    return QBXEvaluation2D(
        values=output_,
        error_estimate=error_estimate,
        status=status,
        num_evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
        accuracy_supported=accuracy_supported,
        expansion_order=expansion_order,
        radius=radius_,
    )


__all__ = ["QBXEvaluation2D", "evaluate_qbx_2d"]

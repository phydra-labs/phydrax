#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._interpolation import barycentric_basis
from ...._strict import StrictModule
from ....integration import (
    adaptive_interval_callable,
    AdaptiveQuadraturePlan,
    IntegrationEstimate,
    IntegrationProvenance,
    IntegrationStatus,
)
from ._core import BoundaryPanelization2D


class PanelInteractionReport2D(StrictModule):
    """Conservative target-to-panel regime classification for 2D evaluation."""

    minimum_node_distance: Array
    panel_center_distance: Array
    panel_scales: Array
    near_mask: Array
    far_mask: Array
    source_node_collision: Array
    classification_id: str = eqx.field(static=True)

    def __init__(
        self,
        panelization: BoundaryPanelization2D,
        targets: ArrayLike,
        /,
        *,
        near_ratio: float = 4.0,
    ):
        if not isinstance(panelization, BoundaryPanelization2D):
            raise TypeError("panelization must be BoundaryPanelization2D.")
        ratio = float(near_ratio)
        if not jnp.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("near_ratio must be finite and positive.")
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Interaction targets must have shape (target_count, 2).")
        order = panelization.quadrature_order
        points = panelization.points.reshape((panelization.panel_count, order, 2))
        centers = jnp.mean(points, axis=1)
        scales = jnp.sum(
            panelization.weights.reshape((panelization.panel_count, order)), axis=1
        )
        center_distance = jnp.linalg.norm(
            values[:, None, :] - centers[None, :, :], axis=-1
        )
        node_distance = jnp.linalg.norm(
            values[:, None, None, :] - points[None, :, :, :], axis=-1
        )
        minimum_node_distance = jnp.min(node_distance, axis=-1)
        near = center_distance <= ratio * scales[None, :]
        collision = jnp.any(node_distance == 0.0, axis=(-1, -2))
        self.minimum_node_distance = minimum_node_distance
        self.panel_center_distance = center_distance
        self.panel_scales = scales
        self.near_mask = near
        self.far_mask = ~near
        self.source_node_collision = collision
        self.classification_id = canonical_fingerprint(
            {
                "kind": "layer-panel-interaction-2d-v1",
                "panelization_id": panelization.panelization_id,
                "target_count": int(values.shape[0]),
                "near_ratio": ratio,
            }
        )


class AdaptiveLayerEvaluation2D(StrictModule):
    """Aggregated adaptive panel estimates for one target batch."""

    values: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    near_panel_count: int = eqx.field(static=True)
    far_panel_count: int = eqx.field(static=True)
    failed_panel_count: int = eqx.field(static=True)


def classify_panel_interactions_2d(
    panelization: BoundaryPanelization2D,
    targets: ArrayLike,
    /,
    *,
    near_ratio: float = 4.0,
) -> PanelInteractionReport2D:
    """Classify broad far/near panel regimes without using nodes as support evidence."""
    return PanelInteractionReport2D(
        panelization,
        targets,
        near_ratio=near_ratio,
    )


def _panel_breakpoints(
    plan: AdaptiveQuadraturePlan,
    bounds: Array,
    extra: tuple[float, ...] = (),
    /,
) -> tuple[float, ...]:
    lower, upper = float(bounds[0]), float(bounds[1])
    return tuple(
        sorted({point for point in (*plan.breakpoints, *extra) if lower < point < upper})
    )


def _panel_density(
    node_reference: Array,
    node_density: Array,
    reference: Array,
    /,
) -> Array:
    differences = node_reference[:, None] - node_reference[None, :]
    weights = jnp.reciprocal(jnp.prod(differences + jnp.eye(node_reference.size), axis=1))
    basis = jax.vmap(
        lambda location: barycentric_basis(location, node_reference, weights)
    )(reference)
    return basis @ node_density


def _panel_source_data(
    potential,
    panel_id: int,
    reference: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    panelization = potential.panelization
    order = panelization.quadrature_order
    start = panel_id * order
    stop = start + order
    chart = panelization.panel_chart_indices[panel_id]
    chart_indices = jnp.full(reference.shape, chart, dtype=jnp.int32)
    reference_points = reference[:, None]
    frame = panelization.atlas.frame(chart_indices, reference_points)
    node_reference = panelization.references[start:stop, 0]
    node_density = potential.density[start:stop]
    density = _panel_density(node_reference, node_density, reference)
    return frame.origin, frame.normal, frame.jacobian, density


def _panel_integrand(
    potential, target: Array, panel_id: int, reference: Array, /
) -> Array:
    sources, normals, jacobian, density = _panel_source_data(
        potential,
        panel_id,
        reference,
    )
    if potential.kind == "single":
        kernels = jax.vmap(potential.kernel.value, in_axes=(None, 0))(
            target,
            sources,
        )
    else:
        kernels = jax.vmap(
            potential.kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(target, sources, normals)
    return kernels * density * jacobian


def _negative_log_moment(lower: Array, upper: Array, target: Array, /) -> Array:
    left = target - lower
    right = upper - target

    def primitive(length: Array) -> Array:
        safe = jnp.where(length == 0.0, 1.0, length)
        return jnp.where(length == 0.0, 0.0, -length * jnp.log(safe) + length)

    return (primitive(left) + primitive(right)) / (2.0 * jnp.pi)


def evaluate_laplace_single_layer_self_panel_2d(
    potential,
    panel_id: int,
    target_reference: ArrayLike,
    plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Evaluate one logarithmically singular Laplace single-layer panel."""
    if potential.kind != "single":
        raise ValueError("Self-panel regularization currently supports single layers.")
    target_reference_ = jnp.asarray(target_reference, dtype=float).reshape(())
    panelization = potential.panelization
    bounds = panelization.panel_reference_bounds[panel_id]
    chart = panelization.panel_chart_indices[panel_id]
    target_frame = panelization.atlas.frame(
        jnp.asarray([chart], dtype=jnp.int32),
        target_reference_.reshape((1, 1)),
    )
    target_point = target_frame.origin[0]
    target_jacobian = target_frame.jacobian[0]
    order = panelization.quadrature_order
    start = panel_id * order
    stop = start + order
    node_reference = panelization.references[start:stop, 0]
    node_density = potential.density[start:stop]
    g0 = (
        _panel_density(
            node_reference,
            node_density,
            target_reference_.reshape((1,)),
        )[0]
        * target_jacobian
    )
    breakpoints = _panel_breakpoints(
        plan,
        bounds,
        (float(target_reference_),),
    )
    self_plan = AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=plan.absolute_tolerance,
        relative_tolerance=plan.relative_tolerance,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=breakpoints,
        collect_partition=plan.collect_partition,
        throw=plan.throw,
    )

    def regularized(reference: Array) -> Array:
        sources, _, jacobian, density = _panel_source_data(
            potential,
            panel_id,
            reference,
        )
        difference = reference - target_reference_
        absolute_difference = jnp.abs(difference)
        safe_difference = jnp.where(absolute_difference == 0.0, 1.0, difference)
        distance = jnp.linalg.norm(sources - target_point, axis=-1)
        ratio = jnp.where(
            absolute_difference == 0.0,
            target_jacobian,
            distance / jnp.abs(safe_difference),
        )
        weight_density = density * jacobian
        smooth = -jnp.log(ratio) * weight_density / (2.0 * jnp.pi)
        singular = -jnp.log(jnp.abs(safe_difference)) / (2.0 * jnp.pi)
        remainder = jnp.where(
            absolute_difference == 0.0,
            0.0,
            singular * (weight_density - g0),
        )
        return smooth + remainder

    estimate = adaptive_interval_callable(
        regularized,
        bounds,
        self_plan,
    )
    value = estimate.value + g0 * _negative_log_moment(
        bounds[0],
        bounds[1],
        target_reference_,
    )
    return IntegrationEstimate(
        value,
        status=estimate.status,
        num_evaluations=estimate.num_evaluations,
        error_estimate=estimate.error_estimate,
        error_kind="adaptive-self-regularized",
        diagnostics=estimate.diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-self-layer", "layer", type(plan.rule).__name__
        ),
        precision_evidence=estimate.precision_evidence,
    )


def evaluate_helmholtz_single_layer_self_panel_weights_2d(
    panelization: BoundaryPanelization2D,
    kernel,
    panel_id: int,
    target_reference: ArrayLike,
    plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Evaluate product-integration weights for a Helmholtz self panel."""
    from ._helmholtz2d import HelmholtzLayerKernel2D

    if not isinstance(kernel, HelmholtzLayerKernel2D):
        raise TypeError("Helmholtz self weights require a HelmholtzLayerKernel2D.")
    target_reference_ = jnp.asarray(target_reference, dtype=float).reshape(())
    bounds = panelization.panel_reference_bounds[panel_id]
    chart = panelization.panel_chart_indices[panel_id]
    target_frame = panelization.atlas.frame(
        jnp.asarray([chart], dtype=jnp.int32),
        target_reference_.reshape((1, 1)),
    )
    target_point = target_frame.origin[0]
    target_jacobian = target_frame.jacobian[0]
    order = panelization.quadrature_order
    start = panel_id * order
    stop = start + order
    node_reference = panelization.references[start:stop, 0]
    differences = node_reference[:, None] - node_reference[None, :]
    safe_differences = differences + jnp.eye(order, dtype=node_reference.dtype)
    barycentric_weights = jnp.reciprocal(jnp.prod(safe_differences, axis=1))
    target_index = int(jnp.argmin(jnp.abs(node_reference - target_reference_)))
    basis_at_target = jax.nn.one_hot(target_index, order)
    breakpoints = _panel_breakpoints(
        plan,
        bounds,
        (float(target_reference_),),
    )
    self_plan = AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=plan.absolute_tolerance,
        relative_tolerance=plan.relative_tolerance,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=breakpoints,
        collect_partition=plan.collect_partition,
        throw=plan.throw,
    )

    def basis(reference: Array) -> Array:
        return jax.vmap(
            lambda location: barycentric_basis(
                location,
                node_reference,
                barycentric_weights,
            )
        )(reference)

    def regularized(reference: Array) -> Array:
        chart_indices = jnp.full(reference.shape, chart, dtype=jnp.int32)
        frame = panelization.atlas.frame(
            chart_indices,
            reference[:, None],
        )
        sources = frame.origin
        jacobian = frame.jacobian
        kernel_values = jax.vmap(kernel.value, in_axes=(None, 0))(
            target_point,
            sources,
        )
        difference = reference - target_reference_
        absolute_difference = jnp.abs(difference)
        safe_difference = jnp.where(absolute_difference == 0.0, 1.0, difference)
        singular = -jnp.log(jnp.abs(safe_difference)) / (2.0 * jnp.pi)
        gamma = 0.5772156649015329
        regular_limit = (
            0.25j
            - jnp.log(kernel.wavenumber * target_jacobian / 2.0) / (2.0 * jnp.pi)
            - gamma / (2.0 * jnp.pi)
        )
        regular = jnp.where(
            absolute_difference == 0.0,
            regular_limit,
            kernel_values - singular,
        )
        basis_values = basis(reference)
        return regular[:, None] * basis_values * jacobian[:, None] + jnp.where(
            absolute_difference[:, None] == 0.0,
            0.0,
            singular[:, None]
            * (basis_values * jacobian[:, None] - basis_at_target * target_jacobian),
        )

    estimate = adaptive_interval_callable(regularized, bounds, self_plan)
    value = (
        estimate.value
        + _negative_log_moment(
            bounds[0],
            bounds[1],
            target_reference_,
        )
        * basis_at_target
        * target_jacobian
    )
    return IntegrationEstimate(
        value,
        status=estimate.status,
        num_evaluations=estimate.num_evaluations,
        error_estimate=estimate.error_estimate,
        error_kind="adaptive-helmholtz-self-product",
        diagnostics=estimate.diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-helmholtz-self", "layer", type(plan.rule).__name__
        ),
        precision_evidence=estimate.precision_evidence,
    )


def evaluate_helmholtz_single_layer_self_panel_block_2d(
    panelization: BoundaryPanelization2D,
    kernel,
    panel_id: int,
    target_references: ArrayLike,
    plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Batch product-integration weights for all targets on one self panel."""
    from ._helmholtz2d import HelmholtzLayerKernel2D

    if not isinstance(kernel, HelmholtzLayerKernel2D):
        raise TypeError("Helmholtz self weights require a HelmholtzLayerKernel2D.")
    targets_reference = jnp.asarray(target_references, dtype=float).reshape((-1,))
    panel_order = panelization.quadrature_order
    if targets_reference.shape != (panel_order,):
        raise ValueError("Self-panel target references must match panel order.")
    bounds = panelization.panel_reference_bounds[panel_id]
    chart = panelization.panel_chart_indices[panel_id]
    target_frame = panelization.atlas.frame(
        jnp.full((panel_order,), chart, dtype=jnp.int32),
        targets_reference[:, None],
    )
    target_points = target_frame.origin
    target_jacobian = target_frame.jacobian
    start = panel_id * panel_order
    stop = start + panel_order
    node_reference = panelization.references[start:stop, 0]
    differences = node_reference[:, None] - node_reference[None, :]
    barycentric_weights = jnp.reciprocal(
        jnp.prod(differences + jnp.eye(panel_order), axis=1)
    )
    breakpoints = _panel_breakpoints(
        plan,
        bounds,
        tuple(float(value) for value in targets_reference),
    )
    self_plan = AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=plan.absolute_tolerance,
        relative_tolerance=plan.relative_tolerance,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=breakpoints,
        collect_partition=plan.collect_partition,
        throw=plan.throw,
    )

    def regularized(reference: Array) -> Array:
        chart_indices = jnp.full(reference.shape, chart, dtype=jnp.int32)
        source_frame = panelization.atlas.frame(
            chart_indices,
            reference[:, None],
        )
        sources = source_frame.origin
        jacobian = source_frame.jacobian
        kernel_values = jax.vmap(
            lambda target: jax.vmap(kernel.value, in_axes=(None, 0))(
                target,
                sources,
            )
        )(target_points).T
        difference = reference[:, None] - targets_reference[None, :]
        absolute_difference = jnp.abs(difference)
        safe_difference = jnp.where(absolute_difference == 0.0, 1.0, difference)
        singular = -jnp.log(jnp.abs(safe_difference)) / (2.0 * jnp.pi)
        gamma = 0.5772156649015329
        regular_limit = (
            0.25j
            - jnp.log(kernel.wavenumber * target_jacobian / 2.0) / (2.0 * jnp.pi)
            - gamma / (2.0 * jnp.pi)
        )
        regular = jnp.where(
            absolute_difference == 0.0,
            regular_limit[None, :],
            kernel_values - singular,
        )
        basis_values = jax.vmap(
            lambda location: barycentric_basis(
                location,
                node_reference,
                barycentric_weights,
            )
        )(reference)
        basis_target = jnp.eye(panel_order, dtype=reference.dtype)
        return regular[:, :, None] * basis_values[:, None, :] * jacobian[
            :, None, None
        ] + jnp.where(
            absolute_difference[:, :, None] == 0.0,
            0.0,
            singular[:, :, None]
            * (
                basis_values[:, None, :] * jacobian[:, None, None]
                - target_jacobian[None, :, None] * basis_target[None, :, :]
            ),
        )

    estimate = adaptive_interval_callable(regularized, bounds, self_plan)
    value = estimate.value + (
        _negative_log_moment(
            bounds[0],
            bounds[1],
            targets_reference,
        )[:, None]
        * target_jacobian[:, None]
        * jnp.eye(panel_order)
    )
    return IntegrationEstimate(
        value,
        status=estimate.status,
        num_evaluations=estimate.num_evaluations,
        error_estimate=estimate.error_estimate,
        error_kind="adaptive-helmholtz-self-product-batched",
        diagnostics=estimate.diagnostics,
        provenance=IntegrationProvenance(
            "adaptive-helmholtz-self-batched",
            "layer",
            type(plan.rule).__name__,
        ),
        precision_evidence=estimate.precision_evidence,
    )


def evaluate_double_layer_self_panel_weights_2d(
    panelization: BoundaryPanelization2D,
    kernel,
    panel_id: int,
    target_reference: ArrayLike,
    plan: AdaptiveQuadraturePlan,
    /,
) -> IntegrationEstimate:
    """Evaluate principal-value double-layer self weights by symmetric cancellation."""
    target_reference_ = jnp.asarray(target_reference, dtype=float).reshape(())
    bounds = panelization.panel_reference_bounds[panel_id]
    chart = panelization.panel_chart_indices[panel_id]
    target_frame = panelization.atlas.frame(
        jnp.asarray([chart], dtype=jnp.int32),
        target_reference_[None, None],
    )
    target = target_frame.origin[0]
    order = panelization.quadrature_order
    start = panel_id * order
    stop = start + order
    node_reference = panelization.references[start:stop, 0]
    differences = node_reference[:, None] - node_reference[None, :]
    barycentric_weights = jnp.reciprocal(jnp.prod(differences + jnp.eye(order), axis=1))
    left_span = target_reference_ - bounds[0]
    right_span = bounds[1] - target_reference_
    symmetric_span = jnp.minimum(left_span, right_span)
    self_plan = AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=plan.absolute_tolerance,
        relative_tolerance=plan.relative_tolerance,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=(),
        collect_partition=plan.collect_partition,
        throw=False,
    )

    def values_at(reference: Array) -> Array:
        frame = panelization.atlas.frame(
            jnp.full(reference.shape, chart, dtype=jnp.int32),
            reference[:, None],
        )
        basis_values = jax.vmap(
            lambda location: barycentric_basis(
                location,
                node_reference,
                barycentric_weights,
            )
        )(reference)
        kernels = jax.vmap(
            kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(target, frame.origin, frame.normal)
        return kernels[:, None] * basis_values * frame.jacobian[:, None]

    def symmetric(reference_distance: Array) -> Array:
        plus = target_reference_ + reference_distance
        minus = target_reference_ - reference_distance
        plus_values = values_at(plus)
        minus_values = values_at(minus)
        return jnp.where(
            reference_distance[:, None] == 0.0,
            0.0,
            plus_values + minus_values,
        )

    estimates = []
    if bool(symmetric_span > 0.0):
        estimates.append(
            adaptive_interval_callable(
                symmetric,
                jnp.asarray((0.0, symmetric_span)),
                self_plan,
            )
        )
    if bool(left_span > symmetric_span):
        estimates.append(
            adaptive_interval_callable(
                values_at,
                jnp.asarray((bounds[0], target_reference_ - symmetric_span)),
                self_plan,
            )
        )
    if bool(right_span > symmetric_span):
        estimates.append(
            adaptive_interval_callable(
                values_at,
                jnp.asarray((target_reference_ + symmetric_span, bounds[1])),
                self_plan,
            )
        )
    if not estimates:
        estimates.append(
            adaptive_interval_callable(
                symmetric,
                jnp.asarray((0.0, symmetric_span)),
                self_plan,
            )
        )
    value = jnp.sum(jnp.stack([estimate.value for estimate in estimates]), axis=0)
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
        error_kind="adaptive-principal-value-symmetric-cancellation",
        diagnostics=None,
        provenance=IntegrationProvenance(
            "adaptive-double-self", "layer", type(plan.rule).__name__
        ),
    )


def _panel_direct(potential, target: Array, panel_id: int, /) -> Array:
    order = potential.panelization.quadrature_order
    start = panel_id * order
    stop = start + order
    sources = potential.panelization.points[start:stop]
    normals = potential.panelization.normals[start:stop]
    weights = potential.panelization.weights[start:stop]
    density = potential.density[start:stop]
    if potential.kind == "single":
        kernels = jax.vmap(potential.kernel.value, in_axes=(None, 0))(
            target,
            sources,
        )
    else:
        kernels = jax.vmap(
            potential.kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(target, sources, normals)
    return jnp.sum(kernels * weights * density)


def _panel_plan(plan: AdaptiveQuadraturePlan, panel_count: int) -> AdaptiveQuadraturePlan:
    absolute = (
        None if plan.absolute_tolerance is None else plan.absolute_tolerance / panel_count
    )
    relative = (
        None if plan.relative_tolerance is None else plan.relative_tolerance / panel_count
    )
    return AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=plan.breakpoints,
        collect_partition=plan.collect_partition,
        throw=False,
    )


def evaluate_laplace_adaptive_2d(
    potential,
    targets: ArrayLike,
    plan: AdaptiveQuadraturePlan,
    interactions: PanelInteractionReport2D,
    /,
) -> AdaptiveLayerEvaluation2D:
    """Evaluate Laplace layers with shared adaptive estimates for every panel."""
    if not isinstance(plan, AdaptiveQuadraturePlan):
        raise TypeError("plan must be an AdaptiveQuadraturePlan.")
    values = jnp.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("Adaptive layer targets must have shape (target_count, 2).")
    if interactions.near_mask.shape[0] != values.shape[0]:
        raise ValueError("Interaction report does not match the target batch.")
    panelization = potential.panelization
    panel_plan = _panel_plan(plan, panelization.panel_count)
    output_values = []
    errors = []
    statuses = []
    evaluations = []
    near_count = 0
    far_count = 0
    failed_count = 0
    for target_index, target in enumerate(values):
        target_value = jnp.asarray(0.0)
        target_error = jnp.asarray(0.0)
        target_status = jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32)
        target_evaluations = jnp.asarray(0, dtype=jnp.int32)
        collision = jnp.all(
            target[None, :] == panelization.points,
            axis=-1,
        )
        target_node_index = (
            int(jnp.argmax(collision)) if bool(jnp.any(collision)) else None
        )
        self_panel_id = (
            None
            if target_node_index is None
            else target_node_index // panelization.quadrature_order
        )
        for panel_id in range(panelization.panel_count):
            is_self = self_panel_id == panel_id
            is_near = is_self or bool(interactions.near_mask[target_index, panel_id])
            near_count += int(is_near)
            far_count += int(not is_near)
            bounds = panelization.panel_reference_bounds[panel_id]
            if is_self:
                estimate = evaluate_laplace_single_layer_self_panel_2d(
                    potential,
                    panel_id,
                    panelization.references[target_node_index, 0],
                    panel_plan,
                )
            else:
                estimate = adaptive_interval_callable(
                    lambda reference: _panel_integrand(
                        potential,
                        target,
                        panel_id,
                        reference,
                    ),
                    bounds,
                    panel_plan,
                )
            target_value = target_value + estimate.value
            if estimate.error_estimate is not None:
                target_error = target_error + estimate.error_estimate
            target_status = jnp.maximum(target_status, estimate.status)
            target_evaluations = target_evaluations + estimate.num_evaluations
            failed_count += int(not bool(estimate.successful))
        output_values.append(target_value)
        errors.append(target_error)
        statuses.append(target_status)
        evaluations.append(target_evaluations)
    values_ = jnp.stack(output_values)
    errors_ = jnp.stack(errors)
    statuses_ = jnp.stack(statuses)
    evaluations_ = jnp.stack(evaluations)
    error_estimate = jnp.max(errors_)
    status = jnp.max(statuses_)
    decision_dtype = jnp.real(values_).dtype
    absolute = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if plan.absolute_tolerance is None
        else jnp.asarray(plan.absolute_tolerance, dtype=decision_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if plan.relative_tolerance is None
        else jnp.asarray(plan.relative_tolerance, dtype=decision_dtype)
    )
    magnitude = jnp.max(jnp.abs(values_))
    accuracy_supported = (
        (status == int(IntegrationStatus.CONVERGED))
        & jnp.isfinite(error_estimate)
        & (error_estimate <= absolute + relative * magnitude)
    )
    if plan.throw:
        values_ = eqx.error_if(
            values_,
            ~accuracy_supported,
            "Adaptive layer quadrature failed its global accuracy contract.",
        )
    return AdaptiveLayerEvaluation2D(
        values=values_,
        error_estimate=error_estimate,
        status=status,
        num_evaluations=jnp.sum(evaluations_),
        accuracy_supported=accuracy_supported,
        near_panel_count=near_count,
        far_panel_count=far_count,
        failed_panel_count=failed_count,
    )


__all__ = [
    "AdaptiveLayerEvaluation2D",
    "PanelInteractionReport2D",
    "evaluate_laplace_single_layer_self_panel_2d",
    "classify_panel_interactions_2d",
    "evaluate_laplace_adaptive_2d",
    "evaluate_helmholtz_single_layer_self_panel_weights_2d",
]

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
from ...._interpolation import barycentric_basis
from ...._strict import StrictModule
from ....geometry import GeometryCapability, SignReliability, ZeroSetAccuracy
from ....integration import (
    adaptive_interval_callable,
    AdaptiveQuadraturePlan,
    IntegrationStatus,
)
from ....operators.differential._jet import jet_terms
from ....special import hankel1


class QBXEvaluation2D(StrictModule):
    """Certified target-associated local expansion and coefficient evidence."""

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


def _kernel_value(potential, target: Array, source: Array, normal: Array) -> Array:
    from ._helmholtz2d import HelmholtzCombinedField2D, HelmholtzLayerPotential2D

    if isinstance(potential, HelmholtzCombinedField2D):
        single = potential.kernel.value(target, source)
        double = potential.kernel.source_normal_derivative(target, source, normal)
        return double - 1j * potential.eta * single
    if isinstance(potential, (HelmholtzLayerPotential2D,)):
        if potential.kind == "single":
            return potential.kernel.value(target, source)
        return potential.kernel.source_normal_derivative(target, source, normal)
    if potential.kind == "single":
        return potential.kernel.value(target, source)
    return potential.kernel.source_normal_derivative(target, source, normal)


def _series_multiply(left: Array, right: Array, order: int) -> Array:
    dtype = jnp.result_type(left, right)
    values = []
    for degree in range(order + 1):
        value = jnp.asarray(0.0, dtype=dtype)
        for index in range(degree + 1):
            value = value + left[index] * right[degree - index]
        values.append(value)
    return jnp.stack(values)


def _series_reciprocal(values: Array, order: int) -> Array:
    reciprocal = jnp.zeros((order + 1,), dtype=values.dtype)
    reciprocal = reciprocal.at[0].set(1.0 / values[0])
    for degree in range(1, order + 1):
        value = jnp.asarray(0.0, dtype=values.dtype)
        for index in range(1, degree + 1):
            value = value + values[index] * reciprocal[degree - index]
        reciprocal = reciprocal.at[degree].set(-value / values[0])
    return reciprocal


def _radial_series(
    center: Array,
    source: Array,
    direction: Array,
    order: int,
) -> Array:
    offset = center - source
    squared = jnp.zeros((order + 1,), dtype=center.dtype)
    squared = squared.at[0].set(jnp.dot(offset, offset))
    if order >= 1:
        squared = squared.at[1].set(2.0 * jnp.dot(offset, direction))
    if order >= 2:
        squared = squared.at[2].set(1.0)
    radius = jnp.zeros((order + 1,), dtype=center.dtype)
    radius = radius.at[0].set(jnp.sqrt(squared[0]))
    for degree in range(1, order + 1):
        convolution = jnp.asarray(0.0, dtype=center.dtype)
        for index in range(1, degree):
            convolution = convolution + radius[index] * radius[degree - index]
        radius = radius.at[degree].set(
            (squared[degree] - convolution) / (2.0 * radius[0])
        )
    return radius


def _bessel_derivatives(
    bessel_order: int,
    argument: Array,
    order: int,
) -> Array:
    """Differentiate integer-order Hankel functions by their ODE recurrence."""
    value_zero = hankel1(float(bessel_order), argument)
    derivatives = jnp.zeros((order + 1,), dtype=value_zero.dtype)
    derivatives = derivatives.at[0].set(value_zero)
    if order >= 1:
        first = (
            -hankel1(1.0, argument)
            if bessel_order == 0
            else hankel1(float(bessel_order - 1), argument) - value_zero / argument
        )
        derivatives = derivatives.at[1].set(first)
    for degree in range(order - 1):
        lower_one = derivatives[degree - 1] if degree >= 1 else jnp.asarray(0.0 + 0.0j)
        lower_two = derivatives[degree - 2] if degree >= 2 else jnp.asarray(0.0 + 0.0j)
        next_value = (
            (2.0 * degree + 1.0) * argument * derivatives[degree + 1]
            + (degree * degree + argument * argument - bessel_order**2)
            * derivatives[degree]
            + 2.0 * degree * argument * lower_one
            + degree * (degree - 1.0) * lower_two
        )
        derivatives = derivatives.at[degree + 2].set(-next_value / argument**2)
    return derivatives


def _compose_series(
    derivatives: Array,
    argument: Array,
    order: int,
) -> Array:
    delta = argument.at[0].set(0.0)
    power = jnp.zeros((order + 1,), dtype=argument.dtype)
    power = power.at[0].set(1.0)
    result = jnp.zeros((order + 1,), dtype=derivatives.dtype)
    for degree in range(order + 1):
        result = result + derivatives[degree] * power / math.factorial(degree)
        if degree < order:
            power = _series_multiply(power, delta, order)
    return result


def _raw_derivatives(normalized: Array, order: int) -> Array:
    return jnp.stack(
        [normalized[degree] * math.factorial(degree) for degree in range(order + 1)]
    )


@eqx.filter_jit
def _helmholtz_directional_components(
    potential,
    center: Array,
    source: Array,
    normal: Array,
    direction: Array,
    order: int,
    include_double: bool,
) -> tuple[Array, Array]:
    """Evaluate directional terms without tracing custom Hankel JVPs."""
    wavenumber = jnp.asarray(potential.kernel.wavenumber, dtype=center.dtype)
    radius = _radial_series(center, source, direction, order)
    argument = wavenumber * radius
    single_series = _compose_series(
        _bessel_derivatives(0, argument[0], order),
        argument,
        order,
    )
    single = 0.25j * _raw_derivatives(single_series, order)
    if not include_double:
        return single, jnp.zeros_like(single)

    normal_projection = jnp.zeros((order + 1,), dtype=center.dtype)
    offset = center - source
    normal_projection = normal_projection.at[0].set(jnp.dot(offset, normal))
    if order >= 1:
        normal_projection = normal_projection.at[1].set(jnp.dot(direction, normal))
    geometric_factor = _series_multiply(
        normal_projection,
        _series_reciprocal(radius, order),
        order,
    )
    double_series = _series_multiply(
        _compose_series(
            _bessel_derivatives(1, argument[0], order),
            argument,
            order,
        ),
        geometric_factor,
        order,
    )
    double = 0.25j * wavenumber * _raw_derivatives(double_series, order)
    return single, double


def _directional_terms(
    potential,
    center: Array,
    source: Array,
    normal: Array,
    direction: Array,
    order: int,
) -> Array:
    from ._helmholtz2d import HelmholtzCombinedField2D, HelmholtzLayerPotential2D

    if isinstance(potential, HelmholtzCombinedField2D):
        single, double = _helmholtz_directional_components(
            potential,
            center,
            source,
            normal,
            direction,
            order,
            True,
        )
        return double - 1j * potential.eta * single
    if isinstance(potential, HelmholtzLayerPotential2D):
        single, double = _helmholtz_directional_components(
            potential,
            center,
            source,
            normal,
            direction,
            order,
            potential.kind == "double",
        )
        return double if potential.kind == "double" else single

    def function(distance: Array) -> Array:
        target = center + distance * direction
        return _kernel_value(potential, target, source, normal)

    primal, terms = jet_terms(
        function,
        jnp.asarray(0.0, dtype=center.dtype),
        jnp.asarray(1.0, dtype=center.dtype),
        order=order,
    )
    return jnp.stack((primal, *terms))


def _expand_coefficients(coefficients: Array, displacement: Array, order: int) -> Array:
    distance = jnp.linalg.norm(displacement)
    value = coefficients[0]
    factorial = 1.0
    for degree in range(1, order + 1):
        factorial *= degree
        value = value + coefficients[degree] * distance**degree / factorial
    return value


def _bounded_expansion_tail(
    retained_term: Array,
    first_omitted_term: Array,
    /,
) -> Array:
    retained = jnp.abs(retained_term)
    omitted = jnp.abs(first_omitted_term)
    tiny = jnp.finfo(jnp.real(retained_term).dtype).tiny
    ratio = omitted / jnp.maximum(retained, tiny)
    return jnp.where(
        ratio < 1.0,
        omitted / jnp.maximum(1.0 - ratio, 1e-15),
        jnp.asarray(jnp.inf, dtype=retained.dtype),
    )


def _panel_plan(
    plan: AdaptiveQuadraturePlan,
    panel_count: int,
    bounds: Array,
    /,
) -> AdaptiveQuadraturePlan:
    absolute = (
        None if plan.absolute_tolerance is None else plan.absolute_tolerance / panel_count
    )
    relative = (
        None if plan.relative_tolerance is None else plan.relative_tolerance / panel_count
    )
    lower, upper = (float(bounds[0]), float(bounds[1]))
    breakpoints = tuple(value for value in plan.breakpoints if lower < value < upper)
    return AdaptiveQuadraturePlan(
        plan.rule,
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        max_intervals=plan.max_intervals,
        max_evaluations=plan.max_evaluations,
        breakpoints=breakpoints,
        collect_partition=plan.collect_partition,
        throw=plan.throw,
    )


def _center_clearance(
    panelization,
    center: Array,
    associated_panel: int,
    radius: Array,
) -> Array:
    del associated_panel
    geometry = panelization.geometry
    if geometry is None:
        raise TypeError("QBX requires compiled geometry for continuous clearance.")
    if not geometry.has_capability(GeometryCapability.SIGNED_DISTANCE):
        raise TypeError("QBX requires a signed-distance geometry query.")
    certificate = geometry.field_certificate
    if (
        certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT
        or certificate.sign_reliability is not SignReliability.RELIABLE
        or not certificate.is_signed_distance
    ):
        raise TypeError("QBX requires exact, sign-reliable signed-distance evidence.")
    signed_distance = jnp.asarray(geometry.signed_distance(center[None, :]))[0]
    return jnp.abs(signed_distance) - radius


def _integrate_center_coefficients(
    potential,
    center: Array,
    direction: Array,
    order: int,
    plan: AdaptiveQuadraturePlan,
    /,
    *,
    selected_panels: tuple[int, ...] | None = None,
) -> tuple[Array, Array, Array, Array]:
    panelization = potential.panelization
    panel_ids = (
        tuple(range(panelization.panel_count))
        if selected_panels is None
        else tuple(selected_panels)
    )
    if not panel_ids:
        return (
            jnp.zeros((order + 1,), dtype=potential.density.dtype),
            jnp.asarray(0.0),
            jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
    panel_plan = None
    coefficients = None
    error = jnp.asarray(0.0)
    status = jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32)
    evaluations = jnp.asarray(0, dtype=jnp.int32)
    quadrature_order = panelization.quadrature_order
    for panel_id in panel_ids:
        start = panel_id * quadrature_order
        stop = start + quadrature_order
        chart = panelization.panel_chart_indices[panel_id]
        bounds = panelization.panel_reference_bounds[panel_id]
        panel_plan = _panel_plan(plan, len(panel_ids), bounds)
        node_reference = panelization.references[start:stop, 0]
        differences = node_reference[:, None] - node_reference[None, :]
        barycentric_weights = jnp.reciprocal(
            jnp.prod(differences + jnp.eye(quadrature_order), axis=1)
        )
        node_density = potential.density[start:stop]

        def coefficient_integrand(reference: Array) -> Array:
            chart_indices = jnp.full(reference.shape, chart, dtype=jnp.int32)
            frame = panelization.atlas.frame(chart_indices, reference[:, None])
            density_basis = jax.vmap(
                lambda location: barycentric_basis(
                    location,
                    node_reference,
                    barycentric_weights,
                )
            )(reference)
            density = density_basis @ node_density

            def one(source, normal, jacobian, weight):
                return (
                    _directional_terms(
                        potential,
                        center,
                        source,
                        normal,
                        direction,
                        order,
                    )
                    * jacobian
                    * weight
                )

            return jax.vmap(one)(
                frame.origin,
                frame.normal,
                frame.jacobian,
                density,
            )

        if panel_plan is None:
            raise RuntimeError("QBX panel quadrature plan was not prepared.")
        estimate = adaptive_interval_callable(
            coefficient_integrand,
            bounds,
            panel_plan,
        )
        coefficients = (
            estimate.value if coefficients is None else coefficients + estimate.value
        )
        error = (
            jnp.asarray(jnp.inf)
            if estimate.error_estimate is None
            else error + estimate.error_estimate
        )
        status = jnp.maximum(status, estimate.status)
        evaluations = evaluations + estimate.num_evaluations
    if coefficients is None:
        raise ValueError("QBX coefficient quadrature has no source panels.")
    return coefficients, error, status, evaluations


def evaluate_qbx_2d(
    potential,
    targets: ArrayLike,
    /,
    *,
    target_side: Literal["interior", "exterior", "boundary"],
    order: int,
    radius_factor: float,
    adaptive_plan: AdaptiveQuadraturePlan,
) -> QBXEvaluation2D:
    """Evaluate a layer field with coefficient quadrature and certified association."""
    values = jnp.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("QBX targets must have shape (target_count, 2).")
    expansion_order = int(order)
    factor = float(radius_factor)
    if expansion_order < 1 or not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("QBX order and radius_factor must be positive and finite.")
    if not isinstance(adaptive_plan, AdaptiveQuadraturePlan):
        raise TypeError("adaptive_plan must be an AdaptiveQuadraturePlan.")
    panelization = potential.panelization
    geometry = panelization.geometry
    if geometry is None or not geometry.has_capability(
        GeometryCapability.BOUNDARY_NORMAL
    ):
        raise TypeError("QBX requires certified target boundary normals.")
    panel_scales = jnp.sum(
        panelization.weights.reshape(
            (panelization.panel_count, panelization.quadrature_order)
        ),
        axis=1,
    )
    outputs = []
    truncation_errors = []
    coefficient_errors = []
    clearances = []
    statuses = []
    evaluations = []
    association_records = []
    for target in values:
        node_distances = jnp.linalg.norm(target[None, :] - panelization.points, axis=-1)
        node_index = int(jnp.argmin(node_distances))
        associated_panel = node_index // panelization.quadrature_order
        radius = factor * panel_scales[associated_panel]
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
        center_statuses = []
        center_evaluations = []
        target_clearance = jnp.inf
        for center in centers:
            clearance = _center_clearance(
                panelization,
                center,
                associated_panel,
                radius,
            )
            target_clearance = jnp.minimum(target_clearance, clearance)
            clearance_tolerance = 64.0 * jnp.finfo(values.dtype).eps
            clearance_valid = (
                clearance >= -clearance_tolerance
                if target_side == "boundary"
                else clearance > clearance_tolerance
            )
            if not bool(clearance_valid):
                center_values.append(jnp.asarray(jnp.nan, dtype=potential.density.dtype))
                center_errors.append(jnp.asarray(jnp.inf))
                center_statuses.append(
                    jnp.asarray(int(IntegrationStatus.INVALID_BOUNDS), dtype=jnp.int32)
                )
                center_evaluations.append(jnp.asarray(0, dtype=jnp.int32))
                coefficient_errors.append(jnp.asarray(jnp.inf))
                continue
            displacement = target - center
            direction = displacement / jnp.linalg.norm(displacement)
            coefficients, coefficient_error, status, evaluation_count = (
                _integrate_center_coefficients(
                    potential,
                    center,
                    direction,
                    expansion_order + 1,
                    adaptive_plan,
                )
            )
            center_values.append(
                _expand_coefficients(coefficients, target - center, expansion_order)
            )
            low_value = _expand_coefficients(
                coefficients,
                target - center,
                max(expansion_order - 1, 0),
            )
            extra_value = _expand_coefficients(
                coefficients,
                target - center,
                expansion_order + 1,
            )
            center_errors.append(
                _bounded_expansion_tail(
                    center_values[-1] - low_value,
                    extra_value - center_values[-1],
                )
            )
            center_statuses.append(status)
            center_evaluations.append(evaluation_count)
            coefficient_errors.append(coefficient_error)
        outputs.append(jnp.mean(jnp.stack(center_values)))
        truncation_errors.append(jnp.max(jnp.stack(center_errors)))
        statuses.append(jnp.max(jnp.stack(center_statuses)))
        evaluations.append(jnp.sum(jnp.stack(center_evaluations)))
        clearances.append(target_clearance)
        association_records.append((node_index, associated_panel))
    values_ = jnp.stack(outputs)
    truncation_error = jnp.max(jnp.stack(truncation_errors))
    coefficient_error = (
        jnp.max(jnp.stack(coefficient_errors))
        if coefficient_errors
        else jnp.asarray(jnp.inf)
    )
    error_estimate = truncation_error + coefficient_error
    status = jnp.max(jnp.stack(statuses))
    finite = jnp.all(jnp.isfinite(values_)) & jnp.isfinite(error_estimate)
    decision_dtype = jnp.real(values_).dtype
    absolute = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if adaptive_plan.absolute_tolerance is None
        else jnp.asarray(adaptive_plan.absolute_tolerance, dtype=decision_dtype)
    )
    relative = (
        jnp.sqrt(jnp.finfo(decision_dtype).eps)
        if adaptive_plan.relative_tolerance is None
        else jnp.asarray(adaptive_plan.relative_tolerance, dtype=decision_dtype)
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
        & (status == int(IntegrationStatus.CONVERGED))
        & clearance_supported
        & (error_estimate <= absolute + relative * jnp.max(jnp.abs(values_)))
    )
    if adaptive_plan.throw:
        values_ = eqx.error_if(
            values_,
            ~accuracy_supported,
            "QBX coefficient quadrature or expansion clearance failed.",
        )
    association_id = canonical_fingerprint(
        {
            "kind": "qbx-target-association-2d-v1",
            "panelization_id": panelization.panelization_id,
            "target_count": int(values.shape[0]),
            "target_side": target_side,
            "order": expansion_order,
            "radius_factor": factor,
            "associations": association_records,
        }
    )
    return QBXEvaluation2D(
        values=values_,
        coefficient_quadrature_error=coefficient_error,
        truncation_error=truncation_error,
        error_estimate=error_estimate,
        status=status,
        num_evaluations=jnp.sum(jnp.stack(evaluations)),
        accuracy_supported=accuracy_supported,
        clearance=jnp.stack(clearances),
        association_id=association_id,
        expansion_order=expansion_order,
    )


__all__ = ["QBXEvaluation2D", "evaluate_qbx_2d"]

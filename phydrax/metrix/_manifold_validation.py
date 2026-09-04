#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._manifold import AbstractRiemannianManifold
from ._state_geometry import AbstractStateGeometry


def _maximum_absolute(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return jnp.max(jnp.abs(array), initial=0.0)


def _relative_residual(value: ArrayLike, reference: ArrayLike, /) -> Array:
    value_array = jnp.asarray(value)
    reference_array = jnp.asarray(reference)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=jnp.result_type(value_array, reference_array, float)),
        _maximum_absolute(reference_array),
    )
    return _maximum_absolute(value_array - reference_array) / scale


class ManifoldValidationReport(StrictModule):
    """Aggregate numerical checks of an array-manifold implementation."""

    valid: Array
    contains: Array
    constraint_residual: Array
    projection_idempotence_residual: Array
    tangent_residual: Array
    metric_duality_residual: Array
    retraction_origin_residual: Array
    retraction_differential_residual: Array
    transported_tangent_residual: Array
    identity_transport_residual: Array
    transport_isometry_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        contains: Array,
        constraint_residual: Array,
        projection_idempotence_residual: Array,
        tangent_residual: Array,
        metric_duality_residual: Array,
        retraction_origin_residual: Array,
        retraction_differential_residual: Array,
        transported_tangent_residual: Array,
        identity_transport_residual: Array,
        transport_isometry_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.contains = jnp.asarray(contains, dtype=bool)
        self.constraint_residual = jnp.asarray(constraint_residual)
        self.projection_idempotence_residual = jnp.asarray(
            projection_idempotence_residual
        )
        self.tangent_residual = jnp.asarray(tangent_residual)
        self.metric_duality_residual = jnp.asarray(metric_duality_residual)
        self.retraction_origin_residual = jnp.asarray(retraction_origin_residual)
        self.retraction_differential_residual = jnp.asarray(
            retraction_differential_residual
        )
        self.transported_tangent_residual = jnp.asarray(transported_tangent_residual)
        self.identity_transport_residual = jnp.asarray(identity_transport_residual)
        self.transport_isometry_residual = jnp.asarray(transport_isometry_residual)


class StateGeometryValidationReport(StrictModule):
    """Aggregate numerical checks of a four-space state geometry."""

    valid: Array
    contains: Array
    retraction_origin_residual: Array
    retraction_differential_residual: Array
    inverse_retraction_residual: Array
    inverse_differential_residual: Array
    vjp_duality_residual: Array
    identity_transport_residual: Array
    transport_roundtrip_residual: Array
    transport_duality_residual: Array
    transport_isometry_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        contains: Array,
        retraction_origin_residual: Array,
        retraction_differential_residual: Array,
        inverse_retraction_residual: Array,
        inverse_differential_residual: Array,
        vjp_duality_residual: Array,
        identity_transport_residual: Array,
        transport_roundtrip_residual: Array,
        transport_duality_residual: Array,
        transport_isometry_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.contains = jnp.asarray(contains, dtype=bool)
        self.retraction_origin_residual = jnp.asarray(retraction_origin_residual)
        self.retraction_differential_residual = jnp.asarray(
            retraction_differential_residual
        )
        self.inverse_retraction_residual = jnp.asarray(inverse_retraction_residual)
        self.inverse_differential_residual = jnp.asarray(inverse_differential_residual)
        self.vjp_duality_residual = jnp.asarray(vjp_duality_residual)
        self.identity_transport_residual = jnp.asarray(identity_transport_residual)
        self.transport_roundtrip_residual = jnp.asarray(transport_roundtrip_residual)
        self.transport_duality_residual = jnp.asarray(transport_duality_residual)
        self.transport_isometry_residual = jnp.asarray(transport_isometry_residual)


def validate_manifold(
    manifold: AbstractRiemannianManifold,
    point: ArrayLike,
    ambient_vector: ArrayLike,
    /,
    *,
    ambient_cotangent: ArrayLike | None = None,
    step_scale: float = 1e-3,
    tolerance: float = 1e-5,
    raise_on_error: bool = True,
) -> ManifoldValidationReport:
    """Validate manifold laws at one point without modifying the manifold."""

    if not isinstance(manifold, AbstractRiemannianManifold):
        raise TypeError("manifold must be an AbstractRiemannianManifold.")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be positive.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    point_array = jnp.asarray(point)
    ambient = jnp.asarray(ambient_vector)
    if ambient.shape != point_array.shape:
        raise ValueError("ambient_vector must have the same shape as point.")
    cotangent = ambient if ambient_cotangent is None else jnp.asarray(ambient_cotangent)
    if cotangent.shape != point_array.shape:
        raise ValueError("ambient_cotangent must have the same shape as point.")

    contains = jnp.asarray(manifold.contains(point_array), dtype=bool).reshape(())
    constraint_residual = jnp.asarray(manifold.constraint_residual(point_array)).reshape(
        ()
    )
    tangent = manifold.project_tangent(point_array, ambient)
    projected_twice = manifold.project_tangent(point_array, tangent)
    projection_idempotence_residual = _relative_residual(projected_twice, tangent)
    tangent_residual = _relative_residual(tangent, projected_twice)

    rgradient = manifold.egrad_to_rgrad(point_array, cotangent)
    dual_left = manifold.inner(point_array, rgradient, tangent)
    dual_right = jnp.real(jnp.vdot(cotangent, tangent))
    metric_duality_residual = _relative_residual(dual_left, dual_right)

    zero = jnp.zeros_like(point_array)
    retraction_origin = manifold.retract(point_array, zero)
    retraction_origin_residual = _relative_residual(retraction_origin, point_array)
    _, retraction_velocity = jax.jvp(
        lambda step: manifold.retract(point_array, step),
        (zero,),
        (tangent,),
    )
    retraction_differential_residual = _relative_residual(retraction_velocity, tangent)

    step = jnp.asarray(step_scale, dtype=point_array.dtype) * tangent
    destination = manifold.retract(point_array, step)
    transported = manifold.transport(point_array, step, destination, tangent)
    transported_projection = manifold.project_tangent(destination, transported)
    transported_tangent_residual = _relative_residual(transported_projection, transported)
    identity_transport = manifold.transport(point_array, zero, point_array, tangent)
    identity_transport_residual = _relative_residual(identity_transport, tangent)
    if manifold.transport_is_isometric:
        source_norm = manifold.inner(point_array, tangent, tangent)
        target_norm = manifold.inner(destination, transported, transported)
        transport_isometry_residual = _relative_residual(target_norm, source_norm)
    else:
        transport_isometry_residual = jnp.asarray(0.0, dtype=point_array.dtype)

    residuals = jnp.stack(
        (
            constraint_residual,
            projection_idempotence_residual,
            tangent_residual,
            metric_duality_residual,
            retraction_origin_residual,
            retraction_differential_residual,
            transported_tangent_residual,
            identity_transport_residual,
            transport_isometry_residual,
        )
    )
    valid = contains & jnp.all(jnp.isfinite(residuals)) & jnp.all(residuals <= tolerance)
    report = ManifoldValidationReport(
        valid=valid,
        contains=contains,
        constraint_residual=constraint_residual,
        projection_idempotence_residual=projection_idempotence_residual,
        tangent_residual=tangent_residual,
        metric_duality_residual=metric_duality_residual,
        retraction_origin_residual=retraction_origin_residual,
        retraction_differential_residual=retraction_differential_residual,
        transported_tangent_residual=transported_tangent_residual,
        identity_transport_residual=identity_transport_residual,
        transport_isometry_residual=transport_isometry_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError("Manifold law validation failed.")
    return report


def validate_state_geometry(
    geometry: AbstractStateGeometry,
    state: ArrayLike,
    ambient_vector: ArrayLike,
    /,
    *,
    step_scale: float = 1e-3,
    tolerance: float = 1e-5,
    raise_on_error: bool = True,
) -> StateGeometryValidationReport:
    """Validate inverse, differential, VJP, and transport four-space laws."""

    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must be an AbstractStateGeometry.")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be positive.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    state_array = jnp.asarray(state)
    ambient = jnp.asarray(ambient_vector)
    if ambient.shape != state_array.shape:
        raise ValueError("ambient_vector must have the same shape as state storage.")

    contains = jnp.asarray(geometry.contains(state_array), dtype=bool).reshape(())
    tangent = jnp.asarray(geometry.project_tangent(state_array, ambient))
    zero_local = jnp.asarray(geometry.inverse_retract(state_array, state_array))
    local_velocity = jnp.asarray(
        geometry.retraction_inverse_jvp(state_array, state_array, tangent)
    )
    origin = jnp.asarray(geometry.retract(state_array, zero_local))
    retraction_origin_residual = _relative_residual(origin, state_array)
    recovered_tangent = jnp.asarray(
        geometry.retraction_jvp(state_array, zero_local, local_velocity)
    )
    retraction_differential_residual = _relative_residual(recovered_tangent, tangent)

    step = jnp.asarray(step_scale, dtype=zero_local.dtype) * local_velocity
    destination = jnp.asarray(geometry.retract(state_array, step))
    target_tangent = jnp.asarray(
        geometry.retraction_jvp(state_array, step, local_velocity)
    )
    chart = geometry.chart_evidence(
        state_array,
        step,
        local_velocity,
        target_tangent,
    )
    transport = geometry.transport_evidence(
        state_array,
        destination,
        tangent,
        geometry.transport_tangent(state_array, destination, tangent),
    )
    inverse_differential_residual = jnp.maximum(
        chart.forward_inverse_differential_residual,
        chart.inverse_forward_differential_residual,
    )
    residuals = jnp.stack(
        (
            retraction_origin_residual,
            retraction_differential_residual,
            chart.inverse_roundtrip_residual,
            inverse_differential_residual,
            chart.vjp_duality_residual,
            transport.identity_residual,
            transport.roundtrip_residual,
            transport.duality_residual,
            transport.isometry_residual,
        )
    )
    valid = (
        contains
        & chart.source_membership
        & chart.target_membership
        & transport.source_membership
        & transport.target_membership
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(residuals <= tolerance)
    )
    report = StateGeometryValidationReport(
        valid=valid,
        contains=contains,
        retraction_origin_residual=retraction_origin_residual,
        retraction_differential_residual=retraction_differential_residual,
        inverse_retraction_residual=chart.inverse_roundtrip_residual,
        inverse_differential_residual=inverse_differential_residual,
        vjp_duality_residual=chart.vjp_duality_residual,
        identity_transport_residual=transport.identity_residual,
        transport_roundtrip_residual=transport.roundtrip_residual,
        transport_duality_residual=transport.duality_residual,
        transport_isometry_residual=transport.isometry_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError("State-geometry law validation failed.")
    return report


__all__ = [
    "ManifoldValidationReport",
    "StateGeometryValidationReport",
    "validate_manifold",
    "validate_state_geometry",
]

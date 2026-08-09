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
    """Aggregate numerical checks of a differential-equation state geometry."""

    valid: Array
    contains: Array
    projection_idempotence_residual: Array
    local_roundtrip_residual: Array
    retraction_origin_residual: Array
    retraction_differential_residual: Array
    inverse_retraction_residual: Array
    pullback_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        contains: Array,
        projection_idempotence_residual: Array,
        local_roundtrip_residual: Array,
        retraction_origin_residual: Array,
        retraction_differential_residual: Array,
        inverse_retraction_residual: Array,
        pullback_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.contains = jnp.asarray(contains, dtype=bool)
        self.projection_idempotence_residual = jnp.asarray(
            projection_idempotence_residual
        )
        self.local_roundtrip_residual = jnp.asarray(local_roundtrip_residual)
        self.retraction_origin_residual = jnp.asarray(retraction_origin_residual)
        self.retraction_differential_residual = jnp.asarray(
            retraction_differential_residual
        )
        self.inverse_retraction_residual = jnp.asarray(inverse_retraction_residual)
        self.pullback_residual = jnp.asarray(pullback_residual)


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
    """Validate state-geometry retraction and coordinate laws at one state."""

    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must be an AbstractStateGeometry.")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be positive.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    state_array = jnp.asarray(state)
    ambient = jnp.asarray(ambient_vector)
    if ambient.shape != state_array.shape:
        raise ValueError("ambient_vector must have the same shape as state.")

    contains = jnp.asarray(geometry.contains(state_array), dtype=bool).reshape(())
    tangent = geometry.project_tangent(state_array, ambient)
    projected_twice = geometry.project_tangent(state_array, tangent)
    projection_idempotence_residual = _relative_residual(projected_twice, tangent)
    local = geometry.to_local(state_array, tangent)
    recovered_tangent = geometry.from_local(state_array, local)
    local_roundtrip_residual = _relative_residual(recovered_tangent, tangent)

    zero = jnp.zeros_like(state_array)
    origin = geometry.retract(state_array, zero)
    retraction_origin_residual = _relative_residual(origin, state_array)
    _, retraction_velocity = jax.jvp(
        lambda value: geometry.retract(state_array, value),
        (zero,),
        (local,),
    )
    retraction_differential_residual = _relative_residual(retraction_velocity, tangent)

    step = jnp.asarray(step_scale, dtype=state_array.dtype) * local
    destination = geometry.retract(state_array, step)
    inverse_step = geometry.inverse_retract(state_array, destination)
    inverse_retraction_residual = _relative_residual(inverse_step, step)
    if geometry.supports_exact_pullback:
        target_tangent = geometry.project_tangent(destination, ambient)
        pulled = geometry.pullback(state_array, step, target_tangent)
        _, pushed = jax.jvp(
            lambda value: geometry.retract(state_array, value),
            (step,),
            (pulled,),
        )
        pullback_residual = _relative_residual(pushed, target_tangent)
    else:
        pullback_residual = jnp.asarray(0.0, dtype=state_array.dtype)

    residuals = jnp.stack(
        (
            projection_idempotence_residual,
            local_roundtrip_residual,
            retraction_origin_residual,
            retraction_differential_residual,
            inverse_retraction_residual,
            pullback_residual,
        )
    )
    valid = contains & jnp.all(jnp.isfinite(residuals)) & jnp.all(residuals <= tolerance)
    report = StateGeometryValidationReport(
        valid=valid,
        contains=contains,
        projection_idempotence_residual=projection_idempotence_residual,
        local_roundtrip_residual=local_roundtrip_residual,
        retraction_origin_residual=retraction_origin_residual,
        retraction_differential_residual=retraction_differential_residual,
        inverse_retraction_residual=inverse_retraction_residual,
        pullback_residual=pullback_residual,
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

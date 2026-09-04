#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._fields import IntensityPlane, ScalarPlaneField, TangentialPlaneField


PlaneField = ScalarPlaneField | TangentialPlaneField


def coherent_mode_intensity(
    modes: Sequence[PlaneField],
    weights: ArrayLike,
    active: ArrayLike,
    /,
) -> IntensityPlane:
    """Reduce mutually incoherent coherent modes to a weighted intensity.

    This computes the coherent-mode decomposition sum ``Σ_m w_m |E_m|²``;
    amplitudes from distinct modes are never summed, so no cross terms are
    introduced. Inactive fixed-capacity slots are masked before any arithmetic,
    allowing their amplitudes and weights to contain NaNs without contamination.
    """
    modes_ = tuple(modes)
    if not modes_:
        raise ValueError("modes must contain at least one plane field.")
    reference = modes_[0]
    if not isinstance(reference, (ScalarPlaneField, TangentialPlaneField)):
        raise TypeError("modes must contain scalar or tangential plane fields.")
    field_type = type(reference)
    for mode in modes_[1:]:
        if type(mode) is not field_type:
            raise TypeError("All coherent modes must have the same concrete field type.")
        if mode.space.space_id != reference.space.space_id:
            raise ValueError("All coherent modes must use the same plane space.")

    count = len(modes_)
    supplied_weights = jnp.asarray(weights)
    if (
        jnp.iscomplexobj(supplied_weights)
        or not jnp.issubdtype(supplied_weights.dtype, jnp.number)
        or jnp.issubdtype(supplied_weights.dtype, jnp.bool_)
    ):
        raise TypeError("Coherent-mode weights must be real numeric values.")
    weights_ = supplied_weights.astype(
        jnp.result_type(supplied_weights.dtype, jnp.float32)
    )
    active_ = jnp.asarray(active)
    if active_.dtype != jnp.bool_:
        raise TypeError("active must be a boolean array.")
    if weights_.shape != (count,) or active_.shape != (count,):
        raise ValueError(
            f"weights and active must both have shape ({count},); got "
            f"{weights_.shape} and {active_.shape}."
        )
    invalid_active_weight = active_ & ((~jnp.isfinite(weights_)) | (weights_ < 0.0))
    safe_weights = jnp.where(active_, weights_, 0.0)
    safe_weights = eqx.error_if(
        safe_weights,
        jnp.any(invalid_active_weight),
        "Active coherent-mode weights must be finite and nonnegative.",
    )

    values = jnp.stack(tuple(mode.values for mode in modes_), axis=0)
    mask_shape = (count,) + (1,) * (values.ndim - 1)
    safe_values = jnp.where(active_.reshape(mask_shape), values, 0.0)
    modal_density = jnp.real(safe_values * jnp.conj(safe_values))
    if isinstance(reference, TangentialPlaneField):
        modal_density = jnp.sum(modal_density, axis=-1)
    intensity = jnp.sum(
        safe_weights.reshape((count,) + (1,) * len(reference.space.shape))
        * modal_density,
        axis=0,
    )

    metadata_consistent = jnp.asarray(True)
    for mode in modes_[1:]:
        metadata_consistent = (
            metadata_consistent
            & (mode.angular_frequency == reference.angular_frequency)
            & (mode.longitudinal_coordinate == reference.longitudinal_coordinate)
        )
    intensity = eqx.error_if(
        intensity,
        ~metadata_consistent,
        "All coherent modes must share angular frequency and longitudinal coordinate.",
    )
    return IntensityPlane(
        reference.space,
        intensity,
        reference.angular_frequency,
        reference.longitudinal_coordinate,
    )


__all__ = ["coherent_mode_intensity"]

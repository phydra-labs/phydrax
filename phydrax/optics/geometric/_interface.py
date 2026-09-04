#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class OpticalRayState(StrictModule, NonTrainableState):
    """Narrow fixed-shape state for rays in real isotropic media."""

    origins: Array
    directions: Array
    refractive_indices: Array
    geometric_path_lengths: Array
    optical_path_lengths: Array

    def __init__(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        refractive_indices: ArrayLike,
        geometric_path_lengths: ArrayLike | None = None,
        optical_path_lengths: ArrayLike | None = None,
    ):
        origins_ = jnp.asarray(origins)
        directions_ = jnp.asarray(directions)
        indices_ = jnp.asarray(refractive_indices)
        if origins_.shape != directions_.shape or origins_.shape[-1:] != (3,):
            raise ValueError("origins and directions must have the same shape B + (3,).")
        if any(
            jnp.issubdtype(value.dtype, jnp.complexfloating)
            for value in (origins_, directions_, indices_)
        ):
            raise TypeError("Geometric ray state must be real-valued.")
        batch_shape = origins_.shape[:-1]
        if jnp.broadcast_shapes(indices_.shape, batch_shape) != batch_shape:
            raise ValueError("refractive_indices must be broadcastable to B.")
        indices_ = jnp.broadcast_to(indices_, batch_shape)
        geometric_ = (
            jnp.zeros(batch_shape)
            if geometric_path_lengths is None
            else jnp.asarray(geometric_path_lengths)
        )
        optical_ = (
            jnp.zeros(batch_shape)
            if optical_path_lengths is None
            else jnp.asarray(optical_path_lengths)
        )
        if jnp.issubdtype(geometric_.dtype, jnp.complexfloating) or jnp.issubdtype(
            optical_.dtype, jnp.complexfloating
        ):
            raise TypeError("Ray path lengths must be real-valued.")
        if (
            jnp.broadcast_shapes(geometric_.shape, batch_shape) != batch_shape
            or jnp.broadcast_shapes(optical_.shape, batch_shape) != batch_shape
        ):
            raise ValueError("Ray path lengths must be broadcastable to B.")
        geometric_ = jnp.broadcast_to(geometric_, batch_shape)
        optical_ = jnp.broadcast_to(optical_, batch_shape)

        dtype = jnp.result_type(
            origins_, directions_, indices_, geometric_, optical_, 0.0
        )
        origins_ = origins_.astype(dtype)
        directions_ = directions_.astype(dtype)
        direction_norm = jnp.sqrt(jnp.sum(directions_ * directions_, axis=-1))
        direction_ok = jnp.isfinite(direction_norm) & (direction_norm > 0.0)
        directions_ = jnp.where(
            direction_ok[..., None],
            directions_ / jnp.where(direction_ok, direction_norm, 1.0)[..., None],
            0.0,
        )
        self.origins = origins_
        self.directions = directions_
        self.refractive_indices = indices_.astype(dtype)
        self.geometric_path_lengths = geometric_.astype(dtype)
        self.optical_path_lengths = optical_.astype(dtype)


class RefractiveInterfaceStatus(IntEnum):
    """Status of a real-isotropic interface evaluation."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DIRECTION = 2
    INVALID_NORMAL = 3
    INVALID_REFRACTIVE_INDEX = 4
    WRONG_SIDE_INCIDENCE = 5
    GRAZING_INCIDENCE = 6
    TOTAL_INTERNAL_REFLECTION = 7
    NUMERICAL_FAILURE = 8


class RefractiveInterfaceResult(StrictModule, NonTrainableState):
    """Snell directions, Fresnel coefficients, and branch evidence.

    The final coefficient component is ordered ``(s, p)``. With normals from
    incident to transmitted medium and local ``s`` perpendicular to the plane
    of incidence, each p basis is ``s × direction``. This convention makes the
    normal-incidence p reflection amplitude the negative of the s amplitude.
    """

    transmitted_directions: Array
    reflected_directions: Array
    incident_cosine: Array
    transmitted_cosine: Array
    snell_discriminant: Array
    reflection_amplitudes: Array
    transmission_amplitudes: Array
    reflectance: Array
    transmittance: Array
    energy_balance_error: Array
    transmission_valid: Array
    reflection_valid: Array
    status: Array


def evaluate_refractive_interface(
    directions: ArrayLike,
    normals: ArrayLike,
    incident_index: ArrayLike,
    transmitted_index: ArrayLike,
    /,
    *,
    incidence_tolerance: float = 1e-10,
) -> RefractiveInterfaceResult:
    """Evaluate lossless real-isotropic Snell and Fresnel laws.

    Normals point from the incident medium to the transmitted medium and a ray
    must therefore have positive direction-normal cosine. Complex amplitudes
    use the ``exp(-i omega t)`` convention. During total internal reflection,
    the reflected direction and phase are valid while no real transmitted ray
    is returned.
    """

    if not math.isfinite(incidence_tolerance) or incidence_tolerance < 0.0:
        raise ValueError("incidence_tolerance must be finite and non-negative.")
    directions_ = jnp.asarray(directions)
    normals_ = jnp.asarray(normals)
    incident_ = jnp.asarray(incident_index)
    transmitted_ = jnp.asarray(transmitted_index)
    if directions_.shape[-1:] != (3,) or normals_.shape[-1:] != (3,):
        raise ValueError("directions and normals must have shape B + (3,).")
    if any(
        jnp.issubdtype(value.dtype, jnp.complexfloating)
        for value in (directions_, normals_, incident_, transmitted_)
    ):
        raise TypeError("Real-isotropic interface inputs must be real-valued.")
    batch_shape = jnp.broadcast_shapes(
        directions_.shape[:-1],
        normals_.shape[:-1],
        incident_.shape,
        transmitted_.shape,
    )
    directions_ = jnp.broadcast_to(directions_, batch_shape + (3,))
    normals_ = jnp.broadcast_to(normals_, batch_shape + (3,))
    incident_ = jnp.broadcast_to(incident_, batch_shape)
    transmitted_ = jnp.broadcast_to(transmitted_, batch_shape)

    dtype = jnp.result_type(directions_, normals_, incident_, transmitted_, 0.0)
    directions_ = directions_.astype(dtype)
    normals_ = normals_.astype(dtype)
    incident_ = incident_.astype(dtype)
    transmitted_ = transmitted_.astype(dtype)
    finite = (
        jnp.all(jnp.isfinite(directions_), axis=-1)
        & jnp.all(jnp.isfinite(normals_), axis=-1)
        & jnp.isfinite(incident_)
        & jnp.isfinite(transmitted_)
    )
    safe_directions = jnp.where(finite[..., None], directions_, 0.0)
    safe_normals = jnp.where(finite[..., None], normals_, 0.0)
    direction_norm = jnp.sqrt(jnp.sum(safe_directions * safe_directions, axis=-1))
    normal_norm = jnp.sqrt(jnp.sum(safe_normals * safe_normals, axis=-1))
    direction_ok = direction_norm > 0.0
    normal_ok = normal_norm > 0.0
    index_ok = (incident_ > 0.0) & (transmitted_ > 0.0)
    direction = safe_directions / jnp.where(direction_ok, direction_norm, 1.0)[..., None]
    normal = safe_normals / jnp.where(normal_ok, normal_norm, 1.0)[..., None]

    incident_cosine = jnp.sum(direction * normal, axis=-1)
    wrong_side = incident_cosine < -incidence_tolerance
    grazing = ~wrong_side & (incident_cosine <= incidence_tolerance)
    tangent = direction - incident_cosine[..., None] * normal
    ratio = incident_ / jnp.where(index_ok, transmitted_, 1.0)
    transmitted_tangent = ratio[..., None] * tangent
    snell_discriminant = 1.0 - jnp.sum(transmitted_tangent * transmitted_tangent, axis=-1)
    total_internal_reflection = snell_discriminant < 0.0
    transmitted_cosine = jnp.sqrt(jnp.maximum(snell_discriminant, 0.0))
    candidate_transmitted = transmitted_tangent + transmitted_cosine[..., None] * normal
    candidate_reflected = direction - 2.0 * incident_cosine[..., None] * normal

    complex_dtype = jnp.result_type(dtype, 1j)
    complex_cosine = jnp.sqrt(snell_discriminant.astype(complex_dtype))
    incident_complex = incident_.astype(complex_dtype)
    transmitted_complex = transmitted_.astype(complex_dtype)
    cosine_complex = incident_cosine.astype(complex_dtype)
    denominator_s = (
        incident_complex * cosine_complex + transmitted_complex * complex_cosine
    )
    denominator_p = (
        transmitted_complex * cosine_complex + incident_complex * complex_cosine
    )
    safe_denominator_s = jnp.where(denominator_s != 0.0, denominator_s, 1.0 + 0.0j)
    safe_denominator_p = jnp.where(denominator_p != 0.0, denominator_p, 1.0 + 0.0j)
    reflection_s = (
        incident_complex * cosine_complex - transmitted_complex * complex_cosine
    ) / safe_denominator_s
    reflection_p = (
        transmitted_complex * cosine_complex - incident_complex * complex_cosine
    ) / safe_denominator_p
    transmission_s = 2.0 * incident_complex * cosine_complex / safe_denominator_s
    transmission_p = 2.0 * incident_complex * cosine_complex / safe_denominator_p
    reflection_amplitudes = jnp.stack((reflection_s, reflection_p), axis=-1)
    transmission_amplitudes = jnp.stack((transmission_s, transmission_p), axis=-1)
    reflectance = jnp.real(
        reflection_amplitudes * jnp.conj(reflection_amplitudes)
    ).astype(dtype)
    flux_ratio = (
        transmitted_
        * transmitted_cosine
        / jnp.where(incident_ * incident_cosine > 0.0, incident_ * incident_cosine, 1.0)
    )
    propagating_transmittance = flux_ratio[..., None] * jnp.real(
        transmission_amplitudes * jnp.conj(transmission_amplitudes)
    ).astype(dtype)
    transmittance = jnp.where(
        total_internal_reflection[..., None], 0.0, propagating_transmittance
    )

    optical_inputs_valid = finite & direction_ok & normal_ok & index_ok
    oriented = optical_inputs_valid & ~wrong_side & ~grazing
    coefficient_finite = (
        jnp.all(jnp.isfinite(reflection_amplitudes), axis=-1)
        & jnp.all(jnp.isfinite(transmission_amplitudes), axis=-1)
        & jnp.all(jnp.isfinite(reflectance), axis=-1)
        & jnp.all(jnp.isfinite(transmittance), axis=-1)
        & jnp.isfinite(snell_discriminant)
    )
    numerical_failure = oriented & ~coefficient_finite
    status = jnp.where(
        ~finite,
        int(RefractiveInterfaceStatus.NONFINITE_INPUT),
        jnp.where(
            ~direction_ok,
            int(RefractiveInterfaceStatus.INVALID_DIRECTION),
            jnp.where(
                ~normal_ok,
                int(RefractiveInterfaceStatus.INVALID_NORMAL),
                jnp.where(
                    ~index_ok,
                    int(RefractiveInterfaceStatus.INVALID_REFRACTIVE_INDEX),
                    jnp.where(
                        wrong_side,
                        int(RefractiveInterfaceStatus.WRONG_SIDE_INCIDENCE),
                        jnp.where(
                            grazing,
                            int(RefractiveInterfaceStatus.GRAZING_INCIDENCE),
                            jnp.where(
                                numerical_failure,
                                int(RefractiveInterfaceStatus.NUMERICAL_FAILURE),
                                jnp.where(
                                    total_internal_reflection,
                                    int(
                                        RefractiveInterfaceStatus.TOTAL_INTERNAL_REFLECTION
                                    ),
                                    int(RefractiveInterfaceStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    transmission_valid = status == int(RefractiveInterfaceStatus.SUCCESS)
    reflection_valid = transmission_valid | (
        status == int(RefractiveInterfaceStatus.TOTAL_INTERNAL_REFLECTION)
    )
    coefficient_valid = reflection_valid
    transmitted_directions = jnp.where(
        transmission_valid[..., None], candidate_transmitted, 0.0
    )
    reflected_directions = jnp.where(
        reflection_valid[..., None], candidate_reflected, 0.0
    )
    reflection_amplitudes = jnp.where(
        coefficient_valid[..., None], reflection_amplitudes, 0.0 + 0.0j
    )
    transmission_amplitudes = jnp.where(
        coefficient_valid[..., None], transmission_amplitudes, 0.0 + 0.0j
    )
    reflectance = jnp.where(coefficient_valid[..., None], reflectance, 0.0)
    transmittance = jnp.where(coefficient_valid[..., None], transmittance, 0.0)
    energy_balance_error = jnp.where(
        coefficient_valid,
        jnp.max(jnp.abs(reflectance + transmittance - 1.0), axis=-1),
        0.0,
    )
    incident_cosine = jnp.where(optical_inputs_valid, incident_cosine, 0.0)
    transmitted_cosine = jnp.where(transmission_valid, transmitted_cosine, 0.0)
    snell_discriminant = jnp.where(optical_inputs_valid, snell_discriminant, 0.0)
    return RefractiveInterfaceResult(
        transmitted_directions,
        reflected_directions,
        incident_cosine,
        transmitted_cosine,
        snell_discriminant,
        reflection_amplitudes,
        transmission_amplitudes,
        reflectance,
        transmittance,
        energy_balance_error,
        transmission_valid,
        reflection_valid,
        status,
    )


__all__ = [
    "OpticalRayState",
    "RefractiveInterfaceResult",
    "RefractiveInterfaceStatus",
    "evaluate_refractive_interface",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._phase_field import double_well_free_energy_density
from ..._strict import StrictModule
from ._lattice import LatticeBoltzmannVelocitySet


class InterfacialFields(StrictModule):
    """Isotropic diffuse-interface geometry and continuum surface force."""

    gradient: Array
    gradient_magnitude: Array
    normal: Array
    curvature: Array
    surface_delta: Array
    force_density: Array


def _spatial_axes(velocity_set: LatticeBoltzmannVelocitySet, /) -> tuple[int, ...]:
    return tuple(range(velocity_set.dimension))


def _validate_scalar_field(
    field: ArrayLike, velocity_set: LatticeBoltzmannVelocitySet, /
) -> Array:
    if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
        raise TypeError("velocity_set must be a LatticeBoltzmannVelocitySet.")
    values = jnp.asarray(field)
    if values.ndim != velocity_set.dimension:
        raise ValueError(
            "A lattice scalar field must have one axis per spatial dimension."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise TypeError("Interfacial fields must have an inexact dtype.")
    return values


def _cell_size(value: ArrayLike, dtype, /) -> Array:
    cell_size = jnp.asarray(value, dtype=dtype)
    if cell_size.shape != ():
        raise ValueError("cell_size must be scalar.")
    return eqx.error_if(
        cell_size,
        ~jnp.isfinite(cell_size) | (cell_size <= 0.0),
        "cell_size must be finite and positive.",
    )


def isotropic_gradient(
    field: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    cell_size: ArrayLike = 1.0,
    /,
) -> Array:
    """Return the velocity-quadrature gradient on a periodic lattice."""

    values = _validate_scalar_field(field, velocity_set)
    dx = _cell_size(cell_size, values.dtype)
    weights = jnp.asarray(velocity_set.weights, dtype=values.dtype)
    velocities = jnp.asarray(velocity_set.velocities, dtype=values.dtype)
    axes = _spatial_axes(velocity_set)
    neighbours = jnp.stack(
        tuple(
            jnp.roll(
                values, shift=tuple(-component for component in direction), axis=axes
            )
            for direction in velocity_set.velocity_tuples
        ),
        axis=-1,
    )
    cs2 = jnp.asarray(velocity_set.sound_speed_squared, dtype=values.dtype)
    return oe.contract("...q,q,qd->...d", neighbours, weights, velocities) / (cs2 * dx)


def isotropic_laplacian(
    field: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    cell_size: ArrayLike = 1.0,
    /,
) -> Array:
    """Return the leading-order isotropic lattice Laplacian."""

    values = _validate_scalar_field(field, velocity_set)
    dx = _cell_size(cell_size, values.dtype)
    weights = jnp.asarray(velocity_set.weights, dtype=values.dtype)
    axes = _spatial_axes(velocity_set)
    neighbours = jnp.stack(
        tuple(
            jnp.roll(
                values, shift=tuple(-component for component in direction), axis=axes
            )
            for direction in velocity_set.velocity_tuples
        ),
        axis=-1,
    )
    cs2 = jnp.asarray(velocity_set.sound_speed_squared, dtype=values.dtype)
    return (
        2.0 * jnp.sum(weights * (neighbours - values[..., None]), axis=-1) / (cs2 * dx**2)
    )


def isotropic_divergence(
    vector: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    cell_size: ArrayLike = 1.0,
    /,
) -> Array:
    """Return the matching velocity-quadrature divergence."""

    values = jnp.asarray(vector)
    if (
        values.ndim != velocity_set.dimension + 1
        or values.shape[-1] != velocity_set.dimension
    ):
        raise ValueError(
            "A lattice vector field must have one trailing spatial component axis."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise TypeError("Interfacial fields must have an inexact dtype.")
    dx = _cell_size(cell_size, values.dtype)
    weights = jnp.asarray(velocity_set.weights, dtype=values.dtype)
    velocities = jnp.asarray(velocity_set.velocities, dtype=values.dtype)
    axes = _spatial_axes(velocity_set)
    neighbours = jnp.stack(
        tuple(
            jnp.roll(
                values, shift=tuple(-component for component in direction), axis=axes
            )
            for direction in velocity_set.velocity_tuples
        ),
        axis=-2,
    )
    cs2 = jnp.asarray(velocity_set.sound_speed_squared, dtype=values.dtype)
    return oe.contract("...qd,q,qd->...", neighbours, weights, velocities) / (cs2 * dx)


def normalized_gradient(
    field: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    cell_size: ArrayLike = 1.0,
    /,
    *,
    epsilon: ArrayLike = 1.0e-14,
) -> tuple[Array, Array, Array]:
    """Return gradient, magnitude, and a zero-safe unit normal."""

    gradient = isotropic_gradient(field, velocity_set, cell_size)
    magnitude = jnp.sqrt(oe.contract("...d,...d->...", gradient, gradient))
    threshold = jnp.asarray(epsilon, dtype=gradient.dtype)
    if threshold.shape != ():
        raise ValueError("epsilon must be scalar.")
    threshold = eqx.error_if(
        threshold,
        ~jnp.isfinite(threshold) | (threshold <= 0.0),
        "epsilon must be finite and positive.",
    )
    normal = gradient / jnp.maximum(magnitude, threshold)[..., None]
    normal = jnp.where((magnitude > threshold)[..., None], normal, 0.0)
    return gradient, magnitude, normal


def _normalise_vectors(vectors: Array, epsilon: Array, /) -> tuple[Array, Array]:
    magnitude = jnp.sqrt(oe.contract("...d,...d->...", vectors, vectors))
    unit = vectors / jnp.maximum(magnitude, epsilon)[..., None]
    return jnp.where((magnitude > epsilon)[..., None], unit, 0.0), magnitude


def _fallback_tangent(wall_normal: Array, /) -> Array:
    dimension = wall_normal.shape[-1]
    if dimension == 2:
        return jnp.stack((-wall_normal[..., 1], wall_normal[..., 0]), axis=-1)
    if dimension == 3:
        basis_index = jnp.argmin(jnp.abs(wall_normal), axis=-1)
        basis = jnp.eye(3, dtype=wall_normal.dtype)[basis_index]
        return jnp.cross(wall_normal, basis)
    raise ValueError("Static wetting is defined only in dimension two or three.")


def static_contact_angle_normal(
    interface_normal: ArrayLike,
    wall_normal: ArrayLike,
    contact_angle: ArrayLike,
    wetting_mask: ArrayLike,
    /,
    *,
    epsilon: ArrayLike = 1.0e-14,
) -> Array:
    """Impose a signed static contact angle without changing non-wall cells.

    ``wall_normal`` points out of the fluid.  Swapping phase labels and replacing
    ``contact_angle`` by its supplement negates the returned interface normal.
    """

    interface = jnp.asarray(interface_normal)
    wall = jnp.asarray(wall_normal, dtype=interface.dtype)
    mask = jnp.asarray(wetting_mask, dtype=bool)
    if interface.shape != wall.shape or interface.ndim < 1:
        raise ValueError("Interface and wall normals must have identical shapes.")
    if interface.shape[-1] not in (2, 3) or mask.shape != interface.shape[:-1]:
        raise ValueError("wetting_mask and normal dimensions are incompatible.")
    angle = jnp.asarray(contact_angle, dtype=interface.dtype)
    threshold = jnp.asarray(epsilon, dtype=interface.dtype)
    if angle.shape != () or threshold.shape != ():
        raise ValueError("contact_angle and epsilon must be scalar.")
    angle = eqx.error_if(
        angle,
        ~jnp.isfinite(angle) | (angle <= 0.0) | (angle >= jnp.pi),
        "contact_angle must lie strictly between zero and pi.",
    )
    threshold = eqx.error_if(
        threshold,
        ~jnp.isfinite(threshold) | (threshold <= 0.0),
        "epsilon must be finite and positive.",
    )
    wall_unit, wall_magnitude = _normalise_vectors(wall, threshold)
    wall_unit = eqx.error_if(
        wall_unit,
        jnp.any(mask & (~jnp.isfinite(wall_magnitude) | (wall_magnitude <= threshold))),
        "Every wetting cell requires a finite nonzero wall normal.",
    )
    tangent = (
        interface
        - oe.contract("...d,...d->...", interface, wall_unit)[..., None] * wall_unit
    )
    tangent_unit, tangent_magnitude = _normalise_vectors(tangent, threshold)
    fallback, _ = _normalise_vectors(_fallback_tangent(wall_unit), threshold)
    tangent_unit = jnp.where(
        (tangent_magnitude > threshold)[..., None], tangent_unit, fallback
    )
    imposed = jnp.cos(angle) * wall_unit + jnp.sin(angle) * tangent_unit
    return jnp.where(mask[..., None], imposed, interface)


def continuum_surface_force(
    colour: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    surface_tension: ArrayLike,
    cell_size: ArrayLike = 1.0,
    /,
    *,
    wall_normal: ArrayLike | None = None,
    wetting_mask: ArrayLike | None = None,
    contact_angle: ArrayLike = 0.5 * jnp.pi,
    epsilon: ArrayLike = 1.0e-14,
) -> InterfacialFields:
    """Construct the CSF force ``sigma * curvature * delta_s * normal``."""

    values = _validate_scalar_field(colour, velocity_set)
    sigma = jnp.asarray(surface_tension, dtype=values.dtype)
    if sigma.shape != ():
        raise ValueError("surface_tension must be scalar.")
    sigma = eqx.error_if(
        sigma,
        ~jnp.isfinite(sigma) | (sigma < 0.0),
        "surface_tension must be finite and nonnegative.",
    )
    gradient, magnitude, normal = normalized_gradient(
        values, velocity_set, cell_size, epsilon=epsilon
    )
    if (wall_normal is None) != (wetting_mask is None):
        raise ValueError("wall_normal and wetting_mask must be supplied together.")
    if wall_normal is not None and wetting_mask is not None:
        normal = static_contact_angle_normal(
            normal, wall_normal, contact_angle, wetting_mask, epsilon=epsilon
        )
    curvature = -isotropic_divergence(normal, velocity_set, cell_size)
    surface_delta = 0.5 * magnitude
    force = sigma * curvature[..., None] * surface_delta[..., None] * normal
    return InterfacialFields(
        gradient,
        magnitude,
        normal,
        curvature,
        surface_delta,
        force,
    )


def natural_wetting_gradient(
    phase: ArrayLike,
    gradient: ArrayLike,
    wall_normal: ArrayLike,
    wetting_strength: ArrayLike,
    gradient_coefficient: ArrayLike,
    wetting_mask: ArrayLike,
    /,
    *,
    epsilon: ArrayLike = 1.0e-14,
) -> Array:
    """Impose the natural cubic-wall-energy boundary condition.

    The surface energy is ``h * (phi**3 / 3 - phi)`` and therefore
    ``kappa * d_n(phi) + h * (phi**2 - 1) = 0``.
    """

    phi = jnp.asarray(phase)
    grad = jnp.asarray(gradient, dtype=phi.dtype)
    wall = jnp.asarray(wall_normal, dtype=phi.dtype)
    mask = jnp.asarray(wetting_mask, dtype=bool)
    if grad.shape != (*phi.shape, wall.shape[-1]) or wall.shape != grad.shape:
        raise ValueError("Phase, gradient, and wall-normal shapes are incompatible.")
    if mask.shape != phi.shape:
        raise ValueError("wetting_mask must match the phase field.")
    h = jnp.asarray(wetting_strength, dtype=phi.dtype)
    kappa = jnp.asarray(gradient_coefficient, dtype=phi.dtype)
    threshold = jnp.asarray(epsilon, dtype=phi.dtype)
    if h.shape != () or kappa.shape != () or threshold.shape != ():
        raise ValueError("Wetting coefficients and epsilon must be scalar.")
    kappa = eqx.error_if(
        kappa,
        ~jnp.isfinite(kappa) | (kappa <= 0.0),
        "gradient_coefficient must be finite and positive.",
    )
    h = eqx.error_if(h, ~jnp.isfinite(h), "wetting_strength must be finite.")
    wall_unit, wall_magnitude = _normalise_vectors(wall, threshold)
    wall_unit = eqx.error_if(
        wall_unit,
        jnp.any(mask & (~jnp.isfinite(wall_magnitude) | (wall_magnitude <= threshold))),
        "Every wetting cell requires a finite nonzero wall normal.",
    )
    current = oe.contract("...d,...d->...", grad, wall_unit)
    prescribed = -h * (phi**2 - 1.0) / kappa
    adjusted = grad + (prescribed - current)[..., None] * wall_unit
    return jnp.where(mask[..., None], adjusted, grad)


def korteweg_stress(
    phase: ArrayLike,
    chemical_potential: ArrayLike,
    gradient: ArrayLike,
    bulk_coefficient: ArrayLike,
    gradient_coefficient: ArrayLike,
    /,
) -> Array:
    """Return the explicit symmetric Korteweg stress tensor."""

    phi = jnp.asarray(phase)
    chemical = jnp.asarray(chemical_potential, dtype=phi.dtype)
    grad = jnp.asarray(gradient, dtype=phi.dtype)
    if chemical.shape != phi.shape or grad.shape[:-1] != phi.shape:
        raise ValueError("Korteweg fields have incompatible shapes.")
    kappa = jnp.asarray(gradient_coefficient, dtype=phi.dtype)
    if kappa.shape != ():
        raise ValueError("gradient_coefficient must be scalar.")
    kappa = eqx.error_if(
        kappa,
        ~jnp.isfinite(kappa) | (kappa <= 0.0),
        "gradient_coefficient must be finite and positive.",
    )
    bulk = double_well_free_energy_density(phi, bulk_coefficient)
    gradient_squared = oe.contract("...d,...d->...", grad, grad)
    isotropic_pressure = phi * chemical - bulk - 0.5 * kappa * gradient_squared
    identity = jnp.eye(grad.shape[-1], dtype=phi.dtype)
    return isotropic_pressure[..., None, None] * identity + kappa * oe.contract(
        "...a,...b->...ab", grad, grad
    )


__all__ = [
    "InterfacialFields",
    "continuum_surface_force",
    "isotropic_divergence",
    "isotropic_gradient",
    "isotropic_laplacian",
    "korteweg_stress",
    "natural_wetting_gradient",
    "normalized_gradient",
    "static_contact_angle_normal",
]

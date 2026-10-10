#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Axisymmetric and principal-stretch finite-strain kinematics with Hencky stress.

All tensors use the cylindrical (or spherical) physical-component ordering
``(r, theta, z)``; reference coordinates are ``(R, Theta, Z)``. The hoop stretch
``1 + u_r / R`` is singular on the symmetry axis ``R = 0``; callers evaluate at
off-axis quadrature points.
"""

from __future__ import annotations

from typing import final

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    Eigenproblem,
    LogarithmSpectralFunction,
    self_adjoint_spectral_operator,
    SelfAdjointSpectralOperatorPolicy,
    SelfAdjointSpectralOperatorStatus,
)
from ._finite_strain import finite_strain_kinematics


# ``b = F F^T`` is symmetric by construction (each entry pair sums identical
# products in the same order); positivity is certified by the logarithm's
# domain check, which reports ``DOMAIN_ERROR`` for singular or inverted ``F``.
_LEFT_CAUCHY_GREEN_PROPERTIES = OperatorProperties(
    self_adjoint=True, evidence={"self_adjoint": "construction"}
)
# Loewner divided-difference Frechet derivatives stay finite at repeated
# principal stretches, including the undeformed reference state ``b = I``.
_LOGARITHM_POLICY = SelfAdjointSpectralOperatorPolicy(differentiation="frechet")
_LOGARITHM = LogarithmSpectralFunction()


def axisymmetric_deformation_gradient(
    displacement_gradient: ArrayLike,
    radial_displacement: ArrayLike,
    reference_radius: ArrayLike,
    /,
) -> Array:
    """Assemble the torsionless axisymmetric deformation gradient.

    ``displacement_gradient`` has shape ``(..., 2, 2)`` with rows ``(u_r, u_z)``
    and columns ``(d/dR, d/dZ)``. The result has shape ``(..., 3, 3)`` in
    ``(r, theta, z)`` ordering::

        [[1 + du_r/dR, 0,           du_r/dZ    ],
         [0,           1 + u_r / R, 0          ],
         [du_z/dR,     0,           1 + du_z/dZ]]

    Leading axes of all inputs broadcast against each other.
    """
    gradient = jnp.asarray(displacement_gradient)
    if gradient.ndim < 2 or gradient.shape[-2:] != (2, 2):
        raise ValueError(
            "Axisymmetric displacement gradients must end in 2x2 with rows (u_r, u_z) "
            "and columns (d/dR, d/dZ)."
        )
    radial = jnp.asarray(radial_displacement)
    radius = jnp.asarray(reference_radius)
    leading = jnp.broadcast_shapes(gradient.shape[:-2], radial.shape, radius.shape)
    gradient = jnp.broadcast_to(gradient, leading + (2, 2))
    hoop = 1.0 + jnp.broadcast_to(radial, leading) / jnp.broadcast_to(radius, leading)
    zero = jnp.zeros_like(hoop)
    rows = (
        (1.0 + gradient[..., 0, 0], zero, gradient[..., 0, 1]),
        (zero, hoop, zero),
        (gradient[..., 1, 0], zero, 1.0 + gradient[..., 1, 1]),
    )
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def cylindrical_principal_stretches(
    radial_gradient: ArrayLike,
    radial_displacement: ArrayLike,
    reference_radius: ArrayLike,
    axial_strain: ArrayLike,
    /,
) -> Array:
    """Return generalized-plane-strain principal stretches ``(r, theta, z)``.

    The stretches are ``(1 + du/dR, 1 + u/R, 1 + eps_zz)`` where ``u`` is the
    radial displacement and ``eps_zz`` is the engineering (nominal) axial
    strain ``lambda_z - 1`` shared by the generalized-plane-strain slice. The
    result has shape ``(..., 3)`` over the broadcast leading shape.
    """
    gradient, displacement, radius, axial = jnp.broadcast_arrays(
        jnp.asarray(radial_gradient),
        jnp.asarray(radial_displacement),
        jnp.asarray(reference_radius),
        jnp.asarray(axial_strain),
    )
    return jnp.stack((1.0 + gradient, 1.0 + displacement / radius, 1.0 + axial), axis=-1)


def spherical_principal_stretches(
    radial_gradient: ArrayLike,
    radial_displacement: ArrayLike,
    reference_radius: ArrayLike,
    /,
) -> Array:
    """Return spherically symmetric principal stretches ``(r, theta, phi)``.

    The stretches are ``(1 + du/dR, 1 + u/R, 1 + u/R)``; the two tangential
    stretches are equal by symmetry. The result has shape ``(..., 3)``.
    """
    gradient, displacement, radius = jnp.broadcast_arrays(
        jnp.asarray(radial_gradient),
        jnp.asarray(radial_displacement),
        jnp.asarray(reference_radius),
    )
    tangential = 1.0 + displacement / radius
    return jnp.stack((1.0 + gradient, tangential, tangential), axis=-1)


@final
class CoaxialHenckyStress(StrictModule):
    """Principal isotropic Hencky response of a coaxial stretch state.

    All principal arrays have shape ``(..., 3)`` in the stretch ordering;
    ``jacobian`` and ``admissible`` have the leading shape. ``admissible`` is
    true where every stretch is finite and strictly positive and every returned
    value is finite; outside it the values carry the natural non-finite
    logarithm results and must not be consumed.
    """

    kirchhoff: Array
    first_piola: Array
    elastic_log_strain: Array
    jacobian: Array
    admissible: Array


def coaxial_hencky_stress(
    stretches: ArrayLike,
    eigen_log_strain: ArrayLike,
    bulk_modulus: ArrayLike,
    shear_modulus: ArrayLike,
    /,
) -> CoaxialHenckyStress:
    """Evaluate the quadratic isotropic Hencky law on principal stretches.

    With ``F = F_e F_*`` and the eigen (thermal, swelling, densification,
    creep, ...) deformation ``F_* = exp(diag(eigen_log_strain))`` coaxial with
    the total stretch, the elastic logarithmic strain is exactly
    ``e = ln(stretch) - eigen_log_strain``. The Kirchhoff stress is
    ``tau = K tr(e) 1 + 2 G dev(e)``, the principal first Piola stress is
    ``P_i = tau_i / stretch_i``, and ``J = prod(stretch)``.

    Coaxiality holds for one-dimensional radial (cylindrical generalized plane
    strain) and spherical states and for any principal-aligned state. It does
    not hold for general axisymmetric RZ states with ``du_r/dZ`` or ``du_z/dR``
    shear; use :func:`axisymmetric_hencky_stress` there.

    ``stretches`` and ``eigen_log_strain`` end in 3; ``bulk_modulus`` and
    ``shear_modulus`` are leading-shape arrays. All leading axes broadcast.
    """
    stretch = jnp.asarray(stretches)
    eigen = jnp.asarray(eigen_log_strain)
    if stretch.ndim < 1 or stretch.shape[-1] != 3:
        raise ValueError("Principal stretches must end in 3.")
    if eigen.ndim < 1 or eigen.shape[-1] != 3:
        raise ValueError("Principal eigen log strains must end in 3.")
    bulk = jnp.asarray(bulk_modulus)
    shear = jnp.asarray(shear_modulus)
    leading = jnp.broadcast_shapes(
        stretch.shape[:-1], eigen.shape[:-1], bulk.shape, shear.shape
    )
    stretch = jnp.broadcast_to(stretch, leading + (3,))
    eigen = jnp.broadcast_to(eigen, leading + (3,))
    bulk = jnp.broadcast_to(bulk, leading)
    shear = jnp.broadcast_to(shear, leading)

    elastic = jnp.log(stretch) - eigen
    trace = jnp.sum(elastic, axis=-1, keepdims=True)
    kirchhoff = bulk[..., None] * trace + 2.0 * shear[..., None] * (elastic - trace / 3.0)
    first_piola = kirchhoff / stretch
    jacobian = jnp.prod(stretch, axis=-1)
    admissible = (
        jnp.all(jnp.isfinite(stretch) & (stretch > 0.0), axis=-1)
        & jnp.all(jnp.isfinite(kirchhoff), axis=-1)
        & jnp.all(jnp.isfinite(first_piola), axis=-1)
    )
    return CoaxialHenckyStress(
        kirchhoff=kirchhoff,
        first_piola=first_piola,
        elastic_log_strain=elastic,
        jacobian=jacobian,
        admissible=admissible,
    )


@final
class AxisymmetricHenckyStress(StrictModule):
    """Isotropic Hencky response of a general axisymmetric deformation.

    Tensor fields have shape ``(..., 3, 3)`` in ``(r, theta, z)`` ordering;
    ``jacobian``, ``logarithm_status`` and ``admissible`` have the leading
    shape. ``logarithm_status`` is the native
    :class:`~phydrax.linalg.eigen.SelfAdjointSpectralOperatorStatus` of
    ``log(F F^T)``. ``admissible`` combines the orientation-preserving
    finite-strain kinematic evidence with a successful logarithm; tensor
    fields are NaN outside it.
    """

    kirchhoff: Array
    first_piola: Array
    elastic_log_strain: Array
    jacobian: Array
    logarithm_status: Array
    admissible: Array


def axisymmetric_hencky_stress(
    deformation_gradient: ArrayLike,
    eigen_log_strain: ArrayLike,
    bulk_modulus: ArrayLike,
    shear_modulus: ArrayLike,
    /,
) -> AxisymmetricHenckyStress:
    """Evaluate the quadratic isotropic Hencky law for a general 3x3 ``F``.

    The spatial logarithmic strain ``ln V = log(b) / 2`` with ``b = F F^T`` is
    the spectral logarithm of the native self-adjoint eigendecomposition with
    Loewner divided-difference Frechet derivatives, so tangents remain finite
    at repeated principal stretches. The elastic strain is
    ``e = ln V - eigen_log_strain``; the Kirchhoff stress is
    ``tau = K tr(e) 1 + 2 G dev(e)`` and ``P = tau F^{-T}`` through the native
    finite-strain inverse.

    The additive split ``ln V_e = ln V - eigen`` is exact when the symmetric
    ``eigen_log_strain`` commutes with ``V`` (always for isotropic thermal or
    swelling eigenstrains, and for principal-aligned states). For anisotropic
    eigenstrains not aligned with ``V`` it is the small-rotation approximation
    of the multiplicative split, accurate to first order in the misalignment.

    ``deformation_gradient`` and ``eigen_log_strain`` end in 3x3 (typically
    from :func:`axisymmetric_deformation_gradient`); moduli are leading-shape
    arrays. All leading axes broadcast.
    """
    deformation = jnp.asarray(deformation_gradient)
    eigen = jnp.asarray(eigen_log_strain)
    if deformation.ndim < 2 or deformation.shape[-2:] != (3, 3):
        raise ValueError("Axisymmetric deformation gradients must end in 3x3.")
    if eigen.ndim < 2 or eigen.shape[-2:] != (3, 3):
        raise ValueError("Eigen log strains must end in 3x3.")
    bulk = jnp.asarray(bulk_modulus)
    shear = jnp.asarray(shear_modulus)
    leading = jnp.broadcast_shapes(
        deformation.shape[:-2], eigen.shape[:-2], bulk.shape, shear.shape
    )
    deformation = jnp.broadcast_to(deformation, leading + (3, 3))
    eigen = jnp.broadcast_to(eigen, leading + (3, 3))
    bulk = jnp.broadcast_to(bulk, leading)
    shear = jnp.broadcast_to(shear, leading)

    kinematics = finite_strain_kinematics(deformation)
    logarithm = self_adjoint_spectral_operator(
        Eigenproblem(
            DenseLinearOperator(
                kinematics.left_cauchy_green,
                properties=_LEFT_CAUCHY_GREEN_PROPERTIES,
            )
        ),
        _LOGARITHM,
        policy=_LOGARITHM_POLICY,
    )
    elastic = 0.5 * logarithm.operator - eigen
    identity = jnp.broadcast_to(jnp.eye(3, dtype=elastic.dtype), elastic.shape)
    trace = jnp.trace(elastic, axis1=-2, axis2=-1)[..., None, None]
    kirchhoff = bulk[..., None, None] * trace * identity + 2.0 * shear[
        ..., None, None
    ] * (elastic - trace / 3.0 * identity)
    first_piola = ein.contract(
        "...ik,...jk->...ij", kirchhoff, kinematics.inverse_deformation_gradient
    )
    admissible = kinematics.admissible & (
        logarithm.status == int(SelfAdjointSpectralOperatorStatus.SUCCESS)
    )
    mask = admissible[..., None, None]
    return AxisymmetricHenckyStress(
        kirchhoff=jnp.where(mask, kirchhoff, jnp.nan),
        first_piola=jnp.where(mask, first_piola, jnp.nan),
        elastic_log_strain=jnp.where(mask, elastic, jnp.nan),
        jacobian=kinematics.jacobian,
        logarithm_status=logarithm.status,
        admissible=admissible,
    )


__all__ = [
    "AxisymmetricHenckyStress",
    "CoaxialHenckyStress",
    "axisymmetric_deformation_gradient",
    "axisymmetric_hencky_stress",
    "coaxial_hencky_stress",
    "cylindrical_principal_stretches",
    "spherical_principal_stretches",
]

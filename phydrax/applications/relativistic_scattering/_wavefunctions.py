#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dirac wavefunctions, vector polarizations, and free propagators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ._kinematics import minkowski_dot, MINKOWSKI_METRIC


_COMPLEX = jnp.complex128
_IDENTITY_TWO = jnp.eye(2, dtype=_COMPLEX)
_ZERO_TWO = jnp.zeros((2, 2), dtype=_COMPLEX)
_PAULI = jnp.asarray(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=_COMPLEX,
)
GAMMA_MATRICES = jnp.stack(
    (
        jnp.block([[_IDENTITY_TWO, _ZERO_TWO], [_ZERO_TWO, -_IDENTITY_TWO]]),
        *tuple(jnp.block([[_ZERO_TWO, sigma], [-sigma, _ZERO_TWO]]) for sigma in _PAULI),
    )
)
GAMMA_FIVE = (
    1.0j * GAMMA_MATRICES[0] @ GAMMA_MATRICES[1] @ GAMMA_MATRICES[2] @ GAMMA_MATRICES[3]
)


def slash(momentum: ArrayLike, /) -> Array:
    """Return ``gamma^mu p_mu`` in the Dirac representation."""
    value = jnp.asarray(momentum)
    if value.shape[-1:] != (4,):
        raise ValueError("Dirac slash requires a trailing four-vector axis.")
    lowered = value * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=value.dtype)
    return ein.contract("...m,mab->...ab", lowered, GAMMA_MATRICES)


def dirac_adjoint(spinor: ArrayLike, /) -> Array:
    """Return ``psi dagger gamma^0``."""
    value = jnp.asarray(spinor)
    if value.shape[-1:] != (4,):
        raise ValueError("Dirac adjoints require a trailing spinor axis of length four.")
    return jnp.conj(value) @ GAMMA_MATRICES[0]


def _spin_basis(spin: int, dtype: jnp.dtype, /) -> Array:
    if spin not in (-1, 1):
        raise ValueError("Dirac spin labels are -1 and +1.")
    return jnp.asarray([1.0, 0.0] if spin == 1 else [0.0, 1.0], dtype=dtype)


def dirac_u(momentum: ArrayLike, mass: ArrayLike, spin: int, /) -> Array:
    """Positive-energy spin-z spinor normalized by ``ubar u = 2m``."""
    p = jnp.asarray(momentum)
    mass_ = jnp.asarray(mass)
    if p.shape != (4,):
        raise ValueError("dirac_u requires one four-momentum.")
    chi = _spin_basis(spin, jnp.result_type(p, _COMPLEX))
    denominator = p[0] + mass_
    sigma_p = ein.contract("i,iab->ab", p[1:], _PAULI)
    normalization = jnp.sqrt(denominator)
    return normalization * jnp.concatenate((chi, sigma_p @ chi / denominator))


def dirac_v(momentum: ArrayLike, mass: ArrayLike, spin: int, /) -> Array:
    """Negative-frequency antiparticle spinor normalized by ``vbar v = -2m``."""
    p = jnp.asarray(momentum)
    mass_ = jnp.asarray(mass)
    if p.shape != (4,):
        raise ValueError("dirac_v requires one four-momentum.")
    eta = _spin_basis(spin, jnp.result_type(p, _COMPLEX))
    denominator = p[0] + mass_
    sigma_p = ein.contract("i,iab->ab", p[1:], _PAULI)
    normalization = jnp.sqrt(denominator)
    return normalization * jnp.concatenate((sigma_p @ eta / denominator, eta))


def spinor_outer(spinor: ArrayLike, /) -> Array:
    """Return ``psi psibar``."""
    value = jnp.asarray(spinor)
    return value[:, None] * dirac_adjoint(value)[None, :]


def spinor_completeness(
    momentum: ArrayLike, mass: ArrayLike, /, *, antiparticle: bool = False
) -> Array:
    """Sum the two physical spin projectors."""
    constructor = dirac_v if antiparticle else dirac_u
    return sum(
        (spinor_outer(constructor(momentum, mass, spin)) for spin in (-1, 1)),
        jnp.zeros((4, 4), dtype=_COMPLEX),
    )


def dirac_bilinear(left: ArrayLike, matrix: ArrayLike, right: ArrayLike, /) -> Array:
    """Evaluate ``leftbar matrix right``."""
    return dirac_adjoint(left) @ jnp.asarray(matrix) @ jnp.asarray(right)


def vector_current(left: ArrayLike, right: ArrayLike, /) -> Array:
    """Evaluate the contravariant vector current ``leftbar gamma^mu right``."""
    return ein.contract(
        "a,mab,b->m",
        dirac_adjoint(left),
        GAMMA_MATRICES,
        jnp.asarray(right),
    )


def _transverse_basis(momentum: Array, /) -> tuple[Array, Array, Array]:
    spatial = momentum[1:]
    magnitude = jnp.sqrt(jnp.sum(spatial * spatial))
    default_direction = jnp.asarray([0.0, 0.0, 1.0], dtype=spatial.dtype)
    direction = jnp.where(magnitude > 0.0, spatial / magnitude, default_direction)
    reference = jnp.where(
        jnp.abs(direction[2]) < 0.9,
        default_direction,
        jnp.asarray([0.0, 1.0, 0.0], dtype=direction.dtype),
    )
    first = jnp.cross(reference, direction)
    first = first / jnp.sqrt(jnp.sum(first * first))
    second = jnp.cross(direction, first)
    return direction, first, second


def photon_polarization(momentum: ArrayLike, helicity: int, /) -> Array:
    """Physical circular photon polarization with zero time component."""
    p = jnp.asarray(momentum)
    if p.shape != (4,) or helicity not in (-1, 1):
        raise ValueError("Photon polarization requires one momentum and helicity +/-1.")
    _, first, second = _transverse_basis(p)
    spatial = (first + 1.0j * float(helicity) * second) / jnp.sqrt(2.0)
    return jnp.concatenate((jnp.zeros((1,), dtype=spatial.dtype), spatial))


def photon_linear_polarization(momentum: ArrayLike, axis: int, /) -> Array:
    """Physical real transverse polarization for axis zero or one."""
    p = jnp.asarray(momentum)
    if p.shape != (4,) or axis not in (0, 1):
        raise ValueError("Linear polarization axis must be zero or one.")
    _, first, second = _transverse_basis(p)
    spatial = first if axis == 0 else second
    return jnp.concatenate((jnp.zeros((1,), dtype=spatial.dtype), spatial))


def massive_vector_polarization(
    momentum: ArrayLike, mass: ArrayLike, polarization: int, /
) -> Array:
    """Three physical polarizations (-1, 0, +1) of a massive vector."""
    p = jnp.asarray(momentum)
    mass_ = jnp.asarray(mass)
    if p.shape != (4,) or polarization not in (-1, 0, 1):
        raise ValueError("Massive-vector polarization label must be -1, 0, or +1.")
    direction, first, second = _transverse_basis(p)
    if polarization == 0:
        spatial_magnitude = jnp.linalg.norm(p[1:])
        return jnp.concatenate(
            ((spatial_magnitude / mass_).reshape((1,)), p[0] * direction / mass_)
        )
    spatial = (first + 1.0j * float(polarization) * second) / jnp.sqrt(2.0)
    return jnp.concatenate((jnp.zeros((1,), dtype=spatial.dtype), spatial))


def photon_polarization_sum(momentum: ArrayLike, /) -> Array:
    """Sum physical helicity projectors in the deterministic transverse gauge."""
    polarizations = jnp.stack(
        (photon_polarization(momentum, -1), photon_polarization(momentum, 1))
    )
    return ein.contract("ha,hb->ab", polarizations, jnp.conj(polarizations))


def massive_vector_polarization_sum(momentum: ArrayLike, mass: ArrayLike, /) -> Array:
    """Return ``-g^{mu nu} + p^mu p^nu/m^2`` from explicit polarizations."""
    polarizations = jnp.stack(
        tuple(massive_vector_polarization(momentum, mass, state) for state in (-1, 0, 1))
    )
    return ein.contract("ha,hb->ab", polarizations, jnp.conj(polarizations))


def scalar_propagator(
    momentum: ArrayLike, mass: ArrayLike, /, *, epsilon: float = 1.0e-12
) -> Array:
    """Feynman scalar propagator."""
    denominator = (
        minkowski_dot(momentum, momentum) - jnp.asarray(mass) ** 2 + 1.0j * epsilon
    )
    return 1.0j / denominator


def fermion_propagator(
    momentum: ArrayLike, mass: ArrayLike, /, *, epsilon: float = 1.0e-12
) -> Array:
    """Feynman Dirac propagator."""
    mass_ = jnp.asarray(mass)
    denominator = minkowski_dot(momentum, momentum) - mass_**2 + 1.0j * epsilon
    return 1.0j * (slash(momentum) + mass_ * jnp.eye(4)) / denominator


def photon_propagator(momentum: ArrayLike, /, *, epsilon: float = 1.0e-12) -> Array:
    """Feynman-gauge photon propagator."""
    denominator = minkowski_dot(momentum, momentum) + 1.0j * epsilon
    return -1.0j * MINKOWSKI_METRIC / denominator


__all__ = [
    "GAMMA_FIVE",
    "GAMMA_MATRICES",
    "dirac_adjoint",
    "dirac_bilinear",
    "dirac_u",
    "dirac_v",
    "fermion_propagator",
    "massive_vector_polarization",
    "massive_vector_polarization_sum",
    "photon_linear_polarization",
    "photon_polarization",
    "photon_polarization_sum",
    "photon_propagator",
    "scalar_propagator",
    "slash",
    "spinor_completeness",
    "spinor_outer",
    "vector_current",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg.eigen import SelfAdjointSpectralOperatorStatus
from phydrax.operators.mechanics import (
    axisymmetric_deformation_gradient,
    axisymmetric_hencky_stress,
    coaxial_hencky_stress,
    cylindrical_principal_stretches,
    spherical_principal_stretches,
)


pytestmark = pytest.mark.strict_jax

# Normalized moduli; the laws are linear in (K, G), so the contracts are scale-free.
_BULK = jnp.asarray(2.0)
_SHEAR = jnp.asarray(0.7)
_LAME = _BULK - 2.0 * _SHEAR / 3.0


def _linear_elastic_stress(strain: Array, /) -> Array:
    """Independent small-strain Lame reference ``lambda tr(eps) 1 + 2 mu eps``."""
    return _LAME * jnp.sum(strain, axis=-1, keepdims=True) + 2.0 * _SHEAR * strain


def _rz_rotation(angle: float, /) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return jnp.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def test_rigid_axial_translation_has_identity_deformation_gradient() -> None:
    # u_r = 0 and u_z = constant: every displacement derivative vanishes.
    radius = jnp.asarray([1e-3, 4.1e-3, 4.7e-3])
    deformation = axisymmetric_deformation_gradient(
        jnp.zeros((3, 2, 2)), jnp.zeros(3), radius
    )

    assert deformation.shape == (3, 3, 3)
    np.testing.assert_array_equal(deformation, jnp.broadcast_to(jnp.eye(3), (3, 3, 3)))


def test_uniform_cylindrical_expansion_has_exact_principal_stretches() -> None:
    # Dyadic values keep every product exactly representable.
    expansion, axial = 2.0**-6, 2.0**-7
    radius = jnp.asarray([0.375, 0.75, 1.5])
    displacement = expansion * radius
    gradient = jnp.broadcast_to(jnp.asarray([[expansion, 0.0], [0.0, axial]]), (3, 2, 2))
    expected = jnp.broadcast_to(
        jnp.asarray([1.0 + expansion, 1.0 + expansion, 1.0 + axial]), (3, 3)
    )

    stretches = cylindrical_principal_stretches(
        jnp.full(3, expansion), displacement, radius, jnp.full(3, axial)
    )
    deformation = axisymmetric_deformation_gradient(gradient, displacement, radius)
    response = coaxial_hencky_stress(stretches, jnp.zeros(3), _BULK, _SHEAR)

    np.testing.assert_array_equal(stretches, expected)
    np.testing.assert_array_equal(deformation, jax.vmap(jnp.diag)(expected))
    np.testing.assert_array_equal(
        response.jacobian, jnp.full(3, (1.0 + expansion) ** 2 * (1.0 + axial))
    )


def test_uniform_spherical_expansion_has_exact_principal_stretches() -> None:
    expansion = 2.0**-6
    radius = jnp.asarray([0.375, 0.75])
    stretches = spherical_principal_stretches(
        jnp.full(2, expansion), expansion * radius, radius
    )
    response = coaxial_hencky_stress(stretches, jnp.zeros(3), _BULK, _SHEAR)

    np.testing.assert_array_equal(stretches, jnp.full((2, 3), 1.0 + expansion))
    np.testing.assert_array_equal(response.jacobian, jnp.full(2, (1.0 + expansion) ** 3))
    # Uniform dilatation is purely volumetric: tau = K ln(J) 1.
    np.testing.assert_allclose(
        response.kirchhoff,
        jnp.full((2, 3), _BULK * 3.0 * np.log1p(expansion)),
        rtol=1e-14,
    )


def test_coaxial_hencky_recovers_lame_elasticity_to_second_order() -> None:
    rng = np.random.default_rng(20260929)
    total_direction = jnp.asarray(rng.uniform(-1.0, 1.0, size=(8, 3)))
    eigen_direction = jnp.asarray(rng.uniform(-0.5, 0.5, size=(8, 3)))

    def error(scale: float, /) -> Array:
        strain = scale * total_direction
        eigen = scale * eigen_direction
        response = coaxial_hencky_stress(1.0 + strain, eigen, _BULK, _SHEAR)
        reference = _linear_elastic_stress(strain - eigen)
        return jnp.maximum(
            jnp.max(jnp.abs(response.kirchhoff - reference)),
            jnp.max(jnp.abs(response.first_piola - reference)),
        )

    coarse, fine = error(1e-3), error(5e-4)

    assert coarse < 10.0 * (_BULK + _SHEAR) * 1e-6
    # Halving the strain quarters the discrepancy: the defect is O(eps^2), not O(eps).
    assert 3.5 < coarse / fine < 4.5


def test_eigen_deformation_is_stress_free() -> None:
    # F = exp(diag(eigen)) is the free thermal/swelling state of the multiplicative split.
    eigen = jnp.asarray([[0.012, -0.004, 0.03], [0.02, 0.02, 0.02]])
    stretches = jnp.exp(eigen)

    coaxial = coaxial_hencky_stress(stretches, eigen, _BULK, _SHEAR)
    general = axisymmetric_hencky_stress(
        jax.vmap(jnp.diag)(stretches), jax.vmap(jnp.diag)(eigen), _BULK, _SHEAR
    )

    np.testing.assert_allclose(coaxial.kirchhoff, 0.0, atol=1e-15)
    np.testing.assert_allclose(coaxial.first_piola, 0.0, atol=1e-15)
    np.testing.assert_allclose(general.kirchhoff, 0.0, atol=1e-14)
    np.testing.assert_allclose(general.first_piola, 0.0, atol=1e-14)


def test_coaxial_hencky_stretch_tangent_matches_analytic_derivative() -> None:
    stretches = jnp.asarray([1.04, 0.97, 1.01])
    eigen = jnp.asarray([0.003, 0.003, 0.001])
    tangent = jax.jacfwd(
        lambda value: coaxial_hencky_stress(value, eigen, _BULK, _SHEAR).kirchhoff
    )(stretches)
    # d tau_i / d lambda_j = (lambda_Lame + 2 G delta_ij) / lambda_j.
    expected = (_LAME + 2.0 * _SHEAR * jnp.eye(3)) / stretches[None, :]

    np.testing.assert_allclose(tangent, expected, rtol=1e-13)


def test_coaxial_hencky_jvp_and_vjp_are_dual() -> None:
    rng = np.random.default_rng(7)
    stretches = jnp.asarray(rng.uniform(0.9, 1.1, size=(4, 3)))
    eigen = jnp.asarray(rng.uniform(-0.01, 0.01, size=(4, 3)))
    direction = jnp.asarray(rng.normal(size=(4, 3)))
    cotangent = (
        jnp.asarray(rng.normal(size=(4, 3))),
        jnp.asarray(rng.normal(size=(4, 3))),
        jnp.asarray(rng.normal(size=(4,))),
    )

    def stress(value: Array, /) -> tuple[Array, Array, Array]:
        response = coaxial_hencky_stress(value, eigen, _BULK, _SHEAR)
        return response.kirchhoff, response.first_piola, response.jacobian

    _, forward = jax.jvp(stress, (stretches,), (direction,))
    _, pullback = jax.vjp(stress, stretches)
    (reverse,) = pullback(cotangent)
    forward_pairing = sum(
        jnp.vdot(left, right) for left, right in zip(cotangent, forward, strict=True)
    )

    np.testing.assert_allclose(forward_pairing, jnp.vdot(reverse, direction), rtol=1e-13)


def test_coaxial_hencky_broadcasts_leading_axes() -> None:
    stretches = jnp.asarray([[[1.02, 0.99, 1.0]], [[0.97, 1.03, 1.01]]])
    bulk = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    response = coaxial_hencky_stress(stretches, jnp.zeros(3), bulk, _SHEAR)
    single = coaxial_hencky_stress(stretches[1, 0], jnp.zeros(3), bulk[2], _SHEAR)

    assert response.kirchhoff.shape == (2, 4, 3)
    assert response.jacobian.shape == (2, 4)
    np.testing.assert_array_equal(response.kirchhoff[1, 2], single.kirchhoff)


def test_axisymmetric_hencky_matches_coaxial_for_diagonal_deformation() -> None:
    stretches = jnp.asarray([[1.05, 0.98, 1.02], [0.96, 0.96, 1.01]])
    eigen = jnp.asarray([[0.004, 0.004, 0.002], [0.01, -0.002, 0.0]])
    coaxial = coaxial_hencky_stress(stretches, eigen, _BULK, _SHEAR)
    general = axisymmetric_hencky_stress(
        jax.vmap(jnp.diag)(stretches), jax.vmap(jnp.diag)(eigen), _BULK, _SHEAR
    )

    assert bool(jnp.all(general.admissible))
    np.testing.assert_allclose(
        general.kirchhoff, jax.vmap(jnp.diag)(coaxial.kirchhoff), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        general.first_piola,
        jax.vmap(jnp.diag)(coaxial.first_piola),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(general.jacobian, coaxial.jacobian, rtol=1e-14)


def test_axisymmetric_hencky_is_objective_under_rz_rotation() -> None:
    # F = Q diag(lambda) with Q a rotation in the r-z plane and an isotropic eigen
    # strain: tau = Q tau_coaxial Q^T and P = Q diag(P_coaxial) exactly.
    rotation = _rz_rotation(0.3)
    stretches = jnp.asarray([1.06, 0.97, 1.02])
    swelling = 0.004
    coaxial = coaxial_hencky_stress(stretches, jnp.full(3, swelling), _BULK, _SHEAR)
    general = axisymmetric_hencky_stress(
        rotation @ jnp.diag(stretches), swelling * jnp.eye(3), _BULK, _SHEAR
    )

    np.testing.assert_allclose(
        general.kirchhoff,
        rotation @ jnp.diag(coaxial.kirchhoff) @ rotation.T,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        general.first_piola,
        rotation @ jnp.diag(coaxial.first_piola),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "differentiate",
    [jax.jacfwd, jax.jacrev],
    ids=["forward", "reverse"],
)
def test_axisymmetric_hencky_reference_tangent_is_isotropic_lame(
    differentiate: Callable[[Callable[[Array], Array]], Callable[[Array], Array]],
) -> None:
    # b = I has a triply repeated eigenvalue; the Frechet tangent must stay finite
    # and equal C_ijkl = lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk).
    tangent = differentiate(
        lambda deformation: (
            axisymmetric_hencky_stress(
                deformation, jnp.zeros((3, 3)), _BULK, _SHEAR
            ).first_piola
        )
    )(jnp.eye(3))
    delta = np.eye(3)
    volumetric = delta[:, :, None, None] * delta[None, None, :, :]
    symmetric = (
        delta[:, None, :, None] * delta[None, :, None, :]
        + delta[:, None, None, :] * delta[None, :, :, None]
    )
    expected = float(_LAME) * volumetric + float(_SHEAR) * symmetric

    np.testing.assert_allclose(tangent, expected, rtol=1e-10, atol=1e-12)


def test_inadmissible_deformation_reports_evidence() -> None:
    stretches = jnp.asarray([[1.0, 1.0, 1.0], [-0.5, 1.0, 1.0]])
    coaxial = coaxial_hencky_stress(stretches, jnp.zeros(3), _BULK, _SHEAR)
    deformations = jnp.stack(
        [
            jnp.eye(3),
            jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])),
            jnp.diag(jnp.asarray([0.0, 1.0, 1.0])),
        ]
    )
    general = axisymmetric_hencky_stress(deformations, jnp.zeros((3, 3)), _BULK, _SHEAR)

    np.testing.assert_array_equal(coaxial.admissible, [True, False])
    np.testing.assert_array_equal(general.admissible, [True, False, False])
    assert int(general.logarithm_status[2]) == int(
        SelfAdjointSpectralOperatorStatus.DOMAIN_ERROR
    )
    assert bool(jnp.all(jnp.isnan(general.first_piola[1:])))
    assert bool(jnp.all(jnp.isfinite(general.first_piola[0])))


@pytest.mark.parametrize(
    "evaluate",
    [
        lambda: axisymmetric_deformation_gradient(jnp.zeros((3, 3)), 0.0, 1.0),
        lambda: coaxial_hencky_stress(jnp.ones(2), jnp.zeros(3), _BULK, _SHEAR),
        lambda: coaxial_hencky_stress(jnp.ones(3), jnp.zeros(2), _BULK, _SHEAR),
        lambda: axisymmetric_hencky_stress(jnp.eye(2), jnp.zeros((3, 3)), _BULK, _SHEAR),
        lambda: axisymmetric_hencky_stress(jnp.eye(3), jnp.zeros(3), _BULK, _SHEAR),
    ],
    ids=[
        "displacement-gradient-not-2x2",
        "stretches-not-3",
        "principal-eigen-not-3",
        "deformation-not-3x3",
        "tensor-eigen-not-3x3",
    ],
)
def test_malformed_trailing_axes_are_refused(evaluate: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        evaluate()

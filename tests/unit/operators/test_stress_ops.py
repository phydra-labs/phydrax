#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.operators.differential import (
    deformation_gradient,
    deviatoric_stress,
    hydrostatic_pressure,
    hydrostatic_stress,
    linear_elastic_cauchy_stress_2d,
    linear_elastic_orthotropic_stress_2d,
    maxwell_stress,
    neo_hookean_cauchy,
    neo_hookean_pk1,
    neo_hookean_reference_energy,
    svk_pk2_stress,
    viscous_stress,
)


def test_deviatoric_and_hydrostatic():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    # sigma = p I
    p = 3.0

    @geom.Function("x")
    def sigma_const(x):
        return jnp.array([[p, 0.0], [0.0, p]])

    dev = deviatoric_stress(sigma_const)
    s = jnp.asarray(
        dev(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(s, jnp.zeros((2, 2)))

    hp = hydrostatic_pressure(sigma_const)
    pval = jnp.asarray(
        hp(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.isclose(pval, -p * 1.0)

    hs = hydrostatic_stress(sigma_const)
    sig = jnp.asarray(
        hs(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(
        sig,
        jnp.asarray(
            sigma_const(
                frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})
            ).data
        ),
    )


def test_viscous_stress_symmetry():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    mu = 2.0
    a, b = 0.1, -0.2

    @geom.Function("x")
    def u(x):
        return jnp.array([a * x[0], b * x[1]])

    tau = viscous_stress(u, mu=mu)
    t = jnp.asarray(
        tau(frozendict({"x": cx.Field(jnp.array([0.5, -0.3]), dims=(None,))})).data
    )
    assert jnp.allclose(t, jnp.swapaxes(jnp.asarray(t), -1, -2))


def test_maxwell_stress_E_only():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    eps = 2.0

    @geom.Function("x")
    def E(x):
        return jnp.array([1.0, 0.0])

    T = maxwell_stress(E=E, epsilon=eps)
    T0 = jnp.asarray(
        T(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    # Expected: T = eps [[0.5, 0],[0,-0.5]]
    assert jnp.allclose(T0, jnp.array([[0.5 * eps, 0.0], [0.0, -0.5 * eps]]))


def test_linear_isotropic_plane_stress_simple():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    E, nu = 10.0, 0.25
    a, b = 0.1, -0.2

    @geom.Function("x")
    def u(x):
        return jnp.array([a * x[0], b * x[1]])

    sigma2d = linear_elastic_cauchy_stress_2d(u, E=E, nu=nu, mode2d="plane_stress")
    s = jnp.asarray(
        sigma2d(frozendict({"x": cx.Field(jnp.array([0.1, -0.2]), dims=(None,))})).data
    )
    assert s.shape == (2, 2)


def test_orthotropic_reduces_isotropic():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    # Choose E1=E2=E, nu12=nu, G12=E/(2(1+nu)) -> isotropic equivalence in plane stress
    E, nu = 10.0, 0.3
    G = E / (2 * (1 + nu))
    a, b = 0.1, -0.2

    @geom.Function("x")
    def u(x):
        return jnp.array([a * x[0], b * x[1]])

    sig_iso = linear_elastic_cauchy_stress_2d(u, E=E, nu=nu, mode2d="plane_stress")
    sig_ortho = linear_elastic_orthotropic_stress_2d(
        u, E1=E, E2=E, nu12=nu, G12=G, mode2d="plane_stress"
    )
    pts = frozendict({"x": cx.Field(jnp.array([0.2, -0.4]), dims=(None,))})
    assert jnp.allclose(
        jnp.asarray(sig_iso(pts).data),
        jnp.asarray(sig_ortho(pts).data),
        atol=1e-6,
    )


def test_finite_strain_shapes_zero_disp():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    # At zero displacement, F=I, E=0 => SVK S=0; Neo-Hookean Cauchy σ=0
    @geom.Function("x")
    def uz(x):
        return jnp.array([0.0, 0.0])

    mu, lam = 2.0, 3.0
    S = svk_pk2_stress(uz, lambda_=lam, mu=mu)
    s = jnp.asarray(
        S(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(s, jnp.zeros((2, 2)))

    nh = neo_hookean_cauchy(uz, mu=mu, lambda_=lam)
    sig = jnp.asarray(
        nh(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(sig, jnp.zeros((2, 2)))


def test_neo_hookean_field_energy_and_stresses_match_plane_strain_array_model():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    displacement_gradient = jnp.asarray([[0.08, 0.07], [0.02, -0.06]])
    deformation = jnp.eye(2) + displacement_gradient
    mu = 3.0
    lambda_ = 9.0

    @geom.Function("x")
    def u(x):
        return displacement_gradient @ x

    point = frozendict({"x": cx.Field(jnp.asarray((0.2, -0.4)), dims=(None,))})
    energy = jnp.asarray(
        neo_hookean_reference_energy(u, mu=mu, lambda_=lambda_)(point).data
    )
    first_piola = jnp.asarray(neo_hookean_pk1(u, mu=mu, lambda_=lambda_)(point).data)
    cauchy = jnp.asarray(neo_hookean_cauchy(u, mu=mu, lambda_=lambda_)(point).data)

    parameters = phx.applications.solid_mechanics.NeoHookeanParameters(mu, lambda_)
    embedded = jnp.eye(3).at[:2, :2].set(deformation)
    expected_energy = phx.applications.solid_mechanics.neo_hookean_reference_energy(
        embedded, parameters
    )
    expected_first_piola = phx.applications.solid_mechanics.neo_hookean_first_piola(
        embedded, parameters
    )[:2, :2]
    expected_cauchy = expected_first_piola @ deformation.T / jnp.linalg.det(deformation)

    np.testing.assert_allclose(energy, expected_energy, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(first_piola, expected_first_piola, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(cauchy, expected_cauchy, rtol=2e-11, atol=2e-11)


def test_neo_hookean_field_supports_heterogeneous_scalar_materials():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.asarray((0.05 * x[0], -0.02 * x[1]))

    @geom.Function("x")
    def mu(x):
        return 2.0 + 0.5 * x[0]

    point_value = jnp.asarray((0.4, -0.3))
    point = frozendict({"x": cx.Field(point_value, dims=(None,))})
    actual = jnp.asarray(neo_hookean_reference_energy(u, mu=mu, lambda_=4.0)(point).data)
    deformation = jnp.diag(jnp.asarray((1.05, 0.98, 1.0)))
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters(
        2.0 + 0.5 * point_value[0], 4.0
    )
    expected = phx.applications.solid_mechanics.neo_hookean_reference_energy(
        deformation, parameters
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "mu, lambda_",
    [
        (-1.0, 3.0),
        (2.0, -2.0),
        (jnp.nan, 3.0),
    ],
)
def test_neo_hookean_field_marks_invalid_materials_nonfinite(mu, lambda_):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.asarray((0.0, 0.0))

    point = frozendict({"x": cx.Field(jnp.zeros(2), dims=(None,))})
    value = neo_hookean_reference_energy(u, mu=mu, lambda_=lambda_)(point).data
    assert not bool(jnp.isfinite(value))


def test_neo_hookean_field_marks_nonpositive_jacobian_nonfinite():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def inverted(x):
        return jnp.asarray((-2.0 * x[0], 0.0))

    point = frozendict({"x": cx.Field(jnp.asarray((0.2, 0.1)), dims=(None,))})
    energy = neo_hookean_reference_energy(inverted, mu=2.0, lambda_=3.0)(point).data
    first_piola = neo_hookean_pk1(inverted, mu=2.0, lambda_=3.0)(point).data
    cauchy = neo_hookean_cauchy(inverted, mu=2.0, lambda_=3.0)(point).data
    assert not bool(jnp.isfinite(energy))
    assert not bool(jnp.all(jnp.isfinite(first_piola)))
    assert not bool(jnp.all(jnp.isfinite(cauchy)))


@pytest.mark.parametrize("components", [1, 3])
def test_deformation_gradient_rejects_displacement_dimension_mismatch(components):
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def displacement(x):
        return jnp.zeros((components,))

    gradient = deformation_gradient(displacement)
    point = frozendict({"x": cx.Field(jnp.zeros(2), dims=(None,))})
    with pytest.raises(ValueError, match="displacement gradient"):
        gradient(point)


def test_neo_hookean_field_rejects_retired_kappa_keyword():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def u(x):
        return jnp.asarray((0.0, 0.0))

    with pytest.raises(TypeError, match="kappa"):
        neo_hookean_cauchy(u, mu=2.0, kappa=3.0)

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Square
from phydrax.operators.differential import (
    deviatoric_stress,
    hydrostatic_pressure,
    hydrostatic_stress,
    linear_elastic_cauchy_stress_2d,
    linear_elastic_orthotropic_stress_2d,
    maxwell_stress,
    neo_hookean_cauchy,
    svk_pk2_stress,
    viscous_stress,
)


def test_deviatoric_and_hydrostatic():
    geom = Square(center=(0.0, 0.0), side=2.0)

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
    geom = Square(center=(0.0, 0.0), side=2.0)
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
    geom = Square(center=(0.0, 0.0), side=2.0)
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
    geom = Square(center=(0.0, 0.0), side=2.0)
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
    geom = Square(center=(0.0, 0.0), side=2.0)
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
    geom = Square(center=(0.0, 0.0), side=2.0)

    # At zero displacement, F=I, E=0 => SVK S=0; Neo-Hookean Cauchy σ=0
    @geom.Function("x")
    def uz(x):
        return jnp.array([0.0, 0.0])

    mu, lam, kappa = 2.0, 3.0, 5.0
    S = svk_pk2_stress(uz, lambda_=lam, mu=mu)
    s = jnp.asarray(
        S(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(s, jnp.zeros((2, 2)))

    nh = neo_hookean_cauchy(uz, mu=mu, kappa=kappa)
    sig = jnp.asarray(
        nh(frozendict({"x": cx.Field(jnp.array([0.0, 0.0]), dims=(None,))})).data
    )
    assert jnp.allclose(sig, jnp.zeros((2, 2)))

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _dirichlet_modes():
    domain = phx.domain.Interval1d(0.0, 1.0)
    first = domain.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))
    second = domain.Function("x")(lambda x: jnp.sin(2.0 * jnp.pi * x[0]))
    return domain, first, second


def _energy(left, right):
    left_gradient = phx.operators.conjugate(phx.operators.grad(left, var="x"))
    right_gradient = phx.operators.grad(right, var="x")
    return phx.operators.einsum(
        "...i,...i->...",
        left_gradient,
        right_gradient,
    )


def _term(domain):
    return phx.terms.VariationalEigenspace(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(
            phx.integration.GaussLegendreRule(48)
        ),
        objective_vars=("u0", "u1"),
        stiffness_form=_energy,
        materialization_policy="fixed",
    )


def test_variational_eigenspace_recovers_dirichlet_laplacian_modes():
    domain, first, second = _dirichlet_modes()
    term = _term(domain)
    fields = {"u0": first, "u1": second}

    evaluation = term.assemble(fields)
    result = term.ritz(fields)

    assert bool(evaluation.valid)
    assert bool(result.successful)
    assert jnp.allclose(
        result.eigenvalues,
        jnp.asarray([jnp.pi**2, 4.0 * jnp.pi**2]),
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.allclose(evaluation.mass, 0.5 * jnp.eye(2), atol=2e-12)
    assert jnp.allclose(evaluation.objective, 5.0 * jnp.pi**2, rtol=2e-10)


def test_block_objective_is_basis_invariant_and_cluster_gradient_is_finite():
    stiffness = jnp.diag(jnp.asarray([2.0, 2.0]))
    mass = jnp.eye(2)
    change = jnp.asarray([[2.0, -1.0], [1.0, 3.0]])
    transformed_stiffness = change.T @ stiffness @ change
    transformed_mass = change.T @ mass @ change

    original = phx.linalg.eigen.block_rayleigh_trace(stiffness, mass)
    transformed = phx.linalg.eigen.block_rayleigh_trace(
        transformed_stiffness,
        transformed_mass,
    )
    gradient = jax.grad(
        lambda scale: phx.linalg.eigen.block_rayleigh_trace(
            scale * stiffness,
            mass,
        ).objective
    )(1.0)

    assert jnp.allclose(original.objective, transformed.objective, atol=2e-12)
    assert jnp.allclose(gradient, 4.0, atol=2e-12)


def test_variational_eigenspace_rejects_collapsed_trial_span():
    domain, first, _second = _dirichlet_modes()
    term = _term(domain)
    fields = {"u0": first, "u1": 2.0 * first}

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="invalid|positive-definiteness|Gram|Cholesky",
    ):
        jax.block_until_ready(term.loss(fields))


def test_variational_eigenspace_supports_complex_trial_phases():
    domain, first, second = _dirichlet_modes()
    term = _term(domain)
    fields = {"u0": 1.0j * first, "u1": -second}

    result = term.ritz(fields)

    assert bool(result.successful)
    assert jnp.allclose(
        result.eigenvalues,
        jnp.asarray([jnp.pi**2, 4.0 * jnp.pi**2]),
        rtol=2e-10,
        atol=2e-10,
    )

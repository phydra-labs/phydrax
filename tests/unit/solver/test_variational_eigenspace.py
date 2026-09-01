#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
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
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(48)),
        ),
        objective_vars=("u0", "u1"),
        stiffness_form=_energy,
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
        lambda scale: (
            phx.linalg.eigen.block_rayleigh_trace(
                scale * stiffness,
                mass,
            ).objective
        )
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


def _strong_residual_term(
    domain,
    *,
    objective_vars=("u0", "u1"),
    metric_action=None,
    pairing=None,
    residual_pairing=None,
):
    return phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(48)),
        ),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        metric_action=metric_action,
        pairing=pairing,
        residual_pairing=residual_pairing,
        objective_vars=objective_vars,
    )


def test_invariant_subspace_residual_solves_strong_dirichlet_modes():
    domain, first, second = _dirichlet_modes()
    term = _strong_residual_term(domain)
    fields = {"u0": first, "u1": second}

    evaluation = term.assemble(fields)
    result = term.ritz(fields)

    assert bool(evaluation.valid)
    assert bool(result.successful)
    assert evaluation.objective < 1e-24
    assert evaluation.residual_gram_minimum_eigenvalue > -1e-24
    assert jnp.allclose(
        result.eigenvalues,
        jnp.asarray([jnp.pi**2, 4.0 * jnp.pi**2]),
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.max(result.relative_residuals) < 2e-15


def test_invariant_subspace_residual_supports_positive_generalized_metric():
    domain, first, second = _dirichlet_modes()
    term = _strong_residual_term(
        domain,
        metric_action=lambda field: 2.0 * field,
    )

    result = term.ritz({"u0": first, "u1": second})

    assert bool(result.successful)
    assert jnp.allclose(
        result.eigenvalues,
        jnp.asarray([0.5 * jnp.pi**2, 2.0 * jnp.pi**2]),
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.max(result.relative_residuals) < 2e-15


def test_invariant_residual_is_trial_basis_invariant_and_cluster_safe():
    domain = phx.domain.Interval1d(0.0, 1.0)
    first = domain.Function("x")(lambda x: x[0] * (1.0 - x[0]))
    second = domain.Function("x")(lambda x: x[0] * (1.0 - x[0]) * (2.0 * x[0] - 1.0))
    term = _strong_residual_term(domain)
    original = {"u0": first, "u1": second}
    mixed = {
        "u0": 2.0 * first + second,
        "u1": -first + 3.0 * second,
    }

    original_evaluation = term.assemble(original)
    mixed_evaluation = term.assemble(mixed)
    original_result = term.ritz(original)
    mixed_result = term.ritz(mixed)

    assert original_evaluation.objective > 0.0
    assert jnp.allclose(
        original_evaluation.objective,
        mixed_evaluation.objective,
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.allclose(
        original_result.eigenvalues,
        mixed_result.eigenvalues,
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.allclose(
        original_result.relative_residuals,
        mixed_result.relative_residuals,
        rtol=2e-10,
        atol=2e-10,
    )

    identity_term = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24)),
        ),
        operator_action=lambda field: field,
        objective_vars=("u0", "u1"),
    )

    def clustered_loss(scale):
        return identity_term.loss(
            {
                "u0": first + scale * second,
                "u1": second,
            }
        )

    gradient = jax.grad(clustered_loss)(jnp.asarray(0.2))
    assert jnp.isfinite(gradient)


def test_invariant_residual_rejects_collapsed_trial_span():
    domain, first, _second = _dirichlet_modes()
    term = _strong_residual_term(domain)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="invalid|positive-definiteness|Cholesky",
    ):
        jax.block_until_ready(
            term.loss(
                {
                    "u0": first,
                    "u1": 2.0 * first,
                }
            )
        )


def test_invariant_residual_is_complex_phase_invariant():
    domain, first, second = _dirichlet_modes()
    term = _strong_residual_term(domain)
    reference = term.ritz({"u0": first, "u1": second})
    phased = term.ritz({"u0": 1.0j * first, "u1": -second})

    assert bool(phased.successful)
    assert jnp.allclose(phased.eigenvalues, reference.eigenvalues, atol=2e-12)
    assert jnp.allclose(
        phased.relative_residuals,
        reference.relative_residuals,
        atol=2e-15,
    )


def test_invariant_residual_rejects_non_self_adjoint_projection():
    domain, first, second = _dirichlet_modes()
    term = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(32)),
        ),
        operator_action=lambda field: phx.operators.partial_n(
            field,
            var="x",
            axis=0,
            order=1,
        ),
        objective_vars=("u0", "u1"),
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="Hermitian",
    ):
        jax.block_until_ready(term.loss({"u0": first, "u1": second}))


def test_invariant_residual_rejects_indefinite_residual_pairing():
    domain = phx.domain.Interval1d(0.0, 1.0)
    first = domain.Function("x")(lambda x: x[0] * (1.0 - x[0]))
    term = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(32)),
        ),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        residual_pairing=lambda left, right: -phx.operators.conjugate(left) * right,
        objective_vars=("u",),
    )

    evaluation = term.assemble({"u": first})
    assert not bool(evaluation.residual_gram_positive_semidefinite)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="invalid",
    ):
        jax.block_until_ready(term.loss({"u": first}))


def test_single_mode_residual_exposes_scalar_convenience_only_for_one_field():
    domain, first, second = _dirichlet_modes()
    single = _strong_residual_term(domain, objective_vars=("u",))
    single_result = single.ritz({"u": first})
    block_result = _strong_residual_term(domain).ritz({"u0": first, "u1": second})

    assert jnp.allclose(single_result.eigenvalue, jnp.pi**2, atol=2e-10)
    assert isinstance(single_result.mode, phx.domain.DomainFunction)
    with pytest.raises(ValueError, match="one trial field"):
        _ = block_result.eigenvalue
    with pytest.raises(ValueError, match="one trial field"):
        _ = block_result.mode


def test_invariant_residual_weight_must_be_nonnegative():
    domain, _first, _second = _dirichlet_modes()
    with pytest.raises(ValueError, match="non-negative"):
        phx.terms.InvariantSubspaceResidual(
            source=phx.integration.per_step(
                phx.integration.over(domain.component()),
                phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
            ),
            operator_action=lambda field: field,
            objective_vars=("u",),
            weight=-1.0,
        )


def test_invariant_residual_integration_sources_are_explicit():
    domain, first, _second = _dirichlet_modes()
    target = phx.integration.over(domain.component())
    plan = phx.integration.MonteCarloPlan(128)
    fixed_realization = phx.integration.materialize(target, plan, key=jr.key(1))
    fixed = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.fixed(fixed_realization),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        objective_vars=("u",),
    )
    assert fixed.sample(key=jr.key(2)) is fixed_realization

    realization = phx.integration.materialize(
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24)),
    )
    caller = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.caller(target),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        objective_vars=("u",),
    )
    with pytest.raises(ValueError, match="requires batch"):
        caller.loss({"u": first})
    assert (
        caller.assemble({"u": first}, batch=realization).provenance
        == "FixedQuadraturePlan"
    )


def test_invariant_residual_supports_vector_fields_with_explicit_pairing():
    domain = phx.domain.Interval1d(0.0, 1.0)
    first = domain.Function("x")(lambda x: jnp.asarray([jnp.sin(jnp.pi * x[0]), 0.0]))
    second = domain.Function("x")(
        lambda x: jnp.asarray([0.0, jnp.sin(2.0 * jnp.pi * x[0])])
    )

    def vector_pairing(left, right):
        return phx.operators.einsum(
            "...i,...i->...",
            phx.operators.conjugate(left),
            right,
        )

    term = _strong_residual_term(
        domain,
        pairing=vector_pairing,
        residual_pairing=vector_pairing,
    )
    result = term.ritz({"u0": first, "u1": second})

    assert bool(result.successful)
    assert jnp.allclose(
        result.eigenvalues,
        jnp.asarray([jnp.pi**2, 4.0 * jnp.pi**2]),
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.max(result.relative_residuals) < 2e-15


def test_functional_solver_refines_held_out_strong_eigen_residual():
    domain, first, second = _dirichlet_modes()
    amplitude = domain.Parameter(0.25)
    trial = first + amplitude * second
    training = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24)),
        ),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        objective_vars=("u",),
    )
    held_out = phx.terms.InvariantSubspaceResidual(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(56)),
        ),
        operator_action=lambda field: -phx.operators.laplacian(field, var="x"),
        objective_vars=("u",),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": trial},
        terms=(training,),
    )
    initial = held_out.ritz(solver.functions)
    trained = solver.solve(
        num_iter=40,
        optim=optax.adam(5e-2),
        keep_best=True,
        jit=True,
        log_every=0,
    )
    final = held_out.ritz(trained.functions)

    assert final.relative_residuals[0] < initial.relative_residuals[0]
    assert jnp.abs(final.eigenvalue - jnp.pi**2) < jnp.abs(initial.eigenvalue - jnp.pi**2)
    assert bool(final.successful)

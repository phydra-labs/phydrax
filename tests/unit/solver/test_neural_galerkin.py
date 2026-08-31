#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _constant_growth_problem(*, initial=1.0):
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray(initial))
    component = domain.component()
    batch = component.sample(
        phx.domain.PointSampling(
            8,
            layout=phx.domain.SampleLayout((("x",),)),
            design="uniform",
        ),
        key=jr.key(0),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
    problem = phx.solver.NeuralGalerkinProblem(
        {"u": field},
        lambda _time, functions, _args: {"u": functions["u"]},
        (phx.solver.FieldProjectionMetric("u", realization),),
        problem_id="constant-neural-growth",
    )
    return domain, batch, problem


def test_diffrax_neural_galerkin_evolves_and_reconstructs_constant_field():
    _domain, batch, problem = _constant_growth_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.1, 0.2]),
        time_id="constant-growth-grid",
    )
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        solver=dfx.Euler(),
        dt0=0.01,
        tangent=phx.solver.NeuralTangentSolvePolicy(damping=0.0),
        dense=True,
    )

    expected = (1.0 + 0.01) ** 20
    assert bool(result.successful)
    assert jnp.allclose(result.parameter_solution.states[-1, 0], expected, rtol=2e-6)
    final = result.field_at(2, "u")(batch)
    assert jnp.allclose(final.data, expected)
    dense = result.functions_at_time(jnp.asarray(0.15))["u"](batch)
    assert jnp.all(jnp.isfinite(dense.data))
    assert jnp.all(result.audit.accepted)
    assert jnp.max(result.audit.relative_projection_defect) < 1e-10


def test_default_tsit5_neural_galerkin_matches_exponential_growth():
    _domain, _batch, problem = _constant_growth_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.4, 5),
        time_id="tsit-growth-grid",
    )
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        tangent=phx.solver.NeuralTangentSolvePolicy(damping=0.0),
        rtol=1e-7,
        atol=1e-9,
    )

    assert bool(result.successful)
    assert jnp.allclose(
        result.parameter_solution.states[-1, 0],
        jnp.exp(0.4),
        rtol=2e-5,
        atol=2e-6,
    )
    assert result.parameter_solution.solver_name == "Tsit5"


def test_gram_neural_galerkin_accepts_randomized_nystrom_preconditioning():
    _domain, _batch, problem = _constant_growth_problem()
    damping = 1e-3
    tangent = phx.solver.NeuralTangentSolvePolicy(
        "gram",
        damping=damping,
        preconditioner=phx.linalg.RandomizedNystromPreconditionerBuilder(
            1,
            oversampling=0,
            shift=damping,
        ),
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.05]),
        time_id="gram-growth-grid",
    )
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        tangent=tangent,
        solver=dfx.Euler(),
        dt0=0.01,
    )

    assert bool(result.successful)
    assert jnp.all(result.audit.accepted)
    assert jnp.all(result.audit.matvec_count > 0)


def test_neural_galerkin_rejects_dynamic_and_malformed_metrics():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray(1.0))
    component = domain.component()
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(8),
    )
    with pytest.raises(TypeError, match="IntegrationRealization"):
        phx.solver.FieldProjectionMetric("u", source)

    batch = component.sample(phx.domain.PointSampling(4), key=jr.key(1))
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component), batch
    )
    with pytest.raises(KeyError, match="missing"):
        phx.solver.NeuralGalerkinProblem(
            {"u": field},
            lambda _time, functions, _args: {"u": functions["u"]},
            (phx.solver.FieldProjectionMetric("v", realization),),
        )


def test_neural_galerkin_rejects_rate_field_mismatch_and_backsolve_adjoint():
    domain, _batch, problem = _constant_growth_problem()
    bad = phx.solver.NeuralGalerkinProblem(
        problem.functions,
        lambda _time, _functions, _args: {"v": domain.Parameter(jnp.asarray(1.0))},
        problem.metrics,
        parameter_subspace=problem.parameter_subspace,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.1]),
        time_id="bad-rate-grid",
    )
    with pytest.raises(ValueError, match="exactly the evolved"):
        phx.solver.solve_neural_galerkin(
            bad,
            grid,
            solver=dfx.Euler(),
            dt0=0.1,
        )
    with pytest.raises(NotImplementedError, match="BacksolveAdjoint"):
        phx.solver.solve_neural_galerkin(
            problem,
            grid,
            adjoint=dfx.BacksolveAdjoint(),
        )


def test_neural_field_result_rejects_invalid_indices_and_missing_dense_output():
    _domain, _batch, problem = _constant_growth_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.05]),
        time_id="node-validation-grid",
    )
    result = phx.solver.solve_neural_galerkin(
        problem,
        grid,
        solver=dfx.Euler(),
        dt0=0.05,
    )

    with pytest.raises(IndexError, match="out of range"):
        result.functions_at(2)
    with pytest.raises(KeyError, match="Unknown"):
        result.field_at(0, "missing")
    with pytest.raises(ValueError, match="Dense"):
        result.functions_at_time(0.025)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


class _PartitionedDesign(eqx.Module):
    control: jax.Array
    scale: float = eqx.field(static=True)


def _termination(*, tolerance=1e-7, steps=50):
    return phx.optim.OptimizationTermination(
        absolute_optimality=tolerance,
        relative_optimality=0.0,
        maximum_steps=steps,
    )


def _forbid_explicit_jacobians(monkeypatch):
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("The matrix-free method formed an explicit Jacobian.")

    monkeypatch.setattr(jax, "jacrev", forbidden)
    monkeypatch.setattr(jax, "jacfwd", forbidden)


def test_primal_dual_newton_krylov_solves_mixed_constraints_without_jacobians(
    monkeypatch,
):
    equality = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.array([parameters[0] + parameters[1]]),
        lower=1.0,
        upper=1.0,
        constraint_id="sum",
    )
    inequality = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.array([parameters[0]]),
        upper=1.5,
        constraint_id="upper-first",
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - jnp.array([2.0, -1.0])) ** 2),
        constraints=(equality, inequality),
        problem_id="mixed-matrix-free",
    )
    _forbid_explicit_jacobians(monkeypatch)

    result = phx.optim.minimize(
        problem,
        jnp.array([3.0, 3.0]),
        method=phx.optim.PrimalDualNewtonKrylov(),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.5, -0.5]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.provenance.matrix_free
    assert result.diagnostics.jacobian_evaluations == 0
    assert result.diagnostics.hvp_evaluations > 0
    assert result.diagnostics.setup_refreshes == 1
    assert result.diagnostics.numeric_refreshes > 0
    assert result.certificate is not None
    np.testing.assert_allclose(
        result.certificate.equality_multipliers,
        jnp.array([-0.5]),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.certificate.inequality_multipliers,
        jnp.array([1.0]),
        atol=2e-6,
    )
    assert result.certificate.active_mask.tolist() == [True]
    assert result.certificate.equality_sources == ("constraint:0:0:equality",)
    assert result.certificate.inequality_sources == ("constraint:1:0:upper",)


def test_primal_dual_canonical_layout_includes_parameter_bounds():
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
        problem_id="bound-certificate",
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=phx.optim.PrimalDualNewtonKrylov(),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=2e-6)
    assert result.certificate.inequality_sources == ("bound:0:upper",)
    np.testing.assert_allclose(
        result.certificate.inequality_multipliers,
        jnp.array([2.0]),
        atol=2e-6,
    )
    assert result.certificate.primal_feasibility < 1e-7
    assert result.certificate.dual_feasibility < 1e-7
    assert result.certificate.complementarity < 1e-7


def test_primal_dual_eager_and_filtered_jit_agree_with_large_step_limit():
    problem = phx.optim.MinimizationProblem(
        lambda parameters, target: jnp.sum((parameters - target) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
        problem_id="compiled-primal-dual",
    )
    method = phx.optim.PrimalDualNewtonKrylov()
    termination = _termination(steps=100_000)

    def solve(target):
        return phx.optim.minimize(
            problem,
            jnp.array([0.0]),
            method=method,
            termination=termination,
            args=target,
        )

    eager = solve(jnp.array(2.0))
    compiled = eqx.filter_jit(solve)(jnp.array(2.0))

    np.testing.assert_allclose(compiled.parameters, eager.parameters, atol=2e-6)
    np.testing.assert_allclose(
        compiled.certificate.stationarity_residual,
        eager.certificate.stationarity_residual,
        atol=2e-6,
    )
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.setup_refreshes) == 1
    assert int(compiled.diagnostics.numeric_refreshes) == int(
        eager.diagnostics.numeric_refreshes
    )
    assert int(compiled.diagnostics.numeric_refreshes) == (
        int(compiled.diagnostics.linear_solves) + 1
    )


def test_primal_dual_filtered_jit_supports_function_operator_preconditioner():
    dtype = jnp.asarray(0.0).dtype
    kkt_space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace((1,), dtype=dtype),
            phx.linalg.ArraySpace((0,), dtype=dtype),
        )
    )
    scale = jnp.asarray(1.0, dtype=dtype)
    inverse_operator = phx.linalg.FunctionLinearOperator(
        lambda blocks: (scale * blocks[0], scale * blocks[1]),
        source=kkt_space,
        target=kkt_space,
        operator_id="jit-primal-dual-function-preconditioner",
    )
    preconditioner = phx.linalg.OperatorPreconditioner(
        inverse_operator,
        positive_definite=True,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.MINRES(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-8,
            absolute=1e-8,
            max_steps=200,
        ),
        preconditioning=phx.linalg.PreconditioningPolicy(preconditioner),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum((parameters - 2.0) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
    )
    solve = eqx.filter_jit(
        lambda initial: phx.optim.minimize(
            problem,
            initial,
            method=phx.optim.PrimalDualNewtonKrylov(linear_policy=policy),
            termination=_termination(),
        )
    )

    result = solve(jnp.array([0.0]))

    np.testing.assert_allclose(result.parameters, jnp.array([1.0]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.numeric_refreshes == (result.diagnostics.linear_solves + 1)


def test_primal_dual_final_certificate_promotes_exhausted_budget_to_success():
    equality = phx.optim.NonlinearConstraint(
        lambda parameters, _: parameters,
        lower=1.0,
        upper=1.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - 2.0) ** 2),
        constraints=(equality,),
    )
    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=phx.optim.PrimalDualNewtonKrylov(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-7,
            relative_optimality=0.0,
            maximum_steps=10,
            maximum_evaluations=1,
        ),
    )

    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.accepted_steps == 1
    assert result.diagnostics.objective_evaluations > 1
    assert result.certificate.primal_feasibility <= 1e-7
    assert result.certificate.dual_feasibility <= 1e-7


def test_primal_dual_handles_redundant_equalities_matrix_free():
    constraint = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.repeat(jnp.sum(parameters)[None], 2),
        lower=1.0,
        upper=1.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: 0.5 * jnp.sum((parameters - jnp.array([2.0, -1.0])) ** 2),
        constraints=(constraint,),
    )

    result = phx.optim.minimize(
        problem,
        jnp.zeros(2),
        method=phx.optim.PrimalDualNewtonKrylov(),
        termination=_termination(),
    )

    np.testing.assert_allclose(result.parameters, jnp.array([2.0, -1.0]), atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.diagnostics.primal_feasibility < 1e-7


def test_primal_dual_reports_explicit_restoration_failure():
    impossible = phx.optim.NonlinearConstraint(
        lambda parameters, _: jnp.ones_like(parameters),
        lower=0.0,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda parameters, _: jnp.sum(0.0 * parameters),
        constraints=(impossible,),
    )

    result = phx.optim.minimize(
        problem,
        jnp.array([1.0]),
        method=phx.optim.PrimalDualNewtonKrylov(linear_maximum_steps=3),
        termination=_termination(steps=5),
    )

    assert int(result.status) == int(phx.optim.OptimizationStatus.RESTORATION_FAILED)
    assert result.diagnostics.direction_fallbacks == 1
    assert result.diagnostics.primal_feasibility == 1.0
    np.testing.assert_array_equal(result.parameters, jnp.array([1.0]))
    assert result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_steps == 1
    assert result.diagnostics.setup_refreshes == 1
    assert result.diagnostics.numeric_refreshes == 2
    np.testing.assert_array_equal(
        result.certificate.stationarity_residual,
        jnp.array([0.0]),
    )


def test_reduced_newton_krylov_uses_incremental_state_and_adjoint_actions(
    monkeypatch,
):
    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: jnp.sum((state - 2.0) ** 2) + 0.1 * jnp.sum(design**2),
        problem_id="reduced-newton-linear",
    )
    _forbid_explicit_jacobians(monkeypatch)

    result = phx.optim.solve_state_design(
        problem,
        jnp.array([0.0]),
        jnp.array([0.0]),
        method=phx.optim.ReducedNewtonKrylov(),
        termination=_termination(tolerance=1e-6, steps=10),
    )

    expected = jnp.array([4.0 / 2.2])
    np.testing.assert_allclose(result.state, expected, atol=2e-6)
    np.testing.assert_allclose(result.design, expected, atol=2e-6)
    assert int(result.status) == int(phx.optim.OptimizationStatus.SUCCESS)
    assert result.provenance.matrix_free
    assert "incremental state" in result.provenance.notes
    assert result.diagnostics.linear_solves > 3
    assert result.diagnostics.setup_refreshes > 0
    assert result.diagnostics.numeric_refreshes > 0


def test_reduced_newton_krylov_jit_supports_partitioned_nested_state_design():
    wrapped = _PartitionedDesign(jnp.array([0.0]), 2.0)
    initial_design, static_design = eqx.partition(
        wrapped,
        eqx.is_inexact_array,
    )
    initial_state = {"field": jnp.array([0.0])}

    def physical(design):
        return eqx.combine(design, static_design)

    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: {
            "field": state["field"] - physical(design).scale * physical(design).control
        },
        lambda state, design, target: (
            jnp.sum((state["field"] - target) ** 2)
            + 0.1 * jnp.sum(physical(design).control ** 2)
        ),
        problem_id="partitioned-reduced-newton",
    )
    method = phx.optim.ReducedNewtonKrylov()
    termination = _termination(tolerance=1e-7, steps=100_000)

    def solve(target):
        return phx.optim.solve_state_design(
            problem,
            initial_state,
            initial_design,
            method=method,
            termination=termination,
            args=target,
        )

    eager = solve(jnp.array([2.0]))
    compiled = eqx.filter_jit(solve)(jnp.array([2.0]))
    eager_design = physical(eager.design)
    compiled_design = physical(compiled.design)

    np.testing.assert_allclose(
        compiled_design.control,
        eager_design.control,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compiled.state["field"],
        eager.state["field"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compiled_design.control,
        jnp.array([4.0 / 4.1]),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        compiled.state["field"],
        jnp.array([8.0 / 4.1]),
        atol=2e-6,
    )
    assert (
        int(compiled.status)
        == int(eager.status)
        == int(phx.optim.OptimizationStatus.SUCCESS)
    )
    assert int(compiled.diagnostics.iterations) == int(eager.diagnostics.iterations)
    assert int(compiled.diagnostics.setup_refreshes) == int(
        eager.diagnostics.setup_refreshes
    )
    assert int(compiled.diagnostics.setup_refreshes) > 3
    assert int(compiled.diagnostics.numeric_refreshes) == int(
        eager.diagnostics.numeric_refreshes
    )

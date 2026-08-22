#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem(scale=1.0):
    return phx.optim.QuadraticProgram(
        scale * jnp.eye(2),
        jnp.asarray([-1.0, -2.0]),
        inequality_matrix=-jnp.eye(2),
        inequality_rhs=jnp.zeros(2),
        problem_id="lifecycle-qp",
    )


def test_plan_prepare_refresh_and_solve_match_one_shot():
    initial = _problem()
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.DensePrimalDualQP(),
        termination=phx.optim.ConvexTermination(absolute=1e-8),
    )
    plan = phx.optim.plan_convex_program(initial, policy)
    template = phx.optim.prepare_convex_template(initial, plan)
    prepared = phx.optim.bind_convex_numeric(template, initial)
    first = phx.optim.solve_convex_program(prepared)

    refreshed_problem = _problem(2.0)
    refreshed = phx.optim.refresh_convex_program(prepared, refreshed_problem)
    second = phx.optim.solve_convex_program(refreshed)
    expected = phx.optim.solve_quadratic_program(refreshed_problem)

    np.testing.assert_allclose(first.result.primal, [1.0, 2.0], atol=2e-5)
    np.testing.assert_allclose(second.result.primal, expected.primal, atol=2e-5)
    assert prepared.numeric_version == 0
    assert refreshed.numeric_version == 1
    assert first.plan_id == second.plan_id
    assert first.result.provenance.numeric_version == 0
    assert second.result.provenance.numeric_version == 1
    assert second.result.provenance.structure_id == refreshed_problem.structure_id
    assert second.result.provenance.policy_id == policy.policy_id


def test_refresh_rejects_topology_and_identity_changes():
    prepared = phx.optim.prepare_convex_program(_problem())
    changed_dimension = phx.optim.QuadraticProgram(
        jnp.eye(3),
        jnp.zeros(3),
        problem_id="lifecycle-qp",
    )
    with pytest.raises(ValueError, match="preserve the convex-program structure"):
        phx.optim.refresh_convex_program(prepared, changed_dimension)

    changed_identity = phx.optim.QuadraticProgram(
        jnp.eye(2),
        jnp.zeros(2),
        inequality_matrix=-jnp.eye(2),
        inequality_rhs=jnp.zeros(2),
        problem_id="different-id",
    )
    with pytest.raises(ValueError, match="preserve the convex-program structure"):
        phx.optim.refresh_convex_program(prepared, changed_identity)


def test_prepared_solve_rejects_policy_override():
    prepared = phx.optim.prepare_convex_program(_problem())
    with pytest.raises(ValueError, match="policy must be omitted"):
        phx.optim.solve_convex_program(
            prepared,
            policy=phx.optim.ConvexSolvePolicy(),
        )


def test_dense_planning_and_direct_solve_enforce_resource_contracts():
    problem = _problem()
    materialization_limited = phx.optim.ConvexSolvePolicy(
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=1,
            max_bytes=1024,
        )
    )
    factorization_limited = phx.optim.ConvexSolvePolicy(
        resources=phx.linalg.SolveResourcePolicy(factorization_bytes=1)
    )

    with pytest.raises(ValueError, match="materialization limit"):
        phx.optim.plan_convex_program(problem, materialization_limited)
    with pytest.raises(ValueError, match="factorization estimate"):
        phx.optim.solve_quadratic_program(problem, policy=factorization_limited)
    assert materialization_limited.policy_id != factorization_limited.policy_id


def test_dense_warm_start_is_explicit_and_reuses_audited_state():
    problem = _problem()
    cold = phx.optim.solve_quadratic_program(problem)
    warm = phx.optim.ConvexWarmStart.from_result(cold, interior_margin=1e-7)
    restarted = phx.optim.solve_quadratic_program(problem, warm_start=warm)

    np.testing.assert_allclose(restarted.primal, cold.primal, atol=2e-6)
    assert restarted.status == phx.optim.ConvexProgramStatus.OPTIMAL
    assert restarted.iterations <= cold.iterations


def test_qpax_rejects_warm_start_before_backend_execution():
    problem = _problem()
    cold = phx.optim.solve_quadratic_program(problem)
    warm = phx.optim.ConvexWarmStart.from_result(cold)
    policy = phx.optim.ConvexSolvePolicy(phx.optim.QPaxInteriorPoint())
    prepared = phx.optim.prepare_convex_program(problem, policy)

    with pytest.raises(ValueError, match="does not support warm starts"):
        phx.optim.solve_convex_program(prepared, warm_start=warm)

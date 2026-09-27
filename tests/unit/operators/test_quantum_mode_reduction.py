#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


q = phx.operators.quantum


def test_mode_reduction_contracts() -> None:
    hamiltonian = jnp.diag(jnp.asarray([0.0, 1.0, 3.0]))
    position = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    lowering = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, jnp.sqrt(2.0)], [0.0, 0.0, 0.0]])
    problem = q.ModeReductionProblem(
        hamiltonian,
        (
            q.NamedModeOperator("position", position, hermitian=True),
            q.NamedModeOperator("lowering", lowering),
        ),
    )
    policy = q.ModeReductionPolicy(2, minimum_boundary_gap=1.5)

    prepared = q.prepare_mode_reduction(problem, policy=policy)

    assert bool(prepared.diagnostics.valid)
    assert jnp.allclose(prepared.energies, jnp.asarray([0.0, 1.0]))
    assert jnp.allclose(prepared.operator("position").matrix, position[:2, :2])
    assert jnp.allclose(prepared.operator("lowering").matrix, lowering[:2, :2])
    assert float(prepared.diagnostics.boundary_gap) == pytest.approx(2.0)
    assert (
        prepared.prepared_id
        == q.prepare_mode_reduction(problem, policy=policy).prepared_id
    )
    coarse_problem = q.ModeReductionProblem(
        jnp.diag(jnp.asarray([0.0, 1.0, 4.0])),
        (q.NamedModeOperator("number", jnp.diag(jnp.arange(3.0)), hermitian=True),),
        problem_id="coarse",
    )
    fine_problem = q.ModeReductionProblem(
        jnp.diag(jnp.asarray([0.0, 1.0, 4.0, 8.0])),
        (q.NamedModeOperator("number", jnp.diag(jnp.arange(4.0)), hermitian=True),),
        problem_id="fine",
    )
    policy = q.ModeReductionPolicy(2)
    coarse = q.prepare_mode_reduction(coarse_problem, policy=policy)
    fine = q.prepare_mode_reduction(fine_problem, policy=policy)

    report = q.compare_mode_resolutions(coarse, fine)

    assert bool(report.valid)
    assert not bool(report.subspace_overlap_available)
    assert jnp.isnan(report.minimum_subspace_overlap)
    with pytest.raises(ValueError, match="shape"):
        q.compare_mode_resolutions(
            coarse,
            fine,
            coarse_to_fine=jnp.eye(3),
        )
    with pytest.raises(ValueError, match="unique"):
        q.ModeReductionProblem(
            jnp.eye(2),
            (
                q.NamedModeOperator("x", jnp.eye(2)),
                q.NamedModeOperator("x", jnp.eye(2)),
            ),
        )
    problem = q.ModeReductionProblem(jnp.asarray([[0.0, 1.0], [0.0, 1.0]]))
    prepared = q.prepare_mode_reduction(
        problem,
        policy=q.ModeReductionPolicy(1),
    )
    assert not bool(prepared.diagnostics.valid)

    with pytest.raises(ValueError, match="must not exceed"):
        q.plan_mode_reduction(problem, q.ModeReductionPolicy(3))
    with pytest.raises(ValueError, match="maximum_raw_dimension"):
        q.plan_mode_reduction(
            problem,
            q.ModeReductionPolicy(1, maximum_raw_dimension=1),
        )


def test_mode_reduction_refresh_tracks_labels_and_is_differentiable() -> None:
    coupling = jnp.asarray(0.15)

    def problem(detuning: Any) -> Any:
        hamiltonian = jnp.asarray(
            [[0.0, coupling], [coupling, detuning]], dtype=jnp.complex128
        )
        return q.ModeReductionProblem(
            hamiltonian,
            (
                q.NamedModeOperator(
                    "z", jnp.diag(jnp.asarray([1.0, -1.0])), hermitian=True
                ),
            ),
            problem_id="tracked-mode",
        )

    prepared = q.prepare_mode_reduction(
        problem(jnp.asarray(2.0)),
        policy=q.ModeReductionPolicy(2),
    )
    refreshed = jax.jit(q.refresh_mode_reduction)(
        prepared,
        problem(jnp.asarray(2.1)),
    )
    derivative = jax.grad(
        lambda value: q.refresh_mode_reduction(prepared, problem(value)).energies[0]
    )(jnp.asarray(2.0))

    assert bool(refreshed.diagnostics.valid)
    assert int(refreshed.numeric_version) == 1
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert jnp.isfinite(derivative)
    assert derivative > 0.0

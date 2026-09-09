#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._multipoint_state_design import (
    MultipointStateDesignProblem,
    StateDesignCase,
)


opt = phx.optim


def _policy():
    return opt.StateAcceptancePolicy(
        state_relative_tolerance=1e-8,
        state_absolute_tolerance=1e-8,
        adjoint_relative_tolerance=1e-8,
        adjoint_absolute_tolerance=1e-8,
    )


def _bind(shared, local, args):
    del args
    return shared + local


def test_failed_heterogeneous_block_retains_separate_final_evidence():
    healthy = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: state**2 + design**2,
        acceptance_policy=_policy(),
    )
    inconsistent = opt.StateDesignProblem(
        lambda state, design, _: {"response": state["response"] * 0.0 + design},
        lambda state, design, _: jnp.sum(state["response"] ** 2) + design**2,
        acceptance_policy=_policy(),
    )
    multipoint = MultipointStateDesignProblem(
        (
            StateDesignCase("loaded", healthy, jnp.asarray(0.0), _bind),
            StateDesignCase(
                "inconsistent",
                inconsistent,
                {"response": jnp.zeros((2,))},
                _bind,
            ),
        )
    )
    result = multipoint.to_state_design_problem().solve_state(
        {"shared": jnp.asarray(1.0), "local": (jnp.asarray(1.0), jnp.asarray(0.0))},
        multipoint.initial_state,
    )
    assert not bool(result.successful)
    assert result.acceptance.block_ids == ("loaded", "inconsistent")
    good, bad = result.acceptance.blocks
    assert bool(good.accepted)
    assert not bool(bad.accepted)
    assert float(good.residual_norm) < 1e-7
    assert float(bad.residual_norm) == pytest.approx(2.0**0.5)
    assert float(result.state[0]) == pytest.approx(2.0, abs=1e-7)


def test_raw_child_status_failure_survives_residual_recertification():
    healthy = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: state**2,
        acceptance_policy=_policy(),
    )
    exhausted = opt.StateDesignProblem(
        lambda state, design, _: jnp.exp(state) - design,
        lambda state, design, _: state**2,
        state_solver=opt.LeastSquaresStateSolver(
            termination=opt.OptimizationTermination(
                absolute_optimality=1e-14,
                relative_optimality=0.0,
                maximum_steps=1,
            ),
        ),
        acceptance_policy=opt.StateAcceptancePolicy(
            state_absolute_tolerance=1.0,
            state_relative_tolerance=0.0,
        ),
    )
    multipoint = MultipointStateDesignProblem(
        (
            StateDesignCase("converged", healthy, jnp.asarray(0.0), _bind),
            StateDesignCase("exhausted", exhausted, jnp.asarray(0.0), _bind),
        )
    )
    problem = multipoint.to_state_design_problem()
    design = {"shared": jnp.asarray(2.0), "local": (jnp.asarray(0.0), jnp.asarray(0.0))}
    result = problem.solve_state(design, multipoint.initial_state)
    good, bad = result.acceptance.blocks
    assert bool(good.accepted)
    assert float(bad.residual_norm) <= float(bad.threshold)
    assert not bool(bad.status_accepted)
    assert not bool(result.successful)
    # Even a later successful outer status cannot re-label a failed child solve.
    fresh = problem.state_evidence(
        result.state,
        design,
        problem.residual(result.state, design),
        opt.OptimizationStatus.SUCCESS,
        reference_norm=0.0,
        solver_acceptance=result.acceptance,
    )
    assert bool(fresh.blocks[0].accepted)
    assert not bool(fresh.blocks[1].status_accepted)
    assert not bool(fresh.accepted)


def test_large_case_cannot_hide_small_state_or_transpose_defect():
    child = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: state**2 + design**2,
        acceptance_policy=_policy(),
    )
    multipoint = MultipointStateDesignProblem(
        (
            StateDesignCase("large", child, jnp.asarray(0.0), _bind),
            StateDesignCase("small", child, jnp.asarray(0.0), _bind),
        )
    )
    problem = multipoint.to_state_design_problem()
    design = {"shared": jnp.asarray(0.0), "local": (jnp.asarray(1e9), jnp.asarray(0.0))}
    state = (jnp.asarray(1e9), jnp.asarray(1e-3))
    evidence = problem.state_evidence(
        state,
        design,
        problem.residual(state, design),
        opt.OptimizationStatus.SUCCESS,
        reference_norm=1e30,
    )
    assert bool(evidence.blocks[0].accepted)
    assert not bool(evidence.blocks[1].accepted)
    assert not bool(evidence.accepted)
    assert float(evidence.blocks[0].reference_norm) == pytest.approx(1e9)
    assert float(evidence.blocks[1].reference_norm) == 0.0
    assert float(evidence.blocks[1].threshold) == pytest.approx(1e-8)

    transpose = problem.acceptance_policy.adjoint_evidence(
        (jnp.asarray(1e9), jnp.asarray(1.0)),
        (jnp.asarray(1e9), jnp.asarray(1.001)),
        (jnp.asarray(1e9), jnp.asarray(1.0)),
        phx.linalg.LinearSolveStatus.SUCCESS,
        admissible=True,
        realization_matches=True,
    )
    assert bool(transpose.blocks[0].accepted)
    assert not bool(transpose.blocks[1].accepted)
    assert not bool(transpose.accepted)
    assert float(transpose.blocks[1].transpose_defect_norm) == pytest.approx(
        1e-3, rel=1e-3
    )


def test_child_vector_equalities_bound_bindings_and_cross_case_constraints_survive():
    child = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: state**2,
        design_bounds=opt.Bounds(0.25, 2.0),
        constraints=(
            opt.StateDesignConstraint(
                lambda state, design, _: jnp.stack((state + design, design**2)),
                lower=jnp.asarray((1.0, -jnp.inf)),
                upper=jnp.asarray((1.0, 1.5)),
                constraint_id="response",
            ),
        ),
    )
    vector_child = opt.StateDesignProblem(
        lambda state, design, _: {"response": state["response"] - design},
        lambda state, design, _: jnp.sum(state["response"] ** 2),
        design_bounds=opt.Bounds(jnp.asarray((0.1, 0.2)), jnp.asarray((1.0, 2.0))),
    )
    multipoint = MultipointStateDesignProblem(
        (
            StateDesignCase("scalar", child, jnp.asarray(0.0), _bind),
            StateDesignCase(
                "vector",
                vector_child,
                {"response": jnp.zeros((2,))},
                lambda shared, local, scale: shared * local * scale,
                args=jnp.asarray(2.0),
            ),
        ),
        constraints=(
            opt.StateDesignConstraint(
                lambda state, design, limit: (
                    state[0] + jnp.sum(state[1]["response"]) - limit
                ),
                lower=0.0,
                upper=0.0,
                constraint_id="cross-case-balance",
            ),
        ),
    )
    problem = multipoint.to_state_design_problem()
    design = {
        "shared": jnp.asarray(0.5),
        "local": (jnp.asarray(0.0), jnp.asarray((0.5, 1.0))),
    }
    state = (jnp.asarray(0.5), {"response": jnp.asarray((0.5, 1.0))})
    assert tuple(item.constraint_id for item in problem.constraints) == (
        "scalar:response",
        "scalar:design-bounds",
        "vector:design-bounds",
        "cross-case-balance",
    )
    values = problem.constraint_values(state, design, jnp.asarray(2.0))
    np.testing.assert_allclose(values[0], (1.0, 0.25))
    np.testing.assert_allclose(values[1], 0.5)
    np.testing.assert_allclose(values[2], (0.5, 1.0))
    np.testing.assert_allclose(values[3], 0.0)
    compilation = opt.compile_structured_state_design(
        problem,
        state,
        design,
        sample_args=jnp.asarray(2.0),
        exact_hessian=False,
    )
    program = compilation.optimization.program
    np.testing.assert_allclose(
        program.constraint_lower,
        (0.0, 0.0, 0.0, 1.0, -jnp.inf, 0.25, 0.1, 0.2, 0.0),
    )
    np.testing.assert_allclose(
        program.constraint_upper,
        (0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 1.0, 2.0, 0.0),
    )
    assert program.equality_indices.tolist() == [0, 1, 2, 3, 8]


def test_case_identity_collision_cannot_alias_constraint_or_acceptance_evidence():
    child = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: state**2,
        design_bounds=opt.Bounds(0.0, 1.0),
    )
    case = StateDesignCase("same", child, jnp.asarray(0.0), _bind)
    with pytest.raises(ValueError, match="unique"):
        MultipointStateDesignProblem((case, case))
    multipoint = MultipointStateDesignProblem(
        (case,),
        constraints=(
            opt.StateDesignConstraint(
                lambda state, design, _: design["shared"],
                upper=0.5,
                constraint_id="same:design-bounds",
            ),
        ),
    )
    with pytest.raises(ValueError, match="unique"):
        multipoint.to_state_design_problem()


def test_rejected_case_rolls_back_entire_shared_local_state_design_pair():
    child = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, target: (state - target) ** 2 + 0.1 * design**2,
        acceptance_policy=_policy(),
    )
    limited = opt.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, target: (state - target) ** 2 + 0.1 * design**2,
        acceptance_policy=_policy(),
        state_admissibility=lambda state, design, _: state <= 0.5,
    )
    multipoint = MultipointStateDesignProblem(
        (
            StateDesignCase(
                "free", child, jnp.asarray(0.0), _bind, args=jnp.asarray(1.0)
            ),
            StateDesignCase(
                "limited", limited, jnp.asarray(0.0), _bind, args=jnp.asarray(2.0)
            ),
        )
    )
    initial_design = {
        "shared": jnp.asarray(0.0),
        "local": (jnp.asarray(0.0), jnp.asarray(0.0)),
    }
    result = opt.solve_state_design(
        multipoint.to_state_design_problem(),
        multipoint.initial_state,
        initial_design,
        method=opt.ReducedAdjoint(line_search=opt.ArmijoLineSearch(maximum_steps=1)),
        termination=opt.OptimizationTermination(maximum_steps=5),
    )
    assert int(result.status) == int(opt.OptimizationStatus.LINE_SEARCH_FAILED)
    np.testing.assert_array_equal(result.state, (0.0, 0.0))
    np.testing.assert_array_equal(result.design["local"], (0.0, 0.0))
    assert float(result.design["shared"]) == 0.0
    assert float(result.objective) == pytest.approx(5.0)
    assert result.state_acceptance.block_ids == ("free", "limited")
    assert all(bool(block.accepted) for block in result.state_acceptance.blocks)

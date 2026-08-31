from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


def _termination(steps=80):
    return phx.optim.OptimizationTermination(
        absolute_optimality=2.0e-5,
        relative_optimality=0.0,
        absolute_step=1.0e-11,
        relative_step=0.0,
        maximum_steps=steps,
    )


def _problem(*, state_dependent=False):
    constraint = phx.optim.StateDesignConstraint(
        (
            (lambda state, design, _: state)
            if state_dependent
            else (lambda state, design, _: design)
        ),
        upper=0.8 if state_dependent else 1.0,
        constraint_id="state-limit" if state_dependent else "design-limit",
        depends_on_state=state_dependent,
    )
    return phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: (state - 2.0) ** 2 + 0.1 * design**2,
        design_bounds=phx.optim.Bounds(0.05, 3.0),
        constraints=(constraint,),
        problem_id="analytic-constrained-state-design",
    )


def test_reduced_mma_solves_design_only_constraint_with_adjoint_gradient():
    result = phx.optim.solve_state_design(
        _problem(),
        jnp.asarray(0.5),
        jnp.asarray(0.5),
        method=phx.optim.ReducedMMA(),
        termination=_termination(),
    )

    assert result.successful
    assert float(result.design) == pytest.approx(1.0, abs=3.0e-4)
    assert float(result.state) == pytest.approx(1.0, abs=3.0e-4)
    assert result.certificate is not None
    assert result.certificate.inequality_sources == ("design-limit:upper",)
    assert result.diagnostics.primal_feasibility <= 2.0e-5
    assert isinstance(result.method_evidence, phx.optim.MMAEvidence)


def test_reduced_mma_solves_state_dependent_constraint():
    result = phx.optim.solve_state_design(
        _problem(state_dependent=True),
        jnp.asarray(0.5),
        jnp.asarray(0.5),
        method=phx.optim.ReducedMMA(),
        termination=_termination(steps=100),
    )

    assert result.successful
    assert float(result.design) == pytest.approx(0.8, abs=4.0e-4)
    assert float(result.state) == pytest.approx(0.8, abs=4.0e-4)
    assert result.diagnostics.linear_solves >= result.diagnostics.iterations


def test_other_state_design_methods_reject_unhandled_constraints():
    problem = _problem()
    with pytest.raises(ValueError, match="use ReducedMMA"):
        phx.optim.solve_state_design(
            problem,
            jnp.asarray(0.5),
            jnp.asarray(0.5),
            method=phx.optim.ReducedAdjoint(),
            termination=_termination(steps=2),
        )
    with pytest.raises(ValueError, match="does not support"):
        phx.optim.solve_state_design(
            problem,
            jnp.asarray(0.5),
            jnp.asarray(0.5),
            method=phx.optim.SimultaneousKKT(),
            termination=_termination(steps=2),
        )


def test_state_design_constraint_requires_scalar_finite_inequality_for_reduced_mma():
    vector_constraint = phx.optim.StateDesignConstraint(
        lambda state, design, _: jnp.stack((design, state)),
        upper=1.0,
        constraint_id="vector",
    )
    problem = phx.optim.StateDesignProblem(
        lambda state, design, _: state - design,
        lambda state, design, _: (state - 1.0) ** 2,
        design_bounds=phx.optim.Bounds(0.0, 2.0),
        constraints=(vector_constraint,),
    )
    with pytest.raises(ValueError, match="scalar"):
        phx.optim.solve_state_design(
            problem,
            jnp.asarray(0.5),
            jnp.asarray(0.5),
            method=phx.optim.ReducedMMA(),
            termination=_termination(steps=2),
        )

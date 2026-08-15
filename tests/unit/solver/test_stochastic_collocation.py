import jax.numpy as jnp
import pytest

import phydrax as phx


def _factors():
    uniform = phx.domain.ProbabilityDomain(
        phx.uq.Uniform(-1.0, 1.0), label="uniform-input"
    )
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(2.0, 3.0), label="normal-input")
    return uniform, normal


def test_stochastic_collocation_fits_surrogate_moments_and_level_difference():
    plan = phx.solver.StochasticCollocationPlan(
        _factors(),
        4,
        plan_id="polynomial-collocation",
    )
    design = phx.solver.materialize_stochastic_collocation(plan)
    calls = []

    def solve_node(node):
        calls.append(node.node_id)
        uniform = node.parameters["uniform-input"]
        normal = node.parameters["normal-input"]
        return phx.solver.StochasticCollocationNodeEvaluation(
            uniform**2 + normal,
            provenance=f"analytic:{node.node_id}",
        )

    result = phx.solver.evaluate_stochastic_collocation(design, solve_node)

    assert len(calls) == design.num_nodes
    assert len(set(calls)) == design.num_nodes
    assert result.successful
    assert result.interpolant is not None
    assert result.previous_interpolant is not None
    assert result.mean == pytest.approx(2.0 + 1.0 / 3.0, abs=1e-11)
    assert result.variance == pytest.approx(9.0 + 4.0 / 45.0, abs=1e-10)
    assert result.interpolant(jnp.asarray(0.2), jnp.asarray(1.5)) == pytest.approx(
        1.54,
        abs=1e-11,
    )
    assert result.diagnostics.current_weight_sum == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics.previous_weight_sum == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics.input_axis_labels == (
        "uniform-input",
        "normal-input",
    )
    assert result.diagnostics.axis_rules == (
        "clenshaw-curtis",
        "gauss-hermite",
    )
    assert result.diagnostics.mean_level_difference_norm is not None
    assert result.diagnostics.variance_level_difference_norm is not None
    assert result.diagnostics.mean_level_difference_norm < 1e-10
    assert result.diagnostics.variance_level_difference_norm < 1e-9


def test_collocation_retains_solver_failures_and_refuses_partial_surrogate():
    uniform = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="coefficient")
    plan = phx.solver.StochasticCollocationPlan(
        (uniform,),
        3,
        plan_id="failure-aware-collocation",
    )
    first_design = phx.solver.materialize_stochastic_collocation(plan)
    replay_design = phx.solver.materialize_stochastic_collocation(plan)

    assert tuple(node.node_id for node in first_design.nodes) == tuple(
        node.node_id for node in replay_design.nodes
    )

    def solve_node(node):
        coefficient = node.parameters["coefficient"]
        failed = coefficient > 0.9
        return phx.solver.StochasticCollocationNodeEvaluation(
            coefficient**2,
            valid=~failed,
            status=jnp.where(failed, 77, phx.solver.COLLOCATION_SUCCESS),
            provenance=f"conditional-pde:{node.node_id}",
        )

    result = phx.solver.evaluate_stochastic_collocation(first_design, solve_node)
    failed_indices = jnp.flatnonzero(result.status == 77)

    assert failed_indices.size > 0
    assert not result.successful
    assert result.interpolant is None
    assert jnp.isnan(result.mean)
    assert result.diagnostics.num_failed_nodes == failed_indices.size
    failed_index = int(failed_indices[0])
    assert result.evaluations[failed_index].provenance.startswith("conditional-pde:")
    assert (
        result.design.nodes[failed_index].node_id
        in result.evaluations[failed_index].provenance
    )


def test_collocation_marks_nonfinite_outputs_with_canonical_status():
    uniform = phx.domain.ProbabilityDomain(phx.uq.Uniform(0.0, 1.0), label="input")
    design = phx.solver.materialize_stochastic_collocation(
        phx.solver.StochasticCollocationPlan((uniform,), 2)
    )
    evaluations = tuple(
        phx.solver.StochasticCollocationNodeEvaluation(
            jnp.inf if node.index == 0 else node.parameters["input"]
        )
        for node in design.nodes
    )

    result = phx.solver.assemble_stochastic_collocation(design, evaluations)

    assert result.status[0] == phx.solver.COLLOCATION_NONFINITE
    assert not result.valid[0]
    assert not result.successful

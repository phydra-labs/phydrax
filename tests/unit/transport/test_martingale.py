import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _target(points, probabilities, provenance):
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        cx.Field(jnp.asarray(probabilities, dtype=float), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


def _transport(source, source_weights, target, target_weights):
    return phx.transport.discrete_problem(
        _target(source, source_weights, "source"),
        _target(target, target_weights, "target"),
        cost=phx.transport.PrecomputedCost(
            (jnp.asarray(source)[:, None] - jnp.asarray(target)[None, :]) ** 2
        ),
    )


def test_two_point_martingale_transport_has_exact_conditional_means_and_dual():
    base = _transport([-1.0, 1.0], [0.5, 0.5], [-2.0, 2.0], [0.5, 0.5])
    problem = phx.transport.MartingaleTransportProblem(base)
    result = phx.transport.solve_martingale_transport(problem)
    expected = jnp.asarray([[0.375, 0.125], [0.125, 0.375]])

    assert bool(result.successful)
    assert result.problem.constraint_kind == "martingale"
    assert jnp.allclose(result.coupling, expected, atol=2e-6)
    assert jnp.allclose(
        result.conditional_mean(), jnp.asarray([[-1.0], [1.0]]), atol=2e-6
    )
    assert result.evidence.martingale_defect <= problem.constraint_tolerance
    assert result.dual.maximum_inequality_violation <= problem.constraint_tolerance


def test_scalar_convex_order_failure_is_explicit_before_optimization():
    base = _transport([-2.0, 2.0], [0.5, 0.5], [-1.0, 1.0], [0.5, 0.5])
    result = phx.transport.solve_martingale_transport(
        phx.transport.MartingaleTransportProblem(base)
    )

    assert result.problem.convex_order.criterion_complete
    assert not bool(result.problem.convex_order.feasible)
    assert result.problem.convex_order.minimum_call_value_gap < 0.0
    assert not bool(result.successful)
    assert result.optimizer_result is None
    assert int(result.status) == int(
        phx.transport.MartingaleTransportStatus.CONVEX_ORDER_VIOLATION
    )


def test_marginals_alone_cannot_hide_a_martingale_defect_or_relabel_classical_ot():
    base = _transport([-1.0, 1.0], [0.5, 0.5], [-1.0, 1.0], [0.5, 0.5])
    problem = phx.transport.MartingaleTransportProblem(base)
    swapped = jnp.asarray([[0.0, 0.5], [0.5, 0.0]])
    evidence = phx.transport.audit_martingale_coupling(problem, swapped)

    assert evidence.source_marginal_residual == 0.0
    assert evidence.target_marginal_residual == 0.0
    assert evidence.martingale_defect == 2.0
    assert not bool(evidence.valid)
    with pytest.raises(TypeError, match="classical"):
        phx.transport.solve_martingale_transport(base)


def _matrix_kernel(matrix):
    probabilities = jnp.asarray(matrix, dtype=float)

    def sample(key, state, _t0, _t1, _context):
        row = jnp.asarray(state, dtype=jnp.int32)
        return jr.categorical(key, jnp.log(probabilities[row])).astype(float)

    def log_prob(next_state, state, _t0, _t1, _context):
        row = jnp.asarray(state, dtype=jnp.int32)
        column = jnp.asarray(next_state, dtype=jnp.int32)
        value = probabilities[row, column]
        return jnp.where(value > 0.0, jnp.log(value), -jnp.inf)

    return phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(),
        process_id="martingale-reference",
        approximation_id="exact-matrix",
        log_prob_fn=log_prob,
    )


def test_martingale_schrodinger_bridge_recovers_endpoints_and_conditional_means():
    support = jnp.asarray([-1.0, 1.0])
    initial = _target([0.0, 1.0], [0.5, 0.5], "initial")
    terminal = _target([0.0, 1.0], [0.5, 0.5], "terminal")
    base = phx.transport.dynamic.SchrodingerBridgeProblem(
        initial,
        terminal,
        jnp.asarray([0.0, 1.0]),
        _matrix_kernel([[0.8, 0.2], [0.2, 0.8]]),
        phx.stochastic.StateSpaceStepContext.empty(),
    )
    problem = phx.transport.dynamic.MartingaleSchrodingerBridgeProblem(
        base,
        martingale_coordinates=support,
        constraint_tolerance=1e-6,
    )
    result = phx.transport.dynamic.MartingaleSchrodingerBridgeSolver(
        max_iterations=50,
        moment_iterations=32,
        tolerance=1e-6,
        max_path_entries=16,
    )(problem)

    assert bool(result.successful)
    assert jnp.allclose(result.initial_marginal(), jnp.asarray([0.5, 0.5]), atol=1e-6)
    assert jnp.allclose(result.terminal_marginal(), jnp.asarray([0.5, 0.5]), atol=1e-6)
    assert result.diagnostics.endpoint_residual <= 1e-6
    assert result.diagnostics.martingale_defect <= 1e-6
    assert jnp.allclose(
        result.controlled_transition_probabilities[0], jnp.eye(2), atol=1e-6
    )

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


CONTEXT = phx.stochastic.StateSpaceStepContext.empty()


def _target(probabilities, *, provenance="endpoint"):
    states = jnp.arange(jnp.asarray(probabilities).shape[-1], dtype=float)
    return phx.integration.discrete(
        states,
        cx.Field(jnp.asarray(probabilities), dims=("state",)),
        axes="state",
        normalized=True,
        provenance=provenance,
    )


def _matrix_kernel(matrix, *, process_id="finite-reference"):
    probabilities = jnp.asarray(matrix, dtype=float)

    def sample(key, state, _t0, _t1, _context):
        index = jnp.asarray(state, dtype=jnp.int32)
        return jr.categorical(key, jnp.log(probabilities[index])).astype(float)

    def log_prob(next_state, state, _t0, _t1, _context):
        source = jnp.asarray(state, dtype=jnp.int32)
        target = jnp.asarray(next_state, dtype=jnp.int32)
        value = probabilities[source, target]
        return jnp.where(value > 0.0, jnp.log(value), -jnp.inf)

    return phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(),
        process_id=process_id,
        approximation_id="exact-matrix",
        log_prob_fn=log_prob,
    )


def _problem(initial, terminal, matrix, *, times=(0.0, 1.0)):
    return phx.transport.dynamic.SchrodingerBridgeProblem(
        _target(initial, provenance="initial"),
        _target(terminal, provenance="terminal"),
        jnp.asarray(times),
        _matrix_kernel(matrix),
        CONTEXT,
    )


def test_reference_equal_endpoints_preserve_stationary_reference():
    matrix = jnp.asarray([[0.75, 0.25], [0.25, 0.75]])
    problem = _problem([0.5, 0.5], [0.5, 0.5], matrix)
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)

    assert bool(result.converged)
    assert not result.approximate
    assert result.provenance.reference_process == "finite-reference"
    assert jnp.allclose(result.controlled_transition_probabilities[0], matrix)
    assert jnp.allclose(result.marginal_probabilities, 0.5)


def test_analytic_two_state_bridge_matches_cross_ratio_solution():
    matrix = jnp.asarray([[0.75, 0.25], [0.25, 0.75]])
    problem = _problem([0.5, 0.5], [0.75, 0.25], matrix)
    solver = phx.transport.dynamic.SchrodingerBridgeSolver(
        max_iterations=1000, tolerance=1e-12
    )
    result = solver.solve(problem)

    upper_left = (11.0 - jnp.sqrt(13.0)) / 16.0
    expected = jnp.asarray(
        [
            [upper_left, 0.5 - upper_left],
            [0.75 - upper_left, upper_left - 0.25],
        ]
    )
    assert bool(result.converged)
    assert jnp.allclose(result.endpoint_coupling, expected, atol=1e-10)
    assert jnp.allclose(
        result.controlled_transition_probabilities[0], expected / 0.5, atol=1e-10
    )
    explicit_kl = jnp.sum(
        jnp.where(
            expected > 0.0,
            expected * jnp.log(expected / (0.5 * matrix)),
            0.0,
        )
    )
    assert jnp.allclose(result.diagnostics.path_kl, explicit_kl, atol=1e-10)


def test_deterministic_feasible_and_infeasible_support_are_explicit():
    identity = jnp.eye(2)
    feasible = phx.transport.dynamic.solve_schrodinger_bridge(
        _problem([1.0, 0.0], [1.0, 0.0], identity)
    )
    infeasible = phx.transport.dynamic.solve_schrodinger_bridge(
        _problem([1.0, 0.0], [0.0, 1.0], identity)
    )

    assert bool(feasible.converged)
    assert jnp.array_equal(feasible.marginal_probabilities[:, 0], jnp.ones((2,)))
    assert not bool(infeasible.converged)
    assert not bool(infeasible.diagnostics.feasible)
    assert int(infeasible.diagnostics.status) == int(
        phx.transport.TransportStatus.INFEASIBLE_SUPPORT
    )
    with pytest.raises(Exception, match="did not converge"):
        phx.transport.dynamic.BridgeInferenceAdapter(infeasible)


def test_zero_probability_support_is_retained_and_reachable():
    flip = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    problem = _problem([1.0, 0.0], [0.0, 1.0], flip)
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)

    assert problem.num_states == 2
    assert jnp.array_equal(problem.state_support, jnp.asarray([0.0, 1.0]))
    assert bool(result.converged)
    assert jnp.allclose(result.initial_marginal(), jnp.asarray([1.0, 0.0]))
    assert jnp.allclose(result.terminal_marginal(), jnp.asarray([0.0, 1.0]))
    assert jnp.allclose(jnp.sum(result.controlled_transition_probabilities, axis=-1), 1.0)


@pytest.mark.parametrize(
    "times",
    [jnp.asarray([0.0]), jnp.asarray([0.0, 0.0]), jnp.asarray([0.0, -1.0]), jnp.asarray([0.0, jnp.nan])],
)
def test_invalid_time_grids_are_rejected(times):
    with pytest.raises(ValueError, match="times"):
        _problem([0.5, 0.5], [0.5, 0.5], jnp.eye(2), times=times)


def test_sampler_only_and_unnormalized_reference_transitions_are_rejected():
    sampler_only = phx.stochastic.CallableTransitionKernel(
        lambda key, state, _t0, _t1, _context: state,
        state_shape=(),
        process_id="sampler-only",
        approximation_id="sampled",
    )
    with pytest.raises(ValueError, match="sampler-only kernels"):
        phx.transport.dynamic.SchrodingerBridgeProblem(
            _target([0.5, 0.5]),
            _target([0.5, 0.5]),
            jnp.asarray([0.0, 1.0]),
            sampler_only,
            CONTEXT,
        )

    problem = _problem([0.5, 0.5], [0.5, 0.5], [[0.6, 0.6], [0.4, 0.4]])
    with pytest.raises(Exception, match="not normalized"):
        phx.transport.dynamic.solve_schrodinger_bridge(problem)


def test_endpoint_recovery_normalization_and_controlled_kernel_density():
    matrix = jnp.asarray([[0.8, 0.2], [0.3, 0.7]])
    problem = _problem([0.2, 0.8], [0.65, 0.35], matrix, times=(0.0, 0.5, 1.0))
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)
    kernel = result.controlled_kernel()

    assert jnp.allclose(result.marginal_probabilities[0], problem.initial_probabilities)
    assert jnp.allclose(result.marginal_probabilities[-1], problem.terminal_probabilities)
    assert jnp.allclose(jnp.sum(result.marginal_probabilities, axis=-1), 1.0)
    assert jnp.allclose(jnp.sum(result.controlled_transition_probabilities, axis=-1), 1.0)
    context = phx.stochastic.StateSpaceStepContext.empty(step_index=0)
    density = jnp.exp(
        jnp.stack(
            [kernel.log_prob(jnp.asarray(next_state), jnp.asarray(0.0), 0.0, 0.5, context) for next_state in (0.0, 1.0)]
        )
    )
    assert jnp.allclose(jnp.sum(density), 1.0)


def test_sampling_replay_prefix_stability_and_empirical_marginals():
    problem = _problem(
        [0.7, 0.3],
        [0.25, 0.75],
        [[0.85, 0.15], [0.1, 0.9]],
        times=(0.0, 0.5, 1.0),
    )
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)
    short = result.sample_state_indices(jr.key(21), sample_shape=(256,))
    replay = result.sample_state_indices(jr.key(21), sample_shape=(256,))
    long = result.sample_state_indices(jr.key(21), sample_shape=(4096,))

    assert jnp.array_equal(short, replay)
    assert jnp.array_equal(short, long[:256])
    empirical = jnp.mean(jax.nn.one_hot(long, 2), axis=0)
    assert jnp.allclose(empirical, result.marginal_probabilities, atol=0.035)
    paths = result.sample_paths(jr.key(8), sample_shape=(64,))
    assert paths.shape == (64, 3)
    assert jnp.all(jnp.isfinite(result.path_log_prob(paths)))


def test_solver_is_jittable_and_path_kl_is_differentiable():
    matrix = jnp.asarray([[0.7, 0.3], [0.2, 0.8]])
    solver = phx.transport.dynamic.SchrodingerBridgeSolver(
        max_iterations=200, tolerance=1e-10
    )
    problem = _problem([0.4, 0.6], [0.6, 0.4], matrix)
    compiled = eqx.filter_jit(solver.solve)(problem)
    assert bool(compiled.converged)

    def objective(logits):
        terminal = jax.nn.softmax(logits)
        dynamic_problem = _problem([0.4, 0.6], terminal, matrix)
        return solver.solve(dynamic_problem).diagnostics.path_kl

    gradient = jax.grad(objective)(jnp.asarray([0.2, -0.2]))
    assert gradient.shape == (2,)
    assert jnp.all(jnp.isfinite(gradient))
    assert not jnp.allclose(gradient, 0.0)


def test_named_cases_are_solved_independently_without_cross_case_mass():
    states = jnp.asarray([0.0, 1.0])
    initial = phx.integration.discrete(
        states,
        cx.Field(jnp.asarray([[0.8, 0.2], [0.1, 0.9]]), dims=("case", "state")),
        axes="state",
        normalized=True,
        provenance="case-initial",
    )
    terminal = phx.integration.discrete(
        states,
        cx.Field(jnp.asarray([[0.6, 0.4], [0.7, 0.3]]), dims=("case", "state")),
        axes="state",
        normalized=True,
        provenance="case-terminal",
    )
    problem = phx.transport.dynamic.SchrodingerBridgeProblem(
        initial,
        terminal,
        jnp.asarray([0.0, 1.0]),
        _matrix_kernel([[0.75, 0.25], [0.25, 0.75]]),
        CONTEXT,
    )
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)

    assert problem.case_axes == ("case",)
    assert problem.case_shape == (2,)
    assert jnp.all(result.converged)
    assert jnp.allclose(result.marginal_probabilities[:, 0], initial.weights.data)
    assert jnp.allclose(result.marginal_probabilities[:, -1], terminal.weights.data)
    first_alone = phx.transport.dynamic.solve_schrodinger_bridge(
        _problem([0.8, 0.2], [0.6, 0.4], [[0.75, 0.25], [0.25, 0.75]])
    )
    assert jnp.allclose(result.endpoint_coupling[0], first_alone.endpoint_coupling)


def test_physical_mass_mask_and_vector_event_shape_are_preserved():
    states = jnp.asarray([[0.0], [1.0], [2.0]])
    mask = cx.Field(jnp.asarray([True, True, False]), dims=("state",))
    initial = phx.integration.discrete(
        states,
        cx.Field(jnp.asarray([2.0, 0.0, 100.0]), dims=("state",)),
        axes="state",
        mask=mask,
        normalized=False,
        provenance="physical-initial",
    )
    terminal = phx.integration.discrete(
        states,
        cx.Field(jnp.asarray([0.0, 2.0, 100.0]), dims=("state",)),
        axes="state",
        mask=mask,
        normalized=False,
        provenance="physical-terminal",
    )
    matrix = jnp.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    def sample(key, state, _t0, _t1, _context):
        index = jnp.asarray(state[0], dtype=jnp.int32)
        selected = jr.categorical(key, jnp.log(matrix[index]))
        return states[selected]

    def log_prob(next_state, state, _t0, _t1, _context):
        source = jnp.asarray(state[0], dtype=jnp.int32)
        target = jnp.asarray(next_state[0], dtype=jnp.int32)
        probability = matrix[source, target]
        return jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)

    reference = phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(1,),
        process_id="vector-state-reference",
        approximation_id="exact-matrix",
        log_prob_fn=log_prob,
    )
    problem = phx.transport.dynamic.SchrodingerBridgeProblem(
        initial,
        terminal,
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0]), time_id="physical-bridge-grid"
        ),
        reference,
        CONTEXT,
    )
    result = phx.transport.dynamic.solve_schrodinger_bridge(problem)

    assert problem.state_shape == (1,)
    assert problem.time_id == "physical-bridge-grid"
    assert jnp.allclose(problem.mass, 2.0)
    assert jnp.array_equal(problem.initial.mask, jnp.asarray([True, True, False]))
    assert jnp.allclose(result.initial_marginal(), jnp.asarray([2.0, 0.0, 0.0]))
    assert jnp.allclose(result.terminal_marginal(), jnp.asarray([0.0, 2.0, 0.0]))
    assert result.provenance.time_grid == "physical-bridge-grid"


def test_dynamic_transport_public_catalog_is_intentional_and_complete():
    expected = {
        "BridgeInferenceAdapter",
        "BridgePathLawDiagnostics",
        "BridgePathSample",
        "BridgeProblemProvenance",
        "BridgeProvenance",
        "ControlledTransitionKernel",
        "FiniteBridgeTarget",
        "SchrodingerBridgeDiagnostics",
        "SchrodingerBridgeProblem",
        "SchrodingerBridgeResult",
        "SchrodingerBridgeSolver",
        "TerminalDistributionControlAdapter",
        "bridge_path_law_diagnostics",
        "bridge_path_log_prob",
        "reference_path_log_prob",
        "require_converged_bridge",
        "sample_bridge",
        "sample_bridge_paths",
        "sample_bridge_state_indices",
        "solve_schrodinger_bridge",
    }
    assert expected == set(phx.transport.dynamic.__all__)
    assert expected | {"dynamic"} <= set(phx.transport.__all__)
    assert all(vars(phx.transport.dynamic)[name] is not None for name in expected)

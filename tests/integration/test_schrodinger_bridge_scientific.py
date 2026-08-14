import itertools

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _target(weights, provenance):
    return phx.integration.discrete(
        jnp.asarray([0.0, 1.0]),
        cx.Field(jnp.asarray(weights), dims=("state",)),
        axes="state",
        normalized=True,
        provenance=provenance,
    )


def _kernel(matrix):
    matrix = jnp.asarray(matrix)

    def sample(key, state, _t0, _t1, _context):
        probabilities = matrix[jnp.asarray(state, dtype=jnp.int32)]
        return jax.random.categorical(key, jnp.log(probabilities)).astype(float)

    def log_prob(next_state, state, _t0, _t1, _context):
        probability = matrix[
            jnp.asarray(state, dtype=jnp.int32),
            jnp.asarray(next_state, dtype=jnp.int32),
        ]
        return jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)

    return phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(),
        process_id="enumerated-reference",
        approximation_id="exact-matrix",
        log_prob_fn=log_prob,
    )


def _result():
    problem = phx.transport.dynamic.SchrodingerBridgeProblem(
        _target([0.65, 0.35], "inference-prior"),
        _target([0.2, 0.8], "terminal-control-target"),
        jnp.asarray([0.0, 0.4, 1.0]),
        _kernel([[0.8, 0.2], [0.25, 0.75]]),
        phx.stochastic.StateSpaceStepContext.empty(args={"experiment": jnp.asarray(1.0)}),
    )
    return phx.transport.dynamic.SchrodingerBridgeSolver(
        max_iterations=1000, tolerance=1e-12
    ).solve(problem)


def test_exact_path_enumeration_recovers_normalization_endpoints_and_kl():
    result = _result()
    paths = jnp.asarray(list(itertools.product((0.0, 1.0), repeat=3)))
    controlled = jnp.exp(result.path_log_prob(paths))
    reference = jnp.exp(result.reference_path_log_prob(paths))

    assert bool(result.converged)
    assert jnp.allclose(jnp.sum(controlled), 1.0, atol=1e-11)
    assert jnp.allclose(jnp.sum(reference), 1.0, atol=1e-11)
    first = jax.nn.one_hot(paths[:, 0].astype(jnp.int32), 2)
    last = jax.nn.one_hot(paths[:, -1].astype(jnp.int32), 2)
    assert jnp.allclose(
        jnp.sum(controlled[:, None] * first, axis=0),
        jnp.asarray([0.65, 0.35]),
        atol=1e-11,
    )
    assert jnp.allclose(
        jnp.sum(controlled[:, None] * last, axis=0), jnp.asarray([0.2, 0.8]), atol=1e-11
    )
    enumerated_kl = jnp.sum(
        jnp.where(controlled > 0.0, controlled * jnp.log(controlled / reference), 0.0)
    )
    assert jnp.allclose(enumerated_kl, result.diagnostics.path_kl, atol=1e-11)


def test_inference_control_and_path_law_adapters_compose_native_contracts():
    result = _result()
    inference = phx.transport.dynamic.BridgeInferenceAdapter(result)
    control = phx.transport.dynamic.TerminalDistributionControlAdapter(result)
    sample = inference.sample(jr.key(91), sample_shape=(6000,))
    diagnostics = phx.transport.dynamic.bridge_path_law_diagnostics(result, sample)

    assert isinstance(inference.transition, phx.stochastic.AbstractTransitionKernel)
    assert isinstance(inference.initial_prior(), phx.stochastic.CategoricalStatePrior)
    assert inference.transition.has_log_density
    assert jnp.allclose(control.terminal_probabilities, jnp.asarray([0.2, 0.8]))
    assert jnp.allclose(control.path_kl_cost, result.diagnostics.path_kl)
    assert bool(sample.valid.all())
    assert bool(diagnostics.valid)
    assert int(diagnostics.num_samples) == 6000
    assert float(diagnostics.empirical_marginal_residual) < 0.04
    assert jnp.allclose(
        diagnostics.mean_log_likelihood_ratio,
        diagnostics.exact_path_kl,
        atol=4.0 * diagnostics.log_likelihood_ratio_standard_error + 0.01,
    )

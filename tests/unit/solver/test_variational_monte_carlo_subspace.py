#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _BasisState(eqx.Module):
    index: int = eqx.field(static=True)
    phase: complex = eqx.field(static=True)

    def __call__(self, configuration):
        index = jnp.where(configuration[0] > 0, 0, 1)
        log_abs = jnp.where(index == self.index, 0.0, -jnp.inf)
        return phx.operators.LogAmplitude(log_abs, self.phase)


class _TableState(eqx.Module):
    parameters: jax.Array

    def __call__(self, configuration):
        index = jnp.where(configuration[0] > 0, 0, 1)
        return phx.operators.LogAmplitude(self.parameters[index], 1.0 + 0.0j)


def _operator():
    def diagonal(configurations):
        return jnp.where(configurations[..., 0] > 0, 0.0, 2.0)

    def connections(configurations):
        connected = (-configurations)[..., None, :]
        shape = configurations.shape[:-1] + (1,)
        return phx.operators.ConnectedConfigurations(
            connected,
            jnp.zeros(shape),
            jnp.zeros(shape, dtype=bool),
            configuration_shape=(1,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(1,),
        operator_id="two-level-diagonal",
    )


def _kernel():
    proposal = phx.sampling.CallableProposal(
        lambda _key, current: -current,
        lambda _proposed, _current: jnp.asarray(0.0),
        proposal_id="two-level-flip",
    )
    return phx.sampling.MetropolisHastings(proposal)


def _initial_configurations():
    return jnp.asarray([[1], [-1]], dtype=jnp.int32)


def test_subspace_vmc_recovers_exact_complex_phased_basis():
    problem = phx.solver.VariationalMonteCarloSubspaceProblem(
        (_BasisState(0, 1.0 + 0.0j), _BasisState(1, 0.0 + 1.0j)),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    state = problem.initial_state(key=jr.key(1))

    estimate, samples = phx.solver.evaluate_variational_monte_carlo_subspace(
        problem,
        state.models,
        state.markov_state,
        key=jr.key(2),
        num_draws=4,
    )

    assert bool(estimate.successful)
    assert estimate.gram_numerical_rank == 2
    assert estimate.active_samples == 8
    assert jnp.allclose(estimate.state_energies, jnp.asarray([0.0, 2.0]), atol=2e-12)
    assert jnp.allclose(estimate.state_variances, 0.0, atol=2e-12)
    assert jnp.allclose(estimate.overlap_matrix, 0.5 * jnp.eye(2), atol=2e-12)
    assert estimate.overlap_hermiticity_residual == 0.0
    assert estimate.hamiltonian_hermiticity_residual == 0.0
    assert samples.final_state.step_index > state.markov_state.step_index


def test_subspace_vmc_reports_collapsed_state_span():
    problem = phx.solver.VariationalMonteCarloSubspaceProblem(
        (_BasisState(0, 1.0 + 0.0j), _BasisState(0, 0.0 + 1.0j)),
        _operator(),
        _kernel(),
        jnp.asarray([[1]], dtype=jnp.int32),
    )
    state = problem.initial_state(key=jr.key(3))

    estimate, _samples = phx.solver.evaluate_variational_monte_carlo_subspace(
        problem,
        state.models,
        state.markov_state,
        key=jr.key(4),
        num_draws=4,
    )

    assert not bool(estimate.successful)
    assert estimate.status == phx.solver.VMC_SUBSPACE_SINGULAR_SPAN
    assert estimate.gram_numerical_rank == 1


def test_subspace_vmc_zero_iterations_returns_certified_ritz_modes():
    problem = phx.solver.VariationalMonteCarloSubspaceProblem(
        (_BasisState(0, 1.0 + 0.0j), _BasisState(1, 1.0 + 0.0j)),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=0,
        draws_per_iteration=2,
        final_evaluation_draws=4,
        final_chain_diagnostics=False,
    )

    result = phx.solver.solve_variational_monte_carlo_subspace(
        problem,
        policy,
        key=jr.key(5),
    )

    assert bool(result.successful)
    assert result.completed_iterations == 0
    assert result.objective_history.shape == (0,)
    assert jnp.allclose(
        result.final_estimate.state_energies,
        jnp.asarray([0.0, 2.0]),
        atol=2e-12,
    )


def test_subspace_vmc_score_corrected_sr_step_is_finite():
    problem = phx.solver.VariationalMonteCarloSubspaceProblem(
        (
            _TableState(jnp.asarray([0.0, -0.7])),
            _TableState(jnp.asarray([-0.6, 0.0])),
        ),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=1,
        draws_per_iteration=4,
        final_evaluation_draws=4,
        learning_rate=0.01,
        damping=0.1,
        final_chain_diagnostics=False,
    )

    result = phx.solver.solve_variational_monte_carlo_subspace(
        problem,
        policy,
        key=jr.key(6),
    )

    assert bool(result.successful)
    assert result.completed_iterations == 1
    assert result.status_history[0] == phx.solver.VMC_SUBSPACE_SUCCESS
    assert jnp.isfinite(result.update_norm_history[0])
    assert jnp.all(jnp.isfinite(result.final_estimate.state_energies))

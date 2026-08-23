import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _TableModel(eqx.Module):
    parameters: jax.Array

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        value = self.parameters[index]
        if jnp.iscomplexobj(value):
            return phx.operators.LogAmplitude(
                jnp.real(value), jnp.exp(1j * jnp.imag(value))
            )
        return phx.operators.LogAmplitude(value, 1.0 + 0.0j)


def _operator():
    def diagonal(configurations):
        return -configurations[..., 0] * configurations[..., 1]

    def connections(configurations):
        first = configurations.at[..., 0].multiply(-1)
        second = configurations.at[..., 1].multiply(-1)
        connected = jnp.stack((first, second), axis=-2)
        shape = configurations.shape[:-1] + (2,)
        return phx.operators.ConnectedConfigurations(
            connected,
            -0.5 * jnp.ones(shape),
            jnp.ones(shape, dtype=bool),
            configuration_shape=(2,),
        )

    return phx.operators.CallableDiscreteQuantumOperator(
        diagonal,
        connections,
        configuration_shape=(2,),
        operator_id="test-ising",
    )


def _kernel():
    def sample(key, current):
        index = jr.randint(key, (), 0, current.shape[0])
        return current.at[index].multiply(-1)

    def log_prob(_proposed, current):
        return -jnp.log(float(current.shape[0]))

    proposal = phx.sampling.CallableProposal(
        sample,
        log_prob,
        proposal_id="single-spin-flip",
    )
    return phx.sampling.MetropolisHastings(proposal)


def _initial_configurations():
    return jnp.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int32)


def _exact_energy(model):
    state = jnp.exp(model.parameters)
    hamiltonian = jnp.asarray(
        [
            [-1.0, -0.5, -0.5, 0.0],
            [-0.5, 1.0, 0.0, -0.5],
            [-0.5, 0.0, 1.0, -0.5],
            [0.0, -0.5, -0.5, -1.0],
        ]
    )
    return jnp.real(jnp.vdot(state, hamiltonian @ state) / jnp.vdot(state, state))


def test_variational_monte_carlo_runs_persistent_sr_and_improves_energy():
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2])),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=2,
        draws_per_iteration=16,
        steps_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=32,
        learning_rate=0.03,
        damping=0.1,
        max_update_norm=5.0,
    )
    result = phx.solver.solve_variational_monte_carlo(
        problem,
        policy,
        key=jr.key(4),
    )

    assert result.successful
    assert result.completed_iterations == 2
    assert result.energy_history.shape == (2,)
    assert result.update_norm_history.shape == (2,)
    assert jnp.all(result.status_history == phx.solver.VMC_SUCCESS)
    assert _exact_energy(result.final_state.model) < _exact_energy(problem.model)
    assert result.final_state.markov_state.step_index > 0
    diagnostics = result.final_estimate.chain_diagnostics
    assert diagnostics is not None
    assert set(diagnostics.rhat) == {
        "configuration",
        "local_energy_imag",
        "local_energy_real",
    }
    assert diagnostics.mean_acceptance_rate == result.final_estimate.acceptance_rate


def test_vmc_zero_iterations_performs_only_frozen_evaluation():
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(jnp.asarray([0.0, 0.1, -0.1, 0.0])),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=0,
        draws_per_iteration=4,
        final_evaluation_draws=8,
        damping=0.1,
    )
    result = phx.solver.solve_variational_monte_carlo(
        problem,
        policy,
        key=jr.key(8),
    )

    assert result.completed_iterations == 0
    assert result.energy_history.shape == (0,)
    assert result.linear_results == ()
    assert result.final_estimate.successful
    assert result.final_estimate.chain_diagnostics is not None


def test_vmc_complex_parameter_modes_are_explicit():
    real_model = _TableModel(jnp.zeros((4,)))
    complex_model = _TableModel(jnp.zeros((4,), dtype=complex))

    with pytest.raises(TypeError, match="holomorphic"):
        phx.solver.VariationalMonteCarloProblem(
            real_model,
            _operator(),
            _kernel(),
            _initial_configurations(),
            complex_parameter_mode="holomorphic",
        )
    with pytest.raises(TypeError, match="real parameter mode"):
        phx.solver.VariationalMonteCarloProblem(
            complex_model,
            _operator(),
            _kernel(),
            _initial_configurations(),
            complex_parameter_mode="real",
        )

    for mode in ("holomorphic", "nonholomorphic"):
        problem = phx.solver.VariationalMonteCarloProblem(
            complex_model,
            _operator(),
            _kernel(),
            _initial_configurations(),
            complex_parameter_mode=mode,
        )
        result = phx.solver.solve_variational_monte_carlo(
            problem,
            phx.solver.VariationalMonteCarloPolicy(
                num_iterations=1,
                draws_per_iteration=8,
                final_evaluation_draws=8,
                damping=0.2,
                learning_rate=0.01,
            ),
            key=jr.key(12 if mode == "holomorphic" else 13),
        )
        assert result.successful
        assert jnp.all(jnp.isfinite(result.final_state.parameter_coordinates))


def test_vmc_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2])),
        _operator(),
        _kernel(),
        _initial_configurations(),
    )
    one_step = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=1,
        draws_per_iteration=12,
        steps_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=8,
        learning_rate=0.03,
        damping=0.1,
    )
    two_steps = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=2,
        draws_per_iteration=12,
        steps_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=8,
        learning_rate=0.03,
        damping=0.1,
    )
    key = jr.key(21)
    direct = phx.solver.solve_variational_monte_carlo(problem, two_steps, key=key)
    first = phx.solver.solve_variational_monte_carlo(problem, one_step, key=key)
    checkpoint = tmp_path / "vmc-state.zip"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint, problem, one_step, first.final_state
    )
    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint, problem, one_step
    )
    resumed = phx.solver.solve_variational_monte_carlo(problem, one_step, state=restored)

    assert resumed.completed_iterations == 2
    assert jnp.array_equal(
        resumed.final_state.parameter_coordinates,
        direct.final_state.parameter_coordinates,
    )
    assert jnp.array_equal(
        resumed.final_state.markov_state.position,
        direct.final_state.markov_state.position,
    )
    assert jnp.array_equal(
        resumed.final_state.markov_state.log_target,
        direct.final_state.markov_state.log_target,
    )
    assert (
        resumed.final_state.markov_state.step_index
        == direct.final_state.markov_state.step_index
    )
    assert jnp.array_equal(jr.key_data(resumed.final_state.root_key), jr.key_data(key))

    incompatible = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=1,
        draws_per_iteration=12,
        steps_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=8,
        learning_rate=0.03,
        damping=0.2,
    )
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.solver.read_variational_monte_carlo_checkpoint(
            checkpoint, problem, incompatible
        )
    with pytest.raises(ValueError, match="Resume key"):
        phx.solver.solve_variational_monte_carlo(
            problem,
            one_step,
            state=restored,
            key=jr.key(22),
        )

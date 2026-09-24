from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _TableModel(eqx.Module):
    parameters: jax.Array = phx.parameter_field()

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        value = self.parameters[index]
        if jnp.iscomplexobj(value):
            return phx.operators.LogAmplitude(
                jnp.real(value), jnp.exp(1j * jnp.imag(value))
            )
        return phx.operators.LogAmplitude(value, 1.0 + 0.0j)


class _StaticTableModel(eqx.Module):
    parameters: jax.Array = phx.parameter_field()
    offset: float = eqx.field(static=True)

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        return phx.operators.LogAmplitude(
            self.parameters[index] + self.offset,
            1.0 + 0.0j,
        )


class _ActivatedTableModel(eqx.Module):
    parameters: jax.Array = phx.parameter_field()
    activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        return phx.operators.LogAmplitude(
            self.activation(self.parameters[index]),
            1.0 + 0.0j,
        )


def _identity_activation(value):
    return value


def _halved_activation(value):
    return 0.5 * value


def _scaled_activation(scale):
    def activation(value):
        return scale * value

    return activation


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
            jnp.ones(shape, dtype="bool"),
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


def _table_log_target(model, configuration):
    return 2.0 * model(configuration).log_abs


def _full_target_factory(model):
    return phx.sampling.FullMarkovTarget(
        lambda configuration: _table_log_target(model, configuration),
        target_id="table-density",
    )


def _incremental_target_factory(model):
    def initialize(configuration):
        value = _table_log_target(model, configuration)
        return value, value

    def propose(_current, cached, proposed, _payload):
        value = _table_log_target(model, proposed)
        return value - cached, value, jnp.asarray(True)

    def select(current, proposed, accepted):
        return jnp.where(accepted, proposed, current)

    return phx.sampling.IncrementalMarkovTarget(
        initialize=initialize,
        propose=propose,
        select=select,
        refresh=initialize,
        target_id="table-density",
        refresh_cadence=3,
        cache_tolerance=1e-6,
    )


def _drifting_incremental_target_factory(model):
    def initialize(configuration):
        value = _table_log_target(model, configuration)
        return value, value

    def propose(_current, cached, proposed, _payload):
        value = _table_log_target(model, proposed)
        return value - cached, value, jnp.asarray(True)

    def select(current, proposed, accepted):
        return jnp.where(accepted, proposed, current)

    def refresh(configuration):
        value = _table_log_target(model, configuration)
        return value, value + 1.0

    return phx.sampling.IncrementalMarkovTarget(
        initialize=initialize,
        propose=propose,
        select=select,
        refresh=refresh,
        target_id="drifting-table-density",
        refresh_cadence=1,
        cache_tolerance=0.0,
    )


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


@pytest.mark.parametrize(
    ("target_factory", "target_factory_id"),
    (
        (_full_target_factory, "table-full"),
        (_incremental_target_factory, "table-incremental"),
    ),
)
def test_vmc_rebinds_full_and_incremental_targets_for_each_frozen_model(
    target_factory,
    target_factory_id,
):
    model = _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2]))
    problem = phx.solver.VariationalMonteCarloProblem(
        model,
        _operator(),
        _kernel(),
        _initial_configurations(),
        target_factory=target_factory,
        target_factory_id=target_factory_id,
    )
    state = problem.initial_state(key=jr.key(17))
    changed_model = _TableModel(model.parameters + 0.4)

    estimate, samples = phx.solver.evaluate_variational_monte_carlo(
        problem,
        changed_model,
        state.markov_state,
        key=jr.key(18),
        num_draws=6,
    )
    expected = jax.vmap(
        lambda configuration: _table_log_target(changed_model, configuration)
    )(samples.final_state.position)

    assert estimate.successful
    assert samples.final_state.target_id == "table-density"
    assert jnp.allclose(samples.final_state.log_target, expected)
    if target_factory is _incremental_target_factory:
        assert jnp.allclose(samples.final_state.cache, expected)


def test_vmc_rejects_estimates_from_a_tainted_incremental_chain():
    model = _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2]))
    problem = phx.solver.VariationalMonteCarloProblem(
        model,
        _operator(),
        _kernel(),
        _initial_configurations(),
        target_factory=_drifting_incremental_target_factory,
        target_factory_id="drifting-table",
    )
    state = problem.initial_state(key=jr.key(51))

    estimate, samples = phx.solver.evaluate_variational_monte_carlo(
        problem,
        model,
        state.markov_state,
        key=jr.key(52),
        num_draws=2,
    )

    assert not jnp.all(samples.final_state.valid)
    assert estimate.status == phx.solver.VMC_INVALID_SAMPLES
    assert not estimate.successful


def test_vmc_target_factory_identity_is_explicit_and_stable():
    model = _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2]))
    arguments = (model, _operator(), _kernel(), _initial_configurations())

    default = phx.solver.VariationalMonteCarloProblem(*arguments)
    assert default.target_factory is None
    assert default.target_factory_id is None
    assert default.initial_state().markov_state.target_id == default.target_id
    assert isinstance(default.target_for_model(model), phx.sampling.FullMarkovTarget)

    with pytest.raises(ValueError, match="target_factory_id"):
        phx.solver.VariationalMonteCarloProblem(
            *arguments,
            target_factory=_full_target_factory,
        )
    with pytest.raises(ValueError, match="requires target_factory"):
        phx.solver.VariationalMonteCarloProblem(
            *arguments,
            target_factory_id="unused-factory",
        )
    with pytest.raises(TypeError, match="must return"):
        phx.solver.VariationalMonteCarloProblem(
            *arguments,
            target_factory=lambda _model: lambda _configuration: 0.0,
            target_factory_id="malformed-factory",
        )

    target_ids = iter(("first-target", "second-target"))

    def unstable_factory(frozen_model):
        return phx.sampling.FullMarkovTarget(
            lambda configuration: _table_log_target(frozen_model, configuration),
            target_id=next(target_ids),
        )

    unstable = phx.solver.VariationalMonteCarloProblem(
        *arguments,
        target_factory=unstable_factory,
        target_factory_id="unstable-factory",
    )
    with pytest.raises(ValueError, match="preserve its target identity"):
        unstable.target_for_model(model)


def test_vmc_complex_parameter_modes_are_explicit():
    real_model = _TableModel(jnp.zeros((4,)))
    complex_model = _TableModel(jnp.zeros((4,), dtype="complex128"))

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


def test_vmc_checkpoint_rejects_changed_static_model_configuration(tmp_path):
    parameters = jnp.asarray([0.2, -0.1, 0.1, -0.2])
    common = (_operator(), _kernel(), _initial_configurations())
    original = phx.solver.VariationalMonteCarloProblem(
        _StaticTableModel(parameters, 0.0),
        *common,
        problem_id="static-model-checkpoint",
    )
    changed = phx.solver.VariationalMonteCarloProblem(
        _StaticTableModel(parameters, 0.5),
        *common,
        problem_id="static-model-checkpoint",
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=0,
        draws_per_iteration=2,
        final_evaluation_draws=2,
        final_chain_diagnostics=False,
    )
    checkpoint = tmp_path / "static-model-vmc.zip"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint,
        original,
        policy,
        original.initial_state(key=jr.key(30)),
    )
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.solver.read_variational_monte_carlo_checkpoint(
            checkpoint,
            changed,
            policy,
        )


def test_vmc_checkpoint_identifies_model_callables_by_content(tmp_path):
    parameters = jnp.asarray([0.2, -0.1, 0.1, -0.2])
    common = (_operator(), _kernel(), _initial_configurations())
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=0,
        draws_per_iteration=2,
        final_evaluation_draws=2,
        final_chain_diagnostics=False,
    )

    def problem(activation):
        return phx.solver.VariationalMonteCarloProblem(
            _ActivatedTableModel(parameters, activation),
            *common,
            problem_id="callable-model-checkpoint",
        )

    original = problem(_identity_activation)
    checkpoint = tmp_path / "callable-model-vmc.zip"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint, original, policy, original.initial_state(key=jr.key(31))
    )
    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint, problem(_identity_activation), policy
    )
    assert int(restored.iteration) == 0
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.solver.read_variational_monte_carlo_checkpoint(
            checkpoint, problem(_halved_activation), policy
        )

    for opaque in (_scaled_activation(1.0), lambda value: value):
        opaque_problem = problem(opaque)
        with pytest.raises(TypeError, match="Opaque callables"):
            phx.solver.write_variational_monte_carlo_checkpoint(
                tmp_path / "opaque-model-vmc.zip",
                opaque_problem,
                policy,
                opaque_problem.initial_state(key=jr.key(31)),
            )
        with pytest.raises(TypeError, match="Opaque callables"):
            phx.solver.read_variational_monte_carlo_checkpoint(
                checkpoint, opaque_problem, policy
            )


def test_incremental_vmc_checkpoint_rebuilds_cache_and_resumes_exactly(tmp_path):
    model = _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2]))
    problem = phx.solver.VariationalMonteCarloProblem(
        model,
        _operator(),
        _kernel(),
        _initial_configurations(),
        target_factory=_incremental_target_factory,
        target_factory_id="table-incremental",
        problem_id="incremental-checkpoint",
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=1,
        draws_per_iteration=8,
        steps_per_draw=2,
        warmup_steps=2,
        final_evaluation_draws=4,
        learning_rate=0.03,
        damping=0.1,
        final_chain_diagnostics=False,
    )
    key = jr.key(33)
    first = phx.solver.solve_variational_monte_carlo(problem, policy, key=key)
    checkpoint = tmp_path / "incremental-vmc-state.zip"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint,
        problem,
        policy,
        first.final_state,
    )

    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint,
        problem,
        policy,
    )
    expected = jax.vmap(
        lambda configuration: _table_log_target(restored.model, configuration)
    )(restored.markov_state.position)

    assert jnp.allclose(restored.markov_state.log_target, expected)
    assert jnp.allclose(restored.markov_state.cache, expected)
    assert restored.markov_state.step_index == first.final_state.markov_state.step_index
    assert jnp.array_equal(
        jr.key_data(restored.root_key),
        jr.key_data(first.final_state.root_key),
    )

    uninterrupted = phx.solver.solve_variational_monte_carlo(
        problem,
        policy,
        state=first.final_state,
    )
    resumed = phx.solver.solve_variational_monte_carlo(
        problem,
        policy,
        state=restored,
    )
    assert jnp.array_equal(
        resumed.final_state.parameter_coordinates,
        uninterrupted.final_state.parameter_coordinates,
    )
    assert jnp.array_equal(
        resumed.final_state.markov_state.position,
        uninterrupted.final_state.markov_state.position,
    )
    assert jnp.array_equal(
        resumed.final_state.markov_state.cache,
        uninterrupted.final_state.markov_state.cache,
    )
    assert (
        resumed.final_state.markov_state.step_index
        == uninterrupted.final_state.markov_state.step_index
    )

    incompatible = phx.solver.VariationalMonteCarloProblem(
        model,
        _operator(),
        _kernel(),
        _initial_configurations(),
        target_factory=_incremental_target_factory,
        target_factory_id="table-incremental-changed",
        problem_id="incremental-checkpoint",
    )
    with pytest.raises(phx.uq.CheckpointCompatibilityError):
        phx.solver.read_variational_monte_carlo_checkpoint(
            checkpoint,
            incompatible,
            policy,
        )


def test_incremental_vmc_checkpoint_preserves_chain_taint(tmp_path):
    problem = phx.solver.VariationalMonteCarloProblem(
        _TableModel(jnp.asarray([0.2, -0.1, 0.1, -0.2])),
        _operator(),
        _kernel(),
        _initial_configurations(),
        target_factory=_incremental_target_factory,
        target_factory_id="table-incremental",
        problem_id="tainted-incremental-checkpoint",
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=0,
        draws_per_iteration=2,
        final_evaluation_draws=2,
        final_chain_diagnostics=False,
    )
    state = problem.initial_state(key=jr.key(61))
    declared_valid = jnp.asarray([False, True, False, True])
    tainted = eqx.tree_at(
        lambda value: value.markov_state.valid,
        state,
        declared_valid,
    )
    checkpoint = tmp_path / "tainted-vmc-state.zip"

    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint,
        problem,
        policy,
        tainted,
    )
    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint,
        problem,
        policy,
    )

    assert jnp.array_equal(restored.markov_state.valid, declared_valid)
    estimate, samples = phx.solver.evaluate_variational_monte_carlo(
        problem,
        restored.model,
        restored.markov_state,
        key=jr.key(62),
        num_draws=2,
    )
    assert jnp.array_equal(samples.final_state.valid, declared_valid)
    assert estimate.status == phx.solver.VMC_INVALID_SAMPLES

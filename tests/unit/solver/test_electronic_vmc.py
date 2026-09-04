import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _Hydrogenic(eqx.Module):
    alpha: jax.Array

    def __call__(self, electrons):
        radius = jnp.sqrt(jnp.sum(electrons[0] ** 2))
        return phx.operators.LogAmplitude(-self.alpha * radius)


class _TwoScaleHydrogenic(eqx.Module):
    alpha: jax.Array

    def __call__(self, electrons):
        radius = jnp.sqrt(jnp.sum(electrons[0] ** 2))
        return phx.operators.LogAmplitude(
            -self.alpha[0] * radius - self.alpha[1] * radius**2
        )


class _FailedLocalOperator(phx.operators.AbstractLocalQuantumOperator):
    configuration_shape: tuple[int, int] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(self):
        self.configuration_shape = (1, 3)
        self.operator_id = "failed-electronic-local"

    def estimate(self, model, configurations, /):
        del model
        shape = configurations.shape[:-2]
        return phx.operators.LocalOperatorEstimate(
            jnp.zeros(shape),
            jnp.ones(shape, dtype=bool),
            jnp.full(
                shape,
                int(phx.operators.LocalOperatorStatus.SINGULAR_CONFIGURATION),
                dtype=jnp.int32,
            ),
            jnp.zeros(shape, dtype=jnp.int32),
            configuration_shape=self.configuration_shape,
            operator_id=self.operator_id,
            method_id="test-failed-local",
            compute_dtype=str(configurations.dtype),
        )


def _atom(charges, positions, *, name):
    scale = phx.atomistic.AtomisticScaleContract(phx.units.BOHR, phx.units.HARTREE)
    return phx.atomistic.AtomicStructure(
        jnp.asarray(charges, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.ones((len(charges),), dtype=jnp.float64),
        scale,
        name=name,
    )


def _hydrogen_problem(*, alpha=0.8, chains=8):
    atom = _atom([1], [[0.0, 0.0, 0.0]], name="H")
    model = _Hydrogenic(jnp.asarray(alpha, dtype=jnp.float64))
    operator = phx.operators.ElectronicCoulombHamiltonian(atom, 1)
    proposal = phx.operators.harmonic_mean_electron_proposal(atom, 1, step_size=0.25)
    walkers = phx.operators.electronic_initial_walkers(jr.key(19), atom, 1, chains)
    return phx.solver.VariationalMonteCarloProblem(
        model,
        operator,
        phx.sampling.MetropolisHastings(proposal),
        walkers,
        problem_id="electronic-hydrogen",
    )


def _policy(iterations):
    return phx.solver.VariationalMonteCarloPolicy(
        num_iterations=iterations,
        draws_per_iteration=8,
        steps_per_draw=2,
        warmup_steps=4,
        final_evaluation_draws=8,
        learning_rate=0.05,
        damping=1e-3,
        max_update_norm=1.0,
        final_chain_diagnostics=True,
        failure_mode="record",
    )


def test_hydrogen_vmc_replay_persistence_diagnostics_and_training():
    problem = _hydrogen_problem()
    policy = _policy(2)
    first = phx.solver.solve_variational_monte_carlo(problem, policy, key=jr.key(123))
    replay = phx.solver.solve_variational_monte_carlo(problem, policy, key=jr.key(123))
    assert jnp.array_equal(first.energy_history, replay.energy_history)
    assert jnp.array_equal(
        first.final_state.parameter_coordinates,
        replay.final_state.parameter_coordinates,
    )
    assert first.final_estimate.chain_diagnostics is not None
    assert first.final_estimate.local.operator_id == problem.operator.operator_id
    assert first.final_estimate.local.configuration_shape == (1, 3)
    assert first.final_estimate.local.work_count.shape == (8, 8)
    assert first.completed_iterations == int(first.final_state.iteration)
    assert jnp.abs(first.final_state.model.alpha - 1.0) < jnp.abs(
        problem.model.alpha - 1.0
    )


def test_electronic_vmc_checkpoint_restart_matches_persistent_continuation(tmp_path):
    problem = _hydrogen_problem()
    one_step = _policy(1)
    uninterrupted = phx.solver.solve_variational_monte_carlo(
        problem, _policy(2), key=jr.key(77)
    )
    prefix = phx.solver.solve_variational_monte_carlo(problem, one_step, key=jr.key(77))
    checkpoint = tmp_path / "electronic-vmc.npz"
    phx.solver.write_variational_monte_carlo_checkpoint(
        checkpoint, problem, one_step, prefix.final_state
    )
    restored = phx.solver.read_variational_monte_carlo_checkpoint(
        checkpoint, problem, one_step
    )
    resumed = phx.solver.solve_variational_monte_carlo(problem, one_step, state=restored)
    assert resumed.completed_iterations == 2
    assert jnp.array_equal(
        resumed.final_state.markov_state.position,
        uninterrupted.final_state.markov_state.position,
    )
    assert jnp.allclose(
        resumed.final_state.parameter_coordinates,
        uninterrupted.final_state.parameter_coordinates,
    )


def test_small_helium_and_hydrogen_molecule_ferminet_vmc_smoke():
    cases = (
        (_atom([2], [[0.0, 0.0, 0.0]], name="He"), 2, 1),
        (
            _atom([1, 1], [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], name="H2"),
            2,
            1,
        ),
    )
    for case_index, (nuclei, electrons, spin_up) in enumerate(cases):
        model = phx.nn.quantum.FermiNet(
            nuclei,
            electrons,
            spin_up,
            hidden_features=8,
            pair_features=4,
            layer_count=1,
            determinant_count=2,
            key=jr.key(case_index),
        )
        problem = phx.solver.VariationalMonteCarloProblem(
            model,
            phx.operators.ElectronicCoulombHamiltonian(
                nuclei,
                electrons,
                kinetic=phx.operators.ElectronicKineticPolicy(
                    trace_method="chunked-exact", coordinate_chunk_size=3
                ),
            ),
            phx.sampling.MetropolisHastings(
                phx.operators.harmonic_mean_electron_proposal(
                    nuclei, electrons, step_size=0.2
                )
            ),
            phx.operators.electronic_initial_walkers(
                jr.fold_in(jr.key(91), case_index), nuclei, electrons, 2
            ),
            problem_id=f"electronic-smoke-{case_index}",
        )
        result = phx.solver.solve_variational_monte_carlo(
            problem,
            phx.solver.VariationalMonteCarloPolicy(
                num_iterations=0,
                draws_per_iteration=1,
                final_evaluation_draws=1,
                final_chain_diagnostics=False,
                failure_mode="record",
            ),
            key=jr.fold_in(jr.key(33), case_index),
        )
        assert result.final_estimate.local.value.shape == (2, 1)
        assert result.final_estimate.local.method_id.startswith("electronic-kinetic")


def test_failed_local_and_linear_actions_record_without_applying_updates():
    baseline = _hydrogen_problem(chains=4)
    failed_local = phx.solver.VariationalMonteCarloProblem(
        baseline.model,
        _FailedLocalOperator(),
        baseline.kernel,
        baseline.initial_configurations,
        problem_id="failed-local-electronic-vmc",
    )
    local_result = phx.solver.solve_variational_monte_carlo(
        failed_local, _policy(1), key=jr.key(501)
    )
    assert jnp.all(local_result.final_estimate.local.valid)
    assert not jnp.any(local_result.final_estimate.local.successful)
    assert local_result.final_estimate.active_samples == 0
    assert local_result.status_history[0] == phx.solver.VMC_INVALID_SAMPLES
    assert local_result.completed_iterations == 0
    assert local_result.linear_results == ()

    failed_linear = phx.solver.VariationalMonteCarloProblem(
        _TwoScaleHydrogenic(jnp.asarray([0.8, 0.01], dtype=jnp.float64)),
        baseline.operator,
        baseline.kernel,
        baseline.initial_configurations,
        problem_id="limited-linear-electronic-vmc",
    )
    linear_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0,
            absolute=0.0,
            max_steps=1,
        ),
    )
    policy = phx.solver.VariationalMonteCarloPolicy(
        num_iterations=1,
        draws_per_iteration=4,
        final_evaluation_draws=4,
        damping=1e-3,
        failure_mode="record",
        final_chain_diagnostics=False,
        linear_policy=linear_policy,
    )
    linear_result = phx.solver.solve_variational_monte_carlo(
        failed_linear, policy, key=jr.key(502)
    )
    assert linear_result.status_history[0] == phx.solver.VMC_LINEAR_FAILURE
    assert linear_result.completed_iterations == 0
    assert linear_result.linear_results

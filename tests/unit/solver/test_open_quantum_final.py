#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from tools.open_system_campaigns import (
    gaussian_campaign,
    read_open_system_artifact,
    write_open_system_artifact,
)


def test_approximation_evidence_requires_quantified_axes():
    with pytest.raises(ValueError):
        phx.operators.quantum.OpenSystemApproximationEvidence(
            "empty", (), (), execution_valid=True
        )
    quantity = phx.operators.quantum.ApproximationQuantity(
        "error",
        0.01,
        0.1,
        units="dimensionless",
        norm_id="absolute",
        estimate_kind="bound",
    )
    evidence = phx.operators.quantum.OpenSystemApproximationEvidence(
        "bounded",
        (phx.operators.quantum.ApproximationAxis("cutoff", 4),),
        (quantity,),
        execution_valid=True,
    )
    assert bool(evidence.valid)


def test_promotion_policy_is_fail_closed_for_unknown_physicality():
    quantity = phx.operators.quantum.ApproximationQuantity(
        "error",
        0.01,
        0.1,
        units="dimensionless",
        norm_id="absolute",
        estimate_kind="estimate",
    )
    approximation = phx.operators.quantum.OpenSystemApproximationEvidence(
        "representation",
        (phx.operators.quantum.ApproximationAxis("cutoff", 4),),
        (quantity,),
        execution_valid=True,
    )
    policy = phx.operators.quantum.OpenSystemPromotionPolicy(
        ("cutoff",),
        ("error",),
        ("trace",),
        policy_id="fail-closed",
    )
    decision = phx.operators.quantum.evaluate_open_system_promotion(
        policy,
        approximation,
        phx.operators.quantum.OpenSystemPhysicalityEvidence(),
        execution_success=True,
        capacity_exhausted=False,
        archive_verified=True,
    )
    assert not bool(decision.promoted)


def test_pseudomode_preserves_pure_initial_state_exactly():
    _, mode, _ = phx.operators.quantum.lorentzian_pseudomode(1.0, 0.5, 0.2, cutoff=3)
    initial = jnp.asarray([[1.0 + 0j, 0j], [0j, 0j]])
    problem = phx.solver.jaynes_cummings_pseudomode_problem(mode, initial)
    reduced = problem.reduced_system_density(problem.lindblad_problem.initial_density)
    assert jnp.allclose(reduced, initial)


def test_memory_execution_and_physicality_are_distinct():
    initial = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    result = phx.solver.solve_memory_kernel(
        phx.solver.exponential_memory_qubit_problem(0.01, 1.0, initial),
        step_size=0.001,
        steps=1,
    )
    assert bool(result.execution_valid)
    assert bool(result.pointwise_density_valid)
    assert not bool(result.production_valid)
    assert result.physicality.status == "unknown"


def test_process_physicality_rejects_negative_initial_state():
    initial = jnp.asarray([[2.0 + 0j, 0j], [0j, -1.0 + 0j]])
    process = phx.tensor_network.markov_process_tensor(
        (jnp.eye(4, dtype=complex),), initial
    )
    assert not bool(process.physicality().valid)


def test_fixed_step_jump_probability_guard():
    problem = phx.solver.amplitude_damping_trajectory_problem(
        100.0, jnp.asarray([0j, 1.0 + 0j])
    )
    with pytest.raises(Exception):
        phx.solver.solve_quantum_jump_ensemble(
            problem,
            jax.random.PRNGKey(0),
            step_size=0.1,
            steps=1,
            trajectory_count=1,
        )


def test_gaussian_hbar_and_generator_conventions():
    state = phx.metrix.BosonicGaussianState(jnp.zeros(2), jnp.eye(2), hbar=2.0)
    channel = phx.metrix.BosonicGaussianChannel(
        jnp.eye(2),
        jnp.zeros((2, 2)),
        jnp.zeros(2),
        channel_id="identity-hbar-2",
        hbar=2.0,
    )
    assert bool(channel.apply(state).valid)
    with pytest.raises(ValueError):
        channel.apply(phx.metrix.BosonicGaussianState(jnp.zeros(2), 0.5 * jnp.eye(2)))
    problem = phx.solver.GaussianLindbladProblem(
        -0.5 * jnp.eye(2),
        jnp.eye(2),
        jnp.zeros(2),
        state,
    )
    assert problem.generator_cp_margin >= -1e-9


def test_open_system_artifact_roundtrip_and_identity(tmp_path):
    path = tmp_path / "artifact.zip"
    record = gaussian_campaign()
    write_open_system_artifact(
        path,
        record,
        problem_id=record.campaign_id,
        plan_id="artifact-test-plan",
        backend="cpu",
        runner_id="artifact-test-runner",
        code_fingerprint="artifact-test-code",
    )
    stored, manifest = read_open_system_artifact(
        path,
        expected_campaign_id=record.campaign_id,
        expected_representation_id=record.representation_id,
    )
    assert manifest["record"]["campaign_id"] == record.campaign_id
    assert stored.artifact_names == record.artifact_names
    with pytest.raises(ValueError):
        read_open_system_artifact(
            path, expected_campaign_id="wrong"
        )


def test_generic_quantum_jump_adapter():
    problem = phx.solver.amplitude_damping_trajectory_problem(
        1.0, jnp.asarray([0j, 1.0 + 0j])
    )
    solution = phx.solver.solve_quantum_jump_generic(
        problem,
        jax.random.PRNGKey(3),
        t0=0.0,
        t1=0.2,
        save_times=jnp.asarray([0.0, 0.1, 0.2]),
        trajectory_count=2,
        maximum_events_per_channel=4,
        dt0=0.01,
    )
    assert jnp.all(solution.valid)


def test_local_lindblad_channel_and_purified_certificate():
    lowering = jnp.asarray([[0, 1], [0, 0]], dtype=complex)
    prepared = phx.tensor_network.prepare_local_lindblad_channel(
        jnp.zeros((2, 2), dtype=complex),
        lowering[None, ...],
        0.01,
    )
    assert bool(prepared.evidence.valid)
    problem = phx.solver.boundary_driven_xxz_problem(2, half_step=0.005)
    result = phx.solver.solve_purified_strang(
        problem,
        step_size=0.01,
        steps=1,
        maximum_bond_dimension=4,
        maximum_purification_dimension=8,
    )
    certificate = phx.solver.diagnose_purified_stationarity(
        result,
        jnp.asarray([[0.0], [0.0]]),
        window=1,
        tolerance=1.0,
        truncation_tolerance=1.0,
    )
    assert bool(certificate.valid)


def test_heom_bdf_grid_and_process_identifiability():
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    expansion = phx.operators.quantum.drude_lorentz_matsubara(0.01, 1.0, 2.0, 1)
    problem = phx.solver.HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion,
        phx.solver.HEOMHierarchy(1, 1),
        density,
    )
    bdf = phx.solver.solve_heom_bdf(problem, step_size=0.001, steps=1, maximum_order=2)
    assert bool(bdf.valid)
    grid = phx.solver.solve_heom_continuation_grid(
        problem,
        (expansion,),
        (1,),
        step_size=0.001,
        steps=1,
    )
    assert bool(grid.valid)


def test_neural_rate_evidence_blocks_uncertain_rates():
    evidence = phx.solver.NeuralRateEvidence(
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([2.0]),
        relative_error_tolerance=0.1,
    )
    assert not bool(evidence.valid)


def test_complex_stiefel_and_sequential_process_tomography():
    manifold = phx.metrix.ComplexStiefelManifold(2, 2)
    isometry = jnp.eye(2, dtype=complex)
    assert bool(manifold.contains(isometry))
    retracted = manifold.retract(isometry, 0.01j * jnp.asarray([[0.0, 1.0], [1.0, 0.0]]))
    assert bool(manifold.contains(retracted))
    spec = phx.tensor_network.CombLegSpec(2, 1, 1)
    model = phx.tensor_network.SequentialStinespringProcess(
        spec,
        jnp.eye(2, dtype=complex),
        (isometry,),
        (1,),
        process_id="identity-stinespring",
    )
    instrument = phx.tensor_network.QuantumInstrument(
        jnp.eye(2, dtype=complex)[None, None, ...],
        jnp.asarray([True]),
        jnp.asarray([[True]]),
        instrument_id="identity",
    )
    experiment = phx.solver.ProcessTomographyExperiment(
        (instrument,), (0,), 10.0, experiment_id="identity-setting"
    )
    result = phx.solver.fit_stinespring_process(
        phx.solver.StinespringTomographyProblem(model, (experiment,)),
        iterations=1,
    )
    assert bool(result.execution_valid)
    assert not bool(result.valid)
    assert bool(result.underidentified)
    source = phx.tensor_network.causal_process_from_lindblad(
        jnp.zeros((2, 2), dtype=complex),
        jnp.zeros((1, 2, 2), dtype=complex),
        0.5 * jnp.eye(2, dtype=complex),
        step_size=0.01,
        slot_count=1,
    )
    assert bool(source.valid)

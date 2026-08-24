#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
import tools.open_system_campaigns as campaigns


def test_artifact_is_exactly_reconstructed_and_verified(tmp_path):
    record = campaigns.gaussian_campaign()
    path = tmp_path / "campaign.zip"
    campaigns.write_open_system_artifact(
        path,
        record,
        problem_id=record.campaign_id,
        plan_id="gaussian-test",
        backend="cpu",
        runner_id="gaussian-test-runner",
        code_fingerprint="gaussian-test-code",
    )
    stored, manifest = campaigns.read_open_system_artifact(
        path,
        expected_campaign_id=record.campaign_id,
        expected_representation_id=record.representation_id,
        expected_runner_id="gaussian-test-runner",
    )
    verified = campaigns.verify_open_system_artifact(
        path,
        campaigns.gaussian_campaign(),
        expected_runner_id="gaussian-test-runner",
    )
    assert manifest["record"]["campaign_id"] == record.campaign_id
    assert stored.artifact_names == record.artifact_names
    assert bool(verified.valid)


def test_evidence_contracts_reject_malformed_values():
    with pytest.raises(ValueError, match="non-negative"):
        phx.operators.quantum.ApproximationQuantity(
            "negative-error",
            -1.0,
            0.1,
            units="error",
            norm_id="absolute",
            estimate_kind="estimate",
        )
    with pytest.raises(ValueError, match="confidence"):
        phx.operators.quantum.ApproximationQuantity(
            "statistical",
            0.01,
            0.1,
            units="error",
            norm_id="absolute",
            estimate_kind="statistical",
        )
    with pytest.raises(ValueError, match="Semantic replay"):
        campaigns.SemanticReplayEvidence(
            variates_equal=True,
            address_schema_equal=True,
            event_time_difference=-1.0,
            channel_disagreement_probability=0.0,
            observable_difference=0.0,
            event_time_tolerance=1.0,
            disagreement_tolerance=1.0,
            observable_tolerance=1.0,
        )


def test_promotion_requires_named_physicality_and_verified_archive():
    record = campaigns.gaussian_campaign()
    policy = phx.operators.quantum.OpenSystemPromotionPolicy(
        ("time-step",),
        ("analytic-covariance-error",),
        ("representation-closure",),
        policy_id="gaussian-test-policy",
    )
    unverified = record.evaluate(policy, archive_verified=False)
    verified = campaigns.VerifiedOpenSystemCampaign(
        record,
        "0" * 64,
        reproduction_verified=True,
    ).evaluate(policy)
    assert not bool(unverified.promoted)
    assert bool(verified.promoted)
    assert not verified.missing_physicality


def test_connected_vmc_campaign_separates_projection_audit_from_stochastic_jump():
    record = campaigns.neural_campaign()
    arrays = dict(zip(record.artifact_names, record.artifact_arrays, strict=True))
    assert bool(record.execution_success)
    assert record.representation_id == "connected-vmc-neural-trajectory"
    assert bool(jnp.any(arrays["jump-decisions"]))
    assert bool(arrays["projected-jump-observed"])
    assert jnp.isfinite(arrays["audit-projection-residual"])
    assert "forced-first-projected-jump" not in arrays


def test_intervention_complete_process_design_identifies_physical_quotient():
    spec = phx.tensor_network.CombLegSpec(2, 1, 1)
    model = phx.tensor_network.SequentialStinespringProcess(
        spec,
        jnp.eye(2, dtype=complex),
        (jnp.eye(2, dtype=complex),),
        (1,),
        process_id="test-intervention-complete",
    )
    experiments = phx.solver.informationally_complete_process_experiments(
        model.materialize(), shots=100.0
    )
    result = phx.solver.fit_stinespring_process(
        phx.solver.StinespringTomographyProblem(model, experiments),
        iterations=1,
        learning_rate=1e-4,
    )
    assert len(experiments) == 64
    assert bool(result.quotient_identified)
    assert bool(result.valid)
    assert result.singular_values.ndim == 1


def test_mps_campaign_exercises_event_root_and_capacity_evidence():
    record = campaigns.mps_campaign()
    arrays = dict(zip(record.artifact_names, record.artifact_arrays, strict=True))
    assert bool(record.execution_success)
    assert bool(jnp.any(arrays["active-events"]))
    assert jnp.max(
        jnp.where(arrays["active-events"], arrays["root-residuals"], 0.0)
    ) <= 1e-8
    assert not bool(record.capacity_exhausted)


def test_adaptive_heom_accepts_steps_and_reaches_final_time():
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    expansion = phx.operators.quantum.drude_lorentz_matsubara(
        0.01, 1.0, 2.0, 1
    )
    problem = phx.solver.HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion,
        phx.solver.HEOMHierarchy(expansion.rank, 1),
        density,
    )
    result = phx.solver.solve_heom_adaptive_bdf(
        problem,
        final_time=0.002,
        initial_step=0.001,
        relative_tolerance=1e-4,
        absolute_tolerance=1e-7,
        maximum_attempts=32,
    )
    assert bool(result.valid)
    assert result.accepted_step_count > 0
    assert jnp.isclose(result.solution.times[-1], 0.002)


def test_tomography_setting_fingerprint_canonicalizes_kraus_gauge():
    identity = jnp.eye(2, dtype=complex)
    phase = jnp.exp(0.37j) * identity
    first = phx.tensor_network.QuantumInstrument(
        identity[None, None, ...],
        jnp.asarray([True]),
        jnp.asarray([[True]]),
        instrument_id="fingerprint-first",
    )
    second = phx.tensor_network.QuantumInstrument(
        phase[None, None, ...],
        jnp.asarray([True]),
        jnp.asarray([[True]]),
        instrument_id="fingerprint-second",
    )
    first_experiment = phx.solver.ProcessTomographyExperiment(
        (first,),
        (0,),
        1.0,
        terminal_effect=0.5 * identity,
        trials=1.0,
        experiment_id="fingerprint-first",
    )
    second_experiment = phx.solver.ProcessTomographyExperiment(
        (second,),
        (0,),
        1.0,
        terminal_effect=0.5 * identity,
        trials=1.0,
        experiment_id="fingerprint-second",
    )
    assert first_experiment.same_setting(second_experiment)
    assert not phx.solver.tomography_designs_disjoint(
        (first_experiment,), (second_experiment,)
    )


def test_inactive_instrument_outcome_is_rejected_everywhere():
    identity = jnp.eye(2, dtype=complex)
    instrument = phx.tensor_network.QuantumInstrument(
        jnp.stack((identity, identity))[:, None, ...],
        jnp.asarray([True, False]),
        jnp.asarray([[True], [False]]),
        instrument_id="inactive-outcome",
    )
    with pytest.raises(ValueError, match="inactive"):
        phx.solver.ProcessTomographyExperiment(
            (instrument,),
            (1,),
            0.0,
            terminal_effect=0.5 * identity,
            trials=1.0,
            experiment_id="inactive-experiment",
        )
    process = phx.tensor_network.CausalProcessTensor(
        phx.tensor_network.CombLegSpec(2, 1, 1),
        0.5 * identity,
        (identity[None, ...],),
        process_id="inactive-contract",
    )
    with pytest.raises(ValueError, match="inactive"):
        process.contract((instrument,), (1,))


def test_analytic_pade_poles_are_stable_and_improve_with_order():
    first = phx.operators.quantum.drude_lorentz_pade(0.01, 1.0, 2.0, 1)
    second = phx.operators.quantum.drude_lorentz_pade(0.01, 1.0, 2.0, 2)
    assert bool(first.valid)
    assert bool(second.valid)
    assert jnp.all(jnp.real(second.exponents) > 0.0)
    assert second.fit_residual <= first.fit_residual


def test_direct_memory_map_certification_checks_cp_and_tp():
    initial = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    problem = phx.solver.exponential_memory_qubit_problem(0.01, 1.0, initial)
    result = phx.solver.certify_memory_kernel_map(
        problem,
        step_size=0.001,
        steps=2,
    )
    assert result.superoperators.shape == (3, 4, 4)
    assert bool(result.valid)


def test_mps_dense_materialization_is_capacity_bounded():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    )
    with pytest.raises(ValueError, match="capacity"):
        state.to_dense(maximum_elements=2)
    assert state.to_dense(maximum_elements=4).shape == (4,)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.applications.battery import _qualification
from phydrax.applications.battery._calibration import (
    BatteryCalibrationExperiment,
    BatteryCalibrationParameterBranches,
    BatteryCalibrationPlan,
)
from phydrax.applications.battery._data import (
    BatteryDuplicateCriterion,
    BatteryGroupSplit,
    BatteryPipelineIDs,
    BatteryPreprocessingPolicy,
    BatteryRawTimeSeries,
    preprocess_battery_time_series,
)
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
)
from phydrax.applications.battery._observations import BatterySourceUnits
from phydrax.applications.battery._oed import (
    BatteryOEDCandidateSet,
    BatteryOEDModelContract,
    BatteryOEDSupport,
    evaluate_battery_oed,
    prepare_battery_oed,
    prepare_battery_oed_continuous_problem,
    prepare_battery_oed_enumeration,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._results import BatteryModelOutput
from phydrax.artifacts import ArtifactManifest
from phydrax.optim import FiniteLandscapePolicy, MinimizationProblem, search_finite
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    QualificationCriterion,
    QualificationEvidence,
    ReleaseGateEvidence,
    ReleaseIndex,
    SupportTuple,
)
from phydrax.solver import DifferentialProblem
from phydrax.uq import IdentityBijector, ParameterSpace


class _SyntheticLedger(StrictModule):
    successful: jax.Array


def _candidate(domain_ok=True, ledger_ok=True):
    support = SupportTuple(
        "battery.simulation",
        {
            "model_id": f"test:linear-oed:{int(domain_ok)}:{int(ledger_ok)}",
            "equation_form": "ode",
            "control": "prescribed-current",
        },
    )
    return CapabilityProfile(
        "battery.oed-model-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    ), support


@pytest.fixture(autouse=True)
def _synthetic_candidate_registry(monkeypatch):
    profiles = tuple(
        _candidate(domain, ledger)[0]
        for domain in (False, True)
        for ledger in (False, True)
    )
    monkeypatch.setattr(
        _qualification,
        "BATTERY_CANDIDATE_PROFILES",
        (*_qualification.BATTERY_CANDIDATE_PROFILES, *profiles),
    )


class LinearOEDAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)
    domain_ok: bool = eqx.field(static=True)
    ledger_ok: bool = eqx.field(static=True)

    def __init__(self, *, domain_ok=True, ledger_ok=True):
        self.domain_ok = bool(domain_ok)
        self.ledger_ok = bool(ledger_ok)
        self.model_id = f"test:linear-oed:{int(self.domain_ok)}:{int(self.ledger_ok)}"
        self.equation_form = "ode"
        self.observable_names = (
            "voltage_v",
            "temperature_k",
            "stoichiometry:negative",
            "stoichiometry:positive",
        )
        self.observable_units = ("V", "K", "1", "1")

    def prepare(self, /):
        return self.model_id

    def initial_state(self, prepared_model, parameters, initial_condition, /):
        assert prepared_model == self.model_id
        del parameters
        return jnp.asarray(initial_condition)

    def problem(self, prepared_model, initial_state, runtime_inputs, /):
        assert prepared_model == self.model_id

        def drift(time_s, state, runtime):
            del state
            return runtime.parameters["offset"] + runtime.parameters[
                "current_gain"
            ] * runtime.current(time_s, jnp.asarray(0.0))

        return DifferentialProblem(
            drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=f"{self.model_id}:ode",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        assert prepared_model == self.model_id
        del runtime_inputs
        voltage = jnp.asarray(states)
        values = jnp.stack(
            (
                voltage,
                jnp.full_like(voltage, 300.0),
                jnp.full_like(voltage, 0.4),
                jnp.full_like(voltage, 0.6),
            ),
            axis=-1,
        )
        valid = jnp.full(times_s.shape, self.domain_ok, dtype=bool)
        return BatteryModelOutput(values, valid)

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        assert prepared_model == self.model_id
        del native_solution, runtime_inputs
        return _SyntheticLedger(jnp.asarray(self.ledger_ok))


def _ledger_success(ledger):
    return ledger.successful


def _log_prior(branches):
    leaves = tuple(jnp.ravel(jnp.asarray(value)) for value in jax.tree.leaves(branches))
    return -0.5 * jnp.sum(jnp.concatenate(leaves) ** 2)


def _record(*, observation_shift=0.0, name="train", cell_id="oed-cell"):
    times = np.arange(4.0)
    return preprocess_battery_time_series(
        BatteryRawTimeSeries(
            record_id=f"oed-record-{name}-{observation_shift}",
            experiment_id=f"oed-experiment-{name}",
            cell_id=cell_id,
            source_id="synthetic-source",
            resource_id=f"oed-resource-{name}-{observation_shift}",
            row_ids=tuple(f"{name}-row-{index}" for index in range(4)),
            rights_id="synthetic-rights",
            chemistry="test-chemistry",
            form_factor="test-cell",
            protocol_role="local-oed-calibration",
            source_units=BatterySourceUnits("s", "A", "V", "K", "passive"),
            artifact_manifest=ArtifactManifest(
                artifact_id=f"local-oed-{name}-{observation_shift}",
                producer="battery-oed-test",
                version="1",
                sha256=("a" if name == "train" else "b" if name == "calibration" else "c")
                * 64,
                byte_size=0,
                source_uri=f"file:///synthetic/oed-{name}-{observation_shift}.csv",
                license_id="CC0-1.0",
                model="synthetic-linear-oed",
                coverage="unit-test",
            ),
            time=times,
            current=np.asarray((1.0, 1.0, 0.0, 2.0)),
            voltage=np.asarray((3.0, 3.1, 3.2, 3.3)) + observation_shift,
            temperature=np.full(4, 300.0),
        ),
        BatteryPreprocessingPolicy(BatteryDuplicateCriterion(), 10.0),
    )


def _split(record):
    calibration = _record(name="calibration", cell_id="oed-calibration-cell")
    test = _record(name="test", cell_id="oed-test-cell")
    corpus = (record, calibration, test)
    pipelines = BatteryPipelineIDs.from_specifications(
        preprocessing_id=record.preprocessing_id,
        normalization={"kind": "none"},
        noise_model={"kind": "known-scale"},
        model_selection={"kind": "local-oed"},
    )
    return BatteryGroupSplit(
        corpus,
        train_cell_ids=(record.cell_id,),
        calibration_cell_ids=(calibration.cell_id,),
        test_cell_ids=(test.cell_id,),
        pipeline_ids=pipelines,
    )


def _prepared_calibration(*, observation_shift=0.0, domain_ok=True, ledger_ok=True):
    record = _record(observation_shift=observation_shift)
    protocol = BatteryProtocolPlan(
        (CurrentStepPlan(1.0), RestStepPlan(1.0), CurrentStepPlan(1.0))
    )
    profile, support_tuple = _candidate(domain_ok, ledger_ok)
    adapter = LinearOEDAdapter(domain_ok=domain_ok, ledger_ok=ledger_ok)
    prepared_experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(
            (
                "voltage_v",
                "temperature_k",
                "stoichiometry:negative",
                "stoichiometry:positive",
            )
        ),
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            adjoint=dfx.DirectAdjoint(),
            dt0=0.1,
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
        ),
        record.time_s,
        profile,
        support_tuple,
    ).prepare()
    experiment = BatteryCalibrationExperiment(
        prepared_experiment,
        record,
        channel_map=("voltage_v",),
        channel_scales=1.0,
        ledger_success=_ledger_success,
        ledger_success_id="test:scalar-ledger-success",
    )
    position = BatteryCalibrationParameterBranches(
        {"offset": jnp.asarray(0.1), "current_gain": jnp.asarray(0.2)},
        (jnp.asarray(3.0),),
        ((),),
    )
    bijectors = BatteryCalibrationParameterBranches(
        {"offset": IdentityBijector(), "current_gain": IdentityBijector()},
        (IdentityBijector(),),
        ((),),
    )
    parameter_space = ParameterSpace(
        position,
        bijectors=bijectors,
        log_prior=_log_prior,
    )
    return BatteryCalibrationPlan(
        (experiment,),
        parameter_space,
        group_split=_split(record),
        fit_partition="train",
        parameter_space_id=f"test:oed-space:{observation_shift}:{domain_ok}:{ledger_ok}",
        problem_id="test:battery-oed",
    ).prepare()


def _support(*, current=(-3.0, 3.0), duration=(0.5, 1.5)):
    return BatteryOEDSupport(
        current[0],
        current[1],
        voltage_v=(-10.0, 20.0),
        temperature_k=(290.0, 310.0),
        stoichiometry={"negative": (0.1, 0.9), "positive": (0.1, 0.9)},
        duration_s=duration,
    )


def _qualification_criterion(role, support):
    metric = (
        "battery-model-smoothness-defect"
        if role == "smoothness"
        else "battery-model-directional-derivative-error"
    )
    applicability = (
        "battery-oed-local-smoothness"
        if role == "smoothness"
        else "battery-oed-fixed-path-derivative"
    )
    return QualificationCriterion(
        support_tuple_id=support.support_tuple_id,
        metric=metric,
        unit="1",
        comparison="less-than-or-equal",
        target=1.0e-6,
        aggregation="maximum",
        uncertainty="none",
        applicability=applicability,
        approval_id=f"test:{role}-approval",
        issued_at=25,
        valid_until=200,
    )


def _campaign_records(role, criterion, support, replay_id):
    start = CampaignStartRecord(
        campaign_spec_id=f"test:{role}-campaign",
        criterion_id=criterion.criterion_id,
        resolved_run_spec_id=replay_id,
        support_tuple_id=support.support_tuple_id,
        started_at=35,
    )
    observation = CampaignObservationRecord(
        start_record_id=start.start_record_id,
        campaign_spec_id=start.campaign_spec_id,
        criterion_id=start.criterion_id,
        resolved_run_spec_id=start.resolved_run_spec_id,
        support_tuple_id=start.support_tuple_id,
        raw_artifact_ids=(f"test:{role}-artifact",),
        observed_at=45,
    )
    return start, observation


def _qualification_evidence(
    role,
    criterion,
    start,
    observation,
    subjects,
    topology,
    replay_id,
    *,
    outcome="passed",
    expires_at=200,
    evidence_kind="scientific",
    criteria_ids=None,
):
    return QualificationEvidence(
        evidence_kind,
        outcome,
        subjects,
        build_id="test:oed-build",
        environment_id="test:oed-environment",
        backend="jax",
        topology=topology,
        precision="float64",
        reduction="none",
        replay_id=replay_id,
        criteria_ids=(
            (criterion.criterion_id,) if criteria_ids is None else criteria_ids
        ),
        raw_artifact_ids=observation.raw_artifact_ids,
        campaign_start_record_ids=(start.start_record_id,),
        campaign_observation_record_ids=(observation.observation_record_id,),
        reviewer_id="test:oed-reviewer",
        issued_at=50,
        expires_at=expires_at,
        reason=f"{role}-{outcome}",
    )


def _trusted_model_contract(calibration, *, trust_key=b"oed-secret"):
    experiment = calibration.plan.experiments[0]
    model_id = experiment.prepared.plan.model.model_id
    release_support = experiment.prepared.plan.support_tuple
    profile_coordinates = {
        "provider": "phydrax-tests",
        "name": "battery.linear-oed-model",
        "version": "1",
    }
    release_subject_id = canonical_fingerprint(
        {
            "kind": "battery-oed-release-subject",
            "model_id": model_id,
            "profile": profile_coordinates,
            "support_tuple_id": release_support.support_tuple_id,
        }
    )
    replay_id = "test:oed-qualified-replay"
    subjects = tuple(
        sorted(
            (
                model_id,
                release_subject_id,
                release_support.support_tuple_id,
                calibration.preparation_id,
                calibration.plan.calibration_plan_id,
                experiment.calibration_experiment_id,
                experiment.prepared.preparation_id,
                replay_id,
            )
        )
    )
    topology = experiment.prepared.plan.experiment_plan_id
    smooth_criterion = _qualification_criterion("smoothness", release_support)
    derivative_criterion = _qualification_criterion("derivative", release_support)
    smooth_start, smooth_observation = _campaign_records(
        "smoothness", smooth_criterion, release_support, replay_id
    )
    derivative_start, derivative_observation = _campaign_records(
        "derivative", derivative_criterion, release_support, replay_id
    )
    smoothness = _qualification_evidence(
        "smoothness",
        smooth_criterion,
        smooth_start,
        smooth_observation,
        subjects,
        topology,
        replay_id,
    )
    derivative = _qualification_evidence(
        "derivative",
        derivative_criterion,
        derivative_start,
        derivative_observation,
        subjects,
        topology,
        replay_id,
    )
    gates = (
        ReleaseGateEvidence(
            "smoothness",
            passed=True,
            evidence_ids=(
                smooth_criterion.criterion_id,
                smoothness.evidence_id,
            ),
            reviewer_id="test:oed-release-reviewer",
            issued_at=55,
            expires_at=200,
        ),
        ReleaseGateEvidence(
            "derivative",
            passed=True,
            evidence_ids=(
                derivative_criterion.criterion_id,
                derivative.evidence_id,
            ),
            reviewer_id="test:oed-release-reviewer",
            issued_at=55,
            expires_at=200,
        ),
    )
    profile = CapabilityProfile(
        profile_coordinates["name"],
        profile_coordinates["provider"],
        profile_coordinates["version"],
        (release_support,),
        required_gates=("smoothness", "derivative"),
        release_evidence=gates,
        released=True,
    )
    signer = HMACSHA256ReleaseSigner("test:oed-signer", b"oed-secret")
    index = ReleaseIndex.sign((profile,), signer, issued_at=75)
    trust = HMACSHA256TrustPolicy(
        {"test:oed-signer": trust_key},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    contract = BatteryOEDModelContract(
        model_id,
        index,
        profile.profile_id,
        release_support,
        trust,
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness,
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative,
        at_time=100,
    )
    return (
        contract,
        index,
        profile,
        release_support,
        trust,
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness,
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative,
    )


def _oed(*, criterion="d_optimal", observation_shift=0.0, domain_ok=True, ledger_ok=True):
    calibration = _prepared_calibration(
        observation_shift=observation_shift,
        domain_ok=domain_ok,
        ledger_ok=ledger_ok,
    )
    contract = _trusted_model_contract(calibration)[0]
    return prepare_battery_oed(
        calibration,
        _support(),
        contract,
        criterion=criterion,
    )


def _expected_jacobian(amplitudes):
    first, second = amplitudes
    # Dict coordinates are current_gain then offset; the initial condition is last.
    return np.asarray(
        (
            (0.0, 0.0, 1.0),
            (first, 1.0, 1.0),
            (first, 2.0, 1.0),
            (first + second, 3.0, 1.0),
        )
    )


@pytest.mark.parametrize("criterion", ("d_optimal", "a_optimal", "e_optimal"))
def test_linear_sensitivity_fisher_and_native_criteria(criterion):
    prepared = _oed(criterion=criterion)
    amplitudes = jnp.asarray((1.0, 2.0))
    result = evaluate_battery_oed(prepared, amplitudes)
    expected_jacobian = _expected_jacobian((1.0, 2.0))
    expected_fisher = expected_jacobian.T @ expected_jacobian
    if criterion == "d_optimal":
        expected_criterion = np.linalg.slogdet(expected_fisher)[1]
    elif criterion == "a_optimal":
        expected_criterion = -np.trace(np.linalg.inv(expected_fisher))
    else:
        expected_criterion = np.linalg.eigvalsh(expected_fisher)[0]

    np.testing.assert_allclose(result.whitened_jacobian, expected_jacobian, rtol=2e-5)
    np.testing.assert_allclose(result.fisher_information, expected_fisher, rtol=3e-5)
    np.testing.assert_allclose(result.information, expected_fisher, rtol=3e-5)
    np.testing.assert_allclose(result.criterion.value, expected_criterion, rtol=5e-5)
    assert bool(result.valid)
    np.testing.assert_allclose(
        prepared.sensitivity_action(amplitudes, jnp.asarray((1.0, 0.0, 0.0))),
        expected_jacobian[:, 0],
        rtol=2e-5,
    )
    assert result.battery_run_id
    assert result.protocol_values_digest
    assert result.initial_state_digest
    assert result.parameters_digest


def test_candidate_ranking_and_native_finite_search_are_exact():
    prepared = _oed()
    candidates = BatteryOEDCandidateSet(jnp.asarray(((0.5, 0.5), (1.0, 1.0), (2.0, 2.0))))
    problem = prepare_battery_oed_enumeration(prepared, candidates)
    search = search_finite(
        problem.evaluator,
        problem.space,
        landscape=FiniteLandscapePolicy(retain=True),
    )

    assert bool(search.successful)
    assert bool(search.exact)
    assert int(search.flat_indices[0]) == 2
    np.testing.assert_allclose(search.points[0], (2.0, 2.0))
    assert search.landscape_valid.tolist() == [True, True, True]
    assert problem.candidate_ids == candidates.candidate_ids


def test_support_failures_are_invalid_and_duration_is_fixed():
    prepared = _oed()
    outside = prepared.evaluate(jnp.asarray((4.0, 0.0)))
    assert not bool(outside.current_valid)
    assert not bool(outside.valid)

    duration_invalid = prepare_battery_oed(
        prepared.calibration,
        _support(duration=(0.1, 0.5)),
        prepared.model_contract,
    ).evaluate(jnp.asarray((1.0, 1.0)))
    assert not bool(duration_invalid.duration_valid)
    assert not bool(duration_invalid.valid)

    with pytest.raises(ValueError, match="fixed current-step count"):
        prepared.evaluate(jnp.asarray((1.0,)))


def test_failed_model_and_ledger_paths_are_infeasible_not_finite_scores():
    model_failure = _oed(domain_ok=False)
    model_result = model_failure.evaluate(jnp.asarray((1.0, 1.0)))
    assert not bool(model_result.simulation_valid)
    assert not bool(model_result.valid)
    model_problem = prepare_battery_oed_enumeration(
        model_failure, jnp.asarray(((1.0, 1.0),))
    )
    model_score, model_valid = model_problem.evaluator(jnp.asarray((1.0, 1.0)))
    assert not bool(model_valid)
    assert bool(jnp.isnan(model_score))

    ledger_failure = _oed(ledger_ok=False)
    ledger_result = ledger_failure.evaluate(jnp.asarray((1.0, 1.0)))
    assert not bool(ledger_result.simulation_valid)
    assert not bool(ledger_result.ledger_valid)
    assert not bool(ledger_result.valid)
    ledger_score, ledger_valid = prepare_battery_oed_enumeration(
        ledger_failure, jnp.asarray(((1.0, 1.0),))
    ).evaluator(jnp.asarray((1.0, 1.0)))
    assert not bool(ledger_valid)
    assert bool(jnp.isnan(ledger_score))


def test_jit_gradients_native_problem_and_provenance_are_deterministic():
    first = _oed()
    second = _oed()
    assert first.plan_id == second.plan_id
    assert first.criterion_id == second.criterion_id
    candidates_a = BatteryOEDCandidateSet(jnp.asarray(((1.0, 1.0), (2.0, 2.0))))
    candidates_b = BatteryOEDCandidateSet(jnp.asarray(((1.0, 1.0), (2.0, 2.0))))
    assert candidates_a.candidate_ids == candidates_b.candidate_ids
    assert candidates_a.candidate_set_id == candidates_b.candidate_set_id

    amplitudes = jnp.asarray((1.0, 1.5))
    compiled = eqx.filter_jit(lambda value: evaluate_battery_oed(first, value))(
        amplitudes
    )
    assert bool(compiled.valid)
    assert compiled.calibration_preparation_id == first.calibration.preparation_id
    assert compiled.model_id == first.model_contract.model_id
    assert compiled.model_contract_id == first.model_contract.contract_id
    assert compiled.oed_support_id == first.support.support_id
    assert compiled.criterion_id == first.criterion_id
    assert compiled.replay_id

    problem = prepare_battery_oed_continuous_problem(first)
    assert isinstance(problem, MinimizationProblem)
    assert bool(problem.bounds.contains(amplitudes))
    (value, auxiliary), gradient = problem.value_and_gradient(amplitudes)
    assert bool(auxiliary.valid)
    assert bool(jnp.isfinite(value))
    assert gradient.shape == amplitudes.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_observed_voltage_values_do_not_leak_into_local_design_information():
    baseline = _oed(observation_shift=0.0)
    shifted_observations = _oed(observation_shift=5.0)
    amplitudes = jnp.asarray((1.25, 1.75))
    baseline_result = baseline.evaluate(amplitudes)
    shifted_result = shifted_observations.evaluate(amplitudes)

    assert (
        baseline.calibration.preparation_id
        != shifted_observations.calibration.preparation_id
    )
    np.testing.assert_allclose(
        baseline_result.whitened_prediction,
        shifted_result.whitened_prediction,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        baseline_result.whitened_jacobian,
        shifted_result.whitened_jacobian,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        baseline_result.fisher_information,
        shifted_result.fisher_information,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        baseline_result.criterion.value,
        shifted_result.criterion.value,
        rtol=1e-6,
    )


def test_release_trust_current_evidence_and_content_ids_are_required():
    calibration = _prepared_calibration()
    model_id = calibration.plan.experiments[0].prepared.plan.model.model_id
    (
        _,
        index,
        profile,
        support,
        trust,
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness,
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative,
    ) = _trusted_model_contract(calibration)

    with pytest.raises(ValueError, match="signature, signer trust, or freshness"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            HMACSHA256TrustPolicy(
                {"test:oed-signer": b"wrong-key"},
                maximum_index_age=100,
                maximum_evidence_age=100,
            ),
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    stale = _qualification_evidence(
        "derivative",
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative.subject_ids,
        derivative.topology,
        derivative.replay_id,
        expires_at=99,
    )
    with pytest.raises(ValueError, match="derivative evidence is not current"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            stale,
            at_time=100,
        )

    forged = _qualification_evidence(
        "smoothness",
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness.subject_ids,
        smoothness.topology,
        smoothness.replay_id,
    )
    object.__setattr__(forged, "evidence_id", "forged-evidence-id")
    with pytest.raises(ValueError, match="invalid content address"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            forged,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )


def test_unrelated_kind_criterion_topology_and_replay_evidence_are_rejected():
    calibration = _prepared_calibration()
    model_id = calibration.plan.experiments[0].prepared.plan.model.model_id
    (
        _,
        index,
        profile,
        support,
        trust,
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness,
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative,
    ) = _trusted_model_contract(calibration)

    unrelated_criterion = QualificationCriterion(
        support_tuple_id=support.support_tuple_id,
        metric="unrelated-passed-metric",
        unit="1",
        comparison="less-than-or-equal",
        target=1.0,
        aggregation="maximum",
        uncertainty="none",
        applicability="battery-oed-local-smoothness",
        approval_id="test:unrelated-approval",
        issued_at=25,
        valid_until=200,
    )
    with pytest.raises(ValueError, match="criterion has incompatible semantics"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            unrelated_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    unrelated = _qualification_evidence(
        "smoothness",
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness.subject_ids,
        smoothness.topology,
        smoothness.replay_id,
        criteria_ids=(derivative_criterion.criterion_id,),
    )
    with pytest.raises(ValueError, match="cite only its exact criterion"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            unrelated,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    wrong_kind = _qualification_evidence(
        "smoothness",
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness.subject_ids,
        smoothness.topology,
        smoothness.replay_id,
        evidence_kind="unit",
    )
    with pytest.raises(ValueError, match="scientific or reference"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            wrong_kind,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    wrong_topology = _qualification_evidence(
        "derivative",
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative.subject_ids,
        "unrelated-topology",
        derivative.replay_id,
    )
    with pytest.raises(ValueError, match="one exact topology"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            wrong_topology,
            at_time=100,
        )

    wrong_replay = _qualification_evidence(
        "derivative",
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative.subject_ids,
        derivative.topology,
        "unrelated-replay",
    )
    with pytest.raises(ValueError, match="replay does not bind its run"):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            wrong_replay,
            at_time=100,
        )


def test_unreleased_profile_calibration_binding_and_asymmetric_prior_are_rejected():
    calibration = _prepared_calibration()
    model_id = calibration.plan.experiments[0].prepared.plan.model.model_id
    (
        contract,
        index,
        profile,
        support,
        trust,
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness,
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative,
    ) = _trusted_model_contract(calibration)

    gate_signer = HMACSHA256ReleaseSigner("test:oed-signer", b"oed-secret")
    zero_gate_profile = CapabilityProfile(
        profile.name,
        profile.provider,
        profile.version,
        (support,),
        released=True,
    )
    zero_gate_index = ReleaseIndex.sign((zero_gate_profile,), gate_signer, issued_at=75)
    with pytest.raises(ValueError, match="must require smoothness and derivative gates"):
        BatteryOEDModelContract(
            model_id,
            zero_gate_index,
            zero_gate_profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    wrong_citation_gates = (
        ReleaseGateEvidence(
            "smoothness",
            passed=True,
            evidence_ids=(
                smooth_criterion.criterion_id,
                "unrelated-evidence",
            ),
            reviewer_id="test:oed-release-reviewer",
            issued_at=55,
            expires_at=200,
        ),
        ReleaseGateEvidence(
            "derivative",
            passed=True,
            evidence_ids=(
                derivative_criterion.criterion_id,
                derivative.evidence_id,
            ),
            reviewer_id="test:oed-release-reviewer",
            issued_at=55,
            expires_at=200,
        ),
    )
    wrong_citation_profile = CapabilityProfile(
        profile.name,
        profile.provider,
        profile.version,
        (support,),
        required_gates=("smoothness", "derivative"),
        release_evidence=wrong_citation_gates,
        released=True,
    )
    wrong_citation_index = ReleaseIndex.sign(
        (wrong_citation_profile,), gate_signer, issued_at=75
    )
    with pytest.raises(
        ValueError, match="gate must cite its exact criterion and evidence"
    ):
        BatteryOEDModelContract(
            model_id,
            wrong_citation_index,
            wrong_citation_profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    unreleased = CapabilityProfile(
        "battery.unreleased-linear-oed",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    )
    signer = HMACSHA256ReleaseSigner("test:oed-signer", b"oed-secret")
    unreleased_index = ReleaseIndex.sign((unreleased,), signer, issued_at=75)
    with pytest.raises(ValueError, match="not admissible"):
        BatteryOEDModelContract(
            model_id,
            unreleased_index,
            unreleased.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            smoothness,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            derivative,
            at_time=100,
        )

    wrong_topology_smooth = _qualification_evidence(
        "smoothness",
        smooth_criterion,
        smooth_start,
        smooth_observation,
        smoothness.subject_ids,
        "wrong-but-shared-topology",
        smoothness.replay_id,
    )
    wrong_topology_derivative = _qualification_evidence(
        "derivative",
        derivative_criterion,
        derivative_start,
        derivative_observation,
        derivative.subject_ids,
        "wrong-but-shared-topology",
        derivative.replay_id,
    )
    with pytest.raises(
        ValueError, match="gate must cite its exact criterion and evidence"
    ):
        BatteryOEDModelContract(
            model_id,
            index,
            profile.profile_id,
            support,
            trust,
            smooth_criterion,
            smooth_start,
            smooth_observation,
            wrong_topology_smooth,
            derivative_criterion,
            derivative_start,
            derivative_observation,
            wrong_topology_derivative,
            at_time=100,
        )

    near_symmetric = np.eye(3)
    near_symmetric[0, 1] = 8.0 * np.finfo(float).eps
    prepared = prepare_battery_oed(
        calibration,
        _support(),
        contract,
        prior_information=near_symmetric,
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.prior_information),
        np.asarray(prepared.prior_information).T,
    )

    asymmetric = np.eye(3)
    asymmetric[0, 1] = 1.0e-6
    with pytest.raises(ValueError, match="finite and symmetric"):
        prepare_battery_oed(
            calibration,
            _support(),
            contract,
            prior_information=asymmetric,
        )

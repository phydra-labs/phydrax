#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from phydrax.applications.battery._identifiability import (
    BatteryIdentifiabilityPlan,
    BatteryIdentifiabilityReport,
    evaluate_battery_identifiability,
)
from phydrax.applications.battery._observations import BatterySourceUnits
from phydrax.applications.battery._protocol import BatteryProtocolPlan, CurrentStepPlan
from phydrax.applications.battery._results import BatteryModelOutput
from phydrax.artifacts import ArtifactManifest
from phydrax.linalg import RankPolicy
from phydrax.qualification import CapabilityProfile, SupportTuple
from phydrax.solver import DifferentialProblem
from phydrax.uq import IdentityBijector, ParameterSpace


class _SyntheticLedger(StrictModule):
    successful: jax.Array


def _candidate():
    support = SupportTuple(
        "battery.simulation",
        {
            "model_id": "test:current-affine-voltage",
            "equation_form": "ode",
            "control": "prescribed-current",
        },
    )
    return CapabilityProfile(
        "battery.identifiability-candidate",
        "phydrax-tests",
        "candidate",
        (support,),
        released=False,
    ), support


@pytest.fixture(autouse=True)
def _synthetic_candidate_registry(monkeypatch):
    profile, _ = _candidate()
    monkeypatch.setattr(
        _qualification,
        "BATTERY_CANDIDATE_PROFILES",
        (*_qualification.BATTERY_CANDIDATE_PROFILES, profile),
    )


class CurrentAffineAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self):
        self.model_id = "test:current-affine-voltage"
        self.equation_form = "ode"
        self.observable_names = ("voltage_v",)
        self.observable_units = ("V",)

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
            parameters = runtime.parameters
            return parameters["intercept_v_s"] + parameters[
                "current_gain_v_a_s"
            ] * runtime.current(time_s, jnp.asarray(0.0))

        return DifferentialProblem(
            drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id="test:current-affine-voltage:ode",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        assert prepared_model == self.model_id
        del runtime_inputs
        voltage = jnp.asarray(states)
        return BatteryModelOutput(
            voltage[:, None], jnp.isfinite(times_s) & jnp.isfinite(voltage)
        )

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        assert prepared_model == self.model_id
        del native_solution, runtime_inputs
        return _SyntheticLedger(jnp.asarray(True))


def _record(
    name,
    current_a,
    initial_v,
    intercept_v_s,
    current_gain_v_a_s,
    *,
    cell_id="identifiability-cell",
):
    times = np.asarray((0.0, 1.0, 2.0))
    rate = intercept_v_s + current_gain_v_a_s * current_a
    raw = BatteryRawTimeSeries(
        record_id=name,
        experiment_id=f"experiment-{name}",
        cell_id=cell_id,
        source_id="synthetic-source",
        resource_id=f"resource-{name}",
        row_ids=tuple(f"{name}-row-{index}" for index in range(3)),
        rights_id="synthetic-rights",
        chemistry="test-chemistry",
        form_factor="test-cell",
        protocol_role="identifiability",
        source_units=BatterySourceUnits("s", "A", "V", "K", "passive"),
        artifact_manifest=ArtifactManifest(
            artifact_id=f"local-{name}",
            producer="battery-identifiability-test",
            version="1",
            sha256=("c" if name == "one" else "d") * 64,
            byte_size=0,
            source_uri=f"file:///synthetic/{name}.csv",
            license_id="CC0-1.0",
            model="synthetic-current-affine",
            coverage="unit-test",
        ),
        time=times,
        current=np.full(3, current_a),
        voltage=initial_v + rate * times,
        temperature=np.full(3, 300.0),
    )
    return preprocess_battery_time_series(
        raw,
        BatteryPreprocessingPolicy(BatteryDuplicateCriterion(), 10.0),
    )


def _split(records):
    calibration_record = _record(
        "split-calibration",
        0.0,
        0.0,
        0.0,
        0.0,
        cell_id="identifiability-calibration-cell",
    )
    test_record = _record(
        "split-test",
        0.0,
        0.0,
        0.0,
        0.0,
        cell_id="identifiability-test-cell",
    )
    corpus = tuple(records) + (calibration_record, test_record)
    pipeline_ids = BatteryPipelineIDs.from_specifications(
        preprocessing_id=corpus[0].preprocessing_id,
        normalization={"kind": "none"},
        noise_model={"kind": "externally-declared"},
        model_selection={"kind": "identifiability"},
    )
    return BatteryGroupSplit(
        corpus,
        train_cell_ids=tuple(sorted({record.cell_id for record in records})),
        calibration_cell_ids=("identifiability-calibration-cell",),
        test_cell_ids=("identifiability-test-cell",),
        pipeline_ids=pipeline_ids,
    )


def _ledger_bool(ledger):
    return ledger.successful


def _prepared(record):
    protocol = BatteryProtocolPlan((CurrentStepPlan(2.0),))
    profile, support = _candidate()
    return BatteryExperimentPlan(
        CurrentAffineAdapter(),
        protocol,
        BatteryOutputPlan(("voltage_v",)),
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            adjoint=dfx.ForwardMode(),
            dt0=0.25,
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-9,
        ),
        record.time_s,
        profile,
        support,
    ).prepare()


def _log_prior(branches):
    leaves = jnp.asarray(
        tuple(jnp.asarray(value) for value in branches.shared_physical.values())
    )
    return -0.5 * jnp.sum(leaves * leaves)


def _calibration(currents):
    records = tuple(
        _record(
            "one" if index == 0 else "two",
            current,
            0.5 + index,
            1.0,
            2.0,
        )
        for index, current in enumerate(currents)
    )
    experiments = tuple(
        BatteryCalibrationExperiment(
            _prepared(record),
            record,
            channel_map=("voltage_v",),
            channel_scales=1.0,
            ledger_success=_ledger_bool,
            ledger_success_id="test:boolean-ledger",
        )
        for record in records
    )
    position = BatteryCalibrationParameterBranches(
        {
            "intercept_v_s": jnp.asarray(1.0),
            "current_gain_v_a_s": jnp.asarray(2.0),
        },
        tuple(jnp.asarray(0.5 + index) for index in range(len(records))),
        tuple(() for _ in records),
    )
    bijectors = BatteryCalibrationParameterBranches(
        {
            "intercept_v_s": IdentityBijector(),
            "current_gain_v_a_s": IdentityBijector(),
        },
        tuple(IdentityBijector() for _ in records),
        tuple(() for _ in records),
    )
    parameter_space = ParameterSpace(
        position,
        bijectors=bijectors,
        log_prior=_log_prior,
    )
    return BatteryCalibrationPlan(
        experiments,
        parameter_space,
        group_split=_split(records),
        fit_partition="train",
        parameter_space_id=f"test:identifiability-space:{len(records)}",
        problem_id=f"test:identifiability:{len(records)}",
    )


def test_rank_deficiency_weak_direction_and_deterministic_provenance():
    calibration = _calibration((1.0,))
    rank_policy = RankPolicy(relative_cutoff=1.0e-5)
    plan = BatteryIdentifiabilityPlan(calibration, rank_policy=rank_policy)
    report = evaluate_battery_identifiability(plan)

    assert isinstance(report, BatteryIdentifiabilityReport)
    assert report.whitened_jacobian.shape == (3, 3)
    np.testing.assert_allclose(
        report.fisher_information,
        report.whitened_jacobian.T @ report.whitened_jacobian,
        rtol=1.0e-6,
    )
    assert int(report.numerical_rank) == 2
    assert np.isinf(np.asarray(report.condition_number))
    np.testing.assert_array_equal(report.weak_direction_mask, (False, False, True))
    weak = np.asarray(report.weak_directions[-1])
    assert abs(abs(weak[0]) - abs(weak[1])) < 2.0e-4
    assert abs(weak[2]) < 2.0e-4
    assert bool(report.successful)
    assert report.problem_id == calibration.problem_id
    assert report.calibration_preparation_id == plan.calibration.preparation_id
    assert (
        report.plan_id
        == BatteryIdentifiabilityPlan(calibration, rank_policy=rank_policy).plan_id
    )
    assert report.report_id == evaluate_battery_identifiability(plan).report_id
    changed_position = eqx.tree_at(
        lambda value: value.initial_conditions[0],
        calibration.parameter_space.initial,
        jnp.asarray(0.75),
    )
    changed_report = evaluate_battery_identifiability(plan, changed_position)
    assert changed_report.position_fingerprint != report.position_fingerprint
    assert changed_report.report_id != report.report_id
    assert len(report.position_fingerprint) == 64
    assert len(report.evidence_fingerprint) == 64
    assert len(report.parameter_ids) == 3


def test_distinct_current_experiments_restore_rank_and_report_compiles():
    calibration = _calibration((1.0, 2.0))
    plan = BatteryIdentifiabilityPlan(
        calibration,
        rank_policy=RankPolicy(relative_cutoff=1.0e-5, require_full_rank=True),
    )
    position = calibration.parameter_space.initial
    report = evaluate_battery_identifiability(plan, position)

    assert report.whitened_jacobian.shape == (6, 4)
    assert int(report.numerical_rank) == 4
    assert np.isfinite(np.asarray(report.condition_number))
    assert not np.any(np.asarray(report.weak_direction_mask))
    assert bool(report.successful)

    compiled = jax.jit(lambda value: plan.evaluate_numerics(value))(position)
    np.testing.assert_allclose(
        compiled.whitened_jacobian,
        report.whitened_jacobian,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(compiled.singular_values, report.singular_values)
    assert compiled.whitened_jacobian.shape == report.whitened_jacobian.shape


def test_full_rank_policy_marks_confounding_unsuccessful_without_hiding_report():
    calibration = _calibration((1.0,))
    plan = BatteryIdentifiabilityPlan(
        calibration,
        rank_policy=RankPolicy(relative_cutoff=1.0e-5, require_full_rank=True),
    )
    report = evaluate_battery_identifiability(plan)

    assert int(report.numerical_rank) == 2
    assert np.any(np.asarray(report.weak_direction_mask))
    assert not bool(report.successful)
    assert np.all(np.isfinite(np.asarray(report.whitened_jacobian)))

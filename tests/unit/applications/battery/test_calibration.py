#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.applications.battery import _qualification
from phydrax.applications.battery._calibration import (
    battery_calibration_coordinate_layout,
    BatteryCalibrationExperiment,
    BatteryCalibrationParameterBranches,
    BatteryCalibrationPlan,
    prepare_battery_least_squares,
    prepare_battery_posterior,
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
from phydrax.applications.battery._protocol import BatteryProtocolPlan, CurrentStepPlan
from phydrax.applications.battery._results import BatteryModelOutput
from phydrax.artifacts import ArtifactManifest
from phydrax.observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    PrecisionCovarianceAction,
)
from phydrax.optim import CompositeLeastSquaresProblem, NonlinearLeastSquaresProblem
from phydrax.qualification import CapabilityProfile, SupportTuple
from phydrax.solver import DifferentialProblem
from phydrax.uq import ExpBijector, IdentityBijector, ParameterSpace, PosteriorProblem


class _SyntheticLedger(StrictModule):
    successful: jax.Array


def _candidate():
    support = SupportTuple(
        "battery.simulation",
        {
            "model_id": "test:linear-voltage",
            "equation_form": "ode",
            "control": "prescribed-current",
            "thermal": False,
        },
    )
    return CapabilityProfile(
        "battery.linear-calibration-candidate",
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


class LinearVoltageAdapter(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    equation_form: str = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)
    domain_limit: float = eqx.field(static=True)

    def __init__(self, *, domain_limit=1.0e6):
        self.model_id = "test:linear-voltage"
        self.equation_form = "ode"
        self.observable_names = ("voltage_v",)
        self.observable_units = ("V",)
        self.domain_limit = float(domain_limit)

    def prepare(self, /):
        return self.model_id

    def initial_state(self, prepared_model, parameters, initial_condition, /):
        assert prepared_model == self.model_id
        del parameters
        return jnp.asarray(initial_condition)

    def problem(self, prepared_model, initial_state, runtime_inputs, /):
        assert prepared_model == self.model_id

        def drift(time_s, state, runtime):
            del time_s, state
            return jnp.asarray(runtime.parameters["rate_v_s"])

        return DifferentialProblem(
            drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id="test:linear-voltage:ode",
        )

    def observe(self, prepared_model, times_s, states, runtime_inputs, /):
        assert prepared_model == self.model_id
        del runtime_inputs
        voltage = jnp.asarray(states)
        valid = (
            jnp.isfinite(times_s)
            & jnp.isfinite(voltage)
            & (jnp.abs(voltage) <= self.domain_limit)
        )
        return BatteryModelOutput(voltage[:, None], valid)

    def ledger(self, prepared_model, native_solution, runtime_inputs, /):
        assert prepared_model == self.model_id
        del native_solution, runtime_inputs
        return _SyntheticLedger(jnp.asarray(True))


def _manifest(record_id):
    return ArtifactManifest(
        artifact_id=f"local-{record_id}",
        producer="battery-calibration-test",
        version="1",
        sha256=("a" if record_id == "one" else "b") * 64,
        byte_size=0,
        source_uri=f"file:///synthetic/{record_id}.csv",
        license_id="CC0-1.0",
        model="synthetic-linear-voltage",
        coverage="unit-test",
    )


def _record(
    record_id,
    initial,
    rate,
    offset,
    *,
    voltage_mask=(True, True, True),
    cell_id="cell-shared",
):
    times = np.asarray((0.0, 1.0, 2.0))
    voltage = initial + rate * times + offset
    raw = BatteryRawTimeSeries(
        record_id=record_id,
        experiment_id=f"experiment-{record_id}",
        cell_id=cell_id,
        source_id="synthetic-source",
        resource_id=f"resource-{record_id}",
        row_ids=tuple(f"{record_id}-row-{index}" for index in range(3)),
        rights_id="synthetic-rights",
        chemistry="test-chemistry",
        form_factor="test-cell",
        protocol_role="calibration",
        source_units=BatterySourceUnits("s", "A", "V", "K", "passive"),
        artifact_manifest=_manifest(record_id),
        time=times,
        current=np.ones(3),
        voltage=voltage,
        temperature=np.full(3, 300.0),
        current_mask=np.ones(3, dtype=bool),
        voltage_mask=np.asarray(voltage_mask),
        temperature_mask=np.ones(3, dtype=bool),
    )
    return preprocess_battery_time_series(
        raw,
        BatteryPreprocessingPolicy(BatteryDuplicateCriterion(), 10.0),
    )


def _split(records):
    calibration_record = _record(
        "split-calibration", 0.0, 0.0, 0.0, cell_id="cell-calibration"
    )
    test_record = _record("split-test", 0.0, 0.0, 0.0, cell_id="cell-test")
    corpus = tuple(records) + (calibration_record, test_record)
    pipeline_ids = BatteryPipelineIDs.from_specifications(
        preprocessing_id=corpus[0].preprocessing_id,
        normalization={"kind": "none"},
        noise_model={"kind": "externally-declared"},
        model_selection={"kind": "synthetic-calibration"},
    )
    return BatteryGroupSplit(
        corpus,
        train_cell_ids=tuple(sorted({record.cell_id for record in records})),
        calibration_cell_ids=("cell-calibration",),
        test_cell_ids=("cell-test",),
        pipeline_ids=pipeline_ids,
    )


def _ledger_bool(ledger):
    return ledger.successful


def _prepared(record, *, domain_limit=1.0e6, maximum_steps=128):
    protocol = BatteryProtocolPlan((CurrentStepPlan(2.0),))
    profile, support = _candidate()
    return BatteryExperimentPlan(
        LinearVoltageAdapter(domain_limit=domain_limit),
        protocol,
        BatteryOutputPlan(("voltage_v",)),
        BatteryDiffraxSolvePlan(
            solver=dfx.Tsit5(),
            adjoint=dfx.ForwardMode(),
            dt0=0.25,
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-9,
            maximum_steps=maximum_steps,
        ),
        record.time_s,
        profile,
        support,
    ).prepare()


def _add_offset(prediction, nuisance):
    return prediction + nuisance


def _log_prior(branches):
    leaves = jax.tree_util.tree_leaves(branches)
    return -0.5 * sum(jnp.sum(value * value) for value in leaves)


def _space(raw_rate, initials, offsets):
    position = BatteryCalibrationParameterBranches(
        {"rate_v_s": jnp.asarray(raw_rate)},
        tuple(jnp.asarray(value) for value in initials),
        tuple(jnp.asarray(value) for value in offsets),
    )
    bijectors = BatteryCalibrationParameterBranches(
        {"rate_v_s": ExpBijector()},
        tuple(IdentityBijector() for _ in initials),
        tuple(IdentityBijector() for _ in offsets),
    )
    return ParameterSpace(position, bijectors=bijectors, log_prior=_log_prior)


def _two_experiment_plan(*, domain_limit=1.0e6):
    record_one = _record("one", 1.0, 2.0, 0.25, voltage_mask=(True, False, True))
    record_two = _record("two", -1.0, 2.0, -0.5)
    first = BatteryCalibrationExperiment(
        _prepared(record_one, domain_limit=domain_limit),
        record_one,
        channel_map={"voltage_v": "voltage_v"},
        channel_scales={"voltage_v": 0.5},
        nuisance_model=_add_offset,
        nuisance_model_id="test:additive-voltage-offset",
        ledger_success=_ledger_bool,
        ledger_success_id="test:boolean-ledger",
    )
    layout = battery_calibration_coordinate_layout(record_two, ("voltage_v",))
    second = BatteryCalibrationExperiment(
        _prepared(record_two, domain_limit=domain_limit),
        record_two,
        channel_map=("voltage_v",),
        covariance_action=CholeskyCovarianceAction(0.25 * jnp.eye(3), layout),
        nuisance_model=_add_offset,
        nuisance_model_id="test:additive-voltage-offset",
        ledger_success=_ledger_bool,
        ledger_success_id="test:boolean-ledger",
    )
    position = _space(np.log(2.0), (1.0, -1.0), (0.25, -0.5))
    return BatteryCalibrationPlan(
        (first, second),
        position,
        group_split=_split((record_one, record_two)),
        fit_partition="train",
        parameter_space_id="test:linear-voltage-parameter-space",
        problem_id="test:multi-experiment-calibration",
    )


def test_multi_experiment_native_problems_share_physics_and_apply_masks_whitening():
    plan = _two_experiment_plan()
    prepared = plan.prepare()
    position = plan.parameter_space.initial
    residual = prepared.residual(position)

    assert prepared.residual_size == 5
    np.testing.assert_allclose(residual, np.zeros(5), atol=2.0e-6)
    assert plan.experiments[0].observation.voltage_v[1] == 0.0
    assert plan.experiments[0].residual_size == 2
    assert plan.experiments[1].residual_size == 3

    changed = eqx.tree_at(
        lambda value: value.shared_physical["rate_v_s"],
        position,
        jnp.log(jnp.asarray(2.5)),
    )
    changed_residual = prepared.residual(changed)
    assert np.any(np.abs(np.asarray(changed_residual[:2])) > 0.1)
    assert np.any(np.abs(np.asarray(changed_residual[2:])) > 0.1)

    least_squares = prepare_battery_least_squares(prepared)
    composite = prepare_battery_least_squares(prepared, include_prior=True)
    posterior = prepare_battery_posterior(prepared)
    assert isinstance(least_squares, NonlinearLeastSquaresProblem)
    assert isinstance(composite, CompositeLeastSquaresProblem)
    assert isinstance(posterior, PosteriorProblem)
    assert least_squares.problem_id == plan.problem_id
    np.testing.assert_allclose(least_squares.value(position)[0], residual)
    np.testing.assert_allclose(posterior.gauss_newton_residual(position), residual)
    physical = plan.parameter_space.constrain(position)
    assert physical.shared_physical["rate_v_s"] == pytest.approx(2.0)
    assert np.isfinite(np.asarray(posterior.log_likelihood(physical)))
    assert np.isfinite(np.asarray(composite.objective(position)))


def test_precision_whitening_jvp_jacobian_and_jit_agree():
    record = _record("one", 1.0, 2.0, 0.0)
    layout = battery_calibration_coordinate_layout(record, ("voltage_v",))
    precision = 4.0 * jnp.eye(3)
    covariance = PrecisionCovarianceAction(
        precision,
        -3.0 * jnp.log(jnp.asarray(4.0)),
        layout,
    )
    experiment = BatteryCalibrationExperiment(
        _prepared(record),
        record,
        channel_map=("voltage_v",),
        covariance_action=covariance,
        ledger_success=_ledger_bool,
        ledger_success_id="test:boolean-ledger",
    )
    position = BatteryCalibrationParameterBranches(
        {"rate_v_s": jnp.asarray(np.log(1.5))},
        (jnp.asarray(0.5),),
        ((),),
    )
    bijectors = BatteryCalibrationParameterBranches(
        {"rate_v_s": ExpBijector()},
        (IdentityBijector(),),
        ((),),
    )
    parameter_space = ParameterSpace(
        position,
        bijectors=bijectors,
        log_prior=_log_prior,
    )
    plan = BatteryCalibrationPlan(
        (experiment,),
        parameter_space,
        group_split=_split((record,)),
        fit_partition="train",
        parameter_space_id="test:precision-space",
        problem_id="test:precision-calibration",
    )
    prepared = plan.prepare()
    flat, unravel = ravel_pytree(position)
    direction = jnp.linspace(0.25, 0.75, flat.size)
    function = lambda value: prepared.residual(unravel(value))
    jacobian = jax.jacfwd(function)(flat)
    _, tangent = jax.jvp(function, (flat,), (direction,))
    np.testing.assert_allclose(tangent, jacobian @ direction, rtol=2.0e-5, atol=2.0e-5)

    compiled = jax.jit(lambda value: prepared.residual(value))(position)
    np.testing.assert_allclose(compiled, prepared.residual(position), rtol=2.0e-5)
    physical = parameter_space.constrain(position)
    prediction = prepared.physical_prediction(physical)[0][:, 0]
    raw = prediction - record.voltage_v
    expected = np.asarray(2.0 * raw)
    np.testing.assert_allclose(prepared.residual(position), expected, rtol=2.0e-5)


def _broken_nuisance(prediction, nuisance):
    del prediction, nuisance
    raise RuntimeError("programming defect")


def test_declared_domain_failure_is_invalid_but_programming_error_propagates():
    invalid_plan = _two_experiment_plan(domain_limit=2.0)
    invalid_residual = invalid_plan.prepare().residual(
        invalid_plan.parameter_space.initial
    )
    assert np.all(np.isposinf(np.asarray(invalid_residual)))
    posterior = prepare_battery_posterior(invalid_plan)
    physical = invalid_plan.parameter_space.constrain(
        invalid_plan.parameter_space.initial
    )
    assert np.isneginf(np.asarray(posterior.log_likelihood(physical)))

    record = _record("one", 1.0, 2.0, 0.0)
    experiment = BatteryCalibrationExperiment(
        _prepared(record),
        record,
        channel_map=("voltage_v",),
        channel_scales=1.0,
        nuisance_model=_broken_nuisance,
        nuisance_model_id="test:broken-nuisance",
        ledger_success=_ledger_bool,
        ledger_success_id="test:boolean-ledger",
    )
    position = BatteryCalibrationParameterBranches(
        {"rate_v_s": jnp.asarray(np.log(2.0))},
        (jnp.asarray(1.0),),
        (jnp.asarray(0.0),),
    )
    bijectors = BatteryCalibrationParameterBranches(
        {"rate_v_s": ExpBijector()},
        (IdentityBijector(),),
        (IdentityBijector(),),
    )
    space = ParameterSpace(position, bijectors=bijectors, log_prior=_log_prior)
    plan = BatteryCalibrationPlan(
        (experiment,),
        space,
        group_split=_split((record,)),
        fit_partition="train",
        parameter_space_id="test:broken-space",
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        plan.prepare().residual(position)


def test_preparation_rejects_unknown_ledgers_test_data_and_covariance_mismatch():
    record = _record("one", 1.0, 2.0, 0.0)
    prepared = _prepared(record)
    with pytest.raises(TypeError, match="Unknown battery adapters"):
        BatteryCalibrationExperiment(
            prepared,
            record,
            channel_map=("voltage_v",),
            channel_scales=1.0,
        )

    wrong_layout = CoordinateLayout(tuple(f"wrong:{index}" for index in range(3)))
    with pytest.raises(ValueError, match="exactly match"):
        BatteryCalibrationExperiment(
            prepared,
            record,
            channel_map=("voltage_v",),
            covariance_action=CholeskyCovarianceAction(jnp.eye(3), wrong_layout),
            ledger_success=_ledger_bool,
            ledger_success_id="test:boolean-ledger",
        )

    exact_layout = battery_calibration_coordinate_layout(record, ("voltage_v",))
    with pytest.raises(ValueError, match="logdet must equal"):
        BatteryCalibrationExperiment(
            prepared,
            record,
            channel_map=("voltage_v",),
            covariance_action=PrecisionCovarianceAction(
                4.0 * jnp.eye(3), jnp.asarray(0.0), exact_layout
            ),
            ledger_success=_ledger_bool,
            ledger_success_id="test:boolean-ledger",
        )

    experiment = BatteryCalibrationExperiment(
        prepared,
        record,
        channel_map=("voltage_v",),
        channel_scales=1.0,
        ledger_success=_ledger_bool,
        ledger_success_id="test:boolean-ledger",
    )
    position = BatteryCalibrationParameterBranches(
        {"rate_v_s": jnp.asarray(2.0)}, (jnp.asarray(1.0),), ((),)
    )
    bijectors = BatteryCalibrationParameterBranches(
        {"rate_v_s": IdentityBijector()}, (IdentityBijector(),), ((),)
    )
    space = ParameterSpace(position, bijectors=bijectors, log_prior=_log_prior)
    split = _split((record,))
    with pytest.raises(ValueError, match="test data"):
        BatteryCalibrationPlan(
            (experiment,),
            space,
            group_split=split,
            fit_partition="test",
            parameter_space_id="test:partition-space",
        )
    foreign = _record("foreign", 0.0, 0.0, 0.0, cell_id="cell-foreign")
    with pytest.raises(ValueError, match="content binding"):
        BatteryCalibrationPlan(
            (experiment,),
            space,
            group_split=_split((foreign,)),
            fit_partition="train",
            parameter_space_id="test:partition-space",
        )
    changed_content = _record("one", 1.0, 2.0, 0.1)
    with pytest.raises(ValueError, match="content binding"):
        BatteryCalibrationPlan(
            (experiment,),
            space,
            group_split=_split((changed_content,)),
            fit_partition="train",
            parameter_space_id="test:partition-space",
        )
    relabeled = _record("one", 1.0, 2.0, 0.0, cell_id="relabeled-cell")
    with pytest.raises(ValueError, match="content binding"):
        BatteryCalibrationPlan(
            (experiment,),
            space,
            group_split=_split((relabeled,)),
            fit_partition="train",
            parameter_space_id="test:partition-space",
        )

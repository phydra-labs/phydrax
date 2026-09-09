# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.geophysics._ensembles import (
    geophysical_analysis_increments,
    GeophysicalAnalysisInventory,
    GeophysicalEnsembleAxis,
    GeophysicalEnsembleLineage,
    prepare_geophysical_assimilation,
)
from phydrax.applications.geophysics._observations import (
    GeophysicalObservationOperator,
    GeophysicalObservationPolicy,
    prepare_geophysical_observations,
    prepare_tensor_observation_operator,
)
from phydrax.applications.geophysics._quantities import GeophysicalQuantity
from phydrax.applications.geophysics._time import GeophysicalTimeSpec, TemporalSupport
from phydrax.discretization import DiscreteFieldSpace, TensorDofLayout
from phydrax.dynamics import StateLayout
from phydrax.linalg import ArraySpace
from phydrax.stochastic import GaussianStatePrior
from phydrax.units import JOULE, KELVIN, KILOPASCAL, PASCAL
from phydrax.uq import (
    CheckpointCompatibilityError,
    ensemble_filter_step,
    ensemble_kalman_smoother,
    ensemble_transform_kalman_filter,
    initialize_ensemble_filter,
    read_ensemble_filter_checkpoint,
    write_ensemble_filter_checkpoint,
)


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def _space(axes=("column",), shape=(2,), support="physical-columns"):
    return DiscreteFieldSpace(
        "temperature",
        support,
        TensorDofLayout(axes, shape),
        ArraySpace(shape, dtype=jnp.float64),
        representation="point_value",
    )


def _lineage(*, internal="heat-noise", initial="initial-prior", reverse=False):
    coordinates = {
        "initial_condition": initial,
        "scenario": "unforced",
        "parameter": "relaxation-01",
        "structural": "two-columns",
        "internal_stochastic": internal,
    }
    axes = tuple(
        GeophysicalEnsembleAxis(
            kind, (label, "other") if not reverse else ("other", label)
        )
        for kind, label in coordinates.items()
    )
    return GeophysicalEnsembleLineage(axes[::-1] if reverse else axes, coordinates)


def _operator(quantity=None):
    quantity = (
        GeophysicalQuantity("temperature", "temperature_anomaly", KELVIN)
        if quantity is None
        else quantity
    )
    space = _space()
    clock = GeophysicalTimeSpec(unit="d")
    return prepare_tensor_observation_operator(
        space,
        {"column": [0.0, 1.0]},
        {"column": [0.0, 1.0]},
        quantity,
        source_support_id=space.support_id,
        time=clock,
        kind="grid",
    )


def _prepare(operator, values, **kwargs):
    defaults = dict(
        quantity=operator.quantity,
        time=operator.time,
        target_support_id=operator.transfer.target.support_id,
    )
    defaults.update(kwargs)
    return prepare_geophysical_observations(
        operator, np.arange(len(values)), values, 0.1, **defaults
    )


def _flow(state, duration):
    common = 0.5 * (state[0] + state[1]) * jnp.exp(-0.1 * duration)
    contrast = 0.5 * (state[0] - state[1]) * jnp.exp(-0.3 * duration)
    return jnp.stack((common + contrast, common - contrast))


def _case(*, stochastic=False, lineage=None, mask=None):
    operator = _operator()
    truth = jnp.stack(
        (_flow(jnp.array([2.0, -0.5]), 0.0), _flow(jnp.array([2.0, -0.5]), 1.0))
    )
    prepared = _prepare(operator, truth, availability=mask)
    layout = StateLayout((2,), component_names=("west_temperature", "east_temperature"))

    def transition(key, state, t0, t1, context):
        del context
        out = _flow(state, t1 - t0)
        return (
            out + 0.02 * jnp.sqrt(t1 - t0) * jax.random.normal(key, state.shape)
            if stochastic
            else out
        )

    problem = prepare_geophysical_assimilation(
        layout,
        prepared,
        source_field=operator.transfer.source,
        prior=GaussianStatePrior(
            jnp.array([0.0, 0.0]), jnp.array([[1.0, 0.25], [0.25, 2.0]]), state_shape=(2,)
        ),
        transition=transition,
        model_id="radiative-mixing",
        approximation_id="exact-linear"
        if not stochastic
        else "exact-flow-noisy-increment",
        model_time=operator.time,
        initial_time=0.0,
        lineage=_lineage() if lineage is None else lineage,
    )
    return problem, truth, layout


def test_station_vertical_interpolation_and_periodic_transpose():
    source = _space(("longitude", "pressure"), (4, 3), "longitude-pressure-grid")
    quantity = GeophysicalQuantity("temperature", "temperature", KELVIN)
    operator = prepare_tensor_observation_operator(
        source,
        {
            "longitude": [0.0, 90.0, 180.0, 270.0],
            "pressure": [10000.0, 50000.0, 100000.0],
        },
        {"longitude": [-45.0, 315.0, 90.0], "pressure": [30000.0, 30000.0, 75000.0]},
        quantity,
        source_support_id=source.support_id,
        time=GeophysicalTimeSpec(),
        kind="profile",
        periods={"longitude": 360.0},
    )
    # The periodic seam must interpolate columns 270 and 0, never extrapolate.
    values = (
        jnp.array([0.0, 1.0, 2.0, 3.0])[:, None]
        + jnp.array([10000.0, 50000.0, 100000.0])[None, :] * 0.001
    )
    np.testing.assert_allclose(operator(values), [31.5, 31.5, 76.0], atol=1e-12)
    dual = jnp.array([0.3, -0.7, 1.2])
    np.testing.assert_allclose(
        jnp.vdot(operator(values), dual),
        jnp.vdot(values, operator.transfer.operator.transpose_mv(dual)),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="extrapolate"):
        prepare_tensor_observation_operator(
            source,
            {
                "longitude": [0.0, 90.0, 180.0, 270.0],
                "pressure": [10000.0, 50000.0, 100000.0],
            },
            {"longitude": [45.0], "pressure": [110000.0]},
            quantity,
            source_support_id=source.support_id,
            time=GeophysicalTimeSpec(),
        )
    with pytest.raises(ValueError, match="support"):
        prepare_tensor_observation_operator(
            source,
            {
                "longitude": [0.0, 90.0, 180.0, 270.0],
                "pressure": [10000.0, 50000.0, 100000.0],
            },
            {"longitude": [45.0], "pressure": [50000.0]},
            quantity,
            source_support_id="different-grid",
            time=GeophysicalTimeSpec(),
        )


def test_missing_qc_representativeness_and_physical_unit_conversion():
    operator = _operator(GeophysicalQuantity("pressure", "pressure", PASCAL))
    prepared = prepare_geophysical_observations(
        operator,
        [0.0, 1.0],
        [[100.0, np.nan], [999.0, 101.0]],
        [[0.1, np.nan], [np.nan, 0.2]],
        quantity=GeophysicalQuantity("barometer", "pressure", KILOPASCAL),
        time=operator.time,
        target_support_id=operator.transfer.target.support_id,
        availability=np.array([[True, False], [True, True]]),
        representativeness_std=0.3,
        policy=GeophysicalObservationPolicy(bounds=(90000.0, 110000.0)),
    )
    np.testing.assert_array_equal(
        prepared.sequence.observation_mask, [[True, False], [False, True]]
    )
    np.testing.assert_allclose(
        prepared.sequence.values, [[100000.0, 0.0], [0.0, 101000.0]]
    )
    np.testing.assert_allclose(
        prepared.error_variance, [[100000.0, 1.0], [1.0, 130000.0]]
    )
    np.testing.assert_array_equal(prepared.qc_rejected, [[False, False], [True, False]])
    with pytest.raises(ValueError, match="finite"):
        _prepare(_operator(), [[1.0, np.nan]])
    masked = _prepare(
        _operator(),
        [[1.0, np.nan]],
        policy=GeophysicalObservationPolicy(nonfinite="mask"),
    )
    np.testing.assert_array_equal(masked.sequence.observation_mask, [[True, False]])
    with pytest.raises(ValueError, match="positive"):
        prepare_geophysical_observations(
            operator,
            [0.0],
            [[100000.0, 100000.0]],
            0.0,
            quantity=operator.quantity,
            time=operator.time,
            target_support_id=operator.transfer.target.support_id,
        )


@pytest.mark.parametrize(
    "mismatch", ["shape", "mask", "quantity", "time", "support", "temporal"]
)
def test_shape_support_quantity_and_time_semantics_rejected(mismatch):
    operator = _operator()
    values = [[1.0, 2.0]]
    kwargs = {}
    if mismatch == "shape":
        values = [[1.0]]
    elif mismatch == "mask":
        kwargs["availability"] = np.array([True, False])
    elif mismatch == "quantity":
        kwargs["quantity"] = GeophysicalQuantity("temperature", "temperature", KELVIN)
    elif mismatch == "time":
        kwargs["time"] = GeophysicalTimeSpec(calendar="360_day", unit="d")
    elif mismatch == "support":
        kwargs["target_support_id"] = "same-shape-other-stations"
    else:
        kwargs["temporal"] = TemporalSupport("mean", [[0.0, 1.0]], "end")
    with pytest.raises(ValueError):
        _prepare(operator, values, **kwargs)
    if mismatch == "temporal":
        with pytest.raises(ValueError, match="instantaneous"):
            GeophysicalObservationOperator(
                operator.transfer,
                operator.quantity,
                time=operator.time,
                temporal=kwargs["temporal"],
            )


def test_linear_gaussian_limit_with_partial_missing_observations():
    mask = np.array([[True, False], [True, True]])
    problem, truth, _ = _case(mask=mask)
    state = initialize_ensemble_filter(jax.random.key(5), problem, ensemble_size=12)
    mean = np.asarray(state.ensemble).mean(axis=0)
    covariance = np.cov(np.asarray(state.ensemble), rowvar=False)
    initial_error = np.linalg.norm(mean - np.asarray(truth[0]))
    for step in range(2):
        flow = (
            np.eye(2)
            if step == 0
            else np.stack(
                [np.asarray(_flow(jnp.eye(2)[index], 1.0)) for index in range(2)], axis=1
            )
        )
        mean = flow @ mean
        covariance = flow @ covariance @ flow.T
        selected = np.flatnonzero(mask[step])
        h = np.eye(2)[selected]
        r = np.eye(len(selected)) * 0.01
        innovation_covariance = h @ covariance @ h.T + r
        gain = np.linalg.solve(innovation_covariance, h @ covariance).T
        mean = mean + gain @ (np.asarray(truth[step])[selected] - h @ mean)
        covariance = covariance - gain @ h @ covariance
        state, record = ensemble_filter_step(problem, state)
        np.testing.assert_allclose(
            np.mean(record.analysis_ensemble, axis=0), mean, atol=2e-11
        )
        np.testing.assert_allclose(
            np.cov(np.asarray(record.analysis_ensemble), rowvar=False),
            covariance,
            atol=2e-11,
        )
        assert int(record.observed_count) == len(selected)
    assert np.linalg.norm(mean - np.asarray(truth[-1])) < initial_error * 0.1
    assert np.trace(covariance) < 0.03


def test_all_missing_keeps_mean_and_native_inflation_scales_covariance():
    problem, _, _ = _case(mask=np.zeros((2, 2), dtype=bool))
    state = initialize_ensemble_filter(
        jax.random.key(8), problem, ensemble_size=10, inflation=1.2
    )
    prior = np.asarray(state.ensemble)
    _, step = ensemble_filter_step(problem, state)
    np.testing.assert_allclose(
        np.mean(step.analysis_ensemble, axis=0), prior.mean(axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        np.cov(np.asarray(step.analysis_ensemble), rowvar=False),
        np.cov(prior, rowvar=False) * 1.2**2,
        atol=1e-12,
    )
    assert int(step.observed_count) == 0


def test_restart_preserves_internal_noise_and_axis_lineage(tmp_path):
    lineage = _lineage()
    key = jax.random.key(27)
    np.testing.assert_array_equal(
        jax.random.key_data(lineage.key(key, "internal_stochastic", member=4, step=7)),
        jax.random.key_data(
            _lineage(reverse=True).key(key, "internal_stochastic", member=4, step=7)
        ),
    )
    assert not np.array_equal(
        jax.random.key_data(lineage.key(key, "initial_condition")),
        jax.random.key_data(lineage.key(key, "internal_stochastic")),
    )
    problem, _, _ = _case(stochastic=True, lineage=lineage)
    altered, _, _ = _case(stochastic=True, lineage=_lineage(internal="other-noise"))
    state = initialize_ensemble_filter(key, problem, ensemble_size=10)
    # Changing internal forcing must not resample the initial-condition axis.
    np.testing.assert_array_equal(
        state.ensemble,
        initialize_ensemble_filter(key, altered, ensemble_size=10).ensemble,
    )
    state, _ = ensemble_filter_step(problem, state)
    path = tmp_path / "restart.npz"
    write_ensemble_filter_checkpoint(path, problem, state)
    restarted = read_ensemble_filter_checkpoint(path, problem, ensemble_size=10)
    uninterrupted, _ = ensemble_filter_step(problem, state)
    resumed, _ = ensemble_filter_step(problem, restarted)
    np.testing.assert_array_equal(resumed.ensemble, uninterrupted.ensemble)
    np.testing.assert_array_equal(
        jax.random.key_data(resumed.root_key), jax.random.key_data(key)
    )
    assert resumed.step_index == 2
    with pytest.raises(CheckpointCompatibilityError):
        read_ensemble_filter_checkpoint(path, altered, ensemble_size=10)


def test_analysis_heat_inventory_and_native_smoothing():
    problem, truth, layout = _case()
    result = ensemble_transform_kalman_filter(
        jax.random.key(9),
        problem,
        ensemble_size=12,
        inflation=1.05,
        raise_on_failure=True,
    )
    quantity = GeophysicalQuantity("heat", "energy", JOULE)
    inventory = GeophysicalAnalysisInventory("heat", quantity, [2.0e8, 3.0e8], layout)
    report = geophysical_analysis_increments(result, (inventory,))["heat"]
    expected = np.sum(
        np.asarray(result.analysis_ensembles - result.forecast_ensembles)
        * np.array([2.0e8, 3.0e8]),
        axis=-1,
    )
    np.testing.assert_allclose(report.increment, expected, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(
        report.mean_increment, expected.mean(axis=-1), rtol=1e-12, atol=1e-6
    )
    assert (
        abs(float(report.mean_increment[0])) > 1.0e7
    )  # Analysis is a real heat impulse.
    smoother = ensemble_kalman_smoother(result)
    np.testing.assert_array_equal(smoother.ensembles[-1], result.analysis_ensembles[-1])
    assert bool(jnp.all(smoother.valid))
    assert float(jnp.mean((jnp.mean(smoother.ensembles, axis=1) - truth) ** 2)) < 0.01
    wrong_layout = StateLayout((2,), component_names=("other-west", "other-east"))
    with pytest.raises(ValueError, match="layout"):
        geophysical_analysis_increments(
            result,
            (
                GeophysicalAnalysisInventory(
                    "heat", quantity, [2.0e8, 3.0e8], wrong_layout
                ),
            ),
        )


def test_native_model_adapter_rejects_wrong_support_clock_and_changing_shape():
    operator = _operator()
    observations = _prepare(operator, [[1.0, 2.0]])
    layout = StateLayout((2,))
    prior = GaussianStatePrior(jnp.zeros(2), jnp.eye(2), state_shape=(2,))
    options = dict(
        source_field=operator.transfer.source,
        prior=prior,
        transition=lambda key, state, t0, t1, context: _flow(state, t1 - t0),
        model_id="physical-two-column",
        approximation_id="exact-linear",
        model_time=operator.time,
        initial_time=0.0,
        lineage=_lineage(),
    )
    with pytest.raises(ValueError, match="support"):
        prepare_geophysical_assimilation(
            layout,
            observations,
            **{**options, "source_field": _space(support="other-grid")},
        )
    with pytest.raises(ValueError, match="calendar"):
        prepare_geophysical_assimilation(
            layout,
            observations,
            **{**options, "model_time": GeophysicalTimeSpec(unit="s")},
        )
    problem = prepare_geophysical_assimilation(
        layout,
        observations,
        **{**options, "transition": lambda key, state, t0, t1, context: jnp.zeros((3,))},
    )
    state = initialize_ensemble_filter(jax.random.key(10), problem, ensemble_size=4)
    with pytest.raises(ValueError, match="layout"):
        ensemble_filter_step(problem, state)

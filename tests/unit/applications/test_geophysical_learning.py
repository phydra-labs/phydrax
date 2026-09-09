"""Behavioral contracts at the host geophysical/native-operator boundary."""

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.geophysics._learning import (
    admit_column_closure,
    column_closure_dataset,
    ColumnClosureBinding,
    GeophysicalLearningExperiment,
)
from phydrax.applications.geophysics._metrics import (
    fit_geophysical_climatology,
    geophysical_extreme_reliability,
    geophysical_forecast_metrics,
    geophysical_spectral_rmse,
)
from phydrax.applications.geophysics._quantities import GeophysicalQuantity


def _column(*, kind="interval_increment", control=False):
    op = phx.nn.operator
    q = GeophysicalQuantity("temperature", "temperature", phx.units.KELVIN)
    fields = [
        op.OperatorFieldSpec("state", role="source", dimension=q.unit.dimension),
        op.OperatorFieldSpec(
            "increment", role="target", query_name="column", dimension=q.unit.dimension
        ),
    ]
    if control:
        fields.append(
            op.OperatorFieldSpec("forcing", role="source", dimension=q.unit.dimension)
        )
    task = op.OperatorTask(
        "column-test",
        fields=tuple(fields),
        dimension_basis=tuple(axis for axis, _, _ in q.unit.dimension.terms),
        queries=(
            op.OperatorQuerySpec(
                "column", geometry_kind="point_cloud", coordinate_components=("level",)
            ),
        ),
        problem=op.OperatorProblemSpec(
            source_query_relation="coincident", query_is_fixed=False
        ),
        metadata={"geophysical_target_kind": kind, "geophysical_step_seconds": 60.0},
    )
    return ColumnClosureBinding(
        task, (q,), ("state",), ("increment",), ("top", "bottom"), "pressure-test", 60.0
    )


def _records():
    return tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"c{i}",
            identities={
                "scenario": f"s{i // 4}",
                "model": "shared-model",
                "member": f"r{(i // 2) % 2}",
            },
        )
        for i in range(12)
    )


def _dataset(binding, *, forcing=None, forcing_bounds=None):
    before = 280.0 + np.arange(24).reshape(12, 2, 1)
    after = before + np.asarray([1.0, -0.5])[None, :, None]
    resolved = np.full_like(before, 0.25)
    return column_closure_dataset(
        binding,
        before,
        after,
        resolved,
        layer_mass=np.broadcast_to([1.0, 2.0], (12, 2)),
        interval_bounds=np.broadcast_to([0.0, 60.0], (12, 2)),
        forcing={} if forcing is None else forcing,
        forcing_bounds={} if forcing_bounds is None else forcing_bounds,
        provenance=_records(),
    )


def test_column_labels_subtract_finite_resolved_increment_and_keep_mass_order():
    binding = _column()
    data = _dataset(binding)
    np.testing.assert_allclose(
        data.targets.field("increment").values, np.broadcast_to([0.75, -0.75], (12, 2))
    )
    # Observable weighted closure budget, not a metadata forwarding assertion.
    total = phx.nn.operator.training.operator_integral(
        data.targets.field("increment").values,
        data.batch.query("column"),
        case_shape=(12,),
    )
    np.testing.assert_allclose(total, -0.75)
    with pytest.raises(ValueError, match="geophysical_target_kind"):
        _column(kind="instantaneous_rhs")
    with pytest.raises(ValueError, match="intervals"):
        replace(binding, interval_seconds=30.0)


def test_column_forcing_shift_is_rejected_even_when_duration_matches():
    binding = _column(control=True)
    samples = phx.nn.operator.FunctionSamples(
        values=jnp.ones((12, 2)),
        coordinates=jnp.asarray([[0.0], [1.0]]),
        quadrature_weights=jnp.ones(2),
    )
    _dataset(
        binding,
        forcing={"forcing": samples},
        forcing_bounds={"forcing": np.broadcast_to([0.0, 60.0], (12, 2))},
    )
    with pytest.raises(ValueError, match="align"):
        _dataset(
            binding,
            forcing={"forcing": samples},
            forcing_bounds={"forcing": np.broadcast_to([60.0, 120.0], (12, 2))},
        )


def test_same_model_simulation_holdout_and_train_only_normalization():
    binding = _column()
    experiment = GeophysicalLearningExperiment.prepare(binding.task, _dataset(binding))
    partitions = (
        experiment.split.train,
        experiment.split.validation,
        experiment.split.test,
    )
    groups = [
        {
            tuple(record.identities[key] for key in ("scenario", "model", "member"))
            for record in part.provenance
        }
        for part in partitions
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(
        {record.identities["model"] for record in part.provenance} == {"shared-model"}
        for part in partitions
    )
    op = phx.nn.operator
    model = op.architectures.DeepONet(
        branch={
            "state": phx.nn.models.MLP(
                in_size=2, out_size=4, width_size=4, depth=1, key=jr.key(1)
            )
        },
        trunk=phx.nn.models.MLP(
            in_size=1, out_size=4, width_size=4, depth=1, key=jr.key(2)
        ),
        coord_dim=1,
        latent_size=4,
    )
    fit = experiment.fit(
        model, steps=2, output_field_map={"output": "increment"}, jit=False
    )
    assert fit.completed_steps == 2
    normalized = fit.normalization.normalize_batch(experiment.split.train.batch)
    values = normalized.input("state").values
    weights = normalized.input("state").quadrature(case_shape=normalized.case_shape)
    np.testing.assert_allclose(
        jnp.sum(values * weights) / jnp.sum(weights), 0.0, atol=2e-5
    )
    # Holding out cases cannot pull the training normalization toward their mean.
    heldout = experiment.split.test.batch
    normalized_test = fit.normalization.normalize_batch(heldout).input("state").values
    train_physical = experiment.split.train.batch.input("state").values
    train_mean = jnp.sum(train_physical * weights) / jnp.sum(weights)
    test_weights = heldout.input("state").quadrature(case_shape=heldout.case_shape)
    test_mean = jnp.sum(heldout.input("state").values * test_weights) / jnp.sum(
        test_weights
    )
    normalized_mean = jnp.sum(normalized_test * test_weights) / jnp.sum(test_weights)
    assert float(normalized_mean) * float(test_mean - train_mean) > 0
    # A model-independent question still requires independent models explicitly.
    with pytest.raises(ValueError):
        GeophysicalLearningExperiment.prepare(
            binding.task,
            _dataset(binding),
            policy=op.training.OperatorSplitPolicy(group_by=("model",)),
        )


def test_declared_simulation_groups_reject_disjoint_cases_from_the_same_simulation():
    binding = _column()
    experiment = GeophysicalLearningExperiment.prepare(binding.task, _dataset(binding))
    train = experiment.split.train
    # Case IDs differ, but these two rows belong to the same simulation.
    split = replace(experiment.split, train=train.take([0]), validation=train.take([1]))
    with pytest.raises(ValueError, match="selected.*groups"):
        GeophysicalLearningExperiment(binding.task, split)


def test_explicit_chronological_window_split_accepts_same_simulation_but_rejects_overlap():
    binding = _column()
    dataset = _dataset(binding)
    records = tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"window-case:{i}",
            identities={
                "scenario": "baseline",
                "model": "shared-model",
                "member": "r1",
                "window": f"block:{i // 2}",
            },
            order={"start": 60.0 * i, "end": 60.0 * (i + 1)},
        )
        for i in range(dataset.size)
    )
    dataset = replace(dataset, provenance=records)
    policy = phx.nn.operator.training.OperatorSplitPolicy(
        group_by=("window",), order_by="start"
    )
    experiment = GeophysicalLearningExperiment.prepare(
        binding.task, dataset, policy=policy, temporal_bounds=("start", "end")
    )
    partitions = (
        experiment.split.train,
        experiment.split.validation,
        experiment.split.test,
    )
    for left, right in zip(partitions[:-1], partitions[1:], strict=True):
        assert max(record.order["end"] for record in left.provenance) <= min(
            record.order["start"] for record in right.provenance
        )
    assert all(
        {record.identities["member"] for record in part.provenance} == {"r1"}
        for part in partitions
    )
    # Start times still sort, but the last training target reaches into validation.
    index = experiment.split.train_indices[-1]
    overlapping = list(records)
    overlapping[index] = replace(
        overlapping[index],
        order={
            **overlapping[index].order,
            "end": min(
                record.order["start"] for record in experiment.split.validation.provenance
            )
            + 1.0,
        },
    )
    with pytest.raises(ValueError, match="temporal windows overlap"):
        GeophysicalLearningExperiment.prepare(
            binding.task,
            replace(dataset, provenance=tuple(overlapping)),
            policy=policy,
            temporal_bounds=("start", "end"),
        )


def test_budget_projection_reports_corrections_and_rejects_unphysical_state():
    binding = _column()
    before = jnp.asarray([[280.0], [280.0]])
    report = admit_column_closure(
        binding,
        before,
        jnp.asarray([[3.0], [0.0]]),
        layer_mass=jnp.asarray([1.0, 2.0]),
        target_budget=jnp.asarray([0.0]),
        domain_admission=lambda state: state > 150.0,
    )
    np.testing.assert_allclose(report.require_state(), [[282.0], [279.0]])
    np.testing.assert_allclose(report.raw_budget, [3.0])
    np.testing.assert_allclose(report.budget_correction, [-3.0])
    rejected = admit_column_closure(
        binding,
        before,
        jnp.asarray([[-600.0], [300.0]]),
        layer_mass=jnp.asarray([1.0, 2.0]),
        target_budget=jnp.asarray([0.0]),
        domain_admission=lambda state: state > 0.0,
    )
    assert not rejected.admitted
    np.testing.assert_allclose(rejected.increment, [[-600.0], [300.0]])
    with pytest.raises(ValueError, match="admission"):
        rejected.require_state()


def test_masked_weighted_metrics_keep_lead_members_and_physical_error_units():
    q = _column().quantities[0]
    truth = jnp.zeros((2, 1, 2, 1, 1))
    forecast = jnp.asarray(
        [[[2.0, np.nan], [4.0, np.nan]], [[3.0, 5.0], [3.0, 5.0]]]
    ).reshape(2, 2, 1, 2, 1, 1)
    mask = jnp.asarray([True, False, True, True]).reshape(truth.shape)
    result = geophysical_forecast_metrics(
        truth,
        forecast,
        quantities=(q,),
        lead_seconds=(60.0, 120.0),
        member_ids=("a", "b"),
        area_weights=[1.0, 3.0],
        layer_weights=[2.0],
        time_weights=[4.0],
        mask=mask,
    )
    field = result.fields[0]
    np.testing.assert_allclose(
        field.rmse.value, [[2.0, 4.0], [np.sqrt(21.0), np.sqrt(21.0)]]
    )
    np.testing.assert_allclose(field.crps.value, [2.5, 4.5])
    np.testing.assert_allclose(field.bias.value, [[2.0, 4.0], [4.5, 4.5]])
    np.testing.assert_allclose(field.drift_per_second.value, [2.5 / 60.0, 0.5 / 60.0])
    assert field.error_unit == phx.units.KELVIN
    assert np.all(np.asarray(field.rmse.valid))
    empty = geophysical_forecast_metrics(
        truth,
        forecast,
        quantities=(q,),
        lead_seconds=(60.0, 120.0),
        member_ids=("a", "b"),
        area_weights=[1.0, 3.0],
        layer_weights=[1.0],
        time_weights=[1.0],
        mask=jnp.zeros_like(mask),
    )
    assert not np.any(np.asarray(empty.fields[0].rmse.valid))


def test_anomaly_provenance_and_explicit_multivariate_scaling():
    q = _column().quantities[0]
    climatology = fit_geophysical_climatology(
        jnp.full((2, 2, 1, 1), 280.0),
        quantities=(q,),
        time_weights=[1.0, 1.0],
        mask=True,
        training_case_ids=("train-a", "train-b"),
        source_id="control-climate",
    )
    truth = jnp.asarray([279.0, 281.0]).reshape(1, 1, 2, 1, 1)
    ensemble = truth[:, None, ...] + 1.0
    kwargs = dict(
        quantities=(q,),
        lead_seconds=(60.0,),
        member_ids=("m",),
        area_weights=[1.0, 1.0],
        layer_weights=[1.0],
        time_weights=[1.0],
        mask=jnp.ones_like(truth, dtype=bool),
        climatology=climatology,
    )
    with pytest.raises(ValueError, match="independent"):
        geophysical_forecast_metrics(
            truth, ensemble, verification_case_ids=("train-a",), **kwargs
        )
    result = geophysical_forecast_metrics(
        truth,
        ensemble,
        verification_case_ids=("test",),
        multivariate_scales=(2.0,),
        **kwargs,
    )
    np.testing.assert_allclose(
        result.fields[0].anomaly_correlation.value, [[1 / np.sqrt(2)]]
    )
    np.testing.assert_allclose(result.scaled_energy_score.value, [0.5])
    assert result.climatology_id == climatology.climatology_id


def test_extreme_reliability_empty_bins_and_native_complex_spectral_error():
    q = _column().quantities[0]
    result = geophysical_extreme_reliability(
        [[0.0, 2.0]],
        [[[0.0, 0.0], [0.0, 2.0]]],
        quantity=q,
        threshold=1.0,
        sample_weights=[1.0, 3.0],
        mask=True,
        bin_edges=(0.0, 0.25, 0.75, 1.0),
    )
    np.testing.assert_allclose(result.brier.value, [0.1875])
    np.testing.assert_allclose(result.bin_frequency[0, :2], [0.0, 1.0])
    assert float(result.bin_weight[0, 2]) == 0 and np.isnan(result.bin_frequency[0, 2])
    spectral = geophysical_spectral_rmse(
        jnp.zeros((1, 2), dtype=complex),
        jnp.asarray([[1j, 3j]]),
        mode_weights=[3.0, 1.0],
        mask=True,
    )
    np.testing.assert_allclose(spectral.value, [np.sqrt(3.0)])


def test_real_trained_sfno_artifact_forecast_restart_and_column_admission(tmp_path):
    from examples.geophysical_operator_forecast import run_example

    result = run_example(steps=3, artifact_directory=tmp_path / "native-sfno")
    assert result["restart_max_error"] < 1e-6
    assert result["column_admitted"]
    assert result["column_budget_max_error"] < 0.05
    assert result["lead_seconds"] == (3600.0, 7200.0, 10800.0, 14400.0)
    assert np.all(np.isfinite(result["rmse_kelvin"]))

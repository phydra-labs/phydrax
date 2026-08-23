#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _dataset(cases=10, resolution=8):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.full((resolution,), 1.0 / resolution),
    )
    offsets = jnp.arange(cases, dtype=float)[:, None]
    values = offsets + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _targets(batch, values):
    return phx.nn.operator.OperatorTargetBatch.from_arrays(
        {"solution": values},
        batch,
    )


def test_normalization_is_training_only_invertible_and_persisted(tmp_path):
    dataset = _dataset()
    split = phx.nn.operator.training.split_operator_dataset(
        dataset,
        policy=phx.nn.operator.training.OperatorSplitPolicy(seed=7),
    )
    policy = phx.nn.operator.training.fit_operator_normalization(
        split.train.batch,
        split.train.targets,
        normalize_coordinates=True,
    )
    train_state = split.train.batch.input("state")
    assert train_state.values is not None
    expected_mean = jnp.mean(train_state.values)
    assert jnp.allclose(policy.input_values["state"].mean, expected_mean)

    normalized = policy.normalize_batch(split.validation.batch)
    restored = policy.denormalize_batch(normalized)
    restored_state = restored.input("state")
    validation_state = split.validation.batch.input("state")
    assert restored_state.values is not None
    assert validation_state.values is not None
    assert jnp.allclose(
        restored_state.values,
        validation_state.values,
    )
    assert jnp.allclose(
        restored.input("state").axes[0].nodes,
        split.validation.batch.input("state").axes[0].nodes,
    )
    normalized_targets = policy.normalize_targets(split.validation.targets)
    restored_targets = policy.denormalize_targets(normalized_targets)
    assert jnp.allclose(
        restored_targets.field("solution").values,
        split.validation.targets.field("solution").values,
    )

    path = phx.nn.operator.training.save_operator_normalization(
        tmp_path / "normalization.json", policy
    )
    loaded = phx.nn.operator.training.load_operator_normalization(path)
    assert loaded.to_dict() == policy.to_dict()
    assert "format_version" not in policy.to_dict()


def test_quadrature_normalization_is_invariant_to_sampling_density():
    def sampled_batch(values, weights):
        count = int(values.shape[0])
        coordinates = jnp.linspace(0.0, 1.0, count)[:, None]
        samples = phx.nn.operator.FunctionSamples(
            values=values,
            coordinates=coordinates,
            quadrature_weights=weights,
        )
        return phx.nn.operator.OperatorBatch(
            inputs={"state": samples},
            queries={
                "query": phx.nn.operator.FunctionSamples(
                    values=None,
                    coordinates=coordinates,
                    quadrature_weights=weights,
                )
            },
        )

    sparse_values = jnp.array([0.0, 1.0])
    dense_values = jnp.array([0.0, 0.0, 0.0, 1.0])
    sparse = sampled_batch(sparse_values, jnp.array([0.5, 0.5]))
    dense = sampled_batch(
        dense_values,
        jnp.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5]),
    )

    sparse_quadrature = phx.nn.operator.training.fit_operator_normalization(
        sparse, _targets(sparse, sparse_values), weighting="quadrature"
    )
    dense_quadrature = phx.nn.operator.training.fit_operator_normalization(
        dense, _targets(dense, dense_values), weighting="quadrature"
    )
    sparse_uniform = phx.nn.operator.training.fit_operator_normalization(
        sparse, _targets(sparse, sparse_values), weighting="uniform"
    )
    dense_uniform = phx.nn.operator.training.fit_operator_normalization(
        dense, _targets(dense, dense_values), weighting="uniform"
    )

    for sparse_statistic, dense_statistic in (
        (
            sparse_quadrature.input_values["state"],
            dense_quadrature.input_values["state"],
        ),
        (
            sparse_quadrature.targets["solution"],
            dense_quadrature.targets["solution"],
        ),
    ):
        assert jnp.allclose(sparse_statistic.mean, dense_statistic.mean)
        assert jnp.allclose(sparse_statistic.scale, dense_statistic.scale)
    assert not jnp.allclose(
        sparse_uniform.input_values["state"].mean,
        dense_uniform.input_values["state"].mean,
    )
    assert not jnp.allclose(
        sparse_uniform.targets["solution"].scale,
        dense_uniform.targets["solution"].scale,
    )


def test_dataset_splitting_and_variable_cardinality_adapter_are_deterministic():
    dataset = _dataset(cases=12)
    policy = phx.nn.operator.training.OperatorSplitPolicy(seed=3)
    first = phx.nn.operator.training.split_operator_dataset(dataset, policy=policy)
    second = phx.nn.operator.training.split_operator_dataset(dataset, policy=policy)
    assert first.train_indices == second.train_indices
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert set(first.validation_indices).isdisjoint(first.test_indices)

    cases = []
    targets = []
    for count in (3, 5, 4):
        coordinates = jnp.linspace(0.0, 1.0, count)[:, None]
        samples = phx.nn.operator.FunctionSamples(
            values=coordinates[:, 0],
            coordinates=coordinates,
        )
        batch = phx.nn.operator.OperatorBatch(
            inputs={"state": samples},
            queries={
                "query": phx.nn.operator.FunctionSamples(
                    values=None,
                    coordinates=coordinates,
                )
            },
        )
        cases.append(batch)
        targets.append(_targets(batch, coordinates[:, 0] ** 2))
    ragged = phx.nn.operator.training.operator_dataset_from_cases(cases, targets)
    assert ragged.batch.query("query").sample_shape == (5,)
    assert ragged.targets.field("solution").values.shape == (3, 5)
    assert jnp.array_equal(
        ragged.batch.query("query").mask_array(case_shape=(3,)),
        jnp.asarray(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, True, True, False],
            ]
        ),
    )


def test_provenance_group_and_chronological_splits_prevent_leakage():
    base = _dataset(cases=12)
    grouped_provenance = tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"case-{index}",
            identities={"simulation": f"simulation-{index // 2}"},
            order={"time": float(index)},
        )
        for index in range(12)
    )
    grouped = phx.nn.operator.training.OperatorDataset(
        base.batch,
        base.targets,
        grouped_provenance,
    )
    split = phx.nn.operator.training.split_operator_dataset(
        grouped,
        train_fraction=0.5,
        validation_fraction=0.25,
        policy=phx.nn.operator.training.OperatorSplitPolicy(
            group_by=("simulation",),
            seed=19,
        ),
    )
    partitions = (
        set(split.train_indices),
        set(split.validation_indices),
        set(split.test_indices),
    )
    for simulation in range(6):
        members = {2 * simulation, 2 * simulation + 1}
        assert sum(bool(members & partition) for partition in partitions) == 1

    chronological = phx.nn.operator.training.OperatorDataset(
        base.batch,
        base.targets,
        tuple(
            phx.nn.operator.OperatorCaseProvenance(
                f"ordered-{index}",
                order={"time": float(index)},
            )
            for index in range(12)
        ),
    )
    ordered = phx.nn.operator.training.split_operator_dataset(
        chronological,
        train_fraction=0.5,
        validation_fraction=0.25,
        policy=phx.nn.operator.training.OperatorSplitPolicy(group_by=(), order_by="time"),
    )
    assert max(ordered.train_indices) < min(ordered.validation_indices)
    assert max(ordered.validation_indices) < min(ordered.test_indices)


def test_dataset_preserves_named_multi_query_target_contracts():
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 3))
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(
                values=jnp.arange(6.0).reshape(2, 3),
                axes=(axis,),
            )
        },
        queries={
            "state-query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
            "flux-query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.linspace(0.0, 1.0, 4)[:, None],
            ),
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {
            "state": jnp.ones((2, 3)),
            "flux": jnp.ones((2, 4, 2)),
        },
        batch,
        query_names={
            "state": "state-query",
            "flux": "flux-query",
        },
        specs={
            "state": phx.nn.operator.OperatorOutputSpec(),
            "flux": phx.nn.operator.OperatorOutputSpec(
                2,
                component_names=("x", "y"),
            ),
        },
    )
    dataset = phx.nn.operator.training.OperatorDataset(batch, targets)
    selected = dataset.take(jnp.array([1]))
    assert tuple(selected.targets.fields) == ("state", "flux")
    assert selected.targets.field("state").values.shape == (1, 3)
    assert selected.targets.field("flux").values.shape == (1, 4, 2)
    assert selected.targets.field("flux").query_name == "flux-query"
    assert selected.targets.field("flux").spec.component_names == ("x", "y")


def test_named_normalization_and_dtype_preserve_complex_fields(tmp_path):
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 3))
    complex_values = jnp.arange(6.0).reshape(2, 3) + 1j * jnp.arange(6.0, 12.0).reshape(
        2, 3
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "wave": phx.nn.operator.FunctionSamples(
                values=complex_values,
                axes=(axis,),
            )
        },
        queries={
            "wave-query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
            "sensor-query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.array([[0.1], [0.9]]),
            ),
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {
            "wave": 2.0 * complex_values,
            "sensor": jnp.arange(8.0).reshape(2, 2, 2),
        },
        batch,
        query_names={
            "wave": "wave-query",
            "sensor": "sensor-query",
        },
    )
    policy = phx.nn.operator.training.fit_operator_normalization(
        batch,
        targets,
        normalize_coordinates=True,
    )
    normalized = policy.normalize_targets(targets)
    restored = policy.denormalize_targets(normalized)
    assert tuple(policy.targets) == ("wave", "sensor")
    assert tuple(policy.query_coordinates) == ("wave-query", "sensor-query")
    assert jnp.allclose(
        restored.field("wave").values,
        targets.field("wave").values,
    )
    assert jnp.allclose(
        restored.field("sensor").values,
        targets.field("sensor").values,
    )
    loaded = phx.nn.operator.training.load_operator_normalization(
        phx.nn.operator.training.save_operator_normalization(
            tmp_path / "complex.json", policy
        )
    )
    assert jnp.allclose(
        loaded.targets["wave"].mean,
        policy.targets["wave"].mean,
    )

    dtype = phx.nn.operator.training.OperatorDTypePolicy(compute_dtype="float32")
    cast_targets = dtype.cast_targets(targets)
    cast_batch = dtype.cast_batch(batch)
    cast_wave = cast_batch.input("wave")
    assert cast_wave.values is not None
    assert cast_targets.field("wave").values.dtype == jnp.complex64
    assert cast_targets.field("sensor").values.dtype == jnp.float32
    assert cast_wave.values.dtype == jnp.complex64


def _stochastic_step(model, state, optimizer, key):
    key, sample_key = jr.split(key)
    x = jr.normal(sample_key, (6, 1))
    y = 1.7 * x - 0.2

    def loss(current):
        prediction = jax.vmap(current)(x)
        return jnp.mean((prediction - y) ** 2)

    _, gradients = eqx.filter_value_and_grad(loss)(model)
    updates, state = optimizer.update(gradients, state, model)
    return eqx.apply_updates(model, updates), state, key


def _assert_trees_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(eqx.filter(left, eqx.is_array))
    right_leaves = jax.tree_util.tree_leaves(eqx.filter(right, eqx.is_array))
    assert len(left_leaves) == len(right_leaves)
    assert all(jnp.array_equal(a, b) for a, b in zip(left_leaves, right_leaves))


def test_checkpoint_restores_exact_optimizer_rng_and_policies(tmp_path):
    model = eqx.nn.Linear(1, 1, key=jr.key(1))
    optimizer = optax.adam(1e-2)
    state = optimizer.init(eqx.filter(model, eqx.is_array))
    key = jr.key(9)
    model, state, key = _stochastic_step(model, state, optimizer, key)

    dataset = _dataset(cases=4)
    normalization = phx.nn.operator.training.fit_operator_normalization(
        dataset.batch,
        dataset.targets,
    )
    dtype_policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float64",
        compute_dtype="float32",
        reduction_dtype="float64",
    )
    schema = phx.nn.operator.training.operator_batch_schema(
        dataset.batch, target=dataset.targets
    )
    path = phx.nn.operator.training.save_operator_training_checkpoint(
        tmp_path / "checkpoint",
        model,
        state,
        step=1,
        key=key,
        normalization=normalization,
        dtype_policy=dtype_policy,
        schema=schema,
        metadata={"dataset": "manufactured"},
    )
    published_state = next(path.glob("state-*.eqx"))
    stale_bytes = published_state.read_bytes()
    (path / "state-orphan.eqx").write_bytes(stale_bytes)
    (path / "state.tmp.eqx").write_bytes(stale_bytes)
    expected_model, expected_state, expected_key = _stochastic_step(
        model, state, optimizer, key
    )

    restored = phx.nn.operator.training.load_operator_training_checkpoint(
        path,
        model,
        state,
        expected_schema=schema,
    )
    actual_model, actual_state, actual_key = _stochastic_step(
        restored.model,
        restored.optimizer_state,
        optimizer,
        restored.key,
    )
    _assert_trees_equal(actual_model, expected_model)
    _assert_trees_equal(actual_state, expected_state)
    assert jnp.array_equal(jr.key_data(actual_key), jr.key_data(expected_key))
    assert restored.normalization is not None
    assert restored.normalization.to_dict() == normalization.to_dict()
    assert restored.dtype_policy == dtype_policy
    assert restored.metadata == {"dataset": "manufactured"}
    assert not (path / "state-orphan.eqx").exists()
    assert not (path / "state.tmp.eqx").exists()

    first_state_files = tuple(path.glob("state-*.eqx"))
    assert len(first_state_files) == 1
    phx.nn.operator.training.save_operator_training_checkpoint(
        path,
        expected_model,
        expected_state,
        step=2,
        key=expected_key,
        normalization=normalization,
        dtype_policy=dtype_policy,
        schema=schema,
        metadata={"dataset": "manufactured"},
    )
    second_state_files = tuple(path.glob("state-*.eqx"))
    assert len(second_state_files) == 1
    assert second_state_files != first_state_files
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "phydrax-operator-training-checkpoint"
    assert manifest["version"] == 3
    manifest.pop("version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.training.load_operator_training_checkpoint(
            path,
            expected_model,
            expected_state,
            expected_schema=schema,
        )


class _IncrementOperator:
    def __call__(self, batch, *, key=None):
        del key
        return batch.input("state").values + 1.0


def _advance(batch, feedback, step):
    del step
    source = batch.input("state")
    updated = phx.nn.operator.FunctionSamples(
        values=feedback,
        axes=source.axes,
        coordinates=source.coordinates,
        quadrature_weights=source.quadrature_weights,
        mask=source.mask,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"state": updated},
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def test_autoregressive_rollout_curricula_and_gradients():
    dataset = _dataset(cases=3, resolution=5)
    initial = phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(
                values=jnp.zeros_like(dataset.batch.input("state").values),
                axes=dataset.batch.input("state").axes,
            )
        },
        queries={"query": dataset.batch.query("query")},
        case_axes=dataset.batch.case_axes,
        case_shape=dataset.batch.case_shape,
    )
    targets = jnp.stack(
        tuple(jnp.full((3, 5), value) for value in (1.0, 2.0, 3.0)),
        axis=0,
    )
    rollout = phx.nn.operator.training.autoregressive_operator_rollout(
        _IncrementOperator(),
        initial,
        3,
        _advance,
        key=jr.key(0),
    )
    assert jnp.array_equal(rollout.predictions, targets)
    schedule = phx.nn.operator.training.RolloutHorizonSchedule(1, 3, transition_steps=10)
    forcing = phx.nn.operator.training.TeacherForcingSchedule(
        1.0, 0.0, transition_steps=10
    )
    loss = phx.nn.operator.training.autoregressive_operator_loss(
        _IncrementOperator(),
        initial,
        targets,
        _advance,
        training_step=10,
        horizon=schedule,
        teacher_forcing=forcing,
        key=jr.key(1),
    )
    assert loss == 0.0


def test_dtype_and_prefetch_loader_apply_explicit_device_policy():
    dataset = _dataset(cases=8)
    dtype_policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float32",
        compute_dtype="float32",
        reduction_dtype="float64",
    )
    sharding = phx.nn.operator.OperatorShardingPolicy(mesh_axis="data")
    loader = phx.nn.operator.training.OperatorBatchLoader(
        dataset,
        batch_size=3,
        seed=11,
        prefetch=2,
        dtype_policy=dtype_policy,
        sharding_policy=sharding,
    )
    first = tuple(loader.epoch(2))
    second = tuple(loader.epoch(2))
    assert tuple(item.indices for item in first) == tuple(item.indices for item in second)
    first_state = first[0].batch.input("state")
    assert first_state.values is not None
    assert isinstance(first_state.values.sharding, jax.sharding.NamedSharding)
    assert first_state.values.dtype == jnp.float32
    assert first[0].targets.field("solution").values.dtype == jnp.float32
    assert first_state.values.sharding.spec[0] == "data"

    model = phx.nn.operator.architectures.FNO(
        width=4, depth=1, n_modes=(3,), key=jr.key(3)
    )
    cast_model = dtype_policy.cast_model(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(cast_model, eqx.is_inexact_array))
    assert leaves and all(
        leaf.dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))
        for leaf in leaves
    )


def _prediction_energy(
    prediction,
    batch,
    targets,
    *,
    model,
    key,
    step,
    training,
    context,
):
    del batch, targets, model, key, step, training
    values = prediction.field("output").values
    assert context.physical_batch.case_shape == values.shape[:1]
    return jnp.mean(values**2)


def _fit_model(*, seed=0):
    return phx.nn.operator.architectures.FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(seed),
    )


def test_fit_operator_compiles_accumulates_normalizes_and_composes_losses():
    dataset = _dataset(cases=8)
    result = phx.nn.operator.training.fit_operator(
        _fit_model(),
        dataset,
        epochs=1,
        batch_size=2,
        gradient_accumulation=2,
        normalization="fit",
        loss_terms=(
            phx.nn.operator.training.SupervisedOperatorLoss(
                prediction_field="output",
                target_field="solution",
            ),
            phx.nn.operator.training.OperatorLossTerm(
                "prediction_energy",
                _prediction_energy,
                weight=1e-3,
                identity="tests.prediction_energy.v1",
            ),
        ),
    )

    assert result.completed_steps == 2
    assert result.normalization is not None
    assert result.history.train_steps == (1, 2)
    assert all(
        set(metrics) == {"loss", "supervised_l2", "prediction_energy"}
        for metrics in result.history.train_metrics
    )
    assert jnp.isfinite(result.final_loss)


def test_fit_operator_resume_is_bitwise_exact_with_shuffle_and_accumulation(tmp_path):
    dataset = _dataset(cases=8)
    model = _fit_model(seed=2)
    common: dict[str, Any] = {
        "batch_size": 2,
        "gradient_accumulation": 2,
        "seed": 17,
        "checkpoint_every": 1,
        "configuration": {"test_contract": "exact-resume-v1"},
    }
    uninterrupted = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        epochs=2,
        steps=4,
        **common,
    )
    checkpoint = tmp_path / "fit-checkpoint"
    phx.nn.operator.training.fit_operator(
        model,
        dataset,
        epochs=1,
        steps=2,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        epochs=2,
        steps=4,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    full_leaves = jax.tree_util.tree_leaves(uninterrupted.last_execution_model)
    resumed_leaves = jax.tree_util.tree_leaves(resumed.last_execution_model)
    for full, restored in zip(full_leaves, resumed_leaves, strict=True):
        if isinstance(full, jax.Array):
            assert jnp.array_equal(full, restored)
    assert resumed.resumed_from_step == 2
    assert resumed.history == uninterrupted.history


def test_fit_operator_returns_task_bound_physical_operator():
    dataset = _dataset(cases=4)
    task = phx.nn.operator.OperatorTask(
        "scaled-map",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="state",
                physical_dimension=(0.0,),
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="query",
                physical_dimension=(0.0,),
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )
    output_pipeline = phx.nn.operator.training.OperatorOutputPipeline(
        phx.nn.operator.training.ConservationProjection("solution", source_name="state")
    )
    result = phx.nn.operator.training.fit_operator(
        _fit_model(seed=4),
        dataset,
        task=task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "solution"},
        epochs=1,
        steps=1,
        batch_size=4,
        output_pipeline=output_pipeline,
        normalization="fit",
    )

    assert result.trained_operator is not None
    prediction = result.trained_operator.predict(dataset.batch)
    assert prediction.field("solution").values.shape == (4, 8)
    assert jnp.all(jnp.isfinite(prediction.field("solution").values))
    assert result.output_pipeline is output_pipeline
    assert result.trained_operator.output_pipeline is output_pipeline
    assert jnp.allclose(
        phx.nn.operator.training.operator_integral(
            prediction.field("solution").values,
            dataset.batch.query("query"),
            case_shape=dataset.batch.case_shape,
        ),
        phx.nn.operator.training.operator_integral(
            dataset.batch.input("state").values,
            dataset.batch.input("state"),
            case_shape=dataset.batch.case_shape,
        ),
    )


def test_fit_operator_callbacks_can_stop_at_an_update_boundary():
    events = []

    def stop_after_first(event):
        events.append(event.name)
        return event.name == "batch_end"

    result = phx.nn.operator.training.fit_operator(
        _fit_model(seed=5),
        _dataset(cases=8),
        epochs=4,
        steps=8,
        batch_size=4,
        callbacks=(stop_after_first,),
    )

    assert result.completed_steps == 1
    assert result.stopped_by_callback
    assert events[0] == "train_begin"
    assert "batch_end" in events
    assert events[-1] == "train_end"

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


class _LinearOperator(phx.nn.operator.AbstractOperatorModel):
    weight: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, weight=1.0):
        self.weight = jnp.asarray([[weight]], dtype=jnp.float32)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("state").values
        assert values is not None
        return (values[..., None] @ self.weight)[..., 0]

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _dataset(cases=4, resolution=8):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.linspace(1.0, 2.0, resolution),
    )
    values = jnp.arange(cases, dtype=float)[:, None] + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"output": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def test_dtype_policy_casts_values_without_quantizing_geometry():
    dataset = _dataset()
    original = dataset.batch.input("state")
    policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        reduction_dtype="float32",
    )

    cast_batch = policy.cast_batch(dataset.batch)
    cast_targets = policy.cast_targets(dataset.targets)
    cast_samples = cast_batch.input("state")
    original_axis = original.axes[0]
    cast_axis = cast_samples.axes[0]

    assert cast_samples.values is not None
    assert cast_samples.values.dtype == jnp.bfloat16
    assert cast_targets.field("output").values.dtype == jnp.bfloat16
    assert jnp.array_equal(cast_axis.nodes, original_axis.nodes)
    assert jnp.array_equal(
        cast_axis.quadrature_weights,
        original_axis.quadrature_weights,
    )
    assert cast_axis.nodes.dtype == original_axis.nodes.dtype
    assert cast_axis.quadrature_weights.dtype == original_axis.quadrature_weights.dtype


def test_point_geometry_and_topology_are_not_compute_cast():
    coordinates = jnp.asarray(
        [[[0.0, 0.1], [0.3, 0.7], [1.0, 0.9]]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([[0.2, 0.3, 0.5]], dtype=jnp.float64)
    samples = phx.nn.operator.FunctionSamples(
        values=jnp.ones((1, 3), dtype=jnp.float32),
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=jnp.asarray([[True, True, False]]),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"state": samples},
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=coordinates,
                quadrature_weights=weights,
                mask=samples.mask,
            )
        },
        case_axes=("case",),
        case_shape=(1,),
    )
    policy = phx.nn.operator.training.OperatorDTypePolicy(compute_dtype="float16")

    cast = policy.cast_batch(batch)
    cast_samples = cast.input("state")

    assert cast_samples.values.dtype == jnp.float16
    assert jnp.array_equal(cast_samples.coordinates, samples.coordinates)
    assert jnp.array_equal(cast_samples.quadrature_weights, samples.quadrature_weights)
    assert jnp.array_equal(cast_samples.mask, samples.mask)


def test_persistent_parameters_and_gradients_remain_float32():
    policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        reduction_dtype="float32",
    )
    storage_model = policy.cast_model(_LinearOperator())
    compute_model = policy.compute_model(storage_model)

    assert storage_model.weight.dtype == jnp.float32
    assert compute_model.weight.dtype == jnp.bfloat16

    inputs = jnp.ones((4, 1), dtype=jnp.bfloat16)

    def objective(weight):
        compute_weight = policy.cast_compute_parameters(weight)
        return jnp.sum(inputs @ compute_weight)

    gradient = jax.grad(objective)(storage_model.weight)
    jaxpr = str(jax.make_jaxpr(objective)(storage_model.weight))

    assert gradient.dtype == jnp.float32
    assert "convert_element_type" in jaxpr
    assert "bf16" in jaxpr


def test_dtype_policy_serializes_effective_complex_precision():
    policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float64",
        compute_dtype="bfloat16",
        reduction_dtype="float64",
        matmul_precision="BF16_BF16_F32",
    )
    restored = phx.nn.operator.training.OperatorDTypePolicy.from_dict(policy.to_dict())
    evidence = policy.precision_evidence
    restored_evidence = phx.nn.operator.training.OperatorPrecisionEvidence.from_dict(
        evidence.to_dict()
    )

    assert restored == policy
    assert restored_evidence == evidence
    assert evidence.parameter_complex_dtype == "complex128"
    assert evidence.compute_complex_dtype == "complex64"
    assert evidence.geometry_mode == "preserve"

    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.training.OperatorDTypePolicy.from_dict(
            {
                "parameter_dtype": "float32",
                "compute_dtype": "float32",
                "reduction_dtype": "float32",
            }
        )
    with pytest.raises(ValueError, match="requires compute_dtype"):
        phx.nn.operator.training.OperatorDTypePolicy(
            compute_dtype="float32",
            matmul_precision="BF16_BF16_F32",
        )


def test_fno_bfloat16_values_preserve_uniform_physical_grid():
    dataset = _dataset(cases=1)
    policy = phx.nn.operator.training.OperatorDTypePolicy(
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        reduction_dtype="float32",
    )
    model = policy.compute_model(
        phx.nn.operator.architectures.FNO(
            width=4,
            depth=1,
            n_modes=(3,),
            coordinate_embedding=False,
            key=jr.key(0),
        )
    )
    batch = policy.cast_batch(dataset.batch)

    output = model(batch)

    assert jnp.all(jnp.isfinite(output))
    assert output.dtype == jnp.bfloat16
    assert jnp.array_equal(
        batch.input("state").axes[0].nodes,
        dataset.batch.input("state").axes[0].nodes,
    )


def test_loss_scale_state_transitions_are_jittable_and_complete():
    policy = phx.nn.operator.training.OperatorLossScalePolicy(
        initial_scale=8.0,
        growth_interval=2,
        growth_factor=2.0,
        backoff_factor=0.5,
        minimum_scale=2.0,
        maximum_scale=16.0,
    )
    state = policy.initial_state(jnp.float32)
    finite = jax.jit(policy.on_finite_update)
    nonfinite = jax.jit(policy.on_nonfinite_microstep)

    state = finite(state)
    assert float(state.scale) == 8.0
    assert int(state.consecutive_finite_updates) == 1
    state = finite(state)
    assert float(state.scale) == 16.0
    assert int(state.consecutive_finite_updates) == 0
    state = finite(finite(state))
    assert float(state.scale) == 16.0
    state = nonfinite(state)
    assert float(state.scale) == 8.0
    assert int(state.consecutive_finite_updates) == 0
    assert int(state.nonfinite_microsteps) == 1

    roundtrip = jax.tree_util.tree_map(lambda value: value, state)
    assert jnp.array_equal(roundtrip.scale, state.scale)
    assert jnp.array_equal(
        roundtrip.consecutive_finite_updates,
        state.consecutive_finite_updates,
    )
    assert jnp.array_equal(roundtrip.nonfinite_microsteps, state.nonfinite_microsteps)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"initial_scale": jnp.inf}, "finite"),
        ({"growth_factor": jnp.inf}, "finite"),
        ({"backoff_factor": 1.0}, "strictly between"),
        ({"minimum_scale": 0.0}, "bounds"),
        ({"growth_interval": 0}, "growth_interval"),
    ),
)
def test_loss_scale_policy_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        phx.nn.operator.training.OperatorLossScalePolicy(**kwargs)


def _overflow_first_batch(
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
    del model, key, step, training, context
    values = prediction.field("output").values
    target = targets.field("output").values
    source = batch.input("state").values
    assert source is not None
    loss = jnp.mean((values - target) ** 2)
    return jnp.where(jnp.mean(source) < 1.0, jnp.inf, loss)


def test_float16_fit_backs_off_and_keeps_float32_master_parameters():
    dataset = _dataset(cases=2, resolution=4)
    result = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=1,
        steps=1,
        batch_size=1,
        shuffle=False,
        loss_terms=(
            phx.nn.operator.training.OperatorLossTerm(
                "overflow_first_batch",
                _overflow_first_batch,
                identity="tests.operator_precision.overflow-first.v1",
            ),
        ),
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
            parameter_dtype="float32",
            compute_dtype="float16",
            reduction_dtype="float32",
        ),
        loss_scale_policy=phx.nn.operator.training.OperatorLossScalePolicy(
            initial_scale=8.0,
            growth_interval=10,
            backoff_factor=0.5,
        ),
    )

    assert result.completed_steps == 1
    assert result.execution_model.weight.dtype == jnp.float32
    assert result.loss_scale_state is not None
    assert float(result.loss_scale_state.scale) == 4.0
    assert int(result.loss_scale_state.nonfinite_microsteps) == 1
    assert result.precision_evidence.compute_dtype == "float16"


def test_training_precision_combinations_are_explicit():
    dataset = _dataset(cases=1, resolution=4)

    with pytest.raises(ValueError, match="requires an explicit loss_scale_policy"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(),
            dataset,
            steps=0,
            dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
                compute_dtype="float16"
            ),
        )
    with pytest.raises(ValueError, match="supported only for float16"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(),
            dataset,
            steps=0,
            dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
                compute_dtype="bfloat16"
            ),
            loss_scale_policy=phx.nn.operator.training.OperatorLossScalePolicy(),
        )
    with pytest.raises(ValueError, match="persistent parameters"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(),
            dataset,
            steps=0,
            dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
                parameter_dtype="float16",
                compute_dtype="float16",
            ),
            loss_scale_policy=phx.nn.operator.training.OperatorLossScalePolicy(),
        )


def _overflow_second_batch(
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
    del model, key, step, training, context
    values = prediction.field("output").values
    target = targets.field("output").values
    source = batch.input("state").values
    assert source is not None
    reduction_axes = tuple(range(1, values.ndim))
    source_axes = tuple(range(1, source.ndim))
    case_loss = jnp.mean((values - target) ** 2, axis=reduction_axes)
    source_mean = jnp.mean(source, axis=source_axes)
    overflow = (source_mean > 1.0) & (source_mean < 2.0)
    return jnp.where(overflow, jnp.inf, case_loss)


def _mixed_precision_fit_kwargs():
    return {
        "batch_size": 1,
        "shuffle": False,
        "optimizer": optax.sgd(1e-2),
        "optimizer_id": "optax.sgd/0.01",
        "dtype_policy": phx.nn.operator.training.OperatorDTypePolicy(
            parameter_dtype="float32",
            compute_dtype="float16",
            reduction_dtype="float32",
        ),
        "loss_scale_policy": phx.nn.operator.training.OperatorLossScalePolicy(
            initial_scale=8.0,
            growth_interval=10,
            backoff_factor=0.5,
        ),
    }


def test_nonfinite_microbatch_discards_the_complete_accumulation_window():
    dataset = _dataset(cases=4, resolution=4)
    term = phx.nn.operator.training.OperatorLossTerm(
        "overflow_second_batch",
        _overflow_second_batch,
        identity="tests.operator_precision.overflow-second.v1",
        case_reduction="per_case",
    )
    mixed = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=1,
        steps=1,
        gradient_accumulation=2,
        loss_terms=(term,),
        **_mixed_precision_fit_kwargs(),
    )
    reference = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset.take(jnp.asarray([2, 3])),
        epochs=1,
        steps=1,
        gradient_accumulation=2,
        loss_terms=(term,),
        **_mixed_precision_fit_kwargs(),
    )

    assert jnp.array_equal(
        mixed.execution_model.weight,
        reference.execution_model.weight,
    )
    assert mixed.loss_scale_state is not None
    assert int(mixed.loss_scale_state.nonfinite_microsteps) == 1


def test_dynamic_loss_scale_resume_is_bitwise_exact(tmp_path):
    dataset = _dataset(cases=2, resolution=4)
    term = phx.nn.operator.training.OperatorLossTerm(
        "overflow_first_batch",
        _overflow_first_batch,
        identity="tests.operator_precision.resume-overflow.v1",
    )
    common = {
        **_mixed_precision_fit_kwargs(),
        "loss_terms": (term,),
        "checkpoint_every": 1,
        "configuration": {"test_contract": "operator-loss-scale-resume-v1"},
    }
    uninterrupted = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=2,
        steps=2,
        **common,
    )
    checkpoint = tmp_path / "precision-checkpoint"
    phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=1,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=2,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    assert jnp.array_equal(
        uninterrupted.execution_model.weight,
        resumed.execution_model.weight,
    )
    assert uninterrupted.history == resumed.history
    assert uninterrupted.progress == resumed.progress
    assert uninterrupted.loss_scale_state is not None
    assert resumed.loss_scale_state is not None
    assert jnp.array_equal(
        uninterrupted.loss_scale_state.scale,
        resumed.loss_scale_state.scale,
    )
    assert jnp.array_equal(
        uninterrupted.loss_scale_state.consecutive_finite_updates,
        resumed.loss_scale_state.consecutive_finite_updates,
    )
    assert jnp.array_equal(
        uninterrupted.loss_scale_state.nonfinite_microsteps,
        resumed.loss_scale_state.nonfinite_microsteps,
    )


def _nonfinite_optimizer():
    def init_fn(parameters):
        del parameters
        return optax.EmptyState()

    def update_fn(updates, state, parameters=None):
        del parameters
        return (
            jax.tree_util.tree_map(
                lambda update: jnp.full_like(update, jnp.inf),
                updates,
            ),
            state,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def test_nonfinite_optimizer_candidate_is_not_treated_as_scale_overflow():
    with pytest.raises(FloatingPointError, match="optimizer produced non-finite"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(),
            _dataset(cases=1, resolution=4),
            epochs=1,
            steps=1,
            batch_size=1,
            optimizer=_nonfinite_optimizer(),
            optimizer_id="tests.nonfinite-optimizer.v1",
        )


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Float64 accumulator coverage requires JAX x64.",
)
def test_float64_accumulation_casts_back_to_float32_optimizer_boundary():
    observed_dtypes = []

    def init_fn(parameters):
        del parameters
        return optax.EmptyState()

    def update_fn(updates, state, parameters=None):
        del parameters
        observed_dtypes.extend(
            leaf.dtype
            for leaf in jax.tree_util.tree_leaves(updates)
            if isinstance(leaf, jax.Array)
        )
        return (
            jax.tree_util.tree_map(lambda update: -1e-2 * update, updates),
            state,
        )

    phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        _dataset(cases=2, resolution=4),
        epochs=1,
        steps=1,
        batch_size=1,
        gradient_accumulation=2,
        optimizer=optax.GradientTransformation(init_fn, update_fn),
        optimizer_id="tests.record-gradient-dtype",
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
            parameter_dtype="float32",
            compute_dtype="float32",
            reduction_dtype="float64",
        ),
        jit=False,
        shuffle=False,
    )

    assert observed_dtypes
    assert set(observed_dtypes) == {jnp.dtype(jnp.float32)}


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="Global finite consensus requires at least two JAX devices.",
)
def test_sharded_overflow_produces_one_replicated_skip_decision():
    dataset = _dataset(cases=4, resolution=4)

    def overflow_first_shard(
        prediction,
        batch,
        targets,
        **kwargs,
    ):
        del kwargs
        values = prediction.field("output").values
        target = targets.field("output").values
        source = batch.input("state").values
        assert source is not None
        loss = jnp.mean((values - target) ** 2)
        return jnp.where(jnp.any(source < 1.0), jnp.inf, loss)

    result = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        epochs=1,
        steps=1,
        batch_size=2,
        shuffle=False,
        loss_terms=(
            phx.nn.operator.training.OperatorLossTerm(
                "overflow_first_shard",
                overflow_first_shard,
                identity="tests.operator_precision.sharded-overflow.v1",
            ),
        ),
        sharding_policy=phx.nn.operator.OperatorShardingPolicy(),
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
            compute_dtype="float16"
        ),
        loss_scale_policy=phx.nn.operator.training.OperatorLossScalePolicy(
            initial_scale=8.0,
            growth_interval=10,
        ),
    )

    assert result.completed_steps == 1
    assert result.loss_scale_state is not None
    assert int(result.loss_scale_state.nonfinite_microsteps) == 1

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _dataset(*, cases=6, size=4):
    nodes = jnp.linspace(0.0, 1.0, size, endpoint=False)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )
    phases = jnp.linspace(0.0, 1.0, cases)
    states = jnp.sin(2.0 * jnp.pi * nodes[None, :] + phases[:, None])
    targets = states + 0.25 * jnp.sign(jnp.sin(5.0 * phases))[:, None]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": states},
        {"output": targets},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _model(dataset, *, seed=0):
    batch = dataset.batch
    size = batch.require_single_query().sample_shape[0]
    location = phx.nn.operator.architectures.FNO(
        n_modes=(1,),
        in_channels="scalar",
        out_channels="scalar",
        width=4,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(seed),
    )
    conditioner = phx.nn.operator.architectures.OperatorBatchConditioner(
        {
            "state": phx.nn.operator.architectures.FixedBranchEncoder(
                phx.nn.models.MLP(
                    in_size=size,
                    out_size=4,
                    width_size=6,
                    depth=1,
                    key=jr.key(seed + 1),
                ),
                4,
            )
        }
    )
    return phx.nn.operator.architectures.conditional_coupling_flow_operator(
        jr.key(seed + 2),
        location_model=location,
        conditioner=conditioner,
        reference_query=batch.require_single_query(),
        uncertainty_source="process",
        flow_layers=2,
        nn_width=8,
        nn_depth=1,
    )


def test_flowjax_operator_distribution_batches_samples_logs_and_differentiates():
    dataset = _dataset()
    model = _model(dataset)
    batch = dataset.batch
    target = dataset.targets.field("output").values
    distribution = model.distribution(batch)
    samples = model.sample(batch, num_samples=5, key=jr.key(10))
    log_prob = distribution.log_prob(target)
    compiled_samples, compiled_log_prob = eqx.filter_jit(
        lambda candidate, item, values: (
            candidate.sample(item, num_samples=3, key=jr.key(11)),
            candidate.distribution(item).log_prob(values),
        )
    )(model, batch, target)
    gradients = eqx.filter_grad(
        lambda candidate: -jnp.mean(candidate.distribution(batch).log_prob(target))
    )(model)
    leaves = tuple(
        leaf for leaf in jax.tree_util.tree_leaves(gradients) if eqx.is_array(leaf)
    )

    assert isinstance(
        distribution, phx.nn.operator.architectures.FlowJAXOperatorDistribution
    )
    assert distribution.event_shape == (4,)
    assert distribution.uncertainty_source == "process"
    assert distribution.condition.shape == (6, 4)
    assert samples.shape == (5, 6, 4)
    assert log_prob.shape == (6,)
    assert compiled_samples.shape == (3, 6, 4)
    assert jnp.allclose(compiled_log_prob, log_prob)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(log_prob))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)

    status = phx.nn.operator.operator_architecture_status(
        "ConditionalFlowFunctionOperator"
    )
    assert status.tier == "experimental"
    assert status.capabilities.requires_fixed_query
    assert not status.capabilities.resolution_transfer
    assert status.capabilities.topology == "unused"
    assert model.operator_contract.capabilities.requires_fixed_query
    configuration = dict(model.operator_contract.configuration)
    assert configuration["condition_inputs"] == ("state",)
    assert configuration["uncertainty_source"] == "process"
    assert model.operator_contract.capabilities.topology == "unused"


def test_flowjax_fixed_query_accepts_loader_broadcast_but_rejects_changed_geometry():
    dataset = _dataset()
    model = _model(dataset)
    loader = phx.nn.operator.training.OperatorBatchLoader(
        dataset, batch_size=2, shuffle=False
    )
    loader_batch = next(loader.epoch()).batch
    distribution = model.distribution(loader_batch)

    assert distribution.location.shape == (2, 4)

    changed_axis = phx.nn.operator.OperatorAxis(
        "x",
        dataset.batch.require_single_query().axes[0].nodes + 0.02,
        quadrature_weights=jnp.full((4,), 0.25),
        periodic=True,
    )
    changed = phx.nn.operator.OperatorBatch(
        inputs=dataset.batch.inputs,
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(changed_axis,))
        },
        case_axes=dataset.batch.case_axes,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="fixed reference"):
        jax.block_until_ready(model.distribution(changed).location)


def test_flowjax_operator_trains_through_fit_operator():
    dataset = _dataset()
    result = phx.nn.operator.training.fit_operator(
        _model(dataset),
        dataset,
        loss_terms=(phx.nn.operator.training.OperatorDistributionNLL(),),
        learning_rate=2e-3,
        steps=2,
        batch_size=3,
        seed=4,
        jit=True,
    )

    assert result.completed_steps == 2
    assert jnp.isfinite(result.initial_loss)
    assert jnp.isfinite(result.final_loss)
    assert result.final_loss < result.initial_loss
    assert isinstance(
        result.execution_model,
        phx.nn.operator.architectures.ConditionalFlowFunctionOperator,
    )


def test_flowjax_fit_checkpoint_resume_is_bitwise_exact(tmp_path):
    dataset = _dataset(cases=4)
    model = _model(dataset, seed=20)
    common = {
        "loss_terms": (phx.nn.operator.training.OperatorDistributionNLL(),),
        "learning_rate": 1e-3,
        "batch_size": 2,
        "seed": 13,
        "checkpoint_every": 1,
        "configuration": {"test_contract": "flowjax-exact-resume-v1"},
        "jit": True,
    }
    uninterrupted = phx.nn.operator.training.fit_operator(
        model, dataset, steps=2, **common
    )
    checkpoint = tmp_path / "flow-checkpoint"
    phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    full_leaves = jax.tree_util.tree_leaves(uninterrupted.last_execution_model)
    resumed_leaves = jax.tree_util.tree_leaves(resumed.last_execution_model)
    for full, restored in zip(full_leaves, resumed_leaves, strict=True):
        if isinstance(full, jax.Array):
            assert jnp.array_equal(full, restored)
    assert resumed.resumed_from_step == 1
    assert resumed.history == uninterrupted.history
    full_prediction = uninterrupted.execution_model.sample(
        dataset.batch,
        num_samples=2,
        key=jr.key(90),
    )
    resumed_prediction = resumed.execution_model.sample(
        dataset.batch,
        num_samples=2,
        key=jr.key(90),
    )
    assert jnp.array_equal(full_prediction, resumed_prediction)

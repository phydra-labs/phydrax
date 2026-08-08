#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.geometry.operator import (
    RegionalPointLatentGeometry,
    TensorGridLatentGeometry,
)
from phydrax.nn.operator.layers import (
    GeometryMomentEmbedding,
    GraphAttentionTransfer,
    GraphKernelTransfer,
    MultiscaleGraphTransfer,
    OperatorTransformerProcessor,
    RegionalGraphProcessor,
)


def _constant_mlp(model, value=1.0):
    model = eqx.tree_at(
        lambda item: tuple(layer.weight for layer in item.layers),
        model,
        tuple(jnp.zeros_like(layer.weight) for layer in model.layers),
    )
    model = eqx.tree_at(
        lambda item: tuple(layer.bias for layer in item.layers),
        model,
        tuple(
            jnp.full_like(layer.bias, value if index == len(model.layers) - 1 else 0.0)
            for index, layer in enumerate(model.layers)
        ),
    )
    return model


def _constant_kernel_transfer(*, reduction):
    transfer = GraphKernelTransfer(
        in_channels=1,
        out_channels=1,
        coord_dim=1,
        neighbors=8,
        reduction=reduction,
        width=4,
        depth=1,
        key=jr.key(0),
    )
    transfer = eqx.tree_at(
        lambda item: item.source_lift,
        transfer,
        _constant_mlp(transfer.source_lift),
    )
    return eqx.tree_at(
        lambda item: item.edge_kernel.model,
        transfer,
        _constant_mlp(transfer.edge_kernel.model),
    )


def test_tensor_grid_latent_geometry_tracks_case_bounds_and_measure():
    source = jnp.array(
        [
            [[0.0, -1.0], [2.0, -1.0], [0.0, 3.0], [2.0, 3.0]],
            [[10.0, 2.0], [11.0, 2.0], [10.0, 4.0], [11.0, 4.0]],
        ]
    )
    geometry = TensorGridLatentGeometry(
        (4, 3),
        bounds_policy="case_bbox",
        margin=0.0,
    )

    coordinates = geometry.coordinates((2,), source_coordinates=source)
    weights = geometry.quadrature((2,), source_coordinates=source)

    assert coordinates.shape == (2, 12, 2)
    assert jnp.allclose(jnp.sum(weights, axis=-1), jnp.array([8.0, 2.0]))
    assert tuple(axis.size for axis in geometry.axes()) == (4, 3)


def test_regional_farthest_points_are_deterministic_case_local_and_masked():
    source = jnp.array(
        [
            [[0.0], [1.0], [2.0], [100.0]],
            [[10.0], [11.0], [12.0], [-100.0]],
        ]
    )
    mask = jnp.array([[True, True, True, False], [True, True, True, False]])
    geometry = RegionalPointLatentGeometry(2, 1)

    first = geometry.coordinates(source, mask)
    repeated = jax.jit(lambda values: geometry.coordinates(values, mask))(source)

    assert jnp.array_equal(first, repeated)
    assert jnp.all(first[0] >= 0.0)
    assert jnp.all(first[0] <= 2.0)
    assert jnp.all(first[1] >= 10.0)
    assert jnp.all(first[1] <= 12.0)


def test_kernel_transfer_distinguishes_integral_and_normalized_measure():
    source = (jnp.arange(8, dtype=float) + 0.5)[:, None] / 8.0
    target = jnp.array([[0.25], [0.75]])
    values = jnp.arange(8, dtype=float)
    measure = jnp.full((8,), 0.25)

    integral = _constant_kernel_transfer(reduction="integral")(
        values,
        source,
        target,
        source_measure=measure,
    )
    normalized = _constant_kernel_transfer(reduction="normalized")(
        values,
        source,
        target,
        source_measure=measure,
    )

    assert jnp.allclose(integral[:, 0], 2.0)
    assert jnp.allclose(normalized[:, 0], 1.0)


def test_integral_transfer_rejects_implicit_point_cloud_measure():
    transfer = GraphKernelTransfer(
        in_channels=1,
        out_channels=2,
        coord_dim=1,
        neighbors=2,
        key=jr.key(1),
    )
    coordinates = jnp.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="explicit source_measure"):
        transfer(
            jnp.ones((2,)),
            coordinates,
            coordinates,
            source_measure=None,
        )


def test_kernel_and_attention_transfers_are_jittable_and_differentiable():
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.2], [0.8]]])
    values = jnp.array([[1.0, 2.0, 3.0]])
    measure = jnp.full((1, 3), 1.0 / 3.0)
    transfers = (
        GraphKernelTransfer(
            in_channels=1,
            out_channels=4,
            coord_dim=1,
            neighbors=3,
            key=jr.key(2),
        ),
        GraphAttentionTransfer(
            in_channels=1,
            out_channels=4,
            coord_dim=1,
            neighbors=3,
            heads=2,
            key=jr.key(3),
        ),
    )

    for transfer in transfers:
        evaluate = eqx.filter_jit(
            lambda model, field: model(
                field,
                source,
                target,
                source_measure=measure,
            )
        )
        output = evaluate(transfer, values)
        _, gradient = eqx.filter_value_and_grad(
            lambda model: jnp.sum(
                model(values, source, target, source_measure=measure) ** 2
            )
        )(transfer)
        gradient_leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(gradient)
            if eqx.is_inexact_array(leaf)
        ]

        assert output.shape == (1, 2, 4)
        assert jnp.all(jnp.isfinite(output))
        assert gradient_leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)


def test_geometry_moments_are_permutation_invariant_and_finite_when_empty():
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.25], [2.0]]])
    measure = jnp.array([[0.2, 0.3, 0.5]])
    embedding = GeometryMomentEmbedding(1, 0.6)

    def evaluate(points, weights):
        neighborhood = __import__("phydrax").graph.query_neighbors(
            points,
            target,
            max_neighbors=3,
            radius=0.6,
        )
        return embedding(neighborhood, weights)

    reference = evaluate(source, measure)
    permutation = jnp.array([2, 0, 1])
    permuted = evaluate(source[:, permutation], measure[:, permutation])

    assert jnp.allclose(reference, permuted)
    assert jnp.all(jnp.isfinite(reference))
    assert reference[0, 1, -1] == 0.0


def test_multiscale_gates_form_a_partition_of_unity():
    transfers = tuple(
        GraphKernelTransfer(
            in_channels=1,
            out_channels=3,
            coord_dim=1,
            neighbors=3,
            radius=radius,
            reduction="normalized",
            key=jr.key(index),
        )
        for index, radius in enumerate((0.4, 1.1), start=4)
    )
    transfer = MultiscaleGraphTransfer(
        transfers,
        fusion="gated",
        key=jr.key(6),
    )
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.25], [0.75]]])
    measure = jnp.full((1, 3), 1.0 / 3.0)

    gates = transfer.scale_weights(
        source,
        target,
        source_measure=measure,
    )
    output = transfer(
        jnp.ones((1, 3)),
        source,
        target,
        source_measure=measure,
    )

    assert gates.shape == (1, 2, 2)
    assert jnp.allclose(jnp.sum(gates, axis=-1), 1.0)
    assert output.shape == (1, 2, 3)
    assert jnp.all(jnp.isfinite(output))


def _gino(*, query_channels=0, in_channels="scalar", source_key="u", key=jr.key(20)):
    return phx.nn.operator.architectures.GINO(
        in_channels=in_channels,
        out_channels="scalar",
        coord_dim=1,
        latent_shape=(8,),
        latent_channels=4,
        modes=(3,),
        fno_width=4,
        fno_depth=1,
        encoder_neighbors=4,
        decoder_neighbors=4,
        transfer_width=4,
        transfer_depth=1,
        source_key=source_key,
        query_channels=query_channels,
        key=key,
    )


def _gino_batch(*, query_covariates=False):
    source_coordinates = jnp.stack(
        (
            jnp.linspace(0.0, 1.0, 8),
            jnp.linspace(0.0, 1.0, 8) ** 1.2,
        ),
        axis=0,
    )[..., None]
    query_coordinates = jnp.stack(
        (
            jnp.linspace(0.05, 0.95, 5),
            jnp.linspace(0.1, 0.9, 5) ** 1.1,
        ),
        axis=0,
    )[..., None]
    query_values = (
        jnp.cos(2.0 * jnp.pi * query_coordinates[..., 0]) if query_covariates else None
    )
    return phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=jnp.sin(2.0 * jnp.pi * source_coordinates[..., 0]),
            coordinates=source_coordinates,
            quadrature_weights=jnp.full((2, 8), 1.0 / 8.0),
        )
    }, queries={"query": phx.nn.operator.FunctionSamples(
        values=query_values,
        coordinates=query_coordinates,
        mask=jnp.array(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
    )}, case_axes=("case",),)


def test_gino_supports_per_case_geometry_independent_queries_and_masks():
    model = _gino()
    batch = _gino_batch()

    output = eqx.filter_jit(lambda item, data: item(data))(model, batch)

    assert output.shape == (2, 5)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.allclose(output[1, 3:], 0.0)


def test_gino_supports_query_covariates_and_source_permutation():
    model = _gino(query_channels=1)
    batch = _gino_batch(query_covariates=True)
    reference = model(batch)
    permutation = jnp.array([7, 1, 5, 0, 3, 6, 2, 4])
    source = batch.input("u")
    permuted = phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=source.values[:, permutation],
            coordinates=source.coordinates[:, permutation],
            quadrature_weights=source.quadrature_weights[:, permutation],
        )
    }, queries={"query": batch.require_single_query()}, case_axes=batch.case_axes,)

    assert jnp.allclose(model(permuted), reference, rtol=1e-10, atol=1e-10)


def test_gino_fuses_independently_sampled_multiple_sources():
    first_coordinates = jnp.linspace(0.0, 1.0, 8)[:, None]
    second_coordinates = jnp.linspace(0.05, 0.95, 9)[:, None]
    query = jnp.linspace(0.1, 0.9, 4)[:, None]
    batch = phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=jnp.sin(first_coordinates[:, 0]),
            coordinates=first_coordinates,
            quadrature_weights=jnp.full((8,), 1.0 / 8.0),
        ),
        "v": phx.nn.operator.FunctionSamples(
            values=jnp.cos(second_coordinates[:, 0]),
            coordinates=second_coordinates,
            quadrature_weights=jnp.full((9,), 1.0 / 9.0),
        ),
    }, queries={"query": phx.nn.operator.FunctionSamples(values=None, coordinates=query)}, )
    model = _gino(
        in_channels={"u": 1, "v": 1},
        source_key=None,
        key=jr.key(21),
    )

    output = model(batch)

    assert output.shape == (4,)
    assert jnp.all(jnp.isfinite(output))


def test_gino_has_finite_parameter_gradients_and_serializes(tmp_path):
    model = _gino(key=jr.key(22))
    batch = _gino_batch()
    loss, gradient = eqx.filter_value_and_grad(lambda item: jnp.mean(item(batch) ** 2))(
        model
    )
    gradient_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]
    path = tmp_path / "gino.eqx"
    eqx.tree_serialise_leaves(path, model)
    restored = eqx.tree_deserialise_leaves(path, model)

    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
    assert jnp.allclose(restored(batch), model(batch))


def test_gino_training_checkpoint_resumes_exactly(tmp_path):
    model = _gino(key=jr.key(23))
    batch = _gino_batch()
    target = jnp.sin(3.0 * batch.require_single_query().coordinates[..., 0])
    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    def step(current, optimizer_state):
        _, gradient = eqx.filter_value_and_grad(
            lambda item: jnp.mean((item(batch) - target) ** 2)
        )(current)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, current)
        return eqx.apply_updates(current, updates), optimizer_state

    model, state = step(model, state)
    schema = phx.nn.operator.training.operator_batch_schema(
        batch,
        target=phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"output": target},
            batch,
        ),
    )
    checkpoint = phx.nn.operator.training.save_operator_training_checkpoint(
        tmp_path / "checkpoint",
        model,
        state,
        step=1,
        key=jr.key(24),
        schema=schema,
        metadata={"architecture": "GINO"},
    )
    expected_model, expected_state = step(model, state)
    restored = phx.nn.operator.training.load_operator_training_checkpoint(
        checkpoint,
        model,
        state,
        expected_schema=schema,
    )
    actual_model, actual_state = step(restored.model, restored.optimizer_state)
    expected_leaves = jax.tree_util.tree_leaves(
        eqx.filter((expected_model, expected_state), eqx.is_array)
    )
    actual_leaves = jax.tree_util.tree_leaves(
        eqx.filter((actual_model, actual_state), eqx.is_array)
    )

    assert restored.step == 1
    assert restored.metadata == {"architecture": "GINO"}
    assert len(actual_leaves) == len(expected_leaves)
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(actual_leaves, expected_leaves)
    )


def _rigno(*, key=jr.key(30)):
    return phx.nn.operator.architectures.RIGNO(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=1,
        regional_count=4,
        latent_channels=4,
        processor_neighbors=3,
        processor_depth=2,
        processor_width=6,
        processor_mlp_depth=1,
        encoder_neighbors=4,
        decoder_neighbors=4,
        transfer_width=6,
        transfer_depth=1,
        source_key="u",
        key=key,
    )


def test_regional_graph_processor_is_measure_scale_invariant_and_jittable():
    processor = RegionalGraphProcessor(
        3,
        1,
        neighbors=3,
        depth=2,
        width=6,
        mlp_depth=1,
        key=jr.key(31),
    )
    coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 5)[None, :, None],
        (2, 5, 1),
    )
    values = jnp.stack(
        (
            jnp.sin(coordinates[..., 0]),
            jnp.cos(coordinates[..., 0]),
            coordinates[..., 0],
        ),
        axis=-1,
    )
    measure = jnp.full((2, 5), 0.2)
    mask = jnp.ones((2, 5), dtype=bool)
    evaluate = eqx.filter_jit(
        lambda model, weights: model(
            values,
            coordinates,
            weights,
            mask,
        )
    )

    reference = evaluate(processor, measure)
    rescaled = evaluate(processor, 7.0 * measure)

    assert reference.shape == values.shape
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(reference, rescaled, rtol=1e-10, atol=1e-10)


def test_rigno_supports_case_geometry_query_masks_and_graph_isolation():
    model = _rigno()
    batch = _gino_batch()
    source = batch.input("u")
    changed_values = jnp.asarray(source.values).at[1].add(100.0)
    changed = phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=changed_values,
            coordinates=source.coordinates,
            quadrature_weights=source.quadrature_weights,
        )
    }, queries={"query": batch.require_single_query()}, case_axes=batch.case_axes,)

    reference = eqx.filter_jit(lambda item, data: item(data))(model, batch)
    modified = model(changed)

    assert reference.shape == (2, 5)
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(reference[1, 3:], 0.0)
    assert jnp.allclose(reference[0], modified[0], rtol=1e-12, atol=1e-12)
    assert not jnp.allclose(reference[1, :3], modified[1, :3])


def test_rigno_has_finite_parameter_gradients_and_serializes(tmp_path):
    model = _rigno(key=jr.key(32))
    batch = _gino_batch()
    loss, gradient = eqx.filter_value_and_grad(lambda item: jnp.mean(item(batch) ** 2))(
        model
    )
    gradient_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]
    path = tmp_path / "rigno.eqx"
    eqx.tree_serialise_leaves(path, model)
    restored = eqx.tree_deserialise_leaves(path, model)

    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
    assert jnp.allclose(restored(batch), model(batch))
    status = phx.nn.operator.operator_architecture_status(
        "resolution invariant graph neural operator"
    )
    assert status.name == "RIGNO"
    assert status.tier == "research"
    assert not status.recommendation_eligible


def test_operator_transformer_patch_roundtrip_measure_scaling_and_masks():
    processor = OperatorTransformerProcessor(
        (4, 4),
        2,
        patch_shape=2,
        model_width=8,
        depth=3,
        heads=2,
        key=jr.key(40),
    )
    values = jnp.arange(64, dtype=float).reshape((2, 16, 2)) / 64.0
    coordinates = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0.0, 1.0, 4),
            jnp.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((16, 2))
    coordinates = jnp.broadcast_to(coordinates, (2, 16, 2))
    measure = jnp.full((2, 16), 1.0 / 16.0)
    mask = jnp.ones((2, 16), dtype=bool).at[1, 10:].set(False)

    tokens = processor.patchify(values)
    reconstructed = processor.unpatchify(tokens, (2,))
    evaluate = eqx.filter_jit(
        lambda model, weights: model(
            values,
            coordinates,
            weights,
            mask,
        )
    )
    reference = evaluate(processor, measure)
    rescaled = evaluate(processor, 11.0 * measure)

    assert jnp.array_equal(reconstructed, values)
    assert reference.shape == values.shape
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(reference, rescaled, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(reference[1, 10:], 0.0)


def test_operator_transformer_has_finite_parameter_gradients():
    processor = OperatorTransformerProcessor(
        (4, 4),
        2,
        patch_shape=(2, 2),
        model_width=8,
        depth=2,
        heads=2,
        long_range_skip=True,
        key=jr.key(41),
    )
    values = jnp.linspace(-1.0, 1.0, 32).reshape((1, 16, 2))
    coordinates = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0.0, 1.0, 4),
            jnp.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((1, 16, 2))
    measure = jnp.full((1, 16), 1.0 / 16.0)
    mask = jnp.ones((1, 16), dtype=bool)

    loss, gradient = eqx.filter_value_and_grad(
        lambda model: jnp.mean(model(values, coordinates, measure, mask) ** 2)
    )(processor)
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]

    assert jnp.isfinite(loss)
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def _gaot_batch():
    base = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0.0, 1.0, 3),
            jnp.linspace(0.0, 1.0, 3),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((9, 2))
    source_coordinates = jnp.stack(
        (
            base,
            base
            + 0.04
            * jnp.stack(
                (
                    jnp.sin(jnp.pi * base[:, 0]) * jnp.sin(jnp.pi * base[:, 1]),
                    jnp.sin(2.0 * jnp.pi * base[:, 0]) * jnp.sin(jnp.pi * base[:, 1]),
                ),
                axis=-1,
            ),
        ),
        axis=0,
    )
    query_coordinates = jnp.array(
        [
            [[0.1, 0.2], [0.35, 0.75], [0.6, 0.4], [0.85, 0.8], [0.5, 0.5]],
            [[0.15, 0.1], [0.4, 0.7], [0.65, 0.35], [0.9, 0.75], [0.45, 0.55]],
        ]
    )
    return phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=jnp.sin(jnp.pi * source_coordinates[..., 0])
            * jnp.cos(jnp.pi * source_coordinates[..., 1]),
            coordinates=source_coordinates,
            quadrature_weights=jnp.full((2, 9), 1.0 / 9.0),
        )
    }, queries={"query": phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        mask=jnp.array(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
    )}, case_axes=("case",),)


def _gaot(*, key=jr.key(50)):
    return phx.nn.operator.architectures.GAOT(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=2,
        latent_shape=(4, 4),
        patch_shape=2,
        transfer_radius=0.45,
        transfer_scales=(1.0, 2.0),
        latent_channels=4,
        transformer_width=8,
        transformer_depth=1,
        transformer_heads=2,
        transfer_neighbors=6,
        transfer_width=4,
        transfer_heads=2,
        transfer_depth=1,
        source_key="u",
        query_chunk_size=8,
        key=key,
    )


def test_gaot_supports_case_geometry_query_masks_and_graph_isolation():
    model = _gaot()
    batch = _gaot_batch()
    source = batch.input("u")
    changed = phx.nn.operator.OperatorBatch(inputs={
        "u": phx.nn.operator.FunctionSamples(
            values=jnp.asarray(source.values).at[1].add(20.0),
            coordinates=source.coordinates,
            quadrature_weights=source.quadrature_weights,
        )
    }, queries={"query": batch.require_single_query()}, case_axes=batch.case_axes,)

    reference = eqx.filter_jit(lambda item, data: item(data))(model, batch)
    modified = model(changed)

    assert reference.shape == (2, 5)
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(reference[1, 3:], 0.0)
    assert jnp.allclose(reference[0], modified[0], rtol=1e-12, atol=1e-12)
    assert not jnp.allclose(reference[1, :3], modified[1, :3])


def test_gaot_has_finite_parameter_gradients_serializes_and_is_research(tmp_path):
    model = _gaot(key=jr.key(51))
    batch = _gaot_batch()
    loss, gradient = eqx.filter_value_and_grad(lambda item: jnp.mean(item(batch) ** 2))(
        model
    )
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]
    path = tmp_path / "gaot.eqx"
    eqx.tree_serialise_leaves(path, model)
    restored = eqx.tree_deserialise_leaves(path, model)
    status = phx.nn.operator.operator_architecture_status("geometry-aware operator transformer")

    assert jnp.isfinite(loss)
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.allclose(restored(batch), model(batch))
    assert status.name == "GAOT"
    assert status.tier == "research"
    assert not status.recommendation_eligible

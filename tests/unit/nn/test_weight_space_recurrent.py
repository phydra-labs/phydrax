import io

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.layers import RecurrentBatch
from phydrax.nn.models import MLP, WeightSpaceRecurrentModel
from phydrax.nn.operator.architectures import WeightSpaceOperator
from phydrax.nn.parameters import ParameterSubspace


class _ComplexRoot(eqx.Module):
    weight: jax.Array

    def __call__(self, x, *, key=None):
        del key
        return jnp.real(jnp.dot(self.weight, x))


def _subspace():
    root = MLP(
        in_size=1,
        out_size="scalar",
        width_size=5,
        depth=2,
        key=jr.key(1),
    )
    paths = ParameterSubspace.array_leaf_paths(root)
    return root, ParameterSubspace.from_leaf_paths(root, paths)


def test_parameter_subspace_pack_unpack_and_reconstruction_are_exact():
    root, subspace = _subspace()
    packed = subspace.pack()
    selected = subspace.unpack(packed)
    reconstructed = subspace.reconstruct_vector(packed)
    query = jnp.array([0.37])

    assert packed.shape == (subspace.total_dimension,)
    assert subspace.pack(selected).dtype == packed.dtype
    assert jnp.array_equal(subspace.pack(selected), packed)
    assert jnp.array_equal(reconstructed(query), root(query))
    assert subspace.leaf_paths == ParameterSubspace.array_leaf_paths(root)
    assert len(subspace.leaf_dtypes) == len(subspace.leaf_shapes)

    wrong_dtype = jax.tree.map(
        lambda leaf: None if leaf is None else leaf.astype(jnp.float32),
        selected,
        is_leaf=lambda value: value is None,
    )
    with pytest.raises(ValueError, match="exact dtype"):
        subspace.pack(wrong_dtype)


def test_parameter_subspace_preserves_the_frozen_complement():
    root, _ = _subspace()
    paths = ParameterSubspace.array_leaf_paths(root)
    subspace = ParameterSubspace.from_leaf_paths(root, paths[:1])
    moved = subspace.pack() + 0.1
    reconstructed = subspace.reconstruct_vector(moved)
    selected_filter = jax.tree_util.tree_map_with_path(
        lambda path, _: jax.tree_util.keystr(path) == paths[0],
        root,
    )
    _, original_frozen = eqx.partition(root, selected_filter)
    _, moved_frozen = eqx.partition(reconstructed, selected_filter)
    assert eqx.tree_equal(original_frozen, moved_frozen)


def test_weight_space_recurrence_matches_serial_execution_and_streaming_continuation():
    _, subspace = _subspace()
    associative = WeightSpaceRecurrentModel(
        subspace,
        2,
        1,
        execution="associative",
        dtype=jnp.float64,
        key=jr.key(2),
    )
    serial = WeightSpaceRecurrentModel(
        subspace,
        2,
        1,
        execution="serial",
        dtype=jnp.float64,
        key=jr.key(2),
    )
    values = jr.normal(jr.key(3), (7, 2), dtype=jnp.float64)
    valid = jnp.ones((7,), dtype=bool)
    queries = jnp.linspace(-1.0, 1.0, 5)
    batch = RecurrentBatch(values, valid)

    full = associative.parameter_trajectory(batch)
    assert jnp.allclose(
        full.states,
        serial.parameter_trajectory(batch).states,
        atol=2e-10,
        rtol=2e-10,
    )
    first = associative.parameter_trajectory(
        RecurrentBatch(values[:3], jnp.ones((3,), dtype=bool))
    )
    second = associative.parameter_trajectory(
        RecurrentBatch(values[3:], jnp.ones((4,), dtype=bool)),
        initial_state=first.final_state,
    )
    assert jnp.allclose(
        jnp.concatenate((first.states, second.states)),
        full.states,
        atol=2e-10,
        rtol=2e-10,
    )
    decoded = associative(batch, queries)
    assert decoded.shape == (7, 5)
    assert jnp.all(jnp.isfinite(decoded))


def test_weight_space_reset_isolates_parameter_and_observation_state():
    _, subspace = _subspace()
    model = WeightSpaceRecurrentModel(
        subspace,
        2,
        1,
        dtype=jnp.float64,
        key=jr.key(4),
    )
    values = jr.normal(jr.key(5), (8, 2), dtype=jnp.float64)
    reset = jnp.array([False, False, False, False, True, False, False, False])
    packed = model.parameter_trajectory(
        RecurrentBatch(values, jnp.ones((8,), dtype=bool), reset=reset)
    ).states
    first_chunk = model.parameter_trajectory(
        RecurrentBatch(
            values[:2],
            jnp.ones((2,), dtype=bool),
            reset=reset[:2],
        )
    )
    second_chunk = model.parameter_trajectory(
        RecurrentBatch(
            values[2:],
            jnp.ones((6,), dtype=bool),
            reset=reset[2:],
        ),
        initial_state=first_chunk.final_state,
    )
    assert jnp.allclose(
        jnp.concatenate((first_chunk.states, second_chunk.states)),
        packed,
        atol=2e-10,
        rtol=2e-10,
    )
    first = model.parameter_trajectory(
        RecurrentBatch(values[:4], jnp.ones((4,), dtype=bool))
    ).states
    second = model.parameter_trajectory(
        RecurrentBatch(values[4:], jnp.ones((4,), dtype=bool))
    ).states
    assert jnp.allclose(packed, jnp.concatenate((first, second)), atol=2e-10, rtol=2e-10)


def test_weight_space_model_is_jittable_differentiable_and_serializable():
    _, subspace = _subspace()
    model = WeightSpaceRecurrentModel(
        subspace,
        2,
        1,
        dtype=jnp.float64,
        key=jr.key(6),
    )
    batch = RecurrentBatch(
        jr.normal(jr.key(7), (2, 5, 2), dtype=jnp.float64),
        jnp.array([[True, True, True, False, False], [True, True, True, True, True]]),
    )
    queries = jnp.linspace(-0.5, 0.5, 4)
    expected = eqx.filter_jit(lambda current: current(batch, queries))(model)

    gradient = eqx.filter_grad(lambda current: jnp.sum(current(batch, queries) ** 2))(
        model
    )
    assert jnp.all(jnp.isfinite(gradient.recurrence.input_weight))
    assert jnp.all(jnp.isfinite(gradient.decoder.subspace.pack()))
    assert not any(
        leaf.ndim == 2
        and leaf.shape == (subspace.total_dimension, subspace.total_dimension)
        for leaf in jax.tree.leaves(model.recurrence)
    )

    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, model)
    buffer.seek(0)
    restored = eqx.tree_deserialise_leaves(buffer, model)
    assert jnp.allclose(restored(batch, queries), expected, rtol=1e-13, atol=1e-13)


def test_weight_space_operator_decodes_final_state_on_independent_queries():
    _, subspace = _subspace()
    model = WeightSpaceOperator(
        subspace,
        observation_size=2,
        query_size=1,
        source_key="history",
        key=jr.key(8),
    )
    times = jnp.linspace(0.0, 1.0, 5)[:, None]
    query_coordinates = jnp.array([[-0.8], [-0.1], [0.6]])
    query_mask = jnp.array([True, True, False])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "history": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 5, 2)),
                coordinates=times,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=query_coordinates,
                mask=query_mask,
            )
        },
        case_axes=("case",),
    )

    output = model(batch)
    assert output.shape == (2, 3)
    assert jnp.array_equal(output[:, 2], jnp.zeros((2,)))
    assert model.operator_contract.architecture == "WeightSpaceOperator"
    assert model.operator_contract.capabilities.resolution_transfer

    mismatched = WeightSpaceOperator(
        subspace,
        observation_size=2,
        query_size=1,
        out_channels=2,
        source_key="history",
        key=jr.key(8),
    )
    with pytest.raises(ValueError, match="does not match out_channels"):
        mismatched(batch)


def test_weight_space_model_rejects_complex_selected_root_parameters():
    root = _ComplexRoot(jnp.array([1.0 + 1.0j]))
    subspace = ParameterSubspace(root, eqx.is_inexact_array)
    with pytest.raises(TypeError, match="real selected parameters"):
        WeightSpaceRecurrentModel(subspace, 1, 1)

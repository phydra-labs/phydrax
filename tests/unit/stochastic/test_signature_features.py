import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _path():
    return jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


def test_path_signature_stream_aligns_identity_and_prefixes():
    path = _path()
    terminal = phx.stochastic.path_signature(path, 3)
    stream = phx.stochastic.path_signature(path, 3, stream=True)

    assert tuple(level.shape for level in stream) == ((3, 2), (3, 2, 2), (3, 2, 2, 2))
    assert all(jnp.allclose(level[0], 0.0) for level in stream)
    assert all(jnp.allclose(level[-1], final) for level, final in zip(stream, terminal))
    for stop in range(1, path.shape[0] + 1):
        prefix = phx.stochastic.path_signature(path[:stop], 3)
        assert all(
            jnp.allclose(level[stop - 1], expected)
            for level, expected in zip(stream, prefix)
        )


def test_piecewise_signature_supports_empty_and_streaming_segments():
    increments = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    streamed = phx.stochastic.piecewise_linear_signature(
        increments,
        2,
        stream=True,
    )
    empty = phx.stochastic.piecewise_linear_signature(jnp.empty((0, 2)), 2)

    assert streamed[0].shape == (2, 2)
    assert jnp.allclose(streamed[0][0], increments[0])
    assert jnp.allclose(streamed[0][-1], jnp.sum(increments, axis=0))
    assert jnp.allclose(empty[0], jnp.zeros((2,)))
    assert jnp.allclose(empty[1], jnp.zeros((2, 2)))


def test_signature_features_flatten_degree_order_and_scalar():
    path = _path()
    signature = phx.stochastic.path_signature(path, 2)
    features = phx.stochastic.SignatureFeatures(2, 2, include_scalar=True)
    actual = features(path)
    expected = jnp.concatenate(
        (jnp.ones((1,)), signature[0], signature[1].reshape((-1,)))
    )

    assert features.output_size == 7
    assert actual.shape == (7,)
    assert jnp.allclose(actual, expected)
    assert features.feature_id.startswith("SignatureFeatures[")


def test_signature_feature_modules_support_batches_streaming_jit_and_gradients():
    paths = jnp.stack((_path(), 2.0 * _path()))
    terminal = phx.stochastic.SignatureFeatures(2, 3)
    streaming = phx.stochastic.SignatureFeatures(2, 3, stream=True)
    compiled = eqx.filter_jit(terminal)(paths)
    streamed = eqx.filter_jit(streaming)(paths)
    gradient = jax.grad(lambda path: jnp.sum(terminal(path) ** 2))(_path())

    assert terminal.output_size == 14
    assert compiled.shape == (2, 14)
    assert streamed.shape == (2, 3, 14)
    assert gradient.shape == _path().shape
    assert jnp.all(jnp.isfinite(gradient))


def test_log_signature_features_use_standard_bracket_coordinates():
    path = _path()
    basis = phx.stochastic.PrimitiveBasis(2, 3)
    expected = basis.tensor_to_primitive(
        phx.stochastic.tensor_logarithm(phx.stochastic.path_signature(path, 3))
    )
    features = phx.stochastic.LogSignatureFeatures(2, 3)
    streaming = phx.stochastic.LogSignatureFeatures(2, 3, stream=True)(path)

    assert features.output_size == basis.size == 5
    assert jnp.allclose(features(path), expected)
    assert streaming.shape == (3, basis.size)
    assert jnp.allclose(streaming[0], 0.0)
    assert jnp.allclose(streaming[-1], expected)


def test_time_augmentation_broadcasts_and_canonicalizes_ragged_suffixes():
    times = jnp.asarray([[0.0, 0.5, jnp.nan], [0.0, 0.4, 1.0]])
    values = jnp.asarray(
        [
            [[0.0], [1.0], [jnp.nan]],
            [[2.0], [3.0], [5.0]],
        ]
    )
    joint = phx.stochastic.time_augment_path(
        times,
        values,
        lengths=jnp.asarray([2, 3]),
    )

    assert joint.shape == (2, 3, 2)
    assert jnp.allclose(joint[0, -1], joint[0, 1])
    assert jnp.allclose(joint[1, :, 0], times[1])
    assert jnp.all(jnp.isfinite(joint))


def test_repeat_last_padding_preserves_valid_prefix_and_rejects_internal_nan():
    padded = jnp.asarray([[[0.0], [1.0], [jnp.nan], [jnp.nan]]])
    actual = phx.stochastic.repeat_last_path_padding(padded, jnp.asarray([2]))

    assert jnp.allclose(actual, jnp.asarray([[[0.0], [1.0], [1.0], [1.0]]]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="Valid path prefixes"):
        invalid = eqx.filter_jit(phx.stochastic.repeat_last_path_padding)(
            padded.at[0, 1, 0].set(jnp.nan),
            jnp.asarray([2]),
        )
        jax.block_until_ready(invalid)


def test_time_augmentation_rejects_invalid_valid_schedule():
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly increasing"):
        actual = phx.stochastic.time_augment_path(
            jnp.asarray([0.0, 0.5, 0.4]),
            jnp.ones((3, 1)),
        )
        jax.block_until_ready(actual)


def test_signature_features_validate_declared_dimension():
    with pytest.raises(ValueError, match="num_knots, 3"):
        phx.stochastic.SignatureFeatures(3, 2)(_path())
    with pytest.raises(ValueError, match="nonempty path axes"):
        phx.stochastic.path_signature(jnp.empty((0, 2)), 2)


def test_signature_recurrent_cell_matches_dense_prefixes_and_padding():
    paths = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
            [[1.0, -1.0], [0.5, 0.0], [9.0, 9.0], [-3.0, 4.0]],
        ]
    )
    valid = jnp.asarray([[True, True, True, True], [True, True, False, False]])
    cell = phx.stochastic.SignatureRecurrentCell(2, 3, include_scalar=True)
    result = phx.nn.layers.run_recurrent(
        cell,
        phx.nn.layers.RecurrentBatch(paths, valid),
    )
    dense = phx.stochastic.SignatureFeatures(2, 3, include_scalar=True)

    for case_index, length in enumerate((4, 2)):
        for stop in range(1, length + 1):
            assert jnp.allclose(
                result.outputs[case_index, stop - 1],
                dense(paths[case_index, :stop]),
            )
        assert jnp.allclose(
            result.final_output[case_index],
            dense(paths[case_index, :length]),
        )
    assert jnp.allclose(result.outputs[1, 2:], 0.0)


def test_signature_recurrent_cell_resets_to_a_new_path_basepoint():
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [10.0, -2.0], [10.0, 1.0]])
    reset = jnp.asarray([False, False, True, False])
    cell = phx.stochastic.SignatureRecurrentCell(2, 2, include_scalar=True)
    result = phx.nn.layers.run_recurrent(
        cell,
        phx.nn.layers.RecurrentBatch(
            points,
            jnp.ones((4,), dtype=bool),
            reset=reset,
        ),
    )
    dense = phx.stochastic.SignatureFeatures(2, 2, include_scalar=True)

    assert jnp.allclose(result.outputs[0], dense(points[:1]))
    assert jnp.allclose(result.outputs[1], dense(points[:2]))
    assert jnp.allclose(result.outputs[2], dense(points[2:3]))
    assert jnp.allclose(result.outputs[3], dense(points[2:4]))


def test_signature_recurrent_cell_streaming_carry_matches_one_pass():
    paths = jnp.asarray(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0], [1.5, 0.5], [2.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [-1.0, 0.5], [-1.0, 1.0]],
        ]
    )
    cell = phx.stochastic.SignatureRecurrentCell(2, 4)
    full = phx.nn.layers.run_recurrent(
        cell,
        phx.nn.layers.RecurrentBatch(paths, jnp.ones(paths.shape[:-1], dtype=bool)),
    )
    first = phx.nn.layers.run_recurrent(
        cell,
        phx.nn.layers.RecurrentBatch(paths[:, :3], jnp.ones((2, 3), dtype=bool)),
    )
    second = phx.nn.layers.run_recurrent(
        cell,
        phx.nn.layers.RecurrentBatch(paths[:, 3:], jnp.ones((2, 2), dtype=bool)),
        initial_state=first.final_state,
    )

    assert jnp.allclose(
        jnp.concatenate((first.outputs, second.outputs), axis=1),
        full.outputs,
    )
    assert jnp.allclose(second.final_output, full.final_output)


def test_signature_recurrent_cell_jit_gradients_and_case_axes():
    points = jnp.arange(48.0).reshape((2, 3, 4, 2)) / 10.0
    valid = jnp.ones((2, 3, 4), dtype=bool)
    cell = phx.stochastic.SignatureRecurrentCell(2, 2, include_scalar=True)

    def terminal(values):
        result = phx.nn.layers.run_recurrent(
            cell,
            phx.nn.layers.RecurrentBatch(values, valid),
        )
        return result.final_output

    output = eqx.filter_jit(terminal)(points)
    gradient = jax.grad(lambda values: jnp.sum(terminal(values)))(points)

    assert output.shape == (2, 3, cell.output_size)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(gradient))

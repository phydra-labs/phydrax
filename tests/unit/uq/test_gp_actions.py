#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _state():
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.Matern32Kernel(length_scale=0.3),
        noise_scale=0.05,
    )


def _dense(operator):
    return phx.linalg.materialize(
        operator,
        phx.linalg.MaterializationPolicy(max_entries=10_000, max_bytes=1_000_000),
    )


def test_fixed_actions_preserve_orientation_and_native_sparse_values():
    points = jnp.linspace(0.0, 1.0, 6)[:, None]
    matrix = jnp.arange(18.0).reshape(6, 3) + 1.0
    dense = phx.uq.FixedGaussianProcessActionPolicy(matrix).resolve(
        points,
        state=_state(),
    )

    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 1, 2, 5], dtype=jnp.int32),
        source_size=3,
        target_size=6,
    )
    sparse_operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        source=phx.linalg.ArraySpace((3,), dtype=float),
        target=phx.linalg.ArraySpace((6,), dtype=float),
    )
    sparse = phx.uq.FixedGaussianProcessActionPolicy(sparse_operator).resolve(
        points,
        state=_state(),
    )

    assert dense.operator.target.size == 6
    assert dense.operator.source.size == 3
    assert jnp.array_equal(_dense(dense.operator), matrix)
    assert sparse.structurally_sparse
    assert sparse.storage_elements == 4
    assert jnp.array_equal(_dense(sparse.operator), sparse_operator.as_dense())


def test_fixed_actions_reject_reversed_complex_and_misaligned_inputs():
    with pytest.raises(ValueError, match="shape"):
        phx.uq.FixedGaussianProcessActionPolicy(jnp.ones(5))
    with pytest.raises(TypeError, match="real"):
        phx.uq.FixedGaussianProcessActionPolicy(jnp.ones((5, 2), dtype=jnp.complex128))

    policy = phx.uq.FixedGaussianProcessActionPolicy(jnp.ones((5, 2)))
    with pytest.raises(ValueError, match="align"):
        policy.resolve(jnp.ones((4, 1)), state=_state())


def test_block_sparse_actions_balance_normalize_and_replay():
    points = jnp.linspace(0.0, 1.0, 10)[:, None]
    first = phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(jr.key(7), 10, 3)
    second = phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(jr.key(7), 10, 3)
    resolved = first.resolve(points, state=_state())
    matrix = _dense(resolved.operator)

    assert jnp.array_equal(first.values, second.values)
    assert resolved.structurally_sparse
    assert resolved.storage_elements == 10
    assert jnp.array_equal(jnp.count_nonzero(matrix, axis=1), jnp.ones(10))
    assert jnp.allclose(jnp.linalg.vector_norm(matrix, axis=0), jnp.ones(3))
    assert tuple(jnp.count_nonzero(matrix, axis=0).tolist()) == (4, 3, 3)


def test_block_sparse_actions_reject_invalid_blocks_and_preserve_gradients():
    with pytest.raises(ValueError, match="between"):
        phx.uq.BlockSparseGaussianProcessActionPolicy(jnp.ones(4), 5)

    points = jnp.linspace(0.0, 1.0, 6)[:, None]
    policy = phx.uq.BlockSparseGaussianProcessActionPolicy(
        jnp.asarray([1.0, -1.0, 0.0, 0.0, 2.0, 3.0]),
        3,
    )
    with pytest.raises(Exception, match="nonzero finite norm"):
        policy.resolve(points, state=_state())

    def objective(values):
        resolved = phx.uq.BlockSparseGaussianProcessActionPolicy(values, 3).resolve(
            points,
            state=_state(),
        )
        return jnp.sum(_dense(resolved.operator) ** 3)

    gradient = jax.grad(objective)(jnp.arange(1.0, 7.0))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.vector_norm(gradient) > 0.0


def test_pseudo_input_actions_match_kernel_sections_and_are_differentiable():
    points = jnp.linspace(0.0, 1.0, 9)[:, None]
    pseudo_inputs = jnp.asarray([[0.1], [0.5], [0.9]])
    state = _state()
    raw_policy = phx.uq.PseudoInputGaussianProcessActionPolicy(
        pseudo_inputs,
        orthogonalize=False,
    )
    raw = raw_policy.resolve(points, state=state)
    orthogonal = phx.uq.PseudoInputGaussianProcessActionPolicy(
        pseudo_inputs,
        orthogonalize=True,
    ).resolve(points, state=state)

    expected = state.kernel.matrix(points, pseudo_inputs)
    assert jnp.allclose(_dense(raw.operator), expected)
    assert jnp.allclose(
        _dense(orthogonal.operator).T @ _dense(orthogonal.operator),
        jnp.eye(3),
    )

    def objective(inputs):
        actions = phx.uq.PseudoInputGaussianProcessActionPolicy(inputs).resolve(
            points,
            state=state,
        )
        return jnp.sum(_dense(actions.operator) ** 3)

    gradient = jax.grad(objective)(pseudo_inputs)
    assert gradient.shape == pseudo_inputs.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_pseudo_input_actions_reject_shape_count_and_rank_failures():
    points = jnp.linspace(0.0, 1.0, 5)[:, None]
    state = _state()
    with pytest.raises(ValueError, match="cannot exceed"):
        phx.uq.PseudoInputGaussianProcessActionPolicy(jnp.ones((6, 1))).resolve(
            points,
            state=state,
        )
    with pytest.raises(ValueError, match="trailing"):
        phx.uq.PseudoInputGaussianProcessActionPolicy(jnp.ones((2, 2))).resolve(
            points,
            state=state,
        )
    duplicate = phx.uq.PseudoInputGaussianProcessActionPolicy(jnp.asarray([[0.3], [0.3]]))
    with pytest.raises(Exception, match="linearly independent"):
        duplicate.resolve(points, state=state)

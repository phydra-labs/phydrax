#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from opt_einsum import contract

import phydrax as phx
from phydrax._trainable import partition_trainable


def test_low_rank_complex_linear_matches_materialized_affine_map():
    layer = phx.nn.layers.LowRankComplexLinear(
        in_size=4,
        out_size=3,
        rank=2,
        key=jr.key(1),
    )
    values = jr.normal(jr.key(2), (5, 4)) + 1j * jr.normal(jr.key(3), (5, 4))
    expected = contract("oi,...i->...o", layer.materialize_weight(), values) + layer.bias

    assert jnp.allclose(layer(values), expected)
    assert int(jnp.linalg.matrix_rank(layer.materialize_weight())) == 2
    assert 0.0 < layer.initialization.retained_energy <= 1.0
    assert layer.initialization.relative_truncation_residual >= 0.0


def test_full_rank_factorization_recovers_dense_initializer_energy():
    layer = phx.nn.layers.LowRankComplexLinear(
        in_size=3,
        out_size=4,
        rank=3,
        key=jr.key(4),
    )
    assert jnp.allclose(layer.initialization.retained_energy, 1.0)
    assert layer.initialization.relative_truncation_residual < 1e-12


def test_low_rank_complex_linear_uses_real_trainable_leaves_and_jax_transforms():
    layer = phx.nn.layers.LowRankComplexLinear(
        in_size=3,
        out_size=2,
        rank=2,
        key=jr.key(5),
    )
    trainable, _ = partition_trainable(layer)
    assert all(not jnp.iscomplexobj(leaf) for leaf in jax.tree.leaves(trainable))

    values = jr.normal(jr.key(6), (7, 3))
    assert jax.jit(layer)(values).shape == (7, 2)
    assert jax.vmap(layer)(values).shape == (7, 2)
    gradient = jax.grad(lambda value: jnp.sum(jnp.abs(layer(value)) ** 2))(values[0])
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("rank", (0, 4, 1.5, True))
def test_low_rank_complex_linear_rejects_invalid_rank(rank):
    with pytest.raises((TypeError, ValueError)):
        phx.nn.layers.LowRankComplexLinear(
            in_size=3,
            out_size=3,
            rank=rank,
        )


def test_factorized_holomorphic_mlp_is_holomorphic_and_binds_architecture():
    model = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=2,
        hidden_sizes=(4, 4),
        linear_ranks=(1, 2, 2),
        key=jr.key(7),
    )
    point = jnp.asarray([0.2, -0.3])

    def real_map(value):
        output = model(value[0] + 1j * value[1])
        return jnp.concatenate((jnp.real(output), jnp.imag(output)))

    jacobian = jax.jacfwd(real_map)(point)
    output_count = model.out_size
    cr_first = jacobian[:output_count, 0] - jacobian[output_count:, 1]
    cr_second = jacobian[:output_count, 1] + jacobian[output_count:, 0]
    certificate = model.holomorphic_certificate()

    assert jnp.max(jnp.abs(cr_first)) < 1e-11
    assert jnp.max(jnp.abs(cr_second)) < 1e-11
    assert "low-rank-complex-affine" in certificate.operations
    assert certificate.construction_dependencies == (model.architecture_id,)
    assert all(
        not jnp.iscomplexobj(leaf)
        for leaf in jax.tree.leaves(partition_trainable(model)[0])
    )


def test_holomorphic_mlp_dense_default_and_rank_plan_validation():
    dense = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=1,
        hidden_sizes=(3,),
        key=jr.key(8),
    )
    assert dense.linear_ranks == (None, None)
    assert "low-rank-complex-affine" not in dense.holomorphic_certificate().operations

    with pytest.raises(ValueError, match="one entry per affine layer"):
        phx.nn.models.HolomorphicMLP(
            in_size=1,
            out_size=1,
            hidden_sizes=(3,),
            linear_ranks=(1,),
        )
    with pytest.raises(TypeError, match="integers or None"):
        phx.nn.models.HolomorphicMLP(
            in_size=1,
            out_size=1,
            hidden_sizes=(3,),
            linear_ranks=(1.5, 1),
        )

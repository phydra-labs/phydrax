#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import NamedSharding

import phydrax as phx
from tools.operator_benchmarks import (
    assert_resolution_independent_parameters,
    periodic_burgers_scenario,
    profile_resolution_scaling,
)


def test_resolution_scaling_profile_uses_one_parameterization():
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=3,
    )
    model = phx.nn.operator.architectures.FNO(
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(0),
    )
    profile = profile_resolution_scaling(
        model,
        scenario.evaluations,
        architecture="fno",
        repeats=1,
    )
    assert profile.parameter_count > 0
    assert tuple(point.sample_shape for point in profile.points) == ((8,), (12,))
    assert profile.inference_exponent is not None
    assert jnp.isfinite(profile.inference_exponent)


def test_resolution_specific_models_have_constant_parameter_count():
    models = tuple(
        phx.nn.operator.architectures.FNO(
            width=4,
            depth=1,
            n_modes=(3,),
            key=jr.key(0),
        )
        for _ in (8, 12, 16)
    )
    assert_resolution_independent_parameters(models)


def test_case_sharding_preserves_operator_values_and_replicates_parameters():
    scenario = periodic_burgers_scenario(
        train_resolution=8,
        test_resolution=12,
        num_cases=4,
    )
    model = phx.nn.operator.architectures.FNO(
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(1),
    )
    policy = phx.nn.operator.OperatorShardingPolicy(mesh_axis="data")
    sharded_batch = phx.nn.operator.shard_operator_batch(scenario.train_batch, policy)
    sharded_model = phx.nn.operator.replicate_operator_model(model, policy)

    values = sharded_batch.input("state").values
    assert values is not None
    assert isinstance(values.sharding, NamedSharding)
    assert values.sharding.spec[0] == "data"
    node_sharding = sharded_batch.input("state").axes[0].nodes.sharding
    assert isinstance(node_sharding, NamedSharding)
    assert tuple(node_sharding.spec) == ()
    leaves = jax.tree_util.tree_leaves(eqx.filter(sharded_model, eqx.is_array))
    assert leaves
    assert all(
        isinstance(leaf.sharding, NamedSharding) and tuple(leaf.sharding.spec) == ()
        for leaf in leaves
    )

    expected = model(scenario.train_batch)
    actual = eqx.filter_jit(lambda current, batch: current(batch))(
        sharded_model,
        sharded_batch,
    )
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

import phydrax as phx
from phydrax._model import AbstractArrayModel, FrozenModel
from phydrax._strict import StrictModule
from phydrax._trainable import (
    ArrayRole,
    combine_parameters,
    model_state_field,
    NonTrainableState,
    partition_parameters,
    require_parameter_roles,
)
from phydrax.nn.layers import GRUCell, RecurrentBatch
from phydrax.nn.models import RecurrentSequenceModel


def _paths_with(resolution, role):
    return {
        path
        for path, leaf_role in zip(resolution.paths, resolution.roles, strict=True)
        if leaf_role is role
    }


def _mlp(key=0):
    return phx.nn.models.MLP(
        in_size=2, out_size=1, width_size=4, depth=1, key=jr.key(key)
    )


def _deeponet():
    latent = 3
    return phx.nn.operator.architectures.DeepONet(
        branch=phx.nn.models.MLP(
            in_size=4, out_size=latent, width_size=5, depth=1, key=jr.key(2)
        ),
        trunk=phx.nn.models.MLP(
            in_size="scalar", out_size=latent, width_size=5, depth=1, key=jr.key(3)
        ),
        coord_dim=1,
        latent_size=latent,
        out_size="scalar",
        in_size=4,
    )


def _fno():
    return phx.nn.operator.architectures.FNO(
        n_modes=(3,), width=4, depth=1, source_key="u", key=jr.key(1)
    )


@pytest.mark.parametrize("build", [_mlp, _fno, _deeponet], ids=["mlp", "fno", "deeponet"])
def test_stock_models_train_their_weights_without_declarations(build):
    model = build()
    resolution = require_parameter_roles(model, context="stock model")

    parameters = _paths_with(resolution, ArrayRole.PARAMETER)
    assert parameters
    assert all(".weight" not in path or path in parameters for path in resolution.paths)
    trained, state, fixed = partition_parameters(model)
    assert jax.tree_util.tree_leaves(state) == []
    restored = combine_parameters(trained, state, fixed)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(model)


def _context_batch():
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 4))
    return phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.linspace(1.0, 2.0, 8).reshape((2, 4)), axes=(axis,)
            )
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )


def test_operator_context_batch_is_never_a_parameter():
    context = phx.nn.operator.adapters.bind_operator_context(
        _deeponet(), _context_batch()
    )
    resolution = require_parameter_roles(context, context="operator context")

    batch_paths = {path for path in resolution.paths if path.startswith(".batch")}
    assert batch_paths
    assert batch_paths <= _paths_with(resolution, ArrayRole.FIXED)
    assert _paths_with(resolution, ArrayRole.PARAMETER)

    parameters, state, fixed = partition_parameters(context)

    def loss(parameters_):
        model = combine_parameters(parameters_, state, fixed)
        return jnp.sum(model(jnp.asarray([0.25])) ** 2)

    gradient = jax.grad(loss)(parameters)
    assert jax.tree_util.tree_leaves(gradient.batch) == []
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in jax.tree_util.tree_leaves(gradient))


class _FixedPlan(StrictModule, NonTrainableState):
    model: AbstractArrayModel
    scale: Array


def test_frozen_model_inside_fixed_plan_is_frozen_on_purpose():
    frozen = _FixedPlan(FrozenModel(_mlp()), jnp.asarray(2.0))
    resolution = require_parameter_roles(frozen, context="plan")
    assert set(resolution.roles) == {ArrayRole.FIXED}

    silent = _FixedPlan(_mlp(), jnp.asarray(2.0))
    with pytest.raises(ValueError, match="parameter-under-fixed-ancestor"):
        require_parameter_roles(silent, context="plan")


def test_low_rank_adapted_model_trains_only_its_factors():
    model = _mlp()
    specs = {
        path: phx.nn.parameters.LowRankSpec(rank=1)
        for path in phx.nn.parameters.low_rank_sites(model)
    }
    adapted, _ = phx.nn.parameters.adapt_low_rank(model, specs, key=jr.key(4))
    resolution = require_parameter_roles(adapted, context="low-rank model")

    weight_paths = {
        path
        for path in _paths_with(resolution, ArrayRole.PARAMETER)
        if ".weight." in path
    }
    assert weight_paths
    assert all(path.endswith((".left", ".right")) for path in weight_paths)
    assert all(
        resolution.role_of(path) is ArrayRole.FIXED
        for path in resolution.paths
        if path.endswith(".weight.base")
    )


def test_recurrent_sequence_model_is_a_trainable_root_and_batches_are_fixed():
    model = RecurrentSequenceModel(GRUCell(2, 3, dtype=jnp.float64, key=jr.key(5)))
    batch = RecurrentBatch(jnp.ones((1, 4, 2)), jnp.ones((1, 4), dtype=bool))

    model_roles = require_parameter_roles(model, context="recurrent model")
    assert set(model_roles.roles) == {ArrayRole.PARAMETER}
    batch_roles = require_parameter_roles(batch, context="recurrent batch")
    assert set(batch_roles.roles) == {ArrayRole.FIXED}


def test_variational_ansatz_is_a_trainable_root():
    amplitude = phx.nn.quantum.RestrictedBoltzmannAmplitude(
        jnp.zeros(3), jnp.zeros(2), jnp.full((2, 3), 0.1)
    )
    resolution = require_parameter_roles(amplitude, context="ansatz")
    assert set(resolution.roles) == {ArrayRole.PARAMETER}


class _StatefulModel(AbstractArrayModel):
    weight: Array
    running_mean: Array = model_state_field()
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.ones(2)
        self.running_mean = jnp.zeros(2)
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.dot(self.weight, x - self.running_mean)


def test_domain_bindings_reject_model_state():
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = _StatefulModel()

    with pytest.raises(ValueError, match=r"Domain\.Model.*MODEL_STATE.*running_mean"):
        domain.Model("x")(model)
    with pytest.raises(ValueError, match=r"Domain\.Function.*MODEL_STATE"):
        domain.Function("x")(model)

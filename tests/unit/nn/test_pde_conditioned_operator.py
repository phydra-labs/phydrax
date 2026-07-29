#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.models.architectures._pde_conditioned import (
    PDEConditionedInput,
    PDEConditionedOperator,
)


def _problem(constant: float, *, constant_first: bool = False):
    field = phx.equations.PDEExpression.field("u")
    expression = constant + field if constant_first else field + constant
    return phx.equations.PDEProblemIR(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=(phx.equations.PDEField("u", coordinates=("x",)),),
        equations=(phx.equations.PDEEquation("governing", expression),),
    )


def _batch(*, cases: int = 2, identical: bool = False):
    source = jnp.array([0.25, -0.5, 1.0])
    if cases:
        values = jnp.broadcast_to(source, (cases, source.size))
        if not identical:
            values = values + jnp.arange(cases, dtype=float)[:, None]
        case_axes = ("case",)
    else:
        values = source
        case_axes = ()
    return phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=values)},
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.linspace(0.0, 1.0, 5)[:, None],
            )
        },
        case_axes=case_axes,
    )


def _model():
    source_key, equation_key, trunk_key, encoder_key = jr.split(jr.key(0), 4)
    operator = phx.nn.DeepONet(
        branch={
            "source": phx.nn.MLP(
                in_size=3,
                out_size=4,
                width_size=8,
                depth=2,
                key=source_key,
            ),
            "pde": phx.nn.MLP(
                in_size=4,
                out_size=4,
                width_size=8,
                depth=2,
                key=equation_key,
            ),
        },
        trunk=phx.nn.MLP(
            in_size=1,
            out_size=4,
            width_size=8,
            depth=2,
            key=trunk_key,
        ),
        coord_dim=1,
        latent_size=4,
        in_size=3,
        fusion="sum",
    )
    encoder = phx.nn.PDEConditionEncoder(
        width=4,
        depth=1,
        dimension_rank=0,
        key=encoder_key,
    )
    return PDEConditionedOperator(operator, encoder, input_name="pde")


def _tokens(constant: float, *, constant_first: bool = False):
    return phx.equations.tokenize_pde_ir(
        _problem(constant, constant_first=constant_first)
    )


def test_pde_conditioned_operator_composes_canonical_tasks_and_contracts():
    model = _model()
    batch = _batch()
    tokens = _tokens(1.0)
    equivalent = _tokens(1.0, constant_first=True)
    changed = _tokens(2.0)
    key = jr.key(20)

    output = model(PDEConditionedInput(batch, tokens), key=key)
    equivalent_output = model(PDEConditionedInput(batch, equivalent), key=key)
    changed_output = model(PDEConditionedInput(batch, changed), key=key)
    conditioned = phx.nn.attach_pde_condition(
        batch,
        tokens,
        model.encoder,
        input_name="pde",
    )
    direct_output = model.__call_operator_batch__(conditioned, key=key)
    inner_output = model.operator.__call_operator_batch__(conditioned, key=key)

    assert tokens.canonical_hashes == equivalent.canonical_hashes
    assert tokens.canonical_hashes != changed.canonical_hashes
    assert model.in_size == model.operator.in_size == 3
    assert model.out_size == model.operator.out_size == "scalar"
    assert (
        model.operator_output_specs["output"].channels
        == model.operator.operator_output_specs["output"].channels
    )
    assert output.shape == (2, 5)
    assert jnp.allclose(output, equivalent_output, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(output, changed_output, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(output, direct_output)
    assert jnp.allclose(output, inner_output)

    with pytest.raises(ValueError, match="already-conditioned input branch 'pde'"):
        model.__call_operator_batch__(batch)
    with pytest.raises(ValueError, match="already contains input 'pde'"):
        model(PDEConditionedInput(conditioned, tokens))


def test_pde_conditioned_operator_supports_scalar_and_heterogeneous_case_tokens():
    model = _model()
    batch = _batch(identical=True)
    scalar_tokens = _tokens(1.0)
    heterogeneous_tokens = phx.equations.stack_pde_tokens((_tokens(1.0), _tokens(2.0)))

    scalar_output = model(PDEConditionedInput(batch, scalar_tokens))
    eager = model(PDEConditionedInput(batch, heterogeneous_tokens))
    compiled = eqx.filter_jit(lambda item, value: item(value))(
        model,
        PDEConditionedInput(batch, heterogeneous_tokens),
    )

    assert scalar_output.shape == eager.shape == (2, 5)
    assert jnp.allclose(scalar_output[0], scalar_output[1])
    assert not jnp.allclose(eager[0], eager[1], rtol=1e-5, atol=1e-6)
    assert jnp.allclose(compiled, eager, rtol=1e-5, atol=1e-6)


def test_pde_conditioned_operator_rejects_invalid_token_case_shape():
    model = _model()
    batch = _batch()
    wrong_case_tokens = phx.equations.stack_pde_tokens(
        (_tokens(1.0), _tokens(2.0), _tokens(3.0))
    )

    with pytest.raises(
        ValueError,
        match="PDE token batch shape must be scalar or match OperatorBatch case_shape",
    ):
        model(PDEConditionedInput(batch, wrong_case_tokens))


def test_pde_conditioned_operator_has_finite_wrapper_gradients():
    model = _model()
    value = PDEConditionedInput(_batch(), _tokens(1.0))

    _, gradients = eqx.filter_value_and_grad(
        lambda item: jnp.mean(jnp.square(item(value)))
    )(model)
    parameter_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(leaf)
    ]
    gradient_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(gradients)
        if eqx.is_inexact_array(leaf)
    ]

    assert gradient_leaves
    assert len(gradient_leaves) == len(parameter_leaves)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)

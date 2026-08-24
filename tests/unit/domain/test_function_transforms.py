#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.domain import (
    expectation_field,
    Interval1d,
    ModelBinding,
    sigmoid_field,
    softmax_field,
)
from phydrax.solver import FunctionalSolver


def _sample_batch(domain: Interval1d):
    layout = phx.domain.SampleLayout((("x",),))
    return domain.component().sample(
        phx.domain.PointSampling(5, layout=layout),
        key=jr.key(0),
    )


def _trainable_arrays(tree):
    trainable, _ = partition_trainable(tree)
    return tuple(
        leaf
        for leaf in jax.tree_util.tree_leaves(trainable)
        if eqx.is_inexact_array(leaf)
    )


def test_field_transforms_preserve_values_axes_and_semantics():
    domain = Interval1d(0.0, 1.0)
    batch = _sample_batch(domain)

    scalar_logits = domain.Function("x")(lambda x: 2.0 * x[0] - 1.0).with_metadata(
        quantity="binary-logit"
    )
    probabilities = sigmoid_field(scalar_logits)

    scalar_raw = scalar_logits(batch)
    scalar_probability = probabilities(batch)
    assert jnp.allclose(scalar_probability.data, jax.nn.sigmoid(scalar_raw.data))
    assert scalar_probability.dims == scalar_raw.dims
    assert probabilities.domain is scalar_logits.domain
    assert probabilities.deps == scalar_logits.deps
    assert probabilities.metadata == scalar_logits.metadata

    vector_logits = domain.Function("x")(
        lambda x: jnp.asarray([x[0], -x[0], 0.5 * x[0]])
    ).with_metadata(quantity="class-logits")
    class_probabilities = softmax_field(vector_logits)
    expected_value = expectation_field(
        class_probabilities,
        jnp.asarray([-2.0, 0.5, 4.0]),
    )

    logits_out = vector_logits(batch)
    probabilities_out = class_probabilities(batch)
    expectation_out = expected_value(batch)
    expected_probabilities = jax.nn.softmax(logits_out.data, axis=-1)
    assert jnp.allclose(probabilities_out.data, expected_probabilities)
    assert probabilities_out.dims == logits_out.dims
    assert jnp.allclose(
        expectation_out.data,
        jnp.sum(expected_probabilities * jnp.asarray([-2.0, 0.5, 4.0]), axis=-1),
    )
    assert expectation_out.dims == probabilities_out.dims[:-1]
    assert expected_value.domain is vector_logits.domain
    assert expected_value.deps == vector_logits.deps
    assert expected_value.metadata == vector_logits.metadata


def test_field_transforms_are_differentiable():
    domain = Interval1d(0.0, 1.0)
    scalar = sigmoid_field(domain.Function("x")(lambda x: x[0] ** 2 - 0.25))
    vector = expectation_field(
        softmax_field(
            domain.Function("x")(lambda x: jnp.asarray([x[0], -2.0 * x[0], 0.5 + x[0]]))
        ),
        jnp.asarray([-1.0, 0.25, 3.0]),
    )
    key = jr.key(1)

    scalar_grad = jax.grad(lambda value: scalar.func(jnp.asarray([value]), key=key))(0.4)
    scalar_reference = jax.grad(lambda value: jax.nn.sigmoid(value**2 - 0.25))(0.4)
    assert jnp.allclose(scalar_grad, scalar_reference)

    vector_grad = jax.grad(lambda value: vector.func(jnp.asarray([value]), key=key))(0.4)

    def reference(value):
        probabilities = jax.nn.softmax(
            jnp.asarray([value, -2.0 * value, 0.5 + value]), axis=-1
        )
        return jnp.sum(probabilities * jnp.asarray([-1.0, 0.25, 3.0]))

    assert jnp.allclose(vector_grad, jax.grad(reference)(0.4))


def test_softmax_rejects_nonterminal_axis():
    domain = Interval1d(0.0, 1.0)
    field = domain.Function("x")(lambda x: jnp.asarray([x[0], -x[0]]))
    with pytest.raises(ValueError, match="terminal output axis"):
        softmax_field(field, axis=0)


def test_expectation_validates_values_axis_and_runtime_class_count():
    domain = Interval1d(0.0, 1.0)
    field = softmax_field(domain.Function("x")(lambda x: jnp.asarray([x[0], -x[0]])))

    with pytest.raises(ValueError, match="terminal output axis"):
        expectation_field(field, [0.0, 1.0], axis=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        expectation_field(field, 1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        expectation_field(field, jnp.empty((0,)))
    with pytest.raises(ValueError, match="finite"):
        expectation_field(field, [0.0, jnp.inf])
    with pytest.raises(ValueError, match="length must match"):
        expectation_field(field, [0.0, 1.0, 2.0])(_sample_batch(domain))


class _LinearLogits(eqx.Module):
    weight: jax.Array

    def __call__(self, inputs, *, key):
        del key
        return self.weight * inputs[0]


def test_derived_views_reuse_one_model_without_new_trainable_leaves():
    domain = Interval1d(0.0, 1.0)
    model = _LinearLogits(jnp.asarray([1.0, -0.5, 0.25]))
    logits = domain.Model(
        "x",
        binding=ModelBinding.pointwise(),
    )(model).with_metadata(quantity="logits")
    probabilities = softmax_field(logits)
    expected_value = expectation_field(probabilities, [-1.0, 0.0, 2.0])

    assert probabilities.func.func is logits.func
    assert expected_value.func.func is probabilities.func
    assert len(_trainable_arrays(logits)) == 1
    assert len(_trainable_arrays(probabilities)) == 1
    assert len(_trainable_arrays(expected_value)) == 1

    solver = FunctionalSolver(functions={"logits": logits}, terms=())
    assert tuple(solver.functions) == ("logits",)
    assert solver.functions["logits"].func.raw_model is model
    assert len(_trainable_arrays(solver.functions)) == 1

    derived = expectation_field(
        softmax_field(solver.functions["logits"]),
        [-1.0, 0.0, 2.0],
    )
    batch = _sample_batch(domain)
    assert derived(batch).dims == (batch.structure.axis_names[0],)

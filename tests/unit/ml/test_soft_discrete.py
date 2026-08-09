#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_masked_softmax_normalizes_active_entries_and_closes_empty_masks():
    logits = jnp.asarray([[1.0, -2.0, 3.0], [4.0, 0.0, -1.0]])
    mask = jnp.asarray([[True, False, True], [False, False, False]])
    probabilities = phx.ml.masked_softmax(logits, mask=mask, axis=1)

    assert jnp.allclose(jnp.sum(probabilities[0]), 1.0)
    assert probabilities[0, 1] == 0.0
    assert jnp.array_equal(probabilities[1], jnp.zeros((3,)))
    assert jnp.allclose(
        phx.ml.masked_softmax(logits + 17.0, mask=mask, axis=1),
        probabilities,
    )

    empty = jnp.zeros((3,), dtype=bool)
    gradient = jax.grad(
        lambda values: jnp.sum(phx.ml.masked_softmax(values, mask=empty) ** 2)
    )(logits[0])
    jacobian = jax.jacfwd(lambda values: phx.ml.masked_softmax(values, mask=empty))(
        logits[0]
    )
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))
    assert jnp.array_equal(jacobian, jnp.zeros_like(jacobian))


def test_soft_discrete_primitives_reject_nonfloating_values():
    with pytest.raises(TypeError, match="logits must have a real floating dtype"):
        phx.ml.masked_softmax(jnp.asarray([1, 2, 3]))
    with pytest.raises(TypeError, match="values must have a real floating dtype"):
        phx.ml.soft_ranks(jnp.asarray([1.0 + 1.0j, 2.0 + 0.0j]), temperature=0.1)


def test_temperature_primitives_are_transformable_and_validate_dynamic_values():
    logits = jnp.asarray([-1.0, 0.3, 2.0])
    temperature = jnp.asarray(0.7)
    probabilities = phx.ml.temperature_softmax(logits, temperature=temperature)
    gates = phx.ml.temperature_sigmoid(logits, temperature=temperature)

    assert jnp.allclose(jnp.sum(probabilities), 1.0)
    assert jnp.all((gates > 0.0) & (gates < 1.0))
    assert jnp.allclose(
        jax.jit(lambda x, t: phx.ml.temperature_softmax(x, temperature=t))(
            logits, temperature
        ),
        probabilities,
    )
    batched = jax.vmap(
        lambda x: phx.ml.temperature_softmax(x, temperature=temperature)
    )(jnp.stack((logits, -logits)))
    assert batched.shape == (2, 3)

    tangent = jax.jvp(
        lambda t: phx.ml.temperature_softmax(logits, temperature=t),
        (temperature,),
        (jnp.asarray(0.2),),
    )[1]
    reverse = jax.grad(
        lambda t: jnp.dot(
            phx.ml.temperature_softmax(logits, temperature=t),
            jnp.asarray([1.0, -2.0, 0.5]),
        )
    )(temperature)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.isfinite(reverse) & (jnp.abs(reverse) > 0.0)

    for invalid in (0.0, -1.0, jnp.nan, jnp.inf):
        with pytest.raises(Exception, match="temperature must be finite and positive"):
            result = jax.jit(
                lambda t: phx.ml.temperature_sigmoid(logits, temperature=t)
            )(jnp.asarray(invalid))
            jax.block_until_ready(result)


def test_gumbel_softmax_replays_keys_and_has_relaxed_gradients():
    key = jax.random.key(4)
    other_key = jax.random.key(5)
    logits = jnp.asarray([0.2, -0.5, 1.3])
    first = phx.ml.gumbel_softmax(key, logits, temperature=0.8)
    replay = phx.ml.gumbel_softmax(key, logits, temperature=0.8)
    other = phx.ml.gumbel_softmax(other_key, logits, temperature=0.8)

    assert jnp.array_equal(first, replay)
    assert not jnp.array_equal(first, other)
    assert jnp.all(first > 0.0)
    assert jnp.allclose(jnp.sum(first), 1.0)

    weights = jnp.asarray([-1.0, 0.5, 2.0])
    logits_gradient = jax.grad(
        lambda values: jnp.dot(
            phx.ml.gumbel_softmax(key, values, temperature=0.8), weights
        )
    )(logits)
    temperature_gradient = jax.grad(
        lambda value: jnp.dot(
            phx.ml.gumbel_softmax(key, logits, temperature=value), weights
        )
    )(jnp.asarray(0.8))
    samples = jax.jit(
        jax.vmap(lambda sample_key: phx.ml.gumbel_softmax(sample_key, logits, temperature=0.8))
    )(jax.random.split(key, 4))
    assert samples.shape == (4, 3)
    assert jnp.all(jnp.isfinite(logits_gradient))
    assert jnp.isfinite(temperature_gradient) & (jnp.abs(temperature_gradient) > 0.0)


def test_soft_ranks_define_one_based_orientation_ties_and_axis_semantics():
    values = jnp.asarray([3.0, 1.0, 4.0, 2.0])
    ascending = phx.ml.soft_ranks(values, temperature=0.01)
    descending = phx.ml.soft_ranks(values, temperature=0.01, descending=True)
    tied = phx.ml.soft_ranks(jnp.asarray([0.0, 0.0, 2.0]), temperature=0.05)
    equal = phx.ml.soft_ranks(jnp.ones((4,)), temperature=0.3)

    assert jnp.allclose(ascending, jnp.asarray([3.0, 1.0, 4.0, 2.0]), atol=1e-6)
    assert jnp.allclose(descending, 5.0 - ascending, atol=1e-12)
    assert tied[0] == tied[1]
    assert jnp.allclose(equal, 2.5)
    assert jnp.allclose(
        phx.ml.soft_ranks(values + 100.0, temperature=0.01), ascending
    )

    matrix = jnp.stack((values, values[::-1]), axis=1)
    along_rows = phx.ml.soft_ranks(matrix, temperature=0.2, axis=0)
    assert along_rows.shape == matrix.shape
    assert jnp.allclose(jnp.sum(along_rows, axis=0), 10.0)

    tangent = jax.jvp(
        lambda candidate: phx.ml.soft_ranks(candidate, temperature=0.2),
        (values,),
        (jnp.asarray([0.2, -0.1, 0.3, 0.4]),),
    )[1]
    gradient = jax.grad(
        lambda candidate: jnp.sum(
            phx.ml.soft_ranks(candidate, temperature=0.2) ** 2
        )
    )(values)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.all(jnp.isfinite(gradient))


def test_soft_topk_weights_are_bounded_memberships_with_separate_temperatures():
    scores = jnp.asarray([1.0, 4.0, 2.0, 3.0])
    memberships = phx.ml.soft_topk_weights(
        scores,
        k=2,
        rank_temperature=0.05,
        gate_temperature=0.1,
    )
    shared_temperature = phx.ml.soft_topk_weights(
        scores,
        k=2,
        rank_temperature=0.05,
    )
    tied = phx.ml.soft_topk_weights(
        jnp.asarray([3.0, 3.0, 1.0]),
        k=1,
        rank_temperature=0.2,
        gate_temperature=0.2,
    )

    assert jnp.all((memberships >= 0.0) & (memberships <= 1.0))
    assert jnp.array_equal(jnp.argsort(memberships)[-2:], jnp.asarray([3, 1]))
    assert not jnp.allclose(memberships, shared_temperature)
    assert tied[0] == tied[1]

    rank_gradient, gate_gradient = jax.grad(
        lambda rank_temperature, gate_temperature: jnp.dot(
            phx.ml.soft_topk_weights(
                scores,
                k=2,
                rank_temperature=rank_temperature,
                gate_temperature=gate_temperature,
            ),
            jnp.arange(4.0),
        ),
        argnums=(0, 1),
    )(jnp.asarray(0.3), jnp.asarray(0.4))
    assert jnp.isfinite(rank_gradient) & (jnp.abs(rank_gradient) > 0.0)
    assert jnp.isfinite(gate_gradient) & (jnp.abs(gate_gradient) > 0.0)

    for invalid in (0, 5):
        with pytest.raises(ValueError, match="selected score axis"):
            phx.ml.soft_topk_weights(
                scores,
                k=invalid,
                rank_temperature=0.1,
            )

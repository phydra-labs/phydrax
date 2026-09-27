from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.nn.layers import (
    ExplicitFourierFeatureEmbeddings,
    HybridFourierFeatureEmbeddings,
    MultiscaleFourierFeatureEmbeddings,
    RandomFourierFeatureEmbeddings,
    TrainableFourierFeatureEmbeddings,
)


def test_random_fourier_scenario_1() -> None:
    cases = (
        ("scalar", 8, jnp.asarray(0.5), jr.key(0)),
        (3, 10, jnp.asarray([0.5, 1.0, 1.5]), jr.key(1)),
        (10, 10, jnp.ones((10,)), jr.key(10)),
        (2, 2, jnp.asarray([0.5, 1.0]), jr.key(11)),
    )
    for in_size, out_size, value, key in cases:
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size,
            out_size=out_size,
            key=key,
        )
        vector = value.reshape((1,)) if value.ndim == 0 else value
        phases = embeddings.embedding_matrix @ vector
        expected = jnp.concatenate((jnp.cos(phases), jnp.sin(phases)))
        assert embeddings.out_size == out_size
        assert embeddings(value).shape == (out_size,)
        np.testing.assert_allclose(embeddings(value), expected)

    augmented = RandomFourierFeatureEmbeddings(
        in_size=2,
        out_size=8,
        passthrough=(1,),
        include_constant=True,
        key=jr.key(13),
    )
    output = augmented(jnp.asarray([0.2, 0.4]))
    assert output.shape == (8,)
    np.testing.assert_array_equal(output[-2:], jnp.asarray([0.4, 1.0]))
    key = jr.key(2)
    standard = RandomFourierFeatureEmbeddings(in_size=2, out_size=6, key=key)
    shifted = RandomFourierFeatureEmbeddings(
        in_size=2,
        out_size=6,
        mu=1.0,
        sigma=2.0,
        key=key,
    )
    expected_wavevectors = 2.0 * standard.embedding_matrix + 1.0
    value = jnp.asarray([0.5, 1.0])
    phases = expected_wavevectors @ value
    np.testing.assert_allclose(shifted.embedding_matrix, expected_wavevectors)
    np.testing.assert_allclose(
        shifted(value),
        jnp.concatenate((jnp.cos(phases), jnp.sin(phases))),
    )

    means = (0.0, 1.0)
    deviations = (1.0, 2.0)
    multiscale = RandomFourierFeatureEmbeddings(
        in_size=2,
        out_size=24,
        mu=means,
        sigma=deviations,
        key=jr.key(3),
    )
    keys = jr.split(jr.key(3), len(means) * len(deviations))
    blocks = tuple(
        jr.normal(subkey, (3, 2)) * deviation + mean
        for subkey, (mean, deviation) in zip(
            keys,
            ((mean, deviation) for mean in means for deviation in deviations),
            strict=True,
        )
    )
    expected_multiscale = jnp.concatenate(blocks, axis=0)
    np.testing.assert_allclose(multiscale.embedding_matrix, expected_multiscale)
    first = RandomFourierFeatureEmbeddings(in_size=2, out_size=6, key=jr.key(6))
    replay = RandomFourierFeatureEmbeddings(in_size=2, out_size=6, key=jr.key(6))
    other = RandomFourierFeatureEmbeddings(in_size=2, out_size=6, key=jr.key(8))
    value = jnp.asarray([0.5, 1.0])
    np.testing.assert_array_equal(first.embedding_matrix, replay.embedding_matrix)
    np.testing.assert_array_equal(first(value), replay(value))
    assert not bool(jnp.array_equal(first.embedding_matrix, other.embedding_matrix))

    scalar = RandomFourierFeatureEmbeddings(in_size="scalar", out_size=8, key=jr.key(9))
    with pytest.raises(ValueError):
        scalar(jnp.asarray([1.0, 2.0]))
    embeddings = ExplicitFourierFeatureEmbeddings(
        in_size=2,
        wavevectors=jnp.asarray([[jnp.pi, 0.0], [0.0, 2.0 * jnp.pi]]),
        phases=jnp.asarray([0.0, jnp.pi / 2.0]),
        passthrough=(1,),
        include_constant=True,
    )
    np.testing.assert_allclose(
        embeddings(jnp.asarray([0.5, 0.25])),
        jnp.asarray([0.0, -1.0, 1.0, 0.0, 0.25, 1.0]),
        atol=1e-7,
    )

    periodic = ExplicitFourierFeatureEmbeddings.from_periodic_modes(
        in_size=2,
        coordinate=0,
        period=2.0,
        modes=(1, 2, 3),
        passthrough=(1,),
    )
    left = jnp.asarray([-1.0, 0.3])
    right = jnp.asarray([1.0, 0.3])
    np.testing.assert_allclose(periodic(left), periodic(right), atol=1e-7)
    jacobian = jax.jacfwd(periodic)
    np.testing.assert_allclose(
        jacobian(left)[:, 0],
        jacobian(right)[:, 0],
        atol=1e-6,
    )

    fixed = ExplicitFourierFeatureEmbeddings(
        in_size=2,
        wavevectors=jnp.asarray([[1.0, 2.0]]),
        phases=jnp.asarray([0.3]),
    )
    gradients = eqx.filter_grad(lambda model: jnp.sum(model(jnp.asarray([0.4, 0.7]))))(
        fixed
    )
    np.testing.assert_array_equal(gradients.embedding_matrix, 0.0)
    np.testing.assert_array_equal(gradients.phases, 0.0)

    tensor = ExplicitFourierFeatureEmbeddings(
        in_size=(2, 2),
        wavevectors=jnp.asarray([[1.0, 0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(
        eqx.filter_jit(tensor)(jnp.asarray([[0.2, 0.4], [0.6, 0.8]])),
        jnp.asarray([jnp.cos(1.0), jnp.sin(1.0)]),
    )
    explicit = MultiscaleFourierFeatureEmbeddings(
        in_size=2,
        scales=(1.0, 3.0),
        base_wavevectors=jnp.asarray([[2.0, 0.0]]),
    )
    assert explicit.out_size == 4
    np.testing.assert_array_equal(
        explicit.embedding_matrix,
        jnp.asarray([[2.0, 0.0], [6.0, 0.0]]),
    )
    np.testing.assert_array_equal(explicit.scales, jnp.asarray([1.0, 3.0]))

    default = MultiscaleFourierFeatureEmbeddings(in_size=3, scales=(1.0, 2.0))
    np.testing.assert_array_equal(
        default.embedding_matrix,
        jnp.concatenate((jnp.eye(3), 2.0 * jnp.eye(3))),
    )


def test_random_feature_trainability_matches_the_declared_flag() -> None:
    value = jnp.asarray([0.5, 1.0])
    target = jnp.ones((6,), dtype=jnp.float32)

    def loss(
        model: RandomFourierFeatureEmbeddings,
        inputs: jax.Array,
        expected: jax.Array,
    ) -> jax.Array:
        return jnp.mean((model(inputs) - expected) ** 2)

    for trainable in (False, True):
        model = RandomFourierFeatureEmbeddings(
            in_size=2,
            out_size=6,
            trainable=trainable,
            key=jr.key(4),
        )
        gradients = eqx.filter_grad(loss)(model, value, target)
        matrix_gradient = eqx.filter(gradients, eqx.is_array).embedding_matrix
        assert model.trainable is trainable
        assert bool(jnp.allclose(matrix_gradient, 0.0)) is not trainable
        if trainable:
            updated = eqx.tree_at(
                lambda embedding: embedding.embedding_matrix,
                model,
                model.embedding_matrix - 0.1 * matrix_gradient,
            )
            assert not bool(
                jnp.allclose(updated.embedding_matrix, model.embedding_matrix)
            )


def test_random_fourier_scenario_2() -> None:
    first = HybridFourierFeatureEmbeddings(
        in_size=2,
        deterministic_wavevectors=jnp.asarray([[jnp.pi, 0.0]]),
        random_out_size=4,
        passthrough=(1,),
        include_constant=True,
        key=jr.key(12),
    )
    replay = HybridFourierFeatureEmbeddings(
        in_size=2,
        deterministic_wavevectors=jnp.asarray([[jnp.pi, 0.0]]),
        random_out_size=4,
        passthrough=(1,),
        include_constant=True,
        key=jr.key(12),
    )
    assert first.deterministic_wavevector_count == 1
    assert first.random_feature_size == 4
    assert first.out_size == 8
    np.testing.assert_array_equal(
        first.embedding_matrix[:1],
        jnp.asarray([[jnp.pi, 0.0]]),
    )
    np.testing.assert_array_equal(first.embedding_matrix, replay.embedding_matrix)
    np.testing.assert_array_equal(
        first(jnp.asarray([0.2, 0.4])),
        replay(jnp.asarray([0.2, 0.4])),
    )
    embeddings = TrainableFourierFeatureEmbeddings(
        in_size=2,
        out_size=4,
        initial_wavevectors=jnp.asarray([[1.0, 0.0], [0.0, 2.0]]),
    )
    gradients = eqx.filter_grad(lambda model: jnp.sum(model(jnp.asarray([0.4, 0.7]))))(
        embeddings
    )
    assert embeddings.trainable is True
    assert not bool(jnp.allclose(gradients.embedding_matrix, 0.0))
    np.testing.assert_array_equal(gradients.phases, 0.0)

    with pytest.raises(ValueError, match="row count"):
        TrainableFourierFeatureEmbeddings(
            in_size=2,
            out_size=6,
            initial_wavevectors=jnp.asarray([[1.0, 0.0]]),
        )

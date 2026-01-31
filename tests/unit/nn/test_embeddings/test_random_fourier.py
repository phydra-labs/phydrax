#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models.embeddings._fourier import RandomFourierFeatureEmbeddings


class TestRandomFourierFeatureEmbeddings:
    def test_scalar_feature(self):
        """Test with scalar feature input."""
        # Create embeddings for scalar input
        key = jr.key(0)
        out_size = 8
        embeddings = RandomFourierFeatureEmbeddings(
            in_size="scalar", out_size=out_size, key=key
        )

        # Check output size
        assert embeddings.out_size == out_size

        # Test with scalar input
        x = jnp.array(0.5)
        output = embeddings(x)

        # Check output shape
        assert output.shape == (out_size,)

        # Calculate expected output manually
        embedding_matrix = embeddings.embedding_matrix
        x_embedded = embedding_matrix @ jnp.array([x])
        expected_cos = jnp.cos(x_embedded)
        expected_sin = jnp.sin(x_embedded)
        expected = jnp.concatenate((expected_cos, expected_sin))

        assert jnp.allclose(output, expected)

    def test_vector_feature(self):
        """Test with vector feature input."""
        # Create embeddings for vector input
        key = jr.key(1)
        in_size = 3
        out_size = 10
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key
        )

        # Check output size
        assert embeddings.out_size == out_size

        # Test with vector input
        x = jnp.array([0.5, 1.0, 1.5])
        output = embeddings(x)

        # Check output shape
        assert output.shape == (out_size,)

        # Calculate expected output manually
        embedding_matrix = embeddings.embedding_matrix
        x_embedded = embedding_matrix @ x
        expected_cos = jnp.cos(x_embedded)
        expected_sin = jnp.sin(x_embedded)
        expected = jnp.concatenate((expected_cos, expected_sin))

        assert jnp.allclose(output, expected)

    def test_custom_mu_sigma(self):
        """Test with custom mu and sigma values."""
        # Create embeddings with custom mu and sigma
        key = jr.key(2)
        in_size = 2
        out_size = 6
        mu = 1.0
        sigma = 2.0
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size,
            out_size=out_size,
            mu=mu,
            sigma=sigma,
            key=key,
        )

        # Test with input
        x = jnp.array([0.5, 1.0])
        output = embeddings(x)

        # Check output shape
        assert output.shape == (out_size,)

        # We can't easily predict the exact output due to randomness,
        # but we can check that the output is consistent for the same input
        output2 = embeddings(x)
        assert jnp.allclose(output, output2)

    def test_multiscale_embeddings(self):
        """Test with multiscale embeddings (multiple mu and sigma values)."""
        # Create embeddings with multiple mu and sigma values
        key = jr.key(3)
        in_size = 2
        out_size = 24
        mu = [0.0, 1.0]
        sigma = [1.0, 2.0]
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size,
            out_size=out_size,
            mu=mu,
            sigma=sigma,
            key=key,
        )

        # Check output size (base_rows * num_mu * num_sigma * 2 for cos and sin)
        assert embeddings.out_size == out_size

        # Test with input
        x = jnp.array([0.5, 1.0])
        output = embeddings(x)

        # Check output shape
        assert output.shape == (out_size,)

    def test_trainable(self):
        """Test that embedding matrix is trainable when specified."""
        # Create embeddings with trainable matrix
        key = jr.key(4)
        in_size = 2
        out_size = 6
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size,
            out_size=out_size,
            trainable=True,
            key=key,
        )

        # Check that trainable flag is set correctly
        assert embeddings.trainable is True

        # Define a simple loss function
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create some data
        x = jnp.array([0.5, 1.0])
        y = jnp.ones(out_size, dtype=jnp.float32)  # Target embedding

        # Get the gradient of the loss with respect to the model parameters
        grad_fn = eqx.filter_grad(loss_fn)
        grads = grad_fn(embeddings, x, y)

        # Check that the gradient for embedding_matrix is non-zero
        assert not jnp.allclose(eqx.filter(grads, eqx.is_array).embedding_matrix, 0.0)

        # Apply the gradient to update the model
        learning_rate = 0.1
        updated_matrix = (
            embeddings.embedding_matrix
            - learning_rate * eqx.filter(grads, eqx.is_array).embedding_matrix
        )
        updated_model = eqx.tree_at(
            lambda e: e.embedding_matrix, embeddings, updated_matrix
        )

        # Check that embedding_matrix has changed
        assert not jnp.allclose(
            updated_model.embedding_matrix, embeddings.embedding_matrix
        )

    def test_non_trainable(self):
        """Test that embedding matrix is not trainable by default."""
        # Create embeddings with non-trainable matrix (default)
        key = jr.key(5)
        in_size = 2
        out_size = 6
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key
        )

        # Check that trainable flag is set correctly
        assert embeddings.trainable is False

        # Define a simple loss function
        def loss_fn(model, x, y):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        # Create some data
        x = jnp.array([0.5, 1.0])
        y = jnp.ones(out_size, dtype=jnp.float32)  # Target embedding

        # Get the gradient of the loss with respect to the model parameters
        grad_fn = eqx.filter_grad(loss_fn)
        grads = grad_fn(embeddings, x, y)

        # Check that the gradient for embedding_matrix is zero (non-trainable)
        assert jnp.allclose(eqx.filter(grads, eqx.is_array).embedding_matrix, 0.0)

    def test_reproducibility(self):
        """Test that embeddings are reproducible with the same key."""
        # Create two embeddings with the same key
        key = jr.key(6)
        in_size = 2
        out_size = 6

        embeddings1 = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key
        )

        embeddings2 = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key
        )

        # The embedding matrices should be identical
        assert jnp.allclose(embeddings1.embedding_matrix, embeddings2.embedding_matrix)

        # The outputs for the same input should be identical
        x = jnp.array([0.5, 1.0])
        output1 = embeddings1(x)
        output2 = embeddings2(x)
        assert jnp.allclose(output1, output2)

    def test_different_keys(self):
        """Test that embeddings are different with different keys."""
        # Create two embeddings with different keys
        key1 = jr.key(7)
        key2 = jr.key(8)
        in_size = 2
        out_size = 6

        embeddings1 = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key1
        )

        embeddings2 = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key2
        )

        # The embedding matrices should be different
        assert not jnp.allclose(
            embeddings1.embedding_matrix, embeddings2.embedding_matrix
        )

    def test_error_on_wrong_input(self):
        """Test that an error is raised when input doesn't match feature_size."""
        # Create embeddings for scalar input
        key = jr.key(9)
        embeddings = RandomFourierFeatureEmbeddings(in_size="scalar", out_size=8, key=key)

        # Test with vector input (should raise error)
        x = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError):
            embeddings(x)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        # Create embeddings for high-dimensional input
        key = jr.key(10)
        in_size = 10
        out_size = 10
        embeddings = RandomFourierFeatureEmbeddings(
            in_size=in_size, out_size=out_size, key=key
        )

        # Test with high-dimensional input
        x = jnp.ones(in_size)
        output = embeddings(x)

        # Check output shape
        assert output.shape == (out_size,)

    def test_scalar_embedding_size(self):
        """Test with scalar embedding_size."""
        # Create embeddings with the smallest possible output size
        key = jr.key(11)
        embeddings = RandomFourierFeatureEmbeddings(in_size=2, out_size=2, key=key)

        # Check output size (1 * 2 for cos and sin)
        assert embeddings.out_size == 2

        # Test with input
        x = jnp.array([0.5, 1.0])
        output = embeddings(x)

        # Check output shape
        assert output.shape == (2,)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

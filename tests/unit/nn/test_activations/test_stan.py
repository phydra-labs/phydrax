#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.nn.activations._stan import Stan


class TestStan:
    def test_scalar_input(self):
        """Test Stan activation with scalar input."""
        # Create Stan activation with default scalar shape
        stan = Stan()

        # Check that beta is a scalar with value 1.0
        assert stan.beta.shape == ()
        assert jnp.isclose(stan.beta, 1.0)

        # Test with various scalar inputs
        x_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        for x in x_values:
            output = stan(x)
            # Stan formula: tanh(x) * (1 + beta * x)
            expected = jnp.tanh(x) * (1 + stan.beta * x)
            assert jnp.isclose(output, expected)

    def test_vector_input(self):
        """Test Stan activation with vector input."""
        # Create Stan activation with vector shape
        shape = (3,)
        stan = Stan(shape=shape)

        # Check that beta has the right shape with values 1.0
        assert stan.beta.shape == shape
        assert jnp.allclose(stan.beta, jnp.ones(shape))

        # Test with vector input
        x = jnp.array([0.5, 1.0, 1.5])
        output = stan(x)

        # Calculate expected output manually
        expected = jnp.tanh(x) * (1 + stan.beta * x)
        assert jnp.allclose(output, expected)

    def test_matrix_input(self):
        """Test Stan activation with matrix input."""
        # Create Stan activation with matrix shape
        shape = (2, 2)
        stan = Stan(shape=shape)

        # Check that beta has the right shape with values 1.0
        assert stan.beta.shape == shape
        assert jnp.allclose(stan.beta, jnp.ones(shape))

        # Test with matrix input
        x = jnp.array([[0.5, -0.5], [1.0, -1.0]])
        output = stan(x)

        # Calculate expected output manually
        expected = jnp.tanh(x) * (1 + stan.beta * x)
        assert jnp.allclose(output, expected)

    def test_broadcasting(self):
        """Test that broadcasting works correctly with different shapes."""
        # Create Stan activation with a vector shape
        shape = (3,)
        stan = Stan(shape=shape)

        # Test with a batch of inputs
        batch_size = 4
        x = jnp.ones((batch_size,) + shape)

        # This should broadcast beta to each example in the batch
        output = stan(x)

        # Expected shape should match input shape
        assert output.shape == x.shape

        # Each example should have the same output (since inputs are all ones)
        for i in range(batch_size):
            assert jnp.allclose(output[i], output[0])

    def test_special_values(self):
        """Test Stan activation with special input values."""
        stan = Stan()

        # Test with zero
        assert jnp.isclose(stan(jnp.array(0.0)), 0.0)  # tanh(0) * (1 + beta * 0) = 0

        # Test with large positive value
        large_pos = 10.0
        expected_pos = jnp.tanh(large_pos) * (1 + stan.beta * large_pos)
        assert jnp.isclose(stan(jnp.array(large_pos)), expected_pos)

        # Test with large negative value
        large_neg = -10.0
        expected_neg = jnp.tanh(large_neg) * (1 + stan.beta * large_neg)
        assert jnp.isclose(stan(jnp.array(large_neg)), expected_neg)

    def test_modified_beta(self):
        """Test Stan activation with modified beta values."""
        # Create Stan activation
        stan = Stan()

        # Modify beta
        new_beta = 2.0
        stan = eqx.tree_at(lambda s: s.beta, stan, new_beta)

        # Test with various inputs
        x_values = jnp.array([-1.0, 0.0, 1.0])

        for x in x_values:
            output = stan(x)
            expected = jnp.tanh(x) * (1 + new_beta * x)
            assert jnp.isclose(output, expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

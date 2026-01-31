#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.nn.activations import AdaptiveActivation


class TestAdaptiveActivation:
    def test_scalar_coefficient(self):
        """Test with a scalar coefficient."""

        # Define a simple activation function
        def relu(x: Array) -> Array:
            return jnp.maximum(0, x)

        # Create an adaptive activation with scalar coefficient
        adaptive_relu = AdaptiveActivation(relu)

        # Check that the coefficient is a scalar with value 1.0
        assert adaptive_relu.alpha.shape == ()
        assert jnp.isclose(adaptive_relu.alpha, 1.0)

        # Test with positive input
        x_pos = jnp.array([1.0, 2.0, 3.0])
        output_pos = adaptive_relu(x_pos)
        expected_pos = jnp.array([1.0, 2.0, 3.0])  # ReLU passes positive values
        assert jnp.allclose(output_pos, expected_pos)

        # Test with negative input
        x_neg = jnp.array([-1.0, -2.0, -3.0])
        output_neg = adaptive_relu(x_neg)
        expected_neg = jnp.array([0.0, 0.0, 0.0])  # ReLU zeros negative values
        assert jnp.allclose(output_neg, expected_neg)

    def test_vector_coefficient(self):
        """Test with a vector coefficient."""

        # Define a simple activation function
        def tanh(x: Array) -> Array:
            return jnp.tanh(x)

        # Create an adaptive activation with vector coefficient
        shape = (3,)
        adaptive_tanh = AdaptiveActivation(tanh, shape=shape)

        # Check that the coefficient has the right shape with values 1.0
        assert adaptive_tanh.alpha.shape == shape
        assert jnp.allclose(adaptive_tanh.alpha, jnp.ones(shape))

        # Test with input
        x = jnp.array([0.5, 1.0, 1.5])
        output = adaptive_tanh(x)
        expected = jnp.tanh(x)  # With coefficient 1.0, should be same as tanh
        assert jnp.allclose(output, expected)

    def test_default_init_value(self):
        """Test the default initialization value."""

        # Define a simple activation function
        def sigmoid(x: Array) -> Array:
            return 1.0 / (1.0 + jnp.exp(-x))

        # Create an adaptive activation with the default coefficient
        adaptive_sigmoid = AdaptiveActivation(sigmoid)

        # Check that the coefficient has the right value
        assert jnp.isclose(adaptive_sigmoid.alpha, 1.0)

        # Test with input
        x = jnp.array([0.0, 1.0, 2.0])
        output = adaptive_sigmoid(x)
        expected = sigmoid(x)
        assert jnp.allclose(output, expected)

    def test_matrix_coefficient(self):
        """Test with a matrix coefficient."""

        # Define a simple activation function
        def elu(x: Array) -> Array:
            return jnp.where(x > 0, x, jnp.exp(x) - 1)

        # Create an adaptive activation with matrix coefficient
        shape = (2, 2)
        adaptive_elu = AdaptiveActivation(elu, shape=shape)

        # Check that the coefficient has the right shape and values
        assert adaptive_elu.alpha.shape == shape
        assert jnp.allclose(adaptive_elu.alpha, jnp.ones(shape))

        # Test with input
        x = jnp.array([[1.0, -1.0], [2.0, -2.0]])
        output = adaptive_elu(x)
        expected = elu(x)
        assert jnp.allclose(output, expected)

    def test_broadcasting(self):
        """Test that broadcasting works correctly with different shapes."""

        # Define a simple activation function
        def swish(x: Array) -> Array:
            return x * jax.nn.sigmoid(x)

        # Create an adaptive activation with a vector coefficient
        shape = (3,)
        adaptive_swish = AdaptiveActivation(swish, shape=shape)

        # Test with a batch of inputs
        batch_size = 4
        x = jnp.ones((batch_size,) + shape)

        # This should broadcast the coefficient to each example in the batch
        output = adaptive_swish(x)

        # Expected shape should match input shape
        assert output.shape == x.shape

        # Each example should have the same output (since inputs are all ones)
        for i in range(batch_size):
            assert jnp.allclose(output[i], output[0])

    def test_different_activation_functions(self):
        """Test with different activation functions."""
        # Test with several common activation functions
        activations = {
            "relu": lambda x: jnp.maximum(0, x),
            "leaky_relu": lambda x: jnp.where(x > 0, x, 0.01 * x),
            "tanh": jnp.tanh,
            "sigmoid": lambda x: 1.0 / (1.0 + jnp.exp(-x)),
            "softplus": lambda x: jnp.log(1.0 + jnp.exp(x)),
        }

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        for name, fn in activations.items():
            # Create adaptive version with the default coefficient
            adaptive_fn = AdaptiveActivation(fn)

            # Check output
            output = adaptive_fn(x)
            expected = fn(x)
            assert jnp.allclose(output, expected), f"Failed for {name}"

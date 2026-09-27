import jax
import jax.numpy as jnp
from jax import Array

from phydrax.nn.activations import AdaptiveActivation


def _relu(value: Array) -> Array:
    return jnp.maximum(0, value)


def _sigmoid(value: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-value))


def _elu(value: Array) -> Array:
    return jnp.where(value > 0, value, jnp.exp(value) - 1)


def test_adaptive_activation_contracts() -> None:
    cases = (
        ((), _relu, jnp.asarray([-1.0, 0.0, 2.0])),
        ((3,), jnp.tanh, jnp.asarray([0.5, 1.0, 1.5])),
        ((), _sigmoid, jnp.asarray([0.0, 1.0, 2.0])),
        ((2, 2), _elu, jnp.asarray([[1.0, -1.0], [2.0, -2.0]])),
    )
    for shape, function, values in cases:
        adaptive = AdaptiveActivation(function, shape=shape)
        assert adaptive.alpha.shape == shape
        assert jnp.allclose(adaptive.alpha, jnp.ones(shape))
        assert jnp.allclose(adaptive(values), function(values)), shape
    swish = lambda value: value * jax.nn.sigmoid(value)
    batched = AdaptiveActivation(swish, shape=(3,))
    values = jnp.ones((4, 3))
    output = batched(values)
    assert output.shape == values.shape
    assert jnp.allclose(output, jnp.broadcast_to(output[0], output.shape))

    activations = {
        "relu": _relu,
        "leaky_relu": lambda value: jnp.where(value > 0, value, 0.01 * value),
        "tanh": jnp.tanh,
        "sigmoid": _sigmoid,
        "softplus": lambda value: jnp.log(1.0 + jnp.exp(value)),
    }
    points = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    for name, function in activations.items():
        assert jnp.allclose(AdaptiveActivation(function)(points), function(points)), name

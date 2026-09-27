import equinox as eqx
import jax.numpy as jnp

from phydrax.nn.activations._stan import Stan


def test_stan_contracts() -> None:
    cases = (
        ((), jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])),
        ((3,), jnp.asarray([0.5, 1.0, 1.5])),
        ((2, 2), jnp.asarray([[0.5, -0.5], [1.0, -1.0]])),
    )
    for shape, values in cases:
        activation = Stan(shape=shape)
        assert activation.beta.shape == shape
        assert jnp.allclose(activation.beta, jnp.ones(shape))
        expected = jnp.tanh(values) * (1 + activation.beta * values)
        assert jnp.allclose(activation(values), expected), shape

    batched = Stan(shape=(3,))
    batch = jnp.ones((4, 3))
    output = batched(batch)
    assert output.shape == batch.shape
    assert jnp.allclose(output, jnp.broadcast_to(output[0], output.shape))
    activation = Stan()
    for value in (0.0, 10.0, -10.0):
        expected = jnp.tanh(value) * (1 + activation.beta * value)
        assert jnp.isclose(activation(jnp.asarray(value)), expected), value

    updated = eqx.tree_at(lambda subject: subject.beta, activation, 2.0)
    values = jnp.asarray([-1.0, 0.0, 1.0])
    expected = jnp.tanh(values) * (1 + 2.0 * values)
    assert jnp.allclose(updated(values), expected)

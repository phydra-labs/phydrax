import math

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.layers import SineLayer
from phydrax.nn.models import SIREN


def test_sine_layer_uses_distinct_first_and_hidden_initialization_bounds():
    first = SineLayer(
        in_size=4,
        out_size=7,
        omega=30.0,
        is_first=True,
        key=jr.key(0),
    )
    hidden = SineLayer(
        in_size=7,
        out_size=7,
        omega=2.0,
        is_first=False,
        key=jr.key(1),
    )

    assert jnp.max(jnp.abs(first.weight)) <= 1.0 / 4.0
    assert jnp.max(jnp.abs(hidden.weight)) <= math.sqrt(6.0 / 7.0) / 2.0


def test_siren_shapes_jit_vmap_and_high_order_derivatives_are_finite():
    model = SIREN(
        in_size=1,
        out_size="scalar",
        width_size=24,
        depth=4,
        first_omega=20.0,
        hidden_omega=1.5,
        key=jr.key(3),
    )

    def scalar_value(value):
        return model(jnp.asarray([value]))

    first = jax.grad(scalar_value)
    second = jax.grad(first)
    third = jax.grad(second)
    points = jnp.linspace(-1.0, 1.0, 17)
    values = jax.jit(jax.vmap(scalar_value))(points)
    derivatives = jax.vmap(lambda value: (first(value), second(value), third(value)))(
        points
    )

    assert values.shape == (17,)
    assert jnp.all(jnp.isfinite(values))
    assert all(jnp.all(jnp.isfinite(order)) for order in derivatives)
    assert jnp.std(derivatives[1]) > 0.0


def test_siren_vector_output_and_projection_follow_hidden_frequency_bound():
    model = SIREN(
        in_size=3,
        out_size=2,
        width_size=10,
        depth=2,
        hidden_omega=2.5,
        key=jr.key(5),
    )
    bound = math.sqrt(6.0 / 10.0) / 2.5

    assert model(jnp.ones(3)).shape == (2,)
    assert jnp.max(jnp.abs(model.projection.weight)) <= bound

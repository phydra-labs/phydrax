import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import InputConvexNetwork, PartiallyInputConvexNetwork


def test_input_convex_network_has_positive_semidefinite_hessians():
    model = InputConvexNetwork(
        in_size=3,
        width_size=12,
        depth=3,
        activation="softplus",
        key=jr.key(0),
    )
    points = jr.normal(jr.key(1), (8, 3))
    hessians = jax.jit(jax.vmap(model.hessian))(points)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(hessians)

    assert jnp.min(eigenvalues) >= -1e-9
    assert jnp.all(jnp.isfinite(hessians))


def test_input_convex_gradient_is_monotone():
    model = InputConvexNetwork(in_size=2, width_size=10, depth=2, key=jr.key(2))
    first = jnp.asarray([-0.5, 0.8])
    second = jnp.asarray([0.7, -0.2])
    monotonicity = jnp.vdot(
        model.gradient(first) - model.gradient(second), first - second
    ).real

    assert monotonicity >= -1e-9


def test_positive_hidden_weights_survive_optimizer_style_raw_updates():
    model = InputConvexNetwork(in_size=2, width_size=8, depth=3, key=jr.key(3))
    updates = jax.tree.map(
        lambda leaf: -0.2 * jnp.ones_like(leaf) if eqx.is_array(leaf) else None,
        model,
    )
    updated = eqx.apply_updates(model, updates)

    for layer in updated.state_layers:
        effective = layer.weight_transform(layer.weight)
        assert jnp.all(effective > 0.0)
    assert (
        jnp.min(jnp.linalg.eigvalsh(updated.hessian(jnp.asarray([0.2, -0.4])))) >= -1e-9
    )


def test_partially_input_convex_network_is_convex_only_in_designated_input():
    model = PartiallyInputConvexNetwork(
        context_size=2,
        convex_size=3,
        width_size=11,
        depth=3,
        key=jr.key(5),
    )
    contexts = jr.normal(jr.key(6), (5, 2))
    convex_inputs = jr.normal(jr.key(7), (5, 3))
    hessians = jax.vmap(model.convex_hessian)(contexts, convex_inputs)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(hessians)
    outputs = jax.jit(jax.vmap(model))((contexts, convex_inputs))

    assert outputs.shape == (5,)
    assert jnp.min(eigenvalues) >= -1e-9
    assert jnp.all(jnp.isfinite(hessians))

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.layers import RandomFourierFeatureEmbeddings
from phydrax.nn.models import PirateNet


def test_piratenet_zero_gates_make_the_deep_body_exactly_identity():
    model = PirateNet(
        in_size=2,
        out_size="scalar",
        width_size=6,
        depth=4,
        initial_alpha=0.0,
        key=jr.key(0),
    )
    x = jnp.asarray([0.2, -0.7])

    expected = model.projection(model.lift(x))
    assert jnp.array_equal(eqx.filter_jit(model)(x), expected)
    assert all(block.alpha == 0.0 for block in model.blocks)


def test_piratenet_zero_gate_blocks_have_alpha_but_not_branch_gradients():
    model = PirateNet(
        in_size=2,
        out_size="scalar",
        width_size=5,
        depth=2,
        initial_alpha=0.0,
        key=jr.key(3),
    )
    x = jnp.asarray([0.4, -0.1])
    gradient = eqx.filter_grad(lambda current: jnp.sum(current(x)))(model)

    assert gradient.blocks[0].alpha != 0.0
    branch_gradient = gradient.blocks[0].branch.func.layers[0].weight
    assert jnp.array_equal(branch_gradient, jnp.zeros_like(branch_gradient))


def test_piratenet_reuses_existing_phydrax_embeddings():
    embedding = RandomFourierFeatureEmbeddings(
        in_size=2,
        out_size=8,
        sigma=2.0,
        key=jr.key(5),
    )
    model = PirateNet(
        in_size=2,
        out_size=3,
        width_size=7,
        depth=2,
        embedding=embedding,
        initial_alpha=0.25,
        key=jr.key(6),
    )
    x = jnp.asarray([0.1, 0.8])

    output = eqx.filter_jit(model)(x)
    assert output.shape == (3,)
    assert jnp.all(jnp.isfinite(output))


def test_piratenet_unit_gates_select_each_nonlinear_branch_exactly():
    model = PirateNet(
        in_size=2,
        out_size=2,
        width_size=4,
        depth=2,
        initial_alpha=1.0,
        key=jr.key(8),
    )
    x = jnp.asarray([-0.3, 0.6])
    hidden = model.lift(x)
    encoder_u = model.encoder_u(x)
    encoder_v = model.encoder_v(x)
    for block in model.blocks:
        hidden = block.branch(hidden, encoder_u, encoder_v)

    assert jnp.allclose(model(x), model.projection(hidden))

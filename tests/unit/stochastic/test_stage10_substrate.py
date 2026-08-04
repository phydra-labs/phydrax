import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_antithetic_paths_share_independence_clusters():
    realization = phx.stochastic.WienerRealization.antithetic(
        jr.key(0), (1,), support=(0.0, 1.0), num_pairs=3
    )

    labels = phx.stochastic.realization_independence_labels(
        realization, realization.sample_shape
    )

    assert len(set(labels)) == 3
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[4] == labels[5]


def test_missing_realization_has_unknown_independence():
    labels = phx.stochastic.realization_independence_labels(None, (3,))
    assert labels == (None, None, None)


def test_jump_generator_is_owned_by_stochastic_namespace():
    process = phx.stochastic.JumpProcess(
        lambda t, state, args: jnp.asarray([2.0]),
        lambda state, channel, mark, args: state + jnp.asarray([1.0]),
        state_shape=(1,),
        num_channels=1,
        process_id="counting",
    )

    value = phx.stochastic.jump_generator_observable(
        process,
        jnp.asarray([3.0]),
        time=0.0,
        observable=lambda state: state**2,
        key=jr.key(1),
    )

    assert jnp.allclose(value, jnp.asarray([14.0]))
    assert not hasattr(phx.nn, "jump_generator_observable")

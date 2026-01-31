#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import FeynmaNN, LatentContractionModel, MLP, Separable


def test_pimlp_keep_output_complex_scalar():
    key = jr.key(0)
    m = FeynmaNN(
        in_size=2,
        out_size="scalar",
        width_size=16,
        depth=2,
        num_paths=3,
        keep_output_complex=True,
        key=key,
    )
    x = jr.normal(jr.key(1), (2,))
    y = m(x)
    assert jnp.iscomplexobj(y)
    assert y.shape == ()


def test_pimlp_keep_output_complex_vector():
    m = FeynmaNN(
        in_size=3,
        out_size=4,
        width_size=16,
        depth=2,
        num_paths=3,
        keep_output_complex=True,
        key=jr.key(2),
    )
    x = jr.normal(jr.key(3), (3,))
    y = m(x)
    assert jnp.iscomplexobj(y)
    assert y.shape == (4,)


def test_separable_wrapper_complex_output_toggle():
    # Underlying scalar models for each coordinate produce complex latents
    L = 2
    out = 1
    models = [
        FeynmaNN(
            in_size="scalar",
            out_size=L * out,
            width_size=12,
            depth=2,
            num_paths=2,
            keep_output_complex=True,
            key=jr.key(10 + i),
        )
        for i in range(2)
    ]
    sep_c = Separable(
        in_size=2,
        out_size="scalar",
        latent_size=L,
        models=models,
        keep_outputs_complex=True,
    )
    x = jr.normal(jr.key(12), (7, 2))
    yc = jax.vmap(sep_c)(x)
    assert jnp.iscomplexobj(yc)
    assert yc.shape == (7,)

    sep_r = Separable(
        in_size=2,
        out_size="scalar",
        latent_size=L,
        models=models,
        keep_outputs_complex=False,
    )
    yr = jax.vmap(sep_r)(x)
    assert not jnp.iscomplexobj(yr)
    assert yr.shape == (7,)


def test_latent_contraction_complex_output_toggle():
    # Space model produces complex latents of size L*out; time model is real L
    L = 3
    d = 2
    space_model = FeynmaNN(
        in_size=d,
        out_size=L * 1,
        width_size=16,
        depth=2,
        num_paths=2,
        keep_output_complex=True,
        key=jr.key(20),
    )
    time_model = MLP(in_size="scalar", out_size=L, width_size=8, depth=1, key=jr.key(21))
    ts_c = LatentContractionModel(
        out_size="scalar",
        latent_size=L,
        x=space_model,
        t=time_model,
        keep_outputs_complex=True,
    )
    Ns, Nt = 4, 5
    x_space = jr.normal(jr.key(22), (Ns, d))
    x_time = jr.normal(jr.key(23), (Nt,))
    space_rep = jnp.repeat(x_space, Nt, axis=0)
    time_rep = jnp.tile(x_time, Ns)
    points = jnp.column_stack([space_rep, time_rep])
    yc = jax.vmap(ts_c)(points).reshape(Ns, Nt)
    assert jnp.iscomplexobj(yc)
    assert yc.shape == (Ns, Nt)

    ts_r = LatentContractionModel(
        out_size="scalar",
        latent_size=L,
        x=space_model,
        t=time_model,
        keep_outputs_complex=False,
    )
    yr = jax.vmap(ts_r)(points).reshape(Ns, Nt)
    assert not jnp.iscomplexobj(yr)
    assert yr.shape == (Ns, Nt)

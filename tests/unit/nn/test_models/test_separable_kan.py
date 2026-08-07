#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import BSplineEdgeBasis, SeparableKAN


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_kan_vector_input_shape(scan):
    model = SeparableKAN(
        in_size=2,
        out_size=3,
        latent_size=4,
        width_size=8,
        depth=2,
        scan=scan,
        key=jr.key(0),
    )
    x = jr.normal(jr.key(1), (2,))
    y = model(x)
    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_kan_coord_separable_shape(scan):
    model = SeparableKAN(
        in_size=2,
        out_size="scalar",
        latent_size=3,
        width_size=8,
        depth=2,
        scan=scan,
        key=jr.key(2),
    )
    x0 = jnp.linspace(0.0, 1.0, 5)
    x1 = jnp.linspace(-1.0, 1.0, 6)
    y = model((x0, x1))
    assert y.shape == (5, 6)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_kan_scalar_requires_split_input(scan):
    with pytest.raises(ValueError, match="requires in_size >= 2"):
        _ = SeparableKAN(
            in_size="scalar",
            out_size="scalar",
            width_size=8,
            depth=1,
            scan=scan,
        )


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_kan_scalar_with_split_input(scan):
    model = SeparableKAN(
        in_size="scalar",
        out_size="scalar",
        latent_size=2,
        split_input=2,
        width_size=6,
        depth=1,
        scan=scan,
        key=jr.key(3),
    )
    x = jnp.asarray(0.25)
    y = model(x)
    assert y.shape == ()
    assert jnp.isfinite(y)


def test_separable_bspline_kan_scan_matches_loop():
    basis = BSplineEdgeBasis(degree=3, num_intervals=5)
    key = jr.key(4)
    loop = SeparableKAN(
        in_size=2,
        out_size="scalar",
        latent_size=3,
        width_size=6,
        depth=3,
        edge_basis=basis,
        scan=False,
        key=key,
    )
    scanned = SeparableKAN(
        in_size=2,
        out_size="scalar",
        latent_size=3,
        width_size=6,
        depth=3,
        edge_basis=basis,
        scan=True,
        key=key,
    )
    coordinates = (
        jnp.linspace(-0.8, 0.8, 4),
        jnp.linspace(-0.7, 0.7, 5),
    )

    loop_value = loop(coordinates)
    scanned_value = scanned(coordinates)

    assert scanned_value.shape == (4, 5)
    assert jnp.allclose(scanned_value, loop_value)
    assert all(model._scan_enabled for model in scanned.model.models)

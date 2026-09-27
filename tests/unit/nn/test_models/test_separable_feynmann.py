#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import SeparableFeynmaNN


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_feynmann_vector_input_shape(scan: Any) -> None:
    model = SeparableFeynmaNN(
        in_size=2,
        out_size=3,
        latent_size=4,
        width_size=8,
        depth=2,
        scan=scan,
        num_paths=2,
        key=jr.key(0),
    )
    x = jr.normal(jr.key(1), (2,))
    y = model(x)
    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_feynmann_coord_separable_shape(scan: Any) -> None:
    model = SeparableFeynmaNN(
        in_size=2,
        out_size="scalar",
        latent_size=3,
        width_size=8,
        depth=2,
        scan=scan,
        num_paths=2,
        key=jr.key(2),
    )
    x0 = jnp.linspace(0.0, 1.0, 5)
    x1 = jnp.linspace(-1.0, 1.0, 6)
    y = model((x0, x1))
    assert y.shape == (5, 6)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_feynmann_scalar_requires_split_input(scan: Any) -> None:
    with pytest.raises(ValueError, match="requires in_size >= 2"):
        _ = SeparableFeynmaNN(
            in_size="scalar",
            out_size="scalar",
            width_size=8,
            depth=1,
            scan=scan,
            num_paths=2,
        )


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_feynmann_scalar_with_split_input(scan: Any) -> None:
    model = SeparableFeynmaNN(
        in_size="scalar",
        out_size="scalar",
        latent_size=2,
        split_input=2,
        width_size=6,
        depth=1,
        scan=scan,
        num_paths=2,
        key=jr.key(3),
    )
    x = jnp.asarray(0.25)
    y = model(x)
    assert y.shape == ()
    assert jnp.isfinite(y)

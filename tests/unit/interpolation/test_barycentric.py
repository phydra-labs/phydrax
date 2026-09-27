#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax._interpolation import (
    barycentric_basis,
    barycentric_differentiation_matrix,
)


def test_barycentric_contracts() -> None:
    nodes = jnp.asarray((0, 1, 2), dtype=jnp.int32)
    weights = jnp.asarray((0.5, -1.0, 0.5))

    basis = barycentric_basis(jnp.asarray(0.5), nodes, weights)
    differentiation = barycentric_differentiation_matrix(nodes, weights=weights)

    assert jnp.issubdtype(basis.dtype, jnp.floating)
    assert jnp.allclose(jnp.sum(basis), 1.0)
    assert jnp.allclose(differentiation @ (nodes.astype(basis.dtype) ** 2), 2.0 * nodes)
    for nodes in (
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((0.0, jnp.nan, 1.0)),
        jnp.asarray((0.0, jnp.inf, 1.0)),
    ):
        weights = jnp.ones(nodes.shape)

        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError), match="finite distinct"
        ):
            barycentric_basis(jnp.asarray(0.5), nodes, weights)
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError), match="finite distinct"
        ):
            barycentric_differentiation_matrix(nodes)
    for weights in (
        jnp.asarray((1.0, 0.0, 1.0)),
        jnp.asarray((1.0, jnp.nan, 1.0)),
    ):
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="finite nonzero"):
            barycentric_basis(jnp.asarray(0.5), jnp.asarray((0.0, 1.0, 2.0)), weights)

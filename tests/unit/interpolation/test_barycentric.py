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


def test_barycentric_primitives_promote_integer_nodes_consistently():
    nodes = jnp.asarray((0, 1, 2), dtype=jnp.int32)
    weights = jnp.asarray((0.5, -1.0, 0.5))

    basis = barycentric_basis(jnp.asarray(0.5), nodes, weights)
    differentiation = barycentric_differentiation_matrix(nodes, weights=weights)

    assert jnp.issubdtype(basis.dtype, jnp.floating)
    assert jnp.allclose(jnp.sum(basis), 1.0)
    assert jnp.allclose(differentiation @ (nodes.astype(basis.dtype) ** 2), 2.0 * nodes)


@pytest.mark.parametrize(
    "nodes",
    (
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((0.0, jnp.nan, 1.0)),
        jnp.asarray((0.0, jnp.inf, 1.0)),
    ),
)
def test_barycentric_primitives_reject_invalid_nodes(nodes):
    weights = jnp.ones(nodes.shape)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="finite distinct"):
        barycentric_basis(jnp.asarray(0.5), nodes, weights)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="finite distinct"):
        barycentric_differentiation_matrix(nodes)


@pytest.mark.parametrize(
    "weights",
    (
        jnp.asarray((1.0, 0.0, 1.0)),
        jnp.asarray((1.0, jnp.nan, 1.0)),
    ),
)
def test_barycentric_basis_rejects_invalid_weights(weights):
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="finite nonzero"):
        barycentric_basis(jnp.asarray(0.5), jnp.asarray((0.0, 1.0, 2.0)), weights)

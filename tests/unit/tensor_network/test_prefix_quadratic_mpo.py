#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _embed(operator, site, sites):
    value = jnp.asarray([[1.0 + 0.0j]])
    for index in range(sites):
        value = jnp.kron(value, operator if index == site else jnp.eye(2))
    return value


def test_prefix_quadratic_mpo_matches_dense_definition():
    charge = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
    generators = (charge, charge, charge)
    offsets = jnp.asarray([0.2, -0.1, 0.3])
    weights = jnp.asarray([1.0, 0.7, 0.0])
    result = phx.tensor_network.build_prefix_quadratic_mpo(
        generators,
        offsets,
        prefix_weights=weights,
    )
    identity = jnp.eye(8, dtype=jnp.complex128)
    embedded = tuple(_embed(charge, site, 3) for site in range(3))
    expected = jnp.zeros((8, 8), dtype=jnp.complex128)
    for end in range(3):
        prefix = offsets[end] * identity
        for site in range(end + 1):
            prefix = prefix + embedded[site]
        expected = expected + weights[end] * (prefix @ prefix)

    assert result.evidence.hermitian
    assert result.evidence.maximum_bond_dimension <= 3
    assert jnp.allclose(result.operator.to_dense(), expected, atol=1e-12)


def test_prefix_quadratic_mpo_bond_dimension_is_chain_independent():
    charge = jnp.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=jnp.complex128)
    result = phx.tensor_network.build_prefix_quadratic_mpo(
        (charge,) * 12,
        jnp.linspace(-0.2, 0.3, 12),
        prefix_weights=jnp.concatenate((jnp.ones((11,)), jnp.zeros((1,)))),
    )

    assert result.evidence.site_count == 12
    assert result.evidence.active_prefix_count == 11
    assert result.evidence.maximum_bond_dimension == 3


def test_prefix_quadratic_mpo_handles_one_site_and_rejects_nonhermitian_input():
    charge = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])
    result = phx.tensor_network.build_prefix_quadratic_mpo(
        (charge,),
        jnp.asarray([0.4]),
        prefix_weights=jnp.asarray([1.3]),
    )
    expected = 1.3 * (0.4 * jnp.eye(2) + charge) @ (0.4 * jnp.eye(2) + charge)

    assert jnp.allclose(result.operator.to_dense(), expected)
    with pytest.raises(ValueError, match="Hermitian"):
        phx.tensor_network.build_prefix_quadratic_mpo(
            (jnp.asarray([[0.0, 1.0], [0.0, 0.0]]),),
            jnp.asarray([0.0]),
        )

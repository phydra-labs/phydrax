#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical local product factors including explicit Jordan--Wigner strings."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ._compile import CompiledMonomial, PreparedQuantumLattice


def monomial_product_factors(
    prepared: PreparedQuantumLattice, monomial: CompiledMonomial, /
) -> tuple[Array, ...]:
    """Return site-ordered matrices whose tensor product is the monomial."""
    if not isinstance(prepared, PreparedQuantumLattice) or not isinstance(
        monomial, CompiledMonomial
    ):
        raise TypeError("Expected a prepared lattice and compiled monomial.")
    spaces = prepared.specification.spaces
    local = [jnp.eye(space.dimension, dtype=jnp.complex128) for space in spaces]
    order = prepared.specification.fermion_mode_order
    for factor in monomial.factors:
        if factor.fermion_parity:
            if order is None or factor.space.fermion_mode_label is None:
                raise ValueError("Odd fermion factors require FermionModeOrder.")
            ordinal = order.ordinal(factor.space.fermion_mode_label)
            predecessors = set(order.labels[:ordinal])
            for index, space in enumerate(spaces):
                if space.fermion_mode_label in predecessors:
                    parity = jnp.diag(jnp.asarray((1.0, -1.0), dtype=jnp.complex128))
                    local[index] = local[index] @ parity
        site = prepared.specification.site_ids.index(factor.space.site_id)
        local[site] = local[site] @ factor.matrix
    return tuple(local)


__all__ = ["monomial_product_factors"]

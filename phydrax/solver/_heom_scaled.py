#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import BathCorrelationExpansion
from ._heom import HEOMHierarchy


class ScaledHEOMTopology(StrictModule):
    hierarchy: HEOMHierarchy
    source_edges: Array
    target_edges: Array
    term_edges: Array
    upward_edges: Array
    scaling_factors: Array
    valid: Array

    def __init__(
        self,
        hierarchy: HEOMHierarchy,
        expansion: BathCorrelationExpansion,
        /,
    ):
        if hierarchy.term_count != expansion.rank:
            raise ValueError("Hierarchy and bath expansion ranks differ.")
        if jnp.any(jnp.abs(expansion.coefficients) == 0.0):
            raise ValueError("Scaled HEOM requires nonzero retained coefficients.")
        sources = []
        targets = []
        terms = []
        upward = []
        for source in range(hierarchy.auxiliary_count):
            for term in range(hierarchy.term_count):
                up = int(hierarchy.upward[source, term])
                down = int(hierarchy.downward[source, term])
                if up >= 0:
                    sources.append(source)
                    targets.append(up)
                    terms.append(term)
                    upward.append(True)
                if down >= 0:
                    sources.append(source)
                    targets.append(down)
                    terms.append(term)
                    upward.append(False)
        occupations = hierarchy.multi_indices
        log_factors = 0.5 * jnp.sum(
            jsp.special.gammaln(occupations + 1.0)
            + occupations * jnp.log(jnp.abs(expansion.coefficients))[None, :],
            axis=-1,
        )
        self.hierarchy = hierarchy
        self.source_edges = jnp.asarray(sources, dtype=jnp.int32)
        self.target_edges = jnp.asarray(targets, dtype=jnp.int32)
        self.term_edges = jnp.asarray(terms, dtype=jnp.int32)
        self.upward_edges = jnp.asarray(upward, dtype=bool)
        self.scaling_factors = jnp.exp(log_factors)
        self.valid = jnp.all(jnp.isfinite(self.scaling_factors)) & jnp.all(
            self.scaling_factors > 0.0
        )

    def scale(self, auxiliaries: ArrayLike, /) -> Array:
        values = jnp.asarray(auxiliaries)
        return values / self.scaling_factors[:, None, None]

    def unscale(self, auxiliaries: ArrayLike, /) -> Array:
        values = jnp.asarray(auxiliaries)
        return values * self.scaling_factors[:, None, None]


def prepare_scaled_heom_topology(
    term_count: int,
    depth: int,
    expansion: BathCorrelationExpansion,
    /,
    *,
    maximum_auxiliaries: int,
    maximum_edges: int,
) -> ScaledHEOMTopology:
    count = comb(int(term_count) + int(depth), int(depth))
    edge_bound = 2 * count * int(term_count)
    if count > int(maximum_auxiliaries):
        raise ValueError(
            f"HEOM auxiliary capacity exceeded: required {count}, "
            f"available {maximum_auxiliaries}."
        )
    if edge_bound > int(maximum_edges):
        raise ValueError(
            f"HEOM edge capacity exceeded: required at most {edge_bound}, "
            f"available {maximum_edges}."
        )
    hierarchy = HEOMHierarchy(term_count, depth)
    topology = ScaledHEOMTopology(hierarchy, expansion)
    if topology.source_edges.shape[0] > int(maximum_edges):
        raise ValueError("Prepared HEOM topology exceeded exact edge capacity.")
    return topology


__all__ = ["ScaledHEOMTopology", "prepare_scaled_heom_topology"]

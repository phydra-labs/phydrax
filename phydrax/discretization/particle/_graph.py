#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._core import ParticleDiscretization
from ._neighborhood import ParticleNeighborhoodState
from ._pairwise import particle_pair_geometry, ParticlePairGeometry


def particle_graph_view(
    particles: ParticleDiscretization,
    neighborhood: ParticleNeighborhoodState,
    position: ArrayLike,
    /,
    *,
    directed: bool = True,
    edge_mask: ArrayLike | None = None,
    geometry: ParticlePairGeometry | None = None,
):
    """Return the exact fixed-capacity particle relation as a GraphIR."""

    from ...graph import GraphIR

    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be a ParticleDiscretization.")
    if not isinstance(neighborhood, ParticleNeighborhoodState):
        raise TypeError("neighborhood must be a ParticleNeighborhoodState.")
    value = jnp.asarray(position)
    expected = (particles.capacity, particles.ambient_dimension)
    if value.shape != expected:
        raise ValueError(f"Particle positions must have shape {expected}.")
    active = particles.active_mask[:, None]
    value = eqx.error_if(
        value,
        jnp.any(jnp.where(active, ~jnp.isfinite(value), False)),
        "Active particle positions must be finite.",
    )
    value = neighborhood.require_success(jnp.where(active, value, 0.0))
    pairs = neighborhood.pair_relation
    geometry_ = (
        particle_pair_geometry(value, pairs, box=neighborhood.box)
        if geometry is None
        else geometry
    )
    if geometry_.relation_schema_id != pairs.relation_schema_id:
        raise ValueError("Pair geometry and neighborhood relation schemas differ.")
    valid = pairs.valid
    if edge_mask is not None:
        mask = jnp.asarray(edge_mask, dtype=bool)
        if mask.shape != pairs.relation.route_shape:
            raise ValueError("edge_mask must have the pair-relation route shape.")
        valid = valid & mask
    senders = pairs.left_indices
    receivers = pairs.right_indices
    displacement = geometry_.displacement
    distance = geometry_.distance[:, None]
    left_ids = pairs.left_particle_ids[:, None]
    right_ids = pairs.right_particle_ids[:, None]
    if directed:
        senders = jnp.concatenate((senders, receivers), axis=0)
        receivers = jnp.concatenate((pairs.right_indices, pairs.left_indices), axis=0)
        displacement = jnp.concatenate((displacement, -displacement), axis=0)
        distance = jnp.concatenate((distance, distance), axis=0)
        left_ids, right_ids = (
            jnp.concatenate((left_ids, right_ids), axis=0),
            jnp.concatenate((right_ids, left_ids), axis=0),
        )
        valid = jnp.concatenate((valid, valid), axis=0)
    edge_capacity = int(senders.shape[0])
    return GraphIR(
        nodes={
            "position": value,
            "mass": jnp.where(
                particles.active_mask,
                particles.safe_masses,
                0.0,
            )[:, None],
            "particle_id": particles.particle_ids[:, None],
        },
        edges={
            "displacement": displacement,
            "distance": distance,
            "left_particle_id": left_ids,
            "right_particle_id": right_ids,
        },
        senders=senders,
        receivers=receivers,
        globals={
            "pair_count": neighborhood.pair_count[None],
            "successful": neighborhood.successful[None],
        },
        n_node=jnp.asarray([particles.capacity], dtype=jnp.int32),
        n_edge=jnp.asarray([edge_capacity], dtype=jnp.int32),
        node_mask=particles.active_mask,
        edge_mask=valid,
        graph_mask=jnp.asarray([True]),
    )


__all__ = ["particle_graph_view"]

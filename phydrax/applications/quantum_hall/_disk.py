#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite quantum Hall disk sectors with explicit confinement and edge weights."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._identity import HallChargeSector, MonopoleLandauLevel
from ._multi_landau import (
    MultiLandauLevelSpherePlan,
    prepare_multi_landau_level_sphere,
    PreparedMultiLandauLevelSphere,
    ProjectedOrbitalTerm,
)


class HallDiskPlan(StrictModule, NonTrainableState):
    particle_count: int = eqx.field(static=True)
    manifold: MonopoleLandauLevel
    twice_projection: int = eqx.field(static=True)
    confinement: tuple[float, ...] = eqx.field(static=True)
    interactions: tuple[ProjectedOrbitalTerm, ...] = eqx.field(static=True)
    edge_orbitals: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        manifold: MonopoleLandauLevel,
        twice_projection: int,
        confinement: Sequence[float],
        interactions: Sequence[ProjectedOrbitalTerm],
        /,
        *,
        edge_orbitals: int = 2,
    ):
        particles = int(particle_count)
        projection = int(twice_projection)
        confinement_ = tuple(float(value) for value in confinement)
        interactions_ = tuple(interactions)
        edge = int(edge_orbitals)
        if not isinstance(manifold, MonopoleLandauLevel):
            raise TypeError("manifold must be MonopoleLandauLevel.")
        if (
            particles < 1
            or particles > manifold.orbital_count
            or len(confinement_) != manifold.orbital_count
            or any(not isfinite(value) for value in confinement_)
            or any(not isinstance(value, ProjectedOrbitalTerm) for value in interactions_)
            or edge < 1
            or edge > manifold.orbital_count
        ):
            raise ValueError(
                "Disk particles, confinement, interactions, or edge width are invalid."
            )
        self.particle_count = particles
        self.manifold = manifold
        self.twice_projection = projection
        self.confinement = confinement_
        self.interactions = interactions_
        self.edge_orbitals = edge
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hall-disk-plan",
                "particle_count": particles,
                "manifold": manifold.manifold_id,
                "twice_projection": projection,
                "confinement": confinement_,
                "interactions": tuple(value.term_id for value in interactions_),
                "edge_orbitals": edge,
            }
        )


class PreparedHallDisk(StrictModule, NonTrainableState):
    plan: HallDiskPlan
    many_body: PreparedMultiLandauLevelSphere
    prepared_id: str = eqx.field(static=True)


class HallDiskObservables(StrictModule, NonTrainableState):
    orbital_occupations: Array
    edge_particle_number: Array
    norm_residual: Array
    result_id: str = eqx.field(static=True)


def prepare_hall_disk(plan: HallDiskPlan, /) -> PreparedHallDisk:
    if not isinstance(plan, HallDiskPlan):
        raise TypeError("plan must be HallDiskPlan.")
    one_body = tuple(
        ProjectedOrbitalTerm((index,), (index,), value)
        for index, value in enumerate(plan.confinement)
        if value != 0.0
    )
    many_body_plan = MultiLandauLevelSpherePlan(
        plan.particle_count,
        (plan.manifold,),
        HallChargeSector(
            {
                "particle-number": plan.particle_count,
                "twice-projection": plan.twice_projection,
            }
        ),
        one_body + plan.interactions,
    )
    prepared = prepare_multi_landau_level_sphere(many_body_plan)
    return PreparedHallDisk(
        plan,
        prepared,
        canonical_fingerprint(
            {
                "kind": "prepared-hall-disk",
                "plan": plan.plan_id,
                "many_body": prepared.prepared_id,
            }
        ),
    )


def evaluate_hall_disk_observables(
    prepared: PreparedHallDisk,
    state: ArrayLike,
    /,
) -> HallDiskObservables:
    if not isinstance(prepared, PreparedHallDisk):
        raise TypeError("prepared must be PreparedHallDisk.")
    vector = jnp.asarray(state)
    dimension = prepared.many_body.basis.dimension
    if vector.shape != (dimension,):
        raise ValueError("Disk state has the wrong sector dimension.")
    norm = jnp.real(jnp.vdot(vector, vector))
    normalized = vector / jnp.sqrt(jnp.maximum(norm, jnp.finfo(norm.dtype).tiny))
    coordinates = jnp.stack(
        tuple(prepared.many_body.basis.coordinate(index) for index in range(dimension))
    )
    probabilities = jnp.abs(normalized) ** 2
    occupations = jnp.sum(probabilities[:, None] * coordinates, axis=0)
    edge = jnp.sum(occupations[-prepared.plan.edge_orbitals :])
    return HallDiskObservables(
        occupations,
        edge,
        jnp.abs(norm - 1.0),
        canonical_fingerprint(
            {"kind": "hall-disk-observables", "prepared": prepared.prepared_id}
        ),
    )


__all__ = [
    "HallDiskObservables",
    "HallDiskPlan",
    "PreparedHallDisk",
    "evaluate_hall_disk_observables",
    "prepare_hall_disk",
]

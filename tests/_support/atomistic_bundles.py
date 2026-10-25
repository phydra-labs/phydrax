#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit neighborhood-bundle resources of atomistic test fixtures.

Every fixture states its image search, skin, search capacity ladder, role
capacity ladder, complete-pair budget and (for finite systems with bounded
requirements) its nonperiodic search frame.  Requirements are taken from the
program's canonical ``requirements.neighborhoods``; nothing here chooses a
resource on the fixture's behalf.
"""

from __future__ import annotations

from typing import Any

from jax.typing import ArrayLike

from phydrax.atomistic._cell_dynamics import bind_atomistic_bundle_geometry
from phydrax.atomistic._neighborhood_bundle import (
    AbstractAtomisticImageSearchPolicy,
    AtomisticCompletePairSource,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)


def search_capacity(
    width: int, /, *, particles_per_cell: int, images: int
) -> ParticleImageCapacity:
    """Single-case segmented search storage of ``width`` directed routes."""
    return ParticleImageCapacity(
        maximum_particles_per_cell=particles_per_cell,
        maximum_edges=width,
        maximum_degree=width,
        maximum_images=images,
        maximum_case_edges=width,
    )


def role_capacity(routes: int, degree: int, /) -> AtomisticRoleCapacity:
    return AtomisticRoleCapacity(maximum_routes=routes, maximum_degree=degree)


def neighborhood_bundle_plan(
    requirements: tuple[AtomisticNeighborhoodRequirement, ...],
    /,
    *,
    search: AbstractAtomisticImageSearchPolicy | None = None,
    skin: float | None = None,
    search_capacities: tuple[ParticleImageCapacity, ...] = (),
    role_capacities: tuple[AtomisticRoleCapacity, ...] = (),
    maximum_pairs: int | None = None,
    frame: PeriodicCell | None = None,
) -> AtomisticNeighborhoodBundlePlan:
    """One image source per searched entity and one complete source per entity.

    Bounded requirements of one entity share one declared image source with
    the given search, skin and search capacity ladder; each bounded
    requirement binds the declared role capacity ladder.  Complete
    (``cutoff=None``) requirements are served by a complete-pair source of the
    explicit ``maximum_pairs`` budget.
    """
    values = tuple(requirements)
    bounded = tuple(value for value in values if not value.complete)
    complete = tuple(value for value in values if value.complete)
    if bounded and (
        search is None or skin is None or not search_capacities or not role_capacities
    ):
        raise ValueError(
            "Bounded requirements need an explicit search, skin, search capacities "
            "and role capacities."
        )
    if complete and maximum_pairs is None:
        raise ValueError("Complete pair requirements need an explicit maximum_pairs.")
    sources: list[Any] = []
    for entity in dict.fromkeys(value.entity for value in bounded):
        assert search is not None and skin is not None
        sources.append(
            AtomisticNeighborhoodSource(
                f"images-{entity.value}",
                entity,
                tuple(value for value in bounded if value.entity is entity),
                search,
                ParticleImageCapacityLadder(tuple(search_capacities)),
                skin=skin,
            )
        )
    for entity in dict.fromkeys(value.entity for value in complete):
        assert maximum_pairs is not None
        sources.append(
            AtomisticCompletePairSource(
                f"complete-{entity.value}",
                entity,
                tuple(value for value in complete if value.entity is entity),
                maximum_pairs=maximum_pairs,
            )
        )
    ladders = tuple(
        AtomisticRoleCapacityLadder(value, tuple(role_capacities)) for value in bounded
    )
    return AtomisticNeighborhoodBundlePlan(values, tuple(sources), ladders, frame=frame)


def prepare_program(
    program: Any,
    system: Any,
    /,
    *,
    graph_execution: Any = None,
    **resources: Any,
) -> Any:
    """Prepare ``program`` with the explicit bundle of its own requirements."""
    plan = neighborhood_bundle_plan(program.requirements.neighborhoods, **resources)
    return program.prepare(
        system, neighborhood_bundle=plan, graph_execution=graph_execution
    )


def prepare_force_field(
    plan: Any, /, *, graph_execution: Any = None, **resources: Any
) -> Any:
    """Prepare a force field with the explicit bundle of its program requirements."""
    bundle = neighborhood_bundle_plan(
        plan.potential.requirements.neighborhoods, **resources
    )
    return plan.prepare(neighborhood_bundle=bundle, graph_execution=graph_execution)


def bundle_state(
    potential: Any,
    positions: ArrayLike,
    /,
    *,
    cell_vectors: ArrayLike | None = None,
    image_counts: ArrayLike | None = None,
) -> AtomisticNeighborhoodBundleState:
    """Initialized single-case bundle state of a prepared Hamiltonian."""
    bundle = potential.neighborhood_bundle
    return bundle.initialize(
        bind_atomistic_bundle_geometry(
            bundle,
            potential.system,
            positions,
            cell_vectors=cell_vectors,
            image_counts=image_counts,
        )
    )

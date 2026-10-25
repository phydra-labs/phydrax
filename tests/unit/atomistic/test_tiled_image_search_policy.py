#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomistic binding and recipe restoration of the scalar/tiled image reference."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._neighborhood_bundle import (
    AbstractAtomisticImageSearchPolicy,
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticNeighborhoodView,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticTiledImageSearch,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.discretization import (
    ImageTilePolicy,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
    TiledImageRouteSearch,
)


Role = AtomisticNeighborhoodRole
Requirement = AtomisticNeighborhoodRequirement
ATOMS = AtomisticSiteDomain.PHYSICAL_ATOMS
CELL = np.asarray([[6.0, 0.0, 0.0], [0.5, 6.2, 0.0], [0.3, -0.4, 6.4]])
IDS = np.asarray([3, 7, 11, 20, 5])
FRACTIONS = np.asarray(
    [
        [0.10, 0.20, 0.30],
        [0.17, 0.24, 0.33],
        [0.55, 0.62, 0.71],
        [0.83, 0.12, 0.88],
        [0.41, 0.93, 0.05],
    ]
)
GRAPH = Requirement("mace", Role.DIRECTED_IMAGE_GRAPH, 3.0)
PAIRS = Requirement("lj", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, 2.0)
TILED = AtomisticTiledImageSearch(
    tiles=ImageTilePolicy(
        receiver_width=2, source_width=3, maximum_tile_candidates=10**6
    ),
    deformation_margin=1.0,
)
DENSE = AtomisticDenseImageSearch(maximum_dense_routes=10**7, deformation_margin=1.0)


def _system() -> PreparedAtomisticSystem:
    return phx.atomistic.AtomisticSystemPlan(
        IDS,
        np.ones(IDS.shape, dtype=np.int32),
        np.ones(IDS.shape),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(IDS.shape, dtype=np.int32),
        cell=PeriodicCell(CELL),
    ).prepare()


def _plan(search: AbstractAtomisticImageSearchPolicy) -> AtomisticNeighborhoodBundlePlan:
    requirements = (GRAPH, PAIRS)
    capacity = ParticleImageCapacity(
        maximum_particles_per_cell=8,
        maximum_edges=1024,
        maximum_degree=1024,
        maximum_images=4096,
        maximum_case_edges=1024,
    )
    source = AtomisticNeighborhoodSource(
        "images",
        ATOMS,
        requirements,
        search,
        ParticleImageCapacityLadder((capacity,)),
        skin=0.4,
    )
    ladders = tuple(
        AtomisticRoleCapacityLadder(
            value, (AtomisticRoleCapacity(maximum_routes=1024, maximum_degree=1024),)
        )
        for value in requirements
    )
    return AtomisticNeighborhoodBundlePlan(requirements, (source,), ladders)


def _routes(
    view: AtomisticNeighborhoodView, positions: np.ndarray
) -> dict[tuple[int, int, tuple[int, ...]], np.ndarray]:
    displacement = np.asarray(
        view.displacement(jnp.asarray(positions), jnp.asarray(CELL[None]))
    )
    relation = view.relation
    mask = np.asarray(view.mask)
    keys = [
        (int(source), int(receiver), tuple(int(value) for value in shift))
        for source, receiver, shift in zip(
            np.asarray(relation.source_particle_ids)[mask],
            np.asarray(relation.receiver_particle_ids)[mask],
            np.asarray(relation.image_shifts)[mask],
            strict=True,
        )
    ]
    assert len(set(keys)) == len(keys)
    return dict(zip(keys, displacement[mask], strict=True))


def test_tiled_policy_prepares_the_tiled_route_and_matches_dense_views() -> None:
    positions = FRACTIONS @ CELL
    states = {}
    for name, search in (("tiled", TILED), ("dense", DENSE)):
        prepared = _plan(search).prepare(_system())
        state = prepared.initialize(prepared.geometry(jnp.asarray(positions[None])))
        assert bool(np.all(np.asarray(state.successful)))
        states[name] = (prepared, state)
    prepared, state = states["tiled"]
    route_search = TILED.search(
        3.4,
        CELL[None],
        np.ones((1, 3), dtype=np.bool_),
        ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=64,
            maximum_degree=64,
            maximum_images=4096,
        ),
        particle_capacity=len(IDS),
    )
    assert isinstance(route_search, TiledImageRouteSearch)
    assert route_search.tiles.policy_id == TILED.tiles.policy_id
    dense_prepared, dense_state = states["dense"]
    for requirement in (GRAPH, PAIRS):
        tiled_routes = _routes(prepared.view(state, requirement), positions)
        dense_routes = _routes(dense_prepared.view(dense_state, requirement), positions)
        assert set(tiled_routes) == set(dense_routes)
        assert tiled_routes
        for key, value in tiled_routes.items():
            np.testing.assert_allclose(value, dense_routes[key], atol=1e-12)


def test_tiled_policy_recipe_replays_through_constructors() -> None:
    plan = _plan(TILED)
    recipe = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    restored = AtomisticNeighborhoodBundlePlan.from_dict(recipe)
    assert restored.plan_id == plan.plan_id
    assert restored.to_dict() == plan.to_dict()
    assert plan.plan_id != _plan(DENSE).plan_id
    (source,) = restored.sources
    assert isinstance(source, AtomisticNeighborhoodSource)
    assert isinstance(source.search, AtomisticTiledImageSearch)
    assert source.search.policy_id == TILED.policy_id
    assert source.search.tiles.policy_id == TILED.tiles.policy_id


def test_tiled_policy_refuses_invalid_margins() -> None:
    with pytest.raises(ValueError, match="deformation_margin"):
        AtomisticTiledImageSearch(tiles=TILED.tiles, deformation_margin=-1.0)

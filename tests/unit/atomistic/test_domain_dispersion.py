"""Domain dispersion terms: LJ-PME and FourierD3 against the single-device program.

`AtomisticDomainLennardJonesPMETerm` executes the screened LJ-PME pairs and
their topology-exception correction owner-locally on the term's own
interaction-site pair role and its reciprocal plus self energy through the
distributed mesh; `AtomisticDomainFourierD3Term` sums the modified
coordination numbers owner-locally on its full directed coordination role and
distributes the coefficient-weighted mesh. Every reference is the SAME
canonical term evaluated by the single-device `AtomisticPotentialProgram` over
its own neighborhood bundle. Owners are lane-reference vmap lanes (the
functional oracle), for the halo and graph-partition strategies and for the
slab and pencil reciprocal schedules.

Tolerances (float64): energy and per-term energies ``rtol=1e-9, atol=1e-10``;
forces and stress ``rtol=1e-7, atol=1e-9``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_state import AtomisticDomainStrategy
from phydrax.atomistic._domain_terms import (
    AtomisticDomainEwaldTerm,
    AtomisticDomainFourierD3Term,
    AtomisticDomainLennardJonesPMETerm,
)
from phydrax.atomistic._electrostatics import ParticleMeshEwaldPotential
from phydrax.atomistic._graph_partition import AtomisticGraphPartitionPlan
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodRole,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
)
from phydrax.atomistic._sites import (
    AtomisticCoordinateMapPlan,
    AtomisticInteractionSitePlan,
    VirtualSiteKind,
    VirtualSiteRule,
)
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.sparse._streamed import StreamedRelationPlan
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_domain import ownership


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
CELL = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
SKIN = 0.4
LOCAL = 24
C6 = np.asarray([[0.1, 0.2], [0.2, 0.5]])
LJ_PME_CUTOFF = 2.8
MESH = (16, 16, 16)

STRATEGIES = pytest.mark.parametrize(
    "strategy",
    [AtomisticDomainStrategy.HALO, AtomisticDomainStrategy.GRAPH_PARTITION],
    ids=["halo", "graph-partition"],
)
# Both canonical distributed FFT layouts: x-slabs over two owners and the
# (2, 2) pencil grid over four owners.
LAYOUTS = pytest.mark.parametrize(
    ("owners", "layout"),
    [(2, {}), (4, {"schedule": "pencil", "pencil_grid": (2, 2)})],
    ids=["slab-2", "pencil-2x2"],
)


def _configuration(seed: int, count: int, cell: np.ndarray, minimum: float) -> np.ndarray:
    """Rejection-sampled positions with minimum-image separations above ``minimum``."""
    rng = np.random.default_rng(seed)
    inverse = np.linalg.inv(cell)
    points: list[np.ndarray] = []
    while len(points) < count:
        candidate = rng.uniform(0.0, 1.0, 3) @ cell
        fractional = (candidate - np.asarray(points).reshape((-1, 3))) @ inverse
        fractional -= np.round(fractional)
        if np.all(np.linalg.norm(fractional @ cell, axis=1) >= minimum):
            points.append(candidate)
    return np.asarray(points)


def _minimum_image(positions: np.ndarray, cell: np.ndarray, i: int, j: int) -> float:
    fractional = (positions[i] - positions[j]) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ cell))


def _plan(
    system: Any,
    terms: tuple[Any, ...],
    owners: int,
    strategy: AtomisticDomainStrategy,
) -> AtomisticDomainPlan:
    """Terms-only domain plan: one role per term requirement (``term_key``)."""
    cell = system.cell
    placement = ownership(owners, local_capacity=LOCAL)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    roles = tuple(
        AtomisticDomainRolePlan(
            requirement,
            cell.image_stencil(
                requirement.bounded_cutoff() + SKIN, maximum_image_count=4096
            ),
            streaming=StreamedRelationPlan(accumulation="deterministic"),
            skin=SKIN,
            edge_capacity=4096,
            alias_capacity=4096,
            halo_capacity=LOCAL,
            term_key=term.term_key,
        )
        for term in terms
        for requirement in term.requirements()
    )
    graph = (
        AtomisticGraphPartitionPlan(
            placement,
            partition,
            roles,
            replicated_memory_bytes=1 << 24,
            receiver_edge_capacity=512,
            maximum_particles_per_cell=32,
            search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 22),
        )
        if strategy is AtomisticDomainStrategy.GRAPH_PARTITION
        else None
    )
    return AtomisticDomainPlan(
        placement,
        partition,
        roles,
        migration_capacity=LOCAL,
        message_capacity_bytes=1 << 24,
        strategy=strategy,
        graph_partition=graph,
        terms=terms,
    )


def _reference(terms: list[Any], system: Any, positions: np.ndarray) -> Any:
    """The same canonical terms on one device over their own bundle."""
    prepared = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(terms),
        system,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
        skin=SKIN,
        search_capacities=(search_capacity(4096, particles_per_cell=32, images=343),),
        role_capacities=(role_capacity(4096, 256),),
    )
    evaluation = prepared.evaluate(
        jnp.asarray(positions), bundle_state(prepared, positions), compute_stress=True
    )
    assert bool(evaluation.successful)
    return evaluation


def _assert_matches(evaluation: Any, reference: Any, state: Any) -> None:
    assert bool(evaluation.successful), int(evaluation.reasons)
    assert bool(jnp.all(evaluation.status.term_capacity_successful))
    np.testing.assert_allclose(
        float(evaluation.energy), float(reference.energy), rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.term_energies),
        np.asarray(reference.term_energies),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(reference.forces),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), np.asarray(reference.stress), rtol=1e-7, atol=1e-9
    )


# --- LJ-PME dispersion ----------------------------------------------------------


def _lj_pme_case() -> tuple[Any, np.ndarray, Any, Any]:
    """Twelve typed charged atoms whose three nearest pairs are scaled exceptions."""
    count = 12
    positions = _configuration(3, count, CELL, 1.0)
    rng = np.random.default_rng(4)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    types = rng.integers(0, 2, count)
    near = sorted(
        (_minimum_image(positions, CELL, i, j), i, j)
        for i in range(count)
        for j in range(i + 1, count)
    )
    # Every exception lies inside the LJ-PME cutoff, so its scaled pair (the
    # exception correction of the reciprocal sum) is part of the real space.
    assert near[2][0] < LJ_PME_CUTOFF
    exceptions = np.asarray([[i, j] for _, i, j in near[:3]], dtype=np.int64)
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        np.ones((count,), dtype=np.int64),
        np.ones((count,)),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=types,
        charges=charges,
        cell=PeriodicCell(CELL),
        topology=MolecularTopologyPlan(
            pair_exceptions=exceptions,
            lennard_jones_scales=np.full((3,), 0.5),
            electrostatic_scales=np.full((3,), 0.8333),
        ),
    ).prepare()
    dispersion = phx.atomistic.LennardJonesPMEPotential(C6, 0.9, LJ_PME_CUTOFF, MESH)
    pme = ParticleMeshEwaldPotential(1.1, 2.8, MESH, spline_degree=4)
    return system, positions, dispersion, pme


@LAYOUTS
@STRATEGIES
def test_lj_pme_with_exceptions_and_pme_match_single_device_program(
    strategy: AtomisticDomainStrategy, owners: int, layout: dict[str, Any]
) -> None:
    system, positions, dispersion, pme = _lj_pme_case()
    prepared_dispersion = dispersion.prepare(system)
    prepared_pme = pme.prepare(system)
    terms = (
        AtomisticDomainLennardJonesPMETerm(
            prepared_dispersion,
            AtomisticReciprocalDistributionPlan(
                prepared_dispersion, reuse_reach=SKIN, **layout
            ),
        ),
        AtomisticDomainEwaldTerm(
            prepared_pme,
            AtomisticReciprocalDistributionPlan(prepared_pme, reuse_reach=SKIN, **layout),
            exception_capacity=8,
        ),
    )
    (requirement,) = terms[0].requirements()
    assert requirement.role is AtomisticNeighborhoodRole.INTERACTION_SITE_PAIRS
    runtime = prepare_atomistic_domain(_plan(system, terms, owners, strategy))
    state = runtime.ingress_global(
        positions,
        np.asarray(system.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    evaluation = runtime.evaluate(state)
    _assert_matches(evaluation, _reference([dispersion, pme], system, positions), state)
    assert not bool(evaluation.atom_energy_complete)


def test_lj_pme_domain_term_refuses_virtual_interaction_sites() -> None:
    """Domain rows are physical atoms: a virtual site has no owned row."""
    ids = np.arange(4, dtype=np.int64)
    sites = AtomisticInteractionSitePlan(
        np.asarray([0, 1, 2, 3, 50]),
        np.asarray([1, 1, 1, 1, 0]),
        np.asarray([0, 1, 0, 1, 1]),
        np.zeros((5,)),
    )
    coordinate_map = AtomisticCoordinateMapPlan(
        ids,
        sites,
        np.asarray([0, 1, 2, 3, -1]),
        virtual_rules=(
            VirtualSiteRule(
                VirtualSiteKind.WEIGHTED, 50, np.asarray([2, 3]), np.asarray([0.5, 0.5])
            ),
        ),
    )
    system = phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones((4,), dtype=np.int64),
        np.ones((4,)),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros((4,), dtype=np.int64),
        cell=PeriodicCell(CELL),
        coordinate_map=coordinate_map,
    ).prepare()
    prepared = phx.atomistic.LennardJonesPMEPotential(
        C6, 0.9, LJ_PME_CUTOFF, MESH
    ).prepare(system)
    reciprocal = AtomisticReciprocalDistributionPlan(prepared, reuse_reach=SKIN)
    with pytest.raises(ValueError, match="identity image"):
        AtomisticDomainLennardJonesPMETerm(prepared, reciprocal)


# --- FourierD3 ------------------------------------------------------------------


def _fourier_d3_case() -> tuple[Any, np.ndarray, Any]:
    """Six separated H/C atoms of the synthetic FourierD3 tables (skew cell)."""
    from tests.unit.atomistic.test_dispersion_fourier_d3 import (
        _system as d3_system,
        _term as d3_term,
        TRICLINIC as D3_CELL,
    )

    numbers = (6, 6, 1, 6, 1, 1)
    positions = _configuration(5, len(numbers), D3_CELL, 1.0)
    return d3_system(numbers, vectors=D3_CELL), positions, d3_term(MESH, degree=4)


@LAYOUTS
@STRATEGIES
def test_fourier_d3_matches_single_device_program(
    strategy: AtomisticDomainStrategy, owners: int, layout: dict[str, Any]
) -> None:
    system, positions, term = _fourier_d3_case()
    prepared = term.prepare(system)
    domain_term = AtomisticDomainFourierD3Term(
        prepared,
        AtomisticReciprocalDistributionPlan(prepared, reuse_reach=SKIN, **layout),
    )
    (requirement,) = domain_term.requirements()
    assert requirement.role is AtomisticNeighborhoodRole.COORDINATION
    runtime = prepare_atomistic_domain(_plan(system, (domain_term,), owners, strategy))
    count = positions.shape[0]
    state = runtime.ingress_global(
        positions, np.zeros((count,), dtype=np.int32), rng_key=jax.random.key(0)
    )
    evaluation = runtime.evaluate(state)
    _assert_matches(evaluation, _reference([term], system, positions), state)

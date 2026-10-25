#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated lifecycle cost of native atomistic neighborhood bundles.

Every row executes the public native ``AtomisticNeighborhoodBundlePlan`` /
``PreparedAtomisticNeighborhoodBundle`` lifecycle and the public
``AtomisticPotentialProgram`` force path on one deterministic workload: a
triclinic periodic box or a finite cluster searched in an explicit
nonperiodic frame, ``--atoms`` atoms on a jittered lattice with optional
smooth density skew and per-case active-set heterogeneity over ``--cases``
cases.  The image search route (dense, cell-list, scalar/tiled or Morton
cluster-tile), the source sharing layout and the capacity scale are swept
independently; searches are executed only through the bundle, never
re-implemented here.

Requirements are the real per-term roles of the selected programs
(Lennard-Jones classical minimum-image pairs, a native PaiNN directed image
graph, and, for ``pair-graph-dispersion``, the D3(BJ) full directed
coordination and explicitly counted image-pair sum) plus bundle-only
requirements: a nearest-neighbor-selected directed graph with a UMA-shaped
``ImageNeighborSelectionPolicy`` (scientific selection, reported separately
from storage overflow), independent same-role coordination tiers and an
optional receiver-subset coordination role.

Lifecycle phases per row, each with host preparation, lowering,
compilation, compiler argument/output/temporary/code bytes, the first
synchronized call and warm repeats: ``initialize``; ``certify`` and
``refresh`` under certified small motion (reuse); ``refresh`` after one case
moves beyond its skin (selective rebuild); ``select_cases`` commit; a cell
deformation refresh (periodic); and the frozen-epoch ``bind`` (role-view
formation without search).  Capacity failures replay the same attempt from
the accepted state through ``grow_capacities``/``rebase`` along the declared
ladders, with every attempt retained; non-finite and over-compressed
candidates retain their scientific failure and the refused replay.  Programs
are prepared on the same resources and measured for state formation,
conservative energy/force(/stress) evaluation before and after certified
reuse.

Masked role routes are compared as exact ``(case, source ID, receiver ID,
image shift)`` multisets per requirement identity (so a selected graph is
only compared with the same selection), and program energy/force
differences are reported across search/sharing/capacity variants of one
workload.  Nothing here ranks routes or claims a speedup: timings,
compiler estimates and logical array bytes belong to the recorded device and
configuration only.  PaiNN parameters are random and D3 tables are
explicitly invented synthetic values; neither qualifies any model or element.

Smoke (small, complete)::

    python -m benchmarks.atomistic_neighbor_routes --smoke \
        --label smoke --output .tmp/atomistic_neighbor_routes_smoke.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import host_allocation_peak, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.atomistic import (
    AbstractAtomisticImageSearchPolicy,
    AtomisticCellListImageSearch,
    AtomisticClusterTileImageSearch,
    AtomisticCompletePairSource,
    AtomisticDenseImageSearch,
    AtomisticGraphExecutionPlan,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticNeighborhoodView,
    AtomisticPairCounting,
    AtomisticPotentialProgram,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticSiteDomain,
    AtomisticSystemPlan,
    AtomisticTiledImageSearch,
    AtomisticUnitSystem,
    D3BJDamping,
    D3BJPotential,
    D3CoordinationPolicy,
    D3DispersionCutoff,
    D3ReferenceTables,
    D3TableProvenance,
    LearnedGraphPotentialTerm,
    LennardJonesPotential,
    PreparedAtomisticNeighborhoodBundle,
    PreparedAtomisticPotentialProgram,
    PreparedAtomisticSystem,
)
from phydrax.discretization import (
    ImageNeighborSelectionPolicy,
    ImageReceiverSubset,
    ImageTilePolicy,
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
    SegmentedImageVerletPlan,
    TiledImageRouteSearch,
)
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import BOHR, HARTREE


Role = AtomisticNeighborhoodRole
Requirement = AtomisticNeighborhoodRequirement
ATOMS = AtomisticSiteDomain.PHYSICAL_ATOMS

_SHEAR = np.asarray([[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [0.07, -0.11, 1.0]])
_STRAIN = np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
_SYSTEMS = ("periodic-triclinic", "finite")
_PROGRAMS = ("pair-graph", "pair-graph-dispersion")
_SHARINGS = ("split", "single", "per-role")
_HALF = (AtomisticPairCounting.MINIMUM_IMAGE_HALF, AtomisticPairCounting.IMAGE_HALF)
_RECEIVER_GROUP = "receiver-probe"
_OWNERS = (
    "phydrax/atomistic/_neighborhood_bundle.py",
    "phydrax/atomistic/_potential_program.py",
    "phydrax/atomistic/_classical.py",
    "phydrax/atomistic/_dispersion_d3.py",
    "phydrax/atomistic/_d3_parameters.py",
    "phydrax/nn/atomistic/_painn.py",
    "phydrax/discretization/particle/_image_neighborhood.py",
    "phydrax/discretization/particle/_image_relation.py",
    "phydrax/discretization/particle/_image_radius_tiers.py",
    "phydrax/discretization/particle/_tiled_image_search.py",
    "phydrax/discretization/particle/_cluster_image_search.py",
    "phydrax/discretization/particle/_verlet.py",
)
_SMOKE: dict[str, Any] = {
    "systems": list(_SYSTEMS),
    "atoms": [16],
    "cases": [2],
    "skews": [0.3],
    "receiver_fractions": [0.5],
    "searches": ["cell-list", "dense", "tiled:4x8", "cluster:4"],
    "sharings": ["split"],
    "capacity_factors": [0.25],
    "programs": list(_PROGRAMS),
    "repeats": 2,
}


# ------------------------------------------------------------------ workload


@dataclass(frozen=True)
class Workload:
    """Deterministic unwrapped case geometry of one system kind and atom count."""

    system_kind: str
    atoms: int
    cases: int
    skew: float
    density: float
    spacing: float
    vectors: np.ndarray
    unwrapped: np.ndarray
    active: np.ndarray
    receivers: np.ndarray | None
    seed: tuple[int, ...]

    @property
    def periodic(self) -> bool:
        return self.system_kind == "periodic-triclinic"


def _skewed(fractions: np.ndarray, skew: float, /) -> np.ndarray:
    """Smooth periodic density skew ``g(f) = f - s sin(2 pi f) / (2 pi)``.

    The per-axis density ratio is bounded by ``1 / (1 - s)``, so no atom pair
    collapses and the map keeps ``[0, 1)`` and its periodic joins.
    """
    return fractions - skew * np.sin(2.0 * math.pi * fractions) / (2.0 * math.pi)


def _skew_factor(skew: float, /) -> float:
    return (1.0 / (1.0 - skew)) ** 3


def _neighbors(density: float, radius: float, skew: float, /) -> float:
    """Upper estimate of the directed routes of one receiver within ``radius``."""
    return density * (4.0 / 3.0) * math.pi * radius**3 * _skew_factor(skew)


def workload(
    system_kind: str,
    atoms: int,
    cases: int,
    skew: float,
    receiver_fraction: float,
    args: argparse.Namespace,
    /,
) -> Workload:
    """Jittered-lattice cases; positions are independent of every route choice."""
    seed = (
        int(args.seed),
        _SYSTEMS.index(system_kind),
        int(atoms),
        int(cases),
        int(round(1000 * skew)),
    )
    rng = np.random.default_rng(seed)
    volume = atoms / args.density
    side = int(math.ceil(atoms ** (1.0 / 3.0) - 1.0e-9))
    sites = np.stack(
        np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    chosen = np.sort(rng.choice(sites.shape[0], size=atoms, replace=False))
    base = (sites[chosen] + 0.5) / side
    jitter = rng.normal(0.0, args.jitter / side, size=(cases, atoms, 3))
    if system_kind == "periodic-triclinic":
        lattice = np.cbrt(volume / np.linalg.det(_SHEAR)) * _SHEAR
        fractions = _skewed(np.mod(base[None] + jitter, 1.0), skew)
        unwrapped = np.einsum("cnd,de->cne", fractions, lattice)
        vectors = lattice
    else:
        length = np.cbrt(volume)
        margin = args.frame_margin
        fractions = _skewed(np.clip(base[None] + jitter, 0.0, 1.0), skew)
        unwrapped = margin + length * fractions
        vectors = (length + 2.0 * margin) * np.eye(3)
    inactive = [
        int(round(args.inactive_fraction * atoms * case / max(cases - 1, 1)))
        for case in range(cases)
    ]
    active = np.ones((cases, atoms), dtype=np.bool_)
    for case, count in enumerate(inactive):
        if count:
            active[case, atoms - count :] = False
    receivers = None
    if receiver_fraction < 1.0:
        rows = max(1, int(math.ceil(receiver_fraction * atoms)))
        always = atoms - max(inactive)
        if rows > always:
            raise ValueError(
                f"{rows} receiver rows exceed the {always} atoms active in every case."
            )
        receivers = np.stack(
            [np.sort(rng.choice(always, size=rows, replace=False)) for _ in range(cases)]
        )
    return Workload(
        system_kind,
        atoms,
        cases,
        float(skew),
        float(args.density),
        float(np.cbrt(1.0 / args.density)),
        np.asarray(vectors, dtype=np.float64),
        np.asarray(unwrapped, dtype=np.float64),
        active,
        receivers,
        seed,
    )


def _particle_ids(atoms: int, /) -> np.ndarray:
    """Stable material IDs deliberately distinct from slot indices."""
    return 5 + 3 * np.arange(atoms, dtype=np.int64)


def _numbers(atoms: int, /) -> np.ndarray:
    return np.where(np.arange(atoms) % 2 == 0, 6, 1).astype(np.int64)


def prepare_system(work: Workload, /) -> PreparedAtomisticSystem:
    numbers = _numbers(work.atoms)
    return AtomisticSystemPlan(
        _particle_ids(work.atoms),
        numbers,
        np.where(numbers == 6, 12.011, 1.008),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=(numbers == 1).astype(np.int64),
        cell=PeriodicCell(work.vectors) if work.periodic else None,
        name=f"neighbor-routes-{work.system_kind}-{work.atoms}",
    ).prepare()


def _frame(work: Workload, /) -> PeriodicCell | None:
    if work.periodic:
        return None
    return PeriodicCell(work.vectors, periodic_axes=(False, False, False))


def _moved(
    unwrapped: np.ndarray,
    amplitudes: np.ndarray,
    rng: np.random.Generator,
    /,
) -> np.ndarray:
    """Per-case random-direction displacements of magnitude in ``[a/2, a]``."""
    direction = rng.normal(size=unwrapped.shape)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    scale = rng.uniform(0.5, 1.0, size=unwrapped.shape[:2] + (1,))
    return unwrapped + direction * scale * amplitudes[:, None, None]


def _deformed(
    unwrapped: np.ndarray, vectors: np.ndarray, deformation: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Affine map of case geometry under the row-lattice change ``H -> H F``."""
    fractions = np.einsum("cnd,cde->cne", unwrapped, np.linalg.inv(vectors))
    changed = np.einsum("cde,ef->cdf", vectors, deformation)
    return np.einsum("cnd,cde->cne", fractions, changed), changed


def _represented(
    unwrapped: np.ndarray, vectors: np.ndarray, periodic: bool, /
) -> tuple[np.ndarray, np.ndarray | None]:
    """Wrapped coordinates and exact integer image counts of the unwrapped ones."""
    if not periodic:
        return unwrapped, None
    fractions = np.einsum("cnd,cde->cne", unwrapped, np.linalg.inv(vectors))
    counts = np.floor(fractions)
    wrapped = np.einsum("cnd,cde->cne", fractions - counts, vectors)
    return wrapped, counts.astype(np.int32)


def _bind(
    bundle: PreparedAtomisticNeighborhoodBundle,
    unwrapped: np.ndarray,
    vectors: np.ndarray,
    periodic: bool,
    active: np.ndarray | None,
    /,
    *,
    nonfinite: bool = False,
) -> Any:
    """Bind case geometry; ``nonfinite`` poisons one coordinate of case 0."""
    wrapped, counts = _represented(unwrapped, vectors, periodic)
    if nonfinite:
        wrapped = wrapped.copy()
        wrapped[0, 0, 0] = np.nan
    return bundle.geometry(
        jnp.asarray(wrapped),
        # A finite system served only by complete pairs has a rank-0 lattice.
        cell_vectors=None if bundle.lattice_rank == 0 else jnp.asarray(vectors),
        image_counts=None if counts is None else jnp.asarray(counts),
        active=None if active is None else jnp.asarray(active),
    )


# ------------------------------------------------------------------ programs


def _d3_tables() -> D3ReferenceTables:
    """Invented two-species (H, C) tables; they qualify no element."""
    valid = np.asarray([[True, True, False], [True, True, True]])
    c6 = np.zeros((2, 2, 3, 3))
    c6[0, 0, :2, :2] = [[3.0, 2.3], [2.3, 1.8]]
    c6[1, 1] = [[47.0, 41.5, 30.2], [41.5, 36.8, 26.0], [30.2, 26.0, 19.5]]
    c6[0, 1, :2, :] = [[11.8, 10.2, 8.1], [8.9, 7.7, 5.9]]
    c6[1, 0] = c6[0, 1].T
    return D3ReferenceTables(
        (1, 6),
        covalent_radii=np.asarray([0.8, 1.9]),
        r4r2=np.asarray([2.0, 3.1]),
        c6_reference=c6,
        reference_coordination=np.asarray([[0.0, 0.95, 0.0], [0.0, 1.2, 2.6]]),
        reference_valid=valid,
        length_unit=BOHR,
        energy_unit=HARTREE,
        covalent_radius_scale=4.0 / 3.0,
        provenance=D3TableProvenance(
            "phydrax neighbor-route benchmark synthetic D3 tables",
            kind="synthetic-fixture",
            license_id="synthetic-benchmark-data",
            use_admitted=True,
            redistribution_admitted=True,
            admission="invented values; resource-lifecycle workload only; "
            "qualifies no element",
        ),
    )


def build_program(
    name: str, work: Workload, args: argparse.Namespace, /
) -> AtomisticPotentialProgram:
    """Real native terms with independent cutoff roles."""
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    terms: list[Any] = [
        LennardJonesPotential(
            np.asarray([0.010, 0.004]),
            np.asarray([2.2, 1.7]),
            args.pair_cutoff,
            name="lennard-jones",
        ),
        LearnedGraphPotentialTerm(
            phx.nn.atomistic.PaiNNPotential(
                units.scale,
                cutoff=args.graph_cutoff,
                feature_count=args.graph_features,
                interaction_count=1,
                radial_basis_count=4,
                key=jr.key(args.seed),
            ),
            name="painn",
            allow_periodic=work.periodic,
        ),
    ]
    if name == "pair-graph-dispersion":
        terms.append(
            D3BJPotential(
                _d3_tables(),
                D3BJDamping(
                    a1=0.4289,
                    a2=4.4407,
                    s8=0.7875,
                    s6=1.0,
                    functional="synthetic BJ parameters (benchmark)",
                ),
                D3CoordinationPolicy("normalized-gaussian"),
                D3DispersionCutoff(args.dispersion_cutoff),
                coordination_cutoff=args.coordination_cutoff,
                route_chunk=args.d3_route_chunk,
                name="d3-bj",
            )
        )
    return AtomisticPotentialProgram(terms)


def _selection_policy(args: argparse.Namespace, /) -> ImageNeighborSelectionPolicy:
    """UMA-shaped declared nearest-neighbor rule (scientific, not storage)."""
    return ImageNeighborSelectionPolicy(
        args.graph_cutoff,
        args.selection_neighbors,
        degeneracy_tolerance=args.selection_tolerance,
        measure="squared-distance",
        strict=False,
        radius_boundary="closed",
        coincidence_threshold=0.0,
    )


def lifecycle_requirements(
    programs: Sequence[AtomisticPotentialProgram],
    work: Workload,
    args: argparse.Namespace,
    /,
) -> tuple[Requirement, ...]:
    """Program term roles plus bundle-only selection, tiers and receiver roles."""
    values = {
        requirement.requirement_id: requirement
        for program in programs
        for requirement in program.requirements.neighborhoods
    }
    extra = [
        Requirement(
            "selected-graph",
            Role.DIRECTED_IMAGE_GRAPH,
            args.graph_cutoff,
            selection=_selection_policy(args),
        )
    ]
    extra += [
        Requirement("coordination-tier", Role.COORDINATION, radius)
        for radius in args.coordination_tiers
    ]
    if work.receivers is not None:
        extra.append(
            Requirement(
                "receiver-probe",
                Role.COORDINATION,
                args.coordination_cutoff,
                receiver_group=_RECEIVER_GROUP,
            )
        )
    for requirement in extra:
        values.setdefault(requirement.requirement_id, requirement)
    return tuple(values.values())


# ------------------------------------------------------------------ resources


def search_policy(
    spec: str, work: Workload, args: argparse.Namespace, /
) -> AbstractAtomisticImageSearchPolicy:
    name, _, parameter = spec.partition(":")
    margin = args.deformation_margin
    match name:
        case "dense":
            return AtomisticDenseImageSearch(
                maximum_dense_routes=args.dense_limit, deformation_margin=margin
            )
        case "cell-list":
            return AtomisticCellListImageSearch(
                maximum_candidate_slots=args.maximum_candidate_slots,
                deformation_margin=margin,
            )
        case "tiled":
            receiver, source = _tiles(parameter)
            return AtomisticTiledImageSearch(
                tiles=ImageTilePolicy(
                    receiver_width=receiver,
                    source_width=source,
                    maximum_tile_candidates=args.maximum_tile_candidates,
                ),
                deformation_margin=margin,
            )
        case "cluster":
            width = int(parameter)
            clusters = int(math.ceil(work.atoms / width))
            radius = args.cluster_radius_factor * work.spacing * width ** (1.0 / 3.0)
            reach = _search_radius(args) + 2.0 * radius
            tiles = _neighbors(work.density / width, reach, work.skew) + 1.0
            case_tiles = max(
                16, int(math.ceil(args.cluster_tile_headroom * clusters * tiles))
            )
            return AtomisticClusterTileImageSearch(
                cluster_width=width,
                maximum_cluster_radius=radius,
                tile_capacity=ParticleImageCapacity(
                    maximum_particles_per_cell=clusters,
                    maximum_edges=work.cases * case_tiles,
                    maximum_degree=max(
                        8, int(math.ceil(args.cluster_tile_headroom * tiles))
                    ),
                    maximum_images=args.maximum_images,
                    maximum_case_edges=case_tiles,
                ),
                maximum_candidate_slots=args.maximum_candidate_slots,
                deformation_margin=margin,
            )
    raise ValueError(f"Unknown search {spec!r}.")


def _search_radius(args: argparse.Namespace, /) -> float:
    radii = [args.pair_cutoff, args.graph_cutoff, args.coordination_cutoff]
    radii += list(args.coordination_tiers)
    if "pair-graph-dispersion" in args.programs:
        radii.append(args.dispersion_cutoff)
    return max(radii) + args.skin


def _grown(base: int, estimate: float, /) -> int:
    """A strictly larger ladder entry than ``base``, covering ``estimate``."""
    return max(2 * base, int(math.ceil(estimate)))


def _search_ladder(
    served: Sequence[Requirement],
    rows: int,
    work: Workload,
    factor: float,
    args: argparse.Namespace,
    /,
) -> ParticleImageCapacityLadder:
    radius = max(value.bounded_cutoff() for value in served) + args.skin
    routes = rows * _neighbors(work.density, radius, work.skew) * 1.25
    degree = 1.5 * _neighbors(work.density, radius, work.skew) + 4.0
    occupancy = 2.0 * work.density * radius**3 * _skew_factor(work.skew) + 8.0
    edges = max(16, int(math.ceil(factor * routes)))
    width = max(8, int(math.ceil(factor * degree)))
    cell = min(work.atoms, max(8, int(math.ceil(factor * occupancy))))

    def entry(case_edges: int, maximum_degree: int, particles: int) -> Any:
        return ParticleImageCapacity(
            maximum_particles_per_cell=particles,
            maximum_edges=work.cases * case_edges,
            maximum_degree=maximum_degree,
            maximum_images=args.maximum_images,
            maximum_case_edges=case_edges,
        )

    return ParticleImageCapacityLadder(
        (
            entry(edges, width, cell),
            entry(
                _grown(edges, args.headroom * routes),
                _grown(width, args.headroom * degree),
                work.atoms,
            ),
        )
    )


def _role_ladder(
    requirement: Requirement,
    rows: int,
    work: Workload,
    factor: float,
    args: argparse.Namespace,
    /,
) -> AtomisticRoleCapacityLadder:
    count = _neighbors(work.density, requirement.bounded_cutoff(), work.skew)
    half = 0.5 if requirement.counting in _HALF else 1.0
    routes = work.cases * rows * count * half * 1.25
    degree = 1.5 * count + 4.0
    base_routes = max(8, int(math.ceil(factor * routes)))
    base_degree = max(4, int(math.ceil(factor * degree)))
    return AtomisticRoleCapacityLadder(
        requirement,
        (
            AtomisticRoleCapacity(maximum_routes=base_routes, maximum_degree=base_degree),
            AtomisticRoleCapacity(
                maximum_routes=_grown(base_routes, args.headroom * routes),
                maximum_degree=_grown(base_degree, args.headroom * degree),
            ),
        ),
    )


def bundle_plan(
    requirements: Sequence[Requirement],
    sharing: str,
    search: AbstractAtomisticImageSearchPolicy,
    work: Workload,
    factor: float,
    cases: int,
    args: argparse.Namespace,
    /,
) -> AtomisticNeighborhoodBundlePlan:
    """Explicit sources, search/role ladders and finite frame; no implicit defaults."""
    case_work = dataclasses.replace(work, cases=cases)
    receiver_rows = None if work.receivers is None else int(work.receivers.shape[1])
    subset = [value for value in requirements if value.receiver_group is not None]
    rest = [value for value in requirements if value.receiver_group is None]
    match sharing:
        case "single":
            groups = [("images", rest)]
        case "split":
            pairs = [
                value
                for value in rest
                if value.role is Role.CLASSICAL_MINIMUM_IMAGE_PAIRS
            ]
            groups = [
                ("pairs", pairs),
                (
                    "images",
                    [
                        value
                        for value in rest
                        if value.role is not Role.CLASSICAL_MINIMUM_IMAGE_PAIRS
                    ],
                ),
            ]
        case "per-role":
            groups = [
                (f"{index:02d}-{value.owner}-{value.role.value}", [value])
                for index, value in enumerate(rest)
            ]
        case _:
            raise ValueError(f"Unknown sharing {sharing!r}.")
    sources = [
        AtomisticNeighborhoodSource(
            name,
            ATOMS,
            tuple(served),
            search,
            _search_ladder(served, work.atoms, case_work, factor, args),
            skin=args.skin,
        )
        for name, served in groups
        if served
    ]
    if subset:
        if receiver_rows is None:
            raise ValueError("A receiver-subset role needs declared receiver rows.")
        sources.append(
            AtomisticNeighborhoodSource(
                "receivers",
                ATOMS,
                tuple(subset),
                search,
                _search_ladder(subset, receiver_rows, case_work, factor, args),
                skin=args.skin,
                receiver_group=_RECEIVER_GROUP,
                receiver_rows=receiver_rows,
            )
        )
    ladders = tuple(
        _role_ladder(
            value,
            work.atoms if value.receiver_group is None else int(receiver_rows or 0),
            case_work,
            factor,
            args,
        )
        for value in requirements
    )
    return AtomisticNeighborhoodBundlePlan(
        tuple(requirements),
        tuple(sources),
        ladders,
        case_count=cases,
        frame=_frame(work),
    )


def _receivers(
    bundle: PreparedAtomisticNeighborhoodBundle, work: Workload, /
) -> dict[str, ImageReceiverSubset] | None:
    if work.receivers is None:
        return None
    ids = _particle_ids(work.atoms)[work.receivers]
    return {
        _RECEIVER_GROUP: ImageReceiverSubset(
            jnp.asarray(ids, dtype=bundle.particle_ids.dtype)
        )
    }


# ------------------------------------------------------------------ evidence


def failure_record(error: BaseException, /) -> dict[str, Any]:
    return {
        "status": "refused" if isinstance(error, (ValueError, TypeError)) else "failed",
        "error_type": type(error).__name__,
        "message": str(error)[:4000],
    }


def _array_values(value: Any, /) -> Any:
    """Keep finite values and explicitly tag exceptional certificate scalars."""
    if isinstance(value, list):
        return [_array_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"kind": "nonfinite", "value": repr(value)}
    return value


def _arrays(value: Any, /) -> dict[str, Any]:
    """Every array field as JSON-safe host values, preserving infinite bounds."""
    return {
        field.name: _array_values(np.asarray(getattr(value, field.name)).tolist())
        for field in dataclasses.fields(value)
        if isinstance(getattr(value, field.name), (jax.Array, np.ndarray))
    }


def _role_evidence(view: AtomisticNeighborhoodView, /) -> dict[str, Any]:
    evidence = view.evidence
    record = _arrays(evidence)
    record.update(
        capacity_failure=np.asarray(evidence.capacity_failure).tolist(),
        scientific_failure=np.asarray(evidence.scientific_failure).tolist(),
        successful=np.asarray(evidence.successful).tolist(),
        selection=None if evidence.selection is None else _arrays(evidence.selection),
    )
    return record


def route_digest(view: AtomisticNeighborhoodView, /) -> dict[str, Any]:
    """Exact sorted multiset of masked ``(case, source, receiver, shift)`` routes."""
    relation = view.relation
    mask = np.asarray(view.mask, dtype=np.bool_)
    count = int(mask.sum())
    keys = np.concatenate(
        [
            np.asarray(relation.route_cases)[mask].reshape(count, 1),
            np.asarray(relation.source_particle_ids)[mask].reshape(count, 1),
            np.asarray(relation.receiver_particle_ids)[mask].reshape(count, 1),
            np.asarray(relation.image_shifts)[mask].reshape(count, -1),
        ],
        axis=1,
    ).astype(np.int64)
    canonical = keys[np.lexsort(keys.T[::-1])] if count else keys
    unique = np.unique(canonical, axis=0).shape[0] if count else 0
    return {
        "routes": count,
        "duplicates": count - unique,
        "sha256": hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest(),
    }


def _views(state: AtomisticNeighborhoodBundleState, /) -> list[dict[str, Any]]:
    return [
        {
            "requirement_id": view.requirement_id,
            "owner": view.owner,
            "role": view.role.value,
            "counting": view.counting.value,
            "selected": view.selected,
            "cutoff": view.cutoff,
            "source": view.source,
            "directed_multiplicity": view.directed_multiplicity,
            "route_slots": int(np.asarray(view.mask).shape[0]),
            "masked_routes": route_digest(view),
            "self_image_routes": int(
                np.sum(np.asarray(view.self_image) & np.asarray(view.mask))
            ),
            "graph_epoch": int(np.asarray(state.graph_epochs)[index]),
            "logical_bytes": logical_array_bytes(view),
            "evidence": _role_evidence(view),
        }
        for index, view in enumerate(state.views)
    ]


def _status(state: AtomisticNeighborhoodBundleState, /) -> dict[str, Any]:
    return {
        "successful": np.asarray(state.successful).tolist(),
        "capacity_failure": np.asarray(state.capacity_failure).tolist(),
        "scientific_failure": np.asarray(state.scientific_failure).tolist(),
        "capacity_only": np.asarray(state.capacity_only).tolist(),
        "uninitialized": np.asarray(state.uninitialized).tolist(),
        "stale": np.asarray(state.stale).tolist(),
        "selection_changed": np.asarray(state.selection_changed).tolist(),
        "source_epochs": np.asarray(state.epochs).tolist(),
        "source_rebuilt": np.asarray(state.rebuilt).tolist(),
        "graph_epochs": np.asarray(state.graph_epochs).tolist(),
        "logical_bytes": logical_array_bytes(state),
        "bundle_id": state.bundle_id,
    }


def _bundle_record(bundle: PreparedAtomisticNeighborhoodBundle, /) -> dict[str, Any]:
    sources = []
    for index, (declared, engine, proof) in enumerate(
        zip(bundle.plan.sources, bundle.engines, bundle.proofs, strict=True)
    ):
        record: dict[str, Any] = {
            "name": declared.name,
            "kind": type(declared).__name__,
            "proof": {
                "served": len(proof.served),
                "shared": proof.shared,
                "countings": list(proof.countings),
                "cutoffs": list(proof.cutoffs),
                "coverage_margins": list(proof.coverage_margins),
                "search_radius": proof.search_radius,
                "interaction_radius": proof.interaction_radius,
                "skin": proof.skin,
                "receiver_group": proof.receiver_group,
                "proof_id": proof.proof_id,
            },
        }
        if isinstance(declared, AtomisticNeighborhoodSource):
            if not isinstance(engine, SegmentedImageVerletPlan):
                raise TypeError("An image source is prepared as a segmented Verlet plan.")
            capacity = bundle.source_capacity(index)
            search = engine.search
            search_record: dict[str, Any] = {
                "kind": type(search).__name__,
                "search_id": search.search_id,
                "candidate_slot_count": search.candidate_slot_count,
                "route_capacity": search.route_capacity,
                "policy_id": declared.search.policy_id,
            }
            if isinstance(search, TiledImageRouteSearch):
                schedule = search.tile_schedule()
                search_record["tile_schedule"] = {
                    "tile_steps": schedule.tile_steps,
                    "workspace_candidates": schedule.workspace_candidates,
                    "traversed_candidates": schedule.traversed_candidates,
                }
            record["search"] = search_record
            record["capacity"] = {
                "level": bundle.source_levels[index],
                "capacity_id": capacity.capacity_id,
                "maximum_particles_per_cell": capacity.maximum_particles_per_cell,
                "maximum_edges": capacity.maximum_edges,
                "maximum_case_edges": capacity.maximum_case_edges,
                "maximum_degree": capacity.maximum_degree,
                "maximum_images": capacity.maximum_images,
            }
        sources.append(record)
    roles = []
    for slot, requirement in enumerate(bundle.plan.requirements):
        role: dict[str, Any] = {
            "label": requirement.label,
            "requirement_id": requirement.requirement_id,
            "cutoff": requirement.cutoff,
            "selection": None
            if requirement.selection is None
            else requirement.selection.policy_id,
            "source": bundle.plan.sources[bundle.plan.role_sources[slot]].name,
            "level": bundle.role_levels[slot],
        }
        if not requirement.complete:
            capacity = bundle.capacity(slot)
            role["maximum_routes"] = capacity.maximum_routes
            role["maximum_degree"] = capacity.maximum_degree
        roles.append(role)
    return {
        "plan_id": bundle.plan.plan_id,
        "layout_id": bundle.layout_id,
        "prepared_id": bundle.prepared_id,
        "source_levels": list(bundle.source_levels),
        "role_levels": list(bundle.role_levels),
        "logical_bytes": logical_array_bytes(bundle),
        "sources": sources,
        "roles": roles,
    }


# ------------------------------------------------------------------ execution


def _phase(
    bundle: PreparedAtomisticNeighborhoodBundle, method: str, /, *, receivers: bool
) -> Callable[..., Any]:
    """Traceable public bundle phase; prepared bundle data are compiled constants.

    With ``receivers`` the last positional argument is the receiver-subset
    mapping (possibly ``None``) forwarded by keyword.
    """
    operation = getattr(bundle, method)

    def phase(*arguments: Any) -> Any:
        if not receivers:
            return operation(*arguments)
        *values, subset = arguments
        return operation(*values, receivers=subset)

    return phase


def _evaluator(
    program: PreparedAtomisticPotentialProgram, stress: bool, /
) -> Callable[[Any, Any], Any]:
    def evaluate(positions: Any, state: Any) -> Any:
        return program.evaluate(positions, state, compute_stress=stress)

    return evaluate


class Executables:
    """Compiled phases keyed by name and prepared layout, each charged once."""

    def __init__(self, repeats: int, /) -> None:
        self.repeats = repeats
        self.compiled: dict[tuple[str, str], Any] = {}
        self.records: list[dict[str, Any]] = []

    def get(
        self, name: str, layout: str, function: Callable[..., Any], *arguments: Any
    ) -> Any:
        key = (name, layout)
        if key not in self.compiled:
            jitted = jax.jit(function)
            compiled, phases = measure_lower_and_compile(
                lambda: jitted.lower(*arguments), lambda lowered: lowered.compile()
            )
            cost = compiled.cost_analysis()
            cost = cost[0] if isinstance(cost, list) else cost
            self.records.append(
                {
                    "phase": name,
                    "layout_id": layout,
                    "phases": asdict(phases),
                    "compiler": asdict(
                        compiler_evidence(cost, compiled.memory_analysis(), source="xla")
                    ),
                    "argument_logical_bytes": logical_array_bytes(arguments),
                }
            )
            self.compiled[key] = compiled
        return self.compiled[key]

    def run(self, compiled: Any, *arguments: Any) -> tuple[Any, dict[str, Any]]:
        result, first = measure_synchronized(lambda: compiled(*arguments))
        _, warm = measure_repeated(
            lambda: compiled(*arguments), warmup=0, repeats=self.repeats
        )
        return result, {"first_seconds": first, "warm": warm.to_dict(unit="seconds")}


@dataclass
class Lifecycle:
    """Mutable host lifecycle of one prepared bundle across capacity growth."""

    bundle: PreparedAtomisticNeighborhoodBundle
    work: Workload
    executables: Executables

    @property
    def receivers(self) -> dict[str, ImageReceiverSubset] | None:
        return _receivers(self.bundle, self.work)

    def geometry(
        self, unwrapped: np.ndarray, vectors: np.ndarray, /, *, nonfinite: bool = False
    ) -> Any:
        return _bind(
            self.bundle,
            unwrapped,
            vectors,
            self.work.periodic,
            self.work.active,
            nonfinite=nonfinite,
        )

    def initialize(self, geometry_of: Callable[[], Any], /) -> tuple[Any, list[Any]]:
        """Form every source; capacity-only failures replay along the ladders."""
        attempts: list[dict[str, Any]] = []
        while True:
            bundle = self.bundle
            geometry = geometry_of()
            compiled = self.executables.get(
                "initialize",
                bundle.layout_id,
                _phase(bundle, "initialize", receivers=True),
                geometry,
                self.receivers,
            )
            state, timing = self.executables.run(compiled, geometry, self.receivers)
            attempt = {**self._levels(), **timing, "status": _status(state)}
            attempts.append(attempt)
            if not self._grow(state, attempt):
                return state, attempts

    def refresh(
        self, accepted: Any, geometry_of: Callable[[], Any], /
    ) -> tuple[Any, Any, list[Any]]:
        """Pure candidate refresh with same-attempt capacity replay."""
        attempts: list[dict[str, Any]] = []
        while True:
            bundle = self.bundle
            geometry = geometry_of()
            compiled = self.executables.get(
                "refresh",
                bundle.layout_id,
                _phase(bundle, "refresh", receivers=True),
                accepted,
                geometry,
                self.receivers,
            )
            candidate, timing = self.executables.run(
                compiled, accepted, geometry, self.receivers
            )
            attempt = {**self._levels(), **timing, "status": _status(candidate)}
            attempts.append(attempt)
            if not self._grow(candidate, attempt):
                return candidate, accepted, attempts
            accepted, seconds = measure_synchronized(
                functools.partial(self.bundle.rebase, accepted)
            )
            attempt["rebase_seconds"] = seconds

    def commit(self, candidate: Any, accepted: Any, /) -> tuple[Any, dict[str, Any]]:
        bundle = self.bundle
        accept = candidate.successful
        compiled = self.executables.get(
            "select_cases",
            bundle.layout_id,
            _phase(bundle, "select_cases", receivers=False),
            accept,
            candidate,
            accepted,
        )
        state, timing = self.executables.run(compiled, accept, candidate, accepted)
        return state, {
            "accepted_cases": np.asarray(accept).tolist(),
            **timing,
            "status": _status(state),
        }

    def certify(self, accepted: Any, geometry: Any, /) -> dict[str, Any]:
        bundle = self.bundle
        compiled = self.executables.get(
            "certify",
            bundle.layout_id,
            _phase(bundle, "certify", receivers=True),
            accepted,
            geometry,
            self.receivers,
        )
        certificates, timing = self.executables.run(
            compiled, accepted, geometry, self.receivers
        )
        return {
            **timing,
            "certificates": {
                source.name: None if certificate is None else _arrays(certificate)
                for source, certificate in zip(
                    bundle.plan.sources, certificates, strict=True
                )
            },
        }

    def bind(self, accepted: Any, geometry: Any, /) -> tuple[Any, dict[str, Any]]:
        bundle = self.bundle
        compiled = self.executables.get(
            "bind",
            bundle.layout_id,
            _phase(bundle, "bind", receivers=True),
            accepted,
            geometry,
            self.receivers,
        )
        state, timing = self.executables.run(compiled, accepted, geometry, self.receivers)
        return state, {**timing, "status": _status(state)}

    def probe(self, accepted: Any, geometry: Any, /) -> dict[str, Any]:
        """A candidate that is never committed, and the replay decision on it."""
        bundle = self.bundle
        compiled = self.executables.get(
            "refresh",
            bundle.layout_id,
            _phase(bundle, "refresh", receivers=True),
            accepted,
            geometry,
            self.receivers,
        )
        candidate, timing = self.executables.run(
            compiled, accepted, geometry, self.receivers
        )
        record: dict[str, Any] = {
            **timing,
            "status": _status(candidate),
            "views": [
                {
                    "label": f"{view.owner}:{view.role.value}",
                    "evidence": _role_evidence(view),
                }
                for view in candidate.views
            ],
        }
        try:
            grown = bundle.grow_capacities(candidate)
        except (ValueError, RuntimeError) as error:
            record["capacity_replay"] = failure_record(error)
        else:
            record["capacity_replay"] = {
                "status": "grown",
                "source_levels": list(grown.source_levels),
                "role_levels": list(grown.role_levels),
            }
        return record

    def _levels(self) -> dict[str, Any]:
        return {
            "source_levels": list(self.bundle.source_levels),
            "role_levels": list(self.bundle.role_levels),
        }

    def _grow(self, state: Any, attempt: dict[str, Any], /) -> bool:
        """Host capacity transaction; ``True`` when the attempt must replay."""
        if not bool(np.any(np.asarray(state.capacity_failure))):
            return False
        try:
            grown, seconds = measure_host(lambda: self.bundle.grow_capacities(state))
        except (ValueError, RuntimeError) as error:
            attempt["capacity_replay"] = failure_record(error)
            return False
        attempt["capacity_replay"] = {"status": "grown", "host_seconds": seconds}
        self.bundle = grown
        return True


def _prepared[T](operation: Callable[[], T], /) -> tuple[T, dict[str, Any]]:
    """Host preparation time and traced allocation peak (tracing included)."""
    (value, peak), seconds = measure_host(lambda: host_allocation_peak(operation))
    return value, {"seconds": seconds, "host_allocation_peak_bytes": peak}


@dataclass(frozen=True)
class Variant:
    system_kind: str
    atoms: int
    cases: int
    skew: float
    receiver_fraction: float
    search: str
    sharing: str
    capacity_factor: float

    @property
    def workload_key(self) -> tuple[Any, ...]:
        return (
            self.system_kind,
            self.atoms,
            self.cases,
            self.skew,
            self.receiver_fraction,
        )


def _lifecycle(
    variant: Variant,
    work: Workload,
    system: PreparedAtomisticSystem,
    requirements: tuple[Requirement, ...],
    args: argparse.Namespace,
    observation: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Case-batched bundle lifecycle: build, reuse, selective rebuild, views."""
    search = search_policy(variant.search, work, args)
    graphs = {
        value.requirement_id: StreamedRelationPlan()
        for value in requirements
        if value.role is Role.DIRECTED_IMAGE_GRAPH
    }
    bundle, preparation = _prepared(
        lambda: bundle_plan(
            requirements,
            variant.sharing,
            search,
            work,
            variant.capacity_factor,
            work.cases,
            args,
        ).prepare(system, graph_plans=graphs)
    )
    record: dict[str, Any] = {"preparation": preparation}
    executables = Executables(args.repeats)
    life = Lifecycle(bundle, work, executables)
    record["prepared"] = _bundle_record(bundle)
    rng = np.random.default_rng(work.seed + (17,))
    vectors = np.broadcast_to(work.vectors, (work.cases, 3, 3)).copy()
    unwrapped = work.unwrapped

    state, attempts = life.initialize(lambda: life.geometry(unwrapped, vectors))
    record["initialize"] = {"attempts": attempts, "views": _views(state)}
    observation["initial"] = {
        view.requirement_id: route_digest(view)
        for view in state.views
        if bool(np.all(np.asarray(view.evidence.successful)))
    }
    if not bool(np.all(np.asarray(state.successful))):
        record["status"] = "initialization-unsuccessful"
        record["compilations"] = executables.records
        return record

    reuse = _moved(
        unwrapped,
        np.full((work.cases,), 0.5 * args.reuse_fraction * args.skin),
        rng,
    )
    reuse_record: dict[str, Any] = {
        "maximum_displacement": float(np.max(np.linalg.norm(reuse - unwrapped, axis=-1))),
        "certify": life.certify(state, life.geometry(reuse, vectors)),
    }
    record["reuse"] = reuse_record
    candidate, state, attempts = life.refresh(
        state, lambda: life.geometry(reuse, vectors)
    )
    reuse_record["refresh"] = attempts
    state, reuse_record["commit"] = life.commit(candidate, state)

    amplitudes = np.full((work.cases,), 0.5 * args.reuse_fraction * args.skin)
    amplitudes[0] = args.rebuild_fraction * args.skin
    rebuilt = _moved(reuse, amplitudes, rng)
    rebuild_record: dict[str, Any] = {
        "intended_rebuild_cases": [case == 0 for case in range(work.cases)],
        "displacement_amplitudes": amplitudes.tolist(),
    }
    record["selective_rebuild"] = rebuild_record
    candidate, state, attempts = life.refresh(
        state, lambda: life.geometry(rebuilt, vectors)
    )
    rebuild_record["refresh"] = attempts
    state, rebuild_record["commit"] = life.commit(candidate, state)
    current, current_vectors = rebuilt, vectors

    if work.periodic:
        strained, strained_vectors = _deformed(
            rebuilt, vectors, np.eye(3) + args.strain * _STRAIN
        )
        deformation_record: dict[str, Any] = {
            "strain": args.strain,
            "certify": life.certify(state, life.geometry(strained, strained_vectors)),
        }
        record["cell_deformation"] = deformation_record
        candidate, state, attempts = life.refresh(
            state, lambda: life.geometry(strained, strained_vectors)
        )
        deformation_record["refresh"] = attempts
        state, deformation_record["commit"] = life.commit(candidate, state)
        current, current_vectors = strained, strained_vectors

    frozen = _moved(
        current,
        np.full((work.cases,), 0.05 * args.reuse_fraction * args.skin),
        rng,
    )
    bound, record["view_formation"] = life.bind(
        state, life.geometry(frozen, current_vectors)
    )
    record["view_formation"]["views"] = _views(bound)
    record["accepted"] = {"status": _status(state), "views": _views(state)}
    observation["accepted"] = {
        view.requirement_id: route_digest(view)
        for view in state.views
        if bool(np.all(np.asarray(view.evidence.successful)))
    }

    probes = {
        "nonfinite": life.probe(
            state, life.geometry(current, current_vectors, nonfinite=True)
        )
    }
    if work.periodic:
        radius = PeriodicCell(current_vectors[0]).unique_image_radius
        factor = 0.9 * args.pair_cutoff / radius
        if factor < 1.0:
            compressed, compressed_vectors = _deformed(
                current, current_vectors, factor * np.eye(3)
            )
            probes["compressed_below_unique_image"] = {
                "compression": factor,
                **life.probe(state, life.geometry(compressed, compressed_vectors)),
            }
    record["uncommitted_probes"] = probes
    record["final_prepared"] = _bundle_record(life.bundle)
    record["compilations"] = executables.records
    record["status"] = "completed"
    return record


def _evaluation(evaluation: Any, /) -> dict[str, Any]:
    forces = np.asarray(evaluation.forces)
    return {
        "energy": float(np.asarray(evaluation.energy)),
        "term_energies": np.asarray(evaluation.term_energies).tolist(),
        "successful": bool(np.asarray(evaluation.successful)),
        "neighborhood_successful": np.asarray(
            evaluation.neighborhood_successful
        ).tolist(),
        "graph_overflow": np.asarray(evaluation.graph_overflow).tolist(),
        "maximum_force_norm": float(np.max(np.linalg.norm(forces, axis=-1))),
        "net_force_norm": float(np.linalg.norm(np.sum(forces, axis=0))),
        "stress": None
        if evaluation.stress is None
        else np.asarray(evaluation.stress).tolist(),
        "program_id": evaluation.program_id,
    }


def _program_row(
    name: str,
    variant: Variant,
    work: Workload,
    system: PreparedAtomisticSystem,
    args: argparse.Namespace,
    observation: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Single-case Hamiltonian on the same declared resources: state, E/F(/S), reuse."""
    program_plan = build_program(name, work, args)
    search = search_policy(variant.search, work, args)
    requirements = program_plan.requirements.neighborhoods
    single = dataclasses.replace(
        work,
        cases=1,
        unwrapped=work.unwrapped[:1],
        active=np.ones((1, work.atoms), dtype=np.bool_),
        receivers=None,
    )
    graph_degree = max(
        _role_ladder(value, work.atoms, single, variant.capacity_factor, args)
        .entries[-1]
        .maximum_degree
        for value in requirements
        if value.role is Role.DIRECTED_IMAGE_GRAPH
    )
    execution = AtomisticGraphExecutionPlan(graph_degree, backend="particle")
    program, preparation = _prepared(
        lambda: program_plan.prepare(
            system,
            neighborhood_bundle=bundle_plan(
                requirements,
                variant.sharing,
                search,
                single,
                variant.capacity_factor,
                1,
                args,
            ),
            graph_execution=execution,
        )
    )
    record: dict[str, Any] = {
        "preparation": preparation,
        "graph_execution_maximum_neighbors": graph_degree,
        "requirements": [value.label for value in requirements],
        "program_logical_bytes": logical_array_bytes(program),
    }
    executables = Executables(args.repeats)

    def geometry(bundle: PreparedAtomisticNeighborhoodBundle, unwrapped: Any) -> Any:
        return _bind(bundle, unwrapped, single.vectors[None], work.periodic, None)

    attempts: list[dict[str, Any]] = []
    while True:
        bundle = program.neighborhood_bundle
        start = geometry(bundle, single.unwrapped)
        compiled = executables.get(
            "program_initialize",
            bundle.layout_id,
            _phase(bundle, "initialize", receivers=False),
            start,
        )
        state, timing = executables.run(compiled, start)
        attempt = {
            "source_levels": list(bundle.source_levels),
            "role_levels": list(bundle.role_levels),
            **timing,
            "status": _status(state),
        }
        attempts.append(attempt)
        if not bool(np.any(np.asarray(state.capacity_failure))):
            break
        try:
            grown = bundle.grow_capacities(state)
        except (ValueError, RuntimeError) as error:
            attempt["capacity_replay"] = failure_record(error)
            break
        program = program.with_neighborhood_bundle(grown)
        attempt["capacity_replay"] = {"status": "grown"}
    record["initialize"] = attempts
    stress = work.periodic and args.stress

    def evaluate(
        prepared: PreparedAtomisticPotentialProgram, state: Any
    ) -> tuple[Any, dict[str, Any]]:
        compiled = executables.get(
            "evaluate",
            prepared.neighborhood_bundle.layout_id,
            _evaluator(prepared, stress),
            state.geometry.positions[0],
            state,
        )
        result, timing = executables.run(compiled, state.geometry.positions[0], state)
        return result, {**timing, **_evaluation(result)}

    initial, record["evaluate"] = evaluate(program, state)
    observation[name] = {"initial": initial}
    rng = np.random.default_rng(work.seed + (29,))
    moved = _moved(
        single.unwrapped, np.asarray([0.5 * args.reuse_fraction * args.skin]), rng
    )
    bundle = program.neighborhood_bundle
    target = geometry(bundle, moved)
    compiled = executables.get(
        "program_refresh",
        bundle.layout_id,
        _phase(bundle, "refresh", receivers=False),
        state,
        target,
    )
    refreshed, timing = executables.run(compiled, state, target)
    record["reuse_refresh"] = {**timing, "status": _status(refreshed)}
    if bool(np.all(np.asarray(refreshed.successful))):
        reused, record["evaluate_after_reuse"] = evaluate(program, refreshed)
        observation[name]["reused"] = reused
    record["compilations"] = executables.records
    record["status"] = "completed"
    return record


def measure_variant(
    variant: Variant,
    work: Workload,
    system: PreparedAtomisticSystem,
    args: argparse.Namespace,
    /,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One complete row and its in-memory comparison observation."""
    observation: dict[str, Any] = {}
    row: dict[str, Any] = {"variant": asdict(variant), "programs": {}}
    try:
        programs = [build_program(name, work, args) for name in args.programs]
        requirements = lifecycle_requirements(programs, work, args)
        row["lifecycle"] = _lifecycle(
            variant, work, system, requirements, args, observation
        )
    except Exception as error:  # every refusal or failure is a recorded row
        row["lifecycle"] = failure_record(error)
    for name in args.programs:
        try:
            row["programs"][name] = _program_row(
                name, variant, work, system, args, observation
            )
        except Exception as error:  # every refusal or failure is a recorded row
            row["programs"][name] = failure_record(error)
    row["process_peak_rss_bytes"] = process_peak_rss_bytes()
    return row, observation


# ------------------------------------------------------------------ comparison


def _difference(left: Any, right: Any, /) -> dict[str, Any]:
    return {
        "both_successful": bool(np.asarray(left.successful))
        and bool(np.asarray(right.successful)),
        "absolute_energy_difference": float(
            abs(np.asarray(left.energy) - np.asarray(right.energy))
        ),
        "maximum_absolute_force_difference": float(
            np.max(np.abs(np.asarray(left.forces) - np.asarray(right.forces)))
        ),
    }


def comparisons(
    variants: Sequence[Variant], observations: Sequence[dict[str, Any]], /
) -> list[dict[str, Any]]:
    """Exact route multisets per requirement and E/F differences per workload."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, variant in enumerate(variants):
        groups.setdefault(variant.workload_key, []).append(index)
    records = []
    for key, members in groups.items():
        reference = next(
            (index for index in members if observations[index].get("initial")), None
        )
        if reference is None:
            records.append({"workload": list(key), "reference": None, "variants": []})
            continue
        base = observations[reference]
        rows = []
        for index in members:
            if index == reference:
                continue
            other = observations[index]
            routes = {}
            for stage in ("initial", "accepted"):
                left, right = base.get(stage, {}), other.get(stage, {})
                routes[stage] = {
                    identifier: left[identifier]["sha256"] == right[identifier]["sha256"]
                    for identifier in sorted(set(left) & set(right))
                }
            programs = {}
            for name, value in base.items():
                if name in ("initial", "accepted") or name not in other:
                    continue
                programs[name] = {
                    stage: _difference(value[stage], other[name][stage])
                    for stage in ("initial", "reused")
                    if stage in value and stage in other[name]
                }
            rows.append(
                {
                    "variant": asdict(variants[index]),
                    "identical_routes_per_requirement": routes,
                    "programs": programs,
                }
            )
        records.append(
            {
                "workload": list(key),
                "reference": asdict(variants[reference]),
                "variants": rows,
            }
        )
    return records


# ------------------------------------------------------------------ refusals


def _single_bundle(
    plan: AtomisticNeighborhoodBundlePlan,
    system: PreparedAtomisticSystem,
    work: Workload,
    executables: Executables,
    name: str,
    /,
) -> tuple[PreparedAtomisticNeighborhoodBundle, Any, dict[str, Any]]:
    bundle = plan.prepare(system)
    geometry = _bind(
        bundle,
        work.unwrapped[:1],
        work.vectors[None],
        work.periodic,
        None,
    )
    compiled = executables.get(
        name, bundle.layout_id, _phase(bundle, "initialize", receivers=False), geometry
    )
    state, timing = executables.run(compiled, geometry)
    return bundle, state, {**timing, "status": _status(state), "views": _views(state)}


# Each refusal is an independently recorded resource contract of the same workload.
# Keep the explicit probe declarations and their evidence together; numerical work
# remains in the owning prepared bundle rather than a benchmark-side algorithm.
def refusals(
    work: Workload,
    system: PreparedAtomisticSystem,
    args: argparse.Namespace,
    /,
) -> dict[str, Any]:
    """Declared refusals and the selection-versus-storage separation."""
    single = dataclasses.replace(
        work,
        cases=1,
        unwrapped=work.unwrapped[:1],
        active=np.ones((1, work.atoms), dtype=np.bool_),
        receivers=None,
    )
    executables = Executables(args.repeats)
    search = AtomisticCellListImageSearch(
        maximum_candidate_slots=args.maximum_candidate_slots,
        deformation_margin=args.deformation_margin,
    )
    probes: dict[str, Any] = {}

    def attempt(label: str, operation: Callable[[], Any], /) -> None:
        try:
            probes[label] = operation()
        except Exception as error:  # the refusal itself is the evidence
            probes[label] = failure_record(error)

    def top(requirement: Requirement) -> AtomisticRoleCapacity:
        return _role_ladder(requirement, work.atoms, single, 1.0, args).entries[-1]

    def source(served: tuple[Requirement, ...], **kwargs: Any) -> Any:
        ladder = _search_ladder(served, work.atoms, single, 1.0, args)
        return AtomisticNeighborhoodSource(
            "images",
            ATOMS,
            served,
            kwargs.pop("search", search),
            ParticleImageCapacityLadder((ladder.entries[-1],)),
            skin=args.skin,
        )

    selected = Requirement(
        "selected-graph",
        Role.DIRECTED_IMAGE_GRAPH,
        args.graph_cutoff,
        selection=_selection_policy(args),
    )
    adequate = top(selected)
    narrow = AtomisticRoleCapacity(
        maximum_routes=adequate.maximum_routes,
        maximum_degree=max(1, args.selection_neighbors // 2),
    )

    def selection() -> dict[str, Any]:
        _, reference, reference_record = _single_bundle(
            AtomisticNeighborhoodBundlePlan(
                (selected,),
                (source((selected,)),),
                (AtomisticRoleCapacityLadder(selected, (adequate,)),),
                frame=_frame(work),
            ),
            system,
            single,
            executables,
            "selection_adequate_initialize",
        )
        bundle, state, narrow_record = _single_bundle(
            AtomisticNeighborhoodBundlePlan(
                (selected,),
                (source((selected,)),),
                (AtomisticRoleCapacityLadder(selected, (narrow, adequate)),),
                frame=_frame(work),
            ),
            system,
            single,
            executables,
            "selection_narrow_initialize",
        )
        record: dict[str, Any] = {
            "selection_policy_id": selected.selection.policy_id
            if selected.selection is not None
            else None,
            "adequate_storage": reference_record,
            "narrow_storage": narrow_record,
        }
        try:
            grown = bundle.grow_capacities(state)
        except (ValueError, RuntimeError) as error:
            record["narrow_capacity_replay"] = failure_record(error)
            return record
        geometry = _bind(grown, single.unwrapped, work.vectors[None], work.periodic, None)
        compiled = executables.get(
            "selection_replay_initialize",
            grown.layout_id,
            _phase(grown, "initialize", receivers=False),
            geometry,
        )
        replayed, timing = executables.run(compiled, geometry)
        record["narrow_capacity_replay"] = {
            "role_levels": list(grown.role_levels),
            **timing,
            "status": _status(replayed),
            "views": _views(replayed),
            "identical_routes_to_adequate": route_digest(replayed.views[0])["sha256"]
            == route_digest(reference.views[0])["sha256"],
        }
        return record

    attempt("selection_versus_storage", selection)

    pairs = Requirement(
        "lennard-jones", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, args.pair_cutoff
    )

    def exhausted() -> dict[str, Any]:
        tiny = AtomisticRoleCapacity(maximum_routes=1, maximum_degree=1)
        bundle, state, record = _single_bundle(
            AtomisticNeighborhoodBundlePlan(
                (pairs,),
                (source((pairs,)),),
                (AtomisticRoleCapacityLadder(pairs, (tiny,)),),
                frame=_frame(work),
            ),
            system,
            single,
            executables,
            "exhausted_initialize",
        )
        try:
            bundle.grow_capacities(state)
        except (ValueError, RuntimeError) as error:
            record["capacity_replay"] = failure_record(error)
        else:
            record["capacity_replay"] = {"status": "grown"}
        return record

    attempt("exhausted_role_ladder", exhausted)
    attempt(
        "dense_route_limit",
        lambda: (
            AtomisticNeighborhoodBundlePlan(
                (pairs,),
                (
                    source(
                        (pairs,), search=AtomisticDenseImageSearch(maximum_dense_routes=1)
                    ),
                ),
                (AtomisticRoleCapacityLadder(pairs, (top(pairs),)),),
                frame=_frame(work),
            )
            .prepare(system)
            .layout_id
        ),
    )

    complete = Requirement("complete-pairs", Role.CLASSICAL_MINIMUM_IMAGE_PAIRS, None)
    pair_count = work.atoms * (work.atoms - 1) // 2

    def complete_pairs(maximum: int) -> dict[str, Any]:
        _, _, record = _single_bundle(
            AtomisticNeighborhoodBundlePlan(
                (complete,),
                (
                    AtomisticCompletePairSource(
                        "complete", ATOMS, (complete,), maximum_pairs=maximum
                    ),
                ),
                (),
            ),
            system,
            single,
            executables,
            f"complete_pairs_{maximum}",
        )
        return {"declared_pairs": maximum, "expected_pairs": pair_count, **record}

    attempt("complete_pairs", lambda: complete_pairs(pair_count))
    attempt("complete_pairs_under_budget", lambda: complete_pairs(max(0, pair_count - 1)))

    if work.periodic and system.cell is not None:
        radius = system.cell.unique_image_radius

        def unique_image() -> dict[str, Any]:
            beyond = argparse.Namespace(**{**vars(args), "pair_cutoff": 1.02 * radius})
            program = build_program("pair-graph", work, beyond)
            requirements = program.requirements.neighborhoods
            program.prepare(
                system,
                neighborhood_bundle=bundle_plan(
                    requirements, "split", search, single, 1.0, 1, beyond
                ),
                graph_execution=AtomisticGraphExecutionPlan(
                    max(
                        top(value).maximum_degree
                        for value in requirements
                        if value.role is Role.DIRECTED_IMAGE_GRAPH
                    ),
                    backend="particle",
                ),
            )
            return {"status": "admitted", "pair_cutoff": 1.02 * radius}

        probes["unique_image_radius"] = radius
        attempt("classical_cutoff_beyond_unique_image", unique_image)
    probes["compilations"] = executables.records
    return probes


# ------------------------------------------------------------------ CLI


def _tiles(value: str, /) -> tuple[int, int]:
    receiver, _, source = value.partition("x")
    widths = (int(receiver), int(source))
    if min(widths) < 1:
        raise argparse.ArgumentTypeError("Tile widths must be positive RxS integers.")
    return widths


def _search_spec(value: str, /) -> str:
    name, _, parameter = value.partition(":")
    if name in ("dense", "cell-list") and not parameter:
        return value
    if name == "tiled":
        _tiles(parameter)
        return value
    if name == "cluster" and parameter.isdigit() and int(parameter) > 0:
        return value
    raise argparse.ArgumentTypeError(
        "Searches are dense, cell-list, tiled:RxS or cluster:W."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small complete configuration; explicit sweep options still override.",
    )
    parser.add_argument("--systems", nargs="+", choices=_SYSTEMS, default=list(_SYSTEMS))
    parser.add_argument("--atoms", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--skews", nargs="+", type=float, default=[0.0, 0.5])
    parser.add_argument("--receiver-fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--searches",
        nargs="+",
        type=_search_spec,
        default=["cell-list", "dense", "tiled:8x16", "cluster:8"],
    )
    parser.add_argument("--sharings", nargs="+", choices=_SHARINGS, default=["split"])
    parser.add_argument("--capacity-factors", nargs="+", type=float, default=[0.25, 1.0])
    parser.add_argument(
        "--programs", nargs="+", choices=_PROGRAMS, default=list(_PROGRAMS)
    )
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=0.08)
    parser.add_argument("--inactive-fraction", type=float, default=0.25)
    parser.add_argument("--frame-margin", type=float, default=2.0)
    parser.add_argument("--pair-cutoff", type=float, default=3.0)
    parser.add_argument("--graph-cutoff", type=float, default=4.0)
    parser.add_argument("--coordination-cutoff", type=float, default=3.5)
    parser.add_argument("--dispersion-cutoff", type=float, default=5.0)
    parser.add_argument("--coordination-tiers", nargs="*", type=float, default=[2.5, 3.0])
    parser.add_argument("--selection-neighbors", type=int, default=6)
    parser.add_argument("--selection-tolerance", type=float, default=0.01)
    parser.add_argument("--skin", type=float, default=0.4)
    parser.add_argument("--reuse-fraction", type=float, default=0.8)
    parser.add_argument("--rebuild-fraction", type=float, default=1.5)
    parser.add_argument("--strain", type=float, default=0.01)
    parser.add_argument("--deformation-margin", type=float, default=0.5)
    parser.add_argument("--headroom", type=float, default=4.0)
    parser.add_argument("--maximum-images", type=int, default=4096)
    parser.add_argument("--maximum-candidate-slots", type=int, default=10**7)
    parser.add_argument("--dense-limit", type=int, default=2**22)
    parser.add_argument("--maximum-tile-candidates", type=int, default=2**24)
    parser.add_argument("--cluster-radius-factor", type=float, default=2.0)
    parser.add_argument("--cluster-tile-headroom", type=float, default=3.0)
    parser.add_argument("--graph-features", type=int, default=8)
    parser.add_argument("--d3-route-chunk", type=int, default=1024)
    parser.add_argument(
        "--stress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request strain derivative/stress on periodic programs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_arguments(argv: Sequence[str] | None = None, /) -> argparse.Namespace:
    parser = _parser()
    known, _ = parser.parse_known_args(argv)
    if known.smoke:
        parser.set_defaults(**_SMOKE)
    args = parser.parse_args(argv)
    if any(count < 2 for count in args.atoms) or sorted(set(args.atoms)) != args.atoms:
        raise ValueError("Atom counts must be at least 2 and strictly increasing.")
    if any(count < 1 for count in args.cases) or args.repeats < 1:
        raise ValueError("cases and repeats must be positive.")
    if any(not 0.0 <= value < 0.9 for value in args.skews):
        raise ValueError("Density skews must lie in [0, 0.9).")
    if any(not 0.0 < value <= 1.0 for value in args.receiver_fractions):
        raise ValueError("Receiver fractions must lie in (0, 1].")
    if any(value <= 0.0 for value in args.capacity_factors):
        raise ValueError("Capacity factors must be positive.")
    if not 0.0 <= args.inactive_fraction < 0.5:
        raise ValueError("inactive_fraction must lie in [0, 0.5).")
    if not 0.0 < args.reuse_fraction < 1.0 or args.rebuild_fraction <= 1.0:
        raise ValueError(
            "reuse_fraction must lie in (0, 1) of the half skin and rebuild_fraction "
            "must exceed the skin."
        )
    if args.density <= 0.0 or args.skin <= 0.0 or args.headroom < 1.0:
        raise ValueError("density and skin must be positive and headroom at least 1.")
    return args


def run(args: argparse.Namespace, /) -> dict[str, Any]:
    """Every row, comparison and refusal of one configuration (no identity).

    Callable directly for explicitly unattested probes; ``main`` adds the
    standard benchmark identity and environment.
    """
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    root = Path(__file__).resolve().parents[1]
    variants: list[Variant] = []
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    systems: list[dict[str, Any]] = []
    refusal_records: list[dict[str, Any]] = []
    for kind, atoms, cases, skew, fraction in itertools.product(
        args.systems, args.atoms, args.cases, args.skews, args.receiver_fractions
    ):
        work = workload(kind, atoms, cases, skew, fraction, args)
        system, preparation = _prepared(lambda work=work: prepare_system(work))
        systems.append(
            {
                "workload": [kind, atoms, cases, skew, fraction],
                "preparation": preparation,
                "prepared_id": system.prepared_id,
                "spacing": work.spacing,
                "cell_vectors": work.vectors.tolist(),
                "unique_image_radius": None
                if system.cell is None
                else system.cell.unique_image_radius,
                "active_atoms": work.active.sum(axis=1).tolist(),
                "receiver_rows": None
                if work.receivers is None
                else int(work.receivers.shape[1]),
            }
        )
        if (atoms, cases, skew, fraction) == (
            args.atoms[0],
            args.cases[0],
            args.skews[0],
            args.receiver_fractions[0],
        ):
            refusal_records.append(
                {"system_kind": kind, "atoms": atoms, **refusals(work, system, args)}
            )
        for search, sharing, factor in itertools.product(
            args.searches, args.sharings, args.capacity_factors
        ):
            variant = Variant(kind, atoms, cases, skew, fraction, search, sharing, factor)
            row, observation = measure_variant(variant, work, system, args)
            variants.append(variant)
            rows.append(row)
            observations.append(observation)
    configuration = {
        key: value for key, value in vars(args).items() if key not in ("label", "output")
    }
    return {
        "benchmark": "atomistic-neighborhood-bundle-lifecycle",
        "label": args.label,
        "configuration": configuration,
        "scope": {
            "execution": "public native bundle and potential-program paths only; "
            "no source framework, fixture import or mock",
            "parameters": "random native PaiNN parameters and invented synthetic D3 "
            "tables; no model or element fidelity",
            "precision": "float64 (x64 enabled)",
            "timing": "every timed call synchronizes all outputs; grow_capacities "
            "and rebase are host lifecycle transactions timed separately",
            "memory": "compiler bytes are XLA estimates; logical bytes are array "
            "payloads; host peaks include tracing overhead",
            "comparison": "exact masked route multisets per requirement identity; "
            "selected graphs compare only with the same selection; energy/force "
            "differences are reported, not certified",
            "claims": "no ranking, crossover or speedup claim; only the recorded "
            "device and configuration",
        },
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in _OWNERS
        },
        "systems": systems,
        "rows": rows,
        "comparisons": comparisons(variants, observations),
        "refusals": refusal_records,
    }


_EVIDENCE_FIELDS = (
    "comparisons",
    "configuration",
    "environment",
    "refusals",
    "rows",
    "scope",
    "source_sha256",
    "systems",
)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(root, Path(__file__), _EVIDENCE_FIELDS)
    body = run(args)
    record = {
        **body,
        "environment": capture_environment().to_dict(),
        "identity": identity.to_dict(),
    }
    write_json_atomic(args.output, record)
    print(args.output)


__all__ = [
    "Variant",
    "Workload",
    "bundle_plan",
    "build_program",
    "comparisons",
    "lifecycle_requirements",
    "measure_variant",
    "parse_arguments",
    "refusals",
    "run",
    "workload",
]


if __name__ == "__main__":
    main()

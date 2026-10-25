#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated native dispersion: conventional D3(BJ), FourierD3 and AIMNet2 TS.

Three distinct scientific definitions are measured, never substituted for one
another and never compared with one another:

* ``d3-bj``: :class:`phydrax.atomistic.D3BJPotential`, conventional two-body
  D3(BJ) with Grimme's truncated logistic coordination number over the full
  directed ``coordination`` role and the explicitly counted ``image-pair-sum``
  role (hard or S5-switched dispersion cutoff; no ATM/C9 term);
* ``fourier-d3``: :class:`phydrax.atomistic.FourierD3Potential`, the FourierD3
  *modified* coordination number, signed low-rank factors of the reference
  species/slot C6 block, the ``k = 0`` mode and the coefficient-dependent self
  energy on a term-owned particle mesh (no real-space cutoff). Its CN differs
  from conventional D3, so its energies are not D3(BJ) energies;
* ``ts-aimnet2``: :class:`phydrax.atomistic.TSDispersionPotential` with
  :class:`phydrax.atomistic.TSDispersionReference`, the AIMNetcentral
  ``DispParam``/``D3TS`` source definition (bounded multipliers, TS combination
  rule, BJ-like damping, hard pair cutoff) over a native full directed
  ``directed-image-graph`` role.

Every row executes the actual native public term/program: D3 and FourierD3
through ``AtomisticPotentialProgram.prepare(system, neighborhood_bundle=...)``
and ``evaluate(..., compute_stress=True)``; TS through ``route_energies`` on the
routes of an ``AtomisticNeighborhoodBundlePlan`` state. The driver's own code
only builds deterministic inputs, applies the declared homogeneous-strain
chart to the TS scalar, and records evidence.

Tables and parameters. By default all reference data are deterministic,
caller-declared **synthetic diagnostic** values labelled by atomic numbers
(``D3TableProvenance(kind="synthetic-fixture")``, synthetic TS ``(C6, alpha)``
and ``r4r2`` tables, synthetic BJ damping). They are not Grimme DFT-D3 data,
not AIMNet2 parameters and qualify no element or functional. No GPL table is
bundled and nothing is downloaded. A caller who holds Grimme's ``dftd3.f`` and
``pars.f`` and the right to use them may pass them with their SHA-256 digests,
species, license id and admission text (``--grimme-*``); the native
``D3ReferenceTables.from_grimme_fortran`` verifies the digests and parses them,
and preparation refuses unless ``--admit-table-use`` is given. Real tables then
require an explicitly named ``--functional`` with its ``--damping`` values.

Each row records, separately: host preparation (tables, C6 decomposition, term
construction, program/bundle preparation; time and traced host allocation
peak), the neighborhood ``initialize`` kernel, and per workload lowering,
compilation, first and warm executions, XLA compiler argument/output/temporary/
generated-code bytes, the sampled host RSS peak above the warm-phase baseline
(an observation, never a bound) and logical retained array bytes. Workloads:

* ``efs``: energy, conservative forces and the tension-positive Cauchy stress
  ``sym(dE/dstrain)/V`` (always run; it is the base point of the others);
* ``parameter-gradient``: reverse derivative of the same scalar in the term's
  admitted dynamic parameters (D3: damping and table leaves; FourierD3: damping,
  signed eigenvalues/factors, ``r4r2``, reference CN and covalent radii; TS:
  bounded raw multipliers, damping, ``r4r2`` and the ``(C6, alpha)`` table);
* ``hvp``: forward-over-reverse coordinate Hessian-vector product;
* ``refresh``: deterministic atom move plus symmetric cell strain, pure bundle
  refresh (``bundle.refresh``) and re-evaluation with the compiled ``efs``
  executable: coordination numbers (and hence C6) and the FourierD3 reciprocal
  geometry are recomputed from the current coordinates and cell inside the
  scalar;
* ``derivative-check``: central differences of the same native scalar along one
  coordinate and one strain direction, each point re-certified through a bundle
  refresh, against the automatic directional derivative. This is native
  self-consistency, not an independent physical reference;
* ``c6-refresh`` (TS only): new raw multipliers on retained routes through the
  already compiled ``efs`` executable. Conventional D3/FourierD3 C6 values are
  functions of CN (refreshed in every evaluation) and of identity-bound tables;
  a changed table is a new preparation, so this workload is recorded as not
  applicable there. No dispersion definition here consumes charges.

Status is classified from the actual native evidence: ``measured``,
``capacity-failure`` (role/search storage, repairable by a larger declared
capacity), ``scientific-failure`` (bundle scientific status) or
``term-failure`` (term/geometry status such as coincident atoms). Unsuccessful
outputs are NaN by the native contract and are recorded as non-finite, never
replaced by zero. Refused preparations keep their exception. ``--refusal-probes``
run the native admission paths that must refuse (uncovered species, unadmitted
table use, capped signed rank, finite FourierD3 geometry, mesh retained budget,
TS species without ``r4r2``, role capacity overflow, coincident atoms).

Comparisons are made only between rows of one matched definition: D3 execution
variants (route chunk, capacity) of one term identity; FourierD3 mesh/rank-chunk
self-convergence against the most refined mesh of the sweep (same tables,
decomposition, damping, CN definition and cutoff; a discrepancy, not an error
bound); FourierD3 signed-rank truncation at one mesh against the smallest
achieved coefficient residual; TS capacity variants of one definition.

``main`` captures the canonical benchmark identity first and fails closed when
its source-build prerequisites are absent. ``probe`` runs the same measured
kernels as an explicitly unattested probe and prints the record (no file, no
identity); ``dispersion_cases``/``measure_case``/``refusal_probes`` are the
reusable entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    synchronize,
)
from phydrax._validation import finite_real_scalar, integer
from phydrax.atomistic import (
    AtomisticCellListImageSearch,
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticNeighborhoodSource,
    AtomisticPairCounting,
    AtomisticPotentialProgram,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticSiteDomain,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    bind_atomistic_bundle_geometry,
    D3BJDamping,
    D3BJEvidence,
    D3BJPotential,
    D3CoordinationPolicy,
    D3DispersionCutoff,
    D3InterpolationRule,
    D3ReferenceTables,
    D3TableProvenance,
    FourierC6Decomposition,
    FourierD3CoordinationDefinition,
    FourierD3Evidence,
    FourierD3Potential,
    NVALCHEMI_FOURIER_EPSILON,
    ReciprocalBudget,
    ReciprocalMeshModulus,
    ReciprocalNyquistPolicy,
    TSDispersionPotential,
    TSDispersionReference,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.execution import PhaseMemorySampler
from phydrax.typing import parse
from phydrax.units import BOHR, HARTREE


Method: TypeAlias = Literal["d3-bj", "fourier-d3", "ts-aimnet2"]
Workload: TypeAlias = Literal[
    "parameter-gradient", "hvp", "refresh", "derivative-check", "c6-refresh"
]
Truncation: TypeAlias = Literal["hard", "s5"]
Search: TypeAlias = Literal["cell-list", "dense"]
Probe: TypeAlias = Literal[
    "uncovered-species",
    "unadmitted-table-use",
    "capped-rank",
    "fourier-finite-geometry",
    "mesh-retained-budget",
    "ts-uncovered-species",
    "role-capacity-overflow",
    "coincident-atoms",
]
_METHODS: tuple[str, ...] = get_args(Method)
_WORKLOADS: tuple[str, ...] = get_args(Workload)
_PROBES: tuple[str, ...] = get_args(Probe)
_INTERPOLATIONS: tuple[str, ...] = get_args(D3InterpolationRule)

UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
PHYSICAL_ATOMS = AtomisticSiteDomain.PHYSICAL_ATOMS
#: Atomic numbers used as labels of the synthetic diagnostic tables (row order).
SYNTHETIC_SPECIES_LABELS = (1, 6, 7, 8, 9, 14, 15, 16, 17, 35)
SYNTHETIC_FUNCTIONAL = "synthetic diagnostic BJ parameters (no published functional)"
_SHEAR = np.asarray([[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [0.07, -0.11, 1.0]])
_SAMPLE_INTERVAL = 0.005
_TS_OWNER = "ts-aimnet2-dispersion"
_OWNERS = (
    "benchmarks/atomistic_dispersion.py",
    "phydrax/atomistic/_d3_parameters.py",
    "phydrax/atomistic/_dispersion_d3.py",
    "phydrax/atomistic/_dispersion_fourier_d3.py",
    "phydrax/atomistic/_dispersion_ts.py",
    "phydrax/atomistic/_reciprocal_mesh.py",
    "phydrax/atomistic/_neighborhood_bundle.py",
    "phydrax/atomistic/_potential_program.py",
)
_EVIDENCE_FIELDS = (
    "cases",
    "comparisons",
    "configuration",
    "environment",
    "refusal_probes",
    "scope",
    "source_sha256",
)


# ------------------------------------------------------------ configuration


@dataclass(frozen=True, slots=True)
class GrimmeTables:
    """Caller-held Grimme ``dftd3.f``/``pars.f`` admission (never bundled)."""

    dftd3: Path
    pars: Path
    dftd3_sha256: str
    pars_sha256: str
    species: tuple[int, ...]
    source: str
    license_id: str
    admission: str
    use_admitted: bool


@dataclass(frozen=True, slots=True)
class Damping:
    a1: float
    a2: float
    s8: float
    s6: float


@dataclass(frozen=True, slots=True)
class Settings:
    """Shared scientific and resource declarations of one benchmark run."""

    seed: int
    density: float
    skin: float
    deformation_margin: float
    search: Search
    search_capacity_factor: float
    maximum_candidate_slots: int
    maximum_dense_routes: int
    coordination_cutoff: float
    d3_cutoff: float
    d3_switch_on: float
    fourier_tolerance: float
    fourier_modulus: ReciprocalMeshModulus
    fourier_nyquist: ReciprocalNyquistPolicy
    fourier_epsilon: float
    ts_cutoff: float
    damping: Damping
    functional: str
    ts_damping: Damping
    move: float
    strain: float
    fd_step: float
    grimme: GrimmeTables | None

    def species_labels(self, count: int, /) -> tuple[int, ...]:
        labels = SYNTHETIC_SPECIES_LABELS if self.grimme is None else self.grimme.species
        if not 0 < count <= len(labels):
            raise ValueError(
                f"Species count {count} exceeds the {len(labels)} declared species."
            )
        return tuple(labels[:count])


@dataclass(frozen=True, slots=True)
class D3Axes:
    """Swept conventional D3(BJ) selectors of one row."""

    interpolation: D3InterpolationRule
    truncation: Truncation
    route_chunk: int


@dataclass(frozen=True, slots=True)
class FourierAxes:
    """Swept FourierD3 mesh, rank and budget selectors of one row.

    ``max_rank=None`` scans every populated rank, ``rank_chunk=None`` keeps the
    full decomposition rank resident and ``retained_limit=None`` is the native
    default ``ReciprocalBudget``.
    """

    grid: int
    spline_degree: int
    max_rank: int | None
    rank_chunk: int | None
    retained_limit: int | None


@dataclass(frozen=True, slots=True)
class DispersionCase:
    """One row: method, controlling scales, method axes and resource capacities.

    Exactly the owning method's axes are present (``d3`` for ``d3-bj``,
    ``fourier`` for ``fourier-d3``, neither for ``ts-aimnet2``); the others are
    ``None`` and not swept.
    """

    method: Method
    atoms: int
    species: int
    slots: int | None
    capacity_factor: float
    d3: D3Axes | None = None
    fourier: FourierAxes | None = None
    workloads: tuple[Workload, ...] = ()
    repeats: int = 3

    def __post_init__(self) -> None:
        if (self.d3 is not None) != (self.method == "d3-bj") or (
            self.fourier is not None
        ) != (self.method == "fourier-d3"):
            raise ValueError(f"A {self.method} row carries exactly its own method axes.")

    def scales(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in ("method", "workloads", "repeats")
        }


#: Fixed selectors of the refusal probes (they test admission, not the sweep).
_PROBE_D3 = D3Axes("normalized-gaussian", "hard", 256)
_PROBE_FOURIER = FourierAxes(8, 3, None, None, None)


@dataclass(slots=True)
class PreparedCase:
    case: DispersionCase
    settings: Settings
    system: Any
    positions: np.ndarray
    vectors: np.ndarray
    origin: np.ndarray
    numbers: np.ndarray
    bundle: Any
    program: Any | None
    ts: TSDispersionPotential | None
    reference: TSDispersionReference | None
    raw: np.ndarray | None
    slot: int | None
    preparation: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------ numeric helpers


def failure_record(error: BaseException, /) -> dict[str, Any]:
    return {
        "status": "refused" if isinstance(error, (ValueError, TypeError)) else "failed",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception_only(error),
    }


def _finite(value: Any, /) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.all(np.isfinite(array)):
        return array.tolist()
    return {"kind": "nonfinite", "finite_mask": np.isfinite(array).tolist()}


def _bools(value: Any, /) -> Any:
    return np.asarray(value, dtype=np.bool_).tolist()


def _ints(value: Any, /) -> Any:
    return np.asarray(value, dtype=np.int64).tolist()


def _key(value: Any, /) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _host_phase[T](operation: Callable[[], T], /) -> tuple[T, dict[str, Any]]:
    """Host preparation time and traced allocation peak (tracing overhead included)."""
    (value, peak), seconds = measure_host(
        lambda: host_allocation_peak(lambda: synchronize(operation()))
    )
    return value, {"seconds": seconds, "host_allocation_peak_bytes": peak}


def _volume(vectors: np.ndarray, /) -> float:
    return float(abs(np.linalg.det(vectors)))


def _expected_degree(radius: float, density: float, /) -> float:
    return 4.0 / 3.0 * math.pi * radius**3 * density


# ------------------------------------------------------------ deterministic inputs


def synthetic_reference_tables(
    species: Sequence[int], slots: int, seed: int, /
) -> dict[str, Any]:
    """Deterministic invented D3 reference blocks; values qualify no element.

    The ``(species x slot)`` block is a positive rank-one outer product times
    ``1 + noise`` with symmetric ``|noise| <= 0.15``: exactly symmetric under
    ``(a, p) <-> (b, q)``, positive on populated slot pairs, generally full
    rank with signed eigenvalues. Odd species rows leave their last slot
    unpopulated (zero coordination and C6) when ``slots > 1``.
    """
    labels = tuple(int(value) for value in species)
    count = len(labels)
    rng = np.random.default_rng([seed, count, slots, 3])
    valid = np.zeros((count, slots), dtype=np.bool_)
    for row in range(count):
        valid[row, : slots - (row % 2 if slots > 1 else 0)] = True
    coordination = np.where(
        valid,
        1.1 * np.arange(slots)[None, :] + rng.uniform(0.0, 0.3, (count, slots)),
        0.0,
    )
    strength = rng.uniform(1.5, 7.0, (count * slots,))
    noise = rng.uniform(-0.15, 0.15, (count * slots, count * slots))
    noise = 0.5 * (noise + noise.T)
    block = np.outer(strength, strength) * (1.0 + noise)
    flat = valid.reshape(-1)
    block = np.where(flat[:, None] & flat[None, :], block, 0.0)
    c6 = block.reshape(count, slots, count, slots).transpose(0, 2, 1, 3).copy()
    return {
        "species": labels,
        "covalent_radii": rng.uniform(0.8, 2.6, (count,)),
        "r4r2": rng.uniform(2.0, 5.0, (count,)),
        "c6_reference": c6,
        "reference_coordination": coordination,
        "reference_valid": valid,
    }


def _synthetic_provenance(seed: int, /, *, use: bool = True) -> D3TableProvenance:
    return D3TableProvenance(
        f"phydrax benchmarks/atomistic_dispersion.py synthetic diagnostic tables "
        f"(seed {seed})",
        kind="synthetic-fixture",
        license_id="synthetic-diagnostic-data",
        use_admitted=use,
        redistribution_admitted=True,
        admission="deterministic invented values labelled by atomic numbers; not "
        "Grimme DFT-D3 reference data; qualifies no element",
    )


def _tables(
    settings: Settings, species: int, slots: int | None, /, *, use: bool = True
) -> D3ReferenceTables:
    grimme = settings.grimme
    if grimme is None:
        if slots is None:
            raise ValueError("Synthetic tables need a reference slot count.")
        data = synthetic_reference_tables(
            settings.species_labels(species), slots, settings.seed
        )
        return D3ReferenceTables(
            data["species"],
            covalent_radii=data["covalent_radii"],
            r4r2=data["r4r2"],
            c6_reference=data["c6_reference"],
            reference_coordination=data["reference_coordination"],
            reference_valid=data["reference_valid"],
            length_unit=BOHR,
            energy_unit=HARTREE,
            covalent_radius_scale=4.0 / 3.0,
            provenance=_synthetic_provenance(settings.seed, use=use),
        )
    provenance = D3TableProvenance(
        grimme.source,
        kind="grimme-dftd3-reference",
        license_id=grimme.license_id,
        use_admitted=grimme.use_admitted,
        redistribution_admitted=False,
        admission=grimme.admission,
        digests=(("dftd3.f", grimme.dftd3_sha256), ("pars.f", grimme.pars_sha256)),
    )
    # The archive's units (bohr, hartree bohr^6) are declared by this caller.
    return D3ReferenceTables.from_grimme_fortran(
        grimme.dftd3,
        grimme.pars,
        species=grimme.species,
        length_unit=BOHR,
        energy_unit=HARTREE,
        provenance=provenance,
    )


def synthetic_ts_tables(
    species: Sequence[int], seed: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Invented TS ``(C6, alpha)`` rows and ``r4r2`` (bohr), indexed by atomic number.

    Row 0 of the reference is the source padding dummy ``(0, 1)``; rows of
    undeclared atomic numbers stay zero and are never implemented.
    """
    labels = tuple(int(value) for value in species)
    rng = np.random.default_rng([seed, len(labels), 5])
    size = max(labels) + 1
    reference = np.zeros((size, 2), dtype=np.float64)
    reference[0] = (0.0, 1.0)
    reference[np.asarray(labels), 0] = rng.uniform(5.0, 60.0, len(labels))
    reference[np.asarray(labels), 1] = rng.uniform(3.0, 30.0, len(labels))
    r4r2 = np.zeros((size,), dtype=np.float64)
    r4r2[np.asarray(labels)] = rng.uniform(2.0, 5.0, len(labels))
    return reference, r4r2


def _ts_raw(atoms: int, seed: int, variant: int, /) -> np.ndarray:
    """Caller-declared raw ``DispParam`` outputs standing in for a network output."""
    rng = np.random.default_rng([seed, atoms, 7, variant])
    return rng.uniform(-0.5, 0.5, (atoms, 2))


def periodic_geometry(
    atoms: int, labels: Sequence[int], density: float, seed: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered-lattice atoms in a sheared cell at ``density`` atoms per cubic angstrom.

    Every species appears at least once; the remaining species are drawn from
    the seed. Fractional coordinates lie strictly inside ``[0, 1)``.
    """
    rng = np.random.default_rng([seed, atoms, len(labels), 11])
    volume = atoms / density
    vectors = np.cbrt(volume / np.linalg.det(_SHEAR)) * _SHEAR
    side = int(math.ceil(atoms ** (1.0 / 3.0) - 1.0e-12))
    lattice = np.stack(
        np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    chosen = np.sort(rng.choice(lattice.shape[0], size=atoms, replace=False))
    fractional = (lattice[chosen] + 0.5 + rng.uniform(-0.1, 0.1, (atoms, 3))) / side
    positions = fractional @ vectors
    count = len(labels)
    first = min(count, atoms)
    rows = np.concatenate(
        [np.arange(first), rng.integers(0, count, atoms - first)]
    ).astype(np.int64)
    return positions, vectors, np.asarray(labels, dtype=np.int64)[rows]


def _system(numbers: np.ndarray, vectors: np.ndarray | None, /) -> Any:
    count = int(numbers.shape[0])
    return AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        numbers,
        np.ones((count,), dtype=np.float64),
        UNITS,
        cell=None if vectors is None else PeriodicCell(vectors),
    ).prepare()


def _wrapped(
    positions: np.ndarray, vectors: np.ndarray, origin: np.ndarray, /
) -> np.ndarray:
    fractional = (positions - origin) @ np.linalg.inv(vectors)
    fractional -= np.floor(fractional)
    return origin + fractional @ vectors


def _unit_symmetric(rng: np.random.Generator, /) -> np.ndarray:
    value = rng.uniform(-1.0, 1.0, (3, 3))
    value = 0.5 * (value + value.T)
    return value / np.linalg.norm(value)


def _strained(
    positions: np.ndarray, vectors: np.ndarray, origin: np.ndarray, strain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    deformation = np.eye(3) + strain
    return origin + (positions - origin) @ deformation.T, vectors @ deformation.T


# ------------------------------------------------------------ resources


def _search_policy(settings: Settings, /) -> Any:
    if settings.search == "dense":
        return AtomisticDenseImageSearch(
            maximum_dense_routes=settings.maximum_dense_routes,
            deformation_margin=settings.deformation_margin,
        )
    return AtomisticCellListImageSearch(
        maximum_candidate_slots=settings.maximum_candidate_slots,
        deformation_margin=settings.deformation_margin,
    )


def bundle_plan(
    requirements: Sequence[AtomisticNeighborhoodRequirement],
    vectors: np.ndarray,
    atoms: int,
    capacity_factor: float,
    settings: Settings,
    /,
) -> tuple[AtomisticNeighborhoodBundlePlan, dict[str, Any]]:
    """One image source serving every requirement; declared capacity ladders.

    Mean-density estimates ``N 4/3 pi r^3 rho`` set the storage: the search
    and its cell occupancy by ``search_capacity_factor``, each role's routes
    and degree by ``capacity_factor`` (half-counted roles store half the
    directed routes). Image counts follow the complete stencil bound
    ``prod(2 floor(1 + R (1 + margin) reach_i) + 1)``.
    """
    values = tuple(requirements)
    density = atoms / _volume(vectors)
    radius = max(value.bounded_cutoff() for value in values) + settings.skin
    reach = np.linalg.norm(np.linalg.inv(vectors), axis=0)
    extents = np.floor(1.0 + radius * (1.0 + settings.deformation_margin) * reach)
    images = int(np.prod(2 * extents + 1))
    heights = 1.0 / reach
    cells = int(np.prod([max(1, int(np.floor(height / radius))) for height in heights]))
    occupancy = min(
        atoms, int(math.ceil(settings.search_capacity_factor * atoms / cells)) + 8
    )
    degree = _expected_degree(radius, density)
    edges = int(math.ceil(settings.search_capacity_factor * atoms * degree)) + 16
    search = ParticleImageCapacity(
        maximum_particles_per_cell=occupancy,
        maximum_edges=edges,
        maximum_degree=int(math.ceil(settings.search_capacity_factor * degree)) + 16,
        maximum_images=images,
        maximum_case_edges=edges,
    )
    source = AtomisticNeighborhoodSource(
        "dispersion-images",
        PHYSICAL_ATOMS,
        values,
        _search_policy(settings),
        ParticleImageCapacityLadder((search,)),
        skin=settings.skin,
    )
    ladders = []
    roles = []
    for value in values:
        mean = _expected_degree(value.bounded_cutoff(), density)
        share = 0.5 if value.counting is AtomisticPairCounting.IMAGE_HALF else 1.0
        capacity = AtomisticRoleCapacity(
            maximum_routes=max(4, int(math.ceil(capacity_factor * share * atoms * mean))),
            maximum_degree=max(2, int(math.ceil(capacity_factor * 1.5 * mean))),
        )
        ladders.append(AtomisticRoleCapacityLadder(value, (capacity,)))
        roles.append(
            {
                "requirement": value.label,
                "role": value.role.value,
                "counting": value.counting.value,
                "cutoff": value.cutoff,
                "mean_density_degree_estimate": mean,
                "maximum_routes": capacity.maximum_routes,
                "maximum_degree": capacity.maximum_degree,
            }
        )
    plan = AtomisticNeighborhoodBundlePlan(values, (source,), tuple(ladders))
    return plan, {
        "plan_id": plan.plan_id,
        "search": settings.search,
        "search_radius": radius,
        "skin": settings.skin,
        "deformation_margin": settings.deformation_margin,
        "search_capacity": {
            "maximum_particles_per_cell": occupancy,
            "maximum_edges": edges,
            "maximum_degree": search.maximum_degree,
            "maximum_images": images,
            "capacity_id": search.capacity_id,
        },
        "roles": roles,
    }


def _role_evidence(bundle: Any, state: Any, /) -> dict[str, Any]:
    """Per-role capacity versus requirement and the split capacity/scientific status."""
    roles = []
    for slot, requirement in enumerate(bundle.plan.requirements):
        evidence = state.views[slot].evidence
        capacity = bundle.capacity(slot)
        roles.append(
            {
                "requirement": requirement.label,
                "maximum_routes": capacity.maximum_routes,
                "maximum_degree": capacity.maximum_degree,
                **_module_record(evidence, exclude=("selection",)),
                "capacity_failure": _bools(evidence.capacity_failure),
                "scientific_failure": _bools(evidence.scientific_failure),
                "successful": _bools(evidence.successful),
            }
        )
    return {
        "roles": roles,
        "successful": _bools(state.successful),
        "capacity_failure": _bools(state.capacity_failure),
        "scientific_failure": _bools(state.scientific_failure),
        "stale": _bools(state.stale),
        "graph_epochs": _ints(state.graph_epochs),
    }


def _status(successful: bool, state: Any, /) -> str:
    if successful:
        return "measured"
    if bool(np.any(np.asarray(state.capacity_failure))):
        return "capacity-failure"
    if bool(np.any(np.asarray(state.scientific_failure))):
        return "scientific-failure"
    return "term-failure"


# ------------------------------------------------------------ kernels


def _initialize(bundle: Any, system: Any, positions: Array, cell: Array) -> Any:
    geometry = bind_atomistic_bundle_geometry(
        bundle, system, positions, cell_vectors=cell
    )
    return bundle.initialize(geometry)


def _refresh(bundle: Any, system: Any, state: Any, positions: Array, cell: Array) -> Any:
    geometry = bind_atomistic_bundle_geometry(
        bundle, system, positions, cell_vectors=cell
    )
    return bundle.refresh(state, geometry)


def _program_efs(program: Any, positions: Array, state: Any, cell: Array) -> Any:
    return program.evaluate(positions, state, compute_stress=True, cell_vectors=cell)


def _program_refreshed_energy(
    program: Any, state: Any, positions: Array, cell: Array
) -> tuple[Array, Array, Array, Array]:
    refreshed = _refresh(
        program.neighborhood_bundle, program.system, state, positions, cell
    )
    energy, auxiliary = program.energy(positions, refreshed, cell_vectors=cell)
    return (
        energy,
        auxiliary.successful,
        jnp.all(refreshed.successful),
        jnp.any(refreshed.capacity_failure),
    )


def _program_hvp(
    program: Any, positions: Array, state: Any, cell: Array, direction: Array
) -> tuple[Array, Array]:
    def energy(value: Array) -> Array:
        return program.energy(value, state, cell_vectors=cell)[0]

    return jax.jvp(jax.grad(energy), (positions,), (direction,))


def _program_parameter_gradient(
    program: Any,
    where: Callable[[Any], Any],
    parameters: Any,
    positions: Array,
    state: Any,
    cell: Array,
) -> Any:
    def energy(values: Any) -> Array:
        bound = eqx.tree_at(where, program, values)
        return bound.energy(positions, state, cell_vectors=cell)[0]

    return eqx.filter_value_and_grad(energy)(parameters)


def _d3_parameters(program: Any, /) -> Any:
    term = program.terms[0]
    return (term.plan.damping, term.plan.tables)


def _fourier_parameters(program: Any, /) -> Any:
    plan = program.terms[0].plan
    return (
        plan.damping,
        plan.decomposition.eigenvalues,
        plan.decomposition.factors,
        plan.tables.r4r2,
        plan.tables.reference_coordination,
        plan.tables.covalent_radii,
    )


_PARAMETER_NAMES = {
    "d3-bj": ("damping", "tables"),
    "fourier-d3": (
        "damping",
        "decomposition.eigenvalues",
        "decomposition.factors",
        "tables.r4r2",
        "tables.reference_coordination",
        "tables.covalent_radii",
    ),
    "ts-aimnet2": ("raw_multipliers", "potential", "reference"),
}
_PARAMETERS: dict[str, Callable[[Any], Any]] = {
    "d3-bj": _d3_parameters,
    "fourier-d3": _fourier_parameters,
}


def _ts_scalar(
    ts: TSDispersionPotential,
    reference: TSDispersionReference,
    raw: Array,
    numbers: Array,
    slot: int,
    state: Any,
    positions: Array,
    cell: Array,
    origin: Array,
    strain: Array,
) -> tuple[Array, Array]:
    """TS energy (eV) of the certified directed routes under the strain chart."""
    view = state.view(slot)
    deformation = jnp.eye(3, dtype=positions.dtype) + strain
    x = origin + (positions - origin) @ deformation.T
    h = cell @ deformation.T
    displacement = view.displacement(x, h)
    counted = view.cutoff_mask(x, h)
    squared = jnp.sum(displacement * displacement, axis=-1)
    distance = jnp.sqrt(jnp.where(counted, squared, 1.0))
    coefficients = reference.coefficients(raw, numbers)
    routes = ts.route_energies(
        coefficients,
        numbers,
        view.relation.receiver_indices,
        view.relation.source_indices,
        distance,
        counted,
    )
    energy = jnp.sum(routes)
    certified = jnp.all(
        jax.lax.stop_gradient(positions) == state.geometry.positions[0]
    ) & jnp.all(jax.lax.stop_gradient(cell) == state.geometry.cell_vectors[0])
    successful = (
        jnp.all(view.evidence.successful)
        & certified
        & jnp.all(~counted | jax.lax.stop_gradient(squared > 0.0))
        & jnp.isfinite(energy)
    )
    return energy, successful


def _ts_efs(
    ts: Any,
    reference: Any,
    raw: Array,
    numbers: Array,
    slot: int,
    state: Any,
    positions: Array,
    cell: Array,
    origin: Array,
) -> dict[str, Array]:
    def closure(value: Array, strain: Array) -> tuple[Array, Array]:
        return _ts_scalar(
            ts, reference, raw, numbers, slot, state, value, cell, origin, strain
        )

    (energy, successful), (gradient, strain_derivative) = jax.value_and_grad(
        closure, argnums=(0, 1), has_aux=True
    )(positions, jnp.zeros((3, 3), dtype=positions.dtype))
    forces = -gradient
    # The program's tension-positive Cauchy formula: sym(dE/dstrain) / V.
    stress = (
        0.5 * (strain_derivative + strain_derivative.T) / jnp.abs(jnp.linalg.det(cell))
    )
    accepted = (
        successful
        & jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(forces))
        & jnp.all(jnp.isfinite(strain_derivative))
    )
    nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
    return {
        "energy": jnp.where(accepted, energy, nan),
        "forces": jnp.where(accepted, forces, nan),
        "strain_derivative": jnp.where(accepted, strain_derivative, nan),
        "stress": jnp.where(accepted, stress, nan),
        "successful": accepted,
    }


def _ts_refreshed_energy(
    bundle: Any,
    system: Any,
    ts: Any,
    reference: Any,
    raw: Array,
    numbers: Array,
    slot: int,
    state: Any,
    positions: Array,
    cell: Array,
    origin: Array,
) -> tuple[Array, Array, Array, Array]:
    refreshed = _refresh(bundle, system, state, positions, cell)
    energy, successful = _ts_scalar(
        ts,
        reference,
        raw,
        numbers,
        slot,
        refreshed,
        positions,
        cell,
        origin,
        jnp.zeros((3, 3), dtype=positions.dtype),
    )
    return (
        energy,
        successful,
        jnp.all(refreshed.successful),
        jnp.any(refreshed.capacity_failure),
    )


def _ts_hvp(
    ts: Any,
    reference: Any,
    raw: Array,
    numbers: Array,
    slot: int,
    state: Any,
    positions: Array,
    cell: Array,
    origin: Array,
    direction: Array,
) -> tuple[Array, Array]:
    zero = jnp.zeros((3, 3), dtype=positions.dtype)

    def energy(value: Array) -> Array:
        return _ts_scalar(
            ts, reference, raw, numbers, slot, state, value, cell, origin, zero
        )[0]

    return jax.jvp(jax.grad(energy), (positions,), (direction,))


def _ts_parameter_gradient(
    ts: Any,
    reference: Any,
    raw: Array,
    numbers: Array,
    slot: int,
    state: Any,
    positions: Array,
    cell: Array,
    origin: Array,
) -> Any:
    zero = jnp.zeros((3, 3), dtype=positions.dtype)

    def energy(values: Any) -> Array:
        multipliers, potential, table = values
        return _ts_scalar(
            potential,
            table,
            multipliers,
            numbers,
            slot,
            state,
            positions,
            cell,
            origin,
            zero,
        )[0]

    return eqx.filter_value_and_grad(energy)((raw, ts, reference))


_KERNELS: dict[str, Any] = {
    "initialize": eqx.filter_jit(_initialize),
    "refresh": eqx.filter_jit(_refresh),
    "program-efs": eqx.filter_jit(_program_efs),
    "program-refreshed-energy": eqx.filter_jit(_program_refreshed_energy),
    "program-hvp": eqx.filter_jit(_program_hvp),
    "program-parameter-gradient": eqx.filter_jit(_program_parameter_gradient),
    "ts-efs": eqx.filter_jit(_ts_efs),
    "ts-refreshed-energy": eqx.filter_jit(_ts_refreshed_energy),
    "ts-hvp": eqx.filter_jit(_ts_hvp),
    "ts-parameter-gradient": eqx.filter_jit(_ts_parameter_gradient),
}


def _measure(
    name: str, arguments: tuple[Any, ...], repeats: int, /
) -> tuple[Any, dict[str, Any], Any]:
    """Lower, compile, first and warm executions of one native kernel."""
    lowering = _KERNELS[name].lower
    executable, phases = measure_lower_and_compile(
        partial(lowering, *arguments), methodcaller("compile")
    )
    invocation = partial(executable, *arguments)
    output, first = measure_synchronized(invocation)
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        name, interval_seconds=_SAMPLE_INTERVAL, devices=devices
    ) as sampler:
        _, warm = measure_repeated(invocation, warmup=1, repeats=repeats)
    compiled = executable.compiled
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return (
        output,
        {
            "kernel": name,
            "lowering_seconds": phases.lowering_seconds,
            "compilation_seconds": phases.compilation_seconds,
            "first_seconds": first,
            "warm": warm.to_dict(unit="seconds"),
            "compiler": asdict(
                compiler_evidence(
                    cost,
                    compiled.memory_analysis(),
                    source="xla",
                    unavailable_reason="The selected backend did not expose "
                    "compiler estimates.",
                )
            ),
            "sampled_peak_increase_bytes": sampler.evidence.sampled_peak_increase_bytes,
            "memory": sampler.evidence.to_payload(),
            "output_retained_bytes": logical_array_bytes(output),
        },
        executable,
    )


def _reuse(
    executable: Any, arguments: tuple[Any, ...], repeats: int, /
) -> tuple[Any, dict[str, Any]]:
    """Execute an already compiled kernel at new arguments (no recompilation)."""
    invocation = partial(executable, *arguments)
    output, first = measure_synchronized(invocation)
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        "reused-kernel", interval_seconds=_SAMPLE_INTERVAL, devices=devices
    ) as sampler:
        _, warm = measure_repeated(invocation, warmup=0, repeats=repeats)
    return output, {
        "first_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "sampled_peak_increase_bytes": sampler.evidence.sampled_peak_increase_bytes,
        "memory": sampler.evidence.to_payload(),
        "output_retained_bytes": logical_array_bytes(output),
    }


# ------------------------------------------------------------ method adapters


def _arrays(positions: np.ndarray, vectors: np.ndarray, /) -> tuple[Array, Array]:
    return jnp.asarray(positions), jnp.asarray(vectors)


def _efs_call(
    prepared: PreparedCase, state: Any, positions: Array, cell: Array, /
) -> tuple[str, tuple[Any, ...]]:
    if prepared.program is not None:
        return "program-efs", (prepared.program, positions, state, cell)
    return "ts-efs", (
        *_ts_arguments(prepared, state),
        positions,
        cell,
        jnp.asarray(prepared.origin),
    )


def _ts_arguments(
    prepared: PreparedCase, state: Any, /, *, raw: np.ndarray | None = None
) -> tuple[Any, ...]:
    return (
        prepared.ts,
        prepared.reference,
        jnp.asarray(prepared.raw if raw is None else raw),
        jnp.asarray(prepared.numbers),
        prepared.slot,
        state,
    )


def _refreshed_call(
    prepared: PreparedCase, state: Any, positions: Array, cell: Array, /
) -> tuple[str, tuple[Any, ...]]:
    if prepared.program is not None:
        return "program-refreshed-energy", (prepared.program, state, positions, cell)
    return "ts-refreshed-energy", (
        prepared.bundle,
        prepared.system,
        *_ts_arguments(prepared, state),
        positions,
        cell,
        jnp.asarray(prepared.origin),
    )


def _efs_values(prepared: PreparedCase, output: Any, /) -> dict[str, Any]:
    if prepared.program is None:
        values = {key: np.asarray(value) for key, value in output.items()}
        values["evidence"] = ()
        values["neighborhood_successful"] = None
        values["successful"] = bool(values["successful"])
        return values
    return {
        "energy": np.asarray(output.energy),
        "forces": np.asarray(output.forces),
        "strain_derivative": np.asarray(output.strain_derivative),
        "stress": np.asarray(output.stress),
        "successful": bool(np.asarray(output.successful)),
        "neighborhood_successful": _bools(output.neighborhood_successful),
        "stress_convention": None
        if output.stress_convention is None
        else str(output.stress_convention.value),
        "evidence": tuple(value for group in output.term_evidence for value in group),
    }


def _coordination(evidence: Sequence[Any], /) -> np.ndarray | None:
    for value in evidence:
        if isinstance(value, (D3BJEvidence, FourierD3Evidence)):
            return np.asarray(value.coordination_numbers)
    return None


def _module_record(
    module: eqx.Module, /, *, exclude: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Host evidence fields of a native module, recursing into nested modules.

    Exact identity words (``*_identity`` fields) are omitted; callers record
    their host-decoded digests explicitly.
    """
    record: dict[str, Any] = {}
    for entry in fields(module):
        name = entry.name
        value = getattr(module, name)
        if name in exclude or name.endswith("_identity") or value is None:
            continue
        if isinstance(value, eqx.Module):
            record[name] = _module_record(value)
        elif isinstance(value, (bool, int, float, str)):
            record[name] = value
        elif isinstance(value, (jax.Array, np.ndarray)):
            kind = np.asarray(value).dtype.kind
            record[name] = (
                _bools(value)
                if kind == "b"
                else _ints(value)
                if kind in "iu"
                else _finite(value)
            )
    return record


def _evidence_record(evidence: Sequence[Any], /) -> list[dict[str, Any]]:
    """D3/FourierD3 term evidence: digests, CN summary and every status/component."""
    records = []
    for value in evidence:
        if not isinstance(value, (D3BJEvidence, FourierD3Evidence)):
            continue
        numbers = np.asarray(value.coordination_numbers)
        record = {
            "kind": type(value).__name__,
            "prepared_id": value.prepared_id,
            "coordination_numbers": {
                "minimum": _finite(np.min(numbers)),
                "maximum": _finite(np.max(numbers)),
                "mean": _finite(np.mean(numbers)),
            },
            **_module_record(value, exclude=("coordination_numbers",)),
        }
        if isinstance(value, FourierD3Evidence):
            record["decomposition_id"] = value.decomposition_id
        records.append(record)
    return records


def _outputs(values: dict[str, Any], /) -> dict[str, Any]:
    forces = values["forces"]
    return {
        "energy": _finite(values["energy"]),
        "max_abs_force": _finite(np.max(np.abs(forces))),
        "net_force": _finite(np.sum(forces, axis=0)),
        "strain_derivative": _finite(values["strain_derivative"]),
        "stress": _finite(values["stress"]),
        "successful": values["successful"],
        "neighborhood_successful": values.get("neighborhood_successful"),
        "stress_convention": values.get(
            "stress_convention", "cauchy-tension-positive (driver: sym(dE/dstrain)/V)"
        ),
        "units": {"energy": "eV", "force": "eV/angstrom", "stress": "eV/angstrom^3"},
    }


# ------------------------------------------------------------ preparation


def _damping(values: Damping, functional: str, /) -> D3BJDamping:
    return D3BJDamping(
        a1=values.a1, a2=values.a2, s8=values.s8, s6=values.s6, functional=functional
    )


def _table_record(tables: D3ReferenceTables, /) -> dict[str, Any]:
    provenance = tables.provenance
    return {
        "tables_id": tables.tables_id,
        "species": list(tables.species),
        "reference_slots": tables.reference_slots,
        "populated_slots": int(np.count_nonzero(np.asarray(tables.reference_valid))),
        "length_unit": tables.length_unit.unit_id,
        "energy_unit": tables.energy_unit.unit_id,
        "covalent_radius_scale": float(tables.covalent_radius_scale),
        "retained_bytes": logical_array_bytes(tables),
        "provenance": {
            "provenance_id": provenance.provenance_id,
            "digests": [list(item) for item in provenance.digests],
            **_module_record(provenance, exclude=("digest_identities",)),
        },
    }


def _decomposition_record(decomposition: FourierC6Decomposition, /) -> dict[str, Any]:
    """Signed-rank selection evidence; the stored factors themselves are not dumped."""
    return {
        "decomposition_id": decomposition.decomposition_id,
        "species": list(decomposition.species),
        **_module_record(
            decomposition, exclude=("eigenvalues", "factors", "atomic_numbers")
        ),
    }


def _d3_term(
    axes: D3Axes, settings: Settings, tables: D3ReferenceTables, /
) -> D3BJPotential:
    cutoff = (
        D3DispersionCutoff(settings.d3_cutoff)
        if axes.truncation == "hard"
        else D3DispersionCutoff(
            settings.d3_cutoff,
            switch_on=settings.d3_switch_on,
            switch_off=settings.d3_cutoff,
        )
    )
    return D3BJPotential(
        tables,
        _damping(settings.damping, settings.functional),
        D3CoordinationPolicy(axes.interpolation),
        cutoff,
        coordination_cutoff=settings.coordination_cutoff,
        route_chunk=axes.route_chunk,
    )


def _fourier_term(
    axes: FourierAxes,
    settings: Settings,
    tables: D3ReferenceTables,
    decomposition: FourierC6Decomposition,
    /,
) -> FourierD3Potential:
    return FourierD3Potential(
        tables,
        decomposition,
        _damping(settings.damping, settings.functional),
        (axes.grid, axes.grid, axes.grid),
        coordination_cutoff=settings.coordination_cutoff,
        definition=FourierD3CoordinationDefinition(
            epsilon=settings.fourier_epsilon, epsilon_length_unit=BOHR
        ),
        spline_degree=axes.spline_degree,
        modulus=settings.fourier_modulus,
        nyquist=settings.fourier_nyquist,
        rank_chunk=axes.rank_chunk,
        budget=None
        if axes.retained_limit is None
        else ReciprocalBudget(maximum_retained_bytes=axes.retained_limit),
    )


def _program_preparation(
    term: Any,
    case: DispersionCase,
    settings: Settings,
    system: Any,
    vectors: np.ndarray,
    record: dict[str, Any],
    /,
) -> Any:
    program = AtomisticPotentialProgram([term])
    plan, resources = bundle_plan(
        program.requirements.neighborhoods,
        vectors,
        case.atoms,
        case.capacity_factor,
        settings,
    )
    record["bundle"] = resources
    prepared, record["phases"]["program_prepare"] = _host_phase(
        lambda: program.prepare(system, neighborhood_bundle=plan)
    )
    record["program"] = {
        "prepared_id": prepared.prepared_id,
        "scientific_id": prepared.scientific_id,
        "bundle_prepared_id": prepared.neighborhood_bundle.prepared_id,
    }
    return prepared


@dataclass(frozen=True, slots=True)
class _Binding:
    """Method-specific native objects of one prepared row."""

    bundle: Any
    program: Any | None = None
    ts: TSDispersionPotential | None = None
    reference: TSDispersionReference | None = None
    raw: np.ndarray | None = None
    slot: int | None = None


def _prepare_ts(
    case: DispersionCase,
    settings: Settings,
    system: Any,
    vectors: np.ndarray,
    labels: tuple[int, ...],
    record: dict[str, Any],
    /,
) -> _Binding:
    table, r4r2 = synthetic_ts_tables(labels, settings.seed)
    damping = settings.ts_damping
    (reference, ts), record["phases"]["term"] = _host_phase(
        lambda: (
            TSDispersionReference(table, implemented_species=labels),
            TSDispersionPotential(
                a1=damping.a1,
                a2=damping.a2,
                s8=damping.s8,
                s6=damping.s6,
                pair_cutoff=settings.ts_cutoff,
                r4r2=r4r2,
            ),
        )
    )
    ts.require_coverage(labels)
    record["term"] = {
        "definition": "aimnetcentral-0.2.0-DispParam-D3TS (source-specific TS)",
        "reference_id": reference.reference_id,
        "potential_id": ts.potential_id,
        "pair_cutoff_angstrom": ts.pair_cutoff,
        "damping": asdict(damping),
        "data": "synthetic diagnostic (C6, alpha) and explicit r4r2 tables; raw "
        "multipliers are caller-declared stand-ins for a network output; no AIMNet2 "
        "model or parameters",
        "retained_bytes": logical_array_bytes((ts, reference)),
    }
    requirement = AtomisticNeighborhoodRequirement(
        _TS_OWNER, AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH, settings.ts_cutoff
    )
    plan, record["bundle"] = bundle_plan(
        (requirement,), vectors, case.atoms, case.capacity_factor, settings
    )
    bundle, record["phases"]["bundle_prepare"] = _host_phase(lambda: plan.prepare(system))
    record["bundle"]["prepared_id"] = bundle.prepared_id
    record["comparison_keys"] = {
        "ts-execution-variants": _key(
            [record["geometry"], ts.potential_id, reference.reference_id]
        )
    }
    return _Binding(
        bundle,
        ts=ts,
        reference=reference,
        raw=_ts_raw(case.atoms, settings.seed, 0),
        slot=bundle.slot(requirement),
    )


def _prepare_d3(
    case: DispersionCase,
    settings: Settings,
    system: Any,
    vectors: np.ndarray,
    tables: D3ReferenceTables,
    record: dict[str, Any],
    /,
) -> _Binding:
    axes = case.d3
    if axes is None:
        raise ValueError("A d3-bj row requires its D3 axes.")
    term, record["phases"]["term"] = _host_phase(lambda: _d3_term(axes, settings, tables))
    record["term"] = {
        "definition": "conventional-two-body-d3-bj (truncated logistic CN, no ATM/C9)",
        "term_id": term.term_id,
        "damping_id": term.damping.damping_id,
        "functional": term.damping.functional,
        "damping": asdict(settings.damping),
        "coordination_policy_id": term.coordination.policy_id,
        "coordination_cutoff_angstrom": settings.coordination_cutoff,
        "dispersion_cutoff_angstrom": settings.d3_cutoff,
        "switch_on_angstrom": settings.d3_switch_on if axes.truncation == "s5" else None,
    }
    program = _program_preparation(term, case, settings, system, vectors, record)
    record["term"]["method"] = program.terms[0].method
    record["comparison_keys"] = {
        "d3-execution-variants": _key([record["geometry"], term.term_id])
    }
    return _Binding(program.neighborhood_bundle, program=program)


def _prepare_fourier(
    case: DispersionCase,
    settings: Settings,
    system: Any,
    vectors: np.ndarray,
    tables: D3ReferenceTables,
    labels: tuple[int, ...],
    record: dict[str, Any],
    /,
) -> _Binding:
    axes = case.fourier
    if axes is None:
        raise ValueError("A fourier-d3 row requires its FourierD3 axes.")
    decomposition, record["phases"]["decomposition"] = _host_phase(
        lambda: FourierC6Decomposition(
            tables,
            labels,
            tolerance=settings.fourier_tolerance,
            max_rank=axes.max_rank,
        )
    )
    record["decomposition"] = _decomposition_record(decomposition)
    term, record["phases"]["term"] = _host_phase(
        lambda: _fourier_term(axes, settings, tables, decomposition)
    )
    definition = term.definition
    record["term"] = {
        "definition": "fourier-d3 (modified CN, signed C6 factors, k=0 mode and self "
        "energy included); distinct from conventional D3(BJ)",
        "term_id": term.term_id,
        "damping_id": term.damping.damping_id,
        "functional": term.damping.functional,
        "damping": asdict(settings.damping),
        "definition_id": definition.definition_id,
        "epsilon_bohr2": float(definition.epsilon),
        "coordination_cutoff_angstrom": settings.coordination_cutoff,
        "mesh_plan_id": term.mesh_plan.plan_id,
        "modulus": settings.fourier_modulus,
        "nyquist": settings.fourier_nyquist,
        "rank_chunk": term.rank_chunk,
        "maximum_retained_bytes": term.mesh_plan.budget.maximum_retained_bytes,
    }
    program = _program_preparation(term, case, settings, system, vectors, record)
    record["term"]["method"] = program.terms[0].method
    record["term"]["resources"] = _module_record(program.terms[0].resources)
    definition_key = [
        record["geometry"],
        tables.tables_id,
        term.damping.damping_id,
        definition.definition_id,
        settings.coordination_cutoff,
    ]
    mesh_key = [axes.grid, axes.spline_degree, settings.fourier_modulus]
    record["comparison_keys"] = {
        "fourier-mesh-refinement": _key(
            [*definition_key, decomposition.decomposition_id]
        ),
        "fourier-rank-truncation": _key(
            [*definition_key, *mesh_key, settings.fourier_nyquist]
        ),
    }
    record["refinement"] = {
        "grid": axes.grid,
        "spline_degree": axes.spline_degree,
        "coefficient_residual": float(decomposition.achieved_error),
    }
    return _Binding(program.neighborhood_bundle, program=program)


def prepare_case(
    case: DispersionCase, settings: Settings, record: dict[str, Any], /
) -> PreparedCase:
    """Construct and prepare one row, filling ``record`` as each phase completes."""
    labels = settings.species_labels(case.species)
    positions, vectors, numbers = periodic_geometry(
        case.atoms, labels, settings.density, settings.seed
    )
    record["phases"] = {}
    record["geometry"] = {
        "atoms": case.atoms,
        "atomic_numbers": numbers.tolist(),
        "cell_vectors_angstrom": vectors.tolist(),
        "number_density_per_angstrom3": settings.density,
        "seed": settings.seed,
    }
    system, record["phases"]["system"] = _host_phase(lambda: _system(numbers, vectors))
    if case.method == "ts-aimnet2":
        binding = _prepare_ts(case, settings, system, vectors, labels, record)
    else:
        tables, record["phases"]["tables"] = _host_phase(
            lambda: _tables(settings, case.species, case.slots)
        )
        record["tables"] = _table_record(tables)
        if case.method == "d3-bj":
            binding = _prepare_d3(case, settings, system, vectors, tables, record)
        else:
            binding = _prepare_fourier(
                case, settings, system, vectors, tables, labels, record
            )
        record["program_retained_bytes"] = logical_array_bytes(binding.program)
    return PreparedCase(
        case,
        settings,
        system,
        positions,
        vectors,
        np.asarray(system.cell.origin, dtype=np.float64),
        numbers,
        binding.bundle,
        binding.program,
        binding.ts,
        binding.reference,
        binding.raw,
        binding.slot,
        record,
    )


# ------------------------------------------------------------ workloads


@dataclass(slots=True)
class _Base:
    state: Any
    values: dict[str, Any]
    efs: Any


def _initialize_workload(prepared: PreparedCase, /) -> tuple[Any, dict[str, Any]]:
    positions, cell = _arrays(prepared.positions, prepared.vectors)
    state, record, _ = _measure(
        "initialize",
        (prepared.bundle, prepared.system, positions, cell),
        prepared.case.repeats,
    )
    record["neighborhoods"] = _role_evidence(prepared.bundle, state)
    record["state_retained_bytes"] = logical_array_bytes(state)
    return state, record


def _efs_workload(prepared: PreparedCase, state: Any, /) -> tuple[_Base, dict[str, Any]]:
    positions, cell = _arrays(prepared.positions, prepared.vectors)
    name, arguments = _efs_call(prepared, state, positions, cell)
    output, record, executable = _measure(name, arguments, prepared.case.repeats)
    values = _efs_values(prepared, output)
    record["status"] = _status(values["successful"], state)
    record["outputs"] = _outputs(values)
    record["term_evidence"] = _evidence_record(values["evidence"])
    record["neighborhoods"] = _role_evidence(prepared.bundle, state)
    return _Base(state, values, executable), record


def _gradient_summary(names: Sequence[str], gradient: Any, /) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, part in zip(names, gradient, strict=True):
        leaves = jax.tree_util.tree_flatten_with_path(part)[0]
        summary[name] = {
            f"{name}{jax.tree_util.keystr(path)}": (
                {
                    "shape": list(np.shape(leaf)),
                    "l2": _finite(np.linalg.norm(np.asarray(leaf, dtype=np.float64))),
                    "max_abs": _finite(np.max(np.abs(np.asarray(leaf))))
                    if np.size(leaf)
                    else 0.0,
                }
                if np.ndim(leaf)
                else _finite(leaf)
            )
            for path, leaf in leaves
            if leaf is not None and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
        }
    return summary


def _parameter_gradient_workload(
    prepared: PreparedCase, base: _Base, /
) -> dict[str, Any]:
    positions, cell = _arrays(prepared.positions, prepared.vectors)
    method = prepared.case.method
    if prepared.program is not None:
        where = _PARAMETERS[method]
        arguments = (
            prepared.program,
            where,
            where(prepared.program),
            positions,
            base.state,
            cell,
        )
        name = "program-parameter-gradient"
    else:
        arguments = (
            *_ts_arguments(prepared, base.state),
            positions,
            cell,
            jnp.asarray(prepared.origin),
        )
        name = "ts-parameter-gradient"
    (value, gradient), record, _ = _measure(name, arguments, prepared.case.repeats)
    record["status"] = "measured" if bool(np.isfinite(np.asarray(value))) else "nonfinite"
    record["energy"] = _finite(value)
    record["gradient"] = _gradient_summary(_PARAMETER_NAMES[method], gradient)
    record["meaning"] = (
        "reverse derivative of the same conservative scalar in the dynamic "
        "parameter leaves (fixed by default); integer identity words are not "
        "differentiated"
    )
    return record


def _direction(prepared: PreparedCase, variant: int, /) -> np.ndarray:
    rng = np.random.default_rng(
        [prepared.settings.seed, prepared.case.atoms, 13, variant]
    )
    value = rng.normal(size=prepared.positions.shape)
    return value / np.linalg.norm(value)


def _hvp_workload(prepared: PreparedCase, base: _Base, /) -> dict[str, Any]:
    positions, cell = _arrays(prepared.positions, prepared.vectors)
    direction = jnp.asarray(_direction(prepared, 0))
    if prepared.program is not None:
        name = "program-hvp"
        arguments = (prepared.program, positions, base.state, cell, direction)
    else:
        name = "ts-hvp"
        arguments = (
            *_ts_arguments(prepared, base.state),
            positions,
            cell,
            jnp.asarray(prepared.origin),
            direction,
        )
    (gradient, product), record, _ = _measure(name, arguments, prepared.case.repeats)
    product = np.asarray(product)
    finite = bool(np.all(np.isfinite(product)))
    record["status"] = "measured" if finite else "nonfinite"
    record["outputs"] = {
        "hvp_l2": _finite(np.linalg.norm(product)),
        "direction_curvature": _finite(np.sum(product * np.asarray(direction))),
        "gradient_efs_force_max_abs_difference": _finite(
            np.max(np.abs(-np.asarray(gradient) - base.values["forces"]))
        )
        if base.values["successful"]
        else None,
    }
    record["meaning"] = "forward-over-reverse coordinate Hessian action of the scalar"
    return record


def _refresh_workload(prepared: PreparedCase, base: _Base, /) -> dict[str, Any]:
    settings = prepared.settings
    rng = np.random.default_rng([settings.seed, prepared.case.atoms, 17])
    strain = settings.strain * _unit_symmetric(rng)
    positions, vectors = _strained(
        prepared.positions, prepared.vectors, prepared.origin, strain
    )
    positions = _wrapped(
        positions + rng.uniform(-settings.move, settings.move, positions.shape),
        vectors,
        prepared.origin,
    )
    x, h = _arrays(positions, vectors)
    refreshed, record, _ = _measure(
        "refresh",
        (prepared.bundle, prepared.system, base.state, x, h),
        prepared.case.repeats,
    )
    record["neighborhoods"] = _role_evidence(prepared.bundle, refreshed)
    name, arguments = _efs_call(prepared, refreshed, x, h)
    output, evaluation = _reuse(base.efs, arguments, prepared.case.repeats)
    values = _efs_values(prepared, output)
    before = _coordination(base.values["evidence"])
    after = _coordination(values["evidence"])
    record["evaluation"] = {
        "kernel": name,
        "reused_compiled_executable": True,
        **evaluation,
        "status": _status(values["successful"], refreshed),
        "outputs": _outputs(values),
        "term_evidence": _evidence_record(values["evidence"]),
        "energy_change": _finite(values["energy"] - base.values["energy"]),
        "coordination_max_abs_change": None
        if before is None or after is None
        else _finite(np.max(np.abs(after - before))),
    }
    record["perturbation"] = {
        "symmetric_strain": strain.tolist(),
        "maximum_atom_move_angstrom": settings.move,
        "refreshed_quantities": "coordination numbers and C6 interpolation, "
        "route geometry and (FourierD3) current-cell reciprocal geometry",
        "charge_refresh": "not applicable: no dispersion definition here consumes "
        "charges",
    }
    record["status"] = record["evaluation"]["status"]
    return record


def _derivative_check_workload(prepared: PreparedCase, base: _Base, /) -> dict[str, Any]:
    if not base.values["successful"]:
        return {"status": "not-run", "reason": "the base efs evaluation was unsuccessful"}
    settings = prepared.settings
    step = settings.fd_step
    direction = _direction(prepared, 1)
    shear = _unit_symmetric(
        np.random.default_rng([settings.seed, prepared.case.atoms, 19])
    )
    points = {
        "coordinate+": (
            _wrapped(
                prepared.positions + step * direction, prepared.vectors, prepared.origin
            ),
            prepared.vectors,
        ),
        "coordinate-": (
            _wrapped(
                prepared.positions - step * direction, prepared.vectors, prepared.origin
            ),
            prepared.vectors,
        ),
        "strain+": _strained(
            prepared.positions, prepared.vectors, prepared.origin, step * shear
        ),
        "strain-": _strained(
            prepared.positions, prepared.vectors, prepared.origin, -step * shear
        ),
    }
    executable = None
    record: dict[str, Any] = {"points": {}}
    energies = {}
    for label, (positions, vectors) in points.items():
        x, h = _arrays(positions, vectors)
        name, arguments = _refreshed_call(prepared, base.state, x, h)
        if executable is None:
            output, measured, executable = _measure(
                name, arguments, prepared.case.repeats
            )
            record["kernel"] = measured
        else:
            output, _ = _reuse(executable, arguments, 1)
        energy, successful, neighborhoods, capacity = (np.asarray(v) for v in output)
        energies[label] = float(energy)
        record["points"][label] = {
            "energy": _finite(energy),
            "successful": bool(successful),
            "neighborhood_successful": bool(neighborhoods),
            "capacity_failure": bool(capacity),
        }
    automatic = {
        "coordinate": float(-np.sum(base.values["forces"] * direction)),
        "strain": float(np.sum(base.values["strain_derivative"] * shear)),
    }
    checks = {}
    for kind in ("coordinate", "strain"):
        difference = (energies[f"{kind}+"] - energies[f"{kind}-"]) / (2.0 * step)
        scale = max(abs(automatic[kind]), 1.0e-300)
        checks[kind] = {
            "automatic": _finite(automatic[kind]),
            "central_difference": _finite(difference),
            "absolute_discrepancy": _finite(abs(difference - automatic[kind])),
            "relative_discrepancy": _finite(abs(difference - automatic[kind]) / scale),
        }
    record["checks"] = checks
    record["step"] = step
    record["meaning"] = (
        "central difference of the same native scalar (each point re-certified by "
        "a bundle refresh) against the automatic directional derivative; native "
        "self-consistency, not an independent physical reference; hard cutoff or "
        "threshold steps inside the stencil appear as discrepancies"
    )
    record["status"] = (
        "measured"
        if all(point["successful"] for point in record["points"].values())
        else "term-failure"
    )
    return record


def _c6_refresh_workload(prepared: PreparedCase, base: _Base, /) -> dict[str, Any]:
    if prepared.program is not None:
        return {
            "status": "not-applicable",
            "reason": "conventional D3/FourierD3 C6 values are functions of the "
            "coordination numbers, recomputed in every evaluation (see refresh), and "
            "of identity-bound reference tables; a changed table or decomposition is "
            "a new preparation, never an in-place refresh",
        }
    raw = _ts_raw(prepared.case.atoms, prepared.settings.seed, 1)
    positions, cell = _arrays(prepared.positions, prepared.vectors)
    arguments = (
        *_ts_arguments(prepared, base.state, raw=raw),
        positions,
        cell,
        jnp.asarray(prepared.origin),
    )
    output, record = _reuse(base.efs, arguments, prepared.case.repeats)
    values = _efs_values(prepared, output)
    record.update(
        {
            "reused_compiled_executable": True,
            "routes_retained": True,
            "status": _status(values["successful"], base.state),
            "outputs": _outputs(values),
            "energy_change": _finite(values["energy"] - base.values["energy"]),
            "meaning": "new caller-declared raw DispParam multipliers on the same "
            "certified routes; the coefficient response is connected through the "
            "same scalar",
        }
    )
    return record


_WORKLOAD_FUNCTIONS: dict[str, Callable[[PreparedCase, _Base], dict[str, Any]]] = {
    "parameter-gradient": _parameter_gradient_workload,
    "hvp": _hvp_workload,
    "refresh": _refresh_workload,
    "derivative-check": _derivative_check_workload,
    "c6-refresh": _c6_refresh_workload,
}


def measure_case(
    case: DispersionCase, settings: Settings, /
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Complete row and its numeric comparison payload; every failure is kept."""
    row: dict[str, Any] = {
        "method": case.method,
        "scales": case.scales(),
        "workloads": {},
    }
    preparation: dict[str, Any] = {}
    try:
        prepared = prepare_case(case, settings, preparation)
    except Exception as error:  # every refusal or failure is a recorded row
        row["preparation"] = {**preparation, "failure": failure_record(error)}
        row["process_peak_rss_bytes"] = process_peak_rss_bytes()
        return row, None
    row["preparation"] = preparation
    numeric = None
    try:
        state, row["initialize"] = _initialize_workload(prepared)
        base, row["workloads"]["efs"] = _efs_workload(prepared, state)
    except Exception as error:  # every refusal or failure is a recorded row
        row["workloads"]["efs"] = failure_record(error)
        row["process_peak_rss_bytes"] = process_peak_rss_bytes()
        return row, None
    if base.values["successful"]:
        numeric = {
            "energy": np.asarray(base.values["energy"], dtype=np.float64),
            "forces": np.asarray(base.values["forces"], dtype=np.float64),
            "stress": np.asarray(base.values["stress"], dtype=np.float64),
        }
    for name in case.workloads:
        try:
            record = _WORKLOAD_FUNCTIONS[name](prepared, base)
        except Exception as error:  # every refusal or failure is a recorded row
            record = failure_record(error)
        row["workloads"][name] = record
    row["process_peak_rss_bytes"] = process_peak_rss_bytes()
    return row, numeric


# ------------------------------------------------------------ comparisons


def _comparison_reference(kind: str, members: list[tuple[dict, dict]], /) -> int:
    if kind == "fourier-mesh-refinement":
        return max(
            range(len(members)),
            key=lambda index: (
                members[index][0]["preparation"]["refinement"]["grid"],
                members[index][0]["preparation"]["refinement"]["spline_degree"],
            ),
        )
    if kind == "fourier-rank-truncation":
        return min(
            range(len(members)),
            key=lambda index: members[index][0]["preparation"]["refinement"][
                "coefficient_residual"
            ],
        )
    return 0


_COMPARISON_MEANING = {
    "d3-execution-variants": "one D3(BJ) term identity under different route "
    "chunks/role capacities; must agree to reduction roundoff",
    "ts-execution-variants": "one TS definition under different role capacities",
    "fourier-mesh-refinement": "one FourierD3 definition (tables, decomposition, "
    "damping, modified CN, cutoff) on different meshes/spline degrees/rank chunks, "
    "against the most refined mesh of this sweep; a self-convergence discrepancy, "
    "not an error bound",
    "fourier-rank-truncation": "one FourierD3 mesh and CN definition with different "
    "signed-rank decompositions, against the smallest achieved coefficient "
    "residual; coefficient-approximation discrepancy only",
}


def comparisons(
    rows: Sequence[dict[str, Any]], numeric: Sequence[dict[str, np.ndarray] | None], /
) -> list[dict[str, Any]]:
    """Matched-definition comparisons only; never across methods."""
    groups: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for row, values in zip(rows, numeric, strict=True):
        if values is None:
            continue
        for kind, key in row["preparation"].get("comparison_keys", {}).items():
            groups.setdefault((kind, key), []).append((row, values))
    results = []
    for (kind, key), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        index = _comparison_reference(kind, members)
        reference = members[index][1]
        entries = []
        for row, values in members:
            energy = float(values["energy"])
            entries.append(
                {
                    "row": row["row"],
                    "scales": row["scales"],
                    "energy_relative_difference": _finite(
                        abs(energy - float(reference["energy"]))
                        / max(abs(float(reference["energy"])), 1.0e-300)
                    ),
                    "force_max_abs_difference": _finite(
                        np.max(np.abs(values["forces"] - reference["forces"]))
                    ),
                    "stress_max_abs_difference": _finite(
                        np.max(np.abs(values["stress"] - reference["stress"]))
                    ),
                }
            )
        results.append(
            {
                "kind": kind,
                "meaning": _COMPARISON_MEANING[kind],
                "group": hashlib.sha256(key.encode()).hexdigest(),
                "reference_row": members[index][0]["row"],
                "rows": entries,
            }
        )
    return results


# ------------------------------------------------------------ refusal probes


def _probe_settings(settings: Settings, /) -> Settings:
    """Probes always use the synthetic tables and label (they test native admission)."""
    return replace(settings, grimme=None, functional=SYNTHETIC_FUNCTIONAL)


def _probe_d3_program(
    settings: Settings,
    positions: np.ndarray,
    vectors: np.ndarray,
    numbers: np.ndarray,
    capacity_factor: float,
    /,
) -> tuple[Any, Any, Any]:
    case = DispersionCase(
        "d3-bj", positions.shape[0], 2, 3, capacity_factor, d3=_PROBE_D3
    )
    system = _system(numbers, vectors)
    term = _d3_term(_PROBE_D3, settings, _tables(settings, 2, 3))
    record: dict[str, Any] = {"phases": {}}
    program = _program_preparation(term, case, settings, system, vectors, record)
    x, h = jnp.asarray(positions), jnp.asarray(vectors)
    state = _KERNELS["initialize"](program.neighborhood_bundle, system, x, h)
    evaluation = _KERNELS["program-efs"](program, x, state, h)
    return program, state, evaluation


def _evaluated_probe(
    program: Any, state: Any, evaluation: Any, expected: str, /
) -> dict[str, Any]:
    successful = bool(np.asarray(evaluation.successful))
    status = _status(successful, state)
    evidence = tuple(value for group in evaluation.term_evidence for value in group)
    return {
        "status": "unexpected-success" if successful else status,
        "expected": expected,
        "energy": _finite(evaluation.energy),
        "neighborhoods": _role_evidence(program.neighborhood_bundle, state),
        "term_evidence": _evidence_record(evidence),
    }


def _probe(name: str, settings: Settings, /) -> dict[str, Any]:
    labels = settings.species_labels(2)
    match name:
        case "uncovered-species":
            positions, vectors, _ = periodic_geometry(8, labels, settings.density, 1)
            numbers = np.asarray([labels[0], labels[1], 7, 7, *[labels[0]] * 4])
            term = _d3_term(_PROBE_D3, settings, _tables(settings, 2, 3))
            term.prepare(_system(numbers, vectors))
        case "unadmitted-table-use":
            positions, vectors, numbers = periodic_geometry(
                8, labels, settings.density, 1
            )
            term = _d3_term(_PROBE_D3, settings, _tables(settings, 2, 3, use=False))
            term.prepare(_system(numbers, vectors))
        case "capped-rank":
            tables = _tables(settings, 2, 3)
            decomposition = FourierC6Decomposition(
                tables, labels, tolerance=1.0e-12, max_rank=1
            )
            evidence = _decomposition_record(decomposition)
            try:
                _fourier_term(_PROBE_FOURIER, settings, tables, decomposition)
            except Exception as error:
                return {**failure_record(error), "decomposition": evidence}
            return {"status": "unexpected-success", "decomposition": evidence}
        case "fourier-finite-geometry":
            positions, _, numbers = periodic_geometry(8, labels, settings.density, 1)
            tables = _tables(settings, 2, 3)
            decomposition = FourierC6Decomposition(tables, labels, tolerance=1.0e-12)
            term = _fourier_term(_PROBE_FOURIER, settings, tables, decomposition)
            term.prepare(_system(numbers, None))
        case "mesh-retained-budget":
            positions, vectors, numbers = periodic_geometry(
                8, labels, settings.density, 1
            )
            tables = _tables(settings, 2, 3)
            decomposition = FourierC6Decomposition(tables, labels, tolerance=1.0e-12)
            term = _fourier_term(
                replace(_PROBE_FOURIER, retained_limit=1), settings, tables, decomposition
            )
            term.prepare(_system(numbers, vectors))
        case "ts-uncovered-species":
            _, r4r2 = synthetic_ts_tables(labels, settings.seed)
            damping = settings.ts_damping
            TSDispersionPotential(
                a1=damping.a1,
                a2=damping.a2,
                s8=damping.s8,
                s6=damping.s6,
                pair_cutoff=settings.ts_cutoff,
                r4r2=r4r2,
            ).require_coverage((*labels, 7))
        case "role-capacity-overflow":
            positions, vectors, numbers = periodic_geometry(
                16, labels, settings.density, 1
            )
            return _evaluated_probe(
                *_probe_d3_program(settings, positions, vectors, numbers, 0.05),
                "capacity-failure with NaN outputs (no zero energy)",
            )
        case "coincident-atoms":
            positions, vectors, numbers = periodic_geometry(
                8, labels, settings.density, 1
            )
            positions = positions.copy()
            positions[1] = positions[0]
            return _evaluated_probe(
                *_probe_d3_program(settings, positions, vectors, numbers, 2.0),
                "scientific or term failure (coincident counted route)",
            )
        case _:
            raise ValueError(f"Unknown refusal probe {name!r}.")
    return {"status": "unexpected-success"}


def refusal_probes(settings: Settings, names: Sequence[str], /) -> list[dict[str, Any]]:
    """Run the native admission paths that must refuse; record what actually happened."""
    probe_settings = _probe_settings(settings)
    results = []
    for name in names:
        try:
            record = _probe(name, probe_settings)
        except Exception as error:  # the expected refusal is the recorded result
            record = failure_record(error)
        results.append({"probe": name, **record})
    return results


# ------------------------------------------------------------ cases and CLI


def _integers(values: Sequence[object], name: str, /) -> list[int]:
    return [integer(value, name) for value in values]


def _optional_positive(value: int, name: str, /) -> int | None:
    """CLI encoding: ``0`` declares the owner's unbounded/default selection."""
    if value < 0:
        raise SystemExit(f"{name} values must be nonnegative.")
    return None if value == 0 else value


def _d3_axes(args: argparse.Namespace, /) -> list[D3Axes]:
    interpolations = [
        parse(value, D3InterpolationRule, "--d3-interpolations")
        for value in args.d3_interpolations
    ]
    truncations = [
        parse(value, Truncation, "--d3-truncations") for value in args.d3_truncations
    ]
    chunks = _integers(args.d3_route_chunks, "--d3-route-chunks")
    return [
        D3Axes(interpolation, truncation, chunk)
        for interpolation in interpolations
        for truncation in truncations
        for chunk in chunks
    ]


def _fourier_axes(args: argparse.Namespace, /) -> list[FourierAxes]:
    grids = _integers(args.fourier_grids, "--fourier-grids")
    degrees = _integers(args.fourier_degrees, "--fourier-degrees")
    ranks = [
        _optional_positive(value, "--fourier-max-ranks")
        for value in _integers(args.fourier_max_ranks, "--fourier-max-ranks")
    ]
    chunks = [
        _optional_positive(value, "--fourier-rank-chunks")
        for value in _integers(args.fourier_rank_chunks, "--fourier-rank-chunks")
    ]
    limits = [
        _optional_positive(value, "--fourier-retained-limits")
        for value in _integers(args.fourier_retained_limits, "--fourier-retained-limits")
    ]
    return [
        FourierAxes(grid, degree, rank, chunk, limit)
        for grid in grids
        for degree in degrees
        for rank in ranks
        for chunk in chunks
        for limit in limits
    ]


def dispersion_cases(args: argparse.Namespace, /) -> list[DispersionCase]:
    """The full deterministic sweep of the parsed arguments."""
    methods = [parse(value, Method, "--methods") for value in args.methods]
    workloads = tuple(parse(value, Workload, "--workloads") for value in args.workloads)
    repeats = integer(args.repeats, "--repeats")
    slots: list[int | None] = (
        [None] if args.grimme_dftd3 is not None else [*_integers(args.slots, "--slots")]
    )
    factors = [
        finite_real_scalar(value, "--capacity-factors") for value in args.capacity_factors
    ]
    d3_axes = _d3_axes(args)
    fourier_axes = _fourier_axes(args)
    cases: list[DispersionCase] = []
    for method in methods:
        for atoms in _integers(args.atoms, "--atoms"):
            for species in _integers(args.species, "--species"):
                for factor in factors:
                    if method == "ts-aimnet2":
                        cases.append(
                            DispersionCase(
                                method,
                                atoms,
                                species,
                                None,
                                factor,
                                workloads=workloads,
                                repeats=repeats,
                            )
                        )
                        continue
                    for slot in slots:
                        if method == "d3-bj":
                            cases.extend(
                                DispersionCase(
                                    method,
                                    atoms,
                                    species,
                                    slot,
                                    factor,
                                    d3=axes,
                                    workloads=workloads,
                                    repeats=repeats,
                                )
                                for axes in d3_axes
                            )
                        else:
                            cases.extend(
                                DispersionCase(
                                    method,
                                    atoms,
                                    species,
                                    slot,
                                    factor,
                                    fourier=axes,
                                    workloads=workloads,
                                    repeats=repeats,
                                )
                                for axes in fourier_axes
                            )
    return cases


def _damping_argument(values: Sequence[float], /) -> Damping:
    a1, a2, s8, s6 = (float(value) for value in values)
    return Damping(a1=a1, a2=a2, s8=s8, s6=s6)


def settings_from_arguments(args: argparse.Namespace, /) -> Settings:
    grimme = None
    declared = (
        args.grimme_dftd3,
        args.grimme_pars,
        args.grimme_dftd3_sha256,
        args.grimme_pars_sha256,
        args.grimme_species,
        args.grimme_source,
        args.grimme_license_id,
        args.table_admission,
    )
    if any(value is not None for value in declared):
        if any(value is None for value in declared):
            raise SystemExit(
                "Real Grimme tables need --grimme-dftd3, --grimme-pars, both "
                "--grimme-*-sha256 digests, --grimme-species, --grimme-source, "
                "--grimme-license-id and --table-admission."
            )
        if args.functional == SYNTHETIC_FUNCTIONAL:
            raise SystemExit(
                "Real tables need an explicitly named --functional with its --damping."
            )
        grimme = GrimmeTables(
            dftd3=Path(args.grimme_dftd3),
            pars=Path(args.grimme_pars),
            dftd3_sha256=args.grimme_dftd3_sha256,
            pars_sha256=args.grimme_pars_sha256,
            species=tuple(sorted(set(args.grimme_species))),
            source=args.grimme_source,
            license_id=args.grimme_license_id,
            admission=args.table_admission,
            use_admitted=bool(args.admit_table_use),
        )
    if args.d3_switch_on >= args.d3_cutoff and "s5" in args.d3_truncations:
        raise SystemExit("--d3-switch-on must lie below --d3-cutoff.")
    if (
        any(value < 2 for value in args.atoms)
        or args.repeats < 1
        or min(args.capacity_factors) <= 0.0
    ):
        raise SystemExit("atoms >= 2, repeats >= 1 and capacity factors > 0 required.")
    return Settings(
        seed=args.seed,
        density=args.density,
        skin=args.skin,
        deformation_margin=args.deformation_margin,
        search=parse(args.search, Search, "--search"),
        search_capacity_factor=args.search_capacity_factor,
        maximum_candidate_slots=args.maximum_candidate_slots,
        maximum_dense_routes=args.maximum_dense_routes,
        coordination_cutoff=args.coordination_cutoff,
        d3_cutoff=args.d3_cutoff,
        d3_switch_on=args.d3_switch_on,
        fourier_tolerance=args.fourier_tolerance,
        fourier_modulus=parse(
            args.fourier_modulus, ReciprocalMeshModulus, "--fourier-modulus"
        ),
        fourier_nyquist=parse(
            args.fourier_nyquist, ReciprocalNyquistPolicy, "--fourier-nyquist"
        ),
        fourier_epsilon=args.fourier_epsilon,
        ts_cutoff=args.ts_cutoff,
        damping=_damping_argument(args.damping),
        functional=args.functional,
        ts_damping=_damping_argument(args.ts_damping),
        move=args.move,
        strain=args.strain,
        fd_step=args.fd_step,
        grimme=grimme,
    )


def _parser(*, attested: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=_METHODS, default=list(_METHODS))
    parser.add_argument("--atoms", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--species", nargs="+", type=int, default=[2])
    parser.add_argument("--slots", nargs="+", type=int, default=[3])
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity-factors", nargs="+", type=float, default=[2.0])
    parser.add_argument("--search", choices=get_args(Search), default="cell-list")
    parser.add_argument("--search-capacity-factor", type=float, default=3.0)
    parser.add_argument("--maximum-candidate-slots", type=int, default=1 << 28)
    parser.add_argument("--maximum-dense-routes", type=int, default=1 << 24)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--deformation-margin", type=float, default=0.05)
    parser.add_argument("--coordination-cutoff", type=float, default=5.0)
    parser.add_argument("--d3-cutoff", type=float, default=8.0)
    parser.add_argument("--d3-switch-on", type=float, default=6.5)
    parser.add_argument(
        "--d3-interpolations",
        nargs="+",
        choices=_INTERPOLATIONS,
        default=["normalized-gaussian"],
    )
    parser.add_argument(
        "--d3-truncations",
        nargs="+",
        choices=get_args(Truncation),
        default=["hard", "s5"],
    )
    parser.add_argument("--d3-route-chunks", nargs="+", type=int, default=[256])
    parser.add_argument("--fourier-grids", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--fourier-degrees", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--fourier-max-ranks",
        nargs="+",
        type=int,
        default=[0],
        help="0 scans every populated rank",
    )
    parser.add_argument(
        "--fourier-rank-chunks",
        nargs="+",
        type=int,
        default=[1, 0],
        help="0 keeps the full decomposition rank resident",
    )
    parser.add_argument(
        "--fourier-retained-limits",
        nargs="+",
        type=int,
        default=[0],
        help="ReciprocalBudget maximum_retained_bytes; 0 is the native default",
    )
    parser.add_argument("--fourier-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--fourier-modulus",
        choices=("continuous-sinc-window", "discrete-bspline-modulus"),
        default="continuous-sinc-window",
    )
    parser.add_argument(
        "--fourier-nyquist", choices=("symmetrized", "excluded"), default="symmetrized"
    )
    parser.add_argument(
        "--fourier-epsilon",
        type=float,
        default=NVALCHEMI_FOURIER_EPSILON,
        help="modified-CN regularizer in bohr^2",
    )
    parser.add_argument("--ts-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--damping",
        nargs=4,
        type=float,
        metavar=("A1", "A2_BOHR", "S8", "S6"),
        default=[0.4, 4.5, 0.8, 1.0],
    )
    parser.add_argument("--functional", default=SYNTHETIC_FUNCTIONAL)
    parser.add_argument(
        "--ts-damping",
        nargs=4,
        type=float,
        metavar=("A1", "A2_BOHR", "S8", "S6"),
        default=[0.4, 4.5, 0.8, 1.0],
    )
    parser.add_argument("--grimme-dftd3")
    parser.add_argument("--grimme-pars")
    parser.add_argument("--grimme-dftd3-sha256")
    parser.add_argument("--grimme-pars-sha256")
    parser.add_argument("--grimme-species", nargs="+", type=int)
    parser.add_argument("--grimme-source")
    parser.add_argument("--grimme-license-id")
    parser.add_argument("--table-admission")
    parser.add_argument("--admit-table-use", action="store_true")
    parser.add_argument(
        "--workloads", nargs="*", choices=_WORKLOADS, default=list(_WORKLOADS)
    )
    parser.add_argument(
        "--refusal-probes", nargs="*", choices=_PROBES, default=list(_PROBES)
    )
    parser.add_argument("--move", type=float, default=0.05)
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--repeats", type=int, default=3)
    if attested:
        parser.add_argument("--label", required=True)
        parser.add_argument("--output", type=Path, required=True)
    return parser


def _scope(settings: Settings, /) -> dict[str, Any]:
    return {
        "definitions": "conventional D3(BJ), FourierD3 (modified CN) and AIMNet2 "
        "source TS are distinct definitions; no cross-method comparison or "
        "equivalence claim is made; no ATM/C9 term",
        "tables": "synthetic diagnostic tables labelled by atomic numbers (not "
        "Grimme data; no element or functional fidelity)"
        if settings.grimme is None
        else "caller-held Grimme dftd3.f/pars.f, digest-verified by the native "
        "parser; rights as declared by the caller",
        "cutoffs": "reduced diagnostic radii as configured, not Grimme's default "
        "40/50 bohr radii",
        "precision": "float64 (x64 enabled)",
        "memory": "compiler bytes are XLA estimates; sampled peaks are host RSS "
        "observations above the warm-phase baseline; retained bytes are logical "
        "array payloads; FourierD3 residency is the native admitted static charge",
        "hardware": "only the devices recorded in environment",
    }


def run(args: argparse.Namespace, /) -> dict[str, Any]:
    """Measure every case and refusal probe; return the record without identity."""
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    settings = settings_from_arguments(args)
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    numeric: list[dict[str, np.ndarray] | None] = []
    for index, case in enumerate(dispersion_cases(args)):
        row, values = measure_case(case, settings)
        row["row"] = index
        rows.append(row)
        numeric.append(values)
    configuration = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in ("label", "output")
    }
    configuration["settings"] = json.loads(json.dumps(asdict(settings), default=str))
    return {
        "benchmark": "native-atomistic-dispersion",
        "environment": capture_environment().to_dict(),
        "scope": _scope(settings),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in _OWNERS
        },
        "configuration": configuration,
        "cases": rows,
        "comparisons": comparisons(rows, numeric),
        "refusal_probes": refusal_probes(settings, args.refusal_probes),
    }


def probe(argv: Sequence[str] | None = None, /) -> dict[str, Any]:
    """Explicitly unattested probe of the same kernels; prints JSON, writes no file."""
    record = run(_parser(attested=False).parse_args(argv))
    record = {
        "attestation": "unattested-probe: capture_benchmark_identity was not run; "
        "not a benchmark record",
        **record,
    }
    print(json.dumps(record, indent=1, default=str))
    return record


def main() -> None:
    args = _parser(attested=True).parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(root, Path(__file__), _EVIDENCE_FIELDS)
    record = {"label": args.label, "identity": identity.to_dict(), **run(args)}
    write_json_atomic(args.output, record)
    print(args.output)


__all__ = [
    "D3Axes",
    "DispersionCase",
    "FourierAxes",
    "PreparedCase",
    "Settings",
    "bundle_plan",
    "comparisons",
    "dispersion_cases",
    "measure_case",
    "periodic_geometry",
    "prepare_case",
    "probe",
    "refusal_probes",
    "run",
    "settings_from_arguments",
    "synthetic_reference_tables",
    "synthetic_ts_tables",
]


if __name__ == "__main__":
    sys.exit(main())

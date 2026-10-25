#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Multi-process family scenarios of the distributed atomistic domain runtime.

`tools.atomistic_distributed_smoke` launches the coupled ranks and dispatches
the scenarios of this module to `run_family_rank`; `family_checks` adds the
cross-rank checks of a completed launch. Every rank binds the complete root
`ExecutionGroup` (one owner per device of every process), supplies ONLY its
own atom rows (`process_local_ingress`) and executes, for one declared family
and spatial strategy (halo or replicated-geometry graph partition):

``family-parity``
    Every declared route of the family. A qualified route prepares its
    `AtomisticDomainPlan` (layered model role, canonical domain terms on their
    own roles, optional model epilogue ``plan.reciprocal``), evaluates
    distributed energy, forces, stress, model and per-term energies, and
    compares them with the INDEPENDENT single-device native reference: the same
    scientific model and terms (same weights, same term objects) evaluated by
    one `AtomisticPotentialProgram` over its own neighborhood bundle
    (`LearnedGraphPotentialTerm` + canonical terms) -- never the domain route
    evaluated globally and masked. A route the native runtime does not admit
    is attempted exactly as a qualified one; its typed admission refusal text
    is recorded and must contain the declared refusal, and an admitted
    "refused" route is an error (the declaration is stale).
``family-dynamics``
    A short MTK NPT trajectory of a learned family on a ``dynamic_cell``
    domain plan (`AtomisticDomainDynamicsPlan`, strain response from the
    distributed evaluation) against the single-device P5 `AtomisticDynamicsPlan`
    of the same program and integrator: unwrapped coordinates, momenta, cell,
    energy ledger and MTK controllers within a roundoff tolerance after every
    step. Each family declares its timestep, length and any owner migration
    (`DYNAMICS`). The static-cell MTK refusal of the same domain is recorded.

Families (random-initialized or random source-layout native models; no
pretrained-weight, upstream-parity or ensemble claim):

* ``aimnet2``: per-pass charge normalization system reductions (total charge
  +1) and the model's predicted ``CHARGES`` consumed by a distributed PME
  domain term under the declared ``uniform-background`` neutrality policy (the
  cell is charged; charge cotangents through the layered reverse sweep);
* ``uma``: composition/MoLE routing and channel-balance system reductions with
  the declared ``max_neighbors`` selection binding on every receiver;
* ``mace-les``: the latent-Ewald global payload distributed by
  ``plan.reciprocal``;
* ``lj-pme``: terms-only Lennard-Jones with distributed PME and explicit-mode
  Ewald Coulomb domain terms, and the LJ-PME dispersion domain term (real
  space on its interaction-site pair role, distributed mesh) with PME Coulomb;
* ``fourier-d3``: terms-only FourierD3 dispersion (owner-local modified
  coordination on its coordination role, distributed mesh) under both the
  slab and the pencil reciprocal schedule.

Forced host-CPU devices with gloo collectives (the launcher default) are a
functional oracle of the collective program only: never accelerator,
multi-host or pretrained-model evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.atomistic._classical import (
    LennardJonesPotential,
    PreparedLennardJonesPotential,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from phydrax.atomistic._dispersion_fourier_d3 import PreparedFourierD3Potential
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionPolicy,
    AtomisticDomainRolePlan,
)
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_state import AtomisticDomainStrategy
from phydrax.atomistic._domain_terms import (
    AtomisticDomainEwaldTerm,
    AtomisticDomainFourierD3Term,
    AtomisticDomainLennardJonesPMETerm,
    AtomisticDomainLennardJonesTerm,
)
from phydrax.atomistic._electrostatics import (
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from phydrax.atomistic._force_field import (
    LennardJonesPMEPotential,
    PreparedGeneralForceFieldTerm,
)
from phydrax.atomistic._graph import AtomisticGraphExecutionPlan
from phydrax.atomistic._graph_partition import AtomisticGraphPartitionPlan
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodRole,
)
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
    PreparedReciprocalDistribution,
)
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.execution import ExecutionRuntime
from phydrax.sparse._streamed import StreamedRelationPlan


if TYPE_CHECKING:
    from tools.atomistic_distributed_smoke import Strategy


FamilyScenario = Literal["family-parity", "family-dynamics"]
Family = Literal["aimnet2", "uma", "mace-les", "lj-pme", "fourier-d3"]
ReciprocalSchedule = Literal["slab", "pencil"]

FAMILY_SCENARIOS: tuple[FamilyScenario, ...] = get_args(FamilyScenario)
FAMILIES: tuple[Family, ...] = get_args(Family)
# Dynamic-cell trajectories qualify learned families with a strain response.
LEARNED_FAMILIES: tuple[Family, ...] = ("aimnet2", "uma", "mace-les")
SCENARIO_FAMILIES: dict[FamilyScenario, tuple[Family, ...]] = {
    "family-parity": FAMILIES,
    "family-dynamics": LEARNED_FAMILIES,
}


def _units() -> AtomisticUnitSystem:
    """eV-A-Da-fs units (built on use: importing this module stays array-free)."""
    return AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


MASSES = {1: 1.008, 6: 12.011, 8: 15.999}

# Numerical contract of the distributed route against the single-device
# reference (float64; owner-ordered deterministic reductions on both sides
# differ only in summation order and reciprocal/mesh partitioning).
ENERGY_RELATIVE_TOLERANCE = 1.0e-9
FORCE_ABSOLUTE_TOLERANCE = 1.0e-8
STRESS_ABSOLUTE_TOLERANCE = 1.0e-8
# Per-term energies: |difference| <= TERM_TOLERANCE * max(1, max |reference term|).
TERM_TOLERANCE = 1.0e-9
DYNAMICS_ABSOLUTE_TOLERANCE = 1.0e-8

# Domain resources (generous fixed capacities of these few-atom cells).
SKIN = 0.3
# Absolute image-stencil headroom (A) of every role of a dynamic-cell plan; the
# graph-partition search declares the relative envelope covering it.
DEFORMATION_MARGIN = 0.6
EDGE_CAPACITY = 2048
ALIAS_CAPACITY = 2048
MESSAGE_CAPACITY_BYTES = 1 << 24
REPLICATED_MEMORY_BYTES = 1 << 24
RECEIVER_EDGE_CAPACITY = 512
MAXIMUM_PARTICLES_PER_CELL = 64
CANDIDATE_SLOTS = 1 << 22

# Single-device reference resources.
REFERENCE_SKIN = 0.4

# MTK NPT thermodynamic state and controllers (P5 parameters of the
# single-device MTK tests); timestep and length are per-family (`DYNAMICS`).
TEMPERATURE = 300.0
PRESSURE = 1.0e-3
VELOCITY_SCALE = 0.01

UMA_MAX_NEIGHBORS = 8
AIMNET2_TOTAL_CHARGE = 1.0

EVIDENCE_SCOPE = (
    "random-initialized or random source-layout native models on a forced-CPU "
    "(or observed local) multi-process collective program: functional parity "
    "with the single-device native reference only; no accelerator, multi-host, "
    "pretrained-weight, upstream-parity or ensemble claim"
)


# ------------------------------------------------------------ declarations


@dataclass(frozen=True, slots=True)
class RouteDeclaration:
    """One declared route of a family.

    ``refusal`` is ``None`` for a qualified route; otherwise it is the text the
    native admission refusal must contain (the route is attempted, never
    skipped).
    """

    name: str
    refusal: str | None
    description: str


ROUTES: dict[Family, tuple[RouteDeclaration, ...]] = {
    "aimnet2": (
        RouteDeclaration(
            "model",
            None,
            "layered AIMNet2 with per-pass charge-normalization system reductions",
        ),
        RouteDeclaration(
            "predicted-charge-pme",
            None,
            "AIMNet2 predicted CHARGES consumed by a distributed PME domain term "
            "(uniform-background neutrality of the charged cell)",
        ),
        RouteDeclaration(
            "global-epilogue",
            "declares none",
            "plan.reciprocal epilogue on a model without a global payload",
        ),
        RouteDeclaration(
            "charge-field-without-producer",
            "needs that model",
            "predicted-charge PME domain term without its producing model",
        ),
    ),
    "uma": (
        RouteDeclaration(
            "model",
            None,
            "layered UMA with composition/MoLE and channel-balance reductions and "
            "the declared max_neighbors selection",
        ),
        RouteDeclaration(
            "global-epilogue",
            "declares none",
            "plan.reciprocal epilogue on a model without a global payload",
        ),
    ),
    "mace-les": (
        RouteDeclaration(
            "latent-ewald",
            None,
            "MACE-LES latent-Ewald payload distributed by plan.reciprocal",
        ),
        RouteDeclaration(
            "without-latent-epilogue",
            "plan.reciprocal must distribute",
            "MACE-LES without its distributed latent-Ewald epilogue",
        ),
    ),
    "lj-pme": (
        RouteDeclaration(
            "lj-pme",
            None,
            "terms-only Lennard-Jones plus distributed PME Coulomb",
        ),
        RouteDeclaration(
            "lj-ewald",
            None,
            "terms-only Lennard-Jones plus distributed explicit-mode Ewald Coulomb",
        ),
        RouteDeclaration(
            "lj-pme-dispersion",
            None,
            "terms-only LJ-PME dispersion domain term (interaction-site pair role, "
            "distributed slab mesh) plus distributed PME Coulomb",
        ),
    ),
    "fourier-d3": (
        RouteDeclaration(
            "fourier-d3-slab",
            None,
            "terms-only FourierD3 (owner-local modified coordination, distributed "
            "mesh on the slab schedule)",
        ),
        RouteDeclaration(
            "fourier-d3-pencil",
            None,
            "terms-only FourierD3 on the pencil reciprocal schedule",
        ),
    ),
}
DYNAMICS_ROUTE: dict[Family, str] = {
    "aimnet2": "model",
    "uma": "model",
    "mace-les": "latent-ewald",
}


@dataclass(frozen=True, slots=True)
class DynamicsDeclaration:
    """Declared MTK NPT trajectory of one learned family.

    ``timestep`` (fs) and ``steps`` resolve the family's own force and stress
    scale. ``fractional_shift`` rigidly translates the initial configuration by
    a fractional lattice offset (a periodic translation of the same physical
    state; the MTK cell map is affine about the cell origin and preserves it),
    so an atom starts next to an owner boundary of the ``(owners, 1, 1)``
    fractional partition. ``migrates`` declares that the trajectory must move
    at least one atom to another owner.
    """

    timestep: float
    steps: int
    fractional_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    migrates: bool = False


DYNAMICS: dict[Family, DynamicsDeclaration] = {
    # The random source-layout AIMNet2 starts at |F| up to 2103 eV/A and
    # |dE/d strain| 2287 eV: at 0.5 fs the single-device P5 step itself
    # diverges (2.6 A first step, extended energy 2.9e16 eV). At 0.01 fs over
    # 0.12 fs the P5 trajectory is resolved (refinement to 0.005 fs: equal-time
    # displacement, volume and energies agree; extended-energy drift 0.90 vs
    # 0.21 eV, second order). Atom 1 (H) has the largest initial fractional-a
    # acceleration (negative); the +0.121 shift starts it at a = 0.501, 0.001
    # above the owner boundary at 1/2. The dynamics route is model only.
    "aimnet2": DynamicsDeclaration(0.01, 12, (0.121, 0.0, 0.0), migrates=True),
    "uma": DynamicsDeclaration(0.5, 6),
    "mace-les": DynamicsDeclaration(0.5, 6),
}
STATIC_CELL_REFUSAL = RouteDeclaration(
    "static-cell-mtk",
    "dynamic_cell",
    "MTK cell dynamics on a domain plan without dynamic_cell",
)


def declared_routes(
    scenario: FamilyScenario, family: Family, /
) -> tuple[RouteDeclaration, ...]:
    """The routes one rank of ``scenario`` executes for ``family``, in order."""
    _require_family(scenario, family)
    if scenario == "family-parity":
        return ROUTES[family]
    (route,) = (item for item in ROUTES[family] if item.name == DYNAMICS_ROUTE[family])
    return (route, STATIC_CELL_REFUSAL)


def _require_family(scenario: str, family: str, /) -> None:
    if scenario not in SCENARIO_FAMILIES:
        raise ValueError(f"Unknown family scenario {scenario!r}.")
    if family not in SCENARIO_FAMILIES[scenario]:
        raise ValueError(
            f"Scenario {scenario!r} declares families {SCENARIO_FAMILIES[scenario]}; "
            f"not {family!r}."
        )


# ------------------------------------------------------------ family cases


@dataclass(frozen=True, slots=True)
class RouteBinding:
    """Executable content of one route.

    ``terms`` are canonical (unprepared) energy terms in domain-plan order; the
    reference program evaluates the SAME objects. ``epilogue`` builds the
    model's ``plan.reciprocal`` inside the admission attempt. ``schedule`` is
    the distributed-FFT layout of every reciprocal domain term of the route
    (``"pencil"`` uses the most square ``px x py`` factorization of the
    owners).
    """

    model: bool
    terms: tuple[Any, ...] = ()
    epilogue: Callable[[], AtomisticReciprocalDistributionPlan] | None = None
    schedule: ReciprocalSchedule = "slab"


@dataclass(frozen=True, slots=True)
class FamilyCase:
    """One periodic system, its native model and its declared routes."""

    family: Family
    positions: np.ndarray
    numbers: np.ndarray
    masses: np.ndarray
    species: np.ndarray  # native ingress species (model indices or atom types)
    model: Any | None
    system: Any  # PreparedAtomisticSystem
    local_capacity: int
    streaming: StreamedRelationPlan
    conditioning: Callable[[int], AtomisticConditioning] | None
    routes: Mapping[str, RouteBinding]
    selection_binding: bool = False
    system_reductions: bool = False


def _masses(numbers: np.ndarray, /) -> np.ndarray:
    return np.asarray([MASSES[int(number)] for number in numbers], dtype=np.float64)


def _system(
    numbers: np.ndarray,
    cell: np.ndarray,
    /,
    *,
    charges: np.ndarray | None = None,
    types: np.ndarray | None = None,
) -> Any:
    count = numbers.shape[0]
    return AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        numbers.astype(np.int64),
        _masses(numbers),
        _units(),
        atom_type_ids=None if types is None else types.astype(np.int64),
        charges=np.zeros(count) if charges is None else charges,
        cell=PeriodicCell(cell),
    ).prepare()


def _model_species(model: Any, numbers: np.ndarray, /) -> np.ndarray:
    mask = jnp.ones(numbers.shape, dtype=jnp.bool_)
    return np.asarray(model.species_indices(jnp.asarray(numbers), mask), dtype=np.int32)


def _foreign_epilogue(
    system: Any, /
) -> Callable[[], AtomisticReciprocalDistributionPlan]:
    """A reciprocal distribution offered as the epilogue of a payload-free model."""

    def build() -> AtomisticReciprocalDistributionPlan:
        prepared = ParticleMeshEwaldPotential(
            2.5, 1.2, (16, 16, 16), spline_degree=4
        ).prepare(system)
        return AtomisticReciprocalDistributionPlan(prepared, reuse_reach=SKIN)

    return build


def _aimnet2_case() -> FamilyCase:
    from phydrax.nn.atomistic._aimnet2 import _CHARGE_PORT
    from tests._support.atomistic_model_families import aimnet2_model

    # The single-device AIMNet2 reduction fixture: a charged four-atom cell.
    cell = np.asarray([[3.6, 0.0, 0.0], [0.3, 3.7, 0.0], [0.2, 0.1, 3.8]])
    fractional = np.asarray(
        [[0.12, 0.2, 0.31], [0.38, 0.27, 0.33], [0.55, 0.61, 0.18], [0.81, 0.74, 0.66]]
    )
    numbers = np.asarray([8, 1, 1, 6], dtype=np.int32)
    model = aimnet2_model(3)
    system = _system(numbers, cell)
    spec = model.conditioning_spec

    def conditioning(capacity: int, /) -> AtomisticConditioning:
        return AtomisticConditioning(
            spec,
            {"charge": np.asarray([AIMNET2_TOTAL_CHARGE], dtype=np.float64)},
            case_count=1,
            atom_capacity=capacity,
        )

    field = AtomisticFieldReference(
        "aimnet2",
        AtomisticObservable.CHARGES,
        _CHARGE_PORT,
        level=AtomisticConditionLevel.ATOM,
        unit=_units().charge_unit,
    )
    # The minimum-image real-space cutoff stays below half the cell width. The
    # cell carries the conditioned +1 charge (and the source-compatible
    # finite-epsilon normalization leaves a ~1e-4 residual even at nominal 0),
    # so periodic Coulomb declares the uniform neutralizing background;
    # require-neutral correctly refuses this cell.
    pme = ParticleMeshEwaldPotential(
        2.2,
        1.6,
        (16, 16, 16),
        spline_degree=4,
        neutrality="uniform-background",
        charge_field=field,
    )
    return FamilyCase(
        "aimnet2",
        fractional @ cell,
        numbers,
        _masses(numbers),
        _model_species(model, numbers),
        model,
        system,
        8,
        StreamedRelationPlan(accumulation="deterministic"),
        conditioning,
        {
            "model": RouteBinding(True),
            "predicted-charge-pme": RouteBinding(True, (pme,)),
            "global-epilogue": RouteBinding(True, epilogue=_foreign_epilogue(system)),
            "charge-field-without-producer": RouteBinding(False, (pme,)),
        },
        system_reductions=True,
    )


def _uma_case() -> FamilyCase:
    from tests._support.atomistic_model_families import (
        uma_model,
        uma_spin_index,
        uma_task,
    )

    # Shortest lattice translation above the 2.5 A cutoff; a jittered 2x2x2
    # lattice gives ~18 candidates per receiver, so max_neighbors=8 binds.
    cell = np.asarray([[3.3, 0.0, 0.0], [0.4, 3.4, 0.0], [0.3, -0.2, 3.5]])
    grid = np.stack(np.meshgrid(*(np.arange(2),) * 3, indexing="ij"), axis=-1)
    jitter = np.random.default_rng(41).uniform(-0.04, 0.04, (8, 3))
    fractional = (grid.reshape((-1, 3)) + 0.5) / 2.0 + jitter
    numbers = np.asarray([8, 1, 1, 6, 1, 1, 8, 1], dtype=np.int32)
    streaming = StreamedRelationPlan(
        receiver_tile=8, edge_tile=64, channel_capacity=512, accumulation="deterministic"
    )
    model = uma_model(3, streaming=streaming, max_neighbors=UMA_MAX_NEIGHBORS)
    task = uma_task(periodic=True)
    schema = model.schema
    system = _system(numbers, cell)

    def conditioning(capacity: int, /) -> AtomisticConditioning:
        return schema.conditioning(
            [task],
            atom_capacity=capacity,
            charges=[0],
            spins=[uma_spin_index(task, 1)],
        )

    return FamilyCase(
        "uma",
        fractional @ cell,
        numbers,
        _masses(numbers),
        _model_species(model, numbers),
        model,
        system,
        8,
        streaming,
        conditioning,
        {
            "model": RouteBinding(True),
            "global-epilogue": RouteBinding(True, epilogue=_foreign_epilogue(system)),
        },
        selection_binding=True,
        system_reductions=True,
    )


def _mace_les_case() -> FamilyCase:
    from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
    from tests._support.atomistic_domain import (
        CELL,
        CUTOFF,
        SPECIES,
        streaming,
        structure as domain_structure,
    )
    from tests._support.atomistic_model_families import maceles_model

    # The self-image triclinic domain fixture (shortest translation < cutoff).
    positions, numbers = domain_structure()
    count = positions.shape[0]
    mace = MACEPotential(
        _units().scale,
        MACEArchitecture(
            species=SPECIES,
            cutoff=CUTOFF,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=4,
            hidden_degree=1,
            edge_degree=2,
            interactions=("real-agnostic", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(8,),
            readout_width=4,
            average_neighbor_count=8.0,
            energy_scaling="scale-shift",
        ),
        atomic_energies=np.asarray([[-0.3, -1.1]]),
        energy_scale=np.asarray([1.3]),
        energy_shift=np.asarray([0.2]),
        key=jax.random.key(3),
    )
    model = maceles_model(mace, atom_capacity=count, cell=CELL)

    def epilogue() -> AtomisticReciprocalDistributionPlan:
        return AtomisticReciprocalDistributionPlan(model.latent_ewald, reuse_reach=SKIN)

    return FamilyCase(
        "mace-les",
        positions,
        numbers,
        _masses(numbers),
        _model_species(model, numbers),
        model,
        _system(numbers, CELL),
        8,
        streaming(mace, "deterministic"),
        None,
        {
            "latent-ewald": RouteBinding(True, epilogue=epilogue),
            "without-latent-epilogue": RouteBinding(True),
        },
    )


def _separated(seed: int, count: int, cell: np.ndarray, minimum: float) -> np.ndarray:
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


def _lj_pme_case() -> FamilyCase:
    # The canonical classical domain-term fixture (minimum-image cutoffs).
    cell = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
    count = 12
    positions = _separated(3, count, cell, 1.0)
    rng = np.random.default_rng(4)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    types = rng.integers(0, 2, count).astype(np.int32)
    numbers = np.where(types == 1, 8, 1).astype(np.int32)
    lennard_jones = LennardJonesPotential(
        np.asarray([0.2, 0.1]), np.asarray([1.0, 1.2]), 2.5
    )
    pme = ParticleMeshEwaldPotential(1.1, 2.8, (16, 16, 16), spline_degree=4)
    ewald = EwaldReferencePotential(1.1, 2.8, 8)
    dispersion = LennardJonesPMEPotential(
        np.asarray([[0.1, 0.2], [0.2, 0.5]]), 0.9, 2.8, (16, 16, 16)
    )
    return FamilyCase(
        "lj-pme",
        positions,
        numbers,
        _masses(numbers),
        types,
        None,
        _system(numbers, cell, charges=charges, types=types),
        16,
        StreamedRelationPlan(accumulation="deterministic"),
        None,
        {
            "lj-pme": RouteBinding(False, (lennard_jones, pme)),
            "lj-ewald": RouteBinding(False, (lennard_jones, ewald)),
            "lj-pme-dispersion": RouteBinding(False, (dispersion, pme)),
        },
    )


def _fourier_d3_case() -> FamilyCase:
    from tests.unit.atomistic.test_dispersion_fourier_d3 import (
        _system as d3_system,
        _term as d3_term,
        TRICLINIC as D3_CELL,
    )

    # The FourierD3 fixture (synthetic H/C tables, full signed decomposition)
    # with six separated atoms; the 4 A coordination cutoff exceeds half the
    # cell, so every receiver counts several images of its partners.
    numbers = np.asarray([6, 6, 1, 6, 1, 1], dtype=np.int32)
    count = numbers.shape[0]
    term = d3_term((16, 16, 16), degree=4)
    return FamilyCase(
        "fourier-d3",
        _separated(5, count, D3_CELL, 1.0),
        numbers,
        np.ones((count,), dtype=np.float64),
        np.zeros((count,), dtype=np.int32),  # terms-only: no model species
        None,
        d3_system(tuple(int(number) for number in numbers), vectors=D3_CELL),
        8,
        StreamedRelationPlan(accumulation="deterministic"),
        None,
        {
            "fourier-d3-slab": RouteBinding(False, (term,)),
            "fourier-d3-pencil": RouteBinding(False, (term,), schedule="pencil"),
        },
    )


_BUILDERS: dict[Family, Callable[[], FamilyCase]] = {
    "aimnet2": _aimnet2_case,
    "uma": _uma_case,
    "mace-les": _mace_les_case,
    "lj-pme": _lj_pme_case,
    "fourier-d3": _fourier_d3_case,
}


def family_case(family: Family, /) -> FamilyCase:
    """Build one family case; its routes are exactly the declared ones."""
    case = _BUILDERS[family]()
    declared = [route.name for route in ROUTES[family]]
    if list(case.routes) != declared:
        raise RuntimeError(
            f"Family {family!r} binds routes {list(case.routes)}, not {declared}."
        )
    return case


# ----------------------------------------------------------- domain routes


def pencil_grid(owners: int, /) -> tuple[int, int]:
    """The most square ``(px, py)`` owner grid with ``px * py == owners``."""
    px = max(
        divisor for divisor in range(1, math.isqrt(owners) + 1) if owners % divisor == 0
    )
    return px, owners // px


def _domain_term(
    term: Any, system: Any, schedule: ReciprocalSchedule, owners: int, /
) -> Any:
    """The canonical domain term of ``term`` with its reciprocal distribution."""
    prepared = term.prepare(system)
    if isinstance(prepared, PreparedLennardJonesPotential):
        return AtomisticDomainLennardJonesTerm(prepared)
    layout: dict[str, Any] = (
        {"schedule": "pencil", "pencil_grid": pencil_grid(owners)}
        if schedule == "pencil"
        else {}
    )
    reciprocal = AtomisticReciprocalDistributionPlan(prepared, reuse_reach=SKIN, **layout)
    if isinstance(prepared, PreparedGeneralForceFieldTerm):
        return AtomisticDomainLennardJonesPMETerm(prepared, reciprocal)
    if isinstance(prepared, PreparedFourierD3Potential):
        return AtomisticDomainFourierD3Term(prepared, reciprocal)
    return AtomisticDomainEwaldTerm(prepared, reciprocal)


def _ownership(runtime: ExecutionRuntime, local_capacity: int, /) -> Any:
    from phydrax.discretization.spatial._distributed_relations import (
        DistributedOwnershipPlan,
    )
    from tools.atomistic_distributed_smoke import owner_address

    return DistributedOwnershipPlan(owner_address(), runtime.root_group, local_capacity)


def domain_plan(
    case: FamilyCase,
    route: RouteBinding,
    ownership: Any,
    strategy: Strategy,
    /,
    *,
    dynamic_cell: bool = False,
) -> AtomisticDomainPlan:
    """The route's domain plan: model role, term roles, epilogue and strategy."""
    cell = case.system.cell
    if cell is None:
        raise ValueError("A domain family case needs a periodic system cell.")
    terms = tuple(
        _domain_term(term, case.system, route.schedule, ownership.owner_count)
        for term in route.terms
    )
    declared: list[tuple[Any, str, StreamedRelationPlan]] = []
    if route.model:
        if case.model is None:
            raise ValueError("A model route requires its declared model.")
        declared.append(
            (
                case.model.requirements.neighborhood(
                    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
                ),
                "model",
                case.streaming,
            )
        )
    term_streaming = StreamedRelationPlan(accumulation="deterministic")
    declared += [
        (requirement, term.term_key, term_streaming)
        for term in terms
        for requirement in term.requirements()
    ]
    margin = DEFORMATION_MARGIN if dynamic_cell else 0.0
    roles = tuple(
        AtomisticDomainRolePlan(
            requirement,
            cell.image_stencil(
                requirement.bounded_cutoff() + SKIN + margin, maximum_image_count=4096
            ),
            streaming=streaming,
            skin=SKIN,
            edge_capacity=EDGE_CAPACITY,
            alias_capacity=ALIAS_CAPACITY,
            halo_capacity=ownership.total_capacity,
            term_key=term_key,
        )
        for requirement, term_key, streaming in declared
    )
    # The graph-partition cell list sizes its neighbor-cell offsets from its own
    # reference envelope ``radius * (1 + deformation_margin)``, independently of
    # the role stencils: a dynamic cell compressed inside a role stencil's
    # ``search_radius + margin`` but outside a zero search envelope is refused
    # as a stencil overflow. The relative margin covers every role's stencil.
    search_margin = max((margin / role.search_radius for role in roles), default=0.0)
    partition = FractionalOwnerPartition(cell, (ownership.owner_count, 1, 1))
    kind = AtomisticDomainStrategy(strategy)
    graph = (
        AtomisticGraphPartitionPlan(
            ownership,
            partition,
            roles,
            replicated_memory_bytes=REPLICATED_MEMORY_BYTES,
            receiver_edge_capacity=RECEIVER_EDGE_CAPACITY,
            maximum_particles_per_cell=MAXIMUM_PARTICLES_PER_CELL,
            search=AtomisticCellListImageSearch(
                maximum_candidate_slots=CANDIDATE_SLOTS,
                deformation_margin=search_margin,
            ),
        )
        if kind is AtomisticDomainStrategy.GRAPH_PARTITION
        else None
    )
    return AtomisticDomainPlan(
        ownership,
        partition,
        roles,
        migration_capacity=ownership.local_capacity,
        message_capacity_bytes=MESSAGE_CAPACITY_BYTES,
        strategy=kind,
        reduction=AtomisticDomainReductionPolicy("deterministic"),
        graph_partition=graph,
        reciprocal=None if route.epilogue is None else route.epilogue(),
        dynamic_cell=dynamic_cell,
        terms=terms,
    )


def _refusal(error: ValueError | TypeError, stage: str, /) -> dict[str, str]:
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


def admit_route(
    case: FamilyCase,
    route: RouteBinding,
    ownership: Any,
    strategy: Strategy,
    /,
    *,
    dynamic_cell: bool = False,
) -> tuple[Any | None, dict[str, str] | None]:
    """Attempt the native admission of one route (terms, plan, runtime).

    Returns the prepared runtime, or the typed native refusal of term, plan or
    runtime admission; nothing else is caught.
    """
    try:
        plan = domain_plan(case, route, ownership, strategy, dynamic_cell=dynamic_cell)
        runtime = prepare_atomistic_domain(plan, case.model if route.model else None)
    except (ValueError, TypeError) as error:
        return None, _refusal(error, "prepare")
    return runtime, None


def _ingress(
    case: FamilyCase,
    runtime: Any,
    route: RouteBinding,
    momenta: np.ndarray,
    /,
    *,
    key: int,
    positions: np.ndarray | None = None,
) -> Any:
    """Process-local ingress of this process's rows (index mod process count).

    ``positions`` (global logical order) default to the case's positions.
    """
    from tools.atomistic_distributed_smoke import process_local_ingress

    options: dict[str, Any] = {}
    if route.model and case.conditioning is not None:
        options["conditioning"] = case.conditioning(case.local_capacity)
    return process_local_ingress(
        runtime,
        None,  # species are already native ingress indices
        case.positions if positions is None else positions,
        case.species,
        momenta,
        masses=case.masses,
        rng_key=jax.random.key(key),
        **options,
    )


# --------------------------------------------------- single-device reference


def reference_program(case: FamilyCase, route: RouteBinding, /) -> Any:
    """Independent single-device program of the route's model and terms."""
    from tests._support.atomistic_bundles import (
        prepare_program,
        role_capacity,
        search_capacity,
    )
    from tests._support.atomistic_dynamics_fixtures import SEARCH

    terms: list[Any] = []
    if route.model:
        if case.model is None:
            raise ValueError("A model reference route requires its declared model.")
        terms.append(
            LearnedGraphPotentialTerm(
                case.model,
                name=case.family,
                allow_periodic=True,
                conditioning=None
                if case.conditioning is None
                else case.conditioning(case.system.capacity),
            )
        )
    terms.extend(route.terms)
    return prepare_program(
        AtomisticPotentialProgram(terms),
        case.system,
        graph_execution=AtomisticGraphExecutionPlan(
            256,
            backend="particle",
            streamed=StreamedRelationPlan(accumulation="deterministic"),
        )
        if route.model
        else None,
        search=SEARCH,
        skin=REFERENCE_SKIN,
        search_capacities=(search_capacity(4096, particles_per_cell=32, images=1024),),
        role_capacities=(role_capacity(4096, 512),),
    )


# --------------------------------------------------------------- comparisons


def _scalar(value: Any, /) -> float:
    return float(np.asarray(value).item())


def _maximum_error(observed: np.ndarray, expected: np.ndarray, /) -> float:
    """Largest absolute difference; a single-case reference axis is squeezed."""
    if observed.size != expected.size:
        return float("inf")
    difference = observed - expected.reshape(observed.shape)
    return float(np.max(np.abs(difference))) if observed.size else 0.0


def _owned_rows(active: np.ndarray, /) -> list[int]:
    from jax.experimental import multihost_utils

    counts = multihost_utils.process_allgather(
        np.asarray(active.sum(), np.int64), tiled=False
    )
    return np.asarray(counts).astype(np.int64).tolist()


def route_parity(
    case: FamilyCase, route: RouteBinding, runtime: Any, /
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Distributed E/F/stress/term energies against the single-device program.

    Returns ``(replicated, local, errors)``: values every rank must hold
    bitwise identically, this process's row errors, and failures.
    """
    from tests._support.atomistic_bundles import bundle_state
    from tools.atomistic_distributed_smoke import addressable_rows

    count = case.positions.shape[0]
    state = _ingress(case, runtime, route, np.zeros_like(case.positions), key=7)
    evaluation = runtime.evaluate(state)
    prepared = reference_program(case, route)
    reference = prepared.evaluate(
        jnp.asarray(case.positions),
        bundle_state(prepared, case.positions),
        compute_stress=True,
    )
    errors: list[str] = []
    successful = bool(addressable_rows(evaluation.successful))
    if not successful:
        errors.append(
            f"domain evaluation failed (reasons {int(addressable_rows(evaluation.reasons))})"
        )
    if not bool(np.all(np.asarray(reference.successful))):
        errors.append("the single-device reference evaluation failed")
    energy = _scalar(addressable_rows(evaluation.energy))
    reference_energy = _scalar(reference.energy)
    energy_error = abs(energy - reference_energy)
    if not energy_error <= ENERGY_RELATIVE_TOLERANCE * max(1.0, abs(reference_energy)):
        errors.append(f"energy differs from the reference by {energy_error}")
    ids = addressable_rows(state.layout.stable_ids)
    active = addressable_rows(state.layout.active)
    forces = addressable_rows(evaluation.forces)[active]
    reference_forces = np.asarray(reference.forces).reshape((count, 3))
    force_error = _maximum_error(forces, reference_forces[ids[active]])
    if not force_error <= FORCE_ABSOLUTE_TOLERANCE:
        errors.append(f"forces differ from the reference by {force_error}")
    stress = addressable_rows(evaluation.stress)
    if reference.stress is None:
        errors.append("the periodic reference reported no stress")
        stress_error = float("inf")
    else:
        stress_error = _maximum_error(stress, np.asarray(reference.stress))
    if not stress_error <= STRESS_ABSOLUTE_TOLERANCE:
        errors.append(f"stress differs from the reference by {stress_error}")
    reference_terms = np.asarray(reference.term_energies).reshape((-1,))
    offset = 1 if route.model else 0
    term_energies = addressable_rows(evaluation.term_energies).reshape((-1,))
    term_error = _maximum_error(term_energies, reference_terms[offset:])
    if not term_error <= TERM_TOLERANCE * max(
        1.0, float(np.max(np.abs(reference_terms), initial=0.0))
    ):
        errors.append(f"domain term energies differ from the reference by {term_error}")
    model_energy = model_error = None
    if route.model:
        model_energy = _scalar(addressable_rows(evaluation.local_energy)) + _scalar(
            addressable_rows(evaluation.global_energy)
        )
        model_error = abs(model_energy - float(reference_terms[0]))
        if not model_error <= ENERGY_RELATIVE_TOLERANCE * max(
            1.0, abs(float(reference_terms[0]))
        ):
            errors.append(f"model energy differs from the reference by {model_error}")
    owned = _owned_rows(active)
    if sum(owned) != count:
        errors.append("owned atom rows do not partition the system")
    status = evaluation.status
    capped = int(addressable_rows(status.capped_receivers))
    contexts = len(jax.tree.leaves(evaluation.system_contexts))
    if case.selection_binding and route.model and capped < 1:
        errors.append("the declared nearest-neighbor selection capped no receiver")
    if case.system_reductions and route.model and contexts < 1:
        errors.append("the model's system reductions produced no system context")
    reciprocal = runtime.reciprocal
    if route.epilogue is not None and reciprocal is None:
        errors.append("the declared model epilogue was not prepared")
    replicated = {
        "successful": successful,
        "energy": energy,
        "local_energy": _scalar(addressable_rows(evaluation.local_energy)),
        "global_energy": _scalar(addressable_rows(evaluation.global_energy)),
        "term_energies": term_energies.tolist(),
        "stress": stress.tolist(),
        "strain_derivative": addressable_rows(evaluation.strain_derivative).tolist(),
        "owned_rows_by_process": owned,
        "capped_receivers": capped,
        "maximum_selected_degree": int(addressable_rows(status.maximum_selected_degree)),
        "term_capacity_successful": addressable_rows(
            status.term_capacity_successful
        ).tolist(),
        "system_context_leaves": contexts,
        "reciprocal_kind": None if reciprocal is None else reciprocal.kind.value,
        "term_reciprocal_schedule": route.schedule,
        "term_reciprocal_kinds": [
            term.work.kind.value
            for term in runtime.terms
            if isinstance(term.work, PreparedReciprocalDistribution)
        ],
        "strategy": runtime.plan.strategy.value,
        "owner_count": runtime.plan.ownership.owner_count,
        "owner_processes": list(runtime.plan.ownership.owner_processes),
    }
    local = {
        "reference_energy": reference_energy,
        "reference_term_energies": reference_terms.tolist(),
        "energy_error": energy_error,
        "force_error": force_error,
        "stress_error": stress_error,
        "term_error": term_error,
        "model_energy": model_energy,
        "model_error": model_error,
        "owned_rows": int(active.sum()),
    }
    return replicated, local, errors


# ---------------------------------------------------------------- dynamics


def _integrator(timestep: float, /) -> Any:
    from phydrax.atomistic._cell_dynamics import MTKBarostatPlan, MTKNPTPlan
    from phydrax.atomistic._thermal import NoseHooverChainParameters

    return MTKNPTPlan(
        timestep,
        MTKBarostatPlan(100.0),
        NoseHooverChainParameters(20.0),
        NoseHooverChainParameters(100.0),
    )


def _ensemble(case: FamilyCase, /) -> Any:
    from phydrax.atomistic._domain_dynamics import AtomisticDomainEnsemblePlan

    return AtomisticDomainEnsemblePlan(
        _units(),
        ensemble="npt",
        atom_count=case.positions.shape[0],
        temperature=TEMPERATURE,
        pressure=PRESSURE,
        momentum_restriction="none",
        volume_coordinate_convention="atomic",
    )


@eqx.filter_jit
def _domain_step(dynamics: Any, state: Any) -> Any:
    return dynamics.step_detailed(state)


@eqx.filter_jit
def _reference_step(dynamics: Any, state: Any, table: Any) -> Any:
    return dynamics.step_detailed(state, table)


_LEDGER_FIELDS = (
    "kinetic_energy",
    "potential_energy",
    "extended_energy",
    "thermostat_heat",
    "reservoir_heat",
    "barostat_work",
    "constraint_work",
)
_CONTROLLER_FIELDS = ("particle_chain", "cell_chain", "cell")


def _controller_leaves(controllers: Any, name: str, /) -> list[np.ndarray]:
    from tools.atomistic_distributed_smoke import addressable_rows

    return [
        addressable_rows(leaf) for leaf in jax.tree.leaves(getattr(controllers, name))
    ]


def _atom_owners(state: Any, count: int, local_capacity: int, /) -> np.ndarray:
    """Owner of every atom (by stable ID) of an accepted state (collective)."""
    from jax.experimental import multihost_utils

    ids = np.asarray(
        multihost_utils.process_allgather(state.layout.stable_ids, tiled=True)
    )
    active = np.asarray(
        multihost_utils.process_allgather(state.layout.active, tiled=True)
    ).astype(np.bool_)
    rows = np.flatnonzero(active)
    owners = np.full((count,), -1, dtype=np.int64)
    owners[ids[rows]] = rows // local_capacity
    return owners


def dynamics_parity(
    case: FamilyCase, route: RouteBinding, runtime: Any, /
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """MTK NPT on the dynamic-cell domain against single-device P5 dynamics."""
    from phydrax.atomistic._domain_dynamics import AtomisticDomainDynamicsPlan
    from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
    from phydrax.atomistic._thermodynamic import (
        AtomisticPhaseSpaceMeasurePlan,
        AtomisticThermodynamicStatePlan,
        PreparedThermodynamicStateTable,
    )
    from tools.atomistic_distributed_smoke import addressable_rows

    declaration = DYNAMICS[case.family]
    integrator = _integrator(declaration.timestep)
    count = case.positions.shape[0]
    # A rigid fractional translation: the same periodic physical state.
    shift = np.asarray(declaration.fractional_shift) @ np.asarray(
        case.system.cell.vectors, dtype=np.float64
    )
    positions = case.positions + shift
    velocities = VELOCITY_SCALE * np.random.default_rng(31).normal(size=positions.shape)
    momenta = case.masses[:, None] * velocities
    reference = AtomisticDynamicsPlan(
        case.system, reference_program(case, route), integrator
    ).prepare()
    table = PreparedThermodynamicStateTable(
        reference,
        (
            AtomisticThermodynamicStatePlan(
                AtomisticPhaseSpaceMeasurePlan(
                    case.system,
                    volume_coordinate_convention="atomic",
                    momentum_restriction="none",
                ),
                ensemble="npt",
                temperature=TEMPERATURE,
                pressure=PRESSURE,
            ),
        ),
    )
    single = reference.initialize_state(
        jnp.asarray(positions), table, momentum=momenta, key=jax.random.key(2)
    )
    dynamics = AtomisticDomainDynamicsPlan(
        runtime.plan, integrator, _ensemble(case)
    ).prepare(runtime)
    distributed = dynamics.initialize(
        _ingress(case, runtime, route, momenta, key=2, positions=positions)
    )
    epochs_before = addressable_rows(distributed.layout.owner_epochs).astype(np.int64)
    local_capacity = runtime.plan.ownership.local_capacity
    owners = _atom_owners(distributed, count, local_capacity)
    migrated = np.zeros((count,), dtype=np.bool_)
    owner_changes: list[int] = []
    errors: list[str] = []
    steps = []
    for index in range(declaration.steps):
        single_step = _reference_step(reference, single, table)
        step = _domain_step(dynamics, distributed)
        successful = bool(addressable_rows(step.successful))
        reference_successful = bool(np.all(np.asarray(single_step.successful)))
        steps.append(
            {
                "step": index + 1,
                "successful": successful,
                "rejection_reasons": int(addressable_rows(step.rejection_reasons)),
                "reference_successful": reference_successful,
            }
        )
        if not successful or not reference_successful:
            errors.append(
                f"step {index + 1}: domain successful={successful}, "
                f"reference successful={reference_successful}"
            )
            break
        single = single_step.accepted_state
        distributed = step.accepted_state
        current = _atom_owners(distributed, count, local_capacity)
        owner_changes.append(int(np.sum(current != owners)))
        migrated |= current != owners
        owners = current
    ids = addressable_rows(distributed.layout.stable_ids)
    active = addressable_rows(distributed.layout.active)
    cell = addressable_rows(distributed.cell_vectors)
    unwrapped = (
        addressable_rows(distributed.positions)
        + addressable_rows(distributed.image_counts).astype(np.float64) @ cell
    )[active]
    kinematics = single.kinematics
    reference_cell = np.asarray(single.cell_vectors).reshape((3, 3))
    reference_positions = np.asarray(kinematics.positions).reshape((count, 3))
    reference_images = np.asarray(kinematics.image_counts).reshape((count, 3))
    reference_unwrapped = (
        reference_positions + reference_images.astype(np.float64) @ reference_cell
    )
    rows = ids[active]
    position_error = _maximum_error(unwrapped, reference_unwrapped[rows])
    momentum_error = _maximum_error(
        addressable_rows(distributed.momenta)[active],
        np.asarray(kinematics.momenta).reshape((count, 3))[rows],
    )
    cell_error = _maximum_error(cell, reference_cell)
    if distributed.ledger is None or single.energy is None:
        raise ValueError("A dynamics route requires both accepted-state energy ledgers.")
    ledger = {
        name: _scalar(addressable_rows(getattr(distributed.ledger, name)))
        for name in _LEDGER_FIELDS
    }
    ledger_error = max(
        abs(ledger[name] - _scalar(getattr(single.energy, name)))
        for name in _LEDGER_FIELDS
    )
    controller_error = 0.0
    controllers = distributed.controllers
    if controllers is None:
        raise ValueError("A dynamics route requires its accepted controller state.")
    for name in _CONTROLLER_FIELDS:
        left = _controller_leaves(controllers, name)
        right = [
            np.asarray(leaf)
            for leaf in jax.tree.leaves(getattr(single.controllers, name))
        ]
        if len(left) != len(right):
            errors.append(f"controller {name} differs in structure from P5")
            continue
        for observed, expected in zip(left, right, strict=True):
            controller_error = max(controller_error, _maximum_error(observed, expected))
    for label, value in (
        ("unwrapped coordinates", position_error),
        ("momenta", momentum_error),
        ("cell vectors", cell_error),
        ("energy ledger", ledger_error),
        ("MTK controllers", controller_error),
    ):
        if not value <= DYNAMICS_ABSOLUTE_TOLERANCE:
            errors.append(f"{label} differ from single-device P5 by {value}")
    step_index = int(addressable_rows(distributed.step_index))
    if not errors and step_index != declaration.steps:
        errors.append(f"the trajectory ended at step {step_index}")
    if declaration.migrates and not np.any(migrated):
        errors.append("the declared owner migration did not occur")
    if step_index != int(np.asarray(single.step_index)):
        errors.append("the domain and P5 step indices differ")
    if int(addressable_rows(distributed.ledger.accepted_steps)) != int(
        np.asarray(single.energy.accepted_steps)
    ):
        errors.append("the domain and P5 accepted step counts differ")
    owned = _owned_rows(active)
    if sum(owned) != case.positions.shape[0]:
        errors.append("owned atom rows do not partition the system")
    epochs_after = addressable_rows(distributed.layout.owner_epochs).astype(np.int64)
    replicated = {
        "integrator": "mtk-npt",
        "timestep_femtoseconds": declaration.timestep,
        "declared_steps": declaration.steps,
        "fractional_shift": list(declaration.fractional_shift),
        "steps": steps,
        "owner_changes_by_step": owner_changes,
        "migrated_atoms": int(np.sum(migrated)),
        "topology_epochs": [
            int(addressable_rows(topology.epoch)) for topology in distributed.topologies
        ],
        "step_index": step_index,
        "cell_vectors": cell.tolist(),
        "ledger": ledger,
        "cell_epoch": int(addressable_rows(controllers.cell_epoch)),
        "owned_rows_by_process": owned,
        "strain_response": bool(dynamics.plan.strain_response),
        "dynamic_cell": bool(runtime.plan.dynamic_cell),
        "strategy": runtime.plan.strategy.value,
        "owner_count": runtime.plan.ownership.owner_count,
    }
    local = {
        "position_error": position_error,
        "momentum_error": momentum_error,
        "cell_error": cell_error,
        "ledger_error": ledger_error,
        "controller_error": controller_error,
        "owner_epoch_advances": (epochs_after - epochs_before).tolist(),
    }
    return replicated, local, errors


def static_cell_refusal(
    case: FamilyCase, route: RouteBinding, ownership: Any, strategy: Strategy, /
) -> dict[str, str] | None:
    """Attempt MTK cell dynamics on the same domain without ``dynamic_cell``."""
    from phydrax.atomistic._domain_dynamics import AtomisticDomainDynamicsPlan

    try:
        AtomisticDomainDynamicsPlan(
            domain_plan(case, route, ownership, strategy, dynamic_cell=False),
            _integrator(DYNAMICS[case.family].timestep),
            _ensemble(case),
        )
    except ValueError as error:
        return _refusal(error, "dynamics-plan")
    return None


# -------------------------------------------------------------- rank entry


def _declared_outcome(
    declaration: RouteDeclaration, refusal: dict[str, str] | None, /
) -> tuple[dict[str, Any], list[str]]:
    """Route record of an admission attempt checked against its declaration."""
    record: dict[str, Any] = {
        "declared": "qualified" if declaration.refusal is None else "refused",
        "description": declaration.description,
        "status": "admitted" if refusal is None else "refused",
        "refusal": refusal,
    }
    errors: list[str] = []
    if declaration.refusal is None and refusal is not None:
        errors.append(
            f"route {declaration.name}: the qualified route was refused "
            f"({refusal['type']}: {refusal['message']})"
        )
    if declaration.refusal is not None:
        if refusal is None:
            errors.append(
                f"route {declaration.name}: the native runtime admitted a route "
                "declared refused; the family route declaration is stale"
            )
        elif declaration.refusal not in refusal["message"]:
            errors.append(
                f"route {declaration.name}: refusal {refusal['message']!r} lacks "
                f"the declared {declaration.refusal!r}"
            )
    return record, errors


def run_family_rank(
    scenario: FamilyScenario, directory: Path, strategy: Strategy, family: Family
) -> dict[str, Any]:
    """Execute one family scenario as one rank of a coupled launch."""
    from tools.atomistic_distributed_smoke import runtime_record

    del directory  # family scenarios publish nothing
    _require_family(scenario, family)
    if strategy not in ("halo", "graph-partition"):
        raise ValueError(f"Unknown strategy {strategy!r}.")
    runtime = ExecutionRuntime.current()
    case = family_case(family)
    ownership = _ownership(runtime, case.local_capacity)
    errors: list[str] = []
    routes: dict[str, Any] = {}
    for declaration in declared_routes(scenario, family):
        if declaration is STATIC_CELL_REFUSAL:
            binding = case.routes[DYNAMICS_ROUTE[family]]
            refusal = static_cell_refusal(case, binding, ownership, strategy)
            record, failures = _declared_outcome(declaration, refusal)
            routes[declaration.name] = {"replicated": record, "local": {}}
            errors.extend(failures)
            continue
        binding = case.routes[declaration.name]
        domain, refusal = admit_route(
            case,
            binding,
            ownership,
            strategy,
            dynamic_cell=scenario == "family-dynamics",
        )
        record, failures = _declared_outcome(declaration, refusal)
        errors.extend(failures)
        local: dict[str, Any] = {}
        if domain is not None and declaration.refusal is None:
            check = route_parity if scenario == "family-parity" else dynamics_parity
            replicated, local, failures = check(case, binding, domain)
            record.update(replicated)
            record["status"] = "failed" if failures else "qualified"
            errors.extend(f"route {declaration.name}: {item}" for item in failures)
        routes[declaration.name] = {"replicated": record, "local": local}
    return {
        "scenario": scenario,
        "strategy": strategy,
        "family": family,
        "runtime": runtime_record(runtime),
        "routes": routes,
        "atom_count": int(case.positions.shape[0]),
        "tolerances": {
            "energy_relative": ENERGY_RELATIVE_TOLERANCE,
            "force_absolute": FORCE_ABSOLUTE_TOLERANCE,
            "stress_absolute": STRESS_ABSOLUTE_TOLERANCE,
            "term_scaled": TERM_TOLERANCE,
            "dynamics_absolute": DYNAMICS_ABSOLUTE_TOLERANCE,
        },
        "reference": (
            "single-device AtomisticPotentialProgram (LearnedGraphPotentialTerm and "
            "the same canonical terms) over its own neighborhood bundle"
            + (
                "; P5 AtomisticDynamicsPlan MTK NPT"
                if scenario == "family-dynamics"
                else ""
            )
        ),
        "model_scope": EVIDENCE_SCOPE,
        "errors": errors,
        "passed": not errors,
    }


# ------------------------------------------------------------ cross-rank


def family_checks(
    scenario: FamilyScenario,
    ranks: Sequence[Mapping[str, Any] | None],
    processes: int,
    launch: Mapping[str, Any],
    directory: Path,
) -> list[str]:
    """Family-specific cross-rank checks of one completed coupled launch.

    Launch success, rank presence, process indices, per-rank errors and the
    execution group are the launcher's generic checks.
    """
    del launch, directory
    complete = [rank for rank in ranks if rank is not None]
    if len(complete) != processes:
        return []
    errors: list[str] = []
    families = {rank.get("family") for rank in complete}
    if len(families) != 1:
        return ["ranks executed different families"]
    (family,) = families
    if family not in SCENARIO_FAMILIES.get(scenario, ()):
        return [f"ranks executed undeclared family {family!r}"]
    names = [route.name for route in declared_routes(scenario, family)]
    for index, rank in enumerate(complete):
        if sorted(rank.get("routes", {})) != sorted(names):
            errors.append(f"rank {index} did not record exactly the declared routes")
    if errors:
        return errors
    for name in names:
        records = [rank["routes"][name]["replicated"] for rank in complete]
        if any(record != records[0] for record in records[1:]):
            errors.append(f"ranks disagree on the replicated outcome of route {name}")
        owned = records[0].get("owned_rows_by_process")
        if owned is not None and sum(owned) != complete[0]["atom_count"]:
            errors.append(f"route {name}: owned rows do not partition the system")
    return errors


__all__ = [
    "admit_route",
    "declared_routes",
    "domain_plan",
    "DYNAMICS",
    "dynamics_parity",
    "DynamicsDeclaration",
    "FAMILIES",
    "Family",
    "family_case",
    "family_checks",
    "FAMILY_SCENARIOS",
    "FamilyCase",
    "FamilyScenario",
    "LEARNED_FAMILIES",
    "pencil_grid",
    "ReciprocalSchedule",
    "reference_program",
    "route_parity",
    "RouteBinding",
    "RouteDeclaration",
    "ROUTES",
    "run_family_rank",
    "SCENARIO_FAMILIES",
    "static_cell_refusal",
    "UMA_MAX_NEIGHBORS",
]

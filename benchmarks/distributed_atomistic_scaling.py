#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated strong/weak scaling of distributed atomistic domains.

Rows vary the owner count of one periodic triclinic jittered-lattice crystal at
fixed number density: ``strong`` keeps the atom count fixed, ``weak`` keeps the
atoms per owner fixed. Every row declares its family (native MACE,
conditioned MACE with a case-level joint embedding, MACE-LES with the
distributed latent-Ewald epilogue, AIMNet2, layered UMA, or ``lj-pme``: a
terms-only runtime of the canonical Lennard-Jones and PME domain terms on
their own roles), its spatial strategy (halo or replicated-geometry graph
partition) and its owner execution (device collectives or lane-reference vmap
lanes), and separately measures:

* ``prepare``: plan admission, runtime preparation and state ingress (global
  host ingress in one process; process-local ingress across processes);
* ``evaluate``: lowering, compilation, first and warm E/F/stress execution,
  compiler memory, and the ordered collective program of the compiled
  evaluation (primitive, operand bytes, loop nesting) for device owners;
* ``rebuild``: one forced owner-crossing migration transaction (lower,
  compile, first, warm) and the post-migration evaluation of the compiled
  executable;
* ``md-step``: one distributed velocity-Verlet NVE step (DomainDynamics);
* ``training-step``: one data x domain training-kernel attempt
  (`run_distributed_training_attempt`, compilation inside its first call),
  successful only when the kernel's canonical outcome is ``accepted``;
* ``publication``: collective per-process checkpoint publication with one
  global commit, then same-topology restore and its bitwise check.

``--reciprocal`` adds standalone distributed PME (distributed FFT and the
explicitly named replicated-mesh comparison) and explicit-mode Ewald rows:
preparation, patch/packet resources, energy+force execution phases and, for
the distributed FFT, the isolated forward/inverse transform round trip on the
same spectral layout. Particle meshes use x-slabs; ``--pencil-grids 2x2 2x3``
adds distributed-FFT PME rows on the canonical pencil layout (owners
``PX*PY``, owner ``r`` at pencil coordinate ``(r % PX, r // PX)``), recording
the layout ids, local block shapes, owner coordinates/devices, the declared
collective order and, for device owners, the traced collective program.
One-process PME rows record parity of energy, forces and the cell gradient
against the unchanged single-device term. One-process distributed-FFT round
trips are gated against the global FFT and the input at the declared float64
tolerance (finite error <= 1e-12 times max(1, reference magnitude)); a
mismatch makes the row unsuccessful. ``--reciprocal-only`` skips the
domain campaign rows. The latent-Ewald redistribution runs inside every
``mace-les`` domain evaluation.

Forced-CPU pencil evidence (functional only, never hardware evidence), with
``JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=6`` in the environment::

    python -m benchmarks.distributed_atomistic_scaling --owners devices
        --reciprocal-only --pencil-grids 2x2 2x3 3x2 --repeats 2

Observed-hardware rows run the same command without the CPU overrides; their
``evidence_scope`` names the observed platform and owner devices.

Every requested row and workload is recorded: refused (capacity/admission),
out-of-memory and failed rows keep their status, exception type, message and
traceback tail; nothing is dropped from a table. Memory is sampled per
process (host and allocator counters) after each phase; compiler bytes are
XLA estimates; logical bytes count the retained state arrays. Rows are
strict JSON through the shared `json_safe`: nonfinite numerical evidence (for
example the NaN energies of an unsuccessful native evaluation) is kept as the
string ``"nan"``/``"inf"``/``"-inf"`` beside its success/reason bits.

Evidence scope is recorded per row. Lane-reference rows and forced host-CPU
devices are functional evidence only; accelerator rows describe only the
devices actually observed; one-host multi-process rows are not multi-host
evidence; random-initialized model families say nothing about pretrained
fidelity. No scaling efficiency or universal speed claim is inferred.

Multi-process rows: ``--local-processes P --devices-per-process D`` relaunches
this driver as P coupled workers through `launch_local_processes` (one owner
per device of every process, so owner counts must equal ``P * D``). The
same driver run on every rank of an externally launched job records that
job's observed host topology.

Historical records produced by the predecessor driver
(``benchmarks/distributed_mace_scaling.json``) are retained unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.sharding import PartitionSpec

from benchmarks._io import write_json_atomic
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
from phydrax import config as pxconfig
from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group, ExecutionRuntime
from phydrax._execution_sampling import sample_device_memory, sample_host_memory
from phydrax._fingerprint import canonical_fingerprint
from phydrax.atomistic import AtomisticScaleContract, AtomisticUnitSystem
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionPolicy,
    AtomisticDomainRolePlan,
)
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._domain_state import AtomisticDomainStrategy
from phydrax.atomistic._feature_execution import AtomisticExchangeCapacityError
from phydrax.atomistic._graph_partition import (
    AtomisticGraphPartitionPlan,
    AtomisticGraphPartitionResourceError,
    graph_partition_memory_evidence,
)
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodRole,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
    owner_reciprocal_energy,
    prepare_reciprocal_distribution,
    ReciprocalDistributionCapacityError,
    ReciprocalSitePayload,
)
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.execution import (
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_embedding import MACEEmbeddingFeature
from phydrax.nn.atomistic._mace_readout import MACEEnergyScaling
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.typing import parse
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tools.atomistic_distributed_smoke import (
    addressable_rows,
    domain_repository,
    free_loopback_port,
    json_safe,
    launch_environment,
    owner_address,
    Platform,
    process_local_ingress,
    runtime_record,
    state_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "DISTRIBUTED_ATOMISTIC_SCALING_RESULT "

Family: TypeAlias = Literal[
    "mace", "mace-conditioned", "mace-les", "aimnet2", "uma", "lj-pme"
]
Strategy: TypeAlias = Literal["halo", "graph-partition"]
OwnerExecution: TypeAlias = Literal["devices", "lanes"]
ScalingCampaign: TypeAlias = Literal["strong", "weak"]
Workload: TypeAlias = Literal[
    "evaluate", "rebuild", "md-step", "training-step", "publication"
]
ReciprocalCase: TypeAlias = Literal["pme", "pme-replicated-mesh", "ewald"]
RowStatus: TypeAlias = Literal["ok", "refused", "oom", "failed"]

SKIN = 0.4
DENSITY = 0.08
SHEAR = np.array([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.15, -0.1, 1.0]])
MASSES = {1: 1.008, 6: 12.011, 8: 15.999}
STEP_FEMTOSECONDS = 0.5
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
_COLLECTIVES = frozenset(
    {
        "all_to_all",
        "all_gather",
        "all_gather_invariant",
        "reduce_scatter",
        "psum_scatter",
        "psum",
        "psum_invariant",
        "pmax",
        "pmin",
        "ppermute",
        "pbroadcast",
    }
)
_LOOPS = frozenset({"while", "scan"})


# ------------------------------------------------------------------ failures


def failure(error: BaseException, /) -> dict[str, Any]:
    """Classify and retain one failure; capacity refusals are not OOM."""
    text = str(error)
    status: RowStatus
    if isinstance(
        error,
        (
            AtomisticExchangeCapacityError,
            AtomisticGraphPartitionResourceError,
            ReciprocalDistributionCapacityError,
            ValueError,
        ),
    ):
        status = "refused"
    elif (
        isinstance(error, MemoryError)
        or "RESOURCE_EXHAUSTED" in text
        or ("out of memory" in text.lower())
    ):
        status = "oom"
    else:
        status = "failed"
    tail = traceback.format_exception(type(error), error, error.__traceback__)
    return {
        "status": status,
        "error_type": type(error).__name__,
        "error": text[:4000],
        "traceback_tail": "".join(tail)[-4000:],
    }


def _guarded(operation: Callable[[], dict[str, Any]], /) -> dict[str, Any]:
    try:
        result = operation()
    except Exception as error:
        return failure(error)
    result.setdefault("status", "ok")
    return result


def _memory() -> dict[str, Any]:
    return {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }


def _scalar(value: jax.Array, /) -> Any:
    return addressable_rows(value).item()


# ------------------------------------------------------------------ models


@dataclass(frozen=True, slots=True)
class ModelCase:
    """A declared family: layered model and/or canonical domain terms.

    ``model`` is ``None`` for a terms-only (classical) runtime, whose ingress
    species are the system's atom-type IDs (``type_ids``).
    """

    family: Family
    model: Any | None
    numbers: tuple[int, ...]
    streaming: StreamedRelationPlan
    conditioning: Callable[[int, np.ndarray], AtomisticConditioning | None]
    reciprocal: AtomisticReciprocalDistributionPlan | None
    terms: tuple[Any, ...] = ()
    type_ids: np.ndarray | None = None


def _mace(
    layers: int,
    /,
    *,
    energy_scaling: MACEEnergyScaling = "unscaled",
    embedding_features: Sequence[MACEEmbeddingFeature] = (),
    embedding_readout: bool = False,
) -> MACEPotential:
    scaled = energy_scaling == "scale-shift"
    architecture = MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=8,
        cutoff_power=5,
        channel_count=16,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic",) + ("real-agnostic-residual",) * (layers - 1),
        correlations=(3,) * layers,
        radial_widths=(32, 32),
        readout_width=16 if layers > 1 else None,
        average_neighbor_count=12.0,
        energy_scaling=energy_scaling,
        embedding_features=embedding_features,
        embedding_readout=embedding_readout,
    )
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.array([[-0.3, -1.1]]) if scaled else np.array([-0.3, -1.1]),
        energy_scale=np.array([1.3]) if scaled else None,
        energy_shift=np.array([0.2]) if scaled else None,
        key=jr.key(0),
        acceleration=None,
    )


def _mace_streaming(model: Any, /) -> StreamedRelationPlan:
    return StreamedRelationPlan(
        receiver_tile=32,
        edge_tile=512,
        channel_capacity=model.configuration.message_payload_width(),
        accumulation="deterministic",
    )


def _no_conditioning(local: int, numbers: np.ndarray, /) -> None:
    del local, numbers


def _minimum_width(cell: np.ndarray, /) -> float:
    """Smallest perpendicular width of a (skew) cell."""
    volume = abs(float(np.linalg.det(cell)))
    return min(
        volume / float(np.linalg.norm(np.cross(cell[(a + 1) % 3], cell[(a + 2) % 3])))
        for a in range(3)
    )


def _classical_case(
    atoms: int, cell: np.ndarray, owners: int, mesh_spacing: float, /
) -> ModelCase:
    """Canonical Lennard-Jones + PME terms of one water-like H/O system (eV-Å-fs)."""
    from phydrax.atomistic import AtomisticSystemPlan
    from phydrax.atomistic._classical import LennardJonesPotential
    from phydrax.atomistic._domain_terms import (
        AtomisticDomainEwaldTerm,
        AtomisticDomainLennardJonesTerm,
    )
    from phydrax.atomistic._electrostatics import ParticleMeshEwaldPotential

    _, _, numbers = structure(atoms, (1, 8), atoms)
    types = (numbers == 8).astype(np.int32)
    charges = np.where(numbers == 8, -0.8, 0.4)
    charges = charges - charges.mean()
    system = AtomisticSystemPlan(
        np.arange(atoms, dtype=np.int64),
        numbers.astype(np.int64),
        np.asarray([MASSES[int(number)] for number in numbers], dtype=np.float64),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=types.astype(np.int64),
        charges=charges.astype(np.float64),
        cell=PeriodicCell(cell),
    ).prepare()
    # Minimum-image roles: the cutoffs stay inside half the smallest width.
    cutoff = min(3.0, 0.4 * _minimum_width(cell))
    grid = tuple(
        _mesh_extent(float(np.linalg.norm(row)), owners, mesh_spacing) for row in cell
    )
    lennard_jones = LennardJonesPotential(
        np.asarray([0.005, 0.01]), np.asarray([1.0, 1.4]), cutoff
    ).prepare(system)
    pme = ParticleMeshEwaldPotential(3.0 / cutoff, cutoff, grid, spline_degree=4).prepare(
        system
    )
    terms = (
        AtomisticDomainLennardJonesTerm(lennard_jones),
        AtomisticDomainEwaldTerm(
            pme, AtomisticReciprocalDistributionPlan(pme, reuse_reach=SKIN)
        ),
    )
    return ModelCase(
        "lj-pme",
        None,
        (1, 8),
        StreamedRelationPlan(),
        _no_conditioning,
        None,
        terms=terms,
        type_ids=types,
    )


def model_case(
    family: Family,
    layers: int,
    atoms: int,
    cell: np.ndarray,
    owners: int = 1,
    mesh_spacing: float = 1.0,
) -> ModelCase:
    """Randomly initialized model (no pretrained claim) or canonical terms."""
    if family == "lj-pme":
        return _classical_case(atoms, cell, owners, mesh_spacing)
    match family:
        case "mace":
            model = _mace(layers)
            return ModelCase(
                family, model, (1, 8), _mace_streaming(model), _no_conditioning, None
            )
        case "mace-conditioned":
            model = _mace(
                layers,
                embedding_features=(
                    MACEEmbeddingFeature(
                        "total_charge",
                        "categorical",
                        level="case",
                        width=3,
                        classes=5,
                        offset=2,
                    ),
                    MACEEmbeddingFeature(
                        "total_spin",
                        "categorical",
                        level="case",
                        width=2,
                        classes=3,
                        offset=-1,
                    ),
                ),
                embedding_readout=True,
            )
            mace_conditioning = model.conditioning

            def conditioned(local: int, numbers: np.ndarray, /) -> AtomisticConditioning:
                del numbers
                return AtomisticConditioning(
                    mace_conditioning,
                    {
                        "total_charge": np.asarray([0], dtype=np.int64),
                        "total_spin": np.asarray([2], dtype=np.int64),
                    },
                    case_count=1,
                    atom_capacity=local,
                )

            return ModelCase(
                family, model, (1, 8), _mace_streaming(model), conditioned, None
            )
        case "mace-les":
            from phydrax.atomistic._latent_electrostatics import (
                LatentEwaldDefinition,
                LatentEwaldResources,
            )
            from phydrax.nn.atomistic._mace_les import MACELESPotential

            definition = LatentEwaldDefinition()
            longest = float(np.max(np.linalg.norm(cell, axis=1)))
            model = MACELESPotential.initialize(
                _mace(layers, energy_scaling="scale-shift"),
                definition,
                atom_capacity=atoms,
                resources=LatentEwaldResources(
                    mode_extents=definition.covering_extents(longest)
                ),
                key=jr.key(5),
            )
            return ModelCase(
                family,
                model,
                (1, 8),
                _mace_streaming(model),
                _no_conditioning,
                AtomisticReciprocalDistributionPlan(model.latent_ewald, reuse_reach=SKIN),
            )
        case "aimnet2":
            from phydrax.nn.atomistic._aimnet2_source import (
                aimnet2_potential_from_source,
                externalize_source_long_range,
            )
            from tests._support.aimnet2_source import RC_S, source_inventory

            declaration, tensors = source_inventory(3)
            core, state, _ = externalize_source_long_range(declaration, tensors)
            model = aimnet2_potential_from_source(
                SCALE, core, state, neighbor_cutoff=RC_S
            ).potential
            charge_conditioning = model.conditioning_spec

            def charged(local: int, numbers: np.ndarray, /) -> AtomisticConditioning:
                del numbers
                return AtomisticConditioning(
                    charge_conditioning,
                    {"charge": np.asarray([0.0], dtype=np.float64)},
                    case_count=1,
                    atom_capacity=local,
                )

            return ModelCase(
                family, model, (1, 6, 8), StreamedRelationPlan(), charged, None
            )
        case "uma":
            from phydrax.nn.atomistic._uma_source import (
                parse_uma_source_config,
                uma_initial_source_tensors,
                uma_model_from_source,
            )
            from tests._support.uma_source import uma_config

            configuration, tasks = uma_config()
            declaration = parse_uma_source_config(
                configuration, tasks, roll_gauge="two-chart-nlerp-zero-roll"
            )
            model, _ = uma_model_from_source(
                SCALE,
                declaration,
                uma_initial_source_tensors(declaration, key=jr.key(3)),
                streaming=StreamedRelationPlan(
                    receiver_tile=32, edge_tile=512, channel_capacity=512
                ),
                node_block=32,
            )

            def omol(local: int, numbers: np.ndarray, /) -> AtomisticConditioning:
                # Neutral system; the spin multiplicity matches electron parity.
                multiplicity = 1 if int(np.sum(numbers)) % 2 == 0 else 2
                return model.schema.conditioning(
                    ["omol"], atom_capacity=local, charges=[0], spins=[multiplicity]
                )

            return ModelCase(family, model, (1, 8), StreamedRelationPlan(), omol, None)
    raise ValueError(f"Unknown model family {family!r}.")


# ------------------------------------------------------------- structure/plan


def structure(
    atoms: int, numbers: Sequence[int], seed: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered simple-cubic fractional lattice in a sheared cell (no overlaps)."""
    if atoms < 1:
        raise ValueError("atoms must be positive.")
    cell = (atoms / DENSITY) ** (1.0 / 3.0) * SHEAR
    side = math.ceil(atoms ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    chosen = grid[rng.permutation(grid.shape[0])[:atoms]]
    jitter = rng.uniform(-0.15, 0.15, chosen.shape)
    fractional = np.mod((chosen + 0.5 + jitter) / side, 1.0)
    species = np.asarray(numbers, dtype=np.int32)[rng.integers(0, len(numbers), atoms)]
    return cell, fractional @ cell, species


def _neighbors(radius: float, /) -> int:
    return math.ceil(DENSITY * 4.0 / 3.0 * math.pi * radius**3)


def ownership_plan(owners: int, local: int, execution: OwnerExecution, /) -> Any:
    """Owner placement: a device subset, reference lanes, or every process device."""
    if owners < 1:
        raise ValueError("owners must be positive.")
    multi_process = jax.process_count() > 1
    if multi_process and execution != "devices":
        raise ValueError(
            "Multi-process rows place one owner on every device of every process "
            "(--owners devices); lane-reference owners run in one process."
        )
    runtime = ExecutionRuntime.current()
    parent = runtime.root_group.spec
    if multi_process:
        if owners != parent.device_count:
            raise ValueError(
                "Multi-process rows place one owner on every device of every "
                f"process: {parent.device_count} owners, not {owners}."
            )
        return DistributedOwnershipPlan(owner_address(), runtime.root_group, local)
    if execution == "lanes":
        return DistributedOwnershipPlan(
            owner_address(),
            runtime.child_groups(parent.device_count)[0],
            local,
            owner_lanes=owners,
        )
    if owners > parent.device_count:
        raise ValueError(
            f"{owners} owners require at least {owners} visible devices; "
            f"only {parent.device_count} are available."
        )
    keys = parent.device_keys[:owners]
    # Keep the first child's identity for previously divisible allocations.
    group = bind_execution_group(
        ExecutionGroupSpec(
            canonical_fingerprint(
                {
                    "parent_group_id": parent.group_id,
                    "group_index": 0,
                    "device_keys": [list(key) for key in keys],
                }
            ),
            parent.process_indices,
            keys,
            mesh_axes=(("device", owners),),
            parent_group_id=parent.group_id,
            process_host_ids=parent.process_host_ids,
        )
    )
    return DistributedOwnershipPlan(owner_address(), group, local)


def domain_plan(
    case: ModelCase,
    cell: np.ndarray,
    owners: int,
    local: int,
    execution: OwnerExecution,
    strategy: Strategy,
    /,
    *,
    message_capacity_bytes: int,
    replicated_memory_bytes: int,
) -> AtomisticDomainPlan:
    ownership = ownership_plan(owners, local, execution)
    periodic = PeriodicCell(cell)
    declared = []
    if case.model is not None:
        declared.append(
            (
                case.model.requirements.neighborhood(
                    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
                ),
                "model",
            )
        )
    # Each domain term executes on its own role(s) of its own requirement.
    declared += [
        (requirement, term.term_key)
        for term in case.terms
        for requirement in term.requirements()
    ]
    roles = []
    degree = 32
    for requirement, term_key in declared:
        radius = requirement.bounded_cutoff() + SKIN
        role_degree = max(32, 6 * _neighbors(radius))
        degree = max(degree, role_degree)
        roles.append(
            AtomisticDomainRolePlan(
                requirement,
                periodic.image_stencil(radius, maximum_image_count=4096),
                streaming=case.streaming,
                skin=SKIN,
                alias_capacity=local * 27,
                edge_capacity=local * role_degree,
                halo_capacity=local,
                term_key=term_key,
            )
        )
    partition = FractionalOwnerPartition(periodic, (owners, 1, 1))
    kind = AtomisticDomainStrategy(strategy)
    graph = (
        AtomisticGraphPartitionPlan(
            ownership,
            partition,
            tuple(roles),
            replicated_memory_bytes=replicated_memory_bytes,
            receiver_edge_capacity=degree,
            maximum_particles_per_cell=64,
            search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 24),
        )
        if kind is AtomisticDomainStrategy.GRAPH_PARTITION
        else None
    )
    return AtomisticDomainPlan(
        ownership,
        partition,
        tuple(roles),
        migration_capacity=local,
        message_capacity_bytes=message_capacity_bytes,
        strategy=kind,
        reduction=AtomisticDomainReductionPolicy("deterministic"),
        graph_partition=graph,
        reciprocal=case.reciprocal,
        terms=case.terms,
    )


# ------------------------------------------------------------ evidence


def _aval_bytes(aval: Any, /) -> int | None:
    try:
        itemsize = np.dtype(aval.dtype).itemsize
    except (TypeError, ValueError):
        return None
    return int(np.prod(aval.shape, dtype=np.int64)) * itemsize


def _subjaxprs(value: Any, /) -> list[Any]:
    if hasattr(value, "eqns"):
        return [value]
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        return [value.jaxpr]
    if isinstance(value, (tuple, list)):
        return [item for entry in value for item in _subjaxprs(entry)]
    return []


def collective_program(jaxpr: Any, /, *, loop_depth: int = 0) -> list[dict[str, Any]]:
    """Ordered collectives of a traced program with their operand bytes.

    Loop bodies are listed once with their nesting depth (iteration counts of
    ``while`` loops are data dependent and not multiplied in).
    """
    program = []
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in _COLLECTIVES:
            sizes = [_aval_bytes(variable.aval) for variable in equation.invars]
            program.append(
                {
                    "primitive": name,
                    "operand_bytes": None
                    if any(size is None for size in sizes)
                    else sum(size for size in sizes if size is not None),
                    "loop_depth": loop_depth,
                }
            )
        depth = loop_depth + (1 if name in _LOOPS else 0)
        for value in equation.params.values():
            for nested in _subjaxprs(value):
                program.extend(collective_program(nested, loop_depth=depth))
    return program


def _communication(
    runtime: Any, state: Any, function: Callable[[Any], Any], /
) -> dict[str, Any]:
    plan = runtime.plan
    record: dict[str, Any] = {
        "message_capacity_bytes": plan.message_capacity_bytes,
        "halo_capacity": plan.roles[0].halo_capacity,
    }
    if plan.graph_partition is not None:
        memory = graph_partition_memory_evidence(
            plan, position_dtype=state.positions.dtype
        )
        record["graph_partition_charges"] = [list(item) for item in memory.charges]
        record["graph_partition_replicated_memory_bytes"] = memory.replicated_memory_bytes
        record["topology_replicated_bytes"] = _scalar(
            state.topologies[0].evidence.replicated_bytes
        )
    if plan.ownership.lane_reference:
        record["collectives"] = None
        record["collectives_reason"] = (
            "lane-reference collectives are batched into local reductions"
        )
        return record
    program = collective_program(jax.make_jaxpr(function)(state).jaxpr)
    record["collectives"] = program
    record["collective_operand_bytes"] = sum(
        entry["operand_bytes"] or 0 for entry in program if entry["loop_depth"] == 0
    )
    return record


def _phases(
    function: Callable[[Any], Any],
    argument: Any,
    repeats: int,
    /,
    *,
    analysis: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Lower, compile, first and warm execution of one jitted function.

    Returns the compiled executable, the first result and the phase record.
    """
    jitted = jax.jit(function)
    compiled, timing = measure_lower_and_compile(
        lambda: jitted.lower(argument), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(argument))
    _, warm = measure_repeated(lambda: compiled(argument), warmup=1, repeats=repeats)
    record: dict[str, Any] = {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "warm": warm.to_dict(),
    }
    if analysis:
        record["compiler"] = asdict(
            compiler_evidence(
                compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
            )
        )
    record["memory"] = _memory()
    return compiled, first, record


# ------------------------------------------------------------ workloads


def _blocked(state: Any, logical: np.ndarray, /) -> jax.Array:
    """Owner-blocked rows of a logical-order host table (by stable ID)."""
    count = logical.shape[0]

    @jax.jit
    def rows(ids: jax.Array, active: jax.Array) -> jax.Array:
        values = jnp.asarray(logical)[jnp.clip(ids, 0, count - 1)]
        mask = active.reshape(active.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values, jnp.zeros((), values.dtype))

    return rows(state.layout.stable_ids, state.layout.active)


def _crossing_displacement(
    cell: np.ndarray, positions: np.ndarray, owners: int, /
) -> np.ndarray:
    """Short thermal drift plus one forced crossing into the next owner."""
    displacement = np.random.default_rng(7).normal(scale=0.02, size=positions.shape)
    if owners > 1:
        fractional = np.linalg.solve(cell.T, positions[0] + displacement[0])
        owner = int(np.floor(np.mod(fractional[0], 1.0) * owners))
        target = ((owner + 1) % owners + 0.5) / owners
        displacement[0] += (target - np.mod(fractional[0], 1.0)) * cell[0]
    return displacement


def _rebuild(
    runtime: Any,
    state: Any,
    compiled_evaluate: Any,
    cell: np.ndarray,
    positions: np.ndarray,
    owners: int,
    repeats: int,
    /,
) -> dict[str, Any]:
    displacement = _crossing_displacement(cell, positions, owners)
    moved = runtime.with_dynamics(
        state,
        state.positions + _blocked(state, displacement),
        state.momenta,
        step_index=1,
    )
    _, transition, phases = _phases(
        lambda current: runtime.rebuild(current), moved, repeats
    )
    # The evaluate workload's compiled executable: the migrated state keeps the
    # pytree structure, shapes and owner shardings, so nothing recompiles.
    evaluation, post_seconds = measure_synchronized(
        lambda: compiled_evaluate(transition.state)
    )
    migrated = int(_scalar(transition.migration.migrated))
    committed = bool(_scalar(transition.committed))
    crossed = owners == 1 or migrated > 0
    return {
        **phases,
        "committed": committed,
        "migrated_atoms": migrated,
        "reasons": int(_scalar(transition.reasons)),
        "post_migration_evaluation_seconds": post_seconds,
        "post_migration_successful": bool(_scalar(evaluation.successful)),
        "successful": committed and crossed and bool(_scalar(evaluation.successful)),
    }


def _md_step(runtime: Any, state: Any, atoms: int, repeats: int, /) -> dict[str, Any]:
    from phydrax.atomistic._domain_dynamics import (
        AtomisticDomainDynamicsPlan,
        AtomisticDomainEnsemblePlan,
    )
    from phydrax.atomistic._dynamics import VelocityVerletPlan

    dynamics = AtomisticDomainDynamicsPlan(
        runtime.plan,
        VelocityVerletPlan(STEP_FEMTOSECONDS),
        AtomisticDomainEnsemblePlan(
            AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
            ensemble="nve",
            atom_count=atoms,
        ),
    ).prepare(runtime)
    initialized, initialize_seconds = measure_synchronized(
        lambda: dynamics.initialize(state)
    )
    _, step, phases = _phases(
        lambda current: dynamics.step_detailed(current), initialized, repeats
    )
    return {
        **phases,
        "initialize_seconds": initialize_seconds,
        "integrator": "velocity-verlet",
        "ensemble": "nve",
        "step_femtoseconds": STEP_FEMTOSECONDS,
        "rejection_reasons": int(_scalar(step.rejection_reasons)),
        "successful": bool(_scalar(step.successful)),
    }


def _training_group(runtime: Any, /) -> Any:
    """``(data, owner)`` mesh over the domain devices (data axis of size 1)."""
    group = runtime.plan.ownership.execution_group
    spec = group.spec
    axis = runtime.plan.ownership.axis_name
    return bind_execution_group(
        ExecutionGroupSpec(
            canonical_fingerprint(
                {"parent_group_id": spec.group_id, "kind": "data-domain-training"}
            ),
            spec.process_indices,
            spec.device_keys,
            mesh_axes=(("data", 1), (axis, runtime.plan.ownership.owner_count)),
            parent_group_id=spec.group_id,
            process_host_ids=spec.process_host_ids,
        )
    )


def _training_step(
    runtime: Any, state: Any, evaluation: Any, model: Any, repeats: int, /
) -> dict[str, Any]:
    import optax

    from phydrax._differentiation import ComponentAuthority
    from phydrax._training_kernel import (
        OptaxUpdateRule,
        prepare_training_kernel,
        TrainingAttemptOutcome,
        TrainingKernelSpec,
    )
    from phydrax.atomistic._training_distribution import (
        atomistic_domain_training_objectives,
        AtomisticDataParallelReduction,
        AtomisticDomainTrainingShard,
        AtomisticTrainingDistributionPlan,
        run_distributed_training_attempt,
    )
    from phydrax.atomistic._training_objectives import (
        AtomisticEnergyObjective,
        AtomisticForceObjective,
    )

    ownership = runtime.plan.ownership
    distribution = (
        AtomisticTrainingDistributionPlan(
            ownership.execution_group, data_lanes=1, domain=runtime.plan
        )
        if ownership.lane_reference
        else AtomisticTrainingDistributionPlan(
            _training_group(runtime), domain=runtime.plan
        )
    )
    reduction = AtomisticDataParallelReduction(distribution)
    # Synthetic labels offset from the model's own prediction (a timing
    # workload with a nonzero gradient, not a fit to physical data).
    energy = jnp.asarray([_scalar(evaluation.energy) + 0.1], dtype=jnp.float64)
    shard = AtomisticDomainTrainingShard.from_domain_state(
        runtime,
        state,
        labels={
            "energy": (energy, jnp.asarray([True])),
            "forces": (0.9 * evaluation.forces, state.layout.active),
        },
        label_levels={"energy": "structure", "forces": "atom"},
    )
    kernel = prepare_training_kernel(
        model,
        atomistic_domain_training_objectives(
            runtime,
            (
                AtomisticEnergyObjective(reduction="per-atom-structure"),
                AtomisticForceObjective(reduction="component"),
            ),
            reduction,
            scales=(1.0, 1.0),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(optax.sgd(1.0e-4), rule_id=f"sgd:{reduction.reduction_id}"),
            context="distributed atomistic scaling benchmark",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    batch = reduction.assemble((shard,))
    initial = kernel.init(model, jr.key(3))
    (trained, evidence), first_seconds = measure_synchronized(
        lambda: run_distributed_training_attempt(kernel, reduction, initial, batch)
    )
    _, warm = measure_repeated(
        lambda: run_distributed_training_attempt(kernel, reduction, initial, batch),
        warmup=1,
        repeats=repeats,
    )
    del trained
    outcome, rule_accepted, supported, finite, value = (
        np.asarray(item)
        for item in jax.device_get(
            (
                evidence.outcome,
                evidence.rule_accepted,
                evidence.supported,
                evidence.finite,
                evidence.value,
            )
        )
    )
    attempt = TrainingAttemptOutcome(int(outcome))
    return {
        "first_execution_seconds": first_seconds,
        "compilation": "inside the first attempt (not separable from first execution)",
        "warm": warm.to_dict(),
        "data_shards": distribution.data_shards,
        "reduction_id": reduction.reduction_id,
        "objective_value": float(value),
        "outcome": attempt.name.lower(),
        "rule_accepted": bool(rule_accepted),
        "supported": bool(supported),
        "finite": bool(finite),
        # An update succeeded only when the canonical kernel accepted it.
        "successful": attempt is TrainingAttemptOutcome.ACCEPTED,
        "memory": _memory(),
    }


def _checkpoint_limits(arguments: argparse.Namespace, /) -> ArrayArchiveLimits:
    """Explicit publication/restore admission policy of every checkpoint row.

    ``max_manifest_bytes`` also bounds each process's authenticated shard
    inventory; ``max_members`` bounds its shard count.
    """
    return ArrayArchiveLimits(
        max_members=arguments.checkpoint_members,
        max_manifest_bytes=arguments.checkpoint_inventory_bytes,
    )


def _limits_payload(limits: ArrayArchiveLimits, /) -> dict[str, Any]:
    return {**asdict(limits), "allowed_dtype_kinds": sorted(limits.allowed_dtype_kinds)}


def _publication(
    runtime: Any,
    state: Any,
    directory: Path,
    row_id: str,
    limits: ArrayArchiveLimits,
    /,
) -> dict[str, Any]:
    from phydrax.atomistic._domain_checkpoint import (
        publish_domain_checkpoint,
        restore_domain_checkpoint,
    )
    from phydrax.lifecycle._distributed_checkpoint import _INVENTORY_LOGICAL_NAME

    repository = domain_repository(directory / row_id)
    checkpoint = f"scaling-{row_id}"
    publication, publish_seconds = measure_host(
        lambda: publish_domain_checkpoint(
            repository,
            runtime,
            state,
            checkpoint_id=checkpoint,
            writer_id=f"rank-{jax.process_index()}",
            limits=limits,
        )
    )
    restored, restore_seconds = measure_synchronized(
        lambda: restore_domain_checkpoint(
            repository, runtime, state, checkpoint, limits=limits
        )
    )
    bitwise = state_digest(restored.state) == state_digest(state)
    # This process's committed artifact: its authenticated inventory stream and
    # the generic metadata values that must stay below the fixed ceiling.
    artifact = repository.get_manifest(f"{checkpoint}.process-{jax.process_index()}")
    metadata = dict(artifact.metadata)
    inventory_chunks = [
        chunk
        for chunk in artifact.chunks
        if chunk.logical_name == _INVENTORY_LOGICAL_NAME
    ]
    return {
        "published_shards": len(publication.manifest.shards),
        "process_inventory": {
            "process_index": jax.process_index(),
            "artifact_manifest_id": artifact.manifest_id,
            "shard_count": int(metadata["shard_count"]),
            "inventory_bytes": sum(chunk.plaintext_size for chunk in inventory_chunks),
            "inventory_chunks": len(inventory_chunks),
            "maximum_metadata_value_bytes": max(
                len(value.encode("utf-8")) for value in metadata.values()
            ),
        },
        "publish_seconds": publish_seconds,
        "restore_seconds": restore_seconds,
        "manifest_id": publication.manifest.manifest_id,
        "restart_class": restored.restart_class,
        "restored_bitwise_on_this_process": bitwise,
        "successful": bitwise and restored.restart_class == "bitwise",
        "memory": _memory(),
    }


def _ingress(
    runtime: Any,
    case: ModelCase,
    positions: np.ndarray,
    numbers: np.ndarray,
    local: int,
    /,
) -> tuple[Any, str]:
    masses = np.asarray([MASSES[int(number)] for number in numbers])
    velocities = np.random.default_rng(1).normal(scale=0.005, size=positions.shape)
    momenta = masses[:, None] * velocities
    conditioning = case.conditioning(local, numbers)
    options = {} if conditioning is None else {"conditioning": conditioning}
    if case.model is None:
        # A terms-only runtime takes the system's atom-type IDs as species.
        species = np.asarray(case.type_ids, dtype=np.int32)
    else:
        species = np.asarray(
            case.model.species_indices(
                jnp.asarray(numbers), jnp.ones(numbers.shape, dtype=jnp.bool_)
            )
        )
    if jax.process_count() == 1:
        state = runtime.ingress_global(
            positions,
            species,
            rng_key=jr.key(1),
            momenta=momenta,
            masses=masses,
            **options,
        )
        return state, "global-host"
    state = process_local_ingress(
        runtime,
        None,  # species are already native indices
        positions,
        species,
        momenta,
        masses=masses,
        rng_key=jr.key(1),
        **options,
    )
    return state, "process-local"


def _scope(ownership: Any, /) -> str:
    selected = ownership.execution_group.devices
    platforms = sorted({device.platform for device in selected})
    if ownership.lane_reference:
        scope = "functional lane-reference execution on one device"
    elif platforms == ["cpu"] or pxconfig.snapshot().cpu_device_count is not None:
        scope = "functional CPU owner-device execution"
    else:
        scope = f"observed {platforms} owner-device execution"
    if jax.process_count() > 1:
        scope += f" across {jax.process_count()} processes"
    return (
        scope
        + "; no scaling-efficiency, multi-host (unless observed) or pretrained claim"
    )


def domain_row(
    index: int,
    campaign: ScalingCampaign,
    family: Family,
    strategy: Strategy,
    owners: int,
    atoms: int,
    execution: OwnerExecution,
    workloads: Sequence[Workload],
    arguments: argparse.Namespace,
    /,
) -> dict[str, Any]:
    """One scenario row; every requested workload is recorded with a status."""
    row: dict[str, Any] = {
        "row": index,
        "kind": "domain",
        "campaign": campaign,
        "family": family,
        "strategy": strategy,
        "owners": owners,
        "atoms": atoms,
        "atoms_per_owner": atoms / owners,
        "layers": arguments.layers,
        "execution": execution,
        "process_count": jax.process_count(),
        "workloads": {},
    }
    baseline = _memory()
    try:
        cell, _, _ = structure(atoms, (1,), atoms)
        case, model_seconds = measure_host(
            lambda: model_case(
                family, arguments.layers, atoms, cell, owners, arguments.mesh_spacing
            )
        )
        # The same seed yields the same cell and lattice positions; only the
        # species draw depends on the family.
        _, positions, numbers = structure(atoms, case.numbers, atoms)
        local = int(math.ceil(1.6 * atoms / owners)) + 4
        plan, plan_seconds = measure_host(
            lambda: domain_plan(
                case,
                cell,
                owners,
                local,
                execution,
                strategy,
                message_capacity_bytes=arguments.message_capacity_bytes,
                replicated_memory_bytes=arguments.replicated_memory_bytes,
            )
        )
        runtime, runtime_seconds = measure_host(
            lambda: prepare_atomistic_domain(plan, case.model)
        )
        (state, ingress), ingress_seconds = measure_synchronized(
            lambda: _ingress(runtime, case, positions, numbers, local),
            synchronizer=lambda value: jax.block_until_ready(value[0]),
        )
    except Exception as error:
        row.update(failure(error))
        row["successful"] = False
        row["memory"] = {"baseline": baseline, "after_failure": _memory()}
        return json_safe(row)
    evidence = state.topologies[0].evidence
    halo = state.topologies[0].halo
    row.update(
        status="ok",
        evidence_scope=_scope(plan.ownership),
        interaction_count=None
        if case.model is None
        else int(case.model.interaction_count),
        domain_terms=[type(term).__name__ for term in case.terms],
        roles=[
            {"term_key": role.term_key, "cutoff": float(role.cutoff)}
            for role in plan.roles
        ],
        local_capacity=local,
        ingress=ingress,
        prepare={
            "model_seconds": model_seconds,
            "plan_seconds": plan_seconds,
            "runtime_seconds": runtime_seconds,
            "ingress_seconds": ingress_seconds,
        },
        topology={
            "maximum_search_load": int(_scalar(evidence.maximum_search_load)),
            "maximum_edge_count": int(_scalar(evidence.maximum_edge_count)),
            "maximum_degree": int(_scalar(evidence.maximum_degree)),
            "maximum_halo_load": int(_scalar(evidence.maximum_halo_load)),
            "halo_columns": None
            if halo is None
            else int(_scalar(halo.evidence.halo_columns)),
            "active_edges": int(np.sum(addressable_rows(state.topologies[0].valid))),
            "edges_counted_on": "this process's addressable owners",
        },
        hardware_process_evidence={
            **runtime_record(ExecutionRuntime.current()),
            "owner_lanes": owners if plan.ownership.lane_reference else None,
            "selected_devices": [
                {
                    "id": int(device.id),
                    "process_index": int(device.process_index),
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                }
                for device in plan.ownership.execution_group.devices
            ],
            "owner_processes": list(plan.ownership.owner_processes),
        },
        logical_state_bytes=logical_array_bytes(state),
        memory={"baseline": baseline, "after_preparation": _memory()},
    )

    artifacts: dict[str, Any] = {}

    def evaluate() -> dict[str, Any]:
        compiled, first, phases = _phases(
            lambda current: runtime.evaluate(current), state, arguments.repeats
        )
        artifacts.update(compiled=compiled, first=first)
        return {
            **phases,
            "communication": _communication(
                runtime, state, lambda current: runtime.evaluate(current)
            ),
            "energy": float(_scalar(first.energy)),
            "global_energy": float(_scalar(first.global_energy)),
            "reasons": int(_scalar(first.reasons)),
            "successful": bool(_scalar(first.successful)),
        }

    checkpoint_limits = _checkpoint_limits(arguments)
    # Publication has no numerical-evaluation dependency and can be measured
    # separately after a checkpoint resource refusal.
    evaluation_required = any(workload != "publication" for workload in workloads)
    if evaluation_required:
        row["workloads"]["evaluate"] = _guarded(evaluate)
    operations: dict[str, Callable[[], dict[str, Any]]] = {
        "rebuild": lambda: _rebuild(
            runtime,
            state,
            artifacts["compiled"],
            cell,
            positions,
            owners,
            arguments.repeats,
        ),
        "md-step": lambda: _md_step(runtime, state, atoms, arguments.repeats),
        "training-step": (
            lambda: _training_step(
                runtime, state, artifacts["first"], case.model, arguments.repeats
            )
        )
        if case.model is not None
        else lambda: {
            "status": "not-applicable",
            "reason": "a terms-only runtime has no trainable layered model",
        },
        "publication": lambda: _publication(
            runtime,
            state,
            arguments.checkpoint_directory,
            f"{index}",
            checkpoint_limits,
        ),
    }
    # The admission policy is retained even when publication is refused.
    retained = {"publication": {"archive_limits": _limits_payload(checkpoint_limits)}}
    for workload in workloads:
        if workload == "evaluate":
            continue
        if evaluation_required and not artifacts:
            row["workloads"][workload] = {
                **retained.get(workload, {}),
                "status": "failed",
                "error": "not executed: the evaluation of this row failed",
            }
            continue
        row["workloads"][workload] = {
            **retained.get(workload, {}),
            **_guarded(operations[workload]),
        }
    row["successful"] = all(
        result.get("status") == "not-applicable"
        or (result.get("status") == "ok" and result.get("successful", False))
        for result in row["workloads"].values()
    )
    return json_safe(row)


# ------------------------------------------------------------ reciprocal rows


def _mesh_extent(length: float, owners: int, spacing: float, /) -> int:
    extent = max(8, math.ceil(length / spacing))
    return owners * math.ceil(extent / owners)


# One row owns its preparation, FFT/force phases, reference and retained refusal.
# Keep that chronological measurement record together so cache/profile boundaries
# cannot drift between helpers; the native owners implement all numerical kernels.
def reciprocal_row(
    index: int,
    case: ReciprocalCase,
    owners: int,
    atoms: int,
    execution: OwnerExecution,
    arguments: argparse.Namespace,
    /,
    *,
    pencil_grid: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Standalone distributed reciprocal term: patches, packets and FFT phases.

    ``pencil_grid`` selects the canonical pencil transfer layout (owner ``r``
    at pencil coordinate ``(r % px, r // px)``); otherwise particle meshes
    use x-slabs. PME rows of one process also record parity of energy,
    forces and the cell gradient against the unchanged single-device term.
    """
    row: dict[str, Any] = {
        "row": index,
        "kind": "reciprocal",
        "case": case,
        "owners": owners,
        "atoms": atoms,
        "execution": execution,
        "schedule": "slab" if pencil_grid is None else "pencil",
        "pencil_grid": None if pencil_grid is None else list(pencil_grid),
        "process_count": jax.process_count(),
    }
    try:
        from phydrax.atomistic import AtomisticSystemPlan
        from phydrax.atomistic._electrostatics import (
            EwaldReferencePotential,
            ParticleMeshEwaldPotential,
        )

        cell, positions, _ = structure(atoms, (1,), atoms)
        charges = np.random.default_rng(atoms).normal(size=atoms)
        charges -= charges.mean()
        periodic = PeriodicCell(cell)
        system = AtomisticSystemPlan(
            np.arange(atoms, dtype=np.int64),
            np.ones((atoms,), dtype=np.int64),
            np.ones((atoms,), dtype=np.float64),
            AtomisticUnitSystem.reduced(),
            atom_type_ids=np.zeros((atoms,), dtype=np.int64),
            charges=charges,
            cell=periodic,
        ).prepare()
        # Real-space cutoff inside the minimum-image sphere of the skew cell.
        real_cutoff = 0.4 * _minimum_width(cell)
        alpha = 3.0 / real_cutoff
        if case == "ewald":
            term = EwaldReferencePotential(
                alpha, real_cutoff, arguments.ewald_kmax
            ).prepare(system)
            strategy = "distributed-fft"
        else:
            spacing = arguments.mesh_spacing
            grid = tuple(
                _mesh_extent(float(np.linalg.norm(row_)), owners, spacing)
                for row_ in cell
            )
            term = ParticleMeshEwaldPotential(
                alpha, real_cutoff, grid, spline_degree=4
            ).prepare(system)
            strategy = (
                "replicated-mesh" if case == "pme-replicated-mesh" else "distributed-fft"
            )
        local = int(math.ceil(1.6 * atoms / owners)) + 4
        ownership = ownership_plan(owners, local, execution)
        # The partition must use the term's own cell (same cell identity).
        term_cell = system.cell
        if term_cell is None:
            raise ValueError("The reciprocal benchmark system must be periodic.")
        partition = FractionalOwnerPartition(term_cell, (owners, 1, 1))
        schedule_options: dict[str, Any] = (
            {}
            if pencil_grid is None
            else {"schedule": "pencil", "pencil_grid": pencil_grid}
        )
        prepared, prepare_seconds = measure_host(
            lambda: prepare_reciprocal_distribution(
                AtomisticReciprocalDistributionPlan(
                    term, reuse_reach=SKIN, strategy=strategy, **schedule_options
                ),
                ownership,
                partition,
            )
        )
        wrapped = np.mod(np.linalg.solve(cell.T, positions.T).T, 1.0)
        owner = np.asarray(partition.owners(jnp.asarray(wrapped)))
        slot = np.zeros_like(owner)
        counts = np.zeros((owners,), dtype=np.int64)
        for atom, value in enumerate(owner):
            slot[atom] = counts[value]
            counts[value] += 1
        if np.any(counts > local):
            raise ValueError(f"Owner atom counts {counts.tolist()} exceed {local} slots.")
        rows = owner * local + slot

        def blocked(values: np.ndarray) -> jax.Array:
            out = np.zeros((owners * local,) + values.shape[1:], dtype=values.dtype)
            out[rows] = values
            return ownership.place(out)

        mask = blocked(np.ones((atoms,), dtype=np.bool_))
        payload = ReciprocalSitePayload(values=blocked(charges))
        axis = ownership.axis_name
        owned = PartitionSpec(axis)
        vectors = jnp.asarray(cell)

        def region(
            x: jax.Array, sites: ReciprocalSitePayload, alive: jax.Array, h: jax.Array
        ) -> tuple[jax.Array, ...]:
            share, evidence = owner_reciprocal_energy(prepared, x, sites, alive, h, axis)
            return (
                share[None],
                evidence.successful[None],
                evidence.capacity_successful[None],
                evidence.scientific_successful[None],
                evidence.packets_fit[None],
                evidence.patch_covered[None],
                evidence.maximum_packet_load[None],
            )

        mapped = ownership.map(
            region,
            (owned, jax.tree.map(lambda _: owned, payload), owned, PartitionSpec()),
            (owned,) * 7,
        )

        def energy_forces(x: jax.Array) -> Any:
            def total(value: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
                outputs = mapped(value, payload, mask, vectors)
                return jnp.sum(outputs[0]), outputs[1:]

            (value, status), gradient = jax.value_and_grad(total, has_aux=True)(x)
            return value, -gradient, status

        _, (energy, _, status), phases = _phases(
            energy_forces, blocked(positions), arguments.repeats
        )
        report = prepared.report
        row.update(
            status="ok",
            evidence_scope=_scope(ownership),
            strategy=report.strategy,
            prepare_seconds=prepare_seconds,
            energy_forces=phases,
            energy=float(_scalar(energy)),
            report={
                "grid_shape": report.grid_shape,
                "patch_shape": report.patch_shape,
                "packet_planes": report.packet_planes,
                "packet_lines": report.packet_lines,
                "packet_bytes": report.packet_bytes,
                "patch_bytes": report.patch_bytes,
                "mode_chunk": report.mode_chunk,
                "mode_chunks": report.mode_chunks,
                "channels": report.channels,
                "schedule": report.schedule,
                "spectral_grid": report.spectral_grid,
                "owner_coordinates": report.owner_coordinates,
                "owner_devices": report.owner_devices,
                "physical_layout_id": report.physical_layout_id,
                "modal_layout_id": report.modal_layout_id,
                "physical_block_shape": report.physical_block_shape,
                "modal_block_shape": report.modal_block_shape,
                "collective_order": report.collective_order,
                "spectral_topology_id": report.spectral_topology_id,
                "spectral_plan_id": report.spectral_plan_id,
                "report_id": report.report_id,
            },
            evidence={
                name: addressable_rows(value)[0].item()
                for name, value in zip(
                    (
                        "successful",
                        "capacity_successful",
                        "scientific_successful",
                        "packets_fit",
                        "patch_covered",
                        "maximum_packet_load",
                    ),
                    status,
                    strict=True,
                )
            },
        )
        if ownership.lane_reference:
            row["collectives"] = None
            row["collectives_reason"] = (
                "lane-reference collectives are batched into local reductions"
            )
        else:
            row["collectives"] = collective_program(
                jax.make_jaxpr(energy_forces)(blocked(positions)).jaxpr
            )
        row["parity"] = _guarded(
            lambda: _pme_parity(
                case, term, mapped, payload, mask, blocked, rows, positions, charges, cell
            )
        )
        row["fft_round_trip"] = _guarded(
            lambda: _fft_round_trip(prepared, ownership, arguments.repeats)
        )
        row["successful"] = (
            bool(row["evidence"]["successful"])
            and row["fft_round_trip"].get("status") in ("ok", "not-applicable")
            and row["parity"].get("status") in ("ok", "not-applicable")
        )
    except Exception as error:
        row.update(failure(error))
        row["successful"] = False
    return json_safe(row)


# Relative to max(1, largest reference magnitude) of each quantity.
_PARITY_TOLERANCE = {"energy": 1e-10, "forces": 1e-9, "cell_gradient": 1e-9}


def _pme_parity(
    case: ReciprocalCase,
    term: Any,
    mapped: Callable[..., Any],
    payload: ReciprocalSitePayload,
    mask: jax.Array,
    blocked: Callable[[np.ndarray], jax.Array],
    rows: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray,
    /,
) -> dict[str, Any]:
    """Energy, force and cell-gradient parity with the single-device PME term.

    The reference is the canonical prepared mesh, influence and self energy
    of the SAME term on one device (tin-foil, neutral charges).
    """
    if case == "ewald":
        return {"status": "not-applicable", "reason": "explicit-mode Ewald row"}
    if jax.process_count() != 1:
        return {
            "status": "not-applicable",
            "reason": "parity needs the global rows; no cross-process gather is made",
        }
    from phydrax.atomistic._electrostatics import _coulomb_influence, ewald_self_energy

    settings = term.plan.settings
    constant = term.system.plan.units.coulomb_constant
    values = jnp.asarray(charges)

    def reference(x: jax.Array, h: jax.Array) -> jax.Array:
        geometry = term.mesh.geometry(h)
        routes = term.mesh.build(x, geometry)
        influence = _coulomb_influence(geometry, settings.alpha, constant)
        reciprocal = term.mesh.quadratic_energy(
            routes, values[:, None], geometry, influence, zero_mode="exclude"
        )
        return reciprocal.value + ewald_self_energy(settings.alpha, constant, values)

    def distributed(x: jax.Array, h: jax.Array) -> jax.Array:
        return jnp.sum(mapped(x, payload, mask, h)[0])

    vectors = jnp.asarray(cell)
    x = blocked(positions)
    energy, (gradient, cell_gradient) = jax.value_and_grad(distributed, argnums=(0, 1))(
        x, vectors
    )
    expected, (expected_gradient, expected_cell) = jax.value_and_grad(
        reference, argnums=(0, 1)
    )(jnp.asarray(positions), vectors)

    def relative(value: Any, reference_value: Any) -> float:
        actual, wanted = np.asarray(value), np.asarray(reference_value)
        scale = max(float(np.max(np.abs(wanted))), 1.0)
        return float(np.max(np.abs(actual - wanted))) / scale

    errors = {
        "energy": relative(energy, expected),
        "forces": relative(np.asarray(gradient)[rows], expected_gradient),
        "cell_gradient": relative(cell_gradient, expected_cell),
    }
    passed = all(errors[name] <= _PARITY_TOLERANCE[name] for name in errors)
    return {
        "status": "ok" if passed else "failed",
        "reference": "single-device canonical PME term (mesh, influence, self energy)",
        "errors": errors,
        "tolerances": _PARITY_TOLERANCE,
        "reference_energy": float(expected),
    }


# Declared before execution: the reciprocal spectral plan transforms in float64,
# so both checks use the native complex128 distributed-FFT gate (1e-12) on the
# maximum absolute error relative to max(1, largest reference magnitude).
_FFT_TOLERANCE = {"forward": 1e-12, "round_trip": 1e-12}


def _fft_round_trip(prepared: Any, ownership: Any, repeats: int, /) -> dict[str, Any]:
    """Isolated forward/inverse transform on the prepared slab or pencil layout.

    Owner ``r`` receives the canonical physical block of its spectral
    coordinate; the modal blocks are checked against the global FFT (single
    process) and the round trip against the input. Each check passes only
    when its error is finite and within the declared tolerance of its scale.
    """
    spectral = prepared.spectral
    report = prepared.report
    if spectral is None:
        return {
            "status": "not-applicable",
            "reason": f"strategy {report.strategy!r} has no distributed FFT",
        }
    shape = tuple(report.grid_shape) + (report.channels,)
    mesh = np.random.default_rng(17).normal(size=shape)
    b0, b1 = report.physical_block_shape[0], report.physical_block_shape[1]
    coordinates = [
        (value[0], value[1] if len(value) > 1 else 0)
        for value in report.owner_coordinates
    ]
    physical = np.concatenate(
        [
            mesh[i0 * b0 : (i0 + 1) * b0, i1 * b1 : (i1 + 1) * b1]
            for i0, i1 in coordinates
        ],
        axis=0,
    )
    owned = PartitionSpec(ownership.axis_name)
    round_trip = ownership.map(
        lambda block: (
            spectral.forward_local(block),
            jnp.real(spectral.inverse_local(spectral.forward_local(block))),
        ),
        owned,
        (owned, owned),
    )
    _, (modal, result), phases = _phases(round_trip, ownership.place(physical), repeats)
    record: dict[str, Any] = {
        **phases,
        "transform": f"forward_local then inverse_local ({report.schedule} schedule)",
        "tolerances": _FFT_TOLERANCE,
        "error_convention": (
            "maximum absolute error <= tolerance * max(1, largest reference magnitude)"
        ),
    }
    if jax.process_count() != 1:
        record.update(
            status="not-applicable",
            reason="the global FFT reference needs every block; no cross-process "
            "gather is made",
            maximum_forward_error=None,
            maximum_round_trip_error=None,
        )
        return record
    expected = np.fft.fftn(mesh, axes=(0, 1, 2))
    modal_axis = 2 if report.schedule == "pencil" else 1
    rows = report.modal_block_shape[modal_axis]
    blocks = np.asarray(modal).reshape(
        (report.owner_count,) + tuple(report.modal_block_shape)
    )
    returned = np.asarray(result)
    # np.max propagates NaN; Python max over floats would not.
    forward_error = float(
        np.max(
            [
                np.max(
                    np.abs(
                        blocks[owner]
                        - np.take(
                            expected,
                            range(owner * rows, (owner + 1) * rows),
                            axis=modal_axis,
                        )
                    )
                )
                for owner in range(report.owner_count)
            ]
        )
    )
    round_trip_error = float(np.max(np.abs(returned - physical)))
    scales = {
        "forward": max(1.0, float(np.max(np.abs(expected)))),
        "round_trip": max(1.0, float(np.max(np.abs(physical)))),
    }
    errors = {"forward": forward_error, "round_trip": round_trip_error}
    passed = {
        name: math.isfinite(errors[name])
        and errors[name] <= _FFT_TOLERANCE[name] * scales[name]
        for name in errors
    }
    outputs_finite = bool(np.all(np.isfinite(blocks)) and np.all(np.isfinite(returned)))
    record.update(
        status="ok" if outputs_finite and all(passed.values()) else "failed",
        maximum_forward_error=forward_error,
        maximum_round_trip_error=round_trip_error,
        scales=scales,
        outputs_finite=outputs_finite,
        passed=passed,
    )
    return record


# ------------------------------------------------------------ driver


def _rows(arguments: argparse.Namespace, /) -> list[dict[str, Any]]:
    execution = parse(arguments.owners, OwnerExecution, "owners")
    rows: list[dict[str, Any]] = []
    for campaign in () if arguments.reciprocal_only else arguments.campaigns:
        for family in arguments.families:
            for strategy in arguments.strategies:
                for owners in arguments.owner_counts:
                    atoms = (
                        arguments.total_atoms
                        if campaign == "strong"
                        else arguments.atoms_per_owner * owners
                    )
                    rows.append(
                        domain_row(
                            len(rows),
                            campaign,
                            family,
                            strategy,
                            owners,
                            atoms,
                            execution,
                            arguments.workloads,
                            arguments,
                        )
                    )
    for case in arguments.reciprocal:
        for owners in arguments.owner_counts:
            rows.append(
                reciprocal_row(
                    len(rows),
                    case,
                    owners,
                    arguments.reciprocal_atoms,
                    execution,
                    arguments,
                )
            )
    for grid in arguments.pencil_grids:
        px, py = _pencil_grid(grid)
        rows.append(
            reciprocal_row(
                len(rows),
                "pme",
                px * py,
                arguments.reciprocal_atoms,
                execution,
                arguments,
                pencil_grid=(px, py),
            )
        )
    return rows


def _pencil_grid(text: str, /) -> tuple[int, int]:
    """``"PXxPY"`` owner grid of a pencil reciprocal row."""
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
        raise ValueError(f"pencil grid {text!r} must be PXxPY with positive integers.")
    return int(parts[0]), int(parts[1])


def _configuration(arguments: argparse.Namespace, /) -> dict[str, Any]:
    return {
        "owners": arguments.owners,
        "owner_counts": list(arguments.owner_counts),
        "campaigns": list(arguments.campaigns),
        "families": list(arguments.families),
        "strategies": list(arguments.strategies),
        "workloads": list(arguments.workloads),
        "reciprocal": list(arguments.reciprocal),
        "pencil_grids": list(arguments.pencil_grids),
        "reciprocal_only": arguments.reciprocal_only,
        "total_atoms": arguments.total_atoms,
        "atoms_per_owner": arguments.atoms_per_owner,
        "reciprocal_atoms": arguments.reciprocal_atoms,
        "layers": arguments.layers,
        "repeats": arguments.repeats,
        "skin": SKIN,
        "density": DENSITY,
        "message_capacity_bytes": arguments.message_capacity_bytes,
        "replicated_memory_bytes": arguments.replicated_memory_bytes,
        "mesh_spacing": arguments.mesh_spacing,
        "ewald_kmax": arguments.ewald_kmax,
        "local_processes": arguments.local_processes,
        "devices_per_process": arguments.devices_per_process,
        "platform": arguments.platform,
        "checkpoint_members": arguments.checkpoint_members,
        "checkpoint_inventory_bytes": arguments.checkpoint_inventory_bytes,
    }


def _record(
    arguments: argparse.Namespace, rows: list[dict[str, Any]], /
) -> dict[str, Any]:
    configuration = _configuration(arguments)
    record: dict[str, Any] = {
        "configuration": configuration,
        "rows": rows,
        "passed": all(row["successful"] for row in rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in get_args(RowStatus)
        },
        "environment": capture_environment().to_dict(),
        "host_load": {
            "load_average_1_5_15": list(os.getloadavg()),
            "logical_cpus": os.cpu_count(),
        },
        "evidence_scope": sorted(
            {row["evidence_scope"] for row in rows if "evidence_scope" in row}
        ),
        "process_index": jax.process_index(),
        "memory_scope": "process-local samples; allocator peaks are since process start",
    }
    identity = capture_benchmark_identity(PROJECT_ROOT, Path(__file__), tuple(record))
    record["identity"] = identity.to_dict()
    record["workload_id"] = canonical_fingerprint(
        {"config": configuration, "driver": Path(__file__).name}
    )
    return json_safe(record)


def _launch(arguments: argparse.Namespace, argv: Sequence[str], /) -> dict[str, Any]:
    """Relaunch this driver as coupled workers and assemble every rank record."""
    command = [sys.executable, "-m", "benchmarks.distributed_atomistic_scaling", *argv]
    command += ["--worker", "--checkpoint-directory", str(arguments.checkpoint_directory)]
    error: LocalProcessLaunchError | None = None
    try:
        results = launch_local_processes(
            command,
            LocalProcessLaunchPlan(
                arguments.local_processes,
                f"127.0.0.1:{free_loopback_port()}",
                devices_per_process=arguments.devices_per_process,
            ),
            environment=launch_environment(
                arguments.platform, arguments.devices_per_process
            ),
            cwd=str(PROJECT_ROOT),
            timeout_seconds=arguments.timeout_seconds,
            max_output_bytes=64 * 1024 * 1024,
        )
    except LocalProcessLaunchError as caught:
        error = caught
        results = caught.results
    ranks: list[dict[str, Any] | None] = []
    for result in sorted(results, key=lambda item: item.process_id):
        lines = [
            line[len(RESULT_PREFIX) :]
            for line in result.stdout.decode("utf-8", errors="replace").splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        try:
            ranks.append(json.loads(lines[-1]) if lines else None)
        except json.JSONDecodeError:
            ranks.append(None)
    leader = ranks[0] if ranks and ranks[0] is not None else None
    record = dict(leader) if leader is not None else {"rows": [], "passed": False}
    record["ranks"] = ranks
    record["launch"] = {
        "error": None if error is None else str(error)[:8000],
        "timed_out": False if error is None else bool(error.timed_out),
        "returncodes": [result.returncode for result in results],
        "stderr_tails": [
            result.stderr[-4000:].decode("utf-8", errors="replace") for result in results
        ],
    }
    record["passed"] = (
        error is None
        and all(rank is not None and rank["passed"] for rank in ranks)
        and len(ranks) == arguments.local_processes
    )
    return record


def main(argv: Sequence[str] | None = None) -> None:
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners", default="lanes", choices=get_args(OwnerExecution))
    parser.add_argument("--owner-counts", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--campaigns",
        nargs="+",
        choices=get_args(ScalingCampaign),
        default=["strong", "weak"],
    )
    parser.add_argument(
        "--families", nargs="+", choices=get_args(Family), default=["mace"]
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=get_args(Strategy), default=["halo"]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=get_args(Workload),
        default=["evaluate", "rebuild"],
    )
    parser.add_argument(
        "--reciprocal", nargs="*", choices=get_args(ReciprocalCase), default=[]
    )
    parser.add_argument(
        "--pencil-grids",
        nargs="*",
        default=[],
        metavar="PXxPY",
        help="distributed-FFT PME rows on the canonical pencil layout (owners = PX*PY)",
    )
    parser.add_argument(
        "--reciprocal-only",
        action="store_true",
        help="record only the reciprocal rows (no domain campaign rows)",
    )
    parser.add_argument("--total-atoms", type=int, default=96)
    parser.add_argument("--atoms-per-owner", type=int, default=48)
    parser.add_argument("--reciprocal-atoms", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--message-capacity-bytes", type=int, default=1 << 30)
    parser.add_argument("--replicated-memory-bytes", type=int, default=1 << 30)
    parser.add_argument("--mesh-spacing", type=float, default=1.0)
    parser.add_argument("--ewald-kmax", type=int, default=5)
    parser.add_argument("--local-processes", type=int, default=1)
    parser.add_argument("--devices-per-process", type=int, default=1)
    parser.add_argument("--platform", choices=get_args(Platform), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument(
        "--checkpoint-members",
        type=int,
        default=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_members,
        help="Explicit archive member budget for checkpoint publication and restore.",
    )
    parser.add_argument(
        "--checkpoint-inventory-bytes",
        type=int,
        default=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes,
        help=(
            "Explicit checkpoint shard-inventory byte budget (archive manifest bytes) "
            "for publication and restore."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "distributed_atomistic_scaling.json",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(arguments_list)
    if min(arguments.layers, arguments.repeats, arguments.local_processes) < 1:
        parser.error("layers, repeats and local-processes must be positive")
    if min(arguments.owner_counts) < 1 or arguments.devices_per_process < 1:
        parser.error("owner counts and devices-per-process must be positive")
    if arguments.checkpoint_members < 1:
        parser.error("checkpoint-members must be positive")
    if arguments.checkpoint_inventory_bytes < 1:
        parser.error("checkpoint-inventory-bytes must be positive")
    try:
        pencil_grids = [_pencil_grid(grid) for grid in arguments.pencil_grids]
    except ValueError as error:
        parser.error(str(error))
    if arguments.reciprocal_only and not (arguments.reciprocal or pencil_grids):
        parser.error("--reciprocal-only needs --reciprocal cases or --pencil-grids")
    if arguments.local_processes > 1 and not arguments.worker:
        global_devices = arguments.local_processes * arguments.devices_per_process
        if arguments.owners != "devices" or set(arguments.owner_counts) != {
            global_devices
        }:
            parser.error(
                "multi-process rows need --owners devices and --owner-counts "
                f"{global_devices} (one owner per device of every process)"
            )
        if any(px * py != global_devices for px, py in pencil_grids):
            parser.error(
                f"multi-process pencil grids must factor all {global_devices} devices"
            )
    with tempfile.TemporaryDirectory(prefix="distributed-atomistic-scaling-") as scratch:
        if arguments.checkpoint_directory is None:
            arguments.checkpoint_directory = Path(scratch)
        if arguments.local_processes > 1 and not arguments.worker:
            record = _launch(arguments, arguments_list)
        else:
            # Domain ownership requires 64-bit arrays (launcher workers set it
            # through their rank environment as well).
            pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
            record = _record(arguments, _rows(arguments))
    if arguments.worker:
        print(RESULT_PREFIX + json.dumps(record, allow_nan=False), flush=True)
        if not record["passed"]:
            raise SystemExit(1)
        return
    launcher = arguments.local_processes > 1
    # The launcher parent never initializes a JAX backend of its own.
    if launcher or jax.process_index() == 0:
        write_json_atomic(arguments.output, record)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

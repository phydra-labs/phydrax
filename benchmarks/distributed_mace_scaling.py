#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated owner-local MACE scaling: fixed total and fixed per owner.

Two campaigns vary the owner count of a periodic triclinic random crystal at
fixed number density: ``fixed-total`` keeps the atom count constant (strong
scaling), ``fixed-per-owner`` keeps the atoms per owner constant (weak
scaling). Every row separately measures the owner-local topology epoch
(alias exchange, owner-dense candidate filtering, halo preparation, and
streamed-relation preparation), lowering and compilation of the E/F/S
evaluation, its first execution, warm repeats, and one migration plus rebuild
transaction. Actual alias loads, edge counts, halo loads, deduplicated halo
columns, per-interaction halo message bytes, and migration counts are
retained; no scaling efficiency is inferred.

Owners execute on the first requested real/forced devices in runtime order
(``--owners devices``) or as lane-reference vmap lanes on one device
(``--owners lanes``). The requested device count need not divide the visible
allocation. Each row records the selected devices independently of owner count.
CPU devices and lanes are functional evidence only, never accelerator scaling
evidence. The model uses
native JAX coupling (``acceleration=None``). Compiler bytes are XLA estimates;
allocator counters and retained-array bytes are sampled separately per device.
The host-ingress campaign supports one process only and explicitly refuses
multi-process runs. Fixed-total rows use the same crystal across owner counts.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group, ExecutionRuntime
from phydrax._execution_sampling import sample_device_memory, sample_host_memory
from phydrax._fingerprint import canonical_fingerprint
from phydrax.atomistic import AtomisticScaleContract
from phydrax.atomistic._distributed import (
    evaluate_owner_local_atomistic,
    OwnerLocalAtomisticPlan,
    OwnerLocalAtomisticState,
    prepare_owner_local_atomistic,
    rebuild_owner_local_atomistic,
)
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.sparse._streamed import StreamedRelationPlan
from phydrax.typing import parse
from phydrax.units import ANGSTROM, ELECTRONVOLT


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OwnerExecution: TypeAlias = Literal["devices", "lanes"]
ScalingCampaign: TypeAlias = Literal["fixed-total", "fixed-per-owner"]

CUTOFF = 3.0
SKIN = 0.4
DENSITY = 0.08
SHEAR = np.array([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.15, -0.1, 1.0]])


def _model(layers: int) -> MACEPotential:
    architecture = MACEArchitecture(
        species=(1, 8),
        cutoff=CUTOFF,
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
    )
    return MACEPotential(
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        architecture,
        atomic_energies=np.array([-0.3, -1.1]),
        key=jax.random.key(0),
        acceleration=None,
    )


def _structure(atoms: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = (atoms / DENSITY) ** (1.0 / 3.0)
    cell = length * SHEAR
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, (atoms, 3)) @ cell
    species = np.asarray((1, 8), dtype=np.int32)[rng.integers(0, 2, atoms)]
    return cell, positions, species


def _plan(
    model: MACEPotential,
    cell: np.ndarray,
    owners: int,
    local: int,
    execution: OwnerExecution,
) -> OwnerLocalAtomisticPlan:
    if owners < 1:
        raise ValueError("owners must be positive.")
    if jax.process_count() != 1:
        raise ValueError("This MACE scaling plan supports single-process ingress only.")
    periodic = PeriodicCell(cell)
    address = MortonAddressPlan((0.0,) * 3, (1.0,) * 3, 10, periodic_axes=(True,) * 3)
    runtime = ExecutionRuntime.current()
    parent = runtime.root_group.spec
    match execution:
        case "devices":
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
            ownership = DistributedOwnershipPlan(address, group, local)
        case "lanes":
            ownership = DistributedOwnershipPlan(
                address,
                runtime.child_groups(parent.device_count)[0],
                local,
                owner_lanes=owners,
            )
    edges = local * 96
    return OwnerLocalAtomisticPlan(
        ownership,
        FractionalOwnerPartition(periodic, (owners, 1, 1)),
        periodic.image_stencil(CUTOFF + SKIN, maximum_image_count=4096),
        streaming=StreamedRelationPlan(
            receiver_tile=32,
            edge_tile=512,
            channel_capacity=model.configuration.message_payload_width(),
        ),
        cutoff=CUTOFF,
        skin=SKIN,
        alias_capacity=local * 27,
        edge_capacity=edges,
        halo_capacity=local,
        migration_capacity=local,
        message_capacity_bytes=1 << 30,
    )


def _first_message_bytes(
    plan: OwnerLocalAtomisticPlan, model: MACEPotential, state: OwnerLocalAtomisticState
) -> int:
    """Static per-owner halo message bytes of the first interaction's sources.

    Later interactions are bounded by ``message_capacity_bytes`` through the
    package's trace-time refusal; this row records the layer-0 payload only.
    """
    local = plan.ownership.local_capacity
    species = state.species[:local]
    mask = state.layout.active[:local]
    shapes = jax.eval_shape(
        lambda: model.prepare_sources(
            0, model.init_features(species, mask), species, mask
        )
    )
    rows = plan.ownership.owner_count * plan.halo_capacity
    return sum(
        rows * (leaf.size // leaf.shape[0]) * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(shapes)
    )


def _row(
    campaign: ScalingCampaign,
    owners: int,
    atoms: int,
    layers: int,
    execution: OwnerExecution,
    repeats: int,
) -> dict[str, Any]:
    if min(owners, atoms, layers, repeats) < 1:
        raise ValueError("owners, atoms, layers, and repeats must be positive.")
    if jax.process_count() != 1:
        raise ValueError(
            "This MACE scaling driver supports single-process owner devices or lanes; "
            "multi-process owner-local ingress is not qualified by this driver."
        )
    baseline_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    model = _model(layers)
    cell, positions, species_ids = _structure(atoms, seed=atoms)
    local = int(np.ceil(1.6 * atoms / owners)) + 4
    plan = _plan(model, cell, owners, local, execution)
    mask = jax.numpy.ones(species_ids.shape, dtype=np.bool_)
    species = np.asarray(model.species_indices(jax.numpy.asarray(species_ids), mask))
    velocities = np.random.default_rng(1).normal(scale=4.0, size=positions.shape)
    state, prepare_seconds = measure_synchronized(
        lambda: prepare_owner_local_atomistic(
            plan,
            model,
            positions,
            species,
            rng_key=jax.random.key(1),
            velocities=velocities,
        )
    )
    prepared_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    evidence = state.topology.evidence
    jitted = jax.jit(
        lambda potential, current: evaluate_owner_local_atomistic(
            plan, potential, current
        )
    )
    compiled, compile_timing = measure_lower_and_compile(
        lambda: jitted.lower(model, state), lambda lowered: lowered.compile()
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    compiled_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    first, first_seconds = measure_synchronized(lambda: compiled(model, state))
    _, warm = measure_repeated(
        lambda: compiled(model, state),
        warmup=1,
        repeats=repeats,
    )
    runtime_memory = {
        "host": sample_host_memory().to_payload(),
        "devices": [
            sample.to_payload() for sample in sample_device_memory(jax.local_devices())
        ],
    }
    displacement = 0.05 * velocities
    if owners > 1:
        # Host campaign preparation forces one actual ownership crossing;
        # random short drift alone can otherwise benchmark a zero-move epoch.
        initial_fractional = np.linalg.solve(cell.T, positions[0])
        proposed_fractional = np.linalg.solve(cell.T, positions[0] + displacement[0])
        old_owner = int(np.floor(initial_fractional[0] * owners))
        next_owner_center = ((old_owner + 1) % owners + 0.5) / owners
        displacement[0] += (next_owner_center - proposed_fractional[0]) * cell[0]
    moved = state.with_dynamics(
        state.positions + state.layout.distribute(jax.numpy.asarray(displacement)),
        state.velocities,
        step_index=1,
    )
    transition, rebuild_seconds = measure_synchronized(
        lambda: rebuild_owner_local_atomistic(plan, model, moved)
    )
    after_migration, after_migration_seconds = measure_synchronized(
        lambda: evaluate_owner_local_atomistic(plan, model, transition.state)
    )
    crossed = owners == 1 or int(transition.migration.migrated) > 0
    selected = plan.ownership.execution_group.devices
    platforms = sorted({device.platform for device in selected})
    scope = (
        "functional lane-reference execution on one device"
        if plan.ownership.lane_reference
        else (
            "functional CPU owner-device execution"
            if platforms == ["cpu"]
            else "physical accelerator owner-device execution"
        )
    )
    return {
        "campaign": campaign,
        "owners": owners,
        "atoms": atoms,
        "atoms_per_owner": atoms / owners,
        "layers": layers,
        "execution": execution,
        "acceleration": None,
        "evidence_scope": scope
        + "; no scaling-efficiency or multi-host qualification claim",
        "hardware_process_evidence": {
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "global_device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "owner_lanes": owners if plan.ownership.lane_reference else None,
            "selected_device_count": len(selected),
            "selected_devices": [
                {
                    "id": int(device.id),
                    "process_index": int(device.process_index),
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                }
                for device in selected
            ],
            "execution_group_id": plan.ownership.execution_group.spec.group_id,
            "owner_processes": list(plan.ownership.owner_processes),
        },
        "successful": (
            bool(first.successful)
            and bool(transition.committed)
            and bool(after_migration.successful)
            and crossed
        ),
        "prepare_seconds": prepare_seconds,
        "lowering_seconds": compile_timing.lowering_seconds,
        "compilation_seconds": compile_timing.compilation_seconds,
        "first_execution_seconds": first_seconds,
        "warm": warm.to_dict(),
        "rebuild_seconds": rebuild_seconds,
        "post_migration_execution_seconds": after_migration_seconds,
        "maximum_alias_load": int(evidence.maximum_alias_load),
        "maximum_edge_count": int(evidence.maximum_edge_count),
        "maximum_degree": int(evidence.maximum_degree),
        "maximum_halo_load": int(evidence.maximum_halo_load),
        "halo_columns": int(state.topology.halo.evidence.halo_columns),
        "active_edges": int(jax.numpy.sum(state.topology.edge_valid)),
        "first_interaction_halo_message_bytes": _first_message_bytes(plan, model, state),
        "migrated_atoms": int(transition.migration.migrated),
        "logical_state_bytes": logical_array_bytes(state),
        "compiler": asdict(compiler),
        "memory": {
            "baseline": baseline_memory,
            "after_preparation": prepared_memory,
            "after_compilation": compiled_memory,
            "after_warm_execution": runtime_memory,
            "after_rebuild": {
                "host": sample_host_memory().to_payload(),
                "devices": [
                    sample.to_payload()
                    for sample in sample_device_memory(jax.local_devices())
                ],
            },
            "scope": "process-local samples; allocator peaks are since process start",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners", default="lanes")
    parser.add_argument("--owner-counts", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--total-atoms", type=int, default=96)
    parser.add_argument("--atoms-per-owner", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "distributed_mace_scaling.json",
    )
    arguments = parser.parse_args()
    execution = parse(arguments.owners, OwnerExecution, "owners")
    rows = []
    for campaign in get_args(ScalingCampaign):
        for owners in arguments.owner_counts:
            atoms = (
                arguments.total_atoms
                if campaign == "fixed-total"
                else arguments.atoms_per_owner * owners
            )
            rows.append(
                _row(
                    campaign,
                    owners,
                    atoms,
                    arguments.layers,
                    execution,
                    arguments.repeats,
                )
            )
    configuration = {
        "owners": execution,
        "owner_counts": list(arguments.owner_counts),
        "total_atoms": arguments.total_atoms,
        "atoms_per_owner": arguments.atoms_per_owner,
        "layers": arguments.layers,
        "repeats": arguments.repeats,
        "cutoff": CUTOFF,
        "skin": SKIN,
        "density": DENSITY,
    }
    record: dict[str, Any] = {
        "configuration": configuration,
        "rows": rows,
        "passed": all(row["successful"] for row in rows),
        "environment": capture_environment().to_dict(),
        "host_load": {
            "load_average_1_5_15": (
                None if sys.platform == "win32" else list(os.getloadavg())
            ),
            "load_average_unavailable_reason": (
                "Windows does not provide os.getloadavg"
                if sys.platform == "win32"
                else None
            ),
            "logical_cpus": os.cpu_count(),
        },
        "evidence_scope": sorted({row["evidence_scope"] for row in rows}),
    }
    identity = capture_benchmark_identity(PROJECT_ROOT, Path(__file__), tuple(record))
    record["identity"] = identity.to_dict()
    record["workload_id"] = canonical_fingerprint(
        {"config": configuration, "driver": Path(__file__).name}
    )
    write_json_atomic(arguments.output, record)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

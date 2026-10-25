#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real multi-process smoke scenarios of distributed atomistic domain execution.

Every scenario launches coupled local JAX processes through
`phydrax.execution.launch_local_processes` with a finite whole-job deadline.
Workers bootstrap from the launcher's rank environment, bind the complete root
`ExecutionGroup` (one owner per device of every process) and execute:

``parity``
    Process-local ingress (each process supplies only its own atom rows;
    nothing is gathered on a host) of a two-layer native MACE domain (halo or
    graph-partition strategy), then E/F/strain evaluation compared with the
    independent single-device global MACE oracle of the same model
    (`tests._support.atomistic_domain.global_reference`).
``transport`` / ``transport-kill``
    The stage-transport subgroup probe
    (`phydrax.atomistic._stage_transport.stage_transport_probe_rank`): child
    stage collectives, credit saturation, atomic transfers and a durable
    global commit; ``transport-kill`` SIGKILLs rank 1 after its shard and
    before the global commit.
``publish`` / ``restore`` / ``migrate``
    A migrated continuation is published per process and globally committed
    (`publish_domain_checkpoint`); ``restore`` reloads it bitwise on the SAME
    topology (`restore_domain_checkpoint`); ``migrate`` continues it on a
    CHANGED owner/process topology (`migrate_domain_checkpoint`) after the
    same-topology restore is refused.
``kill``
    Rank 1 SIGKILLs itself before publication while rank 0 publishes: the
    coupled job must fail within its deadline and no global commit may exist.
``domain-campaign`` / ``-kill`` / ``-restart`` / ``-publish`` / ``-migrate``
    A backpressured two-stage domain-physics campaign over process-subset
    child groups (`tools._atomistic_smoke_campaign`): uninterrupted reference,
    killed rank during a commit, bitwise same-topology restart, and
    changed-topology migration of a committed checkpoint.
``dp-domain`` / ``dp-domain-restart``
    Device data × domain training with both axes > 1, unequal and empty
    objective support, an independent single-device objective/gradient
    reference, shared kernel updates and a durable restart
    (`tools._atomistic_smoke_training`).
``family-parity`` / ``family-dynamics`` (``--family``)
    AIMNet2, UMA, MACE-LES and LJ/PME domain routes against independent
    single-device programs, plus dynamic-cell MTK trajectories, with native
    refusals of unsupported routes recorded (`tools._atomistic_smoke_families`).

``--platform cpu`` (default) forces ``--devices-per-process`` host CPU devices
per process with gloo CPU collectives: a functional oracle of the collective
program, never accelerator or multi-host evidence. ``--platform natural`` keeps
the visible accelerator platform and assigns disjoint local devices per rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args, Literal, TypeIs

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import config as pxconfig
from phydrax._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax.execution import (
    ExecutionRuntime,
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
    LocalProcessResult,
)
from phydrax.lifecycle._distributed_checkpoint import (
    IncompleteDistributedCheckpointError,
    load_distributed_checkpoint_commit,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from tools._accelerator_process import write_json_evidence
from tools._atomistic_smoke_campaign import (
    campaign_checks,
    CAMPAIGN_EXPECTED_FAILURE,
    CAMPAIGN_SCENARIOS,
    CampaignScenario,
    run_campaign_rank,
)
from tools._atomistic_smoke_families import (
    FAMILIES,
    Family,
    family_checks,
    FAMILY_SCENARIOS,
    FamilyScenario,
    run_family_rank,
    SCENARIO_FAMILIES,
)
from tools._atomistic_smoke_training import (
    run_training_rank,
    training_checks,
    TRAINING_SCENARIOS,
    TrainingScenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "ATOMISTIC_SMOKE_RESULT "

Scenario = Literal[
    "parity",
    "transport",
    "transport-kill",
    "publish",
    "restore",
    "migrate",
    "kill",
    "domain-campaign",
    "domain-campaign-kill",
    "domain-campaign-restart",
    "domain-campaign-publish",
    "domain-campaign-migrate",
    "dp-domain",
    "dp-domain-restart",
    "family-parity",
    "family-dynamics",
]
Strategy = Literal["halo", "graph-partition"]
Platform = Literal["cpu", "natural"]

EXPECTED_FAILURE: frozenset[str] = frozenset(
    {"transport-kill", "kill", *CAMPAIGN_EXPECTED_FAILURE}
)
DOMAIN_CHECKPOINT = "atomistic-smoke-publish"
KILL_CHECKPOINT = "atomistic-smoke-kill"
TRANSPORT_KILL_CHECKPOINT = "stage-transport-probe-kill"
# The domain scenarios include every resident topology and force-cache shard.
DOMAIN_CHECKPOINT_LIMITS = ArrayArchiveLimits(
    max_members=2048, max_manifest_bytes=8 << 20
)
ENERGY_RELATIVE_TOLERANCE = 1.0e-10
FORCE_ABSOLUTE_TOLERANCE = 1.0e-9
STRAIN_ABSOLUTE_TOLERANCE = 1.0e-9
MIGRATION_RELATIVE_TOLERANCE = 1.0e-10


# --------------------------------------------------------------- repositories


def _repository(
    root: Path, profile_id: str, /, *, maximum_metadata_bytes: int = 1 << 20
) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        profile_id,
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        root,
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=1 << 16,
            maximum_metadata_bytes=maximum_metadata_bytes,
        ),
    )


def domain_repository(directory: Path, /) -> POSIXArtifactRepository:
    """The shared POSIX repository of the domain-checkpoint scenarios."""
    return _repository(Path(directory) / "domain-repository", "posix.atomistic-smoke")


def transport_repository(directory: Path, /) -> POSIXArtifactRepository:
    """The repository the stage-transport probe publishes into."""
    return _repository(Path(directory) / "repository", "posix.stage-transport-probe")


def commit_present(
    repository: POSIXArtifactRepository,
    checkpoint_id: str,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> bool:
    """Whether a verified global commit exists (absence is the typed refusal)."""
    try:
        load_distributed_checkpoint_commit(repository, checkpoint_id, limits=limits)
    except IncompleteDistributedCheckpointError:
        return False
    return True


# ------------------------------------------------------------- rank helpers


def json_safe(value: Any) -> Any:
    """Keep failed (non-finite) numerical evidence in standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def addressable_rows(array: jax.Array, /) -> np.ndarray:
    """This process's view of a global array without any cross-process gather.

    Replicated arrays return their local replica; owner-blocked arrays return
    the addressable owner blocks in global row order (one copy per block).
    """
    if array.is_fully_replicated:
        return np.asarray(array.addressable_data(0))
    blocks: dict[int, np.ndarray] = {}
    for shard in array.addressable_shards:
        leading = shard.index[0]
        start = 0 if leading.start is None else int(leading.start)
        blocks.setdefault(start, np.asarray(shard.data))
    return np.concatenate([blocks[start] for start in sorted(blocks)], axis=0)


def state_digest(state: Any, /) -> dict[str, str]:
    """SHA-256 of this process's addressable continuation leaves."""
    leaves = {
        "positions": state.positions,
        "momenta": state.momenta,
        "masses": state.masses,
        "species": state.species,
        "image_counts": state.image_counts,
        "stable_ids": state.layout.stable_ids,
        "active": state.layout.active,
        "owner_epochs": state.layout.owner_epochs,
        "forces": state.force.forces,
        "cell_vectors": state.cell_vectors,
        "step_index": state.step_index,
        "time": state.time,
        "rng_key": jax.random.key_data(state.rng_key),
    }
    digests = {}
    for name, leaf in leaves.items():
        payload = np.ascontiguousarray(addressable_rows(leaf)).tobytes()
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def runtime_record(runtime: ExecutionRuntime, /) -> dict[str, Any]:
    """Runtime identity and observed process/host topology of this process."""
    hosts = sorted({host for _, host in runtime.inventory.process_host_ids})
    return {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "global_device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "platforms": sorted({device.platform for device in jax.devices()}),
        "device_kinds": sorted({device.device_kind for device in jax.devices()}),
        "execution_group_id": runtime.root_group.spec.group_id,
        "inventory_id": runtime.inventory.inventory_id,
        "process_host_ids": [list(item) for item in runtime.inventory.process_host_ids],
        # Linux kernel boot identities; an empty mapping is unknown, not one host.
        "observed_host_count": len(hosts) if hosts else None,
    }


def owner_address() -> Any:
    """The unit fractional box addressed by every domain ownership plan."""
    from phydrax.discretization.spatial import MortonAddressPlan

    return MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8, periodic_axes=(True, True, True)
    )


def root_ownership(runtime: ExecutionRuntime, /) -> Any:
    """One owner per device of every process of the root group."""
    from phydrax.discretization.spatial._distributed_relations import (
        DistributedOwnershipPlan,
    )
    from tests._support.atomistic_domain import LOCAL_CAPACITY

    return DistributedOwnershipPlan(owner_address(), runtime.root_group, LOCAL_CAPACITY)


def build_domain(ownership: Any, strategy: Strategy, /) -> tuple[Any, Any]:
    """Two-layer native MACE domain (halo or graph partition) on ``ownership``."""
    from phydrax.atomistic._domain_plan import (
        AtomisticDomainPlan,
        AtomisticDomainReductionPolicy,
    )
    from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
    from phydrax.atomistic._domain_state import AtomisticDomainStrategy
    from phydrax.atomistic._graph_partition import AtomisticGraphPartitionPlan
    from phydrax.atomistic._neighborhood_bundle import AtomisticCellListImageSearch
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.discretization.particle._distributed import FractionalOwnerPartition
    from tests._support.atomistic_domain import CELL, mace_model, role_plan

    model = mace_model(2)
    partition = FractionalOwnerPartition(
        PeriodicCell(CELL), (ownership.owner_count, 1, 1)
    )
    role = role_plan(model, CELL, accumulation="deterministic")
    kind = AtomisticDomainStrategy(strategy)
    graph = (
        AtomisticGraphPartitionPlan(
            ownership,
            partition,
            (role,),
            replicated_memory_bytes=1 << 22,
            receiver_edge_capacity=256,
            maximum_particles_per_cell=16,
            search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 22),
        )
        if kind is AtomisticDomainStrategy.GRAPH_PARTITION
        else None
    )
    plan = AtomisticDomainPlan(
        ownership,
        partition,
        (role,),
        migration_capacity=ownership.local_capacity,
        message_capacity_bytes=1 << 20,
        strategy=kind,
        reduction=AtomisticDomainReductionPolicy("deterministic"),
        graph_partition=graph,
    )
    return model, prepare_atomistic_domain(plan, model)


def smoke_structure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The six-atom self-image triclinic MACE oracle structure with momenta."""
    from tests._support.atomistic_domain import structure

    positions, species_ids = structure()
    momenta = np.random.default_rng(23).normal(scale=0.05, size=positions.shape)
    return positions, species_ids, momenta


def _displacement(positions: np.ndarray, /) -> np.ndarray:
    """Deterministic move forcing atom 0 across the x owner boundaries."""
    from tests._support.atomistic_domain import CELL

    displacement = np.random.default_rng(29).normal(scale=0.02, size=positions.shape)
    displacement[0] += 0.3 * CELL[0]
    return displacement


def process_local_ingress(
    runtime: Any,
    model: Any | None,
    positions: np.ndarray,
    species_ids: np.ndarray,
    momenta: np.ndarray,
    /,
    *,
    masses: np.ndarray | None = None,
    rng_key: jax.Array | None = None,
    **options: Any,
) -> Any:
    """Ingress only this process's atoms (index mod process count) as its rows.

    Per-atom inputs are global logical-order host values from which each
    process takes only its own rows; the runtime never gathers or broadcasts
    them. ``species_ids`` are atomic numbers mapped through ``model``, or the
    native species indices when ``model`` is ``None`` (a terms-only runtime).
    ``options`` are forwarded to ``ingress_process_local`` (for example
    ``conditioning``).
    """
    ownership = runtime.plan.ownership
    me = jax.process_index()
    owners_here = sum(1 for process in ownership.owner_processes if process == me)
    rows = owners_here * ownership.local_capacity
    count = positions.shape[0]
    mine = np.flatnonzero(np.arange(count) % jax.process_count() == me)
    if mine.size > rows:
        raise ValueError("This process's atoms exceed its addressable owner rows.")
    native = (
        np.asarray(species_ids, dtype=np.int32)
        if model is None
        else np.asarray(
            model.species_indices(
                jnp.asarray(species_ids), jnp.ones(species_ids.shape, dtype=jnp.bool_)
            )
        )
    )
    active = np.zeros((rows,), dtype=np.bool_)
    active[: mine.size] = True
    local_positions = np.zeros((rows, 3), dtype=np.float64)
    local_positions[: mine.size] = positions[mine]
    local_momenta = np.zeros((rows, 3), dtype=np.float64)
    local_momenta[: mine.size] = momenta[mine]
    local_species = np.zeros((rows,), dtype=np.int32)
    local_species[: mine.size] = native[mine]
    local_masses = np.ones((rows,), dtype=np.float64)
    if masses is not None:
        local_masses[: mine.size] = np.asarray(masses, dtype=np.float64)[mine]
    # Padding rows carry IDs disjoint from every atom and every other process.
    stable = count + me * rows + np.arange(rows, dtype=np.int64)
    stable[: mine.size] = mine
    logical = np.zeros((rows,), dtype=np.int32)
    logical[: mine.size] = mine
    atom_payload: dict[str, np.ndarray] = {}
    for prepared_term in runtime.terms:
        for name, values in sorted(prepared_term.term.atom_fields().items()):
            local = np.zeros((rows, *values.shape[1:]), dtype=values.dtype)
            local[: mine.size] = values[mine]
            atom_payload[name] = local
    return runtime.ingress_process_local(
        local_positions,
        local_species,
        rng_key=jax.random.key(7) if rng_key is None else rng_key,
        active=active,
        stable_ids=stable,
        logical_indices=logical,
        logical_count=count,
        momenta=local_momenta,
        masses=local_masses,
        atom_payload=atom_payload,
        **options,
    )


def oracle_parity(
    runtime: Any, model: Any, state: Any, positions: np.ndarray, /
) -> dict[str, Any]:
    """Compare this process's rows with the single-device global MACE oracle."""
    from jax.experimental import multihost_utils

    from tests._support.atomistic_domain import global_reference

    _, species_ids, _ = smoke_structure()
    evaluation = runtime.evaluate(state)
    successful = bool(addressable_rows(evaluation.successful))
    energy = float(addressable_rows(evaluation.energy))
    reference_energy, reference_forces, reference_strain = global_reference(
        model, positions, species_ids
    )
    reference_energy = float(reference_energy)
    ids = addressable_rows(state.layout.stable_ids)
    active = addressable_rows(state.layout.active)
    forces = addressable_rows(evaluation.forces)
    expected = np.asarray(reference_forces)[ids[active]]
    force_error = (
        float(np.max(np.abs(forces[active] - expected))) if expected.size else 0.0
    )
    strain = addressable_rows(evaluation.strain_derivative)
    strain_error = float(np.max(np.abs(strain - np.asarray(reference_strain))))
    counts = np.asarray(
        multihost_utils.process_allgather(np.asarray(active.sum(), np.int64), tiled=False)
    )
    energy_error = abs(energy - reference_energy)
    errors = []
    if not successful:
        errors.append("domain evaluation failed")
    if not energy_error <= ENERGY_RELATIVE_TOLERANCE * max(1.0, abs(reference_energy)):
        errors.append(f"energy differs from the single-device oracle by {energy_error}")
    if not force_error <= FORCE_ABSOLUTE_TOLERANCE:
        errors.append(f"forces differ from the single-device oracle by {force_error}")
    if not strain_error <= STRAIN_ABSOLUTE_TOLERANCE:
        errors.append(f"strain derivative differs from the oracle by {strain_error}")
    if int(counts.sum()) != positions.shape[0]:
        errors.append("owned atom rows do not partition the system")
    return {
        "successful": successful,
        "energy": energy,
        "reference_energy": reference_energy,
        "energy_error": energy_error,
        "force_error": force_error,
        "strain_error": strain_error,
        "owned_rows_by_process": counts.tolist(),
        "strategy": runtime.plan.strategy.value,
        "owner_count": runtime.plan.ownership.owner_count,
        "owner_processes": list(runtime.plan.ownership.owner_processes),
        "errors": errors,
    }


def _published_state(runtime: Any, model: Any) -> tuple[Any, np.ndarray, int]:
    """Ingress, displace, and commit one migrating rebuild transaction."""
    positions, species_ids, momenta = smoke_structure()
    state = process_local_ingress(runtime, model, positions, species_ids, momenta)
    displacement = _displacement(positions)

    @jax.jit
    def displaced(stored: jax.Array, ids: jax.Array, active: jax.Array) -> jax.Array:
        rows = jnp.asarray(displacement)[jnp.clip(ids, 0, positions.shape[0] - 1)]
        return stored + jnp.where(active[:, None], rows, 0.0)

    moved = runtime.with_dynamics(
        state,
        displaced(state.positions, state.layout.stable_ids, state.layout.active),
        state.momenta,
        step_index=1,
    )
    transition = runtime.rebuild(moved)
    if not bool(addressable_rows(transition.committed)):
        raise RuntimeError("The migrating rebuild transaction was refused.")
    return (
        transition.state,
        positions + displacement,
        int(addressable_rows(transition.migration.migrated)),
    )


# -------------------------------------------------------------- rank scenarios


def _campaign_scenario(scenario: str, /) -> TypeIs[CampaignScenario]:
    return scenario in CAMPAIGN_SCENARIOS


def _training_scenario(scenario: str, /) -> TypeIs[TrainingScenario]:
    return scenario in TRAINING_SCENARIOS


def _family_scenario(scenario: str, /) -> TypeIs[FamilyScenario]:
    return scenario in FAMILY_SCENARIOS


def _family(family: str | None, /) -> TypeIs[Family]:
    return family in FAMILIES


def run_rank(
    scenario: Scenario,
    directory: Path,
    strategy: Strategy,
    family: str | None = None,
) -> dict[str, Any]:
    """Execute one scenario as one rank of a coupled launch."""
    if _campaign_scenario(scenario):
        return run_campaign_rank(scenario, directory, strategy)
    if _training_scenario(scenario):
        return run_training_rank(scenario, directory, strategy)
    if _family_scenario(scenario):
        if not _family(family):
            raise ValueError(f"Scenario {scenario!r} requires one of {FAMILIES}.")
        return run_family_rank(scenario, directory, strategy, family)
    runtime = ExecutionRuntime.current()
    record: dict[str, Any] = {
        "scenario": scenario,
        "strategy": strategy,
        "runtime": runtime_record(runtime),
    }
    errors: list[str] = []
    if scenario in ("transport", "transport-kill"):
        from phydrax.atomistic._stage_transport import stage_transport_probe_rank

        probe = stage_transport_probe_rank(
            str(directory), "complete" if scenario == "transport" else "kill"
        )
        record["probe"] = probe
        if probe.get("restored_bitwise") is not True:
            errors.append("transport state did not restore bitwise")
        record["errors"] = errors
        record["passed"] = not errors
        return record
    model, domain = build_domain(root_ownership(runtime), strategy)
    positions, species_ids, momenta = smoke_structure()
    if scenario == "parity":
        state = process_local_ingress(domain, model, positions, species_ids, momenta)
        parity = oracle_parity(domain, model, state, positions)
        record["parity"] = parity
        errors.extend(parity["errors"])
    elif scenario == "kill":
        state = process_local_ingress(domain, model, positions, species_ids, momenta)
        if jax.process_index() == 1:
            print(RESULT_PREFIX + json.dumps({"rank": 1, "killing": True}), flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        from phydrax.atomistic._domain_checkpoint import publish_domain_checkpoint

        publish_domain_checkpoint(
            domain_repository(directory),
            domain,
            state,
            checkpoint_id=KILL_CHECKPOINT,
            writer_id=f"rank-{jax.process_index()}",
            limits=DOMAIN_CHECKPOINT_LIMITS,
        )
        errors.append("publication returned although a coupled rank was killed")
    elif scenario == "publish":
        from phydrax.atomistic._domain_checkpoint import publish_domain_checkpoint

        published, displaced, migrated = _published_state(domain, model)
        publication = publish_domain_checkpoint(
            domain_repository(directory),
            domain,
            published,
            checkpoint_id=DOMAIN_CHECKPOINT,
            writer_id=f"rank-{jax.process_index()}",
            limits=DOMAIN_CHECKPOINT_LIMITS,
        )
        parity = oracle_parity(domain, model, published, displaced)
        errors.extend(parity["errors"])
        if migrated < 1 and domain.plan.ownership.owner_count > 1:
            errors.append("the published continuation did not migrate any atom")
        record.update(
            parity=parity,
            migrated_atoms=migrated,
            manifest_id=publication.manifest.manifest_id,
            record_id=publication.record.record_id,
            execution_id=publication.record.execution_id,
            digest=state_digest(published),
        )
    elif scenario == "restore":
        from phydrax.atomistic._domain_checkpoint import restore_domain_checkpoint

        template = process_local_ingress(domain, model, positions, species_ids, momenta)
        restored = restore_domain_checkpoint(
            domain_repository(directory),
            domain,
            template,
            DOMAIN_CHECKPOINT,
            limits=DOMAIN_CHECKPOINT_LIMITS,
        )
        displaced = positions + _displacement(positions)
        parity = oracle_parity(domain, model, restored.state, displaced)
        errors.extend(parity["errors"])
        if restored.restart_class != "bitwise":
            errors.append(f"same-topology restart class is {restored.restart_class!r}")
        record.update(
            parity=parity,
            restart_class=restored.restart_class,
            digest=state_digest(restored.state),
        )
    elif scenario == "migrate":
        from phydrax.atomistic._domain_checkpoint import (
            AtomisticDomainCheckpointError,
            migrate_domain_checkpoint,
            restore_domain_checkpoint,
        )
        from phydrax.lifecycle._restart_topology import TopologyRestartPolicy

        repository = domain_repository(directory)
        template = process_local_ingress(domain, model, positions, species_ids, momenta)
        try:
            restore_domain_checkpoint(
                repository,
                domain,
                template,
                DOMAIN_CHECKPOINT,
                limits=DOMAIN_CHECKPOINT_LIMITS,
            )
        except AtomisticDomainCheckpointError as error:
            record["same_topology_restore_refused"] = str(error)
        else:
            errors.append("a changed topology was restored as a same-topology restart")
        migrated = migrate_domain_checkpoint(
            repository,
            domain,
            template,
            DOMAIN_CHECKPOINT,
            policy=TopologyRestartPolicy(
                allow_topology_change=True,
                allow_tolerance_restart=True,
                maximum_relative_tolerance=MIGRATION_RELATIVE_TOLERANCE,
            ),
            relative_tolerance=MIGRATION_RELATIVE_TOLERANCE,
            limits=DOMAIN_CHECKPOINT_LIMITS,
        )
        displaced = positions + _displacement(positions)
        parity = oracle_parity(domain, model, migrated.state, displaced)
        errors.extend(parity["errors"])
        evidence = migrated.migration
        if migrated.restart_class != "tolerance" or evidence is None:
            errors.append("changed-topology continuation lacks tolerance-class evidence")
        else:
            record["migration"] = {
                "source_execution_id": evidence.source_execution_id,
                "target_execution_id": evidence.target_execution_id,
                "source_owner_count": evidence.source_owner_count,
                "target_owner_count": evidence.target_owner_count,
                "source_process_count": evidence.source_process_count,
                "target_process_count": evidence.target_process_count,
                "restored_rows": evidence.restored_rows,
                "read_payload_bytes": evidence.read_payload_bytes,
                "rng_lineage_id": evidence.rng_lineage_id,
            }
        if int(addressable_rows(migrated.state.step_index)) != 1:
            errors.append("the continuation step index was not preserved")
        record.update(
            parity=parity,
            restart_class=migrated.restart_class,
            digest=state_digest(migrated.state),
        )
    else:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    record["errors"] = errors
    record["passed"] = not errors
    return record


# ------------------------------------------------------------ coupled launch


def free_loopback_port() -> int:
    """An ephemeral loopback port for the launcher's coordinator address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def launch_environment(platform: Platform, devices_per_process: int) -> dict[str, str]:
    """Rank environment: x64, explicit platform and gloo CPU collectives."""
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",) if platform == "cpu" else None,
            cpu_device_count=devices_per_process if platform == "cpu" else None,
            cpu_collectives_implementation="gloo",
        )
    )
    environment.update(
        PYTHONPATH=str(PROJECT_ROOT),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    return environment


def _terminal_record(result: LocalProcessResult) -> dict[str, Any] | None:
    lines = [
        line[len(RESULT_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if not lines:
        return None
    record = json.loads(lines[-1])
    return record if isinstance(record, dict) else None


def _checks(
    scenario: Scenario,
    ranks: list[dict[str, Any] | None],
    processes: int,
    launch: dict[str, Any],
    directory: Path,
) -> list[str]:
    if _training_scenario(scenario):
        return training_checks(scenario, ranks, processes, launch, directory)
    if _campaign_scenario(scenario) and scenario in CAMPAIGN_EXPECTED_FAILURE:
        return campaign_checks(scenario, ranks, processes, launch, directory)
    errors: list[str] = []
    if scenario in EXPECTED_FAILURE:
        if launch["error"] is None:
            errors.append("the coupled job survived a killed rank")
        if launch["timed_out"]:
            errors.append("the killed rank was detected only by the whole-job deadline")
        if launch["returncodes"][1:2] != [-signal.SIGKILL]:
            errors.append("rank 1 did not terminate by SIGKILL")
        checkpoint, repository = (
            (KILL_CHECKPOINT, domain_repository(directory))
            if scenario == "kill"
            else (TRANSPORT_KILL_CHECKPOINT, transport_repository(directory))
        )
        if commit_present(
            repository,
            checkpoint,
            limits=DOMAIN_CHECKPOINT_LIMITS
            if scenario == "kill"
            else DEFAULT_ARRAY_ARCHIVE_LIMITS,
        ):
            errors.append("a partial publication became a global commit")
        return errors
    if launch["error"] is not None:
        errors.append(f"coupled launch failed: {launch['error']}")
    if len(ranks) != processes or any(rank is None for rank in ranks):
        errors.append("missing terminal rank records")
        return errors
    complete = [rank for rank in ranks if rank is not None]
    for index, rank in enumerate(complete):
        if rank["runtime"]["process_index"] != index:
            errors.append(f"rank {index} reported another process index")
        if rank["runtime"]["process_count"] != processes:
            errors.append(f"rank {index} observed another process count")
        errors.extend(f"rank {index}: {error}" for error in rank["errors"])
    if len({rank["runtime"]["execution_group_id"] for rank in complete}) != 1:
        errors.append("ranks bound different execution groups")
    if scenario == "transport":
        probes = [rank["probe"] for rank in complete]
        if len({probe["manifest_id"] for probe in probes}) != 1:
            errors.append("ranks loaded different transport commits")
        if len({probe["transfer_digest"] for probe in probes}) != 1:
            errors.append("ranks disagree on the transfer digest")
        if len(probes[1]["received"]) != 3:
            errors.append("the consumer stage did not receive every packet")
        if not any(round_["parked"] > 0 for round_ in probes[0]["rounds"]):
            errors.append("credit saturation (parked upstream packets) was not exercised")
    if scenario == "publish":
        if len({rank["manifest_id"] for rank in complete}) != 1:
            errors.append("ranks committed different manifests")
        if not commit_present(
            domain_repository(directory),
            DOMAIN_CHECKPOINT,
            limits=DOMAIN_CHECKPOINT_LIMITS,
        ):
            errors.append("the global commit is not loadable")
    if _campaign_scenario(scenario):
        errors.extend(campaign_checks(scenario, ranks, processes, launch, directory))
    if _family_scenario(scenario):
        errors.extend(family_checks(scenario, ranks, processes, launch, directory))
    return errors


def run_scenario(
    scenario: Scenario,
    directory: Path,
    /,
    *,
    processes: int = 2,
    devices_per_process: int = 2,
    strategy: Strategy = "halo",
    platform: Platform = "cpu",
    timeout_seconds: float = 600.0,
    family: str | None = None,
) -> dict[str, Any]:
    """Launch one coupled scenario and return its complete, never-dropped record."""
    if scenario not in get_args(Scenario):
        raise ValueError(f"Unknown scenario {scenario!r}.")
    if processes < 2 or devices_per_process < 1:
        raise ValueError("Scenarios need at least two processes and one device each.")
    if scenario in EXPECTED_FAILURE and processes != 2:
        raise ValueError("Kill scenarios are defined on exactly two processes.")
    if _family_scenario(scenario):
        if family not in SCENARIO_FAMILIES[scenario]:
            raise ValueError(
                f"Scenario {scenario!r} requires a family in "
                f"{SCENARIO_FAMILIES[scenario]}."
            )
    elif family is not None:
        raise ValueError(f"Scenario {scenario!r} takes no model family.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        "-m",
        "tools.atomistic_distributed_smoke",
        "--worker",
        "--scenario",
        scenario,
        "--strategy",
        strategy,
        "--directory",
        str(directory),
    ) + (() if family is None else ("--family", family))
    started = time.monotonic()
    error: LocalProcessLaunchError | None = None
    try:
        results = launch_local_processes(
            command,
            LocalProcessLaunchPlan(
                processes,
                f"127.0.0.1:{free_loopback_port()}",
                devices_per_process=devices_per_process,
            ),
            environment=launch_environment(platform, devices_per_process),
            cwd=str(PROJECT_ROOT),
            timeout_seconds=timeout_seconds,
        )
    except LocalProcessLaunchError as caught:
        error = caught
        results = caught.results
    elapsed = time.monotonic() - started
    ordered = sorted(results, key=lambda item: item.process_id)
    ranks: list[dict[str, Any] | None] = []
    parse_errors: list[str] = []
    for result in ordered:
        try:
            ranks.append(_terminal_record(result))
        except (ValueError, TypeError) as caught:
            ranks.append(None)
            parse_errors.append(f"rank {result.process_id} terminal record: {caught}")
    launch = {
        "processes": processes,
        "devices_per_process": devices_per_process,
        "platform": platform,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed,
        "error": None if error is None else str(error),
        "reason": None if error is None else error.reason,
        "timed_out": False if error is None else bool(error.timed_out),
        "returncodes": [result.returncode for result in ordered],
        "stderr": [result.stderr.decode("utf-8", errors="replace") for result in ordered],
        "failed_worker_stdout": None
        if error is None and not parse_errors
        else [result.stdout.decode("utf-8", errors="replace") for result in ordered],
        "output_truncated": [result.output_truncated for result in ordered],
    }
    errors = parse_errors + _checks(scenario, ranks, processes, launch, directory)
    return json_safe(
        {
            "scenario": scenario,
            "strategy": strategy,
            "family": family,
            "expected_failure": scenario in EXPECTED_FAILURE,
            "domain_checkpoint_limits": {
                **asdict(DOMAIN_CHECKPOINT_LIMITS),
                "allowed_dtype_kinds": sorted(
                    DOMAIN_CHECKPOINT_LIMITS.allowed_dtype_kinds
                ),
            }
            if scenario in ("kill", "publish", "restore", "migrate")
            else None,
            "launch": launch,
            "ranks": ranks,
            "errors": errors,
            "passed": not errors,
            "evidence_scope": (
                "functional forced-CPU multi-process collective program; not accelerator, "
                "multi-host or pretrained-model evidence"
                if platform == "cpu"
                else "observed local devices of one host; not multi-host evidence"
            ),
        }
    )


def compare_digests(
    published: Mapping[str, Any], restored: Mapping[str, Any], /
) -> list[str]:
    """Per-rank bitwise comparison of the same-topology restore."""
    errors = []
    for index, (left, right) in enumerate(
        zip(published["ranks"], restored["ranks"], strict=True)
    ):
        if left is None or right is None:
            errors.append(f"rank {index} has no digest")
            continue
        digest = right["digest"]
        differing = sorted(
            name for name, value in left["digest"].items() if value != digest.get(name)
        )
        if differing:
            errors.append(f"rank {index} restored differently: {differing}")
    return errors


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=get_args(Scenario), required=True)
    parser.add_argument("--strategy", choices=get_args(Strategy), default="halo")
    parser.add_argument("--platform", choices=get_args(Platform), default="cpu")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--local-processes", type=int, default=2)
    parser.add_argument("--devices-per-process", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.worker:
        record = json_safe(
            run_rank(
                arguments.scenario,
                arguments.directory,
                arguments.strategy,
                arguments.family,
            )
        )
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        print(RESULT_PREFIX + encoded, flush=True)
        if not record["passed"]:
            raise SystemExit(1)
        return
    record = run_scenario(
        arguments.scenario,
        arguments.directory,
        processes=arguments.local_processes,
        devices_per_process=arguments.devices_per_process,
        strategy=arguments.strategy,
        platform=arguments.platform,
        timeout_seconds=arguments.timeout_seconds,
        family=arguments.family,
    )
    if arguments.output is not None:
        write_json_evidence(arguments.output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

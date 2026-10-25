#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Multi-process stage×domain campaign scenarios of the distributed smoke.

`tools.atomistic_distributed_smoke` launches the coupled ranks and dispatches
the scenarios of this module to `run_campaign_rank`; `campaign_checks` adds
the cross-rank and cross-launch checks. One campaign
(`phydrax.atomistic._stage_transport.AtomisticDomainCampaignPlan`) drives
four six-atom periodic two-layer MACE tasks through two stages bound to
explicit process-subset child groups of the root group:

* ``equilibrate`` (first half of the processes): BAOAB Langevin NVT, 2
  accepted steps per visit in one segment (one round per task);
* ``production`` (second half): velocity-Verlet NVE, 8 accepted steps per
  visit in segments of 2 (four rounds per task).

Every stage runs the native distributed domain runtime and dynamics on its own
child group only (halo or graph partition over one owner per device of its
processes); finished visits leave as the members' `AtomisticDomainPacket`
fragments over the root-group transport (inbox and outbox capacity 1, one
slot per lane pair) and production reports are published to a durable
directory sink that accepts new records only on every third round (explicit
downstream pressure). The slower production stage and the throttled sink park
finished residents upstream without new physics.

``domain-campaign``
    The genuinely uninterrupted reference: no checkpoint at all, run to
    consensus termination, verify every published configuration against the
    single-device global MACE oracle and record parking/refusal evidence.
``domain-campaign-kill``
    The same run; rank 1 SIGKILLs itself after publishing its shard of the
    second checkpoint. The coupled job must fail; that checkpoint must have no
    global commit while the first one stays committed.
``domain-campaign-restart``
    Same-topology restart from the last committed checkpoint of the killed
    run into its own sink: every output must equal the reference's payload
    bitwise (the sink refuses a different payload under a published key).
``domain-campaign-publish`` / ``domain-campaign-migrate``
    The run stops after its first commit; ``migrate`` continues it on a
    CHANGED topology (launch it with another ``--devices-per-process`` or
    process count) through `PreparedAtomisticDomainCampaign.migrate` and must
    reproduce the reference outputs within roundoff tolerances.

Forced host-CPU devices with gloo collectives (the launcher default) are a
functional oracle of the collective program only: never accelerator,
multi-host or pretrained-model evidence.
"""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._array_archive import ArrayArchiveLimits
from phydrax.atomistic._campaign import (
    AbstractAtomisticCampaignSink,
    AtomisticCampaignOutput,
    AtomisticCampaignTask,
    AtomisticDirectorySink,
    AtomisticOutputReceipt,
    AtomisticStageOutcome,
    AtomisticTaskPacket,
)
from phydrax.execution import ExecutionRuntime


if TYPE_CHECKING:
    from phydrax.atomistic._stage_transport import (
        AtomisticDomainCampaignPlan,
        AtomisticDomainCampaignRound,
    )
    from phydrax.lifecycle._repository import POSIXArtifactRepository
    from tools.atomistic_distributed_smoke import Strategy


CampaignScenario = Literal[
    "domain-campaign",
    "domain-campaign-kill",
    "domain-campaign-restart",
    "domain-campaign-publish",
    "domain-campaign-migrate",
]
CAMPAIGN_SCENARIOS: tuple[CampaignScenario, ...] = get_args(CampaignScenario)
CAMPAIGN_EXPECTED_FAILURE: frozenset[str] = frozenset({"domain-campaign-kill"})

CAMPAIGN_ID = "atomistic-smoke-stage-domain"
TASK_SEEDS = (4, 11, 17, 23)
ATOMS = 6
MASSES = {1: 1.008, 8: 15.999}
STEP_FEMTOSECONDS = 0.5
TEMPERATURE = 300.0
FRICTION = 0.05
ENTRY_BUDGET, ENTRY_SEGMENT = 2, 2
PRODUCTION_BUDGET, PRODUCTION_SEGMENT = 8, 2
CHECKPOINT_EVERY = 3
SINK_PERIOD = 3
MAXIMUM_ROUNDS = 240
KILL_AT_CHECKPOINT = 2
# Complete resident leaves include topology and force-cache shards of both stages.
CHECKPOINT_LIMITS = ArrayArchiveLimits(max_members=4096, max_manifest_bytes=1 << 24)
# Published configurations against the independent single-device oracle.
ORACLE_ENERGY_RELATIVE_TOLERANCE = 1.0e-10
ORACLE_FORCE_ABSOLUTE_TOLERANCE = 1.0e-9
# Changed-topology continuation (tolerance class, not bitwise).
MIGRATION_POSITION_TOLERANCE = 1.0e-8
MIGRATION_ENERGY_RELATIVE_TOLERANCE = 1.0e-9

REFERENCE_SINK = "campaign-reference-sink"
RESTART_SINK = "campaign-restart-sink"
MIGRATION_SINK = "campaign-migration-sink"
PREFIXES: dict[str, str] = {
    "domain-campaign": "campaign-reference",
    "domain-campaign-kill": "campaign-restart",
    "domain-campaign-restart": "campaign-restart",
    "domain-campaign-publish": "campaign-migration",
    "domain-campaign-migrate": "campaign-migration",
}
SINKS: dict[str, str] = {
    "domain-campaign": REFERENCE_SINK,
    "domain-campaign-kill": RESTART_SINK,
    "domain-campaign-restart": RESTART_SINK,
    "domain-campaign-publish": MIGRATION_SINK,
    "domain-campaign-migrate": MIGRATION_SINK,
}


class ThrottledDirectorySink(AbstractAtomisticCampaignSink):
    """Durable directory sink accepting new records only on every ``period``-th
    round (deterministic downstream pressure keyed by the transport round).

    A record already durable under its semantic key is always returned (and a
    different payload refused) by the underlying `AtomisticDirectorySink`.
    """

    __slots__ = ("_inner", "period", "sink_id")

    def __init__(self, root: Path, period: int, /) -> None:
        self._inner = AtomisticDirectorySink(root, sink_id="smoke-campaign-sink")
        self.sink_id = self._inner.sink_id
        self.period = int(period)

    def receipt(self, semantic_key: str, /) -> AtomisticOutputReceipt | None:
        return self._inner.receipt(semantic_key)

    def publish(
        self, record: AtomisticCampaignOutput, /, *, round_index: int
    ) -> AtomisticOutputReceipt | None:
        if self._inner.receipt(record.semantic_key) is None and (
            round_index % self.period != self.period - 1
        ):
            return None
        return self._inner.publish(record, round_index=round_index)


def campaign_repository(
    directory: Path, /, *, maximum_metadata_bytes: int = 1 << 20
) -> POSIXArtifactRepository:
    """The POSIX repository every campaign checkpoint commits into."""
    from tools.atomistic_distributed_smoke import _repository

    return _repository(
        Path(directory) / "campaign-repository",
        "posix.atomistic-smoke-campaign",
        maximum_metadata_bytes=maximum_metadata_bytes,
    )


def checkpoint_ids(scenario: str, epoch: int, /) -> tuple[str, ...]:
    """Every checkpoint id the scenario's cadence can produce in one epoch."""
    prefix = PREFIXES[scenario]
    return tuple(
        f"{prefix}-e{epoch:04d}-r{round_index:06d}"
        for round_index in range(CHECKPOINT_EVERY, MAXIMUM_ROUNDS + 1, CHECKPOINT_EVERY)
    )


def committed(repository: Any, checkpoint_id: str, /) -> bool:
    from tools.atomistic_distributed_smoke import commit_present

    return commit_present(repository, checkpoint_id, limits=CHECKPOINT_LIMITS)


def latest_commit(repository: Any, candidates: Sequence[str], /) -> str:
    for candidate in reversed(tuple(candidates)):
        if committed(repository, candidate):
            return candidate
    raise RuntimeError("No committed campaign checkpoint exists to continue from.")


def campaign_model() -> Any:
    from tests._support.atomistic_domain import mace_model

    return mace_model(2)


def campaign_tasks(model: Any, /) -> tuple[Any, ...]:
    """Four six-atom periodic tasks (distinct structures, momenta, ordinals)."""
    from phydrax.atomistic._stage_transport import AtomisticDomainCampaignTask
    from tests._support.atomistic_domain import structure

    tasks = []
    for ordinal, seed in enumerate(TASK_SEEDS):
        positions, numbers = structure(ATOMS, seed)
        native = np.asarray(
            model.species_indices(
                jnp.asarray(numbers), jnp.ones(numbers.shape, dtype=jnp.bool_)
            )
        )
        momenta = np.random.default_rng(100 + seed).normal(
            scale=0.05, size=positions.shape
        )
        tasks.append(
            AtomisticDomainCampaignTask(
                AtomisticCampaignTask(
                    task_id=f"task-{ordinal}",
                    family="mace-six-atom",
                    packet=AtomisticTaskPacket(
                        positions=positions,
                        momenta=momenta,
                        species=numbers,
                        atom_type_ids=native,
                    ),
                    material_id=f"smoke-material-{seed}",
                    ordinal=ordinal,
                ),
                np.asarray([MASSES[int(number)] for number in numbers], np.float64),
            )
        )
    return tuple(tasks)


def _stage_program(strategy: Strategy, integrator: Any, ensemble: Any, /) -> Any:
    """Member-side native program: MACE domain on the stage group + dynamics."""

    def build(group: Any) -> Any:
        from phydrax.atomistic._domain_dynamics import AtomisticDomainDynamicsPlan
        from phydrax.discretization.spatial._distributed_relations import (
            DistributedOwnershipPlan,
        )
        from tests._support.atomistic_domain import LOCAL_CAPACITY
        from tools.atomistic_distributed_smoke import build_domain, owner_address

        ownership = DistributedOwnershipPlan(owner_address(), group, LOCAL_CAPACITY)
        _, runtime = build_domain(ownership, strategy)
        return AtomisticDomainDynamicsPlan(runtime.plan, integrator, ensemble).prepare(
            runtime
        )

    return build


def campaign_plan(
    runtime: ExecutionRuntime, strategy: Strategy, /
) -> tuple[AtomisticDomainCampaignPlan, Any]:
    """The two-stage campaign over process-subset child groups of the root."""
    from phydrax.atomistic._domain_dynamics import AtomisticDomainEnsemblePlan
    from phydrax.atomistic._dynamics import VelocityVerletPlan
    from phydrax.atomistic._stage_transport import (
        AtomisticDomainCampaignPlan,
        AtomisticDomainStage,
        process_subset_group_spec,
    )
    from phydrax.atomistic._thermal import BAOABLangevinPlan
    from phydrax.atomistic._units import AtomisticUnitSystem

    root = runtime.root_group.spec
    processes = tuple(sorted(root.process_indices))
    # Process-subset stages: first half equilibrates, second half produces; a
    # single process (in-process tests) belongs to both stage groups.
    half = max(1, len(processes) // 2)
    entry_processes = processes[:half]
    production_processes = processes[half:] or processes
    units = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    model = campaign_model()
    stages = (
        AtomisticDomainStage(
            "equilibrate",
            process_subset_group_spec(root, entry_processes),
            _stage_program(
                strategy,
                BAOABLangevinPlan(STEP_FEMTOSECONDS, FRICTION),
                AtomisticDomainEnsemblePlan(
                    units, ensemble="nvt", atom_count=ATOMS, temperature=TEMPERATURE
                ),
            ),
            f"mace2-{strategy}-baoab-nvt-{STEP_FEMTOSECONDS}fs-{TEMPERATURE}K",
            ENTRY_BUDGET,
            ENTRY_SEGMENT,
        ),
        AtomisticDomainStage(
            "production",
            process_subset_group_spec(root, production_processes),
            _stage_program(
                strategy,
                VelocityVerletPlan(STEP_FEMTOSECONDS),
                AtomisticDomainEnsemblePlan(units, ensemble="nve", atom_count=ATOMS),
            ),
            f"mace2-{strategy}-velocity-verlet-nve-{STEP_FEMTOSECONDS}fs",
            PRODUCTION_BUDGET,
            PRODUCTION_SEGMENT,
        ),
    )
    plan = AtomisticDomainCampaignPlan(
        CAMPAIGN_ID,
        stages,
        campaign_tasks(model),
        rng_root=29,
        maximum_rounds=MAXIMUM_ROUNDS,
    )
    return plan, model


def _round_record(result: AtomisticDomainCampaignRound, /) -> dict[str, Any]:
    evidence = result.transport
    return {
        "round_index": evidence.round_index,
        "executed_steps": list(result.executed_steps),
        "admitted": list(result.admitted),
        "exported": list(result.exported),
        "parked_residents": list(result.parked),
        "offered": evidence.offered,
        "granted": evidence.granted,
        "committed": evidence.committed,
        "parked": evidence.parked,
        "refused": evidence.refused,
        "credits": list(evidence.credits),
        "receipts": len(evidence.receipts),
        "terminated": evidence.terminated,
        "transfer_digest": evidence.transfer_digest,
    }


def oracle_outputs(directory: Path, sink: str, model: Any, /) -> dict[str, Any]:
    """Durable outputs and their independent single-device MACE oracle errors."""
    from tests._support.atomistic_domain import global_reference

    records: dict[str, Any] = {}
    errors: list[str] = []
    for key, output in sorted(
        AtomisticDirectorySink(Path(directory) / sink).outputs().items()
    ):
        ports = output.ports
        record: dict[str, Any] = {
            "task_id": output.task_id,
            "stage": output.stage,
            "outcome": output.outcome.name,
            "payload_id": output.payload_id,
            "counters": dict(output.counters),
            "reasons": output.reasons,
        }
        if "energy" in ports:
            energy, forces, _ = global_reference(
                model, np.asarray(ports["positions"]), np.asarray(ports["species"])
            )
            energy_error = abs(float(ports["energy"]) - float(energy))
            force_error = float(
                np.max(np.abs(np.asarray(ports["forces"]) - np.asarray(forces)))
            )
            record.update(
                energy=float(ports["energy"]),
                oracle_energy=float(energy),
                energy_error=energy_error,
                force_error=force_error,
            )
            if not energy_error <= ORACLE_ENERGY_RELATIVE_TOLERANCE * max(
                1.0, abs(float(energy))
            ):
                errors.append(f"{key}: energy differs from the oracle by {energy_error}")
            if not force_error <= ORACLE_FORCE_ABSOLUTE_TOLERANCE:
                errors.append(f"{key}: forces differ from the oracle by {force_error}")
        records[key] = record
    return {"outputs": records, "errors": errors}


def run_campaign_rank(
    scenario: CampaignScenario, directory: Path, strategy: Strategy
) -> dict[str, Any]:
    """Execute one campaign scenario as one rank of a coupled launch."""
    from phydrax.atomistic._stage_transport import (
        AtomisticDomainCampaignCheckpointPolicy,
    )
    from tools.atomistic_distributed_smoke import RESULT_PREFIX, runtime_record

    runtime = ExecutionRuntime.current()
    me = jax.process_index()
    record: dict[str, Any] = {
        "scenario": scenario,
        "strategy": strategy,
        "runtime": runtime_record(runtime),
    }
    errors: list[str] = []
    plan, model = campaign_plan(runtime, strategy)
    campaign = plan.prepare(runtime.root_group)
    repository = campaign_repository(directory)
    sink = (
        ThrottledDirectorySink(Path(directory) / SINKS[scenario], SINK_PERIOD)
        if me == campaign.transport.sink_host
        else None
    )
    record["campaign"] = {
        "analysis_id": plan.analysis_id,
        "execution_id": campaign.execution_id,
        "stage_members": [sorted(stage.group.process_indices) for stage in plan.stages],
        "member_stages": [
            index
            for index, binding in enumerate(campaign.bindings)
            if binding is not None
        ],
        "stage_schemas": list(campaign.stage_schemas),
        "slot_bytes": campaign.transport.slot_bytes,
        "round_bytes": campaign.transport.round_bytes,
        "resident_bytes": campaign.resident_bytes,
        "owner_counts": [
            None
            if binding is None
            else binding.dynamics.runtime.plan.ownership.owner_count
            for binding in campaign.bindings
        ],
    }
    if scenario == "domain-campaign-restart":
        source = latest_commit(repository, checkpoint_ids("domain-campaign-kill", 0))
        state = campaign.restore(repository, source, limits=CHECKPOINT_LIMITS)
        record["restored_from"] = source
    elif scenario == "domain-campaign-migrate":
        source = latest_commit(repository, checkpoint_ids("domain-campaign-publish", 0))
        state, migration = campaign.migrate(repository, source, limits=CHECKPOINT_LIMITS)
        record["restored_from"] = source
        record["migration"] = {
            "source_execution_id": migration.source_execution_id,
            "target_execution_id": migration.target_execution_id,
            "source_process_count": migration.source_process_count,
            "target_process_count": migration.target_process_count,
            "source_round_index": migration.source_round_index,
            "source_transfer_digest": migration.source_transfer_digest,
            "residents": list(migration.residents),
            "reissued": list(migration.reissued),
            "read_bytes": migration.read_bytes,
        }
    else:
        state = campaign.start()
    published: list[str] = []

    def after_publish(checkpoint_id: str) -> None:
        published.append(checkpoint_id)
        if (
            scenario == "domain-campaign-kill"
            and me == 1
            and len(published) == KILL_AT_CHECKPOINT
        ):
            print(
                RESULT_PREFIX + json.dumps({"rank": 1, "killing": checkpoint_id}),
                flush=True,
            )
            os.kill(os.getpid(), signal.SIGKILL)

    final, rounds, publications = campaign.run(
        state,
        sink=sink,
        checkpoint=None
        if scenario == "domain-campaign"
        else AtomisticDomainCampaignCheckpointPolicy(
            repository,
            CHECKPOINT_EVERY,
            PREFIXES[scenario],
            f"rank-{me}",
            limits=CHECKPOINT_LIMITS,
        ),
        stop_after_checkpoints=1 if scenario == "domain-campaign-publish" else None,
        after_publish=after_publish,
    )
    record["rounds"] = [_round_record(result) for result in rounds]
    record["publications"] = [
        {
            "checkpoint_id": publication.checkpoint_id,
            "manifest_id": publication.manifest_id,
            "round_index": publication.round_index,
        }
        for publication in publications
    ]
    record["final"] = {
        "round_index": final.round_index,
        "terminated": final.terminated,
        "epoch": final.epoch,
        "cursor": final.cursor,
        "transfer_digest": final.transport.transfer_digest.tobytes().hex(),
    }
    if scenario == "domain-campaign-kill":
        errors.append("the campaign returned although a coupled rank was killed")
    elif scenario == "domain-campaign-publish":
        if len(publications) != 1:
            errors.append("the publishing run did not stop after its first commit")
    elif not final.terminated:
        errors.append("the campaign did not terminate by consensus")
    for index, binding in enumerate(campaign.bindings):
        for previous, result in zip(rounds, rounds[1:]):
            if binding is None and result.executed_steps[index] != 0:
                errors.append(f"a non-member executed stage {index} physics")
            # A resident parked at the start of a round consumes no physics.
            if (
                previous.parked[index]
                and result.parked[index]
                and result.executed_steps[index] != 0
            ):
                errors.append(f"a parked stage {index} resident consumed new physics")
    if (
        scenario != "domain-campaign-publish"
        and final.terminated
        and me == campaign.transport.sink_host
    ):
        oracle = oracle_outputs(Path(directory), SINKS[scenario], model)
        record["outputs"] = oracle["outputs"]
        errors.extend(oracle["errors"])
    record["errors"] = errors
    record["passed"] = not errors
    return record


def _outputs(directory: Path, sink: str, /) -> dict[str, AtomisticCampaignOutput]:
    return AtomisticDirectorySink(Path(directory) / sink).outputs()


# Keep the finite cross-launch acceptance ledger explicit and in diagnostic order.
# Each condition is a distinct failure obligation; shortening this checker must not
# collapse kill, publication, migration and exact-restart guarantees into one flag.
def campaign_checks(
    scenario: CampaignScenario,
    ranks: list[dict[str, Any] | None],
    processes: int,
    launch: Mapping[str, Any],
    directory: Path,
) -> list[str]:
    """Cross-rank and cross-launch checks of a campaign scenario."""
    errors: list[str] = []
    repository = campaign_repository(directory)
    if scenario == "domain-campaign-kill":
        ids = checkpoint_ids(scenario, 0)
        if launch["error"] is None:
            errors.append("the coupled job survived a killed rank")
        if launch["timed_out"]:
            errors.append("the killed rank was detected only by the whole-job deadline")
        if launch["returncodes"][1:2] != [-signal.SIGKILL]:
            errors.append("rank 1 did not terminate by SIGKILL")
        if not committed(repository, ids[0]):
            errors.append("the first campaign checkpoint was not committed")
        if committed(repository, ids[KILL_AT_CHECKPOINT - 1]):
            errors.append("a partial campaign publication became a global commit")
        return errors
    complete = [rank for rank in ranks if rank is not None]
    if len(complete) != processes:
        return errors
    if len({rank["campaign"]["execution_id"] for rank in complete}) != 1:
        errors.append("ranks prepared different campaign executions")
    if len({rank["final"]["transfer_digest"] for rank in complete}) != 1:
        errors.append("ranks disagree on the transfer digest")
    rounds = [round_ for rank in complete for round_ in rank["rounds"]]
    count = len(complete[0]["campaign"]["stage_members"])
    if scenario == "domain-campaign-publish":
        if len({len(rank["publications"]) for rank in complete}) != 1:
            errors.append("ranks committed different checkpoint counts")
        return errors
    sink_rank = complete[0]
    outputs = _outputs(directory, SINKS[scenario])
    finals = {
        key: output
        for key, output in outputs.items()
        if output.outcome is AtomisticStageOutcome.COMPLETED
        and output.stage == "production"
    }
    if len(finals) != len(TASK_SEEDS) or len(outputs) != len(TASK_SEEDS):
        errors.append(
            f"{len(finals)} of {len(TASK_SEEDS)} tasks completed both stages "
            f"({len(outputs)} outputs)"
        )
    if "outputs" not in sink_rank:
        errors.append("the sink host recorded no oracle comparison")
    if scenario == "domain-campaign":
        if not any(round_["parked"] > 0 for round_ in rounds):
            errors.append("transport credit/sink saturation was not exercised")
        if not any(any(round_["parked_residents"]) for round_ in rounds):
            errors.append("no finished resident was parked by a full outbox")
        if not any(round_["refused"] > 0 for round_ in rounds):
            errors.append("sink backpressure refused no granted publication")
        expected = len(TASK_SEEDS) * (ENTRY_BUDGET + PRODUCTION_BUDGET)
        stage_members = complete[0]["campaign"]["stage_members"]
        per_stage = [
            sum(
                rank_round["executed_steps"][index]
                for rank in complete
                if rank["runtime"]["process_index"] == stage_members[index][0]
                for rank_round in rank["rounds"]
            )
            for index in range(count)
        ]
        if sum(per_stage) != expected:
            errors.append(
                f"stage coordinators executed {per_stage} accepted steps, "
                f"expected {expected} in total"
            )
    if scenario == "domain-campaign-restart":
        reference = _outputs(directory, REFERENCE_SINK)
        if set(reference) != set(outputs):
            errors.append("the restarted campaign published other semantic keys")
        differing = sorted(
            key
            for key in set(reference) & set(outputs)
            if reference[key].payload_id != outputs[key].payload_id
        )
        if differing:
            errors.append(f"restarted outputs differ bitwise: {differing}")
        source = {rank.get("restored_from") for rank in complete}
        if source != {checkpoint_ids("domain-campaign-kill", 0)[0]}:
            errors.append(f"the restart continued from {sorted(map(str, source))}")
    if scenario == "domain-campaign-migrate":
        reference = _outputs(directory, REFERENCE_SINK)
        if set(reference) != set(outputs):
            errors.append("the migrated campaign published other semantic keys")
        for key in sorted(set(reference) & set(outputs)):
            left = reference[key]
            right = outputs[key]
            if left.outcome is not right.outcome:
                errors.append(f"{key}: outcome changed under migration")
                continue
            for port in ("positions", "momenta"):
                difference = float(
                    np.max(np.abs(np.asarray(left.ports[port]) - right.ports[port]))
                )
                if not difference <= MIGRATION_POSITION_TOLERANCE:
                    errors.append(f"{key}: migrated {port} differ by {difference}")
            if "energy" in left.ports:
                energy = float(left.ports["energy"])
                difference = abs(energy - float(right.ports["energy"]))
                if not difference <= MIGRATION_ENERGY_RELATIVE_TOLERANCE * max(
                    1.0, abs(energy)
                ):
                    errors.append(f"{key}: migrated energy differs by {difference}")
        migrations = [rank.get("migration") for rank in complete]
        if any(migration is None for migration in migrations):
            errors.append("a rank recorded no migration evidence")
        elif any(
            migration["source_execution_id"] == migration["target_execution_id"]
            for migration in migrations
            if migration is not None
        ):
            errors.append("the migration did not change the execution topology")
    return errors


__all__ = (
    "CAMPAIGN_EXPECTED_FAILURE",
    "CAMPAIGN_SCENARIOS",
    "CampaignScenario",
    "campaign_checks",
    "campaign_plan",
    "campaign_repository",
    "checkpoint_ids",
    "latest_commit",
    "run_campaign_rank",
    "ThrottledDirectorySink",
)

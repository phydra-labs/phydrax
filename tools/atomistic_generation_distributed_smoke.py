#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real multi-process smoke of distributed atomistic generation quotas.

Every scenario launches coupled local JAX processes through
`phydrax.execution.launch_local_processes` with a finite whole-job deadline.
Each rank binds the root `ExecutionGroup`, places ONE generation stream on it
(`AtomisticGenerationPlacement`) and runs the stateful heterogeneous-countdown
sampler of `tests._support.atomistic_generation` (ordinal-dependent completion,
sampler failures and minimum-distance filter refusals):

``quota``
    One global accepted target and attempt budget over every participant, with
    a sink that refuses its first publications (backpressure). Every rank
    reports the ordinals its own lanes ran, the agreed published run, the
    replicated cursor digest, refused publications, the admission window and
    out-of-order completion. Rank 0 (the publication authority) additionally
    compares the published proposals with single-participant local streams of
    other lane counts, with ordinal-chunked local streams, and with an
    independent recomputation from the semantic addresses. A second stream
    exhausts a global attempt budget.
``checkpoint-kill``
    Durable structure-store publication on the authority, a lifecycle global
    commit of the placed continuation mid-run (in-flight lanes present), more
    durable publications, then rank 1 SIGKILLs itself: the coupled job must
    fail within its deadline while the global commit stays loadable.
``restart``
    A fresh launch restores that commit, reconciles the durable store by
    proposal identity and finishes; the store must hold exactly the reference
    proposals of a local single-participant stream, once each, in ordinal
    order.

``--platform cpu`` (default) forces host CPU devices with gloo collectives: a
functional oracle of the collective program on ONE host, never accelerator or
multi-host evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, assert_never, get_args, Literal, TypeAlias

import jax
import jax.random as jr
import numpy as np

from phydrax.atomistic._generation import (
    AbstractAtomisticGenerationSink,
    AtomisticGeneratedProposal,
    AtomisticGenerationState,
    AtomisticGenerationStopReason,
    AtomisticMemoryGenerationSink,
    AtomisticProposalStatus,
    AtomisticStructureStoreGenerationSink,
    PreparedAtomisticGenerationStream,
)
from phydrax.atomistic._generation_distribution import AtomisticGenerationPlacement
from phydrax.atomistic._structure_store import open_atomistic_structure_store
from phydrax.execution import (
    ExecutionRuntime,
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
    LocalProcessResult,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from tests._support.atomistic_generation import (
    expected_positions,
    generation_record,
    generation_store_schema,
    generation_store_writer,
    store_limits,
    stream_plan,
    ThrottledSink,
)
from tools._accelerator_process import write_json_evidence
from tools.atomistic_distributed_smoke import (
    commit_present,
    free_loopback_port,
    json_safe,
    launch_environment,
    Platform,
    PROJECT_ROOT,
    runtime_record,
)


RESULT_PREFIX = "ATOMISTIC_GENERATION_SMOKE_RESULT "

Scenario: TypeAlias = Literal["quota", "checkpoint-kill", "restart"]

EXPECTED_FAILURE: frozenset[str] = frozenset({"checkpoint-kill"})
CHECKPOINT = "generation-smoke-checkpoint"
ROOT = 23
ROUND_BYTES = 1 << 22
REPLICATED_BYTES = 1 << 22
LANES = 2
TARGET = 10
BUDGET = 40
REFUSALS = 3
BUDGET_ONLY = 13
CHUNK = 6
RESTART_TARGET = 20
RESTART_BUDGET = 100
BEFORE_CHECKPOINT = 6
AFTER_CHECKPOINT = 7
INDEPENDENT_TOLERANCE = 1.0e-12


def generation_repository(directory: Path, /) -> POSIXArtifactRepository:
    """The shared POSIX repository of the checkpoint scenarios."""
    profile = HPCFilesystemProfile(
        "posix.generation-smoke",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        Path(directory) / "repository",
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=1 << 16, maximum_metadata_bytes=1 << 20
        ),
    )


def _store_root(directory: Path, /) -> Path:
    return Path(directory) / "store"


# ------------------------------------------------------------- evidence


def _digest(payload: object, /) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _state_digest(state: AtomisticGenerationState, /) -> str:
    """Replicated continuation digest: cursor, pending bank, root key, receipts."""
    shared = jax.device_get(
        (state.root_key_data, state.bank, state.cursor, state.receipt_digest)
    )
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(shared):
        digest.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return digest.hexdigest()


def _summary(records: Sequence[AtomisticGeneratedProposal], /) -> list[dict[str, Any]]:
    return [
        {
            "proposal_id": item.proposal_id,
            "ordinal": item.ordinal,
            "status": item.status,
            "filter_bits": item.filter_bits,
            "positions": hashlib.sha256(
                np.ascontiguousarray(item.positions).tobytes()
            ).hexdigest(),
        }
        for item in records
    ]


def _local_ordinals(state: AtomisticGenerationState, /) -> set[int]:
    """Ordinals admitted to this participant's own lanes."""
    owned: set[int] = set()
    for lanes in state.lanes:
        active = np.asarray(lanes.active)
        owned.update(int(value) for value in np.asarray(lanes.ordinals)[active])
    return owned


def _pending_ordinals(state: AtomisticGenerationState, /) -> set[int]:
    bank = jax.device_get(state.bank)
    return {int(value) for value in np.asarray(bank.ordinals)[np.asarray(bank.occupied)]}


def _drive(
    stream: PreparedAtomisticGenerationStream,
    state: AtomisticGenerationState,
    sink: AbstractAtomisticGenerationSink | None,
    /,
    *,
    maximum_transitions: int,
) -> tuple[AtomisticGenerationState, dict[str, Any], list[AtomisticGeneratedProposal]]:
    """Bounded transitions with publication after each, plus placement evidence."""
    ran: set[int] = set()
    completion: dict[int, int] = {}
    published: list[AtomisticGeneratedProposal] = []
    refused = window = 0
    for transition in range(maximum_transitions):
        if stream.finished(state):
            break
        state = stream.transition(state)
        ran |= _local_ordinals(state)
        for ordinal in _pending_ordinals(state):
            completion.setdefault(ordinal, transition)
        window = max(window, int(state.cursor.next_ordinal) - int(state.cursor.watermark))
        ready = stream.ready(state)
        state, receipt = stream.publish(state, sink)
        if ready and receipt is None:
            refused += 1
        if receipt is not None:
            published.extend(ready[: receipt.count])
    ordinals = sorted(completion)
    out_of_order = any(
        completion[later] < completion[earlier]
        for index, earlier in enumerate(ordinals)
        for later in ordinals[index + 1 :]
    )
    evidence = {
        "ran": sorted(ran),
        "refused_publications": refused,
        "maximum_window": window,
        "out_of_order_completion": out_of_order,
        "transitions": int(state.cursor.transitions),
    }
    return state, evidence, published


def _compare(
    observed: Sequence[AtomisticGeneratedProposal],
    expected: Sequence[AtomisticGeneratedProposal],
    label: str,
    errors: list[str],
    /,
) -> None:
    if [item.proposal_id for item in observed] != [item.proposal_id for item in expected]:
        errors.append(f"{label}: proposal identities differ")
        return
    for left, right in zip(observed, expected, strict=True):
        if not np.array_equal(left.positions, right.positions) or (
            left.status,
            left.filter_bits,
        ) != (right.status, right.filter_bits):
            errors.append(f"{label}: proposal {left.ordinal} differs")
            return


def _local_run(
    lanes: int, *, target: int, budget: int, first: int = 0
) -> list[AtomisticGeneratedProposal]:
    """Single-participant local stream (no collective) of the same definition."""
    stream = stream_plan(lanes_per_member=lanes).prepare()
    sink = AtomisticMemoryGenerationSink(f"local-{lanes}-{first}")
    state = stream.start(
        jr.key(ROOT), target=target, attempt_budget=budget, first_ordinal=first
    )
    stream.run(state, sink, maximum_transitions=1000)
    return list(sink.records)


def _independent(
    records: Sequence[AtomisticGeneratedProposal], errors: list[str], /
) -> None:
    reference = expected_positions(ROOT, [item.ordinal for item in records])
    for item, positions in zip(records, reference, strict=True):
        if not np.allclose(
            item.positions, positions, rtol=0.0, atol=INDEPENDENT_TOLERANCE
        ):
            errors.append(
                f"proposal {item.ordinal} differs from its independent recomputation"
            )
            return


# ------------------------------------------------------------- scenarios


def _quota(
    placement: AtomisticGenerationPlacement, record: dict[str, Any], errors: list[str], /
) -> None:
    stream = stream_plan(lanes_per_member=LANES).prepare(placement=placement)
    throttled = (
        ThrottledSink(AtomisticMemoryGenerationSink("quota"), REFUSALS)
        if placement.is_authority
        else None
    )
    state = stream.start(jr.key(ROOT), target=TARGET, attempt_budget=BUDGET)
    state, evidence, published = _drive(stream, state, throttled, maximum_transitions=400)
    if state.stop_reason is not AtomisticGenerationStopReason.TARGET_REACHED:
        errors.append(f"the target stream stopped with {state.stop_reason.name}")
    if int(state.cursor.committed_accepted) != TARGET:
        errors.append("the committed accepted count differs from the global target")
    if evidence["refused_publications"] < REFUSALS:
        errors.append("sink backpressure was not exercised")
    if evidence["maximum_window"] > stream.plan.pending_capacity:
        errors.append("admission exceeded the bounded pending window")
    if not evidence["out_of_order_completion"]:
        errors.append("heterogeneous out-of-order completion was not exercised")
    statuses = [item.status for item in published]
    if not any(
        status & int(AtomisticProposalStatus.SAMPLER_FAILURE) for status in statuses
    ):
        errors.append("no sampler refusal was published")
    if not any(
        status & int(AtomisticProposalStatus.FILTER_REJECTED) for status in statuses
    ):
        errors.append("no filter refusal was published")
    if [item.ordinal for item in published] != list(range(len(published))):
        errors.append("publication is not ordinal ordered from the first ordinal")

    budget_stream = stream_plan(lanes_per_member=LANES).prepare(placement=placement)
    budget_sink = (
        AtomisticMemoryGenerationSink("budget") if placement.is_authority else None
    )
    budget_state = budget_stream.start(
        jr.key(ROOT), target=1000, attempt_budget=BUDGET_ONLY
    )
    budget_state, budget_evidence, budget_published = _drive(
        budget_stream, budget_state, budget_sink, maximum_transitions=400
    )
    if budget_state.stop_reason is not AtomisticGenerationStopReason.BUDGET_EXHAUSTED:
        errors.append(f"the budget stream stopped with {budget_state.stop_reason.name}")
    if [item.ordinal for item in budget_published] != list(range(BUDGET_ONLY)):
        errors.append("the global attempt budget did not bound the published ordinals")
    record["quota"] = {
        **evidence,
        "published": _summary(published),
        "published_digest": _digest(_summary(published)),
        "state_digest": _state_digest(state),
        "stop_reason": state.stop_reason.name,
        "committed_accepted": int(state.cursor.committed_accepted),
        "committed_rejected": int(state.cursor.committed_rejected),
        "receipts": int(state.cursor.receipts),
        "round_bytes": stream.round_bytes,
        "replicated_bytes": stream.replicated_bytes,
        "authority": placement.is_authority,
        "participant": placement.position,
    }
    record["budget"] = {
        **budget_evidence,
        "published_digest": _digest(_summary(budget_published)),
        "state_digest": _state_digest(budget_state),
        "stop_reason": budget_state.stop_reason.name,
        "shortfall": budget_stream.shortfall(budget_state),
    }
    if not placement.is_authority:
        return
    # Placement independence: other lane counts, one participant, ordinal chunks.
    for lanes in (1, 3):
        _compare(
            published,
            _local_run(lanes, target=TARGET, budget=BUDGET),
            f"local {lanes}-lane reference",
            errors,
        )
    chunked = _local_run(3, target=1000, budget=CHUNK) + _local_run(
        1, target=1000, budget=BUDGET_ONLY - CHUNK, first=CHUNK
    )
    _compare(budget_published, chunked, "ordinal-chunked reference", errors)
    _independent(published, errors)
    _independent(budget_published, errors)


def _store_sink(directory: Path, /) -> AtomisticStructureStoreGenerationSink:
    root = _store_root(directory)
    return AtomisticStructureStoreGenerationSink(
        root, generation_store_writer(root), generation_record, limits=store_limits()
    )


def _checkpoint_kill(
    placement: AtomisticGenerationPlacement,
    directory: Path,
    errors: list[str],
    /,
) -> None:
    sink = None
    if placement.is_authority:
        generation_store_writer(_store_root(directory)).create(
            generation_store_schema(), name="distributed-generation"
        )
        sink = _store_sink(directory)
    stream = stream_plan(lanes_per_member=LANES).prepare(placement=placement)
    state = stream.start(
        jr.key(ROOT), target=RESTART_TARGET, attempt_budget=RESTART_BUDGET
    )
    state, _, _ = _drive(stream, state, sink, maximum_transitions=BEFORE_CHECKPOINT)
    # Harvest without publishing until a ready run is pending, so the commit
    # precedes its durable publication (the restart reconciliation window).
    # Ready runs are replicated: every participant leaves this loop together.
    for _ in range(BEFORE_CHECKPOINT):
        if stream.ready(state):
            break
        state = stream.transition(state)
    if not stream.ready(state):
        errors.append("no unpublished ready run preceded the global commit")
    stream.publish_checkpoint(
        generation_repository(directory),
        state,
        checkpoint_id=CHECKPOINT,
        writer_id=f"participant-{placement.position}",
    )
    state, _, _ = _drive(stream, state, sink, maximum_transitions=AFTER_CHECKPOINT)
    if jax.process_index() == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    # The surviving authority enters the next exchange and blocks there until
    # the launcher fails the coupled job.
    stream.transition(state)
    errors.append("a transition returned although a participant was killed")


def _restart(
    placement: AtomisticGenerationPlacement,
    directory: Path,
    record: dict[str, Any],
    errors: list[str],
    /,
) -> None:
    stream = stream_plan(lanes_per_member=LANES).prepare(placement=placement)
    state = stream.restore_checkpoint(generation_repository(directory), CHECKPOINT)
    watermark = int(state.cursor.watermark)
    active = int(state.cursor.active)
    restored: dict[str, Any] = {
        "watermark": watermark,
        "next_ordinal": int(state.cursor.next_ordinal),
        "active": active,
        "local_active": len(_local_ordinals(state)),
        "state_digest": _state_digest(state),
    }
    if active == 0:
        errors.append("the restored continuation holds no in-flight lanes")
    sink = _store_sink(directory) if placement.is_authority else None
    if sink is not None:
        durable = len(
            open_atomistic_structure_store(
                _store_root(directory), limits=store_limits()
            ).record_ids
        )
        restored["durable_records"] = durable
        if durable <= watermark:
            errors.append("no durable publication followed the global commit")
    state = stream.reconcile(state, sink)
    if int(state.cursor.watermark) <= watermark:
        errors.append("restart reconciliation committed no durable proposal")
    restored["reconciled_watermark"] = int(state.cursor.watermark)
    state, evidence, _ = _drive(stream, state, sink, maximum_transitions=400)
    if state.stop_reason is not AtomisticGenerationStopReason.TARGET_REACHED:
        errors.append(f"the restarted stream stopped with {state.stop_reason.name}")
    record["restart"] = {
        **restored,
        **evidence,
        "state_digest": _state_digest(state),
        "stop_reason": state.stop_reason.name,
    }
    if not placement.is_authority:
        return
    store = open_atomistic_structure_store(_store_root(directory), limits=store_limits())
    ids = store.record_ids
    batch = store.read(np.arange(len(ids)))
    offsets = batch.offsets("atom")
    reference = _local_run(3, target=RESTART_TARGET, budget=RESTART_BUDGET)
    if len(set(ids)) != len(ids):
        errors.append("the durable store holds duplicate proposals")
    if list(ids) != [item.proposal_id for item in reference]:
        errors.append("the durable store differs from the local reference proposals")
    else:
        for index, item in enumerate(reference):
            rows = slice(int(offsets[index]), int(offsets[index + 1]))
            if not np.array_equal(batch.values["positions"][rows], item.positions):
                errors.append(
                    f"stored proposal {item.ordinal} differs from the reference"
                )
                break
    record["restart"]["stored_records"] = len(ids)


def run_rank(scenario: Scenario, directory: Path) -> dict[str, Any]:
    """Execute one scenario as one rank of a coupled launch."""
    runtime = ExecutionRuntime.current()
    record: dict[str, Any] = {"scenario": scenario, "runtime": runtime_record(runtime)}
    errors: list[str] = []
    placement = AtomisticGenerationPlacement(
        runtime.root_group,
        maximum_round_bytes=ROUND_BYTES,
        maximum_replicated_bytes=REPLICATED_BYTES,
    )
    match scenario:
        case "quota":
            _quota(placement, record, errors)
        case "checkpoint-kill":
            _checkpoint_kill(placement, directory, errors)
        case "restart":
            _restart(placement, directory, record, errors)
        case _:
            assert_never(scenario)
    record["errors"] = errors
    record["passed"] = not errors
    return record


# ------------------------------------------------------------ coupled launch


def _terminal_record(result: LocalProcessResult, /) -> dict[str, Any] | None:
    lines = [
        line[len(RESULT_PREFIX) :]
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if not lines:
        return None
    record = json.loads(lines[-1])
    return record if isinstance(record, dict) else None


def _quota_checks(ranks: Sequence[dict[str, Any]], /) -> list[str]:
    errors: list[str] = []
    quotas = [rank["quota"] for rank in ranks]
    for name in ("state_digest", "published_digest"):
        if len({quota[name] for quota in quotas}) != 1:
            errors.append(f"participants disagree on the {name.replace('_', ' ')}")
    for name in ("state_digest", "published_digest"):
        if len({rank["budget"][name] for rank in ranks}) != 1:
            errors.append(f"participants disagree on the budget {name.replace('_', ' ')}")
    ran = [set(quota["ran"]) for quota in quotas]
    if any(not owned for owned in ran):
        errors.append("a participant ran no ordinal")
    if sum(len(owned) for owned in ran) != len(set().union(*ran)):
        errors.append("an ordinal ran on more than one participant")
    published = {item["ordinal"] for item in quotas[0]["published"]}
    if not published <= set().union(*ran):
        errors.append("a published ordinal ran on no participant")
    if [quota["authority"] for quota in quotas] != [True] + [False] * (len(quotas) - 1):
        errors.append("the publication authority is not exactly the first participant")
    return errors


def _checks(
    scenario: Scenario,
    ranks: list[dict[str, Any] | None],
    processes: int,
    launch: dict[str, Any],
    directory: Path,
) -> list[str]:
    errors: list[str] = []
    if scenario in EXPECTED_FAILURE:
        if launch["error"] is None:
            errors.append("the coupled job survived a killed participant")
        if launch["timed_out"]:
            errors.append("the killed participant was detected only by the deadline")
        if launch["returncodes"][1:2] != [-signal.SIGKILL]:
            errors.append("rank 1 did not terminate by SIGKILL")
        if not commit_present(generation_repository(directory), CHECKPOINT):
            errors.append("the mid-run global commit is not loadable")
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
    if errors:
        return errors
    match scenario:
        case "quota":
            errors.extend(_quota_checks(complete))
        case "restart":
            if len({rank["restart"]["state_digest"] for rank in complete}) != 1:
                errors.append("participants disagree on the restarted continuation")
            if (
                sum(rank["restart"]["local_active"] for rank in complete)
                != complete[0]["restart"]["active"]
            ):
                errors.append("restored lane continuations do not match the global count")
        case "checkpoint-kill":
            errors.append("the expected-failure scenario reached the success checks")
        case _:
            assert_never(scenario)
    return errors


def run_scenario(
    scenario: Scenario,
    directory: Path,
    /,
    *,
    processes: int = 2,
    devices_per_process: int = 1,
    platform: Platform = "cpu",
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Launch one coupled scenario and return its complete, never-dropped record."""
    if scenario not in get_args(Scenario):
        raise ValueError(f"Unknown scenario {scenario!r}.")
    if processes < 2 or devices_per_process < 1:
        raise ValueError("Scenarios need at least two processes and one device each.")
    if scenario != "quota" and processes != 2:
        raise ValueError("Checkpoint scenarios are defined on exactly two processes.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable,
        "-m",
        "tools.atomistic_generation_distributed_smoke",
        "--worker",
        "--scenario",
        scenario,
        "--directory",
        str(directory),
    )
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
        "error": None if error is None else str(error)[:4000],
        "reason": None if error is None else error.reason,
        "timed_out": False if error is None else bool(error.timed_out),
        "returncodes": [result.returncode for result in ordered],
        "stderr_tails": [
            result.stderr[-4000:].decode("utf-8", errors="replace") for result in ordered
        ],
    }
    errors = parse_errors + _checks(scenario, ranks, processes, launch, directory)
    return json_safe(
        {
            "scenario": scenario,
            "expected_failure": scenario in EXPECTED_FAILURE,
            "launch": launch,
            "ranks": ranks,
            "errors": errors,
            "passed": not errors,
            "evidence_scope": (
                "functional forced-CPU multi-process collective program on one host; "
                "not accelerator or multi-host evidence"
                if platform == "cpu"
                else "observed local devices of one host; not multi-host evidence"
            ),
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=get_args(Scenario), required=True)
    parser.add_argument("--platform", choices=get_args(Platform), default="cpu")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--local-processes", type=int, default=2)
    parser.add_argument("--devices-per-process", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.worker:
        record = json_safe(run_rank(arguments.scenario, arguments.directory))
        print(
            RESULT_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False),
            flush=True,
        )
        if not record["passed"]:
            raise SystemExit(1)
        return
    record = run_scenario(
        arguments.scenario,
        arguments.directory,
        processes=arguments.local_processes,
        devices_per_process=arguments.devices_per_process,
        platform=arguments.platform,
        timeout_seconds=arguments.timeout_seconds,
    )
    if arguments.output is not None:
        write_json_evidence(arguments.output, record)
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

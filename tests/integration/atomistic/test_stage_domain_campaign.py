#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real multi-process stage×domain campaigns (backpressure, restart, migration).

Each case launches coupled local JAX processes through
`tools.atomistic_distributed_smoke` (finite whole-job deadline, fail-coupled
ranks, forced host-CPU devices with gloo collectives). Two process-subset
child stage groups run native distributed MACE domain dynamics (BAOAB NVT
equilibration, velocity-Verlet NVE production) and exchange complete domain
continuations over the root-group transport. Functional oracles of the
collective program only: not accelerator, multi-host or pretrained evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tools._atomistic_smoke_campaign import (
    checkpoint_ids,
    MIGRATION_SINK,
    REFERENCE_SINK,
    RESTART_SINK,
    TASK_SEEDS,
)
from tools.atomistic_distributed_smoke import run_scenario


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="requires Linux boot identities for observed host mapping",
    ),
]

# Whole-job deadline of one coupled launch (bootstrap plus CPU compilation of
# two stage programs, entry priming and the segment executables).
DEADLINE_SECONDS = 900.0


def _require_passed(record: dict) -> None:
    assert record["passed"], record["errors"]


def test_backpressured_campaign_restarts_bitwise_after_a_killed_rank(
    tmp_path: Path,
) -> None:
    reference = run_scenario(
        "domain-campaign", tmp_path, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(reference)
    producer, consumer = reference["ranks"]
    # Only stage members execute physics: rank 0 equilibrates, rank 1 runs
    # production; parked residents and parked packets were both observed.
    assert producer["campaign"]["member_stages"] == [0]
    assert consumer["campaign"]["member_stages"] == [1]
    assert any(round_["parked_residents"][0] for round_ in producer["rounds"]), (
        "equilibration never parked a finished task"
    )
    assert len(producer["outputs"]) == len(TASK_SEEDS)
    assert all(
        output["outcome"] == "COMPLETED" and output["stage"] == "production"
        for output in producer["outputs"].values()
    )

    killed = run_scenario(
        "domain-campaign-kill", tmp_path, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(killed)
    restarted = run_scenario(
        "domain-campaign-restart", tmp_path, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(restarted)
    assert {rank["restored_from"] for rank in restarted["ranks"]} == {
        checkpoint_ids("domain-campaign-kill", 0)[0]
    }
    # Payload identities match the uninterrupted run bitwise.
    reference_outputs = {
        key: output["payload_id"] for key, output in producer["outputs"].items()
    }
    restart_outputs = {
        key: output["payload_id"]
        for key, output in restarted["ranks"][0]["outputs"].items()
    }
    assert restart_outputs == reference_outputs
    assert (tmp_path / REFERENCE_SINK).is_dir() and (tmp_path / RESTART_SINK).is_dir()


def _in_flight(rounds: list[dict], stage: int, /) -> str | None:
    """The task a stage member holds after its recorded rounds (or ``None``)."""
    resident: str | None = None
    for round_ in rounds:
        if round_["admitted"][stage] is not None:
            resident = round_["admitted"][stage]
        if resident is not None and round_["exported"][stage] == resident:
            resident = None
    return resident


def test_committed_campaign_migrates_to_a_changed_topology(tmp_path: Path) -> None:
    reference = run_scenario(
        "domain-campaign", tmp_path, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(reference)
    published = run_scenario(
        "domain-campaign-publish", tmp_path, timeout_seconds=DEADLINE_SECONDS
    )
    _require_passed(published)
    assert [len(rank["publications"]) for rank in published["ranks"]] == [1, 1]
    migrated = run_scenario(
        "domain-campaign-migrate",
        tmp_path,
        devices_per_process=1,
        timeout_seconds=DEADLINE_SECONDS,
    )
    # Cross-launch checks inside the record: every migrated output keeps its
    # semantic key and outcome, positions/momenta/energy stay within roundoff
    # of the uninterrupted reference, and energies/forces match the
    # independent single-device MACE oracle.
    _require_passed(migrated)
    # The per-process fragment layout really changed (two device owners per
    # process -> one), so the source continuations were decoded under their
    # reconstructed source schema, not the target's.
    assert [rank["campaign"]["owner_counts"] for rank in published["ranks"]] == [
        [2, None],
        [None, 2],
    ]
    owners = [rank["campaign"]["owner_counts"] for rank in migrated["ranks"]]
    assert owners == [[1, None], [None, 1]]
    source_schemas = published["ranks"][0]["campaign"]["stage_schemas"]
    target_schemas = migrated["ranks"][0]["campaign"]["stage_schemas"]
    assert all(
        source != target
        for source, target in zip(source_schemas, target_schemas, strict=True)
    )
    in_flight = 0
    for source, rank in zip(published["ranks"], migrated["ranks"], strict=True):
        evidence = rank["migration"]
        publication = source["publications"][0]
        # Identity: the exact committed source checkpoint, execution, round and
        # transfer chain are continued by a different target execution.
        assert rank["restored_from"] == publication["checkpoint_id"]
        assert evidence["source_execution_id"] == source["campaign"]["execution_id"]
        assert evidence["target_execution_id"] == rank["campaign"]["execution_id"]
        assert evidence["source_execution_id"] != evidence["target_execution_id"]
        assert evidence["source_process_count"] == evidence["target_process_count"] == 2
        assert evidence["source_round_index"] == publication["round_index"]
        assert evidence["source_transfer_digest"] == source["final"]["transfer_digest"]
        assert evidence["read_bytes"] > 0
        # Ownership: each stage's resident resumes on its new stage group.
        stages = len(source["campaign"]["stage_members"])
        assert rank["campaign"]["member_stages"] == source["campaign"]["member_stages"]
        assert evidence["residents"] == [
            _in_flight(source["rounds"], stage)
            if stage in source["campaign"]["member_stages"]
            else None
            for stage in range(stages)
        ]
        in_flight += sum(resident is not None for resident in evidence["residents"])
        in_flight += sum(evidence["reissued"])
        in_flight += sum(round_["receipts"] for round_ in source["rounds"])
    # Conservation: every task the entry stage admitted before the commit is
    # a resumed resident, a recommitted/re-issued continuation or report, or
    # an output published before the commit.
    entry_cursor = max(rank["final"]["cursor"] for rank in published["ranks"])
    assert in_flight == entry_cursor
    assert any(sum(rank["migration"]["reissued"]) for rank in migrated["ranks"])
    # Evaluation identity is layout-independent: outputs finished on two device
    # owners per process (reference) and on one (after migration) carry the
    # same Hamiltonian, numeric revision and evaluated conditions.
    from phydrax.atomistic._campaign import AtomisticDirectorySink

    reference_outputs = AtomisticDirectorySink(tmp_path / REFERENCE_SINK).outputs()
    migrated_outputs = AtomisticDirectorySink(tmp_path / MIGRATION_SINK).outputs()
    assert set(reference_outputs) == set(migrated_outputs)
    for key, output in reference_outputs.items():
        assert output.evaluation_key is not None
        assert migrated_outputs[key].evaluation_key_id == output.evaluation_key_id
        assert migrated_outputs[key].numeric_revision_ids == output.numeric_revision_ids


def test_in_process_campaign_replays_its_outputs_from_a_commit(
    tmp_path: Path,
) -> None:
    """One process holding both stage groups: commits leave the trajectory
    unchanged and a restore replays the uninterrupted run bitwise."""
    from phydrax.atomistic._campaign import AtomisticDirectorySink
    from phydrax.atomistic._stage_transport import (
        AtomisticDomainCampaignCheckpointError,
        AtomisticDomainCampaignCheckpointPolicy,
    )
    from phydrax.execution import ExecutionRuntime
    from phydrax.lifecycle._distributed_checkpoint import (
        load_distributed_checkpoint_commit,
        read_checkpoint_region,
    )
    from tools._atomistic_smoke_campaign import (
        campaign_plan,
        campaign_repository,
        CHECKPOINT_LIMITS,
        SINK_PERIOD,
        ThrottledDirectorySink,
    )

    runtime = ExecutionRuntime.current()
    if len(runtime.root_group.spec.process_indices) != 1:
        pytest.skip("the in-process campaign runs in a single-process job")
    plan, _ = campaign_plan(runtime, "halo")
    campaign = plan.prepare(runtime.root_group)
    # Genuinely uninterrupted: no checkpoint is ever published.
    uninterrupted, rounds, none = campaign.run(
        campaign.start(), sink=ThrottledDirectorySink(tmp_path / "plain", SINK_PERIOD)
    )
    assert uninterrupted.terminated and none == ()
    assert any(any(result.parked) for result in rounds)
    plain = AtomisticDirectorySink(tmp_path / "plain").outputs()
    assert len(plain) == len(TASK_SEEDS)
    # The evaluation key binds the domain Hamiltonian and its numeric revision
    # (never a placement identity) and one evaluated-conditions digest.
    production = campaign.bindings[1]
    assert production is not None
    domain = production.dynamics.runtime
    keys = []
    for output in plain.values():
        assert output.evaluation_key is not None
        keys.append(output.evaluation_key)
    assert {key.hamiltonian_id for key in keys} == {domain.scientific_id}
    assert {key.numeric_revision_ids for key in keys} == {
        (domain.hamiltonian_revision_id,)
    }
    assert len({key.conditions_id for key in keys}) == 1
    assert {output.numeric_revision_ids for output in plain.values()} == {
        (domain.hamiltonian_revision_id,)
    }
    # One process owns every stage group, so admit its complete chunk manifest.
    repository = campaign_repository(
        tmp_path, maximum_metadata_bytes=CHECKPOINT_LIMITS.max_manifest_bytes
    )
    policy = AtomisticDomainCampaignCheckpointPolicy(
        repository, 3, "unit", "rank-0", limits=CHECKPOINT_LIMITS
    )
    final, _, publications = campaign.run(
        campaign.start(),
        sink=ThrottledDirectorySink(tmp_path / "first", SINK_PERIOD),
        checkpoint=policy,
    )
    assert final.terminated and len(publications) >= 2
    first = AtomisticDirectorySink(tmp_path / "first").outputs()
    # Publishing commits changes nothing in the accepted trajectory.
    assert {key: output.payload_id for key, output in first.items()} == {
        key: output.payload_id for key, output in plain.items()
    }
    np.testing.assert_array_equal(
        final.transport.transfer_digest, uninterrupted.transport.transfer_digest
    )
    restored = campaign.restore(
        repository, publications[0].checkpoint_id, limits=CHECKPOINT_LIMITS
    )
    assert restored.epoch == 1 and restored.round_index == publications[0].round_index
    replay, _, _ = campaign.run(
        restored,
        sink=ThrottledDirectorySink(tmp_path / "replay", SINK_PERIOD),
        checkpoint=policy,
    )
    assert replay.terminated
    replayed = AtomisticDirectorySink(tmp_path / "replay").outputs()
    # Everything published after the commit equals the uninterrupted run.
    assert replayed and all(
        plain[key].payload_id == output.payload_id for key, output in replayed.items()
    )
    np.testing.assert_array_equal(
        replay.transport.transfer_digest, uninterrupted.transport.transfer_digest
    )
    with pytest.raises(AtomisticDomainCampaignCheckpointError):
        campaign.migrate(
            repository, publications[0].checkpoint_id, limits=CHECKPOINT_LIMITS
        )
    # Every stage records its derived layout-independent scientific identity
    # (model, terms, roles, cell, integrator, ensemble, schedule, constraints)
    # beside the layout-bound runtime, revision and dynamics-plan rows.
    manifest = load_distributed_checkpoint_commit(
        repository, publications[0].checkpoint_id, limits=CHECKPOINT_LIMITS
    )
    identity = read_checkpoint_region(
        repository,
        manifest,
        "['campaign']['identity']",
        (slice(None),) * 4,
        limits=CHECKPOINT_LIMITS,
    )
    assert identity.shape == (1, len(plan.stages), 4, 8)
    for index, binding in enumerate(campaign.bindings):
        assert binding is not None
        np.testing.assert_array_equal(
            identity[0, index, 3],
            np.asarray(binding.dynamics.scientific_identity.words, np.uint32),
        )
    # NVT equilibration and NVE production are different physics.
    assert not np.array_equal(identity[0, 0, 3], identity[0, 1, 3])


def test_stage_checkpoint_requires_scientific_identity_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.atomistic._stage_transport import (
        AtomisticDomainCampaignCheckpointError,
        AtomisticDomainCampaignState,
        PreparedAtomisticDomainCampaign,
    )
    from phydrax.execution import ExecutionRuntime
    from tools._atomistic_smoke_campaign import (
        campaign_plan,
        campaign_repository,
        CHECKPOINT_LIMITS,
    )

    runtime = ExecutionRuntime.current()
    if len(runtime.root_group.spec.process_indices) != 1:
        pytest.skip("the in-process campaign runs in a single-process job")
    plan, _ = campaign_plan(runtime, "halo")
    campaign = plan.prepare(runtime.root_group)
    repository = campaign_repository(
        tmp_path, maximum_metadata_bytes=CHECKPOINT_LIMITS.max_manifest_bytes
    )
    original = PreparedAtomisticDomainCampaign._blocks

    def without_science(
        self: PreparedAtomisticDomainCampaign, state: AtomisticDomainCampaignState
    ) -> dict[str, np.ndarray]:
        blocks = original(self, state)
        return {**blocks, "identity": blocks["identity"][:, :3]}

    with monkeypatch.context() as patch:
        patch.setattr(PreparedAtomisticDomainCampaign, "_blocks", without_science)
        campaign.checkpoint(
            campaign.start(),
            repository,
            "missing-scientific-identity",
            writer_id="rank-0",
            limits=CHECKPOINT_LIMITS,
        )
    with pytest.raises(AtomisticDomainCampaignCheckpointError):
        campaign.restore(
            repository, "missing-scientific-identity", limits=CHECKPOINT_LIMITS
        )

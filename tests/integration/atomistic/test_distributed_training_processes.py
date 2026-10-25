"""Real multi-process data×domain training of a native MACE model.

Each case launches two coupled local JAX processes with two forced host CPU
devices each (gloo collectives) through `tools.atomistic_distributed_smoke`,
which dispatches to `tools._atomistic_smoke_training`. The training group
composes a data axis of size 2 with a domain owner axis of size 2 on real
devices; the domain plan spans both processes. References are independent
single-device global MACE evaluations, never the distributed route in lanes.
Functional oracles only: not accelerator, multi-host or pretrained-model
evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from tools._atomistic_smoke_training import (
    GRADIENT_RELATIVE_TOLERANCE,
    OBJECTIVE_RELATIVE_TOLERANCE,
    SUPPORT_ABSOLUTE_TOLERANCE,
    TRAINING_CHECKPOINT,
    TRAINING_CHECKPOINT_LIMITS,
)
from tools.atomistic_distributed_smoke import (
    commit_present,
    domain_repository,
    run_scenario,
    Scenario,
)


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="requires Linux boot identities for observed host mapping",
    ),
]

# Whole-job deadline of one coupled launch: JAX bootstrap, CPU compilation of
# the two-layer MACE domain training step and the single-device reference.
DEADLINE_SECONDS = 600.0


def _launch(scenario: Scenario, directory: Path) -> dict[str, Any]:
    record = run_scenario(
        scenario,
        directory,
        processes=2,
        devices_per_process=2,
        timeout_seconds=DEADLINE_SECONDS,
    )
    assert record["passed"], record["errors"]
    assert len(record["ranks"]) == 2
    return record


def _assert_composed_topology(rank: dict[str, Any]) -> None:
    topology = rank["training"]["topology"]
    assert topology["data_shards"] == 2
    assert topology["owner_count"] == 2
    assert topology["mesh_axes"] == [["data", 2], [topology["owner_axis"], 2]]
    assert topology["local_shards"] == [0, 1]
    # One domain plan whose owners span both processes serves every data row.
    assert sorted(topology["owner_processes"]) == [0, 1]
    assert [row["atoms"] for row in topology["rows"]] == [6, 5]


def _assert_matches_reference(step: dict) -> None:
    assert step["objective_error"] <= OBJECTIVE_RELATIVE_TOLERANCE * abs(
        step["reference_value"]
    )
    assert step["gradient_error"] <= GRADIENT_RELATIVE_TOLERANCE * step["gradient_scale"]
    assert step["gradient_scale"] > 0.0
    # Numerators and supports reduce over data shards before normalization:
    # the energy-label-free shard has exactly zero energy support.
    energy_supports = [energy for energy, _ in step["row_supports"]]
    force_supports = [force for _, force in step["row_supports"]]
    assert energy_supports == [1.0, 0.0]
    assert force_supports[0] != force_supports[1]
    assert abs(step["energy_support"] - 1.0) <= SUPPORT_ABSOLUTE_TOLERANCE
    assert abs(step["force_support"] - sum(force_supports)) <= SUPPORT_ABSOLUTE_TOLERANCE
    assert step["capacity_failures"] == 0 and step["scientific_failures"] == 0


def test_data_domain_training_matches_independent_single_device_reference(
    tmp_path: Path,
) -> None:
    record = _launch("dp-domain", tmp_path)
    for rank in record["ranks"]:
        _assert_composed_topology(rank)
        steps = rank["training"]["steps"]
        assert len(steps) == 2
        for step in steps:
            _assert_matches_reference(step)
        assert rank["training"]["inert_empty_support"]["bitwise"]
    # Every process holds the identical shared-kernel state after each step.
    assert len({str(rank["training"]["digests"]) for rank in record["ranks"]}) == 1
    assert commit_present(
        domain_repository(tmp_path),
        TRAINING_CHECKPOINT,
        limits=TRAINING_CHECKPOINT_LIMITS,
    )


def test_data_domain_training_restarts_bitwise_from_durable_kernel_section(
    tmp_path: Path,
) -> None:
    uninterrupted = _launch("dp-domain", tmp_path)
    restarted = _launch("dp-domain-restart", tmp_path)
    for before, after in zip(uninterrupted["ranks"], restarted["ranks"], strict=True):
        _assert_composed_topology(after)
        assert after["training"]["restart_class"] == "bitwise"
        assert after["training"]["manifest_id"] == before["training"]["manifest_id"]
        digests = after["training"]["digests"]
        assert digests["restored_step1"] == before["training"]["digests"]["step1"]
        assert digests["step2"] == before["training"]["digests"]["step2"]
        assert (
            after["training"]["evidence_bits"]["step2"]
            == before["training"]["evidence_bits"]["step2"]
        )
        _assert_matches_reference(after["training"]["steps"][0])

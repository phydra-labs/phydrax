#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax._fingerprint import canonical_fingerprint
from phydrax.lifecycle._event_graph_repository import (
    EventGraphRepository,
    GlobalWorkItem,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    DarkSectorRunCoordinator,
    finalize_dark_sector_epoch,
)


def _digest(name: str) -> str:
    return canonical_fingerprint({"benchmark": name})


def _policy() -> POSIXRepositoryPolicy:
    return POSIXRepositoryPolicy(
        HPCFilesystemProfile(
            "dark-sector-benchmark-posix",
            "local-posix",
            atomic_rename_same_filesystem=True,
            file_fsync=True,
            directory_fsync=True,
            advisory_locking=True,
            attempt_private_staging=True,
        ),
        maximum_chunk_bytes=4096,
        maximum_metadata_bytes=64 * 1024,
    )


def _plan() -> DarkSectorEpochPlan:
    return DarkSectorEpochPlan(
        packet_capacity=2,
        event_capacity=2,
        product_capacity=2,
        radiation_capacity=2,
        work_capacity=1,
        frontier_capacity=1,
        packet_width=4,
        event_width=4,
        product_width=4,
        radiation_width=4,
        work_width=2,
        frontier_width=2,
        species_revision_id=_digest("species"),
        topology_revision_id=_digest("topology"),
    )


def _execute(root: Path, epochs: int) -> dict[str, float | int | str]:
    artifacts = POSIXArtifactRepository(root, _policy())
    graph = EventGraphRepository(artifacts, maximum_record_bytes=8 * 1024 * 1024)
    plan = _plan()
    coordinator = DarkSectorRunCoordinator(
        graph, plan, "benchmark-run", worker_id="benchmark-worker"
    )
    parent_work = None
    work_count = 0
    high_water = 0
    expected_tip_manifest_id = None
    started = time.perf_counter()
    for epoch in range(epochs):
        state = coordinator.resume().state
        new_count = 2 if epoch == 0 else 1
        work_items = []
        values = []
        for local in range(new_count):
            work = GlobalWorkItem(
                "benchmark-cascade",
                (),
                _digest("matrix-element"),
                partition_key=f"partition-{epoch % 2}",
                priority=work_count,
                parent_work_id=None if parent_work is None else parent_work.work_id,
            )
            parent_work = work
            work_items.append(work)
            values.append((float(epoch), float(local)))
            work_count += 1
        admitted = admit_dark_sector_work(
            state,
            tuple(work.work_id for work in work_items),
            jnp.asarray(values),
        )
        result = finalize_dark_sector_epoch(
            state,
            admitted.state,
            complete=True,
            backpressured=admitted.backpressured,
            evidence_ids=(f"benchmark-epoch-{epoch}",),
        )
        receipt = coordinator.commit_epoch(
            result,
            work_items=tuple(work_items),
            matrix_element_revision_id=_digest("matrix-element"),
            committed_at=epoch + 1,
        )
        expected_tip_manifest_id = receipt.tip.epoch_manifest_id
        high_water = max(high_water, max(map(int, result.resident_high_water)))
    elapsed = time.perf_counter() - started
    byte_count = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    recovery_started = time.perf_counter()
    reopened_artifacts = POSIXArtifactRepository(root, _policy())
    reopened_graph = EventGraphRepository(
        reopened_artifacts, maximum_record_bytes=8 * 1024 * 1024
    )
    resumed = DarkSectorRunCoordinator(
        reopened_graph, plan, "benchmark-run", worker_id="benchmark-worker"
    ).resume()
    recovery_seconds = time.perf_counter() - recovery_started
    return {
        "epochs": epochs,
        "work_items": work_count,
        "seconds": elapsed,
        "epochs_per_second": epochs / elapsed,
        "work_items_per_second": work_count / elapsed,
        "repository_bytes": byte_count,
        "bytes_per_epoch": byte_count / epochs,
        "resident_high_water": high_water,
        "recovery_seconds": recovery_seconds,
        "recovered_epoch_sequence": resumed.state.epoch_sequence,
        "tip_manifest_id": resumed.state.parent_epoch_manifest_id,
        "expected_tip_manifest_id": expected_tip_manifest_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("epochs/repeats must be positive and warmup non-negative")

    for _ in range(arguments.warmup):
        with tempfile.TemporaryDirectory(prefix="phydrax-dark-sector-warmup-") as path:
            _execute(Path(path) / "repository", arguments.epochs)
    samples = []
    for _ in range(arguments.repeats):
        with tempfile.TemporaryDirectory(prefix="phydrax-dark-sector-benchmark-") as path:
            samples.append(_execute(Path(path) / "repository", arguments.epochs))

    mean_seconds = sum(float(sample["seconds"]) for sample in samples) / len(samples)
    mean_bytes = sum(int(sample["repository_bytes"]) for sample in samples) / len(samples)
    mean_recovery = sum(float(sample["recovery_seconds"]) for sample in samples) / len(
        samples
    )
    successful = all(
        sample["recovered_epoch_sequence"] == arguments.epochs
        and sample["resident_high_water"] <= 1
        and sample["tip_manifest_id"] == sample["expected_tip_manifest_id"]
        for sample in samples
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "epochs": arguments.epochs,
            "repeats": arguments.repeats,
            "resident_work_capacity": 1,
            "resident_frontier_capacity": 1,
        },
        "execution": {
            "seconds_mean": mean_seconds,
            "epochs_per_second_mean": arguments.epochs / mean_seconds,
            "repository_bytes_mean": mean_bytes,
            "bytes_per_epoch_mean": mean_bytes / arguments.epochs,
            "recovery_seconds_mean": mean_recovery,
            "resident_high_water": max(
                int(sample["resident_high_water"]) for sample in samples
            ),
        },
        "samples": samples,
        "successful": successful,
    }
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if arguments.output is None:
        print(text)
    else:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

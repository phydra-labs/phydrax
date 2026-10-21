# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Real cuDecomp qualification through the authenticated site-MPI launcher.

Set PHYDRAX_TEST_CUDECOMP_MPIEXEC to the exact qualified site Open MPI executable.
Absent opt-in is optional-provider nonqualification, never a physical pass.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from phydrax._execution_launch import (
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
)


_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "tests" / "_support" / "cuda_collective_worker.py"


def _launch(
    ranks: int, dtype: str, failure_case: str | None = None, *, benchmark: bool = False
) -> tuple[dict[str, Any], ...]:
    executable = os.environ.get("PHYDRAX_TEST_CUDECOMP_MPIEXEC")
    if executable is None:
        pytest.skip(
            "cuDecomp requires explicit site GPU-aware MPI and site-built CUDA13 components."
        )
    if not Path(executable).is_absolute() or not Path(executable).is_file():
        raise ValueError(
            "PHYDRAX_TEST_CUDECOMP_MPIEXEC must name the exact absolute site mpiexec."
        )
    with socket.socket() as coordinator:
        coordinator.bind(("127.0.0.1", 0))
        port = coordinator.getsockname()[1]
    arguments = [
        sys.executable,
        str(_WORKER),
        "--case",
        "cudecomp",
        "--dtype",
        dtype,
        "--rounds",
        "2",
        "--failure-timeout-seconds",
        "30",
    ]
    if failure_case is not None:
        arguments.extend(("--failure-case", failure_case))
    if benchmark:
        arguments.append("--benchmark")
    results = launch_local_processes(
        arguments,
        LocalProcessLaunchPlan(ranks, f"127.0.0.1:{port}", devices_per_process=1),
        mpi_launcher=(executable,),
        cwd=str(_ROOT),
        timeout_seconds=360,
        max_output_bytes=1024 * 1024,
        environment={
            **os.environ,
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
    )
    records: list[dict[str, Any]] = []
    for result in results:
        assert result.returncode == 0
        assert not result.output_truncated
        for line in result.stdout.decode("utf-8").splitlines():
            if line.startswith("{"):
                record = json.loads(line)
                if record.get("case") == "cudecomp":
                    records.append(record)
    assert sorted(record["rank"] for record in records) == list(range(ranks))
    assert all(record["process_count"] == ranks for record in records)
    assert len({record["group_id"] for record in records}) == 1
    assert len({record["supervisor_id"] for record in records}) == 1
    return tuple(records)


@pytest.mark.cuda_native
@pytest.mark.parametrize("ranks", (2, 4, 8), ids=("ranks2", "ranks4", "ranks8"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_real_cudecomp_permutation_ad_and_reconstructed_histories(
    ranks: int, dtype: str
) -> None:
    records = _launch(ranks, dtype)
    execution_ids: set[str] = set()
    numerical_ids: set[str] = set()
    for record in records:
        assert record["dtype"] == dtype
        rounds = record["rounds"]
        assert [entry["generation"] for entry in rounds] == [0, 1]
        assert rounds[0]["native_lease_id"] != rounds[1]["native_lease_id"]
        assert rounds[0]["lease_spec_id"] == rounds[1]["lease_spec_id"]
        for entry in rounds:
            assert entry["owner_thread_refused"] is True
            provenance = entry["mpi_provenance"]
            assert provenance["build_identity"]
            assert provenance["library_sha256"]
            assert provenance["header_sha256"]
            evidence = entry["evidence"]
            assert evidence["provider"] == "cudecomp"
            assert evidence["dtype"] == dtype
            assert evidence["coefficient_dtype"] == (
                "complex64" if dtype == "float32" else "complex128"
            )
            assert evidence["closed_reuse_refused"] is True
            assert evidence["exact_permutation"] is True
            assert evidence["zero_cotangent"] is True
            assert evidence["discarded_tangent"] is True
            assert evidence["higher_order_linear_ad"] is True
            assert evidence["composed_payload_history"] is True
            assert evidence["errors"]
            assert all(math.isfinite(value) for value in evidence["errors"].values())
            preparation = evidence["preparation"]
            assert preparation["external_device_scratch_bytes_per_rank"] > 0
            assert (
                preparation["external_device_scratch_bytes_per_rank"]
                <= preparation["maximum_external_scratch_bytes_per_rank"]
            )
            assert preparation["external_resident_device_bytes"] is None
            assert preparation["external_host_bytes"] is None
            execution_ids.add(evidence["execution_id"])
            numerical_ids.add(evidence["numerical_id"])
    assert len(execution_ids) == 1
    assert len(numerical_ids) == 1
    for generation in (0, 1):
        assert (
            len({record["rounds"][generation]["native_lease_id"] for record in records})
            == 1
        )


@pytest.mark.cuda_native
def test_cudecomp_benchmark_preserves_provider_and_scientific_gates() -> None:
    records = _launch(2, "float64", benchmark=True)
    for record in records:
        for entry in record["rounds"]:
            benchmark = entry["evidence"]["benchmark"]
            assert benchmark["campaign"] == "physical-slab"
            assert benchmark["transpose_provider"] == "cudecomp"
            assert benchmark["passed"] is True
            assert all(math.isfinite(value) for value in benchmark["defects"].values())
            assert benchmark["timing"]["plan_and_provider_preparation_seconds"] >= 0.0
            timing = benchmark["timing"]
            assert timing["program_staging_lowering_compilation_seconds"] >= 0.0
            assert (
                timing["inverse_program_preparation_timing"][
                    "program_staging_lowering_compilation_seconds"
                ]
                >= 0.0
            )
            assert timing["lowering_seconds"] is None
            assert timing["compilation_seconds"] is None
            assert (
                benchmark["reference"]["program_preparation_timing"][
                    "program_staging_lowering_compilation_seconds"
                ]
                >= 0.0
            )
            assert (
                benchmark["transpose_preparation_evidence"][
                    "external_device_scratch_bytes_per_rank"
                ]
                > 0
            )
            assert benchmark["reference"]["transpose_preparation_evidence"] is not None
            resource = benchmark["memory"]["logical_resource_report"]
            assert resource["external_device_scratch_bytes"] > 0
            assert resource["external_resident_device_bytes"] is None
            assert resource["external_host_bytes"] is None


@pytest.mark.cuda_native
@pytest.mark.parametrize(
    "failure_case", ("resource-refusal", "unsupported-pencil", "closed-lease-invoke")
)
def test_cudecomp_refuses_unadmitted_preparation_or_closed_lease(
    failure_case: str,
) -> None:
    records = _launch(2, "float64", failure_case)
    for record in records:
        for entry in record["rounds"]:
            evidence = entry["evidence"]
            assert evidence["closed_reuse_refused"] is True
            if failure_case != "closed-lease-invoke":
                assert evidence["case"] == failure_case
                assert evidence["refused"] is True


@pytest.mark.cuda_native
@pytest.mark.parametrize(
    "failure_case",
    (
        "external-scratch-refusal",
        "inactive-cached-call",
        "asymmetric-activity",
        "asymmetric-zero-cotangent",
        "configuration",
        "preparation",
    ),
)
def test_cudecomp_unsafe_entry_is_coupled_and_bounded(failure_case: str) -> None:
    with pytest.raises(LocalProcessLaunchError) as failure:
        _launch(2, "float32", failure_case)
    assert not failure.value.timed_out
    assert any(result.returncode != 0 for result in failure.value.results)

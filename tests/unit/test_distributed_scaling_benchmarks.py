#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded inputs and truthful hardware contracts of distributed drivers."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from benchmarks import distributed_mace_scaling as mace, distributed_spectral as spectral
from phydrax._execution_runtime import ExecutionRuntime
from phydrax.linalg.krylov import KrylovBreakdownStatus


def test_spectral_input_block_matches_global_formula() -> None:
    shape = (8, 12, 4)
    index = (slice(4, 8), slice(0, 12), slice(0, 4))
    grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    phase = 3.0 / grid.size
    expected = np.asarray(
        np.sin((grid + 1.0) * phase) + 1j * np.cos((grid + 0.5) * (phase + 0.125)),
        dtype=np.complex64,
    )
    observed = spectral._input_block(shape, index, 2, np.dtype(np.complex64))
    np.testing.assert_array_equal(observed, expected[index])
    assert observed.shape == (4, 12, 4)


def test_two_devices_do_not_qualify_physical_pencil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    with pytest.raises(ValueError, match="nondegenerate"):
        spectral._campaign_contract("physical-pencil", "pencil", (2, 1))


def test_cpu_inventory_does_not_qualify_physical_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    monkeypatch.setattr(
        spectral.jax, "devices", lambda: (SimpleNamespace(platform="cpu"),) * 2
    )
    with pytest.raises(ValueError, match="accelerator"):
        spectral._campaign_contract("physical-slab", "slab", (2,))


def test_spectral_mesh_must_include_every_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    monkeypatch.setattr(spectral.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        spectral.jax,
        "devices",
        lambda: (SimpleNamespace(platform="gpu", process_index=0),) * 2,
    )
    with pytest.raises(ValueError, match="every JAX process"):
        spectral._campaign_contract("physical-slab", "slab", (2,))


def test_mace_refuses_unimplemented_multi_process_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mace.jax, "process_count", lambda: 2)
    with pytest.raises(ValueError, match="single-process"):
        mace._row("fixed-total", 2, 16, 1, "devices", 1)


def test_mace_rejects_empty_campaign() -> None:
    with pytest.raises(ValueError, match="positive"):
        mace._row("fixed-total", 1, 0, 1, "devices", 1)


@pytest.mark.parametrize("owners", [1, 2, 3, 4, 5, 8])
def test_mace_device_subset_executes_owner_collective(owners: int) -> None:
    if jax.process_count() != 1:
        pytest.skip("The MACE benchmark requires single-process host ingress.")
    runtime = ExecutionRuntime.current()
    available = runtime.root_group.spec.device_count
    if available < owners:
        pytest.skip(f"Requires at least {owners} visible devices.")
    cell, _, _ = mace._structure(16, seed=16)
    plan = mace._plan(mace._model(1), cell, owners, 16, "devices")
    ownership = plan.ownership
    group = ownership.execution_group
    assert group.devices == runtime.root_group.devices[:owners]
    assert ownership.owner_count == owners
    assert not ownership.lane_reference
    if available % owners == 0:
        assert group.spec == runtime.child_groups(available // owners)[0].spec
    values = np.arange(owners * 16, dtype=np.float64)
    placed = ownership.place(values)
    assert placed.sharding.device_set == set(group.devices)

    def owner_sum(block: jax.Array) -> jax.Array:
        return jax.lax.psum(jnp.sum(block), ownership.axis_name)

    result = ownership.map(
        owner_sum, PartitionSpec(ownership.axis_name), PartitionSpec()
    )(placed)
    np.testing.assert_array_equal(np.asarray(result), np.sum(values))


def test_mace_device_subset_refuses_insufficient_devices() -> None:
    if jax.process_count() != 1:
        pytest.skip("The MACE benchmark requires single-process host ingress.")
    cell, _, _ = mace._structure(16, seed=16)
    with pytest.raises(ValueError, match="require at least .* visible devices"):
        mace._plan(mace._model(1), cell, jax.device_count() + 1, 16, "devices")


@pytest.mark.parametrize(
    ("component", "metric", "value"),
    [
        ("weighted_mean", "absolute_error", float("nan")),
        ("weighted_mean", "absolute_error", float("inf")),
        ("weighted_mean", "absolute_error", 1e-3),
        ("distributed_pcg", "maximum_solution_error", float("nan")),
        ("distributed_pcg", "maximum_solution_error", 1e-3),
        ("distributed_pcg", "converged", False),
        ("distributed_pcg", "breakdown", int(KrylovBreakdownStatus.NEAR_BREAKDOWN)),
        ("distributed_pcg", "breakdown", int(KrylovBreakdownStatus.NONFINITE_ACTION)),
        ("distributed_pcg", "breakdown", int(KrylovBreakdownStatus.STAGNATION)),
    ],
)
def test_substrate_scientific_failures_cannot_pass(
    component: str, metric: str, value: float | bool | int
) -> None:
    from tools.distributed_substrate_benchmark import qualification_errors

    record: dict[str, dict[str, float | int | bool]] = {
        "problem": {"group_identity_consistent": True},
        "weighted_mean": {
            "absolute_error": 0.0,
            "seconds": 0.01,
            "observed": 0.5,
            "expected": 0.5,
        },
        "distributed_pcg": {
            "maximum_solution_error": 0.0,
            "seconds": 0.01,
            "converged": True,
            "breakdown": int(KrylovBreakdownStatus.HAPPY),
            "residual_norm": 0.0,
            "residual_absolute_tolerance": 1e-12,
        },
    }
    record[component][metric] = value
    assert qualification_errors(record)


def test_rank_terminal_record_accepts_bootstrap_logs_without_discarding_the_result() -> (
    None
):
    from tools.distributed_substrate_benchmark import _terminal_record, RANK_RESULT_PREFIX

    expected = {"runtime": {"process_index": 2}, "passed": True, "errors": []}
    stdout = (
        b"[bootstrap] joining requested process set\n"
        + RANK_RESULT_PREFIX
        + json.dumps(expected).encode("utf-8")
        + b"\n"
    )
    assert _terminal_record(stdout) == expected


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_duplicate_rank_terminal_records_cannot_pass(count: int) -> None:
    from tools.distributed_substrate_benchmark import _terminal_record, RANK_RESULT_PREFIX

    terminal = RANK_RESULT_PREFIX + b'{"passed":true}\n'
    with pytest.raises(ValueError, match="exactly one terminal"):
        _terminal_record(b"[bootstrap] joined\n" + terminal * count)


@pytest.mark.parametrize("failure_mode", ("supervisor-failure", "unknown-exit"))
def test_successful_rank_records_cannot_hide_failed_process_supervision(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from phydrax.execution import LocalProcessLaunchError, LocalProcessResult
    from tools import distributed_substrate_benchmark as benchmark

    results = tuple(
        LocalProcessResult(
            rank,
            None if failure_mode == "unknown-exit" and rank == 1 else 0,
            benchmark.RANK_RESULT_PREFIX
            + json.dumps(
                {
                    "runtime": {
                        "process_index": rank,
                        "execution_group_id": "same-owned-group",
                    },
                    "passed": True,
                    "errors": [],
                }
            ).encode()
            + b"\n",
            b"",
            False,
        )
        for rank in range(2)
    )

    def launch(*_args: object, **_kwargs: object) -> tuple[LocalProcessResult, ...]:
        if failure_mode == "supervisor-failure":
            raise LocalProcessLaunchError(results, reason="supervision evidence lost")
        return results

    monkeypatch.setattr(benchmark, "launch_local_processes", launch)
    with pytest.raises(SystemExit) as failure:
        benchmark.main(["--local-processes", "2"])
    assert failure.value.code == 1
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["passed"] is False
    if failure_mode == "unknown-exit":
        assert record["rank_outputs"][1]["returncode"] is None
    else:
        assert any("supervision evidence lost" in error for error in record["errors"])

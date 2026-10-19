# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests._support import forced_devices
from tests._support.forced_devices import run_on_forced_devices


_MODULE = "tests._support.distributed_meshing.runner_contracts"


def test_forced_topology_precedes_jax_and_records_ignore_diagnostics() -> None:
    record = run_on_forced_devices(
        _MODULE,
        "record_probe",
        ("argument with spaces",),
        devices=2,
        env={
            "XLA_FLAGS": "--xla_force_host_platform_device_count=7",
            "PHYDRAX_TEST_RUNNER_TOKEN": "child-only",
        },
    )
    assert record == {
        "value": "argument with spaces",
        "devices": 2,
        "platform": "cpu",
        "extra_env": "child-only",
    }


def test_concurrent_ranks_have_separate_records() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(
                run_on_forced_devices,
                _MODULE,
                "record_probe",
                (str(rank),),
                devices=1,
                env={"PHYDRAX_TEST_RUNNER_TOKEN": str(rank)},
            )
            for rank in range(2)
        )
        records = tuple(future.result() for future in futures)
    assert records == (
        {"value": "0", "devices": 1, "platform": "cpu", "extra_env": "0"},
        {"value": "1", "devices": 1, "platform": "cpu", "extra_env": "1"},
    )


def test_child_failure_exposes_stdout_and_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(pytest.fail.Exception, match="fail_with_diagnostics"):
        run_on_forced_devices(_MODULE, "fail_with_diagnostics", (), devices=1, env={})
    captured = capsys.readouterr()
    assert "runner stdout evidence" in captured.out
    assert "runner stderr evidence" in captured.err
    assert "runner deliberate child refusal" in captured.err


def test_forced_topology_refuses_nonpositive_device_count() -> None:
    with pytest.raises(ValueError, match="forced_devices.devices"):
        run_on_forced_devices(_MODULE, "record_probe", ("0",), devices=0, env={})


def test_child_deadline_refuses_without_a_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forced_devices, "_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(pytest.fail.Exception, match="timed out: True"):
        run_on_forced_devices(
            _MODULE,
            "record_probe",
            ("deadline",),
            devices=1,
            env={"PHYDRAX_TEST_RUNNER_TOKEN": "deadline"},
        )


def test_explicit_child_deadline_retains_the_callers_bound() -> None:
    with pytest.raises(pytest.fail.Exception, match="timed out: True"):
        run_on_forced_devices(
            _MODULE,
            "record_probe",
            ("deadline",),
            devices=1,
            env={"PHYDRAX_TEST_RUNNER_TOKEN": "deadline"},
            timeout_seconds=0.001,
        )

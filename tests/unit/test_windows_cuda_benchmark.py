# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Behavioral campaign tests; synthetic records never constitute GPU evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

import jax
import numpy as np
import pytest

from benchmarks import windows_cuda as campaign
from phydrax import config as pxconfig
from phydrax._cuda_native import CudaNativeComponentName


@pytest.mark.parametrize("actual,expected", [(np.nan, 1), (np.inf, np.inf), (2, 1)])
def test_agreement_refuses_incorrect_or_nonfinite_values(
    actual: float, expected: float
) -> None:
    with pytest.raises((ValueError, AssertionError)):
        campaign._agree(actual, expected, 1e-8)


def test_diagnostic_serialization_preserves_native_status() -> None:
    @dataclass
    class Diagnostic:
        native_status: object
        residual: object

    payload = campaign._payload(Diagnostic(np.array([0, 7]), np.array([0.1, 2.0])))
    assert json.loads(json.dumps(payload)) == {
        "native_status": [0, 7],
        "residual": [0.1, 2.0],
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("route", ["fft", "dense-solve", "sparse-apply"])
def test_ordinary_scientific_factory_and_derivative_oracles(
    dtype: type[np.float32] | type[np.float64] | type[np.complex64] | type[np.complex128],
    route: str,
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        case = campaign._dense_or_fft(8, dtype, route)
        case.check(case.function(*case.arguments))
        assert case.derivative is not None and case.derivative_check is not None
        case.derivative_check(case.derivative(*case.arguments))


def test_bad_first_result_never_reaches_warm_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks import _runtime

    def refuse(value: object) -> Never:
        raise ValueError("bad correctness")

    case = campaign.Case("bad", 1, lambda x: x, (np.ones(1),), refuse)
    monkeypatch.setattr(
        _runtime,
        "measure_lower_and_compile",
        lambda *args: (
            lambda *values: np.ones(1),
            SimpleNamespace(lowering_seconds=1, compilation_seconds=2),
        ),
    )
    monkeypatch.setattr(
        _runtime,
        "measure_repeated",
        lambda *args, **kwargs: pytest.fail("warm before correctness"),
    )
    with pytest.raises(ValueError, match="bad correctness"):
        campaign._measure(case, 1, 1)


def _mock_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[campaign.Case], list[str]]:
    from benchmarks import _runtime
    from phydrax import _cuda_native
    from tests._support import cuda_execution_worker
    from tests.unit import test_compensated_reductions
    from tests.unit.discretization import test_mpm_apic_streaming_moments

    monkeypatch.setattr(sys, "platform", "win32")
    device = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(jax, "local_devices", lambda: [device])
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(cuda_execution_worker, "cuda_runtime", lambda: object())
    monkeypatch.setattr(cuda_execution_worker, "inventory", lambda _: {"synthetic": True})
    monkeypatch.setattr(
        _runtime,
        "capture_environment",
        lambda: SimpleNamespace(to_dict=lambda: {"synthetic": True}),
    )
    monkeypatch.setattr(
        _runtime, "measure_synchronized", lambda operation: (operation(), 0.125)
    )
    for owner, names in (
        (
            test_compensated_reductions,
            (
                "test_compensated_contracts",
                "test_long_reduction_nonfinite_lanes_remain_explicit",
                "test_seeded_recurrence_preserves_reference_components",
            ),
        ),
        (
            test_mpm_apic_streaming_moments,
            ("test_masks_nonfinite_metadata_and_zero_weights",),
        ),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, lambda *args: None)
    monkeypatch.setattr(campaign, "_ordinary_cases", lambda: [])

    def fixture(
        name: str, size: int, provider: CudaNativeComponentName | None = None
    ) -> campaign.Case:
        return campaign.Case(
            name, size, lambda: None, (), lambda _: None, provider=provider
        )

    monkeypatch.setattr(
        campaign, "_dense_or_fft", lambda n, dtype, route: fixture(route, n)
    )
    monkeypatch.setattr(
        campaign, "_compensated", lambda n, dtype: fixture("compensated", n)
    )
    monkeypatch.setattr(campaign, "_mpm", lambda n, dtype: fixture("mpm", n))
    monkeypatch.setattr(
        campaign, "_solver", lambda provider, n, dtype: fixture(provider, n, provider)
    )
    monkeypatch.setattr(
        campaign, "_quda", lambda n, dtype, operation: fixture(operation, n, "quda")
    )
    measured: list[campaign.Case] = []
    released: list[str] = []

    def measure(
        case: campaign.Case, preparation: float | None, repeats: int
    ) -> dict[str, Any]:
        measured.append(case)
        return {"name": case.name, "passed": True}

    monkeypatch.setattr(campaign, "_measure", measure)
    monkeypatch.setattr(_cuda_native, "release_cuda_component", released.append)
    return measured, released


def test_selected_optional_failure_cannot_become_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, released = _mock_campaign(monkeypatch)

    def fail(*args: object) -> Never:
        raise RuntimeError("cutensor failed")

    monkeypatch.setattr(campaign, "_cutensor", fail)
    record: dict[str, Any] = {"cases": [], "passed": False}
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        pytest.raises(RuntimeError, match="cutensor failed"),
    ):
        campaign.campaign(1, True, record)
    assert not record["passed"]
    assert released[-1] == "cutensor"


def test_failed_native_measurement_still_releases_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, released = _mock_campaign(monkeypatch)

    def measure(case: campaign.Case, *args: object) -> dict[str, bool]:
        if case.provider == "cudss":
            raise RuntimeError("native failure")
        return {"passed": True}

    monkeypatch.setattr(campaign, "_measure", measure)
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        pytest.raises(RuntimeError, match="native failure"),
    ):
        campaign.campaign(1, False, {"cases": [], "passed": False})
    assert released == ["cudss"]


@pytest.mark.parametrize("repeats", ["0", "21"])
def test_campaign_cli_enforces_bounded_repetitions(tmp_path: Path, repeats: str) -> None:
    with pytest.raises(SystemExit):
        campaign.main(
            [
                "--python",
                str(tmp_path / "python.exe"),
                "--work-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
                "--repeats",
                repeats,
            ]
        )


@pytest.mark.parametrize("returncode,timed_out", [(1, False), (0, True), (None, False)])
def test_launcher_cannot_promote_child_json_over_process_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int | None,
    timed_out: bool,
) -> None:
    from tools import _accelerator_process, qualify_cuda_wheels

    monkeypatch.setattr(
        qualify_cuda_wheels, "copy_harness", lambda _: {"worker.py": "synthetic-digest"}
    )
    monkeypatch.setattr(qualify_cuda_wheels, "clean_environment", lambda **_: {})

    def run(command: list[str], **kwargs: object) -> dict[str, int | bool | None]:
        return {"returncode": returncode, "timed_out": timed_out}

    monkeypatch.setattr(_accelerator_process, "run", run)
    monkeypatch.setattr(
        _accelerator_process,
        "read_json_evidence",
        lambda _: {"passed": True, "errors": []},
    )
    python = tmp_path / "Scripts" / "python.exe"
    python.parent.mkdir()
    python.touch()
    output = tmp_path / "campaign.json"
    assert (
        campaign.main(
            [
                "--python",
                str(python),
                "--work-root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    record = json.loads(output.read_text())
    assert not record["passed"]
    assert record["harness_sha256"] == {"worker.py": "synthetic-digest"}

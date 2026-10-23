# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free TPU lowering preflight contracts (not TPU hardware evidence)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from tools.check_tpu_lowering import equal_check, run_scenario, Scenario, ScenarioSpec


ROOT = Path(__file__).resolve().parents[2]


def _double(value: Any) -> dict[str, Any]:
    return {"value": 2 * value}


def _build_double(scenario: Scenario) -> None:
    result = scenario.run("double", _double, np.arange(3, dtype=np.float32))
    scenario.check(
        equal_check("doubled", result["value"], np.asarray([0, 2, 4], np.float32))
    )


def test_expected_refusal_that_exports_is_a_mismatch() -> None:
    report = run_scenario(
        ScenarioSpec("double", "test", {"double": "host-callback"}, _build_double)
    )
    target = report["tpu_export"]["targets"]["double"]
    assert report["cpu_reference"]["passed"]
    assert target["outcome"] == "exported" and target["platforms"] == ["tpu"]
    assert not target["matched"] and not report["successful"]
    assert report["tpu_export"]["counts"]["mismatched"] == 1
    assert report["tpu_export"]["counts"]["matched_refusals"] == 0


def test_float64_promotion_fails_an_exported_float32_route() -> None:
    def promoted(value: Any) -> dict[str, Any]:
        return {"value": jnp.sum(value.astype(jnp.float64)).astype(jnp.float32)}

    def build(scenario: Scenario) -> None:
        result = scenario.run("promoted", promoted, np.ones(4, np.float32))
        scenario.check(equal_check("sum", result["value"], np.float32(4.0)))

    report = run_scenario(
        ScenarioSpec("promoted", "test", {"promoted": "exported"}, build)
    )
    target = report["tpu_export"]["targets"]["promoted"]
    assert report["cpu_reference"]["passed"]
    assert target["outcome"] == "exported"
    assert "convert_element_type" in target["strong_float64_primitives"]
    assert not target["matched"] and not report["successful"]


def test_weak_scalar_literals_are_not_float64_work() -> None:
    def clipped(value: Any) -> dict[str, Any]:
        return {"value": jnp.where(value > 0, value, 0.0)}

    def build(scenario: Scenario) -> None:
        result = scenario.run("clipped", clipped, np.asarray([-1.0, 2.0], np.float32))
        scenario.check(
            equal_check("clipped", result["value"], np.asarray([0, 2], np.float32))
        )

    report = run_scenario(ScenarioSpec("clipped", "test", {"clipped": "exported"}, build))
    target = report["tpu_export"]["targets"]["clipped"]
    assert target["strong_float64_primitives"] == []
    assert target["matched"] and report["successful"] and report["fully_exported"]


def test_unbuilt_declared_target_is_reported_not_attempted() -> None:
    def build(scenario: Scenario) -> None:
        _build_double(scenario)
        raise RuntimeError("reference stage stopped")

    report = run_scenario(
        ScenarioSpec(
            "partial", "test", {"double": "exported", "later": "exported"}, build
        )
    )
    targets = report["tpu_export"]["targets"]
    assert "RuntimeError" in report["cpu_reference"]["error"]["type"]
    assert not report["cpu_reference"]["passed"] and not report["successful"]
    assert targets["double"]["outcome"] == "exported" and targets["double"]["matched"]
    assert targets["later"]["outcome"] == "not-attempted"
    assert report["tpu_export"]["counts"]["not_attempted"] == 1


def test_failed_cpu_check_fails_scenario_with_matching_exports() -> None:
    def build(scenario: Scenario) -> None:
        result = scenario.run("double", _double, np.arange(3, dtype=np.float32))
        scenario.check(equal_check("wrong", result["value"], np.zeros(3, np.float32)))

    report = run_scenario(ScenarioSpec("double", "test", {"double": "exported"}, build))
    assert report["tpu_export"]["counts"]["mismatched"] == 0
    assert not report["cpu_reference"]["passed"] and not report["successful"]


def test_callback_refusal_cannot_satisfy_a_required_export() -> None:
    # Callback lowering asks for a target client, so CPU selection is forced in
    # a dedicated process before any backend exists.
    source = """
import json
import jax
import numpy as np
import phydrax as phx
phx.config.configure(phx.config.RuntimeConfig(platforms=("cpu",)))
from tools.check_tpu_lowering import run_scenario, ScenarioSpec

def observed(value):
    jax.debug.callback(lambda _: None, value, ordered=True)
    return {"value": value + 1}

def build(scenario):
    scenario.run("observed", observed, np.int32(1))

print(json.dumps(run_scenario(ScenarioSpec("callback", "test", {"observed": "exported"}, build))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    report = json.loads(completed.stdout.splitlines()[-1])
    target = report["tpu_export"]["targets"]["observed"]
    assert report["cpu_reference"]["passed"]
    assert target["outcome"] == "host-callback"
    assert target["callback_primitives"] == ["debug_callback"]
    assert not target["matched"] and not report["successful"]


def test_cli_exports_sparse_routes_and_declares_callback_limitation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.check_tpu_lowering",
            "--output",
            str(output),
            "--scenario",
            "sparse-route-gather-reduce",
            "--scenario",
            "ordered-host-callback",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(output.read_text())
    assert report["kind"] == "phydrax-tpu-lowering-preflight"
    assert report["successful"] and report["refusal"] is None
    assert not report["hardware_qualified"] and not report["tpu_executed"]
    assert report["environment"]["configuration"]["platforms"] == ["cpu"]
    assert report["environment"]["observed_device_platforms"] == ["cpu"]
    sparse = report["scenarios"]["sparse-route-gather-reduce"]["tpu_export"]
    assert sparse["counts"]["matched_exports"] == sparse["counts"]["declared"] == 6
    for target in sparse["targets"].values():
        assert target["platforms"] == ["tpu"] and target["stablehlo_bytes"] > 0
        assert target["disabled_safety_checks"] == []
    callbacks = report["scenarios"]["ordered-host-callback"]["tpu_export"]
    assert callbacks["targets"]["ordered-callback-loop"]["outcome"] == "host-callback"
    assert callbacks["targets"]["callback-free-loops"]["outcome"] == "exported"
    assert report["counts"]["matched_refusals"] == report["counts"]["refused"] == 1
    assert report["counts"]["required_exports"] == report["counts"]["matched_exports"]
    assert report["scenarios"]["sparse-route-gather-reduce"]["fully_exported"]
    assert not report["scenarios"]["ordered-host-callback"]["fully_exported"]
    assert not report["fully_exported"]

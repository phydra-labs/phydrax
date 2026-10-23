# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free evidence parser/refusal policy tests, never qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pytest

from tools import qualify_accelerator_wheels as qualification
from tools._accelerator_process import (
    read_json_evidence,
    run,
    TEXT_LIMIT,
    write_json_evidence,
)
from tools.check_accelerator_extras import Accelerator
from tools.qualify_accelerator_wheels import (
    case_report_errors,
    clean_environment,
    host_errors,
    last_json,
    probe_record,
    REQUIRED_CASES,
    resource_record_errors,
    RESOURCE_REGIONS,
    ROCM_CASES,
    ROCM_ENVIRONMENT,
    runtime_errors,
    wheel_files,
)


NONCE = "0123456789abcdef0123456789abcdef"


def _runtime(accelerator: Accelerator = Accelerator.TPU) -> dict[str, object]:
    return {
        "selected": accelerator.value,
        "rocm_major": accelerator.rocm_major,
        "platform": accelerator.runtime_platform,
        "client_platform": accelerator.runtime_platform,
        "platform_version": (
            f"rocm {accelerator.rocm_major}.1.0"
            if accelerator.rocm_major is not None
            else "upstream-runtime-test-parser-input"
        ),
        "device_kind": "parser-input-not-hardware-evidence",
        "qualified_device_id": 0,
        "execution_device_count": 1,
        "process_index": 0,
        "process_count": 1,
        "global_device_count": 8,
        "local_device_count": 8,
        "x64_enabled": True,
        "matmul_precision": "highest",
        "frontend_index_patch": None,
    }


def _measurement(scope: str = "baseline") -> dict[str, object]:
    return {
        "method": "jax_memory_stats",
        "scope": scope,
        "value_bytes": None,
        "resolution_bytes": None,
        "unavailable_reason": "parser input: provider exposes no allocator statistics",
    }


def _phase(phase: str, runtime: dict[str, object]) -> dict[str, Any]:
    sample = {
        "process_index": 0,
        "device_id": runtime["qualified_device_id"],
        "platform": runtime["platform"],
        "provider_id": runtime["client_platform"],
        "provider_version": runtime["platform_version"],
        "timestamp_ns": 1,
        "allocator_limit": _measurement(),
        "allocator_reserved": _measurement(),
        "allocator_in_use": _measurement(),
        "allocator_peak_in_use": _measurement("peak"),
        "live_array_bytes": {
            "method": "jax_live_arrays",
            "scope": "baseline",
            "value_bytes": 8,
            "resolution_bytes": 1,
            "unavailable_reason": None,
        },
    }
    return {
        "phase": phase,
        "interval_seconds": 0.01,
        "duration_seconds": 0.001,
        "sample_count": 2,
        "maximum_sample_gap_seconds": 0.001,
        "sampled_peak_resident": {
            "method": "interval_sampling",
            "scope": "peak",
            "value_bytes": 4096,
            "resolution_bytes": 4096,
            "unavailable_reason": None,
        },
        "devices": [
            {
                "baseline": sample,
                "final": json.loads(json.dumps(sample)),
                "sampled_peak_in_use": {
                    **_measurement("peak"),
                    "method": "interval_sampling",
                },
            }
        ],
    }


def _resource(region: str, runtime: dict[str, object] | None = None) -> dict[str, Any]:
    """Synthetic parser input shaped like the recorder, never device evidence."""
    runtime = runtime or _runtime()
    signature = [{"shape": [2], "dtype": "float32", "weak_type": False}]
    return {
        "region": region,
        "evidence_nonce": NONCE,
        "runtime": runtime,
        "program_sha256": "6" * 64,
        "input_signature": signature,
        "output_signature": signature,
        "lowering_seconds": 0.01,
        "compilation_seconds": 0.02,
        "first_execution_seconds": 0.003,
        "warm_execution": {
            "count": 1,
            "samples_seconds": [0.001],
            "min_seconds": 0.001,
            "median_seconds": 0.001,
            "mean_seconds": 0.001,
            "std_seconds": 0.0,
            "max_seconds": 0.001,
        },
        "logical_retained_bytes": 8,
        "compiled_memory": {
            "argument_bytes": 8,
            "output_bytes": 8,
            "alias_bytes": 0,
            "temporary_bytes": 16,
            "generated_code_bytes": 0,
            "host_argument_bytes": 0,
            "host_output_bytes": 0,
            "host_alias_bytes": 0,
            "host_temporary_bytes": 0,
            "host_generated_code_bytes": 0,
            "peak_bytes": None,
        },
        "compiled_memory_unavailable_reason": None,
        "phase_memory": {
            "first_execution": _phase("first-execution", runtime),
            "warm_execution": _phase("warm-execution", runtime),
        },
    }


def _xml(
    path: Path,
    names: tuple[str, ...] = REQUIRED_CASES,
    *,
    outcome: str | None = None,
    runtime: dict[str, object] | None = None,
    evidence: bool = True,
    resources: dict[str, list[object]] | None = None,
) -> None:
    suite = ElementTree.Element("testsuite")
    for name in names:
        node = ElementTree.SubElement(
            suite,
            "testcase",
            name=name,
            classname="tests.integration.test_accelerator_scientific_execution",
            time="0.1",
        )
        if outcome is not None:
            ElementTree.SubElement(node, outcome)
        if evidence:
            properties = ElementTree.SubElement(node, "properties")
            ElementTree.SubElement(
                properties,
                "property",
                name="accelerator_runtime",
                value=json.dumps(runtime or _runtime()),
            )
            records = (resources or {}).get(
                name,
                [_resource(region, runtime) for region in RESOURCE_REGIONS.get(name, ())],
            )
            for record in records:
                ElementTree.SubElement(
                    properties,
                    "property",
                    name="accelerator_resources",
                    value=record if isinstance(record, str) else json.dumps(record),
                )
    ElementTree.ElementTree(suite).write(path)


def test_report_requires_exact_successful_case_set_and_runtime(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report)
    errors, cases = case_report_errors(report, _runtime(), NONCE)
    assert errors == [] and len(cases) == len(REQUIRED_CASES)


def _break_resource(failure: str) -> list[object]:
    record = _resource("bounded-pcg-solve")
    warm = record["phase_memory"]["warm_execution"]
    device = warm["devices"][0]
    match failure:
        case "missing":
            return []
        case "duplicate":
            return [record, record]
        case "unexpected-region":
            return [record, _resource("fourier-resample")]
        case "malformed":
            return ["{not json"]
        case "stale-nonce":
            record["evidence_nonce"] = "f" * 32
        case "unbound-nonce":
            record["evidence_nonce"] = None
        case "cpu-runtime":
            record = _resource(
                "bounded-pcg-solve",
                {**_runtime(), "platform": "cpu", "client_platform": "cpu"},
            )
        case "other-device-sample":
            device["final"]["device_id"] = 1
        case "cpu-device-sample":
            device["baseline"]["platform"] = "cpu"
        case "unsampled-device":
            warm["devices"] = []
        case "missing-phase":
            del record["phase_memory"]["first_execution"]
        case "no-compiled-memory":
            record["compiled_memory"] = None
            record["compiled_memory_unavailable_reason"] = "provider exposes none"
        case "missing-lowering":
            record["lowering_seconds"] = None
        case "nan-execution":
            record["first_execution_seconds"] = float("nan")
        case "no-warm-samples":
            record["warm_execution"] = {
                **record["warm_execution"],
                "count": 0,
                "samples_seconds": [],
            }
        case "unexplained-unavailable":
            device["baseline"]["allocator_in_use"]["unavailable_reason"] = None
        case "missing-signature":
            record["input_signature"] = []
        case "missing-retained-bytes":
            del record["logical_retained_bytes"]
        case "inconsistent-warm-summary":
            record["warm_execution"] = {
                "count": 2,
                "samples_seconds": [0.001, 0.003],
                "min_seconds": 0.001,
                "median_seconds": 0.003,
                "mean_seconds": 0.002,
                "std_seconds": 0.001,
                "max_seconds": 0.003,
            }
        case "nan-warm-summary":
            record["warm_execution"]["mean_seconds"] = float("nan")
        case "other-process-sample":
            device["final"]["process_index"] = 1
        case "foreign-provider-sample":
            device["baseline"]["provider_id"] = "cpu"
        case "unhashable-region":
            record["region"] = ["bounded-pcg-solve"]
    return [record]


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "duplicate",
        "unexpected-region",
        "malformed",
        "stale-nonce",
        "unbound-nonce",
        "cpu-runtime",
        "other-device-sample",
        "cpu-device-sample",
        "unsampled-device",
        "missing-phase",
        "no-compiled-memory",
        "missing-lowering",
        "nan-execution",
        "no-warm-samples",
        "unexplained-unavailable",
        "missing-signature",
        "missing-retained-bytes",
        "inconsistent-warm-summary",
        "nan-warm-summary",
        "other-process-sample",
        "foreign-provider-sample",
        "unhashable-region",
    ),
)
def test_resource_gate_refuses_absent_stale_or_foreign_evidence(
    tmp_path: Path, failure: str
) -> None:
    report = tmp_path / "report.xml"
    case = "test_bounded_linear_solve"
    _xml(report, resources={case: _break_resource(failure)})
    errors, _ = case_report_errors(report, _runtime(), NONCE)
    assert errors and all(error.startswith(case) for error in errors)


def test_cpu_recorder_satisfies_gate_only_for_its_own_runtime_and_run() -> None:
    """Exercise the real recorder on CPU; neither accelerator nor performance evidence."""
    import jax
    import jax.numpy as jnp

    from tests._support.accelerator_qualification import compiled_region_evidence

    device = jax.devices("cpu")[0]
    gradient, evidence = compiled_region_evidence(
        jax.grad(lambda point: point[0] ** 3 * point[1] ** 2),
        (jnp.asarray([0.5, 0.25], dtype=jnp.float32),),
        device=device,
    )
    np.testing.assert_array_equal(gradient, np.asarray([0.046875, 0.0625], np.float32))
    assert gradient.devices() == {device}
    runtime = {
        **_runtime(),
        "platform": device.platform,
        "client_platform": device.client.platform,
        "platform_version": device.client.platform_version,
        "qualified_device_id": device.id,
    }
    record = json.loads(
        json.dumps(
            {
                "region": "coordinate-gradient",
                "evidence_nonce": NONCE,
                "runtime": runtime,
                **evidence,
            }
        )
    )
    assert resource_record_errors(record, runtime, NONCE) == []
    assert resource_record_errors(record, _runtime(), NONCE)
    assert resource_record_errors(record, runtime, "f" * 32)


@pytest.mark.parametrize("outcome", ("skipped", "failure", "error"))
def test_skipped_or_failed_cases_never_qualify(tmp_path: Path, outcome: str) -> None:
    report = tmp_path / "report.xml"
    _xml(report, outcome=outcome)
    assert case_report_errors(report, _runtime(), NONCE)[0]


@pytest.mark.parametrize(
    "names",
    (
        REQUIRED_CASES[:-1],
        (*REQUIRED_CASES, REQUIRED_CASES[0]),
        (*REQUIRED_CASES, "unexpected"),
    ),
)
def test_missing_duplicate_or_extra_case_refuses(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    report = tmp_path / "report.xml"
    _xml(report, names)
    assert case_report_errors(report, _runtime(), NONCE)[0]


def test_missing_or_wrong_actual_placement_refuses(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report, evidence=False)
    assert case_report_errors(report, _runtime(), NONCE)[0]
    _xml(report, runtime={**_runtime(), "platform": "cpu"})
    assert case_report_errors(report, _runtime(), NONCE)[0]
    report.write_text("not xml")
    assert case_report_errors(report, _runtime(), NONCE)[0]
    assert case_report_errors(tmp_path / "absent", _runtime(), NONCE)[0]


@pytest.mark.parametrize("accelerator", tuple(Accelerator))
def test_cpu_another_gpu_and_absent_runtime_are_refused(accelerator: Accelerator) -> None:
    for platform in ("cpu", "cuda", "mps"):
        assert runtime_errors(
            {**_runtime(accelerator), "platform": platform, "client_platform": platform},
            accelerator,
        )
    assert runtime_errors({}, accelerator)
    assert not runtime_errors(_runtime(accelerator), accelerator)
    assert runtime_errors({**_runtime(accelerator), "x64_enabled": False}, accelerator)


def test_oneapi_generic_gpu_requires_upstream_oneapi_identity() -> None:
    assert not runtime_errors(
        {**_runtime(Accelerator.ONEAPI), "platform": "gpu"}, Accelerator.ONEAPI
    )
    assert runtime_errors(
        {**_runtime(Accelerator.ONEAPI), "platform": "gpu", "client_platform": "cuda"},
        Accelerator.ONEAPI,
    )


def test_tpu_highest_is_explicit_and_not_inferred_from_x64() -> None:
    assert runtime_errors({**_runtime(), "matmul_precision": "default"}, Accelerator.TPU)


def test_environment_excludes_provider_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "LIBTPU_INIT_ARGS",
        "ONEAPI_DEVICE_SELECTOR",
    ):
        monkeypatch.setenv(name, "hostile-inherited-value")
    environment = clean_environment(tmp_path, Accelerator.TPU)
    assert environment["JAX_PLATFORMS"] == "tpu"
    assert environment["JAX_ENABLE_X64"] == "1"
    assert (
        not {
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "XLA_FLAGS",
            "LIBTPU_INIT_ARGS",
            "ONEAPI_DEVICE_SELECTOR",
        }
        & environment.keys()
    )
    assert host_errors("Darwin", "arm64") and host_errors("Linux", "aarch64")
    assert not host_errors("Linux", "x86_64")


def test_multi_device_host_qualifies_only_one_explicit_execution_device() -> None:
    assert runtime_errors(_runtime(), Accelerator.TPU) == []
    assert runtime_errors({**_runtime(), "execution_device_count": 8}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "process_count": 2}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "local_device_count": 0}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "qualified_device_id": None}, Accelerator.TPU)


@pytest.mark.parametrize("accelerator", (Accelerator.ROCM10,))
def test_rocm_runtime_major_is_not_inferred_from_generic_gpu(
    accelerator: Accelerator,
) -> None:
    record = {**_runtime(accelerator), "platform": "gpu", "client_platform": "rocm"}
    assert not runtime_errors(record, accelerator)
    for runtime in ("CUDA 13.0", "HIP 7.1", "unknown"):
        assert runtime_errors({**record, "platform_version": runtime}, accelerator)
    assert runtime_errors({**record, "rocm_major": 99}, accelerator)
    # HIP numeric version does not identify the packaged SDK.
    packed = "rocm 71626385"
    assert not runtime_errors({**record, "platform_version": packed}, accelerator)


def test_qualification_records_every_selected_device_without_first_device_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", "qualification fixture")
    monkeypatch.setattr(qualification.platform, "system", Mock(return_value="Linux"))
    monkeypatch.setattr(qualification.platform, "machine", Mock(return_value="x86_64"))
    monkeypatch.setattr(qualification, "wheel_metadata", Mock(return_value=Message()))
    monkeypatch.setattr(qualification, "metadata_errors", Mock(return_value=[]))
    monkeypatch.setattr(qualification, "_copy_harness", Mock())
    run_command = Mock(return_value={"returncode": 0, "timed_out": False})
    monkeypatch.setattr(qualification, "run", run_command)
    per_device = [
        {
            "selected_device_id": device_id,
            "errors": ["actual refusal"] if device_id == 2 else [],
            "hardware_qualified": device_id != 2,
            "installed": {"parser_input": "not-hardware-evidence"},
        }
        for device_id in range(4)
    ]
    execute = Mock(side_effect=per_device)
    monkeypatch.setattr(qualification, "_qualify_device", execute)
    args = argparse.Namespace(
        accelerator="rocm10",
        phydrax_wheel=wheel,
        work_dir=tmp_path / "work",
        device_id=[0, 1, 2, 3],
        tpu_autodetect=False,
        uv="uv",
        python=Path(sys.executable),
        timeout=10,
        find_links=[tmp_path / "wheelhouse"],
    )
    monkeypatch.setattr(qualification, "test_requirements", Mock(return_value=[]))
    report = qualification.qualify(args)
    assert report["selected_device_ids"] == [0, 1, 2, 3]
    assert report["devices"] == per_device
    assert [call.args[-1] for call in execute.call_args_list] == [0, 1, 2, 3]
    assert report["hardware_qualified"] is False
    assert report["static_architecture_coverage"] == ["gfx1250", "gfx942", "gfx950"]
    assert report["static_assets_are_hardware_qualification"] is False
    assert report["errors"] == ["Device 2: actual refusal"]
    install = run_command.call_args_list[1]
    command = install.args[0]
    assert command[command.index("--find-links") + 1] == str(tmp_path / "wheelhouse")
    assert "--no-config" in command
    assert "--refresh" in command
    assert not set(ROCM_ENVIRONMENT) & install.kwargs["environment"].keys()
    assert not any("INDEX" in key for key in install.kwargs["environment"])


@pytest.mark.parametrize(
    "accelerator,platforms",
    (
        (Accelerator.ONEAPI, "oneapi"),
        (Accelerator.TPU, "tpu"),
        (Accelerator.ROCM10, "rocm,cpu"),
    ),
)
def test_host_callback_backend_does_not_change_scientific_device_selection(
    tmp_path: Path,
    accelerator: Accelerator,
    platforms: str,
) -> None:
    assert clean_environment(tmp_path, accelerator)["JAX_PLATFORMS"] == platforms
    assert runtime_errors(
        {**_runtime(accelerator), "platform": "cpu", "client_platform": "cpu"},
        accelerator,
    )


def test_rocm_environment_strips_all_manual_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (*ROCM_ENVIRONMENT, "UV_INDEX_URL", "PYTHONPATH", "XLA_FLAGS"):
        monkeypatch.setenv(name, f"configured-{name}")
    environment = clean_environment(tmp_path, Accelerator.ROCM10)
    assert environment["JAX_PLATFORMS"] == "rocm,cpu"
    assert environment["PHYDRAX_ACCELERATOR"] == "rocm10"
    for name in ROCM_ENVIRONMENT:
        assert name not in environment
        assert name not in clean_environment(tmp_path, None)
    assert not {"UV_INDEX_URL", "PYTHONPATH", "XLA_FLAGS"} & environment.keys()


@pytest.mark.parametrize("accelerator", (Accelerator.ROCM10,))
def test_rocm_report_requires_native_precision_on_each_selected_device(
    tmp_path: Path,
    accelerator: Accelerator,
) -> None:
    path = tmp_path / "cases.xml"
    runtime = {**_runtime(accelerator), "qualified_device_id": 3}
    assert len(REQUIRED_CASES + ROCM_CASES) == 20
    _xml(path, REQUIRED_CASES, runtime=runtime)
    assert case_report_errors(path, runtime, NONCE)[0]
    _xml(path, REQUIRED_CASES + ROCM_CASES, runtime=runtime)
    assert not case_report_errors(path, runtime, NONCE)[0]
    assert case_report_errors(path, {**runtime, "qualified_device_id": 0}, NONCE)[0]


def test_core_wheel_content_identity_changes_with_bytes(tmp_path: Path) -> None:
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", "first")
        archive.writestr("phydrax-0.2.2.dist-info/RECORD", "mutable installer record")
    first = wheel_files(wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", "second")
    assert first["phydrax/__init__.py"] != wheel_files(wheel)["phydrax/__init__.py"]
    assert not any(name.endswith("/RECORD") for name in first)


def test_failed_child_or_non_json_never_yields_runtime() -> None:
    assert last_json({"returncode": 1, "stdout": '{"status":"passed"}'}) is None
    assert last_json({"returncode": 0, "stdout": "no evidence"}) is None
    assert last_json({"returncode": 0, "stdout": "[]"}) is None
    assert last_json({"returncode": 0, "stdout": 'log\n{"observed":true}'}) == {
        "observed": True
    }


def _probe_payload(accelerator: Accelerator = Accelerator.TPU) -> dict[str, object]:
    distributions: dict[str, object] = {}
    for name, version in {"phydrax": "0.2.2", **accelerator.distributions}.items():
        files = {
            f"package/member-{index}.so": hashlib.sha256(str(index).encode()).hexdigest()
            for index in range(10_000 if name == "phydrax" else 1)
        }
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        distributions[name] = {
            "version": version,
            "files": files,
            "installed_sha256": digest,
        }
    payload = {
        "provenance": {
            "package": "/parser-input/venv/phydrax/__init__.py",
            "venv": "/parser-input/venv",
            "distributions": distributions,
            "frontend_index_patch": None,
            "rocm_bundle": None,
        },
        "runtime": _runtime(accelerator),
        "diagnostics": {
            "facts": {"device_probed": True, "selected_major": accelerator.rocm_major},
            "issues": [],
        },
    }
    if accelerator is Accelerator.ROCM10:
        _add_bundle_fixture(payload)
    return payload


def _add_bundle_fixture(payload: dict[str, object]) -> None:
    """Synthetic hash-bound parser input, never AMD runtime evidence."""
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    distributions = provenance["distributions"]
    assert isinstance(distributions, dict)
    components = {}
    inventories = {}
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    facts = diagnostics["facts"]
    assert isinstance(facts, dict)
    compiler = {
        "rocm_path": "_rocm_sdk_core",
        "llvm_path": "_rocm_sdk_core/lib/llvm",
        "bitcode_path": "_rocm_sdk_core/lib/llvm/amdgcn/bitcode",
        "lld_path": "_rocm_sdk_core/lib/llvm/bin/ld.lld",
    }
    component_paths = {
        "core": [compiler["lld_path"], "_rocm_sdk_core/lib/libamdhip64.so"],
        "libraries": ["_rocm_sdk_libraries/lib/librocblas.so"],
        "device_gfx1250": ["_rocm_sdk_libraries/.kpack/blas_gfx1250.kpack"],
        "device_gfx942": ["_rocm_sdk_libraries/.kpack/blas_gfx942.kpack"],
        "device_gfx950": ["_rocm_sdk_libraries/.kpack/blas_gfx950.kpack"],
    }
    library_path = component_paths["core"][1]
    for key, paths in component_paths.items():
        name = "phydrax-rocm-" + key.replace("_", "-")
        files = {path: hashlib.sha256(path.encode()).hexdigest() for path in paths}
        inventory = [
            {"path": path, "sha256": digest, "size": 1, "mode": 0o644}
            for path, digest in files.items()
        ]
        text = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(text.encode()).hexdigest()
        inventory_path = name.replace("-", "_") + "-10.1.0.dist-info/phydrax-payload.json"
        files[inventory_path] = digest
        distributions[name]["files"] = files
        inventories[name] = {"path": inventory_path, "text": text}
        components[key] = {
            "distribution": name,
            "version": "10.1.0",
            "module": "_rocm_sdk_core" if key == "core" else "_rocm_sdk_libraries",
            "payload_sha256": digest,
            "upstream": {"url": f"https://example.invalid/{key}.whl", "sha256": "a" * 64},
        }
    manifest = {
        "sdk_version": "10.1.0",
        "architectures": ["gfx1250", "gfx942", "gfx950"],
        "components": components,
        "loader": {
            "compiler": compiler,
            "kpack": {
                arch: component_paths[f"device_{arch}"]
                for arch in ("gfx1250", "gfx942", "gfx950")
            },
            "preload": [library_path],
            "libraries": [
                {
                    "path": library_path,
                    "soname": "libamdhip64.so",
                    "needed": [],
                    "sha256": hashlib.sha256(library_path.encode()).hexdigest(),
                }
            ],
        },
    }
    text = json.dumps(manifest, sort_keys=True)
    distributions["phydrax-rocm-runtime"]["files"] = {
        "phydrax_rocm_runtime/runtime.json": hashlib.sha256(text.encode()).hexdigest()
    }
    for distribution in distributions.values():
        distribution["installed_sha256"] = hashlib.sha256(
            json.dumps(
                distribution["files"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    provenance["rocm_bundle"] = {"manifest": text, "payloads": inventories}
    site = Path(provenance["venv"]) / "site-packages"
    facts.update(
        bundle_version="10.1.0",
        bundle_architectures=["gfx1250", "gfx942", "gfx950"],
        bundle_error=None,
        bundle_loaded=True,
        bundle_manifest=str(site / "phydrax_rocm_runtime/runtime.json"),
        bundle_components=[
            [item["distribution"], item["version"], item["payload_sha256"]]
            for item in components.values()
        ],
    )
    diagnostics["automatic_runtime_environment"] = {
        "ROCM_PATH": str(site / compiler["rocm_path"]),
        "LLVM_PATH": str(site / compiler["llvm_path"]),
        "HIP_DEVICE_LIB_PATH": str(site / compiler["bitcode_path"]),
        "LD_LIBRARY_PATH": None,
    }


@pytest.mark.parametrize(
    "failure",
    (
        "missing-architecture",
        "wrong-architecture",
        "misordered-architectures",
        "duplicate-architecture",
        "wrong-version",
        "missing-payload",
        "missing-inventory",
        "extra-inventory",
        "wrong-payload",
        "wrong-manifest",
        "mixed-runtime",
        "not-loaded",
        "external-sdk",
        "manual-loader-path",
        "missing-compiler",
        "missing-runtime",
    ),
)
def test_rocm_bundle_evidence_refuses_incomplete_or_foreign_runtime(
    tmp_path: Path, failure: str
) -> None:
    payload = _probe_payload(Accelerator.ROCM10)
    provenance = payload["provenance"]
    diagnostics = payload["diagnostics"]
    assert isinstance(provenance, dict)
    assert isinstance(diagnostics, dict)
    facts = diagnostics["facts"]
    distributions = provenance["distributions"]
    bundle = provenance["rocm_bundle"]
    environment = diagnostics["automatic_runtime_environment"]
    assert isinstance(facts, dict)
    assert isinstance(distributions, dict)
    assert isinstance(bundle, dict)
    assert isinstance(environment, dict)
    if failure == "missing-architecture":
        facts["bundle_architectures"] = ["gfx942", "gfx950"]
    elif failure == "wrong-architecture":
        facts["bundle_architectures"] = ["gfx942", "gfx950", "gfx999"]
    elif failure == "misordered-architectures":
        facts["bundle_architectures"] = ["gfx942", "gfx950", "gfx1250"]
    elif failure == "duplicate-architecture":
        facts["bundle_architectures"] = ["gfx1250", "gfx1250", "gfx942", "gfx950"]
    elif failure == "wrong-version":
        distributions["phydrax-rocm-core"]["version"] = "99.0.0"
    elif failure == "missing-payload":
        del distributions["phydrax-rocm-device-gfx1250"]
    elif failure == "wrong-payload":
        bundle["payloads"]["phydrax-rocm-device-gfx1250"]["text"] = "[]"
    elif failure == "missing-inventory":
        del bundle["payloads"]["phydrax-rocm-device-gfx1250"]
    elif failure == "extra-inventory":
        bundle["payloads"]["phydrax-rocm-device-gfx999"] = bundle["payloads"][
            "phydrax-rocm-device-gfx1250"
        ]
    elif failure == "wrong-manifest":
        bundle["manifest"] = "{}"
    elif failure == "mixed-runtime":
        facts["bundle_components"][0][1] = "99.0.0"
    elif failure == "not-loaded":
        facts["bundle_loaded"] = False
    elif failure == "external-sdk":
        facts["bundle_manifest"] = "/checkout/sdk/phydrax_rocm_runtime/runtime.json"
    elif failure == "manual-loader-path":
        environment["LD_LIBRARY_PATH"] = "/checkout/sdk/lib"
    elif failure == "missing-compiler":
        environment["LLVM_PATH"] = None
    else:
        provenance["rocm_bundle"] = None
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.ROCM10) is None


@pytest.mark.parametrize(
    "component", ("core", "libraries", "device-gfx1250", "device-gfx942", "device-gfx950")
)
@pytest.mark.parametrize("failure", ("missing", "wrong-version"))
def test_rocm_provenance_requires_all_five_payload_distributions(
    tmp_path: Path, component: str, failure: str
) -> None:
    payload = _probe_payload(Accelerator.ROCM10)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    distributions = provenance["distributions"]
    assert isinstance(distributions, dict)
    name = f"phydrax-rocm-{component}"
    if failure == "missing":
        del distributions[name]
    else:
        distribution = distributions[name]
        assert isinstance(distribution, dict)
        distribution["version"] = "99.0.0"
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.ROCM10) is None


@pytest.mark.parametrize(
    "failure",
    (
        "wrong-architecture-assets",
        "missing-architecture",
        "misordered-architectures",
        "missing-device-component",
        "extra-device-component",
        "missing-architecture-assets",
        "wrong-sdk",
        "external-compiler",
    ),
)
def test_rocm_manifest_content_is_validated_beyond_record_hash(
    tmp_path: Path, failure: str
) -> None:
    payload = _probe_payload(Accelerator.ROCM10)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    bundle = provenance["rocm_bundle"]
    assert isinstance(bundle, dict)
    manifest = json.loads(bundle["manifest"])
    if failure == "wrong-architecture-assets":
        manifest["loader"]["kpack"]["gfx1250"] = manifest["loader"]["kpack"]["gfx942"]
    elif failure == "missing-architecture":
        manifest["architectures"] = ["gfx942", "gfx950"]
    elif failure == "misordered-architectures":
        manifest["architectures"] = ["gfx942", "gfx950", "gfx1250"]
    elif failure == "missing-device-component":
        del manifest["components"]["device_gfx1250"]
    elif failure == "extra-device-component":
        manifest["components"]["device_gfx999"] = manifest["components"]["device_gfx1250"]
    elif failure == "missing-architecture-assets":
        del manifest["loader"]["kpack"]["gfx1250"]
    elif failure == "wrong-sdk":
        manifest["sdk_version"] = "99.0.0"
    else:
        manifest["loader"]["compiler"]["llvm_path"] = "/checkout/sdk/llvm"
    text = json.dumps(manifest)
    bundle["manifest"] = text
    distribution = provenance["distributions"]["phydrax-rocm-runtime"]
    distribution["files"]["phydrax_rocm_runtime/runtime.json"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    distribution["installed_sha256"] = hashlib.sha256(
        json.dumps(distribution["files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.ROCM10) is None


def test_large_authoritative_probe_sidecar_is_complete_and_hash_validated(
    tmp_path: Path,
) -> None:
    payload = _probe_payload()
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert path.stat().st_size > TEXT_LIMIT
    assert read_json_evidence(path) == payload
    assert probe_record(path, Accelerator.TPU) == payload
    decoded = json.loads(path.read_text())
    decoded["provenance"]["distributions"]["phydrax"]["files"][
        "package/member-9999.so"
    ] = "0" * 64
    write_json_evidence(path, decoded)
    assert probe_record(path, Accelerator.TPU) is None


@pytest.mark.parametrize("accelerator", (Accelerator.ROCM10,))
def test_rocm_provenance_requires_exact_provider_and_no_foreign_family(
    tmp_path: Path,
    accelerator: Accelerator,
) -> None:
    path = tmp_path / "probe.json"
    write_json_evidence(path, _probe_payload(accelerator))
    assert probe_record(path, accelerator) is not None
    payload = json.loads(path.read_text())
    distributions = payload["provenance"]["distributions"]
    distributions[f"jax-{accelerator}-pjrt"]["version"] = "0.11.2"
    write_json_evidence(path, payload)
    assert probe_record(path, accelerator) is None
    distributions[f"jax-{accelerator}-pjrt"]["version"] = "0.12.0"
    distributions["jax-rocm99-plugin"] = distributions[f"jax-{accelerator}-plugin"]
    write_json_evidence(path, payload)
    assert probe_record(path, accelerator) is None


def test_probe_schema_and_size_limits_refuse_instead_of_truncating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    with pytest.raises(ValueError, match="exceeds"):
        write_json_evidence(path, {"large": "x" * 100}, max_bytes=32)
    assert not path.exists()
    path.write_text('{"large":"' + "x" * 100 + '"}')
    with pytest.raises(ValueError, match="exceeds"):
        read_json_evidence(path, max_bytes=32)
    assert probe_record(path, Accelerator.TPU) is None
    write_json_evidence(path, {"provenance": []})
    assert probe_record(path, Accelerator.TPU) is None


def _patched_probe_payload() -> dict[str, object]:
    """Synthetic parser input, not execution or historical qualification evidence."""
    payload = _probe_payload()
    patch = {
        "algorithm": "parser-input-not-compiler-evidence",
        "jax_version": Accelerator.TPU.distributions["jax"],
        "upstream_source_sha256": "1" * 64,
        "upstream_function_sha256": "2" * 64,
        "upstream_dynamic_slice_sha256": "5" * 64,
        "patch_source_sha256": "3" * 64,
    }
    provenance = payload["provenance"]
    runtime = payload["runtime"]
    assert isinstance(provenance, dict) and isinstance(runtime, dict)
    provenance["frontend_index_patch"] = patch
    runtime["frontend_index_patch"] = dict(patch)
    distributions = provenance["distributions"]
    assert isinstance(distributions, dict)
    for name, path, digest in (
        ("jax", "jax/_src/numpy/indexing.py", patch["upstream_source_sha256"]),
        ("phydrax", "phydrax/_jax_indexing_frontend.py", patch["patch_source_sha256"]),
    ):
        distribution = distributions[name]
        assert isinstance(distribution, dict)
        files = distribution["files"]
        assert isinstance(files, dict)
        files[path] = digest
        distribution["installed_sha256"] = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return payload


@pytest.mark.parametrize(
    "field",
    (
        "algorithm",
        "jax_version",
        "upstream_source_sha256",
        "upstream_function_sha256",
        "upstream_dynamic_slice_sha256",
        "patch_source_sha256",
    ),
)
def test_probe_and_cases_require_same_frontend_patch(tmp_path: Path, field: str) -> None:
    payload = _patched_probe_payload()
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.TPU) == payload
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    cases = tmp_path / "cases.xml"
    _xml(cases, runtime=runtime)
    assert not case_report_errors(cases, runtime, NONCE)[0]
    patch = runtime["frontend_index_patch"]
    assert isinstance(patch, dict)
    patch[field] = "mismatched-parser-input"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.TPU) is None
    assert case_report_errors(cases, runtime, NONCE)[0]


@pytest.mark.parametrize(
    "field", ("jax_version", "upstream_source_sha256", "patch_source_sha256")
)
def test_patch_provenance_is_bound_to_installed_distribution_bytes(
    tmp_path: Path, field: str
) -> None:
    payload = _patched_probe_payload()
    value = "0.0.0" if field == "jax_version" else "4" * 64
    provenance = payload["provenance"]
    runtime = payload["runtime"]
    assert isinstance(provenance, dict) and isinstance(runtime, dict)
    provenance_patch = provenance["frontend_index_patch"]
    runtime_patch = runtime["frontend_index_patch"]
    assert isinstance(provenance_patch, dict) and isinstance(runtime_patch, dict)
    provenance_patch[field] = value
    runtime_patch[field] = value
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.TPU) is None


@pytest.mark.parametrize("missing", ("provenance", "runtime"))
def test_patch_presence_is_explicit_even_when_unpatched(
    tmp_path: Path, missing: str
) -> None:
    payload = _probe_payload()
    record = payload[missing]
    assert isinstance(record, dict)
    del record["frontend_index_patch"]
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.TPU) is None


@pytest.mark.parametrize("change", ("missing", "unpatched", "different"))
def test_example_frontend_patch_must_match_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    payload = _patched_probe_payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    example = {
        **runtime,
        "accelerator": Accelerator.TPU.value,
        "status": "passed",
        "results_on_device": True,
    }
    if change == "missing":
        del example["frontend_index_patch"]
    elif change == "unpatched":
        example["frontend_index_patch"] = None
    else:
        patch = runtime["frontend_index_patch"]
        assert isinstance(patch, dict)
        example["frontend_index_patch"] = {
            **patch,
            "patch_source_sha256": "4" * 64,
        }
    write_json_evidence(tmp_path / "installed-runtime-device-0.json", payload)
    _xml(tmp_path / "cases-device-0.xml", runtime=runtime)
    monkeypatch.setattr(
        qualification,
        "run",
        Mock(
            return_value={
                "returncode": 0,
                "timed_out": False,
                "stdout": json.dumps(example),
            }
        ),
    )
    monkeypatch.setattr(qualification, "token_hex", Mock(return_value=NONCE))
    report = qualification._qualify_device(
        argparse.Namespace(tpu_autodetect=False, timeout=1),
        Accelerator.TPU,
        tmp_path,
        tmp_path / "venv",
        tmp_path / "expected-core.json",
        0,
    )
    assert report["errors"] == [
        "Advertised example failed or did not execute on the probed accelerator"
    ]
    assert not report["hardware_qualified"]


@pytest.mark.parametrize("fresh", (True, False))
def test_device_run_binds_resource_evidence_to_its_own_nonce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fresh: bool
) -> None:
    payload = _probe_payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    example = {
        **runtime,
        "accelerator": Accelerator.TPU.value,
        "status": "passed",
        "results_on_device": True,
    }
    write_json_evidence(tmp_path / "installed-runtime-device-0.json", payload)
    # Complete evidence from an earlier run must not qualify a later one.
    _xml(tmp_path / "cases-device-0.xml", runtime=runtime)
    run_command = Mock(
        return_value={"returncode": 0, "timed_out": False, "stdout": json.dumps(example)}
    )
    monkeypatch.setattr(qualification, "run", run_command)
    monkeypatch.setattr(
        qualification, "token_hex", Mock(return_value=NONCE if fresh else "e" * 32)
    )
    report = qualification._qualify_device(
        argparse.Namespace(tpu_autodetect=False, timeout=1),
        Accelerator.TPU,
        tmp_path,
        tmp_path / "venv",
        tmp_path / "expected-core.json",
        0,
    )
    environment = run_command.call_args_list[1].kwargs["environment"]
    assert environment["PHYDRAX_QUALIFICATION_NONCE"] == report["evidence_nonce"]
    assert report["hardware_qualified"] is fresh
    assert fresh == (not report["errors"])
    assert all(
        "not bound to this qualification run" in error for error in report["errors"]
    )


def test_process_logs_are_bounded_without_losing_retained_output(tmp_path: Path) -> None:
    result = run(
        [sys.executable, "-c", f"print('x' * {TEXT_LIMIT * 3})"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == 0 and not result["timed_out"]
    full_log = Path(result["stdout_path"]).read_bytes()
    assert len(result["stdout"].encode()) <= TEXT_LIMIT
    assert len(full_log) > TEXT_LIMIT
    assert hashlib.sha256(full_log).hexdigest() == result["stdout_sha256"]


def test_process_timeout_is_not_a_success_or_hardware_refusal(tmp_path: Path) -> None:
    result = run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=0.1,
    )
    assert result["timed_out"] and result["returncode"] != 0


def test_process_group_cleanup_terminates_descendants_after_parent_exits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "orphan-ran"
    child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).write_text('orphan')"
    source = f"import subprocess,sys; child=subprocess.Popen([sys.executable,'-c',{child!r}]); print(child.pid,flush=True)"
    result = run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == 0 and result["stdout"].strip().isdigit()
    time.sleep(0.7)
    assert not marker.exists()

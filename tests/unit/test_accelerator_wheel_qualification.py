# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free evidence parser/refusal policy tests, never qualification."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from email.message import Message
from pathlib import Path, PurePosixPath
from unittest.mock import Mock
from xml.etree import ElementTree

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
    ROCM_CASES,
    ROCM_ENVIRONMENT,
    runtime_errors,
    wheel_files,
)


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


def _xml(
    path: Path,
    names: tuple[str, ...] = REQUIRED_CASES,
    *,
    outcome: str | None = None,
    runtime: dict[str, object] | None = None,
    evidence: bool = True,
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
    ElementTree.ElementTree(suite).write(path)


def test_report_requires_exact_successful_case_set_and_runtime(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report)
    errors, cases = case_report_errors(report, _runtime())
    assert errors == [] and len(cases) == len(REQUIRED_CASES)


@pytest.mark.parametrize("outcome", ("skipped", "failure", "error"))
def test_skipped_or_failed_cases_never_qualify(tmp_path: Path, outcome: str) -> None:
    report = tmp_path / "report.xml"
    _xml(report, outcome=outcome)
    assert case_report_errors(report, _runtime())[0]


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
    assert case_report_errors(report, _runtime())[0]


def test_missing_or_wrong_actual_placement_refuses(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report, evidence=False)
    assert case_report_errors(report, _runtime())[0]
    _xml(report, runtime={**_runtime(), "platform": "cpu"})
    assert case_report_errors(report, _runtime())[0]
    report.write_text("not xml")
    assert case_report_errors(report, _runtime())[0]
    assert case_report_errors(tmp_path / "absent", _runtime())[0]


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
    assert case_report_errors(path, runtime)[0]
    _xml(path, REQUIRED_CASES + ROCM_CASES, runtime=runtime)
    assert not case_report_errors(path, runtime)[0]
    assert case_report_errors(path, {**runtime, "qualified_device_id": 0})[0]


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
    # These are Linux producer records, not paths on the test runner's host.
    site = PurePosixPath(provenance["venv"]) / "site-packages"
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
        "windows-manifest-path",
        "windows-compiler-path",
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
    elif failure == "windows-manifest-path":
        facts["bundle_manifest"] = (
            r"\parser-input\venv\site-packages\phydrax_rocm_runtime\runtime.json"
        )
    elif failure == "windows-compiler-path":
        environment["LLVM_PATH"] = (
            r"\parser-input\venv\site-packages\_rocm_sdk_core\lib\llvm"
        )
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
    assert not case_report_errors(cases, runtime)[0]
    patch = runtime["frontend_index_patch"]
    assert isinstance(patch, dict)
    patch[field] = "mismatched-parser-input"
    write_json_evidence(path, payload)
    assert probe_record(path, Accelerator.TPU) is None
    assert case_report_errors(cases, runtime)[0]


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


@pytest.mark.parametrize(
    "payload", [b"x", b"\xff", "λ".encode()], ids=["ascii", "invalid-utf8", "utf8"]
)
def test_process_logs_are_bounded_without_losing_retained_output(
    tmp_path: Path, payload: bytes
) -> None:
    source = (
        f"import os; payload={payload!r} * {TEXT_LIMIT * 3}; "
        "os.write(1,payload); os.write(2,payload)"
    )
    result = run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == 0 and not result["timed_out"]
    for text, path, digest in (
        (result["stdout"], result["stdout_path"], result["stdout_sha256"]),
        (result["stderr"], result["stderr_path"], result["stderr_sha256"]),
    ):
        full_log = Path(path).read_bytes()
        assert len(text.encode()) <= TEXT_LIMIT
        assert full_log == payload * (TEXT_LIMIT * 3)
        assert hashlib.sha256(full_log).hexdigest() == digest


def test_process_timeout_is_not_a_success_or_hardware_refusal(tmp_path: Path) -> None:
    result = run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=0.1,
    )
    assert result["timed_out"] and result["returncode"] != 0


@pytest.mark.parametrize("parent_outcome", ["normal", "timeout", "crash"])
def test_process_group_cleanup_terminates_descendants_without_touching_other_processes(
    tmp_path: Path, parent_outcome: str
) -> None:
    ready = tmp_path / "descendant-ready"
    marker = tmp_path / "orphan-ran"
    sentinel = tmp_path / "unrelated-ran"
    child = (
        "import time; from pathlib import Path; "
        f"Path({str(ready)!r}).write_text('ready'); "
        f"time.sleep(2); Path({str(marker)!r}).write_text('orphan'); time.sleep(30)"
    )
    ending = {
        "normal": "pass",
        "timeout": "time.sleep(30)",
        "crash": "os._exit(23)",
    }[parent_outcome]
    source = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"ready=Path({str(ready)!r})\n"
        "while not ready.exists(): time.sleep(0.01)\n"
        "print(child.pid,flush=True)\n"
        f"{ending}\n"
    )
    unrelated_source = (
        "import time; from pathlib import Path; time.sleep(2); "
        f"Path({str(sentinel)!r}).write_text('alive'); time.sleep(30)"
    )
    with subprocess.Popen(
        [sys.executable, "-c", unrelated_source], cwd=tmp_path
    ) as unrelated:
        try:
            result = run(
                [sys.executable, "-c", source],
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout=1.0 if parent_outcome == "timeout" else 10.0,
            )
            assert ready.exists(), "The descendant must actually start before cleanup"
            assert result["stdout"].strip().isdigit()
            assert result["timed_out"] is (parent_outcome == "timeout")
            if parent_outcome == "normal":
                assert result["returncode"] == 0
            elif parent_outcome == "crash":
                assert result["returncode"] == 23
            else:
                assert result["returncode"] != 0
            time.sleep(2.2)
            assert not marker.exists()
            assert sentinel.read_text() == "alive"
            assert unrelated.poll() is None
        finally:
            unrelated.kill()
            unrelated.wait(timeout=10)


@pytest.mark.parametrize("exitcode", [0, 37, 259])
def test_process_exit_status_is_exact(tmp_path: Path, exitcode: int) -> None:
    if exitcode > 255 and os.name != "nt":
        pytest.skip("POSIX exit statuses contain only the low byte")
    result = run(
        [sys.executable, "-c", f"import os; os._exit({exitcode})"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == exitcode
    assert not result["timed_out"]


def test_process_crash_preserves_platform_status(tmp_path: Path) -> None:
    if os.name == "nt":
        source = (
            "import ctypes; k=ctypes.WinDLL('kernel32'); "
            "k.GetCurrentProcess.restype=ctypes.c_void_p; "
            "k.TerminateProcess.argtypes=(ctypes.c_void_p,ctypes.c_uint32); "
            "k.TerminateProcess(k.GetCurrentProcess(),0xC0000005)"
        )
        expected = 0xC0000005
    else:
        source = "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"
        expected = -signal.SIGKILL
    result = run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == expected
    assert not result["timed_out"]


@pytest.mark.skipif(os.name != "nt", reason="Win32 Job Object boundary")
def test_windows_assignment_failure_never_runs_and_terminates_suspended_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import _windows_process as windows

    marker = tmp_path / "unassigned-child-ran"
    kernel = windows._kernel32()
    assign = windows._assign_job
    processes: list[int] = []

    def refuse_job(library: ctypes.CDLL, job: int, process: int) -> None:
        processes.append(windows._duplicate(library, process, inherit=False))
        # Exercise the real Win32 assignment refusal, not a fabricated status.
        assign(library, 0, process)

    monkeypatch.setattr(windows, "_assign_job", refuse_job)
    try:
        with pytest.raises(OSError):
            run(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout=10,
            )
        assert len(processes) == 1
        assert kernel.WaitForSingleObject(processes[0], 0) == 0
        assert not marker.exists()
    finally:
        for process in processes:
            windows._close(kernel, process)


@pytest.mark.skipif(os.name != "nt", reason="Win32 inherited-handle allowlist")
def test_windows_child_does_not_inherit_unlisted_handle(tmp_path: Path) -> None:
    from tools import _windows_process as windows

    kernel = windows._kernel32()
    kernel.CreateEventW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_wchar_p,
    )
    kernel.CreateEventW.restype = ctypes.c_void_p
    event = kernel.CreateEventW(None, True, False, None)
    assert event
    try:
        inherited = windows._duplicate(kernel, event, inherit=True)
        try:
            source = (
                "import ctypes; k=ctypes.WinDLL('kernel32'); "
                "k.SetEvent.argtypes=(ctypes.c_void_p,); "
                f"k.SetEvent({inherited})"
            )
            result = run(
                [sys.executable, "-c", source],
                cwd=tmp_path,
                environment=dict(os.environ),
                timeout=10,
            )
            assert result["returncode"] == 0 and not result["timed_out"]
            # A numeric handle may be reused in the child; test the actual
            # parent's event rather than mere validity of that integer there.
            assert kernel.WaitForSingleObject(event, 0) == 258
        finally:
            windows._close(kernel, inherited)
    finally:
        windows._close(kernel, event)


@pytest.mark.skipif(os.name != "nt", reason="Win32 kill-on-close supervisor cleanup")
def test_windows_supervisor_exit_closes_job_and_terminates_owned_tree(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "owned-child-ready"
    orphan = tmp_path / "owned-child-survived"
    child = (
        "import time; from pathlib import Path; "
        f"Path({str(ready)!r}).touch(); time.sleep(3); "
        f"Path({str(orphan)!r}).touch(); time.sleep(30)"
    )
    source = (
        "import os,sys,threading,time; from pathlib import Path; "
        "from tools._accelerator_process import run\n"
        "def exit_supervisor():\n"
        f"    while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        "    os._exit(17)\n"
        "threading.Thread(target=exit_supervisor,daemon=True).start()\n"
        f"run([sys.executable,'-c',{child!r}],cwd=Path({str(tmp_path)!r}),"
        "environment=dict(os.environ),timeout=20)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        timeout=15,
        capture_output=True,
    )
    assert completed.returncode == 17
    assert ready.exists()
    time.sleep(3.2)
    assert not orphan.exists()


def test_direct_script_runner_supervises_child_without_package_search_path(
    tmp_path: Path,
) -> None:
    script = tmp_path / "direct-command.py"
    marker = tmp_path / "executed"
    child = (
        "import sys; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text('executed'); sys.exit(37)"
    )
    script.write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "from _accelerator_process import run\n"
        f"result=run([sys.executable,'-S','-c',{child!r}],"
        f"cwd=Path({str(tmp_path)!r}),environment=dict(os.environ),timeout=10)\n"
        "print(json.dumps(result))\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "tools")
    completed = subprocess.run(
        [sys.executable, "-S", str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)
    assert evidence["returncode"] == 37
    assert not evidence["timed_out"]
    assert marker.read_text() == "executed"

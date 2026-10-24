#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import _runtime as benchmark_runtime
from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    BenchmarkIdentity,
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    CompilerEvidence,
    DurationDistribution,
    installed_build_fingerprint,
    installed_package_fingerprint,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    source_build_fingerprint,
    synchronize,
    validate_benchmark_record,
)
from phydrax._fingerprint import canonical_fingerprint


def test_benchmark_runtime_scenario_1() -> None:
    distribution = DurationDistribution((0.001, 0.003, 0.002))

    assert distribution.count == 3
    assert distribution.minimum_seconds == pytest.approx(0.001)
    assert distribution.median_seconds == pytest.approx(0.002)
    assert distribution.mean_seconds == pytest.approx(0.002)
    assert distribution.population_std_seconds == pytest.approx(
        float(np.std([0.001, 0.003, 0.002]))
    )
    assert distribution.maximum_seconds == pytest.approx(0.003)
    milliseconds = distribution.to_milliseconds_dict()
    assert milliseconds == {
        "count": 3,
        "samples_ms": [1.0, 3.0, 2.0],
        "min_ms": 1.0,
        "median_ms": 2.0,
        "mean_ms": 2.0,
        "std_ms": pytest.approx(float(np.std([1.0, 3.0, 2.0]))),
        "max_ms": 3.0,
    }
    assert DurationDistribution(()).to_seconds_dict() == {
        "count": 0,
        "samples_seconds": [],
        "min_seconds": None,
        "median_seconds": None,
        "mean_seconds": None,
        "std_seconds": None,
        "max_seconds": None,
    }
    for samples in ((-1.0,), (math.nan,), (math.inf,)):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            DurationDistribution(samples)
    with pytest.raises(ValueError, match="Duration unit"):
        DurationDistribution((1.0,)).to_dict(unit="minutes")  # ty: ignore[invalid-argument-type]
    value = _NestedArrays(
        {"array": jnp.ones((2,), dtype=jnp.float32)},
        (np.ones((3,), dtype=np.float64), "host"),
    )

    assert synchronize(value) is value
    assert logical_array_bytes(value) == 2 * 4 + 3 * 8
    evidence = compiler_evidence(
        {"flops": 101.2, "bytes accessed": 202.4},
        # ty: ignore[invalid-argument-type]
        _MemoryAnalysis(),
        source="xla-cost-analysis",
    )
    assert evidence.flops == 101
    assert evidence.bytes_accessed == 202
    assert evidence.estimated_device_memory_bytes == 60

    unavailable = compiler_evidence(
        None,
        None,
        source="xla-cost-analysis",
        unavailable_reason="unsupported backend",
    )
    assert unavailable.estimated_device_memory_bytes is None
    assert unavailable.unavailable_reason == "unsupported backend"

    not_applicable = CompilerEvidence(0, 0, 0, 0, 0, 0, "not-applicable")
    assert not_applicable.estimated_device_memory_bytes == 0
    with pytest.raises(ValueError, match="requires a reason"):
        CompilerEvidence(None, None, None, None, None, None, "xla-cost-analysis")


@pytest.mark.parametrize(
    "cost,expected_flops,expected_bytes",
    (
        ({"flops": -1.0, "bytes accessed": 123.0}, None, 123),
        ({"flops": 12.0, "bytes accessed": -1.0}, 12, None),
        ({"flops": -1.0, "bytes accessed": -1.0}, None, None),
    ),
)
def test_compiler_custom_call_unknown_cost_is_none_with_reason_and_known_memory(
    cost: dict[str, float], expected_flops: int | None, expected_bytes: int | None
) -> None:
    raw = cost.copy()
    evidence = compiler_evidence(
        cost,
        _KnownCompilerMemory(),
        source="xla-custom-call",
        unavailable_reason="opaque external computation",
    )
    assert cost == raw
    assert evidence.flops == expected_flops
    assert evidence.bytes_accessed == expected_bytes
    assert evidence.estimated_device_memory_bytes == 60
    assert evidence.unavailable_reason is not None
    assert "opaque external computation" in evidence.unavailable_reason
    if expected_flops is None:
        assert "flops" in evidence.unavailable_reason
    if expected_bytes is None:
        assert "bytes accessed" in evidence.unavailable_reason


def test_compiler_known_zero_is_distinct_from_unknown_custom_call() -> None:
    evidence = compiler_evidence(
        {"flops": 0.0, "bytes accessed": 0.0},
        _KnownCompilerMemory(),
        source="xla-known-zero",
    )
    assert evidence.flops == 0
    assert evidence.bytes_accessed == 0
    assert evidence.unavailable_reason is None
    unknown = compiler_evidence(
        {"flops": -1.0, "bytes accessed": -1.0},
        None,
        source="xla-unknown",
    )
    assert unknown.flops is None and unknown.bytes_accessed is None
    assert unknown.unavailable_reason is not None


def test_missing_compiler_field_is_not_mislabeled_as_negative_sentinel() -> None:
    evidence = compiler_evidence(
        {"bytes accessed": 0.0},
        _KnownCompilerMemory(),
        source="xla-omitted",
        unavailable_reason="Compiler omitted the FLOP estimate",
    )
    assert evidence.flops is None
    assert evidence.bytes_accessed == 0
    assert evidence.unavailable_reason == "Compiler omitted the FLOP estimate"


@pytest.mark.parametrize("field", ("flops", "bytes accessed"))
@pytest.mark.parametrize("invalid", (-2.0, -0.5, -1.0001, math.nan, math.inf, -math.inf))
def test_other_negative_or_nonfinite_compiler_costs_remain_invalid(
    field: str, invalid: float
) -> None:
    cost = {"flops": 0.0, "bytes accessed": 0.0}
    cost[field] = invalid
    with pytest.raises(ValueError):
        compiler_evidence(cost, _KnownCompilerMemory(), source="xla-invalid")


@pytest.mark.parametrize("aggregate", (-3.0, 0.0, 128.0))
def test_opaque_custom_calls_never_report_partial_costs_as_totals(
    aggregate: float,
) -> None:
    cost = {"flops": aggregate, "bytes accessed": aggregate}
    evidence = compiler_evidence(
        cost,
        _KnownCompilerMemory(),
        source="xla-mixed-custom-calls",
        opaque_custom_calls=True,
        unavailable_reason="unmodeled vendor kernels",
    )
    assert evidence.flops is None and evidence.bytes_accessed is None
    assert evidence.estimated_device_memory_bytes == 60
    assert evidence.unavailable_reason is not None
    assert "unmodeled vendor kernels" in evidence.unavailable_reason
    assert cost == {"flops": aggregate, "bytes accessed": aggregate}


def test_measure_repeated_synchronizes_warmups_and_retains_every_sample() -> None:
    events = []
    counter = 0

    def operation() -> Any:
        nonlocal counter
        counter += 1
        events.append(("operation", counter))
        return counter

    def synchronizer(value: Any) -> Any:
        events.append(("synchronize", value))
        return value

    result, distribution = measure_repeated(
        operation,
        warmup=2,
        repeats=3,
        synchronizer=synchronizer,
    )

    assert result == 5
    assert distribution.count == 3
    assert events == [
        ("operation", 1),
        ("synchronize", 1),
        ("operation", 2),
        ("synchronize", 2),
        ("operation", 3),
        ("synchronize", 3),
        ("operation", 4),
        ("synchronize", 4),
        ("operation", 5),
        ("synchronize", 5),
    ]
    with pytest.raises(ValueError, match="repeats must be positive"):
        measure_repeated(operation, warmup=0, repeats=0)
    with pytest.raises(ValueError, match="warmup must be nonnegative"):
        measure_repeated(operation, warmup=-1, repeats=1)


def test_repeated_measurement_releases_completed_arrays_without_collection() -> None:
    references: list[weakref.ReferenceType[np.ndarray]] = []

    def operation() -> np.ndarray:
        value = np.zeros((16,), dtype=np.float64)
        references.append(weakref.ref(value))
        return value

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result, _ = measure_repeated(operation, warmup=1, repeats=64)
        assert sum(reference() is not None for reference in references) == 1
        assert references[-1]() is result
    finally:
        if was_enabled:
            gc.enable()


def test_host_and_synchronized_measurement_have_distinct_boundaries() -> None:
    events = []

    def operation() -> str:
        events.append("operation")
        return "value"

    value, host_seconds = measure_host(operation)
    assert value == "value"
    assert host_seconds >= 0.0
    assert events == ["operation"]

    value, synchronized_seconds = measure_synchronized(
        operation,
        synchronizer=lambda result: events.append(f"sync:{result}"),
    )
    assert value == "value"
    assert synchronized_seconds >= 0.0
    assert events == ["operation", "operation", "sync:value"]


def test_lowering_and_compilation_are_ordered_independent_host_phases() -> None:
    events = []

    def lower() -> str:
        events.append("lower")
        return "lowered"

    def compile(lowered: Any) -> str:
        events.append(f"compile:{lowered}")
        return "compiled"

    compiled, timing = measure_lower_and_compile(lower, compile)

    assert compiled == "compiled"
    assert events == ["lower", "compile:lowered"]
    assert timing.lowering_seconds >= 0.0
    assert timing.compilation_seconds >= 0.0


@dataclass(frozen=True)
class _NestedArrays:
    first: object
    second: object


@dataclass(frozen=True)
class _MemoryAnalysis:
    argument_size_in_bytes: int = 10
    output_size_in_bytes: int = 20
    temp_size_in_bytes: int = 30
    generated_code_size_in_bytes: int = 40


@dataclass
class _KnownCompilerMemory:
    """Writable response attributes satisfy the owning compiler-memory protocol."""

    argument_size_in_bytes: int = 10
    output_size_in_bytes: int = 20
    temp_size_in_bytes: int = 30
    generated_code_size_in_bytes: int = 40


class _Distribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


def test_installed_package_fingerprint_normalizes_order_and_spelling(
    monkeypatch: Any,
) -> None:
    first = (_Distribution("A_Package", "1"), _Distribution("b.package", "2"))
    second = (_Distribution("B-PACKAGE", "2"), _Distribution("a-package", "1"))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: first)
    first_fingerprint = installed_package_fingerprint()
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: second)
    assert installed_package_fingerprint() == first_fingerprint

    conflicting = (_Distribution("same", "1"), _Distribution("same", "2"))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: conflicting)
    with pytest.raises(ValueError, match="conflicting versions"):
        installed_package_fingerprint()


def _installed_wheel(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    metadata = root / "phydrax-1.0.dist-info"
    contents = {
        "phydrax/__init__.py": "VALUE = 1\n",
        "phydrax/kernel.pyd": "native-wheel-bytes",
        "phydrax/__pycache__/__init__.cpython-314.pyc": "generated-bytecode",
        "phydrax-1.0.dist-info/METADATA": "Metadata-Version: 2.3\nName: phydrax\nVersion: 1.0\n",
        "phydrax-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp314-cp314-win_amd64\n",
        "phydrax-1.0.dist-info/INSTALLER": "uv\n",
        "phydrax-1.0.dist-info/REQUESTED": "",
        "phydrax-1.0.dist-info/direct_url.json": json.dumps(
            {"url": (root / "phydrax.whl").as_uri()}
        ),
    }
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    record = metadata / "RECORD"
    record.write_text(
        "".join(f"{name},,\n" for name in contents) + "phydrax-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    distribution = importlib.metadata.PathDistribution(metadata)

    def selected_distribution(name: str) -> importlib.metadata.Distribution:
        assert name == "phydrax"
        return distribution

    monkeypatch.setattr(importlib.metadata, "distribution", selected_distribution)
    origin = str(root / "phydrax" / "__init__.py")
    monkeypatch.setattr(benchmark_runtime.phydrax, "__file__", origin)
    monkeypatch.setattr(
        benchmark_runtime.phydrax, "__spec__", SimpleNamespace(origin=origin)
    )
    return record


def test_installed_build_identity_is_prefix_independent_and_hashes_actual_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    record = _installed_wheel(first_root, monkeypatch)
    initial = installed_build_fingerprint()
    assert not (first_root / "uv.lock").exists()
    assert not (first_root / "pyproject.toml").exists()
    record.write_text(
        "\n".join(reversed(record.read_text(encoding="utf-8").splitlines())) + "\n",
        encoding="utf-8",
    )
    cache = first_root / "phydrax" / "__pycache__" / "__init__.cpython-314.pyc"
    cache.unlink()
    assert installed_build_fingerprint() == initial

    second_root = tmp_path / "different-prefix"
    record = _installed_wheel(second_root, monkeypatch)
    (record.parent / "INSTALLER").write_text("pip\n", encoding="utf-8")
    (record.parent / "REQUESTED").unlink()
    record.write_text(
        record.read_text(encoding="utf-8").replace(
            "phydrax-1.0.dist-info/REQUESTED,,\n", ""
        ),
        encoding="utf-8",
    )
    assert installed_build_fingerprint() == initial
    (second_root / "phydrax" / "kernel.pyd").write_text(
        "mutated-wheel-byte", encoding="utf-8"
    )
    assert installed_build_fingerprint() != initial


@pytest.mark.parametrize("filename", ["METADATA", "WHEEL"])
def test_installed_build_identity_hashes_authored_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    record = _installed_wheel(tmp_path, monkeypatch)
    initial = installed_build_fingerprint()
    path = record.parent / filename
    path.write_text(
        path.read_text(encoding="utf-8") + "Build: changed\n", encoding="utf-8"
    )
    assert installed_build_fingerprint() != initial


@pytest.mark.parametrize("unowned", ["checkout", "omitted", "editable"])
def test_installed_build_identity_refuses_unowned_or_editable_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unowned: str,
) -> None:
    record = _installed_wheel(tmp_path / "installed", monkeypatch)
    if unowned == "checkout":
        checkout = tmp_path / "__init__.py"
        checkout.write_text("VALUE = 1\n", encoding="utf-8")
        monkeypatch.setattr(benchmark_runtime.phydrax, "__file__", str(checkout))
        monkeypatch.setattr(
            benchmark_runtime.phydrax, "__spec__", SimpleNamespace(origin=str(checkout))
        )
    elif unowned == "omitted":
        record.write_text(
            record.read_text(encoding="utf-8").replace("phydrax/__init__.py,,\n", ""),
            encoding="utf-8",
        )
    else:
        (record.parent / "direct_url.json").write_text(
            json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="not owned|editable"):
        installed_build_fingerprint()


@pytest.mark.parametrize(
    "name", ["../outside.py", "/outside.py", "C:/outside.py", r"..\outside.py"]
)
def test_installed_build_identity_refuses_escaped_record_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    record = _installed_wheel(tmp_path, monkeypatch)
    record.write_text(
        record.read_text(encoding="utf-8") + f"{name},,\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes"):
        installed_build_fingerprint()


def test_installed_build_identity_refuses_resolved_record_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed_wheel(tmp_path / "installed", monkeypatch)
    outside = tmp_path / "outside.pyd"
    outside.write_text("outside", encoding="utf-8")
    original_resolve = Path.resolve
    payload = tmp_path / "installed" / "phydrax" / "kernel.pyd"

    def escaped_resolve(path: Path, strict: bool = False) -> Path:
        if path == payload:
            return outside
        return original_resolve(path, strict=strict)

    # Model a resolved symlink/junction without requiring Windows symlink privileges.
    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    with pytest.raises(ValueError, match="escapes"):
        installed_build_fingerprint()


@pytest.mark.parametrize("filename", ["RECORD", "WHEEL", "METADATA"])
def test_installed_build_identity_requires_wheel_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    record = _installed_wheel(tmp_path, monkeypatch)
    (record.parent / filename).unlink()
    with pytest.raises(FileNotFoundError):
        installed_build_fingerprint()


def test_installed_build_identity_preserves_malformed_metadata_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _installed_wheel(tmp_path, monkeypatch)
    original_record = record.read_text(encoding="utf-8")
    record.write_text("phydrax/__init__.py\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        installed_build_fingerprint()
    record.write_text(original_record, encoding="utf-8")
    (record.parent / "direct_url.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        installed_build_fingerprint()


def test_installed_build_identity_preserves_missing_distribution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> importlib.metadata.Distribution:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        installed_build_fingerprint()


def test_captured_environment_fingerprint_covers_serialized_runtime_evidence() -> None:
    environment = capture_environment()
    payload = environment.to_dict()
    observed = payload.pop("fingerprint")

    assert canonical_fingerprint(payload) == observed
    assert payload["jax"]["devices"]
    assert payload["package_fingerprint"]


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
def test_frontend_patch_changes_environment_identity(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    identity = {
        "algorithm": "parser-input-not-compiler-evidence",
        "jax_version": "0.12.0",
        "upstream_source_sha256": "1" * 64,
        "upstream_function_sha256": "2" * 64,
        "upstream_dynamic_slice_sha256": "5" * 64,
        "patch_source_sha256": "3" * 64,
    }
    monkeypatch.setattr(benchmark_runtime, "frontend_index_patch_identity", lambda: None)
    unpatched = capture_environment()
    monkeypatch.setattr(
        benchmark_runtime, "frontend_index_patch_identity", lambda: dict(identity)
    )
    patched = capture_environment()
    assert unpatched.to_dict()["frontend_index_patch"] is None
    assert patched.to_dict()["frontend_index_patch"] == identity
    assert patched.fingerprint != unpatched.fingerprint
    identity[field] = "changed-parser-input"
    changed = capture_environment()
    assert changed.fingerprint != patched.fingerprint
    assert changed.package_fingerprint == patched.package_fingerprint
    assert patched.to_dict()["frontend_index_patch"][field] != identity[field]


def test_environment_retains_raw_and_normalized_amd_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_version = "rocm 71626385"
    device = SimpleNamespace(
        platform="gpu",
        device_kind="AMD Instinct MI300X",
        process_index=0,
        id=3,
        local_hardware_id=3,
        client=SimpleNamespace(platform="gpu", platform_version=runtime_version),
    )
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    packages = {
        "jax-rocm10-plugin": "0.12.0",
        "jax-rocm10-pjrt": "0.12.0",
        "phydrax-rocm-runtime": "10.1.0",
        "phydrax-rocm-core": "10.1.0",
        "phydrax-rocm-libraries": "10.1.0",
        "phydrax-rocm-device-gfx942": "10.1.0",
        "phydrax-rocm-device-gfx950": "10.1.0",
    }
    monkeypatch.setattr(
        importlib.metadata,
        "distributions",
        lambda: tuple(_Distribution(name, version) for name, version in packages.items()),
    )
    environment = capture_environment().to_dict()
    assert environment["jax"]["devices"] == [
        {
            "platform": "gpu",
            "kind": "AMD Instinct MI300X",
            "normalized_platform": "rocm",
            "vendor": "amd",
            "runtime_version": runtime_version,
            "process_index": 0,
            "device_id": 3,
            "local_hardware_id": 3,
        }
    ]
    assert environment["rocm_packages"] == packages
    fingerprint = environment.pop("fingerprint")
    assert canonical_fingerprint(environment) == fingerprint


@pytest.mark.parametrize(
    ("name", "first", "second"),
    [
        ("NPROC", "3", "4"),
        ("NCCL_NVLS_ENABLE", "0", "1"),
        ("NCCL_P2P_DISABLE", "0", "1"),
        ("NCCL_LAUNCH_ORDER_IMPLICIT", "0", "1"),
        ("NCCL_LAUNCH_RACE_FATAL", "0", "1"),
        ("UCX_MEMTYPE_CACHE", "n", "y"),
        ("ROCR_VISIBLE_DEVICES", "0,1", "2,3"),
        ("HIP_VISIBLE_DEVICES", "0", "1"),
        ("GPU_DEVICE_ORDINAL", "0", "1"),
        ("LD_LIBRARY_PATH", "/opt/rocm/lib", "/isolated/rocm/lib"),
        ("ROCM_PATH", "/opt/rocm", "/isolated/rocm"),
        ("HSA_OVERRIDE_GFX_VERSION", "9.4.2", "9.4.3"),
        ("PHYDRAX_LOCAL_DEVICE_IDS", "0,1", "2,3"),
    ],
    ids=[
        "cpu-workers",
        "nvls",
        "peer-transport",
        "communicator-order",
        "launch-race-detection",
        "ucx-memory-type-cache",
        "rocm-visibility",
        "hip-visibility",
        "gpu-ordinal",
        "runtime-library-path",
        "rocm-path",
        "gfx-override",
        "local-ownership",
    ],
)
def test_performance_configuration_changes_environment_identity(
    monkeypatch: pytest.MonkeyPatch, name: str, first: str, second: str
) -> None:
    monkeypatch.setenv(name, first)
    before = capture_environment()
    monkeypatch.setenv(name, second)
    after = capture_environment()
    assert before.fingerprint != after.fingerprint


def _benchmark_source_tree(root: Path) -> Path:
    (root / "phydrax").mkdir()
    (root / "benchmarks").mkdir()
    (root / "phydrax" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "benchmarks" / "_runtime.py").write_text("RUNTIME = 1\n", encoding="utf-8")
    driver = root / "benchmarks" / "driver.py"
    driver.write_text("DRIVER = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "uv.lock").write_text("revision = 1\n", encoding="utf-8")
    return driver


def _stored_record(identity: BenchmarkIdentity) -> dict[str, Any]:
    return {
        "identity": identity.to_dict(),
        "cases": [{"last_step_evidence": {"accepted": True, "residual": 0.0}}],
    }


def test_benchmark_identity_detects_source_mutation(tmp_path: Path) -> None:
    driver = _benchmark_source_tree(tmp_path)
    initial = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(initial)

    source = tmp_path / "phydrax" / "model.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))

    assert source_build_fingerprint(tmp_path) != initial.source_build_fingerprint
    with pytest.raises(ValueError, match="source/build fingerprint"):
        validate_benchmark_record(stored, changed)


def test_source_build_identity_still_requires_lockfile(tmp_path: Path) -> None:
    _benchmark_source_tree(tmp_path)
    (tmp_path / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError, match="uv.lock"):
        source_build_fingerprint(tmp_path)


def test_benchmark_identity_detects_driver_mutation(tmp_path: Path) -> None:
    driver = _benchmark_source_tree(tmp_path)
    initial = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(initial)

    driver.write_text("DRIVER = 2\n", encoding="utf-8")
    changed = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))

    assert (
        benchmark_driver_fingerprint(tmp_path, driver)
        != initial.benchmark_driver_fingerprint
    )
    with pytest.raises(ValueError, match="driver fingerprint"):
        validate_benchmark_record(stored, changed)


def test_benchmark_record_rejects_evidence_schema_field_set_mismatch(
    tmp_path: Path,
) -> None:
    driver = _benchmark_source_tree(tmp_path)
    identity = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(identity)
    stored["identity"]["evidence_schema"]["fields"] = ["accepted"]

    with pytest.raises(ValueError, match="evidence field set/signature"):
        validate_benchmark_record(stored, identity)


def test_benchmark_record_rejects_case_evidence_field_set_mismatch(
    tmp_path: Path,
) -> None:
    driver = _benchmark_source_tree(tmp_path)
    identity = capture_benchmark_identity(tmp_path, driver, ("accepted", "residual"))
    stored = _stored_record(identity)
    stored["cases"][0]["last_step_evidence"].pop("residual")

    with pytest.raises(ValueError, match="evidence field set"):
        validate_benchmark_record(stored, identity)


def test_adaptive_sphere_refuses_point_limit_before_preparation() -> None:
    from benchmarks.meshfree_closure_adaptive_restart import (
        measure_adaptive_sphere_zonal_peak,
    )
    from benchmarks.meshfree_scaling import DeclaredCapacityRefusal, MeshfreeConfig

    config = MeshfreeConfig(sizes=(768,), dimension=3, max_points=64)
    with pytest.raises(DeclaredCapacityRefusal, match="max_points=64"):
        measure_adaptive_sphere_zonal_peak(768, 0, config)


def test_windows_runtime_provenance_preserves_unobserved_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    installation = SimpleNamespace(
        distribution="phydrax-cuda13-pjrt",
        version="1.0",
        cuda_major=13,
        jax_version="0.12.0",
        jaxlib_version="0.12.0",
        build_id="build-a",
        manifest_path="C:/synthetic/runtime.json",
        assets=(),
        issues=(),
        capabilities=("cuda",),
    )
    state = SimpleNamespace(
        status="registered", error=None, allocator=None, preallocate=None
    )
    identity = {
        "linkage": "driver-hybrid",
        "owner": "display-driver",
        "runtime_path": "C:/Windows/System32/DriverStore/FileRepository/nvidia/nvcudart_hybrid64.dll",
        "runtime_sha256": "a" * 64,
        "driver_path": "C:/Windows/System32/DriverStore/FileRepository/nvidia/nvcuda64.dll",
        "driver_sha256": "b" * 64,
        "driver_dispatch_path": "C:/Windows/System32/nvcuda.dll",
        "driver_dispatch_sha256": "c" * 64,
    }
    monkeypatch.setattr(benchmark_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0")
    monkeypatch.setitem(
        sys.modules,
        "phydrax_cuda13_pjrt.runtime",
        SimpleNamespace(
            installation_record=lambda: installation,
            runtime_identity=lambda: SimpleNamespace(_asdict=lambda: dict(identity)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "phydrax_cuda13_pjrt.plugin",
        SimpleNamespace(
            bootstrap_state=lambda: state,
        ),
    )
    initial = benchmark_runtime._windows_cuda_provenance()
    assert initial is not None
    assert initial["effective_allocator"] is None
    assert initial["effective_preallocate"] is None
    assert initial["runtime_identity"] == identity
    state.allocator, state.preallocate = "platform", False
    changed = benchmark_runtime._windows_cuda_provenance()
    assert canonical_fingerprint(initial) != canonical_fingerprint(changed)
    installation.build_id = "build-b"
    assert canonical_fingerprint(changed) != canonical_fingerprint(
        benchmark_runtime._windows_cuda_provenance()
    )
    before_driver_change = benchmark_runtime._windows_cuda_provenance()
    identity["runtime_sha256"] = "d" * 64
    assert canonical_fingerprint(before_driver_change) != canonical_fingerprint(
        benchmark_runtime._windows_cuda_provenance()
    )
    state.status = "cpu_selected"
    unobserved = benchmark_runtime._windows_cuda_provenance()
    assert unobserved is not None
    assert unobserved["runtime_identity"] is None


def test_corrupt_windows_runtime_provenance_is_not_silently_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    def corrupt() -> None:
        raise ValueError("corrupt runtime manifest")

    monkeypatch.setattr(benchmark_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0")
    monkeypatch.setitem(
        sys.modules,
        "phydrax_cuda13_pjrt.runtime",
        SimpleNamespace(
            installation_record=corrupt,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "phydrax_cuda13_pjrt.plugin",
        SimpleNamespace(
            bootstrap_state=lambda: None,
        ),
    )
    with pytest.raises(ValueError, match="corrupt runtime manifest"):
        benchmark_runtime._windows_cuda_provenance()

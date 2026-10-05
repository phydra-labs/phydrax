#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from phydrax._fingerprint import canonical_fingerprint


_T = TypeVar("_T")
_Lowered = TypeVar("_Lowered")
_Compiled = TypeVar("_Compiled")
_DurationUnit = Literal["seconds", "milliseconds"]
_PACKAGE_SEPARATOR = re.compile(r"[-_.]+")
_PERFORMANCE_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NPROC",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "JAX_COMPILATION_CACHE_DIR",
)
_SOURCE_BUILD_DIRECTORIES = ("phydrax", "native")
_SOURCE_BUILD_FILES = ("pyproject.toml", "uv.lock")
_IGNORED_SOURCE_PARTS = (
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ty_cache",
    "build",
    "dist",
)


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    """Content identity for code and evidence interpreted by a benchmark record."""

    source_build_fingerprint: str
    benchmark_driver_fingerprint: str
    evidence_fields: tuple[str, ...]
    evidence_schema_signature: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "source_build_fingerprint": self.source_build_fingerprint,
            "benchmark_driver_fingerprint": self.benchmark_driver_fingerprint,
            "evidence_schema": {
                "fields": list(self.evidence_fields),
                "signature": self.evidence_schema_signature,
            },
        }


class CompilerMemoryAnalysis(Protocol):
    argument_size_in_bytes: int
    output_size_in_bytes: int
    temp_size_in_bytes: int
    generated_code_size_in_bytes: int


@dataclass(frozen=True, slots=True)
class DurationDistribution:
    """Finite nonnegative timing samples with derived robust summaries."""

    samples_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        samples = tuple(float(value) for value in self.samples_seconds)
        if any(not math.isfinite(value) or value < 0.0 for value in samples):
            raise ValueError("Duration samples must be finite and nonnegative.")
        object.__setattr__(self, "samples_seconds", samples)

    @property
    def count(self) -> int:
        return len(self.samples_seconds)

    @property
    def minimum_seconds(self) -> float | None:
        return None if not self.samples_seconds else min(self.samples_seconds)

    @property
    def median_seconds(self) -> float | None:
        return (
            None if not self.samples_seconds else statistics.median(self.samples_seconds)
        )

    @property
    def mean_seconds(self) -> float | None:
        return (
            None if not self.samples_seconds else statistics.fmean(self.samples_seconds)
        )

    @property
    def population_std_seconds(self) -> float | None:
        return (
            None if not self.samples_seconds else statistics.pstdev(self.samples_seconds)
        )

    @property
    def maximum_seconds(self) -> float | None:
        return None if not self.samples_seconds else max(self.samples_seconds)

    def to_dict(self, /, *, unit: _DurationUnit = "seconds") -> dict[str, Any]:
        """Serialize with field names and values in one explicit unit."""
        if unit == "seconds":
            scale = 1.0
            suffix = "seconds"
        elif unit == "milliseconds":
            scale = 1_000.0
            suffix = "ms"
        else:
            raise ValueError("Duration unit must be 'seconds' or 'milliseconds'.")
        samples = [scale * value for value in self.samples_seconds]
        if not samples:
            minimum = median = mean = deviation = maximum = None
        else:
            minimum = min(samples)
            median = statistics.median(samples)
            mean = statistics.fmean(samples)
            deviation = statistics.pstdev(samples)
            maximum = max(samples)
        return {
            "count": len(samples),
            f"samples_{suffix}": samples,
            f"min_{suffix}": minimum,
            f"median_{suffix}": median,
            f"mean_{suffix}": mean,
            f"std_{suffix}": deviation,
            f"max_{suffix}": maximum,
        }

    def to_seconds_dict(self) -> dict[str, Any]:
        return self.to_dict(unit="seconds")

    def to_milliseconds_dict(self) -> dict[str, Any]:
        return self.to_dict(unit="milliseconds")


@dataclass(frozen=True, slots=True)
class CompilationTiming:
    """Host lowering and executable compilation durations."""

    lowering_seconds: float
    compilation_seconds: float

    def __post_init__(self) -> None:
        values = (float(self.lowering_seconds), float(self.compilation_seconds))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Compilation timings must be finite and nonnegative.")
        object.__setattr__(self, "lowering_seconds", values[0])
        object.__setattr__(self, "compilation_seconds", values[1])


@dataclass(frozen=True, slots=True)
class CompilerEvidence:
    """Compiler-estimated operation and executable-memory evidence."""

    flops: int | None
    bytes_accessed: int | None
    argument_bytes: int | None
    output_bytes: int | None
    temporary_bytes: int | None
    generated_code_bytes: int | None
    source: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Compiler evidence source must be non-empty.")
        values = (
            self.flops,
            self.bytes_accessed,
            self.argument_bytes,
            self.output_bytes,
            self.temporary_bytes,
            self.generated_code_bytes,
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in values
        ):
            raise ValueError(
                "Compiler evidence values must be nonnegative integers or None."
            )
        if all(value is None for value in values) and not self.unavailable_reason:
            raise ValueError("Unavailable compiler evidence requires a reason.")

    @property
    def estimated_device_memory_bytes(self) -> int | None:
        """Compiler-estimated arguments, outputs, and temporaries resident on device."""
        values = (self.argument_bytes, self.output_bytes, self.temporary_bytes)
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)


@dataclass(frozen=True, slots=True)
class DeviceEnvironment:
    platform: str
    kind: str

    def __post_init__(self) -> None:
        if not self.platform or not self.kind:
            raise ValueError("Device platform and kind must be non-empty.")

    def to_dict(self) -> dict[str, str]:
        return {"platform": self.platform, "kind": self.kind}


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Comparison-relevant software, hardware, and execution configuration."""

    python_version: str
    phydrax_version: str
    numpy_version: str
    jax_version: str
    jaxlib_version: str
    platform: str
    machine: str
    processor: str
    logical_cpus: int
    backend: str
    x64_enabled: bool
    default_float_dtype: str
    devices: tuple[DeviceEnvironment, ...]
    performance_environment: tuple[tuple[str, str | None], ...]
    package_fingerprint: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "python_version": self.python_version,
            "phydrax_version": self.phydrax_version,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "logical_cpus": self.logical_cpus,
            "numpy_version": self.numpy_version,
            "jaxlib_version": self.jaxlib_version,
            "default_float_dtype": self.default_float_dtype,
            "package_fingerprint": self.package_fingerprint,
            "jax": {
                "version": self.jax_version,
                "backend": self.backend,
                "x64_enabled": self.x64_enabled,
                "devices": [device.to_dict() for device in self.devices],
            },
            "performance_environment": dict(self.performance_environment),
        }


def _array_leaves(value: Any, /) -> tuple[jax.Array | np.ndarray, ...]:
    arrays: list[jax.Array | np.ndarray] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)
        if isinstance(item, (jax.Array, np.ndarray)):
            arrays.append(item)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        leaves = jax.tree.leaves(item)
        if not (len(leaves) == 1 and leaves[0] is item):
            for leaf in leaves:
                visit(leaf)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))

    visit(value)
    return tuple(arrays)


def synchronize(value: _T, /) -> _T:
    """Block every JAX array reachable through an acyclic PyTree or dataclass graph."""
    for leaf in _array_leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    return value


def measure_host(operation: Callable[[], _T], /) -> tuple[_T, float]:
    """Measure guaranteed host-only work without a device synchronization step."""
    started = time.perf_counter_ns()
    value = operation()
    return value, (time.perf_counter_ns() - started) / 1_000_000_000.0


def measure_synchronized(
    operation: Callable[[], _T],
    /,
    *,
    synchronizer: Callable[[_T], object] = synchronize,
) -> tuple[_T, float]:
    """Measure one operation through full result synchronization."""
    started = time.perf_counter_ns()
    value = operation()
    synchronizer(value)
    return value, (time.perf_counter_ns() - started) / 1_000_000_000.0


def measure_repeated(
    operation: Callable[[], _T],
    /,
    *,
    warmup: int,
    repeats: int,
    synchronizer: Callable[[_T], object] = synchronize,
) -> tuple[_T, DurationDistribution]:
    """Run synchronized warmups and retain every synchronized steady sample."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive.")
    for _ in range(warmup):
        synchronizer(operation())
    result, elapsed = measure_synchronized(operation, synchronizer=synchronizer)
    samples = [elapsed]
    for _ in range(repeats - 1):
        result, elapsed = measure_synchronized(operation, synchronizer=synchronizer)
        samples.append(elapsed)
    return result, DurationDistribution(tuple(samples))


def measure_lower_and_compile(
    lower: Callable[[], _Lowered],
    compile: Callable[[_Lowered], _Compiled],
    /,
) -> tuple[_Compiled, CompilationTiming]:
    """Measure lowering and compilation as separate host phases."""
    lowered, lowering_seconds = measure_host(lower)
    compiled, compilation_seconds = measure_host(lambda: compile(lowered))
    return compiled, CompilationTiming(lowering_seconds, compilation_seconds)


def compiler_evidence(
    cost_analysis: Mapping[str, float] | None,
    memory_analysis: CompilerMemoryAnalysis | None,
    /,
    *,
    source: str,
    unavailable_reason: str | None = None,
) -> CompilerEvidence:
    """Normalize official compiler cost and memory estimates without heuristics."""
    flops = _analysis_integer(cost_analysis, "flops")
    bytes_accessed = _analysis_integer(cost_analysis, "bytes accessed")
    if memory_analysis is None:
        argument_bytes = output_bytes = temporary_bytes = generated_code_bytes = None
    else:
        argument_bytes = int(memory_analysis.argument_size_in_bytes)
        output_bytes = int(memory_analysis.output_size_in_bytes)
        temporary_bytes = int(memory_analysis.temp_size_in_bytes)
        generated_code_bytes = int(memory_analysis.generated_code_size_in_bytes)
    return CompilerEvidence(
        flops=flops,
        bytes_accessed=bytes_accessed,
        argument_bytes=argument_bytes,
        output_bytes=output_bytes,
        temporary_bytes=temporary_bytes,
        generated_code_bytes=generated_code_bytes,
        source=source,
        unavailable_reason=(
            unavailable_reason
            if flops is None or bytes_accessed is None or memory_analysis is None
            else None
        ),
    )


def logical_array_bytes(value: Any, /) -> int:
    """Return unique logical JAX/NumPy array payload bytes in an object graph."""
    return sum(leaf.nbytes for leaf in _array_leaves(value))


def source_build_fingerprint(project_root: Path, /) -> str:
    """Fingerprint the current package source and repository build inputs."""
    root = _repository_root(project_root)
    required = tuple(root / name for name in _SOURCE_BUILD_FILES)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(
            "Source/build identity requires pyproject.toml and uv.lock."
        )
    paths = list(required)
    for directory_name in _SOURCE_BUILD_DIRECTORIES:
        directory = root / directory_name
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Source/build input must be a directory: {directory}"
            )
        candidates = (
            directory.rglob("*.py")
            if directory_name == "phydrax"
            else directory.rglob("*")
        )
        paths.extend(
            path
            for path in candidates
            if path.is_file()
            and not any(
                part in _IGNORED_SOURCE_PARTS
                for part in path.relative_to(directory).parts
            )
            and path.suffix not in {".pyc", ".pyo"}
        )
    return canonical_fingerprint(
        {
            "kind": "phydrax-current-source-build",
            "files": _file_records(root, paths),
        }
    )


def _driver_source_paths(root: Path, initial: Iterable[Path], /) -> tuple[Path, ...]:
    """Follow first-party scientific imports without executing their modules.

    Local/lazy imports and example oracles are source dependencies too. Package
    and native kernels remain owned by ``source_build_fingerprint``.
    """
    pending = list(initial)
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        package = path.relative_to(root).parent.parts
        for depth in range(1, len(package) + 1):
            initializer = root.joinpath(*package[:depth], "__init__.py")
            if initializer.is_file():
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            modules: tuple[str, ...]
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = (
                    ".".join(package[: len(package) - node.level + 1])
                    if node.level
                    else ""
                )
                module = ".".join(part for part in (prefix, node.module) if part)
                modules = (module, *(f"{module}.{alias.name}" for alias in node.names))
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[0] not in ("benchmarks", "examples"):
                    continue
                module_path = root.joinpath(*parts)
                candidates = (module_path.with_suffix(".py"), module_path / "__init__.py")
                pending.extend(candidate for candidate in candidates if candidate.is_file())
    return tuple(sorted(visited, key=lambda path: path.relative_to(root).as_posix()))


def benchmark_driver_fingerprint(project_root: Path, driver_path: Path, /) -> str:
    """Fingerprint a driver, runtime harness, and imported scientific/oracle sources."""
    root = _repository_root(project_root)
    driver = driver_path.resolve()
    try:
        driver.relative_to(root)
    except ValueError as error:
        raise ValueError("Benchmark driver must be within the project root.") from error
    runtime = root / "benchmarks" / "_runtime.py"
    paths = tuple(dict.fromkeys((driver, runtime)))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            "Benchmark-driver identity requires the driver and benchmarks/_runtime.py."
        )
    return canonical_fingerprint(
        {
            "kind": "phydrax-benchmark-driver",
            "files": _file_records(root, _driver_source_paths(root, paths)),
        }
    )


def capture_benchmark_identity(
    project_root: Path,
    driver_path: Path,
    evidence_fields: Iterable[str],
    /,
) -> BenchmarkIdentity:
    """Capture source, driver, and serialized evidence-schema identity."""
    if isinstance(evidence_fields, (str, bytes)):
        raise TypeError("Evidence fields must be an iterable of field names.")
    normalized_fields = tuple(sorted(evidence_fields))
    if (
        not normalized_fields
        or any(not isinstance(name, str) or not name for name in normalized_fields)
        or len(set(normalized_fields)) != len(normalized_fields)
    ):
        raise ValueError("Evidence fields must be unique non-empty strings.")
    schema_signature = canonical_fingerprint(
        {
            "kind": "benchmark-evidence-field-set",
            "fields": list(normalized_fields),
        }
    )
    identity = {
        "source_build_fingerprint": source_build_fingerprint(project_root),
        "benchmark_driver_fingerprint": benchmark_driver_fingerprint(
            project_root, driver_path
        ),
        "evidence_schema": {
            "fields": list(normalized_fields),
            "signature": schema_signature,
        },
    }
    return BenchmarkIdentity(
        source_build_fingerprint=identity["source_build_fingerprint"],
        benchmark_driver_fingerprint=identity["benchmark_driver_fingerprint"],
        evidence_fields=normalized_fields,
        evidence_schema_signature=schema_signature,
        fingerprint=canonical_fingerprint(identity),
    )


def validate_benchmark_record(
    record: Mapping[str, Any], expected: BenchmarkIdentity, /
) -> None:
    """Reject a stored record that does not describe the current executable evidence."""
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Benchmark record is missing its identity.")
    expected_identity = expected.to_dict()
    if set(identity) != set(expected_identity):
        raise ValueError(
            "Benchmark identity field set does not match the current schema."
        )
    evidence_schema = identity["evidence_schema"]
    if not isinstance(evidence_schema, Mapping) or set(evidence_schema) != {
        "fields",
        "signature",
    }:
        raise ValueError("Benchmark evidence-schema field set is invalid.")
    if identity["source_build_fingerprint"] != expected.source_build_fingerprint:
        raise ValueError(
            "Benchmark record source/build fingerprint does not match current sources."
        )
    if identity["benchmark_driver_fingerprint"] != expected.benchmark_driver_fingerprint:
        raise ValueError(
            "Benchmark record driver fingerprint does not match the current driver."
        )
    if (
        evidence_schema["fields"] != list(expected.evidence_fields)
        or evidence_schema["signature"] != expected.evidence_schema_signature
    ):
        raise ValueError(
            "Benchmark record evidence field set/signature does not match current evidence."
        )
    if identity["fingerprint"] != expected.fingerprint:
        raise ValueError(
            "Benchmark record identity fingerprint does not match current evidence."
        )
    cases = record.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ValueError("Benchmark record cases must be a sequence.")
    expected_fields = set(expected.evidence_fields)
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Benchmark case {index} must be a mapping.")
        evidence = case.get("last_step_evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
            raise ValueError(
                f"Benchmark case {index} evidence field set does not match "
                "the recorded schema."
            )


def _repository_root(project_root: Path, /) -> Path:
    if not isinstance(project_root, Path):
        raise TypeError("Project root must be a pathlib.Path.")
    root = project_root.resolve()
    if not root.is_dir() or not (root / "phydrax").is_dir():
        raise FileNotFoundError(
            "Project root must be a directory containing the phydrax package."
        )
    return root


def _file_records(root: Path, paths: Iterable[Path], /) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        set(paths), key=lambda candidate: candidate.relative_to(root).as_posix()
    ):
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError(f"Source/build input changed while hashing: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": after.st_size,
                "sha256": digest,
            }
        )
    return records


def installed_package_fingerprint() -> str:
    """Fingerprint normalized installed distribution names and versions."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            continue
        name = _PACKAGE_SEPARATOR.sub("-", raw_name).lower()
        version = distribution.version
        if name in packages and packages[name] != version:
            raise ValueError(
                f"Installed distribution {name!r} has conflicting versions "
                f"{packages[name]!r} and {version!r}."
            )
        packages[name] = version
    return canonical_fingerprint(sorted(packages.items()))


def capture_environment() -> RuntimeEnvironment:
    """Capture deterministic comparison-relevant runtime evidence."""
    devices = tuple(
        sorted(
            (
                DeviceEnvironment(device.platform, device.device_kind)
                for device in jax.devices()
            ),
            key=lambda device: (device.platform, device.kind),
        )
    )
    performance_environment = tuple(
        (key, os.environ.get(key)) for key in _PERFORMANCE_ENVIRONMENT_KEYS
    )
    python_version = platform.python_version()
    phydrax_version = importlib.metadata.version("phydrax")
    package_fingerprint = installed_package_fingerprint()
    backend = jax.default_backend()
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    default_float_dtype = str(jnp.asarray(0.0).dtype)
    evidence = {
        "python_version": python_version,
        "phydrax_version": phydrax_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "numpy_version": np.__version__,
        "jaxlib_version": jaxlib.__version__,
        "default_float_dtype": default_float_dtype,
        "package_fingerprint": package_fingerprint,
        "jax": {
            "version": jax.__version__,
            "backend": backend,
            "x64_enabled": x64_enabled,
            "devices": [device.to_dict() for device in devices],
        },
        "performance_environment": dict(performance_environment),
    }
    return RuntimeEnvironment(
        python_version=python_version,
        phydrax_version=phydrax_version,
        numpy_version=np.__version__,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        platform=evidence["platform"],
        machine=evidence["machine"],
        processor=evidence["processor"],
        logical_cpus=evidence["logical_cpus"],
        backend=backend,
        x64_enabled=x64_enabled,
        default_float_dtype=default_float_dtype,
        devices=devices,
        performance_environment=performance_environment,
        package_fingerprint=package_fingerprint,
        fingerprint=canonical_fingerprint(evidence),
    )


def _analysis_integer(analysis: Mapping[str, float] | None, key: str, /) -> int | None:
    if analysis is None or key not in analysis:
        return None
    value = float(analysis[key])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"Compiler analysis field {key!r} must be finite and nonnegative."
        )
    return round(value)


__all__ = [
    "BenchmarkIdentity",
    "CompilationTiming",
    "CompilerEvidence",
    "DeviceEnvironment",
    "DurationDistribution",
    "RuntimeEnvironment",
    "benchmark_driver_fingerprint",
    "capture_benchmark_identity",
    "capture_environment",
    "compiler_evidence",
    "installed_package_fingerprint",
    "logical_array_bytes",
    "measure_host",
    "measure_lower_and_compile",
    "measure_repeated",
    "measure_synchronized",
    "source_build_fingerprint",
    "synchronize",
    "validate_benchmark_record",
]

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import Any, final, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax
import numpy as np

import phydrax as phx
from phydrax import canonical_fingerprint, StrictModule
from phydrax._meshcore import NativeParallelExecution
from phydrax.meshing import (
    load_meshcore,
    meshcore_available,
    meshcore_identity,
    meshcore_runtime_identity,
)
from tools._meshing_cases import python_source_revision, source_tree_revision
from tools.compile_counter import CompileCounter


class _CompiledProgram[T](Protocol):
    compiled: jax.stages.Compiled

    def __call__(self, *args: object, **kwargs: object) -> T: ...


class _LoweredProgram[T](Protocol):
    def compile(self) -> _CompiledProgram[T]: ...


@runtime_checkable
class _CorpusJitEntry(Protocol):
    def lower(
        self, *args: object, **kwargs: object
    ) -> _LoweredProgram[tuple[jax.Array, phx.linalg.LinearSolveResult]]: ...


@runtime_checkable
class _PolyhedralJitEntry(Protocol):
    def lower(
        self, *args: object, **kwargs: object
    ) -> _LoweredProgram[
        tuple[jax.Array, jax.Array, phx.linalg.LinearSolveDiagnostics]
    ]: ...


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _runtime_identity() -> dict[str, object]:
    """Identity of the package, driver, and native binary loaded."""
    available = meshcore_available()
    library = load_meshcore() if available else None
    meshcore = meshcore_identity() if available else "unavailable"
    native = meshcore_runtime_identity() if available else None
    if native is not None and library is not None:
        native = {
            **native,
            "loaded_path": str(library.path.resolve()),
            "identity_id": canonical_fingerprint(
                {
                    "release": native["release"],
                    "source_digest": native["source_digest"],
                    "binary_digest": native["binary_digest"],
                    "configuration": native["configuration"],
                }
            ),
        }
    package_root = Path(phx.__file__).resolve().parent
    distribution = importlib.metadata.distribution("phydrax")
    build = {
        "kind": "meshing-qualification-build",
        "distribution_release": distribution.version,
        "distribution_path": str(distribution.locate_file("")),
        "loaded_python_path": str(package_root),
        "loaded_python_source_digest": python_source_revision(package_root),
        "driver_source_revision": source_tree_revision(),
        "native_binary": native,
    }
    environment = {
        "kind": "meshing-qualification-environment",
        "python_release": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax_release": jax.__version__,
        "numpy_release": np.__version__,
        "native_configuration": None if native is None else native["configuration"],
        "native_library_override": os.environ.get("PHYDRAX_MESHCORE_LIBRARY"),
        "devices": [
            {
                "platform": device.platform,
                "kind": device.device_kind,
                "id": device.id,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
    }
    build_id = canonical_fingerprint(build)
    environment_id = canonical_fingerprint(environment)
    identity = phx.qualification.QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jax.dtypes.canonicalize_dtype(np.dtype(np.float64))),
    )
    return {
        **identity.to_record(),
        "meshcore": meshcore,
        "meshcore_build": native,
        "build": build,
        "environment": environment,
        "source_revision": source_tree_revision(),
    }


def _measured[T](call: Callable[[], T], /) -> tuple[T, dict[str, float]]:
    """Run one stage; report wall seconds and the process peak resident bytes."""
    started = perf_counter()
    value = call()
    elapsed = perf_counter() - started
    peak = peak_resident_bytes(resource.RUSAGE_SELF)
    return value, {"wall_seconds": elapsed, "peak_resident_bytes": peak}


def _missing_dependency(dependency: str, detail: str, /) -> dict[str, object]:
    """Explicit record of a scenario that cannot run in this environment."""
    return {
        "status": "missing-dependency",
        "dependency": dependency,
        "detail": detail,
        "runtime": _runtime_identity(),
    }


def qualification_failure(
    field: str, expectation: str, values: dict[str, Any]
) -> QualificationFailure:
    """Construct a scientific refusal with bounded observed-value evidence."""
    return QualificationFailure(
        f"qualification.{field} must satisfy its scientific contract: {expectation}; "
        f"got {observed_summary(values)}"
    )


class ColdWorkerTimeout(TimeoutError):
    def __init__(self, message: str, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


@final
class QualificationFailure(RuntimeError):
    """A qualification invariant failed with stage and value evidence."""


def observed_summary(values: dict[str, Any]) -> str:
    """Render bounded diagnostic values without copying numerical buffers."""
    summaries: list[str] = []
    for name, value in values.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            summaries.append(f"{name}={value!r}")
        elif isinstance(value, (np.ndarray, jax.Array)):
            summaries.append(f"{name}.shape={value.shape}, {name}.dtype={value.dtype}")
        elif isinstance(value, (tuple, list, dict)):
            summaries.append(f"{name}.length={len(value)}")
        else:
            summaries.append(f"{name}.type={type(value).__qualname__}")
    return ", ".join(summaries)


_ACCOUNTING_MODULES = frozenset(
    {
        "phydrax._native_execution",
        "phydrax._resource_ticket",
        "phydrax.discretization._coordinate_enclosure",
        "phydrax.meshing._native_preparation",
        "phydrax._meshcore",
    }
)


_ACCOUNTING_CALLS = frozenset(
    {
        "consume",
        "batch",
        "reserve",
        "settle",
        "remaining",
        "reserve_bytes",
        "release_bytes",
        "account_work",
        "account_query",
        "account_cavity",
        "execution_consume",
        "execution_remaining",
        "execution_reserve",
        "execution_settle",
        "__enter__",
        "__exit__",
    }
)


@final
class Deadline(StrictModule):
    expires_at: float = eqx.field(static=True)

    def remaining(self) -> float:
        remaining = self.expires_at - perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(
                f"qualification.deadline must remain positive; got {remaining} seconds"
            )
        return remaining


@final
class StageLedger(StrictModule):
    record: dict[str, Any] = eqx.field(static=True)
    deadline: Deadline | None = eqx.field(static=True, default=None)
    cold_phase: str | None = eqx.field(static=True, default=None)

    def inherit_failure(self, other: dict[str, Any]) -> None:
        """Preserve the failed child stage when rethrowing its exception."""
        self.record["active_phase"] = other["active_phase"]

    def run[T](self, name: str, call: Callable[[], T]) -> T:
        """Retain stage evidence; accounting profiling is diagnostic-process only.

        PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS=1 enables inclusive accounting
        call-time profiling. Default timing runs keep that field unmeasured and
        never install a Python profiler. Compiler events and RSS retain their
        seconds/bytes units in both modes.
        PHYDRAX_MESHING_COMPILE_STACKS=1 retains first-party call sites beside
        each kernel's abstract signature; it does not change compile counting.
        PHYDRAX_MESHING_COMPILE_EQUATIONS=1 also retains static lowering equations
        and output/layout signatures for distinguishing primitive parameter variants.
        """
        if self.deadline is not None:
            self.deadline.remaining()
        self.record["active_phase"] = name
        accounting_enabled = (
            os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") == "1"
        )
        compiler = CompileCounter(
            stage=name,
            capture_stack=os.environ.get("PHYDRAX_MESHING_COMPILE_STACKS") == "1",
            capture_equations=os.environ.get("PHYDRAX_MESHING_COMPILE_EQUATIONS") == "1",
        )
        accounting: dict[str, float | int] = {"seconds": 0.0, "depth": 0, "started": 0.0}
        previous_profile = sys.getprofile()

        def profile(
            frame: FrameType,
            event: Literal["call", "return", "c_call", "c_return", "c_exception"],
            arg: object,
        ) -> None:
            if previous_profile is not None:
                previous_profile(frame, event, arg)
            if "__name__" not in frame.f_globals:
                return
            if (
                frame.f_globals["__name__"] not in _ACCOUNTING_MODULES
                or frame.f_code.co_name not in _ACCOUNTING_CALLS
            ):
                return
            if event == "call":
                if accounting["depth"] == 0:
                    accounting["started"] = perf_counter()
                accounting["depth"] += 1
            elif event == "return":
                accounting["depth"] -= 1
                if accounting["depth"] == 0:
                    accounting["seconds"] += perf_counter() - accounting["started"]

        started = perf_counter()
        if accounting_enabled:
            sys.setprofile(profile)
        try:
            with (
                compiler,
                nullcontext()
                if self.deadline is None
                else NativeParallelExecution(None, self.deadline.remaining()),
            ):
                value = call()
            if "attempts" in self.record:
                self.record["attempts"].append({"phase": name, "status": "passed"})
            if self.deadline is not None:
                self.deadline.remaining()
            return value
        finally:
            if accounting_enabled:
                sys.setprofile(previous_profile)
            elapsed = perf_counter() - started
            self.record.setdefault("stages_seconds", {})[name] = elapsed
            if "stage_peak_process_bytes" in self.record:
                self.record["stage_peak_process_bytes"][name] = peak_resident_bytes()
            stages = self.record.setdefault("stage_measurements", {})
            prior = stages.get(name, {})
            stages[name] = {
                "wall_seconds": prior.get("wall_seconds", 0.0) + elapsed,
                "invocations": prior.get("invocations", 0) + 1,
                "eager_xla_compile_count": prior.get("eager_xla_compile_count", 0)
                + compiler.count,
                "eager_xla_compile_seconds": prior.get("eager_xla_compile_seconds", 0.0)
                + compiler.seconds,
                "compile_events": prior.get("compile_events", [])
                + [event.to_record() for event in compiler.events],
                "python_resource_accounting_seconds": (
                    (prior.get("python_resource_accounting_seconds") or 0.0)
                    + accounting["seconds"]
                    if accounting_enabled
                    else None
                ),
                "python_resource_accounting_measurement": (
                    "inclusive-profiled-accounting-call-wall-time"
                    if accounting_enabled
                    else "unmeasured-clean-timing-no-accounting-profiler"
                ),
                "accounting_diagnostics": accounting_enabled,
                "peak_resident_bytes": peak_resident_bytes(),
                "peak_resident_unit": "bytes",
            }


def peak_resident_bytes(who: int = resource.RUSAGE_SELF) -> int:
    """Return process-lifetime peak RSS in bytes on Linux and macOS."""
    peak = resource.getrusage(who).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run_cold_worker(
    entry: str,
    argv: tuple[str, ...],
    *,
    deadline: Deadline,
    ledger: StageLedger,
) -> dict[str, Any]:
    """Run an independent module worker within the remaining deadline."""
    command = [
        sys.executable,
        "-m",
        f"tools.meshing_qualification.{entry}",
        "worker",
        *argv,
    ]

    def launch() -> subprocess.CompletedProcess[str]:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as child:
            try:
                stdout, stderr = child.communicate(timeout=deadline.remaining())
            except (subprocess.TimeoutExpired, TimeoutError) as failure:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
                raise ColdWorkerTimeout(
                    f"qualification.worker.deadline must allow completion; got entry={entry!r}, pid={child.pid}",
                    stdout,
                    stderr,
                ) from failure
            return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)

    phase = ledger.cold_phase or ledger.record.get("active_phase", "cold_worker")
    completed = ledger.run(phase, launch)
    if completed.returncode:
        evidence = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        ledger.record["cold_process_refusal"] = evidence
        raise QualificationFailure(
            f"qualification.worker.returncode must equal zero; got entry={entry!r}, {observed_summary(evidence)}"
        )
    cold = json.loads(completed.stdout)
    if not isinstance(cold, dict):
        raise TypeError(
            f"qualification.worker.record must be a mapping; got {type(cold).__qualname__}"
        )
    process_id = cold.get("process_id")
    if not isinstance(process_id, int) or process_id <= 0 or process_id == os.getpid():
        raise QualificationFailure(
            f"qualification.worker.process_id must identify an independent process; got {process_id!r}"
        )
    return cold


def run_campaign_process(
    command: list[str], *, cwd: Path, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a session-isolated sample and cancel its descendants on a hard timeout."""
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            raise
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)

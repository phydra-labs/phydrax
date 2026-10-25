# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import Any, BinaryIO, final, Literal, Protocol, runtime_checkable, TypedDict

import equinox as eqx
import jax
import numpy as np

import phydrax as phx
from phydrax import canonical_fingerprint, StrictModule
from phydrax._execution_sampling import sample_host_memory
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


# User-approved observation of this installed CPU provider, not a blanket CPU
# exception. A changed provider/version/API surface must be assessed afresh.
_CPU_PEAK_PROBE_SHA256 = "d598707891c0870a12651eb1d398bab74db9b21288ed7d1ffd2c5ae9e83a5303"
_CPU_PEAK_PROBE_API = {
    "jax_release": "0.11.2",
    "jaxlib_release": "0.11.2",
    "device_kind": "cpu",
    "device_memory_api": ["default_memory", "memory", "memory_stats"],
    "client_memory_api": ["heap_profile", "live_arrays", "live_buffers", "live_executables"],
    "runtime_memory_api": [
        "CompiledMemoryStats", "Memory", "check_and_canonicalize_memory_kind",
        "json_to_pprof_profile", "pprof_profile_to_json",
    ],
    "profiler_memory_api": [
        "ProfileOptions", "ProfilerServer", "ProfilerSession",
        "aggregate_profiled_instructions", "get_fdo_profile",
        "get_instructions_profile", "get_profiled_instructions_proto",
        "register_plugin_profiler",
    ],
    "profiler_options": [
        "advanced_configuration", "duration_ms", "enable_hlo_proto",
        "host_tracer_level", "include_dataset_ops", "python_tracer_level",
        "raise_error_on_start_failure", "repository_path", "session_id",
        "start_timestamp_ns",
    ],
    "memory_spaces": ["device", "pinned_host", "unpinned_host"],
}


def _cpu_device_peak_capability(device: jax.Device) -> dict[str, object] | None:
    """Reobserve the approved unsupported capability, never a missing peak value.

    No profiling snapshot or process measurement is converted into a peak. The
    exact installed probe establishes the interpretation of this API surface;
    the live provider must still match it and explicitly return unsupported.
    """
    if device.platform != "cpu":
        return None
    import jaxlib

    if jax.__version__ != _CPU_PEAK_PROBE_API["jax_release"] or (
        jaxlib.__version__ != _CPU_PEAK_PROBE_API["jaxlib_release"]
    ):
        return None
    try:
        runtime = importlib.import_module("jaxlib._jax")
        profiler = importlib.import_module("jaxlib._profiler")

        if device.memory_stats() is not None:
            return None
        signature = {
            "jax_release": jax.__version__,
            "jaxlib_release": jaxlib.__version__,
            "device_kind": device.device_kind,
            "device_memory_api": [
                name for name in dir(device)
                if any(word in name.lower() for word in ("memory", "heap", "alloc", "profile"))
            ],
            "client_memory_api": [
                name for name in dir(device.client)
                if any(word in name.lower() for word in ("memory", "heap", "alloc", "profile", "live"))
            ],
            "runtime_memory_api": [
                name for name in dir(runtime)
                if any(word in name.lower() for word in ("alloc", "memory", "heap", "profile"))
            ],
            "profiler_memory_api": [
                name for name in dir(profiler)
                if any(word in name.lower() for word in ("alloc", "memory", "heap", "profile"))
            ],
            "profiler_options": [
                name for name in dir(jax.profiler.ProfileOptions()) if not name.startswith("_")
            ],
            "memory_spaces": [space.kind for space in device.addressable_memories()],
        }
    except (AttributeError, ImportError, RuntimeError):
        return None
    if signature != _CPU_PEAK_PROBE_API:
        return None
    body = {
        "kind": "verified-cpu-device-allocator-peak-capability",
        "availability": "unsupported",
        "device": {
            "platform": device.platform, "kind": device.device_kind,
            "id": device.id, "process_index": device.process_index,
        },
        "jax_release": jax.__version__,
        "jaxlib_release": jaxlib.__version__,
        "api_signature_id": canonical_fingerprint(signature),
        "probe_evidence_sha256": _CPU_PEAK_PROBE_SHA256,
        "observed_memory_stats": None,
        "reason": (
            "The verified installed CPU provider does not expose a device-allocator "
            "high-water mark through supported APIs. Heap/live-array snapshots, "
            "process RSS and static compiler analysis are not that runtime peak."
        ),
    }
    return {**body, "capability_id": canonical_fingerprint(body)}


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
        "process_memory_measurement_scope": PEAK_RESIDENT_SCOPE,
        "process_memory_acceptance": (
            "approved-macos-measured-process-memory-no-hard-execution-group-cap"
            if sys.platform == "darwin"
            else "process-peak-measurement-with-separately-authored-external-enforcement"
        ),
        "devices": [
            {
                "platform": device.platform,
                "kind": device.device_kind,
                "id": device.id,
                "process_index": device.process_index,
                "device_allocator_peak_capability": _cpu_device_peak_capability(device),
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


class StageMeasurement(TypedDict):
    wall_seconds: float
    peak_resident_bytes: int | None


def _measured[T](call: Callable[[], T], /) -> tuple[T, StageMeasurement]:
    """Run one stage; report wall seconds and the post-exec process peak bytes."""
    started = perf_counter()
    value = call()
    elapsed = perf_counter() - started
    return value, {"wall_seconds": elapsed, "peak_resident_bytes": peak_resident_bytes()}


def epoch_stage_name(stage: str, epoch: int, /) -> str:
    """Name an epoch-owned stage interval; campaign attribution reads exactly this key."""
    return f"{stage}_epoch_{epoch}"


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

LAUNCH_DEADLINE_VARIABLE = "PHYDRAX_QUALIFICATION_LAUNCH_DEADLINE"
"""Campaign launch expiry on the system-wide ``CLOCK_MONOTONIC`` clock."""

REPORTING_GRACE_SECONDS = 10.0
"""Launcher watchdog allowance after the scientific deadline, for the child's report only."""


def _inherited_launch_expiry() -> float | None:
    """Translate an inherited launch expiry into this process's ``perf_counter`` clock."""
    value = os.environ.get(LAUNCH_DEADLINE_VARIABLE)
    if value is None:
        return None
    expires = float(value)
    if not math.isfinite(expires):
        raise ValueError(
            f"qualification.{LAUNCH_DEADLINE_VARIABLE} must be a finite CLOCK_MONOTONIC expiry; got {value!r}"
        )
    return perf_counter() + (expires - time.clock_gettime(time.CLOCK_MONOTONIC))


_LAUNCH_EXPIRES_AT = _inherited_launch_expiry()


def launch_deadline_expired() -> bool | None:
    """Report whether an inherited campaign deadline has passed; None when not launched."""
    return None if _LAUNCH_EXPIRES_AT is None else perf_counter() >= _LAUNCH_EXPIRES_AT


@final
class Deadline(StrictModule):
    """Scientific wall deadline of one qualification process.

    A process started by ``run_campaign_process`` (or descended from one) measures
    the same deadline from the launcher's origin, so import time counts against it
    exactly as it does against the launcher; the earlier expiry always binds.
    """

    expires_at: float = eqx.field(static=True)

    def remaining(self) -> float:
        expires = (
            self.expires_at
            if _LAUNCH_EXPIRES_AT is None
            else min(self.expires_at, _LAUNCH_EXPIRES_AT)
        )
        remaining = expires - perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(
                f"qualification.deadline must remain positive; got {remaining} seconds"
            )
        return remaining

    def monotonic_expiry(self) -> float:
        """Return the binding expiry on the system-wide ``CLOCK_MONOTONIC`` clock.

        A child process given this value measures the same absolute deadline; an
        inherited launch expiry is passed through exactly when it binds.
        """
        own = time.clock_gettime(time.CLOCK_MONOTONIC) + (self.expires_at - perf_counter())
        inherited = os.environ.get(LAUNCH_DEADLINE_VARIABLE)
        return own if inherited is None else min(own, float(inherited))


class StageObserver(Protocol):
    """Optional host-only reporting, excluded from scientific module identity."""

    def __call__(
        self, record: dict[str, Any], name: str, event: str,
        deadline: Deadline | None, value: object, error: BaseException | None,
    ) -> None: ...


_STAGE_OBSERVER: ContextVar[StageObserver | None] = ContextVar(
    "qualification_stage_observer", default=None
)


@contextmanager
def observe_stages(observer: StageObserver) -> Iterator[None]:
    token = _STAGE_OBSERVER.set(observer)
    try:
        yield
    finally:
        _STAGE_OBSERVER.reset(token)


def notify_stage(
    record: dict[str, Any], name: str, event: str, deadline: Deadline | None,
    value: object = None, error: BaseException | None = None,
) -> None:
    observer = _STAGE_OBSERVER.get()
    if observer is not None:
        try:
            observer(record, name, event, deadline, value, error)
        except Exception as failure:
            # Reporting never changes the scientific callback or masks its
            # first exception. The scenario surfaces reporting failure separately.
            record.setdefault("reporting_failures", []).append(
                {"type": type(failure).__name__, "message": str(failure)}
            )


class StageLedgerAggregate(TypedDict):
    """Totals ``StageLedger.run`` accumulates for one stage name over all invocations."""

    wall_seconds: float
    invocations: int
    eager_xla_compile_count: int
    eager_xla_compile_seconds: float
    compile_events: list[dict[str, object]]
    python_resource_accounting_seconds: float | None
    python_resource_accounting_measurement: str
    accounting_diagnostics: bool
    peak_resident_bytes: int | None
    peak_resident_unit: str


@final
class StageLedger(StrictModule):
    record: dict[str, Any] = eqx.field(static=True)
    deadline: Deadline | None = eqx.field(static=True, default=None)
    cold_phase: str | None = eqx.field(static=True, default=None)
    local_boundary: Callable[[str], AbstractContextManager[None]] | None = eqx.field(
        static=True, default=None
    )

    def _local_step(self, name: str) -> AbstractContextManager[None]:
        return nullcontext() if self.local_boundary is None else self.local_boundary(name)

    @contextmanager
    def _execution(
        self, name: str, compiler: CompileCounter, profile: Callable[..., None] | None
    ) -> Iterator[None]:
        # Only context entry/exit is locally admitted. The scientific callback
        # remains outside admission and must own its internal failure protocol.
        with ExitStack() as contexts:
            with self._local_step(f"{name}.native-entry"):
                if profile is not None:
                    sys.setprofile(profile)
                contexts.enter_context(compiler)
                if self.deadline is not None:
                    contexts.enter_context(
                        NativeParallelExecution(None, self.deadline.remaining())
                    )
            yield
            with self._local_step(f"{name}.native-exit"):
                contexts.close()

    def inherit_failure(self, other: dict[str, Any]) -> None:
        """Preserve the failed child stage when rethrowing its exception."""
        self.record["active_phase"] = other["active_phase"]

    def run[T](self, name: str, call: Callable[[], T]) -> T:
        notify_stage(self.record, name, "entry", self.deadline)
        value: object = None
        failure: BaseException | None = None
        try:
            result = self._run(name, call)
            value = result
            return result
        except BaseException as error:
            failure = error
            notify_stage(self.record, name, "failure", self.deadline, error=error)
            raise
        finally:
            notify_stage(self.record, name, "exit", self.deadline, value, failure)

    def _run[T](self, name: str, call: Callable[[], T]) -> T:
        """Retain stage evidence; accounting profiling is diagnostic-process only.

        PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS=1 enables inclusive accounting
        call-time profiling. Default timing runs keep that field unmeasured and
        never install a Python profiler. Compiler events and RSS retain their
        seconds/bytes units in both modes.
        PHYDRAX_MESHING_COMPILE_STACKS=1 retains first-party call sites beside
        each kernel's abstract signature; it does not change compile counting.
        PHYDRAX_MESHING_COMPILE_EQUATIONS=1 also retains static lowering equations
        and output/layout signatures for distinguishing primitive parameter variants.

        Optional local_boundary admission covers only host setup, native context
        entry/exit, deadline checks, and measurement. The mixed scientific
        callback owns all internal collective failure agreement; its exception
        unwinds without introducing an additional collective here.
        """
        with self._local_step(f"{name}.admission"):
            self.record["active_phase"] = name
            if self.deadline is not None:
                self.deadline.remaining()
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
        try:
            with self._execution(name, compiler, profile if accounting_enabled else None):
                value = call()
            with self._local_step(f"{name}.completion"):
                if self.deadline is not None:
                    self.deadline.remaining()
                if "attempts" in self.record:
                    self.record["attempts"].append({"phase": name, "status": "passed"})
            return value
        finally:
            # Never introduce a new collective while unwinding a mixed callback.
            # Already-agreed boundary failures retain local measurements as well.
            with (
                nullcontext()
                if sys.exception() is not None
                else self._local_step(f"{name}.measurement")
            ):
                if accounting_enabled:
                    sys.setprofile(previous_profile)
                elapsed = perf_counter() - started
                self.record.setdefault("stages_seconds", {})[name] = elapsed
                if "stage_peak_process_bytes" in self.record:
                    self.record["stage_peak_process_bytes"][name] = peak_resident_bytes()
                stages = self.record.setdefault("stage_measurements", {})
                prior = stages.get(name, {})
                aggregate: StageLedgerAggregate = {
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
                stages[name] = aggregate


def _status_high_water(path: Path, /) -> int | None:
    """Read Linux ``VmHWM``: the resident high water of the current address space."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in lines:
        if line.startswith("VmHWM:"):
            value, unit = line.split()[1:3]
            if unit != "kB":
                raise ValueError(
                    f"qualification.VmHWM.unit must be 'kB'; got {unit!r} from {path}"
                )
            return int(value) * 1024
    return None


PEAK_RESIDENT_SCOPE = (
    "post-exec-process-high-water:linux-procfs-VmHWM;descendants-excluded"
    if sys.platform == "linux"
    else "post-exec-process-high-water:darwin-getrusage-bytes;descendants-excluded"
    if sys.platform == "darwin"
    else f"unavailable:no-post-exec-high-water-source-on-{sys.platform}"
)
"""Scope of ``peak_resident_bytes`` on this platform."""


def peak_resident_bytes() -> int | None:
    """Return this process's post-exec peak resident bytes, or None if unavailable.

    Linux ``VmHWM`` belongs to the address space created by exec; its raw
    ``getrusage`` counter retains pre-exec high water. Darwin's ``getrusage``
    reports the current Mach task's resident high water in bytes and resets
    across exec. Neither measurement includes descendants or enforces a cap.
    """
    if sys.platform == "darwin":
        return sample_host_memory().process_peak_resident.value_bytes
    if sys.platform == "linux":
        return _status_high_water(Path("/proc/self/status"))
    return None


def measured_peak_resident_bytes() -> int:
    """Return the post-exec peak for consumers whose contract requires a measurement."""
    peak = peak_resident_bytes()
    if peak is None:
        raise RuntimeError(
            f"qualification.peak_resident_bytes must be measured; got {PEAK_RESIDENT_SCOPE}"
        )
    return peak


COLD_WORKER_BOOTSTRAP = "tools.meshing_qualification._cold_worker"
"""Module a prepared cold worker runs: bind to its launcher, import, serve one packet."""

COLD_CONTROL_PACKET_BYTES = 1 << 20
"""Bound of the one newline-terminated JSON control packet, terminator included."""

_COLD_PACKET_FIELDS = frozenset({"archive_path", "content_id", "controls"})
_COLD_PROCESS_BINDING = (
    "module",
    "process_id",
    "parent_process_id",
    "session_id",
    "expires_monotonic",
    "packet_bytes",
    "packet_sha256",
    "content_id",
)


def _monotonic() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


@final
class ColdControlPacket(StrictModule):
    """The one validated control packet a prepared cold worker executes.

    ``deadline`` is the absolute expiry the worker inherited at launch; receiving the
    packet never resets it.
    """

    archive_path: Path = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    controls: dict[str, Any] = eqx.field(static=True)
    deadline: Deadline = eqx.field(static=True)


@final
class ColdWorkerReport(StrictModule):
    """A cold scenario record and its measured archive read-and-restore seconds.

    The runtime charges the rest of the scenario call, on its own clock, as solve
    seconds. ``record`` must not carry ``process_id``: the runtime binds the process.
    """

    record: dict[str, Any] = eqx.field(static=True)
    read_seconds: float = eqx.field(static=True)


def _read_cold_control_packet(descriptor: int, deadline: Deadline, /) -> bytes:
    """Read every byte up to the launcher's EOF, bounded in size and by the expiry."""
    packet = bytearray()
    while True:
        readable, _, _ = select.select([descriptor], [], [], deadline.remaining())
        if not readable:
            raise TimeoutError(
                f"qualification.cold_worker.packet must arrive before the inherited expiry; got {len(packet)} bytes"
            )
        chunk = os.read(descriptor, COLD_CONTROL_PACKET_BYTES + 1 - len(packet))
        if not chunk:
            break
        packet += chunk
        if len(packet) > COLD_CONTROL_PACKET_BYTES:
            raise ValueError(
                f"qualification.cold_worker.packet must not exceed {COLD_CONTROL_PACKET_BYTES} bytes; got more"
            )
    if packet.count(b"\n") != 1 or not packet.endswith(b"\n"):
        raise ValueError(
            f"qualification.cold_worker.packet must be exactly one newline-terminated line; got {len(packet)} bytes"
        )
    return bytes(packet)


def _reject_constant(value: str) -> float:
    raise ValueError(
        f"qualification.cold_worker.packet must contain only finite numbers; got {value}"
    )


def _decode_cold_control_packet(packet: bytes, deadline: Deadline, /) -> ColdControlPacket:
    value = json.loads(packet, parse_constant=_reject_constant)
    if not isinstance(value, dict) or set(value) != _COLD_PACKET_FIELDS:
        raise ValueError(
            f"qualification.cold_worker.packet must hold exactly {sorted(_COLD_PACKET_FIELDS)}; got {sorted(value) if isinstance(value, dict) else type(value).__qualname__}"
        )
    archive, content_id, controls = (
        value["archive_path"],
        value["content_id"],
        value["controls"],
    )
    if not isinstance(archive, str) or not Path(archive).is_absolute():
        raise ValueError(
            f"qualification.cold_worker.archive_path must be absolute; got {archive!r}"
        )
    if not Path(archive).is_file():
        raise FileNotFoundError(
            f"qualification.cold_worker.archive_path must name a published archive; got {archive!r}"
        )
    if not isinstance(content_id, str) or not content_id:
        raise ValueError(
            f"qualification.cold_worker.content_id must be a nonempty string; got {content_id!r}"
        )
    if not isinstance(controls, dict):
        raise ValueError(
            f"qualification.cold_worker.controls must be a JSON object; got {type(controls).__qualname__}"
        )
    return ColdControlPacket(
        archive_path=Path(archive),
        content_id=content_id,
        controls=controls,
        deadline=deadline,
    )


def serve_cold_worker(module: str, /, *, started: float, result_descriptor: int) -> None:
    """Import ``module``, execute its one control packet, and emit one bound result.

    Only ``COLD_WORKER_BOOTSTRAP`` calls this, after recording ``started`` on
    ``CLOCK_MONOTONIC`` and reserving the launcher's stdout as ``result_descriptor``.
    ``module.cold_worker`` receives a ``ColdControlPacket`` and returns a
    ``ColdWorkerReport``. The inherited launch expiry bounds the packet wait and the
    scenario call; the packet carries no clock. Any failure propagates as a nonzero
    exit with its traceback on stderr.
    """
    expires = os.environ.get(LAUNCH_DEADLINE_VARIABLE)
    if expires is None or _LAUNCH_EXPIRES_AT is None:
        raise ValueError(
            f"qualification.cold_worker must inherit {LAUNCH_DEADLINE_VARIABLE}; got None"
        )
    deadline = Deadline(_LAUNCH_EXPIRES_AT)
    program = getattr(importlib.import_module(module), "cold_worker", None)
    if not callable(program):
        raise TypeError(
            f"qualification.cold_worker.module must define callable cold_worker; got {module!r}"
        )
    imported = _monotonic()
    packet = _read_cold_control_packet(sys.stdin.fileno(), deadline)
    received = _monotonic()
    decoded = _decode_cold_control_packet(packet, deadline)
    deadline.remaining()
    execute_started = perf_counter()
    from tools.meshing_qualification._progress import run_with_progress

    report = run_with_progress(
        {"module": module, **decoded.controls}, lambda: program(decoded), cold=True,
    )
    execute_seconds = perf_counter() - execute_started
    if type(report) is not ColdWorkerReport:
        raise TypeError(
            f"qualification.cold_worker.report must be ColdWorkerReport; got {type(report).__qualname__}"
        )
    read_seconds = report.read_seconds
    if not math.isfinite(read_seconds) or not 0.0 <= read_seconds <= execute_seconds:
        raise ValueError(
            f"qualification.cold_worker.read_seconds must lie in [0, {execute_seconds}]; got {read_seconds!r}"
        )
    if "process_id" in report.record:
        raise ValueError(
            "qualification.cold_worker.record must leave process_id to the runtime; got a scenario value"
        )
    process = {
        "module": module,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "session_id": os.getsid(0),
        "expires_monotonic": float(expires),
        "packet_bytes": len(packet),
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "content_id": decoded.content_id,
        "bootstrap_started_monotonic": started,
        "import_completed_monotonic": imported,
        "packet_received_monotonic": received,
        "execute_completed_monotonic": _monotonic(),
        "import_seconds": imported - started,
        "packet_wait_seconds": received - imported,
        "read_seconds": read_seconds,
        "solve_seconds": execute_seconds - read_seconds,
        "peak_resident_bytes": peak_resident_bytes(),
        "peak_resident_scope": PEAK_RESIDENT_SCOPE,
    }
    result = memoryview(
        (json.dumps({"process": process, "record": report.record}) + "\n").encode()
    )
    while result:
        result = result[os.write(result_descriptor, result) :]
    os.close(result_descriptor)


@final
class ColdWorker:
    """A fresh scenario process prepared before its archive is published.

    The process imports its scenario and then blocks for one control packet, so
    preparation overlaps the durable write without receiving parent numerical state.
    It inherited the absolute expiry at launch; ``execute`` never resets it. Leaving
    the context kills the worker's whole session unless it already exited, so an
    archive failure, parent cancellation or timeout always ends the worker; a
    killed launcher kills it through the bootstrap's parent-death binding.
    """

    __slots__ = (
        "_deadline",
        "_expires",
        "_launched",
        "_ledger",
        "_module",
        "_process",
        "_state",
        "_stderr",
    )

    def __init__(
        self,
        module: str,
        process: subprocess.Popen[bytes],
        stderr: BinaryIO,
        *,
        deadline: Deadline,
        ledger: StageLedger,
        launched: float,
        expires: float,
    ) -> None:
        self._module = module
        self._process = process
        self._stderr = stderr
        self._deadline = deadline
        self._ledger = ledger
        self._launched = launched
        self._expires = expires
        self._state = "prepared"

    @property
    def process_id(self) -> int:
        """Process ID, and session ID, of the launched worker."""
        return self._process.pid

    def __enter__(self) -> ColdWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _stderr_text(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")

    def _end(self, status: str) -> None:
        """Record a worker that ended without a validated result."""
        self._state = status
        self._ledger.record["cold_process"] = {
            "status": status,
            "module": self._module,
            "process_id": self._process.pid,
            "launched_monotonic": self._launched,
            "ended_monotonic": _monotonic(),
            "expires_monotonic": self._expires,
            "returncode": self._process.returncode,
        }

    def _kill(self) -> None:
        """Kill the worker session while its leader is unreaped, so the group is ours."""
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGKILL)
            self._process.wait()

    def _retain_input(self, archive_path: Path, packet: bytes | None) -> None:
        """Preserve the real archive even when a control packet is inadmissible."""
        from tools.meshing_qualification._progress import evidence_path

        control_path = evidence_path("cold-control.json", process_id=self.process_id)
        if control_path is None:
            return
        archive_copy = control_path.with_name(
            control_path.name.removesuffix("control.json") + "archive"
        )
        with archive_path.open("rb") as source, archive_copy.open("xb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
            target.flush()
            os.fsync(target.fileno())
        if packet is not None:
            with control_path.open("xb") as control:
                control.write(packet)
                control.flush()
                os.fsync(control.fileno())
        self._ledger.record["cold_artifacts"] = {
            "archive": str(archive_copy),
            "control_packet": None if packet is None else str(control_path),
            "original_archive": str(archive_path.resolve()),
            "scientific_restart_authority": False,
        }

    def _retain_output(self, stdout: bytes) -> None:
        """Keep raw child evidence even when decoding or scientific work refuses."""
        from tools.meshing_qualification._progress import evidence_path

        try:
            output_path = evidence_path("cold.stdout", process_id=self.process_id)
            if output_path is None:
                return
            self._stderr.seek(0)
            for path, content in (
                (output_path, stdout),
                (output_path.with_suffix(".stderr"), self._stderr.read()),
            ):
                with path.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
        except OSError as error:
            # Reporting must not replace the child's first scientific failure.
            self._ledger.record.setdefault("reporting_failures", []).append({
                "phase": "cold-output-retention",
                "type": type(error).__name__,
                "message": str(error),
            })
            self._ledger.record["reporting_status"] = "failed"

    def execute(
        self, archive_path: Path, content_id: str, controls: dict[str, Any]
    ) -> dict[str, Any]:
        """Send the one control packet after publication and collect the bound result.

        Returns the scenario record with the verified worker ``process_id``; the
        process timing evidence lands in ``ledger.record["cold_process"]``.
        """
        if self._state != "prepared":
            raise RuntimeError(
                f"ColdWorker.execute must run once on a prepared worker; got state={self._state!r}"
            )
        self._state = "executing"
        packet = (
            json.dumps(
                {
                    "archive_path": str(Path(archive_path).resolve(strict=True)),
                    "content_id": content_id,
                    "controls": controls,
                },
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        if len(packet) > COLD_CONTROL_PACKET_BYTES:
            # The accepted checkpoint is independent of this transport refusal.
            # Keep it replayable; never persist an oversized packet as admitted.
            try:
                self._retain_input(archive_path, None)
            except OSError as error:
                self._ledger.record["reporting_status"] = "failed"
                self._ledger.record.setdefault("reporting_failures", []).append(
                    {"phase": "cold_input_retention", "type": type(error).__name__, "message": str(error)}
                )
            self._ledger.record["cold_packet_refusal"] = {
                "packet_bytes": len(packet), "maximum_packet_bytes": COLD_CONTROL_PACKET_BYTES
            }
            raise ValueError(
                f"ColdWorker.packet must not exceed {COLD_CONTROL_PACKET_BYTES} bytes; got {len(packet)}"
            )
        self._retain_input(archive_path, packet)
        ledger = self._ledger
        phase = ledger.cold_phase or ledger.record.get("active_phase", "cold_worker")

        def collect() -> tuple[float, float, bytes]:
            sent = _monotonic()
            try:
                stdout, _ = self._process.communicate(
                    packet, timeout=self._deadline.remaining()
                )
            except (subprocess.TimeoutExpired, TimeoutError) as failure:
                self._kill()
                stdout, _ = self._process.communicate()
                self._retain_output(stdout)
                self._end("timeout")
                raise ColdWorkerTimeout(
                    f"qualification.worker.deadline must allow completion; got module={self._module!r}, pid={self._process.pid}",
                    stdout.decode(errors="replace"),
                    self._stderr_text(),
                ) from failure
            self._retain_output(stdout)
            return sent, _monotonic(), stdout

        sent, returned, stdout = ledger.run(phase, collect)
        if self._process.returncode:
            self._end("refused")
            evidence = {
                "returncode": self._process.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": self._stderr_text(),
            }
            ledger.record["cold_process_refusal"] = evidence
            raise QualificationFailure(
                f"qualification.worker.returncode must equal zero; got module={self._module!r}, {observed_summary(evidence)}"
            )
        response = json.loads(stdout)
        if not isinstance(response, dict) or set(response) != {"process", "record"}:
            raise TypeError(
                f"qualification.worker.response must hold process and record; got {type(response).__qualname__}"
            )
        process, record = response["process"], response["record"]
        binding = {
            "module": self._module,
            "process_id": self._process.pid,
            "parent_process_id": os.getpid(),
            "session_id": self._process.pid,
            "expires_monotonic": self._expires,
            "packet_bytes": len(packet),
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
            "content_id": content_id,
        }
        observed = {key: process.get(key) for key in _COLD_PROCESS_BINDING}
        ordered = (
            self._launched,
            process.get("bootstrap_started_monotonic"),
            process.get("import_completed_monotonic"),
            process.get("packet_received_monotonic"),
            process.get("execute_completed_monotonic"),
            returned,
        )
        if (
            observed != binding
            or not all(isinstance(value, float) for value in ordered)
            or list(ordered) != sorted(ordered)
            or sent > process["packet_received_monotonic"]
            or not isinstance(record, dict)
            or "process_id" in record
        ):
            self._end("unbound")
            raise QualificationFailure(
                f"qualification.worker.result must bind the launched process, packet and expiry; got {observed_summary({'binding': binding, 'observed': observed, 'ordered': ordered})}"
            )
        self._state = "completed"
        ledger.record["cold_process"] = {
            "status": "completed",
            **process,
            "launched_monotonic": self._launched,
            "packet_sent_monotonic": sent,
            "result_returned_monotonic": returned,
            "launch_to_bootstrap_seconds": process["bootstrap_started_monotonic"]
            - self._launched,
            "prepared_before_packet_seconds": sent - self._launched,
            "execute_wall_seconds": returned - sent,
        }
        return {**record, "process_id": self._process.pid}

    def close(self) -> None:
        """End the worker session and release its pipes; idempotent."""
        if self._state == "closed":
            return
        self._kill()
        if self._state in ("prepared", "executing"):
            self._end("cancelled")
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()
        self._state = "closed"


def prepare_cold_worker(
    module: str, /, *, deadline: Deadline, ledger: StageLedger
) -> ColdWorker:
    """Launch a fresh ``module`` cold worker before its archive is written.

    The worker inherits ``deadline``'s absolute expiry and nothing else from this
    process. Use the result as a context manager and call ``execute`` once the
    archive is durably published; single-phase consumers call it immediately. On
    Linux the worker is killed when the launching thread exits, so the thread that
    prepares it must outlive ``execute``.
    """
    deadline.remaining()
    expires = deadline.monotonic_expiry()
    stderr = tempfile.TemporaryFile()
    try:
        launched = _monotonic()
        process = subprocess.Popen(
            [sys.executable, "-m", COLD_WORKER_BOOTSTRAP, module, str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env={**os.environ, LAUNCH_DEADLINE_VARIABLE: repr(expires)},
            start_new_session=True,
        )
    except BaseException:
        stderr.close()
        raise
    return ColdWorker(
        module,
        process,
        stderr,
        deadline=deadline,
        ledger=ledger,
        launched=launched,
        expires=expires,
    )


@final
class CampaignProcessTimeout(TimeoutError):
    """The watchdog killed a sample that outlived its deadline and reporting grace."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str,
        stderr: str,
        peak_resident_bytes: int | None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.peak_resident_bytes = peak_resident_bytes


@final
class CampaignProcessResult(StrictModule):
    """A child that exited before the watchdog, judged on the launcher's own clock.

    ``deadline_expired`` is true when the child exited after the scientific expiry,
    i.e. inside the reporting grace. The launcher sets it whether or not the
    child kept the inherited deadline variable (``env -i`` children do not).
    """

    returncode: int = eqx.field(static=True)
    stdout: str = eqx.field(static=True)
    stderr: str = eqx.field(static=True)
    deadline_expired: bool = eqx.field(static=True)


def run_campaign_process(
    command: list[str], *, cwd: Path, timeout: float
) -> CampaignProcessResult:
    """Run a session-isolated sample under one launch-origin deadline.

    The child inherits the launch expiry (``LAUNCH_DEADLINE_VARIABLE``) and enforces
    ``timeout`` itself, so a missed deadline still emits its phase evidence. The
    watchdog adds only ``REPORTING_GRACE_SECONDS`` for that report, then reads the
    child's post-exec peak and kills its whole session. An exit inside the grace is
    reported as ``deadline_expired``; the grace never extends the scientific limit.
    """
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            f"run_campaign_process.timeout must be finite and positive; got {timeout!r}"
        )
    expires = time.clock_gettime(time.CLOCK_MONOTONIC) + timeout
    inherited = os.environ.get(LAUNCH_DEADLINE_VARIABLE)
    if inherited is not None:
        expires = min(expires, float(inherited))
    environment = {**os.environ, LAUNCH_DEADLINE_VARIABLE: repr(expires)}
    watchdog = (
        expires - time.clock_gettime(time.CLOCK_MONOTONIC) + REPORTING_GRACE_SECONDS
    )
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=max(watchdog, 0.0))
        except subprocess.TimeoutExpired as expired:
            peak = _status_high_water(Path(f"/proc/{child.pid}/status"))
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate()
            raise CampaignProcessTimeout(
                f"qualification.campaign_process must report within its {timeout} s deadline plus {REPORTING_GRACE_SECONDS} s reporting grace; got pid={child.pid} killed",
                stdout=stdout,
                stderr=stderr,
                peak_resident_bytes=peak,
            ) from expired
        return CampaignProcessResult(
            returncode=child.returncode,
            stdout=stdout,
            stderr=stderr,
            deadline_expired=time.clock_gettime(time.CLOCK_MONOTONIC) >= expires,
        )

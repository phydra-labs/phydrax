#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Safe rank environments and local multi-process launch without shell execution."""

from __future__ import annotations

import json
import math
import os
import secrets
import selectors
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO


@dataclass(frozen=True, slots=True)
class RankEnvironment:
    coordinator_address: str
    num_processes: int
    process_id: int
    local_device_ids: tuple[int, ...] | None = None
    coordinator_bind_address: str | None = None

    def __post_init__(self) -> None:
        if not self.coordinator_address.strip():
            raise ValueError("coordinator_address must be non-empty")
        if self.num_processes <= 1:
            raise ValueError("rank environments require num_processes > 1")
        if not 0 <= self.process_id < self.num_processes:
            raise ValueError("process_id must be in [0, num_processes)")
        if self.local_device_ids is not None:
            if not self.local_device_ids or min(self.local_device_ids) < 0:
                raise ValueError("local_device_ids must be non-empty and non-negative")
            if len(set(self.local_device_ids)) != len(self.local_device_ids):
                raise ValueError("local_device_ids must be unique")

    def apply(self, environment: Mapping[str, str] | None = None) -> dict[str, str]:
        from ._config_environment import bootstrap_timeout_overrides, runtime_environment
        from ._config_types import RuntimeBootstrap, RuntimeConfig

        values = dict(os.environ if environment is None else environment)
        timeouts = bootstrap_timeout_overrides(values)
        defaults = RuntimeBootstrap()
        bootstrap = RuntimeBootstrap(
            coordinator_address=self.coordinator_address,
            num_processes=self.num_processes,
            process_id=self.process_id,
            local_device_ids=self.local_device_ids,
            coordinator_bind_address=self.coordinator_bind_address,
            heartbeat_timeout_seconds=(
                defaults.heartbeat_timeout_seconds
                if timeouts.heartbeat_timeout_seconds is None
                else timeouts.heartbeat_timeout_seconds
            ),
            initialization_timeout_seconds=(
                defaults.initialization_timeout_seconds
                if timeouts.initialization_timeout_seconds is None
                else timeouts.initialization_timeout_seconds
            ),
        )
        return runtime_environment(RuntimeConfig(bootstrap=bootstrap), values, "override")


@dataclass(frozen=True, slots=True)
class LocalProcessLaunchPlan:
    """One homogeneous local process set with disjoint device assignments."""

    num_processes: int
    coordinator_address: str
    devices_per_process: int | None = None

    def __post_init__(self) -> None:
        if self.num_processes <= 1:
            raise ValueError("local process launch requires num_processes > 1")
        if not self.coordinator_address.startswith(("127.0.0.1:", "[::1]:")):
            raise ValueError("local coordinator_address must bind to loopback")
        if self.devices_per_process is not None and self.devices_per_process <= 0:
            raise ValueError("devices_per_process must be positive")

    def rank(self, process_id: int) -> RankEnvironment:
        local_devices = None
        if self.devices_per_process is not None:
            start = int(process_id) * self.devices_per_process
            local_devices = tuple(range(start, start + self.devices_per_process))
        return RankEnvironment(
            self.coordinator_address,
            self.num_processes,
            int(process_id),
            local_device_ids=local_devices,
            coordinator_bind_address=(
                self.coordinator_address if int(process_id) == 0 else None
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalProcessResult:
    process_id: int
    # An MPI monitor lost before reporting cannot provide a truthful exit code.
    returncode: int | None
    stdout: bytes
    stderr: bytes
    output_truncated: bool = False


class LocalProcessLaunchError(RuntimeError):
    def __init__(
        self,
        results: Sequence[LocalProcessResult],
        /,
        *,
        reason: str = "",
        timed_out: bool = False,
        launcher_returncode: int | None = None,
        launcher_stdout: bytes = b"",
        launcher_stderr: bytes = b"",
        launcher_output_truncated: bool = False,
    ) -> None:
        self.results = tuple(results)
        self.timed_out = timed_out
        self.reason = reason
        self.launcher_returncode = launcher_returncode
        self.launcher_stdout = launcher_stdout
        self.launcher_stderr = launcher_stderr
        self.launcher_output_truncated = launcher_output_truncated
        failed = tuple(result for result in self.results if result.returncode != 0)
        summary = "; ".join(
            f"rank {result.process_id} exited {result.returncode} "
            f"(stderr_bytes={len(result.stderr)}, output_truncated={result.output_truncated})"
            for result in failed
        )
        diagnostics = []
        for result in self.results:
            if result.stderr:
                diagnostics.append(
                    f"rank {result.process_id} stderr tail "
                    f"(last {min(2000, len(result.stderr))} of {len(result.stderr)} bytes):\n"
                    + result.stderr[-2000:].decode("utf-8", errors="replace")
                )
        if launcher_returncode is not None or launcher_stderr:
            diagnostics.append(
                f"MPI launcher exited {launcher_returncode} "
                f"(output_truncated={launcher_output_truncated})"
            )
            if launcher_stderr:
                diagnostics.append(
                    "MPI launcher stderr tail "
                    f"(last {min(2000, len(launcher_stderr))} of {len(launcher_stderr)} bytes):\n"
                    + launcher_stderr[-2000:].decode("utf-8", errors="replace")
                )
        super().__init__(
            f"local distributed launch failed: {reason + '; ' if reason else ''}{summary}"
            + ("\n" + "\n".join(diagnostics) if diagnostics else "")
        )


_SUPERVISOR_FD = "PHYDRAX_LOCAL_SUPERVISOR_FD"
_SUPERVISOR_PID = "PHYDRAX_LOCAL_SUPERVISOR_PID"
_SUPERVISOR_MESSAGE_BYTES = 4096
_SUPERVISOR_LOCK = threading.Lock()


def _launch_timeout(value: float) -> float:
    from ._external_runtime import _positive_timeout

    if type(value) not in (int, float):
        raise ValueError("timeout must be positive and finite.")
    try:
        return _positive_timeout(value)
    except OverflowError as error:
        raise ValueError("timeout must be positive and finite.") from error


@dataclass(frozen=True, slots=True)
class LocalProcessSupervision:
    """Live trusted-local coupling, not a sandbox or escaped-session containment.

    The launcher owns a separate session/process group for every worker rank.
    Its private socket confirms supervisor and actual parent ownership plus the
    whole-job deadline on every admission. Cached identity alone is not admission.
    """

    supervisor_id: str
    parent_pid: int
    supervisor_pid: int
    pid: int
    process_group_id: int
    process_id: int
    num_processes: int
    deadline_monotonic: float

    def require_active(
        self, process_indices: tuple[int, ...], timeout_seconds: float
    ) -> None:
        timeout = _launch_timeout(timeout_seconds)
        if (
            not process_indices
            or any(type(index) is not int for index in process_indices)
            or len(set(process_indices)) != len(process_indices)
            or any(index < 0 or index >= self.num_processes for index in process_indices)
            or self.process_id not in process_indices
        ):
            raise ValueError("Collective membership must belong to the supervised ranks.")
        current = require_local_process_supervision()
        if current != self:
            raise RuntimeError("Local process supervision identity changed.")
        if time.monotonic() + timeout > self.deadline_monotonic:
            raise RuntimeError("Collective timeout exceeds the live launcher deadline.")

    def fail(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty coupled failure reason is required.")
        _supervisor_exchange("fail", reason=reason[:1024])


def _supervisor_exchange(
    operation: str,
    *,
    reason: str = "",
    registration: Mapping[str, str | int] | None = None,
) -> dict[str, str | int | float]:
    """Exchange a bounded challenge on the launcher's inherited private channel."""
    if os.name != "posix":
        raise RuntimeError(
            "Live local process supervision requires POSIX process groups."
        )
    descriptor = os.environ.get(_SUPERVISOR_FD, "")
    if not descriptor.isascii() or not descriptor.isdecimal():
        raise RuntimeError("No live local process supervisor channel is available.")
    challenge = secrets.token_hex(16)
    request: dict[str, str | int] = {
        "operation": operation,
        "challenge": challenge,
        "reason": reason,
    }
    if registration is not None:
        request.update(registration)
    packet = json.dumps(request).encode() + b"\n"
    try:
        with _SUPERVISOR_LOCK, socket.socket(fileno=os.dup(int(descriptor))) as channel:
            if hasattr(socket, "SO_PEERCRED"):
                peer_pid, _, _ = struct.unpack(
                    "3i",
                    channel.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                    ),
                )
                expected_parent = os.environ.get(_SUPERVISOR_PID, str(os.getppid()))
                if str(peer_pid) != expected_parent:
                    raise RuntimeError(
                        "Supervisor channel is not owned by the expected parent supervisor."
                    )
            channel.settimeout(1.0)
            channel.sendall(packet)
            response = bytearray()
            while b"\n" not in response:
                block = channel.recv(_SUPERVISOR_MESSAGE_BYTES - len(response))
                if not block:
                    raise RuntimeError("Local supervisor closed its live channel.")
                response.extend(block)
                if len(response) >= _SUPERVISOR_MESSAGE_BYTES:
                    raise RuntimeError("Local supervisor response exceeds its bound.")
    except (OSError, ValueError) as error:
        raise RuntimeError("Local process supervisor is not live.") from error
    payload = json.loads(response)
    if not isinstance(payload, dict) or payload.get("challenge") != challenge:
        raise RuntimeError("Local process supervisor challenge mismatch.")
    return payload


def require_local_process_supervision() -> LocalProcessSupervision:
    """Acquire a freshly confirmed supervisor for an explicit collective lease."""
    if os.name != "posix":
        raise RuntimeError(
            "Live local process supervision requires POSIX process groups."
        )
    payload = _supervisor_exchange("active")
    integer_fields = (
        "parent_pid",
        "supervisor_pid",
        "pid",
        "process_group_id",
        "process_id",
        "num_processes",
    )
    if (
        any(type(payload.get(name)) is not int for name in integer_fields)
        or not isinstance(payload.get("supervisor_id"), str)
        or not payload["supervisor_id"]
        or type(payload.get("deadline_monotonic")) is not float
    ):
        raise RuntimeError("Malformed local process supervision identity.")
    # The wire boundary is dynamic; construct only after validating every field.
    identity = LocalProcessSupervision(
        supervisor_id=str(payload["supervisor_id"]),
        parent_pid=int(payload["parent_pid"]),
        supervisor_pid=int(payload["supervisor_pid"]),
        pid=int(payload["pid"]),
        process_group_id=int(payload["process_group_id"]),
        process_id=int(payload["process_id"]),
        num_processes=int(payload["num_processes"]),
        deadline_monotonic=float(payload["deadline_monotonic"]),
    )
    if (
        identity.parent_pid != os.getppid()
        or str(identity.supervisor_pid)
        != os.environ.get(_SUPERVISOR_PID, str(os.getppid()))
        or identity.pid != os.getpid()
        or identity.process_group_id != os.getpgrp()
        or identity.process_group_id != identity.pid
        or not 0 <= identity.process_id < identity.num_processes
        or os.environ.get("PHYDRAX_PROCESS_ID") != str(identity.process_id)
        or os.environ.get("PHYDRAX_NUM_PROCESSES") != str(identity.num_processes)
        or not math.isfinite(identity.deadline_monotonic)
        or time.monotonic() >= identity.deadline_monotonic
    ):
        raise RuntimeError(
            "Local process supervision is stale or does not own this rank."
        )
    return identity


@dataclass(slots=True)
class _LaunchJob:
    plan: LocalProcessLaunchPlan
    timeout_seconds: float
    max_output_bytes: int
    deadline: float = field(init=False)
    supervisor_id: str = field(init=False)
    reason: str = ""
    timed_out: bool = False

    def __post_init__(self) -> None:
        self.deadline = time.monotonic() + self.timeout_seconds
        self.supervisor_id = secrets.token_hex(16)

    def identity(
        self, pid: int, parent_pid: int, group: int, process_id: int
    ) -> LocalProcessSupervision:
        return LocalProcessSupervision(
            self.supervisor_id,
            parent_pid,
            os.getpid(),
            pid,
            group,
            process_id,
            self.plan.num_processes,
            self.deadline,
        )


def _send_supervisor_response(
    channel: socket.socket, identity: LocalProcessSupervision, challenge: str
) -> None:
    response = {
        "challenge": challenge,
        "supervisor_id": identity.supervisor_id,
        "parent_pid": identity.parent_pid,
        "supervisor_pid": identity.supervisor_pid,
        "pid": identity.pid,
        "process_group_id": identity.process_group_id,
        "process_id": identity.process_id,
        "num_processes": identity.num_processes,
        "deadline_monotonic": identity.deadline_monotonic,
    }
    channel.sendall(json.dumps(response).encode() + b"\n")


def _receive_supervisor_data(
    channel: socket.socket, buffer: bytearray, overflow_message: str
) -> bool:
    block = channel.recv(_SUPERVISOR_MESSAGE_BYTES)
    if not block:
        return False
    buffer.extend(block)
    if len(buffer) > _SUPERVISOR_MESSAGE_BYTES:
        raise RuntimeError(overflow_message)
    return True


def _supervisor_requests(
    buffer: bytearray, *, malformed_message: str = "Malformed supervisor request."
) -> Iterator[dict[str, Any]]:
    while b"\n" in buffer:
        packet, _, rest = buffer.partition(b"\n")
        buffer[:] = rest
        request: Any = json.loads(packet)
        if not isinstance(request, dict) or not isinstance(request.get("challenge"), str):
            raise RuntimeError(malformed_message)
        yield request


def _kill_launch_group(group: int) -> None:
    if os.name != "posix":
        raise RuntimeError("Local launch teardown requires POSIX process groups.")
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclass(slots=True)
class _LocalLaunchState:
    job: _LaunchJob
    selector: selectors.BaseSelector
    processes: list[subprocess.Popen[bytes]] = field(default_factory=list)
    channels: list[socket.socket] = field(default_factory=list)
    streams: list[tuple[int, int, IO[bytes]]] = field(default_factory=list)
    output: list[list[bytearray]] = field(init=False)
    truncated: list[bool] = field(init=False)
    requests: list[bytearray] = field(init=False)

    def __post_init__(self) -> None:
        count = self.job.plan.num_processes
        self.output = [[bytearray(), bytearray()] for _ in range(count)]
        self.truncated = [False for _ in range(count)]
        self.requests = [bytearray() for _ in range(count)]


def _spawn_local_rank(
    state: _LocalLaunchState,
    process_id: int,
    arguments: tuple[str, ...],
    environment: Mapping[str, str] | None,
    cwd: str | None,
) -> None:
    parent, child = socket.socketpair()
    state.channels.append(parent)
    try:
        values = state.job.plan.rank(process_id).apply(environment)
        values[_SUPERVISOR_FD] = str(child.fileno())
        values[_SUPERVISOR_PID] = str(os.getpid())
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=values,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
            pass_fds=(child.fileno(),),
        )
        state.processes.append(process)
    finally:
        child.close()
    parent.setblocking(False)
    state.selector.register(parent, selectors.EVENT_READ, (process_id, 2))
    for stream_id, stream in enumerate((process.stdout, process.stderr)):
        if stream is None:
            raise RuntimeError("Rank output pipe was not created.")
        os.set_blocking(stream.fileno(), False)
        state.streams.append((process_id, stream_id, stream))
        state.selector.register(stream, selectors.EVENT_READ, (process_id, stream_id))


def _local_launch_finished(state: _LocalLaunchState) -> bool:
    failed = [
        index
        for index, process in enumerate(state.processes)
        if process.poll() not in (None, 0)
    ]
    if failed:
        state.job.reason = f"rank {failed[0]} failed"
        return True
    if all(process.poll() is not None for process in state.processes):
        return True
    if time.monotonic() >= state.job.deadline:
        state.job.timed_out = True
        state.job.reason = (
            f"whole-job deadline exceeded {state.job.timeout_seconds:g} seconds"
        )
        return True
    return False


def _append_local_output(
    state: _LocalLaunchState, process_id: int, stream_id: int, block: bytes
) -> bool:
    remaining = state.job.max_output_bytes - sum(map(len, state.output[process_id]))
    state.output[process_id][stream_id].extend(block[:remaining])
    overflow = len(block) > remaining
    if overflow:
        state.truncated[process_id] = True
        if not state.job.reason:
            state.job.reason = f"rank {process_id} output exceeds max_output_bytes"
    return overflow


def _read_local_output(
    state: _LocalLaunchState, key: selectors.SelectorKey, process_id: int, stream_id: int
) -> None:
    block = os.read(key.fd, 65536)
    if not block:
        state.selector.unregister(key.fileobj)
        return
    _append_local_output(state, process_id, stream_id, block)


def _reply_local_requests(state: _LocalLaunchState, process_id: int) -> None:
    channel = state.channels[process_id]
    for request in _supervisor_requests(state.requests[process_id]):
        if request.get("operation") not in ("active", "fail"):
            raise RuntimeError("Malformed supervisor request.")
        process = state.processes[process_id]
        identity = state.job.identity(process.pid, os.getpid(), process.pid, process_id)
        try:
            _send_supervisor_response(channel, identity, request["challenge"])
        except (BrokenPipeError, ConnectionResetError):
            state.selector.unregister(channel)
            break
        if request["operation"] == "fail":
            state.job.reason = f"rank {process_id} reported failure: {str(request.get('reason', ''))[:1024]}"
            break


def _read_local_supervision(state: _LocalLaunchState, process_id: int) -> None:
    channel = state.channels[process_id]
    try:
        alive = _receive_supervisor_data(
            channel,
            state.requests[process_id],
            f"rank {process_id} supervisor request exceeds its bound",
        )
    except ConnectionResetError:
        state.selector.unregister(channel)
        return
    except RuntimeError as error:
        state.job.reason = str(error)
        return
    if not alive:
        state.selector.unregister(channel)
        return
    try:
        _reply_local_requests(state, process_id)
    except (ValueError, UnicodeDecodeError, RuntimeError):
        state.job.reason = f"rank {process_id} sent malformed supervisor data"


def _monitor_local_ranks(state: _LocalLaunchState) -> None:
    while not _local_launch_finished(state):
        events = sorted(
            state.selector.select(
                min(0.02, max(0.0, state.job.deadline - time.monotonic()))
            ),
            key=lambda event: event[0].data,
        )
        for key, _ in events:
            process_id, stream_id = key.data
            if stream_id < 2:
                _read_local_output(state, key, process_id, stream_id)
            else:
                _read_local_supervision(state, process_id)
            if state.job.reason:
                return


def _drain_local_output(
    state: _LocalLaunchState, process_id: int, stream_id: int, stream: IO[bytes]
) -> None:
    try:
        while block := os.read(stream.fileno(), 65536):
            if _append_local_output(state, process_id, stream_id, block):
                break
    except BlockingIOError:
        pass
    finally:
        stream.close()


def _teardown_local_ranks(state: _LocalLaunchState) -> None:
    from ._external_runtime import _kill_process_group

    # Signal every group before reaping a leader: no rank can hold another
    # rank hostage in a collective while cleanup waits.
    for process in state.processes:
        _kill_launch_group(process.pid)
    for process in state.processes:
        _kill_process_group(process)
    for process_id, stream_id, stream in state.streams:
        _drain_local_output(state, process_id, stream_id, stream)
    for channel in state.channels:
        channel.close()


def _local_launch_results(state: _LocalLaunchState) -> tuple[LocalProcessResult, ...]:
    collected: list[LocalProcessResult] = []
    for index, (process, (stdout, stderr)) in enumerate(
        zip(state.processes, state.output, strict=True)
    ):
        returncode = process.returncode
        if returncode is None:
            raise RuntimeError("Launcher cleanup did not reap a rank process.")
        collected.append(
            LocalProcessResult(
                index, returncode, bytes(stdout), bytes(stderr), state.truncated[index]
            )
        )
    results = tuple(collected)
    if state.job.reason or any(result.returncode for result in results):
        raise LocalProcessLaunchError(
            results, reason=state.job.reason, timed_out=state.job.timed_out
        )
    return results


def _launch_direct_processes(
    arguments: tuple[str, ...],
    plan: LocalProcessLaunchPlan,
    *,
    environment: Mapping[str, str] | None,
    cwd: str | None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[LocalProcessResult, ...]:
    job = _LaunchJob(plan, timeout_seconds, max_output_bytes)
    with selectors.DefaultSelector() as selector:
        state = _LocalLaunchState(job, selector)
        try:
            for process_id in range(plan.num_processes):
                _spawn_local_rank(state, process_id, arguments, environment, cwd)
            _monitor_local_ranks(state)
        finally:
            _teardown_local_ranks(state)
    return _local_launch_results(state)


def launch_local_processes(
    argv: Sequence[str],
    plan: LocalProcessLaunchPlan,
    /,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: str | None = None,
    timeout_seconds: float = 300.0,
    max_output_bytes: int = 1024 * 1024,
    mpi_launcher: Sequence[str] | None = None,
) -> tuple[LocalProcessResult, ...]:
    """Run rank-ordered, deadline-bound, fail-coupled trusted-local processes.

    Output is bounded per rank across stdout and stderr. Exceeding the bound
    fails the job rather than silently reporting truncated success. All rank
    process groups are killed on failure, timeout, interruption, and after
    successful leader exits (including descendants that retain output pipes).
    Processes that deliberately escape their session are not contained.
    ``mpi_launcher`` explicitly selects one local site Open MPI executable and
    flags. The launcher supplies the rank count and authenticated bootstrap;
    no remote-host or other MPI implementation support is implied.
    """
    timeout = _launch_timeout(timeout_seconds)
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer.")
    arguments = tuple(str(value) for value in argv)
    if not arguments or any(not value or "\x00" in value for value in arguments):
        raise ValueError("argv must contain non-empty NUL-free arguments")
    if os.name != "posix":
        raise RuntimeError(
            "Coupled local process launching requires POSIX process groups."
        )
    if mpi_launcher is not None:
        return _launch_mpi_processes(
            arguments,
            plan,
            mpi_launcher,
            environment=environment,
            cwd=cwd,
            timeout_seconds=timeout,
            max_output_bytes=max_output_bytes,
        )
    return _launch_direct_processes(
        arguments,
        plan,
        environment=environment,
        cwd=cwd,
        timeout_seconds=timeout,
        max_output_bytes=max_output_bytes,
    )


__all__ = (
    "LocalProcessLaunchError",
    "LocalProcessLaunchPlan",
    "LocalProcessResult",
    "LocalProcessSupervision",
    "RankEnvironment",
    "launch_local_processes",
    "require_local_process_supervision",
)


@dataclass(slots=True)
class _MPISupervisorChannel:
    socket: socket.socket
    peer_pid: int
    buffer: bytearray
    process_id: int | None = None
    role: str = ""


def _linux_process_identity(pid: int) -> tuple[int, int]:
    """Return actual parent PID and start ticks, not caller-declared ancestry."""
    fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b")", 1)[1].split()
    return int(fields[1]), int(fields[19])


def _mpi_owned_descendant(pid: int, launcher_pid: int) -> bool:
    visited: set[int] = set()
    while pid > 1 and pid not in visited:
        if pid == launcher_pid:
            return True
        visited.add(pid)
        try:
            pid, _ = _linux_process_identity(pid)
        except (OSError, ValueError, IndexError):
            return False
    return False


def _bounded_launch_logs(
    stdout: IO[bytes], stderr: IO[bytes], maximum: int
) -> tuple[bytes, bytes, bool]:
    total = os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size
    stdout.seek(0)
    out = stdout.read(maximum)
    stderr.seek(0)
    err = stderr.read(maximum - len(out))
    return out, err, total > maximum


def _mpi_bootstrap_command(config_path: str, *, monitor: bool) -> tuple[str, ...]:
    # Isolated/no-site startup excludes both the script's package directory
    # (which contains stdlib-shadowing signal/typing names) and site imports.
    # sys.executable still identifies the selected worker's venv interpreter.
    return (
        sys.executable,
        "-I",
        "-S",
        str(Path(__file__).resolve()),
        "--mpi-monitor" if monitor else "--mpi-exec-rank",
        config_path,
    )


@dataclass(slots=True)
class _MPILaunchState:
    """One owner for MPI admission, registered identities and teardown evidence."""

    job: _LaunchJob
    selector: selectors.BaseSelector
    listener: socket.socket
    root: Path
    launcher_out: IO[bytes]
    launcher_err: IO[bytes]
    logs: list[tuple[IO[bytes], IO[bytes]]]
    statuses: list[int | None] = field(init=False)
    identities: list[LocalProcessSupervision | None] = field(init=False)
    monitors: list[int | None] = field(init=False)
    spawned: list[int | None] = field(init=False)
    failures: list[str] = field(init=False)
    environment: dict[str, str] = field(default_factory=dict)
    pidfds: dict[int, int] = field(default_factory=dict)
    groups: set[int] = field(default_factory=set)
    connections: list[_MPISupervisorChannel] = field(default_factory=list)
    process: subprocess.Popen[bytes] | None = None
    tearing_down: bool = False

    def __post_init__(self) -> None:
        count = self.job.plan.num_processes
        self.statuses = [None] * count
        self.identities = [None] * count
        self.monitors = [None] * count
        self.spawned = [None] * count
        self.failures = [""] * count


def _admit_mpi_launcher(
    mpi_launcher: Sequence[str], plan: LocalProcessLaunchPlan
) -> tuple[str, ...]:
    arguments = tuple(str(value) for value in mpi_launcher)
    if (
        not arguments
        or any(not value or "\x00" in value for value in arguments)
        or not os.path.isabs(arguments[0])
        or not os.path.isfile(arguments[0])
        or not os.access(arguments[0], os.X_OK)
    ):
        raise ValueError(
            "mpi_launcher requires an explicit absolute executable and NUL-free arguments."
        )
    if not sys.platform.startswith("linux") or not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError(
            "Authenticated site MPI launching requires Linux peer credentials and /proc."
        )
    if plan.devices_per_process != 1:
        raise ValueError(
            "The site MPI launch profile requires exactly one device per process."
        )
    if any(value in ("-n", "-np", "--np", "--n") for value in arguments[1:]):
        raise ValueError(
            "The MPI rank count belongs to LocalProcessLaunchPlan, not mpi_launcher."
        )
    return arguments


def _open_mpi_rank_logs(
    root: Path, count: int, resources: ExitStack
) -> list[tuple[IO[bytes], IO[bytes]]]:
    return [
        (
            resources.enter_context((root / f"rank-{rank}.stdout").open("w+b")),
            resources.enter_context((root / f"rank-{rank}.stderr").open("w+b")),
        )
        for rank in range(count)
    ]


def _prepare_mpi_bootstrap(
    state: _MPILaunchState,
    arguments: tuple[str, ...],
    environment: Mapping[str, str] | None,
) -> Path:
    from ._config_environment import _BOOTSTRAP_ENVIRONMENT

    plan = state.job.plan
    address = str(state.root / "supervisor.socket")
    state.listener.bind(address)
    state.listener.listen(max(16, 2 * plan.num_processes))
    state.listener.setblocking(False)
    state.selector.register(state.listener, selectors.EVENT_READ, None)
    state.environment = dict(os.environ if environment is None else environment)
    # Only the parent imports configuration owners. MPI monitor/pre-exec
    # scripts stay stdlib-only until canonical settings reach worker exec.
    rank_bootstrap = []
    for rank in range(plan.num_processes):
        values = plan.rank(rank).apply(state.environment)
        rank_bootstrap.append(
            {
                name: values[name]
                for name in sorted(_BOOTSTRAP_ENVIRONMENT)
                if name in values
            }
        )
    config_path = state.root / "bootstrap.json"
    config_path.write_text(
        json.dumps(
            {
                "argv": list(arguments),
                "num_processes": plan.num_processes,
                "coordinator_address": plan.coordinator_address,
                "devices_per_process": plan.devices_per_process,
                "bootstrap_environment_keys": sorted(_BOOTSTRAP_ENVIRONMENT),
                "rank_bootstrap": rank_bootstrap,
                "socket": address,
                "supervisor_pid": os.getpid(),
                "supervisor_id": state.job.supervisor_id,
                "directory": str(state.root),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _spawn_mpi_launcher(
    state: _MPILaunchState,
    launcher_arguments: tuple[str, ...],
    config_path: Path,
    cwd: str | None,
) -> None:
    state.process = subprocess.Popen(
        (
            *launcher_arguments,
            "-n",
            str(state.job.plan.num_processes),
            *_mpi_bootstrap_command(str(config_path), monitor=True),
        ),
        cwd=cwd,
        env=state.environment,
        stdin=subprocess.DEVNULL,
        stdout=state.launcher_out,
        stderr=state.launcher_err,
        shell=False,
        start_new_session=True,
    )


def _accept_mpi_channels(state: _MPILaunchState) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Authenticated MPI channels require Linux peer credentials.")
    while True:
        try:
            channel, _ = state.listener.accept()
        except BlockingIOError:
            return
        peer_pid, _, _ = struct.unpack(
            "3i",
            channel.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            ),
        )
        channel.setblocking(False)
        connection = _MPISupervisorChannel(channel, peer_pid, bytearray())
        state.connections.append(connection)
        state.selector.register(channel, selectors.EVENT_READ, connection)


def _validated_mpi_registration(
    state: _MPILaunchState, connection: _MPISupervisorChannel, request: dict[str, Any]
) -> tuple[int, str, int, int]:
    if sys.platform != "linux":
        raise RuntimeError(
            "Authenticated MPI registration requires Linux process groups."
        )
    rank = request.get("process_id")
    role = request.get("role")
    process = state.process
    if (
        connection.process_id is not None
        or type(rank) is not int
        or not 0 <= rank < state.job.plan.num_processes
        or role not in ("monitor", "rank")
        or request.get("supervisor_id") != state.job.supervisor_id
        or process is None
        or not _mpi_owned_descendant(connection.peer_pid, process.pid)
    ):
        raise RuntimeError("MPI peer does not belong to the owned local launch.")
    parent_pid, _ = _linux_process_identity(connection.peer_pid)
    group = os.getpgid(connection.peer_pid)
    if group == os.getpgrp():
        raise RuntimeError("MPI peer shares the supervising parent's process group.")
    return rank, role, parent_pid, group


def _register_mpi_monitor(
    state: _MPILaunchState, connection: _MPISupervisorChannel, rank: int
) -> None:
    if state.tearing_down:
        raise RuntimeError("MPI launch is already tearing down.")
    if state.monitors[rank] is not None:
        raise RuntimeError("Duplicate MPI rank monitor registration.")
    state.monitors[rank] = connection.peer_pid


def _register_mpi_worker(
    state: _MPILaunchState,
    connection: _MPISupervisorChannel,
    rank: int,
    parent_pid: int,
    group: int,
) -> None:
    if (
        state.identities[rank] is not None
        or state.monitors[rank] != parent_pid
        or state.spawned[rank] not in (None, connection.peer_pid)
        or group != connection.peer_pid
    ):
        raise RuntimeError("MPI worker identity, parent or process group mismatch.")
    state.identities[rank] = state.job.identity(
        connection.peer_pid, parent_pid, group, rank
    )


def _retain_mpi_pid(state: _MPILaunchState, pid: int) -> None:
    if hasattr(os, "pidfd_open") and pid not in state.pidfds:
        state.pidfds[pid] = os.pidfd_open(pid)


def _register_mpi_channel(
    state: _MPILaunchState, connection: _MPISupervisorChannel, request: dict[str, Any]
) -> None:
    rank, role, parent_pid, group = _validated_mpi_registration(
        state, connection, request
    )
    if role == "monitor":
        _register_mpi_monitor(state, connection, rank)
    else:
        _register_mpi_worker(state, connection, rank, parent_pid, group)
    _retain_mpi_pid(state, connection.peer_pid)
    state.groups.add(group)
    connection.process_id = rank
    connection.role = role


def _record_mpi_spawn(
    state: _MPILaunchState,
    connection: _MPISupervisorChannel,
    rank: int,
    request: dict[str, Any],
) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Authenticated MPI workers require Linux process groups.")
    child_pid = request.get("worker_pid")
    if (
        connection.role != "monitor"
        or type(child_pid) is not int
        or state.spawned[rank] is not None
        or _linux_process_identity(child_pid)[0] != connection.peer_pid
        or os.getpgid(child_pid) != child_pid
    ):
        raise RuntimeError("MPI monitor does not own its reported worker.")
    identity = state.identities[rank]
    if identity is not None and identity.pid != child_pid:
        raise RuntimeError("MPI worker registration differs from its monitor report.")
    state.spawned[rank] = child_pid
    state.groups.add(child_pid)
    _retain_mpi_pid(state, child_pid)


def _kill_mpi_rank(state: _MPILaunchState, rank: int) -> None:
    identity = state.identities[rank]
    group = identity.process_group_id if identity is not None else state.spawned[rank]
    if group is not None:
        _kill_launch_group(group)


def _record_mpi_exit(
    state: _MPILaunchState,
    connection: _MPISupervisorChannel,
    rank: int,
    request: dict[str, Any],
) -> None:
    code = request.get("returncode")
    if (
        connection.role != "monitor"
        or type(code) is not int
        or state.statuses[rank] is not None
    ):
        raise RuntimeError("Invalid MPI rank exit report.")
    state.statuses[rank] = code
    _kill_mpi_rank(state, rank)


def _require_mpi_rank_live(
    state: _MPILaunchState, connection: _MPISupervisorChannel, rank: int
) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Authenticated MPI supervision requires Linux process groups.")
    identity = state.identities[rank]
    if connection.role != "rank" or identity is None:
        raise RuntimeError("MPI supervision requires the registered worker channel.")
    parent_pid, _ = _linux_process_identity(connection.peer_pid)
    if (
        parent_pid != identity.parent_pid
        or os.getpgid(connection.peer_pid) != identity.process_group_id
        or time.monotonic() >= state.job.deadline
    ):
        raise RuntimeError("MPI rank supervision is no longer live.")


def _dispatch_mpi_request(
    state: _MPILaunchState, connection: _MPISupervisorChannel, request: dict[str, Any]
) -> int:
    operation = request.get("operation")
    if operation == "register":
        _register_mpi_channel(state, connection, request)
    rank = connection.process_id
    if rank is None:
        raise RuntimeError("Unregistered MPI supervisor channel.")
    match operation:
        case "spawn":
            _record_mpi_spawn(state, connection, rank, request)
        case "exit":
            _record_mpi_exit(state, connection, rank, request)
        case "active" | "fail":
            _require_mpi_rank_live(state, connection, rank)
            if operation == "fail":
                state.failures[rank] = str(request.get("reason", ""))[:1024]
        case "register":
            pass
        case _:
            raise RuntimeError("Unknown MPI supervisor operation.")
    return rank


def _reply_mpi_request(
    state: _MPILaunchState,
    connection: _MPISupervisorChannel,
    rank: int,
    challenge: str,
) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Authenticated MPI supervision requires Linux process groups.")
    parent_pid, _ = _linux_process_identity(connection.peer_pid)
    identity = state.job.identity(
        connection.peer_pid, parent_pid, os.getpgid(connection.peer_pid), rank
    )
    _send_supervisor_response(connection.socket, identity, challenge)
    if state.tearing_down:
        _kill_mpi_rank(state, rank)


def _service_mpi_channel(
    state: _MPILaunchState, connection: _MPISupervisorChannel
) -> None:
    try:
        if not _receive_supervisor_data(
            connection.socket,
            connection.buffer,
            "MPI supervisor request exceeds its bound.",
        ):
            state.selector.unregister(connection.socket)
            return
        for request in _supervisor_requests(
            connection.buffer, malformed_message="Malformed MPI supervisor request."
        ):
            rank = _dispatch_mpi_request(state, connection, request)
            _reply_mpi_request(state, connection, rank, request["challenge"])
    except (OSError, ValueError, RuntimeError, IndexError) as error:
        if not state.job.reason:
            state.job.reason = f"MPI supervision failed: {error}"
        try:
            state.selector.unregister(connection.socket)
        except KeyError:
            pass
        connection.socket.close()


def _service_mpi_launch(state: _MPILaunchState, wait_seconds: float) -> None:
    for key, _ in state.selector.select(wait_seconds):
        if key.fileobj is state.listener:
            _accept_mpi_channels(state)
            continue
        connection = key.data
        if not isinstance(connection, _MPISupervisorChannel):
            raise RuntimeError("MPI supervisor selector lost its channel identity.")
        _service_mpi_channel(state, connection)


def _scan_mpi_rank_failures(state: _MPILaunchState) -> None:
    for rank, (out, err) in enumerate(state.logs):
        if state.failures[rank]:
            state.job.reason = f"rank {rank} reported failure: {state.failures[rank]}"
            return
        if state.statuses[rank] not in (None, 0):
            state.job.reason = f"rank {rank} failed"
            return
        if (
            os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size
            > state.job.max_output_bytes
        ):
            state.job.reason = f"rank {rank} output exceeds max_output_bytes"
            return


def _mpi_completion_reason(state: _MPILaunchState, status: int) -> str:
    if status != 0:
        return f"MPI launcher exited {status}"
    if any(code is None for code in state.statuses):
        return "MPI launcher exited without complete rank status evidence"
    return ""


def _monitor_mpi_launch(state: _MPILaunchState) -> None:
    process = state.process
    if process is None:
        raise RuntimeError("MPI launcher monitoring requires an owned process.")
    while not state.job.reason:
        _service_mpi_launch(
            state, min(0.02, max(0.0, state.job.deadline - time.monotonic()))
        )
        _scan_mpi_rank_failures(state)
        status = process.poll()
        if not state.job.reason and status is not None:
            state.job.reason = _mpi_completion_reason(state, status)
            break
        if not state.job.reason and time.monotonic() >= state.job.deadline:
            state.job.timed_out = True
            state.job.reason = (
                f"whole-job deadline exceeded {state.job.timeout_seconds:g} seconds"
            )
        if (
            not state.job.reason
            and os.fstat(state.launcher_out.fileno()).st_size
            + os.fstat(state.launcher_err.fileno()).st_size
            > state.job.max_output_bytes
        ):
            state.job.reason = "MPI launcher output exceeds max_output_bytes"


def _teardown_mpi_launch(state: _MPILaunchState) -> None:
    from ._external_runtime import _kill_process_group

    state.tearing_down = True
    # Kill workers before the bounded monitor grace, retaining exact wait()
    # status reports without allowing a rank to hold a collective hostage.
    for rank in range(state.job.plan.num_processes):
        _kill_mpi_rank(state, rank)
    drain_deadline = time.monotonic() + 0.75
    while (
        state.process is not None
        and any(code is None for code in state.statuses)
        and time.monotonic() < drain_deadline
    ):
        _service_mpi_launch(state, min(0.02, max(0.0, drain_deadline - time.monotonic())))
    for group in state.groups:
        _kill_launch_group(group)
    if state.process is not None:
        _kill_process_group(state.process)
    for connection in state.connections:
        connection.socket.close()
    for descriptor in state.pidfds.values():
        os.close(descriptor)


def _mpi_launch_results(state: _MPILaunchState) -> tuple[LocalProcessResult, ...]:
    maximum = state.job.max_output_bytes
    results = tuple(
        LocalProcessResult(
            rank, state.statuses[rank], *_bounded_launch_logs(out, err, maximum)
        )
        for rank, (out, err) in enumerate(state.logs)
    )
    stdout, stderr, truncated = _bounded_launch_logs(
        state.launcher_out, state.launcher_err, maximum
    )
    if not state.job.reason and any(result.output_truncated for result in results):
        state.job.reason = "MPI rank output exceeds max_output_bytes"
    if not state.job.reason and truncated:
        state.job.reason = "MPI launcher output exceeds max_output_bytes"
    if state.job.reason or any(result.returncode != 0 for result in results):
        raise LocalProcessLaunchError(
            results,
            reason=state.job.reason,
            timed_out=state.job.timed_out,
            launcher_returncode=None
            if state.process is None
            else state.process.returncode,
            launcher_stdout=stdout,
            launcher_stderr=stderr,
            launcher_output_truncated=truncated,
        )
    return results


def _launch_mpi_processes(
    arguments: tuple[str, ...],
    plan: LocalProcessLaunchPlan,
    mpi_launcher: Sequence[str],
    *,
    environment: Mapping[str, str] | None,
    cwd: str | None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[LocalProcessResult, ...]:
    if sys.platform != "linux":
        raise RuntimeError("Authenticated site MPI launching requires Linux.")
    launcher_arguments = _admit_mpi_launcher(mpi_launcher, plan)
    job = _LaunchJob(plan, timeout_seconds, max_output_bytes)
    with (
        tempfile.TemporaryDirectory(prefix="phydrax-mpi-") as directory,
        ExitStack() as resources,
    ):
        root = Path(directory)
        launcher_out = resources.enter_context((root / "launcher.stdout").open("w+b"))
        launcher_err = resources.enter_context((root / "launcher.stderr").open("w+b"))
        logs = _open_mpi_rank_logs(root, plan.num_processes, resources)
        listener = resources.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        )
        selector = resources.enter_context(selectors.DefaultSelector())
        state = _MPILaunchState(
            job, selector, listener, root, launcher_out, launcher_err, logs
        )
        config_path = _prepare_mpi_bootstrap(state, arguments, environment)
        try:
            _spawn_mpi_launcher(state, launcher_arguments, config_path, cwd)
            _monitor_mpi_launch(state)
        finally:
            _teardown_mpi_launch(state)
        return _mpi_launch_results(state)


def _mpi_bootstrap(config_path: str, *, monitor: bool) -> int:
    """Preserve the real MPI slot environment; never initialize another world."""
    if sys.platform != "linux":
        raise RuntimeError("Authenticated site MPI bootstrap requires Linux.")
    config: Any = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = LocalProcessLaunchPlan(
        config["num_processes"],
        config["coordinator_address"],
        devices_per_process=config["devices_per_process"],
    )
    rank_text = os.environ.get("OMPI_COMM_WORLD_RANK", "")
    size_text = os.environ.get("OMPI_COMM_WORLD_SIZE", "")
    if (
        not rank_text.isascii()
        or not rank_text.isdecimal()
        or not size_text.isascii()
        or not size_text.isdecimal()
        or str(int(rank_text)) != rank_text
        or size_text != str(plan.num_processes)
        or not 0 <= int(rank_text) < plan.num_processes
    ):
        raise RuntimeError(
            "Site MPI bootstrap requires the authoritative Open MPI rank world."
        )
    rank = int(rank_text)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(1.0)
        channel.connect(config["socket"])
        os.environ[_SUPERVISOR_FD] = str(channel.fileno())
        os.environ[_SUPERVISOR_PID] = str(config["supervisor_pid"])
        _supervisor_exchange(
            "register",
            registration={
                "process_id": rank,
                "role": "monitor" if monitor else "rank",
                "supervisor_id": config["supervisor_id"],
            },
        )
        if monitor:
            worker = subprocess.Popen(
                _mpi_bootstrap_command(config_path, monitor=False),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                shell=False,
            )
            _supervisor_exchange("spawn", registration={"worker_pid": worker.pid})
            code = worker.wait()
            _supervisor_exchange("exit", registration={"returncode": code})
            return code if code >= 0 else 128 - code
        values = dict(os.environ)
        for name in config["bootstrap_environment_keys"]:
            values.pop(name, None)
        values.update(config["rank_bootstrap"][rank])
        values[_SUPERVISOR_FD] = str(channel.fileno())
        values[_SUPERVISOR_PID] = str(config["supervisor_pid"])
        os.set_inheritable(channel.fileno(), True)
        directory = Path(config["directory"])
        with (
            (directory / f"rank-{rank}.stdout").open("ab", buffering=0) as stdout,
            (directory / f"rank-{rank}.stderr").open("ab", buffering=0) as stderr,
        ):
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
        arguments = config["argv"]
        os.execvpe(arguments[0], arguments, values)
    raise RuntimeError("MPI worker exec unexpectedly returned.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--mpi-monitor", "--mpi-exec-rank"):
        raise SystemExit("This module is the private supervised MPI rank bootstrap.")
    raise SystemExit(_mpi_bootstrap(sys.argv[2], monitor=sys.argv[1] == "--mpi-monitor"))

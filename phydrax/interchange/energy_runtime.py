#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only, pinned external energy execution. This is not a security sandbox."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax

from .._fingerprint import canonical_fingerprint, canonical_json
from ..artifacts import ScientificArtifactEnvelope
from ..backends import BackendUnavailableError
from ._energy_worker import (
    _DEFAULT_BYTES,
    _digest_file,
    _receive_packet,
    _relative_path,
    _send_packet,
)
from ._resource import read_bounded_resource, ResourceLimits


def _host_only(*values: Any) -> None:
    # A zero-argument host operation inside jit must also be rejected, not just
    # calls whose arguments happen to contain a tracer.
    if not jax.core.trace_ctx.is_top_level():
        raise TypeError(
            "External energy operations cannot execute inside JAX transformations."
        )
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values)
    ):
        raise TypeError("External energy operations require concrete host values.")


def _positive_timeout(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be positive and finite.")
    return value


def _limits(max_bytes: int) -> ResourceLimits:
    return ResourceLimits(
        max_bytes=max_bytes,
        max_depth=32,
        max_nodes=100000,
        max_attributes=100000,
        max_losses=0,
    )


@dataclass(frozen=True, slots=True)
class PinnedExecutable:
    """Caller-declared release/license and exact executable bytes, rechecked per run.

    The digest pins this file, not every dynamic dependency. ``source_url`` is
    provenance, never evidence of permission to redistribute a tool or its inputs.
    """

    path: str
    sha256: str
    version: str
    license_id: str
    source_url: str = ""

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("The pinned executable must be an executable regular file.")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest.")
        if not self.version.strip() or not self.license_id.strip():
            raise ValueError("Explicit version and license_id are required.")
        object.__setattr__(self, "path", str(path))


def pin_energy_executable(
    path: str | os.PathLike[str], *, version: str, license_id: str, source_url: str = ""
) -> PinnedExecutable:
    """Identify a caller-selected local executable without guessing its release."""
    _host_only()
    resolved = Path(path).expanduser().resolve(strict=True)
    return PinnedExecutable(
        str(resolved), _digest_file(resolved), version, license_id, source_url
    )


@dataclass(frozen=True, slots=True)
class EnergyOutput:
    path: str
    data: bytes
    artifact: ScientificArtifactEnvelope


@dataclass(frozen=True, slots=True)
class EnergyRunResult:
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float
    timed_out: bool
    stdout: bytes
    stderr: bytes
    outputs: tuple[EnergyOutput, ...]
    artifact: ScientificArtifactEnvelope
    error: str = ""

    def output(self, path: str) -> bytes:
        for output in self.outputs:
            if output.path == path:
                return output.data
        raise KeyError(path)

    def require_success(self) -> EnergyRunResult:
        if self.error or self.timed_out or self.returncode != 0:
            raise EnergyRuntimeError(
                self.error or "External command failed.", result=self
            )
        return self


class EnergyRuntimeError(RuntimeError):
    """Execution failure retaining bounded diagnostic and artifact evidence."""

    def __init__(
        self,
        message: str,
        *,
        result: EnergyRunResult | None = None,
        evidence: Mapping[str, Any] | None = None,
    ):
        self.result = result
        self.evidence = dict(evidence or {})
        super().__init__(message)


def _artifact(
    kind: str,
    payload: Any,
    *,
    producer: str,
    version: str,
    build_id: str,
    license_id: str,
    resource_id: str,
    error: str = "",
    parents: tuple[str, ...] = (),
) -> ScientificArtifactEnvelope:
    return ScientificArtifactEnvelope(
        artifact_kind=kind,
        content_digest=hashlib.sha256(
            payload if isinstance(payload, bytes) else canonical_json(payload).encode()
        ).hexdigest(),
        producer=producer,
        producer_version=version,
        build_id=build_id,
        license_id=license_id,
        resource_id=resource_id,
        status="failed" if error else "complete",
        failure_reason=error or "none",
        parent_artifact_ids=parents,
    )


def _stage_inputs(
    root: Path, inputs: Mapping[str, bytes], max_bytes: int
) -> dict[str, str]:
    if max_bytes <= 0 or len(inputs) > 100000:
        raise ValueError("Invalid input resource bounds.")
    if sum(len(data) for data in inputs.values()) > max_bytes:
        raise ValueError("Combined input resources exceed max_output_bytes.")
    identities = {}
    for name, data in inputs.items():
        name = _relative_path(name)
        if name.startswith(".phydrax-"):
            raise ValueError("The .phydrax- prefix is reserved for runtime evidence.")
        if not isinstance(data, bytes):
            raise TypeError("Input resources must be exact bytes.")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        identities[name] = hashlib.sha256(data).hexdigest()
    return identities


def _kill_process_group(process: subprocess.Popen) -> None:
    # A child may have exited while its descendants still hold resources.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_energy_command(
    executable: PinnedExecutable,
    args: Sequence[str],
    *,
    inputs: Mapping[str, bytes],
    outputs: Sequence[str] = (),
    stdin: bytes = b"",
    timeout: float = 120,
    max_output_bytes: int = _DEFAULT_BYTES,
    environment: Mapping[str, str] | None = None,
) -> EnergyRunResult:
    """Execute argv, never a shell, in a private directory; detach bounded outputs.

    Timeout, nonzero exit, pin mutation, missing outputs and output-limit failures
    raise ``EnergyRuntimeError`` with ``result``. The private directory and process
    group are removed on every exit. Engine model files are trusted executable
    inputs: isolation is operational, not a restriction on their host access.
    """
    _host_only(args, inputs, timeout)
    timeout = _positive_timeout(timeout)
    if not isinstance(executable, PinnedExecutable):
        raise TypeError("executable must be a PinnedExecutable.")
    if not isinstance(stdin, bytes) or len(stdin) > max_output_bytes:
        raise ValueError("stdin must be bounded bytes.")
    output_names = tuple(_relative_path(name) for name in outputs)
    if len(output_names) != len(set(output_names)):
        raise ValueError("Requested output paths must be unique.")
    command = (executable.path, *tuple(str(arg) for arg in args))
    if any("\x00" in arg for arg in command):
        raise ValueError("Command arguments cannot contain NUL.")
    start = time.monotonic()
    returncode = None
    timed_out = False
    error = ""
    detached = []
    with tempfile.TemporaryDirectory(prefix="phydrax-energy-") as directory:
        root = Path(directory)
        identities = _stage_inputs(root, inputs, max_output_bytes)
        resource_id = canonical_fingerprint(
            {
                "inputs": identities,
                "stdin": hashlib.sha256(stdin).hexdigest(),
                "command": command,
                "environment_overrides": dict(environment or {}),
                "source_url": executable.source_url,
            }
        )
        out_path, err_path = root / ".phydrax-stdout", root / ".phydrax-stderr"
        in_path = root / ".phydrax-stdin"
        in_path.write_bytes(stdin)
        env = dict(os.environ)
        env.update(
            {"HOME": directory, "TMPDIR": directory, "TMP": directory, "TEMP": directory}
        )
        env.update(environment or {})
        with (
            in_path.open("rb") as inp,
            out_path.open("wb") as out,
            err_path.open("wb") as err,
        ):
            process = None
            try:
                if _digest_file(Path(executable.path)) != executable.sha256:
                    raise ValueError("Executable SHA-256 no longer matches its pin.")
                process = subprocess.Popen(
                    command,
                    cwd=directory,
                    env=env,
                    stdin=inp,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                while process.poll() is None:
                    if time.monotonic() - start >= timeout:
                        timed_out = True
                        error = f"Command exceeded {timeout:g} seconds."
                        break
                    if (
                        out_path.stat().st_size + err_path.stat().st_size
                        > max_output_bytes
                    ):
                        error = "Command logs exceed max_output_bytes."
                        break
                    time.sleep(0.01)
            except (OSError, ValueError) as failure:
                error = f"{type(failure).__name__}: {failure}"
            finally:
                if process is not None:
                    _kill_process_group(process)
                    returncode = process.returncode
        with out_path.open("rb") as stream:
            stdout = stream.read(max_output_bytes)
        with err_path.open("rb") as stream:
            stderr = stream.read(max(0, max_output_bytes - len(stdout)))
        if out_path.stat().st_size + err_path.stat().st_size > max_output_bytes:
            error = error or "Command logs exceed max_output_bytes."
        if not error and returncode != 0:
            error = f"Command exited with status {returncode}."
        if not error:
            try:
                if _digest_file(Path(executable.path)) != executable.sha256:
                    error = "Executable changed during execution."
            except OSError as failure:
                error = f"Executable identity could not be rechecked: {failure}"
        remaining = max_output_bytes - len(stdout) - len(stderr)
        for name in output_names:
            try:
                resource = read_bounded_resource(
                    name, trusted_root=root, limits=_limits(max(1, remaining))
                )
                if len(resource.data) > remaining:
                    raise ValueError("Combined outputs exceed max_output_bytes.")
                remaining -= len(resource.data)
                detached.append(
                    EnergyOutput(
                        name,
                        resource.data,
                        _artifact(
                            "energy-engine-output",
                            resource.data,
                            producer=Path(executable.path).name,
                            version=executable.version,
                            build_id=executable.sha256,
                            license_id=executable.license_id,
                            resource_id=resource.manifest.manifest_id,
                            error=error,
                        ),
                    )
                )
            except (OSError, ValueError) as failure:
                error = error or f"Output {name!r}: {failure}"
        elapsed = time.monotonic() - start
        evidence = {
            "command": command,
            "executable_sha256": executable.sha256,
            "input_id": resource_id,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "error": error,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "outputs": [(item.path, item.artifact.artifact_id) for item in detached],
        }
        artifact = _artifact(
            "energy-engine-run",
            evidence,
            producer=Path(executable.path).name,
            version=executable.version,
            build_id=executable.sha256,
            license_id=executable.license_id,
            resource_id=resource_id,
            error=error,
        )
        result = EnergyRunResult(
            command,
            returncode,
            elapsed,
            timed_out,
            stdout,
            stderr,
            tuple(detached),
            artifact,
            error,
        )
    return result.require_success()


def run_energyplus(
    executable: PinnedExecutable,
    model: bytes,
    weather: bytes,
    *,
    model_format: str = "idf",
    outputs: Sequence[str] = ("eplusout.csv", "eplusout.err"),
    inputs: Mapping[str, bytes] | None = None,
    timeout: float = 120,
    max_output_bytes: int = _DEFAULT_BYTES,
) -> EnergyRunResult:
    """Run a pinned EnergyPlus CLI with exact IDF/epJSON and EPW bytes."""
    if model_format not in ("idf", "epjson"):
        raise ValueError("model_format must be 'idf' or 'epjson'.")
    staged = dict(inputs or {})
    model_name = "model.idf" if model_format == "idf" else "model.epJSON"
    if model_name in staged or "weather.epw" in staged:
        raise ValueError("Additional inputs collide with the model/weather paths.")
    staged.update({model_name: model, "weather.epw": weather})
    requested = tuple(dict.fromkeys((*outputs, "eplusout.err")))
    result = run_energy_command(
        executable,
        ("--weather", "weather.epw", "--output-directory", ".", "--readvars", model_name),
        inputs=staged,
        outputs=requested,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
    )
    # EnergyPlus can report fatal/severe model errors independently of CLI status.
    for item in result.outputs:
        if item.path.endswith(".err") and (
            b"**  Fatal  **" in item.data or b"** Severe  **" in item.data
        ):
            error = "EnergyPlus reported severe/fatal model errors."
            failed = _artifact(
                "energyplus-model-validation",
                {"run": result.artifact.artifact_id, "error": error},
                producer="EnergyPlus",
                version=executable.version,
                build_id=executable.sha256,
                license_id=executable.license_id,
                resource_id=result.artifact.resource_id,
                error=error,
                parents=(result.artifact.artifact_id,),
            )
            raise EnergyRuntimeError(
                error, result=replace(result, artifact=failed, error=error)
            )
    return result


def run_radiance_command(
    executable: PinnedExecutable,
    args: Sequence[str],
    *,
    inputs: Mapping[str, bytes],
    outputs: Sequence[str] = (),
    stdin: bytes = b"",
    timeout: float = 120,
    max_output_bytes: int = _DEFAULT_BYTES,
    environment: Mapping[str, str] | None = None,
) -> EnergyRunResult:
    """Run oconv/rtrace/rfluxmtx/etc.; explicitly pass prior-stage bytes, not a shell pipe."""
    return run_energy_command(
        executable,
        args,
        inputs=inputs,
        outputs=outputs,
        stdin=stdin,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        environment=environment,
    )


def _require_optional(module: str, requirement: str) -> None:
    _host_only()
    if importlib.util.find_spec(module) is None:
        raise BackendUnavailableError(
            module,
            "host-execution",
            requirement,
            f"optional Python dependency {module!r} is not installed",
        )


class _HostWorker:
    """Private process transport for native calls that cannot be interrupted in Python."""

    def __init__(
        self,
        kind: str,
        config: Mapping[str, Any],
        *,
        inputs: Mapping[str, bytes],
        timeout: float,
        max_bytes: int = _DEFAULT_BYTES,
    ):
        _host_only(config)
        if os.name != "posix":
            raise OSError(
                "Isolated optional energy sessions currently require a POSIX host."
            )
        self.timeout = _positive_timeout(timeout)
        self.max_bytes = int(max_bytes)
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self._temporary = tempfile.TemporaryDirectory(prefix=f"phydrax-{kind}-")
        self.root = Path(self._temporary.name)
        self._process = None
        self._socket = None
        self._logs = None
        try:
            self.input_ids = _stage_inputs(self.root, inputs, max_bytes)
            self._logs = (self.root / ".phydrax-worker.log").open("w+b")
            parent, child = socket.socketpair()
            self._socket = parent
            environment = dict(os.environ)
            environment.update({"HOME": str(self.root), "TMPDIR": str(self.root)})
            try:
                # Do not put interchange/ on sys.path: its helics/ adapter would
                # shadow the optional native helics distribution in the child.
                self._process = subprocess.Popen(
                    [
                        sys.executable,
                        "-P",
                        str(Path(__file__).with_name("_energy_worker.py")),
                        str(child.fileno()),
                    ],
                    pass_fds=(child.fileno(),),
                    cwd=self.root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._logs,
                    stderr=self._logs,
                    start_new_session=True,
                )
            finally:
                child.close()
            self.info = self.call("open", {"kind": kind, "config": dict(config)})
            self.info["build_id"] = canonical_fingerprint(self.info.pop("build_evidence"))
        except BaseException:
            self.abort()
            raise

    def call(self, operation: str, payload: Mapping[str, Any] | None = None) -> Any:
        _host_only(payload)
        if self.closed:
            raise RuntimeError("External session is closed.")
        started = time.monotonic()
        request = {"operation": operation, "payload": dict(payload or {})}
        evidence = {"operation": operation, "request_id": canonical_fingerprint(request)}
        try:
            self._socket.settimeout(self.timeout)
            _send_packet(self._socket, request)
            if os.fstat(self._logs.fileno()).st_size > self.max_bytes:
                raise ValueError(
                    "External runtime logs exceed the configured byte limit."
                )
            deadline = started + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"External {operation} exceeded {self.timeout:g} seconds."
                    )
                if os.fstat(self._logs.fileno()).st_size > self.max_bytes:
                    raise ValueError(
                        "External runtime logs exceed the configured byte limit."
                    )
                ready, _, _ = select.select([self._socket], [], [], min(remaining, 0.05))
                if ready:
                    self._socket.settimeout(max(0.001, deadline - time.monotonic()))
                    response = _receive_packet(self._socket, self.max_bytes)
                    break
            if not response["ok"]:
                raise EnergyRuntimeError(response["error"], evidence=response)
            if os.fstat(self._logs.fileno()).st_size > self.max_bytes:
                raise ValueError(
                    "External runtime logs exceed the configured byte limit."
                )
            evidence.update(
                status="complete",
                elapsed_seconds=time.monotonic() - started,
                response_id=canonical_fingerprint(response["value"]),
            )
            self.calls.append(evidence)
            return response["value"]
        except BaseException as failure:
            evidence.update(
                status="failed",
                elapsed_seconds=time.monotonic() - started,
                timed_out=isinstance(failure, TimeoutError),
                error=f"{type(failure).__name__}: {failure}",
            )
            self._logs.seek(0)
            evidence["log"] = self._logs.read(self.max_bytes).decode(
                "utf-8", errors="replace"
            )
            self.calls.append(evidence)
            self.abort()
            evidence["returncode"] = self._process.returncode
            if not isinstance(failure, Exception):
                raise
            raise EnergyRuntimeError(str(failure), evidence=evidence) from failure

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.call("close")
        finally:
            self.abort()

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._process is not None:
            _kill_process_group(self._process)
        if self._socket is not None:
            self._socket.close()
        if self._logs is not None:
            self._logs.close()
        self._temporary.cleanup()


@dataclass(frozen=True, slots=True)
class OpenDSSRunResult:
    """Raw multiphase RMS OpenDSS output; never silently balanced or rescaled.

    Node voltages are (real, imaginary) volts, ordered by ``node_names``. Total
    power is the engine's power into circuit sources in kW/kvar (negative for
    ordinary supplying sources); losses are positive-consumption W/var.
    Element powers are terminal-major, conductor-minor kW/kvar, inward-positive.
    """

    converged: bool
    bus_names: tuple[str, ...]
    node_names: tuple[str, ...]
    node_voltages: tuple[tuple[float, float], ...]
    node_voltages_pu: tuple[float, ...]
    total_power: tuple[float, float]
    losses: tuple[float, float]
    element_powers: tuple[tuple[str, int, int, tuple[tuple[float, float], ...]], ...]
    outputs: tuple[EnergyOutput, ...]
    artifact: ScientificArtifactEnvelope
    engine_version: str


def run_opendss(
    commands: Sequence[str],
    *,
    license_id: str,
    inputs: Mapping[str, bytes] | None = None,
    outputs: Sequence[str] = (),
    timeout: float = 120,
    max_output_bytes: int = _DEFAULT_BYTES,
    expected_version: str | None = None,
    source_url: str = "",
) -> OpenDSSRunResult:
    """Execute real OpenDSSDirect.py commands, including an explicit Solve command.

    ``expected_version`` enforces a caller pin; observed package/native-build
    identities are always recorded. Missing optional libraries fail explicitly.
    """
    _require_optional(
        "opendssdirect", "install opendssdirect.py>=0.9 with its DSS native engine"
    )
    if not license_id.strip() or not commands:
        raise ValueError("An explicit license_id and nonempty commands are required.")
    output_names = tuple(_relative_path(name) for name in outputs)
    worker = _HostWorker(
        "opendss",
        {"expected_version": expected_version},
        inputs=inputs or {},
        timeout=timeout,
        max_bytes=max_output_bytes,
    )
    try:
        data = worker.call("run", {"commands": list(commands)})
        error = "" if data["converged"] else "OpenDSS solution did not converge."
        resource_id = canonical_fingerprint(
            {
                "commands": list(commands),
                "inputs": worker.input_ids,
                "source_url": source_url,
            }
        )
        detached = []
        remaining = max_output_bytes
        for name in output_names:
            resource = read_bounded_resource(
                name, trusted_root=worker.root, limits=_limits(max(1, remaining))
            )
            remaining -= len(resource.data)
            if remaining < 0:
                raise ValueError("Combined OpenDSS outputs exceed the byte limit.")
            detached.append(
                EnergyOutput(
                    name,
                    resource.data,
                    _artifact(
                        "opendss-output",
                        resource.data,
                        producer="OpenDSSDirect.py",
                        version=worker.info["version"],
                        build_id=worker.info["build_id"],
                        license_id=license_id,
                        resource_id=resource.manifest.manifest_id,
                        error=error,
                    ),
                )
            )
        worker.close()
        artifact = _artifact(
            "opendss-run",
            {
                "data": data,
                "calls": worker.calls,
                "outputs": [item.artifact.artifact_id for item in detached],
                "engine": worker.info,
            },
            producer="OpenDSSDirect.py",
            version=worker.info["version"],
            build_id=worker.info["build_id"],
            license_id=license_id,
            resource_id=resource_id,
            error=error,
        )
        result = OpenDSSRunResult(
            data["converged"],
            tuple(data["bus_names"]),
            tuple(data["node_names"]),
            tuple(tuple(value) for value in data["node_voltages"]),
            tuple(data["node_voltages_pu"]),
            tuple(data["total_power"]),
            tuple(data["losses"]),
            tuple(
                (name, terminals, conductors, tuple(tuple(p) for p in powers))
                for name, terminals, conductors, powers in data["element_powers"]
            ),
            tuple(detached),
            artifact,
            worker.info["engine_version"],
        )
        if error:
            raise EnergyRuntimeError(
                error, evidence={"artifact_id": artifact.artifact_id, "data": data}
            )
        return result
    finally:
        worker.close()


__all__ = [
    "PinnedExecutable",
    "EnergyOutput",
    "EnergyRunResult",
    "EnergyRuntimeError",
    "OpenDSSRunResult",
    "pin_energy_executable",
    "run_energy_command",
    "run_energyplus",
    "run_radiance_command",
    "run_opendss",
]

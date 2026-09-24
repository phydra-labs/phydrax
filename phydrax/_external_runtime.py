#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only external execution: pinned engines, host inference, staged adjoints.

This is not a security sandbox.
"""

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
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from ._external_resource import read_bounded_resource, ResourceLimits
from ._external_worker import (
    _DEFAULT_BYTES,
    _ERROR_PACKET_BYTES,
    _receive_packet,
    _relative_path,
    _send_packet,
)
from ._fingerprint import canonical_fingerprint, canonical_json
from ._host_io import open_regular_file
from ._identity import ArtifactBindingIdentity
from ._model._component import ExecutionCapabilities
from .artifacts import ScientificArtifactEnvelope
from .backends._types import BackendUnavailableError
from .logging import emit


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


def _require_execution(capabilities: ExecutionCapabilities, /, *values: Any) -> None:
    """Admit one external invocation before anything reaches the runtime.

    The declared capabilities are checked first. A host-only model then refuses
    every active JAX transformation (`jit`, `vmap`, `grad`, `jvp`, `vjp`) and,
    through `_host_only`, traced values, so a refused call never invokes the
    runner.
    """
    if not isinstance(capabilities, ExecutionCapabilities):
        raise TypeError("External execution requires declared ExecutionCapabilities.")
    if not capabilities.host_only:
        return
    if not jax.core.trace_ctx.is_top_level():
        raise TypeError(
            f"Host-only {capabilities.tier!r} execution cannot run inside JAX "
            "transformations (jit, vmap, grad, jvp, vjp); call it eagerly with "
            "concrete values."
        )
    _host_only(*values)


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


ExternalIsolation: type = Literal["trusted-local"]
ExternalEnforcement: type = Literal[
    "trusted-local-direct-descriptor",
    "trusted-local-private-snapshot",
    "trusted-local-verified-path",
]


@dataclass(frozen=True, slots=True)
class ExternalExecutionPolicy:
    """Truthful policy for a direct process trusted to access the local host."""

    isolation: ExternalIsolation = "trusted-local"
    network_access: Literal[True] = True
    inherit_environment: bool = True
    allowed_environment_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.isolation != "trusted-local":
            raise ValueError(
                "Container and sandbox isolation require an enforcing launcher; "
                "this runtime supports trusted-local execution only."
            )
        if self.network_access is not True:
            raise ValueError(
                "Network denial requires an enforcing launcher; trusted-local "
                "execution has unrestricted host network access."
            )
        if not isinstance(self.inherit_environment, bool):
            raise TypeError("inherit_environment must be a bool.")
        variables = tuple(str(value) for value in self.allowed_environment_variables)
        if len(set(variables)) != len(variables) or any(
            not value or not value.replace("_", "").isalnum() for value in variables
        ):
            raise ValueError(
                "Allowed environment variables must be unique canonical names."
            )
        if self.inherit_environment and variables:
            raise ValueError(
                "An environment allowlist requires inherit_environment=False."
            )
        object.__setattr__(
            self, "allowed_environment_variables", tuple(sorted(variables))
        )

    @property
    def policy_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "external-execution-policy",
                "isolation": self.isolation,
                "network_access": self.network_access,
                "inherit_environment": self.inherit_environment,
                "allowed_environment_variables": list(self.allowed_environment_variables),
            }
        )


@dataclass(frozen=True, slots=True)
class PinnedExecutable:
    """Caller-declared release/license and exact descriptor-executed bytes.

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
    """Identify exact bytes of a caller-selected trusted-local executable."""
    _host_only()
    resolved = Path(path).expanduser().resolve(strict=True)
    with open_regular_file(resolved) as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return PinnedExecutable(str(resolved), digest, version, license_id, source_url)


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
    execution_policy_id: str
    isolation: ExternalIsolation
    network_access: Literal[True]
    enforcement: ExternalEnforcement
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


def _descriptor_execution_path(descriptor: int, /) -> str:
    if os.name != "posix" or not os.path.isdir("/proc/self/fd"):
        raise OSError("The host does not expose executable descriptor paths.")
    return f"/proc/self/fd/{descriptor}"


def _execution_enforcement(executable: PinnedExecutable, /) -> ExternalEnforcement:
    if sys.platform != "darwin":
        return "trusted-local-direct-descriptor"
    with open_regular_file(executable.path) as stream:
        script = stream.read(2) == b"#!"
    return "trusted-local-private-snapshot" if script else "trusted-local-verified-path"


def _snapshot_executable(
    executable: PinnedExecutable,
    root: Path,
    /,
) -> Path:
    destination = root / ".phydrax-executable"
    digest = hashlib.sha256()
    with (
        open_regular_file(executable.path) as source,
        destination.open("xb") as target,
    ):
        while block := source.read(1024 * 1024):
            digest.update(block)
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
        os.fchmod(target.fileno(), 0o500)
    if digest.hexdigest() != executable.sha256:
        destination.unlink(missing_ok=True)
        raise ValueError("Executable SHA-256 no longer matches its pin.")
    return destination


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
    execution_policy: ExternalExecutionPolicy | None = None,
) -> EnergyRunResult:
    """Execute trusted local argv after verifying a private executable snapshot.

    Linux executes through the held snapshot descriptor. Darwin executes scripts
    from the private snapshot; path-sensitive native binaries use their verified
    configured path under the declared trusted-local threat model.
    """
    _host_only(args, inputs, timeout)
    timeout = _positive_timeout(timeout)
    policy = ExternalExecutionPolicy() if execution_policy is None else execution_policy
    if not isinstance(policy, ExternalExecutionPolicy):
        raise TypeError("execution_policy must be ExternalExecutionPolicy or None.")
    overrides = dict(environment or {})
    disallowed = tuple(
        sorted(set(overrides).difference(policy.allowed_environment_variables))
    )
    if disallowed and not policy.inherit_environment:
        raise ValueError(
            "Environment overrides exceed the execution allowlist: "
            + ", ".join(disallowed)
        )
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer.")
    if not isinstance(executable, PinnedExecutable):
        raise TypeError("executable must be a PinnedExecutable.")
    if not isinstance(stdin, bytes) or len(stdin) > max_output_bytes:
        raise ValueError("stdin must be bounded bytes.")
    output_names = tuple(_relative_path(name) for name in outputs)
    if any(name.startswith(".phydrax-") for name in output_names):
        raise ValueError("The .phydrax- prefix is reserved for runtime evidence.")
    if len(output_names) != len(set(output_names)):
        raise ValueError("Requested output paths must be unique.")
    command = (executable.path, *tuple(str(arg) for arg in args))
    if any("\x00" in arg for arg in command):
        raise ValueError("Command arguments cannot contain NUL.")
    start = time.monotonic()
    returncode = None
    timed_out = False
    error = ""
    stdout = b""
    stderr = b""
    detached = []
    enforcement = _execution_enforcement(executable)
    with tempfile.TemporaryDirectory(prefix="phydrax-energy-") as directory:
        root = Path(directory)
        identities = _stage_inputs(root, inputs, max_output_bytes)
        resource_id = canonical_fingerprint(
            {
                "inputs": identities,
                "stdin": hashlib.sha256(stdin).hexdigest(),
                "command": command,
                "environment_overrides": overrides,
                "execution_policy_id": policy.policy_id,
                "isolation": policy.isolation,
                "network_access": policy.network_access,
                "source_url": executable.source_url,
                "enforcement": enforcement,
            }
        )
        emit(
            "DEBUG",
            "provider.execution.started",
            "Energy provider execution started",
            executable=Path(executable.path).name,
            input_count=len(inputs),
            output_count=len(output_names),
            resource_id=resource_id,
        )
        out_path, err_path = root / ".phydrax-stdout", root / ".phydrax-stderr"
        in_path = root / ".phydrax-stdin"
        in_path.write_bytes(stdin)
        env = (
            dict(os.environ)
            if policy.inherit_environment
            else {
                key: os.environ[key]
                for key in policy.allowed_environment_variables
                if key in os.environ
            }
        )
        env.update(
            {"HOME": directory, "TMPDIR": directory, "TMP": directory, "TEMP": directory}
        )
        env.update(overrides)
        with (
            in_path.open("rb") as inp,
            out_path.open("w+b") as out,
            err_path.open("w+b") as err,
        ):
            process = None
            try:
                executable_snapshot = _snapshot_executable(executable, root)
                with open_regular_file(executable_snapshot) as executable_stream:
                    executable_descriptor = executable_stream.fileno()
                    execution_path = (
                        str(executable_snapshot)
                        if enforcement == "trusted-local-private-snapshot"
                        else executable.path
                        if enforcement == "trusted-local-verified-path"
                        else _descriptor_execution_path(executable_descriptor)
                    )
                    inherited_descriptors = (
                        (executable_descriptor,)
                        if enforcement == "trusted-local-direct-descriptor"
                        else ()
                    )
                    process = subprocess.Popen(
                        command,
                        executable=execution_path,
                        pass_fds=inherited_descriptors,
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
                            os.fstat(out.fileno()).st_size
                            + os.fstat(err.fileno()).st_size
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
            total_log_bytes = (
                os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size
            )
            out.seek(0)
            stdout = out.read(max_output_bytes)
            err.seek(0)
            stderr = err.read(max(0, max_output_bytes - len(stdout)))
            if total_log_bytes > max_output_bytes:
                error = error or "Command logs exceed max_output_bytes."
        if not error and returncode != 0:
            error = f"Command exited with status {returncode}."
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
            "execution_policy_id": policy.policy_id,
            "isolation": policy.isolation,
            "network_access": policy.network_access,
            "enforcement": enforcement,
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
            policy.policy_id,
            policy.isolation,
            policy.network_access,
            enforcement,
            artifact,
            error,
        )
    emit(
        "ERROR" if result.error else "INFO",
        ("provider.execution.failed" if result.error else "provider.execution.completed"),
        "Energy provider execution finished",
        elapsed_seconds=result.elapsed_seconds,
        executable=Path(executable.path).name,
        output_artifact_count=len(result.outputs),
        resource_id=resource_id,
        return_code=result.returncode,
        stderr_bytes=len(result.stderr),
        stdout_bytes=len(result.stdout),
        timed_out=result.timed_out,
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
    """Private trusted-local process transport for blocking native calls."""

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
                "Optional native energy sessions currently require a POSIX host."
            )
        self.timeout = _positive_timeout(timeout)
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        self.max_bytes = max_bytes
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self._temporary = tempfile.TemporaryDirectory(prefix=f"phydrax-{kind}-")
        self.root = Path(self._temporary.name)
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._logs: BinaryIO | None = None
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
                        str(Path(__file__).with_name("_external_worker.py")),
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
            self.info["execution"] = {
                "isolation": "trusted-local",
                "network_access": True,
                "enforcement": "trusted-local-direct-process",
            }
        except BaseException:
            self.abort()
            raise

    def call(self, operation: str, payload: Mapping[str, Any] | None = None) -> Any:
        _host_only(payload)
        if self.closed:
            raise RuntimeError("External session is closed.")
        connection = self._socket
        logs = self._logs
        process = self._process
        if connection is None or logs is None or process is None:
            raise RuntimeError("External session transport is not initialized.")
        started = time.monotonic()
        request = {
            "operation": operation,
            "payload": dict(payload or {}),
            "maximum_response_bytes": self.max_bytes,
        }
        evidence = {
            "operation": operation,
            "request_id": canonical_fingerprint(request),
            "isolation": "trusted-local",
            "network_access": True,
            "enforcement": "trusted-local-direct-process",
        }
        try:
            _send_packet(connection, request, self.max_bytes)
            if os.fstat(logs.fileno()).st_size > self.max_bytes:
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
                if os.fstat(logs.fileno()).st_size > self.max_bytes:
                    raise ValueError(
                        "External runtime logs exceed the configured byte limit."
                    )
                ready, _, _ = select.select([connection], [], [], min(remaining, 0.05))
                if ready:
                    connection.settimeout(max(0.001, deadline - time.monotonic()))
                    response, response_bytes = _receive_packet(
                        connection,
                        max(self.max_bytes, _ERROR_PACKET_BYTES),
                        include_size=True,
                    )
                    break
            if response["ok"] and response_bytes > self.max_bytes:
                raise ValueError(
                    "External runtime response exceeds the configured byte limit."
                )
            if not response["ok"]:
                raise EnergyRuntimeError(response["error"], evidence=response)
            if os.fstat(logs.fileno()).st_size > self.max_bytes:
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
            logs.seek(0)
            evidence["log"] = logs.read(self.max_bytes).decode("utf-8", errors="replace")
            self.calls.append(evidence)
            self.abort()
            evidence["returncode"] = process.returncode
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
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer.")
    _require_optional(
        "opendssdirect", "install opendssdirect.py>=0.9 with its DSS native engine"
    )
    if not license_id.strip() or not commands:
        raise ValueError("An explicit license_id and nonempty commands are required.")
    output_names = tuple(_relative_path(name) for name in outputs)
    command_list = list(commands)
    response_limits = {
        "max_bytes": max_output_bytes,
        "max_items": max(1, min(1_000_000, max_output_bytes // 16)),
    }
    worker = _HostWorker(
        "opendss",
        {"expected_version": expected_version},
        inputs=inputs or {},
        timeout=timeout,
        max_bytes=max_output_bytes,
    )
    try:
        data = worker.call(
            "run",
            {
                "commands": command_list,
                "response_limits": response_limits,
            },
        )
        error = "" if data["converged"] else "OpenDSS solution did not converge."
        resource_id = canonical_fingerprint(
            {
                "commands": command_list,
                "inputs": worker.input_ids,
                "response_limits": response_limits,
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


# External model tiers ---------------------------------------------------------------

ExternalTransport: type = Literal["copy", "dlpack"]
ExternalDerivativeRoute: type = Literal["external-adjoint", "none"]

# Phydrax methods that need only function values. A provider without an adjoint
# reports them; none is ever selected on the caller's behalf.
_DERIVATIVE_FREE_ALTERNATIVES = (
    "phydrax.optim.OpenEvolutionStrategy",
    "phydrax.optim.POUNDERS",
    "phydrax.optim.search_differential_evolution",
    "phydrax.uq.fit_eki",
)


@dataclass(frozen=True, slots=True)
class ExternalDerivativeSupport:
    """Derivative route of one external provider; nothing is selected implicitly.

    `route="external-adjoint"` means the provider applies staged adjoint actions
    (`ExternalAdjointAction`). `route="none"` means it has no adjoint:
    `alternatives` then lists the Phydrax derivative-free methods that can drive
    it through eager function values. Phydrax never switches to one silently.
    """

    route: ExternalDerivativeRoute
    alternatives: tuple[str, ...]

    def __post_init__(self) -> None:
        match self.route:
            case "external-adjoint":
                if self.alternatives:
                    raise ValueError("An adjoint provider reports no alternatives.")
            case "none":
                if not self.alternatives:
                    raise ValueError(
                        "A provider without an adjoint reports derivative-free "
                        "alternatives."
                    )
            case _:
                raise ValueError(f"Unknown external derivative route {self.route!r}.")


@dataclass(frozen=True, slots=True)
class ExternalTensorSpec:
    """Declared name, exact shape, and exact dtype of one host tensor.

    Values are never cast or reshaped to fit: a mismatch is refused. `dtype`
    accepts any NumPy dtype specification and is stored as its canonical
    `numpy.dtype.str`.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tensor spec names must be non-empty strings.")
        if isinstance(self.shape, (str, bytes)) or not isinstance(self.shape, Sequence):
            raise TypeError("Tensor spec shapes must be integer sequences.")
        shape = tuple(self.shape)
        if any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer))
            for size in shape
        ):
            raise TypeError("Tensor spec shape dimensions must be integers.")
        if any(size < 0 for size in shape):
            raise ValueError("Tensor spec shape dimensions must be nonnegative.")
        dtype = np.dtype(self.dtype)
        if dtype.hasobject:
            raise TypeError("Tensor specs require a numeric dtype.")
        object.__setattr__(self, "shape", tuple(int(size) for size in shape))
        object.__setattr__(self, "dtype", dtype.str)

    def check(self, array: np.ndarray, role: str, /) -> np.ndarray:
        """Return `array` after refusing any shape or dtype mismatch."""
        if tuple(array.shape) != self.shape:
            raise ValueError(
                f"{role} {self.name!r} must have shape {self.shape}; got {array.shape}."
            )
        if array.dtype.str != self.dtype:
            raise TypeError(
                f"{role} {self.name!r} must have dtype {self.dtype}; "
                f"got {array.dtype.str}."
            )
        return array


def _tensor_schema(
    specs: Sequence[ExternalTensorSpec], name: str, /
) -> tuple[ExternalTensorSpec, ...]:
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError(f"{name} must be a sequence of ExternalTensorSpec.")
    schema = tuple(specs)
    if not schema or any(not isinstance(spec, ExternalTensorSpec) for spec in schema):
        raise TypeError(f"{name} must be a non-empty sequence of ExternalTensorSpec.")
    if len({spec.name for spec in schema}) != len(schema):
        raise ValueError(f"{name} names must be unique.")
    return schema


def _schema_record(schema: tuple[ExternalTensorSpec, ...], /) -> list[list[Any]]:
    return [[spec.name, list(spec.shape), spec.dtype] for spec in schema]


def _require_finite(array: np.ndarray, role: str, name: str, /) -> np.ndarray:
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise RuntimeError(f"{role} {name!r} contains non-finite values.")
    return array


def _read_only(value: Any, /) -> np.ndarray:
    # An owned read-only array is already detached; anything else is copied.
    if isinstance(value, np.ndarray) and value.base is None and not value.flags.writeable:
        return value
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _detached(value: Any, spec: ExternalTensorSpec, role: str, /) -> np.ndarray:
    """Return a read-only host copy of `value` checked against `spec`."""
    return spec.check(_read_only(value), role)


def _array_record(array: np.ndarray, /) -> list[Any]:
    contiguous = np.ascontiguousarray(array)
    return [
        list(array.shape),
        array.dtype.str,
        hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    ]


def _host_transport_input(
    value: Any, spec: ExternalTensorSpec, transport: ExternalTransport, /
) -> np.ndarray:
    match transport:
        case "copy":
            array = np.array(value, copy=True)
        case "dlpack":
            if not hasattr(value, "__dlpack__"):
                raise TypeError(
                    f"DLPack transport requires input {spec.name!r} to export __dlpack__."
                )
            array = np.from_dlpack(value)
        case _:
            raise ValueError(f"Unknown host transport {transport!r}.")
    return spec.check(array, "Host inference input")


def _host_transport_output(
    value: Any, spec: ExternalTensorSpec, transport: ExternalTransport, /
) -> jax.Array:
    role = "Host inference output"
    match transport:
        case "copy":
            host = _require_finite(spec.check(np.asarray(value), role), role, spec.name)
            return jnp.array(host, copy=True)
        case "dlpack":
            host = _require_finite(
                spec.check(np.from_dlpack(value), role), role, spec.name
            )
            return jnp.from_dlpack(host)
        case _:
            raise ValueError(f"Unknown host transport {transport!r}.")


_HOST_INFERENCE_CAPABILITIES = ExecutionCapabilities("host-inference", host_only=True)
_EXTERNAL_ADJOINT_CAPABILITIES = ExecutionCapabilities("external-adjoint", host_only=True)


# A weak-reference slot lets `jax.jit(adapter)` trace far enough to be refused
# by the capability guard instead of failing on the callable itself.
@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class HostInferenceAdapter:
    """Eager inference through a host runtime outside JAX.

    `runner` receives one host NumPy array per `input_schema` entry, in schema
    order, and returns one array per `output_schema` entry. Both sides are checked
    exactly against the schemas and outputs must be finite. Results are detached
    concrete JAX arrays: they carry no derivative relation to the inputs.

    `transport="copy"` copies across the host boundary. `transport="dlpack"`
    exchanges buffers without copying through DLPack: inputs must export
    `__dlpack__` and outputs alias the buffers the runtime returned.

    The adapter is host-only (`capabilities.tier == "host-inference"`): `jit`,
    `vmap`, `grad`, `jvp`, and `vjp` are refused before the runtime is invoked.
    It offers no adjoint, so `derivative_support` reports derivative-free
    alternatives without selecting one. `binding` is the artifact binding
    identity of the loaded model.
    """

    runner: Callable[[tuple[np.ndarray, ...]], Sequence[Any]]
    input_schema: tuple[ExternalTensorSpec, ...]
    output_schema: tuple[ExternalTensorSpec, ...]
    binding: ArtifactBindingIdentity
    transport: ExternalTransport = "copy"

    def __post_init__(self) -> None:
        if not callable(self.runner):
            raise TypeError("runner must be callable.")
        object.__setattr__(
            self, "input_schema", _tensor_schema(self.input_schema, "input_schema")
        )
        object.__setattr__(
            self, "output_schema", _tensor_schema(self.output_schema, "output_schema")
        )
        if not isinstance(self.binding, ArtifactBindingIdentity):
            raise TypeError("binding must be an ArtifactBindingIdentity.")
        if self.transport not in ("copy", "dlpack"):
            raise ValueError("transport must be 'copy' or 'dlpack'.")

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Host-only `"host-inference"` execution capabilities."""
        return _HOST_INFERENCE_CAPABILITIES

    @property
    def derivative_support(self) -> ExternalDerivativeSupport:
        """No adjoint; the derivative-free alternatives, none selected."""
        return ExternalDerivativeSupport("none", _DERIVATIVE_FREE_ALTERNATIVES)

    def __call__(self, *inputs: Any) -> jax.Array | tuple[jax.Array, ...]:
        _require_execution(self.capabilities, inputs)
        if len(inputs) != len(self.input_schema):
            raise ValueError(
                f"Host inference expects {len(self.input_schema)} inputs; "
                f"got {len(inputs)}."
            )
        host = tuple(
            _host_transport_input(value, spec, self.transport)
            for value, spec in zip(inputs, self.input_schema, strict=True)
        )
        raw = self.runner(host)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("Host inference runners must return a sequence of arrays.")
        if len(raw) != len(self.output_schema):
            raise RuntimeError(
                f"Host inference returned {len(raw)} outputs; the output schema "
                f"declares {len(self.output_schema)}."
            )
        outputs = tuple(
            _host_transport_output(value, spec, self.transport)
            for value, spec in zip(raw, self.output_schema, strict=True)
        )
        return outputs[0] if len(outputs) == 1 else outputs


def _record_pairs(value: Any, name: str, /) -> tuple[tuple[str, Any], ...]:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if any(
        not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
        for item in items
    ):
        raise TypeError(f"{name} must be a mapping or a sequence of (name, value).")
    keys = [key for key, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} names must be unique.")
    return tuple(sorted(items, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True, eq=False)
class ExternalPrimalStage:
    """One detached external primal evaluation staged for its adjoint.

    `inputs` and `outputs` are read-only host copies; a failed primal
    (`failure_reason` non-empty) exposes no outputs and no replay data.
    `realization_id` names the provider realization the primal produced (for
    example state, mesh, and design digests); `replay_data` holds the
    provider's detached arrays its adjoint replays against; `evidence` names
    artifact and convergence evidence IDs. `replay_id` content-addresses all of
    it together with the owning action's `action_id`.
    """

    action_id: str
    inputs: tuple[np.ndarray, ...]
    outputs: tuple[np.ndarray, ...]
    realization_id: str
    replay_data: tuple[tuple[str, np.ndarray], ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()
    failure_reason: str = ""
    replay_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be a non-empty string.")
        if not isinstance(self.failure_reason, str):
            raise TypeError("failure_reason must be a string.")
        if not isinstance(self.realization_id, str):
            raise TypeError("realization_id must be a string.")
        inputs = tuple(_read_only(value) for value in self.inputs)
        outputs = tuple(_read_only(value) for value in self.outputs)
        replay_data = tuple(
            (name, _read_only(value))
            for name, value in _record_pairs(self.replay_data, "replay_data")
        )
        evidence = _record_pairs(self.evidence, "evidence")
        if any(not isinstance(value, str) for _, value in evidence):
            raise TypeError("evidence values must be identifier strings.")
        if self.failure_reason:
            if outputs or replay_data:
                raise ValueError(
                    "A failed external primal exposes no outputs or replay data."
                )
        elif not self.realization_id:
            raise ValueError("An accepted external primal names its realization.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "replay_data", replay_data)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "replay_id",
            canonical_fingerprint(
                {
                    "kind": "external-primal-stage",
                    "action_id": self.action_id,
                    "inputs": [_array_record(value) for value in inputs],
                    "outputs": [_array_record(value) for value in outputs],
                    "realization_id": self.realization_id,
                    "replay_data": [
                        [name, _array_record(value)] for name, value in replay_data
                    ],
                    "evidence": [list(record) for record in evidence],
                    "failure_reason": self.failure_reason,
                }
            ),
        )

    @property
    def accepted(self) -> bool:
        """Whether the provider accepted the primal."""
        return not self.failure_reason

    def replay(self, name: str, /) -> np.ndarray:
        """Return one named replay array."""
        for key, value in self.replay_data:
            if key == name:
                return value
        raise KeyError(name)


class ExternalAdjointAction(ABC):
    """Staged host-boundary VJP of one external provider (no `pure_callback`).

    `stage_primal(*inputs)` evaluates the provider eagerly on concrete values
    checked against `input_schema` and returns an `ExternalPrimalStage`.
    Downstream JAX code differentiates with respect to the stage outputs, and
    `apply_adjoint(stage, *output_cotangents)` returns the input cotangents
    `Jᵀȳ` formed at exactly that staged realization; upstream JAX code continues
    the chain with its own VJP. Both calls are host-only and refuse every JAX
    transformation before reaching the provider.

    Providers implement `_primal(inputs)`, returning the stage built with this
    action's `action_id`, and `_adjoint(stage, output_cotangents)`, returning
    `(input_cotangents, realization_id)` where `realization_id` names the
    realization the adjoint was formed at. A realization that differs from the
    staged one is a replay mismatch and is refused; so is a stage of another
    action or a failed primal. `action_id` content-addresses the provider,
    version, schemas, and `configuration_id`.
    """

    provider: str
    version: str
    input_schema: tuple[ExternalTensorSpec, ...]
    output_schema: tuple[ExternalTensorSpec, ...]
    configuration_id: str
    action_id: str

    def __init__(
        self,
        *,
        provider: str,
        version: str,
        input_schema: Sequence[ExternalTensorSpec],
        output_schema: Sequence[ExternalTensorSpec],
        configuration_id: str,
    ):
        identifiers = {
            "provider": provider,
            "version": version,
            "configuration_id": configuration_id,
        }
        for name, value in identifiers.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        inputs = _tensor_schema(input_schema, "input_schema")
        outputs = _tensor_schema(output_schema, "output_schema")
        self.provider = provider
        self.version = version
        self.input_schema = inputs
        self.output_schema = outputs
        self.configuration_id = configuration_id
        self.action_id = canonical_fingerprint(
            {
                "kind": "external-adjoint-action",
                "provider": provider,
                "version": version,
                "input_schema": _schema_record(inputs),
                "output_schema": _schema_record(outputs),
                "configuration_id": configuration_id,
            }
        )

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Host-only `"external-adjoint"` execution capabilities."""
        return _EXTERNAL_ADJOINT_CAPABILITIES

    @property
    def derivative_support(self) -> ExternalDerivativeSupport:
        """Staged external adjoint; no derivative-free alternative is needed."""
        return ExternalDerivativeSupport("external-adjoint", ())

    @abstractmethod
    def _primal(self, inputs: tuple[np.ndarray, ...], /) -> ExternalPrimalStage:
        raise NotImplementedError

    @abstractmethod
    def _adjoint(
        self,
        stage: ExternalPrimalStage,
        output_cotangents: tuple[np.ndarray, ...],
        /,
    ) -> tuple[Sequence[Any], str]:
        raise NotImplementedError

    def stage_primal(self, *inputs: Any) -> ExternalPrimalStage:
        """Evaluate the provider once and stage the detached primal."""
        _require_execution(self.capabilities, inputs)
        if len(inputs) != len(self.input_schema):
            raise ValueError(
                f"{self.provider} expects {len(self.input_schema)} inputs; "
                f"got {len(inputs)}."
            )
        host = tuple(
            _detached(value, spec, "External input")
            for value, spec in zip(inputs, self.input_schema, strict=True)
        )
        stage = self._primal(host)
        if not isinstance(stage, ExternalPrimalStage):
            raise TypeError(f"{self.provider} did not return an ExternalPrimalStage.")
        if stage.action_id != self.action_id:
            raise ValueError(f"{self.provider} staged a primal of another action.")
        if len(stage.inputs) != len(host) or any(
            staged.dtype != given.dtype or not np.array_equal(staged, given)
            for staged, given in zip(stage.inputs, host, strict=True)
        ):
            raise ValueError(f"{self.provider} staged different inputs.")
        if stage.accepted:
            if len(stage.outputs) != len(self.output_schema):
                raise RuntimeError(
                    f"{self.provider} staged {len(stage.outputs)} outputs; the "
                    f"output schema declares {len(self.output_schema)}."
                )
            for output, spec in zip(stage.outputs, self.output_schema, strict=True):
                _require_finite(
                    spec.check(output, "External output"), "Output", spec.name
                )
        return stage

    def apply_adjoint(
        self, stage: ExternalPrimalStage, /, *output_cotangents: Any
    ) -> tuple[np.ndarray, ...]:
        """Return the input cotangents `Jᵀȳ` at the staged realization."""
        _require_execution(self.capabilities, output_cotangents)
        if not isinstance(stage, ExternalPrimalStage):
            raise TypeError("stage must be an ExternalPrimalStage.")
        if stage.action_id != self.action_id:
            raise ValueError(
                f"Replay mismatch: the stage belongs to another action, not this "
                f"{self.provider} action."
            )
        if not stage.accepted:
            raise ValueError(
                f"The staged {self.provider} primal was not accepted "
                f"({stage.failure_reason}); it has no adjoint."
            )
        if len(output_cotangents) != len(self.output_schema):
            raise ValueError(
                f"{self.provider} expects {len(self.output_schema)} output "
                f"cotangents; got {len(output_cotangents)}."
            )
        cotangents = tuple(
            _detached(value, spec, "Output cotangent")
            for value, spec in zip(output_cotangents, self.output_schema, strict=True)
        )
        result = self._adjoint(stage, cotangents)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"{self.provider} adjoint must return (input_cotangents, realization_id)."
            )
        values, realization_id = result
        if realization_id != stage.realization_id:
            raise ValueError(
                f"Replay mismatch: the {self.provider} adjoint was formed at "
                f"realization {realization_id!r}, not the staged "
                f"{stage.realization_id!r}."
            )
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{self.provider} adjoint must return a sequence.")
        if len(values) != len(self.input_schema):
            raise RuntimeError(
                f"{self.provider} adjoint returned {len(values)} cotangents; the "
                f"input schema declares {len(self.input_schema)}."
            )
        return tuple(
            _require_finite(
                _detached(value, spec, "Input cotangent"), "Input cotangent", spec.name
            )
            for value, spec in zip(values, self.input_schema, strict=True)
        )


__all__ = [
    "EnergyOutput",
    "EnergyRunResult",
    "EnergyRuntimeError",
    "ExternalAdjointAction",
    "ExternalDerivativeSupport",
    "ExternalExecutionPolicy",
    "ExternalIsolation",
    "ExternalPrimalStage",
    "ExternalTensorSpec",
    "OpenDSSRunResult",
    "PinnedExecutable",
    "pin_energy_executable",
    "run_energy_command",
    "run_energyplus",
    "run_opendss",
    "run_radiance_command",
]

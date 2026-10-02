#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only external execution: pinned engines, host inference, staged adjoints.

This is not a security sandbox.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, NoReturn, TYPE_CHECKING, TypeAlias

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ._external_exchange import read_exchange, write_exchange
from ._external_resource import read_bounded_resource, ResourceLimits
from ._external_worker import (
    _DEFAULT_BYTES,
    _ERROR_PACKET_BYTES,
    _receive_packet,
    _relative_path,
    _send_packet,
)
from ._fingerprint import canonical_fingerprint, canonical_json
from ._host_io import open_regular_beneath, open_regular_file
from ._identity import ArtifactBindingIdentity
from ._jax_context import inside_jax_transformation
from ._model._component import ExecutionCapabilities
from ._publication import publish_file
from .artifacts import ScientificArtifactEnvelope
from .backends._types import BackendUnavailableError
from .logging import emit
from .typing import checked, parse


def _host_only(*values: Any) -> None:
    # A zero-argument host operation inside jit must also be rejected, not just
    # calls whose arguments happen to contain a tracer.
    if inside_jax_transformation():
        raise TypeError(
            "External host operations cannot execute inside JAX transformations."
        )
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(values)
    ):
        raise TypeError("External host operations require concrete host values.")


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
    if inside_jax_transformation():
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


_ARTIFACT_COPY_BYTES = 1024 * 1024


def _limits(max_bytes: int) -> ResourceLimits:
    return ResourceLimits(
        max_bytes=max_bytes,
        max_depth=32,
        max_nodes=100000,
        max_attributes=100000,
        max_losses=0,
    )


ExternalIsolation: TypeAlias = Literal["trusted-local"]
ExternalEnforcement: TypeAlias = Literal[
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


def pin_executable(
    path: str | os.PathLike[str], *, version: str, license_id: str, source_url: str = ""
) -> PinnedExecutable:
    """Identify exact bytes of a caller-selected trusted-local executable."""
    _host_only()
    resolved = Path(path).expanduser().resolve(strict=True)
    with open_regular_file(resolved) as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return PinnedExecutable(str(resolved), digest, version, license_id, source_url)


@dataclass(frozen=True, slots=True)
class PinnedOutput:
    """One small declared output file detached into memory."""

    path: str
    data: bytes
    artifact: ScientificArtifactEnvelope


@dataclass(frozen=True, slots=True)
class PinnedFileRequest:
    """One declared file artifact: working-directory path and its byte cap."""

    path: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError("File artifact paths must be text.")
        path = _relative_path(self.path)
        if path.startswith(".phydrax-"):
            raise ValueError("The .phydrax- prefix is reserved for runtime evidence.")
        if type(self.maximum_bytes) is not int or self.maximum_bytes <= 0:
            raise ValueError("File artifact maximum_bytes must be a positive integer.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class PinnedFileOutputs:
    """Declared file artifacts published into one caller-owned directory.

    Files stay on disk, so provider output may far exceed ``max_output_bytes``;
    each declared path has its own cap and all of them share one total cap.
    """

    destination: str
    requests: tuple[PinnedFileRequest, ...]
    maximum_total_bytes: int

    def __post_init__(self) -> None:
        destination = Path(self.destination).expanduser().resolve(strict=True)
        if not destination.is_dir():
            raise ValueError("File artifact destination must be an existing directory.")
        requests = tuple(self.requests)
        if not requests or any(
            not isinstance(request, PinnedFileRequest) for request in requests
        ):
            raise TypeError("requests must be a nonempty sequence of PinnedFileRequest.")
        paths = [request.path for request in requests]
        if len(set(paths)) != len(paths):
            raise ValueError("Declared file artifact paths must be unique.")
        if type(self.maximum_total_bytes) is not int or self.maximum_total_bytes <= 0:
            raise ValueError("maximum_total_bytes must be a positive integer.")
        object.__setattr__(self, "destination", str(destination))
        object.__setattr__(
            self, "requests", tuple(sorted(requests, key=lambda item: item.path))
        )


@dataclass(frozen=True, slots=True)
class PinnedFileArtifact:
    """One published file artifact with its verified size and SHA-256 digest."""

    path: str
    location: str
    size_bytes: int
    sha256: str
    artifact: ScientificArtifactEnvelope


@dataclass(frozen=True, slots=True)
class PinnedRunResult:
    command: tuple[str, ...]
    returncode: int | None
    elapsed_seconds: float
    timed_out: bool
    stdout: bytes
    stderr: bytes
    outputs: tuple[PinnedOutput, ...]
    file_artifacts: tuple[PinnedFileArtifact, ...]
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

    def file_artifact(self, path: str) -> PinnedFileArtifact:
        for artifact in self.file_artifacts:
            if artifact.path == path:
                return artifact
        raise KeyError(path)

    def require_success(self) -> PinnedRunResult:
        if self.error or self.timed_out or self.returncode != 0:
            raise ExternalRuntimeError(
                self.error or "External command failed.", result=self
            )
        return self


class ExternalRuntimeError(RuntimeError):
    """Execution failure retaining bounded diagnostic and artifact evidence."""

    def __init__(
        self,
        message: str,
        *,
        result: PinnedRunResult | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
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
    # Darwin reports EPERM instead of ESRCH when only an unreaped zombie
    # remains in the group; wait() below reaps it.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
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


def _admit_file_artifacts(root: Path, declared: PinnedFileOutputs, /) -> str:
    """Return why declared artifacts are refused, or ``""`` before any publication."""
    total = 0
    for request in declared.requests:
        try:
            with open_regular_beneath(
                request.path, trusted_root=root, maximum_depth=32
            ) as opened:
                size = opened.file_status.st_size
        except FileNotFoundError:
            return f"File artifact {request.path!r} is missing."
        except (OSError, OverflowError, ValueError, RuntimeError) as failure:
            return f"File artifact {request.path!r}: {failure}"
        if size > request.maximum_bytes:
            return (
                f"File artifact {request.path!r} exceeds its "
                f"{request.maximum_bytes}-byte limit."
            )
        total += size
        if total > declared.maximum_total_bytes:
            return "File artifacts exceed maximum_total_bytes."
    return ""


def _publish_file_artifact(
    root: Path,
    destination: str,
    request: PinnedFileRequest,
    created: list[Path],
    /,
) -> tuple[int, str]:
    location = Path(destination, *PurePosixPath(request.path).parts)
    with open_regular_beneath(
        request.path, trusted_root=root, maximum_depth=32
    ) as opened:

        def copy(stream: BinaryIO) -> None:
            # The cap bounds bytes written, not only bytes admitted beforehand.
            remaining = request.maximum_bytes
            with opened.duplicate_stream() as source:
                source.seek(0)
                while block := source.read(min(_ARTIFACT_COPY_BYTES, remaining + 1)):
                    if len(block) > remaining:
                        raise ValueError(
                            f"File artifact {request.path!r} exceeds its "
                            f"{request.maximum_bytes}-byte limit."
                        )
                    remaining -= len(block)
                    stream.write(block)

        receipt = publish_file(
            location, copy, maximum_bytes=request.maximum_bytes, mode="exclusive"
        )
        created.append(location)
        if receipt.size_bytes != opened.file_status.st_size:
            raise RuntimeError(f"File artifact {request.path!r} changed while published.")
    return receipt.size_bytes, receipt.content_sha256


def _collect_file_artifacts(
    root: Path,
    declared: PinnedFileOutputs,
    executable: PinnedExecutable,
    resource_id: str,
    /,
) -> tuple[tuple[PinnedFileArtifact, ...], str]:
    """Publish every declared artifact, or none of them, with verified digests."""
    refusal = _admit_file_artifacts(root, declared)
    if refusal:
        return (), refusal
    created: list[Path] = []
    published: list[PinnedFileArtifact] = []
    try:
        for request in declared.requests:
            size, digest = _publish_file_artifact(
                root, declared.destination, request, created
            )
            envelope = ScientificArtifactEnvelope(
                artifact_kind="pinned-command-file-artifact",
                content_digest=digest,
                producer=Path(executable.path).name,
                producer_version=executable.version,
                build_id=executable.sha256,
                license_id=executable.license_id,
                resource_id=resource_id,
                status="complete",
            )
            published.append(
                PinnedFileArtifact(request.path, str(created[-1]), size, digest, envelope)
            )
    except (OSError, OverflowError, ValueError, RuntimeError) as failure:
        # A refused set never leaves a partial publication behind.
        for location in created:
            location.unlink(missing_ok=True)
        return (), f"File artifact publication failed: {failure}"
    return tuple(published), ""


def run_pinned_command(
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
    artifacts: PinnedFileOutputs | None = None,
) -> PinnedRunResult:
    """Execute trusted local argv after verifying a private executable snapshot.

    Linux executes through the held snapshot descriptor. Darwin executes scripts
    from the private snapshot; path-sensitive native binaries use their verified
    configured path under the declared trusted-local threat model.

    ``outputs`` are small files detached into memory under ``max_output_bytes``.
    ``artifacts`` declares on-disk file outputs: after a successful command, each
    declared file is checked against its own cap and the shared total cap, then
    exclusively published under the caller's destination with its SHA-256
    digest. A missing or oversize artifact refuses the whole set, and a failed
    command publishes nothing.
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
    if artifacts is not None and not isinstance(artifacts, PinnedFileOutputs):
        raise TypeError("artifacts must be PinnedFileOutputs or None.")
    artifact_requests = () if artifacts is None else artifacts.requests
    if set(output_names).intersection(request.path for request in artifact_requests):
        raise ValueError("A path cannot be both a detached output and a file artifact.")
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
    with tempfile.TemporaryDirectory(prefix="phydrax-pinned-") as directory:
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
                "file_artifacts": [
                    (request.path, request.maximum_bytes) for request in artifact_requests
                ],
                "file_artifact_total_bytes": (
                    0 if artifacts is None else artifacts.maximum_total_bytes
                ),
            }
        )
        emit(
            "DEBUG",
            "provider.execution.started",
            "Pinned provider execution started",
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
                    PinnedOutput(
                        name,
                        resource.data,
                        _artifact(
                            "pinned-command-output",
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
        file_artifacts: tuple[PinnedFileArtifact, ...] = ()
        if artifacts is not None and not error:
            file_artifacts, error = _collect_file_artifacts(
                root, artifacts, executable, resource_id
            )
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
            "file_artifacts": [
                (item.path, item.size_bytes, item.sha256) for item in file_artifacts
            ],
        }
        artifact = _artifact(
            "pinned-command-run",
            evidence,
            producer=Path(executable.path).name,
            version=executable.version,
            build_id=executable.sha256,
            license_id=executable.license_id,
            resource_id=resource_id,
            error=error,
        )
        result = PinnedRunResult(
            command,
            returncode,
            elapsed,
            timed_out,
            stdout,
            stderr,
            tuple(detached),
            file_artifacts,
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
        "Pinned provider execution finished",
        elapsed_seconds=result.elapsed_seconds,
        executable=Path(executable.path).name,
        output_artifact_count=len(result.outputs) + len(result.file_artifacts),
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
) -> PinnedRunResult:
    """Run a pinned EnergyPlus CLI with exact IDF/epJSON and EPW bytes."""
    if model_format not in ("idf", "epjson"):
        raise ValueError("model_format must be 'idf' or 'epjson'.")
    staged = dict(inputs or {})
    model_name = "model.idf" if model_format == "idf" else "model.epJSON"
    if model_name in staged or "weather.epw" in staged:
        raise ValueError("Additional inputs collide with the model/weather paths.")
    staged.update({model_name: model, "weather.epw": weather})
    requested = tuple(dict.fromkeys((*outputs, "eplusout.err")))
    result = run_pinned_command(
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
            raise ExternalRuntimeError(
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
) -> PinnedRunResult:
    """Run oconv/rtrace/rfluxmtx/etc.; explicitly pass prior-stage bytes, not a shell pipe."""
    return run_pinned_command(
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
    ) -> None:
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
                raise ExternalRuntimeError(response["error"], evidence=response)
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
            raise ExternalRuntimeError(str(failure), evidence=evidence) from failure

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
    outputs: tuple[PinnedOutput, ...]
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
                PinnedOutput(
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
            raise ExternalRuntimeError(
                error, evidence={"artifact_id": artifact.artifact_id, "data": data}
            )
        return result
    finally:
        worker.close()


# Persistent native workers ----------------------------------------------------------

# Every control line of a native worker carries this prefix followed by one
# canonical JSON object (native/providers/common/phydrax_worker.hpp). Other
# stdout lines are upstream library noise and are retained as log evidence.
_WORKER_PREFIX = b"@phydrax-worker "
_WORKER_REJECTIONS = (
    "invalid_request",
    "unsupported",
    "resource_exhausted",
    "library_failure",
)
# Rejections after which the native library state is not trusted for reuse.
_WORKER_FATAL_REJECTIONS = ("resource_exhausted", "library_failure")

NativeWorkerFailureKind: TypeAlias = Literal[
    "unavailable", "startup", "timeout", "resource", "protocol", "exited", "rejected"
]


class NativeWorkerError(RuntimeError):
    """Failure of one persistent native worker, retaining bounded evidence.

    ``kind`` is the runtime failure class; for ``"rejected"`` the worker's own
    rejection class is ``evidence["worker_kind"]``.
    """

    def __init__(
        self, kind: NativeWorkerFailureKind, message: str, /, *, evidence: Mapping
    ) -> None:
        self.kind = kind
        self.evidence = dict(evidence)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class NativeWorkerPolicy:
    """Lifetime and resource bounds of one persistent native worker session.

    Memory is enforced as an address-space limit where the host honors one
    (Linux) and otherwise audited against the collective peak resident size the
    worker reports after each call; the session records which applies.
    """

    startup_timeout_seconds: float = 60.0
    maximum_memory_bytes: int = 16 * 1024**3
    maximum_log_bytes: int = 16 * 1024**2
    maximum_message_bytes: int = 1024**2
    maximum_calls: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "startup_timeout_seconds",
            _positive_timeout(self.startup_timeout_seconds),
        )
        for name in (
            "maximum_memory_bytes",
            "maximum_log_bytes",
            "maximum_message_bytes",
            "maximum_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def policy_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "native-worker-policy",
                "startup_timeout_seconds": self.startup_timeout_seconds,
                "maximum_memory_bytes": self.maximum_memory_bytes,
                "maximum_log_bytes": self.maximum_log_bytes,
                "maximum_message_bytes": self.maximum_message_bytes,
                "maximum_calls": self.maximum_calls,
            }
        )


@dataclass(frozen=True, slots=True)
class NativeWorkerIdentity:
    """Exact runtime identity, probed once when the worker session starts."""

    executable: str
    executable_sha256: str
    command: tuple[str, ...]
    reported: Mapping[str, Any]
    ranks: int
    memory_enforcement: str
    identity_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class NativeWorkerCall:
    """One completed worker operation with verified read-only output arrays."""

    operation: str
    sequence: int
    result: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    parts: Mapping[str, Mapping[str, np.ndarray]]
    evidence: Mapping[str, Any]


def _terminate_worker(process: subprocess.Popen, logs: BinaryIO, root: str) -> None:
    _kill_process_group(process)
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            stream.close()
    logs.close()
    shutil.rmtree(root, ignore_errors=True)


def _worker_environment(
    policy: ExternalExecutionPolicy,
    overrides: Mapping[str, str],
    root: str,
    memory: int,
) -> dict[str, str]:
    disallowed = sorted(set(overrides).difference(policy.allowed_environment_variables))
    if disallowed and not policy.inherit_environment:
        raise ValueError(
            "Environment overrides exceed the execution allowlist: "
            + ", ".join(disallowed)
        )
    environment = (
        dict(os.environ)
        if policy.inherit_environment
        else {
            key: os.environ[key]
            for key in policy.allowed_environment_variables
            if key in os.environ
        }
    )
    environment.update({"HOME": root, "TMPDIR": root, "TMP": root, "TEMP": root})
    environment.update(overrides)
    environment["PHYDRAX_WORKER_MEMORY_LIMIT_BYTES"] = str(memory)
    return environment


class NativeWorker:
    """Persistent trusted-local native worker speaking the exchange protocol.

    The executable bytes are digested and the worker's self-reported identity is
    recorded once at startup; calls reuse the same process until ``close``, a
    fatal failure, or ``NativeWorkerPolicy.maximum_calls``. A launcher such as
    ``("mpiexec", "-n", "4")`` runs a collective worker whose rank zero owns the
    control channel. Calls, ``close``, and ``abort`` are serialized by one
    reentrant lock, so a concurrent close waits for the call in flight. This is
    not a security sandbox.
    """

    def __init__(
        self,
        executable: str | os.PathLike[str],
        /,
        *,
        launcher: Sequence[str] = (),
        arguments: Sequence[str] = (),
        policy: NativeWorkerPolicy | None = None,
        environment: Mapping[str, str] | None = None,
        execution_policy: ExternalExecutionPolicy | None = None,
    ) -> None:
        _host_only()
        if os.name != "posix":
            raise OSError("Persistent native workers require a POSIX host.")
        self.policy = NativeWorkerPolicy() if policy is None else policy
        if not isinstance(self.policy, NativeWorkerPolicy):
            raise TypeError("policy must be NativeWorkerPolicy or None.")
        execution = (
            ExternalExecutionPolicy() if execution_policy is None else execution_policy
        )
        if not isinstance(execution, ExternalExecutionPolicy):
            raise TypeError("execution_policy must be ExternalExecutionPolicy or None.")
        launch = tuple(str(value) for value in launcher)
        extra = tuple(str(value) for value in arguments)
        if any(not value or "\x00" in value for value in (*launch, *extra)):
            raise ValueError(
                "Worker launcher and arguments must be nonempty argv tokens."
            )
        path = Path(executable).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise NativeWorkerError(
                "unavailable",
                f"Worker executable is unavailable: {executable}",
                evidence={"executable": str(executable)},
            )
        resolved = str(path.resolve(strict=True))
        if launch:
            located = shutil.which(launch[0])
            if located is None:
                raise NativeWorkerError(
                    "unavailable",
                    f"Worker launcher is unavailable: {launch[0]}",
                    evidence={"launcher": list(launch)},
                )
            launch = (str(Path(located).resolve()), *launch[1:])
        with open_regular_file(resolved) as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self._lock = threading.RLock()
        self.closed = False
        self.final_log = ""
        self.calls: list[dict[str, Any]] = []
        self._sequence = 0
        self._buffer = bytearray()
        self._root = tempfile.mkdtemp(prefix="phydrax-worker-")
        self._logs = (Path(self._root) / ".phydrax-worker.log").open("w+b")
        command = (*launch, resolved, *extra)
        # Evidence of the operation in flight, merged into every failure record.
        self._context: dict[str, Any] = {"command": list(command), "stage": "startup"}
        environment_ = _worker_environment(
            execution,
            dict(environment or {}),
            self._root,
            self.policy.maximum_memory_bytes,
        )
        # Popen reports exec failures only by raising; release the session first.
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self._root,
                env=environment_,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._logs,
                start_new_session=True,
            )
        except OSError as error:
            self._logs.close()
            shutil.rmtree(self._root, ignore_errors=True)
            raise NativeWorkerError(
                "unavailable",
                f"Worker could not be launched: {error}",
                evidence={"command": list(command)},
            ) from error
        self._finalizer = weakref.finalize(
            self, _terminate_worker, self._process, self._logs, self._root
        )
        started = time.monotonic()
        hello = self._receive(started + self.policy.startup_timeout_seconds, "startup")
        record = hello.get("hello")
        if (
            set(hello) != {"hello", "ok"}
            or hello["ok"] is not True
            or not isinstance(record, dict)
            or set(record) != {"identity", "memory_enforcement", "ranks"}
            or not isinstance(record["identity"], dict)
            or type(record["ranks"]) is not int
            or record["ranks"] < 1
            or record["memory_enforcement"]
            not in ("address-space-rlimit", "peak-rss-audit")
        ):
            self._fail("protocol", "Worker sent an invalid hello record.", {})
        identity_id = canonical_fingerprint(
            {
                "kind": "native-worker-identity",
                "executable_sha256": digest,
                "reported": record["identity"],
                "ranks": record["ranks"],
            }
        )
        self.identity = NativeWorkerIdentity(
            resolved,
            digest,
            command,
            record["identity"],
            record["ranks"],
            record["memory_enforcement"],
            identity_id,
            canonical_fingerprint(
                {
                    "kind": "native-worker-session",
                    "identity": identity_id,
                    "command": list(command),
                    "policy": self.policy.policy_id,
                    "execution_policy": execution.policy_id,
                    "process": self._process.pid,
                    "started_ns": time.time_ns(),
                }
            ),
        )
        emit(
            "DEBUG",
            "provider.worker.started",
            "Native provider worker started",
            executable=Path(resolved).name,
            ranks=record["ranks"],
            session_id=self.identity.session_id,
            elapsed_seconds=time.monotonic() - started,
        )

    @property
    def call_count(self) -> int:
        return self._sequence

    @property
    def exhausted(self) -> bool:
        """Whether the session reached its bounded call lifetime."""
        return self._sequence >= self.policy.maximum_calls

    def _log_tail(self) -> str:
        self._logs.flush()
        size = os.fstat(self._logs.fileno()).st_size
        self._logs.seek(max(0, size - 16384))
        tail = self._logs.read(16384).decode("utf-8", errors="replace")
        self._logs.seek(0, os.SEEK_END)
        return tail

    def _fail(
        self, kind: NativeWorkerFailureKind, message: str, evidence: Mapping[str, Any]
    ) -> NoReturn:
        self.abort()
        record = {
            **self._context,
            **evidence,
            "kind": kind,
            "returncode": self._process.returncode,
            "log": self.final_log,
        }
        raise NativeWorkerError(kind, message, evidence=record)

    def _receive(self, deadline: float, stage: str) -> dict[str, Any]:
        stdout = self._process.stdout
        assert stdout is not None
        descriptor = stdout.fileno()
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if line.startswith(_WORKER_PREFIX):
                    return self._decode(line[len(_WORKER_PREFIX) :], stage)
                self._logs.write(line + b"\n")
                continue
            if len(self._buffer) > self.policy.maximum_message_bytes:
                self._fail("resource", "Worker control line exceeds its byte bound.", {})
            if os.fstat(self._logs.fileno()).st_size > self.policy.maximum_log_bytes:
                self._fail("resource", "Worker logs exceed maximum_log_bytes.", {})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail(
                    "timeout",
                    f"Worker {stage} exceeded its deadline.",
                    {"stage": stage},
                )
            ready, _, _ = select.select([descriptor], [], [], min(remaining, 0.05))
            if ready:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    self._process.wait()
                    self._fail(
                        "startup" if stage == "startup" else "exited",
                        f"Worker exited during {stage} with status "
                        f"{self._process.returncode}.",
                        {"stage": stage},
                    )
                self._buffer.extend(chunk)

    def _decode(self, data: bytes, stage: str) -> dict[str, Any]:
        def pairs(items: Any) -> Any:
            record = {}
            for key, value in items:
                if key in record:
                    raise ValueError(f"duplicate field {key!r}")
                record[key] = value
            return record

        # A malformed control line is a protocol failure, never a partial result.
        try:
            value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
        except (UnicodeDecodeError, ValueError) as error:
            self._fail("protocol", f"Worker sent malformed control JSON: {error}", {})
        if not isinstance(value, dict):
            self._fail("protocol", f"Worker {stage} record is not an object.", {})
        return value

    def _send(self, request: Mapping[str, Any], stage: str) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        line = canonical_json(request).encode("ascii") + b"\n"
        if len(line) > self.policy.maximum_message_bytes:
            raise ValueError("Worker request exceeds maximum_message_bytes.")
        # A worker that died closes its pipe; report that as a worker exit.
        try:
            stdin.write(line)
            stdin.flush()
        except BrokenPipeError:
            self._process.wait()
            self._fail(
                "exited",
                f"Worker exited before {stage} with status {self._process.returncode}.",
                {"stage": stage},
            )

    def call(
        self,
        operation: str,
        parameters: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        /,
        *,
        timeout: float,
        maximum_input_bytes: int,
        maximum_output_bytes: int,
    ) -> NativeWorkerCall:
        """Run one operation; any failure raises NativeWorkerError with evidence."""
        _host_only(parameters, arrays)
        timeout = _positive_timeout(timeout)
        if not isinstance(operation, str) or not operation or operation == "close":
            raise ValueError("operation must be a nonempty worker operation name.")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a JSON mapping.")
        if not isinstance(arrays, Mapping) or any(
            not isinstance(value, np.ndarray) for value in arrays.values()
        ):
            raise TypeError("arrays must map exchange names to NumPy arrays.")
        if sum(value.nbytes for value in arrays.values()) > maximum_input_bytes:
            raise ValueError("Worker input arrays exceed maximum_input_bytes.")
        with self._lock:
            if self.closed:
                raise NativeWorkerError(
                    "exited",
                    "Worker session is closed.",
                    evidence={"operation": operation},
                )
            if self.exhausted:
                raise NativeWorkerError(
                    "resource",
                    "Worker session reached maximum_calls.",
                    evidence={"operation": operation, "calls": self._sequence},
                )
            return self._exchange(
                operation,
                parameters,
                arrays,
                timeout,
                maximum_input_bytes,
                maximum_output_bytes,
            )

    def _exchange(
        self,
        operation: str,
        parameters: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        timeout: float,
        maximum_input_bytes: int,
        maximum_output_bytes: int,
    ) -> NativeWorkerCall:
        # Caller holds `_lock`: sequence, pipes, and staging belong to one call.
        self._sequence += 1
        sequence = self._sequence
        started = time.monotonic()
        directory = Path(self._root) / f"call-{sequence}"
        input_directory, output_directory = directory / "input", directory / "output"
        input_directory.mkdir(parents=True)
        output_directory.mkdir()
        inputs = write_exchange(
            input_directory, arrays, maximum_bytes=maximum_input_bytes
        )
        request = {
            "input": str(input_directory),
            "maximum_input_bytes": maximum_input_bytes,
            "maximum_output_bytes": maximum_output_bytes,
            "operation": operation,
            "output": str(output_directory),
            "parameters": dict(parameters),
            "sequence": sequence,
        }
        evidence: dict[str, Any] = {
            "operation": operation,
            "sequence": sequence,
            "session_id": self.identity.session_id,
            "input_manifest_sha256": inputs.manifest_sha256,
            "parameters_id": canonical_fingerprint(dict(parameters)),
        }
        self._context = evidence
        self._send(request, operation)
        response = self._receive(started + timeout, operation)
        peak = response.get("peak_rss_bytes")
        if response.get("sequence") != sequence or type(peak) is not int:
            self._fail(
                "protocol", "Worker response does not match its request.", evidence
            )
        evidence.update(peak_rss_bytes=peak, elapsed_seconds=time.monotonic() - started)
        if response.get("ok") is not True:
            if (
                set(response) != {"error", "kind", "ok", "peak_rss_bytes", "sequence"}
                or response["kind"] not in _WORKER_REJECTIONS
                or not isinstance(response["error"], str)
            ):
                self._fail("protocol", "Worker sent an invalid rejection.", evidence)
            evidence.update(worker_kind=response["kind"], error=response["error"])
            self.calls.append({**evidence, "status": "rejected"})
            if response["kind"] in _WORKER_FATAL_REJECTIONS:
                self._fail("rejected", response["error"], evidence)
            shutil.rmtree(directory)
            raise NativeWorkerError("rejected", response["error"], evidence=evidence)
        if set(response) != {
            "elapsed_seconds",
            "ok",
            "peak_rss_bytes",
            "result",
            "sequence",
        }:
            self._fail("protocol", "Worker sent an invalid response.", evidence)
        if peak > self.policy.maximum_memory_bytes:
            self._fail("resource", "Worker peak memory exceeds its limit.", evidence)
        # Unverifiable output is a protocol failure, never a partial result.
        try:
            contents = read_exchange(output_directory, maximum_bytes=maximum_output_bytes)
        except (OSError, ValueError) as error:
            self._fail("protocol", f"Worker output is invalid: {error}", evidence)
        # Unlinking keeps the verified memory maps valid on POSIX hosts.
        shutil.rmtree(directory)
        evidence.update(
            output_manifest_sha256=contents.manifest.manifest_sha256,
            output_bytes=contents.manifest.total_bytes,
            worker_elapsed_seconds=response["elapsed_seconds"],
        )
        self.calls.append({**evidence, "status": "complete"})
        return NativeWorkerCall(
            operation,
            sequence,
            response["result"],
            contents.arrays,
            contents.parts,
            evidence,
        )

    def close(self) -> None:
        """End the session gracefully when possible, then release every resource."""
        with self._lock:
            if self.closed:
                return
            stdin = self._process.stdin
            if self._process.poll() is None and stdin is not None:
                self._sequence += 1
                line = canonical_json({"operation": "close", "sequence": self._sequence})
                # Shutdown of an already-dead worker is not an error to surface.
                try:
                    stdin.write(line.encode("ascii") + b"\n")
                    stdin.close()
                    self._process.wait(timeout=5.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    pass
            self.abort()

    def abort(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self._logs.flush()
            tail = self._log_tail()
            self._finalizer.detach()
            _kill_process_group(self._process)
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()
            self.final_log = tail
            self._logs.close()
            shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> NativeWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# External model tiers ---------------------------------------------------------------

ExternalTransport: TypeAlias = Literal["copy", "dlpack"]
ExternalDerivativeRoute: TypeAlias = Literal["external-adjoint", "none"]

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

    if TYPE_CHECKING:
        # __post_init__ normalizes shape to a tuple and dtype to `numpy.dtype.str`.
        def __init__(
            self, name: str, shape: Sequence[int | np.integer], dtype: DTypeLike
        ) -> None: ...

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
        object.__setattr__(
            self, "transport", parse(self.transport, ExternalTransport, "transport")
        )

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
    ) -> None:
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

    @checked
    def apply_adjoint(
        self, stage: ExternalPrimalStage, /, *output_cotangents: Any
    ) -> tuple[np.ndarray, ...]:
        """Return the input cotangents `Jᵀȳ` at the staged realization."""
        _require_execution(self.capabilities, output_cotangents)
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
    "ExternalAdjointAction",
    "ExternalDerivativeSupport",
    "ExternalExecutionPolicy",
    "ExternalIsolation",
    "ExternalPrimalStage",
    "ExternalRuntimeError",
    "ExternalTensorSpec",
    "NativeWorker",
    "NativeWorkerCall",
    "NativeWorkerError",
    "NativeWorkerFailureKind",
    "NativeWorkerIdentity",
    "NativeWorkerPolicy",
    "OpenDSSRunResult",
    "PinnedExecutable",
    "PinnedFileArtifact",
    "PinnedFileOutputs",
    "PinnedFileRequest",
    "PinnedOutput",
    "PinnedRunResult",
    "pin_executable",
    "run_energyplus",
    "run_opendss",
    "run_pinned_command",
    "run_radiance_command",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Audited host-only subprocess boundaries for medical-image tooling."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import monotonic

from .._fingerprint import canonical_fingerprint
from ..artifacts import ScientificArtifactEnvelope
from ..logging import emit
from ..qualification import ReferenceArtifactManifest


def _executable(value: str, /) -> str:
    path = shutil.which(value)
    if path is None:
        raise FileNotFoundError(f"Medical-image tool executable is unavailable: {value}")
    return str(Path(path).resolve())


def _digest(path: Path, /) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MedicalToolResult:
    provider_id: str
    provider_version: str
    output_paths: tuple[Path, ...]
    artifacts: tuple[ScientificArtifactEnvelope, ...]
    elapsed_seconds: float
    stdout: str
    stderr: str
    result_id: str


@dataclass(frozen=True, slots=True)
class MedicalToolProvider:
    provider_id: str
    executable: str
    version_arguments: tuple[str, ...]
    license_id: str
    build_id: str = "external"
    provider_identity: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            (self.provider_id, "provider_id"),
            (self.executable, "executable"),
            (self.license_id, "license_id"),
            (self.build_id, "build_id"),
        )
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value, _ in values
        ):
            invalid = next(
                name
                for value, name in values
                if not isinstance(value, str) or not value or value != value.strip()
            )
            raise ValueError(f"{invalid} must be a canonical non-empty string.")
        arguments = tuple(str(value) for value in self.version_arguments)
        if any(not value for value in arguments):
            raise ValueError("version_arguments must contain non-empty arguments.")
        object.__setattr__(self, "version_arguments", arguments)
        object.__setattr__(
            self,
            "provider_identity",
            canonical_fingerprint(
                {
                    "kind": "medical-tool-provider",
                    "provider": self.provider_id,
                    "executable": self.executable,
                    "version_arguments": list(arguments),
                    "license": self.license_id,
                    "build": self.build_id,
                }
            ),
        )

    def version(self) -> str:
        executable = _executable(self.executable)
        result = subprocess.run(
            [executable, *self.version_arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        if result.returncode:
            raise RuntimeError(
                f"{self.provider_id} version command failed with code {result.returncode}."
            )
        version = (result.stdout or result.stderr).strip().splitlines()[0]
        if not version:
            raise RuntimeError(f"{self.provider_id} returned no version identity.")
        return version

    def execute(
        self,
        arguments: tuple[str, ...],
        expected_outputs: tuple[str | Path, ...],
        input_manifests: tuple[ReferenceArtifactManifest, ...],
        /,
        *,
        working_directory: str | Path,
        output_license_id: str,
        timeout_seconds: float,
        commercial_use: bool = False,
        training_use: bool = False,
        export: bool = False,
        retain_logs: bool = False,
    ) -> MedicalToolResult:
        if not input_manifests or any(
            not isinstance(value, ReferenceArtifactManifest) for value in input_manifests
        ):
            raise ValueError(
                "input_manifests must contain ReferenceArtifactManifest values."
            )
        for manifest in input_manifests:
            manifest.require_rights(
                commercial_use=commercial_use,
                training_use=training_use,
                export=export,
            )
        if not isinstance(retain_logs, bool):
            raise TypeError("retain_logs must be boolean.")
        timeout = float(timeout_seconds)
        if timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        directory = Path(working_directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        output_paths = tuple(
            (Path(value) if Path(value).is_absolute() else directory / value).resolve()
            for value in expected_outputs
        )
        if not output_paths or len(set(output_paths)) != len(output_paths):
            raise ValueError("expected_outputs must be unique and non-empty.")
        executable = _executable(self.executable)
        version = self.version()
        command = [executable, *(str(value) for value in arguments)]
        started = monotonic()
        emit(
            "DEBUG",
            "provider.execution.started",
            "Medical-image provider execution started",
            input_artifact_count=len(input_manifests),
            output_count=len(output_paths),
            provider=self.provider_id,
        )
        result = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        elapsed = monotonic() - started
        if result.returncode:
            emit(
                "ERROR",
                "provider.execution.failed",
                "Medical-image provider execution failed",
                elapsed_seconds=elapsed,
                failure_category="nonzero_exit",
                provider=self.provider_id,
                return_code=result.returncode,
                stderr_bytes=len(result.stderr.encode("utf-8")),
                stdout_bytes=len(result.stdout.encode("utf-8")),
            )
            raise RuntimeError(
                f"{self.provider_id} failed with code {result.returncode}; "
                "inspect logs only inside the controlled working directory."
            )
        if any(not path.is_file() or path.stat().st_size == 0 for path in output_paths):
            emit(
                "ERROR",
                "provider.execution.failed",
                "Medical-image provider output validation failed",
                elapsed_seconds=elapsed,
                failure_category="missing_output",
                provider=self.provider_id,
                return_code=result.returncode,
                stderr_bytes=len(result.stderr.encode("utf-8")),
                stdout_bytes=len(result.stdout.encode("utf-8")),
            )
            raise RuntimeError(
                f"{self.provider_id} did not produce every declared output."
            )
        parents = tuple(value.manifest_id for value in input_manifests)
        artifacts = tuple(
            ScientificArtifactEnvelope(
                artifact_kind="medical-image-tool-output",
                content_digest=_digest(path),
                producer=self.provider_id,
                producer_version=version,
                build_id=self.build_id,
                license_id=output_license_id,
                resource_id=f"{self.provider_id}:output:{index}",
                status="complete",
                parent_artifact_ids=parents,
            )
            for index, path in enumerate(output_paths)
        )
        result_id = canonical_fingerprint(
            {
                "kind": "medical-tool-result",
                "provider": self.provider_identity,
                "version": version,
                "arguments": list(arguments),
                "outputs": [value.artifact_id for value in artifacts],
                "parents": list(parents),
            }
        )
        provider_result = MedicalToolResult(
            self.provider_id,
            version,
            output_paths,
            artifacts,
            elapsed,
            result.stdout if retain_logs else "",
            result.stderr if retain_logs else "",
            result_id,
        )
        emit(
            "INFO",
            "provider.execution.completed",
            "Medical-image provider execution completed",
            elapsed_seconds=elapsed,
            output_artifact_ids=tuple(value.artifact_id for value in artifacts),
            provider=self.provider_id,
            result_id=result_id,
            return_code=result.returncode,
            stderr_bytes=len(result.stderr.encode("utf-8")),
            stdout_bytes=len(result.stdout.encode("utf-8")),
        )
        return provider_result


class Dcm2NiixProvider(MedicalToolProvider):
    def __init__(self, executable: str = "dcm2niix", /):
        super().__init__("dcm2niix", executable, ("--version",), "BSD-3-Clause")


class GreedyRegistrationProvider(MedicalToolProvider):
    def __init__(self, executable: str = "greedy", /):
        super().__init__("greedy", executable, ("-version",), "GPL-3.0-or-later")


class ANTsRegistrationProvider(MedicalToolProvider):
    def __init__(self, executable: str = "antsRegistration", /):
        super().__init__("ants-registration", executable, ("--version",), "Apache-2.0")


class FreeSurferProvider(MedicalToolProvider):
    def __init__(self, executable: str = "recon-all", /):
        super().__init__(
            "freesurfer", executable, ("-version",), "FreeSurfer-Software-License"
        )


class FastSurferProvider(MedicalToolProvider):
    def __init__(self, executable: str = "run_fastsurfer.sh", /):
        super().__init__("fastsurfer", executable, ("--version",), "Apache-2.0")


class SynthSegProvider(MedicalToolProvider):
    def __init__(self, executable: str = "mri_synthseg", /):
        super().__init__("synthseg", executable, ("--version",), "Apache-2.0")


__all__ = [
    "ANTsRegistrationProvider",
    "Dcm2NiixProvider",
    "FastSurferProvider",
    "FreeSurferProvider",
    "GreedyRegistrationProvider",
    "MedicalToolProvider",
    "MedicalToolResult",
    "SynthSegProvider",
]

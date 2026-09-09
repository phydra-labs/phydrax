#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Private host-only source pinning and evidence for device engine adapters."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .._fingerprint import canonical_fingerprint
from ..artifacts import ScientificArtifactEnvelope
from ._resource import read_bounded_resource, ResourceLimits
from .energy_runtime import _artifact, _host_only, EnergyRunResult


def _source_path(path: object) -> str:
    if (
        not isinstance(path, str)
        or not path
        or "\x00" in path
        or "\\" in path
        or any(ord(character) < 32 for character in path)
    ):
        raise ValueError("Device source paths must be nonempty POSIX text paths.")
    parsed = PurePosixPath(path)
    parts = parsed.parts
    if (
        parsed.is_absolute()
        or path.startswith(":")
        or any(part in ("", ".", "..") for part in parts)
        or path != "/".join(parts)
    ):
        raise ValueError("Unsafe device source selector.")
    return path


@dataclass(frozen=True, slots=True)
class DeviceSource:
    """Local upstream Git checkout; its selected files must match ``commit`` exactly.

    This does not download or grant redistribution rights. License declarations
    are caller evidence, not a license check. Dirty selected files, symlinks and
    submodules are rejected. Execution uses detached, verified bytes, not imports
    from the caller's mutable checkout.
    """

    directory: str
    commit: str
    license_id: str

    def __post_init__(self) -> None:
        _host_only()
        if re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise ValueError("commit must be a full lowercase Git SHA-1.")
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise ValueError(
                "An explicit source license/permission identifier is required."
            )
        if not isinstance(self.directory, str) or "\x00" in self.directory:
            raise ValueError("Source directory must be a local text path.")
        root = Path(self.directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Source directory must be a local Git checkout.")
        object.__setattr__(self, "directory", str(root))

    def snapshot(self, paths: Sequence[str], *, max_bytes: int) -> dict[str, bytes]:
        _host_only()
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Device source byte limit must be a positive integer.")
        if isinstance(paths, (str, bytes)) or not paths or len(paths) > 100000:
            raise ValueError("Select a finite nonempty set of device source paths.")
        selected: list[str] = []
        for path in paths:
            selected.append(_source_path(path))
        if len(selected) != len(set(selected)):
            raise ValueError("Device source selectors must be unique.")
        root = Path(self.directory)
        resolved = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "rev-parse",
                    "--verify",
                    self.commit + "^{commit}",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            .stdout.decode()
            .strip()
        )
        if resolved != self.commit:
            raise ValueError("Source commit identity mismatch.")
        tree = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "-z", self.commit, "--", *selected],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        files: dict[str, bytes] = {}
        remaining = max_bytes
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, name = record.split(b"\t", 1)
            mode, kind, digest = metadata.decode().split()
            if kind != "blob" or mode not in ("100644", "100755"):
                raise ValueError("Device sources cannot contain symlinks or submodules.")
            path = _source_path(name.decode("utf-8"))
            resource = read_bounded_resource(
                path,
                trusted_root=root,
                limits=ResourceLimits(
                    max_bytes=max(1, remaining),
                    max_depth=32,
                    max_nodes=100000,
                    max_attributes=100000,
                    max_losses=0,
                ),
            )
            data = resource.data
            remaining -= len(data)
            if remaining < 0:
                raise ValueError("Device source exceeds the input byte limit.")
            git_digest = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
            if git_digest != digest:
                raise ValueError(f"Source file differs from pinned commit: {path}")
            files[path] = data
        missing = [
            selector
            for selector in selected
            if not any(
                path == selector or path.startswith(selector + "/") for path in files
            )
        ]
        if missing:
            raise ValueError(
                f"Selected source paths are absent from the pinned commit: {missing}"
            )
        return files


class DeviceQualificationError(ValueError):
    """Unusable scientific output, retaining the successful or failed engine runs.

    No numerical objective, noise estimate or training penalty is synthesized.
    """

    def __init__(self, message: str, *, runs: Sequence[EnergyRunResult] = ()):
        self.runs = tuple(runs)
        self.artifact = _artifact(
            "device-design-qualification",
            {"error": message},
            producer="phydrax",
            version="native",
            build_id="device-design",
            license_id="LicenseRef-PHYDRA",
            resource_id="none",
            error=message,
            parents=tuple(run.artifact.artifact_id for run in self.runs),
        )
        super().__init__(message)


def device_artifact(
    kind: str,
    payload: object,
    source: DeviceSource,
    runs: Sequence[EnergyRunResult],
    *,
    error: str = "",
) -> ScientificArtifactEnvelope:
    return _artifact(
        kind,
        payload,
        producer="phydrax",
        version="native",
        build_id=source.commit,
        license_id=source.license_id,
        error=error,
        resource_id=canonical_fingerprint({"commit": source.commit, "payload": payload}),
        parents=tuple(run.artifact.artifact_id for run in runs),
    )

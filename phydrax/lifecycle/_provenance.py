#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic source/build provenance and SPDX 2.3 software bills of materials."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from .._host_io import open_regular_beneath
from ..logging import emit


def _canonical_json(value: object, /) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")
    return value


@dataclass(frozen=True, slots=True, order=True)
class InstalledPackage:
    name: str
    version: str
    license_expression: str = "NOASSERTION"
    supplier: str = "NOASSERTION"

    def __post_init__(self) -> None:
        _identifier(self.name, "Package name")
        _identifier(self.version, "Package version")
        _identifier(self.license_expression, "Package license")
        _identifier(self.supplier, "Package supplier")


class DistributionLike(Protocol):
    @property
    def metadata(self) -> Mapping[str, str]: ...
    @property
    def version(self) -> str: ...


def installed_packages(
    distributions: Iterable[DistributionLike] | None = None,
    /,
) -> tuple[InstalledPackage, ...]:
    """Snapshot installed metadata. No package indexes or network services are queried."""

    source = (
        importlib.metadata.distributions() if distributions is None else distributions
    )
    by_identity: dict[tuple[str, str], InstalledPackage] = {}
    for distribution in source:
        metadata = distribution.metadata
        name = metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        declared_expression = metadata.get("License-Expression")
        license_expression = (
            " ".join(declared_expression.split())
            if declared_expression and declared_expression.strip()
            else _spdx_license(metadata.get("License")) or "NOASSERTION"
        )
        author = metadata.get("Author") or metadata.get("Author-email")
        supplier = f"Organization: {author}" if author else "NOASSERTION"
        package = InstalledPackage(name, version, license_expression, supplier)
        identity = (name.casefold().replace("_", "-"), version)
        previous = by_identity.get(identity)
        if previous is not None and previous != package:
            # Metadata ambiguity is preserved deterministically rather than order-selected.
            package = min(previous, package)
        by_identity[identity] = package
    return tuple(by_identity[key] for key in sorted(by_identity))


def _spdx_license(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    aliases = {
        "Apache 2": "Apache-2.0",
        "Apache License 2.0": "Apache-2.0",
        "BSD": "BSD-3-Clause",
        "BSD License": "BSD-3-Clause",
        "MIT": "MIT",
        "MIT License": "MIT",
        "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    }
    return aliases.get(cleaned, "NOASSERTION")


def digest_paths(
    root: str | Path,
    paths: Sequence[str | Path],
    /,
    *,
    maximum_total_bytes: int = 1_073_741_824,
) -> str:
    """Hash names, modes, and bytes through held descriptor-relative paths."""

    if type(maximum_total_bytes) is not int or maximum_total_bytes < 0:
        raise ValueError("maximum_total_bytes must be a non-negative integer.")
    root_path = Path(root).resolve()
    normalized: list[tuple[str, Path]] = []
    for item in paths:
        raw = root_path / item if not Path(item).is_absolute() else Path(item)
        absolute = raw.absolute()
        try:
            relative_path = absolute.relative_to(root_path)
        except ValueError as error:
            raise ValueError("Provenance paths must remain beneath root.") from error
        if not relative_path.parts or any(
            component in ("", ".", "..") for component in relative_path.parts
        ):
            raise ValueError("Provenance paths must be canonical relative files.")
        normalized.append((relative_path.as_posix(), relative_path))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("Provenance paths must be unique.")

    digest = hashlib.sha256()
    remaining = maximum_total_bytes
    for relative, relative_path in sorted(normalized, key=lambda item: item[0]):
        try:
            with open_regular_beneath(
                relative_path,
                trusted_root=root_path,
                maximum_depth=max(1, len(relative_path.parts)),
            ) as opened:
                before = opened.file_status
                if before.st_size > remaining:
                    raise ValueError("Provenance paths exceed maximum_total_bytes.")
                name = relative.encode("utf-8")
                digest.update(len(name).to_bytes(8, "big"))
                digest.update(name)
                digest.update((before.st_mode & 0o111).to_bytes(2, "big"))
                digest.update(before.st_size.to_bytes(8, "big"))
                observed = 0
                while True:
                    block = os.read(
                        opened.descriptor,
                        min(1 << 20, before.st_size - observed + 1),
                    )
                    if not block:
                        break
                    observed += len(block)
                    if observed > before.st_size:
                        raise ValueError(
                            f"Provenance path {relative!r} changed while being hashed."
                        )
                    digest.update(block)
                after = opened.verify_stable()
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"Provenance path {relative!r} changed while being hashed."
            ) from error
        if observed != before.st_size or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Provenance path {relative!r} changed while being hashed.")
        remaining -= observed
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BuildProvenance:
    project_name: str
    project_version: str
    source_digest: str
    lock_digest: str
    repository_revision: str
    builder_id: str
    packages: tuple[InstalledPackage, ...]
    parameters: Mapping[str, object]
    provenance_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.project_name, "Project name"),
            (self.project_version, "Project version"),
            (self.repository_revision, "Repository revision"),
            (self.builder_id, "Builder ID"),
        ):
            _identifier(value, name)
        for value in (self.source_digest, self.lock_digest, self.provenance_id):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("Provenance digests must be lowercase SHA-256 values.")
        packages = tuple(
            sorted(
                self.packages,
                key=lambda package: (package.name.casefold(), package.version),
            )
        )
        if len({(value.name.casefold(), value.version) for value in packages}) != len(
            packages
        ):
            raise ValueError("Provenance package identities must be unique.")
        parameters = json.loads(_canonical_json(dict(self.parameters)))
        if not isinstance(parameters, dict):
            raise TypeError("Build parameters must be a JSON object.")
        object.__setattr__(self, "packages", packages)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        if (
            hashlib.sha256(
                _canonical_json(self._content_record()).encode("utf-8")
            ).hexdigest()
            != self.provenance_id
        ):
            raise ValueError(
                "Build provenance ID does not match its deterministic content."
            )

    def _content_record(self) -> dict[str, object]:
        return {
            "builder_id": self.builder_id,
            "lock_digest": self.lock_digest,
            "packages": [
                {
                    "license_expression": value.license_expression,
                    "name": value.name,
                    "supplier": value.supplier,
                    "version": value.version,
                }
                for value in self.packages
            ],
            "parameters": dict(self.parameters),
            "project_name": self.project_name,
            "project_version": self.project_version,
            "repository_revision": self.repository_revision,
            "source_digest": self.source_digest,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "build-provenance",
            **self._content_record(),
            "provenance_id": self.provenance_id,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_record())


def create_build_provenance(
    project_name: str,
    project_version: str,
    /,
    *,
    source_digest: str,
    lock_digest: str,
    repository_revision: str,
    builder_id: str,
    packages: Sequence[InstalledPackage],
    parameters: Mapping[str, object] | None = None,
) -> BuildProvenance:
    ordered = tuple(
        sorted(packages, key=lambda package: (package.name.casefold(), package.version))
    )
    values = dict(parameters or {})
    content = {
        "builder_id": builder_id,
        "lock_digest": lock_digest,
        "packages": [
            {
                "license_expression": value.license_expression,
                "name": value.name,
                "supplier": value.supplier,
                "version": value.version,
            }
            for value in ordered
        ],
        "parameters": values,
        "project_name": project_name,
        "project_version": project_version,
        "repository_revision": repository_revision,
        "source_digest": source_digest,
    }
    identifier = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
    provenance = BuildProvenance(
        project_name,
        project_version,
        source_digest,
        lock_digest,
        repository_revision,
        builder_id,
        ordered,
        values,
        identifier,
    )
    emit(
        "INFO",
        "lifecycle.provenance.captured",
        "Build provenance captured",
        package_count=len(ordered),
        provenance_id=identifier,
        repository_revision=repository_revision,
    )
    return provenance


def build_provenance_from_paths(
    project_name: str,
    project_version: str,
    project_root: str | Path,
    /,
    *,
    source_paths: Sequence[str | Path],
    lock_paths: Sequence[str | Path],
    repository_revision: str,
    builder_id: str,
    distributions: Iterable[DistributionLike] | None = None,
    parameters: Mapping[str, object] | None = None,
) -> BuildProvenance:
    """Snapshot explicit source, lock, and installed-distribution inputs."""

    return create_build_provenance(
        project_name,
        project_version,
        source_digest=digest_paths(project_root, source_paths),
        lock_digest=digest_paths(project_root, lock_paths),
        repository_revision=repository_revision,
        builder_id=builder_id,
        packages=installed_packages(distributions),
        parameters=parameters,
    )


def generate_spdx_sbom(
    name: str,
    packages: Sequence[InstalledPackage] | None = None,
    /,
    *,
    created_at: int = 0,
    creator: str = "Tool: phydrax",
    provenance: BuildProvenance | None = None,
) -> dict[str, object]:
    """Return deterministic SPDX-2.3 JSON. Timestamp is explicit and defaults to epoch."""

    _identifier(name, "SBOM name")
    _identifier(creator, "SBOM creator")
    if created_at < 0:
        raise ValueError("SPDX creation timestamp must be nonnegative.")
    values = installed_packages() if packages is None else tuple(packages)
    ordered = tuple(
        sorted(values, key=lambda value: (value.name.casefold(), value.version))
    )
    if len({(value.name.casefold(), value.version) for value in ordered}) != len(ordered):
        raise ValueError("SPDX package identities must be unique.")
    package_records: list[dict[str, object]] = []
    for package in ordered:
        identity_digest = hashlib.sha256(
            f"{package.name}\x00{package.version}".encode("utf-8")
        ).hexdigest()[:16]
        package_records.append(
            {
                "SPDXID": f"SPDXRef-Package-{identity_digest}",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": package.license_expression,
                "name": package.name,
                "supplier": package.supplier,
                "versionInfo": package.version,
            }
        )
    namespace_seed = {
        "created_at": created_at,
        "creator": creator,
        "name": name,
        "packages": package_records,
        "provenance_id": (None if provenance is None else provenance.provenance_id),
    }
    document_digest = hashlib.sha256(
        _canonical_json(namespace_seed).encode("utf-8")
    ).hexdigest()
    created = datetime.fromtimestamp(created_at, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    document: dict[str, object] = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {"created": created, "creators": [creator]},
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://phydrax.dev/spdx/{document_digest}",
        "name": name,
        "packages": package_records,
        "spdxVersion": "SPDX-2.3",
    }
    if provenance is not None:
        document["annotations"] = [
            {
                "annotationDate": created,
                "annotationType": "OTHER",
                "annotator": creator,
                "comment": _canonical_json(
                    {
                        "build_provenance_id": provenance.provenance_id,
                        "lock_sha256": provenance.lock_digest,
                        "source_sha256": provenance.source_digest,
                    }
                ),
            }
        ]
    return document


def spdx_json(document: Mapping[str, object], /) -> str:
    return _canonical_json(dict(document))


__all__ = [
    "BuildProvenance",
    "DistributionLike",
    "InstalledPackage",
    "build_provenance_from_paths",
    "create_build_provenance",
    "digest_paths",
    "generate_spdx_sbom",
    "installed_packages",
    "spdx_json",
]

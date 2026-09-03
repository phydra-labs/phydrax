#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Audit on-disk prerequisites for cardiovascular release qualification.

This is a preflight auditor, not a licence issuer or release approver.  It
returns a nonzero status while any required record is absent or malformed and
keeps the independent release decision outside the G0-G7 evidence result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from phydrax.applications.cardiovascular._commercial import (
    CardiovascularArtifactKind,
    CardiovascularReleaseGate,
)


_DEFAULT_ARTIFACT_PATHS = {
    CardiovascularArtifactKind.SBOM: Path("release/cardiovascular/sbom.spdx.json"),
    CardiovascularArtifactKind.BUILD_PROVENANCE: Path(
        "release/cardiovascular/build-provenance.json"
    ),
    CardiovascularArtifactKind.COMMERCIAL_LICENSE: Path(
        "release/cardiovascular/commercial-license-authorization.json"
    ),
    CardiovascularArtifactKind.NOTICE_AUDIT: Path(
        "release/cardiovascular/notice-audit.json"
    ),
    CardiovascularArtifactKind.DATA_RIGHTS: Path(
        "release/cardiovascular/data-rights.json"
    ),
    CardiovascularArtifactKind.SUPPLY_CHAIN_ATTESTATION: Path(
        "release/cardiovascular/supply-chain-attestation.json"
    ),
}
_REQUIRED_NON_CLAIMS = (
    "clinical-decision-support",
    "diagnosis",
    "regulated-medical-device",
    "treatment",
)

_NOASSERTION_VALUES = frozenset({"", "NOASSERTION", "NONE", "UNKNOWN", "UNRESOLVED"})
_DISTRIBUTION_ARTIFACT_KINDS = frozenset({"wheel", "sdist", "container"})
_EXTERNAL_RECORD_KINDS = {
    "commercial-license": "cardiovascular-commercial-license-authorization",
    "data-rights": "cardiovascular-data-rights-determination",
    "signer": "cardiovascular-release-signer",
    "verifier": "cardiovascular-signature-verification",
    "vulnerability-report": "cardiovascular-vulnerability-scan",
    "license-report": "cardiovascular-license-scan",
    "supply-chain-attestation": "cardiovascular-supply-chain-attestation",
}


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_record(path: Path, /) -> tuple[dict[str, object], tuple[str, ...]]:
    result: dict[str, object] = {"path": str(path)}
    blockers: list[str] = []
    if not path.exists():
        result["status"] = "missing"
        return result, ("missing",)
    if path.is_symlink() or not path.is_file():
        result["status"] = "not-regular-file"
        return result, ("not-regular-file",)
    byte_size = path.stat().st_size
    result["byte_size"] = byte_size
    if byte_size == 0:
        result["status"] = "empty"
        return result, ("empty",)
    result["status"] = "present"
    result["sha256"] = _sha256(path)
    return result, tuple(blockers)


def _inspect_signed_record(
    path: Path,
    required_fields: Sequence[str],
    /,
    *,
    reviewer_signed: bool = False,
) -> tuple[dict[str, object], tuple[str, ...]]:
    result, failures = _inspect_record(path)
    if failures:
        return result, failures
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["status"] = "invalid-json"
        return result, ("invalid-json",)
    if not isinstance(document, Mapping):
        result["status"] = "invalid-record"
        return result, ("invalid-record",)
    missing = tuple(field for field in required_fields if field not in document)
    if missing:
        result["status"] = "missing-signed-fields"
        return result, tuple(f"missing-field:{field}" for field in missing)
    signature = str(document["signature"]).lower()
    if (
        not signature
        or len(signature) % 2
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        result["status"] = "invalid-signature-encoding"
        return result, ("invalid-signature-encoding",)
    if reviewer_signed and document["reviewer_id"] != document["signer_id"]:
        result["status"] = "reviewer-signer-mismatch"
        return result, ("reviewer-signer-mismatch",)
    result["status"] = "present-signature-not-cryptographically-evaluated"
    result["signer_id"] = str(document["signer_id"])
    result["signature_algorithm"] = str(document["signature_algorithm"])
    return result, ()


def _is_asserted(value: object, /) -> bool:
    return isinstance(value, str) and value.strip().upper() not in _NOASSERTION_VALUES


def _sha256_value(value: object, /) -> str | None:
    if isinstance(value, Mapping):
        algorithm = str(value.get("algorithm", value.get("alg", ""))).lower()
        value = value.get("content", value.get("value", ""))
        if algorithm not in {"sha256", "sha-256"}:
            return None
    if not isinstance(value, str):
        return None
    value = value.lower()
    if value.startswith("sha256:"):
        value = value[7:]
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        return None
    return value


def _hashes_from_record(value: object, /) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(digest for item in value if (digest := _sha256_value(item)) is not None)


def _source_uri(source: object, repository_root: Path, /) -> str | None:
    if not isinstance(source, Mapping) or len(source) != 1:
        return None
    kind, value = next(iter(source.items()))
    if not _is_asserted(value):
        return None
    if kind in {"editable", "path", "virtual"}:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = repository_root / path
        return path.resolve().as_uri()
    if kind in {"registry", "git", "url"}:
        return str(value).strip()
    return None


def _load_bound_external_record(
    value: Path | str | None,
    /,
    *,
    expected_kind: str,
    source_commit: str,
    lock_sha256: str,
) -> tuple[dict[str, object], tuple[str, ...], Mapping[str, object] | None]:
    if value is None:
        return {"status": "missing"}, ("missing",), None
    path = Path(value).expanduser().absolute()
    result, failures = _inspect_record(path)
    if failures:
        return result, failures, None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["status"] = "invalid-json"
        return result, ("invalid-json",), None
    if not isinstance(document, Mapping):
        result["status"] = "invalid-record"
        return result, ("invalid-record",), None

    blockers: list[str] = []
    if document.get("kind") != expected_kind:
        blockers.append("kind-mismatch")
    if document.get("source_commit") != source_commit:
        blockers.append("source-commit-mismatch")
    if document.get("lock_sha256") != lock_sha256:
        blockers.append("lock-sha256-mismatch")
    for field in ("signer_id", "signature_algorithm"):
        if not _is_asserted(document.get(field)):
            blockers.append(f"missing-field:{field}")
    signature = str(document.get("signature", "")).lower()
    if (
        not signature
        or len(signature) % 2
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        blockers.append("invalid-signature-encoding")

    result["kind"] = str(document.get("kind", ""))
    result["signer_id"] = str(document.get("signer_id", ""))
    result["signature_algorithm"] = str(document.get("signature_algorithm", ""))
    result["status"] = (
        "valid-bound-signed-record" if not blockers else "invalid-bound-record"
    )
    return result, tuple(blockers), document


def _subject_hashes(
    document: Mapping[str, object] | None,
    /,
) -> tuple[dict[str, str], tuple[str, ...]]:
    if document is None:
        return {}, ()
    subjects = document.get("subjects")
    if not isinstance(subjects, list):
        return {}, ("subjects-invalid",)
    hashes: dict[str, str] = {}
    blockers: list[str] = []
    for subject in subjects:
        if not isinstance(subject, Mapping):
            blockers.append("subject-invalid")
            continue
        name = str(subject.get("name", "")).strip()
        digest = _sha256_value(subject.get("sha256"))
        if not name:
            blockers.append("subject-name-missing")
        elif digest is None:
            blockers.append(f"subject-hash-invalid:{name}")
        elif name in hashes:
            blockers.append(f"subject-duplicate:{name}")
        else:
            hashes[name] = digest
    return hashes, tuple(blockers)


def _validate_subject_bindings(
    document: Mapping[str, object] | None,
    expected: Mapping[str, str],
    /,
) -> tuple[str, ...]:
    subjects, blockers = _subject_hashes(document)
    failures = list(blockers)
    for name, expected_hash in expected.items():
        actual_hash = subjects.get(name)
        if actual_hash is None:
            failures.append(f"subject-missing:{name}")
        elif actual_hash != expected_hash:
            failures.append(f"subject-hash-mismatch:{name}")
    return tuple(failures)


def _normalize_artifact_paths(
    repository_root: Path,
    overrides: Mapping[CardiovascularArtifactKind | str, Path | str],
    /,
) -> dict[CardiovascularArtifactKind, Path]:
    paths = {
        kind: repository_root / relative_path
        for kind, relative_path in _DEFAULT_ARTIFACT_PATHS.items()
    }
    for kind, path in overrides.items():
        kind_ = CardiovascularArtifactKind(kind)
        path_ = Path(path).expanduser()
        paths[kind_] = path_ if path_.is_absolute() else repository_root / path_
    return paths


def _write_json(path: Path, record: Mapping[str, object], /) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "byte_size": path.stat().st_size,
    }


def build_cardiovascular_release_artifacts(
    repository_root: Path | str,
    output_directory: Path | str,
    /,
    *,
    commercial_license_record: Path | str | None = None,
    data_rights_record: Path | str | None = None,
    signer_record: Path | str | None = None,
    verifier_record: Path | str | None = None,
    vulnerability_report: Path | str | None = None,
    license_report: Path | str | None = None,
    supply_chain_attestation: Path | str | None = None,
    distribution_artifacts: Mapping[str, Path | str] | None = None,
) -> dict[str, object]:
    """Derive release evidence while keeping every external authority external."""
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory.")
    if distribution_artifacts is not None and not isinstance(
        distribution_artifacts, Mapping
    ):
        raise TypeError("distribution_artifacts must be a mapping.")

    blockers: list[str] = []
    lock_path = root / "uv.lock"
    project_path = root / "pyproject.toml"
    notice_path = root / "NOTICE"
    license_path = root / "LICENSE"
    for label, path in (
        ("dependency-lock", lock_path),
        ("project-metadata", project_path),
        ("notice", notice_path),
        ("repository-license", license_path),
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            blockers.append(f"{label}:missing-or-invalid")

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        commit = "unavailable"
        clean_tree = False
        blockers.append("source-control:git-metadata-unavailable")
    else:
        commit = head.stdout.strip() if head.returncode == 0 else "unavailable"
        clean_tree = status.returncode == 0 and not status.stdout
        if head.returncode != 0 or status.returncode != 0:
            blockers.append("source-control:git-metadata-unavailable")
        elif not clean_tree:
            blockers.append("source-control:working-tree-not-clean")

    lock_sha = (
        _sha256(lock_path)
        if lock_path.is_file() and not lock_path.is_symlink()
        else "unavailable"
    )
    project_sha = (
        _sha256(project_path)
        if project_path.is_file() and not project_path.is_symlink()
        else "unavailable"
    )
    lock: Mapping[str, object] = {}
    if lock_sha != "unavailable":
        try:
            parsed_lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            blockers.append("dependency-lock:invalid-toml")
        else:
            lock = parsed_lock
    project: Mapping[str, object] = {}
    if project_sha != "unavailable":
        try:
            parsed_project = tomllib.loads(project_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            blockers.append("project-metadata:invalid-toml")
        else:
            project = parsed_project

    project_table = project.get("project", {})
    if not isinstance(project_table, Mapping):
        project_table = {}
        blockers.append("project-metadata:project-table-invalid")
    project_name = str(project_table.get("name", "")).strip()
    project_version = str(project_table.get("version", "")).strip()
    if not project_name or not project_version:
        blockers.append("project-metadata:project-identity-missing")

    distributions: dict[str, dict[str, object]] = {}
    for kind, value in sorted((distribution_artifacts or {}).items()):
        if kind not in _DISTRIBUTION_ARTIFACT_KINDS:
            blockers.append(f"distribution-artifact-kind:unsupported:{kind}")
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        record, failures = _inspect_record(path.absolute())
        record["kind"] = kind
        distributions[kind] = record
        blockers.extend(f"distribution-artifact-{kind}:{item}" for item in failures)
    if not distributions:
        blockers.append("distribution-artifact:missing")
    distribution_hashes = {
        kind: str(record["sha256"])
        for kind, record in distributions.items()
        if "sha256" in record
    }

    external_values = {
        "commercial-license": commercial_license_record,
        "data-rights": data_rights_record,
        "signer": signer_record,
        "verifier": verifier_record,
        "vulnerability-report": vulnerability_report,
        "license-report": license_report,
        "supply-chain-attestation": supply_chain_attestation,
    }
    external: dict[str, object] = {}
    external_documents: dict[str, Mapping[str, object] | None] = {}
    for label, value in external_values.items():
        result, failures, document = _load_bound_external_record(
            value,
            expected_kind=_EXTERNAL_RECORD_KINDS[label],
            source_commit=commit,
            lock_sha256=lock_sha,
        )
        external[label] = result
        external_documents[label] = document
        blockers.extend(f"external-{label}-record:{item}" for item in failures)

    required_statuses = {
        "commercial-license": ("authorization_status", "authorized"),
        "data-rights": ("rights_status", "authorized"),
        "signer": ("signer_status", "active"),
        "verifier": ("verification_status", "verified"),
        "vulnerability-report": ("scan_status", "passed"),
        "license-report": ("scan_status", "passed"),
        "supply-chain-attestation": ("attestation_status", "verified"),
    }
    for label, (field, expected) in required_statuses.items():
        document = external_documents[label]
        if document is not None and document.get(field) != expected:
            blockers.append(f"external-{label}-record:{field}-not-{expected}")
    for label in ("vulnerability-report", "license-report"):
        document = external_documents[label]
        scanner = None if document is None else document.get("scanner")
        if document is not None and (
            not isinstance(scanner, Mapping)
            or not _is_asserted(scanner.get("name"))
            or not _is_asserted(scanner.get("version"))
        ):
            blockers.append(f"external-{label}-record:scanner-identity-missing")

    external_hashes = {
        label: str(record["sha256"])
        for label, record in external.items()
        if isinstance(record, Mapping) and "sha256" in record
    }
    attestation_subjects = {
        "dependency-lock": lock_sha,
        **{
            label: external_hashes[label]
            for label in (
                "commercial-license",
                "data-rights",
                "signer",
                "license-report",
                "vulnerability-report",
            )
            if label in external_hashes
        },
        **{
            f"distribution:{kind}": digest for kind, digest in distribution_hashes.items()
        },
    }
    for failure in _validate_subject_bindings(
        external_documents["supply-chain-attestation"], attestation_subjects
    ):
        blockers.append(f"external-supply-chain-attestation-record:{failure}")
    verifier_subjects = {
        label: digest for label, digest in external_hashes.items() if label != "verifier"
    }
    for failure in _validate_subject_bindings(
        external_documents["verifier"], verifier_subjects
    ):
        blockers.append(f"external-verifier-record:{failure}")

    raw_records = lock.get("package", ())
    package_items: list[dict[str, object]] = []
    if not isinstance(raw_records, list):
        blockers.append("dependency-lock:package-table-invalid")
        raw_records = []
    for record_index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            blockers.append(f"dependency-lock:package-record-invalid:{record_index}")
            continue
        name = str(record.get("name", "")).strip()
        version = str(record.get("version", "")).strip()
        if not name or not version:
            blockers.append(f"dependency-lock:package-identity-missing:{record_index}")
            continue
        source = _source_uri(record.get("source"), root)
        if source is None:
            blockers.append(f"dependency-metadata:{name}@{version}:source-unresolved")

        archives: list[tuple[str, Mapping[str, object]]] = []
        sdist = record.get("sdist")
        if sdist is not None:
            if isinstance(sdist, Mapping):
                archives.append(("sdist", sdist))
            else:
                blockers.append(
                    f"dependency-metadata:{name}@{version}:sdist-record-invalid"
                )
        wheels = record.get("wheels", [])
        if not isinstance(wheels, list):
            blockers.append(f"dependency-metadata:{name}@{version}:wheels-invalid")
            wheels = ()
        for wheel_index, wheel in enumerate(wheels):
            if isinstance(wheel, Mapping):
                archives.append((f"wheel-{wheel_index}", wheel))
            else:
                blockers.append(
                    f"dependency-metadata:{name}@{version}:wheel-record-invalid:{wheel_index}"
                )

        archive_hashes: list[str] = []
        archive_urls: list[str] = []
        for archive_kind, archive in archives:
            digest = _sha256_value(archive.get("hash"))
            url = archive.get("url")
            if digest is None:
                blockers.append(
                    f"dependency-metadata:{name}@{version}:{archive_kind}-hash-unresolved"
                )
            else:
                archive_hashes.append(digest)
            if not _is_asserted(url):
                blockers.append(
                    f"dependency-metadata:{name}@{version}:{archive_kind}-source-unresolved"
                )
            else:
                archive_urls.append(str(url))

        markers = record.get("resolution-markers", ())
        identity = json.dumps(
            {
                "name": name,
                "version": version,
                "source": source,
                "resolution_markers": markers,
                "record_index": record_index,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        package_items.append(
            {
                "name": name,
                "version": version,
                "source": source,
                "download": archive_urls[0] if archive_urls else source,
                "hashes": tuple(sorted(set(archive_hashes))),
                "record": record,
                "spdx_id": f"SPDXRef-Package-{identity_digest[:24]}",
                "bom_ref": f"urn:phydrax:uv-lock:{identity_digest}",
            }
        )
    package_items.sort(
        key=lambda item: (
            str(item["name"]).lower(),
            str(item["version"]),
            str(item["source"]),
            str(item["bom_ref"]),
        )
    )

    normalized_project_name = project_name.lower().replace("_", "-")
    root_packages = [
        item
        for item in package_items
        if str(item["name"]).lower().replace("_", "-") == normalized_project_name
        and str(item["version"]) == project_version
    ]
    if len(root_packages) != 1:
        blockers.append("dependency-lock:project-package-unresolved")
    root_package = root_packages[0] if len(root_packages) == 1 else None
    if root_package is not None:
        root_package["hashes"] = tuple(
            sorted(set(root_package["hashes"]) | set(distribution_hashes.values()))
        )

    scan_indexes: dict[str, dict[tuple[str, str], Mapping[str, object]]] = {}
    for label in ("license-report", "vulnerability-report"):
        document = external_documents[label]
        entries = None if document is None else document.get("packages")
        index: dict[tuple[str, str], Mapping[str, object]] = {}
        if document is not None and not isinstance(entries, list):
            blockers.append(f"external-{label}-record:packages-invalid")
        elif isinstance(entries, list):
            for entry_index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    blockers.append(
                        f"external-{label}-record:package-invalid:{entry_index}"
                    )
                    continue
                key = (
                    str(entry.get("name", "")).strip().lower().replace("_", "-"),
                    str(entry.get("version", "")).strip(),
                )
                if not key[0] or not key[1]:
                    blockers.append(
                        f"external-{label}-record:package-identity-missing:{entry_index}"
                    )
                elif key in index:
                    blockers.append(
                        f"external-{label}-record:package-duplicate:{key[0]}@{key[1]}"
                    )
                else:
                    index[key] = entry
        scan_indexes[label] = index

    locked_keys = {
        (
            str(item["name"]).lower().replace("_", "-"),
            str(item["version"]),
        )
        for item in package_items
    }
    for label, index in scan_indexes.items():
        for name, version in sorted(set(index) - locked_keys):
            blockers.append(
                f"external-{label}-record:package-not-locked:{name}@{version}"
            )

    spdx_packages: list[dict[str, object]] = []
    cyclonedx_components: list[dict[str, object]] = []
    for item in package_items:
        name = str(item["name"])
        version = str(item["version"])
        identity = f"{name}@{version}"
        key = (name.lower().replace("_", "-"), version)
        expected_hashes = set(item["hashes"])
        if not expected_hashes:
            blockers.append(f"dependency-metadata:{identity}:hash-unresolved")

        license_entry = scan_indexes["license-report"].get(key)
        vulnerability_entry = scan_indexes["vulnerability-report"].get(key)
        for label, entry in (
            ("license-report", license_entry),
            ("vulnerability-report", vulnerability_entry),
        ):
            if entry is None:
                blockers.append(f"external-{label}-record:package-missing:{identity}")
                continue
            entry_source = entry.get("source")
            if entry_source is not None and entry_source != item["source"]:
                blockers.append(
                    f"external-{label}-record:package-source-mismatch:{identity}"
                )
            raw_reported_hashes = entry.get("hashes")
            if not isinstance(raw_reported_hashes, list):
                blockers.append(
                    f"external-{label}-record:package-hashes-invalid:{identity}"
                )
            else:
                for reported_hash in raw_reported_hashes:
                    if _sha256_value(reported_hash) is None:
                        blockers.append(
                            f"external-{label}-record:package-hash-invalid:{identity}"
                        )
                        break
            reported_hashes = set(_hashes_from_record(entry.get("hashes")))
            for digest in sorted(expected_hashes - reported_hashes):
                blockers.append(
                    f"external-{label}-record:package-hash-missing:{identity}:{digest}"
                )

        locked_record = item["record"]
        assert isinstance(locked_record, Mapping)
        locked_license = locked_record.get("license")
        if isinstance(locked_license, Mapping):
            locked_license = locked_license.get(
                "expression", locked_license.get("id", "")
            )
        license_concluded = (
            license_entry.get("license_concluded")
            if license_entry is not None
            else locked_record.get("license_concluded", locked_license)
        )
        license_declared = (
            license_entry.get("license_declared")
            if license_entry is not None
            else locked_record.get("license_declared", locked_license)
        )
        copyright_text = (
            license_entry.get("copyright_text")
            if license_entry is not None
            else locked_record.get("copyright")
        )
        for field, value in (
            ("license-concluded", license_concluded),
            ("license-declared", license_declared),
            ("copyright", copyright_text),
        ):
            if not _is_asserted(value):
                blockers.append(f"dependency-metadata:{identity}:{field}-unresolved")
        if (
            _is_asserted(locked_license)
            and _is_asserted(license_declared)
            and str(locked_license).strip() != str(license_declared).strip()
        ):
            blockers.append(f"dependency-metadata:{identity}:declared-license-mismatch")

        if vulnerability_entry is not None:
            if vulnerability_entry.get("status") != "passed":
                blockers.append(
                    f"external-vulnerability-report-record:package-not-passed:{identity}"
                )
            vulnerabilities = vulnerability_entry.get("vulnerabilities")
            if not isinstance(vulnerabilities, list):
                blockers.append(
                    f"external-vulnerability-report-record:"
                    f"package-vulnerabilities-invalid:{identity}"
                )
            elif vulnerabilities:
                blockers.append(
                    f"external-vulnerability-report-record:"
                    f"package-vulnerabilities-present:{identity}"
                )

        download = item["download"]
        if not _is_asserted(download):
            blockers.append(
                f"dependency-metadata:{identity}:download-location-unresolved"
            )
        asserted_download = str(download) if _is_asserted(download) else "NOASSERTION"
        asserted_concluded = (
            str(license_concluded).strip()
            if _is_asserted(license_concluded)
            else "NOASSERTION"
        )
        asserted_declared = (
            str(license_declared).strip()
            if _is_asserted(license_declared)
            else "NOASSERTION"
        )
        asserted_copyright = (
            str(copyright_text).strip() if _is_asserted(copyright_text) else "NOASSERTION"
        )
        checksums = [
            {"algorithm": "SHA256", "checksumValue": digest}
            for digest in sorted(expected_hashes)
        ]
        spdx_package: dict[str, object] = {
            "SPDXID": item["spdx_id"],
            "name": name,
            "versionInfo": version,
            "downloadLocation": asserted_download,
            "filesAnalyzed": False,
            "licenseConcluded": asserted_concluded,
            "licenseDeclared": asserted_declared,
            "copyrightText": asserted_copyright,
            "checksums": checksums,
        }
        if _is_asserted(item["source"]):
            spdx_package["sourceInfo"] = str(item["source"])
        if str(item["source"]).startswith("https://pypi.org/"):
            spdx_package["externalRefs"] = [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{name}@{version}",
                }
            ]
        spdx_packages.append(spdx_package)

        component: dict[str, object] = {
            "type": "application" if item is root_package else "library",
            "name": name,
            "version": version,
            "bom-ref": item["bom_ref"],
            "hashes": [
                {"alg": "SHA-256", "content": digest}
                for digest in sorted(expected_hashes)
            ],
            "licenses": [{"expression": asserted_concluded}],
            "properties": [
                {"name": "phydrax:uv-lock:source", "value": str(item["source"])},
            ],
        }
        if _is_asserted(download):
            component["externalReferences"] = [
                {"type": "distribution", "url": asserted_download}
            ]
        cyclonedx_components.append(component)

    packages_by_name: dict[str, list[dict[str, object]]] = {}
    for item in package_items:
        packages_by_name.setdefault(
            str(item["name"]).lower().replace("_", "-"), []
        ).append(item)
    dependency_edges: dict[str, set[str]] = {
        str(item["bom_ref"]): set() for item in package_items
    }
    spdx_relationships: list[dict[str, object]] = []
    if root_package is not None:
        spdx_relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": root_package["spdx_id"],
            }
        )

    for item in package_items:
        record = item["record"]
        assert isinstance(record, Mapping)
        dependencies: list[tuple[str, Mapping[str, object]]] = []
        direct_dependencies = record.get("dependencies", [])
        if not isinstance(direct_dependencies, list):
            blockers.append(
                f"dependency-graph:{item['name']}@{item['version']}:dependencies-invalid"
            )
        else:
            dependencies.extend(
                ("runtime", dependency) for dependency in direct_dependencies
            )
        for table_name in ("optional-dependencies", "dev-dependencies"):
            table = record.get(table_name, {})
            if not isinstance(table, Mapping):
                blockers.append(
                    f"dependency-graph:{item['name']}@{item['version']}:"
                    f"{table_name}-invalid"
                )
                continue
            for group, group_dependencies in table.items():
                if not isinstance(group_dependencies, list):
                    blockers.append(
                        f"dependency-graph:{item['name']}@{item['version']}:"
                        f"{table_name}-{group}-invalid"
                    )
                    continue
                dependencies.extend(
                    (f"{table_name}:{group}", dependency)
                    for dependency in group_dependencies
                )

        for dependency_group, dependency in dependencies:
            if not isinstance(dependency, Mapping):
                blockers.append(
                    f"dependency-graph:{item['name']}@{item['version']}:"
                    "dependency-record-invalid"
                )
                continue
            dependency_name = (
                str(dependency.get("name", "")).strip().lower().replace("_", "-")
            )
            if not dependency_name:
                blockers.append(
                    f"dependency-graph:{item['name']}@{item['version']}:"
                    "dependency-name-missing"
                )
                continue
            candidates = list(packages_by_name.get(dependency_name, ()))
            dependency_version = dependency.get("version")
            if dependency_version is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["version"] == str(dependency_version)
                ]
            dependency_source = dependency.get("source")
            if dependency_source is not None:
                required_source = _source_uri(dependency_source, root)
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["source"] == required_source
                ]
            if not candidates:
                version_suffix = (
                    f"@{dependency_version}" if dependency_version is not None else ""
                )
                blockers.append(
                    f"dependency-graph:{item['name']}@{item['version']}:"
                    f"unresolved:{dependency_name}{version_suffix}"
                )
                continue
            for candidate in candidates:
                dependency_edges[str(item["bom_ref"])].add(str(candidate["bom_ref"]))
                relationship: dict[str, object] = {
                    "spdxElementId": item["spdx_id"],
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": candidate["spdx_id"],
                }
                relationship_context = {
                    "group": dependency_group,
                    **{
                        field: dependency[field]
                        for field in ("marker", "extra")
                        if field in dependency
                    },
                }
                relationship["comment"] = json.dumps(
                    relationship_context, sort_keys=True, separators=(",", ":")
                )
                spdx_relationships.append(relationship)

    unique_relationships = {
        json.dumps(relationship, sort_keys=True, separators=(",", ":")): relationship
        for relationship in spdx_relationships
    }
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "phydrax-cardiovascular-release",
        "documentNamespace": f"https://phydra.ai/spdx/{commit}/{lock_sha}",
        "creationInfo": {"creators": ["Tool: cardiovascular_release_qualification.py"]},
        "packages": spdx_packages,
        "relationships": [
            unique_relationships[key] for key in sorted(unique_relationships)
        ],
    }
    root_component = (
        next(
            component
            for component in cyclonedx_components
            if root_package is not None
            and component["bom-ref"] == root_package["bom_ref"]
        )
        if root_package is not None
        else {
            "type": "application",
            "name": project_name or "phydrax",
            "version": project_version,
            "bom-ref": "urn:phydrax:unresolved-project",
        }
    )
    cyclonedx = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"component": root_component},
        "components": [
            component
            for component in cyclonedx_components
            if component is not root_component
        ],
        "dependencies": [
            {"ref": reference, "dependsOn": sorted(dependencies)}
            for reference, dependencies in sorted(dependency_edges.items())
        ],
    }

    provenance = {
        "kind": "cardiovascular-build-provenance",
        "source_commit": commit,
        "source_tree_clean": clean_tree,
        "dependency_lock": {"path": "uv.lock", "sha256": lock_sha},
        "project_metadata": {"path": "pyproject.toml", "sha256": project_sha},
        "distribution_artifacts": distributions,
        "builder": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
    }

    notice_entries: list[dict[str, object]] = []
    if notice_path.is_file() and not notice_path.is_symlink():
        try:
            notice_text = notice_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            blockers.append("notice:invalid-encoding")
        else:
            filenames = sorted(
                set(re.findall(r"LICENSES/([A-Za-z0-9_-]+\.txt)", notice_text))
            )
            for filename in filenames:
                path = root / "LICENSES" / filename
                present = (
                    path.is_file() and not path.is_symlink() and path.stat().st_size > 0
                )
                entry: dict[str, object] = {"file": filename, "present": present}
                if present:
                    entry["sha256"] = _sha256(path)
                else:
                    blockers.append(f"notice-license-text-missing:{filename}")
                notice_entries.append(entry)
    for required in ("SING-MIT.txt", "ASDEX-MIT.txt"):
        required_path = root / "LICENSES" / required
        if (
            not required_path.is_file()
            or required_path.is_symlink()
            or required_path.stat().st_size == 0
        ):
            blocker = f"notice-license-text-missing:{required}"
            if blocker not in blockers:
                blockers.append(blocker)
    notice_audit = {
        "kind": "cardiovascular-notice-audit",
        "notice_sha256": (
            _sha256(notice_path)
            if notice_path.is_file() and not notice_path.is_symlink()
            else "unavailable"
        ),
        "referenced_license_texts": notice_entries,
        "complete": not any(
            blocker.startswith("notice-license-text-missing:")
            or blocker == "notice:invalid-encoding"
            for blocker in blockers
        ),
    }
    if license_path.is_file() and not license_path.is_symlink():
        try:
            repository_license = license_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            blockers.append("repository-license:invalid-encoding")
        else:
            if "PHYDRA NON-PRODUCTION LICENSE" in repository_license:
                blockers.append(
                    "commercial-license-grant-absent:repository-license-is-pnpl"
                )

    blockers = list(dict.fromkeys(blockers))
    g5_evidence_ready = not blockers
    generated: dict[str, dict[str, object]] = {
        "sbom-spdx": _write_json(output / "sbom.spdx.json", spdx),
        "sbom-cyclonedx": _write_json(output / "sbom.cyclonedx.json", cyclonedx),
        "build-provenance": _write_json(output / "build-provenance.json", provenance),
        "notice-audit": _write_json(output / "notice-audit.json", notice_audit),
    }
    supply_chain_manifest = {
        "kind": "cardiovascular-supply-chain-evidence-manifest",
        "source_commit": commit,
        "lock_sha256": lock_sha,
        "dependency_package_count": len(package_items),
        "dependency_graph_complete": not any(
            blocker.startswith("dependency-") for blocker in blockers
        ),
        "g5_evidence_ready": g5_evidence_ready,
        "generated_artifacts": {
            name: {
                "sha256": record["sha256"],
                "byte_size": record["byte_size"],
            }
            for name, record in generated.items()
        },
        "external_records": {
            label: {
                field: record[field]
                for field in ("path", "sha256", "byte_size", "kind", "signer_id")
                if field in record
            }
            for label, record in external.items()
            if isinstance(record, Mapping)
        },
        "distribution_artifacts": distributions,
        "dependencies": {
            "sbom-spdx": ["dependency-lock", "license-report"],
            "sbom-cyclonedx": [
                "dependency-lock",
                "license-report",
                "vulnerability-report",
            ],
            "build-provenance": [
                "dependency-lock",
                "project-metadata",
                *[f"distribution:{kind}" for kind in sorted(distribution_hashes)],
            ],
            "notice-audit": ["notice", "repository-license"],
            "supply-chain-attestation": sorted(attestation_subjects),
            "signature-verification": sorted(verifier_subjects),
        },
    }
    generated["supply-chain-evidence-manifest"] = _write_json(
        output / "supply-chain-evidence-manifest.json", supply_chain_manifest
    )

    dossier_seed = {
        "kind": "cardiovascular-unsigned-gate-dossier",
        "source_commit": commit,
        "lock_sha256": lock_sha,
        "gates": [gate.gate_key for gate in CardiovascularReleaseGate],
        "g5_evidence_ready": g5_evidence_ready,
        "supply_chain_manifest_sha256": generated["supply-chain-evidence-manifest"][
            "sha256"
        ],
        "signed": False,
        "commercial_ready": False,
    }
    dossier_id = hashlib.sha256(
        json.dumps(dossier_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    generated["unsigned-gate-dossier"] = _write_json(
        output / "unsigned-gate-dossier.json",
        {**dossier_seed, "dossier_id": dossier_id},
    )
    hashes = {
        name: {"sha256": record["sha256"], "byte_size": record["byte_size"]}
        for name, record in generated.items()
    }
    generated["artifact-hashes"] = _write_json(
        output / "artifact-hashes.json",
        {
            "kind": "cardiovascular-release-artifact-hashes",
            "artifacts": hashes,
        },
    )
    return {
        "kind": "cardiovascular-release-artifact-build",
        "generated": generated,
        "external_records": external,
        "distribution_artifacts": distributions,
        "g5_evidence_ready": g5_evidence_ready,
        "blockers": blockers,
        "commercial_ready": False,
        "grants_commercial_license": False,
    }


def cardiovascular_release_preflight(
    repository_root: Path | str,
    /,
    *,
    artifact_paths: Mapping[CardiovascularArtifactKind | str, Path | str] | None = None,
    gate_evidence_directory: Path | str = "release/cardiovascular/gates",
    non_claim_directory: Path | str = "release/cardiovascular/non-claims",
    release_decision_path: Path | str = "release/cardiovascular/release-decision.json",
) -> dict[str, object]:
    """Return a deterministic, JSON-ready preflight report for current files."""
    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory.")
    if artifact_paths is not None and not isinstance(artifact_paths, Mapping):
        raise TypeError("artifact_paths must be a mapping.")
    blockers: list[str] = []

    repository_records: dict[str, object] = {}
    for name in ("LICENSE", "NOTICE"):
        record, failures = _inspect_record(root / name)
        repository_records[name.lower()] = record
        blockers.extend(f"repository-{name.lower()}:{failure}" for failure in failures)
    license_path = root / "LICENSE"
    if license_path.is_file() and not license_path.is_symlink():
        license_text = license_path.read_text(encoding="utf-8")
        if "PHYDRA NON-PRODUCTION LICENSE" in license_text or (
            "Commercial Use requires a separate license" in license_text
        ):
            blockers.append(
                "commercial-license-grant-absent:repository-license-is-non-production-only"
            )

    artifacts: dict[str, object] = {}
    paths = _normalize_artifact_paths(
        root, {} if artifact_paths is None else artifact_paths
    )
    for kind in CardiovascularArtifactKind:
        record, failures = _inspect_record(paths[kind])
        artifacts[kind.value] = record
        blockers.extend(f"artifact-{kind.value}:{failure}" for failure in failures)

    gate_root = Path(gate_evidence_directory).expanduser()
    if not gate_root.is_absolute():
        gate_root = root / gate_root
    gates: dict[str, object] = {}
    for gate in CardiovascularReleaseGate:
        record, failures = _inspect_signed_record(
            gate_root / f"{gate.gate_key}.json",
            (
                "dossier_id",
                "reviewer_id",
                "evidence_ids",
                "signer_id",
                "signature_algorithm",
                "signature",
            ),
            reviewer_signed=True,
        )
        gates[gate.gate_key] = record
        blockers.extend(
            f"gate-evidence-{gate.gate_key}:{failure}" for failure in failures
        )

    non_claim_root = Path(non_claim_directory).expanduser()
    if not non_claim_root.is_absolute():
        non_claim_root = root / non_claim_root
    non_claims: dict[str, object] = {}
    for excluded_use in _REQUIRED_NON_CLAIMS:
        record, failures = _inspect_signed_record(
            non_claim_root / f"{excluded_use}.json",
            ("signer_id", "signature_algorithm", "signature"),
        )
        non_claims[excluded_use] = record
        blockers.extend(
            f"signed-non-claim-{excluded_use}:{failure}" for failure in failures
        )

    decision_path = Path(release_decision_path).expanduser()
    if not decision_path.is_absolute():
        decision_path = root / decision_path
    release_decision, decision_failures = _inspect_signed_record(
        decision_path,
        ("approver_id", "signer_id", "signature_algorithm", "signature"),
    )
    decision_status = "not-issued" if decision_failures else "present-not-evaluated"

    blockers = list(dict.fromkeys(blockers))
    preflight_passed = not blockers
    readiness_blockers = list(blockers)
    if preflight_passed:
        readiness_blockers.append(
            "technical-release-assessment-required:run-typed-g0-g7-evaluation"
        )
    if decision_failures:
        readiness_blockers.append("release-decision:not-issued")
    else:
        readiness_blockers.append("release-decision:not-evaluated")
    return {
        "kind": "cardiovascular-release-qualification-preflight",
        "repository_root": str(root),
        "support_boundary": {
            "deployment": "local",
            "data_classification": "non-phi",
            "regulated_device": False,
        },
        "repository_records": repository_records,
        "artifact_references": artifacts,
        "gates": gates,
        "signed_non_claims": non_claims,
        "preflight_passed": preflight_passed,
        "preflight_blockers": blockers,
        "release_decision": {
            "status": decision_status,
            "record": release_decision,
            "separate_from_qualification": True,
        },
        "commercial_ready": False,
        "commercial_ready_blockers": readiness_blockers,
        "grants_commercial_license": False,
        "regulated_device_claim": False,
    }


def _artifact_override(value: str, /) -> tuple[CardiovascularArtifactKind, Path]:
    kind, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("artifact overrides use KIND=PATH")
    try:
        kind_ = CardiovascularArtifactKind(kind)
    except ValueError as error:
        choices = ", ".join(item.value for item in CardiovascularArtifactKind)
        raise argparse.ArgumentTypeError(
            f"unknown artifact kind {kind!r}; choose one of {choices}"
        ) from error
    return kind_, Path(path)


def _distribution_artifact(value: str, /) -> tuple[str, Path]:
    kind, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("distribution artifacts use KIND=PATH")
    if kind not in _DISTRIBUTION_ARTIFACT_KINDS:
        choices = ", ".join(sorted(_DISTRIBUTION_ARTIFACT_KINDS))
        raise argparse.ArgumentTypeError(
            f"unknown distribution kind {kind!r}; choose one of {choices}"
        )
    return kind, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit cardiovascular release prerequisites without issuing a licence, "
            "regulated claim, or release approval."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to audit (defaults to this checkout).",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_artifact_override,
        metavar="KIND=PATH",
        help="Override a required external artifact path; repeat per kind.",
    )
    parser.add_argument(
        "--gate-evidence-directory",
        type=Path,
        default=Path("release/cardiovascular/gates"),
    )
    parser.add_argument(
        "--non-claim-directory",
        type=Path,
        default=Path("release/cardiovascular/non-claims"),
    )
    parser.add_argument(
        "--release-decision",
        type=Path,
        default=Path("release/cardiovascular/release-decision.json"),
    )
    parser.add_argument(
        "--build-artifacts",
        type=Path,
        metavar="DIRECTORY",
        help=(
            "Derive dependency-complete SBOM, provenance, supply-chain manifest, "
            "hashes, and unsigned dossier."
        ),
    )
    parser.add_argument("--commercial-license-record", type=Path)
    parser.add_argument("--data-rights-record", type=Path)
    parser.add_argument("--signer-record", type=Path)
    parser.add_argument("--verifier-record", type=Path)
    parser.add_argument("--vulnerability-report", type=Path)
    parser.add_argument("--license-report", type=Path)
    parser.add_argument("--supply-chain-attestation", type=Path)
    parser.add_argument(
        "--distribution-artifact",
        action="append",
        default=[],
        type=_distribution_artifact,
        metavar="KIND=PATH",
        help="Hash an exact wheel, sdist, or container input; repeat per kind.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write canonical JSON to this file instead of stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    overrides = dict(arguments.artifact)
    distribution_artifacts = dict(arguments.distribution_artifact)
    if len(distribution_artifacts) != len(arguments.distribution_artifact):
        parser.error("each distribution artifact kind may be supplied only once")
    if arguments.build_artifacts is None:
        report = cardiovascular_release_preflight(
            arguments.repository_root,
            artifact_paths=overrides,
            gate_evidence_directory=arguments.gate_evidence_directory,
            non_claim_directory=arguments.non_claim_directory,
            release_decision_path=arguments.release_decision,
        )
        succeeded = bool(report["preflight_passed"])
    else:
        report = build_cardiovascular_release_artifacts(
            arguments.repository_root,
            arguments.build_artifacts,
            commercial_license_record=arguments.commercial_license_record,
            data_rights_record=arguments.data_rights_record,
            signer_record=arguments.signer_record,
            verifier_record=arguments.verifier_record,
            vulnerability_report=arguments.vulnerability_report,
            license_report=arguments.license_report,
            supply_chain_attestation=arguments.supply_chain_attestation,
            distribution_artifacts=distribution_artifacts,
        )
        succeeded = not report["blockers"]
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())

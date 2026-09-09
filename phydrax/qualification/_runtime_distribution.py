#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Live imported-package integrity, bound by an authorized clean-install executor."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .._fingerprint import canonical_fingerprint
from ._trust import QualificationRoleTrust, SignedQualificationRecord


def parse_distribution_manifest(content: str, /) -> dict[str, object]:
    """Validate the retained source → artifact content-addressed mapping, not execution."""

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Distribution records cannot contain duplicate JSON fields.")
        return result

    record = json.loads(content, object_pairs_hook=unique_object)
    expected = {
        "kind",
        "source_build_id",
        "freeze_id",
        "wheel",
        "sdist",
        "sbom",
        "sbom_id",
        "distribution_id",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected
        or record["kind"] != "battery-unpromoted-distribution"
    ):
        raise ValueError("Distribution manifest has an unexpected schema.")
    for name in ("source_build_id", "freeze_id", "sbom_id", "distribution_id"):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "Distribution identities must be exact canonical SHA-256 digests."
            )
    for name in ("wheel", "sdist", "sbom"):
        artifact = record[name]
        if not isinstance(artifact, dict) or set(artifact) != {
            "filename",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("Distribution artifact descriptor is invalid.")
        filename, size, digest = (
            artifact["filename"],
            artifact["size_bytes"],
            artifact["sha256"],
        )
        if (
            not isinstance(filename, str)
            or filename in ("", ".", "..")
            or Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError("Distribution artifact filename must be an exact basename.")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "Distribution artifacts require positive measured sizes and SHA-256 digests."
            )
    if (
        canonical_fingerprint(
            {key: value for key, value in record.items() if key != "distribution_id"}
        )
        != record["distribution_id"]
    ):
        raise ValueError("Distribution manifest content does not match its identity.")
    return record


def _runtime_package_files() -> tuple[tuple[str, str, int], ...]:
    root = Path(__file__).resolve().parents[1]
    for name, module in tuple(sys.modules.items()):
        if name == "phydrax" or name.startswith("phydrax."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(root):
                raise ValueError(
                    "Imported Phydrax modules originate from different distributions."
                )
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError(
                "Runtime distribution files must not be mutable symlink indirections."
            )
        if not path.is_file():
            continue
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(
                "Runtime distribution changed during integrity verification."
            )
        records.append((relative.as_posix(), digest, after.st_size))
    if not records:
        raise ValueError("Imported runtime package has no verifiable retained files.")
    return tuple(records)


@dataclass(frozen=True, slots=True)
class RuntimeDistributionAttestation:
    distribution_id: str
    package_files: tuple[tuple[str, str, int], ...]
    python_implementation: str
    python_version: str
    signature: SignedQualificationRecord

    def content_record(self) -> dict[str, object]:
        return {
            "kind": "installed-runtime-distribution",
            "distribution_id": self.distribution_id,
            "package_files": [list(row) for row in self.package_files],
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.content_record(), "signature": self.signature.to_record()}

    @classmethod
    def from_record(cls, record, /):
        if (
            set(record)
            != {
                "kind",
                "distribution_id",
                "package_files",
                "python_implementation",
                "python_version",
                "signature",
            }
            or record["kind"] != "installed-runtime-distribution"
        ):
            raise ValueError("Invalid installed runtime attestation record.")
        return cls(
            record["distribution_id"],
            tuple(tuple(row) for row in record["package_files"]),
            record["python_implementation"],
            record["python_version"],
            SignedQualificationRecord.from_record(record["signature"]),
        )

    @classmethod
    def attest_verified_install(
        cls, distribution_id, signer, /, *, issued_at: int, expires_at: int
    ):
        """Executor signs only after independently verifying the wheel/install manifest.

        This captures actual imported bytes; it does not itself certify that a
        caller-supplied distribution label belongs to a wheel. That assertion is
        authenticated by the externally provisioned executor authority.
        """
        if (
            not isinstance(distribution_id, str)
            or not distribution_id
            or distribution_id != distribution_id.strip()
        ):
            raise ValueError("Runtime attestation requires an exact distribution ID.")
        files = _runtime_package_files()
        content = {
            "kind": "installed-runtime-distribution",
            "distribution_id": distribution_id,
            "package_files": [list(row) for row in files],
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        }
        signature = SignedQualificationRecord.sign(
            content, signer, role="executor", issued_at=issued_at, expires_at=expires_at
        )
        return cls(
            distribution_id,
            files,
            content["python_implementation"],
            content["python_version"],
            signature,
        )

    def verify(
        self, trust: QualificationRoleTrust, /, *, distribution_id: str, at_time: int
    ) -> int:
        trust.verify(
            self.content_record(), self.signature, role="executor", at_time=at_time
        )
        if (
            self.distribution_id != distribution_id
            or self.python_implementation != platform.python_implementation()
            or self.python_version != platform.python_version()
        ):
            raise ValueError(
                "Runtime distribution or interpreter identity is mismatched."
            )
        if self.package_files != _runtime_package_files():
            raise ValueError(
                "Imported runtime bytes no longer match the authenticated distribution."
            )
        return trust.expiry(self.signature)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..._fingerprint import canonical_fingerprint
from ...qualification._reference import ReferenceArtifactManifest
from ...qualification._trust import QualificationRoleTrust, SignedQualificationRecord


@dataclass(frozen=True, slots=True)
class ReferenceRightsAttestation:
    manifest_id: str
    source_id: str
    license_id: str
    notice_id: str
    commercial_execution: bool
    publication: bool
    derived_output: bool
    export: bool
    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        for value in (self.manifest_id, self.source_id, self.license_id, self.notice_id):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("Rights require exact source/license/NOTICE identities.")
        if any(
            type(value) is not bool
            for value in (
                self.commercial_execution,
                self.publication,
                self.derived_output,
                self.export,
            )
        ):
            raise TypeError("Rights grants must be booleans.")
        if (
            type(self.issued_at) is not int
            or type(self.expires_at) is not int
            or not 0 <= self.issued_at < self.expires_at
        ):
            raise ValueError("Rights validity interval is invalid.")

    def to_record(self) -> dict[str, object]:
        return {"kind": "battery-reference-rights", **asdict(self)}

    @property
    def rights_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(cls, record, /):
        if record.get("kind") != "battery-reference-rights":
            raise ValueError("Invalid reference rights record.")
        return cls(**{name: record[name] for name in cls.__dataclass_fields__})

    def verify(
        self,
        manifest: ReferenceArtifactManifest,
        signature: SignedQualificationRecord,
        trust: QualificationRoleTrust,
        /,
        *,
        at_time: int,
    ) -> str:
        ReferenceArtifactManifest.from_record(manifest.to_record())
        if (
            self.manifest_id != manifest.manifest_id
            or self.license_id != manifest.license_id
        ):
            raise ValueError(
                "Rights attestation does not bind the exact reference/license."
            )
        if not self.issued_at <= at_time < self.expires_at:
            raise ValueError("Reference rights are stale or not active.")
        if not all(
            (
                self.commercial_execution,
                self.publication,
                self.derived_output,
                self.export,
            )
        ):
            raise ValueError("Reference rights do not cover every production use.")
        manifest.require_rights(commercial_use=True, redistribution=True, export=True)
        trust.verify(self, signature, role="scientific-reviewer", at_time=at_time)
        return self.rights_id

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned external source, license, publication, and absorption records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .._fingerprint import canonical_fingerprint
from ._closure_taxonomy import SourceReuseClass
from ._registry import _identifier


class SourceReview(StrEnum):
    UNREVIEWED = "unreviewed"
    TECHNICALLY_REVIEWED = "technically-reviewed"
    LEGAL_REVIEWED = "legal-reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


def _strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(str(value), label) for value in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class PublicationReference:
    title: str
    locator: str

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.locator.strip():
            raise ValueError("Publication title and locator are required.")


@dataclass(frozen=True, slots=True)
class SourceReference:
    source_id: str
    repository_url: str
    revision: str
    archive_digest: str
    license: str
    license_digest: str
    reuse_class: SourceReuseClass
    concepts: tuple[str, ...]
    relevant_paths: tuple[str, ...] = ()
    relevant_documents: tuple[str, ...] = ()
    publications: tuple[PublicationReference, ...] = ()
    code_inspected: bool = False
    behavior_inspected: bool = False
    copying_permitted: bool = False
    provider_only: bool = False
    notice_required: bool = True
    data_rights: str = "not-applicable"
    technical_reviewer: str = "unreviewed"
    legal_reviewer: str = "unreviewed"
    review_status: SourceReview = SourceReview.UNREVIEWED

    @classmethod
    def create(
        cls,
        source_id,
        repository_url,
        revision,
        license,
        reuse_class,
        /,
        *,
        concepts,
        archive_digest="unresolved",
        license_digest="unresolved",
        relevant_paths=(),
        relevant_documents=(),
        publications=(),
        code_inspected=False,
        behavior_inspected=False,
        copying_permitted=False,
        provider_only=False,
        notice_required=True,
        data_rights="not-applicable",
        technical_reviewer="unreviewed",
        legal_reviewer="unreviewed",
        review_status=SourceReview.UNREVIEWED,
    ):
        return cls(
            _identifier(source_id, "source ID"),
            str(repository_url).strip(),
            _identifier(revision, "revision"),
            _identifier(archive_digest, "archive digest"),
            _identifier(license, "license"),
            _identifier(license_digest, "license digest"),
            SourceReuseClass(reuse_class),
            _strings(concepts, "source concept"),
            _strings(relevant_paths, "source path"),
            _strings(relevant_documents, "source document"),
            tuple(publications),
            bool(code_inspected),
            bool(behavior_inspected),
            bool(copying_permitted),
            bool(provider_only),
            bool(notice_required),
            _identifier(data_rights, "data rights"),
            _identifier(technical_reviewer, "technical reviewer"),
            _identifier(legal_reviewer, "legal reviewer"),
            SourceReview(review_status),
        )

    def __post_init__(self) -> None:
        if not self.repository_url.startswith(("https://", "http://")):
            raise ValueError("Source URL must be absolute HTTP(S).")
        if not self.concepts:
            raise ValueError("Source reference requires concepts.")
        if (
            self.reuse_class
            in (SourceReuseClass.STRONG_COPYLEFT, SourceReuseClass.PROPRIETARY)
            and self.copying_permitted
        ):
            raise ValueError("Strong-copyleft/proprietary sources cannot permit copying.")
        if self.review_status is SourceReview.APPROVED:
            unresolved = ("unpinned", "unresolved", "unreviewed")
            if (
                self.revision in unresolved
                or self.archive_digest in unresolved
                or self.license_digest in unresolved
                or self.technical_reviewer in unresolved
                or self.legal_reviewer in unresolved
            ):
                raise ValueError("Approved sources require pinned digests and reviewers.")

    @property
    def source_reference_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True) -> dict[str, object]:
        record = {
            "kind": "source-reference",
            "source_id": self.source_id,
            "repository_url": self.repository_url,
            "revision": self.revision,
            "archive_digest": self.archive_digest,
            "license": self.license,
            "license_digest": self.license_digest,
            "reuse_class": self.reuse_class.value,
            "concepts": list(self.concepts),
            "relevant_paths": list(self.relevant_paths),
            "relevant_documents": list(self.relevant_documents),
            "publications": [
                {"title": value.title, "locator": value.locator}
                for value in self.publications
            ],
            "code_inspected": self.code_inspected,
            "behavior_inspected": self.behavior_inspected,
            "copying_permitted": self.copying_permitted,
            "provider_only": self.provider_only,
            "notice_required": self.notice_required,
            "data_rights": self.data_rights,
            "technical_reviewer": self.technical_reviewer,
            "legal_reviewer": self.legal_reviewer,
            "review_status": self.review_status.value,
        }
        return (
            {**record, "source_reference_id": self.source_reference_id}
            if include_id
            else record
        )


@dataclass(frozen=True, slots=True)
class SourceAbsorptionLedger:
    sources: tuple[SourceReference, ...]

    @classmethod
    def create(cls, sources):
        return cls(tuple(sorted(sources, key=lambda item: item.source_id)))

    def __post_init__(self) -> None:
        identities = tuple(value.source_id for value in self.sources)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("Source ledger requires unique sources.")

    @property
    def ledger_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True) -> dict[str, object]:
        record = {
            "kind": "source-absorption-ledger",
            "sources": [value.to_record() for value in self.sources],
        }
        return {**record, "ledger_id": self.ledger_id} if include_id else record


__all__ = [
    "PublicationReference",
    "SourceAbsorptionLedger",
    "SourceReference",
    "SourceReview",
]

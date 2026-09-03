#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..._fingerprint import canonical_fingerprint


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,254}\Z")
_METADATA_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_METADATA_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}\Z")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE_OR_SSN = re.compile(r"(?:\+?\d[\d(). -]{6,}\d|\b\d{3}-\d{2}-\d{4}\b)")
_DATE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
_PHI_TOKENS = frozenset(
    {
        "address",
        "birth",
        "dob",
        "email",
        "firstname",
        "lastname",
        "medicalrecord",
        "mrn",
        "name",
        "nhs",
        "participant",
        "patient",
        "person",
        "phone",
        "socialsecurity",
        "ssn",
        "subject",
    }
)

CARDIOVASCULAR_CASE_METADATA_KEYS = frozenset(
    {
        "cohort_definition",
        "consent_basis",
        "data_classification",
        "governance_policy",
        "intended_use",
        "jurisdiction",
        "pipeline",
        "purpose",
        "quality_policy",
        "retention_policy",
        "source_modality",
    }
)


def _contains_phi_marker(value: str, /) -> bool:
    collapsed = "".join(character for character in value.lower() if character.isalnum())
    return any(token in collapsed for token in _PHI_TOKENS)


def _identity(value: str, role: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} must be a string identity.")
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{role} must be a non-empty canonical technical identity.")
    if _contains_phi_marker(value) or _EMAIL.search(value) or _PHONE_OR_SSN.search(value):
        raise ValueError(f"{role} must not contain PHI or a linkable person identifier.")
    return value


def _identity_tuple(values: Iterable[str], role: str, /) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{role} must be an iterable of identities, not one string.")
    result = tuple(_identity(value, role) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{role} must not contain duplicate identities.")
    return tuple(sorted(result))


def _metadata_items(
    metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None, /
) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    items = tuple(metadata.items()) if isinstance(metadata, Mapping) else tuple(metadata)
    normalized: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Metadata must contain key/value pairs.")
        key, value = item
        if not isinstance(key, str) or _METADATA_KEY.fullmatch(key) is None:
            raise ValueError("Metadata keys must be canonical lower_snake_case tokens.")
        if key not in CARDIOVASCULAR_CASE_METADATA_KEYS:
            raise ValueError(
                f"Metadata key {key!r} is not in the non-identifying allowlist."
            )
        if not isinstance(value, str) or _METADATA_VALUE.fullmatch(value) is None:
            raise ValueError("Metadata values must be bounded canonical policy tokens.")
        if (
            _contains_phi_marker(key)
            or _contains_phi_marker(value)
            or _EMAIL.search(value)
            or _PHONE_OR_SSN.search(value)
            or _DATE.search(value)
        ):
            raise ValueError(
                "Cardiovascular case metadata must not contain PHI or linkable identifiers."
            )
        normalized.append((key, value))
    keys = tuple(key for key, _ in normalized)
    if len(keys) != len(set(keys)):
        raise ValueError("Cardiovascular case metadata keys must be unique.")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True, init=False)
class CardiovascularCaseManifest:
    """Host-only immutable identity binding for a cardiovascular case definition."""

    case_id: str
    anatomy_id: str
    model_id: str
    protocol_id: str
    observation_ids: tuple[str, ...]
    support_profile_id: str
    release_id: str
    build_id: str
    sbom_id: str
    license_ids: tuple[str, ...]
    data_rights_ids: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]
    manifest_id: str = field(init=False)

    def __init__(
        self,
        case_id: str,
        anatomy_id: str,
        model_id: str,
        protocol_id: str,
        support_profile_id: str,
        release_id: str,
        build_id: str,
        sbom_id: str,
        observation_ids: Iterable[str] = (),
        license_ids: Iterable[str] = (),
        data_rights_ids: Iterable[str] = (),
        metadata: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
    ):
        case = _identity(case_id, "case_id")
        anatomy = _identity(anatomy_id, "anatomy_id")
        model = _identity(model_id, "model_id")
        protocol = _identity(protocol_id, "protocol_id")
        support_profile = _identity(support_profile_id, "support_profile_id")
        release = _identity(release_id, "release_id")
        build = _identity(build_id, "build_id")
        sbom = _identity(sbom_id, "sbom_id")
        observations = _identity_tuple(observation_ids, "observation_ids")
        licenses = _identity_tuple(license_ids, "license_ids")
        rights = _identity_tuple(data_rights_ids, "data_rights_ids")
        metadata_items = _metadata_items(metadata)

        bound_ids = (
            case,
            anatomy,
            model,
            protocol,
            support_profile,
            release,
            build,
            sbom,
            *observations,
            *licenses,
            *rights,
        )
        if len(bound_ids) != len(set(bound_ids)):
            raise ValueError(
                "Every identity bound by a cardiovascular case must be unique."
            )

        manifest_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-case-manifest",
                "case_id": case,
                "anatomy_id": anatomy,
                "model_id": model,
                "protocol_id": protocol,
                "observation_ids": list(observations),
                "support_profile_id": support_profile,
                "release_id": release,
                "build_id": build,
                "sbom_id": sbom,
                "license_ids": list(licenses),
                "data_rights_ids": list(rights),
                "metadata": dict(metadata_items),
            }
        )
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "anatomy_id", anatomy)
        object.__setattr__(self, "model_id", model)
        object.__setattr__(self, "protocol_id", protocol)
        object.__setattr__(self, "observation_ids", observations)
        object.__setattr__(self, "support_profile_id", support_profile)
        object.__setattr__(self, "release_id", release)
        object.__setattr__(self, "build_id", build)
        object.__setattr__(self, "sbom_id", sbom)
        object.__setattr__(self, "license_ids", licenses)
        object.__setattr__(self, "data_rights_ids", rights)
        object.__setattr__(self, "metadata", metadata_items)
        object.__setattr__(self, "manifest_id", manifest_id)

    @property
    def content_id(self) -> str:
        """Return the manifest's deterministic content identity."""
        return self.manifest_id

    @property
    def metadata_mapping(self) -> Mapping[str, str]:
        """Return an immutable mapping view of canonical non-identifying metadata."""
        return MappingProxyType(dict(self.metadata))


__all__ = [
    "CARDIOVASCULAR_CASE_METADATA_KEYS",
    "CardiovascularCaseManifest",
]

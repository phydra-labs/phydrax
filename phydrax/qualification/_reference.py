#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_CHECKSUM_LENGTHS = {"sha256": 64, "sha384": 96, "sha512": 128}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _strict_bool(value: bool, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(normalized))


def _finite_mapping(
    values: Mapping[str, int | float],
    name: str,
    /,
    *,
    positive: bool,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError(f"{name} must be a non-empty mapping.")
    normalized: list[tuple[str, float]] = []
    for coordinate, value in values.items():
        coordinate_ = _identifier(coordinate, f"{name} coordinate")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be real numbers.")
        value_ = float(value)
        if not math.isfinite(value_) or (value_ <= 0.0 if positive else value_ < 0.0):
            condition = "positive" if positive else "non-negative"
            raise ValueError(f"{name} values must be finite and {condition}.")
        value_ = abs(value_)
        normalized.append((coordinate_, value_))
    normalized.sort()
    if len({coordinate for coordinate, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} coordinates must be unique.")
    return tuple(normalized)


class ReferenceArtifactManifest(StrictModule, NonTrainableState):
    """Offline-verifiable identity, rights, and scientific lineage for reference data."""

    artifact_name: str = eqx.field(static=True)
    checksum_algorithm: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    size_bytes: int = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    commercial_use_permitted: bool = eqx.field(static=True)
    redistribution_permitted: bool = eqx.field(static=True)
    training_use_permitted: bool = eqx.field(static=True)
    export_permitted: bool = eqx.field(static=True)
    export_classification: str = eqx.field(static=True)
    nondimensionalization: tuple[tuple[str, float], ...] = eqx.field(static=True)
    uncertainty: tuple[tuple[str, float], ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        artifact_name: str,
        /,
        *,
        checksum_algorithm: str,
        checksum: str,
        size_bytes: int,
        license_id: str,
        commercial_use_permitted: bool,
        redistribution_permitted: bool,
        training_use_permitted: bool,
        export_permitted: bool,
        export_classification: str,
        nondimensionalization: Mapping[str, int | float],
        uncertainty: Mapping[str, int | float],
        lineage_ids: Sequence[str],
    ):
        algorithm = _identifier(checksum_algorithm, "checksum algorithm").lower()
        if algorithm not in _CHECKSUM_LENGTHS:
            raise ValueError("Checksum algorithm must be sha256, sha384, or sha512.")
        digest = _identifier(checksum, "checksum").lower()
        if len(digest) != _CHECKSUM_LENGTHS[algorithm] or any(
            character not in _HEX_DIGITS for character in digest
        ):
            raise ValueError(
                f"{algorithm} checksum must be exactly "
                f"{_CHECKSUM_LENGTHS[algorithm]} hexadecimal characters."
            )
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise TypeError("size_bytes must be an integer.")
        if size_bytes <= 0:
            raise ValueError("Reference artifact size must be positive.")
        self.artifact_name = _identifier(artifact_name, "reference artifact name")
        self.checksum_algorithm = algorithm
        self.checksum = digest
        self.size_bytes = size_bytes
        self.license_id = _identifier(license_id, "reference artifact license ID")
        self.commercial_use_permitted = _strict_bool(
            commercial_use_permitted, "commercial_use_permitted"
        )
        self.redistribution_permitted = _strict_bool(
            redistribution_permitted, "redistribution_permitted"
        )
        self.training_use_permitted = _strict_bool(
            training_use_permitted, "training_use_permitted"
        )
        self.export_permitted = _strict_bool(export_permitted, "export_permitted")
        self.export_classification = _identifier(
            export_classification, "export classification"
        )
        self.nondimensionalization = _finite_mapping(
            nondimensionalization,
            "nondimensionalization scales",
            positive=True,
        )
        self.uncertainty = _finite_mapping(
            uncertainty,
            "reference uncertainty",
            positive=False,
        )
        self.lineage_ids = _identifiers(lineage_ids, "reference lineage IDs")
        self.manifest_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "reference-artifact-manifest",
            "artifact_name": self.artifact_name,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "license_id": self.license_id,
            "commercial_use_permitted": self.commercial_use_permitted,
            "redistribution_permitted": self.redistribution_permitted,
            "training_use_permitted": self.training_use_permitted,
            "export_permitted": self.export_permitted,
            "export_classification": self.export_classification,
            "nondimensionalization": dict(self.nondimensionalization),
            "uncertainty": dict(self.uncertainty),
            "lineage_ids": list(self.lineage_ids),
        }

    def to_record(self) -> dict[str, object]:
        """Return the deterministic manifest without reading the referenced artifact."""
        return {**self._content_record(), "manifest_id": self.manifest_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ReferenceArtifactManifest:
        """Reconstruct and content-verify an offline reference manifest."""
        if not isinstance(record, Mapping):
            raise TypeError("Reference-artifact manifest record must be a mapping.")
        nondimensionalization = record["nondimensionalization"]
        uncertainty = record["uncertainty"]
        lineage_ids = record["lineage_ids"]
        if not isinstance(nondimensionalization, Mapping) or not isinstance(
            uncertainty, Mapping
        ):
            raise TypeError(
                "Serialized nondimensionalization and uncertainty must be mappings."
            )
        if not isinstance(lineage_ids, Sequence) or isinstance(lineage_ids, str):
            raise TypeError("Serialized reference lineage IDs must be a sequence.")
        value = cls(
            record["artifact_name"],
            checksum_algorithm=record["checksum_algorithm"],
            checksum=record["checksum"],
            size_bytes=record["size_bytes"],
            license_id=record["license_id"],
            commercial_use_permitted=record["commercial_use_permitted"],
            redistribution_permitted=record["redistribution_permitted"],
            training_use_permitted=record["training_use_permitted"],
            export_permitted=record["export_permitted"],
            export_classification=record["export_classification"],
            nondimensionalization=nondimensionalization,
            uncertainty=uncertainty,
            lineage_ids=tuple(lineage_ids),
        )
        recorded_id = record.get("manifest_id")
        if recorded_id is not None and str(recorded_id) != value.manifest_id:
            raise ValueError(
                "Serialized reference-artifact manifest has an invalid content address."
            )
        return value

    def rights_refusal_reasons(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ) -> tuple[str, ...]:
        """Return exact requested-use refusals without network or filesystem access."""
        commercial = _strict_bool(commercial_use, "commercial_use")
        redistribute = _strict_bool(redistribution, "redistribution")
        training = _strict_bool(training_use, "training_use")
        export_ = _strict_bool(export, "export")
        reasons: list[str] = []
        if commercial and not self.commercial_use_permitted:
            reasons.append("commercial-use-not-permitted")
        if redistribute and not self.redistribution_permitted:
            reasons.append("redistribution-not-permitted")
        if training and not self.training_use_permitted:
            reasons.append("training-use-not-permitted")
        if export_ and not self.export_permitted:
            reasons.append(f"export-not-permitted:{self.export_classification}")
        return tuple(reasons)

    def require_rights(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ) -> str:
        """Require all rights requested for a use and return the governed manifest ID."""
        reasons = self.rights_refusal_reasons(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        if reasons:
            raise PermissionError(
                f"Reference artifact {self.manifest_id} is not admissible: "
                + "; ".join(reasons)
            )
        return self.manifest_id


__all__ = ["ReferenceArtifactManifest"]

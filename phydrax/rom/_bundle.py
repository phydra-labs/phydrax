#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import equinox as eqx

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ROMArtifactReference(StrictModule, NonTrainableState):
    role: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)
    archive_uri: str = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        role: str,
        artifact_id: str,
        recipe_id: str,
        archive_uri: str,
        /,
        *,
        required: bool = True,
    ):
        values = tuple(
            str(value) for value in (role, artifact_id, recipe_id, archive_uri)
        )
        if any(not value for value in values):
            raise ValueError("ROM artifact reference values must be non-empty.")
        self.role, self.artifact_id, self.recipe_id, self.archive_uri = values
        self.required = bool(required)
        self.reference_id = canonical_fingerprint(
            {
                "kind": "rom-artifact-reference",
                "role": values[0],
                "artifact": values[1],
                "recipe": values[2],
                "uri": values[3],
                "required": self.required,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "recipe_id": self.recipe_id,
            "archive_uri": self.archive_uri,
            "required": self.required,
            "reference_id": self.reference_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> "ROMArtifactReference":
        value = cls(
            str(record["role"]),
            str(record["artifact_id"]),
            str(record["recipe_id"]),
            str(record["archive_uri"]),
            required=bool(record["required"]),
        )
        if value.reference_id != record["reference_id"]:
            raise ValueError("ROM artifact reference identity mismatch.")
        return value


class ROMDeploymentBundle(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)
    capability_profile_ids: tuple[str, ...] = eqx.field(static=True)
    references: tuple[ROMArtifactReference, ...] = eqx.field(static=True)
    build_provenance_id: str = eqx.field(static=True)
    execution_requirements_id: str = eqx.field(static=True)
    resource_policy_id: str = eqx.field(static=True)
    qualification_ids: tuple[str, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        references: Sequence[ROMArtifactReference],
        /,
        *,
        capability_profile_ids: Sequence[str],
        build_provenance_id: str,
        execution_requirements_id: str,
        resource_policy_id: str,
        qualification_ids: Sequence[str],
    ):
        model = str(model_id)
        refs = tuple(references)
        capabilities = tuple(str(value) for value in capability_profile_ids)
        qualifications = tuple(str(value) for value in qualification_ids)
        identifiers = tuple(
            str(value)
            for value in (
                build_provenance_id,
                execution_requirements_id,
                resource_policy_id,
            )
        )
        if (
            not model
            or not refs
            or any(not isinstance(value, ROMArtifactReference) for value in refs)
            or not capabilities
            or not qualifications
            or any(not value for value in (*capabilities, *qualifications, *identifiers))
        ):
            raise ValueError("ROM deployment bundle identities must be complete.")
        roles = tuple(value.role for value in refs)
        if len(set(roles)) != len(roles):
            raise ValueError("ROM deployment artifact roles must be unique.")
        self.model_id = model
        self.capability_profile_ids = capabilities
        self.references = refs
        self.build_provenance_id = identifiers[0]
        self.execution_requirements_id = identifiers[1]
        self.resource_policy_id = identifiers[2]
        self.qualification_ids = qualifications
        self.bundle_id = canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "rom-deployment-bundle",
            "model_id": self.model_id,
            "capability_profile_ids": list(self.capability_profile_ids),
            "references": [value.to_record() for value in self.references],
            "build_provenance_id": self.build_provenance_id,
            "execution_requirements_id": self.execution_requirements_id,
            "resource_policy_id": self.resource_policy_id,
            "qualification_ids": list(self.qualification_ids),
        }
        if include_id:
            record["bundle_id"] = self.bundle_id
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> "ROMDeploymentBundle":
        references = record["references"]
        if not isinstance(references, Sequence) or isinstance(references, str):
            raise TypeError("Serialized ROM references must be a sequence.")
        value = cls(
            str(record["model_id"]),
            tuple(ROMArtifactReference.from_record(item) for item in references),
            capability_profile_ids=tuple(
                str(item) for item in record["capability_profile_ids"]
            ),
            build_provenance_id=str(record["build_provenance_id"]),
            execution_requirements_id=str(record["execution_requirements_id"]),
            resource_policy_id=str(record["resource_policy_id"]),
            qualification_ids=tuple(str(item) for item in record["qualification_ids"]),
        )
        if value.bundle_id != record["bundle_id"]:
            raise ValueError("ROM deployment bundle identity mismatch.")
        return value


def write_rom_deployment_bundle(
    path: str | os.PathLike[str],
    bundle: ROMDeploymentBundle,
    /,
) -> Path:
    if not isinstance(bundle, ROMDeploymentBundle):
        raise TypeError("bundle must be ROMDeploymentBundle.")
    return write_array_archive(
        path,
        manifest=bundle.to_record(),
        arrays={},
    )


def read_rom_deployment_bundle(
    path: str | os.PathLike[str],
    /,
) -> ROMDeploymentBundle:
    manifest, arrays = read_array_archive(path)
    if arrays:
        raise ValueError("ROM deployment bundle archive must not contain numeric arrays.")
    if manifest.get("kind") != "rom-deployment-bundle":
        raise ValueError("Archive is not a ROM deployment bundle.")
    record = dict(manifest)
    record.pop("arrays")
    return ROMDeploymentBundle.from_record(record)


__all__ = [
    "ROMArtifactReference",
    "ROMDeploymentBundle",
    "read_rom_deployment_bundle",
    "write_rom_deployment_bundle",
]

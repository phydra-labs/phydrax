#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import equinox as eqx

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


class DifferentiationContract(StrictModule, NonTrainableState):
    upstream_physical_parameters: bool = eqx.field(static=True)
    stored_values: bool = eqx.field(static=True)
    query_coordinates: bool = eqx.field(static=True)
    local_parameters: bool = eqx.field(static=True)
    stochastic_realization: bool = eqx.field(static=True)
    higher_order: bool = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        upstream_physical_parameters: bool,
        stored_values: bool,
        query_coordinates: bool,
        local_parameters: bool,
        stochastic_realization: bool = False,
        higher_order: bool = True,
    ):
        values = tuple(
            bool(value)
            for value in (
                upstream_physical_parameters,
                stored_values,
                query_coordinates,
                local_parameters,
                stochastic_realization,
                higher_order,
            )
        )
        (
            self.upstream_physical_parameters,
            self.stored_values,
            self.query_coordinates,
            self.local_parameters,
            self.stochastic_realization,
            self.higher_order,
        ) = values
        self.contract_id = canonical_fingerprint(
            {"kind": "differentiation-contract", "capabilities": list(values)}
        )

    @classmethod
    def native(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=True,
            stored_values=True,
            query_coordinates=True,
            local_parameters=True,
        )

    @classmethod
    def coordinate_only(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=False,
            stored_values=False,
            query_coordinates=True,
            local_parameters=False,
        )

    @classmethod
    def constant(cls) -> DifferentiationContract:
        return cls(
            upstream_physical_parameters=False,
            stored_values=False,
            query_coordinates=False,
            local_parameters=False,
            higher_order=False,
        )

    @classmethod
    def from_label(cls, label: str, /) -> DifferentiationContract:
        value = str(label).strip()
        if value == "native-parameter":
            return cls.native()
        if value == "coordinate-only":
            return cls.coordinate_only()
        if value == "constant":
            return cls.constant()
        raise ValueError("Unknown differentiation contract label.")

    def meet(self, *others: DifferentiationContract) -> DifferentiationContract:
        contracts = (self, *others)
        return DifferentiationContract(
            upstream_physical_parameters=all(
                value.upstream_physical_parameters for value in contracts
            ),
            stored_values=all(value.stored_values for value in contracts),
            query_coordinates=all(value.query_coordinates for value in contracts),
            local_parameters=any(value.local_parameters for value in contracts),
            stochastic_realization=any(
                value.stochastic_realization for value in contracts
            ),
            higher_order=all(value.higher_order for value in contracts),
        )


class ScientificArtifactEnvelope(StrictModule, NonTrainableState):
    artifact_kind: str = eqx.field(static=True)
    content_digest: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    parent_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    status: str = eqx.field(static=True)
    failure_reason: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        artifact_kind: str,
        content_digest: str,
        producer: str,
        producer_version: str,
        build_id: str,
        license_id: str,
        resource_id: str,
        status: str,
        failure_reason: str = "none",
        parent_artifact_ids: tuple[str, ...] = (),
    ):
        values = tuple(
            str(value).strip()
            for value in (
                artifact_kind,
                content_digest,
                producer,
                producer_version,
                build_id,
                license_id,
                resource_id,
                failure_reason,
            )
        )
        parents = tuple(str(value).strip() for value in parent_artifact_ids)
        status_ = str(status).strip()
        if (
            any(not value for value in values)
            or any(not value for value in parents)
            or status_ not in ("complete", "failed")
            or (status_ == "complete" and values[7] != "none")
        ):
            raise ValueError("Scientific artifact envelope is invalid.")
        (
            self.artifact_kind,
            self.content_digest,
            self.producer,
            self.producer_version,
            self.build_id,
            self.license_id,
            self.resource_id,
            self.failure_reason,
        ) = values
        self.parent_artifact_ids = parents
        self.status = status_
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "scientific-artifact",
                "values": list(values),
                "parents": list(parents),
                "status": status_,
            }
        )


class ArtifactManifest(StrictModule, NonTrainableState):
    artifact_id: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    model: str = eqx.field(static=True)
    coverage: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        artifact_id: str,
        producer: str,
        version: str,
        sha256: str,
        byte_size: int,
        source_uri: str,
        license_id: str,
        model: str,
        coverage: str,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                artifact_id,
                producer,
                version,
                sha256,
                source_uri,
                license_id,
                model,
                coverage,
            )
        )
        size = int(byte_size)
        if (
            any(not value for value in values)
            or len(values[3]) != 64
            or any(character not in "0123456789abcdef" for character in values[3])
            or size < 0
        ):
            raise ValueError("Artifact manifest is invalid.")
        (
            self.artifact_id,
            self.producer,
            self.version,
            self.sha256,
            self.source_uri,
            self.license_id,
            self.model,
            self.coverage,
        ) = values
        self.byte_size = size
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "artifact-manifest",
                "values": list(values),
                "byte_size": size,
            }
        )

    def as_json(self) -> str:
        return json.dumps(
            {
                "artifact_id": self.artifact_id,
                "producer": self.producer,
                "version": self.version,
                "sha256": self.sha256,
                "byte_size": self.byte_size,
                "source_uri": self.source_uri,
                "license_id": self.license_id,
                "model": self.model,
                "coverage": self.coverage,
            },
            sort_keys=True,
        )


__all__ = [
    "ArtifactManifest",
    "DifferentiationContract",
    "ScientificArtifactEnvelope",
]

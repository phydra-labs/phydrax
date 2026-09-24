#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ._differentiation import DerivativeContract
from ._fingerprint import canonical_fingerprint
from ._identity import ArtifactBindingIdentity
from ._strict import StrictModule
from ._trainable import NonTrainableState


class DerivativeEstimatorKind(StrEnum):
    """Estimator used for one declared derivative-support region."""

    UNSUPPORTED = "unsupported"
    ANALYTIC = "analytic"
    AUTOMATIC = "automatic"
    PATHWISE = "pathwise"
    SCORE = "score"
    CUSTOM = "custom"
    SURROGATE = "surrogate"
    PIECEWISE = "piecewise"


class DerivativeEvidence(StrictModule, NonTrainableState):
    """Parameter- and event-level evidence augmenting a derivative contract.

    `contract` declares the derivative surfaces; `estimator` records how the
    declared derivatives are estimated. An `UNSUPPORTED` estimator requires a
    contract that declares no supported surface.
    """

    contract: DerivativeContract
    estimator: DerivativeEstimatorKind = eqx.field(static=True)
    differentiable_parameters: tuple[str, ...] = eqx.field(static=True)
    discrete_parameters: tuple[str, ...] = eqx.field(static=True)
    stopped_events: tuple[str, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        contract: DerivativeContract,
        /,
        *,
        estimator: DerivativeEstimatorKind,
        differentiable_parameters: Sequence[str] = (),
        discrete_parameters: Sequence[str] = (),
        stopped_events: Sequence[str] = (),
        support_id: str,
        evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(contract, DerivativeContract):
            raise TypeError("contract must be DerivativeContract.")
        if not isinstance(estimator, DerivativeEstimatorKind):
            raise TypeError("estimator must be DerivativeEstimatorKind.")

        def identifiers(values: Sequence[str], name: str) -> tuple[str, ...]:
            result = tuple(str(value).strip() for value in values)
            if any(not value for value in result) or len(set(result)) != len(result):
                raise ValueError(f"{name} must contain distinct non-empty values.")
            return result

        differentiable = identifiers(
            differentiable_parameters, "differentiable_parameters"
        )
        discrete = identifiers(discrete_parameters, "discrete_parameters")
        stopped = identifiers(stopped_events, "stopped_events")
        evidence = identifiers(evidence_ids, "evidence_ids")
        support = str(support_id).strip()
        if not support:
            raise ValueError("support_id must be non-empty.")
        if set(differentiable) & set(discrete):
            raise ValueError("A parameter cannot be both differentiable and discrete.")
        if estimator is not DerivativeEstimatorKind.UNSUPPORTED and not differentiable:
            raise ValueError("A supported estimator requires differentiable parameters.")
        if estimator is not DerivativeEstimatorKind.UNSUPPORTED and not evidence:
            raise ValueError("A supported estimator requires evidence.")
        if (
            estimator is DerivativeEstimatorKind.UNSUPPORTED
            and contract.supported_surfaces
        ):
            raise ValueError(
                "An unsupported estimator requires a contract without supported surfaces."
            )
        self.contract = contract
        self.estimator = estimator
        self.differentiable_parameters = differentiable
        self.discrete_parameters = discrete
        self.stopped_events = stopped
        self.support_id = support
        self.evidence_ids = evidence
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "derivative-evidence",
                "contract": contract.contract_id,
                "estimator": estimator.value,
                "differentiable": list(differentiable),
                "discrete": list(discrete),
                "stopped_events": list(stopped),
                "support": support,
                "evidence": list(evidence),
            }
        )


class ScientificArtifactEnvelope(StrictModule, NonTrainableState):
    """Producer, status, and lineage envelope of one scientific artifact.

    `binding` is the complete `ArtifactBindingIdentity` of the model an artifact
    freezes (semantic, numeric, and executable IDs together) or `None` for
    artifacts that do not carry a model.
    """

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
    binding: ArtifactBindingIdentity | None
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
        binding: ArtifactBindingIdentity | None = None,
    ):
        if binding is not None and not isinstance(binding, ArtifactBindingIdentity):
            raise TypeError("binding must be an ArtifactBindingIdentity or None.")
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
        self.binding = binding
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "scientific-artifact",
                "values": list(values),
                "parents": list(parents),
                "status": status_,
                **({} if binding is None else {"binding": binding.binding_id}),
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
    "DerivativeEstimatorKind",
    "DerivativeEvidence",
    "ScientificArtifactEnvelope",
]

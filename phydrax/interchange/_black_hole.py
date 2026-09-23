#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only, rights-aware interchange for black-hole scientific artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import equinox as eqx

from .._artifact_security import (
    admit_external_artifact,
    AdmittedExternalArtifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    ResourceLimits,
    ResourceManifest,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..artifacts import ArtifactManifest
from ._report import (
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
)


BlackHoleArtifactKind: TypeAlias = Literal[
    "field", "image", "visibility", "waveform", "numeric-model"
]
_ARTIFACT_KINDS = frozenset(("field", "image", "visibility", "waveform", "numeric-model"))
_NUMERIC_MODEL_FORMATS = frozenset(("onnx", "safetensors", "npz", "phydrax-ml-artifact"))
_REQUIRED_SEMANTICS: dict[str, frozenset[str]] = {
    "field": frozenset(
        ("chart_id", "coordinate_frame_id", "quantity_id", "topology_id", "unit_id")
    ),
    "image": frozenset(("observable_id", "screen_frame_id", "unit_id")),
    "visibility": frozenset(
        ("baseline_frame_id", "frequency_axis_id", "polarization_basis_id", "unit_id")
    ),
    "waveform": frozenset(
        ("mode_basis_id", "quantity_id", "time_reference_id", "unit_id")
    ),
    "numeric-model": frozenset(
        ("architecture_id", "input_schema_id", "output_schema_id", "precision_id")
    ),
}


def _identifier(value: object, owner: str, /) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{owner} must be a non-empty canonical text identifier.")
    return value


def _sha256(value: object, owner: str, /) -> str:
    digest = _identifier(value, owner)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{owner} must be a lowercase SHA-256 digest.")
    return digest


def _exact_bool(value: object, owner: str, /) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{owner} must be Boolean.")
    return value


def _identifiers(
    values: Sequence[str], owner: str, /, *, nonempty: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{owner} must be a sequence of identifiers.")
    result = tuple(_identifier(value, owner) for value in values)
    if (nonempty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{owner} must be {'non-empty and ' if nonempty else ''}unique.")
    return tuple(sorted(result))


def _semantic_bindings(
    values: Mapping[str, str] | Sequence[tuple[str, str]], /
) -> tuple[tuple[str, str], ...]:
    raw = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    bindings = tuple(
        (_identifier(name, "Semantic name"), _identifier(value, "Semantic identity"))
        for name, value in raw
    )
    names = tuple(name for name, _ in bindings)
    if len(set(names)) != len(names):
        raise ValueError("Artifact semantic names must be unique.")
    return tuple(sorted(bindings))


class BlackHoleArtifactRights(StrictModule, NonTrainableState):
    """Content-bound license evidence and explicit permitted uses.

    This record is caller-supplied evidence, not a license inference. It grants
    no use not represented by a true permission bit.
    """

    artifact_kind: BlackHoleArtifactKind = eqx.field(static=True)
    source_artifact_id: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    coverage: str = eqx.field(static=True)
    content_sha256: str = eqx.field(static=True)
    size_bytes: int = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    attribution_id: str = eqx.field(static=True)
    commercial_use: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    derivative_use: bool = eqx.field(static=True)
    model_execution: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)

    def __init__(
        self,
        artifact_kind: BlackHoleArtifactKind,
        source_artifact_id: str,
        content_sha256: str,
        size_bytes: int,
        license_id: str,
        source_uri: str,
        attribution_id: str,
        /,
        *,
        producer: str,
        producer_version: str,
        model_id: str,
        coverage: str,
        commercial_use: bool,
        training_use: bool,
        redistribution: bool,
        derivative_use: bool,
        model_execution: bool,
        export: bool,
    ):
        if artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError("Unknown black-hole artifact kind.")
        if type(size_bytes) is not int:
            raise TypeError("Artifact rights byte size must be an exact integer.")
        size = size_bytes
        if size <= 0:
            raise ValueError("Artifact rights require a positive exact byte size.")
        identifiers = (
            _identifier(source_artifact_id, "Source artifact ID"),
            _identifier(producer, "Producer"),
            _identifier(producer_version, "Producer version"),
            _identifier(model_id, "Model ID"),
            _identifier(coverage, "Coverage"),
            _sha256(content_sha256, "Artifact content checksum"),
            _identifier(license_id, "License ID"),
            _identifier(source_uri, "Source URI"),
            _identifier(attribution_id, "Attribution ID"),
        )
        permissions = tuple(
            _exact_bool(value, name)
            for value, name in (
                (commercial_use, "commercial_use"),
                (training_use, "training_use"),
                (redistribution, "redistribution"),
                (derivative_use, "derivative_use"),
                (model_execution, "model_execution"),
                (export, "export"),
            )
        )
        self.artifact_kind = artifact_kind
        (
            self.source_artifact_id,
            self.producer,
            self.producer_version,
            self.model_id,
            self.coverage,
            self.content_sha256,
            self.license_id,
            self.source_uri,
            self.attribution_id,
        ) = identifiers
        self.size_bytes = size
        (
            self.commercial_use,
            self.training_use,
            self.redistribution,
            self.derivative_use,
            self.model_execution,
            self.export,
        ) = permissions
        self.rights_id = canonical_fingerprint(
            {
                "kind": "black-hole-artifact-rights",
                "artifact_kind": artifact_kind,
                "source_artifact_id": self.source_artifact_id,
                "producer": self.producer,
                "producer_version": self.producer_version,
                "model_id": self.model_id,
                "coverage": self.coverage,
                "content_sha256": self.content_sha256,
                "size_bytes": size,
                "license_id": self.license_id,
                "source_uri": self.source_uri,
                "attribution_id": self.attribution_id,
                "permissions": list(permissions),
            }
        )

    def require(self, policy: BlackHoleArtifactUsePolicy, /) -> None:
        if not isinstance(policy, BlackHoleArtifactUsePolicy):
            raise TypeError("Artifact use requires BlackHoleArtifactUsePolicy.")
        if self.license_id not in policy.accepted_license_ids:
            raise PermissionError("Artifact license is outside the explicit use policy.")
        requested = (
            (policy.commercial_use, self.commercial_use, "commercial use"),
            (policy.training_use, self.training_use, "training use"),
            (policy.redistribution, self.redistribution, "redistribution"),
            (policy.derivative_use, self.derivative_use, "derivative use"),
            (policy.model_execution, self.model_execution, "model execution"),
            (policy.export, self.export, "export"),
        )
        denied = tuple(
            name for required, granted, name in requested if required and not granted
        )
        if denied:
            raise PermissionError(
                "Artifact rights do not permit requested " + ", ".join(denied) + "."
            )


def _artifact_manifest(
    rights: BlackHoleArtifactRights, artifact_kind: BlackHoleArtifactKind, /
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=rights.source_artifact_id,
        producer=rights.producer,
        version=rights.producer_version,
        sha256=rights.content_sha256,
        byte_size=rights.size_bytes,
        source_uri=rights.source_uri,
        license_id=rights.license_id,
        model=rights.model_id,
        coverage=rights.coverage,
    )


class BlackHoleArtifactUsePolicy(StrictModule, NonTrainableState):
    """Exact intended-use admission applied after byte identity is verified."""

    intended_use: str = eqx.field(static=True)
    accepted_license_ids: tuple[str, ...] = eqx.field(static=True)
    commercial_use: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    derivative_use: bool = eqx.field(static=True)
    model_execution: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        intended_use: str,
        accepted_license_ids: Sequence[str],
        /,
        *,
        commercial_use: bool = False,
        training_use: bool = False,
        redistribution: bool = False,
        derivative_use: bool = False,
        model_execution: bool = False,
        export: bool = False,
    ):
        intended = _identifier(intended_use, "Intended use")
        licenses = _identifiers(
            accepted_license_ids, "Accepted license IDs", nonempty=True
        )
        permissions = tuple(
            _exact_bool(value, name)
            for value, name in (
                (commercial_use, "commercial_use"),
                (training_use, "training_use"),
                (redistribution, "redistribution"),
                (derivative_use, "derivative_use"),
                (model_execution, "model_execution"),
                (export, "export"),
            )
        )
        self.intended_use = intended
        self.accepted_license_ids = licenses
        (
            self.commercial_use,
            self.training_use,
            self.redistribution,
            self.derivative_use,
            self.model_execution,
            self.export,
        ) = permissions
        self.policy_id = canonical_fingerprint(
            {
                "kind": "black-hole-artifact-use-policy",
                "intended_use": intended,
                "accepted_license_ids": list(licenses),
                "permissions": list(permissions),
            }
        )


class BlackHoleArtifactSchema(StrictModule, NonTrainableState):
    """Neutral semantic bindings for one non-executable artifact representation."""

    artifact_kind: BlackHoleArtifactKind = eqx.field(static=True)
    source_format: str = eqx.field(static=True)
    semantic_bindings: tuple[tuple[str, str], ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        artifact_kind: BlackHoleArtifactKind,
        source_format: str,
        semantic_bindings: Mapping[str, str] | Sequence[tuple[str, str]],
        /,
    ):
        if artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError("Unknown black-hole artifact kind.")
        source = _identifier(source_format, "Source format").lower()
        if artifact_kind == "numeric-model" and source not in _NUMERIC_MODEL_FORMATS:
            raise ValueError(
                "Numeric models must use ONNX, SafeTensors, NPZ, or the native pickle-free ML artifact."
            )
        bindings = _semantic_bindings(semantic_bindings)
        actual = frozenset(name for name, _ in bindings)
        required = _REQUIRED_SEMANTICS[artifact_kind]
        missing = tuple(sorted(required - actual))
        if missing:
            raise ValueError(
                "Neutral artifact mapping is missing required semantics: "
                + ", ".join(missing)
            )
        self.artifact_kind = artifact_kind
        self.source_format = source
        self.semantic_bindings = bindings
        self.schema_id = canonical_fingerprint(
            {
                "kind": "black-hole-neutral-artifact-schema",
                "artifact_kind": artifact_kind,
                "source_format": source,
                "semantic_bindings": [list(item) for item in bindings],
            }
        )


class NeutralBlackHoleArtifact(StrictModule, NonTrainableState):
    """Exact local bytes plus neutral semantics, provenance, rights, and losses."""

    data: bytes = eqx.field(static=True)
    admission: AdmittedExternalArtifact = eqx.field(static=True)
    resource: ResourceManifest = eqx.field(static=True)
    rights: BlackHoleArtifactRights = eqx.field(static=True)
    use_policy: BlackHoleArtifactUsePolicy = eqx.field(static=True)
    schema: BlackHoleArtifactSchema = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        resource: BoundedResource,
        admission: AdmittedExternalArtifact,
        rights: BlackHoleArtifactRights,
        use_policy: BlackHoleArtifactUsePolicy,
        schema: BlackHoleArtifactSchema,
        report: AdapterReport,
        artifact_id: str,
        /,
    ):
        if not isinstance(resource, BoundedResource) or not isinstance(
            admission, AdmittedExternalArtifact
        ):
            raise TypeError(
                "resource and admission must be verified external-artifact records."
            )
        if (
            not isinstance(rights, BlackHoleArtifactRights)
            or not isinstance(use_policy, BlackHoleArtifactUsePolicy)
            or not isinstance(schema, BlackHoleArtifactSchema)
            or not isinstance(report, AdapterReport)
        ):
            raise TypeError("Neutral artifact contracts have incompatible types.")
        artifact = _identifier(artifact_id, "Neutral artifact ID")
        expected_artifact = canonical_fingerprint(
            {
                "kind": "neutral-black-hole-artifact",
                "content_sha256": rights.content_sha256,
                "size_bytes": rights.size_bytes,
                "rights": rights.rights_id,
                "use_policy": use_policy.policy_id,
                "schema": schema.schema_id,
                "losses": [loss.loss_id for loss in report.losses],
                "waivers": [waiver.waiver_id for waiver in report.waivers],
                "preserved_fields": list(report.preserved_fields),
            }
        )
        expected_coordinates = tuple(
            f"{name}={value}" for name, value in schema.semantic_bindings
        )
        if (
            hashlib.sha256(resource.data).hexdigest() != resource.manifest.content_sha256
            or len(resource.data) != resource.manifest.size_bytes
            or admission.sha256 != rights.content_sha256
            or admission.byte_size != rights.size_bytes
            or admission.license_id != rights.license_id
            or admission.manifest_id
            != _artifact_manifest(rights, schema.artifact_kind).manifest_id
            or rights.artifact_kind != schema.artifact_kind
            or resource.manifest.content_sha256 != rights.content_sha256
            or resource.manifest.size_bytes != rights.size_bytes
            or not report.valid
            or report.source_id != rights.content_sha256
            or report.target_id != artifact
            or report.stage != f"black-hole-{schema.artifact_kind}-mapping"
            or report.source_format != schema.source_format
            or report.target_format != "phydrax-neutral-black-hole-artifact"
            or report.coordinate_mapping != expected_coordinates
            or bool(report.assumptions)
            or bool(report.requirements)
            or bool(report.capabilities)
            or bool(report.stages)
            or artifact != expected_artifact
        ):
            raise ValueError(
                "Neutral artifact contracts do not bind the same valid bytes."
            )
        rights.require(use_policy)
        self.data = resource.data
        self.resource = resource.manifest
        self.admission = admission
        self.rights = rights
        self.use_policy = use_policy
        self.schema = schema
        self.report = report
        self.artifact_id = artifact


def map_black_hole_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    artifact_kind: BlackHoleArtifactKind,
    source_format: str,
    semantic_bindings: Mapping[str, str] | Sequence[tuple[str, str]],
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    """Map checksum-pinned local bytes without decoding or executing their format."""

    if (
        not isinstance(limits, ResourceLimits)
        or not isinstance(rights, BlackHoleArtifactRights)
        or not isinstance(use_policy, BlackHoleArtifactUsePolicy)
    ):
        raise TypeError(
            "Artifact mapping requires resource limits, content-bound rights, and a use policy."
        )
    relative = Path(path)
    if len(relative.parts) > limits.max_depth:
        raise ValueError("Artifact path nesting exceeds the configured bound.")
    suffix = relative.suffix.lower()
    if not suffix:
        raise ValueError("External artifacts require an explicit non-pickle file suffix.")
    security_policy = ExternalArtifactPolicy(
        trusted_root,
        maximum_bytes=limits.max_bytes,
        allowed_license_ids=use_policy.accepted_license_ids,
        allowed_suffixes=(suffix,),
    )
    source_manifest = _artifact_manifest(rights, artifact_kind)
    admission = admit_external_artifact(path, source_manifest, policy=security_policy)
    data = read_admitted_artifact(admission, source_manifest, policy=security_policy)
    resource = bounded_resource_from_bytes(
        data, limits=limits, source_path=admission.relative_path
    )
    # Generic admission checks size and SHA-256 before license or semantics.
    if rights.artifact_kind != artifact_kind:
        raise ValueError("Artifact rights do not cover the requested artifact kind.")
    rights.require(use_policy)
    schema = BlackHoleArtifactSchema(artifact_kind, source_format, semantic_bindings)
    losses_ = tuple(losses)
    waivers_ = tuple(waivers)
    if any(not isinstance(value, AdapterLoss) for value in losses_):
        raise TypeError("losses must contain AdapterLoss values.")
    if any(not isinstance(value, AdapterWaiver) for value in waivers_):
        raise TypeError("waivers must contain AdapterWaiver values.")
    losses_ = tuple(sorted(losses_, key=lambda value: value.loss_id))
    waivers_ = tuple(sorted(waivers_, key=lambda value: value.waiver_id))
    if len(losses_) > limits.max_losses:
        raise ValueError("Artifact mapping losses exceed the configured bound.")
    resource = account_bounded_resource(
        resource,
        depth=1,
        nodes=1 + len(schema.semantic_bindings),
        attributes=len(schema.semantic_bindings),
        losses=len(losses_),
    )
    preserved = _identifiers(preserved_fields, "Preserved fields")
    artifact_id = canonical_fingerprint(
        {
            "kind": "neutral-black-hole-artifact",
            "content_sha256": rights.content_sha256,
            "size_bytes": rights.size_bytes,
            "rights": rights.rights_id,
            "use_policy": use_policy.policy_id,
            "schema": schema.schema_id,
            "losses": [loss.loss_id for loss in losses_],
            "waivers": [waiver.waiver_id for waiver in waivers_],
            "preserved_fields": list(preserved),
        }
    )
    status = AdapterStatus.LOSSLESS if not losses_ else AdapterStatus.DECLARED_LOSS
    source_profile = AdapterFormatProfile(
        schema.source_format, qualifiers={"artifact_kind": artifact_kind}
    )
    target_profile = AdapterFormatProfile(
        "phydrax-neutral-black-hole-artifact",
        qualifiers={"artifact_kind": artifact_kind},
    )
    report = AdapterReport(
        status,
        schema.source_format,
        target_profile.format,
        source_id=rights.content_sha256,
        target_id=artifact_id,
        coordinate_mapping=tuple(
            f"{name}={value}" for name, value in schema.semantic_bindings
        ),
        preserved_fields=preserved,
        losses=losses_,
        waivers=waivers_,
        stage=f"black-hole-{artifact_kind}-mapping",
        source_profile=source_profile,
        target_profile=target_profile,
    )
    if not report.valid:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Artifact mapping has unwaived interpretation-changing loss.",
        )
    return NeutralBlackHoleArtifact(
        resource, admission, rights, use_policy, schema, report, artifact_id
    )


def map_field_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    source_format: str,
    chart_id: str,
    coordinate_frame_id: str,
    quantity_id: str,
    topology_id: str,
    unit_id: str,
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    return map_black_hole_artifact(
        path,
        trusted_root=trusted_root,
        limits=limits,
        rights=rights,
        use_policy=use_policy,
        artifact_kind="field",
        source_format=source_format,
        semantic_bindings={
            "chart_id": chart_id,
            "coordinate_frame_id": coordinate_frame_id,
            "quantity_id": quantity_id,
            "topology_id": topology_id,
            "unit_id": unit_id,
        },
        preserved_fields=preserved_fields,
        losses=losses,
        waivers=waivers,
    )


def map_image_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    source_format: str,
    observable_id: str,
    screen_frame_id: str,
    unit_id: str,
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    return map_black_hole_artifact(
        path,
        trusted_root=trusted_root,
        limits=limits,
        rights=rights,
        use_policy=use_policy,
        artifact_kind="image",
        source_format=source_format,
        semantic_bindings={
            "observable_id": observable_id,
            "screen_frame_id": screen_frame_id,
            "unit_id": unit_id,
        },
        preserved_fields=preserved_fields,
        losses=losses,
        waivers=waivers,
    )


def map_visibility_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    source_format: str,
    baseline_frame_id: str,
    frequency_axis_id: str,
    polarization_basis_id: str,
    unit_id: str,
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    return map_black_hole_artifact(
        path,
        trusted_root=trusted_root,
        limits=limits,
        rights=rights,
        use_policy=use_policy,
        artifact_kind="visibility",
        source_format=source_format,
        semantic_bindings={
            "baseline_frame_id": baseline_frame_id,
            "frequency_axis_id": frequency_axis_id,
            "polarization_basis_id": polarization_basis_id,
            "unit_id": unit_id,
        },
        preserved_fields=preserved_fields,
        losses=losses,
        waivers=waivers,
    )


def map_waveform_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    source_format: str,
    mode_basis_id: str,
    quantity_id: str,
    time_reference_id: str,
    unit_id: str,
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    return map_black_hole_artifact(
        path,
        trusted_root=trusted_root,
        limits=limits,
        rights=rights,
        use_policy=use_policy,
        artifact_kind="waveform",
        source_format=source_format,
        semantic_bindings={
            "mode_basis_id": mode_basis_id,
            "quantity_id": quantity_id,
            "time_reference_id": time_reference_id,
            "unit_id": unit_id,
        },
        preserved_fields=preserved_fields,
        losses=losses,
        waivers=waivers,
    )


def map_numeric_model_artifact(
    path,
    /,
    *,
    trusted_root,
    limits: ResourceLimits,
    rights: BlackHoleArtifactRights,
    use_policy: BlackHoleArtifactUsePolicy,
    source_format: str,
    architecture_id: str,
    input_schema_id: str,
    output_schema_id: str,
    precision_id: str,
    preserved_fields: Sequence[str] = (),
    losses: Sequence[AdapterLoss] = (),
    waivers: Sequence[AdapterWaiver] = (),
) -> NeutralBlackHoleArtifact:
    """Map inert numeric model bytes; this function never deserializes or executes them."""

    return map_black_hole_artifact(
        path,
        trusted_root=trusted_root,
        limits=limits,
        rights=rights,
        use_policy=use_policy,
        artifact_kind="numeric-model",
        source_format=source_format,
        semantic_bindings={
            "architecture_id": architecture_id,
            "input_schema_id": input_schema_id,
            "output_schema_id": output_schema_id,
            "precision_id": precision_id,
        },
        preserved_fields=preserved_fields,
        losses=losses,
        waivers=waivers,
    )


__all__ = [
    "BlackHoleArtifactKind",
    "BlackHoleArtifactRights",
    "BlackHoleArtifactSchema",
    "BlackHoleArtifactUsePolicy",
    "NeutralBlackHoleArtifact",
    "map_black_hole_artifact",
    "map_field_artifact",
    "map_image_artifact",
    "map_numeric_model_artifact",
    "map_visibility_artifact",
    "map_waveform_artifact",
]

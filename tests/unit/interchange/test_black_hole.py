#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phydrax.interchange._black_hole import (
    BlackHoleArtifactRights,
    BlackHoleArtifactUsePolicy,
    map_field_artifact,
    map_image_artifact,
    map_numeric_model_artifact,
    map_visibility_artifact,
    map_waveform_artifact,
)
from phydrax.interchange._report import (
    AdapterError,
    AdapterLoss,
    AdapterStatus,
    AdapterWaiver,
)
from phydrax.interchange._resource import ResourceLimits


def _limits(max_bytes: int = 4096, max_losses: int = 4) -> ResourceLimits:
    return ResourceLimits(max_bytes, 4, 32, 16, max_losses)


def _rights(kind, payload, *, checksum=None, size=None, commercial_use=True):
    return BlackHoleArtifactRights(
        kind,
        f"source:{kind}",
        hashlib.sha256(payload).hexdigest() if checksum is None else checksum,
        len(payload) if size is None else size,
        "CC-BY-4.0",
        f"archive:example/{kind}",
        "attribution:example",
        producer="producer:example",
        producer_version="release:1",
        model_id=f"model:{kind}",
        coverage=f"coverage:{kind}:bounded-test",
        commercial_use=commercial_use,
        training_use=kind == "numeric-model",
        redistribution=True,
        derivative_use=True,
        model_execution=kind == "numeric-model",
        export=True,
    )


def _policy(*, license_id="CC-BY-4.0", commercial_use=True, model_execution=False):
    return BlackHoleArtifactUsePolicy(
        "bounded scientific interchange",
        (license_id,),
        commercial_use=commercial_use,
        model_execution=model_execution,
    )


def test_neutral_mappings_bind_all_five_artifact_semantics(tmp_path: Path):
    cases = (
        (
            "field",
            map_field_artifact,
            {
                "source_format": "openpmd-hdf5",
                "chart_id": "chart:cartesian",
                "coordinate_frame_id": "frame:simulation",
                "quantity_id": "quantity:z4c-state",
                "topology_id": "topology:grid-8",
                "unit_id": "units:geometric",
            },
        ),
        (
            "image",
            map_image_artifact,
            {
                "source_format": "fits",
                "observable_id": "observable:stokes-i",
                "screen_frame_id": "frame:camera",
                "unit_id": "units:specific-intensity",
            },
        ),
        (
            "visibility",
            map_visibility_artifact,
            {
                "source_format": "uvfits",
                "baseline_frame_id": "frame:uvw",
                "frequency_axis_id": "frequency:230-ghz",
                "polarization_basis_id": "polarization:linear",
                "unit_id": "units:jansky",
            },
        ),
        (
            "waveform",
            map_waveform_artifact,
            {
                "source_format": "hdf5",
                "mode_basis_id": "basis:spin-weight-minus-two",
                "quantity_id": "quantity:strain",
                "time_reference_id": "time:retarded-geometric",
                "unit_id": "units:dimensionless",
            },
        ),
        (
            "numeric-model",
            map_numeric_model_artifact,
            {
                "source_format": "safetensors",
                "architecture_id": "architecture:surrogate-v2",
                "input_schema_id": "schema:black-hole-parameters",
                "output_schema_id": "schema:waveform-coefficients",
                "precision_id": "precision:float32",
            },
        ),
    )

    mapped = []
    for index, (kind, mapper, semantics) in enumerate(cases):
        payload = f"bounded-{kind}-payload-{index}".encode()
        path = tmp_path / f"artifact-{index}.bin"
        path.write_bytes(payload)
        artifact = mapper(
            path.name,
            trusted_root=tmp_path,
            limits=_limits(),
            rights=_rights(kind, payload),
            use_policy=_policy(model_execution=kind == "numeric-model"),
            preserved_fields=("values",),
            **semantics,
        )
        mapped.append(artifact)
        assert artifact.data == payload
        assert artifact.schema.artifact_kind == kind
        assert artifact.rights.producer == "producer:example"
        assert artifact.rights.producer_version == "release:1"
        assert artifact.rights.coverage == f"coverage:{kind}:bounded-test"
        assert artifact.resource.content_sha256 == hashlib.sha256(payload).hexdigest()
        assert artifact.report.status == AdapterStatus.LOSSLESS

    assert len({artifact.artifact_id for artifact in mapped}) == len(cases)


def test_mapping_fails_closed_on_checksum_size_license_path_and_byte_bounds(
    tmp_path: Path,
):
    payload = b"trusted-field"
    (tmp_path / "field.bin").write_bytes(payload)
    field = {
        "trusted_root": tmp_path,
        "limits": _limits(),
        "source_format": "openpmd-hdf5",
        "chart_id": "chart:cartesian",
        "coordinate_frame_id": "frame:simulation",
        "quantity_id": "quantity:chi",
        "topology_id": "topology:grid",
        "unit_id": "units:dimensionless",
    }

    with pytest.raises(ValueError, match="SHA-256"):
        map_field_artifact(
            "field.bin",
            rights=_rights("field", payload, checksum="0" * 64, commercial_use=False),
            use_policy=_policy(license_id="LicenseRef-Denied"),
            **field,
        )
    with pytest.raises(ValueError, match="size"):
        map_field_artifact(
            "field.bin",
            rights=_rights("field", payload, size=len(payload) + 1),
            use_policy=_policy(),
            **field,
        )
    with pytest.raises(PermissionError, match="license"):
        map_field_artifact(
            "field.bin",
            rights=_rights("field", payload),
            use_policy=_policy(license_id="MIT"),
            **field,
        )
    with pytest.raises(PermissionError, match="commercial use"):
        map_field_artifact(
            "field.bin",
            rights=_rights("field", payload, commercial_use=False),
            use_policy=_policy(commercial_use=True),
            **field,
        )
    outside_name = f"{tmp_path.name}-outside-black-hole.bin"
    (tmp_path.parent / outside_name).write_bytes(payload)
    with pytest.raises(ValueError, match="confined relative path"):
        map_field_artifact(
            f"../{outside_name}",
            rights=_rights("field", payload),
            use_policy=_policy(),
            **field,
        )
    with pytest.raises(ValueError, match="manifest exceeds"):
        map_field_artifact(
            "field.bin",
            rights=_rights("field", payload),
            use_policy=_policy(),
            **{**field, "limits": _limits(max_bytes=4)},
        )


def test_mapping_loss_is_explicit_bounded_and_waived_when_interpretation_changes(
    tmp_path: Path,
):
    payload = b"image"
    (tmp_path / "image.bin").write_bytes(payload)
    common = {
        "trusted_root": tmp_path,
        "limits": _limits(max_losses=1),
        "rights": _rights("image", payload),
        "use_policy": _policy(),
        "source_format": "fits",
        "observable_id": "observable:stokes-i",
        "screen_frame_id": "frame:camera",
        "unit_id": "units:intensity",
    }
    declared = AdapterLoss(
        "metadata.comment",
        "import",
        "dropped",
        "Free-form producer comment has no neutral semantic role.",
        changes_interpretation=False,
    )
    mapped = map_image_artifact("image.bin", losses=(declared,), **common)
    assert mapped.report.status == AdapterStatus.DECLARED_LOSS
    assert mapped.report.losses == (declared,)
    assert mapped.resource.observed_losses == 1

    interpretation = AdapterLoss(
        "screen.orientation",
        "import",
        "transformed",
        "The source screen orientation is changed.",
        changes_interpretation=True,
    )
    with pytest.raises(AdapterError):
        map_image_artifact("image.bin", losses=(interpretation,), **common)
    waiver = AdapterWaiver(
        interpretation, "The consuming analysis requires this orientation."
    )
    admitted = map_image_artifact(
        "image.bin", losses=(interpretation,), waivers=(waiver,), **common
    )
    assert admitted.report.valid
    assert admitted.report.negotiation.waived_losses == (interpretation,)


def test_numeric_model_mapping_rejects_pickle_formats_without_deserializing(
    tmp_path: Path,
):
    payload = b"opaque-model-bytes"
    (tmp_path / "model.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="pickle-free"):
        map_numeric_model_artifact(
            "model.bin",
            trusted_root=tmp_path,
            limits=_limits(),
            rights=_rights("numeric-model", payload),
            use_policy=_policy(model_execution=True),
            source_format="pickle",
            architecture_id="architecture:test",
            input_schema_id="schema:input",
            output_schema_id="schema:output",
            precision_id="precision:float32",
        )

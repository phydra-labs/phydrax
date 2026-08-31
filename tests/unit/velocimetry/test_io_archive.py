#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import zipfile

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.velocimetry.imaging import DenseDisplacementField2D
from phydrax.velocimetry.io import (
    read_learned_piv_artifact,
    read_velocimetry_archive,
    save_learned_piv_artifact,
    write_velocimetry_archive,
)
from phydrax.velocimetry.piv import (
    PIVQuality2D,
    PIVResult,
    PIVRetention,
    PIVStatus2D,
    PIVUncertainty2D,
    ReplacementEvidence2D,
    ValidationEvidence2D,
)
from phydrax.velocimetry.piv._learned_model import (
    CorrelationPyramidPIV,
    LearnedDensePIVPlan,
)


def _piv_result() -> PIVResult:
    row, column = np.meshgrid(np.array([1.0, 3.0]), np.array([2.0, 4.0]), indexing="ij")
    positions = np.stack((row, column), axis=-1)
    displacement = np.array([[[0.0, 0.0], [1.0, -2.0]], [[7.0, 8.0], [-1.0, 3.0]]])
    valid = np.array([[True, True], [False, True]])
    raw = DenseDisplacementField2D(
        positions,
        displacement,
        valid,
        geometry_id="geometry",
        field_id="raw",
        provenance=("pair",),
    )
    validated = DenseDisplacementField2D(
        positions,
        displacement,
        valid,
        geometry_id="geometry",
        field_id="validated",
        provenance=("raw",),
    )
    replaced = DenseDisplacementField2D(
        positions,
        np.where(valid[..., None], displacement, 0.0),
        np.ones_like(valid),
        geometry_id="geometry",
        field_id="replaced",
        provenance=("validated",),
    )
    shape = valid.shape
    quality = PIVQuality2D(
        jnp.ones(shape),
        jnp.full(shape, 0.5),
        jnp.full(shape, 2.0),
        jnp.full(shape, 3.0),
        jnp.ones(shape),
    )
    uncertainty = PIVUncertainty2D(
        jnp.broadcast_to(jnp.eye(2), shape + (2, 2)),
        jnp.asarray(valid),
        "curvature",
    )
    validation = ValidationEvidence2D(
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        jnp.asarray(valid),
        jnp.ones(shape, dtype=jnp.int32),
        jnp.zeros(shape + (2,)),
        jnp.zeros(shape),
        jnp.ones(shape),
        jnp.asarray(valid),
    )
    replacement = ReplacementEvidence2D(
        jnp.asarray(valid),
        jnp.asarray(~valid),
        jnp.where(jnp.asarray(valid), 0, 1),
        jnp.ones(shape, dtype=jnp.int32),
        jnp.zeros(shape, dtype=bool),
    )
    status = PIVStatus2D(
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        jnp.asarray(valid),
        jnp.asarray(~valid),
    )
    retention = PIVRetention(
        jnp.empty((0,)),
        jnp.empty((0,)),
        jnp.empty((0, 2)),
        False,
        "pair",
        "prepared",
        "float32",
        "float32",
        "complex64",
    )
    return PIVResult(
        raw,
        validated,
        replaced,
        quality,
        uncertainty,
        validation,
        replacement,
        status,
        retention,
        "pair",
        "plan",
        "prepared",
        "float32",
        "float32",
        "complex64",
    )


def test_native_piv_archive_round_trip_preserves_all_stages_and_invalid_payload(tmp_path):
    result = _piv_result()
    path = tmp_path / "result.phxv"
    write_velocimetry_archive(
        path,
        result,
        value_kind="piv-result",
        provenance={"experiment": "zero-versus-invalid"},
    )

    restored = read_velocimetry_archive(
        path,
        expected_kind="piv-result",
        expected_type=PIVResult,
    )

    assert np.array_equal(restored.value.raw.valid, result.raw.valid)
    assert np.array_equal(restored.value.raw.displacement_rc, result.raw.displacement_rc)
    assert np.asarray(restored.value.raw.displacement_rc)[0, 0].tolist() == [0.0, 0.0]
    assert bool(np.asarray(restored.value.raw.valid)[0, 0])
    assert not bool(np.asarray(restored.value.raw.valid)[1, 0])
    assert np.asarray(restored.value.raw.displacement_rc)[1, 0].tolist() == [7.0, 8.0]
    assert np.array_equal(
        restored.value.validation_evidence.local_consistency_accepted,
        result.validation_evidence.local_consistency_accepted,
    )
    assert restored.provenance["experiment"] == "zero-versus-invalid"

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert "schema_version" not in manifest
    assert "version" not in manifest


def test_native_archive_rejects_duplicate_or_corrupt_payload_member(tmp_path):
    path = tmp_path / "result.phxv"
    write_velocimetry_archive(path, _piv_result(), value_kind="piv-result")
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("arrays/000000.npy", b"not-the-original-payload")

    with pytest.raises(ArrayArchiveCorruptionError):
        read_velocimetry_archive(path)


def test_learned_artifact_restores_prediction_identity(tmp_path):
    plan = LearnedDensePIVPlan(
        (8, 8),
        level_count=2,
        search_radius=1,
        cost_volume_chunk_size=4,
    )
    model = CorrelationPyramidPIV(
        plan,
        feature_channels=2,
        refinement_channels=3,
        key=jr.key(17),
    )
    first = jnp.linspace(0.0, 1.0, 64).reshape((8, 8, 1))
    second = jnp.roll(first, 1, axis=1)
    prepared = plan.prepare(first, second)
    expected = model(prepared)
    path = tmp_path / "learned-piv.phxv"

    save_learned_piv_artifact(
        path,
        model,
        normalization={"intensity": "unit-interval"},
        training_data_id="synthetic-split:test",
        qualification={"scenario": "translation", "passed": True},
        provenance={"seed": 17},
    )
    artifact = read_learned_piv_artifact(path)
    actual = artifact.model(prepared)

    for expected_level, actual_level in zip(
        expected.displacement_pyramid_rc,
        actual.displacement_pyramid_rc,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_level, expected_level)
    for expected_valid, actual_valid in zip(
        expected.valid_pyramid,
        actual.valid_pyramid,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_valid, expected_valid)
    assert artifact.manifest.training_data_id == "synthetic-split:test"
    assert artifact.manifest.coordinate_convention == "row-down-column-right"

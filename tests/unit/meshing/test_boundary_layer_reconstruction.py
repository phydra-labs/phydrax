"""Stored layer records replay scientific constructors and preserve content."""

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._meshcore import meshcore_available
from phydrax.lifecycle import register_meshing_source_artifacts
from phydrax.lifecycle._meshing_source_records import content_equal
from tests._support.boundary_layer_walls import control, plate, wall
from tests._support.recipes import restore_with_identity


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="layer reconstruction requires meshcore"
)


@pytest.fixture(scope="module", autouse=True)
def registered_artifacts() -> None:
    register_meshing_source_artifacts()


@pytest.fixture(scope="module")
def layers() -> phx.meshing.BoundaryLayerMesh:
    source = wall(*plate(1))
    return phx.meshing.prepare_boundary_layers(source, control(source))


def _with_columns(
    value: phx.meshing.BoundaryLayerMesh, columns: np.ndarray
) -> phx.meshing.BoundaryLayerMesh:
    return phx.meshing.BoundaryLayerMesh(
        value.mesh,
        value.cap,
        value.wall_vertices,
        value.cap_vertices,
        value.layer_index,
        value.validity,
        value.evidence,
        column_index=columns,
        control=value.control,
        source_wall=value.source_wall,
        wall_association=value.wall_association,
        source_domain=value.source_domain,
        policy_id=value.policy_id,
    )


def test_column_integer_inputs_share_stored_content_and_result_identity(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    narrow = _with_columns(layers, np.asarray(layers.column_index, dtype=np.int32))
    wide = _with_columns(layers, np.asarray(layers.column_index, dtype=np.int64))
    assert narrow.result_id == wide.result_id == layers.result_id
    assert narrow.column_index.dtype == np.dtype(np.int64)
    assert wide.column_index.dtype == np.dtype(np.int64)
    assert content_equal(narrow, wide, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_layer_evidence_rebuild_preserves_full_stored_content(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    rebuilt = phx.meshing.BoundaryLayerEvidence.from_stored_fields(layers.evidence)
    assert rebuilt is not layers.evidence
    assert rebuilt.evidence_id == layers.evidence.evidence_id
    assert content_equal(rebuilt, layers.evidence, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_layer_mesh_rebuild_preserves_full_stored_content(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    rebuilt = phx.meshing.BoundaryLayerMesh.from_stored_fields(layers)
    assert rebuilt is not layers
    assert rebuilt.result_id == layers.result_id
    assert content_equal(rebuilt, layers, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)


def test_layer_evidence_rebuild_refuses_forged_support_counts(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    forged = eqx.tree_at(
        lambda evidence: evidence.active_column_counts,
        layers.evidence,
        layers.evidence.active_column_counts + 1,
    )
    with pytest.raises(ValueError, match="BoundaryLayerEvidence.active_column_counts"):
        phx.meshing.BoundaryLayerEvidence.from_stored_fields(forged)


def test_layer_evidence_rebuild_refuses_forged_identity(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    with pytest.raises(ValueError, match="BoundaryLayerEvidence.evidence_id"):
        restore_with_identity(layers.evidence, "evidence_id", "forged-evidence")


def test_layer_mesh_rebuild_refuses_forged_identity(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    with pytest.raises(ValueError, match="BoundaryLayerMesh.result_id"):
        restore_with_identity(layers, "result_id", "forged-layer-mesh")


def test_column_integer_inputs_refuse_lossy_storage_conversion(
    layers: phx.meshing.BoundaryLayerMesh,
) -> None:
    columns = np.asarray(layers.column_index, dtype=np.uint64)
    columns[0] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError, match="BoundaryLayerMesh.column_index"):
        _with_columns(layers, columns)

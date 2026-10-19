from typing import assert_never, Literal

import numpy as np
import pytest

from phydrax._meshcore import (
    exact_orient3d,
    meshcore_available,
    TetMesh3D,
    TetMeshSourceComplex,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)


def _pyramid(
    linear: np.ndarray,
    shift: np.ndarray,
    defect: Literal[
        "none", "source", "gap", "overlap", "orientation", "feature"
    ] = "none",
) -> TetMesh3D:
    points = (
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)),
            dtype=np.float64,
        )
        @ linear.T
        + shift
    )
    cells = np.asarray(((0, 1, 3, 4), (1, 2, 3, 4)), dtype=np.int32)
    q = points[cells]
    negative = exact_orient3d(q[:, 0], q[:, 1], q[:, 2], q[:, 3]) < 0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    faces = np.asarray(
        ((0, 1, 3), (1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4)),
        dtype=np.int32,
    )
    original = np.asarray(
        ((0, 1, 2), (0, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4)),
        dtype=np.int32,
    )
    face_groups = np.asarray((0, 0, 1, 2, 3, 4), dtype=np.int32)
    edges = np.asarray(
        ((0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4)),
        dtype=np.int32,
    )
    edge_groups = np.arange(8, dtype=np.int32)
    match defect:
        case "none":
            pass
        case "source":
            face_groups[1] = 1
        case "gap":
            original = np.delete(original, 1, axis=0)
            face_groups = np.delete(face_groups, 1)
        case "overlap":
            original = np.concatenate((original, original[:1]))
            face_groups = np.concatenate((face_groups, face_groups[:1]))
        case "orientation":
            original[1] = original[1, ::-1]
        case "feature":
            edge_groups[1] = 0
        case _:
            assert_never(defect)
    source = TetMeshSourceComplex(
        points.copy(),
        original,
        face_groups,
        np.zeros(original.shape[0], dtype=np.float64),
        edges.copy(),
        edge_groups,
        np.zeros(8, dtype=np.float64),
        np.zeros(5, dtype=np.int8),
        np.full(5, -1, dtype=np.int32),
        np.zeros((5, 2), dtype=np.float64),
        np.asarray((7, 11, 13, 17, 19), dtype=np.int64),
        np.arange(101, 109, dtype=np.int64),
    )
    return TetMesh3D(
        points,
        cells,
        np.zeros(2, dtype=np.int32),
        faces,
        np.asarray((7, 7, 11, 13, 17, 19), dtype=np.int64),
        edges,
        np.arange(101, 109, dtype=np.int64),
        boundary_policy="conforming",
        max_vertices=32,
        max_tetrahedra=128,
        source=source,
    )


@pytest.mark.parametrize(
    ("linear", "shift"),
    (
        (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
        (
            np.asarray(((0, -2, 0), (0, 0, 2), (2, 0, 0)), dtype=np.float64),
            np.asarray((4, -8, 16), dtype=np.float64),
        ),
        (
            np.diag(np.asarray((-0.5, 0.5, 0.5), dtype=np.float64)),
            np.asarray((-2, 4, 8), dtype=np.float64),
        ),
    ),
)
def test_export_retains_logical_source_coverage_and_primitive_witnesses(
    linear: np.ndarray, shift: np.ndarray
) -> None:
    mesh = _pyramid(linear, shift)
    try:
        arrays = mesh.arrays()
        evidence = mesh.source_evidence()
        assert evidence.face_ancestor_offsets.dtype == np.int64
        assert evidence.face_ancestor_rows.dtype == np.int32
        assert evidence.face_ancestor_offsets.shape == (arrays.faces.shape[0] + 1,)
        assert evidence.face_ancestor_offsets[0] == 0
        assert evidence.face_ancestor_offsets[-1] == evidence.face_ancestor_rows.size == 8
        assert not evidence.face_ancestor_offsets.flags.writeable
        assert not evidence.face_ancestor_rows.flags.writeable
        for row, source in enumerate(arrays.face_sources):
            first, last = evidence.face_ancestor_offsets[row : row + 2]
            contributors = evidence.face_ancestor_rows[first:last]
            expected = (0, 1) if source == 7 else ((11, 13, 17, 19).index(source) + 2,)
            np.testing.assert_array_equal(contributors, expected)
        np.testing.assert_array_equal(
            evidence.segment_ancestors, arrays.segment_sources - 101
        )
        assert np.all(evidence.witness_strata == 1)
        assert np.all((evidence.witness_entities >= 0) & (evidence.witness_entities < 8))
        assert np.all(evidence.witness_deviations == 0)
        assert evidence.requested_bound == evidence.achieved_bound == 0
    finally:
        mesh.close()


@pytest.mark.parametrize("defect", ("source", "gap", "overlap", "orientation", "feature"))
def test_original_source_identity_and_union_defects_are_refused(
    defect: Literal["source", "gap", "overlap", "orientation", "feature"],
) -> None:
    with pytest.raises(ValueError):
        _pyramid(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), defect)

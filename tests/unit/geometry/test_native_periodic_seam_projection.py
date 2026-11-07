"""Native face projection of points whose carrier minimum rounds onto a periodic seam."""

from pathlib import Path

import numpy as np

import phydrax as phx
from phydrax.geometry.brep._projection_contracts import BRepProjectionStatus
from tools.meshing_qualification.scenarios import native_cad_profiles


# Actual straight-P2 seam nodes of the original STEP quadric-routine profile:
# edge midpoints on the transverse cylinder's seam line (z = 0.6), whose
# rounded y (one or one half of -0.6 sin(2 pi)) puts the radial angle just
# across the seam, so its periodic representative rounds onto the 2 pi bound.
_SEAM_NODES = (
    ("-0x1.7000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.a000000000000p-2", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.e000000000000p-2", "0x1.52dc7707ed4d5p-53"),
    ("0x1.6000000000000p-2", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.9000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("0x1.4000000000000p-3", "0x1.52dc7707ed4d5p-53"),
    ("0x1.7fffffffffff8p-4", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.8000000000000p-4", "0x1.52dc7707ed4d5p-53"),
    ("0x1.b000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.5000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("0x1.bfffffffffffcp-3", "0x1.52dc7707ed4d5p-53"),
    ("0x1.3000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.b000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.f000000000000p-1", "0x1.52dc7707ed4d5p-54"),
    ("0x1.1fffffffffffep-2", "0x1.52dc7707ed4d5p-53"),
    ("0x1.8fffffffffffep-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.1000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.2000000000000p-2", "0x1.52dc7707ed4d5p-53"),
    ("0x1.5000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("0x1.fffffffffffe0p-6", "0x1.52dc7707ed4d5p-53"),
    ("0x1.1000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.3000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.0000000000000p-5", "0x1.52dc7707ed4d5p-53"),
    ("0x1.efffffffffffep-1", "0x1.52dc7707ed4d5p-54"),
    ("-0x1.c000000000000p-3", "0x1.52dc7707ed4d5p-53"),
    ("0x1.cfffffffffffep-1", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.d000000000000p-1", "0x1.52dc7707ed4d5p-53"),
    ("0x1.9fffffffffffep-2", "0x1.52dc7707ed4d5p-53"),
    ("0x1.dfffffffffffep-2", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.6000000000000p-2", "0x1.52dc7707ed4d5p-53"),
    ("-0x1.4000000000000p-3", "0x1.52dc7707ed4d5p-53"),
    ("0x1.6fffffffffffep-1", "0x1.52dc7707ed4d5p-53"),
)
_ORIGINAL_QUADRIC_MODEL_ID = (
    "ecc1a3570ae49f8f92993023c6f3be3e05ed8b6309ad7af2f2776405e02d6d2b"
)


def test_original_quadric_seam_nodes_project_to_actual_seam_points(
    tmp_path: Path,
) -> None:
    model, _, _, _ = native_cad_profiles._quadric_case("step", tmp_path)
    assert model.model_id == _ORIGINAL_QUADRIC_MODEL_ID
    face = 2
    assert type(model.patches[face]).__name__ == "RevolutionSurface"
    queries = np.asarray(
        [
            (float.fromhex(x), float.fromhex(y), float.fromhex("0x1.3333333333333p-1"))
            for x, y in _SEAM_NODES
        ]
    )
    projection = phx.geometry.prepare_brep_projection(model)
    count = queries.shape[0]
    result = projection.project(
        queries, np.full((count,), 2, dtype=np.int64), np.full((count,), face)
    )
    status = np.asarray(result.status)
    assert np.all(status == int(BRepProjectionStatus.SEAM))
    points = np.asarray(result.points)
    residuals = np.asarray(result.residuals)
    assert np.all(np.isfinite(points))
    # The proved source point is the actual seam-coedge point (y = 0 on the
    # authored seam line), not the query or a clamped chart point.
    np.testing.assert_array_equal(points[:, 1], 0.0)
    np.testing.assert_allclose(
        points[:, (0, 2)], queries[:, (0, 2)], rtol=0.0, atol=1.0e-15
    )
    # Rigorous distance upper bounds, including rounding slack, far below the
    # original 1e-7 curving residual acceptance bound.
    assert np.all(residuals <= 1.0e-13)
    # Slightly off-seam interior points remain ordinary unique projections.
    inside = queries.copy()
    inside[:, 1] = -1.0e-3
    interior = projection.project(
        inside, np.full((count,), 2, dtype=np.int64), np.full((count,), face)
    )
    assert np.all(np.asarray(interior.status) == int(BRepProjectionStatus.UNIQUE))

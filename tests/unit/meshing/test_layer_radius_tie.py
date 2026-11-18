import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing._boundary_layer import (
    _analyze_wall,
    _build_front,
    _merge_pairs,
    _sector_directions,
    _sectors,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)


def test_merge_radius_encloses_rounded_distance_tie() -> None:
    # Root 2 probes x=-1. Both distances round to 1, although root 0 is
    # strictly positive: expanding only the final box corner would exclude it.
    tiny = 2.0**-54
    points = np.asarray(
        (
            (tiny, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (-2.0, 0.0, 0.0),
            (tiny, 1.0, 0.0),
            (tiny, 0.0, -1.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-2.0, 1.0, 0.0),
            (-2.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 3, 4), (1, 5, 6), (2, 7, 8)), dtype=np.int64)
    source = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("triangles", "triangle", triangles),),
    )
    cells = source.entity_set(2)
    scope = phx.meshing.MeshingScope(
        source.mesh_id,
        source.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )
    control = phx.meshing.BoundaryLayerControl(
        scope,
        phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        smoothing_iterations=0,
    )
    wall = _analyze_wall(
        points,
        np.pad(triangles, ((0, 0), (0, 1)), constant_values=-1),
        np.full(3, 3, dtype=np.int64),
        np.ones(3, dtype=np.bool_),
        control.feature_angle,
    )
    split = np.zeros(wall.edges.shape[0], dtype=np.bool_)
    sectors = _sectors(wall, split)
    directions = _sector_directions(wall, sectors, control)
    front = _build_front(wall, sectors, split, directions, control, source, None)
    candidates = np.zeros(points.shape[0], dtype=np.bool_)
    candidates[:3] = True
    medial = np.zeros(points.shape[0], dtype=np.float64)
    medial[:3] = (1.0, 1.0, 0.5)

    assert tiny + 1.0 == 1.0
    assert np.nextafter(-1.0 + 1.0, np.inf) < tiny
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        _merge_pairs(wall, front, candidates, medial)
    assert failure.value.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
    assert failure.value.entity_ids == (1,)

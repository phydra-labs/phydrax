from __future__ import annotations

import numpy as np

from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry.simplicial._regions import PlanarMeshRegion
from phydrax.meshing._plc_planar_support import planar_region_domain
from phydrax.meshing.providers._native_sources import NativePlanarSource


def test_planar_plc_domain_preserves_outer_and_hole_orientation() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (3.0, 3.0),
            (3.0, 1.0),
        ),
        dtype=np.float64,
    )
    region = PlanarMeshRegion(
        vertices,
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="room-with-courtyard",
    )
    source = NativePlanarSource(
        region,
        "revision-1",
        vertex_entity_ids=np.asarray((101, 102, 103, 104, 201, 202, 203, 204)),
        edge_entity_ids=np.asarray((301, 302, 303, 304, 401, 402, 403, 404)),
    )

    domain = planar_region_domain(source)

    assert isinstance(domain, PiecewiseLinearDomain)
    np.testing.assert_array_equal(domain.vertices, vertices)
    np.testing.assert_array_equal(domain.facets, np.asarray(region.edges))
    np.testing.assert_array_equal(
        domain.facet_regions,
        np.asarray(((0, -1),) * 4 + ((-1, 0),) * 4, dtype=np.int64),
    )
    assert domain.region_ids == (source.region_entity_id,)
    assert domain.source_id == source.source_id

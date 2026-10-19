"""Public carrier construction without a realized query tessellation."""

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import brep_sphere, BRepTessellationPolicy


def test_unrealized_sphere_retains_carriers_and_empty_query_artifact() -> None:
    model = brep_sphere(
        1.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    assert model.geometry is not None
    assert model.topology.num_solids == 1
    assert model.topology.num_faces > 0
    assert model.patches
    assert model.mesh_vertices.shape == (0, 3)
    assert model.mesh_faces.shape == (0, 3)
    assert np.asarray(model.tessellation_deviation_bounds).shape == (0,)
    assert np.asarray(model.tessellation_normal_bounds).shape == (0,)
    assert model.report.num_triangles == 0

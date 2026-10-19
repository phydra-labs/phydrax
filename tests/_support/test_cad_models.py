"""Scientific identity of the shared native-solid catalogs."""

import numpy as np
import pytest

from phydrax.geometry.brep import BRepTessellationPolicy, prepare_brep_query
from tests._support.cad_models import (
    BREP_TEXT_SOLIDS,
    BREP_TEXT_VOLUMES,
    CONTRACT,
    EXACT_VOLUMES,
    geometry_of,
    native_solid,
    SOLIDS,
)


_COARSE = BRepTessellationPolicy(linear_deflection=0.05, angular_deflection=0.4)
# Retain test_native_brep.test_native_solid_measures_match_analytic_values's
# empirical tolerance for the same native analytic solids and tessellation.
_NATIVE_MEASURE_RELATIVE_TOLERANCE = 1e-11
# Preserve test_cad_brep_text's analytic/OCCT volume comparison budget for these
# same source dimensions; changing the fixture owner does not change its oracle.
_BREP_TEXT_VOLUME_RELATIVE_TOLERANCE = 1.0e-6
_GEOMETRY_SOLIDS = (
    ("box", (8, 12, 6, 2), 6.0),
    ("cylinder", (2, 3, 3, 2), 2 * np.pi),
    ("cone", (2, 2, 2, 2), 2 * np.pi / 3),
    ("sphere", (2, 1, 1, 2), 4 * np.pi / 3),
    ("torus", (1, 2, 1, 0), np.pi**2),
    ("extruded_plate", (10, 15, 7, 0), 0.5 * (16 - np.pi)),
    ("revolved_ring", (4, 6, 4, 0), 3 * np.pi),
    ("revolved_quarter", (8, 12, 6, 2), 0.75 * np.pi),
)


@pytest.mark.parametrize(
    "name,counts,volume", _GEOMETRY_SOLIDS, ids=tuple(row[0] for row in _GEOMETRY_SOLIDS)
)
def test_native_solid_geometry_catalog_retains_exact_topology_and_volume(
    name: str,
    counts: tuple[int, int, int, int],
    volume: float,
) -> None:
    model = native_solid(name, catalog="geometry", tessellation=_COARSE)
    geometry = model.geometry
    assert geometry is not None
    assert model.coordinate_contract.spatial_id == CONTRACT.spatial_id
    assert (
        model.topology.num_vertices,
        sum(index != -1 for index in geometry.edge_curves),
        model.topology.num_faces,
        geometry.euler_characteristic(),
    ) == counts
    assert model.topology.num_solids == 1
    balance = geometry.edge_use_balance(model.orientation)
    assert all(
        value == 0
        for value, degenerate in zip(balance, geometry.degenerate_edges, strict=True)
        if not degenerate
    )
    measured = prepare_brep_query(model).measures.solid_volumes[0]
    assert float(measured) == pytest.approx(
        volume, rel=_NATIVE_MEASURE_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", SOLIDS)
def test_native_solid_default_catalog_retains_interchange_dimensions(name: str) -> None:
    model = native_solid(name, tessellation=_COARSE)
    assert float(prepare_brep_query(model).measures.solid_volumes[0]) == pytest.approx(
        EXACT_VOLUMES[name],
        rel=_NATIVE_MEASURE_RELATIVE_TOLERANCE,
    )


def test_native_solid_tessellation_does_not_change_exact_geometry_identity() -> None:
    coarse = native_solid("cylinder", tessellation=_COARSE)
    finer = native_solid(
        "cylinder",
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.01, angular_deflection=0.2
        ),
    )
    assert coarse.geometry is not None and finer.geometry is not None
    assert coarse.geometry.geometry_id == finer.geometry.geometry_id
    assert coarse.model_id == finer.model_id
    assert finer.mesh_faces.shape[0] > coarse.mesh_faces.shape[0]


def test_native_solid_refuses_names_from_another_catalog() -> None:
    with pytest.raises(ValueError, match="cad_models.native_solid.name"):
        native_solid("revolved_quarter")


@pytest.mark.parametrize("name", BREP_TEXT_SOLIDS, ids=BREP_TEXT_SOLIDS)
def test_brep_text_catalog_retains_source_dimensions(name: str) -> None:
    model = native_solid(name, catalog="brep-text", tessellation=_COARSE)
    geometry = geometry_of(model)
    assert model.coordinate_contract.spatial_id == CONTRACT.spatial_id
    assert model.topology.num_solids == 1
    assert geometry.solid_shells
    assert float(prepare_brep_query(model).measures.solid_volumes[0]) == pytest.approx(
        BREP_TEXT_VOLUMES[name], rel=_BREP_TEXT_VOLUME_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", ("plate", "ring", "revolved_quarter"))
def test_brep_text_catalog_refuses_other_catalog_names(name: str) -> None:
    with pytest.raises(ValueError, match="cad_models.native_solid.name"):
        native_solid(name, catalog="brep-text")

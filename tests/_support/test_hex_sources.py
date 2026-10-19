"""Pin independent cut carriers and detect erased oblique chart preparation."""

import numpy as np
import pytest

from phydrax.meshing import MeshingLimits, NativeMappedHexSource, UniformSizeControl
from phydrax.meshing._hex_frame_map import prepare_source_hex_frame
from phydrax.meshing.qualification import (
    curved_occupied_source,
)
from tests._support.coordinate_references import native_meshing_request
from tests._support.hex_sources import (
    pentagonal_cut_prisms,
    sheared_hex_specification,
    sheared_mapped_hex_source,
)


def _assert_oblique_reference(source: NativeMappedHexSource) -> None:
    # The fixture's two unit roots occupy x=1..3, y/z=0..1. The authored map
    # (x,y,z) -> (x+y/2, y+z/4, z) is exactly representable in binary64.
    expected = np.asarray(
        [(x + y / 2, y + z / 4, z) for x in (1, 2, 3) for y in (0, 1) for z in (0, 1)],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(source.domain.reference_mesh.coordinates, expected)


def test_sheared_hex_source_retains_the_independent_oblique_reference() -> None:
    source = sheared_mapped_hex_source()
    _assert_oblique_reference(source)
    assert source.domain.region_ids == ("left", "right")
    np.testing.assert_array_equal(source.domain.cell_regions, (0, 1))


def test_oblique_reference_assertion_detects_erased_shear() -> None:
    source = curved_occupied_source({(1, 0, 0), (2, 0, 0)}, materials=True)
    with pytest.raises(AssertionError):
        _assert_oblique_reference(source)


def test_pentagonal_cut_fixture_retains_layer_coordinates_and_material_ids() -> None:
    mesh, _, domain = pentagonal_cut_prisms(2)
    expected_polygon = ((0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (1.0, 3.0), (0.0, 2.0))
    expected = np.asarray(
        [(x, y, float(z)) for z in range(3) for x, y in expected_polygon],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(mesh.coordinates, expected)
    assert domain.region_ids == ("original-material-0", "original-material-1")
    assert mesh.topology.entities(3).count == 2


def test_sheared_hex_request_retains_declared_materials_interface_and_size_bounds() -> (
    None
):
    source = sheared_mapped_hex_source()
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        prepared = prepare_source_hex_frame(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            scope=scope,
            budget=budget,
        )
        specification = sheared_hex_specification(source, prepared)
    assert specification.target.cell_families.required == ("hexahedron",)
    assert specification.target.geometry_order == 2
    assert {control.material_id for control in specification.region_controls} == {
        "material-left",
        "material-right",
    }
    assert tuple(control.name for control in specification.patch_controls) == (
        "original-material-interface",
    )
    size = specification.size_controls[0]
    assert isinstance(size, UniformSizeControl)
    assert size.target_size == 0.5
    assert size.maximum_size == 1.0
    assert specification.size_compliance.relative_tolerance == 0.95

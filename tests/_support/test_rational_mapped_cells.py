"""Consumer contracts and deliberate corruption checks for rational sources."""

import equinox as eqx
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from tests._support.rational_mapped_cells import cylinder_source, six_quad_source


def _assert_cylinder_controls(controls: NDArray[np.float64]) -> None:
    np.testing.assert_array_equal(
        controls,
        np.asarray(
            [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
            dtype=np.float64,
        ).reshape((6, 3)),
    )


def _assert_quad_partition(
    mesh: CellMesh,
    root: CellGeometrySpec,
    geometry: CellGeometrySpec,
    root_id: str,
) -> None:
    assert sum(block.cell_count for block in mesh.blocks) == 6
    assert all(block.cell_kind == "quadrilateral" for block in mesh.blocks)
    np.testing.assert_array_equal(geometry.coordinates, root.coordinates)
    ancestry = geometry.restriction_source
    assert ancestry is not None
    assert ancestry.source_geometry_id == root_id


def test_cylinder_source_retains_authored_controls_and_root_cell() -> None:
    mesh, geometry, _root, controls = cylinder_source()
    assert mesh.coordinates.shape == (4, 3)
    assert len(mesh.blocks) == 1 and mesh.blocks[0].cell_count == 1
    np.testing.assert_array_equal(mesh.blocks[0].vertices, ((0, 1, 2, 3),))
    _assert_cylinder_controls(controls)
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_cylinder_control_oracle_detects_corruption() -> None:
    _mesh, _geometry, _root, controls = cylinder_source()
    corrupted = controls.copy()
    corrupted[1, 2] += 0.125
    with pytest.raises(AssertionError):
        _assert_cylinder_controls(corrupted)


def test_six_quad_source_retains_root_identity_and_partition() -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    _assert_quad_partition(fine_mesh, geometry, fine_geometry, cell_geometry_id(geometry))


def test_six_quad_coordinate_oracle_detects_corrupted_controls() -> None:
    mesh, geometry, root, controls = cylinder_source()
    fine_mesh, fine_geometry = six_quad_source(mesh, geometry, root, controls)
    corrupted = np.asarray(fine_geometry.coordinates).copy()
    corrupted[1, 2] += 0.125
    changed = eqx.tree_at(lambda value: value.coordinates, fine_geometry, corrupted)
    with pytest.raises(AssertionError):
        _assert_quad_partition(fine_mesh, geometry, changed, cell_geometry_id(geometry))

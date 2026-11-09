#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest
from jax import Array

from phydrax import _meshcore
from phydrax.discretization import CellBlock, CellMesh, lagrange_element
from phydrax.discretization._cell_complex import polyhedral_connectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._mapped_geometry_measure import certified_cell_moments
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.finite_volume._field_view import (
    prepare_finite_volume_field_reconstruction,
)
from phydrax.discretization.finite_volume._mapped_geometry import (
    mapped_constant_quadrature_exact,
)
from phydrax.discretization.finite_volume._reconstruction import (
    PiecewiseConstantReconstruction,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_motion import (
    FixedConnectivityMotionPlan,
)
from tests._support.mixed_material_geometry import nested_material_geometry


pytestmark = pytest.mark.skipif(
    not _meshcore.meshcore_available(), reason="requires phydrax-meshcore"
)


def test_native_restriction_prepares_mapped_fv_geometry() -> None:
    with jax.enable_x64():
        fixture = nested_material_geometry()
        transition = fixture.transition
        source_plan = UnstructuredFiniteVolumePlan.from_cell_mesh(
            transition.source_mesh, coordinate_spec=transition.source_geometry
        )
        target_plan = UnstructuredFiniteVolumePlan.from_cell_mesh(
            transition.target_mesh, coordinate_spec=transition.target_geometry
        )
        source = source_plan.prepare()
        target = target_plan.prepare()
        assert source.cell_quadrature_degree >= 0
        assert source.coordinate_validity.all_certified
        assert target.coordinate_validity.all_certified
        assert source.cell_measure_errors is not None
        assert target.cell_measure_errors is not None
        for region in (0, 1):
            np.testing.assert_allclose(
                np.sum(np.asarray(source.cell_volumes)[fixture.source_regions == region]),
                np.sum(np.asarray(target.cell_volumes)[fixture.target_regions == region]),
                rtol=1e-13,
                atol=0.0,
            )
        for prepared in (source, target):
            np.testing.assert_allclose(
                np.sum(prepared.face_quadrature_weights, axis=1),
                prepared.face_measures,
                rtol=1e-13,
                atol=1e-16,
            )
            # A constant vector flux has face-average density F·A/|face|.
            # The owner/neighbor routing must telescope to each cell's closure.
            vector = np.asarray((0.3, -0.2, 0.7))
            flux = np.asarray(prepared.area_vectors) @ vector
            owner = np.asarray(prepared.owner_cells)
            neighbor = np.asarray(prepared.neighbor_cells)
            balance = np.zeros(prepared.cell_count)
            np.add.at(balance, owner, flux)
            mask = neighbor >= 0
            np.add.at(balance, neighbor[mask], -flux[mask])
            np.testing.assert_allclose(balance, 0.0, atol=1e-15)
            np.testing.assert_allclose(
                np.sum(prepared.cell_quadrature_weights, axis=1),
                prepared.cell_volumes,
                rtol=1e-13,
                atol=0.0,
            )
        carrier = UnstructuredFiniteVolumePlan.from_cell_mesh(
            transition.source_mesh
        ).prepare()
        assert carrier.coordinate_spec is None
        assert carrier.coordinate_validity is None
        assert carrier.cell_measure_errors is None
        assert carrier.geometry_id != source.geometry_id
        assert carrier.support.support_id != source.support.support_id


def test_mapped_fv_refuses_carrier_motion_and_field_support() -> None:
    def identity_motion(vertices: Array, time: Array, args: object) -> Array:
        return vertices

    with jax.enable_x64():
        fixture = nested_material_geometry()
        transition = fixture.transition
        plan = UnstructuredFiniteVolumePlan.from_cell_mesh(
            transition.target_mesh, coordinate_spec=transition.target_geometry
        )
        with pytest.raises(ValueError, match="Vertex-only FV motion"):
            FixedConnectivityMotionPlan(plan, identity_motion, mapping_id="identity")
        prepared = plan.prepare()
        with pytest.raises(ValueError, match="requires mapped field support"):
            prepare_finite_volume_field_reconstruction(
                prepared, PiecewiseConstantReconstruction()
            )
        with pytest.raises(ValueError):
            UnstructuredFiniteVolumePlan.from_cell_mesh(
                transition.source_mesh, coordinate_spec=transition.target_geometry
            )


def test_mapped_fv_rejects_distinct_neighbor_face_maps() -> None:
    with jax.enable_x64():
        transition = nested_material_geometry().transition
        original = transition.source_geometry
        elements = dict(zip(original.block_names, original.elements, strict=True))
        routes = {
            name: np.array(route)
            for name, route in zip(
                original.block_names, original.geometry_dofs, strict=True
            )
        }
        coordinates = np.asarray(original.coordinates)
        # Prism corner 3 lies on the prism/tet interface. Give just this
        # occurrence a distinct coordinate without modifying the shared carrier.
        prism_route = routes["prisms"]
        old = int(prism_route[0, 3])
        changed = coordinates[old].copy()
        changed[2] += 1.0e-8
        prism_route[0, 3] = coordinates.shape[0]
        geometry = CellGeometrySpec(elements, routes, np.vstack((coordinates, changed)))
        plan = UnstructuredFiniteVolumePlan.from_cell_mesh(
            transition.source_mesh, coordinate_spec=geometry
        )
        with pytest.raises(ValueError, match="different exact face geometry"):
            plan.prepare()


def test_quadrature_exactness_uses_certified_jacobian_degrees() -> None:
    with jax.enable_x64():
        transition = nested_material_geometry().transition
        mesh = transition.source_mesh
        moments = certified_cell_moments(mesh, transition.source_geometry)
        assert mapped_constant_quadrature_exact(mesh, moments)
        assert not mapped_constant_quadrature_exact(
            mesh,
            moments._replace(jacobian_degrees=(None,) * len(moments.jacobian_degrees)),
        )
        assert not mapped_constant_quadrature_exact(
            mesh,
            moments._replace(jacobian_degrees=(100,) * len(moments.jacobian_degrees)),
        )
        with pytest.raises(ValueError, match="does not cover"):
            mapped_constant_quadrature_exact(mesh, moments._replace(jacobian_degrees=()))


def test_curved_coordinate_map_changes_physical_fv_centroid_and_face_measure() -> None:
    with jax.enable_x64():
        reference = reference_cell_topology("hexahedron")
        cells = np.arange(8, dtype=np.int32)[None, :]
        mesh = CellMesh(
            np.asarray(reference.vertices, dtype=np.float64),
            (
                CellBlock(
                    "volume",
                    "hexahedron",
                    cells,
                ),
            ),
            polyhedral_connectivity=polyhedral_connectivity(
                (("hexahedron", cells),), len(reference.vertices)
            ),
        )
        element = lagrange_element("hexahedron", 2)
        coordinates = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        x, y = coordinates[:, 0], coordinates[:, 1]
        coordinates[:, 2] += 0.5 * x * (1.0 - x) * y * (1.0 - y)
        geometry = CellGeometrySpec(
            {"volume": element},
            {"volume": np.arange(coordinates.shape[0], dtype=np.int32)[None, :]},
            coordinates,
        )
        mapped = UnstructuredFiniteVolumePlan.from_cell_mesh(
            mesh, coordinate_spec=geometry
        ).prepare()
        # The triangular shear has unit Jacobian and mean vertical shift 1/72;
        # its two curved horizontal faces have strictly more than unit area.
        np.testing.assert_allclose(mapped.cell_volumes, (1.0,), rtol=0.0, atol=4e-16)
        np.testing.assert_allclose(
            mapped.cell_centers, ((0.5, 0.5, 37.0 / 72.0),), rtol=0.0, atol=4e-16
        )
        assert np.count_nonzero(np.asarray(mapped.face_measures) > 1.0001) == 2
        carrier = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
        np.testing.assert_allclose(carrier.cell_centers, ((0.5, 0.5, 0.5),))

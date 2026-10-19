#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import dataclass
from itertools import permutations, product

import equinox as eqx
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import certify_domain_coverage
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    MappedDomainBoundarySource,
)
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._contracts import CellFamilyPolicy, MeshingLimits
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._hex_dominant import extract_hex_dominant
from phydrax.meshing._hex_generation import (
    extract_volume_hexes,
    generate_integer_grid_hexes,
    NativeHexGridRoute,
    NativeHexGridSchedule,
    realize_mapped_grid_hexes,
)
from phydrax.meshing._host_banks import entities
from phydrax.meshing._organization_records import MeshZone, MeshZoneRole
from phydrax.meshing._quad_generation import remap_dual_metadata
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.hex_fields import (
    allfield_inventory,
    allfield_state,
    allfield_transfer,
    quadratic_source,
    quadratic_value,
    TRANSFER_TOLERANCE,
    TransferEvidence,
)
from tests._support.hex_sources import (
    hex_volumes,
    mapped_occupied_source,
    nonunit_material_publication,
    occupied_cells,
    two_tetrahedra,
)


# For the small positive-volume fixtures, sequential quadrature and contraction
# roundoff is bounded by gamma_4096 < 1e-12 in binary64 (Higham, Accuracy and
# Stability of Numerical Algorithms, second edition, section 3.1). Scaled
# coverage measures use rational integration with one directed endpoint rounding.
# Preserve the existing absolute/relative comparison limit.
VOLUME_TOLERANCE = 1.0e-12
# Fixed mapped-FV qualification bounds retained from
# tools/meshing_qualification/scenarios/family.py:_family_mapped_fv.
FV_VOLUME_RTOL = 1.0e-10
STATIONARY_EULER_ATOL = 1.0e-10


def test_all_hex_template_has_positive_maps_and_exact_volume() -> None:
    source = two_tetrahedra()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert result.validity.certified_valid_count == 8
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        volumes = hex_volumes(result.mesh)
        np.testing.assert_allclose(
            volumes, np.full((8,), 1.0 / 24.0), rtol=VOLUME_TOLERANCE
        )
        assert np.sum(volumes) == pytest.approx(1.0 / 3.0)
        assert result.mesh.topology.entities(2).count == 33
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        np.testing.assert_array_equal(
            np.diff(result.mesh.connectivity.face_vertex_offsets), 4
        )
        np.testing.assert_array_equal(
            np.diff(result.mesh.connectivity.cell_face_offsets), 6
        )
        assert isinstance(source.connectivity, TetrahedralConnectivity)
        assert np.max(np.asarray(result.mesh.connectivity.face_cell_counts)) == 2
        source_edges = np.asarray(source.coordinates)[
            np.asarray(source.connectivity.edges)
        ]
        target_edges = np.asarray(result.mesh.coordinates)[
            np.asarray(result.mesh.connectivity.edges)
        ]
        source_maximum = np.max(
            np.linalg.norm(source_edges[:, 1] - source_edges[:, 0], axis=1)
        )
        target_maximum = np.max(
            np.linalg.norm(target_edges[:, 1] - target_edges[:, 0], axis=1)
        )
        assert target_maximum <= 0.5 * source_maximum


def test_dual_hex_maps_preserve_original_source_bank_and_exact_fractional_corners() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import (
        coordinate_corner_images,
        prepared_coordinate_source_bank,
        rounded_point,
    )

    source = two_tetrahedra()
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert result.geometry is not None and result.source_geometry is not None
        np.testing.assert_array_equal(
            np.asarray(result.geometry.coordinates).view(np.uint64),
            np.asarray(result.source_geometry.coordinates).view(np.uint64),
        )
        bank = prepared_coordinate_source_bank(result.geometry, budget=reference_budget)
        elements, routes, _ = result.geometry.resolve(result.mesh)
        source_points = np.asarray(source.coordinates)
        for block, element, block_routes in zip(
            result.mesh.blocks, elements, routes, strict=True
        ):
            for cell, route in zip(
                np.asarray(block.vertices), np.asarray(block_routes), strict=True
            ):
                images = coordinate_corner_images(
                    element,
                    tuple(bank[int(index)] for index in route),
                    budget=reference_budget,
                )
                assert images is not None
                for vertex, actual in zip(cell, images, strict=True):
                    support = result.vertex_supports[vertex]
                    support = support[support >= 0]
                    expected = tuple(
                        sum(
                            (
                                Fraction(float(source_points[index, axis]))
                                for index in support
                            ),
                            Fraction(0),
                        )
                        / support.size
                        for axis in range(3)
                    )
                    assert actual == expected
                    np.testing.assert_array_equal(
                        result.mesh.coordinates[vertex], rounded_point(expected)
                    )


def test_exact_hex_quality_accepts_authoritative_unit_map_at_quality_boundary() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality

    points = np.asarray(reference_cell_topology("hexahedron").vertices)
    mesh = CellMesh(points, (CellBlock("authored", "hexahedron", np.arange(8)[None, :]),))
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0,
        minimum_mean_ratio=1.0,
        maximum_aspect_ratio=1.0,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        scaled, mean, aspect = exact_mapped_volume_quality(
            mesh,
            CellGeometrySpec.affine(mesh),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert np.all(scaled >= schedule.minimum_scaled_jacobian)
        assert np.all(mean >= schedule.minimum_mean_ratio)
        assert np.all(aspect <= schedule.maximum_aspect_ratio)


def test_exact_hex_aspect_detects_curved_interior_despite_regular_corners() -> None:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from phydrax.meshing._hex_generation import exact_mapped_volume_quality
    from phydrax.meshing._quality import evaluate_cell_quality

    points = np.asarray(reference_cell_topology("hexahedron").vertices)
    mesh = CellMesh(points, (CellBlock("authored", "hexahedron", np.arange(8)[None, :]),))
    element = coordinate_lagrange_element("hexahedron", 2)
    controls = np.asarray(element.reference_nodes).copy()
    controls[:, 2] *= 1.0 + 8.0 * controls[:, 0] * (1.0 - controls[:, 0])
    geometry = CellGeometrySpec(
        {"authored": element},
        {"authored": np.arange(controls.shape[0])[None, :]},
        controls,
    )
    schedule = NativeHexGridSchedule("frame_grid", maximum_aspect_ratio=4.0)
    assert np.all(
        np.asarray(evaluate_cell_quality(mesh).aspect_ratios)
        <= schedule.maximum_aspect_ratio
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        _, _, aspect = exact_mapped_volume_quality(
            mesh, geometry, scope=execution_scope, budget=coordinate_budget
        )
        jacobian = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (4.0, 0.0, 2.5)))
        actual_interior_condition = np.linalg.cond(jacobian)
        assert actual_interior_condition > schedule.maximum_aspect_ratio
        assert np.all(aspect >= actual_interior_condition)


def test_material_face_subdivision_and_region_identity_are_exact() -> None:
    source = two_tetrahedra()
    cells = source.topology.entities(3)
    zones = tuple(
        MeshZone(
            name,
            MeshZoneRole.REGION,
            MeshingScope(
                source.mesh_id,
                source.numeric_version,
                MeshingEntityKind.MESH,
                3,
                cells.entity_set_id,
                np.asarray(cells.entity_ids)[[row]],
            ),
        )
        for row, name in enumerate(("left", "right"))
    )
    faces = source.topology.entities(2)
    assert isinstance(source.connectivity, TetrahedralConnectivity)
    face_vertices = np.asarray(source.connectivity.faces)
    interface = np.flatnonzero(
        np.all(np.sort(face_vertices, axis=1) == np.asarray((0, 1, 2)), axis=1)
    )
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "authoritative-material-solid",
        "material-source",
        faces.entity_set_id,
        np.asarray(faces.entity_ids)[interface],
        ("material-source:interface:0",),
        np.zeros((1,), dtype=np.float64),
        exact=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        new_zones, _, _, new_associations = remap_dual_metadata(
            result, zones=zones, associations=(association,)
        )
        assert tuple(zone.name for zone in new_zones) == ("left", "right")
        assert tuple(zone.scope.entity_ids.size for zone in new_zones) == (4, 4)
        face_association = next(
            value
            for value in new_associations
            if value.target_entity_set_id
            == result.mesh.topology.entities(2).entity_set_id
        )
        assert (
            face_association.source_entities.entity_ids()
            == ("material-source:interface:0",) * 3
        )
        assert face_association.exact
        np.testing.assert_array_equal(
            face_association.residuals, np.zeros((3,), dtype=np.float64)
        )
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        assert np.all(
            np.asarray(result.mesh.connectivity.face_cell_counts)[
                (result.entity_parent_dimensions[2] == 2)
                & (result.entity_parent_rows[2] == interface[0])
            ]
            == 2
        )


def test_exact_association_does_not_survive_rounded_off_parent_face() -> None:
    source = two_tetrahedra()
    assert isinstance(source.connectivity, TetrahedralConnectivity)
    faces = source.topology.entities(2)
    interface = np.flatnonzero(
        np.all(
            np.sort(np.asarray(source.connectivity.faces), axis=1)
            == np.asarray((0, 1, 2), dtype=np.int64),
            axis=1,
        )
    )
    association = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "material-solid",
        "material-source",
        faces.entity_set_id,
        np.asarray(faces.entity_ids)[interface],
        ("material-source:interface:0",),
        np.zeros((1,), dtype=np.float64),
        exact=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        extraction = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        center = np.flatnonzero(
            (extraction.entity_parent_dimensions[0] == 2)
            & (extraction.entity_parent_rows[0] == interface[0])
        )[0]
        coordinates = np.asarray(extraction.mesh.coordinates, dtype=np.float64).copy()
        coordinates[center, 2] = np.spacing(np.float64(1.0))
        perturbed = CellMesh(
            coordinates,
            extraction.mesh.blocks,
            vertex_global_ids=extraction.mesh.vertex_global_ids,
            numeric_version="off-parent-face",
        )
        candidate = eqx.tree_at(lambda value: value.mesh, extraction, perturbed)
        _, _, _, associations = remap_dual_metadata(
            candidate, associations=(association,)
        )
        face_association = next(
            value
            for value in associations
            if value.target_entity_set_id == perturbed.topology.entities(2).entity_set_id
        )
        assert not face_association.exact
        assert np.all(
            np.asarray(face_association.residuals) >= np.spacing(np.float64(1.0))
        )


@pytest.mark.parametrize("domain", ("cavity", "nonsweepable"))
def test_general_all_hex_positive_domains(domain: str) -> None:
    if domain == "cavity":
        occupied = set(product(range(3), range(3), range(3))) - {(1, 1, 1)}
    else:
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    source = occupied_cells(occupied)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_volume_hexes(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert result.validity.certified_valid_count == 24 * len(occupied)
        assert np.sum(hex_volumes(result.mesh)) == pytest.approx(
            len(occupied), rel=VOLUME_TOLERANCE
        )
        points = np.asarray(result.mesh.coordinates)
        corners = points[entities(result.mesh, 3)]
        if domain == "cavity":
            assert not np.any(
                np.all(
                    (np.mean(corners, axis=1) > 1.0) & (np.mean(corners, axis=1) < 2.0),
                    axis=1,
                )
            )


def test_hex_dominant_pyramid_transition_closes_interface() -> None:
    source = two_tetrahedra()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        counts: dict[str, int] = {}
        for block in result.mesh.blocks:
            counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
        assert counts["hexahedron"] == 4
        assert counts["pyramid"] == 3
        assert counts["tetrahedron"] == 12
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        assert isinstance(result.mesh.connectivity, PolyhedralConnectivity)
        assert np.max(np.asarray(result.mesh.connectivity.face_cell_counts)) == 2
        assert result.geometry is not None and result.source_geometry is not None
        np.testing.assert_array_equal(
            np.asarray(result.geometry.coordinates).view(np.uint64),
            np.asarray(result.source_geometry.coordinates).view(np.uint64),
        )
        from phydrax.discretization._cell_geometry import (
            RationalComposedCellGeometryElement,
        )
        from phydrax.discretization._coordinate_enclosure import (
            coordinate_corner_images,
            prepared_coordinate_source_bank,
        )

        bank = prepared_coordinate_source_bank(result.geometry, budget=reference_budget)
        elements, routes, _ = result.geometry.resolve(result.mesh)
        for block, element, block_routes in zip(
            result.mesh.blocks, elements, routes, strict=True
        ):
            if block.cell_kind != "pyramid":
                continue
            assert isinstance(element, RationalComposedCellGeometryElement)
            points = np.asarray(
                ((0.5, 0.5, 0.5), reference_cell_topology("pyramid").vertices[4])
            )
            np.testing.assert_array_equal(
                element.reference_derivative_status(points), (True, False)
            )
            for route in np.asarray(block_routes):
                corners = coordinate_corner_images(
                    element,
                    tuple(bank[int(index)] for index in route),
                    budget=reference_budget,
                )
                assert corners is not None
                assert all(
                    np.isfinite(float(value)) for corner in corners for value in corner
                )


def test_mixed_source_maps_join_actual_family_fem_and_fv_consumers() -> None:
    import jax.numpy as jnp

    source = two_tetrahedra()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        source_corners = np.asarray(source.coordinates)[entities(source, 3)]
        volume = float(
            np.sum(np.linalg.det(source_corners[:, 1:] - source_corners[:, :1])) / 6.0
        )
        elements = {
            kind: phx.discretization.lagrange_element(kind, 1)
            for kind in {block.cell_kind for block in result.mesh.blocks}
        }
        field = phx.discretization.FiniteElementFieldSpec(
            "u", {block.name: elements[block.cell_kind] for block in result.mesh.blocks}
        )
        space = phx.discretization.FiniteElementPlan(
            result.mesh, field, coordinate_spec=result.geometry
        ).prepare()
        form = phx.equations.FiniteElementForm(
            "exact-family-manufactured-diffusion",
            "u",
            (
                phx.equations.DiffusionAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        quadratic_source, coefficient_id="quadratic-source"
                    ),
                ),
            ),
        )
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        phx.equations.compile_finite_element_problem(
            form, space, constraint=constraint, dirichlet_values=quadratic_value
        )
        values = jnp.ones(
            (space.dof_maps[0].global_dof_count,), dtype=result.mesh.coordinates.dtype
        )
        line = phx.integration.GaussLegendreRule(6)
        quadratures = {
            "hexahedron": phx.integration.reference_rule_data(
                phx.integration.ReferenceHexahedronRule(line)
            ),
            "tetrahedron": phx.integration.reference_rule_data(
                phx.integration.ReferenceTetrahedronRule(line)
            ),
            "pyramid": phx.integration.reference_rule_data(
                phx.integration.ReferencePyramidRule(line)
            ),
        }
        inventory = 0.0
        for index, block in enumerate(space.mesh.blocks):
            quadrature = quadratures[block.cell_kind]
            geometry = space.evaluate_block_geometry(
                "u",
                index,
                space.default_runtime.coordinates,
                quadrature.points,
                quadrature.weights,
            )
            dofs = space.dof_maps[0].cell_dofs[index]
            discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
            inventory += float(
                np.sum(np.asarray(geometry.physical_weights) * np.asarray(discrete))
            )
        assert inventory == pytest.approx(volume, rel=0.0, abs=VOLUME_TOLERANCE)
        system = phx.equations.EulerSystem(result.mesh.ambient_dimension)
        fv = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            result.mesh, component_names=system.component_names
        ).prepare(cell_geometry=result.geometry)
        volumes = np.asarray(fv.cell_volumes)
        assert np.all(np.isfinite(volumes))
        assert np.all(volumes > 0.0)
        assert abs(float(np.sum(volumes)) - volume) <= FV_VOLUME_RTOL * volume
        assert float(np.sum(volumes)) == pytest.approx(
            volume, rel=0.0, abs=VOLUME_TOLERANCE
        )
        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            fv.boundary_patch_names,
            {
                name: phx.discretization.SlipWallBoundary()
                for name in fv.boundary_patch_names
            },
        )
        method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        )
        dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
            system, fv, method, boundaries
        )
        primitive = jnp.zeros(
            (fv.cell_count, result.mesh.ambient_dimension + 2),
            dtype=result.mesh.coordinates.dtype,
        )
        primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
        state = system.primitive_to_conserved(primitive)
        rate = dynamics(jnp.asarray(0.0, dtype=state.dtype), state)
        defect = float(jnp.max(jnp.abs(rate)))
        assert np.isfinite(defect)
        assert defect <= STATIONARY_EULER_ATOL


@pytest.mark.parametrize("row_order", tuple(permutations(range(3))))
def test_hex_dominant_partial_face_fans_close_across_source_cell_order(
    row_order: tuple[int, ...],
) -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, -1.0, 0.0),
        )
    )
    tetrahedra = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4), (0, 4, 1, 5)), dtype=np.int64)
    source = CellMesh.from_tetrahedra(coordinates, tetrahedra[list(row_order)])
    core = row_order.index(0)
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[core]],
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {
            "hexahedron",
            "pyramid",
            "tetrahedron",
        }
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        if not isinstance(source.connectivity, TetrahedralConnectivity):
            raise TypeError(
                "fixture.source.connectivity must be tetrahedral; got incompatible connectivity"
            )
        if not isinstance(result.mesh.connectivity, PolyhedralConnectivity):
            raise TypeError(
                "fixture.result.connectivity must be polyhedral; got incompatible connectivity"
            )
        source_incidence = np.asarray(source.connectivity.face_cell_counts)
        dimensions, rows = (
            result.entity_parent_dimensions[2],
            result.entity_parent_rows[2],
        )
        expected = np.full(dimensions.shape, 2, dtype=np.int64)
        inherited = dimensions == 2
        expected[inherited] = source_incidence[rows[inherited]]
        np.testing.assert_array_equal(result.mesh.connectivity.face_cell_counts, expected)
        # The complement's shared face has only one marked edge. Its full fan
        # must agree from both sides, not merely the selected core's quad interface.
        source_faces = entities(source, 2)
        partial_face = np.flatnonzero(
            np.all(np.sort(source_faces, axis=1) == (0, 1, 4), axis=1)
        )[0]
        partial_children = inherited & (rows == partial_face)
        assert np.count_nonzero(partial_children) > 1
        assert np.all(
            np.asarray(result.mesh.connectivity.face_cell_counts)[partial_children] == 2
        )
        # Independently integrate each cell family, then group by retained parent
        # ancestry. This catches a closed but duplicated/missing transition cone.
        family_volumes = [hex_volumes(result.mesh)]
        target_points = np.asarray(result.mesh.coordinates)
        for block in result.mesh.blocks:
            if block.cell_kind == "hexahedron":
                continue
            vertices = target_points[np.asarray(block.vertices)]
            simplices = (
                ((0, 1, 2, 4), (0, 2, 3, 4))
                if block.cell_kind == "pyramid"
                else ((0, 1, 2, 3),)
            )
            volumes = np.zeros(block.cell_count)
            for simplex in simplices:
                corners = vertices[:, simplex, :]
                edges = corners[:, 1:] - corners[:, :1]
                determinants = np.linalg.det(edges)
                assert np.all(determinants > 0.0)
                volumes += determinants / 6.0
            family_volumes.append(volumes)
        source_corners = coordinates[tetrahedra[list(row_order)]]
        source_volumes = (
            np.linalg.det(source_corners[:, 1:] - source_corners[:, :1]) / 6.0
        )
        transferred_volumes = np.bincount(
            result.parent_cells, weights=np.concatenate(family_volumes), minlength=3
        )
        np.testing.assert_allclose(
            transferred_volumes,
            source_volumes,
            rtol=0.0,
            # The unit source-volume determinant comparison budgets 32 edge/
            # determinant roundoffs per side plus accumulation: 128 * eps.
            atol=128 * np.finfo(np.float64).eps,
        )


def test_hex_dominant_never_weakens_required_pure_family() -> None:
    source = two_tetrahedra()
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_hex_dominant(
                source,
                MeshingLimits(),
                CellFamilyPolicy(required=("hexahedron",)),
                hex_core_cells=np.asarray(source.topology.entities(3).entity_ids)[[0]],
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED


@pytest.mark.parametrize("dimension,limit", ((1, "maximum_edges"), (2, "maximum_faces")))
def test_hex_dominant_actual_topology_capacity_refuses_without_mutation(
    dimension: int,
    limit: str,
) -> None:
    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    policy = CellFamilyPolicy(
        required=("hexahedron",),
        allowed_transitions=("pyramid", "tetrahedron"),
        allow_mixed=True,
    )
    core = np.asarray(source.topology.entities(3).entity_ids)[[0]]
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        accepted = extract_hex_dominant(
            source,
            MeshingLimits(),
            policy,
            hex_core_cells=core,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        cap = accepted.mesh.topology.entities(dimension).count - 1
        with pytest.raises(MeshingFailure) as failed:
            extract_hex_dominant(
                source,
                MeshingLimits(**{limit: cap}),
                policy,
                hex_core_cells=core,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(source.coordinates, original)


def test_hex_budget_and_immutable_boundary_refuse_without_source_change() -> None:
    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    with native_meshing_request(MeshingLimits(maximum_cells=7)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                source,
                MeshingLimits(maximum_cells=7),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                source,
                MeshingLimits(),
                boundary_subdivision=False,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
        np.testing.assert_array_equal(source.coordinates, original)


def test_invalid_native_topology_preserves_the_constructor_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._meshcore as meshcore

    source = two_tetrahedra()
    original = np.asarray(source.coordinates).copy()
    extract = meshcore.tetrahedron_dual_hexes

    def faulty(
        nodes: np.ndarray, vertex_count: int, /, *, maximum_cells: int
    ) -> np.ndarray:
        cells = extract(nodes, vertex_count, maximum_cells=maximum_cells)
        cells[0] = cells[0, (0, 3, 2, 1, 4, 7, 6, 5)]
        return cells

    monkeypatch.setattr(meshcore, "tetrahedron_dual_hexes", faulty)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(ValueError, match="orientation|face"):
            extract_volume_hexes(
                source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
            )
    np.testing.assert_array_equal(source.coordinates, original)


def test_misoriented_native_template_fails_independent_map_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._meshcore as meshcore

    extract = meshcore.tetrahedron_dual_hexes

    def faulty(
        nodes: np.ndarray, vertex_count: int, /, *, maximum_cells: int
    ) -> np.ndarray:
        cells = extract(nodes, vertex_count, maximum_cells=maximum_cells)
        cells[:] = cells[:, (0, 3, 2, 1, 4, 7, 6, 5)]
        return cells

    monkeypatch.setattr(meshcore, "tetrahedron_dual_hexes", faulty)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_volume_hexes(
                two_tetrahedra(),
                MeshingLimits(),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_independent_curved_source_restrictions_have_zero_fidelity(
    route: NativeHexGridRoute,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    schedule = NativeHexGridSchedule(route)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            schedule,
            MeshingLimits(),
            0.5,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        result = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
        assert sum(block.cell_count for block in result.mesh.blocks) == 8
        assert np.min(result.scaled_jacobian_lower) > 0.9
        assert np.min(result.mean_ratio_lower) > 0.9
        embedding = certify_global_embedding(
            result.mesh,
            result.geometry,
            certify_cell_geometry_validity(result.geometry, mesh=result.mesh),
        )
        coverage = certify_domain_coverage(
            result.mesh, result.geometry, domain, result.cell_regions, embedding=embedding
        )
        assert embedding.status == "certified"
        assert coverage.status == "certified"
        source = MappedDomainBoundarySource(domain, result.cell_regions)
        fidelity = certify_source_fidelity(
            result.mesh, result.geometry, source, tolerance=0.0
        )
        assert fidelity.status == "certified"
        assert fidelity.mesh_to_source_lower == fidelity.mesh_to_source_upper == 0.0
        assert fidelity.source_to_mesh_lower == fidelity.source_to_mesh_upper == 0.0
        assert (
            fidelity.domain_coverage is not None
            and fidelity.domain_coverage.status == "certified"
        )


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
@pytest.mark.parametrize("kind", ("cavity", "nonsweepable", "material"))
def test_grid_portfolio_closes_nontrivial_curved_domains(
    route: NativeHexGridRoute, kind: str
) -> None:
    if kind == "cavity":
        occupied = set(product(range(3), range(3), range(3))) - {(1, 1, 1)}
    elif kind == "nonsweepable":
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    else:
        occupied = {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    complex_, domain = mapped_occupied_source(occupied, materials=kind == "material")
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule(route),
            MeshingLimits(),
            1.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert {block.cell_kind for block in mapped.mesh.blocks} == {"hexahedron"}
        assert sum(block.cell_count for block in mapped.mesh.blocks) == len(occupied)
        assert np.min(mapped.scaled_jacobian_lower) >= 0.85
        assert np.min(mapped.mean_ratio_lower) >= 0.85
        assert domain.exact_region_measures() == tuple(
            np.sum(domain.cell_regions == region)
            for region in range(len(domain.region_ids))
        )
        embedding = certify_global_embedding(
            mapped.mesh,
            mapped.geometry,
            certify_cell_geometry_validity(mapped.geometry, mesh=mapped.mesh),
        )
        coverage = certify_domain_coverage(
            mapped.mesh, mapped.geometry, domain, mapped.cell_regions, embedding=embedding
        )
        assert embedding.status == coverage.status == "certified"
        if route == "balanced_grid":
            assert np.all(grid.octree_levels == grid.closure_rounds)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
@pytest.mark.parametrize(
    "limit", ("maximum_edges", "maximum_faces", "maximum_data_bytes")
)
def test_integer_grid_rejects_actual_entity_and_publication_capacity(
    route: NativeHexGridRoute,
    limit: str,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    source_coordinates = np.asarray(domain.reference_mesh.coordinates).copy()
    limits = MeshingLimits(**{limit: 1})
    with native_meshing_request(limits) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            generate_integer_grid_hexes(
                complex_,
                domain,
                NativeHexGridSchedule(route),
                limits,
                1.0,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(
            domain.reference_mesh.coordinates, source_coordinates
        )


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_fine_grid_capacity_refusal_preserves_original_source(
    route: NativeHexGridRoute,
) -> None:
    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    original = np.asarray(domain.reference_mesh.coordinates).copy()
    with native_meshing_request(MeshingLimits(maximum_cells=8)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            generate_integer_grid_hexes(
                complex_,
                domain,
                NativeHexGridSchedule(route, maximum_depth=20),
                MeshingLimits(maximum_cells=8),
                1.0e-6,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(domain.reference_mesh.coordinates, original)


@pytest.mark.parametrize("route", ("balanced_grid", "frame_grid"))
def test_integer_grid_entity_capacity_counts_retained_cavity_not_bounding_box(
    route: NativeHexGridRoute,
) -> None:
    occupied = {(0, 0, 0), (2, 0, 0)}
    complex_, domain = mapped_occupied_source(occupied, curvature=0.125)
    limits = MeshingLimits(maximum_edges=24, maximum_faces=12)
    with native_meshing_request(limits) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule(route),
            limits,
            1.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        assert sum(block.cell_count for block in grid.mesh.blocks) == len(occupied)
        assert entities(grid.mesh, 1).shape[0] == 24
        assert entities(grid.mesh, 2).shape[0] == 12


def test_original_zero_tolerance_curved_size_request_remains_unmet() -> None:
    """The positive policy must not masquerade as satisfying the stricter request."""
    import phydrax as phx

    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    reference = phx.meshing.NativePlcSource(
        complex_, domain.reference_domain.source_id, "reference-state"
    )
    source = phx.meshing.NativeMappedHexSource(reference, domain)
    faces = domain.reference_mesh.topology.entities(2)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        faces.entity_ids,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.5,
                maximum_size=0.6,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(),
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("mapped_frame_grid_hex")
    ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
    with pytest.raises(MeshingFailure) as failed:
        plan.execute()
    assert failed.value.category is phx.meshing.MeshingFailureCategory.COMPLIANCE_FAILED
    achieved = dict(failed.value.evidence.achieved)
    key = f"size:{request.size_controls[0].control_id}"
    assert achieved[f"{key}:p50_edge"] < 0.5
    assert achieved[f"{key}:p95_edge"] < 0.5
    assert dict(failed.value.evidence.requested)[f"{key}:target_size"] == 0.5
    assert (
        failed.value.stage == phx.meshing.MeshingStageKind.SPECIFICATION_COMPLIANCE.value
    )


@dataclass(frozen=True, slots=True)
class _MaterialPublication:
    result: CellMeshingResult
    domain: MappedReferenceDomain
    expected_measures: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _MaterialRoundTrip:
    coarsening: MeshAdaptationResult
    fields: dict[str, Array]
    restored_fields: dict[str, Array]
    refined_evidence: TransferEvidence
    restored_evidence: TransferEvidence


@pytest.fixture(
    scope="module",
    params=(
        pytest.param((2.0, 2.0, (2.0, 1.0)), id="binary-inverse"),
        pytest.param((3.0, 2.5, (3.0, 1.5)), id="nonbinary-inverse"),
    ),
)
def material_publication(request: pytest.FixtureRequest) -> _MaterialPublication:
    axis_scale, maximum_size, expected_measures = request.param
    result, domain = nonunit_material_publication(axis_scale, maximum_size)
    return _MaterialPublication(result, domain, expected_measures)


@pytest.fixture(scope="module")
def material_refinement(
    material_publication: _MaterialPublication,
) -> MeshAdaptationResult:
    result, domain = material_publication.result, material_publication.domain
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in result.mesh.blocks]
    )
    refinement = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MarkedMeshAdaptation(cell_ids),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=transfer,
                audit_policy=phx.meshing.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    assert refinement.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    return refinement


@pytest.fixture(scope="module")
def material_round_trip(
    material_publication: _MaterialPublication,
    material_refinement: MeshAdaptationResult,
) -> _MaterialRoundTrip:
    result, domain = material_publication.result, material_publication.domain
    refinement = material_refinement
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    fields = allfield_state(result)
    refined_fields, refined_evidence = allfield_transfer(refinement, fields, refine=True)
    assert set(refined_evidence) == set(fields)
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in refinement.target.mesh.blocks]
    )
    coarsening = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            refinement.target,
            phx.meshing.MarkedMeshAdaptation(
                np.empty((0,), dtype=np.int64), fine_ids, hierarchy=refinement.hierarchy
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=transfer,
                audit_policy=phx.meshing.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    assert coarsening.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    restored_fields, restored_evidence = allfield_transfer(
        coarsening, refined_fields, refine=False
    )
    return _MaterialRoundTrip(
        coarsening, fields, restored_fields, refined_evidence, restored_evidence
    )


def test_nonunit_curved_roots_publish_materials_and_preserve_source_strata(
    material_publication: _MaterialPublication,
) -> None:
    result, domain, expected_measures = (
        material_publication.result,
        material_publication.domain,
        material_publication.expected_measures,
    )
    assert result.audit.passed and result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert result.certification.coverage is not None
    for actual, expected in zip(
        result.certification.coverage.achieved_region_measures,
        expected_measures,
        strict=True,
    ):
        assert actual is not None
        assert actual == pytest.approx(expected, rel=0.0, abs=VOLUME_TOLERANCE)
    assert {zone.material_id for zone in result.zones} == {
        "material-left",
        "material-right",
    }
    assert "material-interface" in {patch.name for patch in result.patches}
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    vertex, dimensions = transfer.source_associations(result)
    assert dimensions == (1, 2, 3)
    assert vertex.exact


def test_nonunit_curved_roots_refine_with_positive_pure_hex_maps(
    material_publication: _MaterialPublication,
    material_refinement: MeshAdaptationResult,
) -> None:
    refinement = material_refinement
    assert refinement.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    transfer = phx.meshing.MappedReferenceAssociationTransfer(material_publication.domain)
    transfer.source_associations(refinement.target)
    assert (
        refinement.target.certification is not None
        and refinement.target.certification.passed
    )
    assert {block.cell_kind for block in refinement.target.mesh.blocks} == {"hexahedron"}


def test_nonunit_curved_roots_continuation_retains_all_declared_fields(
    material_round_trip: _MaterialRoundTrip,
) -> None:
    cycle = material_round_trip
    assert cycle.coarsening.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert set(cycle.refined_evidence) == set(cycle.fields)
    assert set(cycle.restored_evidence) == set(cycle.fields)
    assert set(allfield_inventory(cycle.coarsening.target, cycle.restored_fields)) == set(
        cycle.fields
    )


@pytest.mark.parametrize("kind", ("H1", "DG", "Hcurl", "Hdiv"))
def test_nonunit_curved_roots_form_history_round_trip(
    material_round_trip: _MaterialRoundTrip,
    kind: str,
) -> None:
    cycle = material_round_trip
    np.testing.assert_allclose(
        cycle.restored_fields[kind],
        cycle.fields[kind],
        rtol=TRANSFER_TOLERANCE,
        atol=TRANSFER_TOLERANCE,
    )


def test_allfield_inventory_refuses_a_missing_role(
    material_publication: _MaterialPublication,
) -> None:
    fields = allfield_state(material_publication.result)
    del fields["DG"]
    with pytest.raises(AssertionError):
        allfield_inventory(material_publication.result, fields)


def test_allfield_inventory_refuses_truncated_history(
    material_publication: _MaterialPublication,
) -> None:
    fields = allfield_state(material_publication.result)
    fields["H1"] = fields["H1"][:, :1]
    with pytest.raises(AssertionError, match="H1"):
        allfield_inventory(material_publication.result, fields)


def test_mapped_corners_use_correct_rounding_of_the_original_polynomial() -> None:
    from fractions import Fraction

    complex_, domain = mapped_occupied_source({(0, 0, 0)}, curvature=0.125)
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        grid = generate_integer_grid_hexes(
            complex_,
            domain,
            NativeHexGridSchedule("frame_grid"),
            MeshingLimits(),
            1.0 / 3.0,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        mapped = realize_mapped_grid_hexes(
            grid,
            domain.reference_mesh,
            domain.source_geometry,
            domain.cell_regions,
            MeshingLimits(),
            scope=execution_scope,
            budget=coordinate_budget,
        )
        expected = []
        for point in np.asarray(grid.mesh.coordinates):
            x, y, z = (Fraction(float(value)) for value in point)
            expected.append((float(x), float(y), float(z + x * x * y / 8)))
        np.testing.assert_array_equal(
            mapped.mesh.coordinates, np.asarray(expected, dtype=np.float64)
        )

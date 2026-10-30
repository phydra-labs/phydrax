from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization.finite_volume._polyhedral import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._polyhedral_generation import NativePolyhedralSchedule
from phydrax.meshing._periodic import periodic_entity_orbit_map
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing.providers._native_sources import NativePolyhedralSource
from tests.unit.meshing.test_polyhedral_adaptation import _plc_material_l


def _periodic_box_source(
    lengths: tuple[float, float, float], /
) -> tuple[NativePolyhedralSource, tuple[phx.meshing.PeriodicConstraint, ...]]:
    lx, ly, lz = lengths
    vertices = np.asarray(
        tuple(
            (float(x) * lx, float(y) * ly, float(z) * lz)
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ),
        dtype=np.float64,
    )
    row = {
        tuple(int(value) for value in point): index
        for index, point in enumerate(
            tuple(
                (x, y, z)
                for z in range(2)
                for y in range(2)
                for x in range(2)
            )
        )
    }
    polygons = (
        (row[(0, 0, 0)], row[(0, 1, 0)], row[(1, 1, 0)], row[(1, 0, 0)]),
        (row[(0, 0, 1)], row[(1, 0, 1)], row[(1, 1, 1)], row[(0, 1, 1)]),
        (row[(0, 0, 0)], row[(0, 0, 1)], row[(0, 1, 1)], row[(0, 1, 0)]),
        (row[(1, 0, 0)], row[(1, 1, 0)], row[(1, 1, 1)], row[(1, 0, 1)]),
        (row[(0, 0, 0)], row[(1, 0, 0)], row[(1, 0, 1)], row[(0, 0, 1)]),
        (row[(0, 1, 0)], row[(0, 1, 1)], row[(1, 1, 1)], row[(1, 1, 0)]),
    )
    segments = np.asarray(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % len(face)])))
                for face in polygons
                for index in range(len(face))
            }
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(face, dtype=np.int64) for face in polygons),
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1)),
        ("bulk",),
        segments=segments,
        vertex_ids=tuple(f"box-vertex-{index}" for index in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"box-facet-{index}" for index in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=tuple(f"box-edge-{index}" for index in range(segments.shape[0])),
        segment_occurrence_paths=((),) * segments.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePolyhedralSource(
        complex_,
        "periodic-stretched-box",
        "r1",
        sites=np.asarray(((lx / 2.0, ly / 2.0, lz / 2.0),), dtype=np.float64),
        weights=np.zeros((1,), dtype=np.float64),
        site_ids=("bulk-site",),
    )
    facet_rows = {
        identifier: row
        for row, identifier in enumerate(complex_.facet_entity_ids)
    }

    def facet_scope(facet_row: int, /) -> MeshingScope:
        return MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray((complex_.facet_entity_ids[facet_row],), dtype=np.int64),
        )

    pairs = ((4, 3, 0), (2, 5, 1), (0, 1, 2))
    constraints = []
    for source_row, target_row, axis in pairs:
        transform = np.eye(4, dtype=np.float64)
        transform[axis, 3] = lengths[axis]
        constraints.append(
            phx.meshing.PeriodicConstraint(
                facet_scope(facet_rows[complex_.facet_ids[source_row]]),
                facet_scope(facet_rows[complex_.facet_ids[target_row]]),
                transform,
                tolerance=0.0,
            )
        )
    return source, tuple(constraints)


def _periodic_box_specification(
    source: NativePolyhedralSource,
    constraints: tuple[phx.meshing.PeriodicConstraint, ...],
    lengths: tuple[float, float, float],
    /,
) -> VolumeMeshingSpec:
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        region_seeds=(
            RegionSeed(
                np.asarray(tuple(length / 2.0 for length in lengths), dtype=np.float64),
                "bulk",
                "bulk-material",
                RegionRole.SOLID,
            ),
        ),
        periodic_constraints=constraints,
    )


def _independent_polyhedral_volumes(mesh: CellMesh, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Independent measure oracle requires polyhedral connectivity.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    volumes = np.zeros((connectivity.cell_count,), dtype=np.float64)
    for cell in range(connectivity.cell_count):
        first, last = map(int, cell_offsets[cell : cell + 2])
        for face, sign in zip(cell_faces[first:last], cell_signs[first:last], strict=True):
            face_first, face_last = map(
                int, face_offsets[int(face) : int(face) + 2]
            )
            polygon = coordinates[face_vertices[face_first:face_last]]
            for index in range(1, polygon.shape[0] - 1):
                volumes[cell] += (
                    sign
                    * np.dot(
                        polygon[0],
                        np.cross(polygon[index], polygon[index + 1]),
                    )
                    / 6.0
                )
    return np.abs(volumes)


def _independent_face_area(mesh: CellMesh, face: int, /) -> float:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Independent coverage oracle requires polyhedral connectivity.")
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    first, last = map(int, offsets[face : face + 2])
    polygon = np.asarray(mesh.coordinates, dtype=np.float64)[vertices[first:last]]
    vector = 0.5 * np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    return float(np.linalg.norm(vector))




def _assert_reciprocal_internal_faces(mesh: CellMesh, /) -> None:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Reciprocal-face oracle requires polyhedral connectivity.")
    offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
    incident: dict[int, dict[int, int]] = {}
    for cell in range(connectivity.cell_count):
        first, last = map(int, offsets[cell : cell + 2])
        for face, sign in zip(faces[first:last], signs[first:last], strict=True):
            incident.setdefault(int(face), {})[cell] = int(sign)
    owners = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    internal_count = 0
    for face, (owner, neighbor) in enumerate(zip(owners, neighbors, strict=True)):
        if neighbor < 0:
            continue
        internal_count += 1
        sides = incident[int(face)]
        if (
            set(sides) != {int(owner), int(neighbor)}
            or sides[int(owner)] != -sides[int(neighbor)]
        ):
            raise ValueError("An internal polyhedral face lost reciprocal cell ancestry.")
    if internal_count == 0:
        raise ValueError("The material-L fixture must retain its material-interface face.")
def test_restricted_power_periodic_plc_preserves_quotient_and_fv_measure() -> None:
    lengths = (2.0, 3.0, 1.5)
    source, constraints = _periodic_box_source(lengths)
    specification = _periodic_box_specification(source, constraints, lengths)
    result = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "plc_restricted_power",
            polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
        )
    ).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()

    mesh = result.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Restricted-power output must retain polyhedral connectivity.")
    periodic = mesh.periodic_topology
    if periodic is None or not isinstance(periodic.cell, PeriodicCell):
        raise ValueError("Periodic PLC generation lost its canonical quotient topology.")
    periodic.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    periodic.require_coordinates(mesh.coordinates)
    assert tuple(entity.count for entity in periodic.quotient.entity_sets) == (1, 3, 3, 1)

    finite_volume = prepare_polyhedral_finite_volume_geometry(mesh)
    independent_volumes = _independent_polyhedral_volumes(mesh)
    np.testing.assert_allclose(
        np.asarray(finite_volume.cell_volumes), independent_volumes, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(independent_volumes), np.prod(lengths), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(finite_volume.cell_volumes) * 2.5),
        2.5 * np.prod(lengths),
        rtol=0.0,
        atol=1e-12,
    )
    assert result.certification is not None and result.certification.passed
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification.inputs.source is source

    face_association = next(
        association
        for association in result.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    row_by_face_id = {int(identifier): row for row, identifier in enumerate(face_ids)}
    face_orbits = periodic_entity_orbit_map(mesh, 2)
    lifted_to_quotient = np.asarray(face_orbits.lifted_to_quotient, dtype=np.int64)
    winding_shifts = np.asarray(face_orbits.winding_shifts, dtype=np.int64)
    facet_areas: dict[str, float] = {}
    facet_face_ids: dict[str, list[int]] = {}
    face_bounds = {
        0: (2, 0.0),
        1: (2, lengths[2]),
        2: (1, 0.0),
        3: (0, lengths[0]),
        4: (0, 0.0),
        5: (1, lengths[1]),
    }
    for target_id, source_id, role in zip(
        np.asarray(face_association.target_global_ids, dtype=np.int64),
        face_association.source_entity_ids,
        face_association.source_roles,
        strict=True,
    ):
        if role is not GeometrySourceEntityRole.FACET:
            continue
        row = row_by_face_id[int(target_id)]
        facet_areas[source_id] = facet_areas.get(source_id, 0.0) + _independent_face_area(
            mesh, row
        )
        facet_face_ids.setdefault(source_id, []).append(int(target_id))
        facet_row = source.complex.facet_ids.index(source_id)
        axis, coordinate = face_bounds[facet_row]
        connectivity_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        connectivity_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        first, last = map(int, connectivity_offsets[row : row + 2])
        face_points = np.asarray(mesh.coordinates, dtype=np.float64)[
            connectivity_vertices[first:last]
        ]
        np.testing.assert_allclose(face_points[:, axis], coordinate, rtol=0.0, atol=1e-12)
        for other_axis, length in enumerate(lengths):
            assert np.min(face_points[:, other_axis]) >= -1e-12
            assert np.max(face_points[:, other_axis]) <= length + 1e-12

    expected_face_areas = {
        source.complex.facet_ids[0]: lengths[0] * lengths[1],
        source.complex.facet_ids[1]: lengths[0] * lengths[1],
        source.complex.facet_ids[2]: lengths[0] * lengths[2],
        source.complex.facet_ids[3]: lengths[1] * lengths[2],
        source.complex.facet_ids[4]: lengths[1] * lengths[2],
        source.complex.facet_ids[5]: lengths[0] * lengths[2],
    }
    assert set(facet_areas) == set(expected_face_areas)
    for facet_id, area in expected_face_areas.items():
        np.testing.assert_allclose(facet_areas[facet_id], area, rtol=0.0, atol=1e-12)

    for constraint in constraints:
        source_facet = source.source_entity_key_for_scope(
            2, int(np.asarray(constraint.source_scope.entity_ids)[0])
        )[0]
        target_facet = source.source_entity_key_for_scope(
            2, int(np.asarray(constraint.target_scope.entity_ids)[0])
        )[0]
        source_rows = np.asarray(
            [row_by_face_id[identifier] for identifier in facet_face_ids[source_facet]],
            dtype=np.int64,
        )
        target_rows = np.asarray(
            [row_by_face_id[identifier] for identifier in facet_face_ids[target_facet]],
            dtype=np.int64,
        )
        source_orbits = lifted_to_quotient[source_rows]
        target_orbits = lifted_to_quotient[target_rows]
        assert np.array_equal(np.sort(source_orbits), np.sort(target_orbits))
        lattice = np.asarray(periodic.cell.vectors, dtype=np.float64)
        translation = np.asarray(constraint.transform, dtype=np.float64)[:3, 3]
        generator = int(np.flatnonzero(np.all(lattice == translation, axis=1))[0])
        source_windings = {
            int(orbit): winding_shifts[row]
            for row, orbit in zip(source_rows, source_orbits, strict=True)
        }
        target_windings = {
            int(orbit): winding_shifts[row]
            for row, orbit in zip(target_rows, target_orbits, strict=True)
        }
        for orbit in source_windings:
            delta = target_windings[orbit] - source_windings[orbit]
            expected = np.zeros((periodic.cell.rank,), dtype=np.int64)
            expected[generator] = 1
            np.testing.assert_array_equal(delta, expected)


def _material_l_periodic_request() -> tuple[
    NativePolyhedralSource,
    VolumeMeshingSpec,
    phx.meshing.PeriodicConstraint,
]:
    complex_ = _plc_material_l()
    source = NativePolyhedralSource(
        complex_,
        "periodic-material-l",
        "r1",
        sites=np.asarray(
            ((0.5, 0.5, 0.5), (0.5, 1.5, 0.5)),
            dtype=np.float64,
        ),
        weights=np.zeros((2,), dtype=np.float64),
        site_ids=("base-site", "upper-site"),
    )
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    polygon_offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    source_facet_rows: list[int] = []
    target_facet_rows: list[int] = []
    for polygon, facet_row in enumerate(polygon_facets.tolist()):
        first, last = map(int, polygon_offsets[polygon : polygon + 2])
        points = np.asarray(complex_.vertices, dtype=np.float64)[
            polygon_vertices[first:last]
        ]
        if np.all(points[:, 2] == 0.0):
            source_facet_rows.append(facet_row)
        if np.all(points[:, 2] == 1.0):
            target_facet_rows.append(facet_row)
    if not source_facet_rows or len(source_facet_rows) != len(target_facet_rows):
        raise ValueError("The material-L source must have matching top and bottom facets.")

    def facet_scope(rows: list[int], /) -> MeshingScope:
        return MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray(
                [complex_.facet_entity_ids[row] for row in rows], dtype=np.int64
            ),
        )

    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 1.0
    periodic = phx.meshing.PeriodicConstraint(
        facet_scope(source_facet_rows),
        facet_scope(target_facet_rows),
        transform,
        tolerance=0.0,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                "base",
                "base-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.5, 1.5, 0.5), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
        ),
        periodic_constraints=(periodic,),
    )
    return source, specification, periodic


def test_periodic_material_l_polyhedral_fill_preserves_material_and_coverage() -> None:
    source, specification, periodic_constraint = _material_l_periodic_request()
    result = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "plc_restricted_power",
            polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
        )
    ).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()

    mesh = result.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Material-L restricted-power fill must remain polyhedral.")
    _assert_reciprocal_internal_faces(mesh)
    periodic = mesh.periodic_topology
    if periodic is None or not isinstance(periodic.cell, PeriodicCell):
        raise ValueError("Material-L fill lost its periodic quotient.")
    assert periodic.cell.rank == 1
    finite_volume = prepare_polyhedral_finite_volume_geometry(mesh)
    volumes = _independent_polyhedral_volumes(mesh)
    np.testing.assert_allclose(
        np.asarray(finite_volume.cell_volumes), volumes, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(np.sum(volumes), 3.0, rtol=0.0, atol=1e-12)
    assert result.certification is not None and result.certification.passed
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification.inputs.source is source
    assert {zone.material_id for zone in result.zones} == {
        "base-material",
        "upper-material",
    }
    cell_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        )
    }
    volumes_by_material = {
        zone.material_id: float(
            np.sum(
                np.asarray(finite_volume.cell_volumes)[
                    np.asarray(
                        [
                            cell_rows[int(identifier)]
                            for identifier in np.asarray(zone.scope.entity_ids, dtype=np.int64)
                        ],
                        dtype=np.int64,
                    )
                ]
            )
        )
        for zone in result.zones
        if zone.material_id is not None
    }
    np.testing.assert_allclose(volumes_by_material["base-material"], 2.0, atol=1e-12)
    np.testing.assert_allclose(volumes_by_material["upper-material"], 1.0, atol=1e-12)

    face_association = next(
        association
        for association in result.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    face_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        )
    }
    actual_areas: dict[str, float] = {}
    for target_id, source_id, role in zip(
        np.asarray(face_association.target_global_ids, dtype=np.int64),
        face_association.source_entity_ids,
        face_association.source_roles,
        strict=True,
    ):
        if role is GeometrySourceEntityRole.FACET:
            actual_areas[source_id] = actual_areas.get(source_id, 0.0) + (
                _independent_face_area(mesh, face_rows[int(target_id)])
            )
    expected_areas = {
        identifier: 0.0 for identifier in source.complex.facet_ids
    }
    polygon_facets = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    polygon_offsets = np.asarray(source.complex.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(source.complex.polygon_vertices, dtype=np.int64)
    for polygon, facet_row in enumerate(polygon_facets.tolist()):
        first, last = map(int, polygon_offsets[polygon : polygon + 2])
        points = np.asarray(source.complex.vertices, dtype=np.float64)[
            polygon_vertices[first:last]
        ]
        area_vector = 0.5 * np.sum(
            np.cross(points, np.roll(points, -1, axis=0)), axis=0
        )
        expected_areas[source.complex.facet_ids[facet_row]] += float(
            np.linalg.norm(area_vector)
        )
    assert set(actual_areas) == set(expected_areas)
    for facet_id, area in expected_areas.items():
        np.testing.assert_allclose(actual_areas[facet_id], area, rtol=0.0, atol=1e-12)

    orbit_map = periodic_entity_orbit_map(mesh, 2)
    quotient_rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    windings = np.asarray(orbit_map.winding_shifts, dtype=np.int64)
    source_keys = {
        source.source_entity_key_for_scope(2, int(identifier))
        for identifier in np.asarray(periodic_constraint.source_scope.entity_ids).tolist()
    }
    target_keys = {
        source.source_entity_key_for_scope(2, int(identifier))
        for identifier in np.asarray(periodic_constraint.target_scope.entity_ids).tolist()
    }
    source_faces = [
        int(identifier)
        for identifier, source_id, role, path in zip(
            np.asarray(face_association.target_global_ids, dtype=np.int64),
            face_association.source_entity_ids,
            face_association.source_roles,
            face_association.source_occurrence_paths,
            strict=True,
        )
        if role is GeometrySourceEntityRole.FACET
        and (source_id, path) in source_keys
    ]
    target_faces = [
        int(identifier)
        for identifier, source_id, role, path in zip(
            np.asarray(face_association.target_global_ids, dtype=np.int64),
            face_association.source_entity_ids,
            face_association.source_roles,
            face_association.source_occurrence_paths,
            strict=True,
        )
        if role is GeometrySourceEntityRole.FACET
        and (source_id, path) in target_keys
    ]
    source_rows = np.asarray([face_rows[identifier] for identifier in source_faces])
    target_rows = np.asarray([face_rows[identifier] for identifier in target_faces])
    assert np.array_equal(
        np.sort(quotient_rows[source_rows]), np.sort(quotient_rows[target_rows])
    )
    source_windings = {
        int(quotient_rows[row]): windings[row] for row in source_rows
    }
    target_windings = {
        int(quotient_rows[row]): windings[row] for row in target_rows
    }
    for quotient_row, source_winding in source_windings.items():
        np.testing.assert_array_equal(
            target_windings[quotient_row] - source_winding,
            np.asarray((1,), dtype=np.int64),
        )


def _rotational_wedge_request() -> tuple[
    NativePolyhedralSource,
    VolumeMeshingSpec,
    phx.meshing.PeriodicConstraint,
]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    facets = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (0, 3, 5, 2),
    )
    segments = np.asarray(
        sorted(
            {
                tuple(sorted((face[index], face[(index + 1) % len(face)])))
                for face in facets
                for index in range(len(face))
            }
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        tuple(np.asarray(face, dtype=np.int64) for face in facets),
        np.arange(len(facets), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(facets), 1)),
        ("wedge",),
        segments=segments,
        vertex_ids=tuple(f"wedge-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"wedge-facet-{row}" for row in range(len(facets))),
        facet_occurrence_paths=((),) * len(facets),
        segment_ids=tuple(f"wedge-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * segments.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePolyhedralSource(
        complex_,
        "rotational-periodic-wedge",
        "r1",
        sites=np.asarray(((0.25, 0.25, 0.5),), dtype=np.float64),
        weights=np.zeros((1,), dtype=np.float64),
        site_ids=("wedge-site",),
    )

    def facet_scope(facet_row: int, /) -> MeshingScope:
        return MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray((complex_.facet_entity_ids[facet_row],), dtype=np.int64),
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64)
    periodic = phx.meshing.PeriodicConstraint(
        facet_scope(2),
        facet_scope(4),
        transform,
        tolerance=0.0,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        region_seeds=(
            RegionSeed(
                np.asarray((0.25, 0.25, 0.5), dtype=np.float64),
                "wedge",
                "wedge-material",
                RegionRole.SOLID,
            ),
        ),
        periodic_constraints=(periodic,),
    )
    return source, specification, periodic


def test_rotational_periodic_plc_uses_finite_isometry_quotient() -> None:
    source, specification, periodic_constraint = _rotational_wedge_request()
    result = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions(
            "plc_restricted_power",
            polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
        )
    ).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()
    mesh = result.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Rotational periodic output must remain polyhedral.")
    periodic = mesh.periodic_topology
    if periodic is None or not isinstance(
        periodic.cell, PeriodicIsometryGroup
    ):
        raise ValueError("The rotational PLC lost its finite isometry quotient.")
    assert periodic.cell.orders == (4,)
    assert result.certification is not None and result.certification.passed
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification.inputs.source is source

    finite_volume = prepare_polyhedral_finite_volume_geometry(mesh)
    independent = _independent_polyhedral_volumes(mesh)
    np.testing.assert_allclose(
        np.asarray(finite_volume.cell_volumes), independent, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(np.sum(independent), 0.5, rtol=0.0, atol=1e-12)

    face_association = next(
        association
        for association in result.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    face_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        )
    }
    source_facet = source.source_entity_key_for_scope(
        2, int(np.asarray(periodic_constraint.source_scope.entity_ids)[0])
    )[0]
    target_facet = source.source_entity_key_for_scope(
        2, int(np.asarray(periodic_constraint.target_scope.entity_ids)[0])
    )[0]
    source_faces = [
        int(identifier)
        for identifier, source_id, role in zip(
            np.asarray(face_association.target_global_ids, dtype=np.int64),
            face_association.source_entity_ids,
            face_association.source_roles,
            strict=True,
        )
        if role is GeometrySourceEntityRole.FACET and source_id == source_facet
    ]
    target_faces = [
        int(identifier)
        for identifier, source_id, role in zip(
            np.asarray(face_association.target_global_ids, dtype=np.int64),
            face_association.source_entity_ids,
            face_association.source_roles,
            strict=True,
        )
        if role is GeometrySourceEntityRole.FACET and source_id == target_facet
    ]
    face_orbits = periodic_entity_orbit_map(mesh, 2)
    quotient_rows = np.asarray(face_orbits.lifted_to_quotient, dtype=np.int64)
    windings = np.asarray(face_orbits.winding_shifts, dtype=np.int64)
    source_rows = np.asarray([face_rows[identifier] for identifier in source_faces])
    target_rows = np.asarray([face_rows[identifier] for identifier in target_faces])
    assert np.array_equal(
        np.sort(quotient_rows[source_rows]), np.sort(quotient_rows[target_rows])
    )
    source_windings = {
        int(quotient_rows[row]): windings[row] for row in source_rows
    }
    target_windings = {
        int(quotient_rows[row]): windings[row] for row in target_rows
    }
    assert source_windings.keys() == target_windings.keys()
    for quotient_row, source_winding in source_windings.items():
        delta = (
            target_windings[quotient_row] - source_winding
        ) % periodic.cell.orders[0]
        np.testing.assert_array_equal(delta, np.asarray((1,), dtype=np.int64))

# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from phydrax import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    SimplicialLocationPolicy,
)
from phydrax.discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry.brep import PreparedBRepQuery
from phydrax.meshing import CellMeshingResult, certify_cell_mesh, MeshingScope
from phydrax.meshing._assembly import MeshAssembly, MeshPart
from phydrax.meshing._coupling import CouplingSearchStatus, OversetCoupling
from phydrax.meshing._overset import (
    _resolve_donors,
    OversetCellStatus,
    OversetConnectivity,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
)


jax.config.update("jax_enable_x64", True)
_CONTRACT = SpatialCoordinateContract.si()


class _SphereWallBound(TypedDict):
    part_id: str
    geometry_id: str
    solid_revision: str
    maximum_distance: float


def _carrier(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            "The overset geometry fixture requires a certified cell-mesh producer."
        )
    return carrier


def _part(
    name: str, mesh: CellMesh, geometry: CellGeometrySpec | None = None
) -> MeshPart:
    return MeshPart(name, certify_cell_mesh(mesh, _CONTRACT, geometry=geometry))


def _boundary(part: MeshPart, *, inner: bool = False) -> MeshingScope:
    mesh = _carrier(part).mesh
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        mask = np.asarray(connectivity.boundary_edges)
    elif isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        mask = np.asarray(connectivity.boundary_faces).copy()
        if inner:
            points = np.asarray(mesh.coordinates)
            if isinstance(connectivity, PolyhedralConnectivity):
                offsets = np.asarray(connectivity.face_vertex_offsets)
                values = np.asarray(connectivity.face_vertex_values)
                selected = np.asarray(
                    [
                        np.max(np.abs(points[values[start:stop]])) < 0.3
                        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
                    ]
                )
            else:
                selected = (
                    np.max(np.abs(points[np.asarray(connectivity.faces)]), axis=(1, 2))
                    < 0.3
                )
            mask &= selected
    else:
        raise TypeError(
            "The wall fixture requires two- or three-dimensional connectivity."
        )
    return part.scope(
        mesh.ambient_dimension - 1,
        np.asarray(mesh.entity_set(mesh.ambient_dimension - 1).entity_ids)[mask],
    )


def _receptor(centers: ArrayLike) -> MeshPart:
    centers = np.asarray(centers)
    dimension = centers.shape[1]
    corners = np.concatenate((np.zeros((1, dimension)), 0.025 * np.eye(dimension)))
    points = (centers[:, None] + corners[None]).reshape((-1, dimension))
    cells = np.arange(points.shape[0]).reshape((-1, dimension + 1))
    kind = "triangle" if dimension == 2 else "tetrahedron"
    return _part("receptor", CellMesh(points, (CellBlock("receptors", kind, cells),)))


def _register(
    source: MeshPart,
    receptor: MeshPart,
    *,
    policy: OversetPolicy | None = None,
) -> OversetConnectivity:
    return prepare_overset_connectivity(
        MeshAssembly((source, receptor)),
        (
            OversetPartSpec(source.name),
            OversetPartSpec(receptor.name, boundary=_boundary(receptor)),
        ),
        policy=policy or OversetPolicy(fringe_layers=1),
    )


@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "prism", "hexahedron", "pyramid")
)
def test_regular_mapped_family_uses_actual_inverse_and_scientific_ids(kind: str) -> None:
    reference = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    element = coordinate_lagrange_element(kind, 2)

    def physical(points: ArrayLike) -> NDArray[np.float64]:
        points = np.asarray(points, dtype=np.float64).copy()
        points[:, -1] += 0.08 * points[:, 0] * (1 - points[:, 0])
        return points

    nodes = physical(element.reference_nodes)
    mesh = CellMesh(
        physical(reference),
        (
            CellBlock(
                "mapped",
                kind,
                np.arange(len(reference))[None],
                global_ids=np.asarray([701]),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"mapped": element}, {"mapped": np.arange(len(nodes))[None]}, nodes
    )
    source = _part("donor", mesh, geometry)
    receptor = _receptor(physical(np.mean(reference, axis=0)[None]))
    connectivity = _register(source, receptor)
    connectivity.require_complete()
    np.testing.assert_array_equal(connectivity.receptors_of("receptor").donor_cells, 701)
    (packet,) = connectivity.packets
    assert packet.donor_block == "mapped"
    np.testing.assert_array_equal(packet.donor_cell_rows, 0)
    assert np.all(np.asarray(connectivity.receptors_of("receptor").residuals) < 1e-10)
    assert not connectivity.assembly.couplings


def test_mixed_blocks_preserve_flat_ids_and_block_local_packet_routes() -> None:
    points, blocks, centers = [], [], []
    for row, kind in enumerate(("hexahedron", "prism", "pyramid", "tetrahedron")):
        vertices = np.asarray(
            reference_cell_topology(kind).vertices, dtype=np.float64
        ) + (3 * row, 0, 0)
        first = len(points)
        points.extend(vertices)
        blocks.append(
            CellBlock(
                kind,
                kind,
                np.arange(first, first + len(vertices))[None],
                global_ids=np.asarray([91 - 7 * row]),
            )
        )
        centers.append(np.mean(vertices, axis=0))
    source = _part("mixed", CellMesh(np.asarray(points), tuple(blocks)))
    connectivity = _register(source, _receptor(centers))
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.blanking_of("mixed").cell_ids, [91, 84, 77, 70]
    )
    for packet in connectivity.packets:
        block = _carrier(source).mesh.block(packet.donor_block)
        np.testing.assert_array_equal(packet.donor_cells, block.global_ids[0])
        np.testing.assert_array_equal(packet.donor_cell_rows, 0)
    assert {packet.donor_block for packet in connectivity.packets} == {
        block.name for block in blocks
    }
    (affine_overlay,) = connectivity.assembly.couplings
    assert isinstance(affine_overlay, OversetCoupling)
    assert affine_overlay.search_evidence is not None
    original_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in _carrier(source).mesh.blocks]
    )
    np.testing.assert_array_equal(
        original_ids[np.asarray(affine_overlay.search_evidence.source_cells)], 70
    )


def test_polyhedral_original_cells_are_not_private_fan_tetrahedra() -> None:
    topology = reference_cell_topology("hexahedron")
    mesh = CellMesh.from_polyhedra(
        np.asarray(topology.vertices, dtype=np.float64),
        (tuple(np.asarray(face) for face in topology.entities[2]),),
        cell_global_ids=np.asarray([802]),
    )
    source = _part("polyhedral", mesh)
    connectivity = _register(source, _receptor([[0.3, 0.3, 0.3]]))
    connectivity.require_complete()
    np.testing.assert_array_equal(connectivity.receptors_of("receptor").donor_cells, 802)
    (packet,) = connectivity.packets
    np.testing.assert_array_equal(packet.donor_cell_rows, 0)
    assert (
        connectivity.geometries[connectivity.part_index(source.name)].discretization
        is None
    )
    assert not connectivity.assembly.couplings


def test_polyhedral_blocks_with_different_face_widths_keep_original_row_identity() -> (
    None
):
    box = reference_cell_topology("hexahedron")
    tet = reference_cell_topology("tetrahedron")
    points = np.concatenate(
        (np.asarray(box.vertices), np.asarray(tet.vertices) + [3.0, 0.0, 0.0])
    )
    faces = {
        "box": (tuple(np.asarray(face) for face in box.entities[2]),),
        "tet": (tuple(np.asarray(face) + 8 for face in tet.entities[2]),),
    }
    mesh = CellMesh.from_mixed_3d(
        points,
        (),
        polyhedra=faces,
        polyhedral_cell_global_ids={"box": np.asarray([83]), "tet": np.asarray([27])},
    )
    source = _part("polyhedral", mesh)
    connectivity = _register(source, _receptor([[0.3, 0.3, 0.3], [3.15, 0.15, 0.15]]))
    connectivity.require_complete()
    assert {
        (packet.donor_block, int(np.asarray(packet.donor_cells)[0]))
        for packet in connectivity.packets
    } == {("box", 83), ("tet", 27)}
    for packet in connectivity.packets:
        np.testing.assert_array_equal(packet.donor_cell_rows, 0)


def test_closed_mapped_wall_classifies_beyond_its_affine_corner_hull() -> None:
    from phydrax.discretization._view_support import mapped_mesh_support_geometry
    from phydrax.meshing._overset import _part_geometry

    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    element = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(element.reference_nodes).copy()
    nodes[:, 2] -= 0.2 * nodes[:, 0] * (1 - nodes[:, 0])
    mesh = CellMesh(reference, (CellBlock("solid", "hexahedron", np.arange(8)[None]),))
    geometry = CellGeometrySpec(
        {"solid": element}, {"solid": np.arange(len(nodes))[None]}, nodes
    )
    body = _part("body", mesh, geometry)
    owner = _part_geometry(body, OversetPartSpec(body.name), OversetPolicy(), None)
    authority = mapped_mesh_support_geometry(owner.locators[0], "closed-curved-solid")
    background = _receptor([[0.45, 0.45, -0.03]])
    connectivity = prepare_overset_connectivity(
        MeshAssembly((body, background)),
        (
            OversetPartSpec(body.name, wall=_boundary(body), solid_query=authority),
            OversetPartSpec(background.name),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.blanking_of(background.name).vertex_status,
        int(OversetVertexStatus.HOLE),
    )
    assert not np.any(np.asarray(connectivity.blanking_of(body.name).protected_conflict))
    wall_owner = connectivity.geometries[connectivity.part_index(body.name)]
    wall_bvh = wall_owner.wall_bvh
    if wall_bvh is None:
        raise ValueError("The declared mapped wall must publish its actual wall BVH.")
    assert np.min(np.asarray(wall_bvh.item_bbox_min)[:, 2]) < -0.03


def test_restricted_parent_coefficients_are_not_child_corner_geometry() -> None:
    kind = "hexahedron"
    reference = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    parent = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(parent.reference_nodes).copy()
    nodes[:, 2] += 0.15 * nodes[:, 0] * (1 - nodes[:, 0])
    restricted = RestrictedCellGeometryElement(parent, kind, 0.5 * np.eye(3), np.zeros(3))
    basis, _ = restricted.tabulate(jnp.asarray(reference))
    mesh = CellMesh(
        np.asarray(basis) @ nodes,
        (CellBlock("child", kind, np.arange(8)[None], global_ids=np.asarray([62])),),
    )
    geometry = CellGeometrySpec(
        {"child": restricted}, {"child": np.arange(len(nodes))[None]}, nodes
    )
    source = _part("restricted", mesh, geometry)
    reference_site = np.asarray([[0.3, 0.3, 0.3]])
    basis, _ = restricted.tabulate(jnp.asarray(reference_site))
    connectivity = _register(source, _receptor(np.asarray(basis) @ nodes))
    connectivity.require_complete()
    np.testing.assert_array_equal(connectivity.receptors_of("receptor").donor_cells, 62)
    owner = connectivity.geometries[connectivity.part_index(source.name)]
    assert np.max(np.asarray(owner.cell_upper)[:, 2]) > np.max(
        np.asarray(mesh.coordinates)[:, 2]
    )


def test_candidate_truncation_never_selects_an_unproven_mixed_donor() -> None:
    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    # Adjacent cells meet at x=1; both contain the closed boundary point.
    points = np.concatenate((reference, reference + [1.0, 0.0, 0.0]))
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    source = _part(
        "donor",
        CellMesh(
            unique,
            (
                CellBlock(
                    "boxes",
                    "hexahedron",
                    inverse.reshape((2, 8)),
                    global_ids=np.asarray([90, 10]),
                ),
            ),
        ),
    )
    receptor = _receptor([[0.2, 0.2, 0.2]])
    complete = _register(source, receptor)
    geometry = complete.geometries
    specs = complete.specs
    index = complete.part_index("receptor")
    masks = tuple(jnp.ones(item.cell_ids.shape, dtype=jnp.bool_) for item in geometry)
    selected = _resolve_donors(
        np.asarray([[1.0, 0.5, 0.5]]), index, specs, geometry, masks, complete.policy
    )
    donor_geometry = geometry[complete.part_index(source.name)]
    expected = np.lexsort(
        (np.asarray(donor_geometry.cell_ids), np.asarray(donor_geometry.cell_measures))
    )[0]
    assert selected.cells[0] == expected
    policy = OversetPolicy(
        fringe_layers=1, location_policy=SimplicialLocationPolicy(1, 16, 1)
    )
    truncated = _register(source, receptor, policy=policy)
    result = _resolve_donors(
        np.asarray([[1.0, 0.5, 0.5]]),
        index,
        truncated.specs,
        truncated.geometries,
        masks,
        policy,
    )
    np.testing.assert_array_equal(
        result.status, int(CouplingSearchStatus.RESOURCE_EXCEEDED)
    )
    np.testing.assert_array_equal(result.parts, -1)


def test_equal_mapped_parts_use_name_before_scientific_cell_id() -> None:
    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    alpha = _part(
        "alpha",
        CellMesh(
            reference,
            (
                CellBlock(
                    "hex", "hexahedron", np.arange(8)[None], global_ids=np.asarray([90])
                ),
            ),
        ),
    )
    zeta = _part(
        "zeta",
        CellMesh(
            reference,
            (
                CellBlock(
                    "hex", "hexahedron", np.arange(8)[None], global_ids=np.asarray([10])
                ),
            ),
        ),
    )
    receptor = _receptor([[0.3, 0.3, 0.3]])
    connectivity = prepare_overset_connectivity(
        MeshAssembly((zeta, receptor, alpha)),
        (
            OversetPartSpec(zeta.name),
            OversetPartSpec(receptor.name, boundary=_boundary(receptor)),
            OversetPartSpec(alpha.name),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    np.testing.assert_array_equal(
        connectivity.receptors_of(receptor.name).candidate_parts, 2
    )
    np.testing.assert_array_equal(
        connectivity.receptors_of(receptor.name).donor_cells, 90
    )
    assert {packet.donor_part for packet in connectivity.packets} == {"alpha"}


def test_actual_candidate_capacity_honors_cell_count_and_hard_budget() -> None:
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory

    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    source = _part(
        "donor",
        CellMesh(reference, (CellBlock("hex", "hexahedron", np.arange(8)[None]),)),
    )
    receptor = _receptor([[0.3, 0.3, 0.3]])
    policy = OversetPolicy(
        fringe_layers=1,
        location_policy=SimplicialLocationPolicy(64, 16, 1),
        maximum_donor_candidate_pairs=4,
    )
    connectivity = _register(source, receptor, policy=policy)
    connectivity.require_complete()
    with pytest.raises(MeshingFailure) as error:
        _register(
            source,
            receptor,
            policy=OversetPolicy(
                fringe_layers=1,
                location_policy=policy.location_policy,
                maximum_donor_candidate_pairs=3,
            ),
        )
    assert error.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


def _grid(values: Sequence[float], *, shell: bool = False) -> CellMesh:
    reference = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    points = np.asarray(tuple(product(values, repeat=3)), dtype=np.float64)
    index = {tuple(point): row for row, point in enumerate(points)}
    cells = []
    for i, j, k in product(range(len(values) - 1), repeat=3):
        if shell and (i, j, k) == (len(values) // 2 - 1,) * 3:
            continue
        cells.append(
            [
                index[(values[i + x], values[j + y], values[k + z])]
                for x, y, z in reference
            ]
        )
    if shell:
        inner = np.max(np.abs(points), axis=1) < 0.3
        points[inner] *= 0.2 / np.linalg.norm(points[inner], axis=1)[:, None]
    return CellMesh(points, (CellBlock("hexes", "hexahedron", np.asarray(cells)),))


def _mapped_sphere_body() -> MeshPart:
    """Actual quadratic shell source, not a vertex/chord interpolation surrogate."""
    values = (-1.0, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 1.0)
    mesh = _grid(values, shell=True)
    element = coordinate_lagrange_element("hexahedron", 2)
    reference = np.asarray(element.reference_nodes)
    logical = np.asarray(tuple(product(values, repeat=3)))
    index = {value: row for row, value in enumerate(values)}
    origins = np.asarray(
        [
            [index[float(value)] for value in logical[cell[0]]]
            for cell in np.asarray(mesh.blocks[0].vertices)
        ],
        dtype=np.int64,
    )
    keys = (
        2 * origins[:, None, :] + np.rint(2 * reference).astype(np.int64)[None]
    ).reshape((-1, 3))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    axis = np.asarray(
        [
            value
            for first, last in zip(values[:-1], values[1:], strict=True)
            for value in (first, 0.5 * (first + last))
        ]
        + [values[-1]]
    )
    coordinates = axis[unique]
    inner = np.max(np.abs(coordinates), axis=1) <= 0.2
    coordinates[inner] *= 0.2 / np.linalg.norm(coordinates[inner], axis=1)[:, None]
    geometry = CellGeometrySpec(
        {"hexes": element},
        {"hexes": inverse.reshape((-1, element.local_dof_count))},
        coordinates,
    )
    return _part("body", mesh, geometry)


def _sphere_wall_bound(
    part: MeshPart,
    wall: MeshingScope,
    query: PreparedBRepQuery,
    *,
    subdivisions: int = 8,
) -> _SphereWallBound:
    """Bound the whole polynomial wall's physical distance to its exact sphere."""
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._coordinate_enclosure import (
        add,
        affine_arguments,
        compose,
        constant,
        coordinate_polynomials,
        multiply,
        Polynomial,
        polynomial_bounds,
        restrict_coordinates,
        sum_polynomials,
    )
    from phydrax.geometry.brep._patches import SpherePatch
    from phydrax.meshing._overset import _facet_rows

    if len(query.faces) != 1:
        raise TypeError(
            "The independent comparison authority must be the complete sphere."
        )
    patch = query.faces[0].patch
    if not isinstance(patch, SpherePatch):
        raise TypeError(
            "The independent comparison authority must be the complete sphere."
        )
    center, radius = np.asarray(patch.center), float(patch.radius)
    mesh = _carrier(part).mesh
    elements, routes, coordinates = _carrier(part).geometry.resolve(mesh)
    topology = reference_cell_topology("hexahedron")
    selected = {tuple(sorted(row)) for row in _facet_rows(part, wall)}
    bounds: list[float] = []
    sources: dict[int, tuple[Polynomial, ...]] = {}
    for cell, vertices in enumerate(np.asarray(mesh.blocks[0].vertices)):
        for facet in topology.entities[2]:
            if tuple(sorted(vertices[list(facet)])) not in selected:
                continue
            if cell not in sources:
                source = coordinate_polynomials(
                    elements[0], np.asarray(coordinates)[np.asarray(routes[0])[cell]]
                )
                if source is None:
                    raise ValueError(
                        "The wall bound requires the actual polynomial coordinate source."
                    )
                sources[cell] = source
            reference = np.asarray(topology.vertices)[list(facet)]
            axes = np.stack(
                (reference[1] - reference[0], reference[3] - reference[0]), axis=1
            )
            expressions = restrict_coordinates(
                sources[cell], "hexahedron", reference[0], axes
            )
            if expressions is None:
                raise ValueError(
                    "The actual wall restriction must retain its polynomial coordinates."
                )
            relative = tuple(
                add(value, constant(-Fraction.from_float(float(origin)), 2))
                for value, origin in zip(expressions, center, strict=True)
            )
            norm_squared = sum_polynomials(
                tuple(multiply(value, value) for value in relative)
            )
            error = 0.0
            for i, j in product(range(subdivisions), repeat=2):
                arguments = affine_arguments(
                    np.asarray((i, j)) / subdivisions, np.eye(2) / subdivisions
                )
                lower, upper = polynomial_bounds(
                    compose(norm_squared, arguments), "box", 2
                )
                radial_lower = np.nextafter(np.sqrt(max(0.0, lower)), -np.inf)
                radial_upper = np.nextafter(np.sqrt(upper), np.inf)
                error = float(
                    max(
                        error,
                        np.nextafter(
                            max(radius - radial_lower, radial_upper - radius), np.inf
                        ),
                    )
                )
            bounds.append(error)
    if len(bounds) != len(selected):
        raise ValueError("The bound must cover every declared wall facet exactly once.")
    return {
        "part_id": part.part_id,
        "geometry_id": cell_geometry_id(_carrier(part).geometry),
        "solid_revision": query.source_revision,
        "maximum_distance": max(bounds),
    }


def _sphere_query(center: tuple[float, float, float]) -> PreparedBRepQuery:
    """Real native sphere authority; display realization is explicitly unrequested."""
    from phydrax.geometry.brep import (
        brep_sphere,
        BRepTessellationPolicy,
        prepare_brep_query,
    )

    return prepare_brep_query(
        brep_sphere(
            0.2,
            center=center,
            coordinate_contract=_CONTRACT,
            tessellation=BRepTessellationPolicy(realize=False),
        )
    )


def test_native_sphere_wall_holes_protection_and_moving_donor_refresh() -> None:
    """Curved BRep source classification on a chordal body; no exact-face fidelity claim."""
    query = _sphere_query((0.0, 0.0, 0.0))
    background = _part(
        "background", _grid((-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6))
    )
    body = _part("body", _grid((-1.0, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 1.0), shell=True))
    wall = _boundary(body, inner=True)
    exterior = _boundary(body)
    outer = body.scope(
        2, np.setdiff1d(np.asarray(exterior.entity_ids), np.asarray(wall.entity_ids))
    )
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, body)),
        (
            OversetPartSpec("background"),
            OversetPartSpec("body", wall=wall, boundary=outer, solid_query=query),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    initial = connectivity.blanking_of("background")
    assert np.any(np.asarray(initial.vertex_status) == int(OversetVertexStatus.HOLE))
    assert np.any(np.asarray(initial.cell_status) == int(OversetCellStatus.HOLE))
    assert not np.any(np.asarray(connectivity.blanking_of("body").protected_conflict))
    assert np.all(np.asarray(connectivity.receptors_of("background").found))
    moved_query = _sphere_query((0.3, 0.0, 0.0))
    moved = connectivity.moved(
        {"body": _carrier(body).mesh.coordinates + jnp.asarray((0.3, 0.0, 0.0))},
        solid_queries={"body": moved_query},
    )
    moved.require_complete()
    assert moved.epoch == connectivity.epoch + 1
    assert not np.any(np.asarray(moved.blanking_of("body").protected_conflict))
    assert np.all(np.asarray(moved.receptors_of("background").found))
    assert np.any(
        np.asarray(initial.vertex_status)
        != np.asarray(moved.blanking_of("background").vertex_status)
    )
    assert np.any(
        np.asarray(initial.cell_status)
        != np.asarray(moved.blanking_of("background").cell_status)
    )
    assert {packet.packet_id for packet in moved.packets}.isdisjoint(
        {packet.packet_id for packet in connectivity.packets}
    )
    # A source-box contact without any enclosed vertex remains ambiguous,
    # not falsely promoted to a certified sphere/cell intersection.
    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    cut_part = _part(
        "cut",
        CellMesh(
            0.4 * reference - 0.2, (CellBlock("cut", "hexahedron", np.arange(8)[None]),)
        ),
    )
    underresolved = prepare_overset_connectivity(
        MeshAssembly((body, cut_part)),
        (
            OversetPartSpec(body.name, wall=wall, solid_query=query),
            OversetPartSpec(cut_part.name),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    underresolved.require_complete()
    cut = underresolved.blanking_of(cut_part.name)
    np.testing.assert_array_equal(cut.ambiguous_cells, True)
    np.testing.assert_array_equal(cut.cell_status, int(OversetCellStatus.HOLE))
    assert not np.any(np.asarray(cut.vertex_status) == int(OversetVertexStatus.HOLE))
    assert not np.any(np.asarray(cut.wall_contact))


def test_quadratic_mapped_sphere_wall_fidelity_and_two_sided_motion() -> None:
    body = _mapped_sphere_body()
    background = _part(
        "background", _grid((-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6))
    )
    wall = _boundary(body, inner=True)
    outer = body.scope(
        2,
        np.setdiff1d(np.asarray(_boundary(body).entity_ids), np.asarray(wall.entity_ids)),
    )
    query = _sphere_query((0.0, 0.0, 0.0))
    bound = _sphere_wall_bound(body, wall, query)
    assert bound["maximum_distance"] <= 0.01
    connectivity = prepare_overset_connectivity(
        MeshAssembly((background, body)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(body.name, wall=wall, boundary=outer, solid_query=query),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    elements, routes, coordinates = _carrier(body).geometry.resolve(_carrier(body).mesh)
    shift = jnp.asarray((0.3, 0.0, 0.0))
    successor_geometry = CellGeometrySpec(
        {"hexes": elements[0]}, {"hexes": routes[0]}, coordinates + shift
    )
    moved_query = _sphere_query((0.3, 0.0, 0.0))
    moved = connectivity.moved(
        {body.name: _carrier(body).mesh.coordinates + shift},
        geometry={body.name: successor_geometry},
        solid_queries={body.name: moved_query},
    )
    moved.require_complete()
    moved_body = moved.assembly.part(body.name)
    moved_wall = moved.specs[moved.part_index(body.name)].wall
    if moved_wall is None:
        raise ValueError(
            "The successor sphere fixture must retain its declared wall scope."
        )
    moved_bound = _sphere_wall_bound(moved_body, moved_wall, moved_query)
    assert moved_bound["maximum_distance"] <= 0.01
    assert not np.any(np.asarray(moved.blanking_of(body.name).protected_conflict))
    assert np.any(
        np.asarray(connectivity.blanking_of(background.name).vertex_status)
        != np.asarray(moved.blanking_of(background.name).vertex_status)
    )
    assert np.any(
        np.asarray(connectivity.blanking_of(background.name).cell_status)
        != np.asarray(moved.blanking_of(background.name).cell_status)
    )
    assert {packet.packet_id for packet in moved.packets}.isdisjoint(
        {packet.packet_id for packet in connectivity.packets}
    )


def test_cubic_curved_donor_field_reproduction_and_motion_transpose() -> None:
    from phydrax.discretization.fem import (
        FiniteElementFieldSpec,
        FiniteElementPlan,
        lagrange_element,
    )
    from phydrax.meshing._overset import prepare_overset_field_transfer

    reference = np.asarray(reference_cell_topology("triangle").vertices, dtype=np.float64)
    element = coordinate_lagrange_element("triangle", 3)

    def physical(points: np.ndarray) -> np.ndarray:
        output = points.copy()
        output[:, 1] += 0.12 * output[:, 0] * (1.0 - output[:, 0])
        return output

    nodes = physical(np.asarray(element.reference_nodes))
    mesh = CellMesh(
        physical(reference),
        (
            CellBlock(
                "curved", "triangle", np.arange(3)[None], global_ids=np.asarray([719])
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": element}, {"curved": np.arange(len(nodes))[None]}, nodes
    )
    source = _part("donor", mesh, geometry)
    registration = _register(source, _receptor(physical(np.asarray([[0.2, 0.25]]))))
    field = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", lagrange_element("triangle", 3)),
        coordinate_spec=geometry,
    ).prepare()
    coordinates = np.asarray(field.dof_maps[0].dof_coordinates)
    values = {
        source.name: jnp.asarray(
            coordinates[:, 0] ** 3 - coordinates[:, 0] ** 2 + 2 * coordinates[:, 1]
        )
    }
    for epoch in range(2):
        registration.require_complete()
        route = prepare_overset_field_transfer(registration, {source.name: field}, "u")
        rows = np.asarray(registration.receptors_of("receptor").receptor_rows)
        points = np.asarray(
            _carrier(registration.assembly.part("receptor")).mesh.coordinates
        )[rows]
        output = route.apply(values)
        np.testing.assert_allclose(
            output["receptor"],
            points[:, 0] ** 3 - points[:, 0] ** 2 + 2 * points[:, 1],
            atol=1e-11,
        )
        cotangents = {
            name: jnp.arange(value.size, dtype=jnp.float64) + 1.0
            for name, value in output.items()
        }
        assert bool(route.duality_evidence(values, cotangents).valid)
        assert not route.conservative
        if epoch == 0:
            receptor = registration.assembly.part("receptor")
            registration = registration.moved(
                {
                    receptor.name: _carrier(receptor).mesh.coordinates
                    + jnp.asarray([0.03, 0.02])
                }
            )


def test_polyhedral_sliver_donor_keeps_exact_cell_identity_and_bounded_support() -> None:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64) * [1.0, 1.0, 1e-5]
    mesh = CellMesh.from_polyhedra(
        points,
        (tuple(np.asarray(face) for face in topology.entities[2]),),
        cell_global_ids=np.asarray([983]),
    )
    source = _part("thin-polyhedron", mesh)
    corners = np.asarray(
        [[0.3, 0.3, 3e-6], [0.31, 0.3, 3e-6], [0.3, 0.31, 3e-6], [0.3, 0.3, 4e-6]]
    )
    receptor = _part(
        "receptor",
        CellMesh(
            corners, (CellBlock("thin-receptor", "tetrahedron", np.arange(4)[None]),)
        ),
    )
    registration = _register(source, receptor)
    registration.require_complete()
    evidence = registration.receptors_of(receptor.name)
    np.testing.assert_array_equal(evidence.donor_cells, 983)
    (packet,) = registration.packets
    np.testing.assert_array_equal(packet.donor_cell_rows, 0)
    assert np.all(np.asarray(evidence.residuals) < 1e-10)

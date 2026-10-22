"""Small oriented cell meshes, native bisection, and coordinate storage fixtures."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._meshcore import exact_orient2d, exact_orient3d
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import nonrepresentable_reference_ancestry
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMeshStorage
from phydrax.discretization._coordinate_enclosure import (
    coordinate_reference_chain,
    coordinate_source_signature,
    RationalPolynomial,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing import (
    BisectionHierarchy,
    CellMeshingResult,
    certify_cell_mesh,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.typing import ConvertibleToArray
from tests._support.coordinate_references import coordinate_reference_budget


def unit_tetrahedron() -> CellMesh:
    """Return the positive unit simplex with default tetrahedral IDs and metadata."""
    return CellMesh.from_tetrahedra(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
    )


def _grid_points(
    count: int,
    dimension: int,
    jitter: float,
    seed: int,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> NDArray[np.float64]:
    if count < 1:
        raise ValueError(f"cell_meshes.count must be positive; got {count}")
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError(
            f"cell_meshes.jitter must be finite and nonnegative; got {jitter}"
        )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"cell_meshes.bounds must be finite and increasing; got {(lower, upper)}"
        )
    axis = np.linspace(lower, upper, count + 1, dtype=np.float64)
    points = np.stack(np.meshgrid(*((axis,) * dimension), indexing="ij"), -1).reshape(
        (-1, dimension)
    )
    # Preserve the outer boundary and the midplane used for oriented fluxes.
    interior = np.all((points > lower) & (points < upper), axis=1)
    shift = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape)
    shift[points[:, 0] == 0.5 * (lower + upper), 0] = 0.0
    return points + np.where(interior[:, None], jitter / count * shift, 0.0)


def _positive_cell_rows(
    cells: NDArray[np.int32],
    signs: np.ndarray,
    orientation_swap: tuple[int, int],
) -> NDArray[np.int32]:
    """Orient cells while retaining the requested local vertex convention."""
    first, second = orientation_swap
    if (
        first == second
        or not 0 <= first < cells.shape[1]
        or not 0 <= second < cells.shape[1]
    ):
        raise ValueError(
            f"cell_meshes.orientation_swap must select two distinct local vertices; got {orientation_swap}"
        )
    order = list(range(cells.shape[1]))
    order[first], order[second] = order[second], order[first]
    cells[signs < 0] = cells[signs < 0][:, order]
    return cells


def unit_square(
    count: int = 1,
    jitter: float = 0.0,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
    orientation_swap: tuple[int, int] = (1, 2),
) -> CellMesh:
    """Return two positive triangles per square; retain boundary and midplane nodes.

    Grid indexing and seed 11 follow the compatible-mesh-transfer fixture.
    Explicit global IDs are passed to the mesh constructor without renumbering.
    The orientation swap selects the local ordering of initially negative cells.
    """
    points = _grid_points(count, 2, jitter, 11)
    cells: list[tuple[int, int, int]] = []
    for i, j in np.ndindex(count, count):
        a, b = i * (count + 1) + j, (i + 1) * (count + 1) + j
        cells.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    triangles = np.asarray(cells, dtype=np.int32)
    corners = points[triangles]
    signs = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
    if np.any(signs == 0):
        raise ValueError(
            "cell_meshes.unit_square must have nondegenerate cells; got zero orientation"
        )
    triangles = _positive_cell_rows(triangles, signs, orientation_swap)
    return CellMesh.from_triangles(
        points,
        triangles,
        vertex_global_ids=vertex_global_ids,
        cell_global_ids=cell_global_ids,
    )


def kuhn_cube_arrays(
    count: int = 1,
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    jitter: float = 0.0,
    orientation_swap: tuple[int, int] = (1, 2),
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Return positive Kuhn coordinates and rows without preparing a CellMesh."""
    points = _grid_points(count, 3, jitter, 7, lower=lower, upper=upper)
    cells: list[list[int]] = []
    for corner in np.ndindex(count, count, count):
        for order in permutations(range(3)):
            vertex = np.asarray(corner, dtype=np.int64)
            path = [vertex.copy()]
            for axis in order:
                vertex[axis] += 1
                path.append(vertex.copy())
            cells.append([np.ravel_multi_index(tuple(v), (count + 1,) * 3) for v in path])
    tetrahedra = np.asarray(cells, dtype=np.int32)
    corners = points[tetrahedra]
    signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    if np.any(signs == 0):
        raise ValueError(
            "cell_meshes.kuhn_cube must have nondegenerate cells; got zero orientation"
        )
    tetrahedra = _positive_cell_rows(tetrahedra, signs, orientation_swap)
    return points, tetrahedra


def kuhn_cube(
    count: int = 1,
    jitter: float = 0.0,
    *,
    vertex_global_ids: ArrayLike | None = None,
    cell_global_ids: ArrayLike | None = None,
    orientation_swap: tuple[int, int] = (1, 2),
) -> CellMesh:
    """Return six positive Kuhn tetrahedra per cube with boundary and midplane fixed.

    Permutation paths and seed 7 follow the compatible-mesh-transfer fixture;
    orientation is decided by the native exact predicate, not a local determinant.
    The orientation swap selects the local ordering of initially negative cells.
    """
    points, tetrahedra = kuhn_cube_arrays(
        count, jitter=jitter, orientation_swap=orientation_swap
    )
    return CellMesh.from_tetrahedra(
        points,
        tetrahedra,
        vertex_global_ids=vertex_global_ids,
        cell_global_ids=cell_global_ids,
    )


def red_refinement() -> tuple[CellMesh, CellMesh, NDArray[np.int64]]:
    """Authored two-triangle red refinement with independent parent ordering."""
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.2, 0.9), (0.1, 1.0)), dtype=np.float64
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    vertices = [tuple(point) for point in points]
    midpoints: dict[tuple[int, int], int] = {}

    def midpoint(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key not in midpoints:
            midpoints[key] = len(vertices)
            vertices.append(tuple(0.5 * (points[first] + points[second])))
        return midpoints[key]

    children: list[tuple[int, int, int]] = []
    parents: list[int] = []
    for parent, (a, b, c) in enumerate(triangles.tolist()):
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        children += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        parents += [parent] * 4
    return (
        CellMesh.from_triangles(points, triangles),
        CellMesh.from_triangles(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(children, dtype=np.int32),
        ),
        np.asarray(parents, dtype=np.int64),
    )


def tetrahedral_bisection() -> tuple[CellMesh, CellMesh, NDArray[np.int64]]:
    """Authored edge bisection of two face-sharing positive tetrahedra."""
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.5, 0.0),
        ),
        dtype=np.float64,
    )
    source = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int32)
    target = np.asarray(
        ((0, 1, 5, 3), (0, 5, 2, 3), (1, 5, 3, 4), (5, 2, 3, 4)),
        dtype=np.int32,
    )
    # Preserve the determinant reduction and local swap: both are
    # identity-bearing conventions of the compatible-transfer reference.
    for cells in (source, target):
        corners = points[cells]
        signs = np.linalg.det(corners[:, 1:] - corners[:, :1])
        _positive_cell_rows(cells, signs, (0, 1))
    return (
        CellMesh.from_tetrahedra(points[:5], source),
        CellMesh.from_tetrahedra(points, target),
        np.asarray((0, 0, 1, 1), dtype=np.int64),
    )


def triangle_grid(columns: int, rows: int, height: float = 1.0) -> CellMesh:
    """Return a rectangular triangular grid in row-major scientific ID order.

    The diagonal runs from each lower-left to upper-right corner. Coordinates
    and cell ordering retain the bisection fixture's rectangle convention.
    """
    xs = np.linspace(0.0, 1.0, columns + 1, dtype=np.float64)
    ys = np.linspace(0.0, height, rows + 1, dtype=np.float64)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    cells: list[tuple[int, int, int]] = []
    for j, i in np.ndindex(rows, columns):
        a = j * (columns + 1) + i
        b, c, d = a + 1, a + columns + 2, a + columns + 1
        cells.extend(((a, b, c), (a, c, d)))
    return CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))


def incompatible_simplex(dimension: int) -> CellMesh:
    """Return two oriented simplices whose longest-edge labels do not match."""
    match dimension:
        case 2:
            return CellMesh.from_triangles(
                np.asarray(
                    ((0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)), dtype=np.float64
                ),
                np.asarray(((0, 1, 2), (0, 3, 1)), dtype=np.int32),
            )
        case 3:
            return CellMesh.from_tetrahedra(
                np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (2.0, 0.0, 0.0),
                        (1.0, 0.5, 0.0),
                        (1.0, 0.2, 1.0),
                        (1.0, -3.0, -1.0),
                    ),
                    dtype=np.float64,
                ),
                np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int32),
            )
        case _:
            raise ValueError(
                f"cell_meshes.incompatible_simplex.dimension must be two or three; got {dimension}"
            )


def proposal_source() -> CellMeshingResult:
    """Return a certified four-cell source with explicit non-storage-order IDs."""
    mesh = CellMesh.from_triangles(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.7, 0.3)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((7, 3, 9, 1, 5), dtype=np.int64),
        cell_global_ids=np.asarray((10, 20, 30, 40), dtype=np.int64),
    )
    return certify_cell_mesh(mesh, SpatialCoordinateContract.si())


def reference_cell(kind: str) -> CellMesh:
    """Return one cell using the vertex ordering declared by its reference topology."""
    topology = reference_cell_topology(kind)
    coordinates = np.asarray(topology.vertices, dtype=np.float64)
    return CellMesh(
        coordinates,
        (CellBlock("cells", kind, np.arange(len(coordinates), dtype=np.int32)[None]),),
    )


def bisect(
    source: CellMeshingResult,
    refine: ConvertibleToArray = (),
    coarsen: ConvertibleToArray = (),
    hierarchy: BisectionHierarchy | None = None,
) -> MeshAdaptationResult:
    """Execute native bisection using stable cell IDs and the retained forest."""
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.asarray(refine, dtype=np.int64),
                np.asarray(coarsen, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
        )
    )


def _digest(value: object) -> NDArray[np.uint8]:
    return np.frombuffer(bytes.fromhex(canonical_fingerprint(value)), dtype=np.uint8)


def storage(mesh: CellMesh, geometry: CellGeometrySpec) -> CellMeshStorage:
    """Return a rank-zero closure of a two-partition logical coordinate bank.

    Every entity is owned by rank zero. Basis, parent IDs, affine restrictions,
    and coefficient IDs are retained so storage restores the represented map.
    This is host interchange preparation from test_cell_geometry_storage,
    not a geometry evaluator or a collective coverage certificate.
    """
    coordinate_ids = np.arange(geometry.coordinates.shape[0], dtype=np.int64) + 100
    owners = np.zeros(coordinate_ids.shape, dtype=np.int32)
    arrays: dict[str, Array] = {
        "geometry/coordinate_ids": jnp.asarray(coordinate_ids),
        "geometry/coordinate_owners": jnp.asarray(owners),
        "geometry/coordinates": geometry.coordinates,
    }
    origin = geometry.restriction_source
    if origin is not None:
        arrays["geometry/source_geometry_id"] = jnp.asarray(
            _digest(origin.source_geometry_id)
        )
        arrays["geometry/source_topology_id"] = jnp.asarray(
            _digest(origin.source_topology_id)
        )
    for block, element, route in zip(
        mesh.blocks, geometry.elements, geometry.geometry_dofs, strict=True
    ):
        name = block.name
        with coordinate_reference_budget() as budget:
            root, expressions = coordinate_reference_chain(element, budget=budget)
        dimension = block.topological_dimension
        if any(isinstance(value, RationalPolynomial) for value in expressions):
            raise ValueError(
                "cell_meshes.storage must have polynomial reference actions; got rational action"
            )
        arguments = tuple(
            value for value in expressions if not isinstance(value, RationalPolynomial)
        )
        if any(sum(index) > 1 for value in arguments for index in value):
            raise ValueError(
                "cell_meshes.storage must have affine reference actions; got nonlinear action"
            )
        matrix = np.asarray(
            [
                [
                    float(
                        value.get(
                            tuple(int(axis == column) for axis in range(dimension)),
                            Fraction(),
                        )
                    )
                    for column in range(dimension)
                ]
                for value in arguments
            ],
            dtype=np.float64,
        )
        offset = np.asarray(
            [float(value.get((0,) * dimension, Fraction())) for value in arguments],
            dtype=np.float64,
        )
        arrays[f"geometry/cell_ids/{name}"] = block.global_ids
        arrays[f"geometry/routes/{name}"] = jnp.asarray(coordinate_ids[np.asarray(route)])
        arrays[f"geometry/matrix/{name}"] = jnp.asarray(
            np.broadcast_to(matrix, (block.cell_count,) + matrix.shape)
        )
        arrays[f"geometry/offset/{name}"] = jnp.asarray(
            np.broadcast_to(offset, (block.cell_count,) + offset.shape)
        )
        ancestry = nonrepresentable_reference_ancestry(element, matrix, offset)
        if ancestry is not None:
            digest = np.frombuffer(bytes.fromhex(ancestry), dtype=np.uint8)
            arrays[f"geometry/reference_ancestry/{name}"] = jnp.asarray(
                np.broadcast_to(digest, (block.cell_count, 32))
            )
        signature = {
            "signature": coordinate_source_signature(root),
            "arrays": array_tree_fingerprint(root),
        }
        arrays[f"geometry/source_basis/{name}"] = jnp.asarray(
            np.broadcast_to(_digest(signature), (block.cell_count, 32))
        )
        if origin is not None:
            arrays[f"geometry/parent_cell_ids/{name}"] = origin.block_parent_cell_ids[
                name
            ]
            arrays[f"geometry/parent_vertex_ids/{name}"] = origin.block_parent_vertex_ids[
                name
            ]
    entity_ids = tuple(entity.entity_ids for entity in mesh.topology.entity_sets)
    return CellMeshStorage(
        tuple(entity.count for entity in mesh.topology.entity_sets),
        entity_ids,
        tuple(np.zeros(value.shape, dtype=np.int32) for value in entity_ids),
        partition_index=0,
        partition_count=2,
        logical_topology_id=mesh.topology_id,
        logical_geometry_id=mesh.geometry_id,
        logical_coordinate_geometry_id=cell_geometry_id(geometry),
        evidence_id=canonical_fingerprint("collective"),
        logical_arrays=tuple(sorted(arrays.items())),
        local_coordinates=mesh.coordinates,
        local_blocks=mesh.blocks,
        global_coordinate_count=len(coordinate_ids),
        coordinate_global_ids=coordinate_ids,
        coordinate_owner=owners,
        local_geometry=geometry,
        local_physical_boundary_entities=tuple(
            entity.subset("boundary").mask for entity in mesh.topology.entity_sets
        ),
    )

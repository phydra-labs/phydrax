from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolygonalConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..geometry._source_identity import source_entity_id
from ..linalg._exact import determinant_exact, solve_exact
from ..typing import checked
from ._global_embedding import GlobalEmbeddingCertificate
from ._volume_material_map import require_polyhedral_carrier_geometry
from .implicit._analytic_profile import AnalyticImplicitProfile


if TYPE_CHECKING:
    from ..discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
        PreparedSphereChartPiece,
    )


_Point = tuple[Fraction, Fraction]
_Triangle = tuple[_Point, _Point, _Point]
_Ray = tuple[Fraction, Fraction, Fraction]
_RayTriangle = tuple[_Ray, _Ray, _Ray]
_Interval = tuple[Fraction, Fraction]
_ConeBounds = tuple[_Interval, _Interval, _Interval]


@dataclass(frozen=True, slots=True)
class SphereProjectiveTriangleBounds:
    denominator_lower: Fraction
    denominator_upper: Fraction
    jacobian_lower: Fraction
    jacobian_upper: Fraction


class SphereProjectiveReferenceMap(StrictModule, NonTrainableState):
    """Exact projective action between two radial sphere reference cells."""

    source_atlas_id: str = eqx.field(static=True)
    target_atlas_id: str = eqx.field(static=True)
    source_cell_global_id: int = eqx.field(static=True)
    target_cell_global_id: int = eqx.field(static=True)
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    exact_coefficients: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    exact_denominator: tuple[Fraction, ...] = eqx.field(static=True)
    orientation_ratio: Fraction = eqx.field(static=True)
    coefficients: Array
    coefficient_lower: Array
    coefficient_upper: Array
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_atlas: SphereMaterialCellAtlas,
        source_cell_global_id: int,
        target_atlas: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        /,
    ) -> None:
        if source_atlas.domain_id != target_atlas.domain_id:
            raise ValueError(
                "Sphere projective maps require the same source material domain."
            )
        source_row = source_atlas.cell_row(source_cell_global_id)
        target_row = target_atlas.cell_row(target_cell_global_id)
        source_rays = source_atlas.exact_directions[source_row]
        target_rays = target_atlas.exact_directions[target_row]
        target_matrix = tuple(
            tuple(target_rays[column][axis] for column in range(3)) for axis in range(3)
        )
        source_matrix = tuple(
            tuple(source_rays[column][axis] for column in range(3)) for axis in range(3)
        )
        solved = solve_exact(target_matrix, source_matrix)
        coefficients = tuple(tuple(value for value in row) for row in solved)
        denominator = tuple(
            sum((coefficients[row][column] for row in range(3)), Fraction(0))
            for column in range(3)
        )
        determinant = determinant_exact(coefficients)
        if not determinant:
            raise ValueError("Sphere projective reference map must be nonsingular.")
        execution = np.asarray(
            [[float(value) for value in row] for row in coefficients], dtype=np.float64
        )
        lower = np.empty_like(execution)
        upper = np.empty_like(execution)
        for row in range(3):
            for column in range(3):
                lower[row, column], upper[row, column] = _outward(
                    coordinates=coefficients[row][column]
                )
        entity = source_atlas.geometry_entity_id
        occurrence = source_atlas.occurrence_path
        self.source_atlas_id = source_atlas.atlas_id
        self.target_atlas_id = target_atlas.atlas_id
        self.source_cell_global_id = int(source_cell_global_id)
        self.target_cell_global_id = int(target_cell_global_id)
        self.geometry_entity_id = entity
        self.occurrence_path = occurrence
        self.exact_coefficients = coefficients
        self.exact_denominator = denominator
        self.orientation_ratio = determinant
        self.coefficients = jnp.asarray(execution)
        self.coefficient_lower = jnp.asarray(lower)
        self.coefficient_upper = jnp.asarray(upper)
        self.map_id = canonical_fingerprint(
            {
                "kind": "sphere-projective-reference-map",
                "source_atlas": self.source_atlas_id,
                "target_atlas": self.target_atlas_id,
                "source_cell": self.source_cell_global_id,
                "target_cell": self.target_cell_global_id,
                "entity": entity,
                "occurrence": occurrence,
                "coefficients": tuple(
                    tuple((value.numerator, value.denominator) for value in row)
                    for row in coefficients
                ),
            }
        )

    def evaluate(self, point: _Point, /) -> _Point:
        x, y = point
        barycentric = (Fraction(1) - x - y, x, y)
        homogeneous = tuple(
            sum(
                (
                    self.exact_coefficients[row][column] * barycentric[column]
                    for column in range(3)
                ),
                Fraction(0),
            )
            for row in range(3)
        )
        denominator = sum(homogeneous, Fraction(0))
        if not denominator:
            raise ValueError("Sphere projective map has a zero homogeneous denominator.")
        return homogeneous[1] / denominator, homogeneous[2] / denominator

    def triangle_bounds(self, triangle: _Triangle, /) -> SphereProjectiveTriangleBounds:
        denominator_values = tuple(
            sum(
                (
                    weight * self.exact_denominator[corner]
                    for corner, weight in enumerate((1 - x - y, x, y))
                ),
                Fraction(0),
            )
            for x, y in triangle
        )
        lower, upper = min(denominator_values), max(denominator_values)
        if lower <= 0 <= upper:
            raise ValueError(
                "Sphere projective denominator is unresolved on a retained material triangle."
            )
        values = tuple(self.orientation_ratio / (value**3) for value in (lower, upper))
        return SphereProjectiveTriangleBounds(lower, upper, min(values), max(values))


def _dot(first: _Ray, second: _Ray, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _sqrt_fraction_bounds(value: Fraction, /) -> tuple[Fraction, Fraction]:
    if value < 0:
        raise ValueError("Exact norm square must be nonnegative.")
    if not value:
        return Fraction(0), Fraction(0)
    scale_ = 1 << 128
    floor = math.isqrt((value.numerator << 256) // value.denominator)
    lower = Fraction(floor, scale_)
    return lower, lower if lower * lower == value else Fraction(floor + 1, scale_)


def _triangle_radial_deviation_bound(
    triangle: _RayTriangle,
    radius: Fraction,
    /,
) -> Fraction:
    origin, second, third = triangle
    edge_a: _Ray = (
        second[0] - origin[0],
        second[1] - origin[1],
        second[2] - origin[2],
    )
    edge_b: _Ray = (
        third[0] - origin[0],
        third[1] - origin[1],
        third[2] - origin[2],
    )
    candidates: list[Fraction] = [_dot(point, point) for point in triangle]
    for first, next_ in ((origin, second), (second, third), (third, origin)):
        direction: _Ray = (
            next_[0] - first[0],
            next_[1] - first[1],
            next_[2] - first[2],
        )
        length = _dot(direction, direction)
        parameter = max(Fraction(0), min(Fraction(1), -_dot(first, direction) / length))
        point: _Ray = (
            first[0] + parameter * direction[0],
            first[1] + parameter * direction[1],
            first[2] + parameter * direction[2],
        )
        candidates.append(_dot(point, point))
    gram = (
        (_dot(edge_a, edge_a), _dot(edge_a, edge_b)),
        (_dot(edge_b, edge_a), _dot(edge_b, edge_b)),
    )
    rhs = ((-_dot(origin, edge_a),), (-_dot(origin, edge_b),))
    try:
        projection = solve_exact(gram, rhs)
    except ValueError:
        raise ValueError(
            "Sphere material triangle has singular affine geometry."
        ) from None
    u, v = projection[0][0], projection[1][0]
    if u >= 0 and v >= 0 and u + v <= 1:
        nearest: _Ray = (
            origin[0] + u * edge_a[0] + v * edge_b[0],
            origin[1] + u * edge_a[1] + v * edge_b[1],
            origin[2] + u * edge_a[2] + v * edge_b[2],
        )
        candidates.append(_dot(nearest, nearest))
    minimum = min(candidates)
    maximum = max(_dot(point, point) for point in triangle)
    minimum_bounds = _sqrt_fraction_bounds(minimum)
    maximum_bounds = _sqrt_fraction_bounds(maximum)
    return max(abs(value - radius) for value in (*minimum_bounds, *maximum_bounds))


def _cone_bounds(triangle: _RayTriangle, /) -> _ConeBounds:
    norms = tuple(_sqrt_fraction_bounds(_dot(ray, ray)) for ray in triangle)
    unit_components: list[tuple[_Interval, _Interval, _Interval]] = []
    for ray, (norm_lower, norm_upper) in zip(triangle, norms, strict=True):
        if not norm_lower:
            raise ValueError("Sphere radial direction must have positive norm.")
        components: list[_Interval] = []
        for value in ray:
            if value > 0:
                components.append((value / norm_upper, value / norm_lower))
            elif value < 0:
                components.append((value / norm_lower, value / norm_upper))
            else:
                components.append((Fraction(0), Fraction(0)))
        unit_components.append((components[0], components[1], components[2]))

    center: _Ray = (
        sum((ray[0] for ray in triangle), Fraction(0)),
        sum((ray[1] for ray in triangle), Fraction(0)),
        sum((ray[2] for ray in triangle), Fraction(0)),
    )
    center_norm_upper = _sqrt_fraction_bounds(_dot(center, center))[1]
    lower_dots: list[Fraction] = []
    for ray, (_, norm_upper) in zip(triangle, norms, strict=True):
        dot = _dot(center, ray)
        if dot <= 0 or not norm_upper:
            return (
                (Fraction(-1), Fraction(1)),
                (Fraction(-1), Fraction(1)),
                (Fraction(-1), Fraction(1)),
            )
        lower_dots.append(dot / norm_upper)
    lower_norm = min(lower_dots) / center_norm_upper
    if lower_norm <= 0:
        return (
            (Fraction(-1), Fraction(1)),
            (Fraction(-1), Fraction(1)),
            (Fraction(-1), Fraction(1)),
        )
    result: list[_Interval] = []
    for axis in range(3):
        lower = min(component[axis][0] for component in unit_components)
        upper = max(component[axis][1] for component in unit_components)
        if lower >= 0:
            bounds = (lower, min(Fraction(1), upper / lower_norm))
        elif upper <= 0:
            bounds = (max(Fraction(-1), lower / lower_norm), upper)
        else:
            bounds = (
                max(Fraction(-1), lower / lower_norm),
                min(Fraction(1), upper / lower_norm),
            )
        result.append(bounds)
    return (result[0], result[1], result[2])


@dataclass(frozen=True, slots=True)
class _ConeNode:
    bounds: _ConeBounds
    rows: tuple[int, ...]
    left: _ConeNode | None = None
    right: _ConeNode | None = None


def _boxes_intersect(first: _ConeBounds, second: _ConeBounds, /) -> bool:
    return all(
        a_lower <= b_upper and b_lower <= a_upper
        for (a_lower, a_upper), (b_lower, b_upper) in zip(first, second, strict=True)
    )


def _merge_cone_boxes(
    bounds: tuple[_ConeBounds, ...], rows: tuple[int, ...], /
) -> _ConeBounds:
    return (
        (
            min(bounds[row][0][0] for row in rows),
            max(bounds[row][0][1] for row in rows),
        ),
        (
            min(bounds[row][1][0] for row in rows),
            max(bounds[row][1][1] for row in rows),
        ),
        (
            min(bounds[row][2][0] for row in rows),
            max(bounds[row][2][1] for row in rows),
        ),
    )


def _build_cone_tree(
    bounds: tuple[_ConeBounds, ...],
    rows: tuple[int, ...],
    /,
) -> _ConeNode:
    node_bounds = _merge_cone_boxes(bounds, rows)
    if len(rows) <= 8:
        return _ConeNode(node_bounds, rows)
    axis = max(
        range(3),
        key=lambda index: node_bounds[index][1] - node_bounds[index][0],
    )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                bounds[row][axis][0] + bounds[row][axis][1],
                row,
            ),
        )
    )
    middle = len(ordered) // 2
    return _ConeNode(
        node_bounds,
        (),
        _build_cone_tree(bounds, ordered[:middle]),
        _build_cone_tree(bounds, ordered[middle:]),
    )


def _query_cone_tree(
    node: _ConeNode,
    query: _ConeBounds,
    bounds: tuple[_ConeBounds, ...],
    /,
) -> tuple[int, ...]:
    if not _boxes_intersect(node.bounds, query):
        return ()
    if node.rows:
        return tuple(row for row in node.rows if _boxes_intersect(bounds[row], query))
    left, right = node.left, node.right
    if left is None or right is None:
        raise ValueError("An internal sphere cone-tree node must have both children.")
    return tuple(
        sorted(
            (
                *_query_cone_tree(left, query, bounds),
                *_query_cone_tree(right, query, bounds),
            )
        )
    )


class _SphereMaterialFidelityBinding(StrictModule, NonTrainableState):
    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AnalyticImplicitProfile,
        /,
    ) -> None:
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_id = cell_geometry_id(geometry)
        self.source_id = source.source_id
        self.source_revision = source.source_revision
        self.profile_id = source.profile_id
        self.binding_id = canonical_fingerprint(
            {
                "kind": "sphere-material-fidelity-binding",
                "mesh": self.mesh_id,
                "topology": self.topology_id,
                "geometry": self.geometry_id,
                "source": self.source_id,
                "revision": self.source_revision,
                "profile": self.profile_id,
            }
        )

    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AnalyticImplicitProfile,
        /,
    ) -> None:
        if (
            mesh.mesh_id != self.mesh_id
            or mesh.topology_id != self.topology_id
            or cell_geometry_id(geometry) != self.geometry_id
            or (source.source_id, source.source_revision, source.profile_id)
            != (self.source_id, self.source_revision, self.profile_id)
        ):
            raise ValueError(
                "Sphere material fidelity proof is stale for its source/map endpoints."
            )


class SphereMaterialFidelityCertificate(StrictModule, NonTrainableState):
    """Exact radial distance bounds between a P1 atlas and an analytic sphere."""

    binding: _SphereMaterialFidelityBinding
    source: AnalyticImplicitProfile
    deviation_bounds: Array
    mesh_to_source_lower: float = eqx.field(static=True)
    mesh_to_source_upper: float = eqx.field(static=True)
    source_to_mesh_lower: float = eqx.field(static=True)
    source_to_mesh_upper: float = eqx.field(static=True)
    status: Literal["certified"] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AnalyticImplicitProfile,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        exact_directions: tuple[_RayTriangle, ...],
        source_validity: CellValidityCertificate,
        source_embedding: GlobalEmbeddingCertificate,
        /,
    ) -> None:
        radius = Fraction(float(source.radius))
        bounds = tuple(
            _triangle_radial_deviation_bound(triangle, radius)
            for triangle in exact_directions
        )
        rounded = np.asarray(
            [_upper_fraction(value) for value in bounds], dtype=np.float64
        )
        if not np.all(np.isfinite(rounded)):
            raise ValueError("Sphere radial fidelity bounds must fit finite binary64.")
        binding = _SphereMaterialFidelityBinding(mesh, geometry, source)
        upper = float(np.max(rounded, initial=0.0))
        self.binding = binding
        self.source = source
        self.deviation_bounds = jnp.asarray(np.stack((rounded, rounded), axis=1))
        self.mesh_to_source_lower = 0.0
        self.mesh_to_source_upper = upper
        self.source_to_mesh_lower = 0.0
        self.source_to_mesh_upper = upper
        self.status = "certified"
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "sphere-radial-fidelity-certificate",
                "binding": binding.binding_id,
                "deviation_bounds": array_tree_fingerprint(rounded),
                "validity": source_validity.certificate_id,
                "embedding": source_embedding.certificate_id,
                "radius": source.radius,
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AnalyticImplicitProfile,
        /,
    ) -> None:
        self.binding.require(mesh, geometry, source)
        source.validate_source_integrity()


class SphereMaterialCellAtlas(StrictModule, NonTrainableState):
    """Exact radial projective atlas for an embedded affine triangulation of a sphere."""

    source_profile: AnalyticImplicitProfile
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_fidelity_certificate: SphereMaterialFidelityCertificate
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    cell_global_ids: Array
    physical_rows: Array
    directions: Array
    exact_directions: tuple[_RayTriangle, ...]
    cone_bounds: tuple[_ConeBounds, ...]
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)
    atlas_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_profile: AnalyticImplicitProfile,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source_validity: CellValidityCertificate,
        source_embedding: GlobalEmbeddingCertificate,
        /,
    ) -> None:
        if (
            not isinstance(source_profile, AnalyticImplicitProfile)
            or source_profile.family != "sphere"
        ):
            raise ValueError(
                "Sphere material atlas requires an exact nominal analytic sphere source."
            )
        if (
            not isinstance(mesh, CellMesh)
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
        ):
            raise ValueError(
                "Sphere material atlas requires a two-dimensional surface mesh in 3D."
            )
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise ValueError(
                "Sphere material atlas requires triangular surface incidence."
            )
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError(
                "Sphere material atlas requires a fixed triangular surface mesh."
            )
        connectivity = mesh.connectivity
        if (
            int(connectivity.triangle_count)
            != sum(block.cell_count for block in mesh.blocks)
            or np.any(np.asarray(connectivity.boundary_edges, dtype=np.bool_))
            or np.any(np.asarray(connectivity.edge_cell_counts, dtype=np.int32) != 2)
        ):
            raise ValueError(
                "Sphere material mesh must be a closed two-manifold triangulation."
            )
        cell_count = sum(block.cell_count for block in mesh.blocks)
        cell_vertices = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        used_vertices = np.unique(cell_vertices)
        if used_vertices.size - np.asarray(connectivity.edges).shape[0] + cell_count != 2:
            raise ValueError(
                "Sphere material surface must have Euler characteristic two."
            )
        edge_cells: dict[int, list[tuple[int, int]]] = {}
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int64)
        edge_signs = np.asarray(connectivity.cell_edge_signs, dtype=np.int8)
        edge_valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        for cell_row in range(cell_count):
            for edge, sign in zip(
                cell_edges[cell_row, edge_valid[cell_row]],
                edge_signs[cell_row, edge_valid[cell_row]],
                strict=True,
            ):
                if int(sign) not in (-1, 1):
                    raise ValueError(
                        "Sphere surface edge incidence needs an exact orientation sign."
                    )
                edge_cells.setdefault(int(edge), []).append((cell_row, int(sign)))
        adjacency = [set() for _ in range(cell_count)]
        for incident in edge_cells.values():
            if len(incident) != 2 or incident[0][1] + incident[1][1] != 0:
                raise ValueError(
                    "Sphere surface edges must have oppositely oriented two-cell incidence."
                )
            first, second = incident[0][0], incident[1][0]
            adjacency[first].add(second)
            adjacency[second].add(first)
        reached = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current].difference(reached):
                reached.add(neighbor)
                frontier.append(neighbor)
        if len(reached) != cell_count:
            raise ValueError("Sphere material surface must be connected.")
        require_polyhedral_carrier_geometry(mesh, geometry)
        source_validity.require_bound(geometry, mesh=mesh)
        if not source_validity.all_certified:
            raise ValueError(
                "Sphere material atlas requires certified whole-cell validity."
            )
        source_embedding.binding.require(mesh, geometry)
        if source_embedding.status != "certified":
            raise ValueError("Sphere material atlas requires certified global embedding.")
        triangle_rows: list[tuple[int, int, int]] = []
        for block in mesh.blocks:
            block_vertices = np.asarray(block.vertices, dtype=np.int64)
            for block_row in range(block.cell_count):
                triangle_rows.append(
                    (
                        int(block_vertices[block_row, 0]),
                        int(block_vertices[block_row, 1]),
                        int(block_vertices[block_row, 2]),
                    )
                )
        triangles = tuple(triangle_rows)
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        exact_center: _Ray = (
            Fraction(float(source_profile.center[0])),
            Fraction(float(source_profile.center[1])),
            Fraction(float(source_profile.center[2])),
        )
        exact_triangles: list[_RayTriangle] = []
        cell_ids: list[int] = []
        for block in mesh.blocks:
            for identifier in np.asarray(block.global_ids, dtype=np.int64):
                cell_ids.append(int(identifier))
        exact_vertex_rays: tuple[_Ray, ...] = tuple(
            (
                Fraction(float(coordinates[index, 0])) - exact_center[0],
                Fraction(float(coordinates[index, 1])) - exact_center[1],
                Fraction(float(coordinates[index, 2])) - exact_center[2],
            )
            for index in range(coordinates.shape[0])
        )
        signs: list[int] = []
        for triangle in triangles:
            rays: _RayTriangle = (
                exact_vertex_rays[triangle[0]],
                exact_vertex_rays[triangle[1]],
                exact_vertex_rays[triangle[2]],
            )
            radial_matrix = (
                (rays[0][0], rays[1][0], rays[2][0]),
                (rays[0][1], rays[1][1], rays[2][1]),
                (rays[0][2], rays[1][2], rays[2][2]),
            )
            determinant = determinant_exact(radial_matrix)
            if not determinant:
                raise ValueError(
                    "Sphere surface triangle maps through the radial center."
                )
            signs.append(1 if determinant > 0 else -1)
            exact_triangles.append(rays)
        if len(set(signs)) != 1:
            raise ValueError("Sphere surface cells do not share one radial orientation.")
        _require_center_winding_one(tuple(exact_triangles))
        cone_bounds = tuple(_cone_bounds(triangle) for triangle in exact_triangles)
        source_fidelity_certificate = SphereMaterialFidelityCertificate(
            source_profile,
            mesh,
            geometry,
            tuple(exact_triangles),
            source_validity,
            source_embedding,
        )

        entity = source_entity_id(
            source_profile.source_id,
            source_profile.source_revision,
            "analytic-sphere",
            source_profile.profile_id,
        )
        domain = canonical_fingerprint(
            {
                "kind": "analytic-sphere-material-domain",
                "source": source_profile.source_id,
                "revision": source_profile.source_revision,
                "profile": source_profile.profile_id,
            }
        )
        cell_ids_array = np.asarray(cell_ids, dtype=np.int64)
        physical_rows = np.arange(cell_ids_array.size, dtype=np.int64)
        direction_array = np.asarray(
            [
                [[float(value) for value in ray] for ray in triangle]
                for triangle in exact_triangles
            ],
            dtype=np.float64,
        )
        for array in (cell_ids_array, physical_rows, direction_array):
            array.setflags(write=False)
        self.source_profile = source_profile
        self.source_mesh = mesh
        self.source_geometry = geometry
        self.source_fidelity_certificate = source_fidelity_certificate
        self.source_validity = source_validity
        self.source_embedding = source_embedding
        self.cell_global_ids = jnp.asarray(cell_ids_array)
        self.physical_rows = jnp.asarray(physical_rows)
        self.directions = jnp.asarray(direction_array)
        self.exact_directions = tuple(exact_triangles)
        self.cone_bounds = cone_bounds
        self.geometry_entity_id = entity
        self.occurrence_path = ()
        self.source_geometry_id = cell_geometry_id(geometry)
        self.source_topology_id = mesh.topology_id
        self.domain_id = domain
        self.coverage_status = "certified"
        self.atlas_id = canonical_fingerprint(
            {
                "kind": "sphere-material-cell-atlas",
                "source": source_profile.source_id,
                "revision": source_profile.source_revision,
                "profile": source_profile.profile_id,
                "domain": domain,
                "mesh": mesh.mesh_id,
                "topology": mesh.topology_id,
                "geometry": self.source_geometry_id,
                "cells": array_tree_fingerprint(cell_ids_array),
                "rays": tuple(
                    tuple(
                        tuple((value.numerator, value.denominator) for value in ray)
                        for ray in triangle
                    )
                    for triangle in exact_triangles
                ),
                "source_fidelity": source_fidelity_certificate.certificate_id,
                "validity": source_validity.certificate_id,
                "embedding": source_embedding.certificate_id,
            }
        )

    def cell_row(self, cell_global_id: int, /) -> int:
        ids = np.asarray(self.cell_global_ids, dtype=np.int64)
        rows = np.flatnonzero(ids == int(cell_global_id))
        if rows.size != 1:
            raise KeyError(
                f"Sphere material atlas has no unique cell ID {cell_global_id}."
            )
        return int(rows[0])

    def require_bound(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        if (
            mesh.mesh_id != self.source_mesh.mesh_id
            or mesh.topology_id != self.source_topology_id
            or cell_geometry_id(geometry) != self.source_geometry_id
        ):
            raise ValueError(
                "Sphere material atlas is stale for this mesh/coordinate epoch."
            )
        self.source_profile.validate_source_integrity()
        self.source_fidelity_certificate.require_current(
            mesh, geometry, self.source_profile
        )
        self.source_validity.require_bound(geometry, mesh=mesh)
        self.source_embedding.binding.require(mesh, geometry)

    @classmethod
    def from_stored_fields(
        cls, value: SphereMaterialCellAtlas, /
    ) -> SphereMaterialCellAtlas:
        if not isinstance(value, SphereMaterialCellAtlas):
            raise TypeError("value must be SphereMaterialCellAtlas.")
        result = cls(
            value.source_profile,
            value.source_mesh,
            value.source_geometry,
            value.source_validity,
            value.source_embedding,
        )
        if result.atlas_id != value.atlas_id:
            raise ValueError(
                "Stored sphere material atlas does not reproduce its exact source map."
            )
        return result


@checked
def sphere_projective_reference_map(
    source: SphereMaterialCellAtlas,
    source_cell_global_id: int,
    target: SphereMaterialCellAtlas,
    target_cell_global_id: int,
    /,
) -> SphereProjectiveReferenceMap:
    return SphereProjectiveReferenceMap(
        source, source_cell_global_id, target, target_cell_global_id
    )


def _outward(*, coordinates: Fraction) -> tuple[float, float]:
    center = float(coordinates)
    if not np.isfinite(center):
        raise ValueError("Exact projective coefficient must fit finite binary64.")
    rounded = Fraction(center)
    return (
        float(np.nextafter(center, -np.inf)) if rounded > coordinates else center,
        float(np.nextafter(center, np.inf)) if rounded < coordinates else center,
    )


def _clip_polygon(
    polygon: tuple[_Point, ...],
    plane: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[_Point, ...]:
    a, b, c = plane

    def value(point: _Point, /) -> Fraction:
        return a * point[0] + b * point[1] + c

    if not polygon:
        return ()
    output: list[_Point] = []
    previous = polygon[-1]
    previous_value = value(previous)
    for current in polygon:
        current_value = value(current)
        if (current_value >= 0) != (previous_value >= 0):
            ratio = previous_value / (previous_value - current_value)
            output.append(
                (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                )
            )
        if current_value >= 0:
            output.append(current)
        previous, previous_value = current, current_value
    result = []
    for point in output:
        if not result or point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return tuple(result)


def _signed_area2(points: tuple[_Point, ...], /) -> Fraction:
    return (
        sum(
            (
                first[0] * second[1] - second[0] * first[1]
                for first, second in zip(points, (*points[1:], points[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def sphere_chart_overlap_pieces(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    /,
) -> tuple[PreparedSphereChartPiece, ...]:
    """Build the exact common refinement in one analytic sphere's radial material chart."""
    if source.domain_id != target.domain_id:
        raise ValueError("Sphere overlap requires one retained analytic source domain.")
    source.require_bound(source.source_mesh, source.source_geometry)
    target.require_bound(target.source_mesh, target.source_geometry)
    from ..discretization._sphere_chart_records import PreparedSphereChartPiece

    source_ids = np.asarray(source.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.cell_global_ids, dtype=np.int64)
    displacement = float(
        source.source_fidelity_certificate.mesh_to_source_upper
        + target.source_fidelity_certificate.mesh_to_source_upper
    )
    source_simplex: _Triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    source_halfplanes = (
        (Fraction(-1), Fraction(-1), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    target_cone_bounds = target.cone_bounds
    target_tree = _build_cone_tree(target_cone_bounds, tuple(range(len(target_ids))))
    pieces: list[PreparedSphereChartPiece] = []
    for source_row, source_id in enumerate(source_ids):
        for target_row in _query_cone_tree(
            target_tree, source.cone_bounds[source_row], target_cone_bounds
        ):
            target_id = target_ids[target_row]
            projective = SphereProjectiveReferenceMap(
                source, int(source_id), target, int(target_id)
            )
            coefficients = projective.exact_coefficients
            target_halfplanes = tuple(
                (
                    coefficients[index][1] - coefficients[index][0],
                    coefficients[index][2] - coefficients[index][0],
                    coefficients[index][0],
                )
                for index in range(3)
            )
            polygon = source_simplex
            for plane in target_halfplanes:
                polygon = _clip_polygon(polygon, plane)
                if len(polygon) < 3:
                    break
            if len(polygon) < 3 or _signed_area2(polygon) == 0:
                continue
            planes = (*source_halfplanes, *target_halfplanes)
            labels: list[int] = []
            for first, second in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
                supporting = tuple(
                    index
                    for index, (a, b, c) in enumerate(planes)
                    if a * first[0] + b * first[1] + c == 0
                    and a * second[0] + b * second[1] + c == 0
                )
                if not supporting:
                    raise ValueError(
                        "Sphere overlap lost its exact supporting cone face."
                    )
                labels.append(supporting[0])
            clip_id = canonical_fingerprint(
                {
                    "kind": "sphere-exact-cone-overlap",
                    "source": int(source_id),
                    "target": int(target_id),
                    "vertices": tuple(
                        tuple((value.numerator, value.denominator) for value in point)
                        for point in polygon
                    ),
                    "labels": tuple(labels),
                }
            )
            for fan_index in range(1, len(polygon) - 1):
                fan = (0, fan_index, fan_index + 1)
                source_triangle: _Triangle = (
                    polygon[0],
                    polygon[fan_index],
                    polygon[fan_index + 1],
                )
                target_triangle: _Triangle = (
                    projective.evaluate(source_triangle[0]),
                    projective.evaluate(source_triangle[1]),
                    projective.evaluate(source_triangle[2]),
                )
                source_area = abs(_signed_area2(source_triangle))
                target_area = abs(_signed_area2(target_triangle))
                if not source_area or not target_area:
                    continue
                piece_id = canonical_fingerprint(
                    {
                        "kind": "sphere-projective-overlap-piece",
                        "source_atlas": source.atlas_id,
                        "target_atlas": target.atlas_id,
                        "source_cell": int(source_id),
                        "target_cell": int(target_id),
                        "projective_map": projective.map_id,
                        "clip": clip_id,
                        "fan": fan,
                        "source_triangle": tuple(
                            tuple((value.numerator, value.denominator) for value in point)
                            for point in source_triangle
                        ),
                        "target_triangle": tuple(
                            tuple((value.numerator, value.denominator) for value in point)
                            for point in target_triangle
                        ),
                    }
                )
                pieces.append(
                    PreparedSphereChartPiece(
                        int(source_id),
                        int(target_id),
                        source_row,
                        target_row,
                        source_row,
                        target_row,
                        source.geometry_entity_id,
                        source.occurrence_path,
                        projective,
                        source_triangle,
                        target_triangle,
                        jnp.asarray(
                            [
                                [float(value) for value in point]
                                for point in source_triangle
                            ],
                            dtype=jnp.float64,
                        ),
                        jnp.asarray(
                            [
                                [float(value) for value in point]
                                for point in target_triangle
                            ],
                            dtype=jnp.float64,
                        ),
                        projective.triangle_bounds(source_triangle),
                        source_area,
                        target_area,
                        tuple(labels),
                        fan,
                        clip_id,
                        displacement,
                        piece_id,
                    )
                )
    if not pieces:
        raise ValueError(
            "Sphere material atlases have no positive-measure common refinement."
        )
    return tuple(pieces)


def _upper_fraction(value: Fraction, /) -> float:
    rounded = float(value)
    if not np.isfinite(rounded):
        raise ValueError("Sphere measure error must fit finite binary64.")
    return float(np.nextafter(rounded, np.inf)) if Fraction(rounded) < value else rounded


def _triangle_measure_data(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    measures: list[float] = []
    errors: list[float] = []
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError("Sphere material measures require triangular cell rows.")
        for row in range(cells.shape[0]):
            cell = cells[row]
            points: _RayTriangle = (
                (
                    Fraction(float(coordinates[int(cell[0]), 0])),
                    Fraction(float(coordinates[int(cell[0]), 1])),
                    Fraction(float(coordinates[int(cell[0]), 2])),
                ),
                (
                    Fraction(float(coordinates[int(cell[1]), 0])),
                    Fraction(float(coordinates[int(cell[1]), 1])),
                    Fraction(float(coordinates[int(cell[1]), 2])),
                ),
                (
                    Fraction(float(coordinates[int(cell[2]), 0])),
                    Fraction(float(coordinates[int(cell[2]), 1])),
                    Fraction(float(coordinates[int(cell[2]), 2])),
                ),
            )
            first: _Ray = (
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
                points[1][2] - points[0][2],
            )
            second: _Ray = (
                points[2][0] - points[0][0],
                points[2][1] - points[0][1],
                points[2][2] - points[0][2],
            )
            cross: _Ray = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            square = sum((value * value for value in cross), Fraction(0))
            if not square:
                raise ValueError(
                    "Sphere material surface cell has zero exact physical area."
                )
            scale_ = 1 << 128
            floor = math.isqrt((square.numerator << 256) // square.denominator)
            lower_length = Fraction(floor, scale_)
            upper_length = (
                lower_length
                if lower_length * lower_length == square
                else Fraction(floor + 1, scale_)
            )
            lower, upper = lower_length / 2, upper_length / 2
            exact_midpoint = (lower + upper) / 2
            rounded = float(exact_midpoint)
            if not np.isfinite(rounded) or rounded <= 0.0:
                raise ValueError(
                    "Sphere material surface measure must fit positive binary64."
                )
            error = max(
                abs(Fraction(rounded) - lower),
                abs(upper - Fraction(rounded)),
            )
            measures.append(rounded)
            errors.append(_upper_fraction(error))
    return np.asarray(measures, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def _fidelity_bounds(certificate: SphereMaterialFidelityCertificate, /) -> Array:
    return jnp.asarray(
        (
            certificate.mesh_to_source_lower,
            certificate.mesh_to_source_upper,
            certificate.source_to_mesh_lower,
            certificate.source_to_mesh_upper,
        ),
        dtype=jnp.float64,
    )


@checked
def prepare_sphere_chart_deformation(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    /,
) -> PreparedSphereChartDeformation:
    """Produce the exact projective common refinement and its bound deformation."""
    if not isinstance(source, SphereMaterialCellAtlas) or not isinstance(
        target, SphereMaterialCellAtlas
    ):
        raise TypeError("source and target must be SphereMaterialCellAtlas owners.")
    source.require_bound(source.source_mesh, source.source_geometry)
    target.require_bound(target.source_mesh, target.source_geometry)
    pieces = sphere_chart_overlap_pieces(source, target)
    from ..discretization._sphere_chart_records import (
        PreparedSphereChartDeformation,
        PreparedSphereChartOccurrence,
    )

    source_ids = np.asarray(source.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.cell_global_ids, dtype=np.int64)
    source_coverage = [Fraction(0) for _ in range(source_ids.size)]
    target_coverage = [Fraction(0) for _ in range(target_ids.size)]
    for piece in pieces:
        source_coverage[piece.source_row] += piece.source_reference_area
        target_coverage[piece.target_row] += piece.target_reference_area
    if any(value != Fraction(1, 2) for value in (*source_coverage, *target_coverage)):
        raise ValueError(
            "Sphere projective overlaps do not completely cover both atlases."
        )
    entity_paths = {(piece.geometry_entity_id, piece.occurrence_path) for piece in pieces}
    occurrences = []
    for entity, path in sorted(entity_paths):
        indices = tuple(
            index
            for index, piece in enumerate(pieces)
            if piece.geometry_entity_id == entity and piece.occurrence_path == path
        )
        rows = tuple(pieces[index] for index in indices)
        occurrences.append(
            PreparedSphereChartOccurrence(
                entity,
                path,
                indices,
                tuple(sorted({piece.source_cell_global_id for piece in rows})),
                tuple(sorted({piece.target_cell_global_id for piece in rows})),
                "certified",
            )
        )
    source_measures, source_errors = _triangle_measure_data(source.source_mesh)
    target_measures, target_errors = _triangle_measure_data(target.source_mesh)
    source_measure_array = jnp.asarray(source_measures)
    target_measure_array = jnp.asarray(target_measures)
    source_error_array = jnp.asarray(source_errors)
    target_error_array = jnp.asarray(target_errors)
    maximum_displacement = max(piece.displacement_bound for piece in pieces)
    source_bounds = _fidelity_bounds(source.source_fidelity_certificate)
    target_bounds = _fidelity_bounds(target.source_fidelity_certificate)
    certificates = (
        source.source_validity.certificate_id,
        target.source_validity.certificate_id,
        source.source_embedding.certificate_id,
        target.source_embedding.certificate_id,
        source.source_fidelity_certificate.certificate_id,
        target.source_fidelity_certificate.certificate_id,
    )
    deformation_id = canonical_fingerprint(
        {
            "kind": "prepared-sphere-chart-deformation",
            "source_atlas": source.atlas_id,
            "target_atlas": target.atlas_id,
            "source_geometry": source.source_geometry_id,
            "target_geometry": target.source_geometry_id,
            "source_mesh": source.source_mesh.mesh_id,
            "target_mesh": target.source_mesh.mesh_id,
            "domain": source.domain_id,
            "pieces": tuple(piece.piece_id for piece in pieces),
            "certificates": certificates,
            "source_fidelity": source_bounds,
            "target_fidelity": target_bounds,
            "source_cell_measures": array_tree_fingerprint(source_measure_array),
            "target_cell_measures": array_tree_fingerprint(target_measure_array),
            "source_measure_errors": array_tree_fingerprint(source_error_array),
            "target_measure_errors": array_tree_fingerprint(target_error_array),
            "occurrences": tuple(
                (
                    item.geometry_entity_id,
                    item.occurrence_path,
                    item.piece_indices,
                    item.source_cell_global_ids,
                    item.target_cell_global_ids,
                )
                for item in occurrences
            ),
            "displacement": maximum_displacement,
        }
    )
    deformation = PreparedSphereChartDeformation(
        source_atlas=source,
        target_atlas=target,
        source_mesh=source.source_mesh,
        target_mesh=target.source_mesh,
        source_geometry=source.source_geometry,
        target_geometry=target.source_geometry,
        source_geometry_id=source.source_geometry_id,
        target_geometry_id=target.source_geometry_id,
        source_topology_id=source.source_topology_id,
        target_topology_id=target.source_topology_id,
        domain_id=source.domain_id,
        pieces=pieces,
        occurrences=tuple(occurrences),
        source_reference_coverage=tuple(source_coverage),
        target_reference_coverage=tuple(target_coverage),
        source_fidelity_bounds=source_bounds,
        target_fidelity_bounds=target_bounds,
        source_cell_measures=jnp.asarray(source_measures),
        target_cell_measures=jnp.asarray(target_measures),
        source_measure_errors=jnp.asarray(source_errors),
        target_measure_errors=jnp.asarray(target_errors),
        source_validity=source.source_validity,
        target_validity=target.source_validity,
        source_embedding=source.source_embedding,
        target_embedding=target.source_embedding,
        source_fidelity_certificate=source.source_fidelity_certificate,
        target_fidelity_certificate=target.source_fidelity_certificate,
        maximum_displacement_bound=maximum_displacement,
        deformation_id=deformation_id,
    )
    deformation.require_bound(
        source.source_mesh,
        source.source_geometry,
        target.source_mesh,
        target.source_geometry,
    )
    return deformation


def _require_center_winding_one(triangles: tuple[_RayTriangle, ...], /) -> None:
    """Prove the exact origin winding by a generic rational ray intersection count."""
    direction_index = 1
    while True:
        direction: _Ray = (
            Fraction(1),
            Fraction(direction_index),
            Fraction(direction_index**2 + 1),
        )
        intersections = 0
        degenerate = False
        for triangle in triangles:
            origin, second, third = triangle
            edge_a: _Ray = (
                second[0] - origin[0],
                second[1] - origin[1],
                second[2] - origin[2],
            )
            edge_b: _Ray = (
                third[0] - origin[0],
                third[1] - origin[1],
                third[2] - origin[2],
            )
            matrix = (
                (edge_a[0], edge_b[0], -direction[0]),
                (edge_a[1], edge_b[1], -direction[1]),
                (edge_a[2], edge_b[2], -direction[2]),
            )
            try:
                solution = solve_exact(
                    matrix,
                    ((-origin[0],), (-origin[1],), (-origin[2],)),
                )
            except ValueError:
                degenerate = True
                break
            first_weight, second_weight, distance = (
                solution[0][0],
                solution[1][0],
                solution[2][0],
            )
            if (
                distance <= 0
                or first_weight < 0
                or second_weight < 0
                or first_weight + second_weight > 1
            ):
                continue
            if (
                first_weight == 0
                or second_weight == 0
                or first_weight + second_weight == 1
            ):
                degenerate = True
                break
            intersections += 1
        if not degenerate:
            if intersections != 1:
                raise ValueError(
                    "Sphere mesh does not enclose its analytic source center with degree one."
                )
            return
        direction_index += 1


__all__ = [
    "SphereMaterialCellAtlas",
    "SphereMaterialFidelityCertificate",
    "SphereProjectiveReferenceMap",
    "SphereProjectiveTriangleBounds",
    "sphere_chart_overlap_pieces",
    "sphere_projective_reference_map",
    "prepare_sphere_chart_deformation",
]

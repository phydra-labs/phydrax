# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Original parametric-source support of exact native surface subdivisions.

Collective preparation joins scientific root IDs on device. Owner-local proofs
consume only addressable receipts, original source expressions and actual native
coordinate restrictions. Coordinate coincidence never establishes root identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import assert_never, cast, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import _coordinate_enclosure as algebra, CellBlock, CellMesh
from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryElement,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ._meshing_domain import (
    _full_sphere_frame,
    _patch_identity,
    _sphere_triangle_distance_bounds,
    MeshingDomain,
    MeshingDomainBoundarySource,
    PatchCurveUse,
    PatchPoleUse,
)


if TYPE_CHECKING:
    from ..meshing._association import GeometryAssociation
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence
    from ..meshing._result import CellMeshingResult
    from ._mesh_certificates import (
        MeshCertificateBinding,
        MeshCertificateLimits,
        PiecewiseLinearDomain,
        SourceBoundaryDistance,
        SourceBoundaryQuery,
        SourceBoundarySamples,
        SourceFidelityCertificate,
    )

type _Corners = tuple[tuple[Fraction, Fraction], ...]


class SurfaceRestrictionRows(NamedTuple):
    """Fresh whole-map restrictions and their original chart/source identities."""

    cell_ids: np.ndarray
    root_cell_ids: np.ndarray
    patches: np.ndarray
    vertex_ids: tuple[tuple[int, ...], ...]
    reference_corners: tuple[_Corners, ...]
    source_corners: tuple[_Corners, ...]
    source_bounds: np.ndarray
    coordinate_corner_errors: np.ndarray
    restriction_deviations: np.ndarray


class SurfaceCellAncestry(NamedTuple):
    """Lineage-proved scientific root of one serial cell and its exact root-reference corners."""

    root_cell_id: int
    reference_corners: _Corners


def curve_owner(domain: MeshingDomain, curve: int, /) -> PatchCurveUse:
    """The owning coedge that defines one original curve's parameter interval."""
    patch, loop, position = map(int, domain.curve_owners[curve])
    match domain.patches[patch].loops[loop][position]:
        case PatchCurveUse() as owner:
            return owner
        case PatchPoleUse():
            raise ValueError(
                "A source curve owner names a pole side, which has no curve interval."
            )
        case invalid:
            assert_never(invalid)


def _restriction(
    element: CellGeometryElement,
) -> tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]:
    """Actual exact reference chart, including nonaffine polynomial composition."""
    from ..discretization._cell_geometry import _require_scalar_coordinate_element
    from ..discretization._coordinate_enclosure import (
        coordinate_partition_unity_reference_chain,
    )

    scalar = _require_scalar_coordinate_element(element, "Surface source reference chart")
    base, arguments = coordinate_partition_unity_reference_chain(scalar)
    if scalar.topological_dimension != 2:
        raise ValueError("A surface source chart requires two reference dimensions.")
    if any(isinstance(value, algebra.RationalPolynomial) for value in arguments):
        raise ValueError("A surface source reference restriction must remain polynomial.")
    polynomial_arguments = cast(tuple[algebra.Polynomial, ...], arguments)
    domain = "simplex" if element.cell_kind == "triangle" else "box"
    determinant = algebra.determinant(
        tuple(
            tuple(algebra.derivative(value, axis) for axis in range(2))
            for value in polynomial_arguments
        )
    )
    if algebra.polynomial_bounds(determinant, domain, 2)[0] <= 0:
        raise ValueError(
            "A native source chart reverses, collapses or has unresolved reference orientation."
        )
    return base, polynomial_arguments


def _require_orientation(matrix: tuple[tuple[Fraction, ...], ...], /) -> None:
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if determinant <= 0:
        raise ValueError(
            "A native source restriction reverses or collapses its reference orientation."
        )


def _corner_restriction(
    corners: _Corners, /
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    """Affine reference map sending the unit triangle corners to ``corners``."""
    origin = corners[0]
    matrix = tuple(
        tuple(corners[j + 1][i] - origin[i] for j in range(2)) for i in range(2)
    )
    _require_orientation(matrix)
    return matrix, origin


def _arguments(
    matrix: tuple[tuple[Fraction, ...], ...], offset: tuple[Fraction, ...]
) -> tuple[algebra.Polynomial, ...]:
    variables = algebra.axes(2)
    return tuple(
        algebra.add(
            algebra.constant(offset[i], 2),
            algebra.sum_polynomials(
                tuple(algebra.scale(variables[j], matrix[i][j]) for j in range(2))
            ),
        )
        for i in range(2)
    )


def _source_corner(
    domain: MeshingDomain, patch: int, dimension: int, index: int, parameters: np.ndarray
) -> np.ndarray:
    if dimension == 2:
        if index != patch:
            raise ValueError("A root corner names a different original source patch.")
        return np.asarray(parameters[:2], dtype=np.float64)
    if dimension == 1:
        uses = [
            use
            for loop in domain.patches[patch].loops
            for use in loop
            if isinstance(use, PatchCurveUse) and use.curve == index
        ]
        if not uses:
            raise ValueError("A root curve corner is not incident on its original patch.")
        value = float(parameters[0])
        return np.stack([use.charts(np.asarray((value,)))[0] for use in uses])
    if dimension == 0:
        return np.asarray(domain.corner_points[index])
    raise ValueError(
        "Surface support admits only original corner, curve and patch strata."
    )


def _addressable_receipt_packets(
    arrays: tuple[tuple[str, Array], ...],
    partition_index: int,
    /,
) -> tuple[tuple[str, np.ndarray], ...]:
    packets = []
    for name, value in arrays:
        packet = None
        for shard in value.addressable_shards:
            selection = shard.index[0]
            if isinstance(selection, slice):
                first = 0 if selection.start is None else selection.start
                last = value.shape[0] if selection.stop is None else selection.stop
                if first <= partition_index < last:
                    packet = np.asarray(
                        jax.device_get(shard.data[partition_index - first])
                    )
                    break
        if packet is None:
            raise ValueError("Surface receipt partition is not process-addressable.")
        packets.append((name, packet))
    return tuple(packets)


def prepare_surface_source_root_atlas(
    domain: MeshingDomain,
    patches: tuple[int, ...],
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    maximum_support_queries: int,
) -> SurfaceSourceRootAtlas:
    """Build independent source charts from original spline spans before meshing."""
    from .brep._patches import BSplineSurfacePatch

    span_rows = []
    work = 0
    for patch in patches:
        surface = domain.patches[patch].surface
        if not isinstance(surface, BSplineSurfacePatch):
            raise ValueError(
                "Exact source quad extraction requires an original rational B-spline patch."
            )
        u, v = np.asarray(surface.u_knots), np.asarray(surface.v_knots)
        spans = [
            (patch, i, j)
            for i in range(surface.u_degree, surface.control_points.shape[0])
            for j in range(surface.v_degree, surface.control_points.shape[1])
            if u[i] < u[i + 1] and v[j] < v[j + 1]
        ]
        span_rows.extend(spans)
        work += 12 * len(spans) + surface.control_points.size
    if not span_rows or work > maximum_support_queries:
        raise ValueError(
            "Original surface source atlas exceeds its declared root/control preparation bound."
        )
    positions: dict[tuple[int, float, float], int] = {}
    coordinates, blocks, elements, routes, coefficients, parameters, patch_rows = (
        [],
        [],
        {},
        {},
        [],
        [],
        [],
    )
    patch_routes: dict[int, np.ndarray] = {}
    offset = 0
    for identifier, (patch, u_span, v_span) in enumerate(span_rows):
        surface = domain.patches[patch].surface
        if not isinstance(surface, BSplineSurfacePatch):
            raise ValueError("Original source atlas lost its spline patch owner.")
        element = SplineCellGeometryElement(
            surface.u_knots,
            surface.v_knots,
            surface.weights,
            surface.u_degree,
            surface.v_degree,
            (u_span, v_span),
            domain.source_id,
            domain.source_revision,
        )
        controls = np.asarray(surface.control_points).reshape(-1, 3)
        values = algebra.coordinate_corner_images(element, controls)
        if values is None:
            raise ValueError(
                "Original spline source chart has unresolved exact corner images."
            )
        u, v = np.asarray(surface.u_knots), np.asarray(surface.v_knots)
        uv = np.asarray(
            (
                (u[u_span], v[v_span]),
                (u[u_span + 1], v[v_span]),
                (u[u_span + 1], v[v_span + 1]),
                (u[u_span], v[v_span + 1]),
            ),
            dtype=np.float64,
        )
        vertices = []
        for chart, point in zip(uv.tolist(), values, strict=True):
            key = (patch, chart[0], chart[1])
            if key not in positions:
                positions[key] = len(coordinates)
                coordinates.append(algebra.rounded_point(point))
            vertices.append(positions[key])
        name = f"source/{identifier}"
        blocks.append(
            CellBlock(
                name,
                "quadrilateral",
                np.asarray((vertices,), dtype=np.int64),
                global_ids=np.asarray((identifier,), dtype=np.int64),
            )
        )
        elements[name] = element
        if patch not in patch_routes:
            patch_routes[patch] = np.arange(
                offset, offset + controls.shape[0], dtype=np.int64
            )[None, :]
            coefficients.append(controls)
            offset += controls.shape[0]
        routes[name] = patch_routes[patch]
        parameters.append(uv)
        patch_rows.append(patch)
    geometry = CellGeometrySpec(elements, routes, np.concatenate(coefficients))
    mesh = CellMesh(
        np.stack(coordinates), tuple(blocks), numeric_version=domain.source_revision
    )
    return SurfaceSourceRootAtlas(
        domain,
        mesh,
        geometry,
        jnp.asarray(np.stack(parameters)),
        jnp.asarray(patch_rows, dtype=jnp.int64),
        coordinate_contract,
    )


class SurfaceSourceRootAtlas(StrictModule, NonTrainableState):
    """Independent exact source charts of original spline knot spans.

    This is an authored source premise, not a generated result or a target
    surrogate. Its geometry keeps the original control grid and rational
    weights; its quad carrier names span corners only.
    """

    domain: MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    root_parameters: Array
    root_patches: Array
    coordinate_contract: SpatialCoordinateContract
    atlas_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root_parameters: Array,
        root_patches: Array,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        from .brep._patches import BSplineSurfacePatch

        if (
            not isinstance(domain, MeshingDomain)
            or not isinstance(mesh, CellMesh)
            or not isinstance(geometry, CellGeometrySpec)
        ):
            raise TypeError(
                "Original surface atlas requires its actual domain, carrier and source maps."
            )
        if (
            mesh.storage is not None
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
        ):
            raise ValueError(
                "Original spline atlas requires an independent serial surface carrier."
            )
        if not coordinate_contract.is_orthonormal_cartesian:
            raise ValueError(
                "Original spline atlas requires the declared orthonormal Cartesian frame."
            )
        parameters, patches = np.asarray(root_parameters), np.asarray(root_patches)
        count = mesh.entity_set(2).count
        if parameters.shape != (count, 4, 2) or patches.shape != (count,):
            raise ValueError(
                "Original spline atlas requires four source-UV corners and one patch per root."
            )
        if (
            not np.all(np.isfinite(parameters))
            or patches.dtype.kind not in "iu"
            or np.any(patches < 0)
            or np.any(patches >= len(domain.patches))
        ):
            raise ValueError(
                "Original spline atlas has invalid source chart or patch indices."
            )
        elements, routes, coordinates = geometry.resolve(mesh)
        controls = np.asarray(coordinates, dtype=np.float64)
        declared: dict[int, set[tuple[int, int]]] = {}
        cursor = 0
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            if block.cell_kind != "quadrilateral" or not isinstance(
                element, SplineCellGeometryElement
            ):
                raise ValueError(
                    "Original spline roots require their actual unrestricted knot-span maps."
                )
            for row, dofs in enumerate(np.asarray(route)):
                patch = int(patches[cursor])
                surface = domain.patches[patch].surface
                if not isinstance(surface, BSplineSurfacePatch):
                    raise ValueError(
                        "A spline source root cannot substitute for another authored surface family."
                    )
                if (element.source_id, element.source_revision) != (
                    domain.source_id,
                    domain.source_revision,
                ):
                    raise ValueError(
                        "A spline source root changes its original source identity or revision."
                    )
                for actual, expected in (
                    (element.u_knots, surface.u_knots),
                    (element.v_knots, surface.v_knots),
                    (element.weights, surface.weights),
                    (controls[dofs], np.asarray(surface.control_points).reshape(-1, 3)),
                ):
                    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                        raise ValueError(
                            "A spline source root changes original knots, weights or control coefficients."
                        )
                if (element.u_degree, element.v_degree) != (
                    surface.u_degree,
                    surface.v_degree,
                ):
                    raise ValueError(
                        "A spline source root changes its authored reference space."
                    )
                bounds = tuple(
                    (Fraction(float(vector[span])), Fraction(float(vector[span + 1])))
                    for vector, span in zip(
                        (np.asarray(element.u_knots), np.asarray(element.v_knots)),
                        element.span_indices,
                        strict=True,
                    )
                )
                expected_uv = np.asarray(
                    (
                        (bounds[0][0], bounds[1][0]),
                        (bounds[0][1], bounds[1][0]),
                        (bounds[0][1], bounds[1][1]),
                        (bounds[0][0], bounds[1][1]),
                    ),
                    dtype=np.float64,
                )
                if not np.array_equal(parameters[cursor], expected_uv):
                    raise ValueError(
                        "A spline root's source chart corners differ from its original knot span."
                    )
                if element.span_indices in declared.setdefault(patch, set()):
                    raise ValueError(
                        "An original spline atlas repeats a scientific knot span."
                    )
                declared[patch].add(element.span_indices)
                images = algebra.coordinate_corner_images(element, controls[dofs])
                if images is None or not np.array_equal(
                    np.asarray(mesh.coordinates)[np.asarray(block.vertices)[row]],
                    np.stack(tuple(algebra.rounded_point(value) for value in images)),
                ):
                    raise ValueError(
                        "A spline root carrier is not bound to its original exact source corner images."
                    )
                cursor += 1
        for patch, spans in declared.items():
            surface = domain.patches[patch].surface
            if not isinstance(surface, BSplineSurfacePatch):
                raise ValueError("Original spline atlas lost its owning patch.")
            u = np.asarray(surface.u_knots)
            v = np.asarray(surface.v_knots)
            required = {
                (i, j)
                for i in range(surface.u_degree, surface.control_points.shape[0])
                for j in range(surface.v_degree, surface.control_points.shape[1])
                if u[i] < u[i + 1] and v[j] < v[j + 1]
            }
            if spans != required:
                raise ValueError(
                    "Original spline source atlas does not cover every authored knot span."
                )
        self.domain, self.mesh, self.geometry = domain, mesh, geometry
        self.root_parameters = jnp.asarray(parameters, dtype=jnp.float64)
        self.root_patches = jnp.asarray(patches, dtype=jnp.int64)
        self.coordinate_contract = coordinate_contract
        self.atlas_id = self._identity()

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "original-spline-surface-root-atlas",
                "domain": self.domain.domain_id,
                "expressions": [_patch_identity(patch) for patch in self.domain.patches],
                "mesh": self.mesh.mesh_id,
                "geometry": cell_geometry_id(self.geometry),
                "parameters": logical_array_value_collection_digest(
                    {"parameters": self.root_parameters, "patches": self.root_patches}
                ),
                "coordinates": self.coordinate_contract.spatial_id,
            }
        )

    def require_current(self) -> None:
        if self._identity() != self.atlas_id:
            raise ValueError(
                "Original spline source atlas has changed its source definition or root data."
            )


def restrict_surface_source_atlas_geometry(
    atlas: SurfaceSourceRootAtlas,
    target: CellMesh,
    cell_patches: np.ndarray,
    cell_parameters: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    canonical_blocks: bool = False,
) -> tuple[CellMesh, CellGeometrySpec]:
    """Restrict whole target triangles to original knot spans and control banks.

    Scientific target cell/vertex IDs and numeric revision are retained.
    Coordinate-element blocks are regrouped only because exact chart sources
    differ per cell; no target polynomial fit or centroid span selection occurs.
    """
    from ..discretization._cell_geometry import (
        _require_scalar_coordinate_element,
        CellGeometryRestrictionSource,
        PolynomialComposedCellGeometryElement,
    )

    atlas.require_current()
    if not isinstance(canonical_blocks, bool):
        raise TypeError("canonical_blocks must be bool.")
    if (
        target.storage is not None
        or target.topological_dimension != 2
        or target.ambient_dimension != 3
    ):
        raise ValueError(
            "Original source restriction requires an actual serial surface target."
        )
    if any(block.cell_kind != "triangle" for block in target.blocks):
        raise ValueError(
            "Original source chart restriction requires actual target triangles."
        )
    count = target.entity_set(2).count
    patches, parameters = (
        np.asarray(cell_patches),
        np.asarray(cell_parameters, dtype=np.float64),
    )
    if patches.shape != (count,) or parameters.shape != (count, 3, 2):
        raise ValueError(
            "Target source charts must retain one patch and three actual UV corners per cell."
        )
    if count * 12 > maximum_support_queries:
        raise ValueError(
            "Original source restriction exceeds its actual cell/chart preparation bound."
        )
    roots, root_routes, coefficients = atlas.geometry.resolve(atlas.mesh)
    source_roots = []
    cursor = 0
    for block, root, route in zip(atlas.mesh.blocks, roots, root_routes, strict=True):
        scalar = _require_scalar_coordinate_element(root, "Original source chart")
        for identifier, vertices, dofs in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(route),
            strict=True,
        ):
            source_roots.append(
                (
                    int(identifier),
                    scalar,
                    dofs,
                    np.asarray(atlas.mesh.vertex_global_ids)[vertices],
                    int(np.asarray(atlas.root_patches)[cursor]),
                    np.asarray(atlas.root_parameters)[cursor],
                )
            )
            cursor += 1
    coordinates = np.asarray(target.coordinates).copy()
    positions: dict[int, tuple[Fraction, ...]] = {}
    blocks, elements, routes, parents, parent_vertices = [], {}, {}, {}, {}
    groups: dict[str, list[tuple[int, np.ndarray, np.ndarray, int, np.ndarray]]] = {}
    triangle = coordinate_lagrange_element("triangle", 1)
    cursor = 0
    for block in target.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            patch, uv = int(patches[cursor]), parameters[cursor]
            # The affine UV image is the entire triangle's convex hull, so all
            # three corner inequalities prove containment in the original span.
            matches = [
                root
                for root in source_roots
                if root[4] == patch
                and np.all(uv >= np.min(root[5], axis=0))
                and np.all(uv <= np.max(root[5], axis=0))
            ]
            if len(matches) != 1:
                raise ValueError(
                    "A target chart crosses original knot spans; exact knot-line decomposition is required."
                )
            root_id, root, dofs, corners, _, bounds = matches[0]
            low = tuple(Fraction(float(value)) for value in bounds[0])
            width = tuple(
                Fraction(float(value)) - origin
                for value, origin in zip(bounds[2], low, strict=True)
            )
            normalized = tuple(
                tuple(
                    (Fraction(float(value)) - origin) / step
                    for value, origin, step in zip(point, low, width, strict=True)
                )
                for point in uv
            )
            element = PolynomialComposedCellGeometryElement(
                root,
                triangle,
                np.asarray(
                    [[value.numerator for value in point] for point in normalized],
                    dtype=object,
                ),
                np.asarray(
                    [[value.denominator for value in point] for point in normalized],
                    dtype=object,
                ),
            )
            images = algebra.coordinate_corner_images(
                element, np.asarray(coefficients)[dofs]
            )
            if images is None:
                raise ValueError(
                    "Original source target has unresolved exact coordinate expressions."
                )
            for vertex, point in zip(vertices.tolist(), images, strict=True):
                if vertex in positions and positions[vertex] != point:
                    raise ValueError(
                        "Original source charts disagree at a shared target vertex."
                    )
                positions[vertex] = point
                coordinates[vertex] = algebra.rounded_point(point)
            name = f"source-chart/{element.element_id}"
            if name not in groups:
                groups[name] = []
                elements[name] = element
            groups[name].append((int(identifier), vertices, dofs, root_id, corners))
            cursor += 1
    names = (
        sorted(groups)
        if canonical_blocks
        else sorted(groups, key=lambda value: min(row[0] for row in groups[value]))
    )
    for name in names:
        rows = sorted(groups[name], key=lambda row: row[0])
        blocks.append(
            CellBlock(
                name,
                "triangle",
                np.stack([row[1] for row in rows]),
                global_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
            )
        )
        routes[name] = np.stack([row[2] for row in rows])
        parents[name] = np.asarray([row[3] for row in rows], dtype=np.int64)
        parent_vertices[name] = np.stack([row[4] for row in rows])
    from ..meshing._topology_edit import entity_keys

    edge_ids_by_key = {
        tuple(map(int, key)): int(identifier)
        for key, identifier in zip(
            entity_keys(target, 1),
            np.asarray(target.entity_set(1).entity_ids),
            strict=True,
        )
    }
    seen_edges: set[tuple[int, int]] = set()
    regrouped_edge_ids: list[int] = []
    global_vertices = np.asarray(target.vertex_global_ids)
    # Polygonal connectivity numbers edges by first appearance in the actual
    # cell/local-side scan. Bind that order to the original incidence SCI keys.
    for block in blocks:
        for vertices in global_vertices[np.asarray(block.vertices)]:
            for first, last in ((0, 1), (1, 2), (2, 0)):
                a, b = int(vertices[first]), int(vertices[last])
                key = min(a, b), max(a, b)
                if key not in seen_edges:
                    regrouped_edge_ids.append(edge_ids_by_key[key])
                    seen_edges.add(key)
    mesh = CellMesh(
        coordinates,
        tuple(blocks),
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids={
            0: target.vertex_global_ids,
            1: np.asarray(regrouped_edge_ids, dtype=np.int64),
            2: np.concatenate([np.asarray(block.global_ids) for block in blocks]),
        },
        numeric_version=target.numeric_version,
        periodic_topology=target.periodic_topology,
    )
    origin = CellGeometryRestrictionSource(
        cell_geometry_id(atlas.geometry), atlas.mesh.topology_id, parents, parent_vertices
    )
    return mesh, CellGeometrySpec(
        elements, routes, coefficients, restriction_source=origin
    )


class SurfaceSourceReceipts(StrictModule, NonTrainableState):
    """Current collectively joined source-root receipts, lowered on their owner."""

    support: PreparedSurfaceSourceSupport
    geometry_projection: CellGeometryStorageProjection
    arrays: tuple[tuple[str, Array], ...]
    receipt_id: str = eqx.field(static=True)
    local_receipt_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        support: PreparedSurfaceSourceSupport,
        logical_arrays: tuple[tuple[str, Array], ...],
        geometry_projection: CellGeometryStorageProjection,
        /,
    ) -> None:
        if not isinstance(support, PreparedSurfaceSourceSupport) or not isinstance(
            geometry_projection, CellGeometryStorageProjection
        ):
            raise TypeError(
                "Surface receipts consume actual original support and a current collective coordinate projection."
            )
        support.require_current()
        geometry_projection.require_source(
            logical_arrays, geometry_projection.logical_coordinate_geometry_id
        )
        projected = dict(geometry_projection.projected_arrays)
        entries: list[tuple[str, Array]] = []
        query_count = 0
        for name, identifiers in geometry_projection.projected_arrays:
            if not name.startswith("geometry/cell_ids/"):
                continue
            bank = name.removeprefix("geometry/cell_ids/")
            active = identifiers >= 0
            parents = projected.get(f"geometry/parent_cell_ids/{bank}", identifiers)
            query_count += identifiers.size
            if query_count > support.maximum_support_queries:
                raise ValueError(
                    "Surface source receipt queries exceed their bound before root lookup."
                )
            position = jnp.searchsorted(support.root_cell_ids, jnp.maximum(parents, 0))
            clipped = jnp.minimum(position, support.root_cell_ids.shape[0] - 1)
            matched = (position < support.root_cell_ids.shape[0]) & (
                support.root_cell_ids[clipped] == parents
            )
            if not bool(jax.device_get(jnp.all(matched | ~active))):
                raise ValueError(
                    "Current native coordinate ancestry references an undeclared scientific source root."
                )
            for field, values in (
                ("parameters", support.root_parameters),
                ("patches", support.root_patches),
                ("corner_ids", support.root_corner_ids),
                ("coordinates", support.root_coordinates),
            ):
                entries.append((f"surface/{field}/{bank}", values[clipped]))
            entries.extend(
                (
                    (f"surface/cell_ids/{bank}", identifiers),
                    (f"surface/root_cell_ids/{bank}", parents),
                )
            )
        if not entries:
            raise ValueError("Current geometry has no actual native source-root queries.")
        self.support = support
        self.geometry_projection = geometry_projection
        self.arrays = tuple(sorted(entries))
        self.receipt_id = canonical_fingerprint(
            {
                "kind": "surface-source-receipts",
                "support": support.support_id,
                "geometry": geometry_projection.logical_coordinate_geometry_id,
                "projection": geometry_projection.source_content_id,
                "content": logical_array_value_collection_digest(dict(self.arrays)),
            }
        )
        addressable: set[int] = set()
        for shard in self.arrays[0][1].addressable_shards:
            selection = shard.index[0]
            if isinstance(selection, slice):
                addressable.update(
                    range(
                        0 if selection.start is None else selection.start,
                        self.arrays[0][1].shape[0]
                        if selection.stop is None
                        else selection.stop,
                    )
                )
        self.local_receipt_ids = tuple(
            (
                part,
                logical_array_value_collection_digest(
                    dict(_addressable_receipt_packets(self.arrays, part))
                ),
            )
            for part in sorted(addressable)
        )

    def addressable_receipts(
        self, partition_index: int, /
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Read only source-joined receipts owned by this addressable partition."""
        if not 0 <= partition_index < self.geometry_projection.partition_count:
            raise ValueError(
                "Surface receipt partition is outside its current geometry placement."
            )
        packets = _addressable_receipt_packets(self.arrays, partition_index)
        expected = dict(self.local_receipt_ids).get(partition_index)
        if (
            expected is None
            or logical_array_value_collection_digest(dict(packets)) != expected
        ):
            raise ValueError(
                "Current owner-local source receipt numerical content was changed."
            )
        return packets


class SurfaceSourceCharts(StrictModule, NonTrainableState):
    """Acyclic original native triangle chart premise, not a spline root atlas."""

    domain: MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    root_parameters: np.ndarray
    root_patches: np.ndarray
    associations: tuple[GeometryAssociation, ...]
    boundary_source: MeshingDomainBoundarySource
    maximum_support_queries: int = eqx.field(static=True)
    source_chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        root_parameters: np.ndarray,
        root_patches: np.ndarray,
        associations: tuple[GeometryAssociation, ...],
        boundary_source: MeshingDomainBoundarySource,
        /,
        *,
        maximum_support_queries: int,
    ) -> None:
        from ..meshing._association import (
            _entity_rows,
            _target_dimension,
            GeometryAssociation,
            GeometryAssociationKind,
        )

        if (
            not isinstance(domain, MeshingDomain)
            or not isinstance(mesh, CellMesh)
            or not isinstance(geometry, CellGeometrySpec)
        ):
            raise TypeError(
                "Original surface charts require their actual domain, native mesh and coordinate geometry."
            )
        if (
            mesh.storage is not None
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
        ):
            raise ValueError(
                "Serial original surface charts require a dense native triangle carrier."
            )
        if (
            not isinstance(coordinate_contract, SpatialCoordinateContract)
            or not coordinate_contract.is_orthonormal_cartesian
        ):
            raise ValueError(
                "Original surface charts require their orthonormal Cartesian coordinate contract."
            )
        if (
            isinstance(maximum_support_queries, bool)
            or not isinstance(maximum_support_queries, int)
            or maximum_support_queries < 1
        ):
            raise ValueError(
                "Original surface chart query capacity must be a positive integer."
            )
        count = mesh.entity_set(2).count
        if 9 * count > maximum_support_queries:
            raise ValueError(
                "Original surface charts exceed their root-bank query bound before expansion."
            )
        parameters = np.asarray(root_parameters)
        patches = np.asarray(root_patches)
        if (
            parameters.dtype != np.float64
            or parameters.shape != (count, 3, 2)
            or patches.dtype != np.int64
            or patches.shape != (count,)
            or not np.all(np.isfinite(parameters))
        ):
            raise ValueError(
                "Original surface charts require actual finite SCI-ordered binary64 corner parameters and patch rows."
            )
        if np.any(patches < 0) or np.any(patches >= len(domain.patches)):
            raise ValueError("Original surface charts name an absent source patch.")
        if (
            not isinstance(boundary_source, MeshingDomainBoundarySource)
            or boundary_source.domain.domain_id != domain.domain_id
        ):
            raise ValueError(
                "Original surface charts require their actual original source boundary premise."
            )
        if set(patches.tolist()) != set(boundary_source.patches):
            raise ValueError(
                "Original surface chart rows and continuous boundary premise select different source patches."
            )
        if not isinstance(associations, tuple) or any(
            not isinstance(value, GeometryAssociation) for value in associations
        ):
            raise TypeError(
                "Original surface chart strata must be actual GeometryAssociation records."
            )
        by_dimension = {_target_dimension(mesh, value): value for value in associations}
        expected_kind = (
            GeometryAssociationKind.BREP
            if domain.source_kinds == ("vertex", "edge", "face")
            else GeometryAssociationKind.SURFACE
        )
        if set(by_dimension) != {0, 1, 2} or any(
            value.association_kind is not expected_kind or not value.complete
            for value in by_dimension.values()
        ):
            raise ValueError(
                "Original surface charts require complete authoritative native associations on every stratum."
            )
        ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        if not np.all(ids[1:] > ids[:-1]):
            raise ValueError(
                "Original surface chart cell scientific IDs must be strictly ordered."
            )
        cell = by_dimension[2]
        rows = cell.target_rows(ids)
        declared_patches = np.asarray(
            [
                PreparedSurfaceSourceSupport._domain_row(
                    domain,
                    int(cell.source_dimensions[row]),
                    int(cell.source_indices[row]),
                    cell.source_occurrence_paths[row],
                )
                for row in rows
            ],
            dtype=np.int64,
        )
        if not np.array_equal(patches, declared_patches):
            raise ValueError(
                "Original surface chart patch rows differ from their actual cell source associations."
            )
        expected_basis = canonical_fingerprint(
            algebra.coordinate_source_signature(
                coordinate_lagrange_element("triangle", 1)
            )
        )
        elements, routes, _ = geometry.resolve(mesh)
        all_ids = []
        corners = []
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            direct = (
                not isinstance(element, RestrictedCellGeometryElement)
                and canonical_fingerprint(algebra.coordinate_source_signature(element))
                == expected_basis
            )
            composed = (
                isinstance(element, PolynomialComposedCellGeometryElement)
                and canonical_fingerprint(
                    algebra.coordinate_source_signature(element.chart_element)
                )
                == expected_basis
            )
            if (
                block.cell_kind != "triangle"
                or not (direct or composed)
                or route.ndim != 2
                or route.shape != (block.cell_count, element.local_dof_count)
            ):
                raise ValueError(
                    "Original surface charts require native P1 triangle charts "
                    "with either direct or exact composed source coordinates."
                )
            all_ids.append(np.asarray(block.global_ids, dtype=np.int64))
            corners.append(np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)])
        inverse = np.argsort(
            _entity_rows(mesh, 2, np.concatenate(all_ids)), kind="stable"
        )
        PreparedSurfaceSourceSupport._validate_serial_parameters(
            domain, patches, parameters, by_dimension[0], np.concatenate(corners)[inverse]
        )
        self.domain, self.mesh, self.geometry = domain, mesh, geometry
        self.coordinate_contract = coordinate_contract
        # np.asarray without a dtype conversion retains the actual admitted
        # producer buffers; source charts are not an independently sampled bank.
        self.root_parameters, self.root_patches = parameters, patches
        self.associations, self.boundary_source = tuple(associations), boundary_source
        self.maximum_support_queries = maximum_support_queries
        self.source_chart_id = self._identity()

    @staticmethod
    def _source_identity(domain: MeshingDomain) -> str:
        from .._fingerprint import array_tree_fingerprint

        return canonical_fingerprint(
            {
                "source": (
                    domain.source_id,
                    domain.source_revision,
                    domain.authority_id,
                    domain.domain_id,
                ),
                "expressions": [_patch_identity(patch) for patch in domain.patches],
                "arrays": array_tree_fingerprint(domain),
                "source_kinds": domain.source_kinds,
                "source_indices": domain.source_indices,
                "source_occurrences": domain.source_occurrences,
                "region_indices": domain.region_source_indices,
                "region_occurrences": domain.region_source_occurrences,
                "regions": [(region.name, region.boundary) for region in domain.regions],
                "incidence": [(curve.start, curve.end) for curve in domain.curves],
                "accuracy": domain.accuracy,
                "tolerance": domain.tolerance,
                "scale": domain.scale,
            }
        )

    def _identity(self) -> str:
        from .._fingerprint import array_tree_fingerprint

        return canonical_fingerprint(
            {
                "kind": "native-original-surface-charts",
                "source": self._source_identity(self.domain),
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "geometry": cell_geometry_id(self.geometry),
                "geometry_layout": self.geometry.geometry_layout_id,
                "mesh_content": array_tree_fingerprint(
                    (
                        self.mesh.coordinates,
                        self.mesh.vertex_global_ids,
                        tuple(
                            (block.vertices, block.global_ids)
                            for block in self.mesh.blocks
                        ),
                    )
                ),
                "charts": array_tree_fingerprint(
                    (self.root_parameters, self.root_patches)
                ),
                "associations": [
                    (
                        value.association_id,
                        value.source_id,
                        value.source_revision,
                        value.source_entity_ids,
                        value.source_occurrence_paths,
                    )
                    for value in self.associations
                ],
                "association_arrays": array_tree_fingerprint(self.associations),
                "boundary": self.boundary_source.source_scope_id,
                "boundary_source": self._source_identity(self.boundary_source.domain),
                "boundary_charts": array_tree_fingerprint(
                    self.boundary_source.chart_triangulations
                ),
                "maximum_queries": self.maximum_support_queries,
            }
        )

    def require_current(self) -> None:
        if self._identity() != self.source_chart_id:
            raise ValueError(
                "Original native surface charts, source incidence or scientific coefficient banks changed."
            )

    def require_root(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        self.require_current()
        if mesh.topology_id != self.mesh.topology_id or cell_geometry_id(
            geometry
        ) != cell_geometry_id(self.geometry):
            raise ValueError(
                "Native surface chart premise is bound to a different original root carrier."
            )

    def require_bound(
        self, mesh: CellMesh, geometry: CellGeometrySpec, source: object, /
    ) -> None:
        """Authenticate a publication's retained original premise and live map."""
        from ..meshing._result import CellMeshingResult
        from ..meshing._surface_association_transfer import (
            SurfaceChartBoundarySource,
            SurfaceSubdivisionBoundarySource,
        )

        self.require_current()
        if isinstance(source, SurfaceNativeRestrictionBoundarySource):
            source.require_current()
            self.require_root(mesh, geometry)
            if source.mesh.mesh_id != mesh.mesh_id or cell_geometry_id(
                source.geometry
            ) != cell_geometry_id(geometry):
                raise ValueError(
                    "Native restriction source is bound to another live map."
                )
            return
        if isinstance(source, MeshingDomainBoundarySource):
            self.require_root(mesh, geometry)
            from .._fingerprint import array_tree_fingerprint

            if (
                self._source_identity(source.domain) != self._source_identity(self.domain)
                or source.source_scope_id != self.boundary_source.source_scope_id
                or array_tree_fingerprint(source.chart_triangulations)
                != array_tree_fingerprint(self.boundary_source.chart_triangulations)
            ):
                raise ValueError(
                    "Original surface publication changed its actual retained source chart premise."
                )
            return
        if isinstance(
            source, (SurfaceChartBoundarySource, SurfaceSubdivisionBoundarySource)
        ):
            source.support.require_current()
            original = source.support.original
            if isinstance(original, SurfaceSourceRootAtlas):
                if not isinstance(source, SurfaceChartBoundarySource):
                    raise ValueError(
                        "Spline-atlas descendants require an owned chart deformation."
                    )
                original.require_current()
                deformation = source.deformation
                self.require_root(deformation.source_mesh, deformation.source_geometry)
                deformation.require_bound(
                    deformation.source_mesh,
                    deformation.source_geometry,
                    mesh,
                    geometry,
                )
                return
            if not isinstance(original, CellMeshingResult):
                raise ValueError(
                    "Surface descendants require their actual accepted original chart carrier."
                )
            charts = original.surface_source
            if (
                not isinstance(charts, SurfaceSourceCharts)
                or charts.source_chart_id != self.source_chart_id
            ):
                raise ValueError(
                    "Surface descendant lost its exact original scientific chart owner."
                )
            charts.require_root(original.mesh, original.geometry)
            if isinstance(source, SurfaceChartBoundarySource):
                deformation = source.deformation
                deformation.require_bound(
                    deformation.source_mesh, deformation.source_geometry, mesh, geometry
                )
            elif (
                not source.targets
                or source.targets[-1].topology_id != mesh.topology_id
                or cell_geometry_id(source.geometry) != cell_geometry_id(geometry)
            ):
                raise ValueError(
                    "Surface subdivision premise is bound to another live target map."
                )
            return
        raise ValueError(
            "Surface publication lacks its actual original chart/deformation/subdivision source graph."
        )


class PreparedSurfaceSourceSupport(StrictModule, NonTrainableState):
    """Original source theorem and scientific root banks, never a dense surrogate."""

    domain: MeshingDomain
    coordinate_contract: SpatialCoordinateContract
    original: CellMeshingResult | SurfaceSourceRootAtlas | InitialCollectiveMeshEvidence
    root_boundary_source: MeshingDomainBoundarySource | InitialCollectiveMeshEvidence
    root_cell_ids: Array
    root_parameters: Array
    root_patches: Array
    root_corner_ids: Array
    root_coordinates: Array
    root_basis_signature: str = eqx.field(static=True)
    root_geometry_id: str = eqx.field(static=True)
    root_topology_id: str = eqx.field(static=True)
    original_evidence_id: str = eqx.field(static=True)
    source_expression_id: str = eqx.field(static=True)
    maximum_support_queries: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: MeshingDomain,
        root_result: CellMeshingResult | SurfaceSourceRootAtlas,
        root_cell_parameters: Array,
        root_boundary_source: MeshingDomainBoundarySource | InitialCollectiveMeshEvidence,
        /,
        *,
        maximum_support_queries: int = 1 << 26,
    ) -> None:
        from ..meshing._association import (
            _entity_rows,
        )
        from ..meshing._initial_certification import InitialCollectiveMeshEvidence
        from ..meshing._result import CellMeshingResult

        if not isinstance(domain, MeshingDomain) or not isinstance(
            root_result, (CellMeshingResult, SurfaceSourceRootAtlas)
        ):
            raise TypeError(
                "Surface support requires an accepted source result or an independent original source atlas."
            )
        if (
            isinstance(maximum_support_queries, bool)
            or not isinstance(maximum_support_queries, int)
            or maximum_support_queries < 1
        ):
            raise ValueError("maximum_support_queries must be a positive integer.")
        if (
            root_result.mesh.topological_dimension != 2
            or root_result.mesh.ambient_dimension != 3
        ):
            raise ValueError(
                "Native source support requires actual surface triangles in ambient three-space."
            )
        if not root_result.coordinate_contract.is_orthonormal_cartesian:
            raise ValueError(
                "Surface source support requires its declared orthonormal Cartesian frame."
            )
        expected_basis = canonical_fingerprint(
            algebra.coordinate_source_signature(
                coordinate_lagrange_element("triangle", 1)
            )
        )
        corner_count, coefficient_count = 3, 3
        if isinstance(root_result, SurfaceSourceRootAtlas):
            root_result.require_current()
            if (
                not isinstance(root_boundary_source, MeshingDomainBoundarySource)
                or SurfaceSourceCharts._source_identity(root_boundary_source.domain)
                != SurfaceSourceCharts._source_identity(domain)
                or SurfaceSourceCharts._source_identity(root_result.domain)
                != SurfaceSourceCharts._source_identity(domain)
            ):
                raise ValueError(
                    "Original spline source atlas must retain its actual authored domain authority."
                )
            if set(np.asarray(root_result.root_patches).tolist()) != set(
                root_boundary_source.patches
            ):
                raise ValueError(
                    "Original source atlas and boundary premise select different source patches."
                )
            original = root_result
            count = root_result.mesh.entity_set(2).count
            ids = root_result.mesh.entity_set(2).entity_ids
            parameters, patches = root_result.root_parameters, root_result.root_patches
            elements, routes, controls = root_result.geometry.resolve(root_result.mesh)
            coefficient_count = max(element.local_dof_count for element in elements)
            coordinates = jnp.concatenate(
                tuple(
                    jnp.pad(
                        jnp.asarray(controls)[route],
                        (
                            (0, 0),
                            (0, coefficient_count - element.local_dof_count),
                            (0, 0),
                        ),
                    )
                    for element, route in zip(elements, routes, strict=True)
                )
            )
            corners = jnp.concatenate(
                tuple(
                    jnp.asarray(root_result.mesh.vertex_global_ids)[block.vertices]
                    for block in root_result.mesh.blocks
                )
            )
            corner_count = 4
            expected_basis = canonical_fingerprint(
                tuple(
                    algebra.coordinate_source_signature(element) for element in elements
                )
            )
            geometry_id, topology_id, evidence_id = (
                cell_geometry_id(root_result.geometry),
                root_result.mesh.topology_id,
                root_result.atlas_id,
            )
        elif isinstance(root_boundary_source, InitialCollectiveMeshEvidence):
            original = root_boundary_source
            original.require_current()
            if not isinstance(root_result, CellMeshingResult):
                raise TypeError(
                    "Collective source support requires its accepted native result."
                )
            if (
                original.compiled.domain.domain_id != domain.domain_id
                or root_result.collective_evidence is None
            ):
                raise ValueError(
                    "Collective source support must bind its genuine original authored source theorem."
                )
            from ..meshing._result import require_original_meshing_source

            actual = require_original_meshing_source(root_result)
            if (
                not isinstance(actual, InitialCollectiveMeshEvidence)
                or actual.evidence_id != original.evidence_id
            ):
                raise ValueError(
                    "The accepted local carrier does not have this original collective source authority."
                )
            if root_result.mesh.topology_id != original.topology_id:
                raise ValueError(
                    "Surface source preparation requires the actual initial scientific root topology."
                )
            banks = dict(original.logical_arrays)
            count = original.global_entity_counts[2]
            ids = original.entity_ids[2][:count]
            parameters = banks["initial/cell_parameters"][:count]
            patches = banks["initial/cell_patches"][:count]
            corners = banks["cell_vertices"][:count]
            coefficient_ids = banks["geometry/coordinate_ids"][
                : original.global_entity_counts[0]
            ]
            position = jnp.searchsorted(coefficient_ids, corners)
            clipped = jnp.minimum(position, coefficient_ids.shape[0] - 1)
            if not bool(
                jax.device_get(
                    jnp.all(
                        (position < coefficient_ids.shape[0])
                        & (coefficient_ids[clipped] == corners)
                    )
                )
            ):
                raise ValueError(
                    "Original scientific source corners do not name their actual coordinate coefficients."
                )
            coordinates = banks["geometry/coordinates"][clipped]
            expected = jnp.asarray(
                np.frombuffer(bytes.fromhex(expected_basis), dtype=np.uint8)
            )
            if not bool(
                jax.device_get(
                    jnp.all(banks["geometry/source_basis/surface"][:count] == expected)
                )
            ):
                raise ValueError(
                    "Original source support requires its actual native P1 coordinate expression."
                )
            geometry_id, topology_id, evidence_id = (
                original.coordinate_geometry_id,
                original.topology_id,
                original.source_evidence_id,
            )
        elif isinstance(root_boundary_source, MeshingDomainBoundarySource):
            if not isinstance(root_result, CellMeshingResult):
                raise TypeError(
                    "Accepted serial source support requires its actual native result."
                )
            if root_result.mesh.storage is not None:
                raise ValueError(
                    "An owner-local carrier cannot supply a whole dense source theorem."
                )
            if SurfaceSourceCharts._source_identity(
                root_boundary_source.domain
            ) != SurfaceSourceCharts._source_identity(domain):
                raise ValueError(
                    "The original continuous chart source has changed authority."
                )
            cover = root_boundary_source.boundary_chart_cover(maximum_support_queries)
            certification = root_result.certification
            if certification is None:
                raise ValueError(
                    "Surface source preparation requires the accepted root certification report."
                )
            fidelity = certification.fidelity
            if (
                not cover.complete
                or cover.semantics != "certified"
                or fidelity is None
                or fidelity.status != "certified"
            ):
                raise ValueError(
                    "Surface source preparation requires original continuous source coverage and fidelity."
                )
            fidelity.binding.require(root_result.mesh, root_result.geometry)
            if (
                root_boundary_source.source_id != domain.source_id
                or root_boundary_source.source_revision != domain.source_revision
            ):
                raise ValueError(
                    "Original source coverage has a different source revision."
                )
            source_charts = root_result.surface_source
            if not isinstance(source_charts, SurfaceSourceCharts):
                raise ValueError(
                    "Accepted native surface roots must retain their actual original source chart premise."
                )
            source_charts.require_root(root_result.mesh, root_result.geometry)
            from .._fingerprint import array_tree_fingerprint

            if (
                SurfaceSourceCharts._source_identity(source_charts.domain)
                != SurfaceSourceCharts._source_identity(domain)
                or source_charts.boundary_source.source_scope_id
                != root_boundary_source.source_scope_id
                or array_tree_fingerprint(
                    source_charts.boundary_source.chart_triangulations
                )
                != array_tree_fingerprint(root_boundary_source.chart_triangulations)
            ):
                raise ValueError(
                    "Surface support must reuse its actual retained continuous root boundary premise."
                )
            original = root_result
            count = root_result.mesh.entity_set(2).count
            ids = jnp.asarray(root_result.mesh.entity_set(2).entity_ids, dtype=jnp.int64)
            parameters = jnp.asarray(source_charts.root_parameters)
            patches = jnp.asarray(source_charts.root_patches)
            corner_rows = []
            corner_ids = []
            elements, routes, controls = root_result.geometry.resolve(root_result.mesh)
            all_ids = []
            for block, element, route in zip(
                root_result.mesh.blocks, elements, routes, strict=True
            ):
                if block.cell_kind != "triangle":
                    raise ValueError(
                        "Original surface roots require actual native triangle expressions."
                    )
                corner_rows.append(jnp.asarray(controls)[jnp.asarray(route)])
                corner_ids.append(
                    jnp.asarray(root_result.mesh.vertex_global_ids)[
                        jnp.asarray(block.vertices)
                    ]
                )
                all_ids.append(np.asarray(block.global_ids, dtype=np.int64))
            order = _entity_rows(root_result.mesh, 2, np.concatenate(all_ids))
            inverse = np.argsort(order, kind="stable")
            coordinates = jnp.concatenate(corner_rows)[inverse]
            corners = jnp.concatenate(corner_ids)[inverse]
            geometry_id, topology_id, evidence_id = (
                cell_geometry_id(root_result.geometry),
                root_result.mesh.topology_id,
                fidelity.certificate_id,
            )
        else:
            raise TypeError(
                "Original source premise must be MeshingDomainBoundarySource or genuine InitialCollectiveMeshEvidence."
            )
        if count * (corner_count * 2 + coefficient_count) > maximum_support_queries:
            raise ValueError(
                "Surface source preparation exceeds its root-bank query bound before receipt expansion."
            )
        supplied = jnp.asarray(root_cell_parameters, dtype=jnp.float64)
        if (
            supplied.shape != (count, corner_count, 2)
            or not bool(jax.device_get(jnp.all(supplied == parameters)))
            or not bool(jax.device_get(jnp.all(jnp.isfinite(parameters))))
        ):
            raise ValueError(
                "Root parameters must be the actual original physical cell-corner chart bank."
            )
        if (
            not bool(jax.device_get(jnp.all(ids[1:] > ids[:-1])))
            or patches.shape != (count,)
            or corners.shape != (count, corner_count)
            or coordinates.shape != (count, coefficient_count, 3)
        ):
            raise ValueError(
                "Original surface scientific root axes are not canonical and aligned."
            )
        self.domain, self.coordinate_contract, self.original = (
            domain,
            root_result.coordinate_contract,
            original,
        )
        self.root_cell_ids, self.root_parameters, self.root_patches = (
            ids,
            parameters,
            patches,
        )
        self.root_boundary_source = root_boundary_source
        self.root_corner_ids, self.root_coordinates = corners, coordinates
        self.root_basis_signature = expected_basis
        self.root_geometry_id, self.root_topology_id, self.original_evidence_id = (
            geometry_id,
            topology_id,
            evidence_id,
        )
        self.source_expression_id = canonical_fingerprint(
            [_patch_identity(patch) for patch in domain.patches]
        )
        self.maximum_support_queries = maximum_support_queries
        self.support_id = self._identity()

    @staticmethod
    def _domain_row(
        domain: MeshingDomain, dimension: int, index: int, path: tuple[str, ...]
    ) -> int:
        if dimension not in (0, 1, 2):
            raise ValueError("A surface stratum must retain its actual source dimension.")
        rows = [
            row
            for row, (label, occurrence) in enumerate(
                zip(
                    domain.source_indices[dimension],
                    domain.source_occurrences[dimension],
                    strict=True,
                )
            )
            if label == index and occurrence == path
        ]
        if len(rows) != 1:
            raise ValueError(
                "A surface association has no unique original occurrence/definition authority."
            )
        return rows[0]

    @staticmethod
    def _validate_serial_parameters(
        domain: MeshingDomain,
        patches: np.ndarray,
        parameters: np.ndarray,
        vertex: GeometryAssociation,
        corners: np.ndarray,
    ) -> None:
        for patch, chart, ids in zip(patches.tolist(), parameters, corners, strict=True):
            rows = vertex.target_rows(ids)
            for uv, row in zip(chart, rows.tolist(), strict=True):
                dimension = int(vertex.source_dimensions[row])
                index = PreparedSurfaceSourceSupport._domain_row(
                    domain,
                    dimension,
                    int(vertex.source_indices[row]),
                    vertex.source_occurrence_paths[row],
                )
                values = np.asarray(vertex.parameters[row])
                if dimension == 0:
                    poles = [
                        use
                        for loop in domain.patches[patch].loops
                        for use in loop
                        if isinstance(use, PatchPoleUse) and use.corner == index
                    ]
                    declared = any(
                        uv[1] == use.start[1]
                        and min(use.start[0], use.end[0])
                        <= uv[0]
                        <= max(use.start[0], use.end[0])
                        for use in poles
                    )
                    if (
                        not declared
                        and np.linalg.norm(
                            domain.evaluate(np.asarray((patch,)), uv[None])[0]
                            - domain.corner_points[index]
                        )
                        > 128 * np.finfo(np.float64).eps * domain.scale
                    ):
                        raise ValueError(
                            "A physical root corner lacks its actual original chart/stratum binding."
                        )
                else:
                    expected = _source_corner(domain, patch, dimension, index, values)
                    candidates = expected[None] if expected.ndim == 1 else expected
                    if not np.any(
                        np.all(
                            np.abs(candidates - uv)
                            <= 128
                            * np.finfo(np.float64).eps
                            * np.maximum(1.0, np.abs(candidates)),
                            axis=1,
                        )
                    ):
                        raise ValueError(
                            "Root source parameters differ from actual original vertex/coedge associations."
                        )

    def _identity(self) -> str:
        from .._fingerprint import array_tree_fingerprint
        from ..meshing._initial_certification import InitialCollectiveMeshEvidence

        boundary = self.root_boundary_source
        if isinstance(boundary, MeshingDomainBoundarySource):
            boundary_identity = canonical_fingerprint(
                {
                    "domain": SurfaceSourceCharts._source_identity(boundary.domain),
                    "scope": boundary.source_scope_id,
                    "resolution": boundary.resolution,
                    "charts": array_tree_fingerprint(boundary.chart_triangulations),
                }
            )
        elif isinstance(boundary, InitialCollectiveMeshEvidence):
            boundary.require_current()
            boundary_identity = boundary.evidence_id
        else:
            raise TypeError(
                "Original surface support lost its actual retained boundary premise."
            )
        arrays = {
            "cell_ids": self.root_cell_ids,
            "parameters": self.root_parameters,
            "patches": self.root_patches,
            "corner_ids": self.root_corner_ids,
            "coordinates": self.root_coordinates,
        }
        return canonical_fingerprint(
            {
                "kind": "prepared-native-surface-source-support",
                "domain": self.domain.domain_id,
                "source": self.domain.source_id,
                "revision": self.domain.source_revision,
                "source_expressions": [
                    _patch_identity(patch) for patch in self.domain.patches
                ],
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "geometry": self.root_geometry_id,
                "topology": self.root_topology_id,
                "original_theorem": self.original_evidence_id,
                "basis": self.root_basis_signature,
                "boundary_premise": boundary_identity,
                "content": logical_array_value_collection_digest(arrays),
                "maximum_queries": self.maximum_support_queries,
            }
        )

    def require_current(self) -> None:
        from ..meshing._result import CellMeshingResult

        if isinstance(self.original, CellMeshingResult):
            charts = self.original.surface_source
            if not isinstance(charts, SurfaceSourceCharts):
                raise ValueError(
                    "Accepted native source support lost its original chart premise."
                )
            charts.require_root(self.original.mesh, self.original.geometry)
        elif isinstance(self.original, SurfaceSourceRootAtlas):
            self.original.require_current()
        if self._identity() != self.support_id:
            raise ValueError(
                "Original surface support maps, source strata or scientific identities have changed."
            )

    def prepare_receipts(
        self,
        logical_arrays: tuple[tuple[str, Array], ...],
        geometry_projection: CellGeometryStorageProjection,
        /,
    ) -> SurfaceSourceReceipts:
        """Collective global-preparation boundary; never called from local guards."""
        return SurfaceSourceReceipts(self, logical_arrays, geometry_projection)

    def _root_packets(
        self, mesh: CellMesh, receipts: SurfaceSourceReceipts | None, /
    ) -> dict[int, tuple[np.ndarray, int, np.ndarray, np.ndarray]]:
        """Root chart/patch/corner/coefficient rows: whole serial banks or owner-local receipts."""
        root_packets: dict[int, tuple[np.ndarray, int, np.ndarray, np.ndarray]] = {}
        if mesh.storage is None:
            self.require_current()
            if not all(
                value.is_fully_addressable
                for value in (
                    self.root_cell_ids,
                    self.root_parameters,
                    self.root_patches,
                    self.root_corner_ids,
                    self.root_coordinates,
                )
            ):
                raise ValueError(
                    "A dense surface proof cannot host-gather collective scientific root banks."
                )
            for identifier, uv, patch, corners, values in zip(
                np.asarray(self.root_cell_ids),
                np.asarray(self.root_parameters),
                np.asarray(self.root_patches),
                np.asarray(self.root_corner_ids),
                np.asarray(self.root_coordinates),
                strict=True,
            ):
                root_packets[int(identifier)] = (uv, int(patch), corners, values)
            return root_packets
        if receipts is None or receipts.support is not self:
            raise ValueError(
                "Owner-local surface support requires current collectively prepared source receipts."
            )
        packets = dict(receipts.addressable_receipts(mesh.storage.partition_index))
        for name, identifiers in packets.items():
            if not name.startswith("surface/cell_ids/"):
                continue
            bank = name.removeprefix("surface/cell_ids/")
            for row in np.flatnonzero(identifiers >= 0).tolist():
                root = int(packets[f"surface/root_cell_ids/{bank}"][row])
                value = (
                    packets[f"surface/parameters/{bank}"][row],
                    int(packets[f"surface/patches/{bank}"][row]),
                    packets[f"surface/corner_ids/{bank}"][row],
                    packets[f"surface/coordinates/{bank}"][row],
                )
                if root in root_packets and any(
                    not np.array_equal(a, b)
                    for a, b in zip(root_packets[root], value, strict=True)
                ):
                    raise ValueError(
                        "Current root receipts disagree on original source data."
                    )
                root_packets[root] = value
        return root_packets

    def _root_element(self, identifier: int, /) -> CellGeometryElement:
        if isinstance(self.original, SurfaceSourceRootAtlas):
            self.original.require_current()
            elements, _, _ = self.original.geometry.resolve(self.original.mesh)
            for block, element in zip(self.original.mesh.blocks, elements, strict=True):
                if identifier in np.asarray(block.global_ids):
                    return element
            raise ValueError("A source root ID is absent from its original spline atlas.")
        return coordinate_lagrange_element("triangle", 1)

    def _root_expression(
        self, identifier: int, packet: tuple[np.ndarray, int, np.ndarray, np.ndarray], /
    ) -> tuple[tuple[algebra.Expression, ...], tuple[algebra.Polynomial, ...], float]:
        """Original whole-map coordinate/chart expressions and their source bound."""
        uv, patch, _, coefficients = packet
        basis = self._root_element(identifier)
        root_expressions = algebra.coordinate_expressions(
            basis, coefficients[: basis.local_dof_count]
        )
        uv_basis = coordinate_lagrange_element(basis.cell_kind, 1)
        uv_polynomials = algebra.coordinate_polynomials(uv_basis, uv)
        if root_expressions is None or uv_polynomials is None:
            raise ValueError("Original source coordinate/chart expression is unresolved.")
        if isinstance(self.original, SurfaceSourceRootAtlas):
            return root_expressions, uv_polynomials, 0.0
        source_nodes = self.domain.evaluate(np.full(3, patch), uv)
        nodal_error = float(np.max(np.linalg.norm(source_nodes - coefficients, axis=1)))
        bound = (
            float(self.domain.interpolation_bounds(patch, uv[None])[0])
            + nodal_error
            + 256 * np.finfo(np.float64).eps * self.domain.scale
        )
        sphere = _full_sphere_frame(self.domain, patch)
        if sphere is not None:
            bound = min(
                bound,
                float(_sphere_triangle_distance_bounds(sphere, coefficients[None])[0]),
            )
        if not np.isfinite(bound):
            raise ValueError(
                "Original source expression has no finite whole-root coordinate-image enclosure."
            )
        return root_expressions, uv_polynomials, bound

    def prove_native_restrictions(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        receipts: SurfaceSourceReceipts | None = None,
        ancestry: Mapping[int, SurfaceCellAncestry] | None = None,
    ) -> SurfaceRestrictionRows:
        """Prove actual whole coordinate maps against original scientific root IDs.

        Restricted maps must be the exact inherited root expression. A serial
        unrestricted map instead takes its root and root-reference corners from
        ``ancestry`` (actual topology lineage); its measured deviation from the
        exact restricted root expression is added to the source bound.
        """
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Surface restriction proof requires actual mesh and coordinate map owners."
            )
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError(
                "Surface source restriction requires surface cells in three-space."
            )
        if (
            canonical_fingerprint(
                [_patch_identity(patch) for patch in self.domain.patches]
            )
            != self.source_expression_id
        ):
            raise ValueError(
                "The original authored surface source expression was changed."
            )
        restrictions = geometry.restriction_source
        if restrictions is not None and (
            restrictions.source_geometry_id != self.root_geometry_id
            or restrictions.source_topology_id != self.root_topology_id
        ):
            raise ValueError(
                "Native surface restrictions do not retain the original scientific root identity."
            )
        if ancestry is not None and (
            restrictions is not None or mesh.storage is not None
        ):
            raise ValueError(
                "Lineage ancestry proves only serial unrestricted native coordinate maps."
            )
        root_packets = self._root_packets(mesh, receipts)
        elements, routes, controls = geometry.resolve(mesh)
        host_controls = np.asarray(controls, dtype=np.float64)
        host_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        host_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        root_cache: dict[
            int,
            tuple[tuple[algebra.Expression, ...], tuple[algebra.Polynomial, ...], float],
        ] = {}
        restriction_cache: dict[
            str, tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]
        ] = {}
        result_ids: list[int] = []
        root_ids: list[int] = []
        patch_ids: list[int] = []
        vertex_ids: list[tuple[int, ...]] = []
        reference_rows: list[_Corners] = []
        source_rows: list[_Corners] = []
        bounds: list[float] = []
        errors: list[float] = []
        deviations: list[float] = []
        budget = algebra.CoordinateEnclosureBudget(
            self.maximum_support_queries, 256 * 1024**2
        )
        with budget.activate():
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
                if block.cell_kind not in ("triangle", "quadrilateral"):
                    raise ValueError(
                        "Native surface source support requires actual triangle or quad cells."
                    )
                original_action = element
                while isinstance(
                    original_action,
                    (
                        RestrictedCellGeometryElement,
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                    ),
                ):
                    original_action = original_action.source_element
                packet_action = isinstance(
                    original_action, BarycentricCellGeometryElement
                )
                if not packet_action:
                    cached_restriction = restriction_cache.get(element.element_id)
                    if cached_restriction is None:
                        cached_restriction = _restriction(element)
                        restriction_cache[element.element_id] = cached_restriction
                        budget.retain_basis(cached_restriction)
                    else:
                        budget.reserve(1)
                    base, block_arguments = cached_restriction
                parents = (
                    np.asarray(block.global_ids)
                    if restrictions is None
                    else np.asarray(restrictions.block_parent_cell_ids[block.name])
                )
                parent_vertices = (
                    None
                    if restrictions is None
                    else np.asarray(restrictions.block_parent_vertex_ids[block.name])
                )
                for row, (identifier, parent, vertices, dofs) in enumerate(
                    zip(
                        np.asarray(block.global_ids).tolist(),
                        parents.tolist(),
                        np.asarray(block.vertices),
                        np.asarray(route),
                        strict=True,
                    )
                ):
                    lineage_corners = None
                    if ancestry is not None:
                        if identifier not in ancestry:
                            raise ValueError(
                                "A serial surface cell has no lineage-proved scientific source root."
                            )
                        parent, lineage_corners = ancestry[identifier]
                    if parent not in root_packets:
                        raise ValueError(
                            "Current surface cell lacks its actual source-root receipt."
                        )
                    packet = root_packets[parent]
                    if parent_vertices is not None and not np.array_equal(
                        parent_vertices[row][: packet[2].size], packet[2]
                    ):
                        raise ValueError(
                            "Native restriction changes ordered original source corner identities."
                        )
                    if parent not in root_cache:
                        # Root expressions stay cached, so their storage stays charged.
                        root_cache[parent] = self._root_expression(parent, packet)
                    with budget.temporary_scope():
                        remainder = None
                        if packet_action:
                            if (
                                restrictions is None
                                or parent_vertices is None
                                or ancestry is not None
                            ):
                                raise ValueError(
                                    "Full coefficient surface actions require actual retained original root SCI ancestry."
                                )
                            base, arguments, remainder = (
                                algebra.prepare_p1_source_packet_pullback(
                                    element,
                                    self._root_element(parent),
                                    packet[0],
                                    packet[3],
                                    host_controls[dofs],
                                    root_cache[parent][0],
                                    root_corner_ids=packet[2],
                                    retained_corner_ids=parent_vertices[row][
                                        : packet[2].size
                                    ],
                                )
                            )
                            determinant = algebra.determinant(
                                tuple(
                                    tuple(
                                        algebra.derivative(value, axis)
                                        for axis in range(2)
                                    )
                                    for value in arguments
                                )
                            )
                            if (
                                algebra.polynomial_bounds(determinant, "simplex", 2)[0]
                                <= 0
                            ):
                                raise ValueError(
                                    "A full coefficient source UV map reverses, collapses or has unresolved orientation."
                                )
                        else:
                            arguments = (
                                block_arguments
                                if lineage_corners is None
                                else _arguments(*_corner_restriction(lineage_corners))
                            )
                        expected = canonical_fingerprint(
                            algebra.coordinate_source_signature(
                                self._root_element(parent)
                            )
                        )
                        if (
                            canonical_fingerprint(
                                algebra.coordinate_source_signature(base)
                            )
                            != expected
                        ):
                            raise ValueError(
                                "A target restriction changed its original source basis expression."
                            )
                        refs, source_corners, deviation, actual_corners = (
                            _cell_restriction(
                                element,
                                host_controls[dofs],
                                arguments,
                                root_cache[parent],
                                root_kind=self._root_element(parent).cell_kind,
                                exact=ancestry is None,
                                physical_remainder=remainder,
                                exact_source_controls=(
                                    packet[3][: base.local_dof_count]
                                    if ancestry is None and remainder is None
                                    else None
                                ),
                            )
                        )
                    # This is the source-definition chord enclosure. Whole
                    # trimmed-source fidelity additionally consumes the original
                    # complete source theorem and CURRENT full subdivision proof.
                    carrier = host_coordinates[vertices]
                    error = max(
                        sum(
                            (
                                abs(value - Fraction(float(rep)))
                                for value, rep in zip(point, represented, strict=True)
                            ),
                            Fraction(0),
                        )
                        for point, represented in zip(
                            actual_corners, carrier, strict=True
                        )
                    )
                    result_ids.append(identifier)
                    root_ids.append(parent)
                    patch_ids.append(packet[1])
                    vertex_ids.append(tuple(host_vertex_ids[vertices].tolist()))
                    reference_rows.append(refs)
                    source_rows.append(source_corners)
                    bound = root_cache[parent][2]
                    source_bound = (
                        bound
                        if isinstance(self.original, SurfaceSourceRootAtlas)
                        else float(np.nextafter(bound, np.inf))
                    )
                    bounds.append(_upper(Fraction(source_bound) + deviation))
                    errors.append(_upper(error))
                    deviations.append(_upper(deviation))
        return SurfaceRestrictionRows(
            np.asarray(result_ids, dtype=np.int64),
            np.asarray(root_ids, dtype=np.int64),
            np.asarray(patch_ids, dtype=np.int64),
            tuple(vertex_ids),
            tuple(reference_rows),
            tuple(source_rows),
            np.asarray(bounds, dtype=np.float64),
            np.asarray(errors, dtype=np.float64),
            np.asarray(deviations, dtype=np.float64),
        )


def _equal_expressions(first: algebra.Expression, second: algebra.Expression, /) -> bool:
    a, b = algebra.expression_parts(first, 2)
    c, d = algebra.expression_parts(second, 2)
    return algebra.multiply(a, d) == algebra.multiply(c, b)


def _cell_restriction(
    element: CellGeometryElement,
    controls: np.ndarray,
    arguments: tuple[algebra.Polynomial, ...],
    expression: tuple[
        tuple[algebra.Expression, ...], tuple[algebra.Polynomial, ...], float
    ],
    /,
    *,
    root_kind: str,
    exact: bool,
    physical_remainder: tuple[algebra.Expression, ...] | None = None,
    exact_source_controls: np.ndarray | None = None,
) -> tuple[_Corners, _Corners, Fraction, tuple[tuple[Fraction, ...], ...]]:
    """Whole-map original chart equality and target corner/source bindings."""
    from ..discretization._reference_cell import reference_cell_topology

    root_expressions, uv_polynomials, _ = expression
    corners = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in reference_cell_topology(element.cell_kind).vertices
    )
    refs = tuple(
        (algebra.evaluate(arguments[0], point), algebra.evaluate(arguments[1], point))
        for point in corners
    )
    domain = "simplex" if element.cell_kind == "triangle" else "box"
    constraints = (
        (
            *arguments,
            algebra.add(
                algebra.constant(1, 2),
                algebra.scale(algebra.sum_polynomials(arguments), -1),
            ),
        )
        if root_kind == "triangle"
        else (
            *arguments,
            *(
                algebra.add(algebra.constant(1, 2), algebra.scale(value, -1))
                for value in arguments
            ),
        )
    )
    if any(
        min(algebra.bernstein_coefficients(value, domain, 2)) < 0 for value in constraints
    ):
        raise ValueError(
            "A target reference image leaves its original source reference domain."
        )
    if (
        exact
        and physical_remainder is None
        and exact_source_controls is not None
        and controls.shape == exact_source_controls.shape
        and np.array_equal(
            np.asarray(controls, dtype=np.float64).view(np.uint64),
            np.asarray(exact_source_controls, dtype=np.float64).view(np.uint64),
        )
    ):
        actual_corners = tuple(
            tuple(algebra.expression_evaluate(value, point) for value in root_expressions)
            for point in refs
        )
        source_corners = tuple(
            (
                algebra.evaluate(uv_polynomials[0], point),
                algebra.evaluate(uv_polynomials[1], point),
            )
            for point in refs
        )
        return refs, source_corners, Fraction(0), actual_corners
    actual = algebra.coordinate_expressions(element, controls)
    if actual is None:
        raise ValueError("Target coordinate expression is unresolved.")
    restricted = tuple(
        algebra.expression_compose(value, arguments) for value in root_expressions
    )
    actual_corners = tuple(
        tuple(algebra.expression_evaluate(value, point) for value in actual)
        for point in corners
    )
    if physical_remainder is not None:
        difference = tuple(
            algebra.expression_add(a, algebra.expression_scale(r, -1))
            for a, r in zip(actual, restricted, strict=True)
        )
        if any(
            not _equal_expressions(a, b)
            for a, b in zip(difference, physical_remainder, strict=True)
        ):
            raise ValueError(
                "Full coefficient source theorem does not bind the actual whole physical remainder."
            )
        deviation = sum(
            (
                max(abs(min(coefficients)), abs(max(coefficients)))
                for value in physical_remainder
                for coefficients in (
                    algebra.expression_bernstein_coefficients(value, domain, 2),
                )
            ),
            Fraction(0),
        )
    elif exact:
        if any(
            not _equal_expressions(first, second)
            for first, second in zip(restricted, actual, strict=True)
        ):
            raise ValueError(
                "Target coordinate expression is not the exact original source restriction."
            )
        deviation = Fraction(0)
    else:
        if element.cell_kind != "triangle" or root_kind != "triangle":
            raise ValueError(
                "Unrestricted source ancestry requires its actual affine triangle premise."
            )
        deviation = max(
            sum(
                (
                    abs(a - algebra.expression_evaluate(r, point))
                    for a, r in zip(corner, restricted, strict=True)
                ),
                Fraction(0),
            )
            for corner, point in zip(actual_corners, corners, strict=True)
        )
    source_corners = tuple(
        (
            algebra.evaluate(uv_polynomials[0], point),
            algebra.evaluate(uv_polynomials[1], point),
        )
        for point in refs
    )
    return refs, source_corners, deviation, actual_corners


def _upper(value: Fraction, /) -> float:
    """Smallest-step binary64 upper bound of an exact rational quantity."""
    result = float(value)
    return result if Fraction(result) >= value else float(np.nextafter(result, np.inf))


class SurfaceNativeRestrictionBoundarySource(StrictModule, NonTrainableState):
    """Original source queries plus exact current source-chart restrictions."""

    root: MeshingDomainBoundarySource
    support: PreparedSurfaceSourceSupport
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    def __init__(
        self,
        root: MeshingDomainBoundarySource,
        support: PreparedSurfaceSourceSupport,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
    ) -> None:
        if not isinstance(root, MeshingDomainBoundarySource) or not isinstance(
            support, PreparedSurfaceSourceSupport
        ):
            raise TypeError(
                "Native restriction fidelity requires original domain queries and source support."
            )
        if not isinstance(support.original, SurfaceSourceRootAtlas):
            raise ValueError(
                "Exact native restriction fidelity requires an original source-definition atlas."
            )
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Native restriction fidelity requires the actual accepted candidate maps."
            )
        support.require_current()
        if root.domain.domain_id != support.domain.domain_id:
            raise ValueError(
                "Native restriction fidelity changes original source authority."
            )
        if set(root.patches) != set(np.asarray(support.root_patches).tolist()):
            raise ValueError(
                "Native restriction fidelity changes the selected original source patches."
            )
        self.root, self.support, self.mesh, self.geometry = root, support, mesh, geometry
        self.source_id, self.source_revision = root.source_id, root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.root.boundary_samples(maximum_samples)

    def require_current(self) -> None:
        self.support.require_current()
        original = self.support.original
        if not isinstance(original, SurfaceSourceRootAtlas):
            raise ValueError(
                "Native source fidelity lost its immutable original source atlas."
            )
        original.require_current()
        if (self.source_id, self.source_revision) != (
            self.root.source_id,
            self.root.source_revision,
        ):
            raise ValueError(
                "Native restriction source identity or revision has changed."
            )


def _source_uv_domain(domain: MeshingDomain, patch: int, /) -> PiecewiseLinearDomain:
    """Independent exact UV trim domain from original line definitions."""
    from ._mesh_certificates import PiecewiseLinearDomain
    from .brep._patches import LineCurve

    positions: dict[tuple[Fraction, Fraction], int] = {}
    points, facets = [], []
    for loop in domain.patches[patch].loops:
        for use in loop:
            if not isinstance(use, PatchCurveUse) or not isinstance(
                use.pcurve, LineCurve
            ):
                raise ValueError(
                    "Exact spline-quad UV coverage requires original affine trim coedges."
                )
            if (
                use.first_root is not None
                or use.last_root is not None
                or use.trim_curve is not None
            ):
                raise ValueError(
                    "Root-valued nonlinear trim authority needs its owning exact UV coverage theorem."
                )
            origin = tuple(
                Fraction(float(value)) for value in np.asarray(use.pcurve.origin)
            )
            direction = tuple(
                Fraction(float(value)) for value in np.asarray(use.pcurve.direction)
            )
            endpoints = tuple(
                (
                    origin[0] + Fraction(parameter) * direction[0],
                    origin[1] + Fraction(parameter) * direction[1],
                )
                for parameter in (use.first, use.last)
            )
            rows = []
            for point in endpoints:
                if any(Fraction(float(value)) != value for value in point):
                    raise ValueError(
                        "Original UV trim endpoint requires exact rational domain coefficients."
                    )
                if point not in positions:
                    positions[point] = len(points)
                    points.append(tuple(float(value) for value in point))
                rows.append(positions[point])
            facets.append(rows)
    return PiecewiseLinearDomain(
        np.asarray(points, dtype=np.float64),
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(facets), 1)),
        ("surface",),
        source_id=domain.source_id,
    )


def certify_original_surface_restriction_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    tolerance: float,
    order: int,
    limits: MeshCertificateLimits,
    /,
) -> SourceFidelityCertificate | None:
    """Exact original-source equality plus independent whole UV-domain coverage."""
    from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
    from ._mapped_reference_coverage import (
        _Chart,
        _exact_chart_mesh,
        _polynomial_reference_chart,
    )
    from ._mesh_certificates import (
        certify_domain_coverage,
        certify_global_embedding,
        SourceFidelityCertificate,
    )

    if not isinstance(source, SurfaceNativeRestrictionBoundarySource):
        return None
    source.require_current()
    binding.require(mesh, geometry)
    if source.mesh.mesh_id != mesh.mesh_id or cell_geometry_id(
        source.geometry
    ) != cell_geometry_id(geometry):
        raise ValueError(
            "Native source fidelity does not bind the actual published coordinate maps."
        )
    support = source.support
    proof = support.prove_native_restrictions(mesh, geometry)
    if np.any(proof.source_bounds != 0.0) or np.any(proof.restriction_deviations != 0.0):
        raise ValueError(
            "Exact source fidelity requires actual original-source equality, not a bounded fitted proxy."
        )
    packets = support._root_packets(mesh, None)
    elements, _, _ = geometry.resolve(mesh)
    record = geometry.restriction_source
    if record is None:
        raise ValueError(
            "Native source fidelity requires explicit original scientific root ancestry."
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    charts: dict[int, list[_Chart]] = {}
    for block, element in zip(mesh.blocks, elements, strict=True):
        for cell, vertices, parent in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(record.block_parent_cell_ids[block.name]),
            strict=True,
        ):
            root = support._root_element(int(parent))
            reference = _polynomial_reference_chart(element, root)
            if reference is None:
                raise ValueError(
                    "Native source fidelity lost the exact original source chart."
                )
            reference_element, _ = reference
            uv, patch, _, _ = packets[int(parent)]
            charts.setdefault(patch, []).append(
                _Chart(
                    block.name,
                    int(cell),
                    block.cell_kind,
                    tuple(int(value) for value in vertex_ids[vertices]),
                    None,
                    None,
                    reference_element,
                    uv,
                )
            )
    findings = []
    from .brep._patches import LineCurve

    for patch in source.root.patches:
        if patch not in charts:
            raise ValueError(
                "Native source fidelity leaves an original selected patch uncovered."
            )
        atlas = _exact_chart_mesh(charts[patch], 2)
        if atlas is None:
            raise ValueError(
                "Native source UV charts do not share exact current vertex images."
            )
        uv_mesh, uv_geometry = atlas
        uv_validity = certify_cell_geometry_validity(uv_geometry, mesh=uv_mesh)
        embedding = certify_global_embedding(
            uv_mesh, uv_geometry, uv_validity, limits=limits
        )
        affine_trim = all(
            isinstance(use, PatchCurveUse)
            and isinstance(use.pcurve, LineCurve)
            and use.first_root is None
            and use.last_root is None
            and use.trim_curve is None
            for loop in support.domain.patches[patch].loops
            for use in loop
        )
        if affine_trim:
            coverage = certify_domain_coverage(
                uv_mesh,
                uv_geometry,
                _source_uv_domain(support.domain, patch),
                np.zeros(uv_mesh.entity_set(2).count, dtype=np.int64),
                embedding=embedding,
                limits=limits,
            )
            findings.extend(coverage.findings)
        else:
            cover = source.root.boundary_chart_cover(
                limits.maximum_source_samples, _topology_only=True
            )
            if not cover.complete or cover.semantics != "certified":
                raise ValueError(
                    "Curved original trims lack certified retained chart coverage."
                )
            records = [
                value for value in source.root.chart_triangulations if value[0] == patch
            ]
            if len(records) != 1:
                raise ValueError(
                    "Curved original trim coverage lacks one exact patch chart."
                )
            _, root_charts, _, root_cells, *_ = records[0]
            current_cells = np.concatenate(
                [np.asarray(block.vertices, dtype=np.int64) for block in uv_mesh.blocks]
            )

            def triangle_tokens(
                values: np.ndarray, cells: np.ndarray, /
            ) -> tuple[tuple[tuple[int, ...], ...], ...]:
                rows = []
                for triangle in values[cells]:
                    points = sorted(
                        tuple(np.asarray(point, dtype=np.float64).view(np.uint64))
                        for point in triangle
                    )
                    rows.append(tuple(points))
                return tuple(sorted(rows))

            if triangle_tokens(
                np.asarray(uv_mesh.coordinates), current_cells
            ) != triangle_tokens(np.asarray(root_charts), np.asarray(root_cells)):
                raise ValueError(
                    "Curved original trim cover differs from exact restriction charts."
                )
            findings.extend(cover.findings)
        findings.extend(embedding.findings)
    return SourceFidelityCertificate(
        binding,
        tuple(findings),
        tolerance=tolerance,
        semantics=("certified", "certified"),
        mesh_to_source=(0.0, 0.0),
        source_to_mesh=(0.0, 0.0),
        sample_order=order,
        sample_counts=(mesh.entity_set(2).count, support.root_cell_ids.shape[0]),
    )


__all__ = [
    "SurfaceSourceCharts",
    "PreparedSurfaceSourceSupport",
    "SurfaceSourceRootAtlas",
    "SurfaceNativeRestrictionBoundarySource",
    "SurfaceCellAncestry",
    "SurfaceRestrictionRows",
    "SurfaceSourceReceipts",
    "curve_owner",
    "prepare_surface_source_root_atlas",
    "certify_original_surface_restriction_fidelity",
    "restrict_surface_source_atlas_geometry",
]

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bind surface coordinate maps to declared source charts.

Collective preparation joins scientific root IDs on device. Local proofs use
addressable root packets, source expressions, and coordinate restrictions.
Coordinate coincidence does not establish root identity."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import assert_never, cast, NamedTuple, Protocol, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax._fingerprint as fingerprint_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.discretization._reference_cell as reference_cell_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_reference_coverage as mapped_reference_coverage_owner
import phydrax.geometry._meshing_domain as meshing_domain_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry.brep._patches as patches_owner
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._surface_boundary_sources as surface_boundary_sources_owner
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryDistance,
    SourceBoundaryQuery,
    SourceBoundarySamples,
)

from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._host_storage import immutable_host_array
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import _coordinate_enclosure as algebra, CellBlock, CellMesh
from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
    SplineCellGeometryElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization.fem._reference_types import FiniteElementSpec
from ..linalg._small_batched import exact_cofactor_determinant_2
from ..typing import checked
from ._domain_numerics import full_sphere_frame, sphere_triangle_distance_bounds
from ._mesh_certificates import (
    MeshCertificateBinding,
)
from ._rational import rational_vector


if TYPE_CHECKING:
    import phydrax.meshing._result as result_owner
    from phydrax.meshing._association_state import GeometryAssociation

    from ..meshing._initial_certification import InitialCollectiveMeshEvidence
    from ..meshing._result import (
        CellMeshingResult,
    )

type _Corners = tuple[tuple[Fraction, Fraction], ...]


class SurfaceRestrictionRows(NamedTuple):
    """Fresh whole-map restrictions and their source chart/source identities."""

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
    """Lineage-proved scientific root of one serial cell and its rational root-reference corners."""

    root_cell_id: int
    reference_corners: _Corners


class _MeshingResultCapabilities(Protocol):
    """Nominal result type and source-integrity request capability."""

    @property
    def CellMeshingResult(self) -> type[result_owner.CellMeshingResult]: ...

    @staticmethod
    def require_original_meshing_source(
        result: result_owner.CellMeshingResult, /
    ) -> result_owner.CellMeshingResult | InitialCollectiveMeshEvidence: ...


def _meshing_result_capabilities() -> _MeshingResultCapabilities:
    # Result publication composes prepared source support; resolve this upper-layer owner once per request.
    from ..meshing import _result

    return _result


def _collective_evidence_type() -> type[InitialCollectiveMeshEvidence]:
    # Initial collective evidence composes source-support geometry; resolve its nominal type after those declarations load.
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence

    return InitialCollectiveMeshEvidence


@checked
def curve_owner(
    domain: meshing_domain_owner.MeshingDomain, curve: int, /
) -> meshing_domain_owner.PatchCurveUse:
    """Return the coedge that defines a source curve parameter interval."""
    patch, loop, position = map(int, domain.curve_owners[curve])
    match domain.patches[patch].loops[loop][position]:
        case meshing_domain_owner.PatchCurveUse() as owner:
            return owner
        case meshing_domain_owner.PatchPoleUse():
            raise ValueError(
                "SurfaceSourceSupport.curve must name a coedge with a parameter interval; got pole side"
            )
        case invalid:
            assert_never(invalid)


def _restriction(
    element: CellGeometryElement, *, budget: CoordinateEnclosureBudget
) -> tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]:
    """Return the polynomial reference chart after proving positive orientation."""

    scalar = cell_geometry_owner.require_scalar_coordinate_element(
        element, "Surface source reference chart"
    )
    base, arguments = (
        coordinate_enclosure_owner.coordinate_partition_unity_reference_chain(
            scalar, budget=budget
        )
    )
    if scalar.topological_dimension != 2:
        raise ValueError(
            "SurfaceSourceSupport.chart must have two reference dimensions; got different dimension"
        )
    if any(isinstance(value, algebra.RationalPolynomial) for value in arguments):
        raise ValueError(
            "SurfaceSourceSupport.chart must have polynomial reference restrictions; got rational restriction"
        )
    polynomial_arguments = cast(tuple[algebra.Polynomial, ...], arguments)
    domain = "simplex" if element.cell_kind == "triangle" else "box"
    determinant = algebra.determinant(
        tuple(
            tuple(algebra.derivative(value, axis, budget=budget) for axis in range(2))
            for value in polynomial_arguments
        ),
        budget=budget,
    )
    if algebra.polynomial_bounds(determinant, domain, 2, budget=budget)[0] <= 0:
        raise ValueError(
            "SurfaceSourceSupport.chart must have positive resolved reference orientation; got nonpositive orientation bound"
        )
    return base, polynomial_arguments


def _require_orientation(matrix: tuple[tuple[Fraction, ...], ...], /) -> None:
    determinant = exact_cofactor_determinant_2(
        (matrix[0][0], matrix[0][1]), (matrix[1][0], matrix[1][1])
    )
    if determinant <= 0:
        raise ValueError(
            "SurfaceSourceSupport.restriction must have positive reference orientation; got nonpositive determinant"
        )


def _corner_restriction(
    corners: _Corners, /
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    """Return the affine reference map from a unit triangle to the supplied corners."""
    origin = corners[0]
    matrix = tuple(
        tuple(corners[j + 1][i] - origin[i] for j in range(2)) for i in range(2)
    )
    _require_orientation(matrix)
    return matrix, origin


def _arguments(
    matrix: tuple[tuple[Fraction, ...], ...],
    offset: tuple[Fraction, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Polynomial, ...]:
    variables = algebra.axes(2, budget=budget)
    return tuple(
        algebra.add(
            algebra.constant(offset[i], 2, budget=budget),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(variables[j], matrix[i][j], budget=budget)
                    for j in range(2)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for i in range(2)
    )


def _source_corner(
    domain: meshing_domain_owner.MeshingDomain,
    patch: int,
    dimension: int,
    index: int,
    parameters: np.ndarray,
) -> np.ndarray:
    match dimension:
        case 2:
            if index != patch:
                raise ValueError(
                    "SurfaceSourceSupport.corner.patch must match the root patch; got different patch"
                )
            return np.asarray(parameters[:2], dtype=np.float64)
        case 1:
            uses = [
                use
                for loop in domain.patches[patch].loops
                for use in loop
                if isinstance(use, meshing_domain_owner.PatchCurveUse)
                and use.curve == index
            ]
            if not uses:
                raise ValueError(
                    "SurfaceSourceSupport.corner.curve must be incident on the root patch; got absent coedge"
                )
            value = float(parameters[0])
            return np.stack([use.charts(np.asarray((value,)))[0] for use in uses])
        case 0:
            return np.asarray(domain.corner_points[index])
        case _:
            raise ValueError(
                "SurfaceSourceSupport.dimension must name a corner, curve, or patch stratum; got unsupported stratum"
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
            raise ValueError(
                "SurfaceSourceReceipts.partition must be process-addressable; got nonaddressable partition"
            )
        packets.append((name, packet))
    return tuple(packets)


@checked
def prepare_surface_source_root_atlas(
    domain: meshing_domain_owner.MeshingDomain,
    patches: tuple[int, ...],
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget | None = None,
) -> SurfaceSourceRootAtlas:
    """Build spline knot-span source charts before meshing."""

    with embedding_state_owner.coordinate_certificate_request(
        embedding_state_owner.MeshCertificateLimits(), budget=budget
    ) as budget:
        span_rows = []
        work = 0
        for patch in patches:
            surface = domain.patches[patch].surface
            if not isinstance(surface, patches_owner.BSplineSurfacePatch):
                raise ValueError(
                    "SurfaceSourceRootAtlas.patch must be a rational B-spline surface; got different surface family"
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
                "SurfaceSourceRootAtlas.work must fit maximum_support_queries and include spans; got empty or oversized preparation"
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
            if not isinstance(surface, patches_owner.BSplineSurfacePatch):
                raise ValueError(
                    "SurfaceSourceRootAtlas.patch must remain a B-spline surface; got different surface family"
                )
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
            values = algebra.coordinate_corner_images(element, controls, budget=budget)
            if values is None:
                raise ValueError(
                    "SurfaceSourceRootAtlas.corners must resolve source corner images; got unresolved expressions"
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
            budget=budget,
        )


class _AtlasSpanDeclarations(StrictModule):
    spans: dict[int, set[tuple[int, int]]]
    surfaces: dict[int, patches_owner.BSplineSurfacePatch]


def _validate_atlas_span(
    domain: meshing_domain_owner.MeshingDomain,
    patch: int,
    element: SplineCellGeometryElement,
    controls: np.ndarray,
    parameters: np.ndarray,
) -> patches_owner.BSplineSurfacePatch:
    surface = domain.patches[patch].surface
    if not isinstance(surface, patches_owner.BSplineSurfacePatch):
        raise ValueError(
            "SurfaceSourceRootAtlas.patch must match the spline root surface family; got different surface family"
        )
    if (element.source_id, element.source_revision) != (
        domain.source_id,
        domain.source_revision,
    ):
        raise ValueError(
            "SurfaceSourceRootAtlas.source must retain domain source identity and revision; got different identity or revision"
        )
    for actual, expected in (
        (element.u_knots, surface.u_knots),
        (element.v_knots, surface.v_knots),
        (element.weights, surface.weights),
        (
            controls,
            np.asarray(surface.control_points).reshape(-1, 3),
        ),
    ):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise ValueError(
                "SurfaceSourceRootAtlas.controls must match domain knots, weights, and control coefficients; got different source coefficients"
            )
    if (element.u_degree, element.v_degree) != (
        surface.u_degree,
        surface.v_degree,
    ):
        raise ValueError(
            "SurfaceSourceRootAtlas.degree must match the patch reference space; got different degrees"
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
    if not np.array_equal(parameters, expected_uv):
        raise ValueError(
            "SurfaceSourceRootAtlas.root_parameters must match the knot-span corners; got different UV corners"
        )
    return surface


def _validate_atlas_roots(
    domain: meshing_domain_owner.MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    parameters: np.ndarray,
    patches: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AtlasSpanDeclarations:
    elements, routes, coordinates = geometry.resolve(mesh)
    controls = np.asarray(coordinates, dtype=np.float64)
    declared: dict[int, set[tuple[int, int]]] = {}
    surfaces: dict[int, patches_owner.BSplineSurfacePatch] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != "quadrilateral" or not isinstance(
            element, SplineCellGeometryElement
        ):
            raise ValueError(
                "SurfaceSourceRootAtlas.geometry must contain unrestricted spline quadrilateral maps; got incompatible root map"
            )
        for row, dofs in enumerate(np.asarray(route)):
            patch = int(patches[cursor])
            surface = _validate_atlas_span(
                domain, patch, element, controls[dofs], parameters[cursor]
            )
            surfaces[patch] = surface
            if element.span_indices in declared.setdefault(patch, set()):
                raise ValueError(
                    "SurfaceSourceRootAtlas.spans must occur once per patch; got duplicate knot span"
                )
            declared[patch].add(element.span_indices)
            images = algebra.coordinate_corner_images(
                element, controls[dofs], budget=budget
            )
            if images is None or not np.array_equal(
                np.asarray(mesh.coordinates)[np.asarray(block.vertices)[row]],
                np.stack(tuple(algebra.rounded_point(value) for value in images)),
            ):
                raise ValueError(
                    "SurfaceSourceRootAtlas.mesh.coordinates must match rounded source corner images; got different carrier coordinates"
                )
            cursor += 1
    return _AtlasSpanDeclarations(declared, surfaces)


def _require_full_atlas_spans(declarations: _AtlasSpanDeclarations) -> None:
    for patch, spans in declarations.spans.items():
        surface = declarations.surfaces[patch]
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
                "SurfaceSourceRootAtlas.spans must cover every nonempty patch knot span; got different span set"
            )


class SurfaceSourceRootAtlas(StrictModule, NonTrainableState):
    """Retain spline knot-span charts with their patch control grids and weights."""

    domain: meshing_domain_owner.MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    root_parameters: Array
    root_patches: Array
    coordinate_contract: SpatialCoordinateContract
    atlas_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: meshing_domain_owner.MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root_parameters: Array,
        root_patches: Array,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        with embedding_state_owner.coordinate_certificate_request(
            embedding_state_owner.MeshCertificateLimits(), budget=budget
        ) as budget:
            if (
                mesh.storage is not None
                or mesh.topological_dimension != 2
                or mesh.ambient_dimension != 3
            ):
                raise ValueError(
                    "SurfaceSourceRootAtlas.mesh must be a serial surface in three-space; got incompatible carrier"
                )
            if not coordinate_contract.is_orthonormal_cartesian:
                raise ValueError(
                    "SurfaceSourceRootAtlas.coordinate_contract must be orthonormal Cartesian; got incompatible frame"
                )
            parameters, patches = np.asarray(root_parameters), np.asarray(root_patches)
            count = mesh.entity_set(2).count
            if parameters.shape != (count, 4, 2) or patches.shape != (count,):
                raise ValueError(
                    "SurfaceSourceRootAtlas.root_parameters must have four UV corners and one patch per root; got misaligned shapes"
                )
            if (
                not np.all(np.isfinite(parameters))
                or patches.dtype.kind not in "iu"
                or np.any(patches < 0)
                or np.any(patches >= len(domain.patches))
            ):
                raise ValueError(
                    "SurfaceSourceRootAtlas.root_parameters must be finite with declared integer patch indices; got invalid parameters or patches"
                )
            declarations = _validate_atlas_roots(
                domain, mesh, geometry, parameters, patches, budget=budget
            )
            _require_full_atlas_spans(declarations)
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
                "expressions": [
                    meshing_domain_owner.patch_identity(patch)
                    for patch in self.domain.patches
                ],
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
                "SurfaceSourceRootAtlas.atlas_id must match source definition and root data; got changed content"
            )


class _SourceRoot(StrictModule):
    cell_id: int = eqx.field(static=True)
    element: ScalarCoordinateElement
    dofs: np.ndarray
    vertex_ids: np.ndarray
    patch: int = eqx.field(static=True)
    parameters: np.ndarray


class _AtlasRoots(StrictModule):
    roots: tuple[_SourceRoot, ...]
    coefficients: Array | np.ndarray


class _AtlasCellRestriction(StrictModule):
    element: CellGeometryElement
    dofs: np.ndarray
    root_cell_id: int = eqx.field(static=True)
    root_vertex_ids: np.ndarray


def _atlas_roots(atlas: SurfaceSourceRootAtlas) -> _AtlasRoots:
    roots, root_routes, coefficients = atlas.geometry.resolve(atlas.mesh)
    source_roots = []
    cursor = 0
    for block, root, route in zip(atlas.mesh.blocks, roots, root_routes, strict=True):
        scalar = cell_geometry_owner.require_scalar_coordinate_element(
            root, "Original source chart"
        )
        for identifier, vertices, dofs in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(route),
            strict=True,
        ):
            source_roots.append(
                _SourceRoot(
                    int(identifier),
                    scalar,
                    dofs,
                    np.asarray(atlas.mesh.vertex_global_ids)[vertices],
                    int(np.asarray(atlas.root_patches)[cursor]),
                    np.asarray(atlas.root_parameters)[cursor],
                )
            )
            cursor += 1
    return _AtlasRoots(tuple(source_roots), coefficients)


def _restrict_atlas_cell(
    roots: tuple[_SourceRoot, ...],
    patch: int,
    uv: np.ndarray,
    triangle: FiniteElementSpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AtlasCellRestriction:
    # The affine UV image is the entire triangle's convex hull, so all
    # three corner inequalities prove containment in the source span.
    matches = [
        root
        for root in roots
        if root.patch == patch
        and np.all(uv >= np.min(root.parameters, axis=0))
        and np.all(uv <= np.max(root.parameters, axis=0))
    ]
    if len(matches) != 1:
        raise ValueError(
            "SurfaceSourceSupport.cell_parameters must lie in one patch knot span; got cross-span chart"
        )
    selected = matches[0]
    root_id, root, dofs, corners, bounds = (
        selected.cell_id,
        selected.element,
        selected.dofs,
        selected.vertex_ids,
        selected.parameters,
    )
    low = rational_vector(bounds[0])
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
    element = cell_geometry_owner.PolynomialComposedCellGeometryElement(
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
        budget=budget,
    )
    return _AtlasCellRestriction(element, dofs, root_id, corners)


def _assemble_atlas_restriction(
    atlas: SurfaceSourceRootAtlas,
    target: CellMesh,
    coordinates: np.ndarray,
    groups: dict[str, list[tuple[int, np.ndarray, np.ndarray, int, np.ndarray]]],
    elements: dict[str, CellGeometryElement],
    coefficients: Array | np.ndarray,
    canonical_blocks: bool,
) -> tuple[CellMesh, CellGeometrySpec]:
    blocks, routes, parents, parent_vertices = [], {}, {}, {}
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

    edge_ids_by_key = {
        tuple(map(int, key)): int(identifier)
        for key, identifier in zip(
            entity_keys_owner.entity_keys(target, 1),
            np.asarray(target.entity_set(1).entity_ids),
            strict=True,
        )
    }
    seen_edges: set[tuple[int, int]] = set()
    regrouped_edge_ids: list[int] = []
    global_vertices = np.asarray(target.vertex_global_ids)
    # Polygonal connectivity numbers edges by first appearance in the measured
    # cell/local-side scan. Bind that order to the source incidence SCI keys.
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
    origin = cell_geometry_owner.CellGeometryRestrictionSource(
        cell_geometry_id(atlas.geometry), atlas.mesh.topology_id, parents, parent_vertices
    )
    return mesh, CellGeometrySpec(
        elements, routes, coefficients, restriction_source=origin
    )


@checked
def restrict_surface_source_atlas_geometry(
    atlas: SurfaceSourceRootAtlas,
    target: CellMesh,
    cell_patches: np.ndarray,
    cell_parameters: np.ndarray,
    /,
    *,
    maximum_support_queries: int,
    canonical_blocks: bool = False,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec]:
    """Restrict target triangles to spline knot spans and their control banks.

    Target scientific IDs and numeric revision are retained. Coordinate-element
    blocks are regrouped by their retained chart sources."""

    atlas.require_current()
    if not isinstance(canonical_blocks, bool):
        raise TypeError(
            "SurfaceSourceSupport.canonical_blocks must be bool; got different input kind"
        )
    if (
        target.storage is not None
        or target.topological_dimension != 2
        or target.ambient_dimension != 3
    ):
        raise ValueError(
            "SurfaceSourceSupport.target must be a serial surface in three-space; got incompatible carrier"
        )
    if any(block.cell_kind != "triangle" for block in target.blocks):
        raise ValueError(
            "SurfaceSourceSupport.target.blocks must contain triangles; got different cell kind"
        )
    count = target.entity_set(2).count
    patches, parameters = (
        np.asarray(cell_patches),
        np.asarray(cell_parameters, dtype=np.float64),
    )
    if patches.shape != (count,) or parameters.shape != (count, 3, 2):
        raise ValueError(
            "SurfaceSourceSupport.cell_parameters must have one patch and three UV corners per target cell; got misaligned shapes"
        )
    if count * 12 > maximum_support_queries:
        raise ValueError(
            "SurfaceSourceSupport.work must fit maximum_support_queries before chart preparation; got exceeded query capacity"
        )
    prepared = _atlas_roots(atlas)
    coordinates = np.asarray(target.coordinates).copy()
    positions: dict[int, tuple[Fraction, ...]] = {}
    elements: dict[str, CellGeometryElement] = {}
    groups: dict[str, list[tuple[int, np.ndarray, np.ndarray, int, np.ndarray]]] = {}
    triangle = coordinate_lagrange_element("triangle", 1)
    cursor = 0
    for block in target.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            patch, uv = int(patches[cursor]), parameters[cursor]
            restriction = _restrict_atlas_cell(
                prepared.roots, patch, uv, triangle, budget=budget
            )
            element, dofs = restriction.element, restriction.dofs
            root_id, corners = restriction.root_cell_id, restriction.root_vertex_ids
            images = algebra.coordinate_corner_images(
                element, np.asarray(prepared.coefficients)[dofs], budget=budget
            )
            if images is None:
                raise ValueError(
                    "SurfaceSourceSupport.target.geometry must resolve coordinate expressions; got unresolved source expressions"
                )
            for vertex, point in zip(vertices.tolist(), images, strict=True):
                point = tuple(
                    rational_coefficient(value, role="SurfaceSourceSupport.corner_image")
                    for value in point
                )
                if vertex in positions and positions[vertex] != point:
                    raise ValueError(
                        "SurfaceSourceSupport.target.vertices must have matching images across charts; got different shared vertex images"
                    )
                positions[vertex] = point
                coordinates[vertex] = algebra.rounded_point(point)
            name = f"source-chart/{element.element_id}"
            if name not in groups:
                groups[name] = []
                elements[name] = element
            groups[name].append((int(identifier), vertices, dofs, root_id, corners))
            cursor += 1
    return _assemble_atlas_restriction(
        atlas,
        target,
        coordinates,
        groups,
        elements,
        prepared.coefficients,
        canonical_blocks,
    )


class SurfaceSourceReceipts(StrictModule, NonTrainableState):
    """Current collectively joined source-root receipts, lowered on their owner."""

    support: PreparedSurfaceSourceSupport
    geometry_projection: CellGeometryStorageProjection
    arrays: tuple[tuple[str, Array], ...]
    receipt_id: str = eqx.field(static=True)
    local_receipt_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        support: PreparedSurfaceSourceSupport,
        logical_arrays: tuple[tuple[str, Array], ...],
        geometry_projection: CellGeometryStorageProjection,
        /,
    ) -> None:
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
                    "SurfaceSourceReceipts.queries must fit maximum_support_queries before root lookup; got exceeded query capacity"
                )
            position = jnp.searchsorted(support.root_cell_ids, jnp.maximum(parents, 0))
            clipped = jnp.minimum(position, support.root_cell_ids.shape[0] - 1)
            matched = (position < support.root_cell_ids.shape[0]) & (
                support.root_cell_ids[clipped] == parents
            )
            if not bool(jax.device_get(jnp.all(matched | ~active))):
                raise ValueError(
                    "SurfaceSourceReceipts.parents must identify declared source roots; got undeclared parent root"
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
            raise ValueError(
                "SurfaceSourceReceipts.arrays must contain source-root queries; got empty root query bank"
            )
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
                "SurfaceSourceReceipts.partition_index must lie in the geometry placement; got out-of-range partition"
            )
        packets = _addressable_receipt_packets(self.arrays, partition_index)
        expected = dict(self.local_receipt_ids).get(partition_index)
        if (
            expected is None
            or logical_array_value_collection_digest(dict(packets)) != expected
        ):
            raise ValueError(
                "SurfaceSourceReceipts.content must match the prepared partition digest; got changed numerical content"
            )
        return packets


class _ChartAssociations(StrictModule):
    by_dimension: dict[int, association_state_owner.GeometryAssociation]


def _source_chart_associations(
    domain: meshing_domain_owner.MeshingDomain,
    mesh: CellMesh,
    patches: np.ndarray,
    associations: tuple[association_state_owner.GeometryAssociation, ...],
) -> _ChartAssociations:
    if not isinstance(associations, tuple) or any(
        not isinstance(value, association_state_owner.GeometryAssociation)
        for value in associations
    ):
        raise TypeError(
            "SurfaceSourceCharts.associations must be a tuple of GeometryAssociation records; got unsupported record kinds"
        )
    by_dimension = {
        association_state_owner.target_dimension(mesh, value): value
        for value in associations
    }
    expected_kind = (
        association_state_owner.GeometryAssociationKind.BREP
        if domain.source_kinds == ("vertex", "edge", "face")
        else association_state_owner.GeometryAssociationKind.SURFACE
    )
    if set(by_dimension) != {0, 1, 2} or any(
        value.association_kind is not expected_kind or not value.complete
        for value in by_dimension.values()
    ):
        raise ValueError(
            "SurfaceSourceCharts.associations must cover each stratum with the declared source kind; got incomplete or incompatible associations"
        )
    ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if not np.all(ids[1:] > ids[:-1]):
        raise ValueError(
            "SurfaceSourceCharts.cell_ids must be strictly increasing; got unordered scientific IDs"
        )
    cell = by_dimension[2]
    rows = cell.target_rows(ids)
    cell_dimensions, cell_indices = jax.device_get(
        (cell.source_dimensions, cell.source_indices)
    )
    declared_patches = np.asarray(
        [
            PreparedSurfaceSourceSupport.domain_row(
                domain,
                int(cell_dimensions[row]),
                int(cell_indices[row]),
                cell.source_entities.occurrence_path(row),
            )
            for row in rows
        ],
        dtype=np.int64,
    )
    if not np.array_equal(patches, declared_patches):
        raise ValueError(
            "SurfaceSourceCharts.root_patches must match cell source associations; got different patch rows"
        )
    return _ChartAssociations(by_dimension)


def _source_chart_corners(mesh: CellMesh, geometry: CellGeometrySpec) -> np.ndarray:
    expected_basis = canonical_fingerprint(
        algebra.coordinate_source_signature(coordinate_lagrange_element("triangle", 1))
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
                "SurfaceSourceCharts.geometry must contain direct or composed P1 triangle charts with aligned routes; got incompatible chart or route"
            )
        all_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        corners.append(np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)])
    inverse = np.argsort(
        association_state_owner.entity_rows(mesh, 2, np.concatenate(all_ids)),
        kind="stable",
    )
    return np.concatenate(corners)[inverse]


class SurfaceSourceCharts(StrictModule, NonTrainableState):
    """Retain triangle charts and scientific associations for a serial surface source."""

    domain: meshing_domain_owner.MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    root_parameters: np.ndarray
    root_patches: np.ndarray
    associations: tuple[GeometryAssociation, ...]
    boundary_source: domain_boundary_owner.MeshingDomainBoundarySource
    maximum_support_queries: int = eqx.field(static=True)
    source_chart_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: meshing_domain_owner.MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        root_parameters: np.ndarray,
        root_patches: np.ndarray,
        associations: tuple[association_state_owner.GeometryAssociation, ...],
        boundary_source: domain_boundary_owner.MeshingDomainBoundarySource,
        /,
        *,
        maximum_support_queries: int,
    ) -> None:

        if (
            mesh.storage is not None
            or mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
        ):
            raise ValueError(
                "SurfaceSourceCharts.mesh must be a serial surface in three-space; got incompatible carrier"
            )
        if not coordinate_contract.is_orthonormal_cartesian:
            raise ValueError(
                "SurfaceSourceCharts.coordinate_contract must be orthonormal Cartesian; got incompatible frame"
            )
        if (
            isinstance(maximum_support_queries, bool)
            or not isinstance(maximum_support_queries, int)
            or maximum_support_queries < 1
        ):
            raise ValueError(
                "SurfaceSourceCharts.maximum_support_queries must be a positive integer; got invalid query capacity"
            )
        count = mesh.entity_set(2).count
        if 9 * count > maximum_support_queries:
            raise ValueError(
                "SurfaceSourceCharts.work must fit maximum_support_queries before root expansion; got exceeded query capacity"
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
                "SurfaceSourceCharts.root_parameters must be finite float64 corner parameters with int64 patch rows in cell-ID order; got invalid dtype, shape, or values"
            )
        if np.any(patches < 0) or np.any(patches >= len(domain.patches)):
            raise ValueError(
                "SurfaceSourceCharts.root_patches must identify declared patches; got absent patch"
            )
        if boundary_source.domain.content_domain_id() != domain.content_domain_id():
            raise ValueError(
                "SurfaceSourceCharts.boundary_source must bind the domain content; got different source content"
            )
        if set(patches.tolist()) != set(boundary_source.patches):
            raise ValueError(
                "SurfaceSourceCharts.root_patches must select the boundary-source patches; got different patch set"
            )
        incidence = _source_chart_associations(domain, mesh, patches, associations)
        corners = _source_chart_corners(mesh, geometry)
        PreparedSurfaceSourceSupport._validate_serial_parameters(
            domain, patches, parameters, incidence.by_dimension[0], corners
        )
        self.domain, self.mesh, self.geometry = domain, mesh, geometry
        self.coordinate_contract = coordinate_contract
        self.root_parameters = immutable_host_array(parameters)
        self.root_patches = immutable_host_array(patches)
        self.associations, self.boundary_source = tuple(associations), boundary_source
        self.maximum_support_queries = maximum_support_queries
        self.source_chart_id = self._identity()

    def _identity(self) -> str:

        return canonical_fingerprint(
            {
                "kind": "native-original-surface-charts",
                "source": self.domain.domain_id,
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "geometry": cell_geometry_id(self.geometry),
                "geometry_layout": self.geometry.geometry_layout_id,
                "mesh_content": fingerprint_owner.array_tree_fingerprint(
                    (
                        self.mesh.coordinates,
                        self.mesh.vertex_global_ids,
                        tuple(
                            (block.vertices, block.global_ids)
                            for block in self.mesh.blocks
                        ),
                    )
                ),
                "charts": fingerprint_owner.array_tree_fingerprint(
                    (self.root_parameters, self.root_patches)
                ),
                "associations": [
                    (
                        value.association_id,
                        value.source_id,
                        value.source_revision,
                        value.source_entities.entity_ids(),
                        value.source_entities.occurrence_paths(),
                    )
                    for value in self.associations
                ],
                "association_arrays": fingerprint_owner.array_tree_fingerprint(
                    self.associations
                ),
                "boundary": self.boundary_source.source_scope_id,
                "boundary_source": self.boundary_source.domain.domain_id,
                "boundary_occurrences": self.boundary_source.boundary_occurrence_id,
                "boundary_charts": fingerprint_owner.array_tree_fingerprint(
                    self.boundary_source.chart_triangulations
                ),
                "maximum_queries": self.maximum_support_queries,
            }
        )

    def require_current(self) -> None:
        if self._identity() != self.source_chart_id:
            raise ValueError(
                "SurfaceSourceCharts.source_chart_id must match chart, incidence, and coefficient content; got changed content"
            )

    @checked
    def require_root(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        self.require_current()
        if mesh.topology_id != self.mesh.topology_id or cell_geometry_id(
            geometry
        ) != cell_geometry_id(self.geometry):
            raise ValueError(
                "SurfaceSourceCharts.root must match bound topology and geometry IDs; got different root carrier"
            )

    @checked
    def require_bound(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SurfaceBoundSource,
        /,
    ) -> None:
        """Authenticate retained source charts against a publication's live map."""
        self.require_current()
        match source:
            case SurfaceNativeRestrictionBoundarySource():
                source.require_current()
                self.require_root(mesh, geometry)
                if source.mesh.mesh_id != mesh.mesh_id or cell_geometry_id(
                    source.geometry
                ) != cell_geometry_id(geometry):
                    raise ValueError(
                        "SurfaceSourceCharts.source must bind the live mesh and geometry; got different live map"
                    )
                return
            case domain_boundary_owner.MeshingDomainBoundarySource():
                self.require_root(mesh, geometry)

                if (
                    source.domain.content_domain_id() != self.domain.content_domain_id()
                    or source.source_scope_id != self.boundary_source.source_scope_id
                    or source.boundary_occurrence_id != self.boundary_source.boundary_occurrence_id
                    or fingerprint_owner.array_tree_fingerprint(
                        source.chart_triangulations
                    )
                    != fingerprint_owner.array_tree_fingerprint(
                        self.boundary_source.chart_triangulations
                    )
                ):
                    raise ValueError(
                        "SurfaceSourceCharts.source must retain domain content, scope, and chart triangulations; got different boundary content"
                    )
                return
            case (
                surface_boundary_sources_owner.SurfaceChartBoundarySource()
                | surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource()
            ):
                self._require_descendant_bound(mesh, geometry, source)
            case invalid:
                assert_never(invalid)

    def _require_descendant_bound(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: surface_boundary_sources_owner.SurfaceChartBoundarySource
        | surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource,
        /,
    ) -> None:
        source.support.require_current()
        original = source.support.original
        match original:
            case SurfaceSourceRootAtlas():
                match source:
                    case surface_boundary_sources_owner.SurfaceChartBoundarySource():
                        original.require_current()
                        deformation = source.deformation
                        self.require_root(
                            deformation.source_mesh, deformation.source_geometry
                        )
                        source.require_target_binding(mesh, geometry)
                        return
                    case surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource():
                        raise ValueError(
                            "SurfaceSourceCharts.source must contain a chart deformation for spline-atlas descendants; got subdivision-only source"
                        )
                    case invalid:
                        assert_never(invalid)
            case _:
                result_type = _meshing_result_capabilities().CellMeshingResult
                match original:
                    case result_type():
                        charts = original.surface_source
                    case _:
                        raise ValueError(
                            "SurfaceSourceCharts.source.support.original must be an accepted cell meshing result; got different root carrier"
                        )
        match charts:
            case SurfaceSourceCharts():
                if charts.source_chart_id != self.source_chart_id:
                    raise ValueError(
                        "SurfaceSourceCharts.source.support.original.surface_source must retain the bound source_chart_id; got absent or different charts"
                    )
            case _:
                raise ValueError(
                    "SurfaceSourceCharts.source.support.original.surface_source must retain the bound source_chart_id; got absent or different charts"
                )
        charts.require_root(original.mesh, original.geometry)
        match source:
            case surface_boundary_sources_owner.SurfaceChartBoundarySource():
                source.require_target_binding(mesh, geometry)
            case surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource():
                if (
                    not source.targets
                    or source.targets[-1].topology_id != mesh.topology_id
                    or cell_geometry_id(source.geometry) != cell_geometry_id(geometry)
                ):
                    raise ValueError(
                        "SurfaceSourceCharts.source must bind the final subdivision target and geometry; got different target map"
                    )
            case invalid:
                assert_never(invalid)


class _RootBinding(StrictModule):
    """Prepared root banks and the evidence that binds their scientific axes."""

    original: CellMeshingResult | SurfaceSourceRootAtlas | InitialCollectiveMeshEvidence
    count: int = eqx.field(static=True)
    ids: Array
    parameters: Array
    patches: Array
    corners: Array
    coordinates: Array
    basis_signature: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    corner_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)


class _CellPullback(StrictModule):
    base: CellGeometryElement
    arguments: tuple[algebra.Polynomial, ...]
    remainder: tuple[algebra.Expression, ...] | None


class _SurfaceCellProof(StrictModule):
    cell_id: int = eqx.field(static=True)
    root_cell_id: int = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    vertex_ids: tuple[int, ...] = eqx.field(static=True)
    reference_corners: _Corners
    source_corners: _Corners
    bound: float = eqx.field(static=True)
    corner_error: float = eqx.field(static=True)
    deviation: float = eqx.field(static=True)


class _SurfaceRootFrame(StrictModule):
    domain: meshing_domain_owner.MeshingDomain
    coordinate_contract: SpatialCoordinateContract
    basis_signature: str = eqx.field(static=True)


@checked
def _prepare_surface_root_frame(
    domain: meshing_domain_owner.MeshingDomain,
    mesh: CellMesh,
    coordinate_contract: SpatialCoordinateContract,
    maximum_support_queries: int,
) -> _SurfaceRootFrame:
    if (
        isinstance(maximum_support_queries, bool)
        or not isinstance(maximum_support_queries, int)
        or maximum_support_queries < 1
    ):
        raise ValueError(
            "PreparedSurfaceSourceSupport.maximum_support_queries must be a positive integer; got invalid query capacity"
        )
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise ValueError(
            "PreparedSurfaceSourceSupport.mesh must be a surface in three-space; got incompatible carrier"
        )
    if not coordinate_contract.is_orthonormal_cartesian:
        raise ValueError(
            "PreparedSurfaceSourceSupport.coordinate_contract must be orthonormal Cartesian; got incompatible frame"
        )
    expected_basis = canonical_fingerprint(
        algebra.coordinate_source_signature(coordinate_lagrange_element("triangle", 1))
    )
    return _SurfaceRootFrame(domain, coordinate_contract, expected_basis)


class PreparedSurfaceSourceSupport(StrictModule, NonTrainableState):
    """Bind surface cells to source root banks and their certification evidence."""

    domain: meshing_domain_owner.MeshingDomain
    coordinate_contract: SpatialCoordinateContract
    original: CellMeshingResult | SurfaceSourceRootAtlas | InitialCollectiveMeshEvidence
    root_boundary_source: (
        domain_boundary_owner.MeshingDomainBoundarySource | InitialCollectiveMeshEvidence
    )
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
        domain: meshing_domain_owner.MeshingDomain,
        root_result: CellMeshingResult | SurfaceSourceRootAtlas,
        root_cell_parameters: ArrayLike,
        root_boundary_source: domain_boundary_owner.MeshingDomainBoundarySource
        | InitialCollectiveMeshEvidence,
        /,
        *,
        maximum_support_queries: int = 1 << 26,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        # Resolve the declared class at the request boundary, not a subtype-valued protocol property.
        from ..meshing._result import CellMeshingResult

        if not isinstance(
            root_result,
            (CellMeshingResult, SurfaceSourceRootAtlas),
        ):
            raise TypeError(
                "PreparedSurfaceSourceSupport.root_result must be an accepted result or source atlas; got unsupported root kind"
            )
        frame = _prepare_surface_root_frame(
            domain,
            root_result.mesh,
            root_result.coordinate_contract,
            maximum_support_queries,
        )
        expected_basis = frame.basis_signature
        match root_result:
            case SurfaceSourceRootAtlas():
                match root_boundary_source:
                    case domain_boundary_owner.MeshingDomainBoundarySource():
                        binding = self._bind_atlas_roots(
                            domain, root_result, root_boundary_source
                        )
                    case _:
                        raise ValueError(
                            "PreparedSurfaceSourceSupport.root_boundary_source must bind the atlas domain; got incompatible source"
                        )
            case CellMeshingResult():
                collective_type = _collective_evidence_type()
                match root_boundary_source:
                    case collective_type():
                        binding = self._bind_collective_roots(
                            domain, root_result, root_boundary_source, expected_basis
                        )
                    case domain_boundary_owner.MeshingDomainBoundarySource():
                        binding = self._bind_serial_roots(
                            domain,
                            root_result,
                            root_boundary_source,
                            expected_basis,
                            maximum_support_queries,
                            budget,
                        )
                    case _:
                        raise TypeError(
                            "PreparedSurfaceSourceSupport.root_boundary_source must be a domain source or collective evidence; got unsupported source"
                        )
            case invalid:
                assert_never(invalid)
        count, corner_count, coefficient_count = (
            binding.count,
            binding.corner_count,
            binding.coefficient_count,
        )
        ids, parameters, patches = binding.ids, binding.parameters, binding.patches
        corners, coordinates = binding.corners, binding.coordinates
        if count * (corner_count * 2 + coefficient_count) > maximum_support_queries:
            raise ValueError(
                "PreparedSurfaceSourceSupport.work must fit maximum_support_queries before root expansion; got exceeded query capacity"
            )
        if (
            isinstance(binding.original, _meshing_result_capabilities().CellMeshingResult)
            and binding.original.mesh.storage is None
        ):
            supplied, host_parameters = jax.device_get((root_cell_parameters, parameters))
            supplied = np.asarray(supplied, dtype=np.float64)
            parameters_match = (
                supplied.shape == (count, corner_count, 2)
                and np.all(supplied == host_parameters)
                and np.all(np.isfinite(host_parameters))
            )
        else:
            supplied = jnp.asarray(root_cell_parameters, dtype=jnp.float64)
            parameters_match = (
                supplied.shape == (count, corner_count, 2)
                and bool(jax.device_get(jnp.all(supplied == parameters)))
                and bool(jax.device_get(jnp.all(jnp.isfinite(parameters))))
            )
        if not parameters_match:
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_cell_parameters must match the finite root corner bank; got different parameters or shape"
            )
        if (
            not bool(jax.device_get(jnp.all(ids[1:] > ids[:-1])))
            or patches.shape != (count,)
            or corners.shape != (count, corner_count)
            or coordinates.shape != (count, coefficient_count, 3)
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_axes must be strictly ordered and aligned; got invalid bank shapes or IDs"
            )
        self.domain, self.coordinate_contract, self.original = (
            frame.domain,
            frame.coordinate_contract,
            binding.original,
        )
        self.root_cell_ids, self.root_parameters, self.root_patches = (
            ids,
            parameters,
            patches,
        )
        self.root_boundary_source = root_boundary_source
        self.root_corner_ids, self.root_coordinates = corners, coordinates
        self.root_basis_signature = binding.basis_signature
        self.root_geometry_id, self.root_topology_id, self.original_evidence_id = (
            binding.geometry_id,
            binding.topology_id,
            binding.evidence_id,
        )
        self.source_expression_id = canonical_fingerprint(
            [meshing_domain_owner.patch_identity(patch) for patch in domain.patches]
        )
        self.maximum_support_queries = maximum_support_queries
        self.support_id = self._identity()

    @classmethod
    def from_stored_fields(
        cls,
        value: PreparedSurfaceSourceSupport,
        /,
        *,
        root_result: CellMeshingResult | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedSurfaceSourceSupport:
        """Replay stored support through its constructor with a retained root carrier.

        Serial results and spline atlases supply their retained carrier.
        Collective support requires the accepted root result from the archive
        context; global evidence alone does not contain that local publication.
        """
        # The accepted-result declaration participates in the constructor's publication cycle.
        from ..meshing._result import CellMeshingResult

        value.require_current()
        match value.original:
            case SurfaceSourceRootAtlas() | CellMeshingResult():
                retained_root = value.original
            case _:
                collective_type = _collective_evidence_type()
                match value.original:
                    case collective_type():
                        if root_result is None:
                            raise ValueError(
                                "PreparedSurfaceSourceSupport.root_result must supply the accepted collective root context; got missing root carrier"
                            )
                        retained_root = root_result
                    case _:
                        raise TypeError(
                            "PreparedSurfaceSourceSupport.original must retain an accepted result, source atlas, or collective evidence; got unsupported root kind"
                        )
        return cls(
            value.domain,
            retained_root,
            value.root_parameters,
            value.root_boundary_source,
            maximum_support_queries=value.maximum_support_queries,
            budget=budget,
        )

    @staticmethod
    def _bind_atlas_roots(
        domain: meshing_domain_owner.MeshingDomain,
        root_result: SurfaceSourceRootAtlas,
        root_boundary_source: domain_boundary_owner.MeshingDomainBoundarySource,
    ) -> _RootBinding:
        root_result.require_current()
        if (
            root_boundary_source.domain.content_domain_id() != domain.content_domain_id()
            or root_result.domain.content_domain_id() != domain.content_domain_id()
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_boundary_source must bind the atlas and domain content; got different source content"
            )
        if set(np.asarray(root_result.root_patches).tolist()) != set(
            root_boundary_source.patches
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_patches must select the boundary-source patches; got different patch set"
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
            tuple(algebra.coordinate_source_signature(element) for element in elements)
        )
        geometry_id, topology_id, evidence_id = (
            cell_geometry_id(root_result.geometry),
            root_result.mesh.topology_id,
            root_result.atlas_id,
        )
        return _RootBinding(
            original,
            count,
            ids,
            parameters,
            patches,
            corners,
            coordinates,
            expected_basis,
            geometry_id,
            topology_id,
            evidence_id,
            corner_count,
            coefficient_count,
        )

    @staticmethod
    def _bind_collective_roots(
        domain: meshing_domain_owner.MeshingDomain,
        root_result: CellMeshingResult,
        root_boundary_source: InitialCollectiveMeshEvidence,
        expected_basis: str,
    ) -> _RootBinding:
        corner_count, coefficient_count = 3, 3
        # Initial collective evidence composes source-support geometry; resolve its premise type after those declarations load.
        from ..meshing._initial_certification import InitialSurfacePremise

        original = root_boundary_source
        original.require_current()
        match original.premise:
            case InitialSurfacePremise():
                compiled = original.premise.compiled
            case _:
                raise ValueError(
                    "PreparedSurfaceSourceSupport.collective_source must come from an initial surface premise; got a non-surface initial premise"
                )
        if (
            compiled.domain.content_domain_id() != domain.content_domain_id()
            or root_result.collective_evidence is None
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.collective_source must bind domain content and collective evidence; got different source content or absent evidence"
            )

        actual = _meshing_result_capabilities().require_original_meshing_source(
            root_result
        )
        if (
            not isinstance(actual, _collective_evidence_type())
            or actual.evidence_id != original.evidence_id
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.collective_source must match the accepted result evidence_id; got different collective evidence"
            )
        if root_result.mesh.topology_id != original.topology_id:
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_topology_id must match the initial collective topology; got different root topology"
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
                "PreparedSurfaceSourceSupport.root_corner_ids must identify coordinate coefficients; got unmatched coefficient IDs"
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
                "PreparedSurfaceSourceSupport.root_basis must match the P1 surface signature; got different coordinate expression"
            )
        geometry_id, topology_id, evidence_id = (
            original.coordinate_geometry_id,
            original.topology_id,
            original.source_evidence_id,
        )
        return _RootBinding(
            original,
            count,
            ids,
            parameters,
            patches,
            corners,
            coordinates,
            expected_basis,
            geometry_id,
            topology_id,
            evidence_id,
            corner_count,
            coefficient_count,
        )

    @staticmethod
    def _bind_serial_roots(
        domain: meshing_domain_owner.MeshingDomain,
        root_result: CellMeshingResult,
        root_boundary_source: domain_boundary_owner.MeshingDomainBoundarySource,
        expected_basis: str,
        maximum_support_queries: int,
        budget: CoordinateEnclosureBudget | None,
    ) -> _RootBinding:
        corner_count, coefficient_count = 3, 3
        if root_result.mesh.storage is not None:
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_result.mesh must be serial for whole-surface coverage; got owner-local storage"
            )
        if root_boundary_source.domain.content_domain_id() != domain.content_domain_id():
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_boundary_source.domain must match domain content; got different source content"
            )
        cover = root_boundary_source.boundary_chart_cover(
            maximum_support_queries, budget=budget
        )
        certification = root_result.certification
        if certification is None:
            raise ValueError(
                "PreparedSurfaceSourceSupport.certification must retain the accepted root report; got absent certification"
            )
        fidelity = certification.fidelity
        if (
            not cover.complete
            or cover.semantics != "certified"
            or fidelity is None
            or fidelity.status != "certified"
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.certification must contain certified continuous coverage and fidelity; got incomplete or uncertified evidence"
            )
        fidelity.binding.require(root_result.mesh, root_result.geometry)
        if (
            root_boundary_source.source_id != domain.source_id
            or root_boundary_source.source_revision != domain.source_revision
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_boundary_source must match domain source identity and revision; got different source revision"
            )
        source_charts = root_result.surface_source
        if not isinstance(source_charts, SurfaceSourceCharts):
            raise ValueError(
                "PreparedSurfaceSourceSupport.surface_source must retain SurfaceSourceCharts; got absent or incompatible charts"
            )
        source_charts.require_root(root_result.mesh, root_result.geometry)

        if (
            source_charts.domain.content_domain_id() != domain.content_domain_id()
            or source_charts.boundary_source.source_scope_id
            != root_boundary_source.source_scope_id
            or source_charts.boundary_source.boundary_occurrence_id
            != root_boundary_source.boundary_occurrence_id
            or fingerprint_owner.array_tree_fingerprint(
                source_charts.boundary_source.chart_triangulations
            )
            != fingerprint_owner.array_tree_fingerprint(
                root_boundary_source.chart_triangulations
            )
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_boundary_source must match retained content, scope, and triangulations; got different boundary source"
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
                    "PreparedSurfaceSourceSupport.root_result.mesh.blocks must contain triangles; got different cell kind"
                )
            corner_rows.append(jnp.asarray(controls)[jnp.asarray(route)])
            corner_ids.append(
                jnp.asarray(root_result.mesh.vertex_global_ids)[
                    jnp.asarray(block.vertices)
                ]
            )
            all_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        order = association_state_owner.entity_rows(
            root_result.mesh, 2, np.concatenate(all_ids)
        )
        inverse = np.argsort(order, kind="stable")
        coordinates = jnp.concatenate(corner_rows)[inverse]
        corners = jnp.concatenate(corner_ids)[inverse]
        geometry_id, topology_id, evidence_id = (
            cell_geometry_id(root_result.geometry),
            root_result.mesh.topology_id,
            fidelity.certificate_id,
        )
        return _RootBinding(
            original,
            count,
            ids,
            parameters,
            patches,
            corners,
            coordinates,
            expected_basis,
            geometry_id,
            topology_id,
            evidence_id,
            corner_count,
            coefficient_count,
        )

    @staticmethod
    def domain_row(
        domain: meshing_domain_owner.MeshingDomain,
        dimension: int,
        index: int,
        path: tuple[str, ...],
    ) -> int:
        if dimension not in (0, 1, 2):
            raise ValueError(
                "PreparedSurfaceSourceSupport.dimension must name a surface source stratum; got unsupported dimension"
            )
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
                "PreparedSurfaceSourceSupport.association must identify one source occurrence and definition; got nonunique or absent source row"
            )
        return rows[0]

    @staticmethod
    def _validate_serial_parameters(
        domain: meshing_domain_owner.MeshingDomain,
        patches: np.ndarray,
        parameters: np.ndarray,
        vertex: GeometryAssociation,
        corners: np.ndarray,
    ) -> None:
        targets, dimensions, indices, parameter_values = jax.device_get(
            (
                vertex.target_global_ids,
                vertex.source_dimensions,
                vertex.source_indices,
                vertex.parameters,
            )
        )
        target_keys = np.asarray(targets, dtype=np.int64)[:, None]
        for patch, chart, ids in zip(patches.tolist(), parameters, corners, strict=True):
            rows = entity_keys_owner.key_rows(
                target_keys, np.asarray(ids, dtype=np.int64)[:, None]
            )
            if np.any(rows < 0):
                raise ValueError(
                    "SurfaceSourceCharts.associations must classify every root corner ID; got an unclassified corner"
                )
            for uv, row in zip(chart, rows.tolist(), strict=True):
                dimension = int(dimensions[row])
                index = PreparedSurfaceSourceSupport.domain_row(
                    domain,
                    dimension,
                    int(indices[row]),
                    vertex.source_entities.occurrence_path(row),
                )
                values = parameter_values[row]
                if dimension == 0:
                    poles = [
                        use
                        for loop in domain.patches[patch].loops
                        for use in loop
                        if isinstance(use, meshing_domain_owner.PatchPoleUse)
                        and use.corner == index
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
                            "SurfaceSourceCharts.root_parameters must bind each corner to its source stratum; got unmatched physical corner"
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
                            "SurfaceSourceCharts.root_parameters must match vertex and coedge associations; got different source parameters"
                        )

    def _identity(self) -> str:

        boundary = self.root_boundary_source
        match boundary:
            case domain_boundary_owner.MeshingDomainBoundarySource():
                boundary_identity = canonical_fingerprint(
                    {
                        "domain": boundary.domain.domain_id,
                        "scope": boundary.source_scope_id,
                        "resolution": boundary.resolution,
                        "occurrences": boundary.boundary_occurrence_id,
                        "charts": fingerprint_owner.array_tree_fingerprint(
                            boundary.chart_triangulations
                        ),
                    }
                )
            case _:
                collective_type = _collective_evidence_type()
                match boundary:
                    case collective_type():
                        boundary.require_current()
                        boundary_identity = boundary.evidence_id
                    case _:
                        raise TypeError(
                            "PreparedSurfaceSourceSupport.root_boundary_source must retain a domain source or collective evidence; got unsupported source kind"
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
                    meshing_domain_owner.patch_identity(patch)
                    for patch in self.domain.patches
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

        result_type = _meshing_result_capabilities().CellMeshingResult
        match self.original:
            case result_type():
                charts = self.original.surface_source
                match charts:
                    case SurfaceSourceCharts():
                        charts.require_root(self.original.mesh, self.original.geometry)
                    case _:
                        raise ValueError(
                            "PreparedSurfaceSourceSupport.original.surface_source must retain SurfaceSourceCharts; got absent or incompatible charts"
                        )
            case SurfaceSourceRootAtlas():
                self.original.require_current()
            case _:
                # Collective evidence is authenticated by _identity below.
                pass
        if self._identity() != self.support_id:
            raise ValueError(
                "PreparedSurfaceSourceSupport.support_id must match maps, strata, and scientific identities; got changed content"
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
                    "PreparedSurfaceSourceSupport.root_banks must be fully addressable for a serial proof; got nonaddressable collective arrays"
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
                "PreparedSurfaceSourceSupport.receipts must bind this support for an owner-local proof; got absent or different support"
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
                        "PreparedSurfaceSourceSupport.receipts must agree on source root data; got different root-packet rows"
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
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_cell_id must identify a spline-atlas root; got absent root ID"
            )
        return coordinate_lagrange_element("triangle", 1)

    def _root_expression(
        self,
        identifier: int,
        packet: tuple[np.ndarray, int, np.ndarray, np.ndarray],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[algebra.Expression, ...], tuple[algebra.Polynomial, ...], float]:
        """Return root coordinate and UV expressions with their source-image bound."""
        uv, patch, _, coefficients = packet
        basis = self._root_element(identifier)
        root_expressions = algebra.coordinate_expressions(
            basis, coefficients[: basis.local_dof_count], budget=budget
        )
        uv_basis = coordinate_lagrange_element(basis.cell_kind, 1)
        uv_polynomials = algebra.coordinate_polynomials(uv_basis, uv, budget=budget)
        if root_expressions is None or uv_polynomials is None:
            raise ValueError(
                "PreparedSurfaceSourceSupport.root_expression must resolve coordinate and chart expressions; got unresolved expressions"
            )
        if isinstance(self.original, SurfaceSourceRootAtlas):
            return root_expressions, uv_polynomials, 0.0
        source_nodes = self.domain.evaluate(np.full(3, patch), uv)
        nodal_error = float(np.max(np.linalg.norm(source_nodes - coefficients, axis=1)))
        bound = (
            float(self.domain.interpolation_bounds(patch, uv[None])[0])
            + nodal_error
            + 256 * np.finfo(np.float64).eps * self.domain.scale
        )
        sphere = full_sphere_frame(self.domain, patch, budget=budget)
        if sphere is not None:
            bound = min(
                bound,
                float(sphere_triangle_distance_bounds(sphere, coefficients[None])[0]),
            )
        if not np.isfinite(bound):
            raise ValueError(
                "PreparedSurfaceSourceSupport.source_bound must be a finite whole-root enclosure; got nonfinite bound"
            )
        return root_expressions, uv_polynomials, bound

    def _cell_pullback(
        self,
        element: CellGeometryElement,
        parent: int,
        packet: tuple[np.ndarray, int, np.ndarray, np.ndarray],
        root_expressions: tuple[algebra.Expression, ...],
        local_controls: np.ndarray,
        retained_vertices: np.ndarray | None,
        exact: bool,
        lineage_corners: _Corners | None,
        block_restriction: tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]
        | None,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _CellPullback:
        remainder = None
        if block_restriction is None:
            if retained_vertices is None or not exact:
                raise ValueError(
                    "PreparedSurfaceSourceSupport.packet_action must retain source corner ancestry without lineage fitting; got absent or incompatible ancestry"
                )
            base, arguments, remainder = algebra.prepare_p1_source_packet_pullback(
                element,
                self._root_element(parent),
                packet[0],
                packet[3],
                local_controls,
                root_expressions,
                root_corner_ids=packet[2],
                retained_corner_ids=retained_vertices[: packet[2].size],
                budget=budget,
            )
            determinant = algebra.determinant(
                tuple(
                    tuple(
                        algebra.derivative(value, axis, budget=budget)
                        for axis in range(2)
                    )
                    for value in arguments
                ),
                budget=budget,
            )
            if (
                algebra.polynomial_bounds(determinant, "simplex", 2, budget=budget)[0]
                <= 0
            ):
                raise ValueError(
                    "PreparedSurfaceSourceSupport.packet_action must have positive resolved UV orientation; got nonpositive orientation bound"
                )
        else:
            base, block_arguments = block_restriction
            arguments = (
                block_arguments
                if lineage_corners is None
                else _arguments(*_corner_restriction(lineage_corners), budget=budget)
            )
        expected = canonical_fingerprint(
            algebra.coordinate_source_signature(self._root_element(parent))
        )
        if canonical_fingerprint(algebra.coordinate_source_signature(base)) != expected:
            raise ValueError(
                "PreparedSurfaceSourceSupport.restriction.base must match the root basis signature; got different source basis"
            )
        return _CellPullback(base, arguments, remainder)

    def _prove_surface_cell(
        self,
        element: CellGeometryElement,
        identifier: int,
        parent: int,
        vertices: np.ndarray,
        local_controls: np.ndarray,
        carrier: np.ndarray,
        vertex_ids: np.ndarray,
        packet: tuple[np.ndarray, int, np.ndarray, np.ndarray],
        root_expression: tuple[
            tuple[algebra.Expression, ...], tuple[algebra.Polynomial, ...], float
        ],
        retained_vertices: np.ndarray | None,
        lineage_corners: _Corners | None,
        block_restriction: tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]
        | None,
        exact: bool,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _SurfaceCellProof:
        with budget.temporary_scope():
            pullback = self._cell_pullback(
                element,
                parent,
                packet,
                root_expression[0],
                local_controls,
                retained_vertices,
                exact,
                lineage_corners,
                block_restriction,
                budget=budget,
            )
            refs, source_corners, deviation, actual_corners = _cell_restriction(
                element,
                local_controls,
                pullback.arguments,
                root_expression,
                root_kind=self._root_element(parent).cell_kind,
                exact=exact,
                physical_remainder=pullback.remainder,
                exact_source_controls=(
                    packet[3][: pullback.base.local_dof_count]
                    if exact and pullback.remainder is None
                    else None
                ),
                budget=budget,
            )
        error = max(
            sum(
                (
                    abs(value - Fraction(float(rep)))
                    for value, rep in zip(point, represented, strict=True)
                ),
                Fraction(0),
            )
            for point, represented in zip(actual_corners, carrier, strict=True)
        )
        bound = root_expression[2]
        source_bound = (
            bound
            if isinstance(self.original, SurfaceSourceRootAtlas)
            else float(np.nextafter(bound, np.inf))
        )
        return _SurfaceCellProof(
            identifier,
            parent,
            packet[1],
            tuple(vertex_ids[vertices].tolist()),
            refs,
            source_corners,
            upper(Fraction(source_bound) + deviation),
            upper(error),
            upper(deviation),
        )

    def _native_block_restriction(
        self,
        element: CellGeometryElement,
        restriction_cache: dict[
            str, tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]]
        ],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[CellGeometryElement, tuple[algebra.Polynomial, ...]] | None:
        """Cache non-packet source actions while retaining their charged basis."""
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
        packet_action = isinstance(original_action, BarycentricCellGeometryElement)
        if packet_action:
            return None
        cached_restriction = restriction_cache.get(element.element_id)
        if cached_restriction is None:
            cached_restriction = _restriction(element, budget=budget)
            restriction_cache[element.element_id] = cached_restriction
            budget.retain_basis(cached_restriction)
        else:
            budget.reserve(1)
        return cached_restriction

    @checked
    def prove_native_restrictions(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        receipts: SurfaceSourceReceipts | None = None,
        ancestry: Mapping[int, SurfaceCellAncestry] | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> SurfaceRestrictionRows:
        """Prove source-root coordinate restrictions and return per-cell bounds.

        Restricted maps retain root coefficient laws. Unrestricted serial maps
        use lineage-proved root corners and add their measured map deviation."""
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError(
                "PreparedSurfaceSourceSupport.proof.mesh must be a surface in three-space; got incompatible carrier"
            )

        if (
            canonical_fingerprint(
                [
                    meshing_domain_owner.patch_identity(patch)
                    for patch in self.domain.patches
                ]
            )
            != self.source_expression_id
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.source_expression_id must match patch expressions; got changed source content"
            )
        restrictions = geometry.restriction_source
        if restrictions is not None and (
            restrictions.source_geometry_id != self.root_geometry_id
            or restrictions.source_topology_id != self.root_topology_id
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.restrictions must retain source geometry and topology IDs; got different root identity"
            )
        if ancestry is not None and (
            restrictions is not None or mesh.storage is not None
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.ancestry must apply only to serial unrestricted maps; got restricted or owner-local carrier"
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

        proofs: list[_SurfaceCellProof] = []

        with (
            budget.activate(),
            budget.bound_stage(self.maximum_support_queries, 256 * 1024**2),
        ):
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
                if block.cell_kind not in ("triangle", "quadrilateral"):
                    raise ValueError(
                        "PreparedSurfaceSourceSupport.proof.blocks must contain triangle or quadrilateral cells; got different cell kind"
                    )
                block_restriction = self._native_block_restriction(
                    element, restriction_cache, budget=budget
                )
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
                                "PreparedSurfaceSourceSupport.ancestry must identify a root for every serial cell; got absent cell ancestry"
                            )
                        parent, lineage_corners = ancestry[identifier]
                    if parent not in root_packets:
                        raise ValueError(
                            "PreparedSurfaceSourceSupport.receipts must contain the current cell's source root; got absent root packet"
                        )
                    packet = root_packets[parent]
                    if parent_vertices is not None and not np.array_equal(
                        parent_vertices[row][: packet[2].size], packet[2]
                    ):
                        raise ValueError(
                            "PreparedSurfaceSourceSupport.parent_vertex_ids must retain ordered source corners; got different corner IDs"
                        )
                    if parent not in root_cache:
                        # Root expressions stay cached, so their storage stays charged.
                        root_cache[parent] = self._root_expression(
                            parent, packet, budget=budget
                        )
                    proof = self._prove_surface_cell(
                        element,
                        identifier,
                        parent,
                        vertices,
                        host_controls[dofs],
                        host_coordinates[vertices],
                        host_vertex_ids,
                        packet,
                        root_cache[parent],
                        None if parent_vertices is None else parent_vertices[row],
                        lineage_corners,
                        block_restriction,
                        ancestry is None,
                        budget=budget,
                    )
                    proofs.append(proof)
        return SurfaceRestrictionRows(
            np.asarray([proof.cell_id for proof in proofs], dtype=np.int64),
            np.asarray([proof.root_cell_id for proof in proofs], dtype=np.int64),
            np.asarray([proof.patch for proof in proofs], dtype=np.int64),
            tuple(proof.vertex_ids for proof in proofs),
            tuple(proof.reference_corners for proof in proofs),
            tuple(proof.source_corners for proof in proofs),
            np.asarray([proof.bound for proof in proofs], dtype=np.float64),
            np.asarray([proof.corner_error for proof in proofs], dtype=np.float64),
            np.asarray([proof.deviation for proof in proofs], dtype=np.float64),
        )


def _equal_expressions(
    first: algebra.Expression,
    second: algebra.Expression,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    a, b = algebra.expression_parts(first, 2, budget=budget)
    c, d = algebra.expression_parts(second, 2, budget=budget)
    return algebra.multiply(a, d, budget=budget) == algebra.multiply(c, b, budget=budget)


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
    budget: CoordinateEnclosureBudget,
) -> tuple[_Corners, _Corners, Fraction, tuple[tuple[Fraction, ...], ...]]:
    """Return source-reference corners, source UV corners, and coordinate restriction error."""

    root_expressions, uv_polynomials, _ = expression
    corners = tuple(
        rational_vector(row)
        for row in reference_cell_owner.reference_cell_topology(
            element.cell_kind
        ).vertices
    )
    refs = tuple(
        (
            rational_coefficient(
                algebra.evaluate(arguments[0], point, budget=budget),
                role="PreparedSurfaceSourceSupport.reference_corner",
            ),
            rational_coefficient(
                algebra.evaluate(arguments[1], point, budget=budget),
                role="PreparedSurfaceSourceSupport.reference_corner",
            ),
        )
        for point in corners
    )
    domain = "simplex" if element.cell_kind == "triangle" else "box"
    constraints = (
        (
            *arguments,
            algebra.add(
                algebra.constant(1, 2, budget=budget),
                algebra.scale(
                    algebra.sum_polynomials(arguments, budget=budget), -1, budget=budget
                ),
                budget=budget,
            ),
        )
        if root_kind == "triangle"
        else (
            *arguments,
            *(
                algebra.add(
                    algebra.constant(1, 2, budget=budget),
                    algebra.scale(value, -1, budget=budget),
                    budget=budget,
                )
                for value in arguments
            ),
        )
    )
    if any(
        min(algebra.bernstein_coefficients(value, domain, 2, budget=budget)) < 0
        for value in constraints
    ):
        raise ValueError(
            "PreparedSurfaceSourceSupport.reference_image must lie in the source reference domain; got outside reference image"
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
            tuple(
                rational_coefficient(
                    algebra.expression_evaluate(value, point, budget=budget),
                    role="PreparedSurfaceSourceSupport.physical_corner",
                )
                for value in root_expressions
            )
            for point in refs
        )
        source_corners = tuple(
            (
                rational_coefficient(
                    algebra.evaluate(uv_polynomials[0], point, budget=budget),
                    role="PreparedSurfaceSourceSupport.source_corner",
                ),
                rational_coefficient(
                    algebra.evaluate(uv_polynomials[1], point, budget=budget),
                    role="PreparedSurfaceSourceSupport.source_corner",
                ),
            )
            for point in refs
        )
        return refs, source_corners, Fraction(0), actual_corners
    actual = algebra.coordinate_expressions(element, controls, budget=budget)
    if actual is None:
        raise ValueError(
            "PreparedSurfaceSourceSupport.target_expression must resolve coordinate expressions; got unresolved expressions"
        )
    restricted = tuple(
        algebra.expression_compose(value, arguments, budget=budget)
        for value in root_expressions
    )
    actual_corners = tuple(
        tuple(
            rational_coefficient(
                algebra.expression_evaluate(value, point, budget=budget),
                role="PreparedSurfaceSourceSupport.physical_corner",
            )
            for value in actual
        )
        for point in corners
    )
    if physical_remainder is not None:
        difference = tuple(
            algebra.expression_add(
                a, algebra.expression_scale(r, -1, budget=budget), budget=budget
            )
            for a, r in zip(actual, restricted, strict=True)
        )
        if any(
            not _equal_expressions(a, b, budget=budget)
            for a, b in zip(difference, physical_remainder, strict=True)
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.remainder must bind the whole physical coefficient action; got different physical remainder"
            )
        deviation = sum(
            (
                max(abs(min(coefficients)), abs(max(coefficients)))
                for value in physical_remainder
                for coefficients in (
                    algebra.expression_bernstein_coefficients(
                        value, domain, 2, budget=budget
                    ),
                )
            ),
            Fraction(0),
        )
    elif exact:
        if any(
            not _equal_expressions(first, second, budget=budget)
            for first, second in zip(restricted, actual, strict=True)
        ):
            raise ValueError(
                "PreparedSurfaceSourceSupport.target_expression must equal the source restriction; got different coordinate expression"
            )
        deviation = Fraction(0)
    else:
        if element.cell_kind != "triangle" or root_kind != "triangle":
            raise ValueError(
                "PreparedSurfaceSourceSupport.ancestry must bind an affine triangle restriction; got incompatible source chart"
            )
        deviation = max(
            sum(
                (
                    abs(a - algebra.expression_evaluate(r, point, budget=budget))
                    for a, r in zip(corner, restricted, strict=True)
                ),
                Fraction(0),
            )
            for corner, point in zip(actual_corners, corners, strict=True)
        )
    source_corners = tuple(
        (
            rational_coefficient(
                algebra.evaluate(uv_polynomials[0], point, budget=budget),
                role="PreparedSurfaceSourceSupport.source_corner",
            ),
            rational_coefficient(
                algebra.evaluate(uv_polynomials[1], point, budget=budget),
                role="PreparedSurfaceSourceSupport.source_corner",
            ),
        )
        for point in refs
    )
    return (
        refs,
        source_corners,
        rational_coefficient(deviation, role="PreparedSurfaceSourceSupport.deviation"),
        actual_corners,
    )


def upper(value: Fraction, /) -> float:
    """Return a binary64 upper bound of a rational quantity, rounding outward if needed."""
    result = float(value)
    return result if Fraction(result) >= value else float(np.nextafter(result, np.inf))


class SurfaceNativeRestrictionBoundarySource(StrictModule, NonTrainableState):
    """Source queries plus rational current source-chart restrictions."""

    root: domain_boundary_owner.MeshingDomainBoundarySource
    support: PreparedSurfaceSourceSupport
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: domain_boundary_owner.MeshingDomainBoundarySource,
        support: PreparedSurfaceSourceSupport,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
    ) -> None:
        if not isinstance(support.original, SurfaceSourceRootAtlas):
            raise ValueError(
                "SurfaceNativeRestrictionBoundarySource.support.original must be a source-definition atlas; got different root kind"
            )
        support.require_current()
        if root.domain.content_domain_id() != support.domain.content_domain_id():
            raise ValueError(
                "SurfaceNativeRestrictionBoundarySource.root.domain must match support domain content; got different source content"
            )
        if set(root.patches) != set(np.asarray(support.root_patches).tolist()):
            raise ValueError(
                "SurfaceNativeRestrictionBoundarySource.root.patches must match support patches; got different patch set"
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
                "SurfaceNativeRestrictionBoundarySource.support.original must retain the source atlas; got different root kind"
            )
        original.require_current()
        if (self.source_id, self.source_revision) != (
            self.root.source_id,
            self.root.source_revision,
        ):
            raise ValueError(
                "SurfaceNativeRestrictionBoundarySource.source must match root identity and revision; got different source identity"
            )


def _source_uv_domain(
    domain: meshing_domain_owner.MeshingDomain, patch: int, /
) -> domain_coverage_owner.PiecewiseLinearDomain:
    """Independent rational UV trim domain from source line definitions."""

    positions: dict[tuple[Fraction, Fraction], int] = {}
    points, facets = [], []
    for loop in domain.patches[patch].loops:
        for use in loop:
            if not isinstance(use, meshing_domain_owner.PatchCurveUse) or not isinstance(
                use.pcurve, patches_owner.LineCurve
            ):
                raise ValueError(
                    "SurfaceSourceFidelity.trim must contain affine line coedges; got different trim family"
                )
            if (
                use.first_root is not None
                or use.last_root is not None
                or use.trim_curve is not None
            ):
                raise ValueError(
                    "SurfaceSourceFidelity.trim must have scalar affine endpoints; got root-valued or nonlinear trim"
                )
            origin = rational_vector(np.asarray(use.pcurve.origin))
            direction = rational_vector(np.asarray(use.pcurve.direction))
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
                        "SurfaceSourceFidelity.trim.endpoints must be exactly representable domain coefficients; got nonbinary rational endpoint"
                    )
                if point not in positions:
                    positions[point] = len(points)
                    points.append(tuple(float(value) for value in point))
                rows.append(positions[point])
            facets.append(rows)
    return domain_coverage_owner.PiecewiseLinearDomain(
        np.asarray(points, dtype=np.float64),
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(facets), 1)),
        ("surface",),
        source_id=domain.source_id,
    )


class _FidelityCharts(StrictModule):
    charts: dict[int, list[mapped_reference_coverage_owner.Chart]]


class _UVPatchEvidence(StrictModule):
    findings: tuple[embedding_state_owner.MeshCertificateFinding, ...]


def _restriction_fidelity_charts(
    support: PreparedSurfaceSourceSupport,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FidelityCharts:
    packets = support._root_packets(mesh, None)
    elements, _, _ = geometry.resolve(mesh)
    record = geometry.restriction_source
    if record is None:
        raise ValueError(
            "SurfaceSourceFidelity.geometry must retain source root ancestry; got absent restriction record"
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    charts: dict[int, list[mapped_reference_coverage_owner.Chart]] = {}
    for block, element in zip(mesh.blocks, elements, strict=True):
        for cell, vertices, parent in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(record.block_parent_cell_ids[block.name]),
            strict=True,
        ):
            root = support._root_element(int(parent))
            reference = mapped_reference_coverage_owner.polynomial_reference_chart(
                element, root, budget=budget
            )
            if reference is None:
                raise ValueError(
                    "SurfaceSourceFidelity.chart must retain the source reference chain; got unresolved chart"
                )
            reference_element, _ = reference
            uv, patch, _, _ = packets[int(parent)]
            charts.setdefault(patch, []).append(
                mapped_reference_coverage_owner.ComposedReferenceChart(
                    block.name,
                    int(cell),
                    block.cell_kind,
                    tuple(int(value) for value in vertex_ids[vertices]),
                    reference_element,
                    uv,
                )
            )
    return _FidelityCharts(charts)


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


def _certify_uv_patch(
    source: SurfaceNativeRestrictionBoundarySource,
    patch: int,
    charts: dict[int, list[mapped_reference_coverage_owner.Chart]],
    limits: embedding_state_owner.MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> _UVPatchEvidence:
    support = source.support
    findings: list[embedding_state_owner.MeshCertificateFinding] = []
    if patch not in charts:
        raise ValueError(
            "SurfaceSourceFidelity.charts must cover every selected patch; got uncovered patch"
        )
    atlas = mapped_reference_coverage_owner.exact_chart_mesh(
        charts[patch], 2, budget=budget
    )
    if atlas is None:
        raise ValueError(
            "SurfaceSourceFidelity.charts must share current vertex images; got incompatible UV chart images"
        )
    uv_mesh, uv_geometry = atlas
    uv_validity = cell_geometry_validity_owner.certify_cell_geometry_validity(
        uv_geometry, mesh=uv_mesh
    )
    embedding = global_embedding_owner.certify_global_embedding(
        uv_mesh, uv_geometry, uv_validity, limits=limits, budget=budget
    )
    affine_trim = all(
        isinstance(use, meshing_domain_owner.PatchCurveUse)
        and isinstance(use.pcurve, patches_owner.LineCurve)
        and use.first_root is None
        and use.last_root is None
        and use.trim_curve is None
        for loop in support.domain.patches[patch].loops
        for use in loop
    )
    if affine_trim:
        coverage = domain_coverage_owner.certify_domain_coverage(
            uv_mesh,
            uv_geometry,
            _source_uv_domain(support.domain, patch),
            np.zeros(uv_mesh.entity_set(2).count, dtype=np.int64),
            embedding=embedding,
            limits=limits,
            budget=budget,
        )
        findings.extend(coverage.findings)
    else:
        cover = source.root.boundary_chart_cover(
            limits.maximum_source_samples, _topology_only=True, budget=budget
        )
        if not cover.complete or cover.semantics != "certified":
            raise ValueError(
                "SurfaceSourceFidelity.trim_cover must certify coverage of every selected patch; got uncertified or incomplete cover"
            )
        records = [
            value for value in source.root.chart_triangulations if value[0] == patch
        ]
        if len(records) != 1:
            raise ValueError(
                "SurfaceSourceFidelity.trim_cover must retain one chart for the patch; got absent or duplicate patch charts"
            )
        _, root_charts, _, root_cells, *_ = records[0]
        current_cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in uv_mesh.blocks]
        )

        if triangle_tokens(
            np.asarray(uv_mesh.coordinates), current_cells
        ) != triangle_tokens(np.asarray(root_charts), np.asarray(root_cells)):
            raise ValueError(
                "SurfaceSourceFidelity.trim_cover must match restriction triangle content; got different chart triangles"
            )
        findings.extend(cover.findings)
    findings.extend(embedding.findings)
    return _UVPatchEvidence(tuple(findings))


@checked
def certify_original_surface_restriction_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    tolerance: float,
    order: int,
    limits: embedding_state_owner.MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> source_fidelity_owner.SourceFidelityCertificate | None:
    """Certify coordinate restriction equality and whole UV-domain coverage."""

    match source:
        case SurfaceNativeRestrictionBoundarySource():
            source.require_current()
        case _:
            return None
    binding.require(mesh, geometry)
    if source.mesh.mesh_id != mesh.mesh_id or cell_geometry_id(
        source.geometry
    ) != cell_geometry_id(geometry):
        raise ValueError(
            "SurfaceSourceFidelity.source must bind the published mesh and geometry; got different coordinate maps"
        )
    support = source.support
    proof = support.prove_native_restrictions(mesh, geometry, budget=budget)
    if np.any(proof.source_bounds != 0.0) or np.any(proof.restriction_deviations != 0.0):
        raise ValueError(
            "SurfaceSourceFidelity.proof must have zero source bounds and restriction deviations; got nonzero approximation error"
        )
    charts = _restriction_fidelity_charts(support, mesh, geometry, budget=budget).charts
    findings: list[embedding_state_owner.MeshCertificateFinding] = []
    for patch in source.root.patches:
        findings.extend(
            _certify_uv_patch(source, patch, charts, limits, budget=budget).findings
        )
    return source_fidelity_owner.SourceFidelityCertificate(
        binding,
        tuple(findings),
        tolerance=tolerance,
        bounds=source_fidelity_owner.SourceFidelityBounds(
            semantics=("certified", "certified"),
            mesh_to_source=(0.0, 0.0),
            source_to_mesh=(0.0, 0.0),
            sample_counts=(mesh.entity_set(2).count, support.root_cell_ids.shape[0]),
        ),
        sample_order=order,
    )


type SurfaceBoundSource = (
    SurfaceNativeRestrictionBoundarySource
    | domain_boundary_owner.MeshingDomainBoundarySource
    | surface_boundary_sources_owner.SurfaceChartBoundarySource
    | surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource
)


def require_surface_bound_source(
    source: SourceBoundaryQuery | None, name: str, /
) -> SurfaceBoundSource:
    """Return a supported nominal source graph without changing its identity."""
    match source:
        case (
            SurfaceNativeRestrictionBoundarySource()
            | domain_boundary_owner.MeshingDomainBoundarySource()
            | surface_boundary_sources_owner.SurfaceChartBoundarySource()
            | surface_boundary_sources_owner.SurfaceSubdivisionBoundarySource()
        ):
            return source
        case _:
            raise TypeError(
                f"{name} must be a surface source binding; got {type(source).__name__}"
            )


__all__ = [
    "require_surface_bound_source",
    "SurfaceBoundSource",
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

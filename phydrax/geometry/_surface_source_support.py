#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable source-chart banks and exact whole-map restrictions for surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import final, NamedTuple, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometryCoordinateMap,
    CellGeometrySpec,
    ScalarCoordinateElement,
    SplineCellGeometryElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    axes,
    compose_expression,
    constant,
    coordinate_corner_images,
    coordinate_map_expressions,
    coordinate_reference_chain,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    multiply,
    Polynomial,
    RationalPolynomial,
    rounded_point,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import coefficient_payload, RadicalElement
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._surface_chart_records import (
    _exact_chart_from_tokens,
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ._domain_boundary import MeshingDomainBoundarySource
from ._meshing_domain import MeshingDomain
from ._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundaryQuery,
    SourceBoundarySamples,
)


if TYPE_CHECKING:
    from ..meshing._association import GeometryAssociation
    from ..meshing._result import CellMeshingResult
    from .brep._patches import BSplineSurfacePatch
Corners = tuple[tuple[Fraction, Fraction], ...]
type _SurfaceChartToken = tuple[int, int]
type _SurfaceChartPoint = tuple[_SurfaceChartToken, _SurfaceChartToken]
type _SurfaceChartTriangle = tuple[
    _SurfaceChartPoint, _SurfaceChartPoint, _SurfaceChartPoint
]
type _SurfaceChartRationalBank = tuple[_SurfaceChartTriangle, ...]


def _fraction_coordinate(value: object, name: str, /) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an exact rational coordinate.")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    raise TypeError(f"{name} must be an exact rational coordinate.")


def _exact_uv(point: Sequence[object], name: str, /) -> tuple[Fraction, Fraction]:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates.")
    return (
        _fraction_coordinate(point[0], name),
        _fraction_coordinate(point[1], name),
    )


def _exact_coordinate_rows(
    values: Sequence[Sequence[object]], /
) -> tuple[tuple[ExactScalar, ...], ...]:
    rows: list[tuple[ExactScalar, ...]] = []
    for row in values:
        exact_row: list[ExactScalar] = []
        for value in row:
            if not isinstance(value, (Fraction, RadicalElement)):
                raise TypeError(
                    "Exact coordinate maps require rational or algebraic scalars."
                )
            exact_row.append(value)
        rows.append(tuple(exact_row))
    return tuple(rows)


def _exact_surface_chart_bank(
    realization: np.ndarray,
    count: int,
    supplied: Sequence[Corners] | None,
    /,
) -> tuple[Corners, ...]:
    from ..discretization._surface_chart_records import _area, exact_chart_bank

    if supplied is None:
        exact_rows = exact_chart_bank(realization, count)
    else:
        if len(supplied) != count:
            raise ValueError("Exact source charts must align with their cell rows.")
        rows: list[Corners] = []
        for row in supplied:
            if len(row) != 3:
                raise ValueError(
                    "Surface source charts must retain three exact UV corners."
                )
            points = tuple(_exact_uv(point, "Exact source UV point") for point in row)
            rows.append((points[0], points[1], points[2]))
        exact_rows = tuple(rows)
    rounded = np.asarray(
        [[[float(value) for value in point] for point in row] for row in exact_rows],
        dtype=np.float64,
    )
    if rounded.shape != realization.shape or not np.array_equal(rounded, realization):
        raise ValueError(
            "Exact source UV charts do not round to their retained float realization."
        )
    if any(not _area(row) for row in exact_rows):
        raise ValueError("Exact source chart triangles must be nondegenerate.")
    return tuple(exact_rows)


def _exact_reference_cell_vertices(cell_kind: str, /) -> Corners:
    reference = reference_cell_topology(cell_kind)
    values: list[tuple[Fraction, Fraction]] = []
    for point in reference.vertices:
        if len(point) != 2:
            raise ValueError("Surface reference vertices must be two-dimensional.")
        coordinates: list[Fraction] = []
        for value in point:
            if value == 0.0:
                coordinates.append(Fraction(0))
            elif value == 1.0:
                coordinates.append(Fraction(1))
            elif value == 0.5:
                coordinates.append(Fraction(1, 2))
            else:
                raise ValueError(
                    f"Reference cell {cell_kind!r} has a vertex without a canonical rational coordinate."
                )
        values.append((coordinates[0], coordinates[1]))
    return tuple(values)


def _root_reference_point(
    corners: Corners, point: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    if len(corners) == 3:
        origin, first, second = corners
        first_u = first[0] - origin[0]
        first_v = first[1] - origin[1]
        second_u = second[0] - origin[0]
        second_v = second[1] - origin[1]
        offset_u = point[0] - origin[0]
        offset_v = point[1] - origin[1]
        determinant = first_u * second_v - second_u * first_v
        if determinant == 0:
            raise ValueError("Surface root triangle chart must be nondegenerate.")
        local_u = (offset_u * second_v - second_u * offset_v) / determinant
        local_v = (first_u * offset_v - offset_u * first_v) / determinant
        if local_u < 0 or local_v < 0 or local_u + local_v > 1:
            raise ValueError(
                "Exact source UV point lies outside its root triangle chart."
            )
        return local_u, local_v
    if len(corners) != 4:
        raise ValueError(
            "A surface source root chart must have three or four UV corners."
        )
    lower_u, lower_v = corners[0]
    upper_u, _ = corners[1]
    opposite_u, upper_v = corners[2]
    _, opposite_v = corners[3]
    if (
        lower_u >= upper_u
        or lower_v >= upper_v
        or (opposite_u, opposite_v) != (upper_u, upper_v)
        or corners[3][0] != lower_u
        or corners[1][1] != lower_v
    ):
        raise ValueError(
            "Spline source root chart must be an exact axis-aligned rectangle."
        )
    if not (lower_u <= point[0] <= upper_u and lower_v <= point[1] <= upper_v):
        raise ValueError(
            "Exact source UV point lies outside its retained spline root chart."
        )
    return (
        (point[0] - lower_u) / (upper_u - lower_u),
        (point[1] - lower_v) / (upper_v - lower_v),
    )


def _exact_reference_expression(
    value: Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial, /
) -> Expression:
    return value if isinstance(value, RationalPolynomial) else dict(value)


def _same_tree_content(first: object, second: object, /) -> bool:
    if type(first) is not type(second):
        return False
    first_leaves, first_definition = jax.tree_util.tree_flatten(first)
    second_leaves, second_definition = jax.tree_util.tree_flatten(second)
    if first_definition != second_definition or len(first_leaves) != len(second_leaves):
        return False
    for left, right in zip(first_leaves, second_leaves, strict=True):
        if isinstance(left, (jax.Array, np.ndarray, np.generic)) or isinstance(
            right, (jax.Array, np.ndarray, np.generic)
        ):
            left_array, right_array = np.asarray(left), np.asarray(right)
            if (
                left_array.shape != right_array.shape
                or left_array.dtype != right_array.dtype
            ):
                return False
            try:
                if not np.array_equal(left_array, right_array, equal_nan=True):
                    return False
            except TypeError:
                if not np.array_equal(left_array, right_array):
                    return False
        else:
            try:
                equal = left == right
            except (TypeError, ValueError):
                return False
            if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
                return False
    return True


def validate_surface_source_pieces(
    domain: MeshingDomain,
    patches: ArrayLike,
    charts: ArrayLike,
    source_root_cell_ids: ArrayLike,
    source_piece_ids: Sequence[Sequence[int]],
    /,
    *,
    exact_charts: Sequence[Corners],
) -> None:
    """Bind every surface chart to one current exact source-piece identity."""
    if not isinstance(domain, MeshingDomain):
        raise TypeError("domain must be MeshingDomain.")
    domain.require_current(domain.source_id, domain.source_revision)
    patch_rows = np.asarray(patches)
    chart_rows = np.asarray(charts, dtype=np.float64)
    root_rows = np.asarray(source_root_cell_ids)
    pieces = tuple(tuple(row) for row in source_piece_ids)
    if (
        patch_rows.ndim != 1
        or patch_rows.dtype.kind not in "iu"
        or np.any(patch_rows < 0)
        or np.any(patch_rows >= len(domain.patches))
        or chart_rows.shape != (patch_rows.size, 3, 2)
        or not np.all(np.isfinite(chart_rows))
        or root_rows.shape != patch_rows.shape
        or root_rows.dtype.kind not in "iu"
        or np.any(root_rows < 0)
        or np.any(root_rows > np.iinfo(np.int64).max)
        or len(pieces) != patch_rows.size
        or any(
            not piece
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                or int(value) > np.iinfo(np.int64).max
                for value in piece
            )
            for piece in pieces
        )
    ):
        raise ValueError("Surface source-piece identities and chart rows must align.")
    exact_rows = _exact_surface_chart_bank(chart_rows, patch_rows.size, exact_charts)
    from ..meshing._plc_source import _numeric_entity_ids
    from .brep._patches import BSplineSurfacePatch
    from .brep._tspline import TSplineSurfacePatch

    source_patch_scope_ids = {}
    for scope_id in domain.scope_entity_ids(2).tolist():
        key = domain.source_entity_key_for_scope(2, int(scope_id))
        if key in source_patch_scope_ids:
            raise ValueError("Source patch scope identity resolves to multiple IDs.")
        source_patch_scope_ids[key] = int(scope_id)
    patch_keys = domain.stratum_entity_ids(2)
    source_patch_root_ids = np.asarray(
        [
            source_patch_scope_ids[
                (patch_keys[patch], domain.source_occurrences[2][patch])
            ]
            for patch in range(len(domain.patches))
        ],
        dtype=np.int64,
    )

    root_to_piece: dict[int, tuple[int, ...]] = {}
    piece_to_root: dict[tuple[int, ...], int] = {}
    checked_t_spline_patches: set[int] = set()
    for patch_value, chart, root_value, piece in zip(
        patch_rows.tolist(), exact_rows, root_rows.tolist(), pieces, strict=True
    ):
        patch = int(patch_value)
        root_cell_id = int(root_value)
        if piece[0] != patch:
            raise ValueError("A source-piece key must begin with its source patch ID.")
        previous_piece = root_to_piece.setdefault(root_cell_id, piece)
        previous_root = piece_to_root.setdefault(piece, root_cell_id)
        if previous_piece != piece or previous_root != root_cell_id:
            raise ValueError("Source root-cell and extraction-piece identities disagree.")
        surface = domain.patches[patch].surface
        bounds: tuple[Fraction, Fraction, Fraction, Fraction] | None
        if isinstance(surface, BSplineSurfacePatch):
            if len(piece) != 3:
                raise ValueError(
                    "Tensor spline source-piece keys must be (patch, u-span, v-span)."
                )
            u_span, v_span = piece[1:]
            if not (
                surface.u_degree <= u_span < surface.control_points.shape[0]
                and surface.v_degree <= v_span < surface.control_points.shape[1]
            ):
                raise ValueError(
                    "Tensor spline source-piece keys name invalid knot spans."
                )
            u_knots = surface.exact_u_knots
            v_knots = surface.exact_v_knots
            bounds = (
                u_knots[u_span],
                u_knots[u_span + 1],
                v_knots[v_span],
                v_knots[v_span + 1],
            )
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError(
                    "Tensor spline source-piece key names an empty knot span."
                )
        elif isinstance(surface, TSplineSurfacePatch):
            if len(piece) != 3:
                raise ValueError(
                    "T-spline source-piece keys must be (patch, control-cell-ID, extraction-piece-ID)."
                )
            if patch not in checked_t_spline_patches:
                surface.validate_source_bank()
                checked_t_spline_patches.add(patch)
            control_cell_id, extraction_piece_id = piece[1:]
            extraction = surface.realization.extraction_piece(extraction_piece_id)
            if extraction.cell_id != control_cell_id:
                raise ValueError(
                    "T-spline source-piece key changes its control-cell or extraction-piece ID."
                )
            extraction_bounds = extraction.parameter_bounds
            if len(extraction_bounds) != 4:
                raise ValueError(
                    "T-spline extraction bounds must contain four endpoints."
                )
            bounds = (
                _fraction_coordinate(extraction_bounds[0], "T-spline u lower bound"),
                _fraction_coordinate(extraction_bounds[1], "T-spline u upper bound"),
                _fraction_coordinate(extraction_bounds[2], "T-spline v lower bound"),
                _fraction_coordinate(extraction_bounds[3], "T-spline v upper bound"),
            )
        else:
            if len(piece) != 2 or piece[1] != int(domain.source_indices[2][patch]):
                raise ValueError(
                    "Non-spline source-piece keys must retain (patch, source-facet-ID)."
                )
            if root_cell_id != int(source_patch_root_ids[patch]):
                raise ValueError(
                    "Non-spline source roots must retain the canonical source-facet scope ID."
                )
            bounds = None
        if isinstance(surface, (BSplineSurfacePatch, TSplineSurfacePatch)):
            if bounds is None:
                raise ValueError(
                    "Spline source-piece bounds are required for exact validation."
                )
            spline_corners = (
                (bounds[0], bounds[2]),
                (bounds[1], bounds[2]),
                (bounds[1], bounds[3]),
                (bounds[0], bounds[3]),
            )
            entity_id, occurrence = domain.source_entity_key_for_scope(
                2, int(source_patch_root_ids[patch])
            )
            spline_key = canonical_fingerprint(
                {
                    "kind": "native-surface-root-cell",
                    "source": (domain.source_id, domain.source_revision),
                    "patch": (entity_id, occurrence),
                    "piece": piece,
                    "bounds": tuple(
                        (str(first), str(last)) for first, last in spline_corners
                    ),
                }
            )
            expected_root_id = _numeric_entity_ids(
                "native-surface-root-cell", (spline_key,), (occurrence,)
            )[0]
            if root_cell_id != expected_root_id:
                raise ValueError(
                    "Spline source-root ID must match its canonical patch and extraction-piece identity."
                )
        if bounds is not None and any(
            not (bounds[0] <= point[0] <= bounds[1])
            or not (bounds[2] <= point[1] <= bounds[3])
            for point in chart
        ):
            raise ValueError("A surface chart crosses its exact source-piece bounds.")


class SurfaceRestrictionRows(NamedTuple):
    """Exact target maps linked to root mesh cells and source spline pieces."""

    cell_ids: np.ndarray
    root_cell_ids: np.ndarray
    root_atlas_cell_ids: np.ndarray
    source_piece_ids: tuple[tuple[int, ...], ...]
    patches: np.ndarray
    vertex_ids: tuple[tuple[int, ...], ...]
    reference_corners: tuple[Corners, ...]
    source_corners: tuple[Corners, ...]
    source_bounds: np.ndarray
    coordinate_corner_errors: np.ndarray
    restriction_deviations: np.ndarray


@runtime_checkable
class SurfaceSourceCellAtlas(Protocol):
    """Current source-piece chart inventory consumed by surface construction."""

    root_cell_ids: tuple[int, ...]
    root_source_cell_ids: tuple[tuple[int, ...], ...]
    root_parameter_bounds: tuple[Corners, ...]
    root_patches: np.ndarray
    root_parameters: np.ndarray

    def require_current(self) -> None: ...
    def root_parameter_chart(self, row: int, /) -> Corners: ...


@final
class SurfaceSourceCharts(StrictModule, NonTrainableState):
    """Canonical source charts, extraction-piece ancestry, associations, and query."""

    domain: MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    root_parameters: np.ndarray
    root_patches: np.ndarray
    root_parameter_exact: tuple[Corners, ...] = eqx.field(static=True)
    root_parameter_rational_tokens: _SurfaceChartRationalBank = eqx.field(static=True)
    source_root_cell_ids: np.ndarray
    source_piece_ids: tuple[tuple[int, ...], ...]
    associations: tuple[GeometryAssociation, ...]
    boundary_source: MeshingDomainBoundarySource
    source_id: str
    source_revision: str
    cell_global_ids: tuple[int, ...]
    source_chart_id: str

    def __init__(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        root_parameters: ArrayLike,
        root_patches: ArrayLike,
        associations: tuple[GeometryAssociation, ...],
        boundary_source: MeshingDomainBoundarySource,
        /,
        *,
        source_root_cell_ids: ArrayLike,
        source_piece_ids: Sequence[Sequence[int]],
        root_parameter_exact: Sequence[Corners] | None = None,
    ) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "surface source charts require CellMesh and CellGeometrySpec."
            )
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError("SurfaceSourceCharts.mesh must be a surface in three-space.")
        if any(block.cell_kind != "triangle" for block in mesh.blocks):
            raise ValueError("SurfaceSourceCharts roots must be triangular cells.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "Surface source charts require an orthonormal Cartesian frame."
            )
        if not isinstance(boundary_source, MeshingDomainBoundarySource):
            raise TypeError("boundary_source must be MeshingDomainBoundarySource.")
        if not _same_tree_content(boundary_source.domain, domain):
            raise ValueError(
                "boundary_source must retain equivalent exact MeshingDomain content."
            )
        cell_ids = tuple(
            int(value)
            for block in mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        )
        if len(set(cell_ids)) != len(cell_ids) or any(value < 0 for value in cell_ids):
            raise ValueError(
                "SurfaceSourceCharts cell IDs must be unique and nonnegative."
            )
        parameters = np.asarray(root_parameters, dtype=np.float64)
        patches_raw = np.asarray(root_patches)
        if (
            parameters.shape != (len(cell_ids), 3, 2)
            or not np.all(np.isfinite(parameters))
            or patches_raw.shape != (len(cell_ids),)
            or patches_raw.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Root surface charts must align as finite (cells, 3, 2) UV triangles."
            )
        patches = np.asarray(patches_raw, dtype=np.int64)
        source_root_cell_ids_ = np.asarray(source_root_cell_ids)
        source_piece_ids_ = tuple(tuple(row) for row in source_piece_ids)
        exact_parameters = _exact_surface_chart_bank(
            parameters, len(cell_ids), root_parameter_exact
        )
        from ..discretization._surface_chart_records import rational_chart_tokens

        parameter_tokens = rational_chart_tokens(exact_parameters)
        if np.any(patches < 0) or np.any(patches >= len(domain.patches)):
            raise ValueError("Root chart patch IDs must name declared source patches.")
        validate_surface_source_pieces(
            domain,
            patches,
            parameters,
            source_root_cell_ids_,
            source_piece_ids_,
            exact_charts=exact_parameters,
        )
        if set(patches.tolist()) != set(boundary_source.patches):
            raise ValueError(
                "Root chart patches must equal the boundary source patch set."
            )
        from ..meshing._association import GeometryAssociation, GeometrySourceEntityRole

        association_values = tuple(associations)
        if not association_values or any(
            not isinstance(value, GeometryAssociation) for value in association_values
        ):
            raise TypeError("associations must be canonical GeometryAssociation values.")
        entity_sets = {
            mesh.entity_set(dimension).entity_set_id: tuple(
                int(value)
                for value in np.asarray(
                    mesh.entity_set(dimension).entity_ids, dtype=np.int64
                )
            )
            for dimension in range(mesh.topological_dimension + 1)
        }
        cells_by_id = {
            int(cell_id): int(patch)
            for cell_id, patch in zip(cell_ids, patches, strict=True)
        }
        assigned_surface_cells: set[int] = set()
        for association in association_values:
            if (
                association.source_id != domain.source_id
                or association.source_revision != domain.source_revision
                or association.target_entity_set_id not in entity_sets
            ):
                raise ValueError(
                    "Surface associations must bind this source revision and mesh entity set."
                )
            target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
            if np.setdiff1d(
                target_ids,
                np.asarray(entity_sets[association.target_entity_set_id], dtype=np.int64),
            ).size:
                raise ValueError(
                    "Surface association targets entities outside its declared mesh set."
                )
            if association.target_entity_set_id != mesh.entity_set(2).entity_set_id:
                continue
            dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
            source_indices = np.asarray(association.source_indices, dtype=np.int64)
            resolved = np.asarray(association.resolved, dtype=np.bool_)
            ambiguous = np.asarray(association.ambiguous, dtype=np.bool_)
            if not all(
                array.shape == target_ids.shape
                for array in (dimensions, source_indices, resolved, ambiguous)
            ):
                raise ValueError(
                    "Surface face association columns must align with target cell IDs."
                )
            for row, target_id in enumerate(target_ids.tolist()):
                patch = cells_by_id.get(int(target_id))
                if patch is None or not resolved[row] or ambiguous[row]:
                    raise ValueError(
                        "Every surface cell must have one resolved source-face association."
                    )
                if (
                    dimensions[row] != 2
                    or source_indices[row] != domain.source_indices[2][patch]
                    or association.source_occurrence_paths[row]
                    != domain.source_occurrences[2][patch]
                ):
                    raise ValueError(
                        "Surface face association changes source stratum or occurrence identity."
                    )
                role = association.source_roles[row]
                if role is not GeometrySourceEntityRole.FACET:
                    raise ValueError(
                        "Surface cell associations must retain the source facet role."
                    )
                if int(target_id) in assigned_surface_cells:
                    raise ValueError(
                        "Surface cell has duplicate source-face associations."
                    )
                assigned_surface_cells.add(int(target_id))
        if assigned_surface_cells != set(cell_ids):
            raise ValueError(
                "SurfaceSourceCharts requires an explicit source-face association for every cell."
            )
        domain.require_current(domain.source_id, domain.source_revision)
        parameters = np.array(parameters, copy=True)
        patches = np.array(patches, copy=True)
        parameters.setflags(write=False)
        patches.setflags(write=False)
        source_root_cell_ids_ = np.array(source_root_cell_ids_, dtype=np.int64, copy=True)
        source_root_cell_ids_.setflags(write=False)
        source_piece_ids_ = tuple(
            tuple(int(value) for value in row) for row in source_piece_ids_
        )
        self.domain = domain
        self.mesh = mesh
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.root_parameters = parameters
        self.root_patches = patches
        self.source_root_cell_ids = source_root_cell_ids_
        self.root_parameter_exact = exact_parameters
        self.root_parameter_rational_tokens = parameter_tokens
        self.source_piece_ids = source_piece_ids_
        self.associations = association_values
        self.boundary_source = boundary_source
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.cell_global_ids = cell_ids
        self.source_chart_id = self._identity()

    def root_parameter_chart(self, row: int, /) -> Corners:
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < len(self.root_parameter_exact)
        ):
            raise IndexError("surface root chart row is out of range.")
        return self.root_parameter_exact[row]

    @property
    def domain_id(self) -> str:
        return self.domain.domain_id

    @property
    def source_scope_id(self) -> str:
        return self.boundary_source.source_scope_id

    @property
    def source_patch_entity_ids(self) -> tuple[str, ...]:
        patch_keys = self.domain.stratum_entity_ids(2)
        return tuple(patch_keys[int(patch)] for patch in self.root_patches)

    @property
    def source_occurrence_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            self.domain.source_occurrences[2][int(patch)] for patch in self.root_patches
        )

    @property
    def chart_witness(self) -> SurfaceChartWitness:

        self.require_current()
        return SurfaceChartWitness(
            geometry_id=cell_geometry_id(self.geometry),
            topology_id=self.mesh.topology_id,
            domain_id=self.domain.domain_id,
            cell_global_ids=np.asarray(self.cell_global_ids, dtype=np.int64),
            patches=self.root_patches,
            charts=self.root_parameters,
            source_root_cell_ids=self.source_root_cell_ids,
            root_mesh_cell_bindings=tuple(
                SurfaceRootMeshCellBinding(
                    cell_geometry_id(self.geometry),
                    self.mesh.topology_id,
                    cell_id,
                )
                for cell_id in self.cell_global_ids
            ),
            source_piece_ids=self.source_piece_ids,
            geometry_entity_ids=self.source_patch_entity_ids,
            occurrence_paths=self.source_occurrence_paths,
            chart_rational_tokens=self.root_parameter_rational_tokens,
        )

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "surface-source-charts",
                "source": (self.domain.source_id, self.domain.source_revision),
                "domain": self.domain.domain_id,
                "mesh": self.mesh.mesh_id,
                "topology": self.mesh.topology_id,
                "geometry": cell_geometry_id(self.geometry),
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "cells": self.cell_global_ids,
                "parameters": array_tree_fingerprint(self.root_parameters),
                "exact_parameters": self.root_parameter_rational_tokens,
                "patches": array_tree_fingerprint(self.root_patches),
                "source_root_cells": array_tree_fingerprint(self.source_root_cell_ids),
                "source_pieces": self.source_piece_ids,
                "associations": tuple(
                    (
                        value.association_id,
                        array_tree_fingerprint(value),
                    )
                    for value in self.associations
                ),
                "boundary_scope": self.boundary_source.source_scope_id,
                "boundary_cover": self.boundary_source.domain.domain_id,
            }
        )

    def require_current(self) -> None:
        self.domain.require_current(self.source_id, self.source_revision)
        self.boundary_source.require_current()
        if self._identity() != self.source_chart_id:
            raise ValueError("SurfaceSourceCharts source identity or chart bank changed.")

    def require_root(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        self.require_current()
        if (
            not _same_tree_content(mesh, self.mesh)
            or not _same_tree_content(geometry, self.geometry)
            or mesh.topology_id != self.mesh.topology_id
            or cell_geometry_id(geometry) != cell_geometry_id(self.geometry)
        ):
            raise ValueError(
                "SurfaceSourceCharts root must retain its original mesh and coordinate map."
            )

    def require_bound(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        self.require_current()
        from ..meshing._surface_boundary_sources import (
            SurfaceChartBoundarySource,
            SurfaceSubdivisionBoundarySource,
        )

        if isinstance(source, MeshingDomainBoundarySource):
            source.require_current()
            if (
                not _same_tree_content(source.domain, self.domain)
                or source.source_scope_id != self.boundary_source.source_scope_id
                or source.patches != self.boundary_source.patches
                or not _same_tree_content(source, self.boundary_source)
            ):
                raise ValueError(
                    "Surface source query does not retain the chart source scope."
                )
            return
        if isinstance(source, SurfaceNativeRestrictionBoundarySource):
            source.require_current()
            support = source.support
            if (
                not _same_tree_content(source.root, self.boundary_source)
                or not _same_tree_content(source.mesh, mesh)
                or not _same_tree_content(source.geometry, geometry)
            ):
                raise ValueError(
                    "Native restriction query must bind this root scope and target map."
                )
        elif isinstance(source, SurfaceChartBoundarySource):
            source.require_target_binding(mesh, geometry)
            support = source.support
            if not _same_tree_content(source.root, self.boundary_source):
                raise ValueError(
                    "Chart deformation query must retain the exact root boundary source."
                )
        elif isinstance(source, SurfaceSubdivisionBoundarySource):
            source.require_current()
            support = source.support
            if (
                not _same_tree_content(source.root, self.boundary_source)
                or not source.targets
                or not _same_tree_content(source.targets[-1], mesh)
                or not _same_tree_content(source.geometry, geometry)
            ):
                raise ValueError(
                    "Subdivision query must retain the exact root and final target map."
                )
        else:
            raise TypeError("Unsupported surface source binding.")
        support.require_current()
        if not _same_tree_content(support.domain, self.domain):
            raise ValueError(
                "Descendant source support changes the retained MeshingDomain."
            )


@final
class _SplineSurfaceCoordinateMap(StrictModule, NonTrainableState):
    """Exact spline control bank bound to its rounded source-root mesh."""

    source_id: str
    source_revision: str
    rounded_vertices: np.ndarray
    exact_coordinates: tuple[tuple[ExactScalar, ...], ...]
    coefficient_keys: tuple[tuple[int, int], ...]
    source_map_id: str

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        rounded_vertices: ArrayLike,
        exact_coordinates: Sequence[Sequence[ExactScalar]],
        numeric_coordinates: ArrayLike,
        coefficient_keys: Sequence[tuple[int, int]],
        /,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        exact = tuple(tuple(row) for row in exact_coordinates)
        raw_keys = tuple(tuple(row) for row in coefficient_keys)
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        numeric = np.asarray(numeric_coordinates, dtype=np.float64)
        if not source or not revision or not exact or any(len(row) != 3 for row in exact):
            raise ValueError(
                "Exact spline coordinate rows require source identity and three components."
            )
        if any(
            not isinstance(value, (Fraction, RadicalElement))
            for row in exact
            for value in row
        ):
            raise TypeError("Exact spline coordinate banks require exact scalar values.")
        if len(raw_keys) != len(exact) or any(
            len(key) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in key
            )
            for key in raw_keys
        ):
            raise ValueError("Spline control identity keys must align with exact rows.")
        keys = tuple((int(row[0]), int(row[1])) for row in raw_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("Spline control identity keys must be unique.")
        try:
            exact_shadow = np.asarray(
                [[float(value) for value in row] for row in exact],
                dtype=np.float64,
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "Exact spline coordinates must have finite binary64 shadows."
            ) from error
        if (
            rounded.ndim != 2
            or rounded.shape[1] != 3
            or not rounded.shape[0]
            or not np.all(np.isfinite(rounded))
            or numeric.shape != exact_shadow.shape
            or not np.all(np.isfinite(numeric))
            or not np.all(np.isfinite(exact_shadow))
            or not np.array_equal(exact_shadow, numeric)
        ):
            raise ValueError(
                "Exact spline coordinate banks and their rounded numeric views must align."
            )
        payload = tuple(
            tuple(coefficient_payload(value) for value in row) for row in exact
        )
        rounded = np.array(rounded, copy=True)
        rounded.setflags(write=False)
        self.source_id = source
        self.source_revision = revision
        self.rounded_vertices = rounded
        self.exact_coordinates = exact
        self.coefficient_keys = keys
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "exact-spline-surface-coordinate-map",
                "source_id": source,
                "source_revision": revision,
                "rounded_vertices": array_tree_fingerprint(rounded),
                "exact_coordinates": payload,
                "coefficient_keys": keys,
            }
        )

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[tuple[ExactScalar, ...], ...]:
        if indices is None:
            rows = range(len(self.exact_coordinates))
        else:
            requested = np.asarray(indices)
            if (
                requested.ndim != 1
                or requested.dtype.kind not in "iu"
                or np.any(requested < 0)
                or np.any(requested >= len(self.exact_coordinates))
            ):
                raise IndexError(
                    "Spline coordinate source index is outside its exact bank."
                )
            rows = (int(value) for value in requested)
        return tuple(self.exact_coordinates[row] for row in rows)


@final
class SurfaceSourceRootAtlas(StrictModule, NonTrainableState):
    """Exact spline source-piece atlas with validated root maps."""

    domain: MeshingDomain
    mesh: CellMesh
    geometry: CellGeometrySpec
    root_parameters: np.ndarray
    root_patches: np.ndarray
    root_cell_ids: tuple[int, ...]
    root_source_cell_ids: tuple[tuple[int, ...], ...]
    root_parameter_bounds: tuple[Corners, ...]
    coordinate_contract: SpatialCoordinateContract
    source_id: str
    source_revision: str
    atlas_id: str

    def __init__(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root_parameters: ArrayLike,
        root_patches: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        root_source_cell_ids: tuple[tuple[int, ...], ...],
        root_parameter_bounds: tuple[Corners, ...],
    ) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        domain.require_current(domain.source_id, domain.source_revision)
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "SurfaceSourceRootAtlas requires CellMesh and CellGeometrySpec."
            )
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise ValueError(
                "SurfaceSourceRootAtlas.mesh must be a surface in three-space."
            )
        if (
            not isinstance(coordinate_contract, SpatialCoordinateContract)
            or coordinate_contract.coordinate_system != "cartesian"
        ):
            raise ValueError(
                "SurfaceSourceRootAtlas requires an orthonormal Cartesian coordinate contract."
            )
        cell_ids = tuple(
            int(value)
            for block in mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        )
        if len(set(cell_ids)) != len(cell_ids) or any(value < 0 for value in cell_ids):
            raise ValueError(
                "Surface root cell global IDs must be unique and nonnegative."
            )
        parameters = np.asarray(root_parameters, dtype=np.float64)
        patches_raw = np.asarray(root_patches)
        if (
            parameters.shape != (len(cell_ids), 4, 2)
            or not np.all(np.isfinite(parameters))
            or patches_raw.shape != (len(cell_ids),)
            or patches_raw.dtype.kind not in "iu"
            or len(root_source_cell_ids) != len(cell_ids)
            or len(root_parameter_bounds) != len(cell_ids)
        ):
            raise ValueError("Surface root atlas banks must align by root cell ID.")
        patches = np.asarray(patches_raw, dtype=np.int64)
        if np.any(patches < 0) or np.any(patches >= len(domain.patches)):
            raise ValueError("Surface root atlas patch IDs must name declared patches.")
        raw_pieces = tuple(tuple(row) for row in root_source_cell_ids)
        if any(
            not row
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in row
            )
            for row in raw_pieces
        ):
            raise ValueError(
                "Each root cell must retain a nonempty nonnegative source-piece identity."
            )
        pieces = tuple(tuple(int(value) for value in row) for row in raw_pieces)
        if any(
            piece[0] != int(patch) for piece, patch in zip(pieces, patches, strict=True)
        ):
            raise ValueError(
                "Each root source-piece identity must name its retained source patch."
            )
        exact_bounds: list[Corners] = []
        for row in root_parameter_bounds:
            corners: list[tuple[Fraction, Fraction]] = []
            for point in row:
                corners.append(_exact_uv(point, "Spline root UV corner"))
            exact_bounds.append(tuple(corners))
        bounds = tuple(exact_bounds)
        if any(len(row) != 4 for row in bounds):
            raise ValueError(
                "Spline root parameter bounds must retain four ordered UV corners."
            )
        for row, root_corners in enumerate(bounds):
            expected = np.asarray(
                [[float(u), float(v)] for u, v in root_corners], dtype=np.float64
            )
            if not np.array_equal(parameters[row], expected):
                raise ValueError(
                    "Root parameter float bank must realize its exact rational corners."
                )
        _validate_root_atlas(domain, mesh, geometry, parameters, patches, pieces, bounds)
        parameters = np.array(parameters, copy=True)
        patches = np.array(patches, copy=True)
        parameters.setflags(write=False)
        patches.setflags(write=False)
        self.domain = domain
        self.mesh = mesh
        self.geometry = geometry
        self.root_parameters = parameters
        self.root_patches = patches
        self.root_cell_ids = cell_ids
        self.root_source_cell_ids = pieces
        self.root_parameter_bounds = bounds
        self.coordinate_contract = coordinate_contract
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.atlas_id = self._identity()

    @property
    def domain_id(self) -> str:
        return self.domain.domain_id

    def root_parameter_chart(self, row: int, /) -> Corners:
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < len(self.root_parameter_bounds)
        ):
            raise IndexError("surface source root chart row is out of range.")
        return self.root_parameter_bounds[row]

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "surface-source-root-atlas",
                "domain": self.domain.domain_id,
                "source": (self.source_id, self.source_revision),
                "mesh": self.mesh.mesh_id,
                "topology": self.mesh.topology_id,
                "geometry": cell_geometry_id(self.geometry),
                "root_cells": self.root_cell_ids,
                "source_pieces": self.root_source_cell_ids,
                "bounds": tuple(
                    tuple((str(u), str(v)) for u, v in corners)
                    for corners in self.root_parameter_bounds
                ),
                "parameters": array_tree_fingerprint(self.root_parameters),
                "patches": array_tree_fingerprint(self.root_patches),
                "coordinate_contract": self.coordinate_contract.spatial_id,
            }
        )

    def require_current(self) -> None:
        self.domain.require_current(self.source_id, self.source_revision)
        if self._identity() != self.atlas_id:
            raise ValueError(
                "SurfaceSourceRootAtlas source or root-chart identity changed."
            )


@final
class PreparedSurfaceSourceSupport(StrictModule, NonTrainableState):
    """Current root chart bank and exact restriction verifier for surface cells."""

    domain: MeshingDomain
    coordinate_contract: SpatialCoordinateContract
    original: CellMeshingResult | SurfaceSourceRootAtlas
    root_boundary_source: MeshingDomainBoundarySource
    root_cell_ids: tuple[int, ...]
    root_parameters: np.ndarray
    root_patches: np.ndarray
    root_corner_ids: np.ndarray
    root_coordinates: np.ndarray
    root_geometry_id: str
    root_topology_id: str
    root_source_cell_ids: tuple[tuple[int, ...], ...]
    root_atlas_cell_ids: tuple[int, ...]
    root_parameter_bounds: tuple[Corners, ...]
    support_id: str

    def __init__(
        self,
        domain: MeshingDomain,
        root_result: CellMeshingResult | SurfaceSourceRootAtlas,
        root_cell_parameters: ArrayLike,
        root_boundary_source: MeshingDomainBoundarySource,
        /,
    ) -> None:
        if not isinstance(domain, MeshingDomain):
            raise TypeError("domain must be MeshingDomain.")
        if not isinstance(root_boundary_source, MeshingDomainBoundarySource):
            raise TypeError("root_boundary_source must be MeshingDomainBoundarySource.")
        root_boundary_source.require_current()
        if not _same_tree_content(root_boundary_source.domain, domain):
            raise ValueError(
                "Root boundary query must retain equivalent source-domain content."
            )
        if isinstance(root_result, SurfaceSourceRootAtlas):
            root_result.require_current()
            root_mesh, root_geometry = root_result.mesh, root_result.geometry
            root_parameters = np.asarray(root_result.root_parameters, dtype=np.float64)
            root_patches = np.asarray(root_result.root_patches, dtype=np.int64)
            root_ids = root_result.root_cell_ids
            root_atlas_ids = root_result.root_cell_ids
            root_source_ids = root_result.root_source_cell_ids
            bounds = tuple(map(root_result.root_parameter_chart, range(len(root_ids))))
            coordinate_contract = root_result.coordinate_contract
        else:
            from ..meshing._result import CellMeshingResult

            if not isinstance(root_result, CellMeshingResult):
                raise TypeError(
                    "root_result must be CellMeshingResult or SurfaceSourceRootAtlas."
                )
            root_mesh, root_geometry = root_result.mesh, root_result.geometry
            source_charts = root_result.surface_source
            if not isinstance(source_charts, SurfaceSourceCharts):
                raise ValueError("Accepted root result must retain SurfaceSourceCharts.")
            source_charts.require_current()
            source_charts.require_root(root_mesh, root_geometry)
            if not _same_tree_content(source_charts.domain, domain):
                raise ValueError(
                    "SurfaceSourceCharts must retain equivalent exact support-domain content."
                )
            root_parameters = np.asarray(source_charts.root_parameters, dtype=np.float64)
            root_patches = np.asarray(source_charts.root_patches, dtype=np.int64)
            root_ids = source_charts.cell_global_ids
            root_source_ids = source_charts.source_piece_ids
            root_atlas_ids = tuple(
                int(value) for value in source_charts.source_root_cell_ids
            )
            bounds = tuple(map(source_charts.root_parameter_chart, range(len(root_ids))))
            coordinate_contract = source_charts.coordinate_contract
        if len(root_atlas_ids) != len(root_ids):
            raise ValueError(
                "Root atlas cell IDs must align with retained root mesh cells."
            )
        supplied = np.asarray(root_cell_parameters, dtype=np.float64)
        if supplied.shape != root_parameters.shape or not np.array_equal(
            supplied, root_parameters
        ):
            raise ValueError(
                "root_cell_parameters must exactly match the retained source chart bank."
            )
        if set(root_patches.tolist()) != set(root_boundary_source.patches):
            raise ValueError(
                "Root boundary query must cover exactly the retained source patches."
            )
        root_vertex_ids = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
        root_corner_ids = np.concatenate(
            [
                root_vertex_ids[np.asarray(block.vertices, dtype=np.int64)]
                for block in root_mesh.blocks
            ],
            axis=0,
        )
        root_coordinates = np.asarray(root_mesh.coordinates, dtype=np.float64)
        root_geometry_id = cell_geometry_id(root_geometry)
        root_topology_id = root_mesh.topology_id
        for array in (root_parameters, root_patches, root_corner_ids, root_coordinates):
            array = np.array(array, copy=True)
            array.setflags(write=False)
        self.domain = domain
        self.coordinate_contract = coordinate_contract
        self.original = root_result
        self.root_boundary_source = root_boundary_source
        self.root_cell_ids = tuple(root_ids)
        self.root_atlas_cell_ids = tuple(root_atlas_ids)
        self.root_parameters = np.array(root_parameters, copy=True)
        self.root_patches = np.array(root_patches, copy=True)
        self.root_corner_ids = np.array(root_corner_ids, copy=True)
        self.root_coordinates = np.array(root_coordinates, copy=True)
        for array in (
            self.root_parameters,
            self.root_patches,
            self.root_corner_ids,
            self.root_coordinates,
        ):
            array.setflags(write=False)
        self.root_geometry_id = root_geometry_id
        self.root_topology_id = root_topology_id
        self.root_source_cell_ids = tuple(root_source_ids)
        self.root_parameter_bounds = tuple(bounds)
        self.support_id = self._identity()

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "prepared-surface-source-support",
                "domain": self.domain.domain_id,
                "source": (self.domain.source_id, self.domain.source_revision),
                "root_mesh": self.root_topology_id,
                "root_geometry": self.root_geometry_id,
                "root_cells": self.root_cell_ids,
                "root_atlas_cells": self.root_atlas_cell_ids,
                "root_source_pieces": self.root_source_cell_ids,
                "root_parameters": array_tree_fingerprint(self.root_parameters),
                "root_parameter_bounds": tuple(
                    tuple((str(u), str(v)) for u, v in chart)
                    for chart in self.root_parameter_bounds
                ),
                "root_patches": array_tree_fingerprint(self.root_patches),
                "root_vertices": array_tree_fingerprint(self.root_corner_ids),
                "root_coordinates": array_tree_fingerprint(self.root_coordinates),
                "boundary": self.root_boundary_source.source_scope_id,
            }
        )

    def require_current(self) -> None:
        self.domain.require_current(self.domain.source_id, self.domain.source_revision)
        self.root_boundary_source.require_current()
        if isinstance(self.original, SurfaceSourceRootAtlas):
            self.original.require_current()
        else:
            source_charts = self.original.surface_source
            if source_charts is None:
                raise ValueError("Accepted surface root has no retained source charts.")
            source_charts.require_current()
        if self._identity() != self.support_id:
            raise ValueError(
                "PreparedSurfaceSourceSupport source or root charts changed."
            )

    def require_root(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        self.require_current()
        original_mesh = self.original.mesh
        original_geometry = self.original.geometry
        if (
            not _same_tree_content(mesh, original_mesh)
            or not _same_tree_content(geometry, original_geometry)
            or mesh.topology_id != self.root_topology_id
            or cell_geometry_id(geometry) != self.root_geometry_id
        ):
            raise ValueError(
                "Surface source root must retain its actual root mesh and geometry."
            )

    def prove_native_restrictions(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        chart_witness: SurfaceChartWitness | None = None,
    ) -> SurfaceRestrictionRows:
        """Verify every target whole map is an exact retained root-cell restriction."""
        self.require_current()
        root_mesh = self.original.mesh
        root_geometry = self.original.geometry
        root_id_to_row = {
            identifier: row for row, identifier in enumerate(self.root_cell_ids)
        }
        witness_row_by_cell: dict[int, int] = {}
        witness_root_row_by_cell: dict[int, int] = {}
        if chart_witness is not None:
            if (
                not isinstance(chart_witness, SurfaceChartWitness)
                or chart_witness.geometry_id != cell_geometry_id(geometry)
                or chart_witness.topology_id != mesh.topology_id
                or chart_witness.domain_id != self.domain.domain_id
            ):
                raise ValueError(
                    "Surface chart witness must bind the exact target geometry, topology, and domain."
                )
            chart_cell_ids = np.asarray(chart_witness.cell_global_ids, dtype=np.int64)
            target_cell_ids = tuple(
                int(value)
                for block in mesh.blocks
                for value in np.asarray(block.global_ids, dtype=np.int64)
            )
            chart_roots = np.asarray(chart_witness.source_root_cell_ids, dtype=np.int64)
            chart_root_bindings = tuple(chart_witness.root_mesh_cell_bindings)
            chart_patches = np.asarray(chart_witness.patches, dtype=np.int64)
            exact_charts = tuple(
                map(chart_witness.exact_chart, range(chart_cell_ids.size))
            )
            if (
                chart_cell_ids.shape != (len(target_cell_ids),)
                or len(set(chart_cell_ids.tolist())) != chart_cell_ids.size
                or set(chart_cell_ids.tolist()) != set(target_cell_ids)
                or chart_roots.shape != chart_cell_ids.shape
                or len(chart_root_bindings) != chart_cell_ids.size
                or chart_patches.shape != chart_cell_ids.shape
            ):
                raise ValueError(
                    "Surface chart witness must cover each target cell by its global ID."
                )
            validate_surface_source_pieces(
                self.domain,
                chart_patches,
                np.asarray(chart_witness.charts, dtype=np.float64),
                chart_roots,
                chart_witness.source_piece_ids,
                exact_charts=exact_charts,
            )
            patch_keys = self.domain.stratum_entity_ids(2)
            for row, cell_id in enumerate(chart_cell_ids.tolist()):
                binding = chart_root_bindings[row]
                if binding is None:
                    raise ValueError(
                        "A descendant restriction requires an explicit root CellMesh binding."
                    )
                if (
                    binding.root_geometry_id != self.root_geometry_id
                    or binding.root_topology_id != self.root_topology_id
                ):
                    raise ValueError(
                        "Surface chart root-mesh binding names a different root geometry or topology."
                    )
                root_row = root_id_to_row.get(binding.root_cell_global_id)
                patch = int(chart_patches[row])
                if (
                    root_row is None
                    or int(chart_roots[row]) != self.root_atlas_cell_ids[root_row]
                    or chart_witness.source_piece_ids[row]
                    != self.root_source_cell_ids[root_row]
                    or patch != int(self.root_patches[root_row])
                    or chart_witness.geometry_entity_ids[row] != patch_keys[patch]
                    or chart_witness.occurrence_paths[row]
                    != self.domain.source_occurrences[2][patch]
                ):
                    raise ValueError(
                        "Surface chart witness changes root-mesh, atlas-cell, extraction-piece, entity, or occurrence identity."
                    )
                witness_row_by_cell[int(cell_id)] = row
                witness_root_row_by_cell[int(cell_id)] = root_row
        if _same_tree_content(mesh, root_mesh) and _same_tree_content(
            geometry, root_geometry
        ):
            cell_ids: list[int] = []
            root_atlas_cell_ids: list[int] = []
            source_piece_ids: list[tuple[int, ...]] = []
            vertex_ids: list[tuple[int, ...]] = []
            reference_corners: list[Corners] = []
            root_row_by_id = {
                identifier: row for row, identifier in enumerate(self.root_cell_ids)
            }
            for block in mesh.blocks:
                reference = _exact_reference_cell_vertices(block.cell_kind)
                for identifier, vertices in zip(
                    np.asarray(block.global_ids, dtype=np.int64),
                    np.asarray(block.vertices, dtype=np.int64),
                    strict=True,
                ):
                    cell_id = int(identifier)
                    root_row = root_row_by_id[cell_id]
                    if chart_witness is not None and (
                        _exact_chart_from_tokens(
                            chart_witness.chart_rational_tokens,
                            witness_row_by_cell[cell_id],
                        )
                        != self.root_parameter_bounds[root_row]
                    ):
                        raise ValueError(
                            "Root SurfaceChartWitness does not reproduce the canonical root source chart."
                        )
                    cell_ids.append(cell_id)
                    root_atlas_cell_ids.append(self.root_atlas_cell_ids[root_row])
                    source_piece_ids.append(self.root_source_cell_ids[root_row])
                    vertex_ids.append(
                        tuple(
                            int(value)
                            for value in np.asarray(mesh.vertex_global_ids)[vertices]
                        )
                    )
                    reference_corners.append(reference)
            source_corners = self.root_parameter_bounds
            return _surface_restriction_rows(
                tuple(cell_ids),
                tuple(cell_ids),
                tuple(root_atlas_cell_ids),
                tuple(source_piece_ids),
                self.root_patches,
                tuple(vertex_ids),
                tuple(reference_corners),
                source_corners,
            )
        restriction = geometry.restriction_source
        if restriction is None and chart_witness is None:
            raise ValueError(
                "Descendant surface maps require an exact restriction or canonical source chart witness."
            )
        if restriction is not None and (
            restriction.source_geometry_id != self.root_geometry_id
            or restriction.source_topology_id != self.root_topology_id
        ):
            raise ValueError(
                "Surface restriction source IDs do not match retained root geometry and topology."
            )
        root_elements, root_routes, _ = root_geometry.resolve(root_mesh)
        root_coefficients = _exact_coordinate_rows(root_geometry.source_coordinates())
        root_locations: dict[
            int,
            tuple[
                int,
                int,
                ScalarCoordinateElement,
                tuple[tuple[ExactScalar, ...], ...],
            ],
        ] = {}
        for block_index, (root_block, root_element, root_route) in enumerate(
            zip(root_mesh.blocks, root_elements, root_routes, strict=True)
        ):
            if not isinstance(root_element, ScalarCoordinateElement):
                raise TypeError(
                    "Surface root geometry requires a scalar coordinate element."
                )
            root_ids = np.asarray(root_block.global_ids, dtype=np.int64)
            route_rows = np.asarray(root_route, dtype=np.int64)
            if (
                root_ids.ndim != 1
                or route_rows.ndim != 2
                or route_rows.shape[0] != root_ids.size
                or route_rows.shape[1] != root_element.local_dof_count
            ):
                raise ValueError(
                    "Surface root DOF routes must align with their cell rows."
                )
            for root_row in range(root_ids.size):
                dofs = tuple(
                    int(route_rows[root_row, local_index])
                    for local_index in range(route_rows.shape[1])
                )
                root_local = tuple(root_coefficients[index] for index in dofs)
                root_locations[int(root_ids[root_row])] = (
                    block_index,
                    root_row,
                    root_element,
                    root_local,
                )
        target_elements, target_routes, _ = geometry.resolve(mesh)
        target_coefficients = _exact_coordinate_rows(geometry.source_coordinates())
        cell_ids: list[int] = []
        root_cell_ids: list[int] = []
        root_atlas_cell_ids: list[int] = []
        source_piece_ids: list[tuple[int, ...]] = []
        patches: list[int] = []
        target_vertex_ids: list[tuple[int, ...]] = []
        reference_corners: list[Corners] = []
        source_corners: list[Corners] = []
        source_bounds: list[tuple[float, float, float, float]] = []
        root_block_names = tuple(block.name for block in root_mesh.blocks)
        parent_cells = {} if restriction is None else restriction.block_parent_cell_ids
        parent_vertices = (
            {} if restriction is None else restriction.block_parent_vertex_ids
        )
        parent_reference_maps = (
            {} if restriction is None else restriction.block_parent_reference_maps
        )
        for block, element_candidate, route in zip(
            mesh.blocks, target_elements, target_routes, strict=True
        ):
            if not isinstance(element_candidate, ScalarCoordinateElement):
                raise TypeError(
                    "Surface target geometry requires a scalar coordinate element."
                )
            element = element_candidate
            cell_global_ids = np.asarray(block.global_ids, dtype=np.int64)
            route_rows = np.asarray(route, dtype=np.int64)
            vertex_rows = np.asarray(block.vertices, dtype=np.int64)
            if (
                cell_global_ids.ndim != 1
                or route_rows.ndim != 2
                or route_rows.shape[0] != cell_global_ids.size
                or route_rows.shape[1] != element.local_dof_count
                or vertex_rows.ndim != 2
                or vertex_rows.shape[0] != cell_global_ids.size
            ):
                raise ValueError("Surface target routes must align with their cell rows.")
            parent_ids = (
                None
                if restriction is None
                else np.asarray(parent_cells[block.name], dtype=np.int64)
            )
            if parent_ids is not None and parent_ids.shape != cell_global_ids.shape:
                raise ValueError("Surface parent-cell IDs must align with target cells.")
            source_block: str | None = None
            if restriction is not None:
                source_block = block.name
                if restriction.source_block_names is not None:
                    source_block = restriction.source_block_names[
                        restriction.block_names.index(block.name)
                    ]
                if source_block not in root_block_names:
                    raise ValueError("Surface restriction names an absent root block.")
            reference_map = (
                None if restriction is None else parent_reference_maps[block.name]
            )
            for row_index in range(cell_global_ids.size):
                cell_id = int(cell_global_ids[row_index])
                dofs = tuple(
                    int(route_rows[row_index, local_index])
                    for local_index in range(route_rows.shape[1])
                )
                target_vertices = vertex_rows[row_index]
                witness_row: int | None = None
                if chart_witness is None:
                    if parent_ids is None:
                        raise ValueError("Surface restrictions require root-cell IDs.")
                    parent = int(parent_ids[row_index])
                else:
                    witness_row = witness_row_by_cell[cell_id]
                    binding = chart_root_bindings[witness_row]
                    if binding is None:
                        raise ValueError(
                            "A descendant restriction requires an explicit root CellMesh binding."
                        )
                    parent = binding.root_cell_global_id
                    if restriction is not None:
                        if parent_ids is None:
                            raise ValueError(
                                "Surface restriction parent-cell IDs are absent."
                            )
                        if parent != int(parent_ids[row_index]):
                            raise ValueError(
                                "Surface restriction parent differs from its canonical root-mesh binding."
                            )
                root_row_in_bank = root_id_to_row.get(parent)
                if root_row_in_bank is None:
                    raise ValueError(
                        "Surface ancestry names an absent root source chart."
                    )
                if (
                    chart_witness is not None
                    and witness_root_row_by_cell[cell_id] != root_row_in_bank
                ):
                    raise ValueError(
                        "Surface root-mesh binding resolves inconsistently by global cell ID."
                    )
                if parent not in root_locations:
                    raise ValueError(
                        "Surface restriction names an absent explicit root cell ID."
                    )
                root_block_index, root_row, root_element, root_local = root_locations[
                    parent
                ]
                if (
                    source_block is not None
                    and root_mesh.blocks[root_block_index].name != source_block
                ):
                    raise ValueError(
                        "Surface restriction source block differs from its actual root cell."
                    )
                if restriction is None:
                    if chart_witness is None or witness_row is None:
                        raise ValueError(
                            "Descendant restriction requires an exact canonical source chart."
                        )
                    root_corners = tuple(
                        _root_reference_point(
                            self.root_parameter_bounds[root_row_in_bank], point
                        )
                        for point in _exact_chart_from_tokens(
                            chart_witness.chart_rational_tokens, witness_row
                        )
                    )
                    arguments = _reference_chart_arguments(root_corners, block.cell_kind)
                elif (
                    reference_map is not None
                    and reference_map.source_reference_chart_arguments is not None
                ):
                    if (
                        reference_map.target_cell_kind != block.cell_kind
                        or reference_map.source_cell_kind
                        != root_mesh.blocks[root_block_index].cell_kind
                    ):
                        raise ValueError(
                            "Exact surface ancestry map cell kinds do not match the target and root blocks."
                        )
                    arguments = tuple(
                        _exact_reference_expression(value)
                        for value in reference_map.source_reference_chart_arguments[
                            row_index
                        ]
                    )
                    if affine_map(arguments, 2) is None:
                        raise ValueError(
                            "Surface source charts require an exact affine retained reference map."
                        )
                else:
                    root_owner, arguments = coordinate_reference_chain(
                        element,
                        ancestor=root_element,
                        cell_row=row_index,
                    )
                    if root_owner is not root_element:
                        raise ValueError(
                            "Restricted coordinate element names a different root basis."
                        )
                target_local = tuple(target_coefficients[index] for index in dofs)
                root_coordinates = coordinate_map_expressions(
                    root_element, root_local, cell_row=root_row
                )
                target_coordinates = coordinate_map_expressions(
                    element, target_local, cell_row=row_index
                )
                composed = tuple(
                    _compose_expression(value, arguments) for value in root_coordinates
                )
                if len(composed) != len(target_coordinates) or any(
                    not expression_equal(first, second)
                    for first, second in zip(composed, target_coordinates, strict=True)
                ):
                    raise ValueError(
                        "Surface target coordinate map is not the exact retained root restriction."
                    )
                if restriction is not None:
                    parent_vertex_row = np.asarray(
                        parent_vertices[block.name][row_index], dtype=np.int64
                    )
                    valid_parent_vertices = parent_vertex_row[parent_vertex_row >= 0]
                    root_vertices = np.asarray(
                        root_mesh.vertex_global_ids[
                            np.asarray(root_mesh.blocks[root_block_index].vertices)[
                                root_row
                            ]
                        ],
                        dtype=np.int64,
                    )
                    if np.setdiff1d(valid_parent_vertices, root_vertices).size:
                        raise ValueError(
                            "Restriction parent vertex IDs must belong to the declared source root cell."
                        )
                reference_vertices = _exact_reference_cell_vertices(block.cell_kind)
                cell_root_corners = tuple(
                    tuple(evaluate_expression(argument, point) for argument in arguments)
                    for point in reference_vertices
                )
                root_parameter_corners = self.root_parameter_bounds[root_row_in_bank]
                root_parameter_kind = root_mesh.blocks[root_block_index].cell_kind
                source_uv = tuple(
                    _root_parameter_point(
                        root_parameter_corners, root_parameter_kind, point
                    )
                    for point in cell_root_corners
                )
                if chart_witness is not None and (
                    witness_row is None
                    or source_uv
                    != _exact_chart_from_tokens(
                        chart_witness.chart_rational_tokens, witness_row
                    )
                ):
                    raise ValueError(
                        "Retained target geometry restriction disagrees with its canonical source chart."
                    )
                cell_ids.append(cell_id)
                root_cell_ids.append(parent)
                root_atlas_cell_ids.append(self.root_atlas_cell_ids[root_row_in_bank])
                source_piece_ids.append(self.root_source_cell_ids[root_row_in_bank])
                patches.append(int(self.root_patches[root_row_in_bank]))
                target_vertex_ids.append(
                    tuple(
                        int(value)
                        for value in np.asarray(mesh.vertex_global_ids)[target_vertices]
                    )
                )
                reference_corners.append(
                    tuple(
                        _exact_uv(point, "Root reference chart corner")
                        for point in cell_root_corners
                    )
                )
                source_corners.append(source_uv)
                source_bounds.append(
                    (
                        float(min(point[0] for point in source_uv)),
                        float(max(point[0] for point in source_uv)),
                        float(min(point[1] for point in source_uv)),
                        float(max(point[1] for point in source_uv)),
                    )
                )
        return _surface_restriction_rows(
            tuple(cell_ids),
            tuple(root_cell_ids),
            tuple(root_atlas_cell_ids),
            tuple(source_piece_ids),
            np.asarray(patches, dtype=np.int64),
            tuple(target_vertex_ids),
            tuple(reference_corners),
            tuple(source_corners),
            np.asarray(source_bounds, dtype=np.float64),
        )


@final
class SurfaceNativeRestrictionBoundarySource(StrictModule, NonTrainableState):
    """A source query that retains exact root restrictions for a live surface map."""

    support: PreparedSurfaceSourceSupport
    root: MeshingDomainBoundarySource
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_id: str
    source_revision: str

    def __init__(
        self,
        root: MeshingDomainBoundarySource,
        support: PreparedSurfaceSourceSupport,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
    ) -> None:
        support.require_current()
        if (
            not _same_tree_content(root.domain, support.domain)
            or root.patches != support.root_boundary_source.patches
        ):
            raise ValueError(
                "Native restriction source must retain equivalent support domain and patch identities."
            )
        if not isinstance(geometry, CellGeometrySpec) or not isinstance(mesh, CellMesh):
            raise TypeError(
                "Native restriction source requires CellMesh and CellGeometrySpec."
            )
        support.prove_native_restrictions(mesh, geometry)
        self.root, self.support, self.mesh, self.geometry = root, support, mesh, geometry
        self.source_id, self.source_revision = root.source_id, root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    @property
    def domain(self) -> MeshingDomain:
        return self.support.domain

    @property
    def source_scope_id(self) -> str:
        return self.root.source_scope_id

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        return self.root.boundary_cover(maximum_radius)

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover:
        return self.root.boundary_chart_cover()

    def require_current(self) -> None:
        self.support.require_current()
        self.root.require_current()
        self.support.prove_native_restrictions(self.mesh, self.geometry)


def _compose_expression(
    expression: Expression, arguments: tuple[Expression, ...], /
) -> Expression:
    return compose_expression(expression, arguments)


def _reference_chart_arguments(
    corners: Corners, cell_kind: str, /
) -> tuple[Polynomial, ...]:
    variables = axes(2)
    if cell_kind == "triangle" and len(corners) == 3:
        origin = corners[0]
        return tuple(
            sum_polynomials(
                (
                    constant(origin[axis], 2),
                    scale(variables[0], corners[1][axis] - origin[axis]),
                    scale(variables[1], corners[2][axis] - origin[axis]),
                ),
                2,
            )
            for axis in range(2)
        )
    if cell_kind == "quadrilateral" and len(corners) == 4:
        origin = corners[0]
        mixed = tuple(
            corners[0][axis] - corners[1][axis] + corners[2][axis] - corners[3][axis]
            for axis in range(2)
        )
        product = multiply(variables[0], variables[1])
        return tuple(
            sum_polynomials(
                (
                    constant(origin[axis], 2),
                    scale(variables[0], corners[1][axis] - origin[axis]),
                    scale(variables[1], corners[3][axis] - origin[axis]),
                    scale(product, mixed[axis]),
                ),
                2,
            )
            for axis in range(2)
        )
    raise ValueError(f"Unsupported target reference chart {cell_kind!r}.")


def _root_parameter_point(
    corners: Corners,
    cell_kind: str,
    point: tuple[ExactScalar, ...],
    /,
) -> tuple[Fraction, Fraction]:
    if len(point) != 2:
        raise ValueError("Surface root reference points must be two-dimensional.")
    u = _fraction_coordinate(point[0], "Root reference u")
    v = _fraction_coordinate(point[1], "Root reference v")
    chart = corners
    match cell_kind:
        case "triangle":
            return (
                chart[0][0]
                + u * (chart[1][0] - chart[0][0])
                + v * (chart[2][0] - chart[0][0]),
                chart[0][1]
                + u * (chart[1][1] - chart[0][1])
                + v * (chart[2][1] - chart[0][1]),
            )
        case "quadrilateral":
            if any(
                chart[2][axis] != chart[1][axis] + chart[3][axis] - chart[0][axis]
                for axis in range(2)
            ):
                raise ValueError(
                    "Surface root chart is not an exact affine quadrilateral parameter cell."
                )
            return (
                chart[0][0]
                + u * (chart[1][0] - chart[0][0])
                + v * (chart[3][0] - chart[0][0]),
                chart[0][1]
                + u * (chart[1][1] - chart[0][1])
                + v * (chart[3][1] - chart[0][1]),
            )
        case _:
            raise ValueError(f"Unsupported surface root chart cell {cell_kind!r}.")


def _surface_restriction_rows(
    cell_ids: tuple[int, ...],
    root_cell_ids: tuple[int, ...],
    root_atlas_cell_ids: tuple[int, ...],
    source_piece_ids: tuple[tuple[int, ...], ...],
    patches: np.ndarray,
    vertex_ids: tuple[tuple[int, ...], ...],
    reference_corners: tuple[Corners, ...],
    source_corners: tuple[Corners, ...],
    source_bounds: np.ndarray | None = None,
) -> SurfaceRestrictionRows:
    count = len(cell_ids)
    if not (
        len(root_cell_ids)
        == len(root_atlas_cell_ids)
        == len(source_piece_ids)
        == len(patches)
        == len(vertex_ids)
        == len(reference_corners)
        == len(source_corners)
        == count
    ):
        raise ValueError("Surface restriction ancestry rows must align by target cell.")
    if source_bounds is None:
        source_bounds = np.asarray(
            [
                (
                    float(min(point[0] for point in corners)),
                    float(max(point[0] for point in corners)),
                    float(min(point[1] for point in corners)),
                    float(max(point[1] for point in corners)),
                )
                for corners in source_corners
            ],
            dtype=np.float64,
        )
    return SurfaceRestrictionRows(
        np.asarray(cell_ids, dtype=np.int64),
        np.asarray(root_cell_ids, dtype=np.int64),
        np.asarray(root_atlas_cell_ids, dtype=np.int64),
        source_piece_ids,
        np.asarray(patches, dtype=np.int64),
        vertex_ids,
        reference_corners,
        source_corners,
        np.asarray(source_bounds, dtype=np.float64),
        np.zeros((count,), dtype=np.float64),
        np.zeros((count,), dtype=np.float64),
    )


def _control_point_id_bank(surface: BSplineSurfacePatch, /) -> tuple[int, ...]:
    from .brep._patches import BSplineSurfacePatch

    if not isinstance(surface, BSplineSurfacePatch):
        raise TypeError("B-spline root atlases require BSplineSurfacePatch.")
    controls = np.asarray(surface.control_points)
    supplied_ids = surface.control_point_ids
    if supplied_ids is None:
        raise ValueError(
            "B-spline root atlases require persistent source control_point_ids; "
            "coefficient row positions are not source identities."
        )
    identifiers = np.asarray(supplied_ids)
    if identifiers.shape != controls.shape[:2] or identifiers.dtype.kind not in "iu":
        raise ValueError(
            "control_point_ids must be an aligned integer control-grid bank."
        )
    flat = np.asarray(identifiers, dtype=np.int64).reshape((-1,))
    if np.any(flat < 0) or np.unique(flat).size != flat.size:
        raise ValueError("control_point_ids must be unique nonnegative source IDs.")
    return tuple(int(value) for value in flat)


def _exact_tspline_source_banks(
    surface: object,
    /,
) -> tuple[
    tuple[tuple[ExactScalar, ...], ...],
    tuple[ExactScalar, ...],
    np.ndarray,
    np.ndarray,
]:
    from .brep._tspline import TSplineSurfacePatch

    if not isinstance(surface, TSplineSurfacePatch):
        raise TypeError("surface must be TSplineSurfacePatch.")
    exact_homogeneous = surface.exact_target_homogeneous
    if any(
        len(row) != 4
        or any(not isinstance(value, (Fraction, RadicalElement)) for value in row)
        for row in exact_homogeneous
    ):
        raise ValueError(
            "T-spline source homogeneous rows require exact XYZ and weight scalars."
        )
    points, weights = surface._current_geometry()
    controls = np.asarray(points, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        controls.shape != (len(exact_homogeneous), 3)
        or weight_values.shape != (len(exact_homogeneous),)
        or not np.all(np.isfinite(controls))
        or not np.all(np.isfinite(weight_values))
        or np.any(weight_values <= 0.0)
    ):
        raise ValueError("T-spline numeric source banks must remain finite and positive.")
    numeric_homogeneous = np.concatenate(
        (controls * weight_values[:, None], weight_values[:, None]), axis=1
    )
    error = float(surface.maximum_homogeneous_realization_error)
    if (
        not np.all(np.isfinite(numeric_homogeneous))
        or not np.isfinite(error)
        or error < 0.0
    ):
        raise ValueError("T-spline homogeneous realization evidence must remain finite.")
    exact_error_bound = Fraction.from_float(error)
    if any(
        abs(Fraction.from_float(float(numeric_homogeneous[row, column])) - exact_value)
        > exact_error_bound
        for row, exact_row in enumerate(exact_homogeneous)
        for column, exact_value in enumerate(exact_row)
    ):
        raise ValueError(
            "T-spline numeric homogeneous coefficients exceed their retained "
            "exact realization bound."
        )
    exact_weights = tuple(row[-1] for row in exact_homogeneous)
    exact_controls = tuple(
        tuple(value / row[-1] for value in row[:-1]) for row in exact_homogeneous
    )
    controls_shadow = np.asarray(
        [[float(value) for value in point] for point in exact_controls],
        dtype=np.float64,
    )
    weights_shadow = np.asarray(
        [float(value) for value in exact_weights], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(controls_shadow))
        or not np.all(np.isfinite(weights_shadow))
        or np.any(weights_shadow <= 0.0)
    ):
        raise ValueError("T-spline exact source bank must have finite positive shadows.")
    return exact_controls, exact_weights, controls_shadow, weights_shadow


def _validate_root_atlas(
    domain: MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    parameters: np.ndarray,
    patches: np.ndarray,
    source_cell_ids: tuple[tuple[int, ...], ...],
    bounds: tuple[Corners, ...],
    /,
) -> None:
    from .._interpolation import tensor_bspline_span_extraction
    from .brep._patches import BSplineSurfacePatch
    from .brep._tspline import TSplineSurfacePatch

    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    source_map = geometry.source_map
    if not isinstance(source_map, _SplineSurfaceCoordinateMap):
        raise ValueError(
            "Spline root geometry must retain its canonical exact source map."
        )
    if (
        source_map.source_id != domain.source_id
        or source_map.source_revision != domain.source_revision
        or source_map.exact_coordinates != coefficients
    ):
        raise ValueError(
            "Spline root source map must bind the current exact coefficient bank."
        )
    expected_keys: list[tuple[int, int]] = []
    patch_values = tuple(int(patches[index]) for index in range(patches.shape[0]))
    for patch in sorted(set(patch_values)):
        surface = domain.patches[patch].surface
        if isinstance(surface, BSplineSurfacePatch):
            expected_keys.extend(
                (patch, int(identifier)) for identifier in _control_point_id_bank(surface)
            )
        elif isinstance(surface, TSplineSurfacePatch):
            expected_keys.extend(
                (patch, int(anchor.anchor_id)) for anchor in surface.mesh.anchors
            )
        else:
            raise ValueError("Spline root source map contains an unsupported carrier.")
    if source_map.coefficient_keys != tuple(expected_keys):
        raise ValueError(
            "Spline root source coefficient keys must match authored control IDs."
        )
    source_row_by_key = {key: row for row, key in enumerate(expected_keys)}
    exact_shadow = np.asarray(
        [[float(value) for value in row] for row in coefficients], dtype=np.float64
    )
    if not np.array_equal(np.asarray(geometry.coordinates), exact_shadow):
        raise ValueError(
            "Spline root numeric control bank must be the exact bank's binary64 shadow."
        )
    if not np.array_equal(
        np.asarray(source_map.rounded_vertices), np.asarray(mesh.coordinates)
    ):
        raise ValueError(
            "Spline root source map must bind the rounded root mesh vertices."
        )
    covered: dict[int, set[tuple[int, ...]]] = {}
    exact_tspline_banks: dict[
        int,
        tuple[
            tuple[tuple[ExactScalar, ...], ...],
            tuple[ExactScalar, ...],
        ],
    ] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != "quadrilateral" or not isinstance(
            element, SplineCellGeometryElement
        ):
            raise ValueError(
                "Surface source atlases require unrestricted spline quadrilateral maps."
            )
        route_rows = np.asarray(route, dtype=np.int64)
        if (
            route_rows.ndim != 2
            or route_rows.shape[0] != block.cell_count
            or route_rows.shape[1] != element.local_dof_count
        ):
            raise ValueError("Spline root coefficient routes must align with cell rows.")
        for row_index in range(route_rows.shape[0]):
            route_row = tuple(
                int(route_rows[row_index, local_index])
                for local_index in range(route_rows.shape[1])
            )
            patch = int(patches[cursor])
            source_piece = source_cell_ids[cursor]
            surface = domain.patches[patch].surface
            if isinstance(surface, BSplineSurfacePatch):
                if len(source_piece) != 3 or source_piece[0] != patch:
                    raise ValueError(
                        "Tensor spline roots must retain (patch, u-span, v-span) ancestry."
                    )
                _, u_span, v_span = source_piece
                control_ids = _control_point_id_bank(surface)
                u_knots = surface.exact_u_knots
                v_knots = surface.exact_v_knots
                anchors, extraction = tensor_bspline_span_extraction(
                    u_knots,
                    v_knots,
                    surface.u_degree,
                    surface.v_degree,
                    u_span,
                    v_span,
                    control_ids,
                )
                anchors = tuple(int(value) for value in anchors)
                extraction = tuple(
                    tuple(
                        _fraction_coordinate(value, "B-spline extraction coefficient")
                        for value in row
                    )
                    for row in extraction
                )
                positions = {
                    int(identifier): row for row, identifier in enumerate(control_ids)
                }
                if any(anchor not in positions for anchor in anchors):
                    raise ValueError(
                        "Tensor spline extraction names an absent source control ID."
                    )
                coefficient_rows = tuple(positions[anchor] for anchor in anchors)
                exact_weight_bank = tuple(
                    surface.exact_weights[u][v]
                    for u in range(surface.control_points.shape[0])
                    for v in range(surface.control_points.shape[1])
                )
                exact_control_bank = tuple(
                    tuple(surface.exact_control_points[u][v])
                    for u in range(surface.control_points.shape[0])
                    for v in range(surface.control_points.shape[1])
                )
                expected_weights = tuple(
                    exact_weight_bank[index] for index in coefficient_rows
                )
                expected_controls = tuple(
                    exact_control_bank[index] for index in coefficient_rows
                )
                expected_bounds = (
                    u_knots[u_span],
                    u_knots[u_span + 1],
                    v_knots[v_span],
                    v_knots[v_span + 1],
                )
                expected_degrees = (surface.u_degree, surface.v_degree)
                coverage_key = (u_span, v_span)
            elif isinstance(surface, TSplineSurfacePatch):
                if len(source_piece) != 3 or source_piece[0] != patch:
                    raise ValueError(
                        "T-spline roots must retain (patch, control-cell-ID, "
                        "extraction-piece-ID) ancestry."
                    )
                source_cell_id, piece_id = source_piece[1:]
                extraction_piece = surface.realization.extraction_piece(piece_id)
                if extraction_piece.cell_id != source_cell_id:
                    raise ValueError(
                        "T-spline extraction-piece identity must name its actual control cell."
                    )
                anchors = tuple(int(value) for value in extraction_piece.anchor_ids)
                extraction = tuple(
                    tuple(
                        _fraction_coordinate(value, "T-spline extraction coefficient")
                        for value in row
                    )
                    for row in extraction_piece.exact_extraction_operator
                )
                coefficient_rows = tuple(
                    int(value)
                    for value in np.asarray(
                        extraction_piece.coefficient_rows, dtype=np.int64
                    )
                )
                if patch not in exact_tspline_banks:
                    control_bank, weight_bank, _, _ = _exact_tspline_source_banks(surface)
                    exact_tspline_banks[patch] = control_bank, weight_bank
                exact_control_bank, exact_weight_bank = exact_tspline_banks[patch]
                expected_weights = tuple(
                    exact_weight_bank[index] for index in coefficient_rows
                )
                expected_controls = tuple(
                    exact_control_bank[index] for index in coefficient_rows
                )
                parameter_bounds = extraction_piece.parameter_bounds
                if len(parameter_bounds) != 4:
                    raise ValueError(
                        "T-spline source-piece bounds must have four endpoints."
                    )
                expected_bounds = (
                    _fraction_coordinate(parameter_bounds[0], "T-spline u lower bound"),
                    _fraction_coordinate(parameter_bounds[1], "T-spline u upper bound"),
                    _fraction_coordinate(parameter_bounds[2], "T-spline v lower bound"),
                    _fraction_coordinate(parameter_bounds[3], "T-spline v upper bound"),
                )
                coverage_key = (source_cell_id, piece_id)
                degree = surface.mesh.degree
                if len(degree) != 2:
                    raise ValueError("T-spline source mesh degree must have two axes.")
                expected_degrees = (int(degree[0]), int(degree[1]))
            else:
                raise ValueError(
                    "Root atlas source patch has no exact shared spline extraction."
                )
            try:
                expected_route = tuple(
                    source_row_by_key[(patch, anchor)] for anchor in anchors
                )
            except KeyError as error:
                raise ValueError(
                    "Spline extraction anchors must resolve to authored source IDs."
                ) from error
            if tuple(int(value) for value in route_row) != expected_route:
                raise ValueError(
                    "Spline root coefficient routes must resolve through source IDs."
                )
            if (
                element.source_id != domain.source_id
                or element.source_revision != domain.source_revision
            ):
                raise ValueError(
                    "Spline root element must bind the exact source identity "
                    "and revision."
                )
            if element.degrees != expected_degrees:
                raise ValueError(
                    "Spline root element degrees must match the source spline profile."
                )
            if (
                element.anchor_ids != anchors
                or element.exact_extraction_operator != extraction
            ):
                raise ValueError(
                    "Spline root element must retain the actual shared source extraction."
                )
            if element.parameter_bounds != expected_bounds:
                raise ValueError(
                    "Spline root element bounds must match the source-piece "
                    "parameter cell."
                )
            expected_corners = (
                (expected_bounds[0], expected_bounds[2]),
                (expected_bounds[1], expected_bounds[2]),
                (expected_bounds[1], expected_bounds[3]),
                (expected_bounds[0], expected_bounds[3]),
            )
            if bounds[cursor] != expected_corners or not np.array_equal(
                parameters[cursor],
                np.asarray([[float(u), float(v)] for u, v in expected_corners]),
            ):
                raise ValueError(
                    "Spline root parameters must realize the exact source-piece corners."
                )
            if element.exact_weights != expected_weights or not np.array_equal(
                np.asarray(element.weights),
                np.asarray([float(value) for value in expected_weights]),
            ):
                raise ValueError(
                    "Spline root weights must retain the exact source bank and its binary64 shadow."
                )
            local = tuple(coefficients[int(index)] for index in route_row)
            exact_controls = tuple(tuple(point) for point in expected_controls)
            if local != exact_controls:
                raise ValueError(
                    "Spline root coordinates must match their source coefficient rows."
                )
            images = coordinate_corner_images(element, local, cell_row=row_index)
            if images is None or not np.array_equal(
                np.asarray(mesh.coordinates)[np.asarray(block.vertices)[row_index]],
                np.stack(tuple(rounded_point(image) for image in images)),
            ):
                raise ValueError(
                    "Spline root mesh corners must match rounded exact source-map images."
                )
            spans = covered.setdefault(patch, set())
            if coverage_key in spans:
                raise ValueError("Spline root atlas repeats a source-piece identity.")
            spans.add(coverage_key)
            cursor += 1
    for patch in sorted(set(patch_values)):
        surface = domain.patches[patch].surface
        if isinstance(surface, BSplineSurfacePatch):
            u = surface.exact_u_knots
            v = surface.exact_v_knots
            required = {
                (u_span, v_span)
                for u_span in range(surface.u_degree, surface.control_points.shape[0])
                for v_span in range(surface.v_degree, surface.control_points.shape[1])
                if u[u_span] < u[u_span + 1] and v[v_span] < v[v_span + 1]
            }
        elif isinstance(surface, TSplineSurfacePatch):
            required = {
                (extraction.cell_id, extraction.piece_id)
                for extraction in surface.realization.extractions
            }
        else:
            raise ValueError(
                "Surface atlas contains an unsupported source spline carrier."
            )
        if covered.get(patch, set()) != required:
            raise ValueError(
                "Surface root atlas must cover every source span/cell exactly once."
            )


def prepare_surface_source_root_atlas(
    domain: MeshingDomain,
    patches: tuple[int, ...],
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> SurfaceSourceRootAtlas:
    """Build exact source root cells for tensor B-splines or bicubic T-splines."""
    from .._interpolation import tensor_bspline_span_extraction
    from .brep._patches import BSplineSurfacePatch
    from .brep._tspline import TSplineSurfacePatch

    domain.require_current(domain.source_id, domain.source_revision)
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    if coordinate_contract.coordinate_system != "cartesian":
        raise ValueError(
            "Surface source root atlas requires an orthonormal Cartesian frame."
        )
    if isinstance(patches, (str, bytes)):
        raise TypeError("patches must contain source patch indices.")
    patch_values = tuple(patches)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in patch_values
    ):
        raise TypeError("patches must contain integer source patch indices.")
    selected = tuple(sorted(set(int(value) for value in patch_values)))
    if not selected or any(
        value < 0 or value >= len(domain.patches) for value in selected
    ):
        raise ValueError("patches must select existing source surface patches.")
    point_rows: list[np.ndarray] = []
    blocks: list[CellBlock] = []
    elements: dict[str, SplineCellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    coefficient_parts: list[np.ndarray] = []
    exact_coordinate_rows: list[tuple[ExactScalar, ...]] = []
    coefficient_keys: list[tuple[int, int]] = []
    parameters: list[np.ndarray] = []
    patch_rows: list[int] = []
    source_cells: list[tuple[int, ...]] = []
    source_bounds: list[Corners] = []
    point_ids: dict[tuple[int, Fraction, Fraction], int] = {}
    coordinate_offset = 0
    root_cell_id = 0

    def add_root(
        patch: int,
        source_piece: tuple[int, ...],
        parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction],
        degrees: tuple[int, int],
        anchors: tuple[int, ...],
        extraction: tuple[tuple[Fraction, ...], ...],
        local_weights: np.ndarray,
        exact_weights: tuple[ExactScalar, ...],
        local_positions: tuple[int, ...],
        control_rows: tuple[tuple[ExactScalar, ...], ...],
        /,
    ) -> None:
        nonlocal root_cell_id
        element = SplineCellGeometryElement(
            anchors,
            parameter_bounds,
            extraction,
            local_weights,
            domain.source_id,
            domain.source_revision,
            degrees=degrees,
            exact_weights=exact_weights,
        )
        block_name = f"source_span_{root_cell_id}"
        route = coordinate_offset + np.asarray(local_positions, dtype=np.int64)
        routes[block_name] = route[None, :]
        elements[block_name] = element
        local = tuple(control_rows[index] for index in local_positions)
        images = coordinate_corner_images(element, local, cell_row=0)
        if images is None:
            raise ValueError("Exact spline root corner images are unresolved.")
        uv_corners = (
            (parameter_bounds[0], parameter_bounds[2]),
            (parameter_bounds[1], parameter_bounds[2]),
            (parameter_bounds[1], parameter_bounds[3]),
            (parameter_bounds[0], parameter_bounds[3]),
        )
        vertices = []
        for uv, image in zip(uv_corners, images, strict=True):
            key = (patch, uv[0], uv[1])
            vertex = point_ids.get(key)
            if vertex is None:
                vertex = len(point_rows)
                point_ids[key] = vertex
                point_rows.append(rounded_point(image))
            vertices.append(vertex)
        blocks.append(
            CellBlock(
                block_name,
                "quadrilateral",
                np.asarray((vertices,), dtype=np.int64),
                global_ids=np.asarray((root_cell_id,), dtype=np.int64),
            )
        )
        parameters.append(
            np.asarray([[float(u), float(v)] for u, v in uv_corners], dtype=np.float64)
        )
        patch_rows.append(patch)
        source_cells.append(source_piece)
        source_bounds.append(uv_corners)
        root_cell_id += 1

    for patch in selected:
        surface = domain.patches[patch].surface
        if isinstance(surface, BSplineSurfacePatch):
            control_ids = _control_point_id_bank(surface)
            id_positions = {
                int(identifier): row for row, identifier in enumerate(control_ids)
            }
            controls = np.asarray(surface.control_points, dtype=np.float64).reshape(
                (-1, 3)
            )
            weights = np.asarray(surface.weights, dtype=np.float64).reshape((-1,))
            if (
                not np.all(np.isfinite(controls))
                or not np.all(np.isfinite(weights))
                or np.any(weights <= 0.0)
            ):
                raise ValueError(
                    "Spline source control points and weights must be finite with positive weights."
                )
            u_knots = surface.exact_u_knots
            v_knots = surface.exact_v_knots
            exact_control_bank = tuple(
                tuple(surface.exact_control_points[u][v])
                for u in range(surface.control_points.shape[0])
                for v in range(surface.control_points.shape[1])
            )
            exact_weight_bank = tuple(
                surface.exact_weights[u][v]
                for u in range(surface.control_points.shape[0])
                for v in range(surface.control_points.shape[1])
            )
            coefficient_parts.append(controls)
            exact_coordinate_rows.extend(exact_control_bank)
            coefficient_keys.extend(
                (patch, int(identifier)) for identifier in control_ids
            )
            for u_span in range(surface.u_degree, surface.control_points.shape[0]):
                if not u_knots[u_span] < u_knots[u_span + 1]:
                    continue
                for v_span in range(surface.v_degree, surface.control_points.shape[1]):
                    if not v_knots[v_span] < v_knots[v_span + 1]:
                        continue
                    anchors, operator = tensor_bspline_span_extraction(
                        u_knots,
                        v_knots,
                        surface.u_degree,
                        surface.v_degree,
                        u_span,
                        v_span,
                        control_ids,
                    )
                    anchors = tuple(int(value) for value in anchors)
                    positions = tuple(id_positions[anchor] for anchor in anchors)
                    extraction = tuple(
                        tuple(
                            _fraction_coordinate(value, "B-spline extraction coefficient")
                            for value in row
                        )
                        for row in operator
                    )
                    parameter_bounds = (
                        u_knots[u_span],
                        u_knots[u_span + 1],
                        v_knots[v_span],
                        v_knots[v_span + 1],
                    )
                    add_root(
                        patch,
                        (patch, u_span, v_span),
                        parameter_bounds,
                        (surface.u_degree, surface.v_degree),
                        anchors,
                        extraction,
                        weights[np.asarray(positions, dtype=np.int64)],
                        tuple(exact_weight_bank[index] for index in positions),
                        positions,
                        exact_control_bank,
                    )
            coordinate_offset += controls.shape[0]
        elif isinstance(surface, TSplineSurfacePatch):
            (
                exact_control_bank,
                exact_weight_bank,
                controls_exact_shadow,
                weights_exact_shadow,
            ) = _exact_tspline_source_banks(surface)
            coefficient_parts.append(controls_exact_shadow)
            exact_coordinate_rows.extend(exact_control_bank)
            coefficient_keys.extend(
                (patch, int(anchor.anchor_id)) for anchor in surface.mesh.anchors
            )
            for extraction in surface.realization.extractions:
                anchors = tuple(int(value) for value in extraction.anchor_ids)
                positions = tuple(
                    int(value)
                    for value in np.asarray(extraction.coefficient_rows, dtype=np.int64)
                )
                parameter_bounds = extraction.parameter_bounds
                if len(parameter_bounds) != 4:
                    raise ValueError("T-spline root bounds must have four endpoints.")
                bounds = (
                    _fraction_coordinate(parameter_bounds[0], "T-spline u lower bound"),
                    _fraction_coordinate(parameter_bounds[1], "T-spline u upper bound"),
                    _fraction_coordinate(parameter_bounds[2], "T-spline v lower bound"),
                    _fraction_coordinate(parameter_bounds[3], "T-spline v upper bound"),
                )
                operator = tuple(
                    tuple(
                        _fraction_coordinate(value, "T-spline extraction coefficient")
                        for value in row
                    )
                    for row in extraction.exact_extraction_operator
                )
                degree = surface.mesh.degree
                if len(degree) != 2:
                    raise ValueError("T-spline source mesh degree must have two axes.")
                degrees = (int(degree[0]), int(degree[1]))
                add_root(
                    patch,
                    (patch, int(extraction.cell_id), int(extraction.piece_id)),
                    bounds,
                    degrees,
                    anchors,
                    operator,
                    weights_exact_shadow[np.asarray(positions, dtype=np.int64)],
                    tuple(exact_weight_bank[index] for index in positions),
                    positions,
                    exact_control_bank,
                )
            coordinate_offset += controls_exact_shadow.shape[0]
        else:
            raise ValueError(
                "No exact shared span extraction is supported for this surface carrier."
            )
    if not blocks:
        raise ValueError("Selected spline patches contain no nonempty source cells.")
    root_mesh = CellMesh(
        np.stack(point_rows),
        tuple(blocks),
        vertex_global_ids=np.arange(len(point_rows), dtype=np.int64),
        numeric_version=domain.source_revision,
    )
    numeric_controls = np.concatenate(coefficient_parts, axis=0)
    source_map: CellGeometryCoordinateMap = _SplineSurfaceCoordinateMap(
        domain.source_id,
        domain.source_revision,
        root_mesh.coordinates,
        exact_coordinate_rows,
        numeric_controls,
        coefficient_keys,
    )
    root_geometry = CellGeometrySpec(
        elements,
        routes,
        numeric_controls,
        source_map=source_map,
        source_mesh=root_mesh,
    )
    return SurfaceSourceRootAtlas(
        domain,
        root_mesh,
        root_geometry,
        np.stack(parameters),
        np.asarray(patch_rows, dtype=np.int64),
        coordinate_contract,
        root_source_cell_ids=tuple(source_cells),
        root_parameter_bounds=tuple(source_bounds),
    )


__all__ = [
    "PreparedSurfaceSourceSupport",
    "SurfaceNativeRestrictionBoundarySource",
    "SurfaceRestrictionRows",
    "SurfaceSourceCharts",
    "SurfaceSourceRootAtlas",
    "prepare_surface_source_root_atlas",
    "validate_surface_source_pieces",
]

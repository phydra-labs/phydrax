#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native constrained triangulation of source-authoritative surface charts."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry._surface_source_support as surface_support_owner
import phydrax.meshing._association as association_owner
from phydrax.geometry import ConstrainedDelaunayTriangulation
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._interval_enclosure import (
    IntervalExtensionError,
    PreparedIntervalFunction,
)
from phydrax.geometry._rational import fraction_endpoint
from phydrax.geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    ExactLineCurve,
    LineCurve,
    SurfaceMappedCurve,
)
from phydrax.geometry.reconstruction._core import ReconstructedGeometrySource
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellMeshingTarget,
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from phydrax.meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from phydrax.meshing._organization import MeshLabel, MeshPatch
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import (
    resolve_size_controls,
    SizeFieldDomain,
    UniformSizeControl,
)
from phydrax.meshing._surface_construction import (
    source_cell_ancestry,
    SurfaceConstruction,
    SurfaceSourceCellAtlas,
)

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._reference_cell import reference_cell_topology
from .._plc_source import _numeric_entity_ids
from ._native_options import NativeMeshingOptions
from ._native_schedules import NativeSurfaceSchedule
from ._native_surface_association import (
    _surface_edges,
    associations as surface_associations,
)


@final
class _SurfaceRootCells(StrictModule, NonTrainableState):
    """Validated source-piece roots retained by a prepared surface route."""

    domain: domain_owner.MeshingDomain
    root_cell_ids: tuple[int, ...] = eqx.field(static=True)
    root_source_cell_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    root_parameter_bounds: tuple[tuple[tuple[Fraction, Fraction], ...], ...] = eqx.field(
        static=True
    )
    root_patches: np.ndarray
    root_parameters: np.ndarray
    spline_root_atlas: surface_support_owner.SurfaceSourceRootAtlas | None

    def __init__(
        self,
        domain: domain_owner.MeshingDomain,
        root_cell_ids: tuple[int, ...],
        root_source_cell_ids: tuple[tuple[int, ...], ...],
        root_parameter_bounds: tuple[tuple[tuple[Fraction, Fraction], ...], ...],
        root_patches: np.ndarray,
        root_parameters: np.ndarray,
        spline_root_atlas: surface_support_owner.SurfaceSourceRootAtlas | None,
        /,
    ) -> None:
        identifiers = tuple(int(value) for value in root_cell_ids)
        pieces = tuple(tuple(int(value) for value in row) for row in root_source_cell_ids)
        bounds = tuple(
            tuple((Fraction(first), Fraction(last)) for first, last in row)
            for row in root_parameter_bounds
        )
        patches = np.asarray(root_patches, dtype=np.int64)
        parameters = np.asarray(root_parameters, dtype=np.float64)
        if (
            not identifiers
            or len(set(identifiers)) != len(identifiers)
            or any(value < 0 for value in identifiers)
            or len(pieces) != len(identifiers)
            or len(bounds) != len(identifiers)
            or patches.shape != (len(identifiers),)
            or parameters.shape != (len(identifiers), 4, 2)
            or not np.all(np.isfinite(parameters))
            or any(
                len(piece) < 2
                or piece[0] != int(patch)
                or any(value < 0 for value in piece)
                for piece, patch in zip(pieces, patches.tolist(), strict=True)
            )
            or any(len(row) != 4 for row in bounds)
        ):
            raise ValueError(
                "Native surface root cells must retain aligned source-piece charts."
            )
        if spline_root_atlas is not None:
            if not isinstance(
                spline_root_atlas, surface_support_owner.SurfaceSourceRootAtlas
            ):
                raise TypeError(
                    "spline_root_atlas must be SurfaceSourceRootAtlas or None."
                )
            spline_root_atlas.require_current()
            if spline_root_atlas.domain.domain_id != domain.domain_id:
                raise ValueError("Spline root atlas must retain the same source domain.")
        patches = np.array(patches, copy=True)
        parameters = np.array(parameters, copy=True)
        patches.setflags(write=False)
        parameters.setflags(write=False)
        self.domain = domain
        self.root_cell_ids = identifiers
        self.root_source_cell_ids = pieces
        self.root_parameter_bounds = bounds
        self.root_patches = patches
        self.root_parameters = parameters
        self.spline_root_atlas = spline_root_atlas

    def require_current(self) -> None:
        self.domain.require_current(self.domain.source_id, self.domain.source_revision)
        if self.spline_root_atlas is not None:
            self.spline_root_atlas.require_current()


@dataclass(slots=True)
class _ChartMesh:
    patch: int
    points: list[tuple[Fraction, Fraction]]
    triangles: list[tuple[int, int, int]]
    corners: dict[int, int]
    curve_parameters: dict[tuple[int, int], Fraction]
    boundary_edges: dict[
        tuple[int, int], tuple[int, tuple[int, int, int], Fraction, Fraction]
    ]
    pole_deviation_bound: float = 0.0
    turn_rows: frozenset[tuple[int, int]] = frozenset()
    # Source-piece root cells the chart triangles were split at; every cell
    # lies inside one of them.
    root_cells: tuple[_SurfaceRootChart, ...] = ()

    def edge_parameters(
        self,
        edge: tuple[int, int],
        curve: int,
        stored: tuple[Fraction, Fraction],
        /,
    ) -> tuple[Fraction | None, Fraction | None]:
        """Return the source-curve parameters at both ends of one trim edge.

        A turn row (the shared start and end of a self-closing trim) records the
        start parameter; on the edge whose stored interval ends the turn it
        carries the end parameter instead.
        """
        first, last = (
            Fraction(1)
            if (row, curve) in self.turn_rows and Fraction(0) not in stored
            else self.curve_parameters.get((row, curve))
            for row in edge
        )
        return first, last


@dataclass(frozen=True, slots=True)
class _PoleChartSample:
    parameter: Fraction
    chart_point: tuple[Fraction, Fraction]
    ring_point: tuple[Fraction, Fraction]
    cap_bound: float


type _PoleRing = tuple[tuple[Fraction, tuple[Fraction, Fraction]], ...]


@dataclass(frozen=True, slots=True)
class _PoleChart:
    """Chart layout of one collapsed pole trim.

    ``samples`` carry the pole points and the inner cap ring. ``rings`` are the
    graded rings beyond it toward the regular chart, each as ordered pole-use
    parameters with their chart points; every ring's parameters contain those
    of the ring inside it.
    """

    samples: tuple[_PoleChartSample, ...]
    rings: tuple[_PoleRing, ...]


type _PoleCharts = dict[tuple[int, int, int], _PoleChart]


def _edge_key(first: int, last: int, /) -> tuple[int, int]:
    return (first, last) if first <= last else (last, first)


@dataclass(frozen=True, slots=True)
class _SurfaceRootChart:
    """One exact convex parameter cell owned by a source piece."""

    cell_id: int
    source_piece: tuple[int, ...]
    corners: tuple[tuple[Fraction, Fraction], ...]
    lower_u: Fraction
    upper_u: Fraction
    lower_v: Fraction
    upper_v: Fraction
    orientation: int


@dataclass(frozen=True, slots=True)
class _RefinedSurface:
    charts: tuple[_ChartMesh, ...]
    local_rows: tuple[dict[int, int], ...]
    coordinates: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    triangles: np.ndarray
    triangle_patches: np.ndarray
    triangle_parameters: np.ndarray
    triangle_deviation_bounds: np.ndarray
    triangle_normal_bounds: np.ndarray
    curve_edge_bounds: dict[tuple[int, int, int], float]
    minimum_angle: float
    fidelity_tolerance: float | None


def _failure(
    message: str,
    /,
    *,
    category: MeshingFailureCategory = MeshingFailureCategory.UNSUPPORTED_COMBINATION,
    entity_ids: tuple[int, ...] = (),
) -> MeshingFailure:
    return MeshingFailure(
        category,
        message,
        provider_code="native_surface_unresolved",
        stage="surface_meshing",
        entity_ids=entity_ids,
    )


def _scope_banks(
    domain: domain_owner.MeshingDomain, /
) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
    """Project canonical source scopes back onto actual MeshingDomain rows."""
    id_banks: list[np.ndarray] = []
    set_ids: list[str] = []
    for dimension in range(4):
        identifiers = np.asarray(domain.scope_entity_ids(dimension), dtype=np.int64)
        if identifiers.ndim != 1 or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "MeshingDomain source scope identities must be unique vectors."
            )
        if dimension == 3:
            paths = domain.region_source_occurrences
            row_count = len(domain.regions)
        else:
            paths = domain.source_occurrences[dimension]
            row_count = len(domain.source_indices[dimension])

        if identifiers.size != row_count:
            raise ValueError(
                "MeshingDomain scope IDs must cover every source entity row."
            )
        rows_by_key = {
            (domain.entity_id(dimension, row), tuple(paths[row])): row
            for row in range(row_count)
        }
        row_ids = np.empty((row_count,), dtype=np.int64)
        for identifier in identifiers.tolist():
            key = domain.source_entity_key_for_scope(dimension, int(identifier))
            try:
                row = rows_by_key[key]
            except KeyError as error:
                raise ValueError(
                    "MeshingDomain scope API names an entity absent from its source table."
                ) from error
            row_ids[row] = int(identifier)
        if row_count and np.unique(row_ids).size != row_count:
            raise ValueError(
                "MeshingDomain scope IDs must cover every source entity row."
            )
        id_banks.append(row_ids)
        set_ids.append(domain.scope_entity_set_id(dimension))
    return tuple(id_banks), tuple(set_ids)


def _scope_id_for_row(
    scope_ids: tuple[np.ndarray, ...], dimension: int, row: int, /
) -> int:
    return int(scope_ids[dimension][row])


def _scope_rows(scope_ids: tuple[np.ndarray, ...], dimension: int, /) -> dict[int, int]:
    return {
        int(identifier): row
        for row, identifier in enumerate(scope_ids[dimension].tolist())
    }


def _source_key_for_scope(
    domain: domain_owner.MeshingDomain,
    dimension: int,
    entity_id: int,
    /,
) -> tuple[str, tuple[str, ...]]:
    return domain.source_entity_key_for_scope(dimension, entity_id)


def _selected_patches(
    domain: domain_owner.MeshingDomain,
    specification: SurfaceMeshingSpec,
    scope_ids: tuple[np.ndarray, ...],
    scope_set_ids: tuple[str, ...],
    /,
) -> tuple[int, ...]:
    scope = specification.scope
    expected = scope_ids[2]
    if (
        scope.source_id != domain.source_id
        or scope.source_revision != domain.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or scope.entity_set_id != scope_set_ids[2]
        or not np.all(np.isin(np.asarray(scope.entity_ids, dtype=np.int64), expected))
    ):
        raise _failure(
            "parametric_surface requires a source-owned scope of actual MeshingDomain patches",
            category=MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
        )
    id_to_patch = _scope_rows(scope_ids, 2)
    try:
        selected = tuple(
            sorted(
                {
                    id_to_patch[int(identifier)]
                    for identifier in np.asarray(
                        scope.entity_ids, dtype=np.int64
                    ).tolist()
                }
            )
        )
    except KeyError as error:
        raise _failure(
            f"surface scope names absent source patch {error.args[0]}",
            category=MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            entity_ids=(int(error.args[0]),),
        ) from error
    if not selected:
        raise _failure(
            "parametric_surface requires at least one selected source patch",
            category=MeshingFailureCategory.INVALID_SPECIFICATION,
        )
    return selected


def _domain_for_source(
    source: domain_owner.MeshingDomain | ReconstructedGeometrySource, /
) -> domain_owner.MeshingDomain:
    if isinstance(source, domain_owner.MeshingDomain):
        return source
    if isinstance(source, ReconstructedGeometrySource):
        domain = source.meshing_domain
        if domain is None:
            raise _failure(
                "ReconstructedGeometrySource has no retained triangle incidence for parametric charts",
                category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            )
        domain.require_current(source.source_id, source.source_revision)
        return domain
    raise TypeError(
        "parametric_surface requires MeshingDomain or ReconstructedGeometrySource."
    )


def _support_issues(
    domain: domain_owner.MeshingDomain,
    specification: SurfaceMeshingSpec,
    selected: tuple[int, ...],
    scope_ids: tuple[np.ndarray, ...],
    scope_set_ids: tuple[str, ...],
    /,
) -> list[str]:
    issues: list[str] = []
    target: CellMeshingTarget = specification.target

    selected_curve_rows = set(_trim_curves(domain, selected))
    selected_curve_ids = {
        _scope_id_for_row(scope_ids, 1, curve) for curve in selected_curve_rows
    }
    selected_corner_ids = {
        _scope_id_for_row(scope_ids, 0, corner)
        for curve in selected_curve_rows
        for corner in (domain.curves[curve].start, domain.curves[curve].end)
    }
    family = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*family.required, *family.preferred)) != {"triangle"}
        or family.allow_mixed
        or family.allowed_transitions
    ):
        issues.append("a conforming affine triangular target in ambient dimension three")
    if specification.background_metric is not None:
        issues.append("background metrics outside the isotropic physical size schedule")
    if specification.planar_embedding is not None:
        issues.append("planar embeddings on a three-dimensional source surface")
    if specification.periodic_constraints:
        issues.append("periodic surface constraints")
    if specification.layer_controls:
        issues.append("boundary layers on a surface-only output")
    if specification.patch_controls:
        issues.append("explicit patch controls not represented by the source patch chart")
    if specification.region_controls:
        issues.append(
            "explicit region controls not represented by source patch adjacency"
        )
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append("uniform physical size controls on selected source patches")
            continue
        control_scope = control.scope
        if (
            control_scope.source_id != domain.source_id
            or control_scope.source_revision != domain.source_revision
            or control_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or control_scope.entity_dimension != 2
            or control_scope.entity_set_id != scope_set_ids[2]
            or not np.all(
                np.isin(
                    np.asarray(control_scope.entity_ids, dtype=np.int64),
                    scope_ids[2],
                )
            )
        ):
            issues.append("uniform size controls scoped to selected source patches")
        if control.maximum_growth_rate is not None:
            issues.append(
                "patch-size growth controls without a source geodesic size field"
            )
    for feature in specification.protected_features:
        expected_dimension = 1 if feature.feature_kind.value == "curve" else 0
        if feature.feature_kind.value not in ("curve", "corner"):
            issues.append("protected surface features of kind curve or corner")
            continue
        feature_scope = feature.scope
        if (
            feature_scope.source_id != domain.source_id
            or feature_scope.source_revision != domain.source_revision
            or feature_scope.entity_kind is not MeshingEntityKind.GEOMETRY
            or feature_scope.entity_dimension != expected_dimension
            or feature_scope.entity_set_id != scope_set_ids[expected_dimension]
            or not np.all(
                np.isin(
                    np.asarray(feature_scope.entity_ids, dtype=np.int64),
                    scope_ids[expected_dimension],
                )
            )
        ):
            issues.append("protected curves and corners scoped to this source revision")
        selected_source_ids = (
            selected_curve_ids if expected_dimension == 1 else selected_corner_ids
        )
        if not set(
            np.asarray(feature_scope.entity_ids, dtype=np.int64).tolist()
        ).issubset(selected_source_ids):
            issues.append("protected features selected on the meshed source patches")
    uniform_controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if (
        not uniform_controls
        or len(uniform_controls) != len(specification.size_controls)
        or any(control.scope.entity_dimension != 2 for control in uniform_controls)
    ):
        issues.append(
            "an explicit physical resolution request via a uniform size control on selected source patches"
        )
    else:
        active = np.concatenate(
            tuple(
                np.asarray(control.scope.entity_ids, dtype=np.int64)
                for control in uniform_controls
            )
        )
        selected_ids = {_scope_id_for_row(scope_ids, 2, row) for row in selected}
        if not selected_ids.issubset(set(active.tolist())):
            issues.append("uniform size controls covering every selected source patch")
    return list(dict.fromkeys(issues))


def _validate_chart_trims(
    domain: domain_owner.MeshingDomain,
    selected: tuple[int, ...],
    preparation: domain_owner.SourcePreparation,
    /,
) -> None:
    for patch in selected:
        for loop in domain.patches[patch].loops:
            for use in loop:
                try:
                    bounds = preparation.curve_box(
                        use.pcurve, min(use.first, use.last), max(use.first, use.last)
                    )
                except (
                    ArithmeticError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise _failure(
                        f"parametric_surface requires a continuous p-curve enclosure: {error}",
                        category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                        entity_ids=(int(patch),),
                    ) from error
                if (
                    bounds.shape != (2, 2)
                    or np.any(~np.isfinite(bounds))
                    or np.any(bounds[0] > bounds[1])
                ):
                    raise _failure(
                        "parametric_surface p-curves require finite ordered chart enclosures",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )


def _use_source_parameter(
    use: domain_owner.PatchCurveUse | domain_owner.PatchPoleUse,
    parameter: Fraction,
    /,
) -> Fraction:
    first = Fraction.from_float(use.first)
    last = Fraction.from_float(use.last)
    if isinstance(use, domain_owner.PatchCurveUse):
        return first + parameter * (last - first)
    start, stop = (first, last) if use.sense > 0 else (last, first)
    return start + parameter * (stop - start)


def _use_chart_point(
    use: domain_owner.PatchCurveUse | domain_owner.PatchPoleUse,
    parameter: Fraction,
    /,
) -> tuple[Fraction, Fraction]:
    # A use closing its chart loop returns to its start chart point exactly.
    if (
        parameter == 1
        and isinstance(use, domain_owner.PatchCurveUse)
        and use.closes_chart_loop()
    ):
        parameter = Fraction(0)
    native = float(_use_source_parameter(use, parameter))
    uv = np.asarray(
        use.pcurve.evaluate(jnp.asarray(native, dtype=jnp.float64)),
        dtype=np.float64,
    ).reshape((2,))
    if not np.all(np.isfinite(uv)):
        raise ValueError("Source p-curve evaluation must remain finite.")
    return Fraction.from_float(float(uv[0])), Fraction.from_float(float(uv[1]))


def _canonical_use_endpoints(
    use: domain_owner.PatchCurveUse | domain_owner.PatchPoleUse, /
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    return _use_chart_point(use, Fraction(0)), _use_chart_point(use, Fraction(1))


def _curved_use_box_corners(
    use: domain_owner.PatchCurveUse,
    preparation: domain_owner.SourcePreparation,
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Return the outward chart enclosure corners of one curved trim use."""
    if isinstance(use.pcurve, (ExactLineCurve, LineCurve)):
        return ()
    bounds = preparation.curve_box(
        use.pcurve, min(use.first, use.last), max(use.first, use.last)
    )
    return (
        (
            Fraction.from_float(float(bounds[0, 0])),
            Fraction.from_float(float(bounds[0, 1])),
        ),
        (
            Fraction.from_float(float(bounds[1, 0])),
            Fraction.from_float(float(bounds[1, 1])),
        ),
    )


def _interpolate_chart(
    first: tuple[Fraction, Fraction],
    last: tuple[Fraction, Fraction],
    parameter: Fraction,
    /,
) -> tuple[Fraction, Fraction]:
    return (
        first[0] + parameter * (last[0] - first[0]),
        first[1] + parameter * (last[1] - first[1]),
    )


def _chart_inputs(
    domain: domain_owner.MeshingDomain,
    patch: int,
    curves: dict[int, tuple[Fraction, ...]],
    pole_charts: _PoleCharts,
    /,
) -> tuple[
    list[tuple[Fraction, Fraction]],
    np.ndarray,
    tuple[tuple[tuple[Fraction, Fraction], ...], ...],
    tuple[int, ...],
    tuple[tuple[Fraction, Fraction], ...],
    dict[tuple[int, int], Fraction],
    dict[int, int],
    tuple[tuple[int, int, int], ...],
    tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...],
    tuple[int | None, ...],
    float,
    tuple[tuple[int, int, int, int], ...],
]:
    """Create source-curve and singular-pole constraints in the exact chart.

    The last result lists each pole band cell as chart rows ``(inner_first,
    inner_last, outer_first, outer_last)`` over one inner sample interval: the
    strip between the collapsed pole side and the cap ring, or between two
    graded rings. The outer side runs along the outer ring's chart points.
    """
    points: list[tuple[Fraction, Fraction]] = []
    point_rows: dict[tuple[Fraction, Fraction], int] = {}
    point_corners: dict[int, int] = {}
    point_curve_parameters: dict[tuple[int, int], Fraction] = {}
    segments: list[tuple[int, int]] = []
    segment_curves: list[int] = []
    segment_parameters: list[tuple[Fraction, Fraction]] = []
    segment_occurrences: list[tuple[int, int, int]] = []
    segment_chart_endpoints: list[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    ] = []
    segment_pole_corners: list[int | None] = []
    loop_rows: list[tuple[int, ...]] = []
    chart = domain.patches[patch]
    internal_segments: list[tuple[int, int, tuple[int, int, int]]] = []
    pole_deviation_bound = 0.0
    pole_bands: list[tuple[int, int, int, int]] = []

    def point_row(uv: tuple[Fraction, Fraction], /) -> int:
        existing = point_rows.get(uv)
        if existing is None:
            existing = len(points)
            points.append(uv)
            point_rows[uv] = existing
        return existing

    def add_internal_segment(
        first: int,
        last: int,
        occurrence: tuple[int, int, int],
        /,
    ) -> None:
        internal_segments.append((first, last, occurrence))

    for loop_index, loop in enumerate(chart.loops):
        traversed: list[int] = []
        for use_index, use in enumerate(loop):
            if isinstance(use, domain_owner.PatchCurveUse):
                parameter_values = curves.get(use.curve)
                if parameter_values is None:
                    raise _failure(
                        "selected source chart references a curve outside its selected surface scope",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(use.curve),),
                    )
                traversal_parameters = (
                    parameter_values
                    if use.sense > 0
                    else tuple(reversed(parameter_values))
                )
                rows: list[int] = []
                uv_values: list[tuple[Fraction, Fraction]] = []
                for parameter in traversal_parameters:
                    uv = _use_chart_point(use, parameter)
                    row = point_row(uv)
                    rows.append(row)
                    uv_values.append(uv)
                    # The end of a closed chart loop is its start point; the
                    # shared row records the start parameter.
                    recorded = (
                        Fraction(0)
                        if parameter == 1 and use.closes_chart_loop()
                        else parameter
                    )
                    previous_parameter = point_curve_parameters.get((row, use.curve))
                    if previous_parameter is not None and previous_parameter != recorded:
                        raise _failure(
                            "one source trim chart point has conflicting source-curve parameters",
                            category=MeshingFailureCategory.INVALID_SOURCE,
                            entity_ids=(int(use.curve),),
                        )
                    point_curve_parameters[(row, use.curve)] = recorded
                    corner = (
                        domain.curves[use.curve].start
                        if parameter == 0
                        else domain.curves[use.curve].end
                        if parameter == 1
                        else None
                    )
                    if corner is not None:
                        previous = point_corners.get(row)
                        if previous is not None and previous != corner:
                            raise _failure(
                                "distinct source corners occupy one source-chart trim point",
                                category=MeshingFailureCategory.INVALID_SOURCE,
                                entity_ids=(int(corner), int(previous)),
                            )
                        point_corners[row] = corner
                if len(rows) < 2 or any(
                    first == last for first, last in zip(rows, rows[1:])
                ):
                    raise _failure(
                        "source p-curve discretization contains a collapsed trim interval",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(use.curve),),
                    )
                traversed.extend(rows[:-1])
                for (
                    first_row,
                    last_row,
                    first_parameter,
                    last_parameter,
                    first_chart,
                    last_chart,
                ) in zip(
                    rows[:-1],
                    rows[1:],
                    traversal_parameters[:-1],
                    traversal_parameters[1:],
                    uv_values[:-1],
                    uv_values[1:],
                    strict=True,
                ):
                    segments.append((first_row, last_row))
                    segment_curves.append(use.curve)
                    segment_parameters.append((first_parameter, last_parameter))
                    segment_occurrences.append((patch, loop_index, use_index))
                    segment_chart_endpoints.append((first_chart, last_chart))
                    segment_pole_corners.append(None)
                continue

            if not isinstance(use, domain_owner.PatchPoleUse):
                raise RuntimeError("MeshingDomain contains an unknown surface trim use.")
            pole_chart = pole_charts.get((patch, loop_index, use_index))
            if pole_chart is None or len(pole_chart.samples) < 2:
                raise _failure(
                    "collapsed pole trim has no continuous singular-chart ring",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(patch),),
                )
            samples = pole_chart.samples
            rows = []
            ring_rows = []
            for sample in samples:
                row = point_row(sample.chart_point)
                ring_row = point_row(sample.ring_point)
                rows.append(row)
                ring_rows.append(ring_row)
                previous = point_corners.get(row)
                if previous is not None and previous != use.corner:
                    raise _failure(
                        "collapsed pole point conflicts with another source corner",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(use.corner), int(previous)),
                    )
                point_corners[row] = use.corner
                pole_deviation_bound = max(pole_deviation_bound, sample.cap_bound)
            traversed.extend(rows[:-1])
            occurrence = (patch, loop_index, use_index)
            for index, (first_sample, last_sample) in enumerate(
                zip(samples[:-1], samples[1:], strict=True)
            ):
                first_row, last_row = rows[index], rows[index + 1]
                first_ring, last_ring = ring_rows[index], ring_rows[index + 1]
                segments.append((first_row, last_row))
                segment_curves.append(-1)
                segment_parameters.append((first_sample.parameter, last_sample.parameter))
                segment_occurrences.append(occurrence)
                segment_chart_endpoints.append(
                    (first_sample.chart_point, last_sample.chart_point)
                )
                segment_pole_corners.append(use.corner)
                add_internal_segment(first_row, first_ring, occurrence)
                add_internal_segment(last_row, last_ring, occurrence)
                add_internal_segment(first_ring, last_ring, occurrence)
                pole_bands.append((first_row, last_row, first_ring, last_ring))
            inner_parameters = tuple(sample.parameter for sample in samples)
            inner_rows = tuple(ring_rows)
            for ring in pole_chart.rings:
                outer_rows = tuple(point_row(point) for _, point in ring)
                outer_index = {
                    parameter: index for index, (parameter, _) in enumerate(ring)
                }
                for first_outer, last_outer in zip(
                    outer_rows[:-1], outer_rows[1:], strict=True
                ):
                    add_internal_segment(first_outer, last_outer, occurrence)
                for index in range(len(inner_parameters) - 1):
                    first_outer = outer_rows[outer_index[inner_parameters[index]]]
                    last_outer = outer_rows[outer_index[inner_parameters[index + 1]]]
                    add_internal_segment(inner_rows[index], first_outer, occurrence)
                    add_internal_segment(inner_rows[index + 1], last_outer, occurrence)
                    pole_bands.append(
                        (
                            inner_rows[index],
                            inner_rows[index + 1],
                            first_outer,
                            last_outer,
                        )
                    )
                inner_parameters = tuple(parameter for parameter, _ in ring)
                inner_rows = outer_rows
        if len(set(traversed)) < 3:
            raise _failure(
                "a source trim loop must retain at least three distinct chart vertices",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(patch),),
            )
        loop_rows.append(tuple(traversed))
    # A graded ring cell whose radial sides carry source-curve samples much
    # denser than its ring edges is left to the quality triangulation, which
    # grades away from those samples; explicit ears would fan them to a far
    # corner. Pole fan cells always stay explicit.
    fixed_by_free: tuple[dict[Fraction, list[int]], ...] = ({}, {})
    for row, point in enumerate(points):
        for axis in range(2):
            fixed_by_free[axis].setdefault(point[axis], []).append(row)

    def side_rows(first: int, last: int, /) -> list[int]:
        start, stop = points[first], points[last]
        axis = 0 if start[0] == stop[0] else 1
        low, high = sorted((start[1 - axis], stop[1 - axis]))
        return sorted(
            (
                row
                for row in fixed_by_free[axis][start[axis]]
                if low <= points[row][1 - axis] <= high
            ),
            key=lambda row: points[row][1 - axis],
        )

    sides = {
        band: (side_rows(band[0], band[2]), side_rows(band[1], band[3]))
        for band in pole_bands
        if band[0] not in point_corners
    }
    crowded = {
        band: rows for band, rows in sides.items() if len(rows[0]) + len(rows[1]) > 4
    }
    if crowded:
        used = sorted(
            {row for band, rows in crowded.items() for row in (*band, *rows[0], *rows[1])}
        )
        physical = dict(
            zip(
                used,
                np.asarray(
                    domain.evaluate(
                        np.full((len(used),), patch, dtype=np.int64),
                        np.asarray(
                            [[float(value) for value in points[row]] for row in used],
                            dtype=np.float64,
                        ),
                    ),
                    dtype=np.float64,
                ),
                strict=True,
            )
        )

        def length(first: int, last: int, /) -> float:
            return float(np.linalg.norm(physical[first] - physical[last]))

        def explicit(band: tuple[int, int, int, int], /) -> bool:
            spacing = min(
                length(first, last)
                for side in crowded[band]
                for first, last in zip(side[:-1], side[1:], strict=True)
            )
            ring = max(length(band[0], band[1]), length(band[2], band[3]))
            return _POLE_SIDE_DENSITY * spacing >= ring

        pole_bands = [
            band for band in pole_bands if band not in crowded or explicit(band)
        ]
    constrained_pairs = {_edge_key(first, last) for first, last in segments}
    curves_by_point: dict[int, set[int]] = {}
    for row, curve in point_curve_parameters:
        curves_by_point.setdefault(row, set()).add(curve)
    for first, last, occurrence in internal_segments:
        edge = _edge_key(first, last)
        if edge in constrained_pairs or curves_by_point.get(
            first, set()
        ) & curves_by_point.get(last, set()):
            continue
        constrained_pairs.add(edge)
        segments.append((first, last))
        segment_curves.append(-2)
        segment_parameters.append((Fraction(0), Fraction(1)))
        segment_occurrences.append(occurrence)
        segment_chart_endpoints.append((points[first], points[last]))
        segment_pole_corners.append(None)

    for segment, curve in zip(segments, segment_curves, strict=True):
        if segment[0] == segment[1]:
            raise _failure(
                "source trim constraints contain a zero-length chart edge",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),) if curve >= 0 else (int(patch),),
            )
    loop_coordinates = tuple(tuple(points[index] for index in loop) for loop in loop_rows)
    return (
        points,
        np.asarray(segments, dtype=np.int64).reshape((-1, 2)),
        loop_coordinates,
        tuple(segment_curves),
        tuple(segment_parameters),
        point_curve_parameters,
        point_corners,
        tuple(segment_occurrences),
        tuple(segment_chart_endpoints),
        tuple(segment_pole_corners),
        pole_deviation_bound,
        tuple(pole_bands),
    )


def _source_curve_parameters(
    triangulation: ConstrainedDelaunayTriangulation,
    point_curve_parameters: dict[tuple[int, int], Fraction],
    segments: np.ndarray,
    segment_curves: tuple[int, ...],
    segment_parameters: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> dict[tuple[int, int], Fraction]:
    """Compose native CDT constraint ancestry to exact represented source parameters.

    A split whose parent is an end of its own constraint segment takes that
    segment's parameter there: the shared start/end row of a self-closing trim
    records the start, while its closing segment ends at the end of the turn.
    """
    points = np.asarray(triangulation.points, dtype=np.float64)
    input_count = triangulation.input_point_count
    sources = np.asarray(triangulation.constraint_sources, dtype=np.int64)
    parents = np.asarray(triangulation.constraint_parents, dtype=np.int64)
    ratios = np.asarray(triangulation.constraint_ratios, dtype=np.float64)
    if (
        sources.shape != (points.shape[0],)
        or parents.shape != (points.shape[0], 2)
        or ratios.shape != (points.shape[0],)
    ):
        raise ValueError(
            "Native CDT constraint ancestry must align with its point table."
        )
    result = dict(point_curve_parameters)
    composed: dict[int, list[Fraction]] = {}

    def parameter(vertex: int, curve: int, /) -> Fraction:
        try:
            return result[(vertex, curve)]
        except KeyError:
            segment = int(sources[vertex])
            if segment < 0 or segment >= len(segment_curves):
                raise ValueError(
                    f"Native CDT vertex {vertex} has no source-curve parameter ancestry."
                ) from None
            if segment_curves[segment] != curve:
                raise ValueError(
                    f"Native CDT vertex {vertex} names a different source curve."
                )
            first, last = (int(value) for value in parents[vertex])
            if not 0 <= first < vertex or not 0 <= last < vertex:
                raise ValueError(
                    f"Native CDT constraint ancestry is cyclic at vertex {vertex}."
                )
            ratio = float(ratios[vertex])
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"Native CDT source-curve ratio is invalid at vertex {vertex}."
                )
            endpoints = (int(segments[segment, 0]), int(segments[segment, 1]))
            low, high = (
                segment_parameters[segment][endpoints.index(parent)]
                if parent in endpoints
                else parameter(parent, curve)
                for parent in (first, last)
            )
            value = Fraction(float(low + Fraction.from_float(ratio) * (high - low)))
            # The native ratio is only known to binary64 resolution. When one source
            # curve bounds several chart sides (a periodic seam), independent splits
            # of the same source location differ only within that resolution; they
            # name one shared source vertex instead of seeding a sub-ulp feature.
            resolution = Fraction.from_float(
                8.0 * math.ulp(1.0) * float(abs(high - low))
                + 4.0 * math.ulp(float(value))
            )
            siblings = composed.setdefault(curve, [])
            value = next(
                (sibling for sibling in siblings if abs(sibling - value) <= resolution),
                value,
            )
            if value not in siblings:
                siblings.append(value)
            result[(vertex, curve)] = value
            return value

    for vertex in range(input_count, points.shape[0]):
        segment = int(sources[vertex])
        if segment < 0:
            continue
        if segment >= len(segment_curves):
            raise ValueError("Native CDT vertex names an absent source constraint.")
        if segment_curves[segment] < 0:
            continue
        parameter(vertex, segment_curves[segment])
    return result


def _pole_band_triangles(
    domain: domain_owner.MeshingDomain,
    patch: int,
    chart_points: list[tuple[Fraction, Fraction]],
    segments: np.ndarray,
    pole_bands: tuple[tuple[int, int, int, int], ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate each pole band cell by physically best ears.

    A band cell ``(a, b, c, d)`` is the chart rectangle between the inner side
    from ``a`` to ``b`` (the collapsed pole side or a ring) and the outer ring
    side from ``c`` to ``d``. Its boundary holds every chart point on these
    sides and on its radial sides: ring samples of different density and source
    seam samples. The convex cell is clipped one chart-positive ear at a time,
    always the ear of best physical shape, so a cell joins rings of different
    sample density without thin triangles. On the pole side every ear ends at
    the pole and the cell becomes its pole fan. Returns positively oriented
    chart triangles and, per opposite edge, the constraint segment row or
    ``-1``.
    """
    floats = np.asarray(
        [[float(value) for value in point] for point in chart_points],
        dtype=np.float64,
    )

    def chain(first: int, last: int, /) -> tuple[int, ...]:
        start, stop = chart_points[first], chart_points[last]
        low = np.minimum(floats[first], floats[last])
        high = np.maximum(floats[first], floats[last])
        candidates = np.flatnonzero(
            np.all((floats >= low) & (floats <= high), axis=1)
        ).tolist()
        direction = (stop[0] - start[0], stop[1] - start[1])
        rows = [
            row
            for row in candidates
            if (chart_points[row][0] - start[0]) * direction[1]
            == (chart_points[row][1] - start[1]) * direction[0]
        ]
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    (chart_points[row][0] - start[0]) * direction[0]
                    + (chart_points[row][1] - start[1]) * direction[1]
                ),
            )
        )

    polygons: list[list[int]] = []
    for inner_first, inner_last, outer_first, outer_last in pole_bands:
        polygons.append(
            [
                *chain(inner_first, inner_last)[:-1],
                *chain(inner_last, outer_last)[:-1],
                *chain(outer_last, outer_first)[:-1],
                *chain(outer_first, inner_first)[:-1],
            ]
        )
    polygon_rows = {row for polygon in polygons for row in polygon}
    edge_segments: dict[tuple[int, int], int] = {}
    for segment, (first, last) in enumerate(segments.tolist()):
        if first not in polygon_rows and last not in polygon_rows:
            continue
        rows = chain(int(first), int(last))
        for start, stop in zip(rows[:-1], rows[1:], strict=True):
            edge_segments[_edge_key(start, stop)] = segment
    used = sorted(polygon_rows)
    physical = dict(
        zip(
            used,
            np.asarray(
                domain.evaluate(
                    np.full((len(used),), patch, dtype=np.int64), floats[used]
                ),
                dtype=np.float64,
            ),
            strict=True,
        )
    )

    def chart_area(first: int, second: int, third: int, /) -> Fraction:
        a, b, c = chart_points[first], chart_points[second], chart_points[third]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def shape(first: int, second: int, third: int, /) -> float:
        a, b, c = physical[first], physical[second], physical[third]
        squares = float(
            np.sum((b - a) ** 2) + np.sum((c - b) ** 2) + np.sum((a - c) ** 2)
        )
        area = float(np.linalg.norm(np.cross(b - a, c - a)))
        return 0.0 if squares == 0.0 else area / squares

    triangles: list[tuple[int, int, int]] = []
    for polygon in polygons:
        orientation = sum(
            (
                chart_area(polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            ),
            Fraction(0),
        )
        if orientation == 0:
            raise _failure(
                "a pole band cell has zero chart area",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=(int(patch),),
            )
        sign = 1 if orientation > 0 else -1
        remaining = list(polygon)
        while len(remaining) > 3:
            # A clipped ear must keep chart area on both sides: the ear itself
            # and the remaining polygon, which may hold collinear side points.
            ears = {
                index: sign
                * chart_area(
                    remaining[index - 1],
                    remaining[index],
                    remaining[(index + 1) % len(remaining)],
                )
                for index in range(len(remaining))
            }
            best = max(
                (index for index, area in ears.items() if 0 < area < sign * orientation),
                key=lambda index: shape(
                    remaining[index - 1],
                    remaining[index],
                    remaining[(index + 1) % len(remaining)],
                ),
            )
            first, second, third = (
                remaining[best - 1],
                remaining[best],
                remaining[(best + 1) % len(remaining)],
            )
            triangles.append(
                (first, second, third) if sign > 0 else (first, third, second)
            )
            orientation -= sign * ears[best]
            del remaining[best]
        first, second, third = remaining
        triangles.append((first, second, third) if sign > 0 else (first, third, second))
    constraints = np.asarray(
        [
            [
                edge_segments.get(
                    _edge_key(triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3]),
                    -1,
                )
                for opposite in range(3)
            ]
            for triangle in triangles
        ],
        dtype=np.int64,
    )
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3)), constraints.reshape(
        (-1, 3)
    )


def _make_chart_mesh(
    domain: domain_owner.MeshingDomain,
    patch: int,
    parameters_by_curve: dict[int, set[Fraction]],
    pole_charts: _PoleCharts,
    minimum_angle: float,
    /,
) -> _ChartMesh:
    values = {
        curve: tuple(sorted(parameters))
        for curve, parameters in parameters_by_curve.items()
    }
    (
        input_points,
        segments,
        loops,
        segment_curves,
        segment_parameters,
        input_curve_parameters,
        input_corners,
        segment_occurrences,
        segment_chart_endpoints,
        segment_pole_corners,
        pole_deviation_bound,
        pole_bands,
    ) = _chart_inputs(domain, patch, values, pole_charts)
    point_array = np.asarray(
        [[float(component) for component in point] for point in input_points],
        dtype=np.float64,
    )
    # Pole bands are thin chart strips whose physical image is a cap of the
    # pole; chart-space quality refinement would split them into slivers of
    # vanishing physical size. They are holes of the quality triangulation and
    # receive explicit pole fans below.
    hole_points = (
        *(domain_boundary_owner._hole_seed(loop) for loop in loops[1:]),
        *(
            (
                sum((input_points[row][0] for row in band), Fraction(0)) / 4,
                sum((input_points[row][1] for row in band), Fraction(0)) / 4,
            )
            for band in pole_bands
        ),
    )
    holes = (
        None
        if not hole_points
        else np.asarray(
            [[float(component) for component in point] for point in hole_points],
            dtype=np.float64,
        )
    )
    try:
        triangulation = ConstrainedDelaunayTriangulation(
            point_array,
            segments,
            holes=holes,
            keep_convex_hull=False,
            min_angle=minimum_angle,
        )
    except (ValueError, RuntimeError) as error:
        raise _failure(
            f"native constrained chart triangulation failed for source patch {patch}: {error}",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(patch),),
        ) from error
    triangle_points = np.asarray(triangulation.points, dtype=np.float64)
    if triangle_points.shape[0] < len(input_points) or not np.array_equal(
        triangle_points[: len(input_points)], point_array
    ):
        raise _failure(
            "native CDT changed an authored trim chart point before construction",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(patch),),
        )
    source_parameters = _source_curve_parameters(
        triangulation,
        input_curve_parameters,
        segments,
        segment_curves,
        segment_parameters,
    )
    chart_points = list(input_points)
    corner_rows = dict(input_corners)
    constraint_sources = np.asarray(triangulation.constraint_sources, dtype=np.int64)
    constraint_parents = np.asarray(triangulation.constraint_parents, dtype=np.int64)
    constraint_ratios = np.asarray(triangulation.constraint_ratios, dtype=np.float64)
    for vertex in range(len(input_points), triangle_points.shape[0]):
        segment = int(constraint_sources[vertex])
        point = triangle_points[vertex]
        if not np.all(np.isfinite(point)):
            raise _failure(
                "native CDT generated a nonfinite trim point",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=(int(patch),),
            )
        if segment < 0:
            chart_points.append((Fraction(float(point[0])), Fraction(float(point[1]))))
            continue
        if segment >= len(segment_curves):
            raise ValueError("Native CDT vertex names an absent source constraint.")
        curve = segment_curves[segment]
        if curve < 0:
            first, last = (int(value) for value in constraint_parents[vertex])
            ratio = float(constraint_ratios[vertex])
            if (
                not 0 <= first < vertex
                or not 0 <= last < vertex
                or not math.isfinite(ratio)
                or not 0.0 <= ratio <= 1.0
            ):
                raise ValueError("Native CDT internal constraint ancestry is invalid.")
            uv = _interpolate_chart(
                chart_points[first],
                chart_points[last],
                Fraction.from_float(ratio),
            )
            chart_points.append(uv)
            corner = segment_pole_corners[segment]
            if corner is not None:
                corner_rows[vertex] = corner
            continue
        value = source_parameters[(vertex, curve)]
        low, high = segment_parameters[segment]
        start_uv, stop_uv = segment_chart_endpoints[segment]
        if low == high:
            raise _failure(
                "source p-curve constraint has a zero-width source parameter interval",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        fraction = (value - low) / (high - low)
        chart_points.append(_interpolate_chart(start_uv, stop_uv, fraction))
    if len(chart_points) != triangle_points.shape[0]:
        raise RuntimeError("Native surface chart point ancestry lost an output vertex.")
    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.size == 0:
        raise _failure(
            "native CDT produced no source surface triangles",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(patch),),
        )
    constraint_rows = np.asarray(triangulation.segment_ids, dtype=np.int64)
    if constraint_rows.shape != (triangles.shape[0], 3):
        raise ValueError("Native CDT constraint incidence must align with triangles.")
    if pole_bands:
        band_triangles, band_constraints = _pole_band_triangles(
            domain, patch, chart_points, np.asarray(segments, dtype=np.int64), pole_bands
        )
        triangles = np.concatenate((triangles, band_triangles))
        constraint_rows = np.concatenate((constraint_rows, band_constraints))
    signed_area = Fraction(0)
    for triangle in triangles.tolist():
        first, second, third = (chart_points[int(row)] for row in triangle)
        area = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )
        if area <= 0:
            raise _failure(
                "native CDT returned a non-positive source-chart triangle",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(patch),),
            )
        signed_area += area
    trim_area = sum(
        (
            first[0] * last[1] - last[0] * first[1]
            for loop in loops
            for first, last in zip(loop, (*loop[1:], loop[0]), strict=True)
        ),
        Fraction(0),
    )
    if signed_area != trim_area:
        raise _failure(
            "native CDT did not reproduce the complete exact source trim measure",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(patch),),
        )
    boundary_edges: dict[
        tuple[int, int], tuple[int, tuple[int, int, int], Fraction, Fraction]
    ] = {}
    for cell_row, triangle in enumerate(triangles.tolist()):
        for opposite in range(3):
            segment = int(constraint_rows[cell_row, opposite])
            if segment < 0:
                continue
            if segment >= len(segment_curves):
                raise ValueError("Native CDT edge names an absent source trim segment.")
            curve = segment_curves[segment]
            if curve < 0:
                continue
            first_vertex = int(triangle[(opposite + 1) % 3])
            last_vertex = int(triangle[(opposite + 2) % 3])
            # A segment end takes its segment's parameter: the shared row of a
            # self-closing trim ends its closing segment at the end of the turn.
            endpoints = (int(segments[segment, 0]), int(segments[segment, 1]))
            try:
                first_parameter, last_parameter = (
                    segment_parameters[segment][endpoints.index(vertex)]
                    if vertex in endpoints
                    else source_parameters[(vertex, curve)]
                    for vertex in (first_vertex, last_vertex)
                )
            except KeyError as error:
                raise ValueError(
                    "Native CDT trim edge endpoints lack canonical source-curve parameters."
                ) from error
            edge = _edge_key(first_vertex, last_vertex)
            if edge[0] == first_vertex:
                interval = first_parameter, last_parameter
            else:
                interval = last_parameter, first_parameter
            record = (
                curve,
                segment_occurrences[segment],
                interval[0],
                interval[1],
            )
            previous = boundary_edges.get(edge)
            if previous is not None and previous[0] != curve:
                raise _failure(
                    "one source chart edge is constrained by distinct source curves",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(previous[0]), int(curve)),
                )
            boundary_edges[edge] = record
    source_segments = {
        int(value) for value in np.unique(constraint_rows[constraint_rows >= 0]).tolist()
    }
    if source_segments != set(range(len(segment_curves))):
        raise _failure(
            "native CDT did not recover every authored source p-curve segment",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=tuple(sorted(set(range(len(segment_curves))) - source_segments)),
        )
    triangles_list: list[tuple[int, int, int]] = []
    for row in triangles.tolist():
        corners = tuple(corner_rows.get(int(vertex)) for vertex in row)
        retained_corners = tuple(corner for corner in corners if corner is not None)
        if len(set(retained_corners)) != len(retained_corners):
            continue
        triangles_list.append((int(row[0]), int(row[1]), int(row[2])))
    if not triangles_list:
        raise _failure(
            "collapsed source trims leave no positive surface triangles",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(patch),),
        )
    orientation = domain.patches[patch].orientation
    if orientation < 0:
        triangles_list = [
            (first, third, second) for first, second, third in triangles_list
        ]
    turning = {
        use.curve
        for loop in domain.patches[patch].loops
        for use in loop
        if isinstance(use, domain_owner.PatchCurveUse) and use.closes_chart_loop()
    }
    return _ChartMesh(
        patch,
        chart_points,
        triangles_list,
        corner_rows,
        source_parameters,
        boundary_edges,
        pole_deviation_bound,
        frozenset(
            key
            for key, value in input_curve_parameters.items()
            if key[1] in turning and value == 0
        ),
    )


def _trim_curves(
    domain: domain_owner.MeshingDomain, selected: tuple[int, ...], /
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                use.curve
                for patch in selected
                for loop in domain.patches[patch].loops
                for use in loop
                if isinstance(use, domain_owner.PatchCurveUse)
            }
        )
    )


def _quality_angle_degrees(
    coordinates: np.ndarray, triangles: np.ndarray, /
) -> np.ndarray:
    corners = coordinates[triangles]
    values = []
    for vertex in range(3):
        first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
        second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
        first_length = np.linalg.norm(first, axis=1)
        second_length = np.linalg.norm(second, axis=1)
        cosine = np.divide(
            np.sum(first * second, axis=1),
            first_length * second_length,
            out=np.ones((triangles.shape[0],), dtype=np.float64),
            where=(first_length * second_length) > 0.0,
        )
        values.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.min(np.stack(values, axis=1), axis=1)


def _chart_bounds(
    domain: domain_owner.MeshingDomain,
    selected: tuple[int, ...],
    scope_ids: tuple[np.ndarray, ...],
    coordinate_contract: SpatialCoordinateContract,
    preparation: domain_owner.SourcePreparation,
    /,
) -> _SurfaceRootCells:
    from ...geometry.brep._patches import BSplineSurfacePatch
    from ...geometry.brep._tspline import TSplineSurfacePatch

    spline_patches = tuple(
        patch
        for patch in selected
        if isinstance(
            domain.patches[patch].surface,
            (BSplineSurfacePatch, TSplineSurfacePatch),
        )
    )
    spline_atlas = (
        None
        if not spline_patches
        else surface_support_owner.prepare_surface_source_root_atlas(
            domain,
            spline_patches,
            coordinate_contract,
        )
    )
    root_ids: list[int] = []
    pieces: list[tuple[int, ...]] = []
    bounds: list[tuple[tuple[Fraction, Fraction], ...]] = []
    patches: list[int] = []
    spline_keys: list[str] = []
    spline_occurrences: list[tuple[str, ...]] = []
    if spline_atlas is not None:
        for row, (patch_value, piece, cell_bounds) in enumerate(
            zip(
                np.asarray(spline_atlas.root_patches, dtype=np.int64).tolist(),
                spline_atlas.root_source_cell_ids,
                spline_atlas.root_parameter_bounds,
                strict=True,
            )
        ):
            patch = int(patch_value)
            scope_id = _scope_id_for_row(scope_ids, 2, patch)
            entity_id, occurrence = domain.source_entity_key_for_scope(2, scope_id)
            spline_keys.append(
                canonical_fingerprint(
                    {
                        "kind": "native-surface-root-cell",
                        "source": (domain.source_id, domain.source_revision),
                        "patch": (entity_id, occurrence),
                        "piece": piece,
                        "bounds": tuple(
                            (str(first), str(last)) for first, last in cell_bounds
                        ),
                    }
                )
            )
            spline_occurrences.append(occurrence)
            pieces.append(tuple(int(value) for value in piece))
            bounds.append(
                tuple((Fraction(first), Fraction(last)) for first, last in cell_bounds)
            )
            patches.append(patch)
            root_ids.append(int(spline_atlas.root_cell_ids[row]))

    for patch in selected:
        if patch in spline_patches:
            continue
        scope_id = _scope_id_for_row(scope_ids, 2, patch)
        surface = domain.patches[patch].surface
        parameter_domain = surface.parameter_domain
        if parameter_domain is None:
            # Curved trims bulge past their endpoints (a full circle has one), so
            # their outward chart enclosures bound the root cell too.
            loop_points = tuple(
                point
                for loop in domain.patches[patch].loops
                for use in loop
                if isinstance(use, domain_owner.PatchCurveUse)
                for point in (
                    *_canonical_use_endpoints(use),
                    *_curved_use_box_corners(use, preparation),
                )
            )
            lower = tuple(min(point[axis] for point in loop_points) for axis in range(2))
            upper = tuple(max(point[axis] for point in loop_points) for axis in range(2))
        else:
            lower = tuple(
                Fraction.from_float(float(value)) for value in parameter_domain[0]
            )
            upper = tuple(
                Fraction.from_float(float(value)) for value in parameter_domain[1]
            )
        if any(lower[axis] >= upper[axis] for axis in range(2)):
            raise _failure(
                "source surface chart must have positive parameter extent",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(patch),),
            )
        corners = (
            (lower[0], lower[1]),
            (upper[0], lower[1]),
            (upper[0], upper[1]),
            (lower[0], upper[1]),
        )
        if any(
            not (lower[0] <= point[0] <= upper[0] and lower[1] <= point[1] <= upper[1])
            for loop in domain.patches[patch].loops
            for use in loop
            if isinstance(use, domain_owner.PatchCurveUse)
            for point in _canonical_use_endpoints(use)
        ):
            raise _failure(
                "source trim lies outside its authored surface parameter domain",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(patch),),
            )
        root_ids.append(scope_id)
        pieces.append((int(patch), int(domain.source_indices[2][patch])))
        bounds.append(corners)
        patches.append(patch)

    if spline_keys:
        spline_ids = _numeric_entity_ids(
            "native-surface-root-cell",
            tuple(spline_keys),
            tuple(spline_occurrences),
        )
        spline_position = 0
        for row, patch in enumerate(patches):
            if patch in spline_patches:
                root_ids[row] = spline_ids[spline_position]
                spline_position += 1
    if len(set(root_ids)) != len(root_ids):
        raise _failure(
            "source surface root-cell identity projection collides with a canonical patch scope ID",
            category=MeshingFailureCategory.INVALID_SOURCE,
        )
    root_parameters = np.asarray(
        [[[float(u), float(v)] for u, v in corners] for corners in bounds],
        dtype=np.float64,
    )
    return _SurfaceRootCells(
        domain,
        tuple(root_ids),
        tuple(pieces),
        tuple(bounds),
        np.asarray(patches, dtype=np.int64),
        root_parameters,
        spline_atlas,
    )


_ChartUV = tuple[Fraction, Fraction]
_ExactUVTriangle = tuple[_ChartUV, _ChartUV, _ChartUV]
_GlobalVertexKey = (
    tuple[Literal[0], int]
    | tuple[Literal[1], int, Fraction]
    | tuple[Literal[2], int, _ChartUV]
)
_LocalEdgeIdentity = (
    tuple[Literal[0], int, _ChartUV, _ChartUV]
    | tuple[Literal[1], int, Fraction, Fraction]
)


def _signed_uv_area2(points: tuple[_ChartUV, ...] | list[_ChartUV], /) -> Fraction:
    if len(points) < 3:
        return Fraction(0)
    return sum(
        (
            first[0] * last[1] - last[0] * first[1]
            for first, last in zip(points, (*points[1:], points[0]), strict=True)
        ),
        Fraction(0),
    )


def _root_edge_cross(
    first: _ChartUV,
    last: _ChartUV,
    point: _ChartUV,
    /,
) -> Fraction:
    return (last[0] - first[0]) * (point[1] - first[1]) - (last[1] - first[1]) * (
        point[0] - first[0]
    )


def _surface_root_charts(
    roots: SurfaceSourceCellAtlas, /
) -> dict[int, tuple[_SurfaceRootChart, ...]]:
    roots.require_current()
    by_patch: dict[int, list[_SurfaceRootChart]] = {}
    for patch_value, cell_id, piece, bounds in zip(
        roots.root_patches.tolist(),
        roots.root_cell_ids,
        roots.root_source_cell_ids,
        roots.root_parameter_bounds,
        strict=True,
    ):
        corners = tuple((Fraction(u), Fraction(v)) for u, v in bounds)
        if len(corners) != 4:
            raise _failure(
                "source-piece root chart must retain four exact parameter corners",
                category=MeshingFailureCategory.INVALID_SOURCE,
            )
        area2 = _signed_uv_area2(corners)
        if area2 == 0:
            raise _failure(
                "source-piece root chart must have nonzero exact parameter area",
                category=MeshingFailureCategory.INVALID_SOURCE,
            )
        orientation = 1 if area2 > 0 else -1
        for first, last in zip(corners, (*corners[1:], corners[0]), strict=True):
            if any(
                orientation * _root_edge_cross(first, last, point) < 0
                for point in corners
            ):
                raise _failure(
                    "source-piece root chart must be convex for exact triangle partitioning",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                )
        chart = _SurfaceRootChart(
            int(cell_id),
            tuple(int(value) for value in piece),
            corners,
            min(point[0] for point in corners),
            max(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[1] for point in corners),
            orientation,
        )
        by_patch.setdefault(int(patch_value), []).append(chart)
    return {
        patch: tuple(
            sorted(
                charts,
                key=lambda chart: (
                    chart.lower_u,
                    chart.upper_u,
                    chart.lower_v,
                    chart.cell_id,
                ),
            )
        )
        for patch, charts in by_patch.items()
    }


def _clip_uv_polygon(
    points: list[_ChartUV],
    root: _SurfaceRootChart,
    /,
) -> list[_ChartUV]:
    polygon = list(points)
    for first, last in zip(
        root.corners, (*root.corners[1:], root.corners[0]), strict=True
    ):
        if not polygon:
            return []
        previous_points = polygon
        polygon = []
        previous = previous_points[-1]
        previous_distance = root.orientation * _root_edge_cross(first, last, previous)
        previous_inside = previous_distance >= 0
        for current in previous_points:
            current_distance = root.orientation * _root_edge_cross(first, last, current)
            current_inside = current_distance >= 0
            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                if denominator == 0:
                    raise _failure(
                        "exact source-piece clipping encountered a parallel crossing",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                    )
                fraction = previous_distance / denominator
                if not 0 <= fraction <= 1:
                    raise _failure(
                        "exact source-piece clipping produced an invalid edge fraction",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                    )
                intersection = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                polygon.append(intersection)
            if current_inside:
                polygon.append(current)
            previous = current
            previous_distance = current_distance
            previous_inside = current_inside
        deduplicated: list[_ChartUV] = []
        for point in polygon:
            if not deduplicated or point != deduplicated[-1]:
                deduplicated.append(point)
        if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
            deduplicated.pop()
        polygon = deduplicated
    return polygon


def _segment_fraction(
    point: _ChartUV,
    first: _ChartUV,
    last: _ChartUV,
    /,
) -> Fraction | None:
    delta_u = last[0] - first[0]
    delta_v = last[1] - first[1]
    if delta_u != 0:
        fraction = (point[0] - first[0]) / delta_u
        if first[1] + fraction * delta_v != point[1]:
            return None
    elif delta_v != 0:
        fraction = (point[1] - first[1]) / delta_v
        if first[0] + fraction * delta_u != point[0]:
            return None
    else:
        return None
    return fraction if 0 < fraction < 1 else None


def _chart_point_row(
    chart: _ChartMesh,
    point_rows: dict[_ChartUV, int],
    point: _ChartUV,
    /,
) -> int:
    existing = point_rows.get(point)
    if existing is not None:
        return existing
    boundary_matches = [
        (edge, value, fraction)
        for edge, value in tuple(chart.boundary_edges.items())
        if (
            fraction := _segment_fraction(
                point, chart.points[edge[0]], chart.points[edge[1]]
            )
        )
        is not None
    ]
    if len(boundary_matches) > 1:
        raise _failure(
            "one exact source-piece cut point has multiple authored trim-edge ancestries",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(chart.patch),),
        )
    row = len(chart.points)
    chart.points.append(point)
    point_rows[point] = row
    if not boundary_matches:
        return row
    edge, boundary, fraction = boundary_matches[0]
    curve, occurrence, stored_first, stored_last = boundary
    first_parameter, last_parameter = chart.edge_parameters(
        edge, curve, (stored_first, stored_last)
    )
    if first_parameter is None or last_parameter is None:
        raise _failure(
            "source trim cut edge lost its exact endpoint parameter ancestry",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(curve),),
        )
    if (first_parameter, last_parameter) == (stored_first, stored_last):
        source_first_row, source_last_row = edge
        parameter = first_parameter + fraction * (last_parameter - first_parameter)
    elif (first_parameter, last_parameter) == (stored_last, stored_first):
        source_first_row, source_last_row = edge[1], edge[0]
        parameter = last_parameter + (1 - fraction) * (first_parameter - last_parameter)
    else:
        raise _failure(
            "source trim cut edge parameters disagree with its exact vertex ancestry",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(curve),),
        )
    if parameter in (stored_first, stored_last):
        raise _failure(
            "source trim cut reached an existing curve parameter without its canonical vertex",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(curve),),
        )
    chart.curve_parameters[(row, curve)] = parameter
    del chart.boundary_edges[edge]
    chart.boundary_edges[_edge_key(source_first_row, row)] = (
        curve,
        occurrence,
        stored_first,
        parameter,
    )
    chart.boundary_edges[_edge_key(row, source_last_row)] = (
        curve,
        occurrence,
        parameter,
        stored_last,
    )
    return row


def _split_chart_at_source_cells(
    chart: _ChartMesh,
    root_charts: tuple[_SurfaceRootChart, ...],
    /,
) -> None:
    if not root_charts:
        raise _failure(
            "selected surface patch has no exact source-piece root cells",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(chart.patch),),
        )
    point_rows = {point: row for row, point in enumerate(chart.points)}
    if len(point_rows) != len(chart.points):
        raise _failure(
            "surface chart contains duplicate exact parameter vertices",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(chart.patch),),
        )
    lower_us = tuple(root.lower_u for root in root_charts)
    children: list[tuple[int, int, int]] = []
    for triangle in tuple(chart.triangles):
        points = [chart.points[row] for row in triangle]
        triangle_area = _signed_uv_area2(points)
        if triangle_area == 0:
            raise _failure(
                "surface chart contains an exact degenerate parameter triangle",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(chart.patch),),
            )
        lower_u = min(point[0] for point in points)
        upper_u = max(point[0] for point in points)
        lower_v = min(point[1] for point in points)
        upper_v = max(point[1] for point in points)
        covered_area = Fraction(0)
        candidate_count = bisect_right(lower_us, upper_u)
        for root in root_charts[:candidate_count]:
            if root.upper_u < lower_u or root.lower_v > upper_v or root.upper_v < lower_v:
                continue
            polygon = _clip_uv_polygon(points, root)
            if len(polygon) < 3:
                continue
            polygon_area = _signed_uv_area2(polygon)
            if polygon_area == 0:
                continue
            if polygon_area * triangle_area < 0:
                raise _failure(
                    "exact source-piece clipping changed triangle orientation",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(chart.patch), root.cell_id),
                )
            covered_area += polygon_area
            polygon_rows = [
                _chart_point_row(chart, point_rows, point) for point in polygon
            ]
            for index in range(1, len(polygon_rows) - 1):
                child = (
                    polygon_rows[0],
                    polygon_rows[index],
                    polygon_rows[index + 1],
                )
                child_points = [chart.points[row] for row in child]
                child_area = _signed_uv_area2(child_points)
                if child_area == 0:
                    continue
                if child_area * triangle_area < 0:
                    child = (child[0], child[2], child[1])
                children.append(child)
        if covered_area != triangle_area:
            raise _failure(
                "exact source-piece root cells do not partition a refined surface triangle",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(chart.patch),),
            )
    chart.triangles = children
    chart.root_cells = root_charts


# A flip must raise the smaller physical minimum angle of its cell pair by more
# than this many degrees, so rounding in the angle evaluation cannot cycle flips.
_FLIP_ANGLE_GAIN = 1.0e-6


def _root_cell_row(
    roots: tuple[_SurfaceRootChart, ...],
    points: list[_ChartUV],
    root_boxes: np.ndarray,
    point_values: np.ndarray,
    /,
) -> int | None:
    """Row of the first convex root cell containing every chart point.

    ``root_boxes`` (rows ``lower_u, upper_u, lower_v, upper_v`` rounded to
    nearest) and ``point_values`` (the points rounded to nearest) screen out
    root cells in binary64: rounding to nearest is monotone, so a strict
    rounded comparison outside a box proves the exact point is outside it. Only
    the remaining candidates take the exact convexity test, so the first
    containing row is unchanged.
    """
    outside = (
        (point_values[None, :, 0] < root_boxes[:, None, 0])
        | (point_values[None, :, 0] > root_boxes[:, None, 1])
        | (point_values[None, :, 1] < root_boxes[:, None, 2])
        | (point_values[None, :, 1] > root_boxes[:, None, 3])
    )
    for row in np.flatnonzero(~np.any(outside, axis=1)).tolist():
        root = roots[row]
        if all(
            root.orientation * _root_edge_cross(first, last, point) >= 0
            for first, last in zip(
                root.corners, (*root.corners[1:], root.corners[0]), strict=True
            )
            for point in points
        ):
            return row
    return None


def _flip_to_physical_angles(
    chart: _ChartMesh,
    mapping: dict[int, int],
    physical: np.ndarray,
    minimum_angle: float,
    /,
) -> None:
    """Flip interior chart edges where the other diagonal is physically better shaped.

    The chart triangulation is shaped in parameters; where the surface map is
    anisotropic a well-shaped chart cell can be a physical sliver (a sphere chart
    near its poles). An edge shared by two cells whose quad is convex in the chart
    is flipped when the other diagonal raises the smaller physical minimum angle
    of the pair, so each flip raises the chart's sorted physical angle vector and
    the passes terminate. Only edges of a cell below ``minimum_angle`` degrees
    are tried. Trim edges, edges whose cells lie in different source-piece root
    cells, and every cell at a collapsed pole keep their place.
    """
    if not chart.triangles:
        return
    corner_rows: dict[int, list[int]] = {}
    for row, corner in chart.corners.items():
        corner_rows.setdefault(corner, []).append(row)
    pole_rows = {row for rows in corner_rows.values() if len(rows) > 1 for row in rows}
    triangles = chart.triangles

    def angles(cells: list[tuple[int, int, int]], /) -> list[float]:
        rows = np.asarray(
            [[mapping[row] for row in cell] for cell in cells], dtype=np.int64
        ).reshape((-1, 3))
        return _quality_angle_degrees(physical, rows).tolist()

    cell_angles = angles(triangles)
    sign = 1 if _root_edge_cross(*(chart.points[row] for row in triangles[0])) > 0 else -1
    roots = chart.root_cells if len(chart.root_cells) > 1 else ()
    root_boxes = np.asarray(
        [
            [float(root.lower_u), float(root.upper_u)]
            + [float(root.lower_v), float(root.upper_v)]
            for root in roots
        ],
        dtype=np.float64,
    ).reshape((-1, 4))
    point_values = np.asarray(
        [[float(u), float(v)] for u, v in chart.points], dtype=np.float64
    ).reshape((-1, 2))
    # Chart points never move during flipping, so each cell's containing root
    # row is resolved once for the whole routine.
    cell_roots: dict[tuple[int, ...], int | None] = {}

    def cell_root(cell: tuple[int, int, int], /) -> int | None:
        key = tuple(sorted(cell))
        if key not in cell_roots:
            cell_roots[key] = _root_cell_row(
                roots,
                [chart.points[row] for row in cell],
                root_boxes,
                point_values[list(cell)],
            )
        return cell_roots[key]

    while True:
        cells_by_edge: dict[tuple[int, int], list[int]] = {}
        for index, triangle in enumerate(triangles):
            for corner in range(3):
                cells_by_edge.setdefault(
                    _edge_key(triangle[corner], triangle[(corner + 1) % 3]), []
                ).append(index)
        touched: set[int] = set()
        for edge in sorted(cells_by_edge):
            cells = cells_by_edge[edge]
            if len(cells) != 2 or edge in chart.boundary_edges:
                continue
            first_cell, second_cell = cells
            if first_cell in touched or second_cell in touched:
                continue
            old = min(cell_angles[first_cell], cell_angles[second_cell])
            if old >= minimum_angle:
                continue
            first = triangles[first_cell]
            apex = next(row for row in first if row not in edge)
            opposite = next(row for row in triangles[second_cell] if row not in edge)
            if pole_rows.intersection((apex, opposite, *edge)):
                continue
            start = first.index(apex)
            left, right = first[(start + 1) % 3], first[(start + 2) % 3]
            candidates = [(apex, left, opposite), (apex, opposite, right)]
            if any(
                sign * _root_edge_cross(*(chart.points[row] for row in cell)) <= 0
                for cell in candidates
            ):
                continue
            if roots:
                first_root = cell_root(first)
                if first_root is None or first_root != cell_root(triangles[second_cell]):
                    continue
            new_angles = angles(candidates)
            if min(new_angles) <= old + _FLIP_ANGLE_GAIN:
                continue
            triangles[first_cell], triangles[second_cell] = candidates
            cell_angles[first_cell], cell_angles[second_cell] = new_angles
            touched.update((first_cell, second_cell))
        if not touched:
            return


def _size_field(
    domain: domain_owner.MeshingDomain,
    specification: SurfaceMeshingSpec,
    selected: tuple[int, ...],
    scope_ids: tuple[np.ndarray, ...],
    /,
) -> tuple[np.ndarray, tuple[str, ...]]:
    patch_ids = np.asarray(
        [_scope_id_for_row(scope_ids, 2, patch) for patch in selected],
        dtype=np.int64,
    )
    chart_centers = np.empty((len(selected), 2), dtype=np.float64)
    for row, patch in enumerate(selected):
        parameter_domain = domain.patches[patch].surface.parameter_domain
        if parameter_domain is not None:
            lower, upper = parameter_domain
            chart_centers[row] = (np.asarray(lower) + np.asarray(upper)) * 0.5
            continue
        points = tuple(
            point
            for loop in domain.patches[patch].loops
            for use in loop
            for point in _canonical_use_endpoints(use)
        )
        chart_centers[row] = np.mean(
            np.asarray([[float(value) for value in point] for point in points]),
            axis=0,
        )
    centers = domain.evaluate(np.asarray(selected, dtype=np.int64), chart_centers)
    controls = tuple(specification.size_controls)
    try:
        resolved, report = resolve_size_controls(
            controls,
            centers,
            patch_ids,
            SizeFieldDomain.SURFACE_GEODESIC,
            combination=specification.size_combination,
        )
    except (TypeError, ValueError) as error:
        raise _failure(
            f"uniform source-patch size controls do not resolve: {error}",
            category=MeshingFailureCategory.CONTROL_CONFLICT,
            entity_ids=tuple(int(value) for value in patch_ids.tolist()),
        ) from error
    sizes = np.asarray(resolved.values, dtype=np.float64)
    if (
        sizes.shape != (len(selected),)
        or np.any(~np.isfinite(sizes))
        or np.any(sizes <= 0.0)
    ):
        raise _failure(
            "resolved source-patch sizes must be positive and finite",
            category=MeshingFailureCategory.CONTROL_CONFLICT,
            entity_ids=tuple(int(value) for value in patch_ids.tolist()),
        )
    return sizes, (report.report_id,)


def _source_curve_point(
    domain: domain_owner.MeshingDomain,
    curve: int,
    parameter: Fraction,
    /,
) -> np.ndarray:
    owner = tuple(int(value) for value in domain.curve_owners[curve].tolist())
    if owner == (-1, -1, -1):
        raise _failure(
            "source curve has no owning parameter chart",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(curve),),
        )
    patch, loop, use_index = owner
    use = domain.patches[patch].loops[loop][use_index]
    if not isinstance(use, domain_owner.PatchCurveUse):
        raise _failure(
            "source curve owner is not an authored p-curve",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=(int(curve),),
        )
    uv = _use_chart_point(use, parameter)
    return domain.evaluate(
        np.asarray((patch,), dtype=np.int64),
        np.asarray([[float(uv[0]), float(uv[1])]], dtype=np.float64),
    )[0]


def _box_chord_bound(
    bounds: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> float:
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (2, 3) or np.any(~np.isfinite(box)) or np.any(box[0] > box[1]):
        return math.inf
    corners = np.asarray(
        [
            (x, y, z)
            for x in (box[0, 0], box[1, 0])
            for y in (box[0, 1], box[1, 1])
            for z in (box[0, 2], box[1, 2])
        ],
        dtype=np.float64,
    )
    chord = np.asarray(last, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    squared_length = float(np.dot(chord, chord))
    if squared_length == 0.0:
        distances = np.linalg.norm(corners - np.asarray(first, dtype=np.float64), axis=1)
    else:
        ratios = np.clip(
            (corners - np.asarray(first, dtype=np.float64)) @ chord / squared_length,
            0.0,
            1.0,
        )
        distances = np.linalg.norm(
            corners - (np.asarray(first, dtype=np.float64) + ratios[:, None] * chord),
            axis=1,
        )
    maximum = float(np.max(distances, initial=0.0))
    return float(np.nextafter(maximum, math.inf))


def _box_point_distance_upper(
    bounds: np.ndarray,
    point: np.ndarray,
    /,
) -> float:
    box = np.asarray(bounds, dtype=np.float64)
    origin = np.asarray(point, dtype=np.float64)
    if (
        box.shape != (2, 3)
        or origin.shape != (3,)
        or np.any(~np.isfinite(box))
        or np.any(~np.isfinite(origin))
        or np.any(box[0] > box[1])
    ):
        return math.inf
    corners = np.asarray(
        [
            (x, y, z)
            for x in (box[0, 0], box[1, 0])
            for y in (box[0, 1], box[1, 1])
            for z in (box[0, 2], box[1, 2])
        ],
        dtype=np.float64,
    )
    maximum = float(np.max(np.linalg.norm(corners - origin, axis=1), initial=0.0))
    return float(np.nextafter(maximum, math.inf))


def _positive_surface_parameter_box(
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    /,
) -> np.ndarray:
    box = np.array(parameter_box, dtype=np.float64, copy=True)
    if box.shape != (2, 2) or np.any(~np.isfinite(box)) or np.any(box[0] > box[1]):
        raise ValueError("Surface chart enclosure must be a finite ordered UV box.")
    domain = surface.parameter_domain
    for axis in range(2):
        if box[0, axis] != box[1, axis]:
            continue
        lower = float(np.nextafter(box[0, axis], -math.inf))
        upper = float(np.nextafter(box[1, axis], math.inf))
        if domain is not None:
            lower = max(lower, domain[0][axis])
            upper = min(upper, domain[1][axis])
        if not lower < upper:
            raise ValueError(
                "A degenerate p-curve interval has no representable surface-chart neighborhood."
            )
        box[0, axis] = lower
        box[1, axis] = upper
    return box


def _surface_mapped_curve_bounds(
    surface: AbstractSurfacePatch,
    pcurve: AbstractCurve,
    first: float,
    last: float,
    preparation: domain_owner.SourcePreparation,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Chart box of a p-curve interval and the box of its mapped image."""
    chart_box = preparation.curve_box(pcurve, first, last)
    chart_box = _positive_surface_parameter_box(surface, chart_box)
    return chart_box, preparation.surface_box(surface, chart_box)


def _mapped_curve_is_c1(
    surface: AbstractSurfacePatch,
    pcurve: AbstractCurve,
    chart_box: np.ndarray,
    first: float,
    last: float,
    /,
) -> bool:
    """Established C1 continuity of ``surface o pcurve`` on one interval.

    An affine placement keeps the continuity class of its source surface.
    """
    source = (
        surface.source
        if isinstance(surface, domain_owner._PlacedSurfacePatch)
        else surface
    )
    return pcurve.is_c1_on(first, last) and source.is_c1_on(chart_box)


def _second_order_chord_bound(
    program: PreparedIntervalFunction,
    endpoints: tuple[np.ndarray, np.ndarray],
    native: tuple[float, float],
    chord: tuple[np.ndarray, np.ndarray],
    /,
) -> float:
    """Distance bound from a mapped p-curve interval to a chord.

    Componentwise linear interpolation of ``g = surface o pcurve`` over a
    parameter interval of width ``h`` deviates by at most
    ``h^2 / 8 * sup|g_i''|``; the interpolant of ``g``'s endpoints is within
    the larger endpoint offset of the chord between the canonical points.
    """
    lower, upper = min(native), max(native)
    try:
        low, high = program.evaluate(
            np.asarray(((lower,),), dtype=np.float64),
            np.asarray(((upper,),), dtype=np.float64),
        )
    except (IntervalExtensionError, NotImplementedError):
        return math.inf
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return math.inf
    magnitudes = np.maximum(np.abs(low), np.abs(high)).reshape((-1,))
    width = Fraction(upper) - Fraction(lower)
    squared = sum((Fraction(float(value)) ** 2 for value in magnitudes), Fraction(0))
    interpolation = fraction_sqrt_interval(squared)[1] * width * width / 8
    offset = max(
        float(np.linalg.norm(point - target))
        for point, target in zip(endpoints, chord, strict=True)
    )
    bound = float(fraction_endpoint(interpolation, upper=True, saturate=True)) + offset
    return float(np.nextafter(np.nextafter(bound, math.inf), math.inf))


@final
@dataclass(slots=True)
class _CurveChordBounds:
    """Continuous chord deviation and length bounds of source-curve intervals.

    Every authored p-curve use bounds its mapped interval by the smaller of a
    box enclosure of the mapped curve (first order in the interval width) and
    the second-order interpolation bound of `_second_order_chord_bound`, so
    sampling converges at the curvature rate instead of to the tolerance
    itself. Interval bounds are cached here; boxes and second-derivative
    programs come from the caller's source preparation.
    """

    preparation: domain_owner.SourcePreparation
    intervals: dict[tuple[int, Fraction, Fraction], tuple[float, float]] = field(
        default_factory=dict
    )

    @property
    def domain(self) -> domain_owner.MeshingDomain:
        return self.preparation.domain

    def interval(
        self, curve: int, lower: Fraction, upper: Fraction, /
    ) -> tuple[float, float]:
        key = (int(curve), lower, upper)
        cached = self.intervals.get(key)
        if cached is None:
            cached = self._interval(int(curve), lower, upper)
            self.intervals[key] = cached
        return cached

    def _interval(
        self, curve: int, lower: Fraction, upper: Fraction, /
    ) -> tuple[float, float]:
        domain = self.domain
        first = _source_curve_point(domain, curve, lower)
        last = _source_curve_point(domain, curve, upper)
        deviation = 0.0
        length = 0.0
        uses = domain.curve_uses(curve)
        if not uses:
            raise _failure(
                "source curve has no authored chart use",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        for row, (patch, use) in enumerate(uses):
            if not isinstance(use, domain_owner.PatchCurveUse):
                continue
            surface = domain.patches[patch].surface
            native_first = float(_use_source_parameter(use, lower))
            native_last = float(_use_source_parameter(use, upper))
            interval_first = min(native_first, native_last)
            interval_last = max(native_first, native_last)
            try:
                chart_box, bounds = _surface_mapped_curve_bounds(
                    surface, use.pcurve, interval_first, interval_last, self.preparation
                )
            except (ArithmeticError, NotImplementedError, TypeError, ValueError) as error:
                raise _failure(
                    f"source p-curve has no finite mapped continuous interval enclosure: {error}",
                    category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    entity_ids=(int(curve),),
                ) from error
            if bounds.shape != (2, 3) or np.any(~np.isfinite(bounds)):
                raise _failure(
                    "source p-curve has no finite mapped continuous interval enclosure",
                    category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    entity_ids=(int(curve),),
                )
            use_deviation = _box_chord_bound(bounds, first, last)
            program = (
                self.preparation.mapped_curve_hessian(surface, use.pcurve)
                if _mapped_curve_is_c1(
                    surface, use.pcurve, chart_box, interval_first, interval_last
                )
                else None
            )
            if program is not None:
                endpoints = tuple(
                    domain.evaluate(
                        np.asarray((patch,), dtype=np.int64),
                        np.asarray(
                            [
                                [
                                    float(value)
                                    for value in _use_chart_point(use, parameter)
                                ]
                            ],
                            dtype=np.float64,
                        ),
                    )[0]
                    for parameter in (lower, upper)
                )
                use_deviation = min(
                    use_deviation,
                    _second_order_chord_bound(
                        program,
                        (endpoints[0], endpoints[1]),
                        (native_first, native_last),
                        (first, last),
                    ),
                )
            deviation = max(deviation, use_deviation)
            diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
            length = max(length, float(np.nextafter(diagonal, math.inf)))
        if not math.isfinite(deviation) or not math.isfinite(length):
            raise _failure(
                "source p-curve continuous chord or length enclosure is nonfinite",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=(int(curve),),
            )
        return deviation, length


def _curve_parameters(
    domain: domain_owner.MeshingDomain,
    curve_ids: tuple[int, ...],
    selected_patches: tuple[int, ...],
    patch_sizes: np.ndarray,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    preparation: domain_owner.SourcePreparation,
    /,
) -> dict[int, set[Fraction]]:
    sizes = {
        patch: float(size)
        for patch, size in zip(selected_patches, patch_sizes, strict=True)
    }
    parameters: dict[int, set[Fraction]] = {}
    hard_tolerances: dict[int, float] = {}
    for feature in specification.protected_features:
        if feature.feature_kind.value == "curve" and feature.hard:
            for identifier in np.asarray(
                feature.scope.entity_ids, dtype=np.int64
            ).tolist():
                hard_tolerances[int(identifier)] = min(
                    hard_tolerances.get(int(identifier), math.inf),
                    feature.maximum_deviation,
                )
    chords = _CurveChordBounds(preparation)
    for curve in curve_ids:
        selected_uses = tuple(
            (patch, use) for patch, use in domain.curve_uses(curve) if patch in sizes
        )
        if not selected_uses:
            raise _failure(
                "selected source chart references a curve outside its selected surface scope",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        edge_limit = schedule.spacing_fraction * min(
            sizes[patch] for patch, _ in selected_uses
        )
        fidelity_limits = [edge_limit]
        if specification.maximum_deviation is not None:
            fidelity_limits.append(specification.maximum_deviation)
        elif domain.tolerance > 0.0:
            fidelity_limits.append(domain.tolerance)
        if curve in hard_tolerances:
            fidelity_limits.append(hard_tolerances[curve])
        fidelity_limit = min(fidelity_limits)
        initial = {
            Fraction(index, schedule.minimum_edge_samples - 1)
            for index in range(schedule.minimum_edge_samples)
        }
        if domain.curves[curve].start == domain.curves[curve].end:
            initial.add(Fraction(1, 2))
        expanded = set(initial)
        ordered = sorted(initial)
        for first, last in zip(ordered[:-1], ordered[1:], strict=True):
            intervals = [(first, last)]
            while intervals:
                lower, upper = intervals.pop()
                deviation, length = chords.interval(curve, lower, upper)
                if deviation <= fidelity_limit and length <= edge_limit:
                    continue
                midpoint = (lower + upper) / 2
                if midpoint in (lower, upper) or float(midpoint) in (
                    float(lower),
                    float(upper),
                ):
                    raise _failure(
                        "source curve sampling reached parameter representability before meeting continuous chord bounds",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        entity_ids=(int(curve),),
                    )
                expanded.add(midpoint)
                intervals.append((midpoint, upper))
                intervals.append((lower, midpoint))
        parameters[curve] = expanded
    return parameters


def _surface_fidelity_tolerance(
    domain: domain_owner.MeshingDomain, specification: SurfaceMeshingSpec, /
) -> float | None:
    """Source deviation bound every surface cell must meet, if any."""
    tolerance = specification.maximum_deviation
    if (
        tolerance is None
        and domain.accuracy in ("bounded_approximation", "certified_enclosure")
        and domain.tolerance > 0.0
    ):
        tolerance = domain.tolerance
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0.0):
        raise _failure(
            "surface fidelity tolerance must be finite and nonnegative",
            category=MeshingFailureCategory.INVALID_SPECIFICATION,
        )
    return tolerance


# Pole fans and graded rings. The cap ring keeps every fan apex angle at the
# pole within _POLE_FAN_ANGLE; each graded ring at most doubles the physical
# ring radius (or adds one edge target), keeps ring chords within
# _POLE_RING_CHORD_RATIO of its radial gap, and the last ring is the first whose
# chart metric along the ring reaches _POLE_RING_ISOTROPY of the metric across
# it, where chart-isotropic quality triangulation is physically sound again.
_POLE_FAN_ANGLE = math.pi / 3
_POLE_RING_PROBES = 64
_POLE_RING_GROWTH = 2.0
_POLE_RING_CHORD_RATIO = 2.5
_POLE_RING_ISOTROPY = 0.5
_POLE_RING_EXTENT = Fraction(7, 16)
_POLE_RING_BISECTIONS = 24
# A ring cell stays explicit while its radial-side samples are no denser than
# its ring edges divided by _POLE_SIDE_DENSITY; denser sides are left to the
# quality triangulation.
_POLE_SIDE_DENSITY = 4.0


def _pole_cap_bound(
    surface: AbstractSurfacePatch,
    use: domain_owner.PatchPoleUse,
    lower: Fraction,
    upper: Fraction,
    fixed_axis: int,
    pole_fixed: float,
    ring_fixed: float,
    pole_point: np.ndarray,
    preparation: domain_owner.SourcePreparation,
    /,
) -> float:
    """Bound the distance from the pole of the cap between the pole and its ring.

    The cap is the chart box over the interval's p-curve between the pole row and
    the ring row. Every cap point joins the pole row, whose image is the pole,
    along the fixed chart axis, so its distance from the pole is at most the
    enclosed fixed-axis Jacobian norm times the ring offset. A box enclosure of
    the source map widens with the long chart extent of the cap, which this
    bound does not; the smaller of both bounds is returned.
    """
    from phydrax.geometry._source_fidelity import _surface_jacobian_bounds

    native_first = float(_use_source_parameter(use, lower))
    native_last = float(_use_source_parameter(use, upper))
    chart_box = preparation.curve_box(
        use.pcurve, min(native_first, native_last), max(native_first, native_last)
    )
    if (
        chart_box.shape != (2, 2)
        or np.any(~np.isfinite(chart_box))
        or np.any(chart_box[0] > chart_box[1])
    ):
        raise ValueError("Pole p-curve interval has no finite ordered chart box.")
    cap_box = np.array(chart_box, copy=True)
    cap_box[:, fixed_axis] = (min(pole_fixed, ring_fixed), max(pole_fixed, ring_fixed))
    cap_bounds = preparation.surface_box(
        surface, _positive_surface_parameter_box(surface, cap_box)
    )
    if cap_bounds.shape != (2, 3):
        raise ValueError("Pole surface chart boxes must realize finite 3D bounds.")
    bound = _box_point_distance_upper(cap_bounds, pole_point)
    corners = tuple(
        (
            Fraction.from_float(float(cap_box[row, 0])),
            Fraction.from_float(float(cap_box[column, 1])),
        )
        for row, column in ((0, 0), (1, 0), (1, 1), (0, 1))
    )
    halves = tuple(
        _surface_jacobian_bounds(surface, triangle, preparation)
        for triangle in (
            (corners[0], corners[1], corners[2]),
            (corners[0], corners[2], corners[3]),
        )
    )
    if halves[0] is not None and halves[1] is not None:
        speed = float(
            np.linalg.norm(
                np.maximum(
                    np.maximum(np.abs(halves[0][0]), np.abs(halves[0][1])),
                    np.maximum(np.abs(halves[1][0]), np.abs(halves[1][1])),
                )[:, fixed_axis]
            )
        )
        offset = abs(Fraction.from_float(ring_fixed) - Fraction.from_float(pole_fixed))
        # The norm and product carry a few binary64 roundings; widen past them.
        swept = float(Fraction.from_float(speed) * offset) * (
            1.0 + 16.0 * float(np.finfo(np.float64).eps)
        )
        if math.isfinite(swept):
            bound = min(bound, math.nextafter(swept, math.inf))
    return bound


def _pole_ring_extent(
    domain: domain_owner.MeshingDomain,
    patch: int,
    loop_index: int,
    use_index: int,
    fixed_axis: int,
    pole_fixed: float,
    inside_fixed: float,
    preparation: domain_owner.SourcePreparation,
    /,
) -> Fraction:
    """Largest graded ring offset as a fraction of the pole-to-interior chart span.

    Graded rings stay below half the distance to the nearest other trim that
    crosses the pole's open chart strip, so rings of facing poles never meet.
    Trims along the strip ends, such as seams leaving the pole, do not cross it.
    """
    loops = domain.patches[patch].loops
    use = loops[loop_index][use_index]
    free_axis = 1 - fixed_axis
    strip = preparation.curve_box(
        use.pcurve, min(use.first, use.last), max(use.first, use.last)
    )
    low, high = float(strip[0, free_axis]), float(strip[1, free_axis])
    tolerance = 128.0 * np.finfo(np.float64).eps * max(abs(low), abs(high), 1.0)
    span = inside_fixed - pole_fixed
    nearest = 1.0
    for other_loop_index, other_loop in enumerate(loops):
        for other_use_index, other in enumerate(other_loop):
            if (other_loop_index, other_use_index) == (loop_index, use_index):
                continue
            box = preparation.curve_box(
                other.pcurve, min(other.first, other.last), max(other.first, other.last)
            )
            if (
                box[1, free_axis] <= low + tolerance
                or box[0, free_axis] >= high - tolerance
            ):
                continue
            offsets = sorted(
                float(value) for value in (box[:, fixed_axis] - pole_fixed) / span
            )
            if offsets[1] <= 0.0:
                continue
            nearest = min(nearest, max(offsets[0], 0.0))
    return Fraction.from_float(nearest) * _POLE_RING_EXTENT


def _pole_graded_rings(
    domain: domain_owner.MeshingDomain,
    patch: int,
    chart_point: Callable[[Fraction], tuple[Fraction, Fraction]],
    fixed_axis: int,
    ring_distance: Callable[[Fraction], tuple[float, float]],
    inner_alpha: Fraction,
    alpha_limit: Fraction,
    stops: tuple[Fraction, ...],
    inner_parameters: list[Fraction],
    edge_limit: float,
    /,
) -> tuple[_PoleRing, ...]:
    """Grade rings from the cap ring toward the regular chart.

    ``ring_distance(alpha)`` returns the fixed chart coordinate at fraction
    ``alpha`` of the pole-to-interior span and its physical distance from the
    pole. Without these rings the chart-isotropic quality triangulation would
    join the small cap ring to vertices far from the pole by needle triangles.
    ``stops`` are the increasing offsets of source-curve samples on the strip
    ends; a ring snaps onto a stop within its step (or within half a step
    beyond it), so no curve sample sits just beside a ring.
    """
    free_axis = 1 - fixed_axis
    free_values: dict[Fraction, Fraction] = {}

    def free_value(parameter: Fraction, /) -> Fraction:
        value = free_values.get(parameter)
        if value is None:
            value = chart_point(parameter)[free_axis]
            free_values[parameter] = value
        return value

    def ring_point(parameter: Fraction, fixed: Fraction, /) -> tuple[Fraction, Fraction]:
        point = [fixed, fixed]
        point[free_axis] = free_value(parameter)
        return point[0], point[1]

    def evaluate(
        parameters: list[Fraction], fixed: Fraction, /
    ) -> dict[Fraction, np.ndarray]:
        values = np.asarray(
            domain.evaluate(
                np.full((len(parameters),), patch, dtype=np.int64),
                np.asarray(
                    [
                        [float(value) for value in ring_point(parameter, fixed)]
                        for parameter in parameters
                    ],
                    dtype=np.float64,
                ),
            ),
            dtype=np.float64,
        )
        return {parameter: values[row] for row, parameter in enumerate(parameters)}

    rings: list[_PoleRing] = []
    parameters = list(inner_parameters)
    alpha = inner_alpha
    value, radius = ring_distance(alpha)
    while alpha < alpha_limit:
        target = min(_POLE_RING_GROWTH * radius, radius + edge_limit)
        next_alpha = alpha_limit
        next_value, next_radius = ring_distance(alpha_limit)
        if next_radius > target:
            next_alpha, next_value, next_radius = alpha, value, radius
            low, high = alpha, alpha_limit
            for _ in range(_POLE_RING_BISECTIONS):
                middle = (low + high) / 2
                middle_value, middle_radius = ring_distance(middle)
                if middle_radius <= target:
                    low = next_alpha = middle
                    next_value, next_radius = middle_value, middle_radius
                else:
                    high = middle
        # A ring snaps down onto the last curve stop within its step, or up
        # onto the next stop when that lies within half a step beyond it, so
        # no stop sits just beside a ring. Stops denser than the rings stay
        # radial side points of their band cells.
        step = next_radius - radius
        within = [stop for stop in stops if alpha < stop <= next_alpha]
        beyond = next((stop for stop in stops if stop > next_alpha), None)
        if within:
            next_alpha = within[-1]
            next_value, next_radius = ring_distance(next_alpha)
        elif beyond is not None and beyond <= alpha_limit:
            beyond_value, beyond_radius = ring_distance(beyond)
            if beyond_radius - next_radius <= step / 2:
                next_alpha, next_value, next_radius = beyond, beyond_value, beyond_radius
        # A ring leaving less than half its step to the extent would bound a
        # thin last band, whose short radial gap forces a dense ring beside the
        # coarse chart beyond; the ring moves onto the extent instead.
        if next_alpha < alpha_limit:
            limit_value, limit_radius = ring_distance(alpha_limit)
            if limit_radius - next_radius < (next_radius - radius) / 2:
                next_alpha, next_value, next_radius = (
                    alpha_limit,
                    limit_value,
                    limit_radius,
                )
        gap = next_radius - radius
        if next_alpha == alpha or gap <= 0.0 or next_value == value:
            break
        fixed = Fraction.from_float(next_value)
        chord_limit = min(edge_limit, _POLE_RING_CHORD_RATIO * gap)
        physical = evaluate(parameters, fixed)
        while True:
            ordered = sorted(physical)
            midpoints = [
                (first + last) / 2
                for first, last in zip(ordered[:-1], ordered[1:], strict=True)
                if np.linalg.norm(physical[last] - physical[first]) > chord_limit
            ]
            if not midpoints:
                break
            physical.update(evaluate(midpoints, fixed))
        rings.append(
            tuple((parameter, ring_point(parameter, fixed)) for parameter in ordered)
        )
        along = min(
            float(np.linalg.norm(physical[last] - physical[first]))
            / abs(float(free_value(last) - free_value(first)))
            for first, last in zip(ordered[:-1], ordered[1:], strict=True)
            if free_value(last) != free_value(first)
        )
        across = gap / abs(next_value - value)
        parameters, alpha, value, radius = ordered, next_alpha, next_value, next_radius
        if along >= _POLE_RING_ISOTROPY * across:
            break
    return tuple(rings)


def _pole_ring_stops(
    loop: tuple[domain_owner.PatchCurveUse | domain_owner.PatchPoleUse, ...],
    use_index: int,
    fixed_axis: int,
    pole_fixed: float,
    inside_fixed: float,
    alpha_limit: Fraction,
    parameters_by_curve: dict[int, set[Fraction]],
    /,
) -> tuple[Fraction, ...]:
    """Offsets of the samples of the trims leaving the pole along its strip ends."""
    use = loop[use_index]
    free_axis = 1 - fixed_axis
    ends = {point[free_axis] for point in _canonical_use_endpoints(use)}
    pole = Fraction.from_float(pole_fixed)
    span = Fraction.from_float(inside_fixed) - pole
    stops: set[Fraction] = set()
    for neighbor in (loop[use_index - 1], loop[(use_index + 1) % len(loop)]):
        if not isinstance(neighbor, domain_owner.PatchCurveUse):
            continue
        for parameter in parameters_by_curve.get(neighbor.curve, ()):
            point = _use_chart_point(neighbor, parameter)
            offset = (point[fixed_axis] - pole) / span
            if point[free_axis] in ends and 0 < offset < alpha_limit:
                stops.add(offset)
    return tuple(sorted(stops))


def _pole_sampling(
    domain: domain_owner.MeshingDomain,
    selected_patches: tuple[int, ...],
    patch_sizes: np.ndarray,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    parameters_by_curve: dict[int, set[Fraction]],
    preparation: domain_owner.SourcePreparation,
    /,
) -> _PoleCharts:
    sizes = {
        patch: float(size)
        for patch, size in zip(selected_patches, patch_sizes, strict=True)
    }
    result: _PoleCharts = {}
    fidelity_tolerance = _surface_fidelity_tolerance(domain, specification)
    for patch in selected_patches:
        surface = domain.patches[patch].surface
        parameter_domain = surface.parameter_domain
        for loop_index, loop in enumerate(domain.patches[patch].loops):
            for use_index, use in enumerate(loop):
                if not isinstance(use, domain_owner.PatchPoleUse):
                    continue
                pcurve_bounds = preparation.curve_box(
                    use.pcurve, min(use.first, use.last), max(use.first, use.last)
                )
                if (
                    pcurve_bounds.shape != (2, 2)
                    or np.any(~np.isfinite(pcurve_bounds))
                    or np.any(pcurve_bounds[0] > pcurve_bounds[1])
                ):
                    raise _failure(
                        "collapsed pole p-curve has no finite ordered chart enclosure",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                extents = pcurve_bounds[1] - pcurve_bounds[0]
                fixed_axis = int(np.argmin(extents))
                chart_scale = max(float(np.max(extents)), 1.0)
                if (
                    extents[1 - fixed_axis]
                    <= 128.0 * np.finfo(np.float64).eps * chart_scale
                ):
                    raise _failure(
                        "collapsed pole p-curve must traverse a positive chart interval",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                if extents[fixed_axis] > 128.0 * np.finfo(np.float64).eps * chart_scale:
                    raise _failure(
                        "collapsed pole p-curve must follow one singular chart coordinate",
                        category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                        entity_ids=(int(patch),),
                    )
                pole_fixed = float(
                    (pcurve_bounds[0, fixed_axis] + pcurve_bounds[1, fixed_axis]) * 0.5
                )
                pole_point = np.asarray(
                    domain.corner_points[use.corner], dtype=np.float64
                )
                midpoint = _use_chart_point(use, Fraction(1, 2))
                candidate_values = []
                if parameter_domain is not None:
                    candidate_values.extend(
                        (
                            float(parameter_domain[0][fixed_axis]),
                            float(parameter_domain[1][fixed_axis]),
                        )
                    )
                for other_loop_index, other_loop in enumerate(
                    domain.patches[patch].loops
                ):
                    for other_use_index, other_use in enumerate(other_loop):
                        if (
                            other_loop_index == loop_index
                            and other_use_index == use_index
                        ):
                            continue
                        candidate_values.extend(
                            float(point[fixed_axis])
                            for point in _canonical_use_endpoints(other_use)
                        )
                candidate_fixed = tuple(
                    value
                    for value in dict.fromkeys(candidate_values)
                    if value != pole_fixed
                )
                if not candidate_fixed:
                    raise _failure(
                        "collapsed pole chart has no authored regular interior coordinate",
                        category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                        entity_ids=(int(patch),),
                    )
                candidate_distances = []
                for value in candidate_fixed:
                    chart = [float(midpoint[0]), float(midpoint[1])]
                    chart[fixed_axis] = value
                    physical = domain.evaluate(
                        np.asarray((patch,), dtype=np.int64),
                        np.asarray((chart,), dtype=np.float64),
                    )[0]
                    candidate_distances.append(
                        float(np.linalg.norm(np.asarray(physical) - pole_point))
                    )
                inside_fixed = candidate_fixed[int(np.argmax(candidate_distances))]
                if inside_fixed == pole_fixed or not math.isfinite(inside_fixed):
                    raise _failure(
                        "collapsed pole chart has no regular interior direction",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                edge_limit = schedule.spacing_fraction * sizes[patch]
                cap_limits = [edge_limit * 0.25]
                if specification.maximum_deviation is not None:
                    cap_limits.append(specification.maximum_deviation * 0.25)
                elif domain.tolerance > 0.0:
                    cap_limits.append(domain.tolerance * 0.25)
                cap_limit = min(cap_limits)
                if cap_limit <= 128.0 * np.finfo(np.float64).eps * max(domain.scale, 1.0):
                    raise _failure(
                        "collapsed pole cap is below physical representability for the requested fidelity",
                        category=MeshingFailureCategory.COMPLIANCE_FAILED,
                        entity_ids=(int(patch),),
                    )

                ring_distance_limit = cap_limit * 0.5

                def pointwise_ring_distance(alpha: Fraction, /) -> tuple[float, float]:
                    ring_value = float(
                        Fraction.from_float(pole_fixed)
                        + alpha
                        * (
                            Fraction.from_float(inside_fixed)
                            - Fraction.from_float(pole_fixed)
                        )
                    )
                    chart = [float(midpoint[0]), float(midpoint[1])]
                    chart[fixed_axis] = ring_value
                    physical = domain.evaluate(
                        np.asarray((patch,), dtype=np.int64),
                        np.asarray((chart,), dtype=np.float64),
                    )[0]
                    distance = float(np.linalg.norm(np.asarray(physical) - pole_point))
                    return ring_value, float(np.nextafter(distance, math.inf))

                lower_alpha = Fraction(0)
                upper_alpha = Fraction(1)
                ring_fixed, lower_distance = pointwise_ring_distance(lower_alpha)
                if lower_distance > cap_limit:
                    raise _failure(
                        "source pole does not coincide with its authored corner",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                upper_ring, upper_distance = pointwise_ring_distance(upper_alpha)
                if upper_distance <= ring_distance_limit:
                    lower_alpha, ring_fixed = upper_alpha, upper_ring
                else:
                    while True:
                        midpoint_alpha = (lower_alpha + upper_alpha) / 2
                        if midpoint_alpha in (lower_alpha, upper_alpha) or float(
                            midpoint_alpha
                        ) in (float(lower_alpha), float(upper_alpha)):
                            break
                        candidate_ring, candidate_distance = pointwise_ring_distance(
                            midpoint_alpha
                        )
                        if candidate_distance <= ring_distance_limit:
                            lower_alpha = midpoint_alpha
                            ring_fixed = candidate_ring
                        else:
                            upper_alpha = midpoint_alpha
                if lower_alpha == 0:
                    raise _failure(
                        "collapsed pole chart has no representable regular ring within the requested fidelity",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        entity_ids=(int(patch),),
                    )

                def chart_point(parameter: Fraction, /) -> tuple[Fraction, Fraction]:
                    return _use_chart_point(use, parameter)

                def ring_chart_point(parameter: Fraction, /) -> tuple[Fraction, Fraction]:
                    point = list(chart_point(parameter))
                    point[fixed_axis] = Fraction.from_float(ring_fixed)
                    return point[0], point[1]

                # The cap ring count follows the size field: no ring interval
                # longer than the edge target and no pole fan apex angle above
                # _POLE_FAN_ANGLE. Enclosure bounds below may only add samples.
                ring_radius = pointwise_ring_distance(lower_alpha)[1]
                probes = tuple(
                    ring_chart_point(Fraction(index, _POLE_RING_PROBES))
                    for index in range(_POLE_RING_PROBES + 1)
                )
                probe_points = np.asarray(
                    domain.evaluate(
                        np.full((len(probes),), patch, dtype=np.int64),
                        np.asarray(
                            [[float(value) for value in point] for point in probes],
                            dtype=np.float64,
                        ),
                    ),
                    dtype=np.float64,
                )
                ring_length = float(
                    np.sum(np.linalg.norm(np.diff(probe_points, axis=0), axis=1))
                )
                interval_count = max(
                    1,
                    schedule.minimum_edge_samples - 1,
                    math.ceil(ring_length / (ring_radius * _POLE_FAN_ANGLE)),
                    math.ceil(ring_length / edge_limit),
                )
                initial = {
                    Fraction(index, interval_count) for index in range(interval_count + 1)
                }
                accepted_bounds: dict[tuple[Fraction, Fraction], float] = {}
                expanded = set(initial)
                ordered = sorted(initial)
                for first, last in zip(ordered[:-1], ordered[1:], strict=True):
                    intervals = [(first, last)]
                    while intervals:
                        interval_first, interval_last = intervals.pop()
                        ring_first = ring_chart_point(interval_first)
                        ring_last = ring_chart_point(interval_last)
                        physical_first = domain.evaluate(
                            np.asarray((patch,), dtype=np.int64),
                            np.asarray(
                                [[float(ring_first[0]), float(ring_first[1])]],
                                dtype=np.float64,
                            ),
                        )[0]
                        physical_last = domain.evaluate(
                            np.asarray((patch,), dtype=np.int64),
                            np.asarray(
                                [[float(ring_last[0]), float(ring_last[1])]],
                                dtype=np.float64,
                            ),
                        )[0]
                        # An interval is accepted when its ring chord meets the
                        # edge target, its cap stays within the cap limit of the
                        # pole, and its pole fan cells meet the refinement bounds
                        # as laid out: refining a fan would raise the pole degree.
                        fans = np.asarray(
                            [
                                [
                                    [float(value) for value in chart_point(pole)],
                                    [float(value) for value in ring_last],
                                    [float(value) for value in ring_first],
                                ]
                                for pole in (interval_first, interval_last)
                            ],
                            dtype=np.float64,
                        )
                        accepted = (
                            float(np.linalg.norm(physical_last - physical_first))
                            <= edge_limit
                            and (
                                fidelity_tolerance is None
                                or float(
                                    np.max(
                                        domain.interpolation_bounds(
                                            patch, fans, preparation=preparation
                                        )
                                    )
                                )
                                <= fidelity_tolerance
                            )
                            and (
                                specification.maximum_normal_deviation is None
                                or float(
                                    np.max(
                                        domain.normal_turn_bounds(
                                            patch, fans, preparation=preparation
                                        )
                                    )
                                )
                                <= specification.maximum_normal_deviation
                            )
                        )
                        if accepted:
                            cap_bound = _pole_cap_bound(
                                surface,
                                use,
                                interval_first,
                                interval_last,
                                fixed_axis,
                                pole_fixed,
                                ring_fixed,
                                pole_point,
                                preparation,
                            )
                            if cap_bound <= cap_limit:
                                accepted_bounds[(interval_first, interval_last)] = (
                                    cap_bound
                                )
                                continue
                        midpoint_parameter = (interval_first + interval_last) / 2
                        if midpoint_parameter in (interval_first, interval_last) or float(
                            midpoint_parameter
                        ) in (float(interval_first), float(interval_last)):
                            raise _failure(
                                "collapsed pole ring reached parameter representability before meeting continuous size bounds",
                                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                                entity_ids=(int(patch),),
                            )
                        expanded.add(midpoint_parameter)
                        intervals.append((midpoint_parameter, interval_last))
                        intervals.append((interval_first, midpoint_parameter))
                ordered = sorted(expanded)
                samples = tuple(
                    _PoleChartSample(
                        parameter,
                        chart_point(parameter),
                        ring_chart_point(parameter),
                        accepted_bounds.get((parameter, ordered[index + 1]), 0.0)
                        if index + 1 < len(ordered)
                        else 0.0,
                    )
                    for index, parameter in enumerate(ordered)
                )
                if samples[0].chart_point == samples[-1].chart_point:
                    raise _failure(
                        "collapsed pole trim must retain distinct chart endpoints",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                alpha_limit = _pole_ring_extent(
                    domain,
                    patch,
                    loop_index,
                    use_index,
                    fixed_axis,
                    pole_fixed,
                    inside_fixed,
                    preparation,
                )
                result[(patch, loop_index, use_index)] = _PoleChart(
                    samples,
                    _pole_graded_rings(
                        domain,
                        patch,
                        chart_point,
                        fixed_axis,
                        pointwise_ring_distance,
                        lower_alpha,
                        alpha_limit,
                        _pole_ring_stops(
                            loop,
                            use_index,
                            fixed_axis,
                            pole_fixed,
                            inside_fixed,
                            alpha_limit,
                            parameters_by_curve,
                        ),
                        ordered,
                        edge_limit,
                    ),
                )
    return result


def _integrate_pole_ring_parameters(
    domain: domain_owner.MeshingDomain,
    selected_patches: tuple[int, ...],
    pole_charts: _PoleCharts,
    parameters_by_curve: dict[int, set[Fraction]],
    /,
) -> None:
    """Add ring points lying on a loop's trims to those trims' parameters.

    A ring point on a straight trim (constant along one chart axis) is solved
    for its trim parameter; when the trim's own chart point at that parameter
    differs from the ring point only by the rounding of the p-curve evaluation,
    the ring point is replaced by the trim's chart point so the pole spoke and
    the trim segment share the vertex instead of overlapping collinearly.
    """
    for patch in selected_patches:
        for loop_index, loop in enumerate(domain.patches[patch].loops):
            poles = tuple(
                (index, use)
                for index, use in enumerate(loop)
                if isinstance(use, domain_owner.PatchPoleUse)
            )
            if not poles:
                continue
            replaced: dict[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = {}
            for use in loop:
                if not isinstance(use, domain_owner.PatchCurveUse):
                    continue
                parameters = parameters_by_curve.get(use.curve)
                if parameters is None:
                    continue
                first, last = _canonical_use_endpoints(use)
                # A ring point already sampled on the curve keeps its parameter:
                # its chart inverse may differ from it in the last bits.
                known = {_use_chart_point(use, value) for value in parameters}
                for pole_use_index, _pole_use in poles:
                    pole_chart = pole_charts[(patch, loop_index, pole_use_index)]
                    ring_points = (
                        *(sample.ring_point for sample in pole_chart.samples),
                        *(point for ring in pole_chart.rings for _, point in ring),
                    )
                    for point in ring_points:
                        if point in known or point in replaced:
                            continue
                        for axis in range(2):
                            delta = last[axis] - first[axis]
                            if delta == 0:
                                continue
                            parameter = (point[axis] - first[axis]) / delta
                            if not 0 <= parameter <= 1:
                                continue
                            chart = _use_chart_point(use, parameter)
                            if chart != point and not _rounded_straight_trim_point(
                                first, last, axis, point, chart
                            ):
                                continue
                            parameters.add(parameter)
                            known.add(chart)
                            if chart != point:
                                replaced[point] = chart
                            break
            if replaced:
                for pole_use_index, _pole_use in poles:
                    key = (patch, loop_index, pole_use_index)
                    pole_chart = pole_charts[key]
                    pole_charts[key] = _PoleChart(
                        tuple(
                            _PoleChartSample(
                                sample.parameter,
                                sample.chart_point,
                                replaced.get(sample.ring_point, sample.ring_point),
                                sample.cap_bound,
                            )
                            for sample in pole_chart.samples
                        ),
                        tuple(
                            tuple(
                                (parameter, replaced.get(point, point))
                                for parameter, point in ring
                            )
                            for ring in pole_chart.rings
                        ),
                    )


def _rounded_straight_trim_point(
    first: tuple[Fraction, Fraction],
    last: tuple[Fraction, Fraction],
    axis: int,
    point: tuple[Fraction, Fraction],
    chart: tuple[Fraction, Fraction],
    /,
) -> bool:
    """Whether ``chart`` is ``point`` on a straight trim up to p-curve rounding.

    The trim keeps the other chart coordinate constant and ``point`` lies on
    it; the trim's evaluated coordinate along ``axis`` then differs from the
    solved one only by the binary64 rounding of the p-curve evaluation.
    """
    other = 1 - axis
    if not first[other] == last[other] == point[other] == chart[other]:
        return False
    value = float(point[axis])
    return abs(float(chart[axis]) - value) <= 4.0 * math.ulp(value)


def _global_vertex_key(
    domain: domain_owner.MeshingDomain,
    chart: _ChartMesh,
    local_row: int,
    /,
) -> _GlobalVertexKey:
    corner = chart.corners.get(local_row)
    if corner is not None:
        return (0, corner)
    curves = tuple(
        (curve, parameter)
        for (row, curve), parameter in chart.curve_parameters.items()
        if row == local_row
    )
    if len(curves) > 1:
        raise _failure(
            "one source-chart vertex has conflicting canonical source-curve ancestry",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=tuple(sorted(curve for curve, _ in curves)),
        )
    if curves:
        curve, parameter = curves[0]
        return (1, curve, parameter)
    surface = domain.patches[chart.patch].surface
    parameter_domain = surface.parameter_domain
    point = list(chart.points[local_row])
    if parameter_domain is not None:
        symbols = surface.native_period_symbols()
        periods = surface.periods
        for axis, symbol, period in zip(range(2), symbols, periods, strict=True):
            if symbol is None or period is None:
                continue
            lower = Fraction.from_float(parameter_domain[0][axis])
            upper = Fraction.from_float(parameter_domain[1][axis])
            if (
                upper - lower == Fraction.from_float(float(period))
                and point[axis] == upper
            ):
                point[axis] = lower
    return (2, chart.patch, (point[0], point[1]))


def _local_edge_identity(
    chart: _ChartMesh,
    first: int,
    last: int,
    /,
) -> _LocalEdgeIdentity:
    edge = _edge_key(first, last)
    boundary = chart.boundary_edges.get(edge)
    if boundary is not None:
        curve, _, first_parameter, last_parameter = boundary
        return (
            1,
            curve,
            min(first_parameter, last_parameter),
            max(first_parameter, last_parameter),
        )
    first_uv, last_uv = (chart.points[row] for row in edge)
    return (0, chart.patch, first_uv, last_uv)


def _add_global_vertex(
    domain: domain_owner.MeshingDomain,
    chart: _ChartMesh,
    local_row: int,
    global_rows: dict[_GlobalVertexKey, int],
    coordinates: list[np.ndarray],
    source_dimensions: list[int],
    source_indices: list[int],
    source_parameters: list[tuple[float, float]],
    /,
) -> int:
    key = _global_vertex_key(domain, chart, local_row)
    uv = chart.points[local_row]
    if key[0] == 0:
        dimension, index = 0, int(key[1])
        point = np.asarray(domain.corner_points[index], dtype=np.float64)
        parameter = (math.nan, math.nan)
    elif key[0] == 1:
        dimension, index = 1, int(key[1])
        source_parameter = key[2]
        assert isinstance(source_parameter, Fraction)
        point = _source_curve_point(domain, index, source_parameter)
        chart_point = domain.evaluate(
            np.asarray((chart.patch,), dtype=np.int64),
            np.asarray([[float(uv[0]), float(uv[1])]], dtype=np.float64),
        )[0]
        allowance = max(
            domain.tolerance * domain.scale,
            64.0 * np.finfo(np.float64).eps * max(domain.scale, 1.0),
        )
        if np.linalg.norm(point - chart_point) > allowance:
            raise _failure(
                "a source p-curve does not evaluate to its canonical source-curve point",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(index,),
            )
        parameter = (float(source_parameter), math.nan)
    else:
        dimension, index = 2, chart.patch
        point = domain.evaluate(
            np.asarray((chart.patch,), dtype=np.int64),
            np.asarray([[float(uv[0]), float(uv[1])]], dtype=np.float64),
        )[0]
        parameter = (float(uv[0]), float(uv[1]))
    if not np.all(np.isfinite(point)):
        raise _failure(
            "source chart evaluation returned a nonfinite physical vertex",
            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            entity_ids=(int(index),),
        )
    row = global_rows.get(key)
    if row is None:
        row = len(coordinates)
        global_rows[key] = row
        coordinates.append(np.asarray(point, dtype=np.float64))
        source_dimensions.append(dimension)
        source_indices.append(index)
        source_parameters.append(parameter)
    else:
        previous = coordinates[row]
        allowance = max(
            domain.tolerance * domain.scale,
            64.0 * np.finfo(np.float64).eps * max(domain.scale, 1.0),
        )
        if np.linalg.norm(previous - point) > allowance:
            raise _failure(
                "shared source chart uses disagree at one source corner or curve parameter",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(index),),
            )
    return row


def _chart_triangles(
    chart: _ChartMesh,
    local_to_global: dict[int, int],
    /,
) -> np.ndarray:
    return np.asarray(
        [tuple(local_to_global[row] for row in triangle) for triangle in chart.triangles],
        dtype=np.int64,
    )


def _triangle_boundary_curves(
    chart: _ChartMesh, triangle: tuple[int, int, int], /
) -> tuple[int, ...]:
    found = {
        chart.boundary_edges[_edge_key(first, last)][0]
        for first, last in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
        if _edge_key(first, last) in chart.boundary_edges
    }
    return tuple(sorted(found))


def _chart_curve_bounds(
    chart: _ChartMesh, chords: _CurveChordBounds, /
) -> dict[tuple[int, int], float]:
    """Chord deviation bound of every trim edge; shared intervals are bounded once."""
    bounds: dict[tuple[int, int], float] = {}
    for edge, (
        curve,
        _occurrence,
        stored_first,
        stored_last,
    ) in chart.boundary_edges.items():
        first, last = chart.edge_parameters(edge, curve, (stored_first, stored_last))
        if first is None or last is None or first == last:
            raise _failure(
                "source trim edge lacks a positive canonical curve interval",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        bounds[edge] = chords.interval(curve, min(first, last), max(first, last))[0]
    return bounds


_FOLD_PROBE_STEP = 1.0e-6
# A normal aim also probes each cell this far from a corner toward its centroid.
_CORNER_PROBE_WEIGHT = 0.9


def _facet_normal_violations(
    domain: domain_owner.MeshingDomain,
    patches: np.ndarray,
    chart_parameters: np.ndarray,
    corners: np.ndarray,
    normal_aim: float | None,
    /,
) -> np.ndarray:
    """Mark cells whose facet opposes or turns away from the oriented source normal.

    The source normal is sampled by central differences of the source map at the
    chart centroid and, with a normal aim, also near each corner inside the cell.
    A cell is marked when its vertex order opposes the sampled normal or when the
    facet turns from it by more than the aim. A chart cell that is well shaped
    in parameters can still be a physical sliver whose facet normal departs from
    the source although its chord deviation and size meet the schedule. This
    only selects cells to refine; certification does not rely on it.
    """
    centroids = np.mean(chart_parameters, axis=1)
    probe_points = [centroids]
    if normal_aim is not None:
        probe_points.extend(
            _CORNER_PROBE_WEIGHT * chart_parameters[:, corner]
            + (1.0 - _CORNER_PROBE_WEIGHT) * centroids
            for corner in range(chart_parameters.shape[1])
        )
    points = np.concatenate(probe_points)
    steps = _FOLD_PROBE_STEP * np.maximum(1.0, np.abs(points))
    probes = np.concatenate(
        (
            points + steps * np.asarray((1.0, 0.0)),
            points - steps * np.asarray((1.0, 0.0)),
            points + steps * np.asarray((0.0, 1.0)),
            points - steps * np.asarray((0.0, 1.0)),
        )
    )
    values = np.asarray(
        domain.evaluate(np.tile(patches, 4 * len(probe_points)), probes),
        dtype=np.float64,
    ).reshape((4, len(probe_points), centroids.shape[0], 3))
    normal = np.cross(values[0] - values[1], values[2] - values[3])
    orientation = np.asarray(
        [domain.patches[int(patch)].orientation for patch in patches.tolist()],
        dtype=np.float64,
    )
    facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    alignment = orientation * np.sum(facet * normal, axis=-1)
    marked = alignment[0] <= 0.0
    if normal_aim is not None:
        lengths = np.linalg.norm(facet, axis=-1) * np.linalg.norm(normal, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosines = alignment / lengths
        marked |= np.any(~(cosines >= math.cos(normal_aim)), axis=0)
    return marked


type _CellBoundKey = tuple[int, tuple[_ChartUV, ...]]
type _TrimIdentity = tuple[Literal[1], int, Fraction, Fraction]
# One lockstep refinement round of owner-local charts: given this owner's
# requested trim-edge splits and whether it still has schedule violations,
# return the trim splits every incident owner requested and whether any owner
# still has violations.
type _SeamRounds = Callable[
    [frozenset[_TrimIdentity], bool],
    tuple[frozenset[_TrimIdentity], bool],
]


def _cached_cell_bounds(
    preparation: domain_owner.SourcePreparation,
    chart: _ChartMesh,
    parameters: np.ndarray,
    cache: dict[_CellBoundKey, tuple[float, float]],
    normal_turns: bool,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell source interpolation and normal-turn bounds, evaluated once per cell.

    Both source bounds are cellwise functions of the exact UV corners, so cells
    retained across refinement rounds reuse their bound; new cells are bounded
    together in one source batch.
    """
    keys = tuple(
        (chart.patch, tuple(chart.points[row] for row in triangle))
        for triangle in chart.triangles
    )
    missing = [row for row, key in enumerate(keys) if key not in cache]
    if missing:
        rows = np.asarray(missing, dtype=np.int64)
        domain = preparation.domain
        deviations = np.asarray(
            domain.interpolation_bounds(
                chart.patch, parameters[rows], preparation=preparation
            ),
            dtype=np.float64,
        )
        normals = (
            np.asarray(
                domain.normal_turn_bounds(
                    chart.patch, parameters[rows], preparation=preparation
                ),
                dtype=np.float64,
            )
            if normal_turns
            else np.zeros((rows.size,), dtype=np.float64)
        )
        for row, deviation, normal in zip(
            missing, deviations.tolist(), normals.tolist(), strict=True
        ):
            cache[keys[row]] = (deviation, normal)
    values = np.asarray([cache[key] for key in keys], dtype=np.float64).reshape((-1, 2))
    return values[:, 0], values[:, 1]


def _refine_charts(
    domain: domain_owner.MeshingDomain,
    charts: list[_ChartMesh],
    sizes: np.ndarray,
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    preparation: domain_owner.SourcePreparation,
    /,
    *,
    seam_rounds: _SeamRounds | None = None,
) -> _RefinedSurface:
    """Refine source charts until every cell meets the physical schedule.

    Bad cells are bisected along their longest-edge propagation path inside
    their own chart; a trim edge is a chart boundary, so a path ending there
    splits the trim edge in every incident chart. Refinement of a chart therefore
    depends on other charts only through the trim splits they request, which
    `seam_rounds` exchanges between owner-local chart sets in lockstep rounds.
    """
    global_rows: dict[_GlobalVertexKey, int] = {}
    coordinates: list[np.ndarray] = []
    source_dimensions: list[int] = []
    source_indices: list[int] = []
    source_parameters: list[tuple[float, float]] = []
    patch_sizes = {
        chart.patch: float(size) for chart, size in zip(charts, sizes, strict=True)
    }
    local_rows: list[dict[int, int]] = []
    for chart in charts:
        local_rows.append(
            {
                row: _add_global_vertex(
                    domain,
                    chart,
                    row,
                    global_rows,
                    coordinates,
                    source_dimensions,
                    source_indices,
                    source_parameters,
                )
                for row in range(len(chart.points))
            }
        )

    fidelity_tolerance = _surface_fidelity_tolerance(domain, specification)
    protected_curve_tolerance: dict[int, float] = {}
    for feature in specification.protected_features:
        if feature.feature_kind.value == "curve" and feature.hard:
            for identifier in np.asarray(
                feature.scope.entity_ids, dtype=np.int64
            ).tolist():
                protected_curve_tolerance[int(identifier)] = min(
                    protected_curve_tolerance.get(int(identifier), math.inf),
                    feature.maximum_deviation,
                )

    # Trim-edge chord bounds change only when a round splits a trim edge, and each
    # curve interval is bounded once. Unchanged cells keep their exact UV corners
    # and therefore their source bounds.
    chords = _CurveChordBounds(preparation)
    curve_bounds_by_chart = tuple(_chart_curve_bounds(chart, chords) for chart in charts)
    trim_edges = tuple(frozenset(chart.boundary_edges) for chart in charts)
    cell_bounds: dict[_CellBoundKey, tuple[float, float]] = {}
    quality_angle = schedule.quality_angle_degrees
    if specification.quality_target is not None:
        quality_angle = max(
            quality_angle, math.degrees(specification.quality_target.minimum_angle)
        )
    while True:
        # Each round's cells are reshaped physically before they are bounded.
        physical = np.asarray(coordinates, dtype=np.float64)
        for chart, mapping in zip(charts, local_rows, strict=True):
            _flip_to_physical_angles(chart, mapping, physical, quality_angle)
        triangles_by_chart = tuple(
            _chart_triangles(chart, mapping)
            for chart, mapping in zip(charts, local_rows, strict=True)
        )
        triangles = np.concatenate(triangles_by_chart, axis=0)
        patches = np.concatenate(
            tuple(
                np.full(rows.shape[0], chart.patch, dtype=np.int64)
                for chart, rows in zip(charts, triangles_by_chart, strict=True)
            )
        )
        chart_parameters = np.concatenate(
            tuple(
                np.asarray(
                    [
                        [chart.points[row] for row in triangle]
                        for triangle in chart.triangles
                    ],
                    dtype=np.float64,
                )
                for chart in charts
            ),
            axis=0,
        )
        physical = np.asarray(coordinates, dtype=np.float64)
        edge_pairs = triangles[:, ((0, 1), (1, 2), (2, 0))]
        edge_lengths = np.linalg.norm(
            physical[edge_pairs[:, :, 0]] - physical[edge_pairs[:, :, 1]], axis=-1
        )
        maximum_edge = np.max(edge_lengths, axis=1)
        deviations = np.empty((triangles.shape[0],), dtype=np.float64)
        normal_bounds = np.zeros_like(deviations)
        deviation_targets = np.full(
            (triangles.shape[0],),
            math.inf if fidelity_tolerance is None else fidelity_tolerance,
        )
        offset = 0
        for chart, curve_bounds in zip(charts, curve_bounds_by_chart, strict=True):
            count = len(chart.triangles)
            local_parameters = chart_parameters[offset : offset + count]
            deviations[offset : offset + count], normal_values = _cached_cell_bounds(
                preparation,
                chart,
                local_parameters,
                cell_bounds,
                specification.maximum_normal_deviation is not None,
            )
            if specification.maximum_normal_deviation is not None:
                normal_bounds[offset : offset + count] = normal_values
            for row, triangle in enumerate(chart.triangles):
                boundary_bound = max(
                    (
                        curve_bounds.get(_edge_key(first, last), 0.0)
                        for first, last in (
                            (triangle[0], triangle[1]),
                            (triangle[1], triangle[2]),
                            (triangle[2], triangle[0]),
                        )
                    ),
                    default=0.0,
                )
                deviations[offset + row] = max(
                    float(deviations[offset + row]),
                    boundary_bound,
                    chart.pole_deviation_bound,
                )
                limits = tuple(
                    protected_curve_tolerance[curve]
                    for curve in _triangle_boundary_curves(chart, triangle)
                    if curve in protected_curve_tolerance
                )
                if limits:
                    feature_tolerance = min(limits)
                    if fidelity_tolerance is not None:
                        feature_tolerance = min(fidelity_tolerance, feature_tolerance)
                    deviation_targets[offset + row] = feature_tolerance
            offset += count
        if np.any(~np.isfinite(deviations)):
            failed = np.flatnonzero(~np.isfinite(deviations))
            raise _failure(
                "source interpolation enclosure is unresolved over a represented chart cell",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=tuple(
                    int(value) for value in np.unique(patches[failed]).tolist()
                ),
            )
        if specification.maximum_normal_deviation is not None and np.any(
            ~np.isfinite(normal_bounds)
        ):
            failed = np.flatnonzero(~np.isfinite(normal_bounds))
            raise _failure(
                "source normal-turn enclosure is unresolved over a represented chart cell",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=tuple(
                    int(value) for value in np.unique(patches[failed]).tolist()
                ),
            )
        edge_targets = np.asarray(
            [schedule.spacing_fraction * patch_sizes[int(patch)] for patch in patches],
            dtype=np.float64,
        )
        bad = (maximum_edge > edge_targets) | (deviations > deviation_targets)
        if specification.maximum_normal_deviation is not None:
            bad |= normal_bounds > specification.maximum_normal_deviation
        # A chart triangle near a singular chart row can map to a cell whose
        # vertex order reverses the oriented source normal, and an anisotropic
        # chart can map a well-shaped chart triangle to a physical sliver whose
        # facet turns from the source beyond the schedule's normal aim; split
        # both like any other schedule violation.
        bad |= _facet_normal_violations(
            domain,
            patches,
            chart_parameters,
            physical[triangles],
            schedule.normal_deviation_aim,
        )
        local_bad = bool(np.any(bad))

        # Bisect along the longest-edge propagation path of every bad cell
        # inside its chart: the split edge is the longest edge of each cell of
        # the chart sharing it, so no cell is bisected across a shorter interior
        # edge. A trim edge bounds the chart, so a path reaching it as the
        # longest edge splits it; its canonical curve identity gives every chart
        # using that curve interval the same exact midpoint parameter.
        split_edges: set[_LocalEdgeIdentity] = set()
        if local_bad:
            identities: list[tuple[_LocalEdgeIdentity, ...]] = []
            cell_charts: list[int] = []
            for chart_row, chart in enumerate(charts):
                identities.extend(
                    tuple(
                        _local_edge_identity(chart, first, last)
                        for first, last in (
                            (triangle[0], triangle[1]),
                            (triangle[1], triangle[2]),
                            (triangle[2], triangle[0]),
                        )
                    )
                    for triangle in chart.triangles
                )
                cell_charts.extend((chart_row,) * len(chart.triangles))
            cells_by_edge: dict[_LocalEdgeIdentity, list[int]] = {}
            for cell, cell_identities in enumerate(identities):
                for identity in cell_identities:
                    cells_by_edge.setdefault(identity, []).append(cell)

            def longest_identity(cell: int, /) -> _LocalEdgeIdentity:
                index = max(
                    range(3),
                    key=lambda index: (
                        edge_lengths[cell, index],
                        identities[cell][index],
                    ),
                )
                return identities[cell][index]

            for cell in np.flatnonzero(bad).tolist():
                while True:
                    identity = longest_identity(cell)
                    onward = next(
                        (
                            other
                            for other in cells_by_edge[identity]
                            if other != cell
                            and cell_charts[other] == cell_charts[cell]
                            and longest_identity(other) != identity
                        ),
                        None,
                    )
                    if onward is None:
                        split_edges.add(identity)
                        break
                    cell = onward
        if seam_rounds is None:
            pending = local_bad
        else:
            shared_splits, pending = seam_rounds(
                frozenset(
                    (1, identity[1], identity[2], identity[3])
                    for identity in split_edges
                    if identity[0] == 1
                ),
                local_bad,
            )
            split_edges.update(shared_splits)
        if not pending:
            break

        changed = False
        for chart, mapping in zip(charts, local_rows, strict=True):
            old_triangles = tuple(chart.triangles)
            requested: dict[tuple[int, int], int] = {}
            curved_rows: set[int] = set()
            all_edges = {
                _edge_key(edge[0], edge[1])
                for triangle in old_triangles
                for edge in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                )
            }
            for edge in all_edges:
                if _local_edge_identity(chart, *edge) not in split_edges:
                    continue
                first, last = edge
                first_uv, last_uv = chart.points[first], chart.points[last]
                midpoint_uv = (
                    (first_uv[0] + last_uv[0]) / 2,
                    (first_uv[1] + last_uv[1]) / 2,
                )
                boundary = chart.boundary_edges.get(edge)
                if boundary is not None:
                    # A curved trim is split at its p-curve point of the midpoint
                    # parameter; the chord midpoint lies off the source curve.
                    patch_row, loop_row, use_row = boundary[1]
                    use = domain.patches[patch_row].loops[loop_row][use_row]
                    if isinstance(use, domain_owner.PatchCurveUse) and not isinstance(
                        use.pcurve, (ExactLineCurve, LineCurve)
                    ):
                        midpoint_uv = _use_chart_point(
                            use, (boundary[2] + boundary[3]) / 2
                        )
                        curved_rows.add(len(chart.points))
                represented_midpoint = (
                    float(midpoint_uv[0]),
                    float(midpoint_uv[1]),
                )
                if midpoint_uv in (first_uv, last_uv) or represented_midpoint in (
                    (float(first_uv[0]), float(first_uv[1])),
                    (float(last_uv[0]), float(last_uv[1])),
                ):
                    raise _failure(
                        "surface refinement reached binary64 chart resolution before meeting the physical schedule",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        entity_ids=(int(chart.patch),),
                    )
                local_row = len(chart.points)
                chart.points.append(midpoint_uv)
                if boundary is not None:
                    curve, occurrence, source_first, source_last = boundary
                    first_parameter, last_parameter = chart.edge_parameters(
                        (first, last), curve, (source_first, source_last)
                    )
                    if first_parameter is None or last_parameter is None:
                        raise _failure(
                            "source curve refinement lost exact endpoint parameters",
                            category=MeshingFailureCategory.INVALID_SOURCE,
                            entity_ids=(int(curve),),
                        )
                    if (first_parameter, last_parameter) == (source_first, source_last):
                        source_first_row, source_last_row = first, last
                    elif (first_parameter, last_parameter) == (source_last, source_first):
                        source_first_row, source_last_row = last, first
                    else:
                        raise _failure(
                            "source curve refinement parameters disagree with exact edge ancestry",
                            category=MeshingFailureCategory.INVALID_SOURCE,
                            entity_ids=(int(curve),),
                        )
                    midpoint_parameter = (source_first + source_last) / 2
                    if midpoint_parameter in (source_first, source_last) or float(
                        midpoint_parameter
                    ) in (float(source_first), float(source_last)):
                        raise _failure(
                            "source curve refinement reached parameter resolution before meeting its physical schedule",
                            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                            entity_ids=(int(curve),),
                        )
                    chart.curve_parameters[(local_row, curve)] = midpoint_parameter
                    chart.boundary_edges.pop(edge)
                    chart.boundary_edges[_edge_key(source_first_row, local_row)] = (
                        curve,
                        occurrence,
                        source_first,
                        midpoint_parameter,
                    )
                    chart.boundary_edges[_edge_key(local_row, source_last_row)] = (
                        curve,
                        occurrence,
                        midpoint_parameter,
                        source_last,
                    )
                mapping[local_row] = _add_global_vertex(
                    domain,
                    chart,
                    local_row,
                    global_rows,
                    coordinates,
                    source_dimensions,
                    source_indices,
                    source_parameters,
                )
                requested[edge] = local_row
                changed = True
            if not requested:
                continue
            children: list[tuple[int, int, int]] = []
            for triangle in old_triangles:
                a, b, c = triangle
                ab_key, bc_key, ca_key = (
                    _edge_key(a, b),
                    _edge_key(b, c),
                    _edge_key(c, a),
                )
                ab, bc, ca = (
                    requested.get(ab_key),
                    requested.get(bc_key),
                    requested.get(ca_key),
                )
                split_count = sum(value is not None for value in (ab, bc, ca))
                if split_count == 0:
                    children.append(triangle)
                elif split_count == 1:
                    if ab is not None:
                        children.extend(((a, ab, c), (ab, b, c)))
                    elif bc is not None:
                        children.extend(((b, bc, a), (bc, c, a)))
                    else:
                        assert ca is not None
                        children.extend(((c, ca, b), (ca, a, b)))
                elif split_count == 2:
                    if ca is None:
                        center, left, right, left_mid, right_mid = b, a, c, ab, bc
                    elif ab is None:
                        center, left, right, left_mid, right_mid = c, b, a, bc, ca
                    else:
                        center, left, right, left_mid, right_mid = a, c, b, ca, ab
                    assert left_mid is not None and right_mid is not None
                    children.extend(
                        (
                            (center, right_mid, left_mid),
                            (left, left_mid, right),
                            (left_mid, right_mid, right),
                        )
                    )
                else:
                    assert ab is not None and bc is not None and ca is not None
                    children.extend(
                        (
                            (a, ab, ca),
                            (ab, b, bc),
                            (ca, bc, c),
                            (ab, bc, ca),
                        )
                    )
            orientation = domain.patches[chart.patch].orientation
            for child in children:
                if curved_rows.isdisjoint(child):
                    continue
                area = _root_edge_cross(*(chart.points[row] for row in child))
                if area == 0 or (area > 0) != (orientation > 0):
                    raise _failure(
                        "curved trim refinement folded a source chart cell",
                        category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        entity_ids=(int(chart.patch),),
                    )
            chart.triangles = children
        if local_bad and not changed:
            raise _failure(
                "surface refinement is stalled while physical schedule violations remain",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=tuple(
                    int(value) for value in np.unique(patches[bad]).tolist()
                ),
            )
        if any(identity[0] == 1 for identity in split_edges):
            curve_bounds_by_chart = tuple(
                _chart_curve_bounds(chart, chords) for chart in charts
            )
            trim_edges = tuple(frozenset(chart.boundary_edges) for chart in charts)

    triangles_by_chart = tuple(
        _chart_triangles(chart, mapping)
        for chart, mapping in zip(charts, local_rows, strict=True)
    )
    triangles = np.concatenate(triangles_by_chart, axis=0)
    triangle_patches = np.concatenate(
        tuple(
            np.full(rows.shape[0], chart.patch, dtype=np.int64)
            for chart, rows in zip(charts, triangles_by_chart, strict=True)
        )
    )
    triangle_parameters = np.concatenate(
        tuple(
            np.asarray(
                [[chart.points[row] for row in cell] for cell in chart.triangles],
                dtype=np.float64,
            )
            for chart in charts
        ),
        axis=0,
    )
    deviations = np.empty((triangles.shape[0],), dtype=np.float64)
    normal_bounds = np.zeros_like(deviations)
    offset = 0
    for chart in charts:
        count = len(chart.triangles)
        parameters = triangle_parameters[offset : offset + count]
        deviations[offset : offset + count], normal_values = _cached_cell_bounds(
            preparation,
            chart,
            parameters,
            cell_bounds,
            specification.maximum_normal_deviation is not None,
        )
        if specification.maximum_normal_deviation is not None:
            normal_bounds[offset : offset + count] = normal_values
        deviations[offset : offset + count] = np.maximum(
            deviations[offset : offset + count], chart.pole_deviation_bound
        )
        offset += count
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    quality = _quality_angle_degrees(coordinates_array, triangles)
    curve_edge_bounds: dict[tuple[int, int, int], float] = {}
    cell_offset = 0
    for chart, local_to_global, bounds, edges in zip(
        charts, local_rows, curve_bounds_by_chart, trim_edges, strict=True
    ):
        if frozenset(chart.boundary_edges) != edges:
            raise _failure(
                "physical surface refinement changed a fixed source trim edge",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=(int(chart.patch),),
            )
        for edge, (curve, _occurrence, _first, _last) in chart.boundary_edges.items():
            bound = bounds[edge]
            first_global, last_global = local_to_global[edge[0]], local_to_global[edge[1]]
            vertex_pair = tuple(sorted((first_global, last_global)))
            key = (curve, vertex_pair[0], vertex_pair[1])
            curve_edge_bounds[key] = max(curve_edge_bounds.get(key, 0.0), bound)
            for local_cell, triangle in enumerate(chart.triangles):
                if edge in (
                    _edge_key(triangle[0], triangle[1]),
                    _edge_key(triangle[1], triangle[2]),
                    _edge_key(triangle[2], triangle[0]),
                ):
                    deviations[cell_offset + local_cell] = max(
                        float(deviations[cell_offset + local_cell]), bound
                    )
        cell_offset += len(chart.triangles)
    return _RefinedSurface(
        tuple(charts),
        tuple(local_rows),
        coordinates_array,
        np.asarray(source_dimensions, dtype=np.int8),
        np.asarray(source_indices, dtype=np.int64),
        np.asarray(source_parameters, dtype=np.float64),
        triangles,
        triangle_patches,
        triangle_parameters,
        deviations,
        normal_bounds,
        curve_edge_bounds,
        float(np.min(quality, initial=math.inf)),
        fidelity_tolerance,
    )


def _curve_chains(
    domain: domain_owner.MeshingDomain,
    refined: _RefinedSurface,
    curve_ids: tuple[int, ...],
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, np.ndarray], ...],
    tuple[tuple[int, Fraction, Fraction], ...],
]:
    parameters: dict[int, set[Fraction]] = {
        curve: {Fraction(0), Fraction(1)} for curve in curve_ids
    }
    vertices: dict[int, dict[Fraction, int]] = {curve: {} for curve in curve_ids}
    for chart, local_rows in zip(refined.charts, refined.local_rows, strict=True):
        for (local, curve), parameter in chart.curve_parameters.items():
            if curve not in parameters:
                raise _failure(
                    "chart retains a source curve outside the selected trim network",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(curve),),
                )
            global_row = local_rows[local]
            previous = vertices[curve].get(parameter)
            if previous is not None and previous != global_row:
                raise _failure(
                    "selected source patches do not share one canonical source-curve vertex chain",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(curve),),
                )
            vertices[curve][parameter] = global_row
            parameters[curve].add(parameter)
            if (local, curve) in chart.turn_rows:
                # The start corner of a self-closing trim also ends its turn.
                vertices[curve][Fraction(1)] = global_row
    edges: list[tuple[int, int]] = []
    owners: list[int] = []
    curve_parameters: list[tuple[int, np.ndarray]] = []
    intervals: list[tuple[int, Fraction, Fraction]] = []
    for curve in curve_ids:
        ordered = tuple(sorted(parameters[curve]))
        chain = vertices[curve]
        if (
            ordered[0] != 0
            or ordered[-1] != 1
            or any(parameter not in chain for parameter in ordered)
        ):
            raise _failure(
                "native construction does not cover the complete source-curve parameter range",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        rows = tuple(chain[parameter] for parameter in ordered)

        source_curve = domain.curves[curve]
        if (
            refined.vertex_source_dimensions[rows[0]] != 0
            or refined.vertex_source_indices[rows[0]] != source_curve.start
            or refined.vertex_source_dimensions[rows[-1]] != 0
            or refined.vertex_source_indices[rows[-1]] != source_curve.end
        ):
            raise _failure(
                "source-curve chain endpoints lost their authored corner ancestry",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        if source_curve.start == source_curve.end and rows[0] != rows[-1]:
            raise _failure(
                "closed source curve does not close on its authored corner vertex",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        if any(first == last for first, last in zip(rows, rows[1:])):
            raise _failure(
                "native construction collapsed a positive source-curve interval",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            )
        curve_parameters.append(
            (curve, np.asarray([float(value) for value in ordered], dtype=np.float64))
        )
        for first_parameter, last_parameter, first_vertex, last_vertex in zip(
            ordered[:-1], ordered[1:], rows[:-1], rows[1:], strict=True
        ):
            edges.append((first_vertex, last_vertex))
            owners.append(curve)
            intervals.append((curve, first_parameter, last_parameter))
    return (
        np.asarray(edges, dtype=np.int64).reshape((-1, 2)),
        np.asarray(owners, dtype=np.int64),
        tuple(curve_parameters),
        tuple(intervals),
    )


def _curve_edge_samples(
    domain: domain_owner.MeshingDomain,
    first: np.ndarray,
    last: np.ndarray,
    curve: int,
    lower: Fraction,
    upper: Fraction,
    sample_count: int,
    /,
) -> float:
    chord = np.asarray(last, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    maximum = 0.0
    for index in range(1, sample_count + 1):
        fraction = Fraction(index, sample_count + 1)
        parameter = Fraction.from_float(float(lower + fraction * (upper - lower)))
        point = _source_curve_point(domain, curve, parameter)
        along = float(fraction)
        maximum = max(
            maximum,
            float(np.linalg.norm(point - (np.asarray(first) + along * chord))),
        )
    return maximum


def _triangle_sampled_deviations(
    domain: domain_owner.MeshingDomain,
    refined: _RefinedSurface,
    /,
) -> np.ndarray:
    weights = np.asarray(
        (
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
        dtype=np.float64,
    )
    parameters = np.einsum("qv,cvd->cqd", weights, refined.triangle_parameters)
    patches = np.repeat(refined.triangle_patches, weights.shape[0])
    source_points = domain.evaluate(patches, parameters.reshape((-1, 2))).reshape(
        (-1, weights.shape[0], 3)
    )
    triangle_coordinates = refined.coordinates[refined.triangles]
    chord_points = np.einsum("qv,cvd->cqd", weights, triangle_coordinates)
    return np.max(np.linalg.norm(source_points - chord_points, axis=2), axis=1)


def _surface_construction(
    domain: domain_owner.MeshingDomain,
    refined: _RefinedSurface,
    roots: _SurfaceRootCells,
    curve_ids: tuple[int, ...],
    schedule: NativeSurfaceSchedule,
    /,
) -> SurfaceConstruction:
    exact_triangle_parameters = tuple(
        (
            chart.points[triangle[0]],
            chart.points[triangle[1]],
            chart.points[triangle[2]],
        )
        for chart in refined.charts
        for triangle in chart.triangles
    )
    triangle_source_cells, triangle_source_pieces = source_cell_ancestry(
        roots,
        refined.triangle_patches,
        refined.triangle_parameters,
        exact_triangle_parameters,
    )
    curve_edges, curve_edge_curves, curve_parameters, intervals = _curve_chains(
        domain, refined, curve_ids
    )
    curve_edge_deviations = np.empty((curve_edges.shape[0],), dtype=np.float64)
    curve_edge_deviation_bounds = np.empty_like(curve_edge_deviations)
    for row, (curve, lower, upper) in enumerate(intervals):
        first_vertex, last_vertex = curve_edges[row]
        key = (
            curve,
            min(int(first_vertex), int(last_vertex)),
            max(int(first_vertex), int(last_vertex)),
        )
        try:
            bound = refined.curve_edge_bounds[key]
        except KeyError as error:
            raise _failure(
                "a source-curve interval has no adjacent certified source chart cell",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=(int(curve),),
            ) from error
        witnessed = _curve_edge_samples(
            domain,
            refined.coordinates[int(first_vertex)],
            refined.coordinates[int(last_vertex)],
            curve,
            lower,
            upper,
            schedule.curve_schedule.fidelity_samples,
        )
        allowance = 128.0 * np.finfo(np.float64).eps * max(domain.scale, 1.0)
        if witnessed > bound + allowance:
            raise _failure(
                "source-curve samples exceed the adjacent interval enclosure",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                entity_ids=(int(curve),),
            )
        curve_edge_deviations[row] = witnessed
        curve_edge_deviation_bounds[row] = bound

    boundary_occurrences = np.full((refined.triangles.shape[0], 3, 3), -1, dtype=np.int32)
    restriction_vertices: list[tuple[int, int]] = []
    restriction_occurrences: list[tuple[int, int, int]] = []
    restriction_endpoint_parameters: list[tuple[float, float]] = []
    restriction_chart_parameters: list[
        tuple[tuple[float, float], tuple[float, float]]
    ] = []
    cell_offset = 0
    for chart, local_rows in zip(refined.charts, refined.local_rows, strict=True):
        for local_row, triangle in enumerate(chart.triangles):
            for opposite in range(3):
                first, last = triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3]
                source = chart.boundary_edges.get(_edge_key(first, last))
                if source is None:
                    continue
                _, occurrence, first_parameter, last_parameter = source
                boundary_occurrences[cell_offset + local_row, opposite] = occurrence
        for (first, last), (curve, occurrence, source_first, source_last) in sorted(
            chart.boundary_edges.items()
        ):
            first_parameter, last_parameter = chart.edge_parameters(
                (first, last), curve, (source_first, source_last)
            )
            if (
                first_parameter is None
                or last_parameter is None
                or (
                    (first_parameter, last_parameter) != (source_first, source_last)
                    and (first_parameter, last_parameter) != (source_last, source_first)
                )
            ):
                raise _failure(
                    "source trim edge endpoints lost their exact curve parameters",
                    category=MeshingFailureCategory.INVALID_SOURCE,
                    entity_ids=(int(curve),),
                )
            restriction_vertices.append((local_rows[first], local_rows[last]))
            restriction_occurrences.append(occurrence)
            restriction_endpoint_parameters.append(
                (float(first_parameter), float(last_parameter))
            )
            restriction_chart_parameters.append(
                (
                    (float(chart.points[first][0]), float(chart.points[first][1])),
                    (float(chart.points[last][0]), float(chart.points[last][1])),
                )
            )
        cell_offset += len(chart.triangles)
    if cell_offset != refined.triangles.shape[0]:
        raise RuntimeError(
            "Surface chart cell rows do not align with construction triangles."
        )
    return SurfaceConstruction(
        refined.coordinates,
        refined.vertex_source_dimensions,
        refined.vertex_source_indices,
        refined.vertex_parameters,
        np.asarray(restriction_vertices, dtype=np.int64).reshape((-1, 2)),
        np.asarray(restriction_occurrences, dtype=np.int32).reshape((-1, 3)),
        np.asarray(restriction_endpoint_parameters, dtype=np.float64).reshape((-1, 2)),
        np.asarray(restriction_chart_parameters, dtype=np.float64).reshape((-1, 2, 2)),
        refined.triangles,
        refined.triangle_patches,
        np.mean(refined.triangle_parameters, axis=1),
        refined.triangle_parameters,
        triangle_source_cells,
        triangle_source_pieces,
        _triangle_sampled_deviations(domain, refined),
        refined.triangle_deviation_bounds,
        refined.triangle_normal_bounds,
        curve_edges,
        curve_edge_curves,
        curve_edge_deviations,
        curve_edge_deviation_bounds,
        np.zeros((refined.triangles.shape[0],), dtype=np.bool_),
        boundary_occurrences,
        tuple(
            (chart.patch, np.asarray(chart.triangles, dtype=np.int64))
            for chart in refined.charts
        ),
        curve_parameters,
        refined.minimum_angle,
    )


def _mesh_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        identifiers,
    )


def _organization(
    domain: domain_owner.MeshingDomain,
    specification: SurfaceMeshingSpec,
    scope_ids: tuple[np.ndarray, ...],
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    associations: tuple[association_owner.GeometryAssociation, ...],
    /,
) -> NativeMeshOrganization:
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    patch_rows = np.asarray(construction.triangle_patches, dtype=np.int64)
    patch_objects = []
    for patch in sorted(set(patch_rows.tolist())):
        rows = np.flatnonzero(patch_rows == patch)
        source_regions: list[tuple[str, tuple[str, ...]] | None] = [None, None]
        for region_row, region in enumerate(domain.regions):
            for source_patch, orientation in region.boundary:
                if source_patch != patch:
                    continue
                side = 1 if orientation > 0 else 0
                if source_regions[side] is not None:
                    raise _failure(
                        "source patch has ambiguous same-side region adjacency",
                        category=MeshingFailureCategory.INVALID_SOURCE,
                        entity_ids=(int(patch),),
                    )
                source_regions[side] = domain.source_entity_key_for_scope(
                    3, _scope_id_for_row(scope_ids, 3, region_row)
                )
        pair = (
            (
                None if source_regions[0] is None else source_regions[0][0],
                None if source_regions[1] is None else source_regions[1][0],
            )
            if any(value is not None for value in source_regions)
            else None
        )
        pair_paths = (
            (
                None if source_regions[0] is None else source_regions[0][1],
                None if source_regions[1] is None else source_regions[1][1],
            )
            if pair is not None
            else None
        )
        patch_objects.append(
            MeshPatch(
                _source_key_for_scope(
                    domain,
                    2,
                    _scope_id_for_row(scope_ids, 2, patch),
                )[0],
                _mesh_scope(mesh, 2, cell_ids[rows]),
                connected=True,
                source_adjacent_region_ids=pair,
                source_adjacent_region_occurrence_paths=pair_paths,
            )
        )
    labels: list[MeshLabel] = []
    selected_patch_set = set(patch_rows.tolist())
    for region in domain.regions:
        patch_selection = tuple(
            patch for patch, _ in region.boundary if patch in selected_patch_set
        )
        if patch_selection:
            rows = np.flatnonzero(
                np.isin(patch_rows, np.asarray(patch_selection, dtype=np.int64))
            )
            labels.append(
                MeshLabel(
                    f"region:{region.name}",
                    _mesh_scope(mesh, 2, cell_ids[rows]),
                )
            )
    vertex_association = next(
        value
        for value in associations
        if value.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    edge_association = next(
        value
        for value in associations
        if value.target_entity_set_id == mesh.entity_set(1).entity_set_id
    )
    for feature in specification.protected_features:
        source_dimension = 1 if feature.feature_kind.value == "curve" else 0
        association = edge_association if source_dimension == 1 else vertex_association
        source_keys = {
            _source_key_for_scope(domain, source_dimension, int(identifier))[0]
            for identifier in np.asarray(
                feature.scope.entity_ids, dtype=np.int64
            ).tolist()
        }
        rows = np.asarray(
            [
                row
                for row, source_key in enumerate(association.source_entity_ids)
                if source_key in source_keys
            ],
            dtype=np.int64,
        )
        if rows.size:
            labels.append(
                MeshLabel(
                    f"feature:{feature.feature_id}",
                    _mesh_scope(
                        mesh,
                        source_dimension,
                        np.asarray(association.target_global_ids, dtype=np.int64)[rows],
                    ),
                )
            )
    return NativeMeshOrganization(
        patches=tuple(patch_objects),
        labels=tuple(labels),
        associations=associations,
    )


def _compliance(
    specification: SurfaceMeshingSpec,
    schedule: NativeSurfaceSchedule,
    refined: _RefinedSurface,
    sizes: np.ndarray,
    selected_patches: tuple[int, ...],
    /,
) -> MeshingComplianceReport:
    triangles = refined.triangles
    edges = np.stack(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
        axis=1,
    )
    lengths = np.linalg.norm(
        refined.coordinates[edges[:, :, 0]] - refined.coordinates[edges[:, :, 1]],
        axis=-1,
    )
    maximum_edge = float(np.max(lengths, initial=0.0))
    maximum_deviation = float(np.max(refined.triangle_deviation_bounds, initial=0.0))
    maximum_normal = float(np.max(refined.triangle_normal_bounds, initial=0.0))
    patch_size = {
        patch: float(size) for patch, size in zip(selected_patches, sizes, strict=True)
    }
    cell_targets = np.asarray(
        [
            schedule.spacing_fraction * patch_size[int(patch)]
            for patch in refined.triangle_patches
        ],
        dtype=np.float64,
    )
    target_edge = float(np.max(cell_targets, initial=0.0))
    requested = [
        ("maximum_edge_length", target_edge),
        ("minimum_angle_degrees", schedule.quality_angle_degrees),
    ]
    if refined.fidelity_tolerance is not None:
        requested.append(("maximum_deviation", refined.fidelity_tolerance))
    achieved = [
        ("maximum_edge_length", maximum_edge),
        ("maximum_deviation", maximum_deviation),
        ("minimum_angle_degrees", refined.minimum_angle),
    ]
    issues = []
    if np.any(
        np.max(lengths, axis=1) > cell_targets * (1.0 + 64.0 * np.finfo(np.float64).eps)
    ):
        issues.append("surface edge spacing exceeds the native schedule")
    if (
        refined.fidelity_tolerance is not None
        and maximum_deviation > refined.fidelity_tolerance
    ):
        issues.append("source interpolation exceeds the requested physical fidelity")
    if specification.maximum_normal_deviation is not None:
        requested.append(
            ("maximum_normal_deviation", specification.maximum_normal_deviation)
        )
        achieved.append(("maximum_normal_deviation", maximum_normal))
        if maximum_normal > specification.maximum_normal_deviation:
            issues.append("source normal turn exceeds the requested physical bound")
    if specification.quality_target is not None:
        target_angle = math.degrees(specification.quality_target.minimum_angle)
        requested.append(("quality_target_degrees", target_angle))
        achieved.append(("quality_target_degrees", refined.minimum_angle))
        if specification.quality_target.hard and refined.minimum_angle < target_angle:
            issues.append("physical triangle angle misses the hard quality target")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _fraction_point_bounds(
    lower: tuple[Fraction, ...], upper: tuple[Fraction, ...], /
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lower_values = []
    upper_values = []
    for low, high in zip(lower, upper, strict=True):
        low_float = float(low)
        high_float = float(high)
        if Fraction.from_float(low_float) > low:
            low_float = float(np.nextafter(low_float, -np.inf))
        if Fraction.from_float(high_float) < high:
            high_float = float(np.nextafter(high_float, np.inf))
        lower_values.append(low_float)
        upper_values.append(high_float)
    return tuple(lower_values), tuple(upper_values)


def _domain_curve_source_query(
    domain: domain_owner.MeshingDomain,
    selected_entity_ids: np.ndarray | None = None,
    /,
) -> ParametricCurveBoundarySource:
    """Bind every selected source edge to each authored trimmed chart use."""
    available = np.asarray(domain.scope_entity_ids(1), dtype=np.int64)
    selected = (
        available
        if selected_entity_ids is None
        else np.asarray(selected_entity_ids, dtype=np.int64)
    )
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("MeshingDomain curve query needs unique source edge scope IDs.")
    if np.any(~np.isin(selected, available)):
        raise ValueError("MeshingDomain curve query scope names an absent source edge.")
    selected_set = set(int(value) for value in selected.tolist())
    curves = []
    ranges = []
    curve_entity_ids = []
    occurrences = []
    chart_ids = []
    endpoint_keys: list[
        tuple[
            tuple[str, tuple[str, ...]],
            tuple[str, tuple[str, ...]],
        ]
    ] = []
    endpoint_bounds = []
    for numeric_id in available.tolist():
        identifier = int(numeric_id)
        if identifier not in selected_set:
            continue
        entity_id, occurrence = domain.source_entity_key_for_scope(1, identifier)
        chart_uses = domain.source_curve_chart_uses(identifier)
        for chart_id, patch_row, _loop_row, _use_row, use in chart_uses:
            curve = SurfaceMappedCurve(domain.patches[patch_row].surface, use.pcurve)
            curves.append(curve)
            ranges.append((float(use.first), float(use.last)))
            curve_entity_ids.append(entity_id)
            occurrences.append(occurrence)
            chart_ids.append(chart_id)
            source_curve = domain.curves[use.curve]
            # The p-curve chart follows edge parameter order; use.sense only orients the loop.
            corners = (source_curve.start, source_curve.end)
            first_corner, second_corner = corners
            first_key = (
                domain.entity_id(0, first_corner),
                domain.source_occurrences[0][first_corner],
            )
            second_key = (
                domain.entity_id(0, second_corner),
                domain.source_occurrences[0][second_corner],
            )
            first_lower, first_upper = domain.source_corner_point_bounds(first_corner)
            second_lower, second_upper = domain.source_corner_point_bounds(second_corner)
            endpoint_keys.append((first_key, second_key))
            endpoint_bounds.append(
                (
                    _fraction_point_bounds(first_lower, first_upper),
                    _fraction_point_bounds(second_lower, second_upper),
                )
            )
    if not curves:
        raise ValueError("Selected MeshingDomain curve scope has no authored trim uses.")
    return ParametricCurveBoundarySource(
        tuple(curves),
        tuple(ranges),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        curve_entity_ids=tuple(curve_entity_ids),
        occurrence_paths=tuple(occurrences),
        chart_ids=tuple(chart_ids),
        endpoint_source_keys=tuple(endpoint_keys),
        endpoint_point_bounds=tuple(endpoint_bounds),
    )


def _surface_curve_chart_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    domain: domain_owner.MeshingDomain,
    query: ParametricCurveBoundarySource,
    construction: SurfaceConstruction,
    /,
) -> SourceBoundaryChartWitness:
    """Bind each source trim-use chart to actual constrained output edges."""
    curve_scope_ids = {
        domain.source_entity_key_for_scope(1, int(identifier)): int(identifier)
        for identifier in domain.scope_entity_ids(1).tolist()
    }
    chart_records: dict[str, tuple[int, domain_owner.PatchCurveUse]] = {}
    for entity_id, occurrence in set(
        zip(query.curve_entity_ids, query.occurrence_paths, strict=True)
    ):
        numeric_id = curve_scope_ids[(entity_id, occurrence)]
        for (
            chart_id,
            patch_row,
            _loop_row,
            _use_row,
            use,
        ) in domain.source_curve_chart_uses(numeric_id):
            chart_records[chart_id] = (patch_row, use)
    if set(chart_records) != set(query.chart_ids):
        raise ValueError(
            "Surface feature query and authored trim-use chart bank disagree."
        )

    chain_edges = np.asarray(construction.curve_edges, dtype=np.int64)
    chain_curves = np.asarray(construction.curve_edge_curves, dtype=np.int64)
    intervals = construction.curve_edge_parameter_intervals()
    if len(intervals) != chain_edges.shape[0] or chain_curves.shape != (
        chain_edges.shape[0],
    ):
        raise ValueError("Surface curve chain ancestry arrays are misaligned.")
    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_edges = _surface_edges(mesh)
    target_edge_by_pair = {
        _edge_key(int(pair[0]), int(pair[1])): int(identifier)
        for pair, identifier in zip(
            target_edges.tolist(), target_edge_ids.tolist(), strict=True
        )
    }
    topology = reference_cell_topology("triangle")
    reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
    local_edges = topology.entities[1]
    triangles = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    adjacent: dict[int, list[tuple[int, int, tuple[int, int]]]] = {}
    for cell_row, triangle in enumerate(triangles):
        for local_edge, local_vertices in enumerate(local_edges):
            pair = (
                int(triangle[local_vertices[0]]),
                int(triangle[local_vertices[1]]),
            )
            target_edge_id = target_edge_by_pair[_edge_key(pair[0], pair[1])]
            adjacent.setdefault(target_edge_id, []).append(
                (cell_row, local_edge, local_vertices)
            )

    query_chart_rows = {chart_id: row for row, chart_id in enumerate(query.chart_ids)}
    witness_cell_ids = []
    witness_edge_ids = []
    witness_local_edges = []
    target_references = []
    target_reference_rational_rows = []
    source_chart_ids = []
    source_entity_ids = []
    occurrence_paths = []
    source_parameters = []
    source_parameter_rational_rows = []
    orientations = []
    for chart_id in query.chart_ids:
        patch_row, use = chart_records[chart_id]
        curve_row = use.curve
        query_row = query_chart_rows[chart_id]
        source_entity_id = query.curve_entity_ids[query_row]
        occurrence = query.occurrence_paths[query_row]
        lower_parameter = Fraction.from_float(float(use.first))
        upper_parameter = Fraction.from_float(float(use.last))
        native_start, native_end = lower_parameter, upper_parameter
        chain_rows = np.flatnonzero(chain_curves == curve_row)
        if chain_rows.size == 0:
            raise ValueError(
                "Surface construction omitted an authored feature curve chain."
            )
        for chain_row in chain_rows.tolist():
            first, last = intervals[chain_row]
            edge_curve = int(construction.curve_edge_curves[chain_row])
            if edge_curve != curve_row:
                raise ValueError(
                    "Surface curve parameter ancestry changed its source edge row."
                )
            edge_vertices = tuple(int(value) for value in chain_edges[chain_row])
            edge_id = target_edge_by_pair[_edge_key(edge_vertices[0], edge_vertices[1])]
            neighbors = [
                value
                for value in adjacent.get(edge_id, ())
                if int(construction.triangle_patches[value[0]]) == patch_row
            ]
            if not neighbors:
                raise ValueError(
                    "Surface source curve chart has no target facet on its source patch."
                )
            parent_row, local_edge, local_vertices = min(
                neighbors, key=lambda value: int(cell_ids[value[0]])
            )
            ordered_vertices = tuple(
                int(triangles[parent_row, index]) for index in local_vertices
            )
            if ordered_vertices == edge_vertices:
                source_q = (first, last)
            elif ordered_vertices == edge_vertices[::-1]:
                source_q = (last, first)
            else:
                raise ValueError(
                    "Surface local-facet vertices disagree with curve-chain ancestry."
                )
            chart_parameters = tuple(
                native_start + (native_end - native_start) * value for value in source_q
            )
            witness_cell_ids.append(int(cell_ids[parent_row]))
            witness_edge_ids.append(edge_id)
            witness_local_edges.append(local_edge)
            target_reference_row = reference_vertices[
                np.asarray(local_vertices, dtype=np.int64)
            ]
            target_references.append(target_reference_row)
            target_reference_rational_rows.append(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in target_reference_row
                )
            )
            source_chart_ids.append(chart_id)
            source_entity_ids.append(source_entity_id)
            occurrence_paths.append(occurrence)
            source_parameters.append(tuple(float(value) for value in chart_parameters))
            source_parameter_rational_rows.append(
                tuple((value,) for value in chart_parameters)
            )
            orientations.append(1 if chart_parameters[1] > chart_parameters[0] else -1)
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray(witness_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(witness_edge_ids, dtype=np.int64),
        local_entity_indices=np.asarray(witness_local_edges, dtype=np.int32),
        target_reference_vertices=np.asarray(target_references, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            target_reference_rational_rows
        ),
        source_chart_ids=tuple(source_chart_ids),
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameters, dtype=np.float64)[:, :, None],
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _surface_point_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    domain: domain_owner.MeshingDomain,
    query: domain_boundary_owner.MeshingDomainBoundarySource,
    requested_point_ids: np.ndarray,
    refined: _RefinedSurface,
    /,
) -> SourceBoundaryPointWitness:
    identifiers = np.asarray(requested_point_ids, dtype=np.int64)
    if (
        identifiers.ndim != 1
        or identifiers.size == 0
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError("Surface point witness needs unique authored source corner IDs.")
    target_rows = []
    source_entity_ids = []
    occurrences = []
    for identifier in identifiers.tolist():
        entity_id, occurrence = domain.source_entity_key_for_scope(0, int(identifier))
        source_row = domain.resolve_entity_id(0, entity_id, occurrence_path=occurrence)
        matches = np.flatnonzero(
            (np.asarray(refined.vertex_source_dimensions) == 0)
            & (np.asarray(refined.vertex_source_indices) == source_row)
        )
        if matches.size != 1:
            raise ValueError(
                "Every protected source corner must map to one target vertex."
            )
        target_rows.append(int(matches[0]))
        source_entity_ids.append(entity_id)
        occurrences.append(occurrence)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(target_rows, dtype=np.int64)
    ]
    if np.unique(target_ids).size != target_ids.size:
        raise ValueError(
            "Distinct source corners cannot share one target vertex identity."
        )
    return SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_scope_id,
        vertex_global_ids=target_ids,
        source_entity_ids=tuple(source_entity_ids),
        occurrence_paths=tuple(occurrences),
    )


@final
class _PreparedNativeSurface(StrictModule, NonTrainableState):
    domain: domain_owner.MeshingDomain
    specification: SurfaceMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    schedule: NativeSurfaceSchedule
    selected_patches: tuple[int, ...] = eqx.field(static=True)
    scope_ids: tuple[np.ndarray, ...]
    size_values: np.ndarray
    roots: _SurfaceRootCells
    boundary_source: domain_boundary_owner.MeshingDomainBoundarySource
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: domain_owner.MeshingDomain,
        specification: SurfaceMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        schedule: NativeSurfaceSchedule,
        selected_patches: tuple[int, ...],
        scope_ids: tuple[np.ndarray, ...],
        size_values: np.ndarray,
        roots: _SurfaceRootCells,
        boundary_source: domain_boundary_owner.MeshingDomainBoundarySource,
        /,
    ) -> None:
        self.domain = domain
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.schedule = schedule
        self.selected_patches = selected_patches
        self.scope_ids = scope_ids
        self.size_values = size_values
        self.roots = roots
        self.boundary_source = boundary_source
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-parametric-surface",
                "domain": domain.domain_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "selected_patches": selected_patches,
                "sizes": array_tree_fingerprint(size_values),
            }
        )

    def execute(
        self, /, *, preparation: domain_owner.SourcePreparation | None = None
    ) -> NativeMeshConstruction:
        """Generate and certify the surface mesh.

        ``preparation`` is the caller's `SourcePreparation` of this domain;
        without one the domain is prepared for this execution alone.
        """
        domain = self.domain
        domain.require_current(domain.source_id, domain.source_revision)
        self.boundary_source.require_current()
        preparation = domain_owner.SourcePreparation.of(domain, preparation)
        curve_ids = _trim_curves(domain, self.selected_patches)
        parameters_by_curve = _curve_parameters(
            domain,
            curve_ids,
            self.selected_patches,
            self.size_values,
            self.specification,
            self.schedule,
            preparation,
        )
        pole_charts = _pole_sampling(
            domain,
            self.selected_patches,
            self.size_values,
            self.specification,
            self.schedule,
            parameters_by_curve,
            preparation,
        )
        _integrate_pole_ring_parameters(
            domain,
            self.selected_patches,
            pole_charts,
            parameters_by_curve,
        )
        quality_angle = self.schedule.quality_angle_degrees
        if self.specification.quality_target is not None:
            quality_angle = max(
                quality_angle,
                math.degrees(self.specification.quality_target.minimum_angle),
            )
        root_charts_by_patch = _surface_root_charts(self.roots)
        while True:
            charts = [
                _make_chart_mesh(
                    domain,
                    patch,
                    parameters_by_curve,
                    pole_charts,
                    quality_angle,
                )
                for patch in self.selected_patches
            ]
            for chart in charts:
                _split_chart_at_source_cells(
                    chart,
                    root_charts_by_patch.get(chart.patch, ()),
                )
            expanded = {
                curve: set(values) for curve, values in parameters_by_curve.items()
            }
            for chart in charts:
                for (local, curve), parameter in chart.curve_parameters.items():
                    expanded[curve].add(parameter)
            if all(expanded[curve] == parameters_by_curve[curve] for curve in curve_ids):
                break
            parameters_by_curve = expanded
        refined = _refine_charts(
            domain,
            charts,
            self.size_values,
            self.specification,
            self.schedule,
            preparation,
        )
        if (
            self.specification.quality_target is not None
            and self.specification.quality_target.hard
            and refined.minimum_angle
            < math.degrees(self.specification.quality_target.minimum_angle)
        ):
            raise _failure(
                "physical surface triangles miss the hard minimum-angle target",
                category=MeshingFailureCategory.COMPLIANCE_FAILED,
            )
        construction = _surface_construction(
            domain,
            refined,
            self.roots,
            curve_ids,
            self.schedule,
        )
        # Patches separating two regions make a non-manifold multiregion surface:
        # each is triangulated once, and the region-labeled topology admits its
        # junction edges and certifies every region boundary closed.
        multiregion_surface = domain.multiregion_surface(
            refined.coordinates,
            refined.triangles,
            refined.triangle_patches,
            vertex_global_ids=np.arange(refined.coordinates.shape[0], dtype=np.int64),
            cell_global_ids=np.arange(refined.triangles.shape[0], dtype=np.int64),
        )
        mesh = cell_mesh_owner.CellMesh.from_triangles(
            refined.coordinates,
            refined.triangles,
            numeric_version=domain.source_revision,
            multiregion_surface=multiregion_surface,
        )
        geometry = CellGeometrySpec.affine(mesh)
        association_values = surface_associations(domain, mesh, construction)
        boundary_cover = self.boundary_source.boundary_chart_cover(
            preparation=preparation
        )
        if not boundary_cover.complete:
            raise _failure(
                "source boundary query could not certify complete trimmed-chart coverage",
                category=MeshingFailureCategory.INVALID_SOURCE,
                entity_ids=tuple(int(patch) for patch in self.selected_patches),
            )
        root_parameter_exact = tuple(
            tuple(chart.points[point] for point in triangle)
            for chart in refined.charts
            for triangle in chart.triangles
        )
        surface_source = surface_support_owner.SurfaceSourceCharts(
            domain,
            mesh,
            geometry,
            self.coordinate_contract,
            construction.triangle_parameters,
            construction.triangle_patches,
            association_values,
            self.boundary_source,
            source_root_cell_ids=construction.triangle_source_cells,
            source_piece_ids=construction.triangle_source_pieces,
            root_parameter_exact=root_parameter_exact,
        )
        organization = _organization(
            domain,
            self.specification,
            self.scope_ids,
            mesh,
            construction,
            association_values,
        )
        cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
        cell_tags = tuple(
            domain.entity_id(2, int(patch))
            for patch in construction.triangle_patches.tolist()
        )
        metadata = SurfaceMetadata(
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            coordinate_contract=self.coordinate_contract,
            provenance=("native_parametric_surface", domain.domain_id),
            cell_tags=cell_tags,
        )
        # A manifold `SurfaceModel` boundary exists only for manifold surfaces; a
        # multiregion surface publishes its region-labeled topology instead, from
        # which each region's closed boundary is selected by side.
        boundary = (
            SurfaceModel.from_triangles(
                refined.coordinates,
                refined.triangles,
                metadata,
                vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
                cell_global_ids=cell_ids,
                numeric_version=domain.source_revision,
            )
            if multiregion_surface is None
            else None
        )
        organization = NativeMeshOrganization(
            boundary=boundary,
            patches=organization.patches,
            zones=organization.zones,
            labels=organization.labels,
            attributes=organization.attributes,
            associations=association_values,
            region_boundary_evidence=organization.region_boundary_evidence,
            surface_source=surface_source,
            multiregion_surface=multiregion_surface,
        )
        compliance = _compliance(
            self.specification,
            self.schedule,
            refined,
            self.size_values,
            self.selected_patches,
        )
        junctions: dict[int, set[int]] = {}
        for curve in curve_ids:
            source_curve = domain.curves[curve]
            junctions.setdefault(source_curve.start, set()).add(curve)
            junctions.setdefault(source_curve.end, set()).add(curve)
        junction_rows = {
            row
            for row, (dimension, index) in enumerate(
                zip(
                    refined.vertex_source_dimensions.tolist(),
                    refined.vertex_source_indices.tolist(),
                    strict=True,
                )
            )
            if dimension == 0 and len(junctions.get(index, ())) > 1
        }
        junction_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
            np.asarray(sorted(junction_rows), dtype=np.int64)
        ]
        scoped_fidelity = []
        for feature in self.specification.protected_features:
            if not feature.hard:
                continue
            feature_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            if feature.feature_kind.value == "curve":
                feature_query = _domain_curve_source_query(domain, feature_ids)
                feature_witness = _surface_curve_chart_witness(
                    mesh, geometry, domain, feature_query, construction
                )
            elif feature.feature_kind.value == "corner":
                feature_query = self.boundary_source
                feature_witness = _surface_point_witness(
                    mesh,
                    geometry,
                    domain,
                    feature_query,
                    feature_ids,
                    refined,
                )
            else:
                continue
            scoped_fidelity.append(
                (feature_query, feature_witness, feature.maximum_deviation)
            )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            domain.source_id,
            domain.source_revision,
            fidelity_source=self.boundary_source,
            fidelity_tolerance=refined.fidelity_tolerance,
            junction_vertices=junction_ids if junction_ids.size else None,
            protected_features=self.specification.protected_features,
            maximum_normal_deviation=self.specification.maximum_normal_deviation,
            fidelity_witness=surface_source.chart_witness,
            scoped_fidelity=tuple(scoped_fidelity),
            feature_source=domain,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
        )


def prepare_native_surface(
    source: domain_owner.MeshingDomain | ReconstructedGeometrySource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    preparation: domain_owner.SourcePreparation | None = None,
) -> _PreparedNativeSurface:
    """Prepare actual trimmed source charts for canonical mesh acceptance.

    ``preparation`` is the caller's `SourcePreparation` of the source domain;
    without one the domain is prepared for this call alone.
    """
    if options.route not in (None, "parametric_surface"):
        raise ValueError(
            "Native surface preparation requires the parametric_surface route."
        )
    schedule = options.surface_schedule
    if schedule is None:
        schedule = NativeSurfaceSchedule()
    if not isinstance(schedule, NativeSurfaceSchedule):
        raise TypeError("parametric_surface requires a NativeSurfaceSchedule.")
    domain = _domain_for_source(source)
    domain.require_current(domain.source_id, domain.source_revision)
    preparation = domain_owner.SourcePreparation.of(domain, preparation)
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
    if coordinate_contract.coordinate_system != "cartesian":
        raise _failure(
            "parametric surface construction requires an orthonormal Cartesian coordinate contract"
        )
    scope_ids, scope_set_ids = _scope_banks(domain)
    selected = _selected_patches(domain, specification, scope_ids, scope_set_ids)
    unsupported = _support_issues(
        domain,
        specification,
        selected,
        scope_ids,
        scope_set_ids,
    )
    if unsupported:
        raise _failure(
            "; ".join(unsupported),
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        )
    _validate_chart_trims(domain, selected, preparation)
    boundary_source = domain_boundary_owner.MeshingDomainBoundarySource(domain, selected)
    try:
        cover = boundary_source.boundary_chart_cover(preparation=preparation)
    except (TypeError, ValueError) as error:
        raise _failure(
            f"source boundary query cannot retain its trim cover: {error}",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=tuple(int(patch) for patch in selected),
        ) from error
    if not cover.complete:
        raise _failure(
            "source boundary query does not provide complete selected-patch coverage",
            category=MeshingFailureCategory.INVALID_SOURCE,
            entity_ids=tuple(int(patch) for patch in selected),
        )
    size_values, _ = _size_field(domain, specification, selected, scope_ids)
    roots = _chart_bounds(domain, selected, scope_ids, coordinate_contract, preparation)
    return _PreparedNativeSurface(
        domain,
        specification,
        coordinate_contract,
        schedule,
        selected,
        scope_ids,
        size_values,
        roots,
        boundary_source,
    )


__all__ = ["prepare_native_surface"]

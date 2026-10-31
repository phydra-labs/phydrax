#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Input-bound continuous geometric and oriented-normal fidelity proofs."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from typing import final, get_args, Protocol, runtime_checkable, TYPE_CHECKING

import jax
import numpy as np
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from ..discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    derivative,
    evaluate_expression,
    ExactScalar,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
)
from ..discretization._exact_algebraic_geometry import coefficient_endpoint
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._surface_chart_records import (
    _exact_chart_from_tokens,
    SurfaceChartWitness,
)
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._domain_boundary import CurvedTrimLens, MeshingDomainBoundarySource
from ._embedding_state import MeshCertificateFinding
from ._interval_enclosure import (
    Interval,
    interval_cos,
    interval_sin,
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ._mesh_certificates import (
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._meshing_domain import MeshingDomain
from ._planar_primitives import (
    area2 as _polygon_area2,
    cross as _polygon_cross,
    intersection as _polygon_intersection,
    signed_area2 as _polygon_signed_area2,
)
from ._rational import fraction_endpoint, rational_vector
from ._source_boundary_records import (
    _fractions_from_chart_tokens,
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistanceWitness,
    SourceBoundaryPointWitness,
    SourceBoundaryProjectionQuery,
    SourceBoundaryQuery,
    SurfaceEnvelopeDistanceWitness,
)
from ._source_identity import source_entity_id
from .brep._patches import (
    AbstractSurfacePatch,
    CircleCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
)


if TYPE_CHECKING:
    from .._interpolation._bspline_extraction import (
        RationalBSplineSurfaceHessianEnclosure,
        RationalBSplineSurfaceSpan,
    )
    from ..meshing._scope import _SourceEntityScopes
    from .implicit._adaptive_discovery import AdaptiveImplicitBoundarySource
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource


Point2 = tuple[Fraction, Fraction]
ExactPoint2 = tuple[ExactScalar, ExactScalar]
Triangle2 = tuple[Point2, Point2, Point2]
Point3 = tuple[Fraction, Fraction, Fraction]
type ExactPoint3 = tuple[ExactScalar, ExactScalar, ExactScalar]
Triangle3 = tuple[Point3, Point3, Point3]
type _RootCellKey = tuple[int, int, int, int]
type _RootPointKey = tuple[int, int, int]
type _RootEdgeKey = tuple[_RootPointKey, _RootPointKey]
type _ImplicitTriangleKey = tuple[
    _RootCellKey, tuple[_RootEdgeKey, _RootEdgeKey, _RootEdgeKey]
]


def _root_edge_key(points: np.ndarray, /) -> _RootEdgeKey:
    if points.shape != (2, 3):
        raise ValueError(
            "Implicit root edge keys require two three-dimensional endpoints."
        )
    return (
        (int(points[0, 0]), int(points[0, 1]), int(points[0, 2])),
        (int(points[1, 0]), int(points[1, 1]), int(points[1, 2])),
    )


def _affine_triangle_arguments(triangle: Triangle2, /) -> tuple[Polynomial, Polynomial]:
    arguments: list[Polynomial] = []
    for axis in range(2):
        polynomial: Polynomial = {}
        polynomial[(0, 0)] = triangle[0][axis]
        polynomial[(1, 0)] = triangle[1][axis] - triangle[0][axis]
        polynomial[(0, 1)] = triangle[2][axis] - triangle[0][axis]
        arguments.append(polynomial)
    return arguments[0], arguments[1]


def _affine_volume_triangle_arguments(
    triangle: Triangle3, /
) -> tuple[Polynomial, Polynomial, Polynomial]:
    arguments: list[Polynomial] = []
    for axis in range(3):
        polynomial: Polynomial = {
            (0, 0): triangle[0][axis],
            (1, 0): triangle[1][axis] - triangle[0][axis],
            (0, 1): triangle[2][axis] - triangle[0][axis],
        }
        arguments.append(polynomial)
    return arguments[0], arguments[1], arguments[2]


def _exact_triangle2(points: tuple[tuple[Fraction, ...], ...], /) -> Triangle2:
    if len(points) != 3 or any(len(point) != 2 for point in points):
        raise ValueError("Exact surface charts must have three two-dimensional corners.")
    return (
        (Fraction(points[0][0]), Fraction(points[0][1])),
        (Fraction(points[1][0]), Fraction(points[1][1])),
        (Fraction(points[2][0]), Fraction(points[2][1])),
    )


def _exact_triangle3(points: tuple[tuple[Fraction, ...], ...], /) -> Triangle3:
    if len(points) != 3 or any(len(point) != 3 for point in points):
        raise ValueError(
            "Exact volume-boundary charts require three 3-D parent-reference corners."
        )
    return (
        (Fraction(points[0][0]), Fraction(points[0][1]), Fraction(points[0][2])),
        (Fraction(points[1][0]), Fraction(points[1][1]), Fraction(points[1][2])),
        (Fraction(points[2][0]), Fraction(points[2][1]), Fraction(points[2][2])),
    )


type SourceFidelityWitness = (
    SurfaceChartWitness
    | SourceBoundaryChartWitness
    | SourceBoundaryPointWitness
    | SourceBoundaryDistanceWitness
    | SurfaceEnvelopeDistanceWitness
)


def _integer_vector(values: ArrayLike, /) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Witness identity banks must be one-dimensional integer arrays.")
    return tuple(int(array[row]) for row in range(array.size))


def _chart_orientation_vector(witness: SourceBoundaryChartWitness, /) -> tuple[int, ...]:
    orientations = np.asarray(witness.orientations)
    if (
        orientations.ndim != 1
        or not np.issubdtype(orientations.dtype, np.integer)
        or orientations.size != len(witness.source_chart_ids)
    ):
        raise ValueError(
            "Chart witness orientation bank must align with its source charts."
        )
    return tuple(int(orientations[row]) for row in range(orientations.size))


def _area2(polygon: tuple[Point2, ...], /) -> Fraction:
    return _polygon_signed_area2(polygon) / 2


def _interiors_separated(first: Triangle2, second: Triangle2, /) -> bool:
    """Whether an edge line of either triangle exactly separates their interiors.

    Two convex polygons have disjoint interiors exactly when a supporting line
    of one of their edges leaves the other polygon in its closed outer half
    plane, so this decides zero overlap area without clipping.
    """
    for inner, outer in ((first, second), (second, first)):
        orientation = _polygon_signed_area2(inner)
        if orientation == 0:
            return True
        for index in range(3):
            start, end = inner[index], inner[(index + 1) % 3]
            if all(
                _polygon_cross(start, end, point) * orientation <= 0 for point in outer
            ):
                return True
    return False


def _intersection_area(first: Triangle2, second: Triangle2, /) -> Fraction:
    if _interiors_separated(first, second):
        return Fraction(0)
    return _polygon_area2(_polygon_intersection(first, second)) / 2


def _triangle_chart_bvh(triangles: tuple[Triangle2, ...], /) -> PackedBVH | None:
    if not triangles:
        return None
    lower = np.empty((len(triangles), 2), dtype=np.float64)
    upper = np.empty_like(lower)
    for row, triangle in enumerate(triangles):
        for axis in range(2):
            low = fraction_endpoint(min(point[axis] for point in triangle), upper=False)
            high = fraction_endpoint(max(point[axis] for point in triangle), upper=True)
            if not math.isfinite(low) or not math.isfinite(high):
                return None
            lower[row, axis] = low
            upper[row, axis] = high
    return prepare_bvh(lower, upper, dtype=np.float64)


def _triangle_overlap_candidate_pairs(
    first: tuple[Triangle2, ...],
    second: tuple[Triangle2, ...],
    first_bvh: PackedBVH | None,
    second_bvh: PackedBVH | None,
    /,
    *,
    self_pairs: bool,
) -> tuple[tuple[int, int], ...]:
    if self_pairs:
        if len(first) != len(second):
            raise ValueError("Self-overlap chart banks must be the same triangle bank.")
    elif not first or not second:
        return ()
    if first_bvh is None or second_bvh is None:
        candidates = (
            (
                (left, right)
                for left in range(len(first))
                for right in range(left + 1, len(first))
            )
            if self_pairs
            else (
                (left, right)
                for left in range(len(first))
                for right in range(len(second))
            )
        )
        return tuple(candidates)
    blocks: list[np.ndarray] = []
    for first_ids, second_ids in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if self_pairs:
            keep = first_ids < second_ids
            first_ids, second_ids = first_ids[keep], second_ids[keep]
        if first_ids.size:
            blocks.append(
                np.column_stack((first_ids, second_ids)).astype(np.int64, copy=False)
            )
    if not blocks:
        return ()
    pairs = np.concatenate(blocks, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return tuple((int(pairs[row, 0]), int(pairs[row, 1])) for row in order)


def _surface_source_chart_rows(
    source: MeshingDomainBoundarySource,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[np.ndarray, tuple[Triangle2, ...]]:
    if (
        witness.chart_dimension != 2
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != source.source_scope_id
    ):
        raise ValueError(
            "Source surface witness changes its chart dimension, source, or query scope."
        )
    domain = source.domain
    patch_rows = np.empty((len(witness.source_entity_ids),), dtype=np.int64)
    for row, (chart_id, entity_id, occurrence) in enumerate(
        zip(
            witness.source_chart_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
    ):
        patch = domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
        if (
            patch not in source.patches
            or domain.entity_id(2, patch) != entity_id
            or domain.source_occurrences[2][patch] != occurrence
            or chart_id != entity_id
        ):
            raise ValueError(
                "Source surface witness changes its authored chart or occurrence."
            )
        patch_rows[row] = patch
    return patch_rows, tuple(
        _exact_triangle2(
            _fractions_from_chart_tokens(witness.source_parameter_rational_tokens, row)
        )
        for row in range(len(patch_rows))
    )


def _surface_query_root(source: SourceBoundaryQuery, /) -> MeshingDomainBoundarySource:
    if isinstance(source, MeshingDomainBoundarySource):
        return source
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(source, SurfaceNativeRestrictionBoundarySource):
        return source.root
    raise TypeError("Surface fidelity requires a retained MeshingDomain source query.")


def _validate_surface_root_mesh_bindings(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SurfaceChartWitness,
    /,
) -> None:
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(source, SurfaceNativeRestrictionBoundarySource):
        source.support.prove_native_restrictions(mesh, geometry, chart_witness=witness)
        return
    geometry_id = cell_geometry_id(geometry)
    target_cell_ids = _integer_vector(witness.cell_global_ids)
    if len(target_cell_ids) != len(witness.root_mesh_cell_bindings):
        raise ValueError(
            "Surface root-cell bindings must align with chart target-cell identities."
        )
    for row, binding in enumerate(witness.root_mesh_cell_bindings):
        if binding is None:
            continue
        if (
            binding.root_geometry_id != geometry_id
            or binding.root_topology_id != mesh.topology_id
            or binding.root_cell_global_id != target_cell_ids[row]
        ):
            raise ValueError(
                "Surface chart root-cell binding differs from the exact target root mesh."
            )


def _query_source_corner_rows(source: MeshingDomainBoundarySource, /) -> frozenset[int]:
    domain = source.domain
    from ._meshing_domain import PatchCurveUse, PatchPoleUse

    corners: set[int] = set()
    for patch_row in source.patches:
        for loop in domain.patches[patch_row].loops:
            for use in loop:
                if isinstance(use, PatchPoleUse):
                    corners.add(int(use.corner))
                elif isinstance(use, PatchCurveUse):
                    curve = domain.curves[use.curve]
                    corners.update((int(curve.start), int(curve.end)))
                else:
                    raise TypeError(
                        "MeshingDomain trim contains an unsupported source corner authority."
                    )
    return frozenset(corners)


def _reconstructed_domain_query(
    source: SourceBoundaryQuery, /
) -> MeshingDomainBoundarySource:
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource
    from .reconstruction._core import ReconstructedGeometrySource
    from .reconstruction._sheet import AuthoredTriangleSheet

    if not isinstance(source, ReconstructedTriangleBoundarySource):
        raise TypeError(
            "Reconstructed source fidelity requires ReconstructedTriangleBoundarySource."
        )
    if not isinstance(
        source.source, (ReconstructedGeometrySource, AuthoredTriangleSheet)
    ):
        raise TypeError(
            "Reconstructed boundary source must retain ReconstructedGeometrySource "
            "or AuthoredTriangleSheet."
        )
    fresh = ReconstructedTriangleBoundarySource(source.source)
    if (
        source.source_id != fresh.source_id
        or source.source_revision != fresh.source_revision
        or not np.array_equal(source._vertices, fresh._vertices)
        or not np.array_equal(source._faces, fresh._faces)
    ):
        raise ValueError(
            "Reconstructed triangle query no longer reproduces its source mesh."
        )
    domain = source.source.meshing_domain
    if domain is None:
        raise ValueError(
            "Reconstructed source has no authoritative persistent MeshingDomain."
        )
    if (
        domain.source_id != source.source_id
        or domain.source_revision != source.source_revision
        or not np.array_equal(
            np.asarray(domain.source_indices[2], dtype=np.int64),
            np.asarray(source.source.persistent_facet_ids, dtype=np.int64),
        )
        or not np.array_equal(np.asarray(domain.corner_points), source._vertices)
    ):
        raise ValueError(
            "Reconstructed MeshingDomain changes persistent source vertex/facet IDs."
        )
    query = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    original_cover = source.boundary_chart_cover()
    domain_cover = query.boundary_chart_cover()
    if (
        not original_cover.complete
        or not domain_cover.complete
        or original_cover.semantics != "certified"
        or domain_cover.semantics != "certified"
        or np.any(original_cover.deviation_bounds != 0.0)
        or np.any(domain_cover.deviation_bounds != 0.0)
    ):
        raise ValueError(
            "Reconstructed source charts lack complete exact physical coverage."
        )

    def triangle_keys(values: np.ndarray) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
        return tuple(
            sorted(
                tuple(
                    sorted(
                        tuple(Fraction.from_float(float(value)) for value in point)
                        for point in triangle
                    )
                )
                for triangle in values
            )
        )

    if triangle_keys(original_cover.simplices) != triangle_keys(domain_cover.simplices):
        raise ValueError(
            "Reconstructed query and persistent source domain cover different physical triangles."
        )
    return query


def _partition_findings(
    source: MeshingDomainBoundarySource,
    witness: SurfaceChartWitness | SourceBoundaryChartWitness,
    /,
) -> tuple[MeshCertificateFinding, ...]:
    # Currentness fingerprints the whole domain: check once, not per patch.
    source.require_current()
    domain = source.domain
    if isinstance(witness, SurfaceChartWitness):
        patches = np.asarray(witness.patches, dtype=np.int64)
        charts = tuple(
            _exact_chart_from_tokens(witness.chart_rational_tokens, row)
            for row in range(len(patches))
        )
    else:
        patches, charts = _surface_source_chart_rows(source, witness)
    target_triangles: dict[int, list[Triangle2]] = {}
    for patch, chart in zip(patches, charts, strict=True):
        target_triangles.setdefault(int(patch), []).append(chart)
    findings: list[MeshCertificateFinding] = []
    if set(target_triangles) != set(source.patches):
        findings.append(
            MeshCertificateFinding(
                "source_chart_patch_coverage",
                "violated",
                "source_facet",
                detail="Surface chart witness omits or adds a declared source patch.",
            )
        )
        return tuple(findings)
    for patch in source.patches:
        target = tuple(target_triangles.get(patch, ()))
        if not target:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_patch_coverage",
                    "violated",
                    "source_facet",
                    (patch,),
                    detail="A source patch has no complete target chart partition.",
                )
            )
            continue
        if isinstance(witness, SurfaceChartWitness):
            source_orientation = int(domain.patches[patch].orientation)
            invalid_orientation = any(
                source_orientation * _area2(triangle) <= 0 for triangle in target
            )
        else:
            invalid_orientation = any(_area2(triangle) == 0 for triangle in target)
        if invalid_orientation:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_orientation",
                    "violated",
                    "source_facet",
                    (patch,),
                    detail="Target material charts reverse the authored source-patch orientation.",
                )
            )
            continue
        if isinstance(witness, SurfaceChartWitness):
            # Collapsed pole slivers are part of their owner cells' chart preimages.
            target = (
                *target,
                *(sliver for _, sliver in source.collapsed_pole_slivers(patch, target)),
            )
        declared, _ = source._trim_partition_reference_unchecked(patch, target)
        source_areas = tuple(abs(_area2(triangle)) for triangle in declared)
        target_areas = tuple(abs(_area2(triangle)) for triangle in target)
        source_area = sum(source_areas, Fraction(0))
        target_area = sum(target_areas, Fraction(0))
        source_bvh = _triangle_chart_bvh(declared)
        target_bvh = _triangle_chart_bvh(target)
        source_overlap = any(
            _intersection_area(declared[first], declared[second]) > 0
            for first, second in _triangle_overlap_candidate_pairs(
                declared, declared, source_bvh, source_bvh, self_pairs=True
            )
        )
        target_overlap = any(
            _intersection_area(target[first], target[second]) > 0
            for first, second in _triangle_overlap_candidate_pairs(
                target, target, target_bvh, target_bvh, self_pairs=True
            )
        )
        target_covered = [Fraction(0) for _ in target]
        for target_row, source_row in _triangle_overlap_candidate_pairs(
            target, declared, target_bvh, source_bvh, self_pairs=False
        ):
            target_covered[target_row] += _intersection_area(
                target[target_row], declared[source_row]
            )
        target_outside = any(
            covered != area
            for covered, area in zip(target_covered, target_areas, strict=True)
        )
        if source_overlap:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_source_partition",
                    "unresolved",
                    "source_facet",
                    (patch,),
                    detail="Independent source chart triangles have positive-measure overlap.",
                )
            )
        if target_overlap or (
            not source_overlap and (target_outside or target_area != source_area)
        ):
            findings.append(
                MeshCertificateFinding(
                    "source_chart_coverage",
                    "violated",
                    "source_facet",
                    (patch,),
                    detail="Exact chart containment, disjointness, or full trim measure fails.",
                )
            )
    return tuple(findings)


def _poly_is_affine(value: Polynomial | RationalPolynomial, /) -> bool:
    return isinstance(value, dict) and all(sum(power) <= 1 for power in value)


def _box_gap_lower(
    source_lower: float,
    source_upper: float,
    target: ExactScalar,
    /,
) -> Fraction:
    target_lower = Fraction(coefficient_endpoint(target, upper=False))
    target_upper = Fraction(coefficient_endpoint(target, upper=True))
    source_low = Fraction(float(source_lower))
    source_high = Fraction(float(source_upper))
    return max(Fraction(0), source_low - target_upper, target_lower - source_high)


def _box_gap_upper(
    source_lower: float,
    source_upper: float,
    target: ExactScalar,
    /,
) -> Fraction:
    target_lower = Fraction(coefficient_endpoint(target, upper=False))
    target_upper = Fraction(coefficient_endpoint(target, upper=True))
    source_low = Fraction(float(source_lower))
    source_high = Fraction(float(source_upper))
    return max(abs(target_lower - source_high), abs(target_upper - source_low))


def _distance_lower(components: tuple[Fraction, ...], /) -> Fraction:
    return fraction_sqrt_interval(
        sum((value * value for value in components), Fraction(0))
    )[0]


def _distance_upper(components: tuple[Fraction, ...], /) -> Fraction:
    return fraction_sqrt_interval(
        sum((value * value for value in components), Fraction(0))
    )[1]


def _interval_product(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    products = tuple(a * b for a in first for b in second)
    return min(products), max(products)


def _interval_difference(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    return first[0] - second[1], first[1] - second[0]


def _normal_cross_interval(
    jacobian_lower: np.ndarray,
    jacobian_upper: np.ndarray,
    orientation: int,
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    columns = tuple(
        tuple(
            (
                Fraction(float(jacobian_lower[row, column])),
                Fraction(float(jacobian_upper[row, column])),
            )
            for row in range(3)
        )
        for column in range(2)
    )
    first, second = columns
    components = (
        _interval_difference(
            _interval_product(first[1], second[2]), _interval_product(first[2], second[1])
        ),
        _interval_difference(
            _interval_product(first[2], second[0]), _interval_product(first[0], second[2])
        ),
        _interval_difference(
            _interval_product(first[0], second[1]), _interval_product(first[1], second[0])
        ),
    )
    if orientation < 0:
        return tuple((-upper, -lower) for lower, upper in components)
    return components


def _normal_cosine_interval_bounds(
    first_cross: tuple[tuple[Fraction, Fraction], ...],
    second_cross: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[Fraction, Fraction] | None:
    dot_lower = Fraction(0)
    dot_upper = Fraction(0)
    first_minimum = []
    first_maximum = []
    second_minimum = []
    second_maximum = []
    for (first_low, first_high), (second_low, second_high) in zip(
        first_cross, second_cross, strict=True
    ):
        product = _interval_product((first_low, first_high), (second_low, second_high))
        dot_lower += product[0]
        dot_upper += product[1]
        first_minimum.append(
            Fraction(0)
            if first_low <= 0 <= first_high
            else min(abs(first_low), abs(first_high))
        )
        first_maximum.append(max(abs(first_low), abs(first_high)))
        second_minimum.append(
            Fraction(0)
            if second_low <= 0 <= second_high
            else min(abs(second_low), abs(second_high))
        )
        second_maximum.append(max(abs(second_low), abs(second_high)))
    first_norm_lower = _distance_lower(tuple(first_minimum))
    first_norm_upper = _distance_upper(tuple(first_maximum))
    second_norm_lower = _distance_lower(tuple(second_minimum))
    second_norm_upper = _distance_upper(tuple(second_maximum))
    if min(first_norm_lower, second_norm_lower) <= 0:
        return None
    cosine_lower = (
        dot_lower / (first_norm_upper * second_norm_upper)
        if dot_lower >= 0
        else dot_lower / (first_norm_lower * second_norm_lower)
    )
    cosine_upper = (
        dot_upper / (first_norm_lower * second_norm_lower)
        if dot_upper >= 0
        else dot_upper / (first_norm_upper * second_norm_upper)
    )
    return max(Fraction(-1), cosine_lower), min(Fraction(1), cosine_upper)


def _normal_cosine_bounds(
    source_cross: tuple[tuple[Fraction, Fraction], ...],
    target_cross: tuple[ExactScalar, ExactScalar, ExactScalar],
    /,
) -> tuple[Fraction, Fraction] | None:
    target_intervals = tuple(
        (
            Fraction(coefficient_endpoint(value, upper=False)),
            Fraction(coefficient_endpoint(value, upper=True)),
        )
        for value in target_cross
    )
    return _normal_cosine_interval_bounds(source_cross, target_intervals)


def _acos_lower(cosine_upper: Fraction, /) -> Fraction:
    if cosine_upper >= 1:
        return Fraction(0)
    if cosine_upper <= -1:
        return Fraction(math.pi)
    lower, upper = 0.0, float(np.nextafter(math.pi, math.inf))
    while np.nextafter(lower, math.inf) < upper:
        midpoint = lower + (upper - lower) / 2
        if midpoint == lower or midpoint == upper:
            break
        interval = interval_cos((np.asarray(midpoint), np.asarray(midpoint)))
        if Fraction(float(interval[0])) >= cosine_upper:
            lower = midpoint
        else:
            upper = midpoint
    return Fraction(lower)


def _acos_upper(cosine_lower: Fraction, /) -> Fraction:
    if cosine_lower >= 1:
        return Fraction(0)
    if cosine_lower <= -1:
        return Fraction(float(np.nextafter(math.pi, math.inf)))
    lower, upper = 0.0, float(np.nextafter(math.pi, math.inf))
    while np.nextafter(lower, math.inf) < upper:
        midpoint = lower + (upper - lower) / 2
        if midpoint == lower or midpoint == upper:
            break
        interval = interval_cos((np.asarray(midpoint), np.asarray(midpoint)))
        if Fraction(float(interval[1])) <= cosine_lower:
            upper = midpoint
        else:
            lower = midpoint
    return Fraction(upper)


def _asin_upper(sine_upper: Fraction, /) -> Fraction:
    """Outward upper bound of ``asin`` on ``[0, 1]`` by certified bisection."""
    if sine_upper <= 0:
        return Fraction(0)
    half_pi = float(np.nextafter(math.pi / 2, math.inf))
    if sine_upper >= 1:
        return Fraction(half_pi)
    lower, upper = 0.0, half_pi
    while np.nextafter(lower, math.inf) < upper:
        midpoint = lower + (upper - lower) / 2
        if midpoint == lower or midpoint == upper:
            break
        interval = interval_sin((np.asarray(midpoint), np.asarray(midpoint)))
        if Fraction(float(interval[0])) >= sine_upper:
            upper = midpoint
        else:
            lower = midpoint
    return Fraction(upper)


def _normal_angle_upper(
    source_cross: tuple[tuple[Fraction, Fraction], ...],
    target_cross: tuple[ExactScalar, ExactScalar, ExactScalar],
    cosine_lower: Fraction,
    /,
) -> Fraction:
    """Upper normal angle from the cosine and, when acute, the sine bound.

    The cosine of interval normals loses ``O(h)`` to independently maximized
    norms, so ``acos`` alone is ``O(sqrt(h))`` on a curved cell. With a positive
    dot product the angle is acute and ``sin θ <= |S x t| / (|S| |t|)`` from the
    interval cross product is first order in the cell's tangent range.
    """
    upper = _acos_upper(cosine_lower)
    if cosine_lower <= 0:
        return upper
    target = tuple(
        (
            Fraction(coefficient_endpoint(value, upper=False)),
            Fraction(coefficient_endpoint(value, upper=True)),
        )
        for value in target_cross
    )
    cross = (
        _interval_difference(
            _interval_product(source_cross[1], target[2]),
            _interval_product(source_cross[2], target[1]),
        ),
        _interval_difference(
            _interval_product(source_cross[2], target[0]),
            _interval_product(source_cross[0], target[2]),
        ),
        _interval_difference(
            _interval_product(source_cross[0], target[1]),
            _interval_product(source_cross[1], target[0]),
        ),
    )

    def norm_lower(intervals: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
        return _distance_lower(
            tuple(
                Fraction(0) if low <= 0 <= high else min(abs(low), abs(high))
                for low, high in intervals
            )
        )

    source_norm = norm_lower(source_cross)
    target_norm = norm_lower(target)
    if source_norm <= 0 or target_norm <= 0:
        return upper
    cross_norm = _distance_upper(tuple(max(abs(low), abs(high)) for low, high in cross))
    sine_upper = min(Fraction(1), cross_norm / (source_norm * target_norm))
    return min(upper, _asin_upper(sine_upper))


def _source_domain_id(source: SourceBoundaryQuery, /) -> str:
    if isinstance(source, MeshingDomainBoundarySource):
        return source.domain.domain_id
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(source, SurfaceNativeRestrictionBoundarySource):
        return source.domain.domain_id
    from ._curve_boundary import ParametricCurveBoundarySource

    if isinstance(source, ParametricCurveBoundarySource):
        return source.query_id
    from ._mapped_reference_boundary import MappedReferenceBoundarySource

    if isinstance(source, MappedReferenceBoundarySource):
        return source.source_query_id
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    if isinstance(source, ReconstructedTriangleBoundarySource):
        domain = source.source.meshing_domain
        return source.source_revision if domain is None else domain.domain_id
    from .implicit._analytic_profile import AnalyticImplicitProfile

    if isinstance(source, AnalyticImplicitProfile):
        return source.profile_id
    return source.source_revision


@final
class SourceFidelityCertificate(StrictModule, NonTrainableState):
    """Continuous two-directed bounds and optional oriented-normal evidence."""

    __strict_contract__ = True

    binding: MeshCertificateBinding
    source: SourceBoundaryQuery
    source_id: str
    source_revision: str
    domain_id: str
    mesh_id: str
    topology_id: str
    geometry_id: str
    tolerance: float
    maximum_normal_deviation: float | None
    deviation_bounds: np.ndarray
    mesh_to_source_lower: float
    mesh_to_source_upper: float
    source_to_mesh_lower: float
    source_to_mesh_upper: float
    normal_deviation_bounds: np.ndarray | None
    maximum_normal_deviation_lower: float | None
    maximum_normal_deviation_bound: float | None
    source_cell_global_ids: tuple[int, ...]
    source_patch_ids: tuple[int, ...]
    source_occurrence_paths: tuple[tuple[str, ...], ...]
    witness: SourceFidelityWitness | None
    coverage: SourceBoundaryChartCover | None
    status: MeshCertificateStatus
    findings: tuple[MeshCertificateFinding, ...]
    certificate_id: str

    def __init__(
        self,
        binding: MeshCertificateBinding,
        source: SourceBoundaryQuery,
        tolerance: float,
        maximum_normal_deviation: float | None,
        deviation_bounds: ArrayLike,
        mesh_to_source_lower: float,
        mesh_to_source_upper: float,
        source_to_mesh_lower: float,
        source_to_mesh_upper: float,
        normal_deviation_bounds: ArrayLike | None,
        maximum_normal_deviation_lower: float | None,
        source_cell_global_ids: tuple[int, ...],
        source_patch_ids: tuple[int, ...],
        source_occurrence_paths: tuple[tuple[str, ...], ...],
        witness: SourceFidelityWitness | None,
        coverage: SourceBoundaryChartCover | None,
        findings: tuple[MeshCertificateFinding, ...],
        /,
    ) -> None:
        bounds = np.asarray(deviation_bounds, dtype=np.float64)
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or np.any(np.isnan(bounds))
            or np.any(bounds < 0)
        ):
            raise ValueError(
                "SourceFidelityCertificate.deviation_bounds must be nonnegative lower/upper pairs."
            )
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError(
                "SourceFidelityCertificate deviation lower bounds exceed upper bounds."
            )
        bounds = np.array(bounds, copy=True)
        bounds.setflags(write=False)
        normal_bounds = None
        if normal_deviation_bounds is not None:
            normal_bounds = np.asarray(normal_deviation_bounds, dtype=np.float64)
            if (
                normal_bounds.ndim != 1
                or np.any(np.isnan(normal_bounds))
                or np.any(normal_bounds < 0)
            ):
                raise ValueError(
                    "SourceFidelityCertificate normal bounds must be nonnegative radians."
                )
            normal_bounds = np.array(normal_bounds, copy=True)
            normal_bounds.setflags(write=False)
        tolerance_ = float(tolerance)
        maximum_normal_ = (
            None if maximum_normal_deviation is None else float(maximum_normal_deviation)
        )
        if not math.isfinite(tolerance_) or tolerance_ < 0:
            raise ValueError("Source fidelity tolerance must be finite and nonnegative.")
        if maximum_normal_ is not None and (
            not math.isfinite(maximum_normal_)
            or maximum_normal_ < 0
            or maximum_normal_ > math.pi
        ):
            raise ValueError(
                "maximum_normal_deviation must be None or finite radians in [0, pi]."
            )
        values = tuple(
            float(value)
            for value in (
                mesh_to_source_lower,
                mesh_to_source_upper,
                source_to_mesh_lower,
                source_to_mesh_upper,
            )
        )
        if any(math.isnan(value) or value < 0 for value in values):
            raise ValueError(
                "Source fidelity directed bounds must be nonnegative and non-NaN."
            )
        if values[0] > values[1] or values[2] > values[3]:
            raise ValueError("Source fidelity directed lower bounds exceed upper bounds.")
        if (
            len(source_cell_global_ids) != bounds.shape[0]
            or len(source_patch_ids) != bounds.shape[0]
            or len(source_occurrence_paths) != bounds.shape[0]
        ):
            raise ValueError(
                "Source fidelity cell identities must align with per-cell bounds."
            )
        status = certificate_status(findings)
        self.binding = binding
        self.source = source
        self.source_id = binding.source_id or source.source_id
        self.source_revision = binding.source_revision or source.source_revision
        self.domain_id = _source_domain_id(source)
        self.mesh_id = binding.mesh_id
        self.topology_id = binding.topology_id
        self.geometry_id = binding.geometry_id
        self.tolerance = tolerance_
        self.maximum_normal_deviation = maximum_normal_
        self.deviation_bounds = bounds
        self.mesh_to_source_lower, self.mesh_to_source_upper = values[0], values[1]
        self.source_to_mesh_lower, self.source_to_mesh_upper = values[2], values[3]
        self.normal_deviation_bounds = normal_bounds
        self.maximum_normal_deviation_lower = maximum_normal_deviation_lower
        self.maximum_normal_deviation_bound = (
            None
            if normal_bounds is None or normal_bounds.size == 0
            else float(np.max(normal_bounds))
        )
        self.source_cell_global_ids = tuple(
            int(value) for value in source_cell_global_ids
        )
        self.source_patch_ids = tuple(int(value) for value in source_patch_ids)
        self.source_occurrence_paths = tuple(
            tuple(path) for path in source_occurrence_paths
        )
        self.witness = witness
        self.coverage = coverage
        self.status = status
        self.findings = tuple(findings)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "source-fidelity-certificate",
                "binding": binding.binding_id,
                "source": (self.source_id, self.source_revision, self.domain_id),
                "tolerance": tolerance_,
                "maximum_normal_deviation": maximum_normal_,
                "deviation_bounds": array_tree_fingerprint(bounds),
                "directed_bounds": tuple(
                    value if math.isfinite(value) else value.hex() for value in values
                ),
                "normal_bounds": None
                if normal_bounds is None
                else array_tree_fingerprint(normal_bounds),
                "normal_lower": maximum_normal_deviation_lower,
                "cells": self.source_cell_global_ids,
                "patches": self.source_patch_ids,
                "occurrences": self.source_occurrence_paths,
                "witness": None if witness is None else witness.witness_id,
                "coverage": None if coverage is None else coverage.cover_id,
                "findings": tuple(finding.finding_id for finding in findings),
                "status": status,
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        self.binding.require(mesh, geometry, source=source)
        fresh = certify_source_fidelity(
            mesh,
            geometry,
            source,
            tolerance=self.tolerance,
            maximum_normal_deviation=self.maximum_normal_deviation,
            witness=self.witness,
        )
        if fresh.certificate_id != self.certificate_id:
            raise ValueError(
                "Source fidelity proof does not reproduce its current source and map inputs."
            )


def _validate_piecewise_linear_chart_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
) -> None:
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if not isinstance(source, PiecewiseLinearBoundarySource):
        raise TypeError("PWL chart validation requires PiecewiseLinearBoundarySource.")
    if (
        source.entity_dimension != witness.chart_dimension
        or witness.chart_dimension not in (1, 2)
    ):
        raise ValueError(
            "PWL chart witness changes the selected source stratum dimension."
        )
    if mesh.ambient_dimension != 3 or mesh.topological_dimension not in (1, 2, 3):
        raise ValueError("PWL boundary fidelity requires a target mesh in three-space.")
    source.validate_chart_witness(witness)
    if witness.chart_dimension == 1:
        if witness.reference_dimension != mesh.topological_dimension:
            raise ValueError(
                "PWL edge witness changes the target parent-cell reference dimension."
            )
        _curve_target_chart_rows(mesh, witness)
        by_chart: dict[str, list[tuple[Fraction, Fraction]]] = {}
        for row, chart_id in enumerate(witness.source_chart_ids):
            parameters = _fractions_from_chart_tokens(
                witness.source_parameter_rational_tokens, row
            )
            first, last = parameters[0][0], parameters[1][0]
            if first == last:
                raise ValueError(
                    "PWL source edge chart pieces must have positive parameter length."
                )
            by_chart.setdefault(chart_id, []).append((min(first, last), max(first, last)))
        if set(by_chart) != set(source.chart_ids):
            raise ValueError("PWL edge witness omits or adds a selected source edge.")
        for intervals in by_chart.values():
            cursor = Fraction(0)
            for lower, upper in sorted(intervals):
                if lower != cursor:
                    raise ValueError(
                        "PWL source edge parameter pieces overlap or leave a gap."
                    )
                cursor = upper
            if cursor != 1:
                raise ValueError("PWL source edge witness omits part of a complete edge.")
        return
    if mesh.topological_dimension not in (2, 3):
        raise ValueError(
            "PWL surface charts require a two- or three-dimensional target mesh."
        )
    expected_reference_dimension = (
        2
        if np.asarray(witness.target_face_vertex_global_ids).size
        or mesh.topological_dimension == 2
        else mesh.topological_dimension
    )
    if witness.reference_dimension != expected_reference_dimension:
        raise ValueError("PWL chart witness changes the target reference dimension.")
    _, _, target_orientations = _surface_target_chart_rows(
        mesh,
        geometry,
        None,
        witness,
        full_source=source.is_full_scope,
    )
    by_chart: dict[str, list[Triangle2]] = {}
    source_orientations: list[int] = []
    standard: Triangle2 = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for row, chart_id in enumerate(witness.source_chart_ids):
        parameters = _exact_triangle2(
            _fractions_from_chart_tokens(witness.source_parameter_rational_tokens, row)
        )
        by_chart.setdefault(chart_id, []).append(parameters)
        source_orientations.append(1 if _area2(parameters) > 0 else -1)
    if set(by_chart) != set(source.chart_ids):
        raise ValueError("PWL chart witness omits or adds a selected source facet.")
    for pieces in by_chart.values():
        _require_reference_chart_partition(tuple(pieces), (standard,))
    expected_orientations = tuple(
        source_sign * target_sign
        for source_sign, target_sign in zip(
            source_orientations, target_orientations, strict=True
        )
    )
    if expected_orientations != _chart_orientation_vector(witness):
        raise ValueError("PWL chart witness changes source/target orientation ancestry.")


def validate_source_fidelity_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceFidelityWitness,
    /,
) -> None:
    """Revalidate witness identity and its actual source/target mathematical premises."""
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError("source must implement SourceBoundaryQuery.")
    if not isinstance(witness, get_args(SourceFidelityWitness.__value__)):
        raise TypeError(
            "witness must be a member of the exported SourceFidelityWitness union."
        )
    from ._mapped_reference_boundary import MappedReferenceBoundarySource

    if isinstance(source, MappedReferenceBoundarySource):
        from .._physical import SpatialCoordinateContract

        contract = source.coordinate_contract
        if not isinstance(contract, SpatialCoordinateContract):
            raise TypeError(
                "Mapped root fidelity requires its domain-owned SpatialCoordinateContract."
            )
        if contract is not source.domain.coordinate_contract or not contract.spatial_id:
            raise ValueError(
                "Mapped root fidelity must use the identity-bound domain coordinate contract."
            )
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(witness, SourceBoundaryDistanceWitness):
        _validate_analytic_profile_distance_witness(mesh, geometry, source, witness)
        return
    if isinstance(witness, SurfaceEnvelopeDistanceWitness):
        _validate_surface_envelope_witness(mesh, geometry, source, witness)
        return
    if isinstance(witness, SourceBoundaryPointWitness):
        _source_point_correspondences(mesh, geometry, source, witness)
        return
    if isinstance(witness, SurfaceChartWitness):
        domain_source = _surface_query_root(source)
        if isinstance(source, SurfaceNativeRestrictionBoundarySource):
            source.require_current()
        else:
            domain_source.require_current()
        _validate_surface_root_mesh_bindings(mesh, geometry, source, witness)
        renewed = SurfaceChartWitness(
            geometry_id=witness.geometry_id,
            topology_id=witness.topology_id,
            domain_id=witness.domain_id,
            cell_global_ids=np.asarray(witness.cell_global_ids),
            patches=np.asarray(witness.patches),
            charts=np.asarray(witness.charts),
            chart_rational_tokens=witness.chart_rational_tokens,
            source_root_cell_ids=np.asarray(witness.source_root_cell_ids),
            source_piece_ids=witness.source_piece_ids,
            geometry_entity_ids=witness.geometry_entity_ids,
            occurrence_paths=witness.occurrence_paths,
            root_mesh_cell_bindings=witness.root_mesh_cell_bindings,
        )
        from ._surface_source_support import validate_surface_source_pieces

        validate_surface_source_pieces(
            domain_source.domain,
            np.asarray(witness.patches),
            np.asarray(witness.charts),
            np.asarray(witness.source_root_cell_ids),
            witness.source_piece_ids,
            exact_charts=tuple(
                _exact_chart_from_tokens(witness.chart_rational_tokens, row)
                for row in range(len(witness.source_piece_ids))
            ),
        )
        if renewed.witness_id != witness.witness_id:
            raise ValueError(
                "Surface chart witness identity does not reproduce its current arrays."
            )
        findings = _partition_findings(domain_source, witness)
        if any(finding.status != "certified" for finding in findings):
            raise ValueError(
                "Surface chart witness is not a complete exact source partition."
            )
        _chart_point_tables(domain_source, witness, mesh, geometry)
        return
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.topology_id != mesh.topology_id
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
    ):
        raise ValueError(
            "Source chart witness is bound to different source, mesh, or geometry operands."
        )
    renewed = SourceBoundaryChartWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=np.asarray(witness.cell_global_ids),
        entity_global_ids=np.asarray(witness.entity_global_ids),
        local_entity_indices=np.asarray(witness.local_entity_indices),
        target_reference_vertices=np.asarray(witness.target_reference_vertices),
        target_reference_rational_tokens=witness.target_reference_rational_tokens,
        target_face_vertex_global_ids=np.asarray(witness.target_face_vertex_global_ids),
        source_chart_ids=witness.source_chart_ids,
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        source_parameters=np.asarray(witness.source_parameters),
        source_parameter_rational_tokens=witness.source_parameter_rational_tokens,
        orientations=np.asarray(witness.orientations),
        implicit_cell_keys=np.asarray(witness.implicit_cell_keys),
        implicit_root_edge_keys=np.asarray(witness.implicit_root_edge_keys),
    )
    if renewed.witness_id != witness.witness_id:
        raise ValueError(
            "Source chart witness identity does not reproduce its current arrays."
        )

    if isinstance(source, MappedReferenceBoundarySource):
        if witness.source_query_id != source.query_id:
            raise ValueError(
                "Mapped root chart witness changes its source query identity."
            )
        if witness.implicit_cell_keys.shape != (
            0,
            4,
        ) or witness.implicit_root_edge_keys.shape != (0, 3, 2, 3):
            raise ValueError(
                "Mapped root chart witness carries foreign implicit ancestry."
            )
        source.validate_chart_witness(witness)
        if witness.chart_dimension == 1:
            if witness.reference_dimension != mesh.topological_dimension:
                raise ValueError(
                    "Mapped root curve witness changes the target reference dimension."
                )
            _curve_target_chart_rows(mesh, witness)
            return
        if witness.chart_dimension == 2:
            polyhedral = np.asarray(witness.target_face_vertex_global_ids).size != 0
            expected_reference_dimension = (
                2
                if polyhedral or mesh.topological_dimension == 2
                else mesh.topological_dimension
            )
            if witness.reference_dimension != expected_reference_dimension:
                raise ValueError(
                    "Mapped root surface witness changes the target reference dimension."
                )
            _, _, target_orientation = _surface_target_chart_rows(
                mesh,
                geometry,
                None,
                witness,
                full_source=False,
            )
            parameters = tuple(
                _exact_triangle2(
                    _fractions_from_chart_tokens(
                        witness.source_parameter_rational_tokens, row
                    )
                )
                for row in range(len(witness.source_chart_ids))
            )
            source_orientation = tuple(
                1 if _area2(triangle) > 0 else -1 for triangle in parameters
            )
            if tuple(
                source_sign * target_sign
                for source_sign, target_sign in zip(
                    source_orientation, target_orientation, strict=True
                )
            ) != _chart_orientation_vector(witness):
                raise ValueError(
                    "Mapped root chart witness changes source/target orientation ancestry."
                )
            return
        raise ValueError(
            "Mapped root fidelity supports only curve and surface chart witnesses."
        )
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if isinstance(source, PiecewiseLinearBoundarySource):
        if (
            witness.chart_dimension != source.entity_dimension
            or source.entity_dimension not in (1, 2)
        ):
            raise ValueError(
                "PWL chart witness changes the selected source stratum dimension."
            )
        _validate_piecewise_linear_chart_witness(mesh, geometry, source, witness)
        return

    from ._curve_boundary import ParametricCurveBoundarySource

    if isinstance(source, ParametricCurveBoundarySource) and witness.chart_dimension == 1:
        from ._source_interpolation import _curve_chart_partition

        if (
            witness.source_query_id != source.query_id
            or witness.reference_dimension != mesh.topological_dimension
            or mesh.ambient_dimension != source.ambient_dimension
            or witness.implicit_cell_keys.shape != (0, 4)
            or witness.implicit_root_edge_keys.shape != (0, 3, 2, 3)
        ):
            raise ValueError(
                "Curve chart witness changes its source query or mesh dimension."
            )
        source.require_current()
        _curve_chart_partition(source, witness)
        _curve_target_chart_rows(mesh, witness)
        return
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    if (
        isinstance(source, ReconstructedTriangleBoundarySource)
        and witness.chart_dimension == 2
    ):
        domain_source = _reconstructed_domain_query(source)
        if witness.source_query_id != domain_source.source_scope_id:
            raise ValueError(
                "Reconstructed chart witness changes its persistent source scope."
            )
        _, charts = _surface_source_chart_rows(domain_source, witness)
        findings = _partition_findings(domain_source, witness)
        if any(finding.status != "certified" for finding in findings):
            raise ValueError(
                "Reconstructed source chart witness is not a complete exact partition."
            )
        _, _, target_orientation = _surface_target_chart_rows(
            mesh, geometry, domain_source, witness
        )
        source_orientation = tuple(1 if _area2(chart) > 0 else -1 for chart in charts)
        if tuple(
            source_sign * target_sign
            for source_sign, target_sign in zip(
                source_orientation, target_orientation, strict=True
            )
        ) != _chart_orientation_vector(witness):
            raise ValueError(
                "Reconstructed source chart witness changes oriented ancestry."
            )
        return
    if witness.chart_dimension == 2:
        try:
            domain_source = _surface_query_root(source)
        except TypeError:
            domain_source = None
        if domain_source is not None:
            polyhedral = np.asarray(witness.target_face_vertex_global_ids).size != 0
            expected_reference_dimension = 2 if polyhedral else mesh.topological_dimension
            if (
                witness.source_query_id != domain_source.source_scope_id
                or witness.reference_dimension != expected_reference_dimension
                or witness.implicit_cell_keys.shape != (0, 4)
                or witness.implicit_root_edge_keys.shape != (0, 3, 2, 3)
            ):
                raise ValueError("Surface chart witness changes its domain source scope.")
            if isinstance(source, SurfaceNativeRestrictionBoundarySource):
                source.require_current()
            else:
                domain_source.require_current()
            patch_rows, charts = _surface_source_chart_rows(domain_source, witness)
            findings = _partition_findings(domain_source, witness)
            if any(finding.status != "certified" for finding in findings):
                raise ValueError(
                    "Surface source chart witness is not a complete exact partition."
                )
            _, _, target_orientation = _surface_target_chart_rows(
                mesh, geometry, domain_source, witness
            )
            source_orientation = tuple(1 if _area2(chart) > 0 else -1 for chart in charts)
            if tuple(
                source_sign * target_sign
                for source_sign, target_sign in zip(
                    source_orientation, target_orientation, strict=True
                )
            ) != _chart_orientation_vector(witness):
                raise ValueError(
                    "Surface source chart witness changes target/source chart orientation."
                )
            coverage = domain_source.boundary_chart_cover()
            if (
                not coverage.complete
                or coverage.semantics != "certified"
                or coverage.findings
            ):
                raise ValueError(
                    "Surface source query does not reproduce a complete exact chart cover."
                )
            return
    from .implicit._adaptive_discovery import AdaptiveImplicitBoundarySource

    if witness.chart_dimension == 2 and (
        isinstance(source, AdaptiveImplicitBoundarySource)
        or isinstance(source, _AdaptiveImplicitBoundaryCarrier)
    ):
        _validate_implicit_source_witness(mesh, geometry, source, witness)
        return
    raise TypeError("No source fidelity validator supports this witness/source pair.")


__all__ = [
    "SourceFidelityCertificate",
    "SourceFidelityWitness",
    "certify_source_fidelity",
    "polyhedral_face_triangle_vertex_ids",
    "validate_source_fidelity_witness",
    "validate_source_entity_scope",
]


def _chart_point_tables(
    source: MeshingDomainBoundarySource,
    witness: SurfaceChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[
    dict[int, tuple[int, int, tuple[int, Triangle2]]],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[str, ...], ...],
]:

    domain = source.domain
    witness_ids = _integer_vector(witness.cell_global_ids)
    witness_patches = _integer_vector(witness.patches)
    witness_charts = tuple(
        _exact_chart_from_tokens(witness.chart_rational_tokens, row)
        for row in range(len(witness_ids))
    )
    if (
        witness.domain_id != domain.domain_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.topology_id != mesh.topology_id
    ):
        raise ValueError(
            "Surface chart witness is bound to a different domain, map, or topology."
        )
    mesh_cells: dict[int, tuple[int, int, CellBlock]] = {}
    for block_index, block in enumerate(mesh.blocks):
        if not isinstance(block, CellBlock) or block.cell_kind != "triangle":
            raise ValueError(
                "Continuous surface chart fidelity currently requires triangle target cells."
            )
        block_ids = _integer_vector(block.global_ids)
        for row in range(len(block_ids)):
            cell_id = block_ids[row]
            if cell_id in mesh_cells:
                raise ValueError("Target mesh repeats a surface cell identity.")
            mesh_cells[cell_id] = (block_index, row, block)
    if tuple(sorted(mesh_cells)) != witness_ids:
        raise ValueError(
            "Surface chart witness must bind every target cell global ID exactly once."
        )
    if len(witness_patches) != len(witness_ids):
        raise ValueError(
            "Surface chart patch rows must align with target cell identities."
        )
    patch_ids = np.empty((len(witness_ids),), dtype=np.int64)
    occurrence_by_id: dict[int, tuple[str, ...]] = {}
    for witness_row, (cell_id, patch) in enumerate(
        zip(witness_ids, witness_patches, strict=True)
    ):
        if patch < 0 or patch >= len(domain.patches):
            raise ValueError("Surface chart witness names an absent source patch.")
        expected_entity = domain.entity_id(2, patch)
        if witness.geometry_entity_ids[witness_row] != expected_entity:
            raise ValueError(
                "Surface chart witness changes the authored source-face identity."
            )
        expected_path = domain.source_occurrences[2][patch]
        if witness.occurrence_paths[witness_row] != expected_path:
            raise ValueError(
                "Surface chart witness changes the source-face occurrence path."
            )
        patch_ids[witness_row] = patch
        occurrence_by_id[cell_id] = expected_path
    rows = tuple(mesh_cells)
    by_id: dict[int, tuple[int, Triangle2]] = {
        cell_id: (witness_patches[row], witness_charts[row])
        for row, cell_id in enumerate(witness_ids)
    }
    output: dict[int, tuple[int, int, tuple[int, Triangle2]]] = {
        cell_id: (mesh_cells[cell_id][0], mesh_cells[cell_id][1], by_id[cell_id])
        for cell_id in rows
    }
    return (
        output,
        tuple(rows),
        tuple(int(by_id[cell_id][0]) for cell_id in rows),
        tuple(occurrence_by_id[cell_id] for cell_id in rows),
    )


def validate_source_entity_scope(
    query: SourceBoundaryQuery,
    feature_source: _SourceEntityScopes,
    dimension: int,
    entity_ids: ArrayLike,
    /,
) -> None:
    """Require a query to reproduce every authored entity/chart in a feature scope."""
    from ..meshing._scope import _SourceEntityScopes

    if not isinstance(query, SourceBoundaryQuery):
        raise TypeError("query must implement SourceBoundaryQuery.")
    if not isinstance(feature_source, _SourceEntityScopes):
        raise TypeError("feature_source must implement canonical source entity scopes.")
    if (query.source_id, query.source_revision) != (
        feature_source.source_id,
        feature_source.source_revision,
    ):
        raise ValueError(
            "Source query and authored feature scope have different identities."
        )
    requested = np.asarray(entity_ids)
    if requested.ndim != 1 or requested.dtype.kind not in "iu" or requested.size == 0:
        raise TypeError(
            "entity_ids must be a nonempty integer vector from the authored source scope."
        )
    requested = np.asarray(requested, dtype=np.int64)
    if np.any(requested < 0) or np.unique(requested).size != requested.size:
        raise ValueError("entity_ids must be unique nonnegative source-scope IDs.")
    available = np.asarray(feature_source.scope_entity_ids(dimension), dtype=np.int64)
    if not set(requested.tolist()) <= set(available.tolist()):
        raise ValueError(
            "Requested feature entities are absent from the authored source scope."
        )
    keys = tuple(
        feature_source.source_entity_key_for_scope(dimension, int(identifier))
        for identifier in requested
    )
    from ._curve_boundary import ParametricCurveBoundarySource

    if isinstance(query, ParametricCurveBoundarySource):
        if dimension == 1:
            query.validate_scope(
                feature_source,
                requested,
                tuple(path for _, path in keys),
            )
        elif dimension == 0:
            if not isinstance(feature_source, _CurvePointScopeOwner):
                raise TypeError(
                    "Curve feature source cannot validate authored endpoint bounds."
                )
            feature_source.validate_point_boundary_query(query, requested)
            query.require_current()
        else:
            raise ValueError(
                "Parametric curve source queries validate only endpoint or curve scopes."
            )
        return
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    if isinstance(query, ReconstructedTriangleBoundarySource):
        domain_query = _reconstructed_domain_query(query)
        domain = domain_query.domain
        if dimension == 2:
            requested_patches = {
                domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
                for entity_id, occurrence in keys
            }
            if not requested_patches <= set(domain_query.patches):
                raise ValueError(
                    "Reconstructed source query omits a requested authored face."
                )
        elif dimension == 0:
            requested_corners = {
                domain.resolve_entity_id(0, entity_id, occurrence_path=occurrence)
                for entity_id, occurrence in keys
            }
            query_corners = _query_source_corner_rows(domain_query)
            if not requested_corners <= query_corners:
                raise ValueError(
                    "Reconstructed source query omits a requested authored source point."
                )
        else:
            raise ValueError(
                "Reconstructed source queries validate only point or face scopes."
            )
        return
    from ._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if isinstance(query, MappedReferenceBoundarySource):
        if dimension not in (0, 1, 2):
            raise ValueError(
                "Mapped root source queries validate only point, curve, and surface feature scopes."
            )
        query.require_current()
        requested_keys = set(keys)
        if dimension == 0:
            available_ids = np.asarray(
                query.authority.scope_entity_ids(0), dtype=np.int64
            )
            available_keys = {
                query.authority.source_entity_key_for_scope(0, int(identifier))
                for identifier in available_ids
            }
            if not requested_keys <= available_keys:
                raise ValueError(
                    "Mapped root query omits a requested authored source point."
                )
        else:
            actual_keys = set(query.selected_source_keys(dimension))
            if actual_keys != requested_keys:
                raise ValueError(
                    "Mapped root query omits or adds authored feature chart keys."
                )
        return
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if isinstance(query, PiecewiseLinearBoundarySource):
        if dimension != query.entity_dimension:
            raise ValueError(
                "PWL query dimension differs from the requested feature stratum."
            )
        if (query.source_id, query.source_revision) != (
            feature_source.source_id,
            feature_source.source_revision,
        ):
            raise ValueError(
                "PWL boundary query does not bind the requested source revision."
            )
        query.require_current()
        if set(query.selected_source_keys(dimension)) != set(keys):
            raise ValueError(
                "PWL boundary query omits or adds authored feature entity keys."
            )
        return

    if isinstance(
        query, (MeshingDomainBoundarySource, SurfaceNativeRestrictionBoundarySource)
    ):
        domain_query = _surface_query_root(query)
        domain = domain_query.domain
        if domain.source_id != feature_source.source_id or (
            domain.source_revision != feature_source.source_revision
        ):
            raise ValueError(
                "Meshing-domain query does not bind the requested source stratum."
            )
        if dimension == 2:
            expected_patches = {
                domain.resolve_entity_id(2, entity_id, occurrence_path=occurrence)
                for entity_id, occurrence in keys
            }
            if set(domain_query.patches) != expected_patches:
                raise ValueError(
                    "Meshing-domain query omits or adds requested authored faces."
                )
        elif dimension == 0:
            requested_corners = {
                domain.resolve_entity_id(0, entity_id, occurrence_path=occurrence)
                for entity_id, occurrence in keys
            }
            query_corners = _query_source_corner_rows(domain_query)
            if not requested_corners <= query_corners:
                raise ValueError(
                    "Meshing-domain query omits a requested authored source point."
                )
        else:
            raise ValueError(
                "Meshing-domain queries validate only point or face feature scopes."
            )
        query.require_current()
        return
    if isinstance(query, _AdaptiveImplicitBoundaryCarrier):
        if dimension != 2:
            raise ValueError(
                "Adaptive implicit source queries validate only their zero-set surface scope."
            )
        query.require_current()
        surface_key = source_entity_id(
            query.source_id, query.source_revision, "implicit-surface", "zero-set"
        )
        if set(keys) != {(surface_key, ())}:
            raise ValueError(
                "Adaptive implicit query scope must name exactly its zero-set entity."
            )
        return
    raise TypeError("No source-scope validator supports this query authority.")


def _cell_global_rows(
    mesh: CellMesh, /
) -> dict[int, tuple[int, int, CellBlock | PolyhedralBlock]]:
    return {
        int(identifier): (block_index, row, block)
        for block_index, block in enumerate(mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }


@runtime_checkable
class _AdaptiveImplicitBoundaryCarrier(SourceBoundaryProjectionQuery, Protocol):
    @property
    def adaptive_source(self) -> AdaptiveImplicitBoundarySource: ...

    def require_current(self, /) -> None: ...


@runtime_checkable
class _CurvePointScopeOwner(Protocol):
    def validate_point_boundary_query(
        self,
        query: SourceBoundaryQuery,
        requested_point_ids: ArrayLike,
        /,
    ) -> None: ...


def _implicit_chart_id(
    source_id: str,
    source_revision: str,
    cell_key: tuple[int, int, int, int],
    root_edges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-scaffold-triangle",
            "source": (source_id, source_revision),
            "cell_key": cell_key,
            "root_edges": root_edges,
        }
    )


def _validate_implicit_source_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
) -> None:
    from .implicit._adaptive_discovery import AdaptiveImplicitBoundarySource

    if isinstance(source, AdaptiveImplicitBoundarySource):
        adaptive = source
    elif isinstance(source, _AdaptiveImplicitBoundaryCarrier):
        source.require_current()
        adaptive = source.adaptive_source
        if not isinstance(adaptive, AdaptiveImplicitBoundarySource):
            raise TypeError(
                "adaptive_source must retain AdaptiveImplicitBoundarySource authority."
            )
    else:
        raise TypeError(
            "Implicit chart witness requires an adaptive implicit boundary source."
        )
    adaptive.require_current()
    surface = adaptive.surface
    scaffold = surface.scaffold
    if (
        mesh.topological_dimension not in (2, 3)
        or mesh.ambient_dimension != 3
        or witness.chart_dimension != 2
        or witness.reference_dimension != mesh.topological_dimension
        or witness.source_query_id != adaptive.source_revision
    ):
        raise ValueError(
            "Implicit chart witness changes its source query or target dimension."
        )
    if (source.source_id, source.source_revision) != (
        witness.source_id,
        witness.source_revision,
    ) or adaptive.source_id != source.source_id:
        raise ValueError(
            "Implicit source chart keys are bound to a different source revision."
        )
    if (
        surface.cover is None
        or not surface.cover.certified
        or not surface.cover.complete
        or surface.evidence.unresolved_count != 0
        or np.any(scaffold.cell_issues != 0)
        or np.any(scaffold.root_issues != 0)
    ):
        raise ValueError(
            "Implicit source has no complete current certified scaffold/root cover."
        )
    root_lower = np.asarray(scaffold.root_endpoint_value_lower, dtype=np.float64)
    root_upper = np.asarray(scaffold.root_endpoint_value_upper, dtype=np.float64)
    crosses = ((root_upper[:, 0] < 0.0) & (root_lower[:, 1] > 0.0)) | (
        (root_lower[:, 0] > 0.0) & (root_upper[:, 1] < 0.0)
    )
    if not np.all(crosses):
        raise ValueError(
            "Implicit scaffold contains a root bracket without strict opposite endpoint signs."
        )
    gradient_lower = np.asarray(scaffold.cell_gradient_lower, dtype=np.float64)
    gradient_upper = np.asarray(scaffold.cell_gradient_upper, dtype=np.float64)
    triangle_cell_rows = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    if (
        gradient_lower.ndim != 2
        or gradient_lower.shape != gradient_upper.shape
        or triangle_cell_rows.ndim != 1
        or np.any(triangle_cell_rows < 0)
        or np.any(triangle_cell_rows >= gradient_lower.shape[0])
    ):
        raise ValueError(
            "Implicit scaffold gradient and triangle-cell banks are malformed."
        )
    if triangle_cell_rows.size:
        selected_gradient_lower = gradient_lower[triangle_cell_rows]
        selected_gradient_upper = gradient_upper[triangle_cell_rows]
        if not np.all(
            np.any(
                (selected_gradient_lower > 0.0) | (selected_gradient_upper < 0.0),
                axis=1,
            )
        ):
            raise ValueError(
                "Implicit root chart cell lacks a strict regular-gradient enclosure."
            )
    raw_cell_keys = np.asarray(scaffold.cell_indices, dtype=np.int64)
    if (
        raw_cell_keys.ndim != 2
        or raw_cell_keys.shape[1] != 4
        or np.any(triangle_cell_rows >= raw_cell_keys.shape[0])
    ):
        raise ValueError(
            "Implicit scaffold root-cell keys must align with four-dimensional grid cells."
        )
    cell_keys: tuple[_RootCellKey, ...] = tuple(
        (
            int(raw_cell_keys[row, 0]),
            int(raw_cell_keys[row, 1]),
            int(raw_cell_keys[row, 2]),
            int(raw_cell_keys[row, 3]),
        )
        for row in range(raw_cell_keys.shape[0])
    )
    triangle_keys: dict[_ImplicitTriangleKey, list[int]] = {}
    triangle_cells = triangle_cell_rows
    triangle_edges = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)
    root_edge_keys = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    if (
        triangle_cells.ndim != 1
        or triangle_edges.shape != (triangle_cells.size, 3)
        or root_edge_keys.ndim != 3
        or root_edge_keys.shape[1:] != (2, 3)
        or np.any(triangle_edges < 0)
        or np.any(triangle_edges >= root_edge_keys.shape[0])
    ):
        raise ValueError(
            "Implicit scaffold triangle/root-edge ancestry banks are malformed."
        )
    for triangle_row in range(triangle_cells.size):
        cell_row = int(triangle_cells[triangle_row])
        edge_rows = triangle_edges[triangle_row]
        edge_keys: tuple[_RootEdgeKey, _RootEdgeKey, _RootEdgeKey] = (
            _root_edge_key(root_edge_keys[int(edge_rows[0])]),
            _root_edge_key(root_edge_keys[int(edge_rows[1])]),
            _root_edge_key(root_edge_keys[int(edge_rows[2])]),
        )
        key: _ImplicitTriangleKey = (cell_keys[int(cell_row)], edge_keys)
        triangle_keys.setdefault(key, []).append(triangle_row)
    witness_cell_keys = np.asarray(witness.implicit_cell_keys, dtype=np.int64)
    witness_root_edges = np.asarray(witness.implicit_root_edge_keys, dtype=np.int64)
    witness_parameters = tuple(
        _fractions_from_chart_tokens(witness.source_parameter_rational_tokens, row)
        for row in range(len(witness.source_chart_ids))
    )
    standard_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    implicit_scope_key = source_entity_id(
        source.source_id,
        source.source_revision,
        "implicit-surface",
        "zero-set",
    )
    if (
        witness_cell_keys.shape != (len(witness.source_chart_ids), 4)
        or witness_root_edges.shape != (len(witness.source_chart_ids), 3, 2, 3)
        or witness.source_entity_ids
        != (implicit_scope_key,) * len(witness.source_chart_ids)
        or witness.occurrence_paths != tuple(() for _ in witness.source_chart_ids)
        or witness_parameters != tuple(standard_chart for _ in witness.source_chart_ids)
    ):
        raise ValueError(
            "Implicit chart witness omits or changes retained root-cell/piece ancestry."
        )
    seen: set[_ImplicitTriangleKey] = set()
    for row in range(witness_cell_keys.shape[0]):
        cell_key_raw = witness_cell_keys[row]
        edge_array = witness_root_edges[row]
        cell_key: _RootCellKey = (
            int(cell_key_raw[0]),
            int(cell_key_raw[1]),
            int(cell_key_raw[2]),
            int(cell_key_raw[3]),
        )
        edge_keys: tuple[_RootEdgeKey, _RootEdgeKey, _RootEdgeKey] = (
            _root_edge_key(edge_array[0]),
            _root_edge_key(edge_array[1]),
            _root_edge_key(edge_array[2]),
        )
        key: _ImplicitTriangleKey = (cell_key, edge_keys)
        candidates = triangle_keys.get(key, ())
        if len(candidates) != 1 or key in seen:
            raise ValueError(
                "Implicit chart witness changes or duplicates an exact scaffold triangle."
            )
        seen.add(key)
        if witness.source_chart_ids[row] != _implicit_chart_id(
            adaptive.source_id, adaptive.source_revision, cell_key, edge_keys
        ):
            raise ValueError(
                "Implicit source chart ID does not bind its root-triangle keys."
            )
    if seen != set(triangle_keys):
        raise ValueError(
            "Implicit chart witness omits possible source scaffold triangles."
        )
    target_entities = np.asarray(witness.entity_global_ids, dtype=np.int64)
    if np.unique(target_entities).size != target_entities.size:
        raise ValueError("Implicit source chart rows must bind distinct target entities.")
    _, _, target_orientation = _surface_target_chart_rows(
        mesh, geometry, None, witness, full_source=False
    )
    if _chart_orientation_vector(witness) != target_orientation:
        raise ValueError(
            "Implicit root chart changes the oriented target reference triangle."
        )


def _curve_target_chart_rows(
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[tuple[int, int], ...]:
    from ..discretization._cell_complex import (
        PolygonalConnectivity,
        PolyhedralConnectivity,
        TetrahedralConnectivity,
    )
    from ..discretization._cell_mesh import SimplicialConnectivity
    from ..discretization._hexahedral import HexahedralConnectivity

    cells = _cell_global_rows(mesh)
    parent_id_values = np.asarray(witness.cell_global_ids, dtype=np.int64)
    target_id_values = np.asarray(witness.entity_global_ids, dtype=np.int64)
    local_entity_values = np.asarray(witness.local_entity_indices, dtype=np.int64)
    if (
        parent_id_values.ndim != 1
        or target_id_values.shape != parent_id_values.shape
        or local_entity_values.shape != parent_id_values.shape
        or parent_id_values.size != len(witness.source_chart_ids)
    ):
        raise ValueError("Curve target witness row identities are malformed.")
    parent_ids = tuple(int(parent_id_values[row]) for row in range(parent_id_values.size))
    target_ids = tuple(int(target_id_values[row]) for row in range(target_id_values.size))
    local_entities = tuple(
        int(local_entity_values[row]) for row in range(local_entity_values.size)
    )
    references = tuple(
        _fractions_from_chart_tokens(witness.target_reference_rational_tokens, row)
        for row in range(len(witness.source_chart_ids))
    )
    if len(set(zip(target_ids, witness.source_chart_ids, strict=True))) != len(
        target_ids
    ):
        raise ValueError(
            "Curve source chart witness repeats one target-edge/source-chart map."
        )
    if witness.reference_dimension != mesh.topological_dimension:
        raise ValueError(
            "Curve target chart references must use parent-cell coordinates."
        )
    edge_lookup: dict[tuple[int, int], int] = {}
    if mesh.topological_dimension > 1:
        connectivity = mesh.connectivity
        if isinstance(connectivity, SimplicialConnectivity):
            edge_vertices = np.asarray(connectivity.entities[1], dtype=np.int64)
        elif isinstance(
            connectivity,
            (
                PolygonalConnectivity,
                PolyhedralConnectivity,
                TetrahedralConnectivity,
                HexahedralConnectivity,
            ),
        ):
            edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
        else:
            raise TypeError("Cell mesh connectivity has no certified local edge routes.")
        edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        if edge_vertices.shape != (edge_ids.size, 2):
            raise ValueError(
                "Cell mesh edge routes do not align with the entity identity bank."
            )
        for edge_row in range(edge_vertices.shape[0]):
            first_vertex = int(vertex_ids[int(edge_vertices[edge_row, 0])])
            second_vertex = int(vertex_ids[int(edge_vertices[edge_row, 1])])
            edge_lookup[
                (min(first_vertex, second_vertex), max(first_vertex, second_vertex))
            ] = int(edge_ids[edge_row])
    output: list[tuple[int, int]] = []
    for row in range(len(parent_ids)):
        parent_id = parent_ids[row]
        entity_id = target_ids[row]
        local_entity = local_entities[row]
        found = cells.get(parent_id)
        if found is None:
            raise ValueError(
                "Curve source chart witness names an absent parent mesh cell."
            )
        block_index, cell_row, block = found
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Curve chart ancestry requires standard reference-cell blocks."
            )
        target_reference = references[row]
        if mesh.topological_dimension == 1:
            if (
                local_entity != -1
                or entity_id != parent_id
                or block.cell_kind != "interval"
            ):
                raise ValueError(
                    "One-dimensional curve charts must bind interval target cells."
                )
            interval_vertices = np.asarray(
                reference_cell_topology("interval").vertices, dtype=np.float64
            )
            expected = (
                (Fraction.from_float(float(interval_vertices[0, 0])),),
                (Fraction.from_float(float(interval_vertices[1, 0])),),
            )
        else:
            if local_entity < 0:
                raise ValueError(
                    "A curve chart on a higher-dimensional mesh requires a local edge route."
                )
            topology = reference_cell_topology(block.cell_kind)
            edges = topology.entities[1]
            if local_entity >= len(edges):
                raise ValueError(
                    "Curve chart witness names an absent local parent-cell edge."
                )
            edge = edges[int(local_entity)]
            reference_vertices_array = np.asarray(topology.vertices, dtype=np.float64)
            reference_vertices: tuple[tuple[Fraction, ...], ...] = tuple(
                tuple(
                    Fraction.from_float(float(reference_vertices_array[vertex, axis]))
                    for axis in range(reference_vertices_array.shape[1])
                )
                for vertex in range(reference_vertices_array.shape[0])
            )
            expected = tuple(reference_vertices[int(index)] for index in edge)
            cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
            first_vertex = int(mesh.vertex_global_ids[int(cell_vertices[int(edge[0])])])
            second_vertex = int(mesh.vertex_global_ids[int(cell_vertices[int(edge[1])])])
            global_edge = (
                min(first_vertex, second_vertex),
                max(first_vertex, second_vertex),
            )
            if edge_lookup.get(global_edge) != entity_id:
                raise ValueError(
                    "Curve chart witness changes its actual mesh-edge ancestry."
                )
        if set(target_reference) != set(expected):
            raise ValueError(
                "Curve chart reference endpoints do not cover the complete target edge."
            )
        target_orientation = 1 if target_reference == expected else -1
        parameter = _fractions_from_chart_tokens(
            witness.source_parameter_rational_tokens, row
        )
        source_orientation = 1 if parameter[1][0] > parameter[0][0] else -1
        if int(witness.orientations[row]) != source_orientation * target_orientation:
            raise ValueError(
                "Curve chart orientation contradicts its exact target/source edge map."
            )
        output.append((block_index, cell_row))
    return tuple(output)


def _target_vertex_exact_coordinates(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    vertex_global_ids: ArrayLike,
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    requested_vertex_ids = np.asarray(vertex_global_ids, dtype=np.int64)
    if requested_vertex_ids.ndim != 1:
        raise ValueError("Target vertex IDs must form a one-dimensional global-ID bank.")
    if mesh.blocks and all(isinstance(block, PolyhedralBlock) for block in mesh.blocks):
        exact_vertices = _polyhedral_vertex_source_coordinates(mesh, geometry)
        output: list[tuple[ExactScalar, ...]] = []
        for row in range(requested_vertex_ids.size):
            identifier = int(requested_vertex_ids[row])
            point = exact_vertices.get(identifier)
            if point is None:
                raise ValueError(
                    "Source witness names an absent target polyhedral vertex."
                )
            output.append(point)
        return tuple(output)
    vertex_ids = _integer_vector(mesh.vertex_global_ids)
    vertex_rows = {identifier: row for row, identifier in enumerate(vertex_ids)}
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    mesh_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    from ..discretization._cell_geometry import CellVertexGeometryElement

    # One sorted incidence index per block: each requested vertex visits only
    # its incident (cell, position) slots instead of scanning every cell.
    block_incidences: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for block in mesh.blocks:
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        flat = connectivity.reshape(-1)
        order = np.argsort(flat, kind="stable")
        block_incidences.append((connectivity, flat[order], order))
    target_points: list[tuple[ExactScalar, ...]] = []
    for request_row in range(requested_vertex_ids.size):
        vertex_id = requested_vertex_ids[request_row]
        local_vertex = vertex_rows.get(int(vertex_id))
        if local_vertex is None:
            raise ValueError("Source witness names an absent target mesh vertex.")
        incident: list[tuple[ExactScalar, ...]] = []
        for block_index, block in enumerate(mesh.blocks):
            connectivity, sorted_vertices, order = block_incidences[block_index]
            first = int(np.searchsorted(sorted_vertices, local_vertex, side="left"))
            last = int(np.searchsorted(sorted_vertices, local_vertex, side="right"))
            if first == last:
                continue
            slots = tuple(
                divmod(int(slot), connectivity.shape[1]) for slot in order[first:last]
            )
            if isinstance(block, PolyhedralBlock):
                element = elements[block_index]
                if not isinstance(element, CellVertexGeometryElement):
                    raise TypeError(
                        "Polyhedral target vertices require vertex-coordinate maps."
                    )
                route = np.asarray(routes[block_index], dtype=np.int64)
                if route.shape != connectivity.shape:
                    raise ValueError(
                        "Polyhedral target vertex routes must align with block vertices."
                    )
                for cell_row, position in slots:
                    coefficient_row = int(route[cell_row, position])
                    if coefficient_row < 0 or coefficient_row >= len(coefficients):
                        raise ValueError(
                            "Polyhedral target vertex route names no source coordinate."
                        )
                    exact_point = tuple(coefficients[coefficient_row])
                    if len(exact_point) != mesh.ambient_dimension:
                        raise ValueError(
                            "Polyhedral target vertex map has the wrong ambient dimension."
                        )
                    if any(
                        not (
                            coefficient_endpoint(value, upper=False)
                            <= float(mesh_coordinates[local_vertex, axis])
                            <= coefficient_endpoint(value, upper=True)
                        )
                        for axis, value in enumerate(exact_point)
                    ):
                        raise ValueError(
                            "Polyhedral target vertex differs from its exact coordinate route."
                        )
                    incident.append(exact_point)
                continue
            if not isinstance(block, CellBlock):
                raise TypeError(
                    "Target vertex maps require standard or polyhedral cell blocks."
                )
            topology = reference_cell_topology(block.cell_kind)
            route = np.asarray(routes[block_index], dtype=np.int64)
            reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
            for cell_row, position in slots:
                reference_vertex = tuple(
                    Fraction(float(reference_vertices[position, axis]))
                    for axis in range(reference_vertices.shape[1])
                )
                local_coordinates = tuple(
                    coefficients[int(route[cell_row, route_index])]
                    for route_index in range(route.shape[1])
                )
                element = elements[block_index]
                if not isinstance(element, ScalarCoordinateElement):
                    raise TypeError(
                        "Target vertex maps require a scalar coordinate element."
                    )
                expressions = coordinate_map_expressions(
                    element, local_coordinates, cell_row=cell_row
                )
                if len(expressions) != mesh.ambient_dimension:
                    raise ValueError(
                        "Target coordinate map dimension differs from its witness source."
                    )
                incident.append(
                    tuple(
                        evaluate_expression(expression, reference_vertex)
                        for expression in expressions
                    )
                )
        if not incident:
            raise ValueError(
                "Source witness target vertex has no incident mapped mesh cell."
            )
        target = incident[0]
        if any(candidate != target for candidate in incident[1:]):
            raise ValueError(
                "Incident CellGeometrySpec maps do not define one exact target vertex point."
            )
        target_points.append(target)
    return tuple(target_points)


def _source_point_correspondences(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryPointWitness,
    /,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]], ...],
    tuple[tuple[ExactScalar, ...], ...],
]:
    from ._curve_boundary import ParametricCurveBoundarySource
    from ._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    domain_query: MeshingDomainBoundarySource | None = None
    mapped_query: MappedReferenceBoundarySource | None = None
    pwl_query: PiecewiseLinearBoundarySource | None = None
    curve_query: ParametricCurveBoundarySource | None = None
    if isinstance(source, MeshingDomainBoundarySource):
        domain_query = source
        query_id = source.source_scope_id
        source.require_current()
    elif isinstance(source, SurfaceNativeRestrictionBoundarySource):
        domain_query = source.root
        query_id = source.source_scope_id
        source.require_current()
    elif isinstance(source, ReconstructedTriangleBoundarySource):
        domain_query = _reconstructed_domain_query(source)
        query_id = domain_query.source_scope_id
        domain_query.require_current()
    elif isinstance(source, ParametricCurveBoundarySource):
        curve_query = source
        query_id = source.query_id
        source.require_current()
    elif isinstance(source, MappedReferenceBoundarySource):
        mapped_query = source
        query_id = source.query_id
        source.require_current()
    elif isinstance(source, PiecewiseLinearBoundarySource):
        if source.entity_dimension != 0:
            raise TypeError(
                "PWL point witnesses require a zero-dimensional boundary query."
            )
        pwl_query = source
        query_id = source.query_id
        source.require_current()
    else:
        raise TypeError(
            "Source point witnesses require retained domain, reconstructed, surface restriction, parametric-curve, mapped-root, or PWL point authority."
        )
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.topology_id != mesh.topology_id
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != query_id
    ):
        raise ValueError(
            "Source point witness is bound to different source, query, mesh, or geometry operands."
        )
    renewed = SourceBoundaryPointWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        vertex_global_ids=np.asarray(witness.vertex_global_ids),
        source_entity_ids=witness.source_entity_ids,
        occurrence_paths=witness.occurrence_paths,
    )
    if renewed.witness_id != witness.witness_id:
        raise ValueError(
            "Source point witness identity does not reproduce its current arrays."
        )

    source_rows: list[int] = []
    source_points: list[tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]]] = []
    if domain_query is not None:
        domain = domain_query.domain
        covered_corners = _query_source_corner_rows(domain_query)
        for entity_id, occurrence in zip(
            witness.source_entity_ids, witness.occurrence_paths, strict=True
        ):
            source_row = domain.resolve_entity_id(
                0, entity_id, occurrence_path=occurrence
            )
            if source_row not in covered_corners:
                raise ValueError(
                    "Source point witness names a corner outside its retained source query scope."
                )
            source_rows.append(source_row)
            source_points.append(domain.source_corner_point_bounds(source_row))
    elif pwl_query is not None:
        witness_keys = tuple(
            zip(witness.source_entity_ids, witness.occurrence_paths, strict=True)
        )
        query_keys = pwl_query.selected_source_keys(0)
        if len(witness_keys) != len(query_keys) or set(witness_keys) != set(query_keys):
            raise ValueError("PWL point witness omits or adds a selected source vertex.")
        for entity_id, occurrence in zip(
            witness.source_entity_ids, witness.occurrence_paths, strict=True
        ):
            source_row = pwl_query.source_row_for_entity_key(0, (entity_id, occurrence))
            point = pwl_query.point_coordinates(entity_id, occurrence)
            source_rows.append(int(pwl_query.source_rows[source_row]))
            source_points.append((point, point))
    elif mapped_query is not None:
        authority = mapped_query.authority
        complex_ = authority.complex
        source_vertex_ids = np.asarray(complex_.vertex_entity_ids, dtype=np.int64)
        source_paths = tuple(tuple(path) for path in complex_.vertex_occurrence_paths)
        root_vertex_ids = np.asarray(
            authority.root_relation.vertex_root_global_ids, dtype=np.int64
        )
        if (
            len(source_paths) != source_vertex_ids.size
            or root_vertex_ids.shape != source_vertex_ids.shape
        ):
            raise ValueError(
                "Mapped PLC point identities and root ancestry banks are misaligned."
            )
        source_keys = tuple(
            authority.source_entity_key_for_scope(0, int(identifier))
            for identifier in source_vertex_ids
        )
        if any(key[1] != source_paths[row] for row, key in enumerate(source_keys)):
            raise ValueError(
                "Mapped PLC point keys change their authored occurrence paths."
            )
        requested_keys = tuple(
            zip(witness.source_entity_ids, witness.occurrence_paths, strict=True)
        )
        if len(set(requested_keys)) != len(requested_keys):
            raise ValueError("Mapped source point witness repeats an authored point key.")
        mapped_rows: list[int] = []
        for key in requested_keys:
            matches = tuple(
                row for row, source_key in enumerate(source_keys) if source_key == key
            )
            if len(matches) != 1:
                raise ValueError(
                    "Mapped source point key is absent or ambiguous in the retained PLC vertex bank."
                )
            mapped_rows.append(matches[0])
        source_coordinates = _target_vertex_exact_coordinates(
            mapped_query.domain.reference_mesh,
            mapped_query.domain.source_geometry,
            root_vertex_ids[np.asarray(mapped_rows, dtype=np.int64)],
        )
        for source_row, point in zip(mapped_rows, source_coordinates, strict=True):
            source_rows.append(source_row)
            source_points.append((point, point))
    elif curve_query is not None:
        query_keys = tuple(
            zip(
                curve_query.point_entity_ids,
                curve_query.point_occurrence_paths,
                strict=True,
            )
        )
        for entity_id, occurrence in zip(
            witness.source_entity_ids, witness.occurrence_paths, strict=True
        ):
            try:
                source_row = query_keys.index((entity_id, occurrence))
            except ValueError as error:
                raise ValueError(
                    "Source point witness names an endpoint absent from its curve query."
                ) from error
            lower_values, upper_values = curve_query.source_point_bounds(
                entity_id, occurrence
            )
            lower_array = np.asarray(lower_values, dtype=np.float64)
            upper_array = np.asarray(upper_values, dtype=np.float64)
            if (
                lower_array.shape != (curve_query.ambient_dimension,)
                or upper_array.shape != lower_array.shape
                or np.any(~np.isfinite(lower_array))
                or np.any(~np.isfinite(upper_array))
                or np.any(lower_array > upper_array)
            ):
                raise ValueError(
                    "Curve query endpoint has no finite ordered source-point enclosure."
                )
            source_rows.append(source_row)
            source_points.append(
                (
                    tuple(Fraction.from_float(float(value)) for value in lower_array),
                    tuple(Fraction.from_float(float(value)) for value in upper_array),
                )
            )
    else:
        raise TypeError("Source point witness has no resolved source query authority.")
    target_points = _target_vertex_exact_coordinates(
        mesh, geometry, witness.vertex_global_ids
    )
    return tuple(source_rows), tuple(source_points), target_points


def _volume_boundary_facet_lookup(
    mesh: CellMesh, /
) -> dict[tuple[int, int], tuple[int, bool]]:
    from ..discretization._cell_complex import (
        PolyhedralConnectivity,
        TetrahedralConnectivity,
    )
    from ..discretization._cell_mesh import SimplicialConnectivity
    from ..discretization._hexahedral import HexahedralConnectivity

    if mesh.topological_dimension != 3:
        raise ValueError("Volume source charts require a three-dimensional cell mesh.")
    connectivity = mesh.connectivity
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    boundary = np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    if isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, SimplicialConnectivity),
    ):
        if (
            isinstance(connectivity, SimplicialConnectivity)
            and connectivity.dimension != 3
        ):
            raise ValueError("Simplicial volume fidelity requires tetrahedral cells.")
        from ..meshing.providers._native_plc_geometry import volume_cell_entity_rows

        _, local_face_rows = volume_cell_entity_rows(mesh, cell_ids)
        if (
            local_face_rows.ndim != 2
            or local_face_rows.shape[0] != cell_ids.size
            or np.any(local_face_rows < 0)
            or np.any(local_face_rows >= face_ids.size)
        ):
            raise ValueError(
                "Canonical volume cell faces do not bind the target face identity bank."
            )
        return {
            (int(cell_ids[cell_row]), local_face): (
                int(face_ids[int(local_face_rows[cell_row, local_face])]),
                bool(boundary[int(local_face_rows[cell_row, local_face])]),
            )
            for cell_row in range(cell_ids.size)
            for local_face in range(local_face_rows.shape[1])
        }
    if isinstance(connectivity, PolyhedralConnectivity):
        if not np.array_equal(np.asarray(connectivity.cell_global_ids), cell_ids):
            raise ValueError("Polyhedral connectivity changes canonical volume cell IDs.")
        connectivity_face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
        if not np.array_equal(connectivity_face_ids, face_ids):
            raise ValueError(
                "Polyhedral connectivity changes canonical boundary face IDs."
            )
        owners = np.asarray(connectivity.face_owner, dtype=np.int64)
        owner_local = np.asarray(connectivity.face_owner_local, dtype=np.int64)
        neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
        neighbor_local = np.asarray(connectivity.face_neighbor_local, dtype=np.int64)
        routes: dict[tuple[int, int], tuple[int, bool]] = {}
        for face_row, (owner, local) in enumerate(zip(owners, owner_local, strict=True)):
            routes[(int(cell_ids[owner]), int(local))] = (
                int(face_ids[face_row]),
                bool(boundary[face_row]),
            )
            neighbor = int(neighbors[face_row])
            if neighbor >= 0:
                routes[(int(cell_ids[neighbor]), int(neighbor_local[face_row]))] = (
                    int(face_ids[face_row]),
                    bool(boundary[face_row]),
                )
        return routes
    raise TypeError("Cell mesh connectivity has no certified local volume-facet routes.")


def _polyhedral_vertex_source_coordinates(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> dict[int, ExactPoint3]:
    from ..discretization._cell_geometry import CellVertexGeometryElement

    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "Polyhedral target coordinates require a volume mesh in three-space."
        )
    elements, routes, _ = geometry.resolve(mesh)
    if any(not isinstance(element, CellVertexGeometryElement) for element in elements):
        raise TypeError(
            "Polyhedral source coordinates require vertex-defined CellGeometrySpec maps."
        )
    coefficients = geometry.source_coordinates()
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    mesh_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    result: dict[int, ExactPoint3] = {}
    for block_index, block in enumerate(mesh.blocks):
        if not isinstance(block, PolyhedralBlock):
            raise TypeError("Polyhedral source coordinates require polyhedron blocks.")
        cells = np.asarray(block.vertices, dtype=np.int64)
        routes_ = np.asarray(routes[block_index], dtype=np.int64)
        if routes_.shape != cells.shape:
            raise ValueError(
                "Polyhedral CellGeometrySpec routes must map every cell vertex."
            )
        for cell_row in range(cells.shape[0]):
            row = cells[cell_row]
            for local_vertex in range(row.shape[0]):
                vertex_row = int(row[local_vertex])
                coefficient_row = int(routes_[cell_row, local_vertex])
                if coefficient_row < 0 or coefficient_row >= len(coefficients):
                    raise ValueError(
                        "A polyhedral vertex route names no source coordinate."
                    )
                exact_values = tuple(coefficients[coefficient_row])
                if len(exact_values) != 3:
                    raise ValueError(
                        "Polyhedral target vertices must lie in three-space."
                    )
                exact_point: ExactPoint3 = (
                    exact_values[0],
                    exact_values[1],
                    exact_values[2],
                )
                for axis in range(3):
                    value = exact_point[axis]
                    if not (
                        coefficient_endpoint(value, upper=False)
                        <= float(mesh_coordinates[vertex_row, axis])
                        <= coefficient_endpoint(value, upper=True)
                    ):
                        raise ValueError(
                            "A polyhedral mesh vertex is not the represented CellGeometrySpec source-map point."
                        )
                identifier = int(vertex_ids[vertex_row])
                previous = result.setdefault(identifier, exact_point)
                if previous != exact_point:
                    raise ValueError(
                        "Incident polyhedral CellGeometrySpec maps disagree at a shared vertex."
                    )
    return result


def polyhedral_face_triangle_vertex_ids(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    """Return exact-source-validated triangle rows for actual polyhedral faces."""
    from ..discretization._cell_complex import PolyhedralConnectivity
    from ..discretization._polyhedral_locator import _triangulate_face

    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "polyhedron" for block in mesh.blocks)
        or not isinstance(mesh.connectivity, PolyhedralConnectivity)
    ):
        raise TypeError(
            "Polyhedral face triangulations require a three-dimensional PolyhedralConnectivity mesh."
        )
    exact_vertex_coordinates = _polyhedral_vertex_source_coordinates(mesh, geometry)
    _volume_boundary_facet_lookup(mesh)
    connectivity = mesh.connectivity
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    if face_offsets.shape != (face_ids.size + 1,):
        raise ValueError("Polyhedral face loops do not align with target face entities.")

    def sign(value: ExactScalar, /) -> int:
        sign_method = getattr(value, "sign", None)
        if callable(sign_method):
            return int(sign_method())
        return int(value > 0) - int(value < 0)

    def orient2(
        first: ExactPoint2,
        second: ExactPoint2,
        third: ExactPoint2,
        /,
    ) -> ExactScalar:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )

    def cross3(
        first: ExactPoint3,
        second: ExactPoint3,
        /,
    ) -> ExactPoint3:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    def on_segment(
        first: ExactPoint2,
        second: ExactPoint2,
        point: ExactPoint2,
        /,
    ) -> bool:
        return min(first[0], second[0]) <= point[0] <= max(first[0], second[0]) and min(
            first[1], second[1]
        ) <= point[1] <= max(first[1], second[1])

    def intersects(
        first: ExactPoint2,
        second: ExactPoint2,
        third: ExactPoint2,
        fourth: ExactPoint2,
        /,
    ) -> bool:
        first_side = sign(orient2(first, second, third))
        second_side = sign(orient2(first, second, fourth))
        third_side = sign(orient2(third, fourth, first))
        fourth_side = sign(orient2(third, fourth, second))
        if first_side * second_side < 0 and third_side * fourth_side < 0:
            return True
        return (
            (first_side == 0 and on_segment(first, second, third))
            or (second_side == 0 and on_segment(first, second, fourth))
            or (third_side == 0 and on_segment(third, fourth, first))
            or (fourth_side == 0 and on_segment(third, fourth, second))
        )

    output: list[tuple[int, tuple[int, int, int]]] = []
    for face_row, face_id_raw in enumerate(face_ids):
        face_id = int(face_id_raw)
        loop = face_values[face_offsets[face_row] : face_offsets[face_row + 1]]
        if loop.ndim != 1 or loop.size < 3 or np.unique(loop).size != loop.size:
            raise ValueError("Polyhedral faces require unique ordered vertex loops.")
        points = tuple(
            exact_vertex_coordinates[int(vertex_ids[int(vertex)])] for vertex in loop
        )
        origin = points[0]
        normal: tuple[ExactScalar, ExactScalar, ExactScalar] | None = None
        for first in range(1, len(points) - 1):
            for second in range(first + 1, len(points)):
                first_vector: ExactPoint3 = (
                    points[first][0] - origin[0],
                    points[first][1] - origin[1],
                    points[first][2] - origin[2],
                )
                second_vector: ExactPoint3 = (
                    points[second][0] - origin[0],
                    points[second][1] - origin[1],
                    points[second][2] - origin[2],
                )
                candidate = cross3(first_vector, second_vector)
                if any(sign(value) for value in candidate):
                    normal = candidate
                    break
            if normal is not None:
                break
        if normal is None:
            raise ValueError("Polyhedral face source map is degenerate.")
        if any(
            sign(
                sum(
                    (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
                    Fraction(0),
                )
            )
            for point in points
        ):
            raise ValueError(
                "Polyhedral face source-map vertices are not exactly coplanar."
            )
        dropped_axis = next(axis for axis, value in enumerate(normal) if sign(value))
        kept_axes = tuple(axis for axis in range(3) if axis != dropped_axis)
        projected: tuple[ExactPoint2, ...] = tuple(
            (point[kept_axes[0]], point[kept_axes[1]]) for point in points
        )
        polygon_area: ExactScalar = sum(
            (
                projected[index][0] * projected[(index + 1) % len(projected)][1]
                - projected[(index + 1) % len(projected)][0] * projected[index][1]
                for index in range(len(projected))
            ),
            Fraction(0),
        )
        polygon_sign = sign(polygon_area)
        if not polygon_sign:
            raise ValueError("Polyhedral face source-map projection is degenerate.")

        local_triangles = _triangulate_face(coordinates, loop)
        triangle_ids_list: list[tuple[int, int, int]] = []
        for triangle in local_triangles:
            if len(triangle) != 3:
                raise ValueError("Polyhedral face triangulation must contain triangles.")
            triangle_ids_list.append(
                (
                    int(vertex_ids[int(triangle[0])]),
                    int(vertex_ids[int(triangle[1])]),
                    int(vertex_ids[int(triangle[2])]),
                )
            )
        triangle_ids = tuple(triangle_ids_list)
        projected_by_id: dict[int, ExactPoint2] = {
            int(vertex_ids[int(local_vertex)]): projected[index]
            for index, local_vertex in enumerate(loop)
        }
        loop_ids = tuple(int(vertex_ids[int(vertex)]) for vertex in loop)
        edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
        triangle_area: ExactScalar = Fraction(0)
        for triangle in triangle_ids:
            triangle_points: tuple[ExactPoint2, ExactPoint2, ExactPoint2] = (
                projected_by_id[triangle[0]],
                projected_by_id[triangle[1]],
                projected_by_id[triangle[2]],
            )
            area = orient2(*triangle_points)
            if sign(area) != polygon_sign:
                raise ValueError(
                    "Polyhedral face triangulation orientation changes under its exact source map."
                )
            triangle_area += area
            for first, second in zip(
                triangle, (triangle[1], triangle[2], triangle[0]), strict=True
            ):
                edge_key = (min(first, second), max(first, second))
                edge_uses.setdefault(edge_key, []).append((first, second))
        if triangle_area != polygon_area:
            raise ValueError(
                "Polyhedral target triangles do not partition the exact CellGeometrySpec face."
            )
        boundary_edges = {
            tuple(sorted((first, second))): (first, second)
            for first, second in zip(loop_ids, (*loop_ids[1:], loop_ids[0]), strict=True)
        }
        for edge, oriented in edge_uses.items():
            if edge in boundary_edges:
                if oriented != [boundary_edges[edge]]:
                    raise ValueError(
                        "Polyhedral target triangles change the actual face boundary chain."
                    )
            elif len(oriented) != 2 or oriented[0] != oriented[1][::-1]:
                raise ValueError(
                    "Polyhedral target triangulation has an unpaired internal edge."
                )
        segments = tuple(edge_uses)
        for index, first_edge in enumerate(segments):
            for second_edge in segments[index + 1 :]:
                if set(first_edge) & set(second_edge):
                    continue
                if intersects(
                    projected_by_id[first_edge[0]],
                    projected_by_id[first_edge[1]],
                    projected_by_id[second_edge[0]],
                    projected_by_id[second_edge[1]],
                ):
                    raise ValueError(
                        "Polyhedral target triangulation has crossing face edges."
                    )
        output.extend((face_id, triangle) for triangle in triangle_ids)
    return tuple(output)


def _validate_surface_envelope_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SurfaceEnvelopeDistanceWitness,
    /,
) -> None:
    from .._meshcore import surface_envelope
    from .reconstruction._boundary import ReconstructedTriangleBoundarySource

    if not isinstance(source, ReconstructedTriangleBoundarySource):
        raise TypeError(
            "Surface envelope distance evidence requires a reconstructed triangle source."
        )
    source_domain = _reconstructed_domain_query(source)
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.topology_id != mesh.topology_id
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != source_domain.source_scope_id
    ):
        raise ValueError(
            "Surface envelope evidence is bound to different source, query, mesh, or geometry operands."
        )
    renewed = SurfaceEnvelopeDistanceWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        offset=witness.offset,
        spacing=witness.spacing,
        certificate_spacing=witness.certificate_spacing,
        bounds=np.asarray(witness.bounds),
        envelope_vertices=np.asarray(witness.envelope_vertices),
        envelope_triangles=np.asarray(witness.envelope_triangles),
        target_vertex_global_ids=np.asarray(witness.target_vertex_global_ids),
        target_facet_global_ids=np.asarray(witness.target_facet_global_ids),
    )
    if renewed.witness_id != witness.witness_id:
        raise ValueError(
            "Surface envelope witness identity does not reproduce its retained arrays."
        )
    # The envelope is the band-constrained remesh of the kernel's extraction.
    fresh = surface_envelope(
        source._vertices,
        source._faces,
        witness.offset,
        witness.spacing,
        witness.certificate_spacing,
        remesh=True,
    )
    if (
        not np.array_equal(fresh.bounds, witness.bounds)
        or not np.array_equal(fresh.surface_vertices, witness.envelope_vertices)
        or not np.array_equal(fresh.surface_triangles, witness.envelope_triangles)
    ):
        raise ValueError(
            "Surface envelope kernel bounds or PWL output do not reproduce from the retained source."
        )
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "Surface envelope fidelity requires a three-dimensional volume mesh."
        )
    cells = _cell_global_rows(mesh)
    facet_routes = _volume_boundary_facet_lookup(mesh)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    boundary_vertices: dict[int, tuple[int, ...]] = {}
    for (cell_id, local_facet), (facet_id, boundary) in facet_routes.items():
        if not boundary:
            continue
        found = cells.get(cell_id)
        if found is None:
            raise ValueError(
                "Envelope boundary facet ancestry names an absent target cell."
            )
        block_index, cell_row, block = found
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Surface envelope boundary validation requires standard reference cells."
            )
        topology = reference_cell_topology(block.cell_kind)
        facets = topology.entities[2]
        if local_facet < 0 or local_facet >= len(facets):
            raise ValueError(
                "Envelope boundary facet ancestry leaves its target reference cell."
            )
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        face_vertices = tuple(
            int(vertex_ids[int(connectivity[cell_row, int(local_vertex)])])
            for local_vertex in facets[local_facet]
        )
        if facet_id in boundary_vertices:
            raise ValueError("Target exterior facet identity is not unique.")
        boundary_vertices[facet_id] = face_vertices
    target_facet_ids = _integer_vector(witness.target_facet_global_ids)
    target_vertex_ids = _integer_vector(witness.target_vertex_global_ids)
    if set(boundary_vertices) != set(target_facet_ids):
        raise ValueError(
            "Envelope witness does not bind the complete target exterior-facet set."
        )
    if set(vertex for face in boundary_vertices.values() for vertex in face) != set(
        target_vertex_ids
    ):
        raise ValueError(
            "Envelope witness does not bind the complete target exterior-vertex set."
        )
    target_points = _target_vertex_exact_coordinates(
        mesh, geometry, witness.target_vertex_global_ids
    )
    envelope_vertices = np.asarray(witness.envelope_vertices, dtype=np.float64)
    if envelope_vertices.shape != (len(target_points), 3):
        raise ValueError(
            "Envelope witness vertex coordinates do not align with target ancestry."
        )
    for row in range(len(target_points)):
        envelope_point = envelope_vertices[row]
        expected = (
            Fraction.from_float(float(envelope_point[0])),
            Fraction.from_float(float(envelope_point[1])),
            Fraction.from_float(float(envelope_point[2])),
        )
        if target_points[row] != expected:
            raise ValueError(
                "Target CellGeometrySpec boundary vertices differ from the kernel envelope surface."
            )
    triangles = np.asarray(witness.envelope_triangles, dtype=np.int64)
    if triangles.shape != (len(target_facet_ids), 3):
        raise ValueError(
            "Envelope witness triangle bank does not align with target facets."
        )
    for row in range(triangles.shape[0]):
        triangle = triangles[row]
        facet_id = int(target_facet_ids[row])
        expected_vertices = (
            int(target_vertex_ids[int(triangle[0])]),
            int(target_vertex_ids[int(triangle[1])]),
            int(target_vertex_ids[int(triangle[2])]),
        )
        if set(expected_vertices) != set(boundary_vertices[facet_id]):
            raise ValueError(
                "Target boundary facet vertex ancestry differs from its envelope triangle."
            )


def _target_boundary_triangles(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> tuple[tuple[int, ...], tuple[Triangle3, ...]]:
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError(
            "Source distance evidence requires a three-dimensional volume mesh."
        )
    cells = _cell_global_rows(mesh)
    facet_routes = _volume_boundary_facet_lookup(mesh)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    points_by_facet: dict[int, Triangle3] = {}
    for (cell_id, local_facet), (facet_id, boundary) in facet_routes.items():
        if not boundary:
            continue
        found = cells.get(cell_id)
        if found is None:
            raise ValueError("Target exterior facet names an absent volume cell.")
        block_index, cell_row, block = found
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Source distance evidence requires standard reference-cell geometry."
            )
        topology = reference_cell_topology(block.cell_kind)
        face_routes = topology.entities[2]
        if local_facet < 0 or local_facet >= len(face_routes):
            raise ValueError("Target exterior facet leaves its reference volume cell.")
        local_vertices = face_routes[local_facet]
        if len(local_vertices) != 3:
            raise ValueError(
                "Source distance evidence currently requires triangular exterior facets."
            )
        route = np.asarray(routes[block_index][cell_row], dtype=np.int64)
        local = tuple(
            coefficients[int(route[route_index])] for route_index in range(route.shape[0])
        )
        element = elements[block_index]
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Source distance evidence requires a scalar coordinate map element."
            )
        expressions = coordinate_map_expressions(element, local, cell_row=cell_row)
        if len(expressions) != 3 or not all(
            _poly_is_affine(value) for value in expressions
        ):
            raise ValueError(
                "Source distance evidence requires exact affine target facet maps."
            )
        reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
        triangle_points: list[Point3] = []
        for local_vertex in local_vertices:
            arguments = tuple(
                Fraction(float(reference_vertices[int(local_vertex), axis]))
                for axis in range(reference_vertices.shape[1])
            )
            image = tuple(
                evaluate_expression(expression, arguments) for expression in expressions
            )
            if len(image) != 3:
                raise ValueError("Target facet map must take values in three-space.")
            first, second, third = image
            if (
                not isinstance(first, Fraction)
                or not isinstance(second, Fraction)
                or not isinstance(third, Fraction)
            ):
                raise ValueError(
                    "Source distance target triangles must have rational exact coordinates."
                )
            triangle_point: Point3 = (first, second, third)
            triangle_points.append(triangle_point)
        triangle: Triangle3 = (
            triangle_points[0],
            triangle_points[1],
            triangle_points[2],
        )
        if int(facet_id) in points_by_facet:
            raise ValueError("Target exterior face identity appears more than once.")
        points_by_facet[int(facet_id)] = triangle
    if not points_by_facet:
        raise ValueError(
            "Source distance evidence requires a nonempty target exterior surface."
        )
    facet_ids = tuple(sorted(points_by_facet))
    return facet_ids, tuple(points_by_facet[facet_id] for facet_id in facet_ids)


def _triangle_centroid_balls(
    triangles: tuple[Triangle3, ...], /
) -> tuple[list[Point3], list[Fraction]]:
    """Exact centroids and rational upper radii of balls containing each triangle."""
    centers: list[Point3] = []
    radii: list[Fraction] = []
    for triangle in triangles:
        center = (
            sum((point[0] for point in triangle), Fraction(0)) / 3,
            sum((point[1] for point in triangle), Fraction(0)) / 3,
            sum((point[2] for point in triangle), Fraction(0)) / 3,
        )
        centers.append(center)
        radii.append(
            max(
                fraction_sqrt_interval(
                    sum(
                        ((point[axis] - center[axis]) ** 2 for axis in range(3)),
                        Fraction(0),
                    )
                )[1]
                for point in triangle
            )
        )
    return centers, radii


def _nearest_triangle_distance_upper(
    point: Point3,
    point_value: np.ndarray,
    triangles: tuple[Triangle3, ...],
    center_values: np.ndarray,
    radius_values: np.ndarray,
    /,
) -> Fraction:
    """Return ``min_T sqrt_up(d_T^2)`` over every triangle, pruning provable losers.

    Each triangle contains its exact centroid ``c_T`` and lies within its
    rational radius ``r_T`` of it, so ``|p - c_T| - r_T <= d_T <= |p - c_T|``
    and ``m = min_T |p - c_T|`` bounds the minimum distance above. A triangle
    whose float lower bound exceeds the float ``m`` by ``slack`` has
    ``d_T > m + slack / 2``: ``slack`` is a billion times the binary64 error
    of the centroid rounding, the offsets, the radius rounding and the
    nearest-offset, and also of the few-ulp gap between ``sqrt_up`` and the
    exact minimum. A pruned triangle's ``sqrt_up`` therefore exceeds the
    survivors' minimum, which is the minimum over all triangles exactly.
    """
    from ..geometry._domain_boundary import _point_triangle_squared

    offsets = np.linalg.norm(center_values - point_value, axis=1)
    nearest = float(np.min(offsets))
    scale = float(np.max(np.abs(center_values))) + float(np.max(np.abs(point_value)))
    slack = 1.0e-9 * (nearest + float(np.max(radius_values)) + scale)
    candidates = np.flatnonzero(offsets - radius_values <= nearest + slack)
    return min(
        fraction_sqrt_interval(_point_triangle_squared(point, triangles[row]))[1]
        for row in candidates.tolist()
    )


def _analytic_profile_distance_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryDistanceWitness,
    /,
) -> tuple[Fraction, Fraction, Fraction]:
    from .implicit._analytic_profile import AnalyticImplicitProfile

    if not isinstance(source, AnalyticImplicitProfile):
        raise TypeError(
            "A distance-only restricted witness requires AnalyticImplicitProfile authority."
        )
    source.validate_source_integrity()
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.topology_id != mesh.topology_id
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != source.profile_id
    ):
        raise ValueError(
            "Analytic source distance witness is bound to different profile, mesh, or geometry inputs."
        )
    renewed = SourceBoundaryDistanceWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        sample_count=witness.sample_count,
        cover_radius=witness.cover_radius,
        target_facet_global_ids=np.asarray(witness.target_facet_global_ids),
    )
    if renewed.witness_id != witness.witness_id:
        raise ValueError(
            "Analytic source distance witness identity does not reproduce its target extent."
        )
    target_facet_ids, target_triangles = _target_boundary_triangles(mesh, geometry)
    if set(target_facet_ids) != set(_integer_vector(witness.target_facet_global_ids)):
        raise ValueError(
            "Analytic distance witness must bind every target exterior facet exactly once."
        )

    centers, radii = _triangle_centroid_balls(target_triangles)
    center_values = np.asarray(
        [[float(value) for value in center] for center in centers], dtype=np.float64
    )
    if np.any(~np.isfinite(center_values)):
        raise ValueError("Target facet center is outside finite source-query resolution.")
    distances = source.boundary_distance(center_values)
    if (
        distances.semantics != "certified"
        or distances.lower.shape != (len(centers),)
        or distances.upper.shape != distances.lower.shape
        or np.any(~np.isfinite(distances.lower))
        or np.any(~np.isfinite(distances.upper))
        or np.any(distances.lower < 0.0)
        or np.any(distances.upper < distances.lower)
    ):
        raise ValueError(
            "Analytic profile does not provide certified physical distance enclosures."
        )
    mesh_lower = Fraction(0)
    mesh_upper = Fraction(0)
    for row, center in enumerate(centers):
        rounding = fraction_sqrt_interval(
            sum(
                (
                    (Fraction.from_float(float(center_values[row, axis])) - center[axis])
                    ** 2
                    for axis in range(3)
                ),
                Fraction(0),
            )
        )[1]
        distance_lower = Fraction.from_float(float(distances.lower[row]))
        distance_upper = Fraction.from_float(float(distances.upper[row]))
        mesh_lower = max(
            mesh_lower, max(Fraction(0), distance_lower - rounding - radii[row])
        )
        mesh_upper = max(mesh_upper, distance_upper + rounding + radii[row])

    samples = source.boundary_cover(witness.cover_radius)
    sample_points = np.asarray(samples.points, dtype=np.float64)
    covering_radii = np.asarray(samples.covering_radius, dtype=np.float64)
    point_errors = np.asarray(samples.point_error, dtype=np.float64)
    if (
        samples.semantics != "certified"
        or not samples.complete
        or sample_points.shape != (witness.sample_count, 3)
        or sample_points.shape[0] == 0
        or covering_radii.shape != (sample_points.shape[0],)
        or point_errors.shape != covering_radii.shape
        or np.any(~np.isfinite(sample_points))
        or np.any(~np.isfinite(covering_radii))
        or np.any(~np.isfinite(point_errors))
        or np.any(covering_radii < 0.0)
        or np.any(covering_radii > witness.cover_radius)
        or np.any(point_errors < 0.0)
    ):
        raise ValueError(
            "Analytic profile source samples do not reproduce a complete certified boundary cover."
        )
    radius_values = np.asarray([float(radius) for radius in radii], dtype=np.float64)
    reverse_upper = Fraction(0)
    for sample_row in range(sample_points.shape[0]):
        point = (
            Fraction.from_float(float(sample_points[sample_row, 0])),
            Fraction.from_float(float(sample_points[sample_row, 1])),
            Fraction.from_float(float(sample_points[sample_row, 2])),
        )
        cover_radius = covering_radii[sample_row]
        point_error = point_errors[sample_row]
        distance_upper = _nearest_triangle_distance_upper(
            point,
            sample_points[sample_row],
            target_triangles,
            center_values,
            radius_values,
        )
        reverse_upper = max(
            reverse_upper,
            distance_upper
            + Fraction.from_float(float(cover_radius))
            + Fraction.from_float(float(point_error)),
        )
    return mesh_lower, mesh_upper, reverse_upper


def _validate_analytic_profile_distance_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryDistanceWitness,
    /,
) -> None:
    _analytic_profile_distance_evidence(mesh, geometry, source, witness)


def _face_reference_chart(
    cell_kind: str,
    local_facet: int,
    target_vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Triangle2, ...], tuple[Triangle2, ...]]:
    from ..linalg._exact import solve_exact

    topology = reference_cell_topology(cell_kind)
    faces = topology.entities[2]
    if local_facet < 0 or local_facet >= len(faces):
        raise ValueError("A source chart witness names an absent local reference facet.")
    face_indices = faces[local_facet]
    if len(face_indices) not in (3, 4):
        raise ValueError(
            "Only triangular or quadrilateral volume facets have a chart witness."
        )
    reference_vertices_array = np.asarray(topology.vertices, dtype=np.float64)
    reference_vertices: tuple[tuple[Fraction, ...], ...] = tuple(
        tuple(
            Fraction.from_float(float(reference_vertices_array[vertex, axis]))
            for axis in range(reference_vertices_array.shape[1])
        )
        for vertex in range(reference_vertices_array.shape[0])
    )
    face_points = tuple(reference_vertices[int(index)] for index in face_indices)
    origin = face_points[0]
    first = tuple(face_points[1][axis] - origin[axis] for axis in range(3))
    second_point = face_points[2] if len(face_indices) == 3 else face_points[3]
    second = tuple(second_point[axis] - origin[axis] for axis in range(3))
    if len(face_indices) == 4 and any(
        face_points[2][axis] != origin[axis] + first[axis] + second[axis]
        for axis in range(3)
    ):
        raise ValueError(
            "Quadrilateral reference facets must be exact affine parallelograms."
        )
    coordinate_pair = next(
        (
            (left, right)
            for left in range(3)
            for right in range(left + 1, 3)
            if first[left] * second[right] - first[right] * second[left] != 0
        ),
        None,
    )
    if coordinate_pair is None:
        raise ValueError("Reference facet has a singular exact chart basis.")
    left, right = coordinate_pair
    matrix = ((first[left], second[left]), (first[right], second[right]))
    mapped: list[Point2] = []
    for point in target_vertices:
        solution = solve_exact(
            matrix,
            ((point[left] - origin[left],), (point[right] - origin[right],)),
        )
        chart_point = (solution[0][0], solution[1][0])
        if any(
            origin[axis] + chart_point[0] * first[axis] + chart_point[1] * second[axis]
            != point[axis]
            for axis in range(3)
        ):
            raise ValueError(
                "Target reference chart leaves its actual local reference facet."
            )
        mapped.append(chart_point)
    domain: tuple[Triangle2, ...]
    if len(face_indices) == 3:
        domain = (
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            ),
        )
    else:
        domain = (
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(1), Fraction(1)),
            ),
            (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(1)),
            ),
        )
    if len(mapped) != 3:
        raise ValueError("Target reference facet chart requires one exact triangle.")
    source_triangle: Triangle2 = (mapped[0], mapped[1], mapped[2])
    return (source_triangle,), domain


def _require_reference_chart_partition(
    triangles: tuple[Triangle2, ...],
    domain: tuple[Triangle2, ...],
    /,
) -> None:
    if not triangles or any(_area2(triangle) == 0 for triangle in triangles):
        raise ValueError("Target reference chart pieces must be nondegenerate triangles.")
    for triangle in triangles:
        covered_area = sum(
            (_intersection_area(triangle, source) for source in domain), Fraction(0)
        )
        if covered_area != abs(_area2(triangle)):
            raise ValueError("Target reference chart piece leaves its complete facet.")
    if any(
        _intersection_area(first, second) > 0
        for index, first in enumerate(triangles)
        for second in triangles[index + 1 :]
    ):
        raise ValueError("Target reference chart pieces have positive-measure overlap.")
    if sum((abs(_area2(triangle)) for triangle in triangles), Fraction(0)) != sum(
        (abs(_area2(triangle)) for triangle in domain), Fraction(0)
    ):
        raise ValueError("Target reference chart pieces omit part of their mesh facet.")


def _polyhedral_surface_target_chart_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    cells: dict[int, tuple[int, int, CellBlock | PolyhedralBlock]],
    facets: dict[tuple[int, int], tuple[int, bool]],
    /,
    *,
    full_source: bool,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[Triangle2, ...],
    tuple[int, ...],
]:

    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_row_by_id = {int(value): row for row, value in enumerate(face_ids)}
    expected_lists: dict[int, list[tuple[int, int, int]]] = {
        int(value): [] for value in face_ids
    }
    for face_id, triangle in polyhedral_face_triangle_vertex_ids(mesh, geometry):
        expected_lists[face_id].append(triangle)
    expected_triangles = {
        face_id: tuple(triangles) for face_id, triangles in expected_lists.items()
    }
    if full_source:
        exterior_ids = {
            facet_id for facet_id, is_exterior in facets.values() if is_exterior
        }
        witnessed_ids = set(_integer_vector(witness.entity_global_ids))
        if not exterior_ids <= witnessed_ids:
            raise ValueError(
                "Full-source polyhedral chart witness must bind every exterior target facet."
            )

    target_triangles = np.asarray(witness.target_face_vertex_global_ids, dtype=np.int64)
    target_reference = np.asarray(witness.target_reference_vertices, dtype=np.float64)
    standard = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    if target_triangles.shape != (
        len(witness.source_chart_ids),
        3,
    ) or target_reference.shape != (len(witness.source_chart_ids), 3, 2):
        raise ValueError(
            "Polyhedral chart rows require target triangle IDs and 2D face charts."
        )
    reference_rows = tuple(
        _exact_triangle2(
            _fractions_from_chart_tokens(witness.target_reference_rational_tokens, row)
        )
        for row in range(len(witness.source_chart_ids))
    )
    parent_triangles_by_face: dict[int, dict[tuple[int, int, int], list[Triangle2]]] = {}
    side_by_face: dict[int, tuple[int, int]] = {}
    target_rows: list[tuple[int, int]] = []
    for row, (cell_id, facet_id, local_facet) in enumerate(
        zip(
            np.asarray(witness.cell_global_ids, dtype=np.int64),
            np.asarray(witness.entity_global_ids, dtype=np.int64),
            np.asarray(witness.local_entity_indices, dtype=np.int64),
            strict=True,
        )
    ):
        found = cells.get(int(cell_id))
        actual = facets.get((int(cell_id), int(local_facet)))
        if (
            found is None
            or actual is None
            or actual[0] != int(facet_id)
            or int(facet_id) not in face_row_by_id
        ):
            raise ValueError(
                "Polyhedral chart witness changes an actual target face/cell incidence."
            )
        side = (int(cell_id), int(local_facet))
        previous_side = side_by_face.setdefault(int(facet_id), side)
        if previous_side != side:
            raise ValueError(
                "Each polyhedral face must use one consistent incident-cell side."
            )
        parent_triangle = (
            int(target_triangles[row, 0]),
            int(target_triangles[row, 1]),
            int(target_triangles[row, 2]),
        )
        if parent_triangle not in expected_triangles[int(facet_id)]:
            raise ValueError(
                "Polyhedral target triangle is not in the canonical actual face triangulation."
            )
        parent_triangles_by_face.setdefault(int(facet_id), {}).setdefault(
            parent_triangle, []
        ).append(reference_rows[row])
        target_rows.append((found[0], found[1]))
    for facet_id, parent_pieces in parent_triangles_by_face.items():
        if set(parent_pieces) != set(expected_triangles[facet_id]):
            raise ValueError(
                "Polyhedral source chart rows omit an actual target face triangle."
            )
        for pieces in parent_pieces.values():
            _require_reference_chart_partition(tuple(pieces), (standard,))
    return (
        tuple(target_rows),
        reference_rows,
        tuple(1 if _area2(triangle) > 0 else -1 for triangle in reference_rows),
    )


def _surface_target_chart_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MeshingDomainBoundarySource | None,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    full_source: bool | None = None,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[Triangle2, ...],
    tuple[int, ...],
]:
    if mesh.ambient_dimension != 3 or witness.chart_dimension != 2:
        raise ValueError(
            "Surface chart fidelity requires two-dimensional charts in three-space."
        )
    cells = _cell_global_rows(mesh)
    cell_ids = _integer_vector(witness.cell_global_ids)
    entities = np.asarray(witness.entity_global_ids, dtype=np.int64)
    local_entities = np.asarray(witness.local_entity_indices, dtype=np.int64)
    count = len(witness.source_chart_ids)
    if (
        len(cell_ids) != count
        or entities.shape != (count,)
        or local_entities.shape != (count,)
    ):
        raise ValueError(
            "Surface chart target identity rows must align with source chart rows."
        )
    if full_source is None:
        full_source = source is None or set(source.patches) == set(
            range(len(source.domain.patches))
        )
    if mesh.topological_dimension == 2:
        if witness.reference_dimension != 2 or np.any(local_entities != -1):
            raise ValueError(
                "Surface target charts must bind top-dimensional reference cells."
            )
        if not np.array_equal(entities, np.asarray(cell_ids, dtype=np.int64)):
            raise ValueError(
                "Surface target chart entities must equal their global cell IDs."
            )
        if full_source and set(cell_ids) != set(cells):
            raise ValueError(
                "Full-source surface chart witness must bind every target cell."
            )
        reference_triangles = tuple(
            _exact_triangle2(
                _fractions_from_chart_tokens(
                    witness.target_reference_rational_tokens, row
                )
            )
            for row in range(count)
        )
        by_cell: dict[int, list[Triangle2]] = {}
        for row in range(count):
            cell_id = cell_ids[row]
            found = cells.get(cell_id)
            if found is None:
                raise ValueError("Surface chart witness names an absent target cell.")
            block_index, cell_row, block = found
            if block.cell_kind != "triangle":
                raise ValueError(
                    "Surface source charts currently require triangle target cells."
                )
            by_cell.setdefault(cell_id, []).append(reference_triangles[row])
        standard: Triangle2 = (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
        for triangles in by_cell.values():
            _require_reference_chart_partition(tuple(triangles), (standard,))
        return (
            tuple((cells[cell_id][0], cells[cell_id][1]) for cell_id in cell_ids),
            reference_triangles,
            tuple(1 if _area2(triangle) > 0 else -1 for triangle in reference_triangles),
        )

    if mesh.topological_dimension != 3:
        raise ValueError(
            "Surface chart fidelity supports two-dimensional targets and volume boundaries."
        )
    facets = _volume_boundary_facet_lookup(mesh)
    polyhedral = np.asarray(witness.target_face_vertex_global_ids).size != 0
    if polyhedral:
        if witness.reference_dimension != 2:
            raise ValueError(
                "Polyhedral face charts use their explicit two-dimensional face coordinates."
            )
        return _polyhedral_surface_target_chart_rows(
            mesh, geometry, witness, cells, facets, full_source=full_source
        )
    if witness.reference_dimension != 3:
        raise ValueError(
            "Standard volume-boundary charts retain three-dimensional parent-reference corners."
        )
    if any(
        cells[cell_id][2].cell_kind == "polyhedron"
        for cell_id in cell_ids
        if cell_id in cells
    ):
        raise ValueError(
            "A polyhedral volume chart witness requires explicit target face triangle ancestry."
        )
    if full_source:
        boundary_ids = {
            identifier for identifier, boundary in facets.values() if boundary
        }
        target_entity_ids = {int(entities[row]) for row in range(entities.size)}
        if not boundary_ids <= target_entity_ids:
            raise ValueError(
                "Full-source volume chart witness must bind every exterior target facet."
            )

    parent_reference_triangles = tuple(
        _exact_triangle3(
            _fractions_from_chart_tokens(witness.target_reference_rational_tokens, row)
        )
        for row in range(count)
    )
    grouped: dict[tuple[int, int], list[Triangle2]] = {}
    domains: dict[tuple[int, int], tuple[Triangle2, ...]] = {}
    reference_rows: list[Triangle2] = []
    cell_rows: list[tuple[int, int]] = []
    orientation_signs: list[int] = []
    facet_owners: dict[int, tuple[int, int]] = {}
    for row in range(count):
        cell_id = cell_ids[row]
        local_entity = int(local_entities[row])
        entity_id = int(entities[row])
        found = cells.get(cell_id)
        actual = facets.get((cell_id, local_entity))
        if found is None or actual is None or actual[0] != entity_id:
            raise ValueError(
                "Volume chart witness changes an actual target facet ancestry."
            )
        side = (cell_id, local_entity)
        previous_side = facet_owners.setdefault(actual[0], side)
        if previous_side != side:
            raise ValueError(
                "Each volume target facet must use one consistent incident-cell side."
            )
        block_index, cell_row, block = found
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Standard volume-boundary charts require reference-cell blocks."
            )
        target_charts, domain = _face_reference_chart(
            block.cell_kind, local_entity, parent_reference_triangles[row]
        )
        if len(target_charts) != 1:
            raise ValueError("A volume chart row must map one parent-reference triangle.")
        target_chart = target_charts[0]
        key = (cell_id, local_entity)
        grouped.setdefault(key, []).append(target_chart)
        domains[key] = domain
        reference_rows.append(target_chart)
        cell_rows.append((block_index, cell_row))
        orientation_signs.append(1 if _area2(target_chart) > 0 else -1)
    for key, triangles in grouped.items():
        _require_reference_chart_partition(tuple(triangles), domains[key])
    return tuple(cell_rows), tuple(reference_rows), tuple(orientation_signs)


def _exact_surface_point_from_expressions(
    expressions: tuple[Polynomial, ...], uv: Point2, /
) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
    try:
        point = tuple(evaluate_expression(expression, uv) for expression in expressions)
    except (ArithmeticError, FloatingPointError, TypeError, ValueError):
        return None
    if len(point) != 3:
        return None
    return point[0], point[1], point[2]


def _exact_surface_point(
    surface: AbstractSurfacePatch, uv: Point2, /
) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
    if isinstance(surface, PlanePatch):
        origin = rational_vector(np.asarray(surface.origin))
        first = rational_vector(np.asarray(surface.first_axis))
        second = rational_vector(np.asarray(surface.second_axis))
        return (
            origin[0] + uv[0] * first[0] + uv[1] * second[0],
            origin[1] + uv[0] * first[1] + uv[1] * second[1],
            origin[2] + uv[0] * first[2] + uv[1] * second[2],
        )
    from ._source_interpolation import exact_surface_coordinate_expressions

    try:
        expressions = exact_surface_coordinate_expressions(surface)
    except (ArithmeticError, FloatingPointError, TypeError, ValueError):
        return None
    if expressions is None:
        return None
    return _exact_surface_point_from_expressions(expressions, uv)


def _triangle_physical_normal(
    corners: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    first = tuple(corners[1][axis] - corners[0][axis] for axis in range(3))
    second = tuple(corners[2][axis] - corners[0][axis] for axis in range(3))
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _plane_normal_deviation(
    surface: AbstractSurfacePatch,
    orientation: int,
    uv_triangle: Triangle2,
    target_cross: tuple[ExactScalar, ExactScalar, ExactScalar],
    /,
) -> tuple[Fraction, Fraction] | None:
    if not isinstance(surface, PlanePatch):
        return None
    exact_points = tuple(_exact_surface_point(surface, point) for point in uv_triangle)
    if any(point is None for point in exact_points):
        return None
    first = rational_vector(np.asarray(surface.first_axis))
    second = rational_vector(np.asarray(surface.second_axis))
    source_cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    if orientation < 0:
        source_cross = tuple(-value for value in source_cross)
    parallel = (
        source_cross[1] * target_cross[2] - source_cross[2] * target_cross[1],
        source_cross[2] * target_cross[0] - source_cross[0] * target_cross[2],
        source_cross[0] * target_cross[1] - source_cross[1] * target_cross[0],
    )
    dot = sum(
        (source_cross[axis] * target_cross[axis] for axis in range(3)),
        Fraction(0),
    )
    if all(value == 0 for value in parallel):
        if dot > 0:
            return Fraction(0), Fraction(0)
        if dot < 0:
            return Fraction(math.pi), Fraction(float(np.nextafter(math.pi, math.inf)))
    cosine = _normal_cosine_bounds(
        tuple(
            (
                Fraction(coefficient_endpoint(value, upper=False)),
                Fraction(coefficient_endpoint(value, upper=True)),
            )
            for value in source_cross
        ),
        target_cross,
    )
    if cosine is None:
        return None
    if cosine[0] == cosine[1] == 1:
        return Fraction(0), Fraction(0)
    return _acos_lower(cosine[1]), _acos_upper(cosine[0])


def _surface_jacobian_bounds(
    surface: AbstractSurfacePatch, uv_triangle: Triangle2, /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Enclose the physical Jacobian columns of a source surface on a chart triangle."""
    from ._source_interpolation import (
        _has_unexpanded_exact_surface_carrier,
        exact_surface_coordinate_expressions,
    )

    exact_surface = exact_surface_coordinate_expressions(surface)
    if exact_surface is not None:
        arguments = _affine_triangle_arguments(uv_triangle)
        lower = np.empty((3, 2), dtype=np.float64)
        upper = np.empty_like(lower)
        for physical_axis, expression in enumerate(exact_surface):
            for parameter in range(2):
                restricted = compose_expression(
                    derivative(expression, parameter), arguments
                )
                low, high = polynomial_bounds(restricted, "simplex", 2)
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    return None
                lower[physical_axis, parameter] = low
                upper[physical_axis, parameter] = high
        return lower, upper
    if _has_unexpanded_exact_surface_carrier(surface):
        return None
    chart_lower = tuple(min(point[axis] for point in uv_triangle) for axis in range(2))
    chart_upper = tuple(max(point[axis] for point in uv_triangle) for axis in range(2))
    box_lower = np.asarray(
        tuple(fraction_endpoint(value, upper=False) for value in chart_lower),
        dtype=np.float64,
    )
    box_upper = np.asarray(
        tuple(fraction_endpoint(value, upper=True) for value in chart_upper),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(box_lower)) or np.any(~np.isfinite(box_upper)):
        return None
    jacobian = prepare_interval_function(jax.jacfwd(surface.evaluate), 2)
    low, high = jacobian.evaluate(box_lower.reshape((1, 2)), box_upper.reshape((1, 2)))
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return None
    return np.asarray(low[0], dtype=np.float64), np.asarray(high[0], dtype=np.float64)


type _PatchDerivativeSource = (
    tuple[RationalBSplineSurfaceSpan, ...] | PreparedIntervalFunction
)


type _TensorNet = tuple[tuple[ExactScalar, ...], ...]


class _DerivativeCache:
    """Per-certification derivative sources and exact span enclosures.

    A spline cell's Bernstein restriction is the dominant exact cost; its
    enclosures serve both the interpolation and the normal bound of the cell.
    """

    __slots__ = ("enclosures", "intervals", "normal_nets", "spans")

    def __init__(self) -> None:
        self.spans: dict[int, tuple[RationalBSplineSurfaceSpan, ...] | None] = {}
        self.intervals: dict[int, PreparedIntervalFunction] = {}
        self.enclosures: dict[
            tuple[int, Triangle2], tuple[RationalBSplineSurfaceHessianEnclosure, ...]
        ] = {}
        self.normal_nets: dict[
            tuple[int, tuple[int, int]], tuple[_TensorNet, _TensorNet, _TensorNet]
        ] = {}

    def patch_spans(
        self, patch: int, surface: AbstractSurfacePatch, /
    ) -> tuple[RationalBSplineSurfaceSpan, ...] | None:
        if patch not in self.spans:
            from ._meshing_domain import _exact_rational_surface_spans

            self.spans[patch] = _exact_rational_surface_spans(surface)
        return self.spans[patch]

    def source(
        self, patch: int, surface: AbstractSurfacePatch, /
    ) -> _PatchDerivativeSource:
        spans = self.patch_spans(patch, surface)
        if spans is not None:
            return spans
        interval = self.intervals.get(patch)
        if interval is None:
            interval = prepare_interval_function(jax.jacfwd(surface.evaluate), 2)
            self.intervals[patch] = interval
        return interval

    def span_enclosures(
        self,
        patch: int,
        spans: tuple[RationalBSplineSurfaceSpan, ...],
        uv_triangle: Triangle2,
        /,
    ) -> tuple[RationalBSplineSurfaceHessianEnclosure, ...]:
        key = (patch, uv_triangle)
        enclosures = self.enclosures.get(key)
        if enclosures is None:
            from ._meshing_domain import _span_enclosures_for_chart

            enclosures = _span_enclosures_for_chart(spans, uv_triangle)
            self.enclosures[key] = enclosures
        return enclosures

    def span_normal_nets(
        self,
        patch: int,
        spans: tuple[RationalBSplineSurfaceSpan, ...],
        span_indices: tuple[int, int],
        /,
    ) -> tuple[_TensorNet, _TensorNet, _TensorNet]:
        """Bernstein nets of ``N_u x N_v``, positively parallel to ``S_u x S_v``.

        A rational span's first derivatives are ``N_u / W**2`` and ``N_v / W**2``
        with ``W > 0``, so the polynomial cross product of the derivative
        numerators has the oriented normal direction everywhere on the span.
        """
        key = (patch, span_indices)
        nets = self.normal_nets.get(key)
        if nets is None:
            from .._interpolation._bspline_extraction import (
                _tensor_bernstein_linear_combination,
                _tensor_bernstein_product,
            )

            span = next(span for span in spans if span.span_indices == span_indices)
            first, second = span._du_numerators, span._dv_numerators
            if len(first) != 3 or len(second) != 3:
                raise ValueError("Surface spline spans must map into three-space.")

            def component(left: int, right: int) -> _TensorNet:
                return _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(first[left], second[right]),
                        ),
                        (
                            Fraction(-1),
                            _tensor_bernstein_product(first[right], second[left]),
                        ),
                    )
                )

            nets = (component(1, 2), component(2, 0), component(0, 1))
            self.normal_nets[key] = nets
        return nets

    def span_normal_cone_cosine(
        self,
        patch: int,
        spans: tuple[RationalBSplineSurfaceSpan, ...],
        enclosures: tuple[RationalBSplineSurfaceHessianEnclosure, ...],
        orientation: int,
        target: tuple[tuple[Fraction, Fraction], ...],
        /,
    ) -> Fraction | None:
        """Lower cosine bound between the target normal and every source normal.

        On each span sub-box the normal direction is a convex combination of the
        restricted Bernstein control vectors. When every control vector makes
        an acute angle at most ``theta`` with the target, so does each convex
        combination: the set of such vectors is a convex cone.
        """
        from .._interpolation._bspline_extraction import _tensor_bernstein_restrict

        cosine_lower: Fraction | None = None
        for enclosure in enclosures:
            nets = self.span_normal_nets(patch, spans, enclosure.span_indices)
            (span_lower_u, span_upper_u), (span_lower_v, span_upper_v) = (
                enclosure.span_parameter_bounds
            )
            (lower_u, upper_u), (lower_v, upper_v) = enclosure.parameter_bounds
            width_u = span_upper_u - span_lower_u
            width_v = span_upper_v - span_lower_v
            restricted = tuple(
                _tensor_bernstein_restrict(
                    net,
                    (lower_u - span_lower_u) / width_u,
                    (upper_u - span_lower_u) / width_u,
                    (lower_v - span_lower_v) / width_v,
                    (upper_v - span_lower_v) / width_v,
                )
                for net in nets
            )
            for row in zip(*restricted, strict=True):
                for vector in zip(*row, strict=True):
                    intervals = tuple(
                        (
                            Fraction(coefficient_endpoint(value, upper=False)),
                            Fraction(coefficient_endpoint(value, upper=True)),
                        )
                        if orientation > 0
                        else (
                            Fraction(-coefficient_endpoint(value, upper=True)),
                            Fraction(-coefficient_endpoint(value, upper=False)),
                        )
                        for value in vector
                    )
                    cosine = _normal_cosine_interval_bounds(intervals, target)
                    if cosine is None or cosine[0] <= 0:
                        return None
                    if cosine_lower is None or cosine[0] < cosine_lower:
                        cosine_lower = cosine[0]
        return cosine_lower


def _span_jacobian_bounds(
    enclosures: tuple[RationalBSplineSurfaceHessianEnclosure, ...], /
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hull of exact span Jacobian bounds over every span the cell meets."""
    if not enclosures:
        return None
    lower = np.empty((len(enclosures[0].du_bounds), 2), dtype=np.float64)
    upper = np.empty_like(lower)
    for component in range(lower.shape[0]):
        for parameter in range(2):
            bounds = [
                (enclosure.du_bounds if parameter == 0 else enclosure.dv_bounds)[
                    component
                ]
                for enclosure in enclosures
            ]
            lower[component, parameter] = fraction_endpoint(
                min(bound[0] for bound in bounds), upper=False
            )
            upper[component, parameter] = fraction_endpoint(
                max(bound[1] for bound in bounds), upper=True
            )
    return lower, upper


type _FractionInterval = tuple[Fraction, Fraction]


def _interval_sum(*intervals: _FractionInterval) -> _FractionInterval:
    return (
        sum((value[0] for value in intervals), Fraction(0)),
        sum((value[1] for value in intervals), Fraction(0)),
    )


def _interval_scale(value: Fraction, interval: _FractionInterval, /) -> _FractionInterval:
    return _interval_product((value, value), interval)


def _interval_cross(
    first: tuple[_FractionInterval, ...], second: tuple[_FractionInterval, ...], /
) -> tuple[_FractionInterval, ...]:
    return tuple(
        _interval_difference(
            _interval_product(first[(axis + 1) % 3], second[(axis + 2) % 3]),
            _interval_product(first[(axis + 2) % 3], second[(axis + 1) % 3]),
        )
        for axis in range(3)
    )


def _trig_interval(
    function: Callable[[Interval], Interval],
    lower: float,
    upper: float,
    /,
) -> _FractionInterval:
    low, high = function((np.asarray(lower), np.asarray(upper)))
    return Fraction(float(low)), Fraction(float(high))


def _exact_cross(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _exact_dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _sphere_normal_direction(
    surface: SpherePatch, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[_FractionInterval, ...] | None:
    """Enclose a positive multiple of ``X_u x X_v`` over a sphere chart box.

    ``X_u = r cos(v) E'(u)`` and ``X_v = r T(u, v)`` with ``E = cos u f + sin u s``
    and ``T = -sin v E + cos v a``, so ``X_u x X_v = r^2 cos(v) E' x T``. On the
    latitude domain ``|v| <= 0.5 * pi`` (binary64, below the real ``pi / 2``)
    ``cos v > 0``, so ``E' x T`` has the oriented normal's direction, also at the
    poles where ``X_u`` vanishes.
    """
    radius = Fraction(float(np.asarray(surface.radius)))
    half_turn = Fraction(0.5 * math.pi)
    if radius == 0 or lower[1] < -half_turn or upper[1] > half_turn:
        return None
    first_axis, second_axis, axis = (
        rational_vector(np.asarray(value, dtype=np.float64))
        for value in (surface.first_axis, surface.second_axis, surface.axis)
    )
    cos_u = _trig_interval(interval_cos, lower[0], upper[0])
    sin_u = _trig_interval(interval_sin, lower[0], upper[0])
    cos_v = _trig_interval(interval_cos, lower[1], upper[1])
    sin_v = _trig_interval(interval_sin, lower[1], upper[1])
    equatorial = tuple(
        _interval_sum(
            _interval_scale(first_axis[index], cos_u),
            _interval_scale(second_axis[index], sin_u),
        )
        for index in range(3)
    )
    equatorial_derivative = tuple(
        _interval_sum(
            _interval_scale(-first_axis[index], sin_u),
            _interval_scale(second_axis[index], cos_u),
        )
        for index in range(3)
    )
    meridian = tuple(
        _interval_sum(
            _interval_product((-sin_v[1], -sin_v[0]), equatorial[index]),
            _interval_scale(axis[index], cos_v),
        )
        for index in range(3)
    )
    return _interval_cross(equatorial_derivative, meridian)


def _profile_meridian_sign(
    offset: Fraction,
    cosine_weight: Fraction,
    sine_weight: Fraction,
    lower: float,
    upper: float,
    /,
) -> int | None:
    """Sign of ``offset + cosine_weight cos t + sine_weight sin t`` inside ``[lower, upper]``.

    The sign is decided by its enclosure, or, when that contains zero, by a
    derivative enclosure excluding zero (at most one root) and endpoint
    enclosures of one sign: the function then vanishes at most at one end.
    """

    def value(first: float, last: float, /) -> _FractionInterval:
        return _interval_sum(
            (offset, offset),
            _interval_scale(cosine_weight, _trig_interval(interval_cos, first, last)),
            _interval_scale(sine_weight, _trig_interval(interval_sin, first, last)),
        )

    whole = value(lower, upper)
    if whole[0] > 0:
        return 1
    if whole[1] < 0:
        return -1
    slope = _interval_sum(
        _interval_scale(-cosine_weight, _trig_interval(interval_sin, lower, upper)),
        _interval_scale(sine_weight, _trig_interval(interval_cos, lower, upper)),
    )
    if slope[0] <= 0 <= slope[1]:
        return None
    ends = (value(lower, lower), value(upper, upper))
    if all(end[0] >= 0 for end in ends):
        return 1
    if all(end[1] <= 0 for end in ends):
        return -1
    return None


def _revolution_normal_direction(
    surface: RevolutionSurface, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[_FractionInterval, ...] | None:
    """Enclose a positive multiple of ``X_u x X_v`` over a circle-revolution box.

    With an exactly unit axis ``a`` the Rodrigues map is a rotation ``R(u)``, so
    ``X_u = R(u)(a x q)``, ``X_v = R(u) q'`` for the profile offset ``q(t)``. A
    circle profile whose plane contains the axis keeps ``q`` in ``span(w, a)``
    with ``w = n x a`` (``n`` the circle-plane normal), and then
    ``X_u x X_v = (q . w) / |w|^2 R(u)((a . q') w - (w . q') a)``. The rotated
    profile normal stays regular on the axis where ``q . w`` vanishes; the sign
    of ``q . w`` inside the profile interval orients it.
    """
    profile = surface.curve
    if not isinstance(profile, CircleCurve):
        return None
    axis = rational_vector(np.asarray(surface.axis_direction, dtype=np.float64))
    if _exact_dot(axis, axis) != 1:
        return None
    origin = rational_vector(np.asarray(surface.axis_origin, dtype=np.float64))
    center, first_axis, second_axis = (
        rational_vector(np.asarray(value, dtype=np.float64))
        for value in (profile.center, profile.first_axis, profile.second_axis)
    )
    radius = Fraction(float(np.asarray(profile.radius)))
    offset = tuple(center[index] - origin[index] for index in range(3))
    plane_normal = _exact_cross(first_axis, second_axis)
    if (
        radius == 0
        or not any(plane_normal)
        or _exact_dot(plane_normal, axis) != 0
        or _exact_dot(plane_normal, offset) != 0
    ):
        return None
    meridian = _exact_cross(plane_normal, axis)
    sign = _profile_meridian_sign(
        _exact_dot(offset, meridian),
        radius * _exact_dot(first_axis, meridian),
        radius * _exact_dot(second_axis, meridian),
        float(lower[1]),
        float(upper[1]),
    )
    if sign is None:
        return None
    cos_t = _trig_interval(interval_cos, lower[1], upper[1])
    sin_t = _trig_interval(interval_sin, lower[1], upper[1])
    tangent = tuple(
        _interval_sum(
            _interval_scale(-radius * first_axis[index], sin_t),
            _interval_scale(radius * second_axis[index], cos_t),
        )
        for index in range(3)
    )
    tangent_axis = _interval_sum(
        *(_interval_scale(axis[index], tangent[index]) for index in range(3))
    )
    tangent_meridian = _interval_sum(
        *(_interval_scale(meridian[index], tangent[index]) for index in range(3))
    )
    binormal = _exact_cross(axis, meridian)
    cos_u = _trig_interval(interval_cos, lower[0], upper[0])
    sin_u = _trig_interval(interval_sin, lower[0], upper[0])
    direction = tuple(
        _interval_sum(
            _interval_product(
                tangent_axis,
                _interval_sum(
                    _interval_scale(meridian[index], cos_u),
                    _interval_scale(binormal[index], sin_u),
                ),
            ),
            _interval_scale(-axis[index], tangent_meridian),
        )
        for index in range(3)
    )
    if sign < 0:
        return tuple((-high, -low) for low, high in direction)
    return direction


def _closed_form_normal_deviation(
    surface: AbstractSurfacePatch,
    orientation: int,
    uv_triangle: Triangle2,
    target_cross: tuple[ExactScalar, ExactScalar, ExactScalar],
    limit: Fraction | None,
    /,
) -> tuple[Fraction, Fraction] | None:
    """Normal deviation from a closed-form normal direction of a revolution chart.

    The Jacobian cross product of sphere and revolution charts vanishes on the
    axis, so its interval enclosure loses the normal direction near a pole; the
    closed-form direction stays regular there. Other carriers return ``None``.

    A cell's bounding box encloses up to twice its area, and the box enclosure
    of the direction widens with it. While a piece's bound straddles ``limit``
    it is split at its edge midpoints into four exact chart triangles whose
    union is the piece, halving its box overestimate. Splitting stops when every
    piece is decided, when a piece's lower bound exceeds the limit (the cell's
    deviation then exceeds it), or when a piece's edge midpoints round onto
    its corners in binary64, so its chart box can no longer shrink.
    """
    from ._meshing_domain import _PlacedSurfacePatch

    # A placed chart X' = L X + b has X'_u x X'_v = cof(L) (X_u x X_v).
    cofactors: list[tuple[tuple[Fraction, ...], ...]] = []
    while isinstance(surface, _PlacedSurfacePatch):
        linear = tuple(
            rational_vector(np.asarray(row, dtype=np.float64))
            for row in np.asarray(surface.linear)
        )
        # Row r of cof(L) = det(L) L^-T is l_(r+1) x l_(r+2) for rows l of L.
        cofactors.append(
            tuple(
                _exact_cross(linear[(row + 1) % 3], linear[(row + 2) % 3])
                for row in range(3)
            )
        )
        surface = surface.source
    if not isinstance(surface, (SpherePatch, RevolutionSurface)):
        return None
    carrier = surface

    def bound(triangle: Triangle2, /) -> tuple[Fraction, Fraction] | None:
        lower = np.asarray(
            tuple(
                fraction_endpoint(min(point[axis] for point in triangle), upper=False)
                for axis in range(2)
            ),
            dtype=np.float64,
        )
        upper = np.asarray(
            tuple(
                fraction_endpoint(max(point[axis] for point in triangle), upper=True)
                for axis in range(2)
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            return None
        direction = (
            _sphere_normal_direction(carrier, lower, upper)
            if isinstance(carrier, SpherePatch)
            else _revolution_normal_direction(carrier, lower, upper)
        )
        if direction is None:
            return None
        for cofactor in reversed(cofactors):
            direction = tuple(
                _interval_sum(
                    *(
                        _interval_scale(cofactor[row][column], direction[column])
                        for column in range(3)
                    )
                )
                for row in range(3)
            )
        source_cross = (
            direction
            if orientation > 0
            else tuple((-high, -low) for low, high in direction)
        )
        cosine = _normal_cosine_bounds(source_cross, target_cross)
        if cosine is None:
            return None
        return _acos_lower(cosine[1]), _normal_angle_upper(
            source_cross, target_cross, cosine[0]
        )

    def split(triangle: Triangle2, /) -> tuple[Triangle2, ...] | None:
        first, second, third = triangle

        def midpoint(start: Point2, end: Point2, /) -> Point2:
            return (start[0] + end[0]) / 2, (start[1] + end[1]) / 2

        near, middle, far = (
            midpoint(first, second),
            midpoint(second, third),
            midpoint(third, first),
        )
        if all(
            (float(point[0]), float(point[1]))
            in ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
            for point, start, end in (
                (near, first, second),
                (middle, second, third),
                (far, third, first),
            )
        ):
            return None
        return (
            (first, near, far),
            (near, second, middle),
            (far, middle, third),
            (near, middle, far),
        )

    evidence = bound(uv_triangle)
    if evidence is None or limit is None or evidence[1] <= limit or evidence[0] > limit:
        return evidence
    # Every piece's lower bound is attained inside the cell; the cell's upper
    # bound is the largest over a partition of the cell into pieces.
    lower = evidence[0]
    settled_upper = Fraction(0)
    pending = [(uv_triangle, evidence)]
    while pending:
        following: list[tuple[Triangle2, tuple[Fraction, Fraction]]] = []
        for piece, piece_evidence in pending:
            children = split(piece)
            child_evidence = (
                None if children is None else tuple(bound(child) for child in children)
            )
            if children is None or child_evidence is None or None in child_evidence:
                settled_upper = max(settled_upper, piece_evidence[1])
                continue
            for child, child_bound in zip(children, child_evidence, strict=True):
                assert child_bound is not None
                lower = max(lower, child_bound[0])
                if child_bound[0] > limit:
                    return lower, evidence[1]
                if child_bound[1] <= limit:
                    settled_upper = max(settled_upper, child_bound[1])
                else:
                    following.append((child, child_bound))
        pending = following
    return lower, min(evidence[1], settled_upper)


def _normal_deviation_for_cell(
    domain: MeshingDomain,
    patch: int,
    uv_triangle: Triangle2,
    target_cross: tuple[ExactScalar, ExactScalar, ExactScalar],
    derivatives: _DerivativeCache,
    limit: Fraction | None = None,
    /,
) -> tuple[Fraction, Fraction] | None:
    """Enclose one cell's normal deviation; sources are cached per patch.

    Spline carriers use exact Bernstein Jacobian bounds of their extracted
    spans: a naive interval extension of the basis recursion does not shrink
    to the cell's actual tangent range. Closed-form revolution charts split a
    cell whose bound straddles ``limit`` until the bound decides it.
    """
    surface_patch = domain.patches[patch]
    exact_plane = _plane_normal_deviation(
        surface_patch.surface, surface_patch.orientation, uv_triangle, target_cross
    )
    if exact_plane is not None:
        return exact_plane
    closed_form = _closed_form_normal_deviation(
        surface_patch.surface,
        surface_patch.orientation,
        uv_triangle,
        target_cross,
        limit,
    )
    if closed_form is not None:
        return closed_form
    surface = surface_patch.surface
    from ._source_interpolation import (
        _has_unexpanded_exact_surface_carrier,
        exact_surface_coordinate_expressions,
    )

    exact_surface = exact_surface_coordinate_expressions(surface)
    source: _PatchDerivativeSource | None = None
    enclosures: tuple[RationalBSplineSurfaceHessianEnclosure, ...] | None = None
    if exact_surface is not None:
        arguments = _affine_triangle_arguments(uv_triangle)
        low = np.empty((3, 2), dtype=np.float64)
        high = np.empty_like(low)
        for physical_axis, expression in enumerate(exact_surface):
            for parameter in range(2):
                restricted = compose_expression(
                    derivative(expression, parameter), arguments
                )
                lower, upper = polynomial_bounds(restricted, "simplex", 2)
                if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                    return None
                low[physical_axis, parameter] = lower
                high[physical_axis, parameter] = upper
    else:
        if _has_unexpanded_exact_surface_carrier(surface):
            return None
        chart_lower = tuple(
            min(point[axis] for point in uv_triangle) for axis in range(2)
        )
        chart_upper = tuple(
            max(point[axis] for point in uv_triangle) for axis in range(2)
        )
        box_lower = np.asarray(
            tuple(fraction_endpoint(value, upper=False) for value in chart_lower),
            dtype=np.float64,
        )
        box_upper = np.asarray(
            tuple(fraction_endpoint(value, upper=True) for value in chart_upper),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(box_lower)) or np.any(~np.isfinite(box_upper)):
            return None
        source = derivatives.source(patch, surface)
        if isinstance(source, PreparedIntervalFunction):
            bounds = source.evaluate(box_lower.reshape((1, 2)), box_upper.reshape((1, 2)))
            low, high = bounds[0][0], bounds[1][0]
        else:
            enclosures = derivatives.span_enclosures(patch, source, uv_triangle)
            jacobian = _span_jacobian_bounds(enclosures)
            if jacobian is None:
                return None
            low, high = jacobian
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return None
    source_cross = _normal_cross_interval(low, high, surface_patch.orientation)
    cosine = _normal_cosine_bounds(source_cross, target_cross)
    if (
        enclosures is None
        or source is None
        or isinstance(source, PreparedIntervalFunction)
    ):
        if cosine is None:
            return None
        return _acos_lower(cosine[1]), _normal_angle_upper(
            source_cross, target_cross, cosine[0]
        )
    # The componentwise Jacobian-box cross product decouples the tangent
    # components; the exact Bernstein normal cone keeps them coupled.
    cone_cosine = derivatives.span_normal_cone_cosine(
        patch,
        source,
        enclosures,
        surface_patch.orientation,
        tuple(
            (
                Fraction(coefficient_endpoint(value, upper=False)),
                Fraction(coefficient_endpoint(value, upper=True)),
            )
            for value in target_cross
        ),
    )
    if cosine is None and cone_cosine is None:
        return None
    lower = Fraction(0) if cosine is None else _acos_lower(cosine[1])
    upper = min(
        bound
        for bound in (
            None
            if cosine is None
            else _normal_angle_upper(source_cross, target_cross, cosine[0]),
            None if cone_cosine is None else _acos_upper(cone_cosine),
        )
        if bound is not None
    )
    return lower, upper


def _curved_trim_lens_caps(
    surface: AbstractSurfacePatch,
    chart_rows: list[tuple[int, Triangle2]],
    lenses: tuple[CurvedTrimLens, ...],
    /,
) -> dict[int, Fraction]:
    """Bound the physical distance a curved-trim lens adds to each target cell.

    A lens point is within ``parameter_bound`` of its chord and its arc along a
    chart segment inside the lens box, clipped to the convex source parameter
    domain that holds both segment ends. The surface image of that segment is no
    longer than the Jacobian norm bound on the box times the chart distance. A
    straight trim's box is flat across the trim; that axis is widened by
    ``parameter_bound`` before clipping, a superset on which the Jacobian bound
    still holds.
    """
    caps: dict[int, Fraction] = {}
    domain = surface.parameter_domain
    for lens in lenses:
        if lens.parameter_bound == 0:
            # A straight trim coincides with its chord; its lens adds nothing.
            continue
        widths = tuple(
            lens.parameter_bound if lens.lower[axis] == lens.upper[axis] else 0
            for axis in range(2)
        )
        lower = (lens.lower[0] - widths[0], lens.lower[1] - widths[1])
        upper = (lens.upper[0] + widths[0], lens.upper[1] + widths[1])
        if domain is not None:
            lower = (
                max(lower[0], Fraction.from_float(float(domain[0][0]))),
                max(lower[1], Fraction.from_float(float(domain[0][1]))),
            )
            upper = (
                min(upper[0], Fraction.from_float(float(domain[1][0]))),
                min(upper[1], Fraction.from_float(float(domain[1][1]))),
            )
        if lower[0] >= upper[0] or lower[1] >= upper[1]:
            raise ValueError("A curved trim lens leaves the source parameter domain.")
        norm_squared = Fraction(0)
        for triangle in (
            (lower, (upper[0], lower[1]), upper),
            (lower, upper, (lower[0], upper[1])),
        ):
            bounds = _surface_jacobian_bounds(surface, triangle)
            if bounds is None:
                raise ValueError("A curved trim lens has no surface Jacobian enclosure.")
            magnitude = np.maximum(np.abs(bounds[0]), np.abs(bounds[1]))
            norm_squared = max(
                norm_squared,
                sum(
                    (Fraction.from_float(float(value)) ** 2 for value in magnitude.flat),
                    Fraction(0),
                ),
            )
        cap = fraction_sqrt_interval(norm_squared)[1] * lens.parameter_bound
        for cell_id, chart in chart_rows:
            if all(
                min(point[axis] for point in chart) <= upper[axis]
                and max(point[axis] for point in chart) >= lower[axis]
                for axis in range(2)
            ):
                caps[cell_id] = max(caps.get(cell_id, Fraction(0)), cap)
    return caps


def _certify_source_point_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryPointWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    source_rows, source_points, target_points = _source_point_correspondences(
        mesh, geometry, source, witness
    )
    limit = Fraction.from_float(float(tolerance))
    deviations = np.empty((len(source_rows), 2), dtype=np.float64)
    lower_max = Fraction(0)
    upper_max = Fraction(0)
    findings: list[MeshCertificateFinding] = []
    vertex_ids = _integer_vector(witness.vertex_global_ids)
    for row, (source_point, target_point, _source_row, vertex_id) in enumerate(
        zip(source_points, target_points, source_rows, vertex_ids, strict=True)
    ):
        source_lower, source_upper = source_point
        components: list[tuple[Fraction, Fraction]] = []
        for source_low, source_high, target_coordinate in zip(
            source_lower, source_upper, target_point, strict=True
        ):
            difference_lower = target_coordinate - source_high
            difference_upper = target_coordinate - source_low
            lower_float = coefficient_endpoint(difference_lower, upper=False)
            upper_float = coefficient_endpoint(difference_upper, upper=True)
            if not math.isfinite(lower_float) or not math.isfinite(upper_float):
                raise ValueError(
                    "Source point distance enclosure exceeds finite numeric resolution."
                )
            exact_lower = Fraction.from_float(lower_float)
            exact_upper = Fraction.from_float(upper_float)
            absolute_lower = (
                Fraction(0)
                if exact_lower <= 0 <= exact_upper
                else min(abs(exact_lower), abs(exact_upper))
            )
            absolute_upper = max(abs(exact_lower), abs(exact_upper))
            components.append((absolute_lower, absolute_upper))
        distance_lower = _distance_lower(tuple(value[0] for value in components))
        distance_upper = _distance_upper(tuple(value[1] for value in components))
        lower_max = max(lower_max, distance_lower)
        upper_max = max(upper_max, distance_upper)
        deviations[row] = (
            fraction_endpoint(distance_lower, upper=False, saturate=True),
            fraction_endpoint(distance_upper, upper=True, saturate=True),
        )
        if distance_lower > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_point_distance_limit",
                    "violated",
                    "vertex",
                    (vertex_id,),
                    detail="A certified source-point distance lower bound exceeds the requested tolerance.",
                )
            )
        elif distance_upper > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_point_distance_range",
                    "unresolved",
                    "vertex",
                    (vertex_id,),
                    detail="The source-point distance enclosure does not fit within the requested tolerance.",
                )
            )
    if maximum_normal_deviation is not None:
        findings.append(
            MeshCertificateFinding(
                "source_point_normal",
                "unresolved",
                "source_entity",
                detail="A zero-dimensional source point has no oriented normal for this request.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        deviations,
        fraction_endpoint(lower_max, upper=False, saturate=True),
        fraction_endpoint(upper_max, upper=True, saturate=True),
        fraction_endpoint(lower_max, upper=False, saturate=True),
        fraction_endpoint(upper_max, upper=True, saturate=True),
        None,
        None,
        vertex_ids,
        tuple(source_rows),
        witness.occurrence_paths,
        witness,
        None,
        tuple(findings),
    )


def _certify_surface_envelope_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SurfaceEnvelopeDistanceWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    _validate_surface_envelope_witness(mesh, geometry, source, witness)
    source_to_mesh_upper = Fraction.from_float(float(witness.bounds[0]))
    mesh_to_source_upper = Fraction.from_float(float(witness.bounds[1]))
    mesh_to_source_lower = Fraction.from_float(float(witness.bounds[2]))
    limit = Fraction.from_float(float(tolerance))
    findings: list[MeshCertificateFinding] = []
    if mesh_to_source_lower > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "violated",
                "mesh",
                detail="The certified envelope separation lower bound exceeds the requested original-source tolerance.",
            )
        )
    elif mesh_to_source_upper > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "mesh",
                detail="The continuous envelope-to-original-source upper bound exceeds the request.",
            )
        )
    if source_to_mesh_upper > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "mesh",
                detail="The continuous original-source-to-envelope upper bound exceeds the request.",
            )
        )
    if maximum_normal_deviation is not None:
        findings.append(
            MeshCertificateFinding(
                "source_envelope_normal",
                "unresolved",
                "mesh",
                detail="The unsigned-distance envelope witness does not prove oriented-normal deviation.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        np.empty((0, 2), dtype=np.float64),
        fraction_endpoint(mesh_to_source_lower, upper=False, saturate=True),
        fraction_endpoint(mesh_to_source_upper, upper=True, saturate=True),
        0.0,
        fraction_endpoint(source_to_mesh_upper, upper=True, saturate=True),
        None,
        None,
        (),
        (),
        (),
        witness,
        None,
        tuple(findings),
    )


def _certify_analytic_profile_distance_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryDistanceWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    mesh_lower, mesh_upper, source_upper = _analytic_profile_distance_evidence(
        mesh, geometry, source, witness
    )
    limit = Fraction.from_float(float(tolerance))
    findings: list[MeshCertificateFinding] = []
    if mesh_lower > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "violated",
                "mesh",
                detail="A continuous target-boundary distance lower bound exceeds the requested tolerance.",
            )
        )
    elif mesh_upper > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "mesh",
                detail="The continuous target-to-profile distance upper bound exceeds the request.",
            )
        )
    if source_upper > limit:
        findings.append(
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "mesh",
                detail="The continuous profile-to-target distance upper bound exceeds the request.",
            )
        )
    if maximum_normal_deviation is not None:
        findings.append(
            MeshCertificateFinding(
                "source_distance_normal",
                "unresolved",
                "mesh",
                detail="A profile distance witness does not prove oriented-normal deviation.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        np.empty((0, 2), dtype=np.float64),
        fraction_endpoint(mesh_lower, upper=False, saturate=True),
        fraction_endpoint(mesh_upper, upper=True, saturate=True),
        0.0,
        fraction_endpoint(source_upper, upper=True, saturate=True),
        None,
        None,
        (),
        (),
        (),
        witness,
        None,
        tuple(findings),
    )


def _unresolved_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
    witness: SourceFidelityWitness | None,
    coverage: SourceBoundaryChartCover | None,
    findings: tuple[MeshCertificateFinding, ...],
) -> SourceFidelityCertificate:
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        np.empty((0, 2), dtype=np.float64),
        0.0,
        math.inf,
        0.0,
        math.inf,
        None if maximum_normal_deviation is None else np.empty((0,), dtype=np.float64),
        None,
        (),
        (),
        (),
        witness,
        coverage,
        findings,
    )


def _certify_domain_surface_chart_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    from ._source_interpolation import certify_domain_surface_charts

    domain_source = _surface_query_root(source)

    validate_source_fidelity_witness(mesh, geometry, source, witness)
    patch_rows, charts = _surface_source_chart_rows(domain_source, witness)
    target_rows, _, _ = _surface_target_chart_rows(mesh, geometry, domain_source, witness)
    evidence = certify_domain_surface_charts(
        mesh,
        geometry,
        domain_source,
        witness,
        patch_rows,
        target_rows,
        tolerance=tolerance,
    )
    coverage = domain_source.boundary_chart_cover()
    findings = [*coverage.findings, *evidence.findings]
    if not coverage.complete or coverage.semantics != "certified" or coverage.findings:
        findings.append(
            MeshCertificateFinding(
                "source_boundary_cover",
                "unresolved",
                "source_facet",
                detail="Independent source boundary chart coverage is not complete and certified.",
            )
        )
    normal_bounds = None
    normal_lower_max = Fraction(0)
    if maximum_normal_deviation is not None:
        normal_bounds = np.full(
            (len(evidence.cell_global_ids),), np.inf, dtype=np.float64
        )
        normal_limit = Fraction.from_float(float(maximum_normal_deviation))
        derivatives = _DerivativeCache()
        for row, (patch, chart) in enumerate(zip(patch_rows, charts, strict=True)):
            if not evidence.target_affine or not evidence.target_affine[row]:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_range",
                        "unresolved",
                        "cell",
                        (evidence.cell_global_ids[row],),
                        detail="Continuous normal deviation requires an affine target chart map.",
                    )
                )
                continue
            target_cross = _triangle_physical_normal(evidence.target_corners[row])
            try:
                normal_evidence = _normal_deviation_for_cell(
                    domain_source.domain,
                    int(patch),
                    chart,
                    target_cross,
                    derivatives,
                    normal_limit,
                )
            except (IntervalExtensionError, TypeError, ValueError, FloatingPointError):
                normal_evidence = None
            if normal_evidence is None:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_range",
                        "unresolved",
                        "cell",
                        (evidence.cell_global_ids[row],),
                        detail="Source or target oriented-normal range is singular or unresolved.",
                    )
                )
                continue
            lower, upper = normal_evidence
            normal_lower_max = max(normal_lower_max, lower)
            normal_bounds[row] = fraction_endpoint(upper, upper=True, saturate=True)
            if lower > normal_limit:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "violated",
                        "cell",
                        (evidence.cell_global_ids[row],),
                        detail="A certified oriented-normal lower bound exceeds the request.",
                    )
                )
            elif upper > normal_limit:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "unresolved",
                        "cell",
                        (evidence.cell_global_ids[row],),
                        detail="The continuous oriented-normal upper bound exceeds the request.",
                    )
                )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        evidence.deviation_bounds,
        evidence.mesh_to_source_lower,
        evidence.mesh_to_source_upper,
        evidence.source_to_mesh_lower,
        evidence.source_to_mesh_upper,
        normal_bounds,
        None
        if maximum_normal_deviation is None
        else fraction_endpoint(normal_lower_max, upper=False, saturate=True),
        evidence.cell_global_ids,
        tuple(int(value) for value in patch_rows),
        witness.occurrence_paths,
        witness,
        coverage,
        tuple(findings),
    )


def _certify_reconstructed_chart_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: ReconstructedTriangleBoundarySource,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    domain_source = _reconstructed_domain_query(source)
    domain_certificate = _certify_domain_surface_chart_fidelity(
        mesh,
        geometry,
        domain_source,
        witness,
        tolerance=tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
    )
    coverage = source.boundary_chart_cover()
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        domain_certificate.deviation_bounds,
        domain_certificate.mesh_to_source_lower,
        domain_certificate.mesh_to_source_upper,
        domain_certificate.source_to_mesh_lower,
        domain_certificate.source_to_mesh_upper,
        domain_certificate.normal_deviation_bounds,
        domain_certificate.maximum_normal_deviation_lower,
        domain_certificate.source_cell_global_ids,
        domain_certificate.source_patch_ids,
        domain_certificate.source_occurrence_paths,
        witness,
        coverage,
        domain_certificate.findings,
    )


def _certify_curve_chart_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    from ._curve_boundary import ParametricCurveBoundarySource
    from ._source_interpolation import certify_parametric_curve_chart

    if not isinstance(source, ParametricCurveBoundarySource):
        raise TypeError(
            "A one-dimensional source chart witness requires a parametric curve source."
        )
    validate_source_fidelity_witness(mesh, geometry, source, witness)
    target_rows = _curve_target_chart_rows(mesh, witness)
    evidence = certify_parametric_curve_chart(
        mesh, geometry, source, witness, target_rows, tolerance
    )
    findings = list(evidence.findings)
    normal_bounds = None
    if maximum_normal_deviation is not None:
        normal_bounds = np.full(
            (len(evidence.cell_global_ids),), np.inf, dtype=np.float64
        )
        findings.append(
            MeshCertificateFinding(
                "normal_deviation_range",
                "unresolved",
                "mesh",
                detail="A one-dimensional curve chart does not define an oriented surface normal.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        evidence.deviation_bounds,
        evidence.mesh_to_source_lower,
        evidence.mesh_to_source_upper,
        evidence.source_to_mesh_lower,
        evidence.source_to_mesh_upper,
        normal_bounds,
        None,
        evidence.cell_global_ids,
        evidence.chart_rows,
        evidence.occurrence_paths,
        witness,
        None,
        tuple(findings),
    )


def _certify_piecewise_linear_curve_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource
    from ._source_interpolation import certify_piecewise_linear_curve_charts

    if (
        not isinstance(source, PiecewiseLinearBoundarySource)
        or source.entity_dimension != 1
    ):
        raise TypeError("PWL edge fidelity requires a one-dimensional boundary query.")
    validate_source_fidelity_witness(mesh, geometry, source, witness)
    target_rows = _curve_target_chart_rows(mesh, witness)
    evidence = certify_piecewise_linear_curve_charts(
        mesh, geometry, source, witness, target_rows, tolerance
    )
    findings = list(evidence.findings)
    normal_bounds = None
    if maximum_normal_deviation is not None:
        normal_bounds = np.full(
            (len(evidence.cell_global_ids),), math.inf, dtype=np.float64
        )
        findings.append(
            MeshCertificateFinding(
                "normal_deviation_range",
                "unresolved",
                "mesh",
                detail="A one-dimensional PWL curve does not define an oriented surface normal.",
            )
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        evidence.deviation_bounds,
        evidence.mesh_to_source_lower,
        evidence.mesh_to_source_upper,
        evidence.source_to_mesh_lower,
        evidence.source_to_mesh_upper,
        normal_bounds,
        None,
        evidence.cell_global_ids,
        evidence.chart_rows,
        evidence.occurrence_paths,
        witness,
        None,
        tuple(findings),
    )


def _certify_piecewise_linear_surface_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    from ._mapped_reference_interpolation import certify_surface_chart_maps
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if not isinstance(source, PiecewiseLinearBoundarySource):
        raise TypeError("PWL source fidelity requires PiecewiseLinearBoundarySource.")
    validate_source_fidelity_witness(mesh, geometry, source, witness)
    target_rows, _, _ = _surface_target_chart_rows(
        mesh,
        geometry,
        None,
        witness,
        full_source=source.is_full_scope,
    )
    coverage = source.boundary_chart_cover()
    findings: list[MeshCertificateFinding] = list(coverage.findings)
    if not coverage.complete or coverage.semantics != "certified" or coverage.findings:
        findings.append(
            MeshCertificateFinding(
                "source_boundary_cover",
                "unresolved",
                "source_facet",
                detail="Derived PWL boundary charts do not reproduce a complete certified source cover.",
            )
        )
    evidence = certify_surface_chart_maps(
        mesh,
        geometry,
        source,
        witness,
        target_rows,
        tolerance=tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
    )
    findings.extend(evidence.findings)
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        evidence.deviation_bounds,
        evidence.mesh_to_source_lower,
        evidence.mesh_to_source_upper,
        evidence.source_to_mesh_lower,
        evidence.source_to_mesh_upper,
        evidence.normal_deviation_bounds,
        evidence.maximum_normal_deviation_lower,
        evidence.cell_global_ids,
        evidence.chart_rows,
        evidence.occurrence_paths,
        witness,
        coverage,
        tuple(findings),
    )


def _certify_mapped_root_chart_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    witness: SourceBoundaryChartWitness,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SourceFidelityCertificate:
    from ._mapped_reference_boundary import MappedReferenceBoundarySource
    from ._mapped_reference_interpolation import (
        certify_mapped_reference_curve_charts,
        certify_surface_chart_maps,
    )

    if not isinstance(source, MappedReferenceBoundarySource):
        raise TypeError("Mapped-root fidelity requires MappedReferenceBoundarySource.")
    validate_source_fidelity_witness(mesh, geometry, source, witness)
    coverage = None
    if witness.chart_dimension == 2:
        target_rows, _, _ = _surface_target_chart_rows(
            mesh,
            geometry,
            None,
            witness,
            full_source=False,
        )
        try:
            coverage = source.boundary_chart_cover()
        except (ArithmeticError, FloatingPointError, TypeError, ValueError):
            coverage = None
        if (
            coverage is None
            or not coverage.complete
            or coverage.semantics != "certified"
            or coverage.findings
        ):
            cover_findings = (
                MeshCertificateFinding(
                    "source_boundary_cover",
                    "unresolved",
                    "source_facet",
                    detail="Mapped root source charts lack a complete continuous source cover.",
                ),
            )
        else:
            cover_findings = ()
        evidence = certify_surface_chart_maps(
            mesh,
            geometry,
            source,
            witness,
            target_rows,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )
    elif witness.chart_dimension == 1:
        target_rows = _curve_target_chart_rows(mesh, witness)
        cover_findings = ()
        evidence = certify_mapped_reference_curve_charts(
            mesh,
            geometry,
            source,
            witness,
            target_rows,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )
    else:
        raise ValueError(
            "Mapped-root fidelity supports only one- and two-dimensional source charts."
        )
    findings = [*cover_findings, *evidence.findings]
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance,
        maximum_normal_deviation,
        evidence.deviation_bounds,
        evidence.mesh_to_source_lower,
        evidence.mesh_to_source_upper,
        evidence.source_to_mesh_lower,
        evidence.source_to_mesh_upper,
        evidence.normal_deviation_bounds,
        evidence.maximum_normal_deviation_lower,
        evidence.cell_global_ids,
        evidence.chart_rows,
        evidence.occurrence_paths,
        witness,
        coverage,
        tuple(findings),
    )


def certify_source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None = None,
    witness: SourceFidelityWitness | None = None,
) -> SourceFidelityCertificate:
    """Certify continuous physical source fidelity from one typed witness."""
    tolerance_ = float(tolerance)
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("source fidelity tolerance must be finite and nonnegative.")
    normal_limit = (
        None if maximum_normal_deviation is None else float(maximum_normal_deviation)
    )
    if normal_limit is not None and (
        not math.isfinite(normal_limit) or normal_limit < 0.0 or normal_limit > math.pi
    ):
        raise ValueError("maximum_normal_deviation must be None or radians in [0, pi].")
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError(
            "source must implement the immutable SourceBoundaryQuery contract."
        )
    if isinstance(witness, SourceBoundaryDistanceWitness):
        return _certify_analytic_profile_distance_fidelity(
            mesh,
            geometry,
            source,
            witness,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
        )
    if isinstance(witness, SurfaceEnvelopeDistanceWitness):
        return _certify_surface_envelope_fidelity(
            mesh,
            geometry,
            source,
            witness,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
        )
    if isinstance(witness, SourceBoundaryPointWitness):
        return _certify_source_point_fidelity(
            mesh,
            geometry,
            source,
            witness,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
        )
    if isinstance(witness, SourceBoundaryChartWitness):
        from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

        if isinstance(source, PiecewiseLinearBoundarySource):
            if witness.chart_dimension == 1:
                return _certify_piecewise_linear_curve_fidelity(
                    mesh,
                    geometry,
                    source,
                    witness,
                    tolerance=tolerance_,
                    maximum_normal_deviation=normal_limit,
                )
            if witness.chart_dimension == 2:
                return _certify_piecewise_linear_surface_fidelity(
                    mesh,
                    geometry,
                    source,
                    witness,
                    tolerance=tolerance_,
                    maximum_normal_deviation=normal_limit,
                )
            raise ValueError("PWL chart fidelity supports only edge and face strata.")
        from ._mapped_reference_boundary import MappedReferenceBoundarySource

        if isinstance(source, MappedReferenceBoundarySource):
            return _certify_mapped_root_chart_fidelity(
                mesh,
                geometry,
                source,
                witness,
                tolerance=tolerance_,
                maximum_normal_deviation=normal_limit,
            )
        if witness.chart_dimension == 1:
            return _certify_curve_chart_fidelity(
                mesh,
                geometry,
                source,
                witness,
                tolerance=tolerance_,
                maximum_normal_deviation=normal_limit,
            )
        if witness.chart_dimension == 2:
            try:
                _surface_query_root(source)
            except TypeError:
                pass
            else:
                return _certify_domain_surface_chart_fidelity(
                    mesh,
                    geometry,
                    source,
                    witness,
                    tolerance=tolerance_,
                    maximum_normal_deviation=normal_limit,
                )
        from .reconstruction._boundary import ReconstructedTriangleBoundarySource

        if witness.chart_dimension == 2 and isinstance(
            source, ReconstructedTriangleBoundarySource
        ):
            return _certify_reconstructed_chart_fidelity(
                mesh,
                geometry,
                source,
                witness,
                tolerance=tolerance_,
                maximum_normal_deviation=normal_limit,
            )
        if witness.chart_dimension == 2:
            from .implicit._adaptive_discovery import AdaptiveImplicitBoundarySource
            from .implicit._source_charts import (
                _AdaptiveImplicitBoundaryCarrier as _ChartCarrier,
                certify_adaptive_implicit_charts,
            )

            if isinstance(source, AdaptiveImplicitBoundarySource) or isinstance(
                source, _ChartCarrier
            ):
                return certify_adaptive_implicit_charts(
                    mesh,
                    geometry,
                    source,
                    witness,
                    tolerance=tolerance_,
                    maximum_normal_deviation=normal_limit,
                )
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
            witness=witness,
            coverage=None,
            findings=(
                MeshCertificateFinding(
                    "source_fidelity_witness",
                    "unresolved",
                    "mesh",
                    detail="The source boundary chart authority has no supported continuous evaluator.",
                ),
            ),
        )
    if witness is None:
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
            witness=None,
            coverage=None,
            findings=(
                MeshCertificateFinding(
                    "source_fidelity_witness",
                    "unresolved",
                    "mesh",
                    detail="Continuous source fidelity requires one source-bound fidelity witness.",
                ),
            ),
        )
    if not isinstance(witness, SurfaceChartWitness):
        raise TypeError(
            "witness must be a member of the exported SourceFidelityWitness union."
        )
    from ._surface_source_support import SurfaceNativeRestrictionBoundarySource

    if not isinstance(
        source, (MeshingDomainBoundarySource, SurfaceNativeRestrictionBoundarySource)
    ):
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
            witness=witness,
            coverage=None,
            findings=(
                MeshCertificateFinding(
                    "source_fidelity_witness",
                    "unresolved",
                    "mesh",
                    detail="The surface chart witness has no matching authoritative domain source.",
                ),
            ),
        )
    domain_source = _surface_query_root(source)
    if isinstance(source, SurfaceNativeRestrictionBoundarySource):
        source.require_current()
    else:
        domain_source.require_current()
    coverage = domain_source.boundary_chart_cover()
    findings = list(coverage.findings)
    if not coverage.complete or coverage.semantics != "certified" or findings:
        findings.append(
            MeshCertificateFinding(
                "source_boundary_cover",
                "unresolved",
                "source_facet",
                detail="Independent source boundary chart coverage is not complete and certified.",
            )
        )
    try:
        findings.extend(_partition_findings(domain_source, witness))
        _, cell_ids, patch_ids, occurrence_paths = _chart_point_tables(
            domain_source, witness, mesh, geometry
        )
    except (TypeError, ValueError, IndexError) as error:
        findings.append(
            MeshCertificateFinding(
                "source_chart_identity",
                "unresolved",
                "mesh",
                detail=f"Exact source chart coverage is unavailable: {error}",
            )
        )
        return _unresolved_fidelity(
            mesh,
            geometry,
            source,
            tolerance=tolerance_,
            maximum_normal_deviation=normal_limit,
            witness=witness,
            coverage=coverage,
            findings=tuple(findings),
        )
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        findings.append(
            MeshCertificateFinding(
                "source_chart_dimension",
                "unresolved",
                "mesh",
                detail="Surface chart fidelity requires a two-dimensional mesh in three-space.",
            )
        )
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    target_corners: dict[int, tuple[tuple[ExactScalar, ...], ...]] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(block, CellBlock) or block.cell_kind != "triangle":
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "mesh",
                    detail="Continuous surface fidelity requires triangle target blocks.",
                )
            )
            continue
        if not isinstance(element, ScalarCoordinateElement):
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    _integer_vector(block.global_ids),
                    detail="Continuous surface fidelity requires scalar coordinate elements.",
                )
            )
            continue
        route_rows = np.asarray(route, dtype=np.int64)
        for row_index in range(route_rows.shape[0]):
            row = route_rows[row_index]
            cell_id = int(block.global_ids[row_index])
            local = tuple(
                coefficients[int(row[route_index])] for route_index in range(row.shape[0])
            )
            expressions = coordinate_map_expressions(element, local, cell_row=row_index)
            if (
                block.cell_kind != "triangle"
                or len(expressions) != 3
                or not all(_poly_is_affine(expression) for expression in expressions)
            ):
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_target_map",
                        "unresolved",
                        "cell",
                        (cell_id,),
                        detail="Continuous fidelity requires an exact affine target triangle map.",
                    )
                )
                continue
            reference_vertices = reference_cell_topology("triangle").vertices
            target_corners[cell_id] = tuple(
                tuple(
                    evaluate_expression(
                        expression,
                        tuple(Fraction(float(value)) for value in point),
                    )
                    for expression in expressions
                )
                for point in reference_vertices
            )

    cell_count = len(cell_ids)
    row_for_id = {identifier: row for row, identifier in enumerate(cell_ids)}
    deviation_bounds = np.zeros((cell_count, 2), dtype=np.float64)
    deviation_bounds[:, 1] = np.inf
    normal_bounds = (
        None if normal_limit is None else np.full((cell_count,), np.inf, dtype=np.float64)
    )
    normal_lower_max = Fraction(0)
    derivatives = _DerivativeCache()
    mesh_to_source_upper = Fraction(0)
    source_to_mesh_upper = Fraction(0)
    witness_cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    witness_patch_ids = np.asarray(witness.patches, dtype=np.int64)
    if (
        witness_cell_ids.ndim != 1
        or witness_patch_ids.shape != witness_cell_ids.shape
        or witness_cell_ids.size != cell_count
    ):
        raise ValueError(
            "Surface chart witness rows do not align with actual target cells."
        )
    witness_ids = tuple(
        int(witness_cell_ids[row]) for row in range(witness_cell_ids.size)
    )
    witness_patches = tuple(
        int(witness_patch_ids[row]) for row in range(witness_patch_ids.size)
    )
    witness_charts = tuple(
        _exact_chart_from_tokens(witness.chart_rational_tokens, row)
        for row in range(len(witness_ids))
    )
    by_patch: dict[int, list[tuple[int, Triangle2]]] = {}
    for cell_id, patch, chart in zip(
        witness_ids, witness_patches, witness_charts, strict=True
    ):
        by_patch.setdefault(patch, []).append((cell_id, chart))
    for patch, chart_rows in by_patch.items():
        charts_float = np.asarray(
            [
                [[float(value) for value in point] for point in chart]
                for _, chart in chart_rows
            ],
            dtype=np.float64,
        )
        surface = domain_source.domain.patches[patch].surface
        from ._meshing_domain import _outward_surface_parameter_box
        from ._source_interpolation import (
            _exact_surface_interpolation_bound,
            _fraction_float_interval,
            exact_surface_coordinate_expressions,
        )

        try:
            try:
                exact_surface_map = exact_surface_coordinate_expressions(surface)
            except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                exact_surface_map = None
            exact_points_by_row: list[tuple[tuple[ExactScalar, ...], ...] | None] = []
            for _, chart in chart_rows:
                if exact_surface_map is None:
                    exact_points_by_row.append(None)
                    continue
                point_0 = _exact_surface_point_from_expressions(
                    exact_surface_map, chart[0]
                )
                point_1 = _exact_surface_point_from_expressions(
                    exact_surface_map, chart[1]
                )
                point_2 = _exact_surface_point_from_expressions(
                    exact_surface_map, chart[2]
                )
                exact_points_by_row.append(
                    None
                    if point_0 is None or point_1 is None or point_2 is None
                    else (point_0, point_1, point_2)
                )
            spans = (
                None
                if exact_surface_map is not None
                else derivatives.patch_spans(patch, surface)
            )
            if exact_surface_map is not None:
                interpolation = np.asarray(
                    [
                        _exact_surface_interpolation_bound(exact_surface_map, chart)
                        for _, chart in chart_rows
                    ],
                    dtype=np.float64,
                )
            elif spans is not None:
                from ._meshing_domain import (
                    _spline_interpolation_bound_from_enclosures,
                )

                interpolation = np.asarray(
                    [
                        _spline_interpolation_bound_from_enclosures(
                            chart, derivatives.span_enclosures(patch, spans, chart)
                        )
                        for _, chart in chart_rows
                    ],
                    dtype=np.float64,
                )
            else:
                interpolation = np.asarray(
                    domain_source.domain.interpolation_bounds(patch, charts_float),
                    dtype=np.float64,
                )
            # Adjacent cells share corners; each source corner box is computed once.
            corner_boxes: dict[Point2, np.ndarray] = {}
            source_intervals_by_row: list[tuple[np.ndarray, np.ndarray] | None] = []
            for chart_row, (_, chart) in enumerate(chart_rows):
                if exact_points_by_row[chart_row] is not None:
                    source_intervals_by_row.append(None)
                    continue
                point_lower = np.empty((3, 3), dtype=np.float64)
                point_upper = np.empty_like(point_lower)
                for corner, uv in enumerate(chart):
                    point_box = corner_boxes.get(uv)
                    if point_box is None:
                        u_bounds = _fraction_float_interval(uv[0])
                        v_bounds = _fraction_float_interval(uv[1])
                        parameter_box = _outward_surface_parameter_box(
                            surface,
                            (u_bounds[0], v_bounds[0]),
                            (u_bounds[1], v_bounds[1]),
                        )
                        point_box = np.asarray(
                            surface.bounding_box(parameter_box), dtype=np.float64
                        )
                        if (
                            point_box.shape != (2, 3)
                            or np.any(~np.isfinite(point_box))
                            or np.any(point_box[0] > point_box[1])
                        ):
                            raise ValueError(
                                "Source chart corner has no finite ordered physical point box."
                            )
                        corner_boxes[uv] = point_box
                    point_lower[corner] = point_box[0]
                    point_upper[corner] = point_box[1]
                source_intervals_by_row.append((point_lower, point_upper))
            # A sliver's source image stays within its cap bound of the owner's
            # pole corner, whose target image is bounded by the endpoint error.
            sliver_caps: dict[int, Fraction] = {}
            target_charts = tuple(chart for _, chart in chart_rows)
            slivers = domain_source.collapsed_pole_slivers(patch, target_charts)
            for owner_row, sliver in slivers:
                owner = chart_rows[owner_row][0]
                sliver_caps[owner] = max(
                    sliver_caps.get(owner, Fraction(0)),
                    domain_source.pole_sliver_cap_bound(patch, sliver),
                )
            # Target charts follow a curved trim by chords. A chart point between a
            # chord and its arc is within the lens bound of both, so the directed
            # distances of every cell meeting the lens box grow by its surface image.
            # Currentness was validated once above, before the per-patch loop.
            _, lenses = domain_source._trim_partition_reference_unchecked(
                patch, (*target_charts, *(sliver for _, sliver in slivers))
            )
            lens_caps = _curved_trim_lens_caps(surface, chart_rows, lenses)
        except (
            ArithmeticError,
            FloatingPointError,
            IntervalExtensionError,
            NotImplementedError,
            TypeError,
            ValueError,
        ) as error:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_interpolation",
                    "unresolved",
                    "source_facet",
                    (patch,),
                    detail=f"Continuous source interpolation enclosure is unavailable: {error}",
                )
            )
            continue
        source_intervals_valid = len(source_intervals_by_row) == len(chart_rows) and len(
            exact_points_by_row
        ) == len(chart_rows)
        for interval in source_intervals_by_row:
            if interval is None:
                continue
            lower, upper = interval
            if (
                lower.shape != (3, 3)
                or upper.shape != (3, 3)
                or np.any(~np.isfinite(lower))
                or np.any(~np.isfinite(upper))
                or np.any(lower > upper)
            ):
                source_intervals_valid = False
                break
        if (
            interpolation.shape != (len(chart_rows),)
            or np.any(~np.isfinite(interpolation))
            or not source_intervals_valid
        ):
            findings.append(
                MeshCertificateFinding(
                    "source_chart_interpolation",
                    "unresolved",
                    "source_facet",
                    (patch,),
                    detail="Source interval enclosure has invalid shape or nonfinite endpoints.",
                )
            )
            continue
        for chart_row, (cell_id, chart) in enumerate(chart_rows):
            corners = target_corners.get(cell_id)
            if corners is None:
                continue
            endpoint_error = Fraction(0)
            exact_source_points = exact_points_by_row[chart_row]
            if exact_source_points is None:
                source_interval = source_intervals_by_row[chart_row]
                if source_interval is None:
                    findings.append(
                        MeshCertificateFinding(
                            "source_chart_interpolation",
                            "unresolved",
                            "source_facet",
                            (patch,),
                            detail="A source chart corner has no exact map or interval enclosure.",
                        )
                    )
                    continue
                source_point_lower, source_point_upper = source_interval
                for corner in range(3):
                    component_errors = tuple(
                        _box_gap_upper(
                            float(source_point_lower[corner, axis]),
                            float(source_point_upper[corner, axis]),
                            corners[corner][axis],
                        )
                        for axis in range(3)
                    )
                    endpoint_error = max(
                        endpoint_error, _distance_upper(component_errors)
                    )
            else:
                endpoint_error = max(
                    (
                        _distance_upper(
                            tuple(
                                max(
                                    abs(
                                        Fraction(
                                            coefficient_endpoint(
                                                target - source_value, upper=True
                                            )
                                        )
                                    ),
                                    abs(
                                        Fraction(
                                            coefficient_endpoint(
                                                target - source_value, upper=False
                                            )
                                        )
                                    ),
                                )
                                for target, source_value in zip(
                                    corners[corner],
                                    exact_source_points[corner],
                                    strict=True,
                                )
                            )
                        )
                        for corner in range(3)
                    ),
                    default=Fraction(0),
                )
            interpolation_bound = Fraction(float(interpolation[chart_row]))
            deviation = interpolation_bound + endpoint_error
            cap = sliver_caps.get(cell_id)
            if cap is not None:
                deviation = max(deviation, cap + endpoint_error)
            deviation += lens_caps.get(cell_id, Fraction(0))
            output_row = row_for_id[cell_id]
            deviation_upper = fraction_endpoint(deviation, upper=True, saturate=True)
            deviation_bounds[output_row] = (0.0, deviation_upper)
            mesh_to_source_upper = max(mesh_to_source_upper, deviation)
            source_to_mesh_upper = max(source_to_mesh_upper, deviation)
            if deviation > Fraction(tolerance_):
                findings.append(
                    MeshCertificateFinding(
                        "source_fidelity_tolerance",
                        "unresolved",
                        "cell",
                        (cell_id,),
                        detail="The continuous directed deviation upper bound exceeds the requested tolerance.",
                    )
                )
            if normal_limit is None or normal_bounds is None:
                continue
            target_cross = _triangle_physical_normal(corners)
            try:
                normal_evidence = _normal_deviation_for_cell(
                    domain_source.domain,
                    patch,
                    chart,
                    target_cross,
                    derivatives,
                    Fraction.from_float(normal_limit),
                )
            except (IntervalExtensionError, TypeError, ValueError, FloatingPointError):
                normal_evidence = None
            if normal_evidence is None:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_range",
                        "unresolved",
                        "cell",
                        (cell_id,),
                        detail="Source or actual target normal range is singular or unresolved.",
                    )
                )
                continue
            normal_lower, normal_upper = normal_evidence
            normal_lower_max = max(normal_lower_max, normal_lower)
            normal_bounds[output_row] = fraction_endpoint(
                normal_upper, upper=True, saturate=True
            )
            if normal_lower > Fraction(normal_limit):
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "violated",
                        "cell",
                        (cell_id,),
                        detail="A certified pointwise normal-deviation lower bound exceeds the request.",
                    )
                )
            elif normal_upper > Fraction(normal_limit):
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "unresolved",
                        "cell",
                        (cell_id,),
                        detail="The continuous normal-deviation upper bound exceeds the request.",
                    )
                )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        source=source,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    return SourceFidelityCertificate(
        binding,
        source,
        tolerance_,
        normal_limit,
        deviation_bounds,
        0.0,
        fraction_endpoint(mesh_to_source_upper, upper=True, saturate=True),
        0.0,
        fraction_endpoint(source_to_mesh_upper, upper=True, saturate=True),
        normal_bounds,
        None
        if normal_limit is None
        else fraction_endpoint(normal_lower_max, upper=False, saturate=True),
        cell_ids,
        patch_ids,
        occurrence_paths,
        witness,
        coverage,
        tuple(findings),
    )

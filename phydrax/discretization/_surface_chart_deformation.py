"""Prepare exact material overlap and endpoint evidence for surface charts."""

from __future__ import annotations

import heapq
import math
from fractions import Fraction
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._meshing_domain import MeshingDomain
from ..geometry._source_fidelity import certify_source_fidelity
from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
from ..typing import checked
from . import _coordinate_enclosure as algebra
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._mapped_geometry_measure import (
    mapped_coordinate_expressions,
    mapped_geometry_cells,
)
from ._surface_chart_records import (
    material_piece_id,
    material_point,
    material_reference,
    material_triangle,
    PreparedSurfaceChartDeformation,
    PreparedSurfaceChartOccurrence,
    PreparedSurfaceChartPiece,
    SurfaceChartWitness,
)


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _FloatArray = npt.NDArray[np.float64]
type _IntArray = npt.NDArray[np.int64]


class _WitnessBank(NamedTuple):
    physical_ids: _IntArray
    physical_rows_by_patch: dict[int, tuple[int, ...]]
    witness_rows_by_physical: _IntArray
    charts_by_physical: tuple[_Triangle, ...]
    patches_by_physical: _IntArray


class _EndpointProofs(NamedTuple):
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    fidelity: SourceFidelityCertificate


class _OverlapResult(NamedTuple):
    pieces: tuple[PreparedSurfaceChartPiece, ...]
    source_areas: tuple[Fraction, ...]
    target_areas: tuple[Fraction, ...]


def _signed_area(points: tuple[_Point, ...], /) -> Fraction:
    return (
        sum(
            (
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(points, (*points[1:], points[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _cross(first: _Point, second: _Point, point: _Point, /) -> Fraction:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _chart_bounds(
    triangle: _Triangle, /
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        min(point[0] for point in triangle),
        max(point[0] for point in triangle),
        min(point[1] for point in triangle),
        max(point[1] for point in triangle),
    )


def _candidate_pairs(
    source: tuple[_Triangle, ...], target: tuple[_Triangle, ...], /
) -> tuple[tuple[int, int], ...]:
    """Sweep exact chart bounds; retain only overlapping candidate pairs."""
    events = [
        (bounds[0], 0, row, bounds[1], bounds[2], bounds[3])
        for row, bounds in enumerate(map(_chart_bounds, source))
    ]
    events.extend(
        (bounds[0], 1, row, bounds[1], bounds[2], bounds[3])
        for row, bounds in enumerate(map(_chart_bounds, target))
    )
    events.sort()
    active_source: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    active_target: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    source_expiry: list[tuple[Fraction, int]] = []
    target_expiry: list[tuple[Fraction, int]] = []
    pairs: list[tuple[int, int]] = []
    for lower_x, kind, row, upper_x, lower_y, upper_y in events:
        while source_expiry and source_expiry[0][0] < lower_x:
            _, expired = heapq.heappop(source_expiry)
            active_source.pop(expired, None)
        while target_expiry and target_expiry[0][0] < lower_x:
            _, expired = heapq.heappop(target_expiry)
            active_target.pop(expired, None)
        if kind == 0:
            for target_row, (
                target_upper_x,
                target_lower_y,
                target_upper_y,
            ) in active_target.items():
                if (
                    target_upper_x >= lower_x
                    and lower_y <= target_upper_y
                    and target_lower_y <= upper_y
                ):
                    pairs.append((row, target_row))
            active_source[row] = (upper_x, lower_y, upper_y)
            heapq.heappush(source_expiry, (upper_x, row))
        else:
            for source_row, (
                source_upper_x,
                source_lower_y,
                source_upper_y,
            ) in active_source.items():
                if (
                    source_upper_x >= lower_x
                    and lower_y <= source_upper_y
                    and source_lower_y <= upper_y
                ):
                    pairs.append((source_row, row))
            active_target[row] = (upper_x, lower_y, upper_y)
            heapq.heappush(target_expiry, (upper_x, row))
    return tuple(sorted(pairs))


def _mapped_target_triangle(source: _Triangle, target: _Triangle, /) -> _Triangle:
    return (
        material_reference(target, source[0]),
        material_reference(target, source[1]),
        material_reference(target, source[2]),
    )


def _supporting_planes(
    mapped: _Triangle, /
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    origin = mapped[0]
    first = (mapped[1][0] - origin[0], mapped[1][1] - origin[1])
    second = (mapped[2][0] - origin[0], mapped[2][1] - origin[1])
    return (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
        (-first[0], -second[0], origin[0]),
        (-first[1], -second[1], origin[1]),
        (
            first[0] + first[1],
            second[0] + second[1],
            Fraction(1) - origin[0] - origin[1],
        ),
    )


def _clip_halfplane(
    polygon: tuple[_Point, ...],
    plane: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[_Point, ...]:
    if not polygon:
        return ()
    first_coefficient, second_coefficient, bound = plane
    output: list[_Point] = []
    previous = polygon[-1]
    previous_value = (
        bound - first_coefficient * previous[0] - second_coefficient * previous[1]
    )
    for current in polygon:
        current_value = (
            bound - first_coefficient * current[0] - second_coefficient * current[1]
        )
        if current_value >= 0:
            if previous_value < 0:
                fraction = previous_value / (previous_value - current_value)
                output.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
            output.append(current)
        elif previous_value >= 0:
            fraction = previous_value / (previous_value - current_value)
            output.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        previous, previous_value = current, current_value
    compact: list[_Point] = []
    for point in output:
        if not compact or compact[-1] != point:
            compact.append(point)
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    return tuple(compact)


def _strict_convex_polygon(points: tuple[_Point, ...], /) -> tuple[_Point, ...]:
    result = list(points)
    changed = True
    while changed and len(result) >= 3:
        changed = False
        for index in range(len(result)):
            previous = result[index - 1]
            current = result[index]
            following = result[(index + 1) % len(result)]
            if _cross(previous, current, following) == 0:
                result.pop(index)
                changed = True
                break
    if len(result) < 3:
        return ()
    if _signed_area(tuple(result)) < 0:
        result.reverse()
    if _signed_area(tuple(result)) <= 0:
        return ()
    return tuple(result)


def _polygon_labels(
    polygon: tuple[_Point, ...],
    planes: tuple[tuple[Fraction, Fraction, Fraction], ...],
    /,
) -> tuple[int, ...]:
    labels = []
    for first, second in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        support = next(
            (
                index
                for index, (a, b, c) in enumerate(planes)
                if a * first[0] + b * first[1] == c and a * second[0] + b * second[1] == c
            ),
            None,
        )
        if support is None:
            raise ValueError("An exact material polygon edge has no supporting plane.")
        labels.append(support)
    return tuple(labels)


def _overlap_pieces(
    source_charts: tuple[_Triangle, ...],
    target_charts: tuple[_Triangle, ...],
    source_ids: _IntArray,
    target_ids: _IntArray,
    /,
) -> _OverlapResult:
    if source_ids.shape != (len(source_charts),) or target_ids.shape != (
        len(target_charts),
    ):
        raise ValueError("Surface material charts and cell identities do not align.")
    source_areas = [Fraction(0) for _ in source_charts]
    target_areas = [Fraction(0) for _ in target_charts]
    pieces: list[PreparedSurfaceChartPiece] = []
    unit = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    for source_row, target_row in _candidate_pairs(source_charts, target_charts):
        mapped = _mapped_target_triangle(
            source_charts[source_row], target_charts[target_row]
        )
        planes = _supporting_planes(mapped)
        polygon = unit
        for plane in planes:
            polygon = _clip_halfplane(polygon, plane)
            if not polygon:
                break
        polygon = _strict_convex_polygon(polygon)
        if not polygon:
            continue
        labels = _polygon_labels(polygon, planes)
        for fan_index in range(1, len(polygon) - 1):
            source_vertices: _Triangle = (
                polygon[0],
                polygon[fan_index],
                polygon[fan_index + 1],
            )
            target_vertices: _Triangle = (
                material_point(mapped, source_vertices[0]),
                material_point(mapped, source_vertices[1]),
                material_point(mapped, source_vertices[2]),
            )
            source_area = abs(_signed_area(source_vertices))
            target_area = abs(_signed_area(target_vertices))
            if not source_area or not target_area:
                raise ValueError("A surface material overlap piece is collapsed.")
            fan = (0, fan_index, fan_index + 1)
            source_id, target_id = (
                int(source_ids[source_row]),
                int(target_ids[target_row]),
            )
            pieces.append(
                PreparedSurfaceChartPiece(
                    source_row,
                    target_row,
                    source_id,
                    target_id,
                    source_vertices,
                    target_vertices,
                    source_area,
                    target_area,
                    labels,
                    polygon,
                    fan,
                    material_piece_id(
                        source_id,
                        target_id,
                        source_vertices,
                        target_vertices,
                        polygon,
                        labels,
                        fan,
                    ),
                )
            )
            source_areas[source_row] += source_area
            target_areas[target_row] += target_area
    if any(area != Fraction(1, 2) for area in (*source_areas, *target_areas)):
        raise ValueError(
            "Exact material overlaps must partition every source and target triangle."
        )
    return _OverlapResult(tuple(pieces), tuple(source_areas), tuple(target_areas))


def _nearest_integer(value: Fraction, /) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def _align_periodic_charts(
    source_charts: tuple[_Triangle, ...],
    target_charts: tuple[_Triangle, ...],
    source_roots: _IntArray,
    target_roots: _IntArray,
    periods: tuple[float | None, ...],
    /,
) -> tuple[_Triangle, ...]:
    if source_roots.shape != (len(source_charts),) or target_roots.shape != (
        len(target_charts),
    ):
        raise ValueError("Periodic chart ancestry must align with material rows.")
    anchor: tuple[_Triangle, _Triangle] | None = None
    target_by_root: dict[int, int] = {}
    for row, identity in enumerate(target_roots):
        target_by_root.setdefault(int(identity), row)
    for row, identity in enumerate(source_roots):
        target_row = target_by_root.get(int(identity))
        if target_row is not None:
            anchor = (source_charts[row], target_charts[target_row])
            break
    if anchor is None:
        anchor = (source_charts[0], target_charts[0])
    source_anchor, target_anchor = anchor
    shifts = []
    for axis, period_value in enumerate(periods):
        if period_value is None:
            shifts.append(Fraction(0))
            continue
        period = Fraction(float(period_value))
        if period <= 0:
            raise ValueError("Surface chart periods must be strictly positive.")
        source_center = sum((point[axis] for point in source_anchor), Fraction(0)) / 3
        target_center = sum((point[axis] for point in target_anchor), Fraction(0)) / 3
        shifts.append(period * _nearest_integer((source_center - target_center) / period))

    def shift_point(point: _Point, /) -> _Point:
        return (point[0] + shifts[0], point[1] + shifts[1])

    return tuple(
        (
            shift_point(triangle[0]),
            shift_point(triangle[1]),
            shift_point(triangle[2]),
        )
        for triangle in target_charts
    )


def _witness_bank(
    witness: SurfaceChartWitness,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    /,
) -> _WitnessBank:
    if (witness.geometry_id, witness.topology_id, witness.domain_id) != (
        cell_geometry_id(geometry),
        mesh.topology_id,
        domain.domain_id,
    ):
        raise ValueError("Surface chart witness is stale for its endpoint or domain.")
    if witness.cell_kind != "triangle" or any(
        block.cell_kind != "triangle" for block in mesh.blocks
    ):
        raise ValueError("Surface chart deformation requires triangular cell blocks.")
    physical_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    witness_rows = np.searchsorted(witness_ids, physical_ids)
    if (
        witness_rows.shape != physical_ids.shape
        or np.any(witness_rows >= witness_ids.size)
        or not np.array_equal(witness_ids[witness_rows], physical_ids)
    ):
        raise ValueError("Surface chart witness must bind every physical cell ID once.")
    patches = np.asarray(witness.patches, dtype=np.int64)[witness_rows]
    if np.any(patches >= len(domain.patches)):
        raise ValueError("Surface chart witness references an absent domain patch.")
    for row, patch in enumerate(patches):
        if (
            witness.geometry_entity_ids[witness_rows[row]]
            != domain.entity_id(2, int(patch))
            or witness.occurrence_paths[witness_rows[row]]
            != domain.source_occurrences[2][int(patch)]
        ):
            raise ValueError(
                "Surface chart witness changes its source occurrence identity."
            )
    charts = tuple(
        material_triangle(witness.exact_chart(int(row))) for row in witness_rows
    )
    by_patch = {
        int(patch): tuple(int(row) for row in np.flatnonzero(patches == patch))
        for patch in np.unique(patches)
    }
    return _WitnessBank(physical_ids, by_patch, witness_rows, charts, patches)


def _chart_patch_rows(
    witness: SurfaceChartWitness, patch: int, /
) -> tuple[tuple[int, ...], tuple[_Triangle, ...], _IntArray]:
    witness_patches = np.asarray(witness.patches, dtype=np.int64)
    witness_rows = tuple(int(row) for row in np.flatnonzero(witness_patches == patch))

    charts = tuple(material_triangle(witness.exact_chart(row)) for row in witness_rows)
    roots = np.asarray(witness.source_root_cell_ids, dtype=np.int64)[list(witness_rows)]
    return witness_rows, charts, roots


def _affine_arguments(vertices: _Triangle, /) -> tuple[algebra.Expression, ...]:
    variables = algebra.axes(2)
    return tuple(
        algebra.add(
            algebra.constant(vertices[0][axis], 2),
            algebra.add(
                algebra.scale(variables[0], vertices[1][axis] - vertices[0][axis]),
                algebra.scale(variables[1], vertices[2][axis] - vertices[0][axis]),
            ),
        )
        for axis in range(2)
    )


def _absolute_expression_bound(expression: algebra.Expression, /) -> float:
    if isinstance(expression, algebra.RationalPolynomial):
        numerator = algebra.bernstein_coefficients(expression.numerator, "simplex", 2)
        denominator = algebra.bernstein_coefficients(expression.denominator, "simplex", 2)
        if not denominator or (min(denominator) <= 0 <= max(denominator)):
            raise ValueError("A rational surface map has an unresolved denominator.")
        lower = min(abs(value) for value in denominator)
        if lower <= 0:
            raise ValueError("A rational surface map has a singular denominator.")
        bound = max((abs(value) for value in numerator), default=Fraction(0)) / lower
    else:
        controls = algebra.bernstein_coefficients(expression, "simplex", 2)
        bound = max((abs(value) for value in controls), default=Fraction(0))
    result = float(bound)
    if not math.isfinite(result):
        raise ValueError("A surface chart enclosure is not finite.")
    return float(np.nextafter(result, np.inf)) if result else 0.0


def _coordinate_map_bank(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[tuple[algebra.Expression, ...], ...]:
    result = []
    for element, local, cell_row in mapped_geometry_cells(mesh, geometry):
        expressions = mapped_coordinate_expressions(element, local, cell_row=cell_row)
        result.append(expressions)
    if len(result) != sum(block.cell_count for block in mesh.blocks):
        raise ValueError("Surface coordinate maps do not cover every physical cell.")
    return tuple(result)


def _chord_bound(expression: algebra.Expression, /) -> float:
    corners = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    values = tuple(algebra.evaluate_expression(expression, point) for point in corners)
    chord = algebra.add(
        algebra.constant(values[0], 2),
        algebra.add(
            algebra.scale(algebra.axes(2)[0], values[1] - values[0]),
            algebra.scale(algebra.axes(2)[1], values[2] - values[0]),
        ),
    )
    difference = algebra.subtract(expression, chord)
    return _absolute_expression_bound(difference)


def _polynomial_chord_bounds(
    coordinate_maps: tuple[tuple[algebra.Expression, ...], ...], /
) -> _FloatArray:
    return np.asarray(
        [
            sum(_chord_bound(component) for component in values)
            for values in coordinate_maps
        ],
        dtype=np.float64,
    )


def _displacement_bounds(
    source_maps: tuple[tuple[algebra.Expression, ...], ...],
    target_maps: tuple[tuple[algebra.Expression, ...], ...],
    pieces: tuple[PreparedSurfaceChartPiece, ...],
    source_rows: _IntArray,
    target_rows: _IntArray,
    /,
) -> _FloatArray:
    result = np.empty((len(pieces),), dtype=np.float64)
    for index, piece in enumerate(pieces):
        source_arguments = _affine_arguments(piece.exact_source_reference_vertices)
        target_arguments = _affine_arguments(piece.exact_target_reference_vertices)
        old = source_maps[int(source_rows[piece.source_row])]
        new = target_maps[int(target_rows[piece.target_row])]
        if len(old) != len(new):
            raise ValueError(
                "Surface deformation endpoints have different ambient dimensions."
            )
        bound = 0.0
        for source_coordinate, target_coordinate in zip(old, new, strict=True):
            source_on_piece = algebra.compose_expression(
                source_coordinate, source_arguments
            )
            target_on_piece = algebra.compose_expression(
                target_coordinate, target_arguments
            )
            bound += _absolute_expression_bound(
                algebra.subtract(source_on_piece, target_on_piece)
            )
        result[index] = float(np.nextafter(bound, np.inf)) if bound else 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError("Surface material displacement bounds are not finite.")
    return result


def _measure_enclosures(
    validity: CellValidityCertificate,
    mesh: CellMesh,
    /,
) -> tuple[_FloatArray, _FloatArray]:
    lower = np.asarray(validity.determinant_lower, dtype=np.float64)
    upper = np.asarray(validity.determinant_upper, dtype=np.float64)
    count = sum(block.cell_count for block in mesh.blocks)
    if (
        lower.shape != (count,)
        or upper.shape != (count,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower <= 0.0)
        or np.any(upper < lower)
    ):
        raise ValueError(
            "Surface validity proof has no positive mapped-measure enclosure."
        )
    if mesh.ambient_dimension == 2:
        density_lower = lower
        density_upper = upper
    else:
        density_lower = np.sqrt(lower)
        density_upper = np.sqrt(upper)
    scale = 0.5
    low = np.nextafter(scale * density_lower, -np.inf)
    high = np.nextafter(scale * density_upper, np.inf)
    low = np.maximum(low, np.finfo(np.float64).tiny)
    values = low + 0.5 * (high - low)
    errors = 0.5 * (high - low) + 64.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, values
    )
    if np.any(values <= errors) or not np.all(np.isfinite(values + errors)):
        raise ValueError("Surface physical cell measures are not positively enclosed.")
    return values, errors


def _fidelity_upper_by_cell(
    certificate: SourceFidelityCertificate,
    mesh: CellMesh,
    /,
) -> _FloatArray:
    identities = tuple(int(value) for value in certificate.source_cell_global_ids)
    lookup = {identity: row for row, identity in enumerate(identities)}
    physical_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    bounds = np.asarray(certificate.deviation_bounds, dtype=np.float64)
    if bounds.shape != (len(identities), 2):
        raise ValueError("Surface fidelity proof does not align with its cells.")
    try:
        rows = np.asarray(
            [lookup[int(identity)] for identity in physical_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("Surface fidelity proof omits an endpoint cell.") from error
    return bounds[rows, 1]


def _endpoint_proofs(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    witness: SurfaceChartWitness,
    cover: MeshingDomainBoundarySource,
    maximum_fidelity: float,
    /,
    *,
    validity: CellValidityCertificate | None,
    validity_policy: CellValidityPolicy | None,
    embedding: GlobalEmbeddingCertificate | None,
    fidelity: SourceFidelityCertificate | None,
) -> _EndpointProofs:
    if cover.domain.domain_id != domain.domain_id:
        raise ValueError("Surface boundary chart cover belongs to a different domain.")
    local_validity = (
        certify_cell_geometry_validity(geometry, mesh=mesh, policy=validity_policy)
        if validity is None
        else validity
    )
    local_validity.require_bound(geometry, mesh=mesh)
    if not local_validity.all_certified:
        raise ValueError("Surface deformation requires locally valid endpoint cells.")
    global_embedding = (
        certify_global_embedding(mesh, geometry, local_validity)
        if embedding is None
        else embedding
    )
    global_embedding.require_current(mesh, geometry, validity=local_validity)
    if global_embedding.status != "certified":
        raise ValueError("Surface deformation requires globally embedded endpoints.")
    source_fidelity = (
        certify_source_fidelity(
            mesh,
            geometry,
            cover,
            tolerance=maximum_fidelity,
            witness=witness,
        )
        if fidelity is None
        else fidelity
    )
    source_fidelity.require_current(mesh, geometry, cover)
    if source_fidelity.status != "certified":
        raise ValueError("Surface deformation requires certified source fidelity.")
    upper = np.asarray(source_fidelity.deviation_bounds, dtype=np.float64)[:, 1]
    if np.any(upper > maximum_fidelity):
        raise ValueError("Surface endpoint fidelity exceeds maximum_fidelity.")
    return _EndpointProofs(local_validity, global_embedding, source_fidelity)


def _deformation_identity(
    source_witness: SurfaceChartWitness,
    target_witness: SurfaceChartWitness,
    root_witness: SurfaceChartWitness,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    domain: MeshingDomain,
    source_bounds: _FloatArray,
    target_bounds: _FloatArray,
    polynomial_chord_bounds: _FloatArray,
    corner_bounds: _FloatArray,
    continuous_chord_bounds: _FloatArray,
    source_measures: _FloatArray,
    target_measures: _FloatArray,
    source_errors: _FloatArray,
    target_errors: _FloatArray,
    source_proofs: _EndpointProofs,
    target_proofs: _EndpointProofs,
    occurrences: tuple[PreparedSurfaceChartOccurrence, ...],
    maximum_displacement: float,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "prepared-surface-chart-deformation",
            "source": source_witness.witness_id,
            "target": target_witness.witness_id,
            "material_root": root_witness.witness_id,
            "source_geometry": source_witness.geometry_id,
            "target_geometry": target_witness.geometry_id,
            "source_mesh": source_mesh.mesh_id,
            "target_mesh": target_mesh.mesh_id,
            "domain": domain.domain_id,
            "overlaps": tuple(
                tuple(piece.piece_id for piece in item.pieces) for item in occurrences
            ),
            "root_partitions": tuple(
                tuple(piece.piece_id for piece in item.root_pieces)
                for item in occurrences
            ),
            "certificates": (
                source_proofs.validity.certificate_id,
                target_proofs.validity.certificate_id,
                source_proofs.embedding.certificate_id,
                target_proofs.embedding.certificate_id,
                source_proofs.fidelity.certificate_id,
                target_proofs.fidelity.certificate_id,
            ),
            "source_bounds": array_tree_fingerprint(source_bounds),
            "target_bounds": array_tree_fingerprint(target_bounds),
            "source_polynomial_chord_bounds": array_tree_fingerprint(
                polynomial_chord_bounds
            ),
            "source_corner_enclosure_bounds": array_tree_fingerprint(corner_bounds),
            "source_continuous_chord_bounds": array_tree_fingerprint(
                continuous_chord_bounds
            ),
            "source_cell_measures": array_tree_fingerprint(source_measures),
            "target_cell_measures": array_tree_fingerprint(target_measures),
            "source_measure_errors": array_tree_fingerprint(source_errors),
            "target_measure_errors": array_tree_fingerprint(target_errors),
            "occurrences": tuple(
                (
                    item.patch,
                    item.geometry_entity_id,
                    item.occurrence_path,
                    array_tree_fingerprint(item.displacement_bounds),
                )
                for item in occurrences
            ),
            "displacement": maximum_displacement,
        }
    )


@checked
def prepare_surface_chart_deformation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    domain: MeshingDomain,
    /,
    *,
    source_witness: SurfaceChartWitness,
    target_witness: SurfaceChartWitness,
    source_chart_cover: MeshingDomainBoundarySource,
    target_chart_cover: MeshingDomainBoundarySource,
    maximum_fidelity: float,
    maximum_displacement: float,
    material_root_witness: SurfaceChartWitness | None = None,
    material_root_chart_cover: MeshingDomainBoundarySource | None = None,
    prepared_source_validity: CellValidityCertificate | None = None,
    prepared_target_validity: CellValidityCertificate | None = None,
    prepared_source_embedding: GlobalEmbeddingCertificate | None = None,
    prepared_target_embedding: GlobalEmbeddingCertificate | None = None,
    prepared_source_fidelity: SourceFidelityCertificate | None = None,
    prepared_target_fidelity: SourceFidelityCertificate | None = None,
    validity_policy: CellValidityPolicy | None = None,
) -> PreparedSurfaceChartDeformation:
    """Prepare exact source/target material overlaps and endpoint certificates.

    Material-chart cell intersections are clipped as exact rational halfplanes
    in source reference coordinates. Physical displacement is enclosed with the
    current coordinate-map Bernstein expressions; endpoint validity, global
    embedding, source fidelity, and physical measures are retained from their
    owning certificate/integration substrates. No work receipt or request quota
    participates in the result.
    """
    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("Surface chart endpoints must be CellMesh values.")
    if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError("Surface chart endpoints require CellGeometrySpec maps.")
    fidelity_limit = float(maximum_fidelity)
    displacement_limit = float(maximum_displacement)
    if (
        not math.isfinite(fidelity_limit)
        or fidelity_limit < 0.0
        or not math.isfinite(displacement_limit)
        or displacement_limit < 0.0
    ):
        raise ValueError(
            "Surface fidelity and displacement limits must be finite and nonnegative."
        )
    if source_mesh.topological_dimension != 2 or target_mesh.topological_dimension != 2:
        raise ValueError("Surface chart deformations require two-dimensional meshes.")
    source_bank = _witness_bank(source_witness, source_mesh, source_geometry, domain)
    target_bank = _witness_bank(target_witness, target_mesh, target_geometry, domain)
    root_witness = (
        source_witness if material_root_witness is None else material_root_witness
    )
    if root_witness.domain_id != domain.domain_id:
        raise ValueError("Material root witness belongs to a different domain.")
    if root_witness.witness_id == source_witness.witness_id:
        root_cover = (
            source_chart_cover
            if material_root_chart_cover is None
            else material_root_chart_cover
        )
    else:
        if material_root_chart_cover is None:
            raise ValueError(
                "A distinct material root witness requires its retained boundary chart cover."
            )
        root_cover = material_root_chart_cover
    if root_cover.domain.domain_id != domain.domain_id:
        raise ValueError("Material root chart cover belongs to a different domain.")
    source_proofs = _endpoint_proofs(
        source_mesh,
        source_geometry,
        domain,
        source_witness,
        source_chart_cover,
        fidelity_limit,
        validity=prepared_source_validity,
        validity_policy=validity_policy,
        embedding=prepared_source_embedding,
        fidelity=prepared_source_fidelity,
    )
    target_proofs = _endpoint_proofs(
        target_mesh,
        target_geometry,
        domain,
        target_witness,
        target_chart_cover,
        fidelity_limit,
        validity=prepared_target_validity,
        validity_policy=validity_policy,
        embedding=prepared_target_embedding,
        fidelity=prepared_target_fidelity,
    )
    if validity_policy is not None and (
        source_proofs.validity.policy_id != validity_policy.policy_id
        or target_proofs.validity.policy_id != validity_policy.policy_id
    ):
        raise ValueError("Prepared endpoint validity uses a different validity policy.")
    source_measures, source_errors = _measure_enclosures(
        source_proofs.validity, source_mesh
    )
    target_measures, target_errors = _measure_enclosures(
        target_proofs.validity, target_mesh
    )
    source_maps = _coordinate_map_bank(source_mesh, source_geometry)
    target_maps = _coordinate_map_bank(target_mesh, target_geometry)
    source_fidelity_bounds = _fidelity_upper_by_cell(source_proofs.fidelity, source_mesh)
    target_fidelity_bounds = _fidelity_upper_by_cell(target_proofs.fidelity, target_mesh)
    polynomial_chord_bounds = _polynomial_chord_bounds(source_maps)
    continuous_chord_bounds = np.empty_like(polynomial_chord_bounds)
    corner_bounds = source_fidelity_bounds.copy()
    source_occurrences: list[PreparedSurfaceChartOccurrence] = []
    source_patches = tuple(sorted(source_bank.physical_rows_by_patch))
    if source_patches != tuple(sorted(target_bank.physical_rows_by_patch)):
        raise ValueError("Surface chart endpoints cover different source patches.")
    maximum_bound = 0.0
    source_cursor = 0
    for patch in source_patches:
        source_rows = source_bank.physical_rows_by_patch[patch]
        target_rows = target_bank.physical_rows_by_patch[patch]
        source_charts = tuple(source_bank.charts_by_physical[row] for row in source_rows)
        target_charts = tuple(target_bank.charts_by_physical[row] for row in target_rows)
        source_ids = source_bank.physical_ids[list(source_rows)]
        target_ids = target_bank.physical_ids[list(target_rows)]
        source_witness_rows = source_bank.witness_rows_by_physical[list(source_rows)]
        target_witness_rows = target_bank.witness_rows_by_physical[list(target_rows)]
        source_root_ids = np.asarray(source_witness.source_root_cell_ids, dtype=np.int64)[
            source_witness_rows
        ]
        target_root_ids = np.asarray(target_witness.source_root_cell_ids, dtype=np.int64)[
            target_witness_rows
        ]
        root_rows, root_charts, root_root_ids = _chart_patch_rows(root_witness, patch)
        root_ids = np.asarray(root_witness.cell_global_ids, dtype=np.int64)[
            list(root_rows)
        ]
        if root_witness.witness_id == source_witness.witness_id:
            source_charts_aligned = source_charts
            root_pieces: tuple[PreparedSurfaceChartPiece, ...] = ()
        else:
            source_charts_aligned = _align_periodic_charts(
                root_charts,
                source_charts,
                np.asarray(root_root_ids, dtype=np.int64),
                source_root_ids,
                domain.patches[patch].surface.periods,
            )
            root_pieces = _overlap_pieces(
                root_charts,
                source_charts_aligned,
                root_ids,
                source_ids,
            ).pieces
        target_charts_aligned = _align_periodic_charts(
            source_charts_aligned,
            target_charts,
            source_root_ids,
            target_root_ids,
            domain.patches[patch].surface.periods,
        )
        overlap = _overlap_pieces(
            source_charts_aligned,
            target_charts_aligned,
            source_ids,
            target_ids,
        )
        target_charts = target_charts_aligned
        displacement = _displacement_bounds(
            source_maps,
            target_maps,
            overlap.pieces,
            np.asarray(source_rows, dtype=np.int64),
            np.asarray(target_rows, dtype=np.int64),
        )
        maximum_bound = max(maximum_bound, float(np.max(displacement, initial=0.0)))
        if maximum_bound > displacement_limit:
            raise ValueError(
                "Surface material displacement exceeds maximum_displacement."
            )
        source_occurrences.append(
            PreparedSurfaceChartOccurrence(
                patch=patch,
                geometry_entity_id=domain.entity_id(2, patch),
                occurrence_path=domain.source_occurrences[2][patch],
                source_rows=jnp.asarray(source_rows, dtype=jnp.int64),
                target_rows=jnp.asarray(target_rows, dtype=jnp.int64),
                exact_source_material_charts=source_charts_aligned,
                exact_target_material_charts=target_charts,
                pieces=overlap.pieces,
                root_pieces=root_pieces,
                material_root_witness_id=root_witness.witness_id,
                displacement_bounds=jnp.asarray(displacement, dtype=jnp.float64),
                source_domain_coverage="certified",
                target_domain_coverage="certified",
            )
        )
        for local_row, physical_row in enumerate(source_rows):
            continuous = domain.interpolation_bounds(
                patch,
                np.asarray([source_charts_aligned[local_row]], dtype=np.float64),
            )[0]
            continuous_bound = float(continuous)
            if not math.isfinite(continuous_bound) or continuous_bound < 0.0:
                raise ValueError("Surface source lacks a finite continuous chord bound.")
            continuous_chord_bounds[physical_row] = continuous_bound
        source_cursor += len(source_rows)
    if source_cursor != source_bank.physical_ids.size:
        raise ValueError("Surface material occurrences omit source cells.")
    occurrences = tuple(source_occurrences)
    deformation_id = _deformation_identity(
        source_witness,
        target_witness,
        root_witness,
        source_mesh,
        target_mesh,
        domain,
        source_fidelity_bounds,
        target_fidelity_bounds,
        polynomial_chord_bounds,
        corner_bounds,
        continuous_chord_bounds,
        source_measures,
        target_measures,
        source_errors,
        target_errors,
        source_proofs,
        target_proofs,
        occurrences,
        maximum_bound,
    )
    deformation = PreparedSurfaceChartDeformation(
        source_witness=source_witness,
        target_witness=target_witness,
        material_root_witness=root_witness,
        source_chart_cover=source_chart_cover,
        material_root_chart_cover=root_cover,
        target_chart_cover=target_chart_cover,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        occurrences=occurrences,
        source_geometry_id=cell_geometry_id(source_geometry),
        target_geometry_id=cell_geometry_id(target_geometry),
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_fidelity_bounds=jnp.asarray(source_fidelity_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_fidelity_bounds, dtype=jnp.float64),
        source_polynomial_chord_bounds=jnp.asarray(
            polynomial_chord_bounds, dtype=jnp.float64
        ),
        source_corner_enclosure_bounds=jnp.asarray(corner_bounds, dtype=jnp.float64),
        source_continuous_chord_bounds=jnp.asarray(
            continuous_chord_bounds, dtype=jnp.float64
        ),
        source_cell_measures=jnp.asarray(source_measures, dtype=jnp.float64),
        target_cell_measures=jnp.asarray(target_measures, dtype=jnp.float64),
        source_measure_errors=jnp.asarray(source_errors, dtype=jnp.float64),
        target_measure_errors=jnp.asarray(target_errors, dtype=jnp.float64),
        source_validity=source_proofs.validity,
        target_validity=target_proofs.validity,
        source_embedding=source_proofs.embedding,
        target_embedding=target_proofs.embedding,
        source_fidelity_certificate=source_proofs.fidelity,
        target_fidelity_certificate=target_proofs.fidelity,
        maximum_displacement_bound=maximum_bound,
        deformation_id=deformation_id,
    )
    deformation.require_bound(source_mesh, source_geometry, target_mesh, target_geometry)
    return deformation


__all__ = ["prepare_surface_chart_deformation"]

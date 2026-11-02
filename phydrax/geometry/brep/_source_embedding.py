#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact structural embedding proofs for analytic source maps.

This owner proves a property of a carrier on an *entire exact rectangle*.  It
neither proves that a trim or a homotopy lies in that rectangle nor that the
requested quotient is present in CAD topology.  Those are separate obligations
of the consumer.  In particular a rounded numerical period is not a period.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from .._rational import rational_vector
from ._assembly import _determinant3
from ._boolean_core import _chart_sign, chart_add, chart_parts, chart_scale, ChartScalar
from ._carrier_pieces import carrier_content_id
from ._patches import (
    _exact_cross3,
    _exact_dot3,
    _prepare_revolution_line_profile,
    _require_exact_numeric_shadow,
    _validate_surface_source_shadows,
    AbstractCurve,
    AbstractSurfacePatch,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from ._placed import PlacedSurface
from ._trim_helpers import PeriodOffset

if TYPE_CHECKING:
    from .._meshing_domain import SourcePreparation
    from .._surface_source_support import _CadCurveUseBinding, SurfaceSourceCharts


type ExactChartBox = tuple[
    tuple[ChartScalar, ChartScalar], tuple[ChartScalar, ChartScalar]
]
type _Family = Literal[
    "plane", "cylinder", "revolution-cylinder", "revolution-plane", "sphere", "ring-torus"
]

_TAU = PeriodOffset(Fraction(0), Fraction(1))
_QUARTER = PeriodOffset(Fraction(0), Fraction(1, 4))


def _difference(a: ChartScalar, b: ChartScalar, /) -> ChartScalar:
    return chart_add(a, chart_scale(b, Fraction(-1)))


def _chart_scalar(value: object, /) -> bool:
    return isinstance(value, Fraction) or (
        isinstance(value, PeriodOffset)
        and isinstance(value.rational, Fraction)
        and isinstance(value.turns, Fraction)
    )


def _sheet_sign(value: ChartScalar, /) -> int | None:
    """Refine an undecided period comparison using directed high-precision π."""
    sign = _chart_sign(value)
    if sign is not None:
        return sign
    from ._intersection_curve import interval_context

    rational, turns = chart_parts(value)
    endpoints = interval_context(50).pi._mpi_

    def fraction(endpoint: tuple[int, int, int, int], /) -> Fraction:
        negative, mantissa, exponent, _ = endpoint
        value = Fraction(-mantissa if negative else mantissa)
        return value * (2**exponent) if exponent >= 0 else value / (2**-exponent)

    low_pi, high_pi = (fraction(endpoint) for endpoint in endpoints)
    first, last = rational + 2 * turns * low_pi, rational + 2 * turns * high_pi
    low, high = min(first, last), max(first, last)
    if low > 0:
        return 1
    if high < 0:
        return -1
    return None


def _same(a: ChartScalar, b: ChartScalar, /) -> bool:
    return chart_parts(a) == chart_parts(b)


def _positive(value: ExactScalar, /) -> bool:
    return value.sign() > 0 if isinstance(value, RadicalElement) else value > 0


def _represented_vector(value: object, /) -> tuple[Fraction, ...] | None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return rational_vector(array)


def _frame(
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    axis: tuple[ExactScalar, ...],
    /,
) -> bool:
    # Orthogonality is unnecessary: an invertible linear image preserves the
    # embedding and regularity of the standard analytic parameterization.
    return _exact_dot3(_exact_cross3(first, second), axis) != 0


def _family(surface: AbstractSurfacePatch, /) -> _Family | None:
    # Imported lazily because meshing-domain construction itself imports B-Rep.
    from .._meshing_domain import _PlacedSurfacePatch

    patch = surface
    while isinstance(patch, (PlacedSurface, _PlacedSurfacePatch)):
        if type(patch) not in (PlacedSurface, _PlacedSurfacePatch):
            return None
        _validate_surface_source_shadows(patch)
        if isinstance(patch, PlacedSurface):
            matrix = patch.affine.matrix
            child = patch.definition
        else:
            matrix = patch.placement.matrix
            _require_exact_numeric_shadow(
                patch.linear,
                tuple(row[:3] for row in matrix[:3]),
                "_PlacedSurfacePatch.linear",
            )
            _require_exact_numeric_shadow(
                patch.offset,
                tuple(row[3] for row in matrix[:3]),
                "_PlacedSurfacePatch.offset",
            )
            child = patch.source
        if _determinant3(tuple(row[:3] for row in matrix[:3])) == 0:
            return None
        patch = child
    _validate_surface_source_shadows(patch)
    if type(patch) is PlanePatch:
        origin = _represented_vector(patch.origin)
        first = _represented_vector(patch.first_axis)
        second = _represented_vector(patch.second_axis)
        if origin is None or first is None or second is None:
            return None
        return "plane" if any(_exact_cross3(first, second)) else None
    if type(patch) is CylinderPatch:
        if _positive(patch.radius_exact) and _frame(
            patch.first_axis_exact, patch.second_axis_exact, patch.axis_exact
        ):
            return "cylinder"
    elif type(patch) is TorusPatch:
        if (
            _positive(patch.minor_radius_exact)
            and _positive(patch.major_radius_exact - patch.minor_radius_exact)
            and _frame(patch.first_axis_exact, patch.second_axis_exact, patch.axis_exact)
        ):
            return "ring-torus"
    elif type(patch) is SpherePatch:
        center = _represented_vector(patch.center)
        first = _represented_vector(patch.first_axis)
        second = _represented_vector(patch.second_axis)
        axis = _represented_vector(patch.axis)
        radius = np.asarray(patch.radius, dtype=np.float64)
        if (
            center is not None
            and first is not None
            and second is not None
            and axis is not None
            and radius.shape == ()
            and np.isfinite(radius)
            and float(radius) > 0
            and _frame(first, second, axis)
        ):
            return "sphere"
    elif type(patch) is RevolutionSurface and type(patch.curve) is LineCurve:
        line = _prepare_revolution_line_profile(patch, patch.curve)
        if (
            line is not None
            and line.axial_direction != 0
            and all(value == 0 for value in line.radial_direction)
            and any(value != 0 for value in line.radial_origin)
        ):
            return "revolution-cylinder"
        if (
            line is not None
            and line.axial_direction == 0
            and any(value != 0 for value in line.radial_direction)
        ):
            return "revolution-plane"
    return None


def _revolution_plane_pole(surface: AbstractSurfacePatch, /) -> Fraction | None:
    """Exact profile parameter of the unique polar fiber of a radial line."""
    from .._meshing_domain import _PlacedSurfacePatch

    while isinstance(surface, (PlacedSurface, _PlacedSurfacePatch)):
        surface = surface.definition if isinstance(surface, PlacedSurface) else surface.source
    if not isinstance(surface, RevolutionSurface):
        return None
    line = _prepare_revolution_line_profile(surface, surface.curve)
    if line is None or line.axial_direction != 0:
        return None
    for origin, direction in zip(line.radial_origin, line.radial_direction, strict=True):
        if direction:
            root = -origin / direction
            if isinstance(root, Fraction) and all(
                a + root * b == 0
                for a, b in zip(line.radial_origin, line.radial_direction, strict=True)
            ):
                return root
            return None
    return None


@dataclass(frozen=True, slots=True, eq=False)
class SourceEmbeddingProof:
    """Replayable carrier theorem, conditional on domain and quotient authority.

    ``bounds`` contains exact chart coordinates, not numerical enclosures.
    ``deck_axes`` identifies precisely the opposing rectangle sides separated by
    symbolic native τ. ``pole_roots`` identifies complete latitude fibers at
    the listed exact sphere pole roots. Regularity holds away from those fibers.
    The caller must prove that the actual source/swept domain is contained in
    this box and that every declared identification is authorized by the same
    source topology; this record does not assert either fact.
    """

    source: AbstractSurfacePatch
    source_id: str
    family: _Family
    bounds: ExactChartBox
    deck_axes: tuple[int, ...]
    pole_roots: tuple[ChartScalar, ...]

    def replay(
        self,
        surface: AbstractSurfacePatch,
        bounds: ExactChartBox,
        /,
        *,
        deck_axes: tuple[int, ...] = (),
        pole_roots: tuple[ChartScalar, ...] = (),
    ) -> bool:
        if surface is not self.source:
            return False
        proof = certify_source_embedding(
            surface, bounds, deck_axes=deck_axes, pole_roots=pole_roots
        )
        return (
            proof is not None
            and proof.source_id == self.source_id
            and proof.family == self.family
            and proof.bounds == self.bounds
            and proof.deck_axes == self.deck_axes
            and proof.pole_roots == self.pole_roots
        )


def certify_source_embedding(
    surface: AbstractSurfacePatch,
    bounds: ExactChartBox,
    /,
    *,
    deck_axes: tuple[int, ...] = (),
    pole_roots: tuple[ChartScalar, ...] = (),
) -> SourceEmbeddingProof | None:
    """Prove analytic sheet embedding, or return None when not established.

    Inputs use only Fraction and PeriodOffset; floats are intentionally refused.
    Bounds must be strictly ordered. Every full-turn side pairing and every
    sphere pole on the box must be explicitly declared, and no other pairing or
    pole is accepted. Comparison uses directed period bounds and fails closed.
    """
    if not isinstance(surface, AbstractSurfacePatch):
        return None
    if (
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(not isinstance(row, tuple) or len(row) != 2 for row in bounds)
        or any(not _chart_scalar(value) for row in bounds for value in row)
    ):
        return None
    if (
        not isinstance(deck_axes, tuple)
        or any(type(axis) is not int for axis in deck_axes)
        or tuple(sorted(set(deck_axes))) != deck_axes
        or not isinstance(pole_roots, tuple)
        or any(not _chart_scalar(root) for root in pole_roots)
    ):
        return None
    try:
        _validate_surface_source_shadows(surface)
        family = _family(surface)
        if family is None:
            return None
        widths = tuple(_difference(bounds[1][i], bounds[0][i]) for i in range(2))
        if any(_sheet_sign(width) != 1 for width in widths):
            return None
        periodic = (
            (0, 1) if family == "ring-torus" else (() if family == "plane" else (0,))
        )
        expected_decks = []
        for axis in periodic:
            relative = _difference(widths[axis], _TAU)
            sign = _sheet_sign(relative)
            if sign is None or sign > 0:
                return None
            if sign == 0:
                expected_decks.append(axis)
        if tuple(expected_decks) != deck_axes:
            return None
        expected_poles: list[ChartScalar] = []
        if family == "sphere":
            south = chart_scale(_QUARTER, Fraction(-1))
            if _sheet_sign(_difference(bounds[0][1], south)) not in (0, 1) or _sheet_sign(
                _difference(_QUARTER, bounds[1][1])
            ) not in (0, 1):
                return None
            if _same(bounds[0][1], south):
                expected_poles.append(south)
            if _same(bounds[1][1], _QUARTER):
                expected_poles.append(_QUARTER)
        elif family == "revolution-plane":
            root = _revolution_plane_pole(surface)
            low, high = bounds[0][1], bounds[1][1]
            if root is None or not isinstance(low, Fraction) or not isinstance(high, Fraction):
                return None
            if low < root < high:
                return None
            if root in (low, high):
                expected_poles.append(root)
        if len(pole_roots) != len(expected_poles) or any(
            not _same(actual, expected)
            for actual, expected in zip(pole_roots, expected_poles, strict=True)
        ):
            return None
        return SourceEmbeddingProof(
            surface, carrier_content_id(surface), family, bounds, deck_axes, pole_roots
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True, slots=True, eq=False)
class CadTrimSheetProof:
    """Source-sheet proof bound to a current CAD chart and reference chain."""

    charts: object
    chart_id: str
    patch: int
    reference_id: str
    embedding: SourceEmbeddingProof

    def replay(self, charts: object, patch: int, tolerance: float, /) -> bool:
        if charts is not self.charts or patch != self.patch:
            return False
        proof = certify_cad_trim_sheet(charts, patch, tolerance)
        return (
            proof is not None
            and proof.chart_id == self.chart_id
            and proof.reference_id == self.reference_id
            and proof.embedding.source_id == self.embedding.source_id
            and proof.embedding.bounds == self.embedding.bounds
            and proof.embedding.deck_axes == self.embedding.deck_axes
        )


def _constant_chart_line(
    curve: AbstractCurve, axis: int, /
) -> tuple[ChartScalar, ChartScalar, Fraction] | None:
    """Exact constant coordinate and affine transverse coordinate, never a shadow."""
    from ._boolean_core import line_chart_coefficients

    coefficients = line_chart_coefficients(curve)
    if coefficients is None:
        return None
    origin, direction = coefficients
    if len(origin) != 2 or len(direction) != 2 or direction[axis] != 0:
        return None
    other = 1 - axis
    return origin[axis], origin[other], direction[other]


@dataclass(frozen=True, slots=True, eq=False)
class BRepTrimSheetProof:
    """Exact carrier sheet and CAD quotient bound to a B-Rep domain occurrence."""

    domain: object
    domain_id: str
    patch: int
    embedding: SourceEmbeddingProof

    def replay(self, domain: object, patch: int, /) -> bool:
        if domain is not self.domain or patch != self.patch:
            return False
        proof = certify_brep_trim_sheet(domain, patch)
        return (
            proof is not None
            and proof.domain_id == self.domain_id
            and proof.embedding.source_id == self.embedding.source_id
            and proof.embedding.bounds == self.embedding.bounds
            and proof.embedding.deck_axes == self.embedding.deck_axes
        )


def certify_brep_trim_sheet(domain: object, patch: int, /) -> BRepTrimSheetProof | None:
    """Prove a cylindrical or exact polar-plane CAD source sheet.

    The full-turn quotient requires opposite coedge uses of one source edge on
    the exact constant lines 0 and τ. Root identity with those lines fixes an
    overhanging curved use's endpoint coordinates; interval derivatives then
    prove the entire use remains between them. The proof binds this actual
    domain occurrence; it does not identify the trimmed interior, prove a
    separate solver region has the same symbolic full-period domain, or
    establish trim homotopy/partition.
    """
    from .._domain_boundary import (
        _exact_line_junction_box,
        _rooted_patch_chart_endpoints,
        exact_plane_chart,
    )
    from .._meshing_domain import (
        _same_brep_trim_endpoint,
        MeshingDomain,
        PatchCurveUse,
        PatchPoleUse,
        SourcePreparation,
    )

    if not isinstance(domain, MeshingDomain) or type(patch) is not int:
        return None
    if not 0 <= patch < len(domain.patches) or domain.brep_authority is None:
        return None
    try:
        domain.require_current()
        authority = domain.brep_authority
        face = domain.source_indices[2][patch]
        surface = domain.patches[patch].surface
        if _family(surface) == "revolution-plane":
            plane = exact_plane_chart(domain, patch)
            if plane is None or surface.native_period_symbols()[0] != "two_pi":
                return None
            endpoints = _rooted_patch_chart_endpoints(domain, patch)
            values = tuple(point[1] for loop in endpoints for edge in loop for point in edge)
            if not values or min(values) == max(values):
                return None
            low, high = min(values), max(values)
            root = _revolution_plane_pole(surface)
            if root is None:
                return None
            poles = [
                edge for loop, sides in zip(domain.patches[patch].loops, endpoints, strict=True)
                for use, edge in zip(loop, sides, strict=True) if isinstance(use, PatchPoleUse)
            ]
            expected_poles = (root,) if root in (low, high) else ()
            if len(poles) != len(expected_poles) or any(
                a[1] != root or b[1] != root for a, b in poles
            ):
                return None
            embedding = certify_source_embedding(
                surface, ((Fraction(0), low), (_TAU, high)),
                deck_axes=(0,), pole_roots=expected_poles,
            )
            if embedding is None:
                return None
            return BRepTrimSheetProof(domain, domain.domain_id, patch, embedding)
        if _family(surface) not in ("cylinder", "revolution-cylinder"):
            return None
        uses = domain.patches[patch].loops
        coedges = authority.geometry.face_loops[face]
        if (
            not uses
            or len(uses) != len(coedges)
            or any(
                not loop or len(loop) != len(edges)
                for loop, edges in zip(uses, coedges, strict=True)
            )
        ):
            return None
        preparation = SourcePreparation(domain)
        lines = {}
        ranges = {}
        boxes = {}
        for li, loop in enumerate(uses):
            for ui, use in enumerate(loop):
                line = _constant_chart_line(use.pcurve, 0)
                if line is not None:
                    lines[(li, ui)] = line
                first, last = min(use.first, use.last), max(use.first, use.last)
                for root in authority.geometry.coedge_endpoint_roots[coedges[li][ui]]:
                    if root is not None:
                        low, high = root.parameter_enclosure()
                        if not np.isfinite(low) or not np.isfinite(high) or low > high:
                            return None
                        first, last = min(first, low), max(last, high)
                # A root on this carrier is necessarily in its authored domain.
                carrier_domain = use.pcurve.parameter_domain
                if carrier_domain is not None:
                    first = max(first, carrier_domain[0])
                    last = min(last, carrier_domain[1])
                if not first < last:
                    return None
                box = np.asarray(preparation.curve_box(use.pcurve, first, last))
                if (
                    box.shape != (2, 2)
                    or np.any(~np.isfinite(box))
                    or np.any(box[0] > box[1])
                ):
                    return None
                ranges[(li, ui)] = (first, last)
                boxes[(li, ui)] = box
        v_low = min(Fraction.from_float(float(box[0, 1])) for box in boxes.values())
        v_high = max(Fraction.from_float(float(box[1, 1])) for box in boxes.values())
        if v_low == v_high:
            return None
        seam_pairs = []
        for first, line in lines.items():
            if not _same(line[0], Fraction(0)):
                continue
            low_use = uses[first[0]][first[1]]
            low_edge = coedges[first[0]][first[1]]
            for second, other in lines.items():
                if not _same(other[0], _TAU):
                    continue
                high_use = uses[second[0]][second[1]]
                high_edge = coedges[second[0]][second[1]]
                if (
                    isinstance(low_use, PatchCurveUse)
                    and isinstance(high_use, PatchCurveUse)
                    and authority.geometry.coedge_edges[low_edge]
                    == authority.geometry.coedge_edges[high_edge]
                    and low_use.sense == -high_use.sense
                    and low_use.first == high_use.first
                    and low_use.last == high_use.last
                    and _same(line[1], other[1])
                    and line[2] == other[2]
                ):
                    seam_pairs.append((first, second))
        if len(seam_pairs) > 1:
            return None
        full_turn = bool(seam_pairs)

        def endpoint_seam(li: int, ui: int, start: bool) -> ChartScalar | None:
            loop, edges = uses[li], coedges[li]
            neighbor = (ui - 1) % len(loop) if start else (ui + 1) % len(loop)
            line = lines.get((li, neighbor))
            if line is None or not (_same(line[0], Fraction(0)) or _same(line[0], _TAU)):
                return None
            a, b = (neighbor, ui) if start else (ui, neighbor)
            if not _same_brep_trim_endpoint(
                authority, face, edges[a], edges[b], loop[a], loop[b]
            ) and _exact_line_junction_box(
                authority, face, edges[a], edges[b], loop[a], loop[b]
            ) is None:
                return None
            return line[0]

        u_low: Fraction | None = None
        u_high: Fraction | None = None
        for li, loop in enumerate(uses):
            for ui, use in enumerate(loop):
                line = lines.get((li, ui))
                if (
                    full_turn
                    and line is not None
                    and (_same(line[0], Fraction(0)) or _same(line[0], _TAU))
                ):
                    continue
                box = boxes[(li, ui)]
                low, high = (Fraction.from_float(float(box[row, 0])) for row in (0, 1))
                if full_turn and (low < 0 or _sheet_sign(_difference(high, _TAU)) != -1):
                    start, stop = (
                        endpoint_seam(li, ui, True),
                        endpoint_seam(li, ui, False),
                    )
                    if start is None or stop is None or _same(start, stop):
                        return None
                    expected = 1 if _same(start, Fraction(0)) else -1
                    expected *= use.sense * (1 if use.last > use.first else -1)
                    first, last = ranges[(li, ui)]
                    lower, upper = preparation.curve_derivative_bounds(
                        use.pcurve, first, last
                    )
                    lower, upper = np.asarray(lower), np.asarray(upper)
                    if (
                        lower.shape != (2,)
                        or upper.shape != (2,)
                        or np.any(~np.isfinite(lower))
                        or np.any(~np.isfinite(upper))
                        or np.any(lower > upper)
                        or (expected > 0 and not float(lower[0]) > 0)
                        or (expected < 0 and not float(upper[0]) < 0)
                    ):
                        return None
                else:
                    u_low = low if u_low is None else min(u_low, low)
                    u_high = high if u_high is None else max(u_high, high)
        if full_turn:
            embedding = certify_source_embedding(
                surface, ((Fraction(0), v_low), (_TAU, v_high)), deck_axes=(0,)
            )
        else:
            if u_low is None or u_high is None:
                return None
            embedding = certify_source_embedding(
                surface, ((u_low, v_low), (u_high, v_high))
            )
        if embedding is None:
            return None
        return BRepTrimSheetProof(domain, domain.domain_id, patch, embedding)
    except (
        ArithmeticError,
        IndexError,
        KeyError,
        NotImplementedError,
        TypeError,
        ValueError,
    ):
        return None


def certify_cad_trim_sheet(
    charts: object, patch: int, tolerance: float, /
) -> CadTrimSheetProof | None:
    """Compose source-only sheet authority with a bound reference homotopy.

    Homotopy embeddedness, the reference partition, and orientation are
    independent obligations. Polar planes retain their declared pole quotient.
    """
    from .._meshing_domain import SourcePreparation
    from .._surface_source_support import SurfaceSourceCharts, validate_cad_curve_ancestry

    if not isinstance(charts, SurfaceSourceCharts) or type(patch) is not int:
        return None
    try:
        charts.require_current()
        bindings = validate_cad_curve_ancestry(charts, tolerance)
        return _certify_cad_trim_sheet_validated(
            charts, bindings, patch, SourcePreparation(charts.domain)
        )
    except (
        ArithmeticError,
        IndexError,
        KeyError,
        NotImplementedError,
        TypeError,
        ValueError,
    ):
        return None


def _certify_cad_trim_sheet_validated(
    charts: SurfaceSourceCharts,
    bindings: dict[int, tuple[_CadCurveUseBinding, ...]],
    patch: int,
    preparation: SourcePreparation,
    /,
) -> CadTrimSheetProof | None:
    """Compose a sheet using ancestry replayed by the enclosing proof."""
    from .._approximate_source_coverage import _cad_trim_reference_loops
    from ._curve_approximation_topology import _trim_reference_box

    try:
        native = certify_brep_trim_sheet(charts.domain, patch)
        if native is None:
            return None
        reference = _cad_trim_reference_loops(charts, bindings, patch)
        pieces = tuple(segment for loop in reference for segment in loop)
        if not pieces:
            return None
        boxes = tuple(_trim_reference_box(segment, preparation) for segment in pieces)
        if any(
            box.shape != (2, 2) or np.any(~np.isfinite(box)) or np.any(box[0] > box[1])
            for box in boxes
        ):
            return None
        native_bounds = native.embedding.bounds
        native_v_low, native_v_high = native_bounds[0][1], native_bounds[1][1]
        if not isinstance(native_v_low, Fraction) or not isinstance(
            native_v_high, Fraction
        ):
            return None
        if native.embedding.family == "revolution-plane":
            if any(
                not native_v_low <= point[1] <= native_v_high
                for segment in pieces for point in (segment.start, segment.stop)
            ):
                return None
            v_low, v_high = native_v_low, native_v_high
        else:
            v_low = min(
                (native_v_low, *(Fraction.from_float(float(box[0, 1])) for box in boxes))
            )
            v_high = max(
                (native_v_high, *(Fraction.from_float(float(box[1, 1])) for box in boxes))
            )
        full_turn = native.embedding.deck_axes == (0,)
        if full_turn:
            for segment in pieces:
                for point in (segment.start, segment.stop):
                    if point[0] < 0 or _sheet_sign(_difference(point[0], _TAU)) not in (
                        -1,
                        0,
                    ):
                        return None
            bounds = ((Fraction(0), v_low), (_TAU, v_high))
        else:
            u_low, u_high = native_bounds[0][0], native_bounds[1][0]
            for segment in pieces:
                for point in (segment.start, segment.stop):
                    if _sheet_sign(_difference(point[0], u_low)) == -1:
                        u_low = point[0]
                    if _sheet_sign(_difference(point[0], u_high)) == 1:
                        u_high = point[0]
            bounds = ((u_low, v_low), (u_high, v_high))
        embedding = certify_source_embedding(
            charts.domain.patches[patch].surface,
            bounds,
            deck_axes=native.embedding.deck_axes,
            pole_roots=native.embedding.pole_roots,
        )
        if embedding is None:
            return None

        def ratio(value: Fraction) -> tuple[int, int]:
            return value.numerator, value.denominator

        reference_id = canonical_fingerprint(
            tuple(
                tuple(
                    (
                        carrier_content_id(segment.curve),
                        ratio(segment.first),
                        ratio(segment.last),
                        tuple(map(ratio, segment.start)),
                        tuple(map(ratio, segment.stop)),
                        None
                        if segment.start_root_box is None
                        else array_tree_fingerprint(segment.start_root_box),
                        None
                        if segment.stop_root_box is None
                        else array_tree_fingerprint(segment.stop_root_box),
                        segment.start_parameter_box,
                        segment.stop_parameter_box,
                    )
                    for segment in loop
                )
                for loop in reference
            )
        )
        return CadTrimSheetProof(
            charts, charts.source_chart_id, patch, reference_id, embedding
        )
    except (
        ArithmeticError,
        IndexError,
        KeyError,
        NotImplementedError,
        TypeError,
        ValueError,
    ):
        return None


__all__ = [
    "BRepTrimSheetProof",
    "CadTrimSheetProof",
    "ExactChartBox",
    "SourceEmbeddingProof",
    "certify_brep_trim_sheet",
    "certify_cad_trim_sheet",
    "certify_source_embedding",
]

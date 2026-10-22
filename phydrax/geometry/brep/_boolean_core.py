#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared Boolean policies, selection semantics, and rectilinear boundary assembly."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Literal, TypeAlias, TypedDict

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._atlas import AbstractTrimCurve
from .._cad_revision import (
    AssociationGraph,
    CADOccurrence,
    CADRevision,
)
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._coordinate_request import coordinate_source_request
from ._intersection import NativePeriodEndpoint, ParametricIntersectionPolicy
from ._intersection_curve import AffinePCurve, CurveTrimSegment, PeriodicPCurve
from ._model import BRepGeometry, BRepModel
from ._patches import AbstractCurve, LineCurve, PlanePatch
from ._placed_affine import exact_line_descriptor, exact_plane_descriptor
from ._request_preparations import TrimRequestPreparations
from ._sewing import BRepSewingPolicy, sew_brep, shells
from ._trim_helpers import PeriodOffset, period_scalar_bounds


BRepBooleanOperation: TypeAlias = Literal["union", "intersection", "difference"]


type Cell = tuple[int, int, int]


def resolve_node(node: str, aliases: Mapping[str, str], /) -> str:
    """Return the representative of an arrangement node alias chain."""
    seen: set[str] = set()
    while node in aliases:
        if node in seen:
            raise BRepBooleanFailure(
                "BRepBoolean.node_aliases must be acyclic; got a cyclic alias graph"
            )
        seen.add(node)
        node = aliases[node]
    return node


def merge_node_aliases(aliases: dict[str, str], first: str, second: str, /) -> None:
    """Merge alias classes using their lexicographically smallest representative."""
    first = resolve_node(first, aliases)
    second = resolve_node(second, aliases)
    if first != second:
        aliases[max(first, second)] = min(first, second)


class BRepBooleanPolicy(StrictModule):
    """Declare arrangement, sewing, tessellation, and intersection limits."""

    maximum_cells: int = eqx.field(static=True)
    maximum_faces: int = eqx.field(static=True)
    sewing: BRepSewingPolicy
    tessellation: BRepTessellationPolicy
    intersection: ParametricIntersectionPolicy
    construction_tolerance: float = eqx.field(static=True)
    trim_resolution: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        maximum_cells: int = 100_000,
        maximum_faces: int = 100_000,
        sewing: BRepSewingPolicy | None = None,
        tessellation: BRepTessellationPolicy | None = None,
        intersection: ParametricIntersectionPolicy | None = None,
        construction_tolerance: float = 1.0e-9,
        trim_resolution: float = 0.025,
    ) -> None:
        for name, value in (
            ("BRepBooleanPolicy.maximum_cells", maximum_cells),
            ("BRepBooleanPolicy.maximum_faces", maximum_faces),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer; got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        sewing_ = BRepSewingPolicy() if sewing is None else sewing
        tessellation_ = BRepTessellationPolicy() if tessellation is None else tessellation
        intersection_ = (
            ParametricIntersectionPolicy() if intersection is None else intersection
        )
        for name, value in (
            ("BRepBooleanPolicy.construction_tolerance", construction_tolerance),
            ("BRepBooleanPolicy.trim_resolution", trim_resolution),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        self.maximum_cells = maximum_cells
        self.maximum_faces = maximum_faces
        self.sewing = sewing_
        self.tessellation = tessellation_
        self.intersection = intersection_
        self.construction_tolerance = float(construction_tolerance)
        self.trim_resolution = float(trim_resolution)


class BRepBooleanFailure(RuntimeError):
    """Report a scientific failure with the input model revisions."""

    def __init__(self, reason: str, source_entities: tuple[str, ...] = ()) -> None:
        self.reason = reason
        self.source_entities = source_entities
        super().__init__(
            f"Native CAD Boolean unresolved: {reason}; sources={source_entities}."
        )


class BRepBooleanResult(StrictModule):
    """Return the Boolean model and its exhaustive ancestry, including empty sets."""

    model: BRepModel
    association_graph: AssociationGraph = eqx.field(static=True)
    operation: BRepBooleanOperation = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    deleted_source_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        graph: AssociationGraph,
        operation: BRepBooleanOperation,
        certificate_id: str,
        deleted_source_ids: tuple[str, ...],
    ) -> None:
        self.model = model
        self.association_graph = graph
        self.operation = parse(operation, BRepBooleanOperation, "operation")
        self.certificate_id = certificate_id
        self.deleted_source_ids = deleted_source_ids

    @property
    def empty(self) -> bool:
        return self.model.topology.num_solids == 0


type ChartScalar = Fraction | PeriodOffset


def chart_add(first: ChartScalar, second: ChartScalar) -> ChartScalar:
    left = first if isinstance(first, PeriodOffset) else PeriodOffset(first, Fraction(0))
    right = (
        second if isinstance(second, PeriodOffset) else PeriodOffset(second, Fraction(0))
    )
    value = left + right
    if not isinstance(value, (Fraction, PeriodOffset)):
        raise TypeError(
            f"face_cells.chart_sum must be a Fraction or PeriodOffset; got {type(value).__name__}"
        )
    return value


def chart_scale(value: ChartScalar, factor: Fraction) -> ChartScalar:
    source = (
        value if isinstance(value, PeriodOffset) else PeriodOffset(value, Fraction(0))
    )
    result = source * factor
    if not isinstance(result, (Fraction, PeriodOffset)):
        raise TypeError(
            f"face_cells.chart_product must be a Fraction or PeriodOffset; got {type(result).__name__}"
        )
    return result


def chart_parts(value: ChartScalar) -> tuple[Fraction, Fraction]:
    return (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )


def line_chart_coefficients(
    curve: AbstractCurve | AbstractTrimCurve,
) -> tuple[tuple[ChartScalar, ...], tuple[Fraction, ...]] | None:
    """Exact ``origin + t * direction`` UV coefficients in Q + Q*(2*pi), or None."""
    if isinstance(curve, LineCurve):
        return (
            tuple(Fraction(float(value)) for value in np.asarray(curve.origin)),
            tuple(Fraction(float(value)) for value in np.asarray(curve.direction)),
        )
    if isinstance(curve, PeriodicPCurve):
        child = line_chart_coefficients(curve.source_curve)
        if child is None:
            return None
        origin, direction = child
        return (
            tuple(
                chart_add(value, PeriodOffset(Fraction(0), Fraction(shift)))
                for value, shift in zip(origin, curve.period_shifts, strict=True)
            ),
            direction,
        )
    if isinstance(curve, AffinePCurve):
        child = line_chart_coefficients(curve.curve)
        if child is None:
            return None
        origin, direction = child
        transformed = []
        for row in range(2):
            value: ChartScalar = curve.offset[row]
            for column in range(2):
                value = chart_add(
                    value, chart_scale(origin[column], curve.matrix[row][column])
                )
            transformed.append(value)
        return (
            tuple(transformed),
            tuple(
                sum((curve.matrix[i][j] * direction[j] for j in range(2)), Fraction(0))
                for i in range(2)
            ),
        )
    return None


def _chart_sign(value: ChartScalar, /) -> int | None:
    """Exact sign of ``q + t * 2 pi``; None only when directed bounds straddle zero."""
    rational, turns = chart_parts(value)
    if not turns:
        return (rational > 0) - (rational < 0)
    # A nonzero turn coefficient never cancels a rational: pi is irrational.
    lower, upper = period_scalar_bounds(value)
    if lower > 0.0:
        return 1
    if upper < 0.0:
        return -1
    return None


def _coedge_trim(
    geometry: BRepGeometry,
    coedge: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> CurveTrimSegment:
    edge = geometry.coedge_edges[coedge]
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    first_root, last_root = geometry.coedge_endpoint_roots[coedge]
    return CurveTrimSegment(
        geometry.pcurves[coedge],
        first,
        last,
        reversed=geometry.coedge_senses[coedge] < 0,
        first_root=first_root,
        last_root=last_root,
        preparations=preparations,
        budget=budget,
    )


def _coedge_axis_values(
    geometry: BRepGeometry, coedge: int, axis: int, /
) -> tuple[ChartScalar, ChartScalar] | None:
    """Exact chart coordinates of an affine p-curve at its authored endpoints."""
    coefficients = line_chart_coefficients(geometry.pcurves[coedge])
    if coefficients is None:
        return None
    origin, direction = coefficients
    edge = geometry.coedge_edges[coedge]
    values: list[ChartScalar] = []
    for value, root in zip(
        np.asarray(geometry.edge_ranges)[edge],
        geometry.coedge_endpoint_roots[coedge],
        strict=True,
    ):
        match root:
            case None:
                parameter: ChartScalar = Fraction(float(value))
            case NativePeriodEndpoint():
                parameter = root.exact_parameter
            case _:
                return None
        values.append(chart_add(origin[axis], chart_scale(parameter, direction[axis])))
    return values[0], values[1]


def _full_period_seam(
    geometry: BRepGeometry,
    face: int,
    axis: int,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Prove the face trim spans exactly one native turn from an authored seam.

    The premise is the face's own topology: one edge used by two opposite
    coedges whose exact affine p-curves are one isoline in two native period
    gauges differing by one turn, and every trim coedge on the closed strip
    between those gauges. Endpoint coordinates use authored native-period
    roots; other carriers need a certified enclosure strictly inside the strip.
    Box widths, near-period extents and numerical tolerances never prove it.
    """
    coedges = tuple(coedge for loop in geometry.face_loops[face] for coedge in loop)
    uses: dict[int, list[int]] = {}
    for coedge in coedges:
        uses.setdefault(geometry.coedge_edges[coedge], []).append(coedge)
    turn = PeriodOffset(Fraction(0), Fraction(1))
    seams: set[ChartScalar] = set()
    other = 1 - axis
    for pair in uses.values():
        if (
            len(pair) != 2
            or geometry.coedge_senses[pair[0]] == geometry.coedge_senses[pair[1]]
        ):
            continue
        first_line = line_chart_coefficients(geometry.pcurves[pair[0]])
        second_line = line_chart_coefficients(geometry.pcurves[pair[1]])
        if first_line is None or second_line is None:
            continue
        (first_origin, first_direction), (second_origin, second_direction) = (
            first_line,
            second_line,
        )
        if (
            first_direction != second_direction
            or first_direction[axis] != 0
            or first_direction[other] == 0
            or first_origin[other] != second_origin[other]
        ):
            continue
        difference = chart_add(
            second_origin[axis], chart_scale(first_origin[axis], Fraction(-1))
        )
        if difference == turn:
            seams.add(first_origin[axis])
        elif difference == -turn:
            seams.add(second_origin[axis])
    if len(seams) != 1:
        return None
    seam = next(iter(seams))
    if not isinstance(seam, Fraction) or Fraction(float(seam)) != seam:
        return None
    end = chart_add(seam, turn)
    lower_bounds, upper_bounds = period_scalar_bounds(seam), period_scalar_bounds(end)
    for coedge in coedges:
        values = _coedge_axis_values(geometry, coedge, axis)
        if values is not None:
            # An affine coordinate attains its range at the authored endpoints.
            if any(
                _chart_sign(chart_add(value, chart_scale(seam, Fraction(-1))))
                not in (0, 1)
                or _chart_sign(chart_add(end, chart_scale(value, Fraction(-1))))
                not in (0, 1)
                for value in values
            ):
                return None
            continue
        box = np.asarray(
            _coedge_trim(
                geometry, coedge, preparations=preparations, budget=budget
            ).enclosure(
                0.0, 1.0, preparations=preparations, budget=budget
            ),
            dtype=np.float64,
        )
        if not (box[0, axis] >= lower_bounds[1] and box[1, axis] <= upper_bounds[0]):
            return None
    return seam


@checked
def boolean_face_chart_box(
    model: BRepModel,
    face: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Source face parameter box with proved full native turns in their seam chart.

    ``parameter_bounds`` remains the model's outward box. A periodic axis is
    replaced by the fundamental chart ``[seam, seam + period]`` only when
    ``_full_period_seam`` proves the face trim spans that turn and the declared
    box contains it; otherwise the declared box is returned unchanged.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((model, face)) as budget:
            return boolean_face_chart_box(
                model, face, preparations=preparations, budget=budget
            )
    box = np.array(np.asarray(model.parameter_bounds, dtype=np.float64)[face])
    geometry = model.geometry
    if geometry is None:
        return box
    patch = model.patches[face]
    symbols = patch.native_period_symbols()
    for axis, period in enumerate(patch.periods):
        if period is None or symbols[axis] != "two_pi":
            continue
        seam = _full_period_seam(
            geometry, face, axis, preparations=preparations, budget=budget
        )
        if seam is None:
            continue
        lower = float(seam)
        upper = lower + float(period)
        if box[0, axis] <= lower and upper <= box[1, axis]:
            box[0, axis], box[1, axis] = lower, upper
    return box


class _Rectangle(StrictModule):
    model: int = eqx.field(static=True)
    solid: int = eqx.field(static=True)
    face: int = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    coordinate: float = eqx.field(static=True)
    lower: tuple[float, float, float] = eqx.field(static=True)
    upper: tuple[float, float, float] = eqx.field(static=True)


def _rectilinear_coedge_vertex(
    model: BRepModel,
    geometry: BRepGeometry,
    points: np.ndarray,
    coedge: int,
) -> np.ndarray:
    edge = geometry.coedge_edges[coedge]
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.curve must have an explicit line carrier; got an implicit nonlinear trim",
            (f"{model.source_revision}:edge:{edge}",),
        )
    first, last = geometry.edge_vertices[edge]
    source_curve = geometry.curves[curve_index]
    curve = (
        exact_line_descriptor(source_curve)
        if isinstance(source_curve, AbstractCurve)
        else None
    )
    if curve is None:
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.curve must admit an affine line descriptor; got a nonlinear carrier"
        )
    direction = np.asarray(curve.direction)
    origin = np.asarray(curve.origin)
    ranges = np.asarray(geometry.edge_ranges)[edge]
    if np.count_nonzero(direction) != 1 or any(
        Fraction(float(origin[d]))
        + Fraction(float(parameter)) * Fraction(float(direction[d]))
        != Fraction(float(points[vertex, d]))
        for parameter, vertex in zip(ranges, (first, last), strict=True)
        for d in range(3)
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.coedge.endpoints must match rational line evaluation on one axis; got incompatible coordinates",
            (f"{model.source_revision}:edge:{edge}",),
        )
    return points[first if geometry.coedge_senses[coedge] > 0 else last]


def _prepare_rectilinear_face(
    model_index: int,
    model: BRepModel,
    geometry: BRepGeometry,
    points: np.ndarray,
    solid: int,
    face: int,
) -> _Rectangle:
    patch = exact_plane_descriptor(model.patches[face])
    loops = geometry.face_loops[face]
    if patch is None or len(loops) != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face must use a planar carrier with one loop; got another face family",
            (f"{model.source_revision}:face:{face}",),
        )
    coedges = loops[0]
    if len(coedges) != 4:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.coedges must contain four lines; got another coedge count",
            (f"{model.source_revision}:face:{face}",),
        )
    vertices_ = np.stack(
        [
            _rectilinear_coedge_vertex(model, geometry, points, coedge)
            for coedge in coedges
        ]
    )
    lower, upper = vertices_.min(axis=0), vertices_.max(axis=0)
    fixed = np.flatnonzero(lower == upper)
    if fixed.size != 1:
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.vertices must have one fixed coordinate; got another fixed-axis count",
            (f"{model.source_revision}:face:{face}",),
        )
    axis = int(fixed[0])
    if (
        float(patch.origin[axis]) != float(lower[axis])
        or float(patch.first_axis[axis]) != 0.0
        or float(patch.second_axis[axis]) != 0.0
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.patch must support its rectangle coordinates; got incompatible chart axes or origin",
            (f"{model.source_revision}:face:{face}",),
        )
    changing = tuple(index for index in range(3) if index != axis)
    expected = {
        tuple(point)
        for point in product(
            *((lower[d], upper[d]) if d in changing else (lower[d],) for d in range(3))
        )
    }
    differences = np.roll(vertices_, -1, axis=0) - vertices_
    if {tuple(point) for point in vertices_} != expected or np.any(
        np.count_nonzero(differences, axis=1) != 1
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.rectilinear_face.vertices must form an axis-aligned rectangle; got incompatible corners or edges",
            (f"{model.source_revision}:face:{face}",),
        )
    return _Rectangle(
        model_index,
        solid,
        face,
        axis,
        float(lower[axis]),
        (float(lower[0]), float(lower[1]), float(lower[2])),
        (float(upper[0]), float(upper[1]), float(upper[2])),
    )


def rectangles(
    models: tuple[BRepModel, ...],
    selected_solids: tuple[frozenset[int], ...] | None = None,
) -> tuple[_Rectangle, ...]:
    rectangles = []
    for model_index, model in enumerate(models):
        geometry = model.geometry
        if geometry is None or (not geometry.solid_shells and model.patches):
            raise BRepBooleanFailure(
                "BRepBoolean.model.geometry must define closed-solid shells; got missing geometry or shells",
                (model.model_id,),
            )
        points = np.asarray(geometry.vertex_points)
        for solid, faces in enumerate(model.topology.solid_faces):
            if selected_solids is not None and solid not in selected_solids[model_index]:
                continue
            signs = np.asarray(model.orientation).copy()
            for face, sense in zip(
                faces, model.topology.solid_face_orientations[solid], strict=True
            ):
                signs[face] *= sense
            shells(geometry, faces, signs)
            for face in faces:
                rectangles.append(
                    _prepare_rectilinear_face(
                        model_index, model, geometry, points, solid, face
                    )
                )
    return tuple(rectangles)


def membership(
    point: tuple[Fraction, Fraction, Fraction],
    rectangles: tuple[_Rectangle, ...],
    models: tuple[BRepModel, ...],
) -> tuple[tuple[int, int], ...]:
    crossings: dict[tuple[int, int], int] = {}
    for rectangle in rectangles:
        if rectangle.axis != 0 or Fraction(rectangle.coordinate) <= point[0]:
            continue
        if all(
            Fraction(rectangle.lower[d]) < point[d] < Fraction(rectangle.upper[d])
            for d in (1, 2)
        ):
            key = rectangle.model, rectangle.solid
            crossings[key] = crossings.get(key, 0) + 1
    return tuple(
        (model, solid)
        for model, carrier in enumerate(models)
        for solid in range(carrier.topology.num_solids)
        if crossings.get((model, solid), 0) % 2
    )


def selected(
    operation: BRepBooleanOperation, members: tuple[tuple[int, int], ...]
) -> bool:
    first = any(model == 0 for model, _ in members)
    second = any(model == 1 for model, _ in members)
    match operation:
        case "union":
            return first or second
        case "intersection":
            return first and second
        case "difference":
            return first and not second
        case _:
            raise ValueError(
                f"BRepBoolean.operation must be union, intersection, or difference; got {operation!r}"
            )


def precedence_owner(
    precedence: tuple[str, ...],
    voids: tuple[tuple[str, tuple[str, ...]], ...],
    present: frozenset[str],
    /,
) -> str | None:
    """Return the highest-precedence material owner among present operands.

    Ownership is decided first: the highest-precedence present region wins.
    A present void (``(void operand ID, targeted region IDs)``) that targets
    that winner empties the material; a lower present region never refills it.
    Region IDs are their region operand IDs. Planar and solid partitions share
    this rule.
    """
    owner = next((region for region in precedence if region in present), None)
    if owner is not None and any(
        void in present and owner in targets for void, targets in voids
    ):
        return None
    return owner


class RegionSelectionPayload(TypedDict):
    kind: Literal["precedence-region-partition"]
    operands: tuple[str, ...]
    precedence: tuple[str, ...]
    voids: tuple[tuple[str, tuple[str, ...]], ...]
    solids: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class RegionSelection:
    """Partition ownership of open solid material by declared region precedence.

    ``operand_ids`` names the arrangement operands by index; ``precedence`` and
    ``voids`` follow ``precedence_owner``. Only the declared
    ``(operand index, solid)`` selections participate.
    """

    operand_ids: tuple[str, ...]
    precedence: tuple[str, ...]
    voids: tuple[tuple[str, tuple[str, ...]], ...]
    solids: tuple[tuple[int, int], ...]

    @property
    def payload(self) -> RegionSelectionPayload:
        return {
            "kind": "precedence-region-partition",
            "operands": self.operand_ids,
            "precedence": self.precedence,
            "voids": self.voids,
            "solids": self.solids,
        }


type MaterialSelection = BRepBooleanOperation | RegionSelection


def material_owner(
    selection: MaterialSelection, members: tuple[tuple[int, int], ...]
) -> str | None:
    """Return the material owner whose open neighborhood lies in ``members``.

    A Boolean owns its selected material under the operation name; ``None`` is
    empty space. Two sides with different owners bound published material.
    """
    match selection:
        case RegionSelection():
            selected_bank = set(selection.solids)
            present = frozenset(
                selection.operand_ids[model]
                for model, solid in members
                if (model, solid) in selected_bank
            )
            return precedence_owner(selection.precedence, selection.voids, present)
        case str():
            return selection if selected(selection, members) else None
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def selection_payload(
    selection: MaterialSelection,
) -> BRepBooleanOperation | RegionSelectionPayload:
    match selection:
        case RegionSelection():
            return selection.payload
        case str():
            return selection
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def side_payload(
    selection: MaterialSelection,
    negative: str | None,
    positive: str | None,
) -> tuple[str | None, str | None] | tuple[bool, bool]:
    """Boolean sides are identified by selection alone; partition sides by owner."""
    match selection:
        case RegionSelection():
            return negative, positive
        case str():
            return negative is not None, positive is not None
        case _:
            raise ValueError(
                f"BRepBoolean.material.selection must be a Boolean operation or RegionSelection; got {selection!r}"
            )


def components(cells: set[Cell]) -> tuple[tuple[Cell, ...], ...]:
    remaining = set(cells)
    result = []
    while remaining:
        seed = min(remaining)
        queue = deque((seed,))
        remaining.remove(seed)
        component = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for axis, step in product(range(3), (-1, 1)):
                candidate = list(cell)
                candidate[axis] += step
                neighbor = candidate[0], candidate[1], candidate[2]
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(tuple(sorted(component)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BoundaryFace:
    solid: int
    axis: int
    side: int
    vertices: tuple[Cell, ...]
    sources: tuple[tuple[int, int, int], ...]


def boundary(
    components: tuple[tuple[Cell, ...], ...],
    coordinates: tuple[tuple[float, ...], ...],
    rectangles: tuple[_Rectangle, ...],
    policy: BRepBooleanPolicy,
) -> tuple[BoundaryFace, ...]:
    faces = []
    for solid, component in enumerate(components):
        cells = set(component)
        for cell in component:
            for axis, side in product(range(3), (-1, 1)):
                neighbor = list(cell)
                neighbor[axis] += side
                if tuple(neighbor) in cells:
                    continue
                if len(faces) >= policy.maximum_faces:
                    raise BRepBooleanFailure(
                        "BRepBoolean.face.fragments must fit maximum_faces; got face fragmentation budget exhausted"
                    )
                first, second = (axis + 1) % 3, (axis + 2) % 3
                plane = cell[axis] + (1 if side > 0 else 0)
                vertices = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    vertex = list(cell)
                    vertex[axis] = plane
                    vertex[first] += a
                    vertex[second] += b
                    vertices.append((vertex[0], vertex[1], vertex[2]))
                coordinate = coordinates[axis][plane]
                lower = tuple(coordinates[d][vertices[0][d]] for d in range(3))
                upper = tuple(coordinates[d][vertices[2][d]] for d in range(3))
                sources = tuple(
                    sorted(
                        {
                            (rectangle.model, rectangle.solid, rectangle.face)
                            for rectangle in rectangles
                            if rectangle.axis == axis
                            and rectangle.coordinate == coordinate
                            and all(
                                rectangle.lower[d] <= lower[d]
                                and upper[d] <= rectangle.upper[d]
                                for d in (first, second)
                            )
                        }
                    )
                )
                if not sources:
                    raise BRepBooleanFailure(
                        "BRepBoolean.boundary.fragment must retain a source face; got no source face"
                    )
                faces.append(BoundaryFace(solid, axis, side, tuple(vertices), sources))
    return tuple(faces)


def assemble(
    models: tuple[BRepModel, ...],
    coordinates: tuple[tuple[float, ...], ...],
    faces: tuple[BoundaryFace, ...],
    policy: BRepBooleanPolicy,
    certificate: str,
    *,
    shared_faces: bool = False,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    face_count = (
        len({tuple(sorted(face.vertices)) for face in faces})
        if shared_faces
        else len(faces)
    )
    if (
        face_count > policy.maximum_faces
        or 4 * face_count > policy.sewing.maximum_coedges
    ):
        raise BRepBooleanFailure(
            "BRepBoolean.boundary must fit face and coedge limits; got face/coedge allocation budget exhausted"
        )
    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[int, Cell], int] = {}
    curves = []
    edge_vertices = []
    edge_ranges = []
    edge_ids: dict[tuple[int, int], int] = {}
    pcurves = []
    coedge_edges = []
    senses = []
    loops = []
    patches = []
    bounds = []
    orientation = []
    solid_count = max((face.solid for face in faces), default=-1) + 1
    solid_faces: list[list[int]] = [[] for _ in range(solid_count)]
    solid_signs: list[list[int]] = [[] for _ in range(solid_count)]
    face_ids: dict[tuple[Cell, ...], int] = {}
    for face in faces:
        face_key = tuple(sorted(face.vertices))
        if shared_faces and face_key in face_ids:
            face_index = face_ids[face_key]
            solid_faces[face.solid].append(face_index)
            solid_signs[face.solid].append(face.side)
            continue
        face_index = len(loops)
        face_ids[face_key] = face_index
        first, second = (face.axis + 1) % 3, (face.axis + 2) % 3
        indices = []
        for grid in face.vertices:
            key = (-1 if shared_faces else face.solid), grid
            if key not in vertex_ids:
                vertex_ids[key] = len(vertices)
                vertices.append(
                    (
                        coordinates[0][grid[0]],
                        coordinates[1][grid[1]],
                        coordinates[2][grid[2]],
                    )
                )
            indices.append(vertex_ids[key])
        loop = []
        for index, start in enumerate(indices):
            end = indices[(index + 1) % 4]
            p0, p1 = vertices[start], vertices[end]
            changing = next(d for d in range(3) if p0[d] != p1[d])
            lo, hi = (start, end) if p0[changing] < p1[changing] else (end, start)
            key = lo, hi
            if key not in edge_ids:
                origin = np.asarray(vertices[lo], dtype=np.float64).copy()
                origin[changing] = 0.0
                direction = np.zeros(3, dtype=np.float64)
                direction[changing] = 1.0
                edge_ids[key] = len(curves)
                curves.append(LineCurve(origin, direction))
                edge_vertices.append(key)
                edge_ranges.append((vertices[lo][changing], vertices[hi][changing]))
            p_origin = np.asarray(
                (vertices[lo][first], vertices[lo][second]), dtype=np.float64
            )
            p_direction = np.zeros(2, dtype=np.float64)
            parameter_axis = 0 if changing == first else 1
            p_origin[parameter_axis] = 0.0
            p_direction[parameter_axis] = 1.0
            pcurves.append(LineCurve(p_origin, p_direction))
            loop.append(len(coedge_edges))
            coedge_edges.append(edge_ids[key])
            senses.append(1 if start == lo else -1)
        loops.append((tuple(loop),))
        origin = np.zeros(3, dtype=np.float64)
        origin[face.axis] = vertices[indices[0]][face.axis]
        patches.append(PlanePatch(origin, np.eye(3)[first], np.eye(3)[second]))
        bounds.append(
            (
                (vertices[indices[0]][first], vertices[indices[0]][second]),
                (vertices[indices[2]][first], vertices[indices[2]][second]),
            )
        )
        orientation.append(face.side)
        solid_faces[face.solid].append(face_index)
        solid_signs[face.solid].append(face.side)
    geometry = BRepGeometry(
        vertex_points=np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        curves=tuple(curves),
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.asarray(edge_ranges, dtype=np.float64).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(senses),
        face_loops=tuple(loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
        budget=budget,
        preparations=preparations,
    )
    orientation_array = np.asarray(orientation, dtype=np.float64)
    sewed = sew_brep(
        geometry,
        orientation_array,
        tuple(tuple(group) for group in solid_faces),
        policy=policy.sewing,
        solid_orientations=tuple(tuple(group) for group in solid_signs),
        preparations=preparations,
        budget=budget,
    )
    return assemble_brep_model(
        sewed.geometry,
        tuple(patches),
        np.asarray(bounds, dtype=np.float64).reshape((-1, 2, 2)),
        orientation_array,
        tuple("plane" for _ in patches),
        coordinate_contract=models[0].coordinate_contract,
        source_id=f"native-boolean:{certificate}",
        source_format="native",
        source_digest=certificate,
        import_policy_id=certificate,
        tessellation=policy.tessellation,
        preparations=preparations,
        budget=budget,
    )


def source_revision(models: tuple[BRepModel, ...], certificate: str) -> CADRevision:
    occurrences = []
    for model_index, model in enumerate(models):
        for solid, faces in enumerate(model.topology.solid_faces):
            root = f"operand:{model_index}/solid:{solid}"
            occurrences.append(
                CADOccurrence(
                    certificate,
                    root,
                    f"{model.source_revision}:solid:{solid}",
                    "solid",
                    (root,),
                )
            )
            for face, orientation in zip(
                faces, model.topology.solid_face_orientations[solid], strict=True
            ):
                child = f"{root}/face:{face}"
                occurrences.append(
                    CADOccurrence(
                        certificate,
                        child,
                        f"{model.source_revision}:face:{face}",
                        "face",
                        (root, child),
                        root,
                        orientation,
                    )
                )
    return CADRevision(
        certificate,
        f"native-boolean-sources:{certificate}",
        tuple(occurrences),
        certificate,
    )


def rectilinear_family(models: tuple[BRepModel, ...]) -> bool:
    for model in models:
        geometry = model.geometry
        if geometry is None:
            return False
        points = np.asarray(geometry.vertex_points)
        for face, patch in enumerate(model.patches):
            loops = geometry.face_loops[face]
            if (
                exact_plane_descriptor(patch) is None
                or len(loops) != 1
                or len(loops[0]) != 4
            ):
                return False
            vertices = []
            for coedge in loops[0]:
                edge = geometry.coedge_edges[coedge]
                curve_index = geometry.edge_curves[edge]
                if curve_index < 0:
                    return False
                source_curve = geometry.curves[curve_index]
                if (
                    not isinstance(source_curve, AbstractCurve)
                    or exact_line_descriptor(source_curve) is None
                ):
                    return False
                vertices.extend(geometry.edge_vertices[edge])
            box = points[vertices]
            if np.count_nonzero(np.min(box, axis=0) == np.max(box, axis=0)) != 1:
                return False
    return True


__all__ = [
    "BRepBooleanFailure",
    "BRepBooleanOperation",
    "BRepBooleanPolicy",
    "BRepBooleanResult",
]

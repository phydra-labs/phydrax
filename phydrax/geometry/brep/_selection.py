#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified three-valued entity predicates and recipe-bound patch controls."""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np

from ..._strict import StrictModule
from ..._validation import finite_real_scalar, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._interval_enclosure import interval_add, interval_multiply, interval_subtract
from .._meshing_domain import MeshingDomain
from ..design._naming import (
    RecipeEntityKind,
    RecipeEntityName,
    RecipeEntityNames,
    RecipeNamingError,
)
from ._edge_classification import brep_edge_classification
from ._model import BRepEntityId, BRepModel
from ._normal_range import _direction, _HALF_PI, _trim_cell, brep_normal_ranges
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    interval_dot,
    LineCurve,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from ._query_membership import full_rectangle
from ._request_preparations import TrimRequestPreparations


if TYPE_CHECKING:
    from ...meshing._controls import PatchControl
    from ...meshing._scope import MeshingScope

BRepPredicateKind: TypeAlias = Literal[
    "analytic_family",
    "facing",
    "parallel",
    "perpendicular",
    "convexity",
    "extremal",
    "adjacent",
    "and",
    "or",
    "not",
    "neighbors",
]
BRepAnalyticFamily: TypeAlias = Literal[
    "plane", "cylinder", "cone", "sphere", "torus", "nurbs", "line", "circle", "ellipse"
]
BRepConvexity: TypeAlias = Literal["convex", "concave", "sign_change"]


@final
class BRepPredicate(StrictModule):
    """Immutable expression; angle parameters are declared radians, not tolerances."""

    kind: BRepPredicateKind = eqx.field(static=True)
    family: BRepAnalyticFamily = eqx.field(static=True)
    direction: tuple[float, float, float] = eqx.field(static=True)
    angle: float = eqx.field(static=True)
    convexity: BRepConvexity = eqx.field(static=True)
    entities: tuple[BRepEntityId, ...] = eqx.field(static=True)
    operands: tuple[BRepPredicate, ...]

    def __init__(
        self,
        kind: BRepPredicateKind,
        /,
        *,
        family: BRepAnalyticFamily = "plane",
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        angle: float = 0.0,
        convexity: BRepConvexity = "convex",
        entities: tuple[BRepEntityId, ...] = (),
        operands: tuple[BRepPredicate, ...] = (),
    ) -> None:
        kind_ = parse(kind, BRepPredicateKind, "predicate.kind")
        family_ = parse(family, BRepAnalyticFamily, "predicate.family")
        convexity_ = parse(convexity, BRepConvexity, "predicate.convexity")
        vector = _direction(direction)
        angle_ = finite_real_scalar(angle, "predicate.angle")
        if not 0 <= angle_ <= pi / 2:
            raise ValueError(
                "predicate.angle must lie in [0, pi/2]; got an out-of-range angle"
            )
        match kind_:
            case "and" | "or":
                valid = len(operands) >= 2
            case "not" | "neighbors":
                valid = len(operands) == 1
            case _:
                valid = not operands
        if not valid:
            raise ValueError(
                "predicate.operands must match the expression arity; got a mismatched operand count"
            )
        self.kind, self.family, self.convexity = kind_, family_, convexity_
        self.direction = (float(vector[0]), float(vector[1]), float(vector[2]))
        self.angle, self.entities, self.operands = (
            angle_,
            tuple(entities),
            tuple(operands),
        )

    def __and__(self, other: BRepPredicate, /) -> BRepPredicate:
        return BRepPredicate("and", operands=(self, other))

    def __or__(self, other: BRepPredicate, /) -> BRepPredicate:
        return BRepPredicate("or", operands=(self, other))

    def __invert__(self) -> BRepPredicate:
        return BRepPredicate("not", operands=(self,))

    def neighbors(self) -> BRepPredicate:
        """Select one-hop neighbors, excluding an entity's self incidence."""
        return BRepPredicate("neighbors", operands=(self,))


def _inventory(model: BRepModel, kind: RecipeEntityKind, /) -> tuple[BRepEntityId, ...]:
    match kind:
        case "face":
            return model.face_ids
        case "edge":
            return model.edge_ids
        case _:
            raise ValueError(
                "selection.kind must be face or edge; got another entity family"
            )


@final
class BRepSelection(StrictModule):
    """Exhaustive disjoint model-ordered sets, including producer provenance."""

    model_id: str = eqx.field(static=True)
    selected: tuple[BRepEntityId, ...] = eqx.field(static=True)
    rejected: tuple[BRepEntityId, ...] = eqx.field(static=True)
    unresolved: tuple[BRepEntityId, ...] = eqx.field(static=True)
    provenance: tuple[str, ...] = eqx.field(static=True)
    predicates: tuple[BRepPredicate, ...]

    @checked
    def __init__(
        self,
        model: BRepModel,
        selected: tuple[BRepEntityId, ...],
        rejected: tuple[BRepEntityId, ...],
        unresolved: tuple[BRepEntityId, ...],
        /,
        *,
        provenance: tuple[str, ...],
        predicates: tuple[BRepPredicate, ...] = (),
    ) -> None:
        entities = (*selected, *rejected, *unresolved)
        if not entities or len(set(entities)) != len(entities):
            raise ValueError(
                "selection.sets must be nonempty and disjoint; got empty or repeated entities"
            )
        kind = parse(entities[0].kind, RecipeEntityKind, "selection.kind")
        inventory = _inventory(model, kind)
        if set(entities) != set(inventory):
            raise ValueError(
                "selection.sets must exhaust one model face or edge inventory; got mismatched entities"
            )
        if not provenance or any(
            not isinstance(x, str) or not x.strip() for x in provenance
        ):
            raise ValueError(
                "selection.provenance must contain nonempty producer descriptions; got absent or empty descriptions"
            )
        self.model_id = model.model_id
        selected_set, rejected_set, unresolved_set = (
            set(selected),
            set(rejected),
            set(unresolved),
        )
        self.selected = tuple(x for x in inventory if x in selected_set)
        self.rejected = tuple(x for x in inventory if x in rejected_set)
        self.unresolved = tuple(x for x in inventory if x in unresolved_set)
        self.provenance, self.predicates = tuple(provenance), tuple(predicates)

    @checked
    def bind_region(
        self,
        name: str,
        names: RecipeEntityNames,
        domain: MeshingDomain,
        adjacent_region_names: tuple[str, ...],
        /,
        *,
        include_unresolved: bool | None = None,
    ) -> PatchControl:
        """Bind the existing WP-C PatchControl through stable recipe names."""
        # Geometry preparation does not eagerly import the meshing request stack.
        from ...meshing._controls import PatchControl

        if self.model_id != names.model_id:
            raise RecipeNamingError(
                "selection.names must belong to the selected model; got a different model"
            )
        if self.unresolved and include_unresolved is None:
            raise ValueError(
                "selection.unresolved must have an explicit inclusion decision; got no decision"
            )
        match include_unresolved:
            case True | False | None:
                pass
            case _:
                raise TypeError(
                    "selection.include_unresolved must be bool or None; got another object kind"
                )
        entities = self.selected + (self.unresolved if include_unresolved else ())
        if not entities:
            raise ValueError(
                "selection.binding must contain at least one entity; got empty membership"
            )
        binding = RecipeSelectionBinding(names, entities)
        return PatchControl(
            name,
            binding.scope(names, domain),
            adjacent_region_names,
            recipe_binding=binding,
        )


@final
class RecipeSelectionBinding(StrictModule):
    """Stable membership, not a second region or meshing-control vocabulary."""

    kind: RecipeEntityKind = eqx.field(static=True)
    names: tuple[RecipeEntityName, ...] = eqx.field(static=True)
    inventory: tuple[RecipeEntityName, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self, names: RecipeEntityNames, entities: tuple[BRepEntityId, ...], /
    ) -> None:
        if not entities:
            raise RecipeNamingError("selection.binding must be nonempty; got no entities")
        kind = parse(entities[0].kind, RecipeEntityKind, "selection.binding.kind")
        inventory = names.face_names if kind == "face" else names.edge_names
        if inventory is None:
            raise RecipeNamingError(
                "selection.binding must have exhaustive recipe names; got unsupported names"
            )
        selected = tuple(names.name(entity) for entity in entities)
        if any(name.kind != kind for name in selected):
            raise RecipeNamingError(
                "selection.binding must contain one entity family; got mixed entity families"
            )
        self.kind, self.names, self.inventory = kind, selected, inventory

    @checked
    def scope(self, names: RecipeEntityNames, domain: MeshingDomain, /) -> MeshingScope:
        # The scope owner is needed only when a geometric selection becomes a request.
        from ...meshing._scope import MeshingEntityKind, MeshingScope

        inventory = names.face_names if self.kind == "face" else names.edge_names
        if inventory is None or set(inventory) != set(self.inventory):
            raise RecipeNamingError(
                "selection.rebind must preserve recipe inventory; got split, merge, missing, or changed names"
            )
        if domain.authority_id != names.model_id:
            raise RecipeNamingError(
                "selection.domain must retain the named model identity and source revision; got a different model"
            )
        entities = tuple(names.resolve(name) for name in self.names)
        dimension = 2 if self.kind == "face" else 1
        slots = _recipe_domain_slots(entities, domain, dimension)
        indices = domain.scope_indices(dimension)
        return MeshingScope(
            domain.source_id,
            domain.source_revision,
            MeshingEntityKind.GEOMETRY,
            dimension,
            domain.entity_set_id(dimension),
            np.asarray(sorted(indices[slot] for slot in slots), dtype=np.int64),
        )


def _recipe_domain_slots(
    entities: tuple[BRepEntityId, ...], domain: MeshingDomain, dimension: int, /
) -> tuple[int, ...]:
    slots: list[int] = []
    source = tuple(
        zip(
            domain.source_indices[dimension],
            domain.source_occurrences[dimension],
            strict=True,
        )
    )
    for entity in entities:
        images = tuple(
            slot
            for slot, (index, path) in enumerate(source)
            if index == entity.index
            and (not entity.occurrence_path or path == entity.occurrence_path)
        )
        if entity.source_revision != domain.source_revision or not images:
            raise RecipeNamingError(
                "selection.domain must bind the recipe realization's source revision and entities; got a mismatched domain"
            )
        # Empty paths identify definitions; declared occurrence rows, not
        # spatial coincidence or path prefixes, expand a definition selection.
        slots.extend(images)
    return tuple(sorted(set(slots)))


def _adjacency(
    model: BRepModel, kind: RecipeEntityKind, /
) -> tuple[set[BRepEntityId], ...]:
    geometry = model.require_geometry()
    face_edges = [
        set(geometry.coedge_edges[c] for loop in loops for c in loop)
        for loops in geometry.face_loops
    ]
    if kind == "face":
        return tuple(
            {
                model.face_ids[j]
                for j, other in enumerate(face_edges)
                if i != j and edges & other
            }
            | {model.edge_ids[e] for e in edges}
            for i, edges in enumerate(face_edges)
        )
    return _edge_neighbors(model, face_edges)


def _edge_neighbors(
    model: BRepModel, face_edges: list[set[int]], /
) -> tuple[set[BRepEntityId], ...]:
    geometry = model.require_geometry()
    edge_vertices = [set(pair) for pair in geometry.edge_vertices]
    return tuple(
        {
            model.edge_ids[j]
            for j, other in enumerate(edge_vertices)
            if i != j and vertices & other
        }
        | {model.face_ids[f] for f, edges in enumerate(face_edges) if i in edges}
        for i, vertices in enumerate(edge_vertices)
    )


def _family(
    model: BRepModel, kind: RecipeEntityKind, family: BRepAnalyticFamily, /
) -> np.ndarray:
    classes = {
        "plane": PlanePatch,
        "cylinder": CylinderPatch,
        "cone": ConePatch,
        "sphere": SpherePatch,
        "torus": TorusPatch,
        "nurbs": BSplineSurfacePatch,
        "line": LineCurve,
        "circle": CircleCurve,
        "ellipse": EllipseCurve,
    }
    if kind == "face":
        return np.asarray(
            [isinstance(patch, classes[family]) for patch in model.patches], dtype=np.int8
        )
    geometry = model.require_geometry()
    return np.asarray(
        [
            isinstance(
                geometry.curves[curve],
                BSplineCurve if family == "nurbs" else classes[family],
            )
            if curve >= 0
            else False
            for curve in geometry.edge_curves
        ],
        dtype=np.int8,
    )


def _normal_test(
    model: BRepModel,
    predicate: BRepPredicate,
    maximum: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    angle = predicate.angle
    angle_point = (np.float64(angle), np.float64(angle))
    reverse_limit = interval_subtract(
        interval_multiply((np.float64(2), np.float64(2)), _HALF_PI), angle_point
    )
    perpendicular_lower = interval_subtract(_HALF_PI, angle_point)
    perpendicular_upper = interval_add(_HALF_PI, angle_point)
    records = brep_normal_ranges(
        model,
        predicate.direction,
        draft_angle=0.0,
        maximum_cells=maximum,
        coordinate_budget=budget,
        angle_thresholds=(angle, pi - angle, pi / 2 - angle, pi / 2 + angle),
    )
    result = np.full(len(records), -1, dtype=np.int8)
    for i, record in enumerate(records):
        if record.singular_points.shape[0]:
            continue
        lower, upper = np.asarray(record.angle_enclosure)
        match predicate.kind:
            case "facing":
                inside, outside = upper < angle, lower > angle
            case "parallel":
                inside = upper < angle or lower > reverse_limit[1]
                outside = lower > angle and upper < reverse_limit[0]
            case "perpendicular":
                inside = lower > perpendicular_lower[1] and upper < perpendicular_upper[0]
                outside = upper < perpendicular_lower[0] or lower > perpendicular_upper[1]
            case _:
                raise ValueError(
                    "selection.normal must have a normal predicate; got another predicate kind"
                )
        if inside:
            result[i] = 1
        elif outside:
            result[i] = 0
    return result


def _plane_coefficients(
    patch: PlanePatch, direction: np.ndarray, /
) -> tuple[Fraction, ...]:
    d = tuple(Fraction(float(x)) for x in direction)
    return tuple(
        sum(
            (Fraction(float(x)) * y for x, y in zip(np.asarray(vector), d, strict=True)),
            Fraction(0),
        )
        for vector in (patch.origin, patch.first_axis, patch.second_axis)
    )


def _plane_projection(
    patch: PlanePatch, box: np.ndarray, direction: np.ndarray, /
) -> tuple[Fraction, Fraction]:
    coefficients = _plane_coefficients(patch, direction)
    lower = upper = coefficients[0]
    for axis, coefficient in enumerate(coefficients[1:]):
        values = tuple(coefficient * Fraction(float(x)) for x in box[:, axis])
        lower, upper = lower + min(values), upper + max(values)
    return lower, upper


def _planar_trim_projection(
    model: BRepModel,
    face: int,
    patch: PlanePatch,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, Fraction]:
    geometry = model.require_geometry()
    coedges = tuple(c for loop in geometry.face_loops[face] for c in loop)
    if not coedges:
        return _plane_projection(
            patch, np.asarray(model.parameter_bounds)[face], direction
        )
    budget.reserve(64 * len(coedges), 1024 * len(coedges))
    coefficients = _plane_coefficients(patch, direction)
    bounds: list[tuple[Fraction, Fraction]] = []
    for coedge in coedges:
        curve = geometry.pcurves[coedge]
        edge = geometry.coedge_edges[coedge]
        endpoints = geometry.edge_parameter_enclosure(edge, budget=budget)
        if isinstance(curve, LineCurve):
            origin, velocity = np.asarray(curve.origin), np.asarray(curve.direction)
            base = coefficients[0] + sum(
                (
                    c * Fraction(float(x))
                    for c, x in zip(coefficients[1:], origin, strict=True)
                ),
                Fraction(0),
            )
            speed = sum(
                (
                    c * Fraction(float(x))
                    for c, x in zip(coefficients[1:], velocity, strict=True)
                ),
                Fraction(0),
            )
            values = tuple(base + speed * Fraction(float(t)) for t in endpoints.flat)
            bounds.append((min(values), max(values)))
        else:
            box = curve.bounding_box(
                float(endpoints[0, 0]), float(endpoints[1, 1]), budget=budget
            )
            bounds.append(_plane_projection(patch, box, direction))
    # A linear functional on a bounded planar trim attains its extrema on
    # the trim boundary. P-curves keep this bound on the face carrier even
    # when a sewn 3D edge has a declared gap from that carrier.
    return min(lower for lower, _ in bounds), max(upper for _, upper in bounds)


def _extrusion_projection(
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, Fraction] | None:
    if not isinstance(patch, ExtrusionSurface) or not isinstance(
        patch.curve, BSplineCurve
    ):
        return None
    curve = patch.curve
    if not np.all(np.asarray(curve.weights) > 0):
        return None
    controls = np.asarray(curve.control_points)
    budget.reserve(64 * controls.shape[0], 1024 * controls.shape[0])
    d = tuple(Fraction(float(x)) for x in direction)
    projections = tuple(
        sum((Fraction(float(x)) * y for x, y in zip(point, d, strict=True)), Fraction(0))
        for point in controls
    )
    axial = sum(
        (
            Fraction(float(x)) * y
            for x, y in zip(np.asarray(patch.direction), d, strict=True)
        ),
        Fraction(0),
    )
    offsets = tuple(axial * Fraction(float(value)) for value in box[:, 1])
    # Positive rational B-spline weights give a convex combination of controls.
    return min(projections) + min(offsets), max(projections) + max(offsets)


def _face_counterpoint(
    model: BRepModel,
    face: int,
    direction: np.ndarray,
    extreme_lower: Fraction,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    patch = model.patches[face]
    bounds = np.asarray(model.parameter_bounds)[face]
    preparations = TrimRequestPreparations()
    # Each admitted point proves a strict counterexample. Failure to find one
    # remains unresolved; it never certifies a universal statement.
    for u in (0.25, 0.5, 0.75):
        for v in (0.25, 0.5, 0.75):
            point = bounds[0] + np.asarray((u, v)) * (bounds[1] - bounds[0])
            box = np.stack((point, point))
            if _trim_cell(model, face, box, False, 0, budget, preparations) is not True:
                continue
            if isinstance(patch, PlanePatch):
                _, upper = _plane_projection(patch, box, direction)
            else:
                spatial = patch.bounding_box(box, budget=budget)
                _, projection_upper = interval_dot(
                    (spatial[0], spatial[1]), (direction, direction)
                )
                upper = Fraction(float(projection_upper))
            if upper < extreme_lower:
                return True
    return False


def _projection_enclosures(
    model: BRepModel,
    kind: RecipeEntityKind,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[list[tuple[Fraction, Fraction]], list[bool]]:
    geometry = model.require_geometry()
    bounds = np.asarray(model.parameter_bounds)
    enclosures: list[tuple[Fraction, Fraction]] = []
    exact_rectangles: list[bool] = []
    if kind == "face":
        for face, patch in enumerate(model.patches):
            projection = (
                _planar_trim_projection(model, face, patch, direction, budget)
                if isinstance(patch, PlanePatch)
                else _extrusion_projection(patch, bounds[face], direction, budget)
            )
            if projection is not None:
                enclosures.append(projection)
                exact_rectangles.append(
                    isinstance(patch, PlanePatch)
                    and full_rectangle(geometry, face, patch, bounds[face])
                )
            else:
                box = patch.bounding_box(bounds[face], budget=budget)
                lo, hi = interval_dot((box[0], box[1]), (direction, direction))
                enclosures.append((Fraction(float(lo)), Fraction(float(hi))))
                exact_rectangles.append(False)
    else:
        for edge, carrier in enumerate(geometry.edge_curves):
            if carrier < 0:
                point = np.asarray(geometry.vertex_points)[
                    geometry.edge_vertices[edge][0]
                ]
                box = np.stack((point, point))
            else:
                endpoints = geometry.edge_parameter_enclosure(edge, budget=budget)
                box = geometry.curves[carrier].bounding_box(
                    float(endpoints[0, 0]), float(endpoints[1, 1]), budget=budget
                )
            lo, hi = interval_dot((box[0], box[1]), (direction, direction))
            enclosures.append((Fraction(float(lo)), Fraction(float(hi))))
            exact_rectangles.append(False)
    return enclosures, exact_rectangles


def _extremal(
    model: BRepModel,
    kind: RecipeEntityKind,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    enclosures, exact_rectangles = _projection_enclosures(model, kind, direction, budget)
    extreme = max(upper for _, upper in enclosures)
    extreme_lower = max(lower for lower, _ in enclosures)
    extreme_lower = max(
        (
            upper
            for (_, upper), rectangle in zip(enclosures, exact_rectangles, strict=True)
            if rectangle
        ),
        default=extreme_lower,
    )
    # Selection means wholly on the supporting extreme, not merely touching
    # it along an edge. Nonconstant rectangles have a certified counterpoint.
    return np.asarray(
        [
            1
            if lower == upper == extreme
            else 0
            if upper < extreme_lower
            or rectangle
            and lower < extreme_lower
            or kind == "face"
            and _face_counterpoint(model, index, direction, extreme_lower, budget)
            else -1
            for index, ((lower, upper), rectangle) in enumerate(
                zip(enclosures, exact_rectangles, strict=True)
            )
        ],
        dtype=np.int8,
    )


def _combine(
    model: BRepModel,
    kind: RecipeEntityKind,
    predicate: BRepPredicate,
    maximum: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    inventory = _inventory(model, kind)
    match predicate.kind:
        case "not":
            values = _evaluate(model, kind, predicate.operands[0], maximum, budget)
            return np.where(values < 0, -1, 1 - values).astype(np.int8)
        case "and" | "or":
            budget.reserve(
                len(predicate.operands) * len(inventory),
                8 * len(predicate.operands) * len(inventory),
            )
            values = np.stack(
                [
                    _evaluate(model, kind, operand, maximum, budget)
                    for operand in predicate.operands
                ]
            )
            if predicate.kind == "and":
                return np.where(
                    np.any(values == 0, axis=0),
                    0,
                    np.where(np.all(values == 1, axis=0), 1, -1),
                ).astype(np.int8)
            return np.where(
                np.any(values == 1, axis=0),
                1,
                np.where(np.all(values == 0, axis=0), 0, -1),
            ).astype(np.int8)
        case "neighbors":
            values = _evaluate(model, kind, predicate.operands[0], maximum, budget)
            selected = {
                entity
                for entity, value in zip(inventory, values, strict=True)
                if value == 1
            }
            unresolved = {
                entity
                for entity, value in zip(inventory, values, strict=True)
                if value < 0
            }
            return np.asarray(
                [
                    1 if neighbors & selected else -1 if neighbors & unresolved else 0
                    for neighbors in _adjacency(model, kind)
                ],
                dtype=np.int8,
            )
        case _:
            raise ValueError(
                "selection.combinator must have logical operands; got a leaf predicate"
            )


def _evaluate(
    model: BRepModel,
    kind: RecipeEntityKind,
    predicate: BRepPredicate,
    maximum: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    inventory = _inventory(model, kind)
    budget.reserve(16 * len(inventory), 64 * len(inventory))
    match predicate.kind:
        case "analytic_family":
            return _family(model, kind, predicate.family)
        case "facing" | "parallel" | "perpendicular":
            return (
                _normal_test(model, predicate, maximum, budget)
                if kind == "face"
                else np.full(len(inventory), -1, dtype=np.int8)
            )
        case "convexity":
            if kind != "edge":
                return np.full(len(inventory), -1, dtype=np.int8)
            records = brep_edge_classification(
                model,
                model.require_geometry(),
                maximum_cells=maximum,
                coordinate_budget=budget,
            )
            return np.asarray(
                [
                    -1
                    if record.status == "unresolved"
                    else int(record.status == predicate.convexity)
                    for record in records
                ],
                dtype=np.int8,
            )
        case "extremal":
            return _extremal(model, kind, _direction(predicate.direction), budget)
        case "adjacent":
            all_entities = set((*model.face_ids, *model.edge_ids))
            if not set(predicate.entities) <= all_entities:
                raise ValueError(
                    "selection.adjacent must use entities from this model; got foreign entities"
                )
            return np.asarray(
                [
                    bool(neighbors & set(predicate.entities))
                    for neighbors in _adjacency(model, kind)
                ],
                dtype=np.int8,
            )
        case "and" | "or" | "not" | "neighbors":
            return _combine(model, kind, predicate, maximum, budget)
        case _:
            raise ValueError(
                "selection.predicate must have a supported kind; got an invalid selector"
            )


@checked
def select_brep_entities(
    model: BRepModel,
    predicate: BRepPredicate,
    /,
    *,
    kind: RecipeEntityKind = "face",
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
) -> BRepSelection:
    """Evaluate a predicate over one model family using Kleene logic."""
    kind_ = parse(kind, RecipeEntityKind, "selection.kind")
    maximum = positive_integer(maximum_cells, "selection.maximum_cells")
    inventory = _inventory(model, kind_)
    coordinate_budget.reserve(
        len(inventory) * maximum * 64, len(inventory) * maximum * 1024
    )
    values = _evaluate(model, kind_, predicate, maximum, coordinate_budget)
    return BRepSelection(
        model,
        tuple(
            entity for entity, value in zip(inventory, values, strict=True) if value == 1
        ),
        tuple(
            entity for entity, value in zip(inventory, values, strict=True) if value == 0
        ),
        tuple(
            entity for entity, value in zip(inventory, values, strict=True) if value < 0
        ),
        provenance=("certified geometric predicate",),
        predicates=(predicate,),
    )

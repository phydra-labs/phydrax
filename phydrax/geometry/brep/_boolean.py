#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native exact-coordinate CAD Boolean and material-owner arrangements.

Rectilinear solids use authoritative source-coordinate cells. Other surfaces
use certified coupled intersection curves and source-rooted trims. No
tessellation participates in classification, construction, or ancestry, and
unresolved source evidence never becomes a fitted or triangle-derived B-Rep.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import parse
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._intersection import ParametricIntersectionPolicy
from ._model import BRepGeometry, BRepModel
from ._patches import AbstractCurve, LineCurve, PlanePatch
from ._placed_affine import exact_line_descriptor, exact_plane_descriptor
from ._sewing import _shells, BRepSewingPolicy, sew_brep


BRepBooleanOperation: TypeAlias = Literal["union", "intersection", "difference"]
type _Cell = tuple[int, int, int]


class BRepBooleanPolicy(StrictModule):
    """Bounded exact arrangement with no implicit tolerance repair."""

    maximum_cells: int = eqx.field(static=True)
    maximum_faces: int = eqx.field(static=True)
    sewing: BRepSewingPolicy
    tessellation: BRepTessellationPolicy
    intersection: ParametricIntersectionPolicy
    construction_tolerance: float = eqx.field(static=True)
    trim_resolution: float = eqx.field(static=True)

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
            ("maximum_cells", maximum_cells),
            ("maximum_faces", maximum_faces),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        sewing_ = BRepSewingPolicy() if sewing is None else sewing
        tessellation_ = BRepTessellationPolicy() if tessellation is None else tessellation
        if not isinstance(sewing_, BRepSewingPolicy):
            raise TypeError("sewing must be a BRepSewingPolicy.")
        if not isinstance(tessellation_, BRepTessellationPolicy):
            raise TypeError("tessellation must be a BRepTessellationPolicy.")
        intersection_ = (
            ParametricIntersectionPolicy() if intersection is None else intersection
        )
        if not isinstance(intersection_, ParametricIntersectionPolicy):
            raise TypeError("intersection must be a ParametricIntersectionPolicy.")
        for name, value in (
            ("construction_tolerance", construction_tolerance),
            ("trim_resolution", trim_resolution),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.maximum_cells = maximum_cells
        self.maximum_faces = maximum_faces
        self.sewing = sewing_
        self.tessellation = tessellation_
        self.intersection = intersection_
        self.construction_tolerance = float(construction_tolerance)
        self.trim_resolution = float(trim_resolution)


class BRepBooleanFailure(RuntimeError):
    """Scientific failure preserving the complete input model revisions."""

    def __init__(self, reason: str, source_entities: tuple[str, ...] = ()) -> None:
        self.reason = reason
        self.source_entities = source_entities
        super().__init__(
            f"Native CAD Boolean unresolved: {reason}; sources={source_entities}."
        )


class BRepBooleanResult(StrictModule):
    """Canonical exact model, including empty sets, with exhaustive ancestry."""

    model: BRepModel
    association_graph: AssociationGraph = eqx.field(static=True)
    operation: BRepBooleanOperation = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    deleted_source_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        graph: AssociationGraph,
        operation: BRepBooleanOperation,
        certificate_id: str,
        deleted_source_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        if not isinstance(graph, AssociationGraph):
            raise TypeError("graph must be an AssociationGraph.")
        self.model = model
        self.association_graph = graph
        self.operation = parse(operation, BRepBooleanOperation, "operation")
        self.certificate_id = certificate_id
        self.deleted_source_ids = deleted_source_ids

    @property
    def empty(self) -> bool:
        return self.model.topology.num_solids == 0


@dataclass(frozen=True, slots=True)
class _Rectangle:
    model: int
    solid: int
    face: int
    axis: int
    coordinate: float
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]


def _rectangles(
    models: tuple[BRepModel, ...],
    selected_solids: tuple[frozenset[int], ...] | None = None,
) -> tuple[_Rectangle, ...]:
    rectangles = []
    for model_index, model in enumerate(models):
        geometry = model.geometry
        if geometry is None or (not geometry.solid_shells and model.patches):
            raise BRepBooleanFailure(
                "authoritative closed-solid geometry is required", (model.model_id,)
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
            _shells(geometry, faces, signs)
            for face in faces:
                patch = exact_plane_descriptor(model.patches[face])
                loops = geometry.face_loops[face]
                if patch is None or len(loops) != 1:
                    raise BRepBooleanFailure(
                        "certified coupled-curve face fragmentation is unavailable",
                        (f"{model.source_revision}:face:{face}",),
                    )
                coedges = loops[0]
                if len(coedges) != 4:
                    raise BRepBooleanFailure(
                        "rectilinear face requires four authoritative line coedges",
                        (f"{model.source_revision}:face:{face}",),
                    )
                vertices = []
                for coedge in coedges:
                    edge = geometry.coedge_edges[coedge]
                    curve_index = geometry.edge_curves[edge]
                    if curve_index < 0:
                        raise BRepBooleanFailure(
                            "nonlinear trim requires certified curve arrangement",
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
                            "A rectilinear coedge requires an exact affine line descriptor."
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
                            "source line endpoints are not exact rectilinear coordinates",
                            (f"{model.source_revision}:edge:{edge}",),
                        )
                    vertices.append(
                        points[first if geometry.coedge_senses[coedge] > 0 else last]
                    )
                vertices_ = np.stack(vertices)
                lower, upper = vertices_.min(axis=0), vertices_.max(axis=0)
                fixed = np.flatnonzero(lower == upper)
                if fixed.size != 1:
                    raise BRepBooleanFailure(
                        "non-orthogonal planar face arrangement is unavailable",
                        (f"{model.source_revision}:face:{face}",),
                    )
                axis = int(fixed[0])
                if (
                    float(patch.origin[axis]) != float(lower[axis])
                    or float(patch.first_axis[axis]) != 0.0
                    or float(patch.second_axis[axis]) != 0.0
                ):
                    raise BRepBooleanFailure(
                        "source plane does not exactly support its rectangle",
                        (f"{model.source_revision}:face:{face}",),
                    )
                changing = tuple(index for index in range(3) if index != axis)
                expected = {
                    tuple(point)
                    for point in product(
                        *(
                            (lower[d], upper[d]) if d in changing else (lower[d],)
                            for d in range(3)
                        )
                    )
                }
                differences = np.roll(vertices_, -1, axis=0) - vertices_
                if {tuple(point) for point in vertices_} != expected or np.any(
                    np.count_nonzero(differences, axis=1) != 1
                ):
                    raise BRepBooleanFailure(
                        "face is not an exact axis-aligned rectangle",
                        (f"{model.source_revision}:face:{face}",),
                    )
                rectangles.append(
                    _Rectangle(
                        model_index,
                        solid,
                        face,
                        axis,
                        float(lower[axis]),
                        (float(lower[0]), float(lower[1]), float(lower[2])),
                        (float(upper[0]), float(upper[1]), float(upper[2])),
                    )
                )
    return tuple(rectangles)


def _membership(
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


def _selected(
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
            raise ValueError(f"Invalid Boolean operation {operation!r}.")


def _precedence_owner(
    precedence: tuple[str, ...],
    voids: tuple[tuple[str, tuple[str, ...]], ...],
    present: frozenset[str],
    /,
) -> str | None:
    """Canonical partition owner of material covered by ``present`` operand IDs.

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


@dataclass(frozen=True, slots=True)
class _RegionSelection:
    """Partition ownership of open solid material by declared region precedence.

    ``operand_ids`` names the arrangement operands by index; ``precedence`` and
    ``voids`` follow ``_precedence_owner``. Only the declared
    ``(operand index, solid)`` selections participate.
    """

    operand_ids: tuple[str, ...]
    precedence: tuple[str, ...]
    voids: tuple[tuple[str, tuple[str, ...]], ...]
    solids: tuple[tuple[int, int], ...]

    @property
    def payload(self) -> dict[str, object]:
        return {
            "kind": "precedence-region-partition",
            "operands": self.operand_ids,
            "precedence": self.precedence,
            "voids": self.voids,
            "solids": self.solids,
        }


type _MaterialSelection = BRepBooleanOperation | _RegionSelection


def _material_owner(
    selection: _MaterialSelection, members: tuple[tuple[int, int], ...]
) -> str | None:
    """Owner of material whose open neighborhood lies in exactly ``members``.

    A Boolean owns its selected material under the operation name; ``None`` is
    empty space. Two sides with different owners bound published material.
    """
    match selection:
        case _RegionSelection():
            selected = set(selection.solids)
            present = frozenset(
                selection.operand_ids[model]
                for model, solid in members
                if (model, solid) in selected
            )
            return _precedence_owner(selection.precedence, selection.voids, present)
        case str():
            return selection if _selected(selection, members) else None
        case _:
            raise ValueError(f"Invalid material selection {selection!r}.")


def _selection_payload(selection: _MaterialSelection) -> object:
    match selection:
        case _RegionSelection():
            return selection.payload
        case str():
            return selection
        case _:
            raise ValueError(f"Invalid material selection {selection!r}.")


def _side_payload(
    selection: _MaterialSelection,
    negative: str | None,
    positive: str | None,
) -> tuple[object, object]:
    """Boolean sides are identified by selection alone; partition sides by owner."""
    match selection:
        case _RegionSelection():
            return negative, positive
        case str():
            return negative is not None, positive is not None
        case _:
            raise ValueError(f"Invalid material selection {selection!r}.")


def _components(cells: set[_Cell]) -> tuple[tuple[_Cell, ...], ...]:
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
class _BoundaryFace:
    solid: int
    axis: int
    side: int
    vertices: tuple[_Cell, ...]
    sources: tuple[tuple[int, int, int], ...]


def _boundary(
    components: tuple[tuple[_Cell, ...], ...],
    coordinates: tuple[tuple[float, ...], ...],
    rectangles: tuple[_Rectangle, ...],
    policy: BRepBooleanPolicy,
) -> tuple[_BoundaryFace, ...]:
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
                    raise BRepBooleanFailure("face fragmentation budget exhausted")
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
                        "boundary fragment has no authoritative source face"
                    )
                faces.append(_BoundaryFace(solid, axis, side, tuple(vertices), sources))
    return tuple(faces)


def _assemble(
    models: tuple[BRepModel, ...],
    coordinates: tuple[tuple[float, ...], ...],
    faces: tuple[_BoundaryFace, ...],
    policy: BRepBooleanPolicy,
    certificate: str,
    *,
    shared_faces: bool = False,
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
        raise BRepBooleanFailure("face/coedge allocation budget exhausted")
    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[int, _Cell], int] = {}
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
    face_ids: dict[tuple[_Cell, ...], int] = {}
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
    )
    orientation_array = np.asarray(orientation, dtype=np.float64)
    sewed = sew_brep(
        geometry,
        orientation_array,
        tuple(tuple(group) for group in solid_faces),
        policy=policy.sewing,
        solid_orientations=tuple(tuple(group) for group in solid_signs),
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
    )


def _source_revision(models: tuple[BRepModel, ...], certificate: str) -> CADRevision:
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


def _graph(
    source: CADRevision,
    model: BRepModel,
    faces: tuple[_BoundaryFace, ...],
    components: tuple[tuple[_Cell, ...], ...],
    membership: dict[_Cell, tuple[tuple[int, int], ...]],
    certificate: str,
) -> AssociationGraph:
    from ._partition import cad_revision_from_brep_model

    target = cad_revision_from_brep_model(model)
    pairs = set()
    for solid, component in enumerate(components):
        ancestors = {ancestor for cell in component for ancestor in membership[cell]}
        for operand, parent_solid in ancestors:
            pairs.add((f"operand:{operand}/solid:{parent_solid}", f"solid:{solid}"))
    for face_index, face in enumerate(faces):
        for operand, parent_solid, parent_face in face.sources:
            pairs.add(
                (
                    f"operand:{operand}/solid:{parent_solid}/face:{parent_face}",
                    f"solid:{face.solid}/face:{face_index}",
                )
            )
    correspondences = tuple(
        OccurrenceCorrespondence(first, second, certificate)
        for first, second in sorted(pairs)
    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        correspondences,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-rectilinear-cad-arrangement"
        ),
    )
    return AssociationGraph(source, target, transaction)


def _coincident_result(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> BRepBooleanResult:
    from dataclasses import replace

    from ._partition import cad_revision_from_brep_model

    first = models[0]
    certificate = canonical_fingerprint(
        {
            "kind": "native-identical-cad-boolean",
            "source": first.model_id,
            "operation": operation,
        }
    )
    source = _source_revision(models, certificate)
    if operation == "difference":
        model = _assemble(models, ((), (), ()), (), policy, certificate)
        pairs = ()
    else:
        if first.topology.num_faces > policy.maximum_faces:
            raise BRepBooleanFailure("face allocation budget exhausted")
        model = BRepModel(
            patches=first.patches,
            parameter_bounds=first.parameter_bounds,
            orientation=first.orientation,
            trim_domains=first.trim_domains,
            topology=first.topology,
            coordinate_contract=first.coordinate_contract,
            mesh_vertices=first.mesh_vertices,
            mesh_faces=first.mesh_faces,
            triangle_face_ids=first.triangle_face_ids,
            triangle_parameters=first.triangle_parameters,
            physical_tags=first.physical_tags,
            tessellation_deviation_bounds=first.tessellation_deviation_bounds,
            tessellation_normal_bounds=first.tessellation_normal_bounds,
            mesh_vertex_source_dimensions=first.mesh_vertex_source_dimensions,
            mesh_vertex_source_indices=first.mesh_vertex_source_indices,
            mesh_vertex_parameters=first.mesh_vertex_parameters,
            mesh_chart_restriction_vertices=first.mesh_chart_restriction_vertices,
            mesh_chart_restriction_edges=first.mesh_chart_restriction_edges,
            mesh_chart_restriction_endpoint_parameters=(
                first.mesh_chart_restriction_endpoint_parameters
            ),
            mesh_chart_restriction_parameters=first.mesh_chart_restriction_parameters,
            coedge_deviation_bounds=first.coedge_deviation_bounds,
            triangle_occurrence_ids=first.triangle_occurrence_ids,
            vertex_occurrence_ids=first.vertex_occurrence_ids,
            report=replace(
                first.report,
                source_digest=certificate,
                source_id=f"native-boolean:{certificate}",
            ),
            geometry=first.geometry,
        )
        pairs = tuple(
            OccurrenceCorrespondence(
                occurrence.occurrence_id,
                occurrence.occurrence_id.split("/", 1)[1],
                certificate,
            )
            for occurrence in source.occurrences
        )
    target = cad_revision_from_brep_model(model)
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        pairs,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-identical-cad-boolean"
        ),
    )
    graph = AssociationGraph(source, target, transaction)
    deleted = (
        tuple(occurrence.occurrence_id for occurrence in source.occurrences)
        if operation == "difference"
        else ()
    )
    return BRepBooleanResult(model, graph, operation, certificate, deleted)


def _rectilinear_family(models: tuple[BRepModel, ...]) -> bool:
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


def _boolean_world(
    first: BRepModel,
    second: BRepModel,
    operation: BRepBooleanOperation,
    /,
    *,
    policy: BRepBooleanPolicy | None = None,
) -> BRepBooleanResult:
    """Regularized native CAD union, intersection, or difference.

    Tangent boundary-only intersections are empty regularized solids. Coincident
    faces are classified once by arrangement occupancy. Disconnected components
    and enclosed void shells retain separate oriented topology.
    Rectilinear solids use exact source-coordinate cell arrangements. Other
    surfaces use complete coupled intersection curves and source-rooted trims.
    Unresolved discovery, root identity, classification, or sewing never returns
    an approximate fitted or triangle-derived B-Rep.
    """
    if not isinstance(first, BRepModel) or not isinstance(second, BRepModel):
        raise TypeError("Boolean operands must be BRepModel values.")
    if first.coordinate_contract.spatial_id != second.coordinate_contract.spatial_id:
        raise ValueError("Boolean operands must use the same coordinate contract.")
    operation_ = parse(operation, BRepBooleanOperation, "operation")
    policy_ = BRepBooleanPolicy() if policy is None else policy
    if not isinstance(policy_, BRepBooleanPolicy):
        raise TypeError("policy must be a BRepBooleanPolicy.")
    models = first, second
    for model in models:
        if model.geometry is None or (model.patches and not model.topology.num_solids):
            raise BRepBooleanFailure(
                "Boolean operands require authoritative closed solids",
                (model.source_revision,),
            )
    if first.model_id == second.model_id and first.geometry is not None:
        return _coincident_result(models, operation_, policy_)
    if not _rectilinear_family(models):
        from ._boolean_coincidence import coincident_boolean_brep

        coincident = coincident_boolean_brep(models, operation_, policy_)
        if coincident is not None:
            return coincident
        from ._boolean_full_overlay import full_overlay_boolean_brep

        return full_overlay_boolean_brep(models, operation_, policy_)
    rectangles = _rectangles(models)
    coordinates = tuple(
        tuple(
            sorted(
                {rectangle.lower[d] for rectangle in rectangles}
                | {rectangle.upper[d] for rectangle in rectangles}
            )
        )
        for d in range(3)
    )
    count = prod(max(0, len(values) - 1) for values in coordinates)
    if count > policy_.maximum_cells:
        raise BRepBooleanFailure(
            "arrangement cell budget exhausted",
            tuple(model.source_revision for model in models),
        )
    membership: dict[_Cell, tuple[tuple[int, int], ...]] = {}
    selected: set[_Cell] = set()
    for i, j, k in product(*(range(len(values) - 1) for values in coordinates)):
        cell = i, j, k
        point = (
            (Fraction(coordinates[0][i]) + Fraction(coordinates[0][i + 1])) / 2,
            (Fraction(coordinates[1][j]) + Fraction(coordinates[1][j + 1])) / 2,
            (Fraction(coordinates[2][k]) + Fraction(coordinates[2][k + 1])) / 2,
        )
        members = _membership(point, rectangles, models)
        membership[cell] = members
        if _selected(operation_, members):
            selected.add(cell)
    certificate = canonical_fingerprint(
        {
            "kind": "native-rectilinear-cad-boolean",
            "sources": [model.model_id for model in models],
            "operation": operation_,
            "selected_cells": sorted(selected),
            "coordinates": coordinates,
        }
    )
    source = _source_revision(models, certificate)
    components = _components(selected)
    faces = _boundary(components, coordinates, rectangles, policy_)
    model = _assemble(models, coordinates, faces, policy_, certificate)
    graph = _graph(source, model, faces, components, membership, certificate)
    mapped = {edge.source_occurrence_id for edge in graph.transaction.correspondences}
    deleted = tuple(
        occurrence.occurrence_id
        for occurrence in source.occurrences
        if occurrence.occurrence_id not in mapped
    )
    return BRepBooleanResult(model, graph, operation_, certificate, deleted)


def boolean_brep(
    first: BRepModel,
    second: BRepModel,
    operation: BRepBooleanOperation,
    /,
    *,
    policy: BRepBooleanPolicy | None = None,
) -> BRepBooleanResult:
    """Exact native Boolean of authored physical instances with source lineage."""
    from ._placement import _entity_occurrence, _entity_text, materialize_brep_occurrences

    if not isinstance(first, BRepModel) or not isinstance(second, BRepModel):
        raise TypeError("Boolean operands must be BRepModel values.")
    policy_ = BRepBooleanPolicy() if policy is None else policy
    if not isinstance(policy_, BRepBooleanPolicy):
        raise TypeError("policy must be a BRepBooleanPolicy.")
    originals = (first, second)
    physical_instances = False
    for model in originals:
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "Boolean operands require authoritative source geometry."
            )
        physical_instances |= (
            any(
                occurrence.rotation != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                or occurrence.translation != (0.0, 0.0, 0.0)
                for occurrence in geometry.occurrences
            )
            or len(geometry.occurrences) != model.topology.num_solids
        )
    if not physical_instances:
        return _boolean_world(first, second, operation, policy=policy_)
    lifts = tuple(
        materialize_brep_occurrences(model, tessellation=policy_.tessellation)
        for model in originals
    )
    result = _boolean_world(lifts[0].model, lifts[1].model, operation, policy=policy_)
    certificate = canonical_fingerprint(
        {
            "kind": "exact-placed-cad-boolean-lineage",
            "sources": [model.model_id for model in originals],
            "placements": [lift.certificate_id for lift in lifts],
            "boolean": result.certificate_id,
        }
    )
    occurrences: list[CADOccurrence] = []
    replacement: dict[str, tuple[str, ...]] = {}
    for operand, lift in enumerate(lifts):
        reverse: dict[tuple[str, int], list[str]] = {}
        source_inventory = {
            occurrence.occurrence_id: occurrence
            for occurrence in lift.association_graph.source_revision.occurrences
        }
        for source, target in zip(
            lift.source_entities, lift.target_entities, strict=True
        ):
            if source.kind not in ("solid", "face"):
                continue
            identifier = f"operand:{operand}/{_entity_occurrence(source)}"
            source_occurrence = source_inventory[_entity_occurrence(source)]
            occurrences.append(
                CADOccurrence(
                    certificate,
                    identifier,
                    _entity_text(source),
                    source.kind,
                    (identifier,),
                    orientation=source_occurrence.orientation,
                )
            )
            reverse.setdefault((target.kind, target.index), []).append(identifier)
        for source_occurrence in result.association_graph.source_revision.occurrences:
            prefix = f"operand:{operand}/"
            if not source_occurrence.occurrence_id.startswith(prefix):
                continue
            fields = source_occurrence.occurrence_id[len(prefix) :].split("/")
            solid = int(fields[0].removeprefix("solid:"))
            if source_occurrence.kind == "solid":
                replacement[source_occurrence.occurrence_id] = tuple(
                    reverse[("solid", solid)]
                )
            elif source_occurrence.kind == "face":
                face = int(fields[1].removeprefix("face:"))
                replacement[source_occurrence.occurrence_id] = tuple(
                    reverse[("face", face)]
                )
    source_revision = CADRevision(
        certificate,
        f"native-placed-boolean-sources:{certificate}",
        tuple(occurrences),
        certificate,
    )
    pairs = {
        (original, edge.target_occurrence_id)
        for edge in result.association_graph.transaction.correspondences
        for original in replacement[edge.source_occurrence_id]
    }
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source_revision.revision_id,
        result.association_graph.target_revision.revision_id,
        tuple(
            OccurrenceCorrespondence(source, target, certificate)
            for source, target in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-exact-placed-cad-boolean"
        ),
    )
    graph = AssociationGraph(
        source_revision, result.association_graph.target_revision, transaction
    )
    mapped = {source for source, _ in pairs}
    deleted = tuple(
        sorted(
            occurrence.occurrence_id
            for occurrence in occurrences
            if occurrence.occurrence_id not in mapped
        )
    )
    return BRepBooleanResult(result.model, graph, result.operation, certificate, deleted)


__all__ = [
    "BRepBooleanFailure",
    "BRepBooleanOperation",
    "BRepBooleanPolicy",
    "BRepBooleanResult",
    "boolean_brep",
]

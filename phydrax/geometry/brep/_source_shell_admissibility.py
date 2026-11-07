#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-authoritative geometric premises for CAD shell admissibility.

A pair certificate here is only a disjointness premise, not a shell certificate:
other face pairs, individual face embeddings, and material nesting must also be
proved independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._mesh_certificates import _same_tree_content
from ._boolean_coincidence import prove_surface_correspondence
from ._boolean_core import boolean_face_chart_box
from ._correspondence import _periodic_pcurve_source, exact_source_lift
from ._intersection import intersect_surface_regions
from ._intersection_curve import (
    _orthogonal_cylinder_curve_branch_sign,
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    orthogonal_cylinder_family,
    SurfaceRegion,
)
from ._model import BRepModel
from ._source_embedding import certify_source_embedding


if TYPE_CHECKING:
    from .._surface_source_support import SurfaceSourceCharts
    from ._curve_approximation_topology import TrimReferenceSegment
    from ._intersection_curve import IntersectionSide
    from ._patches import CircleCurve


_IntervalBox = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


def _rational_frame(
    values: tuple[tuple[object, ...], ...], /
) -> TypeGuard[tuple[tuple[Fraction, ...], ...]]:
    return all(isinstance(value, Fraction) for vector in values for value in vector)


def _rational_components(
    values: tuple[object, ...], /
) -> TypeGuard[tuple[Fraction, ...]]:
    return all(isinstance(value, Fraction) for value in values)


def _trim_enclosure(model: BRepModel, face: int, /) -> _IntervalBox | None:
    """Enclose the actual bounded trim, including source-root endpoint intervals."""
    domain = model.trim_domains[face]
    if domain is None or domain.numeric_only:
        return None
    boxes: list[np.ndarray] = []
    if domain.curve_loops:
        for loop in domain.curve_loops:
            for segment in loop:
                if not isinstance(segment, CurveTrimSegment):
                    return None
                lower = (
                    segment.first
                    if segment.first_root is None
                    else segment.first_root.parameter_enclosure()[0]
                )
                upper = (
                    segment.last
                    if segment.last_root is None
                    else segment.last_root.parameter_enclosure()[1]
                )
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    return None
                box = np.asarray(
                    segment.curve.bounding_box(lower, upper), dtype=np.float64
                )
                if (
                    box.shape != (2, 2)
                    or not np.all(np.isfinite(box))
                    or np.any(box[0] > box[1])
                ):
                    return None
                boxes.append(box)
    else:
        if domain.outer is None:
            return None
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        if (
            bounds.shape != (2, 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[0] >= bounds[1])
        ):
            return None
        lo = tuple(Fraction.from_float(float(value)) for value in bounds[0])
        hi = tuple(Fraction.from_float(float(value)) for value in bounds[1])
        native_points: list[tuple[Fraction, Fraction]] = []
        for loop in (domain.outer, *domain.holes):
            points = np.asarray(loop, dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not np.all(np.isfinite(points))
            ):
                return None
            native_points.extend(
                (
                    lo[0] + Fraction.from_float(float(point[0])) * (hi[0] - lo[0]),
                    lo[1] + Fraction.from_float(float(point[1])) * (hi[1] - lo[1]),
                )
                for point in points
            )
        if not native_points:
            return None
        return (
            (
                min(point[0] for point in native_points),
                min(point[1] for point in native_points),
            ),
            (
                max(point[0] for point in native_points),
                max(point[1] for point in native_points),
            ),
        )
    if not boxes:
        return None
    lower = np.min(np.stack([box[0] for box in boxes]), axis=0)
    upper = np.max(np.stack([box[1] for box in boxes]), axis=0)
    return (
        (Fraction.from_float(float(lower[0])), Fraction.from_float(float(lower[1]))),
        (Fraction.from_float(float(upper[0])), Fraction.from_float(float(upper[1]))),
    )


@dataclass(frozen=True, slots=True)
class SourceFaceSeparation:
    """Recomputable exact same-chart separation of two bounded source faces."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    trim_boxes: tuple[_IntervalBox, _IntervalBox]
    correspondence_id: str
    embedding_id: str
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        proof = certify_same_chart_face_separation(model, *self.faces)
        return proof == self


def certify_same_chart_face_separation(
    model: BRepModel, first: int, second: int, /
) -> SourceFaceSeparation | None:
    """Prove disjoint trimmed faces of one carrier on one injective chart sheet.

    Trim loop bounds enclose complete source curves using endpoint root
    enclosures. The bounded winding-defined trim interiors lie inside these
    bounds. A native analytic embedding on their union excludes all periodic
    aliases and other carrier self-identifications. Any unsupported source or
    unresolved enclosure returns None, never a disjointness claim.
    """
    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
    ):
        return None
    if first == second or not (
        0 <= first < len(model.patches) and 0 <= second < len(model.patches)
    ):
        return None
    first, second = sorted((first, second))
    try:
        model.require_current()
        correspondence = prove_surface_correspondence(
            model.patches[first], model.patches[second]
        )
        identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
        if (
            correspondence is None
            or correspondence.matrix != identity
            or correspondence.offset != (Fraction(0), Fraction(0))
        ):
            return None
        first_box = _trim_enclosure(model, first)
        second_box = _trim_enclosure(model, second)
        if first_box is None or second_box is None:
            return None
        if not any(
            first_box[1][axis] < second_box[0][axis]
            or second_box[1][axis] < first_box[0][axis]
            for axis in range(2)
        ):
            return None
        bounds: _IntervalBox = (
            (
                min(first_box[0][0], second_box[0][0]),
                min(first_box[0][1], second_box[0][1]),
            ),
            (
                max(first_box[1][0], second_box[1][0]),
                max(first_box[1][1], second_box[1][1]),
            ),
        )
        embedding = certify_source_embedding(model.patches[first], bounds)
        if embedding is None:
            return None
        payload = {
            "kind": "cad-source-same-chart-face-separation",
            "source_id": model.source_id,
            "model_id": model.model_id,
            "source_revision": model.source_revision,
            "faces": (first, second),
            "trim_boxes": tuple(
                tuple(
                    tuple((value.numerator, value.denominator) for value in point)
                    for point in box
                )
                for box in (first_box, second_box)
            ),
            "correspondence": correspondence.certificate_id,
            "embedding": embedding.source_id,
        }
        return SourceFaceSeparation(
            model.source_id,
            model.model_id,
            model.source_revision,
            (first, second),
            (first_box, second_box),
            correspondence.certificate_id,
            embedding.source_id,
            canonical_fingerprint(payload),
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


@dataclass(frozen=True, slots=True)
class SourceCoaxialRadialSeparation:
    """Replayable separation of the complete exact coaxial carriers."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        return certify_coaxial_radial_separation(model, *self.faces) == self


def certify_coaxial_radial_separation(
    model: BRepModel, first: int, second: int, /
) -> SourceCoaxialRadialSeparation | None:
    from ._intersection import _coaxial_radial_separation

    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
        or not 0 <= first < second < len(model.patches)
    ):
        return None
    try:
        model.require_current()
        common = set(model.geometry.topology.face_edges[first]) & set(
            model.geometry.topology.face_edges[second]
        )
        if common or not _coaxial_radial_separation(
            model.patches[first], model.patches[second]
        ):
            return None
        identity = canonical_fingerprint(
            {
                "kind": "cad-source-coaxial-radial-separation",
                "source": model.source_id,
                "model": model.model_id,
                "revision": model.source_revision,
                "faces": (first, second),
            }
        )
        return SourceCoaxialRadialSeparation(
            model.source_id,
            model.model_id,
            model.source_revision,
            (first, second),
            identity,
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


def _opposed_source_uses(
    model: BRepModel, faces: tuple[int, int], coedges: tuple[int, int], /
) -> bool:
    """Bind both edge occurrences to opposite oriented uses of one shell."""
    geometry = model.geometry
    if geometry.coedge_edges[coedges[0]] != geometry.coedge_edges[coedges[1]] or any(
        sum(
            coedge == candidate
            for loop in geometry.face_loops[face]
            for candidate in loop
        )
        != 1
        for face, coedge in zip(faces, coedges, strict=True)
    ):
        return False
    if len(geometry.solid_shells) != 1 or len(geometry.solid_shells[0]) != 1:
        return False
    shell = geometry.solid_shells[0][0]
    shell_faces = geometry.shell_faces[shell]
    signs = geometry.shell_orientations[shell]
    if any(shell_faces.count(face) != 1 for face in faces):
        return False
    oriented = tuple(
        int(model.orientation[face])
        * int(signs[shell_faces.index(face)])
        * int(geometry.coedge_senses[coedge])
        for face, coedge in zip(faces, coedges, strict=True)
    )
    return oriented[0] in (-1, 1) and oriented[0] == -oriented[1]


@dataclass(frozen=True, slots=True)
class SourceRingFacePair:
    """A current, full support-intersection ring bound to both face uses."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    shared_edges: tuple[int, ...]
    coedges: tuple[int, int]
    coverage_id: str
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        return certify_source_ring_face_pair(model, *self.faces) == self


def certify_source_ring_face_pair(
    model: BRepModel, first: int, second: int, /
) -> SourceRingFacePair | None:
    """Prove a complete plane/cylinder support intersection with a native ring."""
    from ._boolean_coincidence import prove_source_ring_pair_coverage
    from ._patches import CylinderPatch, PlanePatch

    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
        or not 0 <= first < second < len(model.patches)
    ):
        return None
    try:
        model.require_current()
        patches = (model.patches[first], model.patches[second])
        cylinder = next((p for p in patches if isinstance(p, CylinderPatch)), None)
        if cylinder is None or not any(isinstance(p, PlanePatch) for p in patches):
            return None
        # The coverage owner's support uses binary shadows. Require those to
        # be the actual exact carrier coefficients, not rounded algebraic data.
        for numeric, exact in (
            (cylinder.origin, cylinder.origin_exact),
            (cylinder.first_axis, cylinder.first_axis_exact),
            (cylinder.second_axis, cylinder.second_axis_exact),
            (cylinder.axis, cylinder.axis_exact),
        ):
            if tuple(Fraction.from_float(float(x)) for x in np.asarray(numeric)) != exact:
                return None
        if Fraction.from_float(float(cylinder.radius)) != cylinder.radius_exact:
            return None
        geometry = model.geometry
        common = set(geometry.topology.face_edges[first]) & set(
            geometry.topology.face_edges[second]
        )
        if len(common) != 1:
            return None
        edge = next(iter(common))
        occurrences = tuple(
            tuple(
                coedge
                for loop in geometry.face_loops[face]
                for coedge in loop
                if geometry.coedge_edges[coedge] == edge
            )
            for face in (first, second)
        )
        if any(len(uses) != 1 for uses in occurrences):
            return None
        coedges = (occurrences[0][0], occurrences[1][0])
        if not _opposed_source_uses(model, (first, second), coedges):
            return None
        coverage = prove_source_ring_pair_coverage(
            (model,),
            ((0, first), (0, second)),
            (0, edge),
            (geometry.pcurves[coedges[0]], geometry.pcurves[coedges[1]]),
        )
        if coverage is None:
            return None
        payload = {
            "kind": "cad-source-ring-face-pair",
            "source": model.source_id,
            "model": model.model_id,
            "revision": model.source_revision,
            "faces": (first, second),
            "edge": edge,
            "coedges": coedges,
            "senses": tuple(geometry.coedge_senses[c] for c in coedges),
            "coverage": coverage.certificate_id,
        }
        return SourceRingFacePair(
            model.source_id,
            model.model_id,
            model.source_revision,
            (first, second),
            (edge,),
            coedges,
            coverage.certificate_id,
            canonical_fingerprint(payload),
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


@dataclass(frozen=True, slots=True)
class SourceTangentCylinderTorusPair:
    """Replayable complete coaxial tangent-cylinder/torus intersection."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    shared_edges: tuple[int, ...]
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        return certify_tangent_cylinder_torus_pair(model, *self.faces) == self


def _shared_two_arc_circle(
    model: BRepModel, first: int, second: int, /
) -> (
    tuple[CircleCurve, tuple[int, ...], tuple[tuple[int, int, int], ...], tuple[str, ...]]
    | None
):
    """Bind two complete adjacent conic arcs to their two source occurrences."""
    from ._intersection import ConicParameterEndpoint
    from ._patches import _validate_conic_parameter_root, CircleCurve

    geometry = model.geometry
    edges = sorted(
        set(geometry.topology.face_edges[first])
        & set(geometry.topology.face_edges[second])
    )
    if len(edges) != 2:
        return None
    arcs: list[
        tuple[CircleCurve, tuple[ConicParameterEndpoint, ConicParameterEndpoint]]
    ] = []
    use_ids: list[tuple[int, int, int]] = []
    for edge in edges:
        index = geometry.edge_curves[edge]
        if index < 0:
            return None
        curve = geometry.curves[index]
        if not isinstance(curve, CircleCurve):
            return None
        raw_roots = geometry.edge_endpoint_roots[edge]
        first_root, second_root = raw_roots
        if not isinstance(first_root, ConicParameterEndpoint) or not isinstance(
            second_root, ConicParameterEndpoint
        ):
            return None
        roots = (first_root, second_root)
        # A Boolean-rebound endpoint retains its raw carrier parameter. Only the
        # exact cosine/sine/turn root below is used, so that parameter is
        # admissible solely as a representative inside the root's isolation.
        if any(
            root.parameter_exact is not None
            and not (
                root.root.parameter_lower
                <= root.parameter_exact
                <= root.root.parameter_upper
            )
            for root in roots
        ):
            return None
        for root in roots:
            _validate_conic_parameter_root(root.root)
            if root.root_id != canonical_fingerprint(root.identity_payload()):
                return None
        if any(
            not _same_tree_content(root.carrier, curve)
            or not _same_tree_content(root.root.conic, curve)
            for root in roots
        ):
            return None
        start, stop = map(float, geometry.edge_ranges[edge])
        if not all(
            root.root.parameter_lower <= value <= root.root.parameter_upper
            for root, value in zip(roots, (start, stop), strict=True)
        ):
            return None
        for face in (first, second):
            uses = [
                c
                for loop in geometry.face_loops[face]
                for c in loop
                if geometry.coedge_edges[c] == edge
            ]
            if len(uses) != 1:
                return None
            coedge = uses[0]
            region = SurfaceRegion(
                model.patches[face], boolean_face_chart_box(model, face)
            )
            if not exact_source_lift(
                curve, geometry.pcurves[coedge], region, start, stop
            ):
                return None
            use_ids.append((face, coedge, geometry.coedge_senses[coedge]))
        if not _opposed_source_uses(
            model, (first, second), (use_ids[-2][1], use_ids[-1][1])
        ):
            return None
        arcs.append((curve, roots))
    arcs.sort(key=lambda arc: float(arc[1][0].root.parameter_lower))
    (curve, start), (other_curve, end) = arcs
    if not _same_tree_content(curve, other_curve):
        return None
    a, b, c, d = start[0].root, start[1].root, end[0].root, end[1].root
    if (
        (a.cosine_exact, a.sine_exact, a.turns) != (1, 0, 0)
        or (d.cosine_exact, d.sine_exact, d.turns) != (1, 0, 1)
        or b.root_id != c.root_id
        or b.parameter_lower <= a.parameter_upper
        or b.parameter_upper >= d.parameter_lower
    ):
        return None
    return (
        curve,
        tuple(edges),
        tuple(use_ids),
        tuple(root.root_id for _, roots in arcs for root in roots),
    )


def certify_tangent_cylinder_torus_pair(
    model: BRepModel, first: int, second: int, /
) -> SourceTangentCylinderTorusPair | None:
    """Prove a coaxial tangent cylinder/torus intersection covered by two arcs."""
    from .._rational import rational_cross, rational_dot
    from ._patches import CylinderPatch, TorusPatch

    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
        or not 0 <= first < second < len(model.patches)
    ):
        return None
    try:
        model.require_current()
        patches = (model.patches[first], model.patches[second])
        cylinder = next((p for p in patches if isinstance(p, CylinderPatch)), None)
        torus = next((p for p in patches if isinstance(p, TorusPatch)), None)
        if cylinder is None or torus is None:
            return None
        for patch in (cylinder, torus):
            frame = (patch.first_axis_exact, patch.second_axis_exact, patch.axis_exact)
            if not _rational_frame(frame):
                return None
            if any(
                rational_dot(frame[i], frame[j]) != (1 if i == j else 0)
                for i in range(3)
                for j in range(3)
            ):
                return None
        if not all(
            isinstance(x, Fraction)
            for x in (
                *cylinder.origin_exact,
                *torus.center_exact,
                cylinder.radius_exact,
                torus.major_radius_exact,
                torus.minor_radius_exact,
            )
        ):
            return None
        if (
            torus.major_radius_exact <= torus.minor_radius_exact
            or torus.minor_radius_exact <= 0
            or cylinder.radius_exact
            not in (
                torus.major_radius_exact - torus.minor_radius_exact,
                torus.major_radius_exact + torus.minor_radius_exact,
            )
            or any(rational_cross(cylinder.axis_exact, torus.axis_exact))
            or any(
                rational_cross(
                    tuple(
                        a - b
                        for a, b in zip(
                            cylinder.origin_exact, torus.center_exact, strict=True
                        )
                    ),
                    torus.axis_exact,
                )
            )
        ):
            return None
        shared = _shared_two_arc_circle(model, first, second)
        if shared is None:
            return None
        _, edges, use_ids, roots = shared
        identity = canonical_fingerprint(
            {
                "kind": "cad-source-tangent-cylinder-torus-pair",
                "source": model.source_id,
                "model": model.model_id,
                "revision": model.source_revision,
                "faces": (first, second),
                "edges": edges,
                "uses": use_ids,
                "roots": roots,
            }
        )
        return SourceTangentCylinderTorusPair(
            model.source_id,
            model.model_id,
            model.source_revision,
            (first, second),
            tuple(edges),
            identity,
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


@dataclass(frozen=True, slots=True)
class SourcePlaneTorusPair:
    """A planar trim excludes the other exact coaxial torus section."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    shared_edges: tuple[int, ...]
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        return certify_plane_torus_pair(model, *self.faces) == self


def certify_plane_torus_pair(
    model: BRepModel, first: int, second: int, /
) -> SourcePlaneTorusPair | None:
    """Prove an annular planar trim retains just its shared torus circle."""
    from .._rational import rational_cross, rational_dot, rational_vector
    from ._model import _validate_curve_trim_domains
    from ._patches import PlanePatch, TorusPatch

    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
        or not 0 <= first < second < len(model.patches)
    ):
        return None
    try:
        model.require_current()
        plane_match = next(
            (
                (f, model.patches[f])
                for f in (first, second)
                if isinstance(model.patches[f], PlanePatch)
            ),
            None,
        )
        torus = next(
            (
                patch
                for patch in (model.patches[first], model.patches[second])
                if isinstance(patch, TorusPatch)
            ),
            None,
        )
        if plane_match is None or torus is None:
            return None
        plane_face, plane = plane_match
        if not isinstance(plane, PlanePatch):
            return None
        frame = (torus.first_axis_exact, torus.second_axis_exact, torus.axis_exact)
        if not _rational_frame(frame):
            return None
        if any(
            rational_dot(frame[i], frame[j]) != (1 if i == j else 0)
            for i in range(3)
            for j in range(3)
        ):
            return None
        torus_center = torus.center_exact
        if (
            not _rational_components(torus_center)
            or not isinstance(torus.major_radius_exact, Fraction)
            or not isinstance(torus.minor_radius_exact, Fraction)
        ):
            return None
        if (
            torus.major_radius_exact <= torus.minor_radius_exact
            or torus.minor_radius_exact <= 0
        ):
            return None
        normal = rational_cross(
            rational_vector(np.asarray(plane.first_axis)),
            rational_vector(np.asarray(plane.second_axis)),
        )
        torus_axis = frame[2]
        if not any(normal) or any(rational_cross(normal, torus_axis)):
            return None
        shared = _shared_two_arc_circle(model, first, second)
        if shared is None:
            return None
        curve, edges, uses, roots = shared
        circle_center = curve.center_exact
        circle_frame = (curve.first_axis_exact, curve.second_axis_exact, torus_axis)
        if (
            not _rational_components(circle_center)
            or not _rational_frame(circle_frame)
            or not isinstance(curve.radius_exact, Fraction)
        ):
            return None
        if any(
            rational_dot(circle_frame[i], circle_frame[j]) != (1 if i == j else 0)
            for i in range(3)
            for j in range(3)
        ):
            return None
        relative = tuple(a - b for a, b in zip(circle_center, torus_center, strict=True))
        if any(rational_cross(relative, torus_axis)):
            return None
        height = rational_dot(relative, torus_axis)
        radius = curve.radius_exact
        if radius <= 0 or (
            (radius - torus.major_radius_exact) ** 2 + height**2
            != torus.minor_radius_exact**2
        ):
            return None
        plane_origin = rational_vector(np.asarray(plane.origin))
        if (
            rational_dot(
                normal,
                tuple(a - b for a, b in zip(circle_center, plane_origin, strict=True)),
            )
            != 0
        ):
            return None
        domain = model.trim_domains[plane_face]
        if domain is None or domain.numeric_only or not domain.curve_loops:
            return None
        _validate_curve_trim_domains(model.geometry, model.trim_domains)
        matching = [
            index
            for index, loop in enumerate(model.geometry.face_loops[plane_face])
            if len(loop) == 2
            and {model.geometry.coedge_edges[c] for c in loop} == set(edges)
        ]
        if len(matching) != 1:
            return None
        loop_index = matching[0]
        # The first loop defines the bounded outer region; later loops are
        # excluded holes. The second torus section lies beyond that boundary.
        if not (
            (loop_index == 0 and radius < torus.major_radius_exact)
            or (loop_index > 0 and radius > torus.major_radius_exact)
        ):
            return None
        identity = canonical_fingerprint(
            {
                "kind": "cad-source-plane-torus-pair",
                "source": model.source_id,
                "model": model.model_id,
                "revision": model.source_revision,
                "faces": (first, second),
                "edges": edges,
                "uses": uses,
                "roots": roots,
                "plane_loop": loop_index,
            }
        )
        return SourcePlaneTorusPair(
            model.source_id,
            model.model_id,
            model.source_revision,
            (first, second),
            edges,
            identity,
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


@dataclass(frozen=True, slots=True)
class SourceFaceBranchCoverage:
    """Replayable complete face-pair branches bound to shared source edges."""

    source_id: str
    model_id: str
    source_revision: str
    faces: tuple[int, int]
    shared_edges: tuple[int, ...]
    retained_branch_contents: tuple[tuple[int, str], ...]
    excluded_branches: tuple[str, ...]
    branch_content_ids: tuple[str, ...]
    certificate_id: str

    def replay(self, model: BRepModel, /) -> bool:
        return certify_retained_branch_pair(model, *self.faces) == self


def _branch_outside_trim(
    branch: IntersectionCurve,
    region: SurfaceRegion,
    side: IntersectionSide,
    trim: _IntervalBox,
    /,
) -> bool:
    """Exclude a branch from one face without merging periodic chart sheets."""
    pcurve = branch.p_curve(side)
    box = np.asarray(pcurve.bounding_box(0.0, float(branch.num_charts)), dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1]):
        return False
    branch_box: _IntervalBox = (
        (Fraction.from_float(float(box[0, 0])), Fraction.from_float(float(box[0, 1]))),
        (Fraction.from_float(float(box[1, 0])), Fraction.from_float(float(box[1, 1]))),
    )
    separated = tuple(
        axis
        for axis in range(2)
        if branch_box[1][axis] < trim[0][axis] or trim[1][axis] < branch_box[0][axis]
    )
    if not separated:
        return False
    union: _IntervalBox = (
        (min(branch_box[0][0], trim[0][0]), min(branch_box[0][1], trim[0][1])),
        (max(branch_box[1][0], trim[1][0]), max(branch_box[1][1], trim[1][1])),
    )
    if certify_source_embedding(region.patch, union) is not None:
        return True
    # Outward-rounded full-turn trims exceed one period, so no rational box is
    # an embedded sheet. Separation on the non-periodic axis holds at every
    # periodic coordinate; the exact full-turn sheet with its declared deck
    # pairing then separates the branch from the trim modulo the period.
    from ._boolean_core import ChartScalar
    from ._source_embedding import _TAU

    periods = region.patch.native_period_symbols()
    for axis in separated:
        other = 1 - axis
        if periods[axis] is not None or periods[other] != "two_pi":
            continue
        lower: list[ChartScalar] = [Fraction(0), Fraction(0)]
        upper: list[ChartScalar] = [_TAU, _TAU]
        lower[axis], upper[axis] = union[0][axis], union[1][axis]
        if (
            certify_source_embedding(
                region.patch,
                ((lower[0], lower[1]), (upper[0], upper[1])),
                deck_axes=(other,),
            )
            is not None
        ):
            return True
    return False


def _shared_branch_edge(
    model: BRepModel,
    faces: tuple[int, int],
    regions: tuple[SurfaceRegion, SurfaceRegion],
    branch: IntersectionCurve,
    /,
) -> int | None:
    """Bind a complete branch to one actual edge and both rooted coedges."""
    geometry = model.geometry
    common = set(geometry.topology.face_edges[faces[0]]) & set(
        geometry.topology.face_edges[faces[1]]
    )
    matches: list[int] = []
    for edge in common:
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            continue
        stored = geometry.curves[curve_index]
        if not isinstance(stored, IntersectionCurve) or not stored.fully_certified:
            continue
        if not _same_tree_content(stored, branch):
            continue
        start, end = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
        if start != 0.0 or end != float(branch.num_charts):
            continue
        roots = geometry.edge_endpoint_roots[edge]
        if any(
            root is None or root.parameter_enclosure() != (value, value)
            for root, value in zip(roots, (start, end), strict=True)
        ):
            continue
        vertices = geometry.edge_vertices[edge]
        if branch.closed and vertices[0] != vertices[1]:
            continue
        valid = True
        for face, region in zip(faces, regions, strict=True):
            coedges = [
                coedge
                for loop in geometry.face_loops[face]
                for coedge in loop
                if geometry.coedge_edges[coedge] == edge
            ]
            if len(coedges) != 1:
                valid = False
                break
            coedge = coedges[0]
            pcurve = geometry.pcurves[coedge]
            periodic = _periodic_pcurve_source(pcurve, region.patch)
            if periodic is None:
                valid = False
                break
            base, _ = periodic
            if (
                not isinstance(base, IntersectionPCurve)
                or not _same_tree_content(base.curve, stored)
                or not exact_source_lift(stored, base, region, start, end)
            ):
                valid = False
                break
            for vertex, endpoint in zip(
                vertices, geometry.coedge_endpoint_roots[coedge], strict=True
            ):
                vertex_root = geometry.vertex_roots[vertex]
                if (
                    vertex_root is None
                    or endpoint is None
                    or not vertex_root.same_uv_endpoint(
                        region.patch, pcurve, endpoint, pcurve, endpoint
                    )
                ):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            matches.append(edge)
    return matches[0] if len(matches) == 1 else None


def _global_orthogonal_branches(
    model: BRepModel,
    regions: tuple[SurfaceRegion, SurfaceRegion],
    curves: tuple[IntersectionCurve, ...],
    /,
) -> bool:
    """Prove both exact cylinder support loops have degree one and opposite signs."""
    family = orthogonal_cylinder_family(regions[0], regions[1])
    if family is None or len(curves) != 2:
        return False
    signs: set[int] = set()
    geometry = model.geometry
    for branch in curves:
        sign = _orthogonal_cylinder_curve_branch_sign(branch, family)
        if (
            sign not in (-1, 1)
            or sign in signs
            or int(branch.closure_period_shifts()[2 * family.small_index]) != 1
        ):
            return False
        signs.add(sign)
        rooted = False
        for edge, curve_index in enumerate(geometry.edge_curves):
            if curve_index < 0:
                continue
            stored = geometry.curves[curve_index]
            if not isinstance(stored, IntersectionCurve) or not _same_tree_content(
                stored, branch
            ):
                continue
            if tuple(
                float(value) for value in np.asarray(geometry.edge_ranges[edge])
            ) != (0.0, float(branch.num_charts)):
                continue
            first_vertex, last_vertex = geometry.edge_vertices[edge]
            if first_vertex != last_vertex:
                continue
            vertex_root = geometry.vertex_roots[first_vertex]
            first_root, last_root = geometry.edge_endpoint_roots[edge]
            if (
                vertex_root is not None
                and first_root is not None
                and last_root is not None
                and first_root.parameter_enclosure() == (0.0, 0.0)
                and last_root.parameter_enclosure()
                == (float(branch.num_charts), float(branch.num_charts))
                and vertex_root.supports_endpoint(first_root)
                and vertex_root.supports_endpoint(last_root)
            ):
                rooted = True
                break
        if not rooted:
            return False
    return signs == {-1, 1}


def certify_retained_branch_pair(
    model: BRepModel, first: int, second: int, /
) -> SourceFaceBranchCoverage | None:
    """Prove all pair intersections are declared, complete source-edge branches.

    Uses the exhaustive native intersection owner on containing face regions;
    unresolved, isolated, coincident or unbound contacts refuse this route.
    Unused branches must be separated from a complete trimmed face on one
    source-injective sheet. This is a face-pair premise, not a shell proof.
    """
    if (
        not isinstance(model, BRepModel)
        or type(first) is not int
        or type(second) is not int
        or first == second
        or not (0 <= first < len(model.patches) and 0 <= second < len(model.patches))
    ):
        return None
    faces = (min(first, second), max(first, second))
    try:
        model.require_current()
        trims = tuple(_trim_enclosure(model, face) for face in faces)
        if any(trim is None for trim in trims):
            return None
        regions = (
            SurfaceRegion(
                model.patches[faces[0]], boolean_face_chart_box(model, faces[0])
            ),
            SurfaceRegion(
                model.patches[faces[1]], boolean_face_chart_box(model, faces[1])
            ),
        )
        needs_global = False
        domain = None
        for face, region, trim in zip(faces, regions, trims, strict=True):
            if trim is None:
                return None
            box = np.asarray(region.parameter_box, dtype=np.float64)
            for axis in range(2):
                if trim[0][axis] >= Fraction.from_float(float(box[0, axis])) and trim[1][
                    axis
                ] <= Fraction.from_float(float(box[1, axis])):
                    continue
                if axis != 0:
                    return None
                from .._meshing_domain import MeshingDomain
                from ._source_embedding import certify_brep_trim_sheet

                if domain is None:
                    domain = MeshingDomain.from_brep(model)
                matches = [
                    patch
                    for patch, source_face in enumerate(domain.source_indices[2])
                    if source_face == face
                ]
                if len(matches) != 1:
                    return None
                sheet = certify_brep_trim_sheet(domain, matches[0])
                if sheet is None or axis not in sheet.embedding.deck_axes:
                    return None
                needs_global = True
        result = intersect_surface_regions(regions[0], regions[1])
        if not result.complete or result.points or result.coincident or not result.curves:
            return None
        if needs_global and not _global_orthogonal_branches(
            model, regions, result.curves
        ):
            return None
        retained: list[int] = []
        excluded: list[str] = []
        branch_contents: list[str] = []
        retained_contents: list[tuple[int, str]] = []
        for branch in result.curves:
            content_id = canonical_fingerprint(
                {
                    "branch_id": branch.branch_id,
                    "array_tree": array_tree_fingerprint(branch),
                }
            )
            branch_contents.append(content_id)
            edge = _shared_branch_edge(model, faces, regions, branch)
            if edge is not None:
                if edge in retained:
                    return None
                retained.append(edge)
                retained_contents.append((edge, content_id))
                continue
            if not any(
                _branch_outside_trim(branch, region, side, trim)
                for region, side, trim in zip(
                    regions, ("first", "second"), trims, strict=True
                )
                if trim is not None
            ):
                return None
            excluded.append(branch.branch_id)
        if not retained:
            return None
        retained_edges = tuple(sorted(retained))
        retained_branch_contents = tuple(sorted(retained_contents))
        branch_content_ids = tuple(sorted(branch_contents))
        excluded_branches = tuple(sorted(excluded))
        certificate = canonical_fingerprint(
            {
                "kind": "cad-source-retained-branch-pair",
                "source_id": model.source_id,
                "model_id": model.model_id,
                "source_revision": model.source_revision,
                "faces": faces,
                "shared_edges": retained_edges,
                "retained_branch_contents": retained_branch_contents,
                "excluded_branches": excluded_branches,
                "branch_content_ids": branch_content_ids,
            }
        )
        return SourceFaceBranchCoverage(
            model.source_id,
            model.model_id,
            model.source_revision,
            faces,
            retained_edges,
            retained_branch_contents,
            excluded_branches,
            branch_content_ids,
            certificate,
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None


@dataclass(frozen=True, slots=True)
class SourceShellAdmissibility:
    """Source-bound proof of one embedded, outward, closed material shell."""

    source_id: str
    model_id: str
    source_revision: str
    source_chart_id: str
    tolerance: float
    pair_certificates: tuple[str, ...]
    sheet_certificates: tuple[str, ...]
    certificate_id: str

    def replay(self, charts: object, /) -> bool:
        return certify_single_source_shell(charts, self.tolerance) == self


def _single_shell_topology(model: BRepModel, domain: object, /) -> bool:
    """Prove edge balance and cyclic vertex links after exact pole quotient."""
    from .._domain_boundary import _exact_line_junction_box
    from .._meshing_domain import _same_brep_trim_endpoint, MeshingDomain
    from ._correspondence import _exact_brep_pole_collapse

    if not isinstance(domain, MeshingDomain) or domain.brep_authority is not model:
        return False
    geometry = model.geometry
    if len(geometry.solid_shells) != 1 or len(geometry.solid_shells[0]) != 1:
        return False
    shell = geometry.solid_shells[0][0]
    if not 0 <= shell < len(geometry.shell_faces):
        return False
    faces = geometry.shell_faces[shell]
    signs = geometry.shell_orientations[shell]
    if len(set(faces)) != len(faces) or set(faces) != set(range(len(model.patches))):
        return False
    orientation = np.asarray(model.orientation, dtype=np.int8)
    face_graph: dict[int, set[int]] = {face: set() for face in faces}
    edge_uses: dict[int, list[tuple[int, int]]] = {}
    links: dict[int, dict[tuple[int, int], list[tuple[int, int]]]] = {}
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    patch_by_face = {
        int(face): patch for patch, face in enumerate(domain.source_indices[2])
    }
    for face, shell_sign in zip(faces, signs, strict=True):
        face_sign = int(orientation[face]) * int(shell_sign)
        if face_sign not in (-1, 1):
            return False
        region = SurfaceRegion(model.patches[face], boolean_face_chart_box(model, face))
        patch = patch_by_face.get(face)
        if patch is None:
            return False
        for loop_index, loop in enumerate(geometry.face_loops[face]):
            if not loop:
                return False
            uses = domain.patches[patch].loops[loop_index]
            for index, (coedge, following) in enumerate(
                zip(loop, (*loop[1:], loop[0]), strict=True)
            ):
                edge, next_edge = (
                    geometry.coedge_edges[coedge],
                    geometry.coedge_edges[following],
                )
                end_slot = 1 if geometry.coedge_senses[coedge] > 0 else 0
                next_slot = 0 if geometry.coedge_senses[following] > 0 else 1
                if (
                    geometry.edge_vertices[edge][end_slot]
                    != geometry.edge_vertices[next_edge][next_slot]
                ):
                    return False
                first_use, second_use = uses[index], uses[(index + 1) % len(uses)]
                if (
                    not _same_brep_trim_endpoint(
                        model, face, coedge, following, first_use, second_use
                    )
                    and _exact_line_junction_box(
                        model, face, coedge, following, first_use, second_use
                    )
                    is None
                ):
                    return False
            regular = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                sense = geometry.coedge_senses[coedge]
                curve_index = geometry.edge_curves[edge]
                if sense not in (-1, 1):
                    return False
                start, stop = (float(value) for value in ranges[edge])
                if curve_index < 0:
                    vertex, other = geometry.edge_vertices[edge]
                    if vertex != other or not _exact_brep_pole_collapse(
                        None,
                        model.patches[face],
                        geometry.pcurves[coedge],
                        start,
                        stop,
                        geometry.vertex_exact_points[vertex],
                        geometry.vertex_roots[vertex],
                        geometry.coedge_endpoint_roots[coedge],
                    ):
                        return False
                    continue
                periodic = _periodic_pcurve_source(geometry.pcurves[coedge], region.patch)
                if periodic is None or not exact_source_lift(
                    geometry.curves[curve_index], periodic[0], region, start, stop
                ):
                    return False
                edge_uses.setdefault(edge, []).append((face, sense * face_sign))
                regular.append(coedge)
            if not regular:
                return False
            for index, coedge in enumerate(regular):
                following = regular[(index + 1) % len(regular)]
                edge, next_edge = (
                    geometry.coedge_edges[coedge],
                    geometry.coedge_edges[following],
                )
                sense, next_sense = (
                    geometry.coedge_senses[coedge],
                    geometry.coedge_senses[following],
                )
                end_slot = 1 if sense > 0 else 0
                next_slot = 0 if next_sense > 0 else 1
                vertex = geometry.edge_vertices[edge][end_slot]
                if geometry.edge_vertices[next_edge][next_slot] != vertex:
                    return False
                first_node, second_node = (edge, end_slot), (next_edge, next_slot)
                graph = links.setdefault(vertex, {})
                graph.setdefault(first_node, []).append(second_node)
                graph.setdefault(second_node, []).append(first_node)
    for uses in edge_uses.values():
        if len(uses) != 2 or uses[0][1] + uses[1][1] != 0:
            return False
        first_face, second_face = uses[0][0], uses[1][0]
        face_graph[first_face].add(second_face)
        face_graph[second_face].add(first_face)
    for graph in links.values():
        if not graph or any(len(neighbors) != 2 for neighbors in graph.values()):
            return False
        reached: set[tuple[int, int]] = set()
        pending = [next(iter(graph))]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(graph[node])
        if reached != set(graph):
            return False
    reached_faces: set[int] = set()
    pending_faces = [faces[0]]
    while pending_faces:
        face = pending_faces.pop()
        if face not in reached_faces:
            reached_faces.add(face)
            pending_faces.extend(face_graph[face])
    return reached_faces == set(faces)


def _reference_patch_partition(
    charts: SurfaceSourceCharts,
    patch: int,
    loops: tuple[tuple[TrimReferenceSegment, ...], ...],
    /,
) -> bool:
    """Prove the reference 2-chain fills exactly the rooted trim loops."""
    from .._approximate_source_coverage import (
        _area2,
        _intersection_area,
        _possibly_overlapping_pairs,
        _triangle2,
    )
    from .._meshing_domain import PatchPoleUse

    triangles = tuple(
        _triangle2(charts.root_parameter_chart(row))
        for row, owner in enumerate(charts.root_patches)
        if int(owner) == patch
    )
    if not triangles:
        return False
    has_pole = any(
        isinstance(use, PatchPoleUse)
        for loop in charts.domain.patches[patch].loops
        for use in loop
    )
    slivers = (
        charts.boundary_source.collapsed_pole_slivers(patch, triangles)
        if has_pole
        else ()
    )
    triangles = (*triangles, *(sliver for _, sliver in slivers))
    signs = tuple(1 if _area2(t) > 0 else -1 if _area2(t) < 0 else 0 for t in triangles)
    if 0 in signs or len(set(signs)) != 1:
        return False
    for first, second in _possibly_overlapping_pairs(triangles, triangles, True):
        if _intersection_area(triangles[first], triangles[second]) > 0:
            return False

    def edge(first: tuple, last: tuple) -> tuple[tuple, int]:
        return ((first, last), 1) if first < last else ((last, first), -1)

    vertices = (
        {point for triangle in triangles for point in triangle}
        | {
            point
            for loop in loops
            for segment in loop
            for point in (segment.start, segment.stop)
        }
        if slivers
        else set()
    )

    def parts(first: tuple, last: tuple) -> tuple[tuple[tuple, int], ...]:
        dx, dy = last[0] - first[0], last[1] - first[1]
        length = dx * dx + dy * dy
        if not length:
            return ()
        if not slivers:
            return (edge(first, last),)
        points = sorted(
            (
                point
                for point in vertices
                if (point[0] - first[0]) * dy == (point[1] - first[1]) * dx
                and 0 <= (point[0] - first[0]) * dx + (point[1] - first[1]) * dy <= length
            ),
            key=lambda point: (point[0] - first[0]) * dx + (point[1] - first[1]) * dy,
        )
        return tuple(edge(a, b) for a, b in zip(points, points[1:]))

    incidence: dict[tuple, list[int]] = {}
    for triangle in triangles:
        for first, last in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
            for key, sign in parts(first, last):
                incidence.setdefault(key, []).append(sign)
    boundary: dict[tuple, int] = {}
    for key, directions in incidence.items():
        if len(directions) == 1:
            boundary[key] = directions[0]
        elif len(directions) != 2 or sum(directions) != 0:
            return False
    authored: dict[tuple, int] = {}
    for loop in loops:
        for segment in loop:
            for key, sign in parts(segment.start, segment.stop):
                if key in authored:
                    return False
                authored[key] = sign
    return boundary == authored or boundary == {
        key: -sign for key, sign in authored.items()
    }


def certify_single_source_shell(
    charts: object, tolerance: float, /
) -> SourceShellAdmissibility | None:
    """Prove one actual source shell, refusing unsupported or unresolved premises.

    Source-bound charts supply only replayable coedge reference chains for the
    exact trim homotopies and sheet proof; no target mesh embedding or mesh
    distance is used as a source-shell premise.
    """
    from ..._bvh import bvh_overlap_pair_blocks, prepare_bvh
    from .._approximate_source_coverage import _cad_trim_reference_loops_prepared
    from .._meshing_domain import _PlacedSurfacePatch, SourcePreparation
    from .._rational import fraction_down, fraction_up
    from .._surface_source_support import (
        _cad_root_edges,
        SurfaceSourceCharts,
        validate_cad_curve_ancestry,
    )
    from ._assembly import BRepAffinePlacement
    from ._curve_approximation_topology import certify_trim_reference_homotopy
    from ._measure import brep_shell_volume_sign
    from ._source_embedding import _certify_cad_trim_sheet_validated

    if (
        not isinstance(charts, SurfaceSourceCharts)
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        return None
    try:
        charts.require_current()
        domain = charts.domain
        model = domain.brep_authority
        if (
            not isinstance(model, BRepModel)
            or domain.accuracy != "exact_represented"
            or domain.tolerance != 0.0
            or model.source_id != charts.source_id
            or model.source_revision != charts.source_revision
        ):
            return None
        model.require_current()
        patch_faces = tuple(int(value) for value in domain.source_indices[2])
        if len(patch_faces) != len(model.patches) or set(patch_faces) != set(
            range(len(model.patches))
        ):
            return None
        identity = BRepAffinePlacement.identity().identity_payload()
        for patch, face in enumerate(patch_faces):
            surface = domain.patches[patch].surface
            if isinstance(surface, _PlacedSurfacePatch):
                if surface.placement.identity_payload() != identity:
                    return None
                surface = surface.source
            if not _same_tree_content(surface, model.patches[face]):
                return None
        if not _single_shell_topology(model, domain):
            return None
        bindings = validate_cad_curve_ancestry(charts, tolerance)
        preparation = SourcePreparation(domain)
        _, adjacency = _cad_root_edges(charts)
        sheet_certificates: list[str] = []
        for patch in range(len(domain.patches)):
            loops = _cad_trim_reference_loops_prepared(
                charts, bindings, patch, preparation, adjacency
            )
            sheet = _certify_cad_trim_sheet_validated(
                charts, bindings, patch, preparation, reference=loops
            )
            if sheet is None:
                return None
            certify_trim_reference_homotopy(loops, preparation)
            if not _reference_patch_partition(charts, patch, loops):
                return None
            sheet_certificates.append(
                canonical_fingerprint(
                    {
                        "face": patch_faces[patch],
                        "source": sheet.embedding.source_id,
                        "reference": sheet.reference_id,
                    }
                )
            )
        boxes = []
        for face, patch in enumerate(model.patches):
            trim = _trim_enclosure(model, face)
            if trim is None:
                return None
            uv = np.asarray(
                (
                    (fraction_down(trim[0][0]), fraction_down(trim[0][1])),
                    (fraction_up(trim[1][0]), fraction_up(trim[1][1])),
                ),
                dtype=np.float64,
            )
            box = np.asarray(patch.bounding_box(uv), dtype=np.float64)
            if (
                box.shape != (2, 3)
                or not np.all(np.isfinite(box))
                or np.any(box[0] > box[1])
            ):
                return None
            boxes.append(box)
        bank = np.stack(boxes)
        tree = prepare_bvh(bank[:, 0, :], bank[:, 1, :], dtype=np.float64)
        candidates: set[tuple[int, int]] = set()
        for left, right in bvh_overlap_pair_blocks(tree, tree, include_touching=True):
            candidates.update(
                (int(a), int(b)) for a, b in zip(left, right, strict=True) if a < b
            )
        for first in range(len(model.patches)):
            for edge in model.geometry.topology.face_edges[first]:
                for second in model.geometry.topology.edge_faces[edge]:
                    if first < second and (first, second) not in candidates:
                        return None
        pair_certificates: list[str] = []
        for first, second in sorted(candidates):
            common = set(model.geometry.topology.face_edges[first]) & set(
                model.geometry.topology.face_edges[second]
            )
            separate = certify_same_chart_face_separation(model, first, second)
            if separate is not None:
                if common:
                    return None
                pair_certificates.append(separate.certificate_id)
                continue
            radial = certify_coaxial_radial_separation(model, first, second)
            if radial is not None:
                pair_certificates.append(radial.certificate_id)
                continue
            ring = certify_source_ring_face_pair(model, first, second)
            if ring is not None:
                pair_certificates.append(ring.certificate_id)
                continue
            tangent = certify_tangent_cylinder_torus_pair(model, first, second)
            if tangent is not None:
                pair_certificates.append(tangent.certificate_id)
                continue
            plane_torus = certify_plane_torus_pair(model, first, second)
            if plane_torus is not None:
                pair_certificates.append(plane_torus.certificate_id)
                continue
            from ._boolean_full_overlay import _coaxial_faces_meet_on_shared_circle

            regions = (
                SurfaceRegion(model.patches[first], boolean_face_chart_box(model, first)),
                SurfaceRegion(
                    model.patches[second], boolean_face_chart_box(model, second)
                ),
            )
            if not common and _coaxial_faces_meet_on_shared_circle(
                model, (first, second), regions
            ):
                pair_certificates.append(
                    canonical_fingerprint(
                        {
                            "kind": "cad-source-coaxial-pair",
                            "model": model.model_id,
                            "faces": (first, second),
                            "edges": tuple(sorted(common)),
                        }
                    )
                )
                continue
            pair = certify_retained_branch_pair(model, first, second)
            if pair is None or set(pair.shared_edges) != common:
                return None
            pair_certificates.append(pair.certificate_id)
        shell = model.geometry.solid_shells[0][0]
        if brep_shell_volume_sign(model, shell) != 1:
            return None
        pairs = tuple(pair_certificates)
        sheets = tuple(sheet_certificates)
        identity = canonical_fingerprint(
            {
                "kind": "cad-source-single-shell-admissibility",
                "source_id": model.source_id,
                "model_id": model.model_id,
                "source_revision": model.source_revision,
                "source_chart_id": charts.source_chart_id,
                "tolerance": float(tolerance).hex(),
                "pairs": pairs,
                "sheets": sheets,
                "volume_sign": 1,
            }
        )
        return SourceShellAdmissibility(
            model.source_id,
            model.model_id,
            model.source_revision,
            charts.source_chart_id,
            float(tolerance),
            pairs,
            sheets,
            identity,
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None

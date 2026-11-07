#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound displacement evidence for explicit trim completion.

The completed interpretation is a repair operation; it is not an assertion
that the original tolerance-bearing coedges formed an exact closed face.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from ._carrier_pieces import carrier_payload
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._model import BRepModel
from ._patches import CircleCurve, LineCurve
from ._repair_bounds import vertex_displacement_bound
from ._root_bindings import BRepConicVertex
from ._source_repair_roots import ExactPoint, line_parameter
from ._source_repair_trim import (
    circle_coedge_completion_bounds,
    line_coedge_completion_bounds,
    phase_interval,
)


@dataclass(frozen=True, slots=True)
class SourceCompletionCertificate:
    """Exact bounds and the explicitly completed interpretation's evidence."""

    max_vertex: Fraction
    max_edge: Fraction
    max_connector: Fraction
    face_source_to_target: Fraction
    face_target_to_source: Fraction
    completion_payload: dict[str, object]


def _fraction_payload(value: Fraction, /) -> list[int]:
    return [value.numerator, value.denominator]


def _vertex_point(model: BRepModel, index: int, /) -> ExactPoint:
    geometry = model.geometry
    root = geometry.vertex_roots[index]
    if root is not None:
        if not isinstance(root.primary, BRepConicVertex):
            raise ValueError("Source completion requires an exact conic vertex root.")
        return root.primary.point_exact
    exact = geometry.vertex_exact_points[index]
    if exact is not None:
        return cast(ExactPoint, exact)
    return cast(
        ExactPoint,
        tuple(
            Fraction.from_float(float(value))
            for value in np.asarray(geometry.vertex_points)[index]
        ),
    )


def _endpoint_interval(
    model: BRepModel,
    edge: int,
    endpoint: int,
    /,
    *,
    repaired: bool = False,
) -> tuple[Fraction, Fraction]:
    geometry = model.geometry
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise ValueError("Degenerate edges are not supported by this completion proof.")
    curve = geometry.curves[curve_index]
    hint = float(np.asarray(geometry.edge_ranges)[edge, endpoint])
    root = geometry.edge_endpoint_roots[edge][endpoint]
    if root is None:
        if repaired:
            if not isinstance(curve, LineCurve):
                raise ValueError(
                    "A repaired circular edge requires exact endpoint witnesses."
                )
            vertex = geometry.edge_vertices[edge][endpoint]
            if (
                geometry.vertex_roots[vertex] is None
                and geometry.vertex_exact_points[vertex] is None
            ):
                raise ValueError(
                    "A repaired line endpoint requires an exact vertex binding."
                )
            value = line_parameter(curve, _vertex_point(model, vertex))
            return value, value
        value = Fraction.from_float(hint)
        return value, value
    if isinstance(root, NativePeriodEndpoint):
        if carrier_payload(root.curve) != carrier_payload(curve):
            raise ValueError("A native edge phase must bind its actual edge carrier.")
        if root.patch is not None:
            patch_payload = carrier_payload(root.patch)
            incident = any(
                carrier_payload(model.patches[face]) == patch_payload
                for face, loops in enumerate(geometry.face_loops)
                for loop in loops
                for coedge in loop
                if geometry.coedge_edges[coedge] == edge
            )
            if (
                not incident
                or root.axis not in (0, 1)
                or root.patch.native_period_symbols()[root.axis] != "two_pi"
            ):
                raise ValueError(
                    "A surface-period edge phase must bind an incident native-period chart."
                )
        elif root.axis is not None:
            raise ValueError("A curve-owned period cannot have a surface chart axis.")
        if repaired:
            if (
                not isinstance(curve, CircleCurve)
                or root.rational != 0
                or root.turns.denominator != 1
            ):
                raise ValueError(
                    "Repaired native phases need an exact cardinal vertex witness."
                )
            vertex = geometry.edge_vertices[edge][endpoint]
            point = tuple(
                center + curve.radius_exact * axis
                for center, axis in zip(
                    curve.center_exact, curve.first_axis_exact, strict=True
                )
            )
            if _vertex_point(model, vertex) != point:
                raise ValueError("Native edge phase does not bind the repaired vertex.")
        return phase_interval(root.rational, turns=root.turns)
    if isinstance(root, ConicParameterEndpoint):
        if carrier_payload(root.carrier) != carrier_payload(curve):
            raise ValueError("An endpoint root must bind its actual edge carrier.")
        if (
            repaired
            and _vertex_point(model, geometry.edge_vertices[edge][endpoint])
            != root.point_exact
        ):
            raise ValueError("The edge endpoint root does not bind its repaired vertex.")
        if isinstance(curve, LineCurve):
            parameter = line_parameter(curve, root.point_exact)
            if root.parameter_exact != parameter:
                raise ValueError(
                    "The line endpoint's exact parameter is not bound to its point."
                )
            return parameter, parameter
        if isinstance(curve, CircleCurve):
            # A circle parameter is rational only at the exact phase-zero root.
            phase_zero = root.parameter_exact == 0 and (
                root.root.cosine_exact,
                root.root.sine_exact,
                root.root.turns,
            ) == (1, 0, 0)
            if (
                (root.parameter_exact is not None and not phase_zero)
                or carrier_payload(root.root.conic) != carrier_payload(curve)
                or root.root.exact_point() != root.point_exact
            ):
                raise ValueError(
                    "Circle endpoint phase must bind its own exact carrier and point."
                )
            return phase_interval(root.root)
    raise ValueError("The source edge endpoint has no supported exact parameter witness.")


def _require_same_incidence(source: BRepModel, target: BRepModel, /) -> None:
    first, second = source.geometry, target.geometry
    for name in (
        "edge_vertices",
        "coedge_edges",
        "coedge_senses",
        "face_loops",
        "shell_faces",
        "shell_orientations",
        "solid_shells",
    ):
        if getattr(first, name) != getattr(second, name):
            raise ValueError(f"Repair changed the authored {name} incidence.")
    if (
        len(first.vertex_roots) != len(second.vertex_roots)
        or len(first.edge_curves) != len(second.edge_curves)
        or len(source.patches) != len(target.patches)
        or not np.array_equal(
            np.asarray(source.orientation), np.asarray(target.orientation)
        )
    ):
        raise ValueError(
            "Repair changed the source entity ordering or face orientations."
        )


def certify_source_completion(
    source: BRepModel,
    target: BRepModel,
    /,
    *,
    tolerance: Fraction,
) -> SourceCompletionCertificate:
    """Prove the declared completion and repair against every source entity.

    Supported occurrences are native line and circle carriers with exact
    endpoint parameters or source-bound native period/conic witnesses. Faces
    must independently satisfy the explicit completion proof; unsupported or
    ambiguous interpretations refuse rather than assigning a source region.
    """
    from ._source_repair_face_completion import certify_face_completions

    if not isinstance(source, BRepModel) or not isinstance(target, BRepModel):
        raise TypeError("Source completion requires two BRepModel values.")
    if not isinstance(tolerance, Fraction) or tolerance <= 0:
        raise ValueError("Source completion tolerance must be a positive exact Fraction.")
    _require_same_incidence(source, target)
    first, second = source.geometry, target.geometry
    vertex_bounds = tuple(
        vertex_displacement_bound(
            _vertex_point(source, index), _vertex_point(target, index)
        )
        for index in range(len(first.vertex_roots))
    )
    edges = []
    edge_bounds = []
    for edge, (source_index, target_index) in enumerate(
        zip(first.edge_curves, second.edge_curves, strict=True)
    ):
        if source_index < 0 or target_index < 0:
            raise ValueError(
                "Degenerate edges are not supported by this completion proof."
            )
        source_curve, target_curve = (
            first.curves[source_index],
            second.curves[target_index],
        )
        source_parameters = (
            _endpoint_interval(source, edge, 0),
            _endpoint_interval(source, edge, 1),
        )
        target_parameters = (
            _endpoint_interval(target, edge, 0, repaired=True),
            _endpoint_interval(target, edge, 1, repaired=True),
        )
        if type(source_curve) is CircleCurve and type(target_curve) is CircleCurve:
            bounds = circle_coedge_completion_bounds(
                source_curve, target_curve, source_parameters, target_parameters
            )
            kind = "circle"
        elif type(source_curve) is LineCurve and type(target_curve) is LineCurve:
            bounds = line_coedge_completion_bounds(
                source_curve, target_curve, source_parameters, target_parameters
            )
            kind = "line"
        else:
            raise ValueError(
                "The source and repaired edge carriers lack a supported correspondence."
            )
        edge_bounds.append(bounds)
        edges.append(
            {
                "edge": edge,
                "vertices": list(first.edge_vertices[edge]),
                "carrier_kind": kind,
                "source_parameters": [
                    [_fraction_payload(v) for v in interval]
                    for interval in source_parameters
                ],
                "target_parameters": [
                    [_fraction_payload(v) for v in interval]
                    for interval in target_parameters
                ],
                "occurrence_bound": _fraction_payload(bounds.occurrence),
                "endpoint_bounds": [
                    _fraction_payload(value) for value in bounds.connectors
                ],
            }
        )
    faces = certify_face_completions(source, target, tolerance=tolerance)
    if len(faces) != len(source.patches):
        raise ValueError("Face completion evidence must cover every authored face.")
    max_vertex = max(vertex_bounds, default=Fraction(0))
    max_edge = max((bound.occurrence for bound in edge_bounds), default=Fraction(0))
    max_connector = max(
        max((face.connector_bound for face in faces), default=Fraction(0)),
        max(
            (value for bound in edge_bounds for value in bound.connectors),
            default=Fraction(0),
        ),
    )
    source_to_target = max((face.source_to_target for face in faces), default=Fraction(0))
    target_to_source = max((face.target_to_source for face in faces), default=Fraction(0))
    if (
        max(max_vertex, max_edge, max_connector, source_to_target, target_to_source)
        > tolerance
    ):
        raise ValueError("Source completion displacement exceeds the declared tolerance.")
    return SourceCompletionCertificate(
        max_vertex,
        max_edge,
        max_connector,
        source_to_target,
        target_to_source,
        {
            "interpretation": "explicit-bounded-trim-completion",
            "original_source_claim": "authored-incidence-and-carriers-only",
            "source_revision": source.source_revision,
            "target_revision": target.source_revision,
            "tolerance": _fraction_payload(tolerance),
            "vertices": [
                {"vertex": index, "displacement_bound": _fraction_payload(bound)}
                for index, bound in enumerate(vertex_bounds)
            ],
            "edges": edges,
            "faces": [face.payload for face in faces],
        },
    )

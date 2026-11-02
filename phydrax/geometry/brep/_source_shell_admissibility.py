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

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ._boolean_coincidence import prove_surface_correspondence
from ._intersection_curve import CurveTrimSegment
from ._model import BRepModel
from ._source_embedding import certify_source_embedding


_IntervalBox = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


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
            model.source_revision,
            (first, second),
            (first_box, second_box),
            correspondence.certificate_id,
            embedding.source_id,
            canonical_fingerprint(payload),
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError, NotImplementedError):
        return None

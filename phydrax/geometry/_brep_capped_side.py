#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Symbolic native-angle charts on cylindrical B-Rep sides of circular caps."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np

from ._brep_circle_rim_proof import CircleRimEdge
from ._meshing_domain import PatchCurveUse
from ._surface_affine_cell_proof import (
    certify_periodic_affine_surface_cell,
    PeriodicTriangle,
)
from ._surface_source_support import (
    _cad_root_edges,
    SurfaceSourceCharts,
    validate_cad_curve_ancestry,
)
from .brep._boolean_core import chart_add, chart_scale, ChartScalar
from .brep._correspondence import exact_source_lift
from .brep._intersection_curve import SurfaceRegion
from .brep._source_embedding import _constant_chart_line, _family, _same, _sheet_sign
from .brep._trim_helpers import PeriodOffset


_TAU = PeriodOffset(Fraction(0), Fraction(1))


def _diff(a: ChartScalar, b: ChartScalar, /) -> ChartScalar:
    return chart_add(a, chart_scale(b, Fraction(-1)))


def _within(value: ChartScalar, low: ChartScalar, high: ChartScalar, /) -> bool:
    return _sheet_sign(_diff(value, low)) in (0, 1) and _sheet_sign(
        _diff(value, high)
    ) in (-1, 0)


def _side_trim(
    charts: SurfaceSourceCharts, patch: int, rim_rows: set[int], /
) -> tuple[Fraction, Fraction, int, dict[int, Fraction]]:
    """Prove one full native revolution rectangle and its seam occurrence."""
    domain = charts.domain
    model = domain.brep_authority
    if model is None:
        raise ValueError("A cylindrical side needs current B-Rep authority.")
    surface = domain.patches[patch].surface
    if _family(surface) != "revolution-cylinder":
        raise ValueError(
            "The side does not have a proved cylindrical revolution carrier."
        )
    loops = domain.patches[patch].loops
    if (
        len(loops) != 1
        or len(loops[0]) != 4
        or any(not isinstance(use, PatchCurveUse) for use in loops[0])
    ):
        raise ValueError("A cylindrical side needs its complete four-coedge trim.")
    face = int(domain.source_indices[2][patch])
    geom = model.geometry
    if len(geom.face_loops[face]) != 1 or len(geom.face_loops[face][0]) != 4:
        raise ValueError("The cylindrical side changes its B-Rep face loop.")
    rings: dict[int, Fraction] = {}
    seams = []
    for i, use in enumerate(loops[0]):
        if not isinstance(use, PatchCurveUse):
            raise ValueError("A cylindrical side needs its complete four-coedge trim.")
        coedge = geom.face_loops[face][0][i]
        edge = geom.coedge_edges[coedge]
        if int(domain.source_indices[1][use.curve]) != edge:
            raise ValueError("The side trim changes its B-Rep coedge occurrence.")
        curve_index = geom.edge_curves[edge]
        if curve_index < 0:
            raise ValueError("The cylindrical side has a collapsed regular edge.")
        pcurve = geom.pcurves[coedge]
        lower, upper = (float(value) for value in np.asarray(geom.edge_ranges[edge]))
        if not exact_source_lift(
            geom.curves[curve_index],
            pcurve,
            SurfaceRegion(model.patches[face], model.parameter_bounds[face]),
            lower,
            upper,
        ):
            raise ValueError("The side coedge lacks its independent exact source lift.")
        if use.curve in rim_rows:
            line = _constant_chart_line(pcurve, 1)
            if (
                line is None
                or not isinstance(line[0], Fraction)
                or not _same(line[1], Fraction(0))
                or line[2] != 1
            ):
                raise ValueError("The rim does not use the side's native angular gauge.")
            if use.curve in rings:
                raise ValueError("The side repeats a rim coedge.")
            rings[use.curve] = line[0]
        else:
            line = _constant_chart_line(pcurve, 0)
            if line is None or not (_same(line[0], Fraction(0)) or _same(line[0], _TAU)):
                raise ValueError("The side seam is not an exact native deck occurrence.")
            a = chart_add(line[1], chart_scale(Fraction.from_float(lower), line[2]))
            b = chart_add(line[1], chart_scale(Fraction.from_float(upper), line[2]))
            seams.append((use.curve, line[0], a, b, geom.coedge_senses[coedge]))
    if len(rings) != 2 or len(set(rings.values())) != 2 or len(seams) != 2:
        raise ValueError("The side must be bounded by two rims and one paired seam.")
    lo, hi = sorted(rings.values())
    if (
        seams[0][0] != seams[1][0]
        or seams[0][4] != -seams[1][4]
        or not (
            (_same(seams[0][1], Fraction(0)) and _same(seams[1][1], _TAU))
            or (_same(seams[1][1], Fraction(0)) and _same(seams[0][1], _TAU))
        )
        or any(
            not (
                (_same(seam[2], lo) and _same(seam[3], hi))
                or (_same(seam[2], hi) and _same(seam[3], lo))
            )
            for seam in seams
        )
    ):
        raise ValueError(
            "The side's seam does not cover the same exact rim interval on both decks."
        )
    return lo, hi, seams[0][0], rings


def prepare_cylindrical_side_charts(
    charts: SurfaceSourceCharts, tolerance: float, rims: dict[int, CircleRimEdge], /
) -> dict[int, PeriodicTriangle]:
    """Prove trimmed, regular side cells with matching exact rim/seam traces.

    Returned rows are keyed by source root cell ID. Native turn expressions are
    derived from source endpoint roots, never from a rounded approximation of π.
    """
    charts.require_current()
    edge_vertices, adjacency = _cad_root_edges(charts)
    bindings = validate_cad_curve_ancestry(charts, tolerance)
    row_of = {cell: row for row, cell in enumerate(charts.cell_global_ids)}
    cell_vertices: dict[int, tuple[int, int, int]] = {}
    for block in charts.mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Cylindrical material charts require triangles.")
        for cell, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            cell_vertices[int(cell)] = (
                int(vertices[0]),
                int(vertices[1]),
                int(vertices[2]),
            )
    patch_rims: dict[int, set[int]] = {}
    phases: dict[tuple[int, int], ChartScalar] = {}
    for edge, rim in rims.items():
        patch = int(charts.root_patches[row_of[rim.side_cell]])
        patch_rims.setdefault(patch, set()).add(rim.curve_row)
        for vertex, phase in zip(edge_vertices[edge], (rim.first, rim.last), strict=True):
            key = (patch, vertex)
            previous = phases.get(key)
            if previous is not None and not (
                _same(previous, phase)
                or (_same(previous, Fraction(0)) and _same(phase, _TAU))
                or (_same(phase, Fraction(0)) and _same(previous, _TAU))
            ):
                raise ValueError(
                    "Incident rim edges disagree on the exact source vertex phase."
                )
            phases[key] = phase
    trim = {patch: _side_trim(charts, patch, rows) for patch, rows in patch_rims.items()}
    seam_vertices: dict[int, set[int]] = {patch: set() for patch in patch_rims}
    for edge, uses in bindings.items():
        for patch, (_, _, seam_row, _) in trim.items():
            if len(uses) == 2 and all(
                use.curve_row == seam_row
                and int(charts.root_patches[row_of[use.root_cell_id]]) == patch
                for use in uses
            ):
                seam_vertices[patch].update(edge_vertices[edge])
    if any(not values for values in seam_vertices.values()):
        raise ValueError("The side is missing its meshed native seam occurrence.")
    output: dict[int, PeriodicTriangle] = {}
    orientation: dict[int, int] = {}
    for cell, row in row_of.items():
        patch = int(charts.root_patches[row])
        if patch not in trim:
            continue
        low, high, _, _ = trim[patch]
        # This is the retained surface's own numeric shadow, consulted only
        # after the coedge proof above authorized the symbolic deck pairing.
        period_shadow = charts.domain.patches[patch].surface.periods[0]
        if period_shadow is None:
            raise ValueError("The side has no declared native period shadow.")
        turn_shadow = Fraction.from_float(float(period_shadow))
        points = []
        for vertex, (angle, height) in zip(
            cell_vertices[cell], charts.root_parameter_chart(row), strict=True
        ):
            phase = phases.get((patch, vertex), angle)
            if vertex in seam_vertices[patch]:
                if angle == 0:
                    if (patch, vertex) in phases and not (
                        _same(phase, Fraction(0)) or _same(phase, _TAU)
                    ):
                        raise ValueError("A rim vertex changes the seam's zero phase.")
                    phase = Fraction(0)
                elif angle == turn_shadow:
                    if (patch, vertex) in phases and not (
                        _same(phase, Fraction(0)) or _same(phase, _TAU)
                    ):
                        raise ValueError("A rim vertex changes the seam's turn phase.")
                    phase = _TAU
                else:
                    raise ValueError(
                        "A seam vertex has no exact declared deck occurrence."
                    )
            if not _within(phase, Fraction(0), _TAU) or not low <= height <= high:
                raise ValueError(
                    "A side material corner leaves its exact trimmed rectangle."
                )
            points.append((phase, height))
        triangle = cast(PeriodicTriangle, tuple(points))
        proof = certify_periodic_affine_surface_cell(
            charts.domain.patches[patch].surface, triangle
        )
        if proof is None:
            raise ValueError(
                "A side material cell has no strict whole-cell embedding proof."
            )
        if patch in orientation and orientation[patch] != proof.orientation:
            raise ValueError(
                "Cylindrical side cells have inconsistent exact orientation."
            )
        orientation[patch] = proof.orientation
        output[cell] = triangle
    for edge, uses in adjacency.items():
        side = [(cell, slot) for cell, slot, _, _ in uses if cell in output]
        if not side:
            continue
        if len(side) == 1:
            if edge not in rims:
                raise ValueError(
                    "A cylindrical side edge has no matching material neighbour."
                )
            rim = rims[edge]
            cell, _ = side[0]
            patch = int(charts.root_patches[row_of[cell]])
            vertices = cell_vertices[cell]
            endpoints = tuple(
                output[cell][vertices.index(vertex)] for vertex in edge_vertices[edge]
            )
            height = trim[patch][3][rim.curve_row]
            if any(point[1] != height for point in endpoints):
                raise ValueError("A side rim changes the exact coedge height.")
            shifts = tuple(
                _diff(point[0], phase)
                for point, phase in zip(endpoints, (rim.first, rim.last), strict=True)
            )
            if not _same(shifts[0], shifts[1]) or not any(
                _same(shifts[0], value)
                for value in (Fraction(0), _TAU, chart_scale(_TAU, Fraction(-1)))
            ):
                raise ValueError(
                    "A side rim changes its exact source-circle parameter trace."
                )
            continue
        if len(side) != 2 or edge in rims:
            raise ValueError("A cylindrical side edge changes its manifold incidence.")
        (a, sa), (b, sb) = side
        va, vb = cell_vertices[a], cell_vertices[b]
        canonical = edge_vertices[edge]
        pa = tuple(output[a][va.index(v)] for v in canonical)
        pb = tuple(output[b][vb.index(v)] for v in canonical)
        if pa[0][1] != pb[0][1] or pa[1][1] != pb[1][1]:
            raise ValueError("A side edge changes its transverse source trace.")
        da, db = _diff(pa[0][0], pb[0][0]), _diff(pa[1][0], pb[1][0])
        if not _same(da, db) or not (
            _same(da, Fraction(0))
            or _same(da, _TAU)
            or _same(da, chart_scale(_TAU, Fraction(-1)))
        ):
            raise ValueError("A side edge changes its pointwise native-angle trace.")
        if not _same(da, Fraction(0)) and not all(
            v in seam_vertices[int(charts.root_patches[row_of[a]])] for v in canonical
        ):
            raise ValueError("A deck trace is not bound to the proved side seam.")
    return output

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from phydrax.geometry import ConstrainedDelaunayTriangulation
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.meshing.providers._native_surface import (
    _accepted_source_splits,
    _chart_cdt_scale,
    _ChartMesh,
    _folded_chart_edge,
    _local_edge_identity,
    _source_curve_parameters,
    _SourceSplitProposal,
)
from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case


def test_repaired_cylinder_chart_uses_exact_dyadic_physical_scaling(
    tmp_path: Path,
) -> None:
    model, _, _, _ = _external_case("external-combined", "step", tmp_path)
    domain = MeshingDomain.from_brep(model)
    patch = int(np.flatnonzero(np.asarray(domain.source_indices[2]) == 0)[0])
    points = np.asarray(model.parameter_bounds[0], dtype=np.float64)
    scale = _chart_cdt_scale(domain, patch, points)
    assert scale == 1.0 / 64.0
    scaled = points * np.asarray((scale, 1.0), dtype=np.float64)
    restored = scaled / np.asarray((scale, 1.0), dtype=np.float64)
    np.testing.assert_array_equal(restored, points)


def test_cdt_source_ancestry_retains_exact_nonbinary_parameters() -> None:
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    segments = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    triangulation = ConstrainedDelaunayTriangulation(
        points, segments, min_angle=20.0, max_area=0.01
    )
    parameter_pairs = tuple((Fraction(0), Fraction(1, 3)) for _ in segments)
    initial = {
        (int(vertex), curve): parameter_pairs[curve][end]
        for curve, segment in enumerate(segments)
        for end, vertex in enumerate(segment)
    }
    derived = _source_curve_parameters(
        triangulation, initial, segments, tuple(range(len(segments))), parameter_pairs
    )
    witnessed = []
    for vertex in range(triangulation.input_point_count, len(triangulation.points)):
        segment = int(triangulation.constraint_sources[vertex])
        if segment < 0:
            continue
        first, last = (int(value) for value in triangulation.constraint_parents[vertex])
        ratio = Fraction.from_float(float(triangulation.constraint_ratios[vertex]))
        exact = derived[(first, segment)] + ratio * (
            derived[(last, segment)] - derived[(first, segment)]
        )
        assert derived[(vertex, segment)] == exact
        witnessed.append(exact)
    assert witnessed
    assert any(value != Fraction.from_float(float(value)) for value in witnessed)


def test_source_split_events_share_only_exact_parent_intervals() -> None:
    lower, upper = Fraction(0), Fraction(1, 16)
    first = Fraction(1, 32)
    competing = first - Fraction(1, 2**58)
    charts = [
        _ChartMesh(
            0,
            [],
            [],
            {},
            {},
            {},
            split_proposals=(
                _SourceSplitProposal(18, lower, upper, competing, (3, 0, 1), 8),
                _SourceSplitProposal(2, lower, upper, competing, (3, 0, 2), 9),
            ),
        ),
        _ChartMesh(
            1,
            [],
            [],
            {},
            {},
            {},
            split_proposals=(
                _SourceSplitProposal(18, lower, upper, first, (1, 0, 0), 11),
                _SourceSplitProposal(18, upper, 2 * upper, upper + first, (1, 0, 1), 12),
            ),
        ),
    ]
    accepted = _accepted_source_splits(charts)
    assert accepted == {18: {first, upper + first}, 2: {competing}}


def test_reversed_source_ancestry_preserves_authored_endpoints_and_descendants() -> None:
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    segments = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    triangulation = ConstrainedDelaunayTriangulation(
        points, segments, min_angle=20.0, max_area=0.01
    )
    pairs = tuple((Fraction(1, 16), Fraction(0)) for _ in segments)
    initial = {
        (int(vertex), curve): pairs[curve][end]
        for curve, segment in enumerate(segments)
        for end, vertex in enumerate(segment)
    }
    result = _source_curve_parameters(
        triangulation, initial, segments, tuple(range(len(segments))), pairs
    )
    assert all(result[key] == value for key, value in initial.items())
    descendants = 0
    for vertex in range(triangulation.input_point_count, len(triangulation.points)):
        curve = int(triangulation.constraint_sources[vertex])
        if curve < 0:
            continue
        first, last = (int(value) for value in triangulation.constraint_parents[vertex])
        ratio = Fraction.from_float(float(triangulation.constraint_ratios[vertex]))
        assert result[(vertex, curve)] == result[(first, curve)] + ratio * (
            result[(last, curve)] - result[(first, curve)]
        )
        descendants += 1
    assert descendants


def test_full_turn_polar_annulus_fold_splits_unwrapped_chart_edge() -> None:
    half_turn = Fraction.from_float(math.pi)
    points = [
        (2 * half_turn, Fraction(5, 32)),
        (half_turn, Fraction(1, 8)),
        (Fraction(0), Fraction(9, 64)),
    ]
    triangle = (0, 1, 2)
    chart = _ChartMesh(6, points, [triangle], {}, {}, {})
    edges = ((0, 1), (1, 2), (2, 0))
    identities = tuple(_local_edge_identity(chart, *edge) for edge in edges)
    chosen = _folded_chart_edge(chart, triangle, identities)
    assert chosen == identities[2]
    midpoint = tuple((points[0][axis] + points[2][axis]) / 2 for axis in range(2))
    assert midpoint[0] == half_turn
    assert points[0][0] - midpoint[0] == midpoint[0] - points[2][0] == half_turn
    assert points[0] != points[2]
    assert identities[0] != identities[1]

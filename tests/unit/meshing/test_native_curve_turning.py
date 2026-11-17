#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.meshing.providers._native_surface as native_surface
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveMeshingSpec,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import CurvatureSizeControl, SizeControlStrength
from phydrax.meshing.providers._native_curve import prepare_native_curve
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeCurveSource


# A strict rational lower bound for pi, independent of platform libm.
_PI_LOWER = Fraction(314159265358979323846264338327950288419716939937510, 10**50)


def _turn_report(
    last: tuple[float, float], /, *, limit: float, batch: bool = False
) -> tuple[float, tuple[str, ...], str]:
    vertices = [(0.0, 0.0), (1.0, 0.0), last]
    edges = [(0, 1), (1, 2)]
    entities = [101, 101]
    names = ["first", "second"]
    ranges = [(0.0, 1.0), (1.0, 2.0)]
    if batch:
        vertices.extend(((4.0, 0.0), (5.0, 0.0)))
        edges.append((3, 4))
        entities.append(202)
        names.append("independent")
        ranges.append((0.0, 1.0))
    segments = SegmentMesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(edges, dtype=np.int32),
        source_id="turning-test",
    )
    source = NativeCurveSource(
        segments,
        "revision",
        curve_entity_ids=np.asarray(entities, dtype=np.int64),
        vertex_entity_ids=np.arange(len(vertices), dtype=np.int64),
        chart_ids=tuple(names),
        chart_parameter_ranges=np.asarray(ranges, dtype=np.float64),
        chart_endpoint_ids=tuple(
            (f"vertex-{first}", f"vertex-{last}") for first, last in edges
        ),
    )
    first_id = int(source.chart_scope_entity_ids[0])

    def scope(ids: np.ndarray) -> MeshingScope:
        return MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            1,
            source.scope_entity_set_id(1),
            ids,
        )

    controlled = scope(np.asarray((first_id,), dtype=np.int64))
    whole = scope(np.unique(np.asarray(source.chart_scope_entity_ids, dtype=np.int64)))
    control = CurvatureSizeControl(
        controlled, limit, maximum_size=5.0, strength=SizeControlStrength.HARD
    )
    # The extra entity needs its own sizing control when present.
    controls = (control,)
    if batch:
        other = scope(
            np.asarray((int(source.chart_scope_entity_ids[-1]),), dtype=np.int64)
        )
        controls += (CurvatureSizeControl(other, 2.0, maximum_size=1.0),)
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        whole,
        size_controls=controls,
    )
    result = prepare_native_curve(
        source,
        specification,
        NativeMeshingOptions("curve_arc_length"),
        SpatialCoordinateContract.si(),
    ).execute()
    key = f"size:{control.control_id}:maximum_chord_turning"
    return (
        dict(result.compliance.achieved)[key],
        result.compliance.issues,
        control.control_id,
    )


@pytest.mark.parametrize(
    ("last", "exact_lower"),
    [((2.0, 0.0), Fraction(0)), ((1.0, 1.0), _PI_LOWER / 2), ((0.0, 0.0), _PI_LOWER)],
)
def test_chord_turn_report_encloses_exact_axis_angles(
    last: tuple[float, float], exact_lower: Fraction
) -> None:
    upper, _, _ = _turn_report(last, limit=2.0)

    def encloses(bound: float) -> bool:
        return Fraction(bound) >= exact_lower

    assert encloses(upper)
    assert upper < float(exact_lower) + 1e-6
    # The identical consumer check rejects a copied bound made inward of the witness.
    inward = math.nextafter(float(exact_lower), -math.inf)
    assert not encloses(inward)


def test_chord_turn_encloses_exact_three_five_cosine() -> None:
    upper, _, _ = _turn_report((4.0, 4.0), limit=2.0)

    def cosine_upper(angle: float) -> Fraction:
        x = Fraction(angle)
        return sum(
            ((-1) ** k * x ** (2 * k) / math.factorial(2 * k) for k in range(13)),
            Fraction(0),
        )

    # Alternating Taylor's even partial sum bounds cos(x) above on [0, pi].
    def encloses(bound: float) -> bool:
        return cosine_upper(bound) <= Fraction(3, 5)

    assert 0.0 < upper < 2.0
    assert encloses(upper)
    assert not encloses(0.5)


def test_hard_turn_limit_and_independent_curve_batch() -> None:
    single, issues, control_id = _turn_report((1.0, 1.0), limit=1.0)
    batched, batch_issues, batch_control_id = _turn_report(
        (1.0, 1.0), limit=1.0, batch=True
    )
    assert f"normal_angle:{control_id}" in issues
    assert f"normal_angle:{batch_control_id}" in batch_issues
    assert single == batched
    _, accepted_issues, accepted_id = _turn_report((1.0, 1.0), limit=2.0)
    assert f"normal_angle:{accepted_id}" not in accepted_issues


@pytest.mark.parametrize("chunk_size", (1, 2, 4, 10))
def test_chord_bounds_preserve_exact_width_and_outward_endpoints(
    chunk_size: int,
) -> None:
    program = prepare_interval_function(
        lambda x: jnp.stack((x[0], x[0] * 0, x[0] * 0)), 1
    )
    zero = np.zeros(3, dtype=np.float64)
    endpoint = np.zeros((2, 3), dtype=np.float64)
    intervals = (
        (Fraction(1, 3), Fraction(2, 3)),
        (Fraction(1, 7), Fraction(4, 7)),
        (Fraction(1, 5), Fraction(3, 5)),
        (Fraction(2, 9), Fraction(7, 9)),
    )
    jobs = [(program, (endpoint, endpoint), native, (zero, zero)) for native in intervals]
    reference = [
        native_surface._second_order_chord_bounds([job])[0] for job in jobs
    ]
    actual = [
        value
        for start in range(0, len(jobs), chunk_size)
        for value in native_surface._second_order_chord_bounds(
            jobs[start : start + chunk_size]
        )
    ]
    assert actual == reference
    for bound, (lower, upper) in zip(actual, intervals, strict=True):
        assert Fraction(bound) >= upper * (upper - lower) ** 2 / 8

    # An inward-rounded proposed bound must fail the same exact witness.
    first, last = intervals[0]
    witness = last * (last - first) ** 2 / 8
    inward = math.nextafter(float(witness), -math.inf)
    assert Fraction(inward) < witness


def test_curve_chord_results_match_separate_intervals() -> None:
    from phydrax.geometry._meshing_domain import SourcePreparation
    from tests.unit.meshing.test_native_surface_routes import _curved_trim_domain

    domain = _curved_trim_domain()
    intervals = tuple(
        (0, Fraction(index, 4), Fraction(index + 1, 4)) for index in range(4)
    )
    separate = native_surface._CurveChordBounds(SourcePreparation(domain))
    expected = tuple(separate.interval(*interval) for interval in intervals)
    together = native_surface._CurveChordBounds(SourcePreparation(domain))
    together.prepare_intervals(intervals)
    assert tuple(together.interval(*interval) for interval in intervals) == expected


def test_surface_mesh_is_deterministic_for_curved_trim() -> None:
    from tests.unit.meshing.test_native_surface_routes import (
        _curved_trim_domain,
        _surface_specification,
    )

    domain = _curved_trim_domain()
    specification = _surface_specification(domain, maximum_deviation=0.05)

    def mesh() -> native_surface.NativeMeshConstruction:
        return native_surface.prepare_native_surface(
            domain,
            specification,
            NativeMeshingOptions("parametric_surface"),
            SpatialCoordinateContract.si(),
        ).execute()

    expected = mesh()
    actual = mesh()
    np.testing.assert_array_equal(actual.mesh.coordinates, expected.mesh.coordinates)
    assert len(actual.mesh.blocks) == len(expected.mesh.blocks)
    for actual_block, expected_block in zip(
        actual.mesh.blocks, expected.mesh.blocks, strict=True
    ):
        np.testing.assert_array_equal(actual_block.vertices, expected_block.vertices)
        np.testing.assert_array_equal(actual_block.global_ids, expected_block.global_ids)
    assert actual.compliance.passed == expected.compliance.passed
    assert actual.compliance.achieved == expected.compliance.achieved
    assert actual.compliance.issues == expected.compliance.issues



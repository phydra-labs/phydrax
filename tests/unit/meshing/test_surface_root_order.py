#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.meshing.providers._native_surface import (
    _ChartMesh,
    _domain_curve_source_query,
    _split_chart_at_source_cells,
    _SurfaceRootCells,
    _surface_root_charts,
)


def _domain(source_id: str, /) -> MeshingDomain:
    points = ((0, 0), (1, 0), (1, 1), (0, 1))
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(MeshingDomainCurve(first, last) for first, last in pairs)
    uses = tuple(
        PatchCurveUse(
            index,
            LineCurve(
                points[first],
                tuple(points[last][axis] - points[first][axis] for axis in range(2)),
            ),
            0.0,
            1.0,
        )
        for index, (first, last) in enumerate(pairs)
    )
    patch = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), (uses,)
    )
    return MeshingDomain(
        (patch,), curves, 4, source_id=source_id, source_revision="revision"
    )


def _ordered_and_split(
    source_id: str, identifiers: tuple[int, int], /
) -> tuple[
    tuple[tuple[tuple[Fraction, Fraction], ...], ...],
    tuple[tuple[Fraction, Fraction], ...],
    tuple[tuple[int, int, int], ...],
]:
    zero, half, one = Fraction(0), Fraction(1, 2), Fraction(1)
    a = ((zero, zero), (one, zero), (one, one), (half, half))
    b = ((zero, zero), (half, half), (one, one), (zero, one))
    roots = _SurfaceRootCells(
        _domain(source_id),
        identifiers,
        ((0, 0), (0, 1)),
        (a, b),
        np.asarray((0, 0), dtype=np.int64),
        np.asarray((a, b), dtype=np.float64),
        None,
    )
    ordered = _surface_root_charts(roots)[0]
    chart = _ChartMesh(
        patch=0,
        points=[(zero, zero), (one, zero), (one, one), (zero, one)],
        triangles=[(0, 1, 2), (0, 2, 3)],
        corners={},
        curve_parameters={},
        boundary_edges={},
    )
    _split_chart_at_source_cells(chart, ordered)
    return tuple(root.corners for root in ordered), tuple(chart.points), tuple(chart.triangles)


def test_source_root_ties_do_not_depend_on_source_identity() -> None:
    # Both roots have lower_u=0, upper_u=1, lower_v=0; their IDs reverse order.
    first = _ordered_and_split("first-source", (29, 5))
    second = _ordered_and_split("second-source", (3, 41))
    assert first == second
    assert first[0][0][1] == (Fraction(1, 2), Fraction(1, 2))


def test_curve_query_rows_follow_domain_not_hashed_scope_ids() -> None:
    first = _domain("first-source")
    second = _domain("second-source")

    def scope_order(domain: MeshingDomain, /) -> tuple[int, ...]:
        rows = []
        for identifier in domain.scope_entity_ids(1):
            key, occurrence = domain.source_entity_key_for_scope(1, int(identifier))
            rows.append(domain.resolve_entity_id(1, key, occurrence_path=occurrence))
        return tuple(rows)

    assert scope_order(first) != scope_order(second)
    query_a = _domain_curve_source_query(first)
    query_b = _domain_curve_source_query(second)
    assert query_a.parameter_ranges == query_b.parameter_ranges
    assert query_a.endpoint_parameter_bounds == query_b.endpoint_parameter_bounds
    assert query_a.endpoint_point_bounds == query_b.endpoint_point_bounds
    values_a = np.stack(
        [np.asarray(curve.evaluate(jnp.asarray((0.0, 1.0)))) for curve in query_a.curves]
    )
    values_b = np.stack(
        [np.asarray(curve.evaluate(jnp.asarray((0.0, 1.0)))) for curve in query_b.curves]
    )
    assert np.array_equal(values_a, values_b)

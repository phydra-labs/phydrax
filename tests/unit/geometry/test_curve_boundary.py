from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._source_boundary_records import SourceBoundaryQuery
from phydrax.geometry.brep._patches import LineCurve


def test_curve_boundary_query_uses_continuous_boxes_and_preserves_occurrences() -> None:
    query = ParametricCurveBoundarySource(
        (
            LineCurve(jnp.asarray((0.0, 0.0)), jnp.asarray((1.0, 0.0))),
            LineCurve(jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))),
        ),
        ((0.0, 1.0), (0.0, 1.0)),
        source_id="curve-source",
        source_revision="revision-1",
        curve_entity_ids=("shared-edge", "shared-edge"),
        occurrence_paths=(("assembly:left",), ("assembly:right",)),
        chart_ids=("left-chart", "right-chart"),
    )

    assert isinstance(query, SourceBoundaryQuery)
    assert query.curve_entity_ids == ("shared-edge", "shared-edge")
    assert query.occurrence_paths == (("assembly:left",), ("assembly:right",))
    assert query.chart_ids == ("left-chart", "right-chart")

    distance = query.boundary_distance(np.asarray(((0.5, 0.5), (1.0, 0.5))))
    assert distance.semantics == "certified"
    assert distance.lower[0] <= 0.5 <= distance.upper[0]
    assert distance.lower[1] <= 0.0 <= distance.upper[1]

    complete = query.boundary_cover(0.5)
    assert complete.complete and complete.semantics == "certified"
    assert complete.points.shape[0] >= 2 and complete.points.shape[1] == 2
    assert np.all(complete.covering_radius <= 0.5)
    refined = query.boundary_cover(0.25)
    assert (
        refined.complete and refined.points.shape[0] >= 4 and refined.points.shape[1] == 2
    )
    assert np.all(refined.covering_radius <= 0.25)


def test_curve_query_uses_source_owned_continuous_enclosure() -> None:
    query = ParametricCurveBoundarySource(
        (
            LineCurve(
                jnp.asarray((0.0, 0.0)),
                jnp.asarray((1.0, 0.0)),
                origin_exact=(Fraction(0), Fraction(0)),
                direction_exact=(Fraction(1), Fraction(0)),
            ),
        ),
        ((0.0, 1.0),),
        source_id="bounded-curve-source",
        source_revision="revision-1",
        curve_entity_ids=("edge",),
        occurrence_paths=((),),
        chart_ids=("edge-chart",),
    )

    distance = query.boundary_distance(np.asarray(((0.5, 1.0),)))
    assert distance.lower[0] == 1.0
    assert distance.upper[0] == 1.0
    samples = query.boundary_cover(0.5)
    assert samples.complete
    assert samples.covering_radius[0] == 0.5

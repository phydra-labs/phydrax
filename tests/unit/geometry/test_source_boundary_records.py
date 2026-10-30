import numpy as np
import pytest

from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundarySamples,
)


def test_source_distance_and_sample_records_keep_semantics_and_coverage() -> None:
    distance = SourceBoundaryDistance((0.0, 1.0), (0.1, 1.1), "certified")
    samples = SourceBoundarySamples(
        ((0.0, 0.0), (1.0, 0.0)),
        (0.1, 0.1),
        (0.0, 0.0),
        "sampled",
        complete=False,
    )

    assert distance.semantics == "certified"
    assert np.all(distance.lower <= distance.upper)
    assert samples.semantics == "sampled"
    assert not samples.complete
    assert not samples.points.flags.writeable


def test_source_distance_and_cover_reject_malformed_bounds() -> None:
    with pytest.raises(ValueError, match="lower <= upper"):
        SourceBoundaryDistance((1.0,), (0.0,), "certified")
    with pytest.raises(ValueError, match="nonnegative"):
        SourceBoundarySamples(((0.0, 0.0),), (-1.0,), (0.0,), "certified", complete=True)


def test_chart_cover_binds_source_revision_and_restriction_witnesses() -> None:
    cover = SourceBoundaryChartCover(
        (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
        (0.01,),
        "certified",
        True,
        "authored-surface",
        "revision-7",
        restriction_ids=("patch-0-root-cell-4",),
    )

    assert cover.complete
    assert cover.source_id == "authored-surface"
    assert cover.source_revision == "revision-7"
    assert cover.restriction_ids == ("patch-0-root-cell-4",)
    assert not cover.simplices.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        SourceBoundaryChartCover(
            (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
            (0.01,),
            "certified",
            True,
            "authored-surface",
            "revision-7",
            restriction_ids=("same", "same"),
        )

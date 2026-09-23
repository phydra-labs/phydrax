#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx
import phydrax._limit_study as limit_study


def _data(coordinates: tuple[float, ...]):
    return tuple(
        phx.uq.ScientificLimitDatum(
            f"datum-{index}",
            {"x": coordinate},
            1.0 + coordinate,
            0.1,
        )
        for index, coordinate in enumerate(coordinates)
    )


def test_limit_study_rejects_unknown_prespecified_datum_ids():
    plan = phx.uq.ScientificLimitStudyPlan(
        (phx.uq.ScientificLimitAxis("x", 0.0),),
        (
            phx.uq.ScientificLimitVariation(
                "linear",
                {"x": 1},
                included_datum_ids=("datum-0", "missing", "datum-2"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown datum IDs"):
        phx.uq.run_scientific_limit_study(plan, _data((1.0, 2.0, 3.0)))


def test_limit_study_applies_minimum_span_in_transformed_coordinates():
    plan = phx.uq.ScientificLimitStudyPlan(
        (
            phx.uq.ScientificLimitAxis(
                "x",
                1.0,
                transform="inverse",
                minimum_span=0.5,
            ),
        ),
        (phx.uq.ScientificLimitVariation("linear", {"x": 1}),),
    )

    result = phx.uq.run_scientific_limit_study(plan, _data((2.0, 4.0, 8.0)))

    assert result.status == "abstained"
    assert result.fits[0].status == "abstained"
    assert result.fits[0].reason == "insufficient-axis-span"


def test_limit_study_abstains_when_covariance_solve_fails(monkeypatch):
    plan = phx.uq.ScientificLimitStudyPlan(
        (phx.uq.ScientificLimitAxis("x", 0.0),),
        (phx.uq.ScientificLimitVariation("linear", {"x": 1}),),
    )
    original = limit_study.np.linalg.lstsq
    call_count = 0

    def fail_second_solve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("synthetic covariance failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(limit_study.np.linalg, "lstsq", fail_second_solve)

    result = phx.uq.run_scientific_limit_study(plan, _data((1.0, 2.0, 3.0)))

    assert result.status == "abstained"
    assert result.fits[0].reason == "covariance-solve-failed"

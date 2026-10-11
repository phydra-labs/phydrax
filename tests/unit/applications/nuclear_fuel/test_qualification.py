#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Qualification-driver case registry and history-refinement acceptance."""

from __future__ import annotations

import pytest

from phydrax.qualification._nuclear_fuel_release_matrix import FUEL_VERIFICATION_CASES


def test_history_refinement_accepts_only_scale_resolved_or_roundoff_limited_rows() -> (
    None
):
    from tools.nuclear_fuel_qualification import _history_refinement_outcome

    temperatures = (
        648.5412621602176,
        648.5412621698455,
        648.5412621879212,
    )
    rows = [
        {
            "fgr": 0.001,
            "maximum_fuel_temperature_K": temperature,
            "pressure_Pa": 2.0e5,
            "mean_outer_diameter_m": 0.0095,
        }
        for temperature in temperatures
    ]
    outcome, evidence = _history_refinement_outcome(rows)
    temperature = evidence["maximum_fuel_temperature_K"]
    assert isinstance(temperature, dict)
    assert outcome == "passed"
    assert temperature["status"] == "roundoff"
    assert temperature["observed_order"] is None
    assert temperature["roundoff_floor"] > temperature["fine_difference"]


def test_history_refinement_refuses_resolved_noncontraction() -> None:
    from tools.nuclear_fuel_qualification import _history_refinement_outcome

    rows = [
        {
            "fgr": 0.001,
            "maximum_fuel_temperature_K": temperature,
            "pressure_Pa": 2.0e5,
            "mean_outer_diameter_m": 0.0095,
        }
        for temperature in (600.0, 601.0, 603.0)
    ]
    outcome, evidence = _history_refinement_outcome(rows)
    temperature = evidence["maximum_fuel_temperature_K"]
    assert isinstance(temperature, dict)
    assert outcome == "failed"
    assert temperature["status"] == "noncontracting"


def test_registry_closure_cases_match_release_matrix_rows() -> None:
    from tools.nuclear_fuel_qualification import QUALIFICATION_CASES

    # Aggregation and the closure matrix accept exactly the package rows, so
    # every row needs one fault-bearing case with its reference and metric.
    closure = [case for case in QUALIFICATION_CASES if case.fault is not None]
    assert [(case.case_id, case.reference, case.metric) for case in closure] == [
        (row.case_id, row.reference, row.metric) for row in FUEL_VERIFICATION_CASES
    ]
    identifiers = [case.case_id for case in QUALIFICATION_CASES]
    assert len(set(identifiers)) == len(identifiers)


@pytest.mark.parametrize("selection", [(), ("V1", "V1"), ("V16",)])
def test_case_selection_refuses_empty_repeated_or_unregistered(
    selection: tuple[str, ...],
) -> None:
    from tools.nuclear_fuel_qualification import select_cases

    with pytest.raises(ValueError, match="registered qualification cases"):
        select_cases(selection)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import Any

import pytest

from tools import (
    biomembrane_remesh_qualification,
    bubble_dynamics_qualification,
    bubbly_flow_qualification,
    color_gradient_lbm_qualification,
    foam_qualification,
    soap_film_tunnel_qualification,
    surface_thin_film_qualification,
    thin_film_optics_qualification,
    threshold_dynamics_qualification,
    threshold_surface_seeding_qualification,
    two_phase_hysing_qualification,
)


def _assert_failed_native_record_cannot_pass(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    registry_name: str,
    record: dict[str, Any],
) -> None:
    registry = {"injected": lambda: record}
    monkeypatch.setattr(module, registry_name, registry)
    report = module.run_qualification(("injected",))
    assert report["successful"] is False


def test_bubble_failed_solve_cannot_pass_on_numerical_closeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_failed_native_record_cannot_pass(
        monkeypatch,
        bubble_dynamics_qualification,
        "CAMPAIGNS",
        {
            "status": "MAX_STEPS",
            "relative_error": 0.0,
            "successful": False,
            "passed": True,
        },
    )


def test_threshold_incomplete_run_cannot_pass_on_numerical_closeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_failed_native_record_cannot_pass(
        monkeypatch,
        threshold_dynamics_qualification,
        "SCENARIOS",
        {
            "status": 0,
            "steps": 3,
            "committed_steps": 2,
            "relative_error": 0.0,
            "successful": False,
            "passed": True,
        },
    )


def test_foam_rolled_back_run_cannot_pass_on_numerical_closeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_failed_native_record_cannot_pass(
        monkeypatch,
        foam_qualification,
        "_SCENARIOS",
        {
            "committed": False,
            "relative_error": 0.0,
            "successful": False,
            "passed": True,
        },
    )


def test_thin_film_rejected_evaluation_cannot_pass_on_numerical_closeness() -> None:
    record = {
        "observer_fit": {"passed": True},
        "white_balance": {"white": {"passed": True}},
        "airy_versus_characteristic_matrix": {
            "film": {"successful": False, "passed": True}
        },
    }
    assert thin_film_optics_qualification._qualification_successful(record) is False


def test_threshold_uniform_coefficients_avoid_label_squared_storage() -> None:
    record = threshold_dynamics_qualification._uniform_dense_resource()
    assert record["successful"] is True
    assert record["kernel_coefficient_shape"] == [2]
    assert record["coefficient_bytes"] == 0
    working = record["working_bytes"]
    avoided = record["avoided_pairwise_materialization_bytes"]
    if not isinstance(working, int) or not isinstance(avoided, int):
        raise AssertionError("Threshold resource byte evidence must be integral.")
    assert working < avoided
    assert record["potentials_finite"] is True


def test_threshold_nonuniform_coefficients_refuse_before_decomposition() -> None:
    record = threshold_dynamics_qualification._nonuniform_resource_refusal()
    assert record["successful"] is True
    assert record["refused_before_decomposition"] is True
    expected = record["expected_coefficient_bytes"]
    maximum = record["maximum_working_bytes"]
    if not isinstance(expected, int) or not isinstance(maximum, int):
        raise AssertionError("Threshold resource byte evidence must be integral.")
    assert expected > maximum




def _exit_code(main: Callable[[], int | None]) -> int:
    try:
        result = main()
    except SystemExit as error:
        if not isinstance(error.code, int):
            raise AssertionError("Qualification CLI exit code must be an integer.")
        return error.code
    return 0 if result is None else result


@pytest.mark.parametrize("successful", [True, False], ids=("successful", "unsuccessful"))
@pytest.mark.parametrize(
    ("module", "report_owner"),
    (
        (bubble_dynamics_qualification, "run_qualification"),
        (threshold_dynamics_qualification, "run_qualification"),
        (thin_film_optics_qualification, "qualify"),
        (foam_qualification, "run_qualification"),
        (biomembrane_remesh_qualification, "run"),
        (soap_film_tunnel_qualification, "run"),
        (surface_thin_film_qualification, "run"),
        (threshold_surface_seeding_qualification, "run"),
        (bubbly_flow_qualification, "run_qualification"),
        (color_gradient_lbm_qualification, "run_qualification"),
    ),
    ids=(
        "bubble",
        "threshold",
        "thin-film-optics",
        "foam",
        "biomembrane-remesh",
        "soap-film-tunnel",
        "surface-thin-film",
        "threshold-surface-seeding",
        "bubbly-flow",
        "color-gradient-lbm",
    ),
)
def test_program_qualification_cli_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    report_owner: str,
    successful: bool,
) -> None:
    monkeypatch.setattr(module, report_owner, lambda *args, **kwargs: {"successful": successful})
    monkeypatch.setattr("sys.argv", ["qualification"])
    assert _exit_code(module.main) == (0 if successful else 1)


@pytest.mark.parametrize("successful", [True, False], ids=("successful", "unsuccessful"))
def test_hysing_qualification_cli_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    successful: bool,
) -> None:
    run = two_phase_hysing_qualification.HysingRun(
        case=1,
        cells_per_unit=1,
        time_refinement=1,
        step_size=0.1,
        steps_requested=1,
        steps_accepted=1,
        completed=True,
        status="COMPLETED",
        failure=None,
        samples=[],
        metrics={},
        residuals={},
        ledger={},
        wall_seconds=0.0,
        compile_seconds=0.0,
    )
    monkeypatch.setattr(two_phase_hysing_qualification, "run_case", lambda *args: run)
    monkeypatch.setattr(
        two_phase_hysing_qualification,
        "_assessment",
        lambda *args: {"passed": successful},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualification",
            "--case",
            "1",
            "--resolutions",
            "1",
            "--time-refinements",
            "1",
        ],
    )
    assert _exit_code(two_phase_hysing_qualification.main) == (
        0 if successful else 1
    )

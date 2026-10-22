"""Optional CUDA qualification refuses missing mathematics and failed evidence."""

from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools._cuda_optional_qualification import collect_profile_coverage, LibraryProfile


@pytest.fixture
def profile() -> LibraryProfile:
    return LibraryProfile(
        ("qualification.py",), ("forward", "reverse"), ("float32", "float64")
    )


def _report(path: Path, names: tuple[str, ...], *, terminal: str | None = None) -> None:
    suite = ElementTree.Element("testsuite")
    for name in names:
        case = ElementTree.SubElement(
            suite, "testcase", name=name, classname="numerical_contract"
        )
        if terminal is not None and name == "reverse[float64]":
            ElementTree.SubElement(case, terminal)
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")


def test_complete_mathematical_precision_matrix_qualifies(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        ("forward[float32]", "reverse[float32]", "forward[float64]", "reverse[float64]"),
    )
    result = collect_profile_coverage(report, profile)
    assert result.passed
    assert result.missing == result.unsuccessful == result.duplicates == ()
    assert result.report_error is None


def test_missing_precision_refuses_an_otherwise_passing_report(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(report, ("forward[float32]", "reverse[float32]", "forward[float64]"))
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.missing == (("reverse", "float64"),)
    assert result.unsuccessful == ()


def test_missing_scenario_is_not_replaced_by_another_precision_case(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(report, ("forward[float32]", "forward[float64]", "unrelated[float64]"))
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.missing == (
        ("reverse", None),
        ("reverse", "float32"),
        ("reverse", "float64"),
    )


@pytest.mark.parametrize("terminal", ["skipped", "failure", "error"])
def test_required_scenario_must_actually_pass(
    tmp_path: Path, profile: LibraryProfile, terminal: str
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        ("forward[float32]", "reverse[float32]", "forward[float64]", "reverse[float64]"),
        terminal=terminal,
    )
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.unsuccessful == ("numerical_contract::reverse[float64]",)
    assert result.missing == ()


def test_duplicate_result_cannot_inflate_completed_coverage(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        (
            "forward[float32]",
            "reverse[float32]",
            "forward[float64]",
            "reverse[float64]",
            "reverse[float64]",
        ),
    )
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.duplicates == ("numerical_contract::reverse[float64]",)


@pytest.mark.parametrize("contents", [None, "not XML", "<testsuite/>"])
def test_absent_or_empty_report_never_qualifies(
    tmp_path: Path, profile: LibraryProfile, contents: str | None
) -> None:
    report = tmp_path / "cases.xml"
    if contents is not None:
        report.write_text(contents, encoding="utf-8")
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.cases == ()
    assert (result.report_error is not None) == (contents != "<testsuite/>")

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Same-compiled-model qualification tiers and deliberately invalid requests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from phydrax import config as pxconfig
from phydrax.applications.robotics._mujoco_model import (
    adapt_mujoco_model,
    MujocoContactExclusion,
    MujocoModelEvidence,
)
from tools.qualify_native_mujoco import DEFAULT_PROFILE, matched_stepping_refusal, qualify


mujoco = pytest.importorskip("mujoco")

_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "robotics" / "native_mujoco"
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return qualify(DEFAULT_PROFILE, repeats=1)


@pytest.fixture
def _float64() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _fixture(report: dict[str, Any], name: str, /) -> dict[str, Any]:
    return next(item for item in report["fixtures"] if item["fixture"] == name)


def test_smooth_tiers_pass_against_the_same_compiled_model(
    report: dict[str, Any],
) -> None:
    smooth = _fixture(report, "arm_smooth")
    assert smooth["source_sha256_matches_profile"]
    for tier in (
        "smooth-model-mapping",
        "smooth-actuation",
        "matched-stepping",
        "fixed-branch-ad",
    ):
        record = smooth["tiers"][tier]
        assert record["status"] == "passed", (tier, record)
    assert smooth["identity"]["source_id"].startswith("mujoco-mjb-sha256:")
    assert smooth["law"]["integrator"] == "euler"


def test_contact_tier_executes_native_plant_invariants_not_trajectories(
    report: dict[str, Any],
) -> None:
    contact = _fixture(report, "arm_contact")
    tier = contact["tiers"]["different-native-contact-physics"]
    assert tier["status"] == "passed", tier
    assert tier["geometry_prerequisite"]["status"] == "passed"
    assert tier["native_step_invariants"]["status"] == "passed"
    quantities = {
        item["quantity"] for item in tier["native_step_invariants"]["comparisons"]
    }
    assert {"contact-established", "native-rest-on-contact"} <= quantities
    assert tier["search_backend"] == "lbvh"
    assert tier["reference_search_backend"] == "exhaustive"
    assert "contact forces" in tier["not_compared"]
    assert tier["law"]["native"] != tier["law"]["provider"]
    assert "matched-stepping" not in contact["tiers"]
    assert contact["tiers"]["smooth-model-mapping"]["status"] == "passed"


@pytest.mark.parametrize(
    ("fixture", "integrator"),
    [("arm_contact", None), ("arm_limited", None), ("arm_smooth", "RK4")],
    ids=["different-contact-law", "different-limit-law", "different-integrator"],
)
def test_invalid_parity_requests_are_refused_not_passed(
    report: dict[str, Any], fixture: str, integrator: str | None
) -> None:
    request = next(
        item
        for item in report["invalid_requests"]
        if item["fixture"] == fixture and item["integrator_override"] == integrator
    )
    assert request["status"] == "refused", request


def test_qualified_report_requires_observed_runtime_identity(
    report: dict[str, Any],
) -> None:
    assert report["status"] == "qualified-evaluated-tiers", report["failed"]
    assert report["runtime_identity_verified"] is True


def test_mjx_tier_is_passed_or_explicitly_unavailable(report: dict[str, Any]) -> None:
    for fixture in report["fixtures"]:
        tier = fixture["tiers"]["mjx-smooth-model-mapping"]
        assert tier["status"] in ("passed", "provider-unavailable"), tier
        if tier["status"] == "provider-unavailable":
            assert (
                f"{fixture['fixture']}:mjx-smooth-model-mapping"
                in report["not_passed_unevaluated"]
            )


@pytest.mark.usefixtures("_float64")
def test_excluded_contact_is_still_not_valid_for_matched_stepping() -> None:
    adaptation = adapt_mujoco_model(
        mujoco.MjModel.from_xml_path(str(_FIXTURES / "arm_contact.xml")),
        contact=MujocoContactExclusion("dropping contact does not make laws equal"),
    )
    evidence = adaptation.evidence
    assert isinstance(evidence, MujocoModelEvidence)
    assert matched_stepping_refusal(evidence) is not None

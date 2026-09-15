#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import pytest

from tools import black_hole_qualification as qualification


def _passing_profile(name):
    return qualification._profile_result(
        name,
        checks={"analytic_identity": True},
        flags={"finite": True},
        residuals={"identity_abs": 0.0},
        statuses={"status": 0},
        identities={"plan_id": f"qualification:{name}"},
    )


def test_requested_profiles_are_independent_and_fail_closed(monkeypatch):
    monkeypatch.setitem(
        qualification.PROFILE_RUNNERS,
        "geometry",
        lambda: _passing_profile("geometry"),
    )

    def fail_thermodynamics():
        raise RuntimeError("bounded scientific failure")

    monkeypatch.setitem(
        qualification.PROFILE_RUNNERS,
        "thermodynamics",
        fail_thermodynamics,
    )
    report = qualification.run_qualification(("geometry", "thermodynamics"), valid_at=10)

    assert report["requested_profiles"] == ["geometry", "thermodynamics"]
    assert report["failed_profiles"] == ["thermodynamics"]
    assert not report["passed"]
    assert report["profiles"]["geometry"]["passed"]
    manifest_id = report["manifest"]["manifest_id"]
    assert report["manifest"]["validity"]["evaluated_at_unix_seconds"] == 10
    assert report["profiles"]["geometry"]["manifest_id"] == manifest_id
    assert report["profiles"]["thermodynamics"]["manifest_id"] == manifest_id
    assert report["report_id"]
    manifest_payload = dict(report["manifest"])
    assert manifest_payload.pop("manifest_id") == qualification._content_id(
        "black-hole-runtime-manifest", manifest_payload
    )
    report_payload = dict(report)
    assert report_payload.pop("report_id") == qualification._content_id(
        "black-hole-qualification-report", report_payload
    )
    failure = report["profiles"]["thermodynamics"]
    assert not failure["passed"]
    assert failure["failure"] == {
        "type": "RuntimeError",
        "message": "bounded scientific failure",
    }
    assert json.loads(qualification.serialize_report(report)) == report


def test_profile_selection_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="Unknown"):
        qualification.run_qualification(("not-a-profile",), valid_at=10)
    with pytest.raises(ValueError, match="unique"):
        qualification.run_qualification(("geometry", "geometry"), valid_at=10)
    with pytest.raises(ValueError, match="nonnegative Unix timestamp"):
        qualification.run_qualification(("geometry",), valid_at=-1)


def test_report_serialization_rejects_nonfinite_json_numbers():
    with pytest.raises(ValueError, match="Out of range float values"):
        qualification.serialize_report({"residual": float("nan")})

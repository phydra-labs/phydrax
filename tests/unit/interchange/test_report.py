#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.util

import jax
import pytest

import phydrax.interchange as interchange
import phydrax.velocimetry.io as velocimetry_io
from phydrax.interchange import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)


def _transformation() -> AdapterLoss:
    return AdapterLoss(
        " intensity.dtype ",
        "import",
        "transformed",
        " normalized to a common floating-point dtype ",
        changes_interpretation=False,
    )


def _report(status: AdapterStatus, *, losses=()) -> AdapterReport:
    return AdapterReport(
        status,
        "source-format",
        "target-format",
        source_id="source-id",
        target_id="target-id",
        coordinate_mapping=("row -> y", "column -> x"),
        preserved_fields=("values", "validity"),
        assumptions=("coordinates are rectilinear",),
        losses=losses,
    )


def test_report_construction_preserves_transformations_as_static_metadata():
    transformation = _transformation()
    report = _report(AdapterStatus.DECLARED_LOSS, losses=(transformation,))

    assert report.valid
    assert transformation.path == "intensity.dtype"
    assert transformation.rationale == "normalized to a common floating-point dtype"
    assert report.losses == (transformation,)
    assert report.coordinate_mapping == ("row -> y", "column -> x")
    assert report.preserved_fields == ("values", "validity")

    leaves, structure = jax.tree_util.tree_flatten(report)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert leaves == []
    assert restored.status == AdapterStatus.DECLARED_LOSS
    assert restored.losses == (transformation,)


def test_report_construction_rejects_inconsistent_loss_accounting():
    transformation = _transformation()

    with pytest.raises(ValueError, match="lossless report"):
        _report(AdapterStatus.LOSSLESS, losses=(transformation,))
    with pytest.raises(ValueError, match="declared-loss report"):
        _report(AdapterStatus.DECLARED_LOSS)


def test_require_lossless_accepts_lossless_and_reports_declared_loss():
    require_lossless(_report(AdapterStatus.LOSSLESS))

    declared = _report(AdapterStatus.DECLARED_LOSS, losses=(_transformation(),))
    with pytest.raises(AdapterError, match="cannot represent") as error:
        require_lossless(declared)
    assert error.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC


def test_interchange_is_the_canonical_report_import_surface():
    expected = {
        "AdapterError",
        "AdapterLoss",
        "AdapterReport",
        "AdapterStatus",
        "require_lossless",
    }
    removed = expected | {"AdapterDirection", "AdapterLossCategory"}

    assert expected <= set(interchange.__all__)
    assert AdapterReport.__module__ == "phydrax.interchange._report"
    assert removed.isdisjoint(vars(velocimetry_io))
    assert importlib.util.find_spec("phydrax.velocimetry.io._report") is None

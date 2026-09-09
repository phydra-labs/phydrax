#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.finance.qualification import (
    archive_finance_result,
    finance_result_manifest,
    reopen_finance_result,
    valuation_support,
)


def test_finance_archive_preserves_named_physical_arrays_replay_and_support(tmp_path):
    arrays = {
        "present_value": np.asarray((10.5, 4.25)),
        "valid": np.asarray((True, True)),
    }
    units = {"present_value": "USD", "valid": "1"}
    manifest = finance_result_manifest("result-a", "run-a", arrays, units)
    support = valuation_support(
        "fourier",
        product="european-option",
        model="heston",
        pricing_law="usd-risk-neutral",
    )
    path = archive_finance_result(
        tmp_path / "result.phx",
        result_manifest=manifest,
        arrays=arrays,
        replay_id="replay-a",
        support_tuples=(support,),
        law_ids=("usd-risk-neutral",),
    )
    reopened = reopen_finance_result(path)

    assert reopened.result_manifest.manifest_id == manifest.manifest_id
    assert reopened.replay_id == "replay-a"
    assert reopened.law_ids == ("usd-risk-neutral",)
    assert reopened.support_tuples[0].support_tuple_id == support.support_tuple_id
    np.testing.assert_array_equal(
        reopened.arrays["present_value"],
        arrays["present_value"],
    )
    np.testing.assert_array_equal(reopened.arrays["valid"], arrays["valid"])


def test_result_manifest_requires_explicit_units_for_every_array():
    with pytest.raises(ValueError, match="cover every physical result array"):
        finance_result_manifest(
            "result-a",
            "run-a",
            {"present_value": np.asarray(1.0), "valid": np.asarray(True)},
            {"present_value": "USD"},
        )

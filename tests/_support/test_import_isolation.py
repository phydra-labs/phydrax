# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import pytest

from tests._support.import_isolation import run_without_imports


_SCENARIOS = "tests._support.import_isolation_scenarios"


def test_import_refusal_harness_admits_unrelated_imports() -> None:
    run_without_imports(_SCENARIOS, "permitted_import", ("xml",))


@pytest.mark.parametrize(
    "function", ("forbidden_root", "forbidden_descendant"), ids=("root", "descendant")
)
def test_import_refusal_harness_rejects_a_forbidden_import(function: str) -> None:
    with pytest.raises(AssertionError, match=r"isolated_scenario\.imports.*got xml"):
        run_without_imports(_SCENARIOS, function, ("xml",))

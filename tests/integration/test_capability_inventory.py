#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

from phydrax.qualification import builtin_capability_catalog
from tools.check_capability_consistency import capability_consistency_errors


def test_checked_capability_inventory_matches_canonical_catalog() -> None:
    root = Path(__file__).parents[2]

    errors = capability_consistency_errors(
        root,
        builtin_capability_catalog(),
        inventory=Path("docs/data/capabilities.json"),
        portfolios=Path("docs/data/application_portfolios.json"),
    )

    assert errors == ()

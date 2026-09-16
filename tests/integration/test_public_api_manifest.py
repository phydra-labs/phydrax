#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

from tools.check_public_api_manifest import public_api_manifest_errors


def test_checked_public_api_manifest_matches_exports() -> None:
    root = Path(__file__).parents[2]

    errors = public_api_manifest_errors(
        root,
        manifest=Path("docs/data/public_api.json"),
    )

    assert errors == ()

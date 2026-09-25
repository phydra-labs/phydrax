#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import types
from collections import deque
from pathlib import Path

import phydrax
from tools.check_public_api_manifest import public_api_manifest_errors
from tools.generate_public_api_manifest import public_api_record


def test_checked_public_api_manifest_matches_exports() -> None:
    root = Path(__file__).parents[2]

    errors = public_api_manifest_errors(
        root,
        manifest=Path("docs/data/public_api.json"),
    )

    assert errors == ()


def test_public_api_record_does_not_depend_on_import_history() -> None:
    before = public_api_record()

    # Resolve every export that loads without optional providers, as unrelated
    # code in the same process would.
    queue = deque((phydrax,))
    seen: set[int] = set()
    while queue:
        module = queue.popleft()
        if id(module) in seen:
            continue
        seen.add(id(module))
        for name in module.__all__:
            try:
                value = getattr(module, name)
            except ImportError:
                continue
            if isinstance(value, types.ModuleType) and value.__name__.startswith(
                "phydrax."
            ):
                queue.append(value)

    assert public_api_record() == before

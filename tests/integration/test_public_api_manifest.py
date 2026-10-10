#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import types
from collections import deque
from pathlib import Path

import phydrax
from phydrax.interchange import AdapterError, AdapterStatus
from tools.check_public_api_manifest import public_api_manifest_errors
from tools.generate_public_api_manifest import public_api_record


def test_public_api_manifest_scenario_1() -> None:
    root = Path(__file__).parents[2]

    errors = public_api_manifest_errors(
        root,
        manifest=Path("docs/data/public_api.json"),
    )

    assert errors == ()
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
            except AdapterError as error:
                if error.status is not AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE:
                    raise
                continue
            except ImportError:
                continue
            if isinstance(value, types.ModuleType) and value.__name__.startswith(
                "phydrax."
            ):
                queue.append(value)

    assert public_api_record() == before

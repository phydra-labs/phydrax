#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate the canonical manifest of explicitly exported public symbols."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import ModuleType

import phydrax
from phydrax._fingerprint import canonical_fingerprint


def public_api_record(*, maximum_depth: int = 4) -> dict[str, object]:
    queue: deque[tuple[ModuleType, int]] = deque(((phydrax, 0),))
    seen: set[str] = set()
    modules: dict[str, list[str]] = {}
    while queue:
        module, depth = queue.popleft()
        if module.__name__ in seen:
            continue
        seen.add(module.__name__)
        exported = getattr(module, "__all__", ())
        if not isinstance(exported, (tuple, list)):
            raise TypeError(f"{module.__name__}.__all__ must be an ordered sequence.")
        names = tuple(str(name) for name in exported)
        if len(set(names)) != len(names):
            raise ValueError(f"{module.__name__}.__all__ contains duplicates.")
        qualified = []
        for name in names:
            try:
                value = getattr(module, name)
            except ImportError:
                # Optional-provider facades deliberately raise until their extra is
                # installed. Their declared symbol remains part of the public API.
                qualified.append(f"{module.__name__}.{name}")
                continue
            except AttributeError as error:
                raise AttributeError(
                    f"{module.__name__}.{name} is exported but missing."
                ) from error
            qualified.append(f"{module.__name__}.{name}")
            if (
                depth < maximum_depth
                and isinstance(value, ModuleType)
                and value.__name__.startswith("phydrax.")
            ):
                queue.append((value, depth + 1))
        modules[module.__name__] = sorted(qualified)
    payload: dict[str, object] = {
        "kind": "public-api-manifest",
        "modules": dict(sorted(modules.items())),
    }
    return {**payload, "manifest_id": canonical_fingerprint(payload)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data/public_api.json"),
    )
    parser.add_argument("--maximum-depth", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.maximum_depth < 0:
        raise ValueError("maximum depth must be non-negative.")
    record = public_api_record(maximum_depth=arguments.maximum_depth)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

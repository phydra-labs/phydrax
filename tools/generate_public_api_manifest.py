#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate the canonical manifest of explicitly exported public symbols."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import deque
from pathlib import Path
from types import ModuleType

import phydrax
from phydrax._fingerprint import canonical_fingerprint


EXPLICIT_PUBLIC_MODULES = (
    "phydrax.applications.geophysics.deformation",
    "phydrax.applications.geophysics.electrical",
    "phydrax.applications.geophysics.electromagnetics",
    "phydrax.applications.geophysics.petrophysics",
    "phydrax.applications.geophysics.potential_fields",
    "phydrax.applications.geophysics.seismic",
    "phydrax.backends.lattice",
    "phydrax.chemistry.spectroscopy",
    "phydrax.discretization.discrete_velocity",
    "phydrax.discretization.dlr",
    "phydrax.discretization.spatial",
    "phydrax.interchange.cosmology",
    "phydrax.nn.quantum.variable_sector",
    "phydrax.nuclear.dosimetry",
    "phydrax.operators.integral",
    "phydrax.operators.quantum.variable_sector",
)


def _validate_public_path(path: str, /) -> None:
    parts = path.split(".")
    if (
        not parts
        or parts[0] != "phydrax"
        or any(part.startswith("_") for part in parts[1:])
    ):
        raise ValueError(f"Canonical public path {path!r} contains a private component.")


def public_api_record() -> dict[str, object]:
    queue: deque[tuple[str, ModuleType]] = deque((("phydrax", phydrax),))
    queue.extend(
        (path, importlib.import_module(path)) for path in EXPLICIT_PUBLIC_MODULES
    )
    seen_paths: set[str] = set()
    module_owners: dict[int, str] = {}
    modules: dict[str, list[str]] = {}
    while queue:
        public_path, module = queue.popleft()
        if public_path in seen_paths:
            continue
        _validate_public_path(public_path)
        previous_owner = module_owners.get(id(module))
        if previous_owner is not None:
            seen_paths.add(public_path)
            continue
        module_owners[id(module)] = public_path
        seen_paths.add(public_path)

        exported = module.__dict__.get("__all__")
        if not isinstance(exported, (tuple, list)):
            raise TypeError(f"{public_path}.__all__ must be an ordered sequence.")
        if any(not isinstance(name, str) for name in exported):
            raise TypeError(f"{public_path}.__all__ must contain only strings.")
        names = tuple(exported)
        if len(set(names)) != len(names):
            raise ValueError(f"{public_path}.__all__ contains duplicates.")

        qualified: list[str] = []
        for name in names:
            qualified_name = f"{public_path}.{name}"
            _validate_public_path(qualified_name)
            try:
                value = getattr(module, name)
            except ImportError:
                qualified.append(qualified_name)
                continue
            except AttributeError as error:
                raise AttributeError(
                    f"{qualified_name} is exported but missing."
                ) from error
            qualified.append(qualified_name)
            if isinstance(value, ModuleType) and value.__name__.startswith("phydrax."):
                queue.append((qualified_name, value))
        modules[public_path] = sorted(qualified)

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
    arguments = parser.parse_args()
    record = public_api_record()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

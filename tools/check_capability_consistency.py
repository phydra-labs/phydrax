#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail closed when checked capability inventory or public references drift."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from phydrax.qualification import (
    application_promotion_portfolios,
    builtin_capability_catalog,
    CapabilityCatalog,
)


def _resolve_symbol(symbol: str, /) -> object:
    parts = symbol.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            value: object = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            continue
        for name in parts[index:]:
            value = getattr(value, name)
        return value
    raise ModuleNotFoundError(symbol)


def capability_consistency_errors(
    root: Path,
    catalog: CapabilityCatalog,
    /,
    *,
    inventory: Path,
    portfolios: Path | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    inventory_path = root / inventory
    if not inventory_path.is_file():
        errors.append(f"missing-generated-inventory:{inventory.as_posix()}")
    else:
        checked = json.loads(inventory_path.read_text(encoding="utf-8"))
        if checked != catalog.to_record():
            errors.append("generated-inventory-drift")
    if portfolios is not None:
        portfolio_path = root / portfolios
        if not portfolio_path.is_file():
            errors.append(
                f"missing-generated-portfolios:{portfolios.as_posix()}"
            )
        else:
            checked_portfolios = json.loads(
                portfolio_path.read_text(encoding="utf-8")
            )
            expected_portfolios = {
                "kind": "application-promotion-portfolios",
                "catalog_id": catalog.catalog_id,
                "portfolios": [
                    value.to_record()
                    for value in application_promotion_portfolios(catalog)
                ],
            }
            if checked_portfolios != expected_portfolios:
                errors.append("generated-application-portfolio-drift")
    for declaration in catalog.declarations:
        for path in (*declaration.documentation, *declaration.examples):
            if not (root / path).is_file():
                errors.append(f"missing-path:{declaration.capability}:{path}")
        for symbol in declaration.public_symbols:
            try:
                _resolve_symbol(symbol)
            except (AttributeError, ModuleNotFoundError) as error:
                errors.append(
                    f"missing-symbol:{declaration.capability}:{symbol}:"
                    f"{type(error).__name__}"
                )
    return tuple(sorted(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/data/capabilities.json"),
    )
    parser.add_argument(
        "--portfolios",
        type=Path,
        default=Path("docs/data/application_portfolios.json"),
    )
    arguments = parser.parse_args()
    errors = capability_consistency_errors(
        arguments.root,
        builtin_capability_catalog(),
        inventory=arguments.inventory,
        portfolios=arguments.portfolios,
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

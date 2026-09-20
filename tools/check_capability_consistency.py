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
    builtin_omniphysics_closure_matrices,
    builtin_source_absorption_ledger,
    CapabilityCatalog,
    validate_closure_catalog,
    validate_source_coverage,
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
    closure: Path | None = None,
    sources: Path | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    matrices = builtin_omniphysics_closure_matrices(catalog)
    ledger = builtin_source_absorption_ledger()
    errors.extend(validate_closure_catalog(catalog, matrices))
    errors.extend(validate_source_coverage(ledger))
    qualification_path = root / "docs/data/omniphysics_qualification.json"
    retained_evidence_ids: set[str] = set()
    retained_provider_ids: set[str] = set()
    if not qualification_path.is_file():
        errors.append("missing-omniphysics-qualification-evidence")
    else:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        for collection in ("controls", "refinements", "applications", "providers"):
            for record in qualification.get(collection, ()):
                evidence_id = record.get("evidence_id")
                if isinstance(evidence_id, str):
                    retained_evidence_ids.add(evidence_id)
                provider_id = record.get("provider_id")
                if isinstance(provider_id, str):
                    retained_provider_ids.add(provider_id)
    source_ids = {source.source_id for source in ledger.sources}
    for matrix in matrices:
        for requirement in matrix.requirements:
            for path in (
                *requirement.required_benchmarks,
                *requirement.required_documents,
            ):
                if not (root / path).is_file():
                    errors.append(f"missing-closure-path:{matrix.family}:{path}")
            for symbol in requirement.required_public_symbols:
                try:
                    _resolve_symbol(symbol)
                except (AttributeError, ModuleNotFoundError) as error:
                    errors.append(
                        f"missing-closure-symbol:{matrix.family}:{symbol}:{type(error).__name__}"
                    )
            for source_id in set(requirement.source_ids).difference(source_ids):
                errors.append(f"missing-closure-source:{matrix.family}:{source_id}")
        for resolution in matrix.resolutions:
            for evidence_id in set(resolution.evidence_ids).difference(
                retained_evidence_ids
            ):
                errors.append(f"missing-retained-evidence:{matrix.family}:{evidence_id}")
            for provider_id in set(resolution.provider_ids).difference(
                retained_provider_ids
            ):
                errors.append(f"missing-retained-provider:{matrix.family}:{provider_id}")
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
            errors.append(f"missing-generated-portfolios:{portfolios.as_posix()}")
        else:
            checked_portfolios = json.loads(portfolio_path.read_text(encoding="utf-8"))
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
    if closure is not None:
        closure_path = root / closure
        ledger = builtin_source_absorption_ledger()
        expected_closure = {
            "kind": "omniphysics-closure-matrices",
            "catalog_id": catalog.catalog_id,
            "source_ledger_id": ledger.ledger_id,
            "matrices": [value.to_record() for value in matrices],
        }
        if not closure_path.is_file():
            errors.append(f"missing-generated-closure:{closure.as_posix()}")
        elif json.loads(closure_path.read_text(encoding="utf-8")) != expected_closure:
            errors.append("generated-closure-drift")
    if sources is not None:
        source_path = root / sources
        expected_sources = builtin_source_absorption_ledger().to_record()
        if not source_path.is_file():
            errors.append(f"missing-generated-sources:{sources.as_posix()}")
        elif json.loads(source_path.read_text(encoding="utf-8")) != expected_sources:
            errors.append("generated-source-ledger-drift")
    for declaration in catalog.declarations:
        for path in (*declaration.documentation, *declaration.examples):
            if not (root / path).is_file():
                errors.append(f"missing-path:{declaration.capability}:{path}")
        for symbol in declaration.public_symbols:
            try:
                _resolve_symbol(symbol)
            except (AttributeError, ModuleNotFoundError) as error:
                errors.append(
                    f"missing-symbol:{declaration.capability}:{symbol}:{type(error).__name__}"
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
    parser.add_argument(
        "--closure",
        type=Path,
        default=Path("docs/data/capability_closure.json"),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("docs/data/source_absorption.json"),
    )
    arguments = parser.parse_args()
    errors = capability_consistency_errors(
        arguments.root,
        builtin_capability_catalog(),
        inventory=arguments.inventory,
        portfolios=arguments.portfolios,
        closure=arguments.closure,
        sources=arguments.sources,
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

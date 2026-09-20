#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate canonical capability inventory data and documentation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from phydrax.qualification import (
    application_promotion_portfolios,
    builtin_capability_catalog,
    builtin_omniphysics_closure_matrices,
    builtin_source_absorption_ledger,
)


def _markdown(record: dict[str, object], /) -> str:
    declarations = record["declarations"]
    counts = Counter(item["disposition"] for item in declarations)
    lines = [
        "# Capability inventory",
        "",
        "This page is generated from `phydrax.qualification` declarations. It is an",
        "inventory, not a release index. Only a trusted signed release index can",
        "authorize a released support tuple.",
        "",
        f"Catalog ID: `{record['catalog_id']}`",
        "",
        "## Dispositions",
        "",
        "| Disposition | Count |",
        "| --- | ---: |",
    ]
    for disposition in ("released", "candidate", "research", "internal", "retired"):
        lines.append(f"| {disposition} | {counts.get(disposition, 0)} |")
    lines.extend(
        [
            "",
            "## Capabilities",
            "",
            "| Capability | Owner | Disposition | Domain maturity | Profiles |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for declaration in declarations:
        lines.append(
            "| "
            f"`{declaration['capability']}` | "
            f"`{declaration['owner']}` | "
            f"{declaration['disposition']} | "
            f"{declaration['domain_maturity']} | "
            f"{len(declaration['profiles'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `internal`: intentionally unavailable as a public capability.",
            "- `research`: implemented for bounded research use without an exact release profile.",
            "- `candidate`: exact unreleased support profiles exist and may collect evidence.",
            "- `released`: every required evidence dimension and signed release profile passed.",
            "- `retired`: removed from the public surface.",
            "",
        ]
    )
    return "\n".join(lines)


def _closure_markdown(matrices, ledger) -> str:
    lines = [
        "# Omniphysics closure matrix",
        "",
        f"Source ledger ID: `{ledger.ledger_id}`",
        "",
        "| Family | Requirements | Resolutions | Classified | Implementation closed | Release closed |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for matrix in matrices:
        lines.append(
            f"| `{matrix.family}` | {len(matrix.requirements)} | "
            f"{len(matrix.resolutions)} | {str(matrix.classified).lower()} | "
            f"{str(matrix.implementation_closed).lower()} | "
            f"{str(matrix.release_closed).lower()} |"
        )
    lines.extend(("", "This generated matrix is an inventory, not a release claim.", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/data/capabilities.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/api/capabilities.md"),
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
    parser.add_argument(
        "--closure-markdown",
        type=Path,
        default=Path("docs/api/capability_closure.md"),
    )
    arguments = parser.parse_args()

    catalog = builtin_capability_catalog()
    record = catalog.to_record()
    portfolios = {
        "kind": "application-promotion-portfolios",
        "catalog_id": catalog.catalog_id,
        "portfolios": [
            value.to_record() for value in application_promotion_portfolios(catalog)
        ],
    }
    ledger = builtin_source_absorption_ledger()
    matrices = builtin_omniphysics_closure_matrices(catalog)
    closure = {
        "kind": "omniphysics-closure-matrices",
        "catalog_id": catalog.catalog_id,
        "source_ledger_id": ledger.ledger_id,
        "matrices": [value.to_record() for value in matrices],
    }
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.portfolios.parent.mkdir(parents=True, exist_ok=True)
    arguments.closure.parent.mkdir(parents=True, exist_ok=True)
    arguments.sources.parent.mkdir(parents=True, exist_ok=True)
    arguments.closure_markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.markdown.write_text(_markdown(record), encoding="utf-8")
    arguments.portfolios.write_text(
        json.dumps(portfolios, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.closure.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.sources.write_text(
        json.dumps(ledger.to_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arguments.closure_markdown.write_text(
        _closure_markdown(matrices, ledger), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

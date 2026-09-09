#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from phydrax.finance.qualification import (
    build_finance_qualification_matrix,
    evaluate_finance_campaign,
)
from phydrax.qualification import QualificationEvidence, SupportTuple


def _reject_nonfinite(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not supported.")


def _read_request(
    path: Path, /
) -> tuple[list[SupportTuple], list[QualificationEvidence], int]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(value, Mapping):
        raise TypeError("Finance qualification request must be a JSON object.")
    if set(value) != {"support_tuples", "evidence", "evaluated_at"}:
        raise ValueError("Finance qualification request fields are not canonical.")
    support_records = value["support_tuples"]
    evidence_records = value["evidence"]
    if not isinstance(support_records, Sequence) or isinstance(support_records, str):
        raise TypeError("support_tuples must be a JSON array.")
    if not isinstance(evidence_records, Sequence) or isinstance(evidence_records, str):
        raise TypeError("evidence must be a JSON array.")
    supports = []
    for record in support_records:
        if not isinstance(record, Mapping):
            raise TypeError("Every support tuple must be a JSON object.")
        supports.append(SupportTuple.from_record(record))
    evidence = []
    for record in evidence_records:
        if not isinstance(record, Mapping):
            raise TypeError("Every qualification evidence item must be a JSON object.")
        evidence.append(QualificationEvidence.from_record(record))
    evaluated_at = value["evaluated_at"]
    if isinstance(evaluated_at, bool) or not isinstance(evaluated_at, int):
        raise TypeError("evaluated_at must be an integer timestamp.")
    return supports, evidence, evaluated_at


def run(path: Path, /) -> dict[str, object]:
    """Evaluate one local, fully materialized finance qualification request."""
    supports, evidence, evaluated_at = _read_request(path)
    matrix = build_finance_qualification_matrix(supports)
    campaign = evaluate_finance_campaign(matrix, evidence, at_time=evaluated_at)
    return campaign.to_record()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reviewed finance qualification records without invoking "
            "external executables, feeds, or networks."
        )
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = run(arguments.request)
    payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    coverage = result["coverage"]
    if not isinstance(coverage, Mapping):
        raise RuntimeError("Qualification campaign coverage invariant was violated.")
    return 0 if coverage["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

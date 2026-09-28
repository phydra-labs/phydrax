#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from pathlib import Path

import pytest

from tools import audit_contract_candidates, audit_selectors


def _package(tmp_path: Path, source: str, /) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    package = root / "phydrax"
    package.mkdir(parents=True)
    (package / "module.py").write_text(source, encoding="utf-8")
    return root, package


def test_selector_audit_reads_pep_695_literal_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package = _package(
        tmp_path,
        """from typing import Literal

type Route = Literal[\"direct\", \"fast\"]
ROUTES = (\"direct\", \"fast\")

def accepts(route: str) -> bool:
    return route in ROUTES
""",
    )
    monkeypatch.setattr(audit_selectors, "ROOT", root)

    findings = audit_selectors.audit(package)

    assert len(findings) == 1
    assert findings[0].kind == "option-table"
    assert findings[0].alias == "Route"


def test_contract_audit_reads_pep_695_literal_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package = _package(
        tmp_path,
        """from typing import Literal

class StrictModule:
    pass

type Route = Literal[\"direct\", \"fast\"]

class Plan(StrictModule):
    route: Route
""",
    )
    monkeypatch.setattr(audit_contract_candidates, "ROOT", root)

    candidates = audit_contract_candidates.audit(package)

    assert len(candidates) == 1
    assert candidates[0].name == "Plan"
    assert candidates[0].selectors == 1

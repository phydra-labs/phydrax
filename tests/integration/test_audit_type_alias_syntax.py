#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import sys
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


def test_signature_audit_separates_redundant_guards_from_retained_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    package = root / "audited_signatures"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "module.py").write_text(
        """import enum

from phydrax.typing import checked


class Operand:
    pass


class Mode(enum.Enum):
    FAST = "fast"


class Owner:
    @checked
    def checked_method(self, operand: Operand, mode: Mode) -> None:
        if not isinstance(operand, Operand):
            raise TypeError("operand")
        if not isinstance(mode, Mode):
            raise TypeError("mode")

    def unchecked_method(self, operand: Operand) -> None:
        if not isinstance(operand, Operand):
            raise TypeError("operand")


def source_function(operand: Operand) -> None:
    if not isinstance(operand, Operand):
        raise TypeError("operand")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_contract_candidates, "ROOT", root)
    monkeypatch.syspath_prepend(str(root))

    guards = audit_contract_candidates.audit_signature_guards(package)
    for name in ("audited_signatures.module", "audited_signatures"):
        sys.modules.pop(name, None)

    assert [(guard.function, guard.parameter, guard.disposition) for guard in guards] == [
        ("Owner.checked_method", "operand", "redundant-at-checked-boundary"),
        ("Owner.checked_method", "mode", "retained-static-only-annotation"),
        ("Owner.unchecked_method", "operand", "retained-unchecked-method"),
        ("source_function", "operand", "retained-source-function"),
    ]

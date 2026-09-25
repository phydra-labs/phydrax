"""Pinned ty fixtures for consumer-visible static typing behavior.

Every case file must type-check cleanly except its deliberately invalid lines, each
of which carries `# ty: ignore[<rule>]`. With `unused-ignore-comment` promoted to an
error, a missing diagnostic fails exactly like an unexpected one, and
`assert_type` pins inferred types.
"""

import sys
from pathlib import Path

import pytest

from tools.check_typing import run_tool, verify_pinned


_ROOT = Path(__file__).resolve().parents[2]
_CASES = tuple(sorted((Path(__file__).parent / "cases").glob("*.py")))


@pytest.mark.typecheck
def test_pinned_ty_matches_every_fixture_expectation():
    verify_pinned(_ROOT, "ty")
    assert _CASES
    result = run_tool(
        _ROOT,
        "ty",
        (
            "check",
            "--output-format",
            "concise",
            "--error",
            "unused-ignore-comment",
            "--python",
            sys.prefix,
            *(str(case) for case in _CASES),
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr

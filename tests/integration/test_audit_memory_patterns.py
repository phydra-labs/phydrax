#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_memory_patterns as memory


@pytest.mark.parametrize(
    ("rule", "bad", "good"),
    [
        (
            "inverse-by-identity",
            "return np.linalg.solve(a, np.eye(n)) @ b",
            "return np.linalg.solve(a, b)",
        ),
        (
            "vmap-identity",
            "return jax.vmap(action)(jnp.eye(n)) @ b",
            "return jax.vmap(action)(batches)",
        ),
        (
            "dense-derivative",
            "return jax.jacfwd(f)(x) @ b",
            "return jax.jvp(f, (x,), (b,))[1]",
        ),
        (
            "pairwise-broadcast",
            "return jnp.sum(x[:, None] - x[None, :])",
            "return x[:, None] - y[None, :]",
        ),
        (
            "array-cache",
            "@lru_cache(maxsize=None)\ndef table(n):\n    return np.ones(n)",
            "@host_table_cache(max_entries=8, max_bytes=1024, weight=payload_bytes)\ndef table(n):\n    return np.ones(n)",
        ),
        (
            "payload-tobytes",
            "return hashlib.sha256(x.tobytes()).hexdigest()",
            "return hashlib.sha256(memoryview(x).cast('B')).hexdigest()",
        ),
        ("payload-read-bytes", "return path.read_bytes()", "return path.open('rb')"),
        ("whole-buffer", "return buffer.getvalue()", "return buffer.getbuffer()"),
        ("identity-tolist", "return x.tolist()", "return iter(x)"),
        (
            "scan-reduce",
            "_, values = lax.scan(step, carry, xs)\n    return jnp.sum(values)",
            "_, values = lax.scan(step, carry, xs)\n    return values",
        ),
        (
            "dense-zero-sparse",
            "a = np.zeros((n, m))\n    a[0, 0] = 1\n    return np.nonzero(a)",
            "a = np.zeros(n)\n    return np.nonzero(a)",
        ),
        (
            "quadratic-prefix",
            "for cursor in range(n):\n        active = sum(buckets[:cursor])",
            "active = 0\n    for bucket in buckets:\n        active += bucket",
        ),
    ],
)
def test_each_rule_distinguishes_a_compliant_variant(
    rule: str, bad: str, good: str
) -> None:
    imports = "import jax\nimport jax.numpy as jnp\nimport numpy as np\nfrom jax import lax\nfrom functools import lru_cache\n"

    def source(body: str) -> str:
        if body.startswith("@"):
            return imports + body
        return imports + "def identity_payload(x):\n    " + body + "\n"

    bad_hits = memory.scan_source(source(bad), "phydrax/runtime/_payload.py")
    good_hits = memory.scan_source(source(good), "phydrax/runtime/_payload.py")
    assert rule in {finding.rule for finding in bad_hits}
    assert rule not in {finding.rule for finding in good_hits}


@pytest.mark.parametrize("name", ["jacfwd", "jacrev", "hessian", "jacobian"])
@pytest.mark.parametrize(
    "consumer",
    ["a @ v", "jnp.trace(a)", "jnp.diagonal(a)", "a[0, 1]", "contract('ij,j->i', a, v)"],
)
def test_derivative_aliases_and_local_consumers(name: str, consumer: str) -> None:
    source = f"from jax import {name} as derivative\nimport jax.numpy as jnp\ndef f(x):\n    a = derivative(fun)(x)\n    return {consumer}\n"
    hits = memory.scan_source(source, "phydrax/runtime.py")
    assert [hit.rule for hit in hits] == ["dense-derivative"]
    assert hits[0].symbol == "f"


def test_dense_outputs_and_unrelated_scopes_are_not_misuse() -> None:
    source = """import jax
import jax.numpy as jnp
from jax import lax

def dense(x):
    a = jax.jacfwd(fun)(x)
    return a

def pairs(x):
    a = x[:, None] - x[None, :]
    return a

def history(xs):
    _, values = lax.scan(step, carry, xs)
    return values

def unrelated(values):
    return jnp.sum(values)
"""
    assert memory.scan_source(source, "phydrax/runtime.py") == []


def test_dictionary_and_count_only_caches_require_review() -> None:
    source = """from functools import cache as memoize, lru_cache
array_cache = {}
@memoize
def first(n):
    return np.ones(n)
@lru_cache(maxsize=4)
def second(n):
    return np.ones(n)
"""
    hits = memory.scan_source(source, "phydrax/runtime.py")
    assert [(hit.symbol, hit.rule) for hit in hits] == [
        ("array_cache", "array-cache"),
        ("first", "array-cache"),
        ("second", "array-cache"),
    ]
    typed = memory.scan_source(
        "import numpy as np\ntables: dict[int, np.ndarray] = {}\n", "phydrax/runtime.py"
    )
    assert [(hit.symbol, hit.rule) for hit in typed] == [("tables", "array-cache")]


def test_exact_allowlist_rejects_stale_duplicate_and_unsorted_entries(
    tmp_path: Path,
) -> None:
    findings = [memory.Finding("phydrax/a.py", 8, "f", "payload-tobytes", "array bytes")]
    header = "path\tsymbol\trule\trationale\n"
    entry = "phydrax/a.py\tf\tpayload-tobytes\tFixed scalar wire output.\n"
    allowlist = tmp_path / "allowlist.tsv"
    allowlist.write_text(header + entry, encoding="utf-8")
    assert memory.check_allowlist(findings, allowlist) == []
    shifted = [memory.Finding("phydrax/a.py", 80, "f", "payload-tobytes", "array bytes")]
    assert memory.check_allowlist(shifted, allowlist) == []
    assert memory.check_allowlist(findings + shifted, allowlist) == []
    allowlist.write_text(header + entry + entry, encoding="utf-8")
    assert any(
        "duplicate" in error for error in memory.check_allowlist(findings, allowlist)
    )
    allowlist.write_text(header + entry.replace("\tf\t", "\tmissing\t"), encoding="utf-8")
    errors = memory.check_allowlist(findings, allowlist)
    assert any("stale" in error for error in errors)
    assert any("unallowlisted" in error for error in errors)
    allowlist.write_text(header + entry.replace("a.py", "*.py"), encoding="utf-8")
    assert any(
        "invalid exact" in error for error in memory.check_allowlist(findings, allowlist)
    )
    allowlist.write_text(header + entry.replace("a.py", "z.py") + entry, encoding="utf-8")
    assert any(
        "unsorted" in error for error in memory.check_allowlist(findings, allowlist)
    )
    allowlist.write_text(
        header + entry.replace("Fixed scalar wire output.", '"Two\nlines"'),
        encoding="utf-8",
    )
    assert any(
        "invalid exact" in error for error in memory.check_allowlist(findings, allowlist)
    )
    allowlist.write_text("path\tsymbol\trule\n", encoding="utf-8")
    assert any("header" in error for error in memory.check_allowlist(findings, allowlist))


def test_cli_json_is_stable_and_unallowlisted_findings_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    (package / "runtime.py").write_text(
        "def f(x):\n    return x.tobytes()\n", encoding="utf-8"
    )
    allowlist = tmp_path / "allowlist.tsv"
    allowlist.write_text("path\tsymbol\trule\trationale\n", encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_memory_patterns",
            "--root",
            str(package),
            "--allowlist",
            str(allowlist),
            "--json",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        memory.main()
    assert failure.value.code == 1
    assert "unallowlisted" in capsys.readouterr().out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data == [
        {
            "file": "phydrax/runtime.py",
            "line": 2,
            "symbol": "f",
            "rule": "payload-tobytes",
            "detail": "x.tobytes",
        }
    ]


def test_repository_memory_candidates_have_exact_dispositions() -> None:
    findings = memory.audit(memory.PACKAGE)
    assert findings == sorted(findings)
    assert memory.check_allowlist(findings, memory.ALLOWLIST) == []

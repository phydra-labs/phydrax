#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import pytest

from phydrax import config as pxconfig


_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_REPO_ROOT = _DOCS_DIR.parent
_LONG_RUNNING_DOCS = frozenset(
    {
        Path("docs/cookbook/uncertainty_quantification.md"),
        Path("docs/cookbook/operator_learning.md"),
        Path("docs/guides_uncertainty.md"),
    }
)


def _extract_python_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    in_block = False
    cur: list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()

        if not in_block:
            if stripped == "```python {.executable}":
                in_block = True
                cur = []
            continue

        if stripped == "```":
            code = _dedent_python_block("\n".join(cur)).strip()
            if code:
                blocks.append(code + "\n")
            in_block = False
            cur = []
            continue

        cur.append(line)

    return blocks


def _expected_stdout(markdown: str) -> str:
    """Unmarked examples are silent; output fences specify printed lines."""
    outputs = re.findall(
        r"^```text \{\.output-expected\}\n(.*?)^```\s*$", markdown, re.M | re.S
    )
    return "".join(output for output in outputs)


def _assert_documented_stdout(stdout: str, markdown: str) -> None:
    assert stdout == _expected_stdout(markdown), (
        "Printed values differ from the documented output"
    )


def _dedent_python_block(code: str) -> str:
    lines = code.splitlines()
    indents: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        indents.append(len(line) - len(stripped))
    if not indents:
        return textwrap.dedent(code)
    trim = min(indents)
    if trim <= 0:
        return code
    return "\n".join(
        line[trim:] if len(line) >= trim else line.lstrip() for line in lines
    )


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        yield path


def _write_blocks_to_temp_script(md_path: Path, blocks: list[str], tmp_dir: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_]+", "_", md_path.as_posix()).strip("_")
    script_path = tmp_dir / f"docs_example__{safe_stem}.py"

    parts: list[str] = [f"# Generated from {md_path}\n"]
    for i, block in enumerate(blocks):
        parts.append(f"\n# --- {md_path.name} block {i} ---\n")
        parts.append(block)

    script_path.write_text("".join(parts), encoding="utf-8")
    return script_path


def _markdown_files_with_python_blocks() -> list[Path]:
    paths: list[Path] = []
    for md_path in _iter_markdown_files(_DOCS_DIR):
        blocks = _extract_python_blocks(md_path.read_text(encoding="utf-8"))
        if blocks:
            paths.append(md_path)
    return paths


@pytest.mark.parametrize(
    "md_path",
    _markdown_files_with_python_blocks(),
    ids=lambda p: Path(p).relative_to(_REPO_ROOT).as_posix(),
)
def test_docs_python_examples_run(tmp_path: Path, md_path: Path) -> None:
    blocks = _extract_python_blocks(md_path.read_text(encoding="utf-8"))
    script_path = _write_blocks_to_temp_script(md_path, blocks, tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env = pxconfig.environment(
        pxconfig.RuntimeConfig(
            platforms=("cpu",),
            launch_environment=pxconfig.LaunchEnvironmentConfig(preallocate=False),
        ),
        base=env,
        mode="defaults",
    )

    relative_path = md_path.relative_to(_REPO_ROOT)
    timeout_s = 60 * (20 if relative_path in _LONG_RUNNING_DOCS else 10)

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    assert completed.returncode == 0, (
        f"{relative_path}: example exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    _assert_documented_stdout(completed.stdout, md_path.read_text(encoding="utf-8"))


def test_expected_output_preserves_printed_values() -> None:
    markdown = "```text {.output-expected}\nvalue: 3\n```\n"
    assert _expected_stdout(markdown) == "value: 3\n"
    assert _expected_stdout("```python {.executable}\nassert 2 + 2 == 4\n```\n") == ""


def test_documented_output_refuses_mismatched_values() -> None:
    markdown = "```text {.output-expected}\nvalue: 3\n```\n"
    with pytest.raises(AssertionError, match="Printed values differ"):
        _assert_documented_stdout("value: 4\n", markdown)
    with pytest.raises(AssertionError, match="Printed values differ"):
        _assert_documented_stdout("unexpected output\n", "")
    _assert_documented_stdout("value: 3\n", markdown)


def test_example_markers_render_as_code_blocks() -> None:
    markdown = pytest.importorskip("markdown")
    pytest.importorskip("pymdownx.superfences")
    for fence in ("```python {.executable}", "```text {.output-expected}"):
        html = markdown.markdown(
            f"Text\n\n{fence}\nvalue = rows[0][:3]\n```\n\n## Next heading\n",
            extensions=["pymdownx.superfences", "attr_list"],
        )
        assert "<pre>" in html and "<h2>Next heading</h2>" in html, fence

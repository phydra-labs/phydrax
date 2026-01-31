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


_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_REPO_ROOT = _DOCS_DIR.parent


def _extract_python_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    in_block = False
    cur: list[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()

        if not in_block:
            if stripped.startswith("```python"):
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

    header = f"# Generated from {md_path}\n"
    parts: list[str] = [header]
    for i, block in enumerate(blocks):
        parts.append(f"\n# --- {md_path.name} block {i} ---\n")
        parts.append(_shrink_doc_iterations(block))

    script_path.write_text("".join(parts), encoding="utf-8")
    return script_path


def _shrink_doc_iterations(code: str) -> str:
    code = re.sub(r"num_iter\s*=\s*\d+", "num_iter=5", code)
    code = re.sub(r"num_points\s*=\s*\d+", "num_points=16", code)
    code = re.sub(r"width_size\s*=\s*\d+", "width_size=8", code)
    code = re.sub(r"depth\s*=\s*\d+", "depth=1", code)
    code = re.sub(r"latent_size\s*=\s*\d+", "latent_size=4", code)
    code = re.sub(r"memory\s*=\s*\d+", "memory=5", code)
    code = re.sub(
        r"boundary_weight_num_reference\s*=\s*\d+",
        "boundary_weight_num_reference=64",
        code,
    )
    code = re.sub(r"n_epochs\s*=\s*\d+", "n_epochs=2", code)
    return code


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
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    timeout_s = 180

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(_REPO_ROOT),
        env=env,
        check=True,
        timeout=timeout_s,
    )

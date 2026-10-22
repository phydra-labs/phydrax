"""Generate static facade exports from their literal deferred-export tables.

Each managed facade declares one sorted literal `_DEFERRED_EXPORTS` table: public
name -> owner module relative to the facade package, or `module:.<name>` for a
submodule exported at its own dotted path. `phydrax._deferred_exports` resolves the
table at runtime; this tool rewrites the checker imports and `__all__` from it
between marker comments, and `--check` reports drift without writing.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = "# BEGIN GENERATED FACADE EXPORTS"
END = "# END GENERATED FACADE EXPORTS"
SUBMODULE = "module:"
FACADES = (
    "applications",
    "backends",
    "discretization",
    "discretization/fem",
    "discretization/finite_difference",
    "discretization/finite_volume",
    "equations",
    "geometry",
    "geometry/brep",
    "geometry/surface",
    "graph",
    "imaging",
    "integration",
    "lifecycle",
    "meshing",
    "meshing/providers",
    "meshing/qualification",
    "operators",
    "solver",
    "tensor_network",
)


def render(owners: dict[str, str]) -> str:
    """Return sorted checker imports and public names."""
    lines = [START, "if TYPE_CHECKING:"]
    for name, owner in sorted(owners.items()):
        if owner.startswith(SUBMODULE):
            lines.append(f"    from . import {name} as {name}")
        else:
            lines.append(f"    from {owner} import {name} as {name}")
    if not owners:
        lines.append("    pass")
    lines.extend(["", "__all__ = ["])
    lines.extend(f'    "{name}",' for name in sorted(owners))
    lines.extend(["]", END])
    source = "\n".join(lines) + "\n"
    ruff = str(ROOT / ".venv/bin/ruff")
    sorted_imports = subprocess.run(
        [
            ruff,
            "check",
            "--select",
            "I001",
            "--fix",
            "--stdin-filename",
            str(ROOT / "phydrax/__init__.py"),
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    ).stdout
    return subprocess.run(
        [ruff, "format", "--stdin-filename", str(ROOT / "phydrax/__init__.py"), "-"],
        input=sorted_imports,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    ).stdout.rstrip("\n")


def deferred_table(path: Path) -> dict[str, str]:
    """Return the literal `_DEFERRED_EXPORTS` table of one facade."""
    tree = ast.parse(path.read_text())
    table = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_DEFERRED_EXPORTS"
        and node.value is not None
    )
    owners = ast.literal_eval(table)
    if list(owners) != sorted(owners):
        raise ValueError(
            f"facade.owners must be sorted; got unordered _DEFERRED_EXPORTS in {path.relative_to(ROOT)}"
        )
    # A submodule is public only at its own dotted path; re-exporting another
    # package's submodule would create an alias spelling.
    aliases = sorted(
        name
        for name, owner in owners.items()
        if owner.startswith(SUBMODULE) and owner != f"{SUBMODULE}.{name}"
    )
    if aliases:
        raise ValueError(
            f"facade.owners must exclude foreign submodules; got {', '.join(aliases)} in {path.relative_to(ROOT)}"
        )
    return owners


def main() -> int:
    """Rewrite generated regions, or report drift without modifying files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift: list[str] = []
    for facade in FACADES:
        path = ROOT / "phydrax" / facade / "__init__.py"
        source = path.read_text()
        if START not in source or END not in source:
            drift.append(str(path.relative_to(ROOT)))
            continue
        start = source.index(START)
        end = source.index(END, start) + len(END)
        generated = render(deferred_table(path))
        if source[start:end] != generated:
            drift.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(source[:start] + generated + source[end:])
    if args.check and drift:
        print("Facade export drift:\n" + "\n".join(drift))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

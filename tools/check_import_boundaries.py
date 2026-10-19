#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail closed when public materials cross private import boundaries."""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.generate_public_api_manifest import (
    EXPLICIT_PUBLIC_MODULES,
    public_api_record,
)


_API_DIRECTIVE = re.compile(r"^::: (phydrax(?:\.[A-Za-z0-9_]+)*)", re.MULTILINE)
_PRIVATE_IMPORT = re.compile(
    r"^(?:from|import)\s+(phydrax(?:\.[A-Za-z0-9_]+)*\._[A-Za-z0-9_.]*)",
    re.MULTILINE,
)
_JAX_CONFIGURATION_NAMES = (
    "config",
    "enable_x64",
    "default_matmul_precision",
    "numpy_dtype_promotion",
    "numpy_rank_promotion",
    "transfer_guard",
    "transfer_guard_host_to_device",
    "transfer_guard_device_to_device",
    "transfer_guard_device_to_host",
    "disable_jit",
    "default_prng_impl",
    "default_device",
)
# Every upstream configuration capability; `phydrax/_jax_config.py` alone uses them.
_BANNED_ORIGINS: Final = frozenset(
    {f"jax.{name}" for name in _JAX_CONFIGURATION_NAMES} | {"jax._src.config"}
)
# Direct spellings, for material that is not parseable Python.
_CONFIGURATION_BYPASS: Final = re.compile(
    r"\bjax\s*\.\s*(?:" + "|".join(_JAX_CONFIGURATION_NAMES) + r")\b"
    r"|\bjax\s*\.\s*_src\s*\.\s*config\b"
    r"|\bfrom\s+jax\s+import\s+(?:\(\s*)?"
    r"(?:[A-Za-z_]\w*(?:\s+as\s+[A-Za-z_]\w*)?\s*,\s*)*"
    r"(?:" + "|".join(_JAX_CONFIGURATION_NAMES) + r")\b"
    r"|\bfrom\s+jax\s*\.\s*_src\s+import\s+(?:\(\s*)?"
    r"(?:[A-Za-z_]\w*(?:\s+as\s+[A-Za-z_]\w*)?\s*,\s*)*config\b"
)
_PYTHON_FENCE: Final = re.compile(
    r"^[ \t]*```[ \t]*(?:python3?|py)\b[^\n]*\n(.*?)^[ \t]*```",
    re.MULTILINE | re.DOTALL,
)
_SCOPES: Final = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

# One name's binding in a scope: a JAX origin, a non-JAX value (`None`), or an
# alias expression resolved against the scope chain.
type _Binding = str | ast.expr | None


class _Scope:
    """One lexical scope; its bindings are collected on first lookup."""

    __slots__ = ("_bindings", "_node")

    def __init__(self, node: ast.AST, /) -> None:
        self._node = node
        self._bindings: Mapping[str, tuple[_Binding, ...]] | None = None

    def bindings(self) -> Mapping[str, tuple[_Binding, ...]]:
        if self._bindings is None:
            self._bindings = _scope_bindings(self._node)
        return self._bindings


@dataclass(frozen=True, slots=True)
class _Fragment:
    """Python material and the containing-file line of each of its lines."""

    text: str
    lines: tuple[int, ...]

    def line(self, relative: int, /) -> int:
        return self.lines[min(max(relative, 1), len(self.lines)) - 1]


def _jax_origin(module: str, /) -> str | None:
    return module if module == "jax" or module.startswith("jax.") else None


def _banned_prefix(origin: str, /) -> str | None:
    for banned in _BANNED_ORIGINS:
        if origin == banned or origin.startswith(f"{banned}."):
            return banned
    return None


def _import_bindings(
    node: ast.Import | ast.ImportFrom, /
) -> Iterator[tuple[str, str | None]]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname is not None:
                yield alias.asname, _jax_origin(alias.name)
            else:
                root = alias.name.partition(".")[0]
                yield root, _jax_origin(root)
        return
    module = _jax_origin(node.module or "") if node.level == 0 else None
    for alias in node.names:
        origin = None if module is None else f"{module}.{alias.name}"
        yield alias.asname or alias.name, origin


def _target_names(target: ast.expr, /) -> Iterator[str]:
    for node in ast.walk(target):
        if isinstance(node, ast.Name):
            yield node.id


def _direct_children(scope: ast.AST, /) -> Iterator[ast.AST]:
    """Nodes evaluated in `scope` itself, excluding the bodies of nested scopes."""
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, _SCOPES):
            pending.extend(ast.iter_child_nodes(node))


def _node_bindings(node: ast.AST, /) -> Iterator[tuple[str, _Binding]]:
    match node:
        case ast.Import() | ast.ImportFrom():
            yield from _import_bindings(node)
        case ast.Assign(targets=targets, value=value):
            simple = len(targets) == 1 and isinstance(targets[0], ast.Name)
            for target in targets:
                for name in _target_names(target):
                    yield name, value if simple else None
        case (
            ast.AnnAssign(target=ast.Name(id=name), value=value)
            | ast.NamedExpr(target=ast.Name(id=name), value=value)
        ):
            yield name, value
        case (
            ast.FunctionDef(name=name)
            | ast.AsyncFunctionDef(name=name)
            | ast.ClassDef(name=name)
        ):
            yield name, None
        case ast.arg(arg=name) | ast.ExceptHandler(name=str() as name):
            yield name, None
        case (
            ast.For(target=target)
            | ast.AsyncFor(target=target)
            | ast.comprehension(target=target)
            | ast.AugAssign(target=target)
        ):
            for name in _target_names(target):
                yield name, None
        case ast.withitem(optional_vars=ast.expr() as target):
            for name in _target_names(target):
                yield name, None
        case ast.Global(names=names) | ast.Nonlocal(names=names):
            for name in names:
                yield name, None


def _scope_bindings(scope: ast.AST, /) -> Mapping[str, tuple[_Binding, ...]]:
    """Every binding made directly in `scope`; any JAX binding of a name wins."""
    bindings: dict[str, list[_Binding]] = {}
    parameters: list[ast.AST] = []
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        parameters.extend(ast.walk(scope.args))
    for node in (*parameters, *_direct_children(scope)):
        if isinstance(node, _SCOPES) and node is not scope:
            # A nested scope binds only its own name here.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.setdefault(node.name, []).append(None)
            continue
        for name, binding in _node_bindings(node):
            bindings.setdefault(name, []).append(binding)
    return {name: tuple(values) for name, values in bindings.items()}


def _lookup(name: str, chain: Sequence[_Scope], active: frozenset[str], /) -> str | None:
    for depth, scope in enumerate(chain):
        bindings = scope.bindings()
        if name not in bindings:
            continue
        for binding in bindings[name]:
            origin = (
                _origin(binding, chain[depth:], active | {name})
                if isinstance(binding, ast.expr) and name not in active
                else binding
                if isinstance(binding, str)
                else None
            )
            if origin is not None:
                return origin
        return None
    # An unbound `jax` in a script or snippet is the upstream package.
    return "jax" if name == "jax" else None


def _origin(
    node: ast.expr, chain: Sequence[_Scope], active: frozenset[str] = frozenset(), /
) -> str | None:
    match node:
        case ast.Name(id=name):
            return _lookup(name, chain, active)
        case ast.Attribute(value=value, attr=attribute):
            base = _origin(value, chain, active)
            return None if base is None else f"{base}.{attribute}"
        case ast.Call(
            func=ast.Name(id="getattr"),
            args=[value, ast.Constant(value=str() as attribute), *_],
        ):
            base = _origin(value, chain, active)
            return None if base is None else f"{base}.{attribute}"
    return None


def _string_parts(node: ast.AST, /) -> list[ast.Constant | ast.JoinedStr] | None:
    """The literal parts of a string literal or a static `+` concatenation."""
    match node:
        case ast.Constant(value=str()) | ast.JoinedStr():
            return [node]
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            first, second = _string_parts(left), _string_parts(right)
            return None if first is None or second is None else [*first, *second]
    return None


def _string_fragment(
    parts: Sequence[ast.Constant | ast.JoinedStr], fragment: _Fragment, /
) -> _Fragment | None:
    """Embedded string material mentioning JAX, with containing-file lines."""
    texts: list[str] = []
    lines: list[int] = []
    for part in parts:
        if isinstance(part, ast.Constant):
            text = str(part.value)
        else:
            # Interpolations are runtime values; static text is analyzed alone.
            text = "".join(
                str(item.value) if isinstance(item, ast.Constant) else "_"
                for item in part.values
            )
        count = text.count("\n")
        # Physical newlines map line by line; escaped ones stay on the literal's line.
        physical = (part.end_lineno or part.lineno) - part.lineno == count
        start = fragment.line(part.lineno)
        part_lines = [start + (index if physical else 0) for index in range(count + 1)]
        # A part's first line continues the previous part's last line.
        lines.extend(part_lines[1:] if lines else part_lines)
        texts.append(text)
    text = "".join(texts)
    return _Fragment(text, tuple(lines)) if "jax" in text else None


def _accesses_capability(node: ast.AST, /) -> bool:
    """Whether `node` names a configuration capability as an attribute."""
    match node:
        case ast.Attribute(attr=attribute):
            return attribute in _JAX_CONFIGURATION_NAMES
        case ast.Call(
            func=ast.Name(id="getattr"),
            args=[_, ast.Constant(value=str() as attribute), *_],
        ):
            return attribute in _JAX_CONFIGURATION_NAMES
    return False


def _import_uses(node: ast.Import | ast.ImportFrom, /) -> Iterator[str]:
    for _, origin in _import_bindings(node):
        banned = None if origin is None else _banned_prefix(origin)
        if banned is not None:
            yield banned
    if isinstance(node, ast.Import):
        for alias in node.names:
            banned = _banned_prefix(alias.name)
            if banned is not None:
                yield banned


def _fragment_uses(tree: ast.Module, fragment: _Fragment, /) -> Iterator[tuple[int, str]]:
    pending: list[tuple[ast.AST, tuple[_Scope, ...]]] = [(tree, (_Scope(tree),))]
    while pending:
        node, chain = pending.pop()
        if isinstance(node, _SCOPES):
            chain = (_Scope(node), *chain)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for capability in _import_uses(node):
                yield fragment.line(node.lineno), capability
        elif isinstance(node, ast.expr) and _accesses_capability(node):
            origin = _origin(node, chain)
            if origin is not None and origin in _BANNED_ORIGINS:
                yield fragment.line(node.lineno), origin
        else:
            parts = _string_parts(node)
            if parts is not None:
                nested = _string_fragment(parts, fragment)
                if nested is not None:
                    yield from _material_uses(nested, embedded=True)
                continue
        pending.extend((child, chain) for child in ast.iter_child_nodes(node))


def _material_uses(
    fragment: _Fragment, /, *, embedded: bool = False
) -> Iterator[tuple[int, str]]:
    """Configuration capabilities used by one piece of Python material."""
    # Every capability access spells its capability name somewhere in the text.
    if not any(name in fragment.text for name in _JAX_CONFIGURATION_NAMES):
        return
    source = textwrap.dedent(fragment.text)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Partial snippets and prose are scanned for direct spellings instead.
        for match in _CONFIGURATION_BYPASS.finditer(fragment.text):
            line = fragment.text.count("\n", 0, match.start()) + 1
            yield fragment.line(line), " ".join(match.group().split())
        return
    if embedded and all(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, (ast.Name, ast.Attribute, ast.Constant))
        for statement in tree.body
    ):
        # A string that only names a capability is data, not a script.
        return
    yield from _fragment_uses(tree, fragment)


def _markdown_uses(text: str, /) -> Iterator[tuple[int, str]]:
    """Python fences are analyzed as code; every other line by direct spelling."""
    prose = list(text)
    for match in _PYTHON_FENCE.finditer(text):
        start = text.count("\n", 0, match.start(1)) + 1
        body = match.group(1)
        lines = tuple(range(start, start + body.count("\n") + 1))
        yield from _material_uses(_Fragment(body, lines))
        prose[match.start(1) : match.end(1)] = [
            character if character == "\n" else " " for character in body
        ]
    remainder = "".join(prose)
    for match in _CONFIGURATION_BYPASS.finditer(remainder):
        line = remainder.count("\n", 0, match.start()) + 1
        yield line, " ".join(match.group().split())


def _is_private_path(path: str, /) -> bool:
    return any(part.startswith("_") for part in path.split(".")[1:])


def _resolves_public_symbol(symbol: str, public_paths: frozenset[str], /) -> bool:
    if symbol in public_paths:
        return True
    if not any(symbol.startswith(public + ".") for public in public_paths):
        return False
    parts = symbol.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            value: object = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # ty: ignore[unsupported-operator]
            if error.name != module_name and not module_name.startswith(error.name + "."):
                raise
            continue
        except ImportError:
            return False
        for name in parts[index:]:
            try:
                value = getattr(value, name)
            except AttributeError:
                return False
        return True
    return False


def _example_private_imports(path: Path, /) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = (node.module,)
        else:
            continue
        for module in modules:
            if module.startswith("phydrax.") and _is_private_path(module):
                errors.append(
                    f"private-example-import:{path.as_posix()}:{node.lineno}:{module}"
                )
    return tuple(errors)


def configuration_bypass_errors(root: Path, /) -> tuple[str, ...]:
    """Audit Python sources, embedded scripts, and documentation fences.

    Imports, aliases, and lexical scopes are analyzed statically; analyzed
    material is parsed, never executed or imported.
    """
    paths = [
        path
        for directory in ("phydrax", "tests", "tools", "benchmarks", "examples")
        for path in (root / directory).rglob("*.py")
        if path.relative_to(root).as_posix() != "phydrax/_jax_config.py"
    ]
    readme = root / "README.md"
    if readme.is_file():
        paths.append(readme)
    paths.extend(
        path
        for path in (root / "docs").rglob("*.md")
        if path.relative_to(root).parts[:2] != ("docs", "plans")
    )
    errors: set[str] = set()
    for path in sorted(paths):
        text = path.read_text()
        if "jax" not in text:
            continue
        if path.suffix == ".md":
            uses = _markdown_uses(text)
        else:
            lines = tuple(range(1, text.count("\n") + 2))
            uses = _material_uses(_Fragment(text, lines))
        relative = path.relative_to(root).as_posix()
        errors.update(
            f"configuration-bypass:{relative}:{line}:{capability}"
            for line, capability in uses
        )
    return tuple(sorted(errors))


# Exceptions name one tool path, package owner, and symbol, with the benchmark
# measurement or workflow instrumentation contract requiring private access.
_MESHING_PRIVATE_IMPORT_EXCEPTIONS: dict[tuple[str, str, str], str] = {
    (
        "tools/meshing_qualification/kernel_performance.py",
        "phydrax",
        "_meshcore",
    ): (
        "The native batch benchmark measures private kernels and their execution "
        "scopes directly; its preparation, timing, and bitwise evidence comparison "
        "are tool-owned measurement, not a public scientific workflow."
    ),
    (
        "tools/meshing_qualification/_runtime.py",
        "phydrax._meshcore",
        "NativeParallelExecution",
    ): (
        "The workflow stage ledger lowers its remaining deadline while inheriting "
        "the active native parallelism; exposing this harness scope would add a "
        "second public thread vocabulary forbidden by plan D7."
    ),
    (
        "tools/meshing_qualification/_records.py",
        "phydrax._meshcore",
        "MeshcoreError",
    ): (
        "The harness reports bare native refusals with their status and optional "
        "elapsed/maximum seconds; public MeshingFailure carries different request "
        "evidence and cannot classify these exceptions without losing fields."
    ),
}


def _meshing_private_imports(
    root: Path,
    path: Path,
    allowed: frozenset[tuple[str, str, str]],
) -> tuple[str, ...]:
    """Refuse each unrecorded private module or private-symbol dependency."""
    relative = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports = tuple((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            imports = tuple((alias.name, "<module>") for alias in node.names)
        else:
            continue
        for module, symbol in imports:
            if (module == "phydrax" or module.startswith("phydrax.")) and (
                _is_private_path(module) or symbol.startswith("_")
            ):
                if (relative, module, symbol) not in allowed:
                    errors.append(
                        f"private-meshing-tool-import:{relative}:{node.lineno}:{module}:{symbol}"
                    )
    return tuple(errors)


def import_boundary_errors(root: Path, /) -> tuple[str, ...]:
    record = public_api_record()
    modules = record["modules"]
    if not isinstance(modules, dict):
        raise TypeError("Public API record modules must be a dictionary.")
    public_modules = frozenset(modules)
    public_paths = public_modules | frozenset(
        symbol
        for values in modules.values()
        for symbol in values
        if isinstance(symbol, str)
    )

    errors = list(configuration_bypass_errors(root))
    for module in EXPLICIT_PUBLIC_MODULES:
        if module not in public_modules:
            errors.append(f"missing-explicit-public-module:{module}")

    for path in (root / "examples").rglob("*.py"):
        errors.extend(_example_private_imports(path))

    allowed = frozenset(_MESHING_PRIVATE_IMPORT_EXCEPTIONS)
    meshing_tools = (
        root / "tools/_meshing_cases.py",
        root / "tools/meshing_benchmarks.py",
        root / "tools/meshing_design_source.py",
        root / "tools/check_native_dependency_isolation.py",
        root / "tools/layer_core_lifecycle_continuation.py",
        root / "tools/native_moving_overset.py",
        *(root / "tools/meshing_qualification").rglob("*.py"),
    )
    for path in meshing_tools:
        if path.exists():
            errors.extend(_meshing_private_imports(root, path, allowed))

    documentation_paths = (root / "README.md", *(root / "docs").rglob("*.md"))
    for path in documentation_paths:
        text = path.read_text()
        for match in _API_DIRECTIVE.finditer(text):
            symbol = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            if _is_private_path(symbol):
                errors.append(
                    f"private-api-directive:{path.relative_to(root)}:{line}:{symbol}"
                )
            elif not _resolves_public_symbol(symbol, public_paths):
                errors.append(
                    f"unknown-api-directive:{path.relative_to(root)}:{line}:{symbol}"
                )
        for match in _PRIVATE_IMPORT.finditer(text):
            module = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"private-documentation-import:{path.relative_to(root)}:{line}:{module}"
            )

    return tuple(sorted(set(errors)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    errors = import_boundary_errors(arguments.root)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report host conversions that may synchronize JAX values.

The report is read-only and never rewrites code. It lists every call of `bool`,
`int`, `float`, `numpy.asarray`/`numpy.array`, and `jax.device_get` in package
code and classifies each hit by its syntactic context:

- `static-shape`: the argument reads static metadata (`.shape`, `.ndim`, `.size`,
  `len(...)`, `.itemsize`, dtype information);
- `host-preparation`: the call runs in a constructor, a preparation or
  validation function, or at module level;
- `explicit-safe-point`: an explicit `jax.device_get`;
- `traced-body`: the call runs inside a function passed to a JAX transformation
  or loop (`jit`, `vmap`, `grad`, `scan`, `fori_loop`, `while_loop`, `cond`,
  `switch`, `filter_jit`, `filter_vmap`, `checkpointed_scan`); these are
  candidates for accidental tracer synchronization;
- `external-provider`: the call is in an interchange, backend, export, or
  provider-worker module;
- `output-observation`: the call is in a reporting, record, representation, or
  summary method;
- `unclassified`: none of the above; requires review.

A classification is evidence for review, not a verdict: a `traced-body` hit on a
static Python value is harmless, and an `unclassified` hit may be a deliberate
host boundary.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
_CONVERSIONS = frozenset(
    {"bool", "int", "float", "np.asarray", "np.array", "numpy.asarray", "numpy.array"}
)
_SAFE_POINTS = frozenset({"jax.device_get"})
_STATIC_ATTRIBUTES = frozenset(
    {"shape", "ndim", "size", "itemsize", "dtype", "nbytes", "kind"}
)
_TRANSFORMS = frozenset(
    {
        "jit",
        "vmap",
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "scan",
        "fori_loop",
        "while_loop",
        "cond",
        "switch",
        "filter_jit",
        "filter_vmap",
        "filter_grad",
        "checkpointed_scan",
        "custom_jvp",
        "custom_vjp",
    }
)
_PREPARATION_PREFIXES = (
    "__init__",
    "__post_init__",
    "__check_init__",
    "prepare",
    "_prepare",
    "validate",
    "_validate",
    "_require",
    "_check",
    "from_",
    "_canonical",
    "_coerce",
    "_as_",
)
_OBSERVATION_PREFIXES = (
    "report",
    "summary",
    "to_record",
    "to_dict",
    "record",
    "__repr__",
    "__str__",
    "describe",
    "_record",
    "_summary",
)
_PROVIDER_PARTS = frozenset({"interchange", "backends", "export", "io"})


@dataclass(frozen=True, order=True, slots=True)
class Hit:
    file: str
    line: int
    call: str
    category: str
    function: str


def _call_name(node: ast.Call, /) -> str:
    return (
        ast.unparse(node.func) if isinstance(node.func, ast.Name | ast.Attribute) else ""
    )


def _reads_static_metadata(node: ast.expr, /) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in _STATIC_ATTRIBUTES:
            return True
        if isinstance(child, ast.Call) and _call_name(child) in ("len", "math.prod"):
            return True
    return False


_Scope: TypeAlias = tuple[tuple[str, str], ...]


def _is_transform(name: str, /) -> bool:
    # `map` is a transformation only as `lax.map`; builtin and tree maps are not.
    return name.rsplit(".", 1)[-1] in _TRANSFORMS or name.endswith("lax.map")


def _qualified(scope: _Scope, /) -> tuple[str, ...]:
    return tuple(name for _, name in scope)


def _function_prefixes(scope: _Scope, /) -> list[tuple[str, ...]]:
    """Scopes in which a bare name may resolve: the module and each enclosing function.

    Class bodies are skipped, as in Python name resolution.
    """
    prefixes = [()]
    for index, (kind, _) in enumerate(scope):
        if kind == "function":
            prefixes.append(_qualified(scope[: index + 1]))
    return prefixes


def _traced_functions(tree: ast.Module, /) -> set[tuple[str, ...]]:
    """Qualified names of functions passed to, or decorated by, JAX transformations."""
    traced: set[tuple[str, ...]] = set()

    def visit(node: ast.AST, scope: _Scope) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = (*scope, ("function", child.name))
                for decorator in child.decorator_list:
                    target = (
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    if _is_transform(ast.unparse(target)):
                        traced.add(_qualified(inner))
                visit(child, inner)
                continue
            if isinstance(child, ast.ClassDef):
                visit(child, (*scope, ("class", child.name)))
                continue
            if isinstance(child, ast.Call):
                if _is_transform(_call_name(child)):
                    for argument in child.args[:1]:
                        if isinstance(argument, ast.Name):
                            traced.update(
                                (*prefix, argument.id)
                                for prefix in _function_prefixes(scope)
                            )
            visit(child, scope)

    visit(tree, ())
    return traced


def _hits(path: Path, /) -> Iterator[Hit]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(ROOT).as_posix()
    provider = bool(_PROVIDER_PARTS & set(path.relative_to(PACKAGE).parts[:-1])) or (
        "worker" in path.stem or "provider" in path.stem
    )
    traced = _traced_functions(tree)

    def visit(node: ast.AST, scope: _Scope) -> Iterator[Hit]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from visit(child, (*scope, ("function", child.name)))
                continue
            if isinstance(child, ast.ClassDef):
                yield from visit(child, (*scope, ("class", child.name)))
                continue
            if isinstance(child, ast.Call):
                name = _call_name(child)
                if name in _CONVERSIONS or name in _SAFE_POINTS:
                    yield Hit(
                        relative,
                        child.lineno,
                        name,
                        _category(child, name, scope, traced, provider),
                        ".".join(_qualified(scope)) or "<module>",
                    )
            yield from visit(child, scope)

    yield from visit(tree, ())


def _category(
    node: ast.Call,
    name: str,
    scope: _Scope,
    traced: set[tuple[str, ...]],
    provider: bool,
    /,
) -> str:
    functions = [function for kind, function in scope if kind == "function"]
    if name in _SAFE_POINTS:
        return "explicit-safe-point"
    if node.args and _reads_static_metadata(node.args[0]):
        return "static-shape"
    if any(
        _qualified(scope[: index + 1]) in traced
        for index, (kind, _) in enumerate(scope)
        if kind == "function"
    ):
        return "traced-body"
    if provider:
        return "external-provider"
    if not functions or any(
        function.startswith(_PREPARATION_PREFIXES) for function in functions
    ):
        return "host-preparation"
    if any(function.startswith(_OBSERVATION_PREFIXES) for function in functions):
        return "output-observation"
    return "unclassified"


def audit(root: Path, /) -> list[Hit]:
    hits: list[Hit] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            hits.extend(_hits(path))
    return sorted(hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report possible host synchronization.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--category", default=None)
    arguments = parser.parse_args()
    hits = audit(PACKAGE)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps([asdict(hit) for hit in hits], indent=2) + "\n", encoding="utf-8"
        )
    for hit in hits:
        if arguments.category in (None, hit.category):
            print(f"{hit.file}:{hit.line}: {hit.category} {hit.call} in {hit.function}")
    counts = Counter(hit.category for hit in hits)
    print(f"total: {len(hits)} {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

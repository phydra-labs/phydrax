#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rank strict modules that would benefit from structural contracts.

The report is read-only. It considers dataclass-based `StrictModule` classes that
have not opted in (``__strict_contract__ = True``) and scores the static facts a
contract could state:

- closed selectors: fields annotated with a `Literal` alias;
- static sizes: `int` static fields whose names count or size something;
- aligned arrays: two or more dynamic array fields;
- identifier tuples: `tuple[str, ...]` static fields;
- host arrays: `numpy.ndarray` fields.

Classes with a generated constructor are listed first because they currently run
no validation. Classes whose `__check_init__` or constructor uses
`equinox.error_if` hold data-dependent numerical predicates and are marked, since
those stay with their owner.

With ``--signatures`` the report instead lists hand-written argument guards that
restate a nominal parameter annotation (``if not isinstance(x, T): raise
TypeError`` where ``x: T``). Inside a `phydrax.typing.checked` function whose
compiled plan enforces ``T`` such a guard is redundant and is a finding; the
report imports those modules to read the plan. Every other guard is retained:
module-level functions are content-addressed by their code, so their bodies
stay unchanged; selector, protocol, and conversion annotations are static-only
under `checked`; and an unchecked method keeps its guard where checking its
signature would not preserve its contract (see
``docs/plans/runtime-contract-dispositions.tsv``).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
_SIZE_WORDS = ("count", "size", "dimension", "rank", "length", "order", "degree")


@dataclass(frozen=True, slots=True)
class Candidate:
    file: str
    line: int
    name: str
    generated_constructor: bool
    selectors: int
    sizes: int
    arrays: int
    identifier_tuples: int
    host_arrays: int
    numerical_predicates: bool

    @property
    def score(self) -> int:
        # Metadata facts are cheap and exact to state; array extents need nominal
        # dimensions chosen by the owner, so aligned arrays count once.
        return (
            3 * self.selectors
            + 3 * self.sizes
            + 3 * self.identifier_tuples
            + self.host_arrays
            + (1 if self.arrays >= 2 else 0)
        )


@dataclass(frozen=True, slots=True)
class SignatureGuard:
    file: str
    line: int
    function: str
    parameter: str
    annotation: str
    disposition: str


# Scalars and containers keep their owner's admission and normalization.
_PRIMITIVE_ANNOTATIONS = frozenset(
    {"bool", "bytes", "complex", "dict", "float", "frozenset", "int", "list", "object"}
    | {"set", "str", "tuple"}
)


def _annotation_guard(statement: ast.If, /) -> tuple[str, str] | None:
    """Return ``(parameter, type)`` of an ``if not isinstance(...): raise TypeError``."""
    match statement:
        case ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id="isinstance"), args=[ast.Name(id=name), kind]
                ),
            ),
            body=[ast.Raise(exc=ast.Call(func=ast.Name(id="TypeError")))],
            orelse=[],
        ):
            return name, ast.unparse(kind)
        case _:
            return None


def _own_scope(node: ast.AST, /) -> list[ast.AST]:
    """Return the nodes of one scope without entering nested functions or classes."""
    nodes: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        nodes.append(child)
        if not isinstance(
            child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef
        ):
            stack.extend(ast.iter_child_nodes(child))
    return nodes


def _function_guards(
    path: Path,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
    method: bool,
    /,
) -> list[SignatureGuard]:
    annotations = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if argument.annotation is not None
    }
    checked = any(
        ast.unparse(decorator).rpartition(".")[2] == "checked"
        for decorator in function.decorator_list
    )
    guards: list[SignatureGuard] = []
    for statement in _own_scope(function):
        if not isinstance(statement, ast.If):
            continue
        found = _annotation_guard(statement)
        if found is None:
            continue
        name, kind = found
        if annotations.get(name) != kind or kind in _PRIMITIVE_ANNOTATIONS:
            continue
        if checked:
            disposition = "checked"
        elif method:
            disposition = "retained-unchecked-method"
        else:
            disposition = "retained-source-function"
        guards.append(
            SignatureGuard(
                str(path.relative_to(ROOT)),
                statement.lineno,
                qualname,
                name,
                kind,
                disposition,
            )
        )
    return guards


def _scope_guards(
    path: Path, body: list[ast.stmt], prefix: str, in_class: bool, /
) -> list[SignatureGuard]:
    guards: list[SignatureGuard] = []
    for statement in body:
        match statement:
            case ast.ClassDef(name=name, body=inner):
                guards.extend(_scope_guards(path, inner, f"{prefix}{name}.", True))
            case (
                ast.FunctionDef(name=name, body=inner)
                | ast.AsyncFunctionDef(name=name, body=inner)
            ):
                guards.extend(
                    _function_guards(path, statement, f"{prefix}{name}", in_class)
                )
                guards.extend(
                    _scope_guards(path, inner, f"{prefix}{name}.<locals>.", False)
                )
            case _:
                pass
    return guards


def _checked_function(file: str, qualname: str, /) -> object:
    parts = list(Path(file).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    owner: object = importlib.import_module(".".join(parts))
    *outer, name = qualname.split(".")
    for part in outer:
        owner = vars(owner)[part]
    function = vars(owner)[name]
    return (
        function.__func__
        if isinstance(function, classmethod | staticmethod)
        else function
    )


def _enforced_by_checked(guard: SignatureGuard, /) -> bool:
    """Return whether the `checked` plan of the guard's function enforces its type."""
    # Imported on use: only the signature report needs the package at runtime.
    from phydrax._typing_plan import NominalContract
    from phydrax._typing_signature import checked_plan

    if "<locals>" in guard.function:
        return True
    function = _checked_function(guard.file, guard.function)
    plan = checked_plan(function) if callable(function) else None
    if plan is None:
        raise TypeError(f"{guard.file}: {guard.function} is not a checked function.")
    return any(
        slot.name == guard.parameter and isinstance(slot.contract, NominalContract)
        for slot in plan.slots
    )


def audit_signature_guards(root: Path, /) -> list[SignatureGuard]:
    """Return every argument guard that restates its parameter's nominal annotation."""
    guards: list[SignatureGuard] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guards.extend(_scope_guards(path, tree.body, "", False))
    classified = [
        guard
        if guard.disposition != "checked"
        else replace(
            guard,
            disposition="redundant-at-checked-boundary"
            if _enforced_by_checked(guard)
            else "retained-static-only-annotation",
        )
        for guard in guards
    ]
    return sorted(classified, key=lambda item: (item.file, item.line, item.parameter))


def _literal_aliases(tree: ast.Module, /) -> set[str]:
    aliases: set[str] = set()
    for node in tree.body:
        value = (
            node.value
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.TypeAlias)
            else None
        )
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else node.target
            if isinstance(node, ast.AnnAssign)
            else node.name
            if isinstance(node, ast.TypeAlias)
            else None
        )
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and ast.unparse(value.value) in ("Literal", "typing.Literal")
        ):
            aliases.add(target.id)
    return aliases


def _is_strict_module(node: ast.ClassDef, /) -> bool:
    return any(ast.unparse(base).endswith("StrictModule") for base in node.bases)


def _static(field: ast.AnnAssign, /) -> bool:
    value = field.value
    return (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) in ("eqx.field", "field", "equinox.field")
        and any(
            keyword.arg == "static"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in value.keywords
        )
    )


def _candidate(path: Path, node: ast.ClassDef, aliases: set[str], /) -> Candidate | None:
    body = node.body
    if any(
        isinstance(item, ast.Assign)
        and any(ast.unparse(target) == "__strict_contract__" for target in item.targets)
        for item in body
    ):
        return None
    fields = [
        item
        for item in body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    if not fields:
        return None
    methods = {item.name: item for item in body if isinstance(item, ast.FunctionDef)}
    selectors = sizes = arrays = identifier_tuples = host_arrays = 0
    for field in fields:
        annotation = ast.unparse(field.annotation).replace('"', "")
        name = field.target.id if isinstance(field.target, ast.Name) else ""
        head = annotation.split("[", 1)[0]
        if head in aliases or head in ("Literal", "typing.Literal"):
            selectors += 1
        elif (
            annotation == "int"
            and _static(field)
            and any(word in name for word in _SIZE_WORDS)
        ):
            sizes += 1
        elif annotation in ("Array", "jax.Array") and not _static(field):
            arrays += 1
        elif annotation in ("tuple[str, ...]", "Tuple[str, ...]"):
            identifier_tuples += 1
        elif head in ("np.ndarray", "numpy.ndarray", "npt.NDArray"):
            host_arrays += 1
    numerical = any(
        isinstance(call, ast.Call) and ast.unparse(call.func).endswith("error_if")
        for method in methods.values()
        for call in ast.walk(method)
    )
    candidate = Candidate(
        path.relative_to(ROOT).as_posix(),
        node.lineno,
        node.name,
        "__init__" not in methods,
        selectors,
        sizes,
        arrays,
        identifier_tuples,
        host_arrays,
        numerical,
    )
    return candidate if candidate.score else None


def audit(root: Path, /) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _literal_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_strict_module(node):
                candidate = _candidate(path, node, aliases)
                if candidate is not None:
                    candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            not item.generated_constructor,
            -item.score,
            item.file,
            item.line,
        ),
    )


def _report_signature_guards() -> None:
    guards = audit_signature_guards(PACKAGE)
    counts: dict[str, int] = {}
    for guard in guards:
        counts[guard.disposition] = counts.get(guard.disposition, 0) + 1
        if guard.disposition == "redundant-at-checked-boundary":
            print(
                f"{guard.file}:{guard.line}: {guard.function} restates "
                f"{guard.parameter}: {guard.annotation}"
            )
    for disposition, count in sorted(counts.items()):
        print(f"{disposition}: {count}")
    if counts.get("redundant-at-checked-boundary", 0):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank structural contract candidates.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--signatures",
        action="store_true",
        help="report argument guards that restate a nominal parameter annotation",
    )
    arguments = parser.parse_args()
    if arguments.signatures:
        _report_signature_guards()
        return
    candidates = audit(PACKAGE)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                [asdict(item) | {"score": item.score} for item in candidates], indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    for item in candidates[: arguments.limit]:
        print(
            f"{item.file}:{item.line}: {item.name} score={item.score}"
            f" generated={item.generated_constructor} selectors={item.selectors}"
            f" sizes={item.sizes} arrays={item.arrays}"
            f" identifier_tuples={item.identifier_tuples} host_arrays={item.host_arrays}"
            f" numerical_predicates={item.numerical_predicates}"
        )
    print(f"total: {len(candidates)} candidates")


if __name__ == "__main__":
    main()

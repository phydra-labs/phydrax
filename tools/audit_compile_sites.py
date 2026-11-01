#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report compiled-program construction that repeats per call.

A compiled program must be built once and keep a stable identity: a
module-level function or method compiled by decorator, a program owned by one
live object (``phydrax._owned_programs.run_owned_program`` or an attribute set
while preparing the object), or a memoized factory keyed on static
configuration. The audit statically reports every other construction:

- ``per-call-compile``: ``jax.jit``, ``jax.pmap``, ``jax.shard_map``,
  ``eqx.filter_jit`` or ``eqx.filter_pmap`` applied (by call or decorator) inside
  a function body. Each call builds a new wrapper; a lambda, nested function,
  ``functools.partial`` or transformed (``jax.vmap``/``jax.grad``) target has a
  new identity per call and so is retraced and recompiled per call.
- ``closure-static-arg``: a lambda, nested function, ``functools.partial`` or
  transformed callable passed as a static argument of a known compiled callable
  (every non-array argument of ``eqx.filter_jit``, ``static_argnums`` /
  ``static_argnames`` of ``jax.jit``). Static arguments hash by identity, so
  each call compiles again.
- ``closure-owned-method``: such a callable passed as the method of
  ``run_owned_program``, whose program key is the owner and phase only.

Sites with an owned lifetime that the rules cannot see, explicit compile APIs,
and deliberate one-shot qualification/calibration programs are recorded with a
rationale in ``tools/audit_compile_sites_allowlist.toml`` (matched by file,
qualified function and kind). A clean report is not a claim that every transform
is cached or that every wrapper is faster: AD, abstract evaluation and eager
control-flow tracing require focused runtime measurements. ``--check`` exits
non-zero on findings outside the allowlist and on stale allowlist entries.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
ALLOWLIST = Path(__file__).resolve().with_name("audit_compile_sites_allowlist.toml")

COMPILERS = frozenset(
    {
        "jax.jit",
        "jax.pmap",
        "jax.shard_map",
        "jax.experimental.shard_map.shard_map",
        "equinox.filter_jit",
        "equinox.filter_pmap",
    }
)
ALL_STATIC_COMPILERS = frozenset({"equinox.filter_jit", "equinox.filter_pmap"})
FUNCTION_TRANSFORMS = frozenset(
    {
        "functools.partial",
        "jax.vmap",
        "jax.grad",
        "jax.value_and_grad",
        "jax.jacfwd",
        "jax.jacrev",
        "jax.hessian",
        "jax.checkpoint",
        "jax.remat",
        "equinox.filter_vmap",
        "equinox.filter_grad",
        "equinox.filter_value_and_grad",
        "equinox.filter_checkpoint",
    }
)
MEMOIZERS = frozenset({"functools.cache", "functools.lru_cache"})
OWNED_PROGRAM = "phydrax._owned_programs.run_owned_program"
PREPARE_METHODS = frozenset({"__init__", "__post_init__", "__check_init__"})


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    file: str
    line: int
    kind: str
    function: str
    detail: str


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    file: str
    function: str
    kind: str
    rationale: str


@dataclass(frozen=True, slots=True)
class Report:
    findings: tuple[Finding, ...]
    allowed: tuple[Finding, ...]
    stale: tuple[AllowlistEntry, ...]


@dataclass(frozen=True, slots=True)
class _Compiled:
    """Static-argument contract of a compiled callable."""

    all_static: bool
    positions: frozenset[int] = frozenset()
    names: frozenset[str] = frozenset()


@dataclass(slots=True)
class _Scope:
    kind: str
    name: str
    node: ast.AST
    closures: set[str] = field(default_factory=set)
    methods: dict[str, _Compiled] = field(default_factory=dict)


def _module_name(path: Path, root: Path, /) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _import_owner(module: str, is_package: bool, node: ast.ImportFrom, /) -> str:
    if node.level == 0:
        return node.module or ""
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".")
    prefix = parts[: len(parts) - node.level + 1]
    return ".".join((*prefix, *((node.module,) if node.module else ())))


def _imports(
    nodes: Sequence[ast.AST], module: str, is_package: bool, /
) -> dict[str, str]:
    """Map every imported local name (any scope) to its qualified origin."""
    names: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    head = alias.name.partition(".")[0]
                    names[head] = head
                else:
                    names[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            owner = _import_owner(module, is_package, node)
            for alias in node.names:
                if alias.name != "*":
                    names[alias.asname or alias.name] = f"{owner}.{alias.name}"
    return names


def _qualify(node: ast.expr, imports: Mapping[str, str], /) -> str | None:
    if isinstance(node, ast.Name):
        return imports.get(node.id)
    if isinstance(node, ast.Attribute):
        head = _qualify(node.value, imports)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _keyword_constants(keywords: Sequence[ast.keyword], name: str, /) -> set[object]:
    for keyword in keywords:
        if keyword.arg != name:
            continue
        value = keyword.value
        elements = value.elts if isinstance(value, ast.Tuple | ast.List) else [value]
        return {e.value for e in elements if isinstance(e, ast.Constant)}
    return set()


@dataclass(frozen=True, slots=True)
class _Compiler:
    name: str
    keywords: tuple[ast.keyword, ...]


def _compiler(node: ast.expr, imports: Mapping[str, str], /) -> _Compiler | None:
    """Return the compiler that ``node`` denotes when applied to a function.

    Recognizes the bare constructor, ``functools.partial(constructor, ...)``
    and a keyword-only constructor call such as ``eqx.filter_jit(donate=...)``.
    """
    name = _qualify(node, imports)
    if name in COMPILERS:
        return _Compiler(name, ())
    if not isinstance(node, ast.Call):
        return None
    callee = _qualify(node.func, imports)
    if callee in COMPILERS and not node.args:
        return _Compiler(callee, tuple(node.keywords))
    if callee == "functools.partial" and node.args:
        inner = _qualify(node.args[0], imports)
        if inner in COMPILERS and len(node.args) == 1:
            return _Compiler(inner, tuple(node.keywords))
    return None


def _contract(
    compiler: _Compiler, function: ast.FunctionDef | ast.AsyncFunctionDef | None, /
) -> _Compiled | None:
    if compiler.name in ALL_STATIC_COMPILERS:
        return _Compiled(all_static=True)
    if compiler.name != "jax.jit":
        return None
    positions = {
        value
        for value in _keyword_constants(compiler.keywords, "static_argnums")
        if isinstance(value, int)
    }
    names = {
        value
        for value in _keyword_constants(compiler.keywords, "static_argnames")
        if isinstance(value, str)
    }
    if function is not None:
        parameters = [*function.args.posonlyargs, *function.args.args]
        for index, parameter in enumerate(parameters):
            if parameter.arg in names:
                positions.add(index)
            if index in positions:
                names.add(parameter.arg)
    if not positions and not names:
        return None
    return _Compiled(False, frozenset(positions), frozenset(names))


def _registry_entries(
    tree: ast.Module, imports: Mapping[str, str], /
) -> dict[str, _Compiled]:
    """Compiled callables bound at module level, keyed by name or ``Class.method``."""
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    entries: dict[str, _Compiled] = {}

    def decorated(
        node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str, shift: int, /
    ) -> None:
        for decorator in node.decorator_list:
            compiler = _compiler(decorator, imports)
            if compiler is None:
                continue
            contract = _contract(compiler, node)
            if contract is None:
                continue
            if shift and not contract.all_static:
                contract = _Compiled(
                    False,
                    frozenset(p - shift for p in contract.positions if p >= shift),
                    contract.names,
                )
            entries[prefix + node.name] = contract

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            decorated(node, "", 0)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    static = any(
                        _qualify(d, imports) == "staticmethod"
                        or (isinstance(d, ast.Name) and d.id == "staticmethod")
                        for d in member.decorator_list
                    )
                    decorated(member, f"{node.name}.", 0 if static else 1)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            compiler = _compiler(call.func, imports)
            if compiler is None:
                name = _qualify(call.func, imports)
                if name not in COMPILERS or not call.args:
                    continue
                compiler = _Compiler(name, tuple(call.keywords))
            target = call.args[0] if call.args else None
            function = functions.get(target.id) if isinstance(target, ast.Name) else None
            contract = _contract(compiler, function)
            if contract is None:
                continue
            for name_node in node.targets:
                if isinstance(name_node, ast.Name):
                    entries[name_node.id] = contract
    return entries


def _unwrap_target(node: ast.expr, imports: Mapping[str, str], /) -> ast.expr:
    """Strip ``partial``/``vmap``/``grad`` wrappers off a compile target."""
    while (
        isinstance(node, ast.Call)
        and node.args
        and _qualify(node.func, imports) in FUNCTION_TRANSFORMS
    ):
        node = node.args[0]
    return node


def _traced_names(
    nodes: Sequence[ast.AST], imports: Mapping[str, str], /
) -> frozenset[str]:
    """Names of functions whose bodies only run under a trace.

    A function is traced when it is the target of a compile construction or the
    method of ``run_owned_program``; constructions in its body (and in functions
    nested in it) are inline in the enclosing program and compile with it.
    """
    names: set[str] = set()
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        target: ast.expr | None = None
        if _qualify(node.func, imports) == OWNED_PROGRAM and len(node.args) >= 3:
            target = node.args[2]
        elif node.args and (
            _compiler(node.func, imports) is not None
            or _qualify(node.func, imports) in COMPILERS
        ):
            target = node.args[0]
        if target is None:
            continue
        target = _unwrap_target(target, imports)
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return frozenset(names)


class _Visitor(ast.NodeVisitor):
    def __init__(
        self,
        relative: str,
        module: str,
        imports: Mapping[str, str],
        registry: Mapping[str, Mapping[str, _Compiled]],
        /,
    ) -> None:
        self.relative = relative
        self.module = module
        self.imports = dict(imports)
        self.registry = registry
        self.local = registry.get(module, {})
        self.scopes: list[_Scope] = []
        self.findings: list[Finding] = []
        self.self_assigned: set[int] = set()
        self.traced: frozenset[str] = frozenset()

    # -- scope bookkeeping -------------------------------------------------

    def _function_scopes(self) -> list[_Scope]:
        return [scope for scope in self.scopes if scope.kind == "function"]

    def _qualname(self) -> str:
        return ".".join(scope.name for scope in self.scopes) or "<module>"

    def _closure_names(self) -> set[str]:
        names: set[str] = set()
        for scope in self._function_scopes():
            names |= scope.closures
        return names

    def _exempt_scope(self) -> bool:
        scopes = self._function_scopes()
        if not scopes:
            return True
        for scope in scopes:
            node = scope.node
            assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            if node.name in self.traced:
                return True
            if any(_compiler(d, self.imports) is not None for d in node.decorator_list):
                return True
        innermost = scopes[-1].node
        assert isinstance(innermost, ast.FunctionDef | ast.AsyncFunctionDef)
        for decorator in innermost.decorator_list:
            head = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _qualify(head, self.imports) in MEMOIZERS:
                return True
        return False

    def _report(self, node: ast.AST, kind: str, detail: str, /) -> None:
        line = getattr(node, "lineno", 0)
        self.findings.append(Finding(self.relative, line, kind, self._qualname(), detail))

    # -- classification ----------------------------------------------------

    def _target_kind(self, node: ast.expr | None, /) -> str:
        if node is None:
            return "decorated function"
        if isinstance(node, ast.Lambda):
            return "lambda"
        if isinstance(node, ast.Name):
            if node.id in self._closure_names():
                return f"closure {node.id}"
            return f"function {node.id}"
        if isinstance(node, ast.Call):
            callee = _qualify(node.func, self.imports)
            if callee in FUNCTION_TRANSFORMS:
                inner = self._target_kind(node.args[0] if node.args else None)
                return f"{callee.rsplit('.', 1)[-1]} of {inner}"
            return "call result"
        if isinstance(node, ast.Attribute):
            return f"attribute {ast.unparse(node)}"
        return type(node).__name__.lower()

    def _closure_like(self, node: ast.expr, /) -> str | None:
        if isinstance(node, ast.Lambda):
            return "lambda"
        if isinstance(node, ast.Name) and node.id in self._closure_names():
            return f"closure {node.id}"
        if isinstance(node, ast.Call):
            callee = _qualify(node.func, self.imports)
            if callee in FUNCTION_TRANSFORMS:
                return f"{callee.rsplit('.', 1)[-1]}(...)"
        return None

    def _owned_by_attribute(self, node: ast.AST, /) -> bool:
        """Construction stored on ``self`` while preparing the object."""
        scopes = self._function_scopes()
        if not scopes or not isinstance(scopes[-1].node, ast.FunctionDef):
            return False
        return scopes[-1].node.name in PREPARE_METHODS and id(node) in self.self_assigned

    def _callee_contract(self, node: ast.expr, /) -> _Compiled | None:
        if isinstance(node, ast.Name):
            if node.id in self.local:
                return self.local[node.id]
            origin = self.imports.get(node.id)
            if origin is not None:
                owner, _, name = origin.rpartition(".")
                return self.registry.get(owner, {}).get(name)
            return None
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                for scope in reversed(self.scopes):
                    if scope.kind == "class":
                        return scope.methods.get(node.attr)
                return None
            origin = _qualify(node.value, self.imports)
            if origin is not None:
                return self.registry.get(origin, {}).get(node.attr)
        return None

    # -- visitors ------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods: dict[str, _Compiled] = {}
        if not self._function_scopes():
            prefix = f"{node.name}."
            for key, contract in self.local.items():
                if key.startswith(prefix):
                    methods[key.removeprefix(prefix)] = contract
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.scopes.append(_Scope("class", node.name, node, methods=methods))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        nested = bool(self._function_scopes())
        if nested:
            self._function_scopes()[-1].closures.add(node.name)
        for decorator in node.decorator_list:
            compiler = _compiler(decorator, self.imports)
            if compiler is not None and nested and not self._exempt_scope():
                self._report(
                    decorator,
                    "per-call-compile",
                    f"{compiler.name} decorating nested def {node.name}",
                )
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self.scopes.append(_Scope("function", node.name, node))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _record_self_assignment(
        self, targets: Sequence[ast.expr], value: ast.expr | None, /
    ) -> None:
        if value is not None and all(
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
            for t in targets
        ):
            self.self_assigned.add(id(value))

    def visit_Assign(self, node: ast.Assign) -> None:
        scopes = self._function_scopes()
        if scopes and (
            isinstance(node.value, ast.Lambda)
            or (
                isinstance(node.value, ast.Call)
                and _qualify(node.value.func, self.imports) in FUNCTION_TRANSFORMS
            )
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scopes[-1].closures.add(target.id)
        self._record_self_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_self_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        compiler = _compiler(node.func, self.imports)
        direct = _qualify(node.func, self.imports)
        constructs = (compiler is not None and bool(node.args)) or (
            direct in COMPILERS and bool(node.args)
        )
        if constructs and not self._exempt_scope() and not self._owned_by_attribute(node):
            name = compiler.name if compiler is not None else direct
            self._report(
                node,
                "per-call-compile",
                f"{name} of {self._target_kind(node.args[0])}",
            )
        if self._function_scopes():
            if direct == OWNED_PROGRAM and len(node.args) >= 3:
                kind = self._closure_like(node.args[2])
                if kind is not None:
                    self._report(node, "closure-owned-method", kind)
            contract = self._callee_contract(node.func)
            if contract is not None:
                callee = ast.unparse(node.func)
                for index, argument in enumerate(node.args):
                    if contract.all_static or index in contract.positions:
                        kind = self._closure_like(argument)
                        if kind is not None:
                            self._report(
                                node,
                                "closure-static-arg",
                                f"{kind} as argument {index} of {callee}",
                            )
                for keyword in node.keywords:
                    if keyword.arg is None:
                        continue
                    if contract.all_static or keyword.arg in contract.names:
                        kind = self._closure_like(keyword.value)
                        if kind is not None:
                            self._report(
                                node,
                                "closure-static-arg",
                                f"{kind} as {keyword.arg}= of {callee}",
                            )
        self.generic_visit(node)


def _sources(root: Path, /) -> tuple[Path, ...]:
    return tuple(
        path for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts
    )


_COMPILE_WORDS = re.compile(r"\b(?:jit|filter_jit|shard_map|pmap|filter_pmap)\b")


@dataclass(frozen=True, slots=True)
class _Parsed:
    module: str
    tree: ast.Module
    imports: dict[str, str]
    traced: frozenset[str]


def _parse(path: Path, base: Path, text: str, /) -> _Parsed:
    module = _module_name(path, base)
    tree = ast.parse(text, filename=str(path))
    nodes = tuple(ast.walk(tree))
    imports = _imports(nodes, module, path.name == "__init__.py")
    return _Parsed(module, tree, imports, _traced_names(nodes, imports))


def audit(root: Path, /, *, base: Path | None = None) -> list[Finding]:
    """Return every compile-site finding under ``root`` (paths relative to ``base``).

    The first pass parses only modules that mention a compile constructor and
    records the compiled callables they bind; the second visits those modules
    and every module that names one of the recorded callables.
    """
    base = ROOT if base is None else base
    texts: dict[Path, str] = {}
    parsed: dict[Path, _Parsed] = {}
    registry: dict[str, dict[str, _Compiled]] = {}
    for path in _sources(root):
        text = path.read_text(encoding="utf-8")
        texts[path] = text
        if _COMPILE_WORDS.search(text) is None:
            continue
        record = _parse(path, base, text)
        parsed[path] = record
        entries = _registry_entries(record.tree, record.imports)
        if entries:
            registry[record.module] = entries
    callables = sorted(
        {name.rpartition(".")[2] for entries in registry.values() for name in entries}
    )
    mentions = re.compile(
        r"\b(?:" + "|".join(map(re.escape, (*callables, "run_owned_program"))) + r")\b"
    )
    findings: list[Finding] = []
    for path, text in texts.items():
        record = parsed.get(path)
        if record is None:
            if mentions.search(text) is None:
                continue
            record = _parse(path, base, text)
        visitor = _Visitor(
            path.relative_to(base).as_posix(), record.module, record.imports, registry
        )
        visitor.traced = record.traced
        visitor.visit(record.tree)
        findings.extend(visitor.findings)
    return sorted(set(findings))


def load_allowlist(path: Path, /) -> tuple[AllowlistEntry, ...]:
    if not path.exists():
        return ()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    entries = []
    for raw in data.get("site", []):
        entry = AllowlistEntry(
            raw["file"], raw["function"], raw["kind"], raw["rationale"].strip()
        )
        if not entry.rationale:
            raise ValueError(f"allowlist entry without rationale: {entry}")
        entries.append(entry)
    return tuple(entries)


def classify(
    findings: Sequence[Finding], allowlist: Sequence[AllowlistEntry], /
) -> Report:
    keys = {(entry.file, entry.function, entry.kind): entry for entry in allowlist}
    used: set[tuple[str, str, str]] = set()
    open_findings: list[Finding] = []
    allowed: list[Finding] = []
    for finding in findings:
        key = (finding.file, finding.function, finding.kind)
        if key in keys:
            used.add(key)
            allowed.append(finding)
        else:
            open_findings.append(finding)
    stale = tuple(entry for key, entry in keys.items() if key not in used)
    return Report(tuple(open_findings), tuple(allowed), stale)


def iter_lines(report: Report, /, *, include_allowed: bool) -> Iterator[str]:
    for finding in report.findings:
        yield (
            f"{finding.file}:{finding.line}: {finding.kind} "
            f"[{finding.function}] {finding.detail}"
        )
    if include_allowed:
        for finding in report.allowed:
            yield (
                f"{finding.file}:{finding.line}: allowed {finding.kind} "
                f"[{finding.function}] {finding.detail}"
            )
    for entry in report.stale:
        yield f"{entry.file}: stale allowlist entry {entry.kind} [{entry.function}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report per-call compile sites.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--allowed", action="store_true", help="list allowed sites")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on findings outside the allowlist or stale entries",
    )
    arguments = parser.parse_args()
    report = classify(audit(PACKAGE), load_allowlist(ALLOWLIST))
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                {
                    "findings": [asdict(finding) for finding in report.findings],
                    "allowed": [asdict(finding) for finding in report.allowed],
                    "stale": [asdict(entry) for entry in report.stale],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    for line in iter_lines(report, include_allowed=arguments.allowed):
        print(line)
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    print(
        f"total: {len(report.findings)} {dict(sorted(counts.items()))} "
        f"allowed: {len(report.allowed)} stale: {len(report.stale)}"
    )
    if arguments.check and (report.findings or report.stale):
        sys.exit(1)


if __name__ == "__main__":
    main()

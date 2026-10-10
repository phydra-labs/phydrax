#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import ast
import typing
from importlib import import_module
from pathlib import Path
from typing import Any

import griffe


typing.GENERATING_DOCUMENTATION = True  # ty: ignore[unresolved-attribute]


def _is_type_expression(value: object, /) -> bool:
    return typing.get_origin(value) is not None and bool(typing.get_args(value))


class _BriefNames(ast.NodeTransformer):
    """Replace qualified references such as `typing.Literal` by their final name."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.Name:
        return ast.copy_location(ast.Name(node.attr, ast.Load()), node)


def _type_expression(
    value: object, parent: griffe.Module | griffe.Class, /
) -> str | griffe.Expr:
    source = value.__qualname__ if isinstance(value, type) else repr(value)
    try:
        node = _BriefNames().visit(ast.parse(source, mode="eval").body)
    except SyntaxError:
        return source
    expression = griffe.safe_get_annotation(
        node, parent, parse_strings=False, log_level=griffe.LogLevel.debug
    )
    return source if expression is None else expression


class RuntimeTypeAliases(griffe.Extension):
    """Document runtime type aliases faithfully under forced inspection.

    Inspection treats every callable member as an import from its `__module__`.
    Parameterized type expressions such as `Literal["a", "b"]` are callable and
    report their origin's module, so `Name: TypeAlias = Literal["a", "b"]` would
    become an unresolvable alias of `typing.Literal`; non-callable ones such as
    `A | B` would become attributes carrying the `UnionType` docstring. Both are
    documented as the equivalent `type Name = ...` alias instead. Every inspected
    type alias shows its value in source form, with quoted literals, `...`, and
    unqualified names, and drops the `TypeAliasType` class docstring that
    inspection reports. A `type` statement whose lazy value names a symbol bound
    only for type checking cannot be evaluated; inspection would fail and drop its
    whole module, so its value is documented from the source statement instead.
    """

    def on_module_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        mod: griffe.Module,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        """Materialize lazy public exports before runtime member inspection."""
        del kwargs
        if not isinstance(node, griffe.ObjectNode) or not isinstance(
            agent, griffe.Inspector
        ):
            return
        runtime_module = node.obj
        symbol_modules = getattr(runtime_module, "_SYMBOL_MODULES", {})
        for name, (module_name, symbol) in symbol_modules.items():
            try:
                owner = import_module(module_name, runtime_module.__package__)
            except ImportError:
                continue
            target = owner.__name__ if symbol is None else f"{owner.__name__}.{symbol}"
            mod.set_member(
                name,
                griffe.Alias(name, target, parent=mod, analysis="dynamic"),
            )
        for module_name in getattr(runtime_module, "_FACADE_EXPORT_MODULES", ()):
            try:
                owner = import_module(module_name, runtime_module.__package__)
            except ImportError:
                continue
            for name in getattr(owner, "__all__", ()):
                if name not in mod.members and name not in runtime_module.__dict__:
                    mod.set_member(
                        name,
                        griffe.Alias(
                            name,
                            f"{owner.__name__}.{name}",
                            parent=mod,
                            analysis="dynamic",
                        ),
                    )
        if runtime_module.__name__ == "phydrax.operators":
            mod.set_member(
                "graph_degree",
                griffe.Alias(
                    "graph_degree",
                    "phydrax.operators.graph.degree",
                    parent=mod,
                    analysis="dynamic",
                ),
            )
        exported = getattr(runtime_module, "__all__", ())
        for name in exported:
            try:
                getattr(runtime_module, name)
            except (AttributeError, ImportError):
                # Optional providers remain documented at their owner boundary.
                continue

    def on_alias_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        alias: griffe.Alias,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, griffe.ObjectNode) and isinstance(agent, griffe.Inspector):
            _document_type_alias(agent.current, alias.name, vars(node.obj)[alias.name])

    def on_attribute_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        attr: griffe.Attribute,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, griffe.ObjectNode) and isinstance(agent, griffe.Inspector):
            _document_type_alias(agent.current, attr.name, node.obj)

    def on_type_alias_node(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if not isinstance(node, griffe.ObjectNode) or not isinstance(
            agent, griffe.Inspector
        ):
            return
        try:
            node.obj.__value__
        except NameError:
            if agent.filepath is None:
                raise
            # The lazy value names a symbol bound only for type checking: rebuild
            # the alias with the forward reference its source statement spells.
            node.obj = type(node.obj)(
                node.name,
                _type_alias_source(agent.filepath, node.name),
                type_params=node.obj.__type_params__,
            )

    def on_type_alias_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        type_alias: griffe.TypeAlias,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(node, griffe.ObjectNode) and isinstance(agent, griffe.Inspector):
            value = node.obj.__value__
            # Griffe already renders string values from source as forward references.
            if not isinstance(value, str):
                type_alias.value = _type_expression(value, agent.current)
            type_alias.docstring = None


def _type_alias_source(module_path: Path, name: str, /) -> str:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for statement in module.body:
        if isinstance(statement, ast.TypeAlias) and statement.name.id == name:
            return ast.unparse(statement.value)
    raise LookupError(f"{module_path} has no module-level `type {name}` statement")


def _document_type_alias(
    parent: griffe.Module | griffe.Class, name: str, value: object, /
) -> None:
    if not _is_type_expression(value):
        return
    parent.set_member(
        name,
        griffe.TypeAlias(
            name,
            value=_type_expression(value, parent),
            parent=parent,
            analysis="dynamic",
        ),
    )

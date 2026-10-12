#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import ast
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import griffe


_PUBLIC_API_MANIFEST = (
    Path(__file__).resolve().parent / "docs" / "data" / "public_api.json"
)
_EQUINOX_MODULES = frozenset({"equinox.Module", "equinox._module._module.Module"})
_ABSTRACT_VARIABLES = frozenset(
    {
        "equinox.AbstractVar",
        "equinox.AbstractClassVar",
        "equinox._module._better_abstract.AbstractVar",
        "equinox._module._better_abstract.AbstractClassVar",
    }
)
# Phydrax array-role helpers accept exactly the keyword arguments of `equinox.field`.
_FIELD_FACTORIES = frozenset(
    {
        "dataclasses.field",
        "equinox.field",
        "equinox._module._field.field",
        "phydrax._trainable.fixed_field",
        "phydrax._trainable.model_state_field",
        "phydrax._trainable.parameter_field",
    }
)


class PublicFacadeExports(griffe.Extension):
    """Bind every public export to its static definition.

    Documentation is collected by static analysis, which never imports Phydrax.
    Static analysis cannot follow four runtime constructs that facades use:

    - a module ``__getattr__`` that imports an exported name on first access, so
      importing a facade does not import every implementation module;
    - an ``__all__`` computed at import time, such as every public global;
    - one assignment that unpacks a tuple into several module constants;
    - an assignment ``Name = Other`` that binds a second name to an object.

    The checked public API manifest owns the exact exported names. Each module's
    exports become its manifest names, and each name a facade does not bind
    statically is bound to its unique static definition below the facade: inside
    the facade package, or in the private sibling modules of a module facade. A
    facade that re-exports a definition from elsewhere binds the unique
    definition in the package. A name with several or no candidate definitions
    stays unbound, so collecting it fails instead of documenting a guessed
    object. Tuple-unpacked module constants become module attributes, and a
    module attribute assigned a name that resolves within Phydrax becomes an
    alias of that object, which it is at runtime.
    """

    def on_module_members(
        self, *, node: ast.AST | griffe.ObjectNode, mod: griffe.Module, **kwargs: Any
    ) -> None:
        if not isinstance(node, ast.Module):
            return
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if not isinstance(target, ast.Tuple):
                    continue
                for element in target.elts:
                    if isinstance(element, ast.Name) and element.id not in mod.members:
                        mod.set_member(
                            element.id,
                            griffe.Attribute(
                                element.id,
                                lineno=statement.lineno,
                                endlineno=statement.end_lineno,
                                parent=mod,
                            ),
                        )

    def on_package(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        if pkg.name != "phydrax":
            return
        manifest = json.loads(_PUBLIC_API_MANIFEST.read_text(encoding="utf-8"))
        # Index definitions first: a `Name = Other` attribute is the definition a
        # facade binds, and it remains so after it becomes an alias of `Other`.
        definitions = _static_definitions(pkg)
        _bind_name_assignments(pkg, loader.modules_collection)
        for module_path, exported in manifest["modules"].items():
            member = loader.modules_collection.get_member(module_path)
            # A public module may be an alias of a private implementation module.
            module = member.final_target if isinstance(member, griffe.Alias) else member
            if not isinstance(module, griffe.Module):
                raise TypeError(f"public API module {module_path} is not a module")
            names = [qualified_name.rpartition(".")[2] for qualified_name in exported]
            module.exports = list(names)
            for name in names:
                if name in module.members:
                    continue
                owners = _export_owners(module, definitions.get(name, ()))
                if len(owners) == 1:
                    module.set_member(
                        name, griffe.Alias(name, f"{owners[0]}.{name}", parent=module)
                    )


def _bind_name_assignments(
    package: griffe.Module, collection: griffe.ModulesCollection, /
) -> None:
    """Turn ``Name = Other`` module attributes into aliases of ``Other``.

    A ``type Name = ...`` statement creates a distinct alias object and stays an
    attribute. A name that does not resolve within the loaded modules stays an
    attribute as well.
    """
    modules = [package]
    while modules:
        module = modules.pop()
        assignments: list[tuple[str, str]] = []
        for name, member in module.members.items():
            if isinstance(member, griffe.Module):
                modules.append(member)
            elif (
                isinstance(member, griffe.Attribute)
                and isinstance(member.value, griffe.ExprName)
                and "type-alias" not in member.labels
            ):
                target = member.value.canonical_path
                if (
                    target != member.path
                    and _lookup(collection, target, set()) is not None
                ):
                    assignments.append((name, target))
        for name, target in assignments:
            module.set_member(name, griffe.Alias(name, target, parent=module))


def _lookup(
    collection: griffe.ModulesCollection, path: str, seen: set[str], /
) -> griffe.Object | None:
    """Return the loaded object a dotted path names, following aliases."""
    if path in seen:
        return None
    seen.add(path)
    head, *parts = path.split(".")
    current = collection.members.get(head)
    for index, part in enumerate(parts):
        if current is None:
            return None
        if isinstance(current, griffe.Alias):
            return _lookup(
                collection, ".".join([current.target_path, *parts[index:]]), seen
            )
        current = current.members.get(part)
    if isinstance(current, griffe.Alias):
        return _lookup(collection, current.target_path, seen)
    return current


def _static_definitions(package: griffe.Module, /) -> Mapping[str, Sequence[str]]:
    """Map each name to the modules that define, rather than import, it."""
    definitions: defaultdict[str, list[str]] = defaultdict(list)
    modules = [package]
    while modules:
        module = modules.pop()
        for name, member in module.members.items():
            if isinstance(member, griffe.Module):
                modules.append(member)
            elif not member.is_alias:
                definitions[name].append(module.path)
    return definitions


def _export_owners(facade: griffe.Module, definitions: Sequence[str], /) -> Sequence[str]:
    if facade.is_init_module:
        local = [owner for owner in definitions if owner.startswith(f"{facade.path}.")]
    else:
        parent = facade.path.rpartition(".")[0]
        local = [
            owner
            for owner in definitions
            if owner.startswith(f"{parent}._") and owner != facade.path
        ]
    return local or definitions


class EquinoxConstructors(griffe.Extension):
    """Document the constructors that static analysis cannot see.

    Equinox generates a dataclass ``__init__`` for a module class unless the class
    defines ``__init__``; otherwise it inherits the decision of the first Equinox
    module in its resolution order, and a non-module class defining ``__init__``
    before it supplies the constructor. A generated signature is rebuilt from the
    fields of every Equinox module in the resolution order, base first: instance
    attributes other than abstract variables, with ``field`` defaults, default
    factories, ``init=False`` exclusions, and ``kw_only`` placement.

    A class that inherits a custom constructor declares the checker-only alias
    ``__init__ = Owner.__init__``; the class documents ``Owner.__init__`` through
    that alias. Abstract classes document the constructor their concrete
    subclasses inherit. A class whose Equinox base lies outside Phydrax, such as a
    Diffrax base, is documented from its own source members only.
    """

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        if pkg.name != "phydrax":
            return
        classes = _package_classes(pkg)
        # Decide every constructor from source members before adding any, so a
        # generated base constructor never reads as a custom one.
        constructors = [
            (cls, constructor)
            for cls in classes
            if (constructor := _documented_constructor(cls)) is not None
        ]
        for cls, constructor in constructors:
            cls.set_member("__init__", constructor)


class ClassMemberListing(griffe.Extension):
    """List class members as the public API presents them.

    A class member without a docstring inherits the docstring of the same member
    in the first documenting class of its resolution order, as ``inspect.getdoc``
    does. An undocumented dataclass field, an annotated instance attribute, is
    omitted when the constructor signature in the class heading states it, or
    when it has no class-level value and so is no class attribute at runtime.
    Fields with class-level values that the constructor does not take,
    named-tuple fields, methods, properties, class constants, enum members, and
    documented fields are listed. Runs after `EquinoxConstructors`, which
    supplies the constructors it reads.

    Static analysis merges an ``__init__`` assignment ``self.name = ...`` into the
    class attribute, so a field's class-level value is recorded while the class
    body is visited.
    """

    def __init__(self) -> None:
        super().__init__()
        self._class_values: set[str] = set()

    def on_attribute_instance(
        self,
        *,
        node: ast.AST | griffe.ObjectNode,
        attr: griffe.Attribute,
        agent: griffe.Visitor | griffe.Inspector,
        **kwargs: Any,
    ) -> None:
        if isinstance(agent, griffe.Visitor) and isinstance(agent.current, griffe.Class):
            if _has_class_value(attr):
                self._class_values.add(attr.path)
            else:
                self._class_values.discard(attr.path)

    def on_package(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        if pkg.name != "phydrax":
            return
        classes = _package_classes(pkg)
        for cls in classes:
            for name, member in cls.members.items():
                if member.docstring is None and isinstance(member, griffe.Object):
                    inherited = _inherited_docstring(cls, name)
                    if inherited is not None:
                        member.docstring = griffe.Docstring(
                            inherited.value,
                            lineno=inherited.lineno,
                            endlineno=inherited.endlineno,
                            parent=member,
                            parser=inherited.parser,
                            parser_options=inherited.parser_options,
                        )
        for cls in classes:
            if _derives_from(cls, frozenset({"typing.NamedTuple"})):
                continue
            parameters = _constructor_parameters(cls, loader.modules_collection)
            undocumented_fields = [
                name
                for name, member in cls.members.items()
                if isinstance(member, griffe.Attribute)
                and member.annotation is not None
                and "instance-attribute" in member.labels
                and "property" not in member.labels
                and member.docstring is None
                and (member.path not in self._class_values or name in parameters)
            ]
            for name in undocumented_fields:
                cls.del_member(name)


def _has_class_value(field: griffe.Attribute, /) -> bool:
    """Whether a dataclass field is a class attribute at runtime.

    A ``field(...)`` call leaves a class attribute only for an explicit default;
    a default factory or no default leaves none.
    """
    value = field.value
    if (
        isinstance(value, griffe.ExprCall)
        and value.function.canonical_path in _FIELD_FACTORIES
    ):
        return any(
            isinstance(argument, griffe.ExprKeyword) and argument.name == "default"
            for argument in value.arguments
        )
    return value is not None


def _constructor_parameters(
    cls: griffe.Class, collection: griffe.ModulesCollection, /
) -> frozenset[str]:
    """Parameter names of the constructor the class heading documents."""
    for owner in (cls, *cls.mro()):
        constructor = owner.members.get("__init__")
        if isinstance(constructor, griffe.Alias):
            constructor = _lookup(collection, constructor.target_path, set())
        if isinstance(constructor, griffe.Function):
            return frozenset(parameter.name for parameter in constructor.parameters)
    return frozenset()


def _inherited_docstring(cls: griffe.Class, name: str, /) -> griffe.Docstring | None:
    for base in cls.mro():
        member = base.members.get(name)
        if isinstance(member, griffe.Object) and member.docstring is not None:
            return member.docstring
    return None


def _package_classes(package: griffe.Module, /) -> list[griffe.Class]:
    classes: list[griffe.Class] = []
    modules = [package]
    while modules:
        module = modules.pop()
        for member in module.members.values():
            if isinstance(member, griffe.Module):
                modules.append(member)
            elif isinstance(member, griffe.Class):
                classes.extend(_nested_classes(member))
    return classes


def _derives_from(cls: griffe.Class, bases: frozenset[str], /) -> bool:
    return any(
        isinstance(base, griffe.Expr) and base.canonical_path in bases
        for owner in (cls, *cls.mro())
        for base in owner.bases
    )


def _nested_classes(cls: griffe.Class, /) -> list[griffe.Class]:
    nested = [cls]
    for member in cls.members.values():
        if isinstance(member, griffe.Class):
            nested.extend(_nested_classes(member))
    return nested


def _documented_constructor(
    cls: griffe.Class, /
) -> griffe.Function | griffe.Alias | None:
    own = cls.members.get("__init__")
    if isinstance(own, griffe.Function) or not _derives_from(cls, _EQUINOX_MODULES):
        return None
    if not _has_dataclass_init(cls):
        if isinstance(own, griffe.Attribute) and isinstance(own.value, griffe.Expr):
            return griffe.Alias("__init__", own.value.canonical_path, parent=cls)
        return None
    resolution = [cls, *cls.mro()]
    positional: list[griffe.Parameter] = [griffe.Parameter("self")]
    keyword: list[griffe.Parameter] = []
    for name, (annotation, default, keyword_only) in _dataclass_fields(
        [
            owner
            for owner in reversed(resolution)
            if _derives_from(owner, _EQUINOX_MODULES)
        ]
    ).items():
        parameter = griffe.Parameter(
            name,
            annotation=annotation,
            kind=griffe.ParameterKind.keyword_only
            if keyword_only
            else griffe.ParameterKind.positional_or_keyword,
            default=default,
        )
        (keyword if keyword_only else positional).append(parameter)
    return griffe.Function(
        "__init__",
        parameters=griffe.Parameters(*positional, *keyword),
        returns="None",
        parent=cls,
    )


def _has_dataclass_init(cls: griffe.Class, /) -> bool:
    """Apply Equinox's rule for generating a dataclass ``__init__``."""
    if isinstance(cls.members.get("__init__"), griffe.Function):
        return False
    for owner in cls.mro():
        if _derives_from(owner, _EQUINOX_MODULES):
            return _has_dataclass_init(owner)
        if isinstance(owner.members.get("__init__"), griffe.Function):
            return False
    return True


type _Field = tuple[str | griffe.Expr | None, str | griffe.Expr | None, bool]


def _dataclass_fields(owners: Sequence[griffe.Class], /) -> dict[str, _Field]:
    """Collect dataclass fields in definition order; redefinitions keep position."""
    fields: dict[str, _Field] = {}
    for owner in owners:
        for name, member in owner.members.items():
            # Griffe strips an explicit `ClassVar` and labels it as a class
            # attribute only; every dataclass field is an instance attribute.
            if (
                not isinstance(member, griffe.Attribute)
                or member.annotation is None
                or "instance-attribute" not in member.labels
                or "property" in member.labels
            ):
                continue
            annotation = member.annotation
            # A subscript's canonical path is that of its subscripted name.
            if (
                isinstance(annotation, griffe.Expr)
                and annotation.canonical_path in _ABSTRACT_VARIABLES
            ):
                continue
            value = member.value
            if (
                isinstance(value, griffe.ExprCall)
                and value.function.canonical_path in _FIELD_FACTORIES
            ):
                options = {
                    argument.name: argument.value
                    for argument in value.arguments
                    if isinstance(argument, griffe.ExprKeyword)
                }
                if str(options.get("init", "True")) == "False":
                    fields.pop(name, None)
                    continue
                factory = options.get("default_factory")
                default = options.get("default") if factory is None else f"{factory}()"
                fields[name] = (
                    annotation,
                    default,
                    str(options.get("kw_only", "False")) == "True",
                )
            else:
                fields[name] = (annotation, value, False)
    return fields

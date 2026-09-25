#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import dis
import enum
import functools
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from enum import StrEnum
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax.extend import core as jax_core
from jaxtyping import PyTree

from ._strict import StrictModule


class ArrayRole(StrEnum):
    """Training role of one array leaf."""

    PARAMETER = "parameter"
    FIXED = "fixed"
    MODEL_STATE = "model_state"


class NonTrainableState:
    """Marker for numeric state that must remain fixed during solver training.

    Every array below a `NonTrainableState` node is FIXED. The marker is audited:
    a trainable component (a `ParameterOwner` or a `parameter_field`) below it is a
    role violation unless an `ExplicitFreeze` node in between freezes it on purpose.
    """


class ExplicitFreeze(NonTrainableState):
    """Marker for holders that intentionally freeze embedded trained components.

    Like `NonTrainableState`, everything below is FIXED; unlike it, the subtree is
    exempt from the silent-freeze audit.
    """


class ParameterOwner:
    """Marker for components whose unannotated inexact arrays are parameters.

    Inexact array leaves below a `ParameterOwner` without an explicit field role
    (from this owner or an intermediate container) resolve to PARAMETER.
    """


_ROLE_METADATA_KEY = "phydrax_role"


def _role_field(role: ArrayRole, kwargs: dict[str, Any], /) -> Any:
    if kwargs.get("static", False):
        raise TypeError(
            f"{role.value} fields cannot be static: static fields are PyTree "
            "structure and carry no arrays."
        )
    metadata = dict(kwargs.pop("metadata", None) or {})
    if _ROLE_METADATA_KEY in metadata:
        raise TypeError(
            f"Field metadata key {_ROLE_METADATA_KEY!r} is reserved for array roles."
        )
    metadata[_ROLE_METADATA_KEY] = role
    return eqx.field(metadata=metadata, **kwargs)


def parameter_field(**kwargs: Any) -> Any:
    """Declare a module field whose array subtree is PARAMETER.

    Accepts the keyword arguments of `equinox.field`; `static=True` is a `TypeError`.
    """
    return _role_field(ArrayRole.PARAMETER, kwargs)


def fixed_field(**kwargs: Any) -> Any:
    """Declare a module field whose array subtree is FIXED.

    Accepts the keyword arguments of `equinox.field`; `static=True` is a `TypeError`.
    """
    return _role_field(ArrayRole.FIXED, kwargs)


def model_state_field(**kwargs: Any) -> Any:
    """Declare a module field whose array subtree is MODEL_STATE.

    Accepts the keyword arguments of `equinox.field`; `static=True` is a `TypeError`.
    """
    return _role_field(ArrayRole.MODEL_STATE, kwargs)


_PARAMETER_UNDER_FIXED_ANCESTOR = "parameter-under-fixed-ancestor"
_ROLE_FIELD_ON_TERMINAL_VALUE = "role-field-on-terminal-value"
_UNCLASSIFIED_REMEDY = (
    "declare the array with parameter_field or fixed_field, hold raw Equinox "
    "modules in phydrax.nn.models.EquinoxModel (a ParameterOwner), select it "
    "explicitly with phydrax.nn.parameters.ParameterSubspace, or freeze its holder "
    "intentionally with an ExplicitFreeze holder such as FrozenModel"
)


class RoleResolution(StrictModule):
    """Static per-leaf array roles of one PyTree structure.

    `paths` and `roles` follow `jax.tree_util` leaf order, so filter specs built
    here are exact `eqx.partition` filters for trees with the resolved structure.
    Unclassified inexact leaves have role `None`. Non-array leaves and non-inexact
    arrays outside a MODEL_STATE declaration are FIXED. Each violation is a
    `(path, kind, remedy)` triple.
    """

    treedef: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    roles: tuple[ArrayRole | None, ...] = eqx.field(static=True)
    violations: tuple[tuple[str, str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        treedef: Any,
        paths: tuple[str, ...],
        roles: tuple[ArrayRole | None, ...],
        violations: tuple[tuple[str, str, str], ...],
    ):
        if not isinstance(treedef, jax.tree_util.PyTreeDef):
            raise TypeError("treedef must be a jax.tree_util.PyTreeDef.")
        if not len(paths) == len(roles) == treedef.num_leaves:
            raise ValueError("paths and roles must match the treedef leaf count.")
        if any(role is not None and not isinstance(role, ArrayRole) for role in roles):
            raise TypeError("roles must be ArrayRole or None.")
        self.treedef = treedef
        self.paths = tuple(paths)
        self.roles = tuple(roles)
        self.violations = tuple(violations)

    @property
    def unclassified(self) -> tuple[str, ...]:
        """Paths of inexact array leaves without a role."""
        return tuple(
            path
            for path, role in zip(self.paths, self.roles, strict=True)
            if role is None
        )

    @property
    def selectable_paths(self) -> tuple[str, ...]:
        """Inexact leaves an explicit selection may declare as parameters.

        These are PARAMETER and unclassified leaves: never leaves below a terminal
        node or declared by `fixed_field` or `model_state_field`.
        """
        return tuple(
            path
            for path, role in zip(self.paths, self.roles, strict=True)
            if role is None or role is ArrayRole.PARAMETER
        )

    def role_of(self, path: str, /) -> ArrayRole | None:
        """Return the role of one leaf path (`None` when unclassified)."""
        try:
            index = self.paths.index(path)
        except ValueError:
            raise ValueError(f"Unknown leaf path {path!r}.") from None
        return self.roles[index]

    def filter_spec(self, *roles: ArrayRole) -> PyTree[bool]:
        """Boolean filter spec marking leaves whose role is one of `roles`."""
        if not roles or any(not isinstance(role, ArrayRole) for role in roles):
            raise TypeError("filter_spec requires one or more ArrayRole values.")
        selected = frozenset(roles)
        return jax.tree_util.tree_unflatten(
            self.treedef, [role in selected for role in self.roles]
        )


@functools.cache
def _terminal_types() -> tuple[type, ...]:
    # Imported lazily: phydrax.domain depends on this module.
    from phydrax.domain import Domain

    return (NonTrainableState, Domain)


def _is_terminal(node: Any, /) -> bool:
    return isinstance(node, _terminal_types())


@functools.cache
def _field_roles(cls: type, /) -> Mapping[str, ArrayRole]:
    if not dataclasses.is_dataclass(cls):
        return {}
    return {
        field.name: ArrayRole(field.metadata[_ROLE_METADATA_KEY])
        for field in dataclasses.fields(cls)
        if _ROLE_METADATA_KEY in field.metadata
    }


def _children(node: Any, /) -> list[tuple[tuple[Any, ...], Any]]:
    return jax.tree_util.tree_flatten_with_path(node, is_leaf=lambda x: x is not node)[0]


def _display(path: str, /) -> str:
    return path or "<root>"


class _Scope(NamedTuple):
    """Inherited walk state of one node."""

    role: ArrayRole | None
    owner: bool
    frozen: bool
    audit_anchor: str | None


class _RoleWalk:
    """Mutable accumulator of one path-aware role walk."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.roles: list[ArrayRole | None] = []
        self.violations: list[tuple[str, str, str]] = []

    def visit(
        self,
        node: Any,
        keys: tuple[Any, ...],
        scope: _Scope,
        field_role: ArrayRole | None,
    ) -> None:
        path = jax.tree_util.keystr(keys)
        scope = self._enter(node, path, scope, field_role)
        children = _children(node)
        if len(children) == 1 and children[0][0] == ():
            self.paths.append(path)
            self.roles.append(_leaf_role(node, scope))
            return
        roles = _field_roles(type(node))
        for child_keys, child in children:
            (key,) = child_keys
            child_role = (
                roles.get(key.name) if isinstance(key, jax.tree_util.GetAttrKey) else None
            )
            self.visit(child, keys + child_keys, scope, child_role)

    def _enter(
        self,
        node: Any,
        path: str,
        scope: _Scope,
        field_role: ArrayRole | None,
    ) -> _Scope:
        terminal = _is_terminal(node)
        role, owner, frozen, anchor = scope
        if frozen:
            # Rule 5: audit below a plain NonTrainableState for silent freezes.
            if anchor is not None and (
                field_role is ArrayRole.PARAMETER
                or (isinstance(node, ParameterOwner) and not terminal)
            ):
                self._record_silent_freeze(path, anchor)
                anchor = None
            elif isinstance(node, ExplicitFreeze):
                anchor = None
            return _Scope(role, owner, frozen, anchor)
        if field_role is not None:
            if terminal and field_role is not ArrayRole.FIXED:
                # Rule 3: a PARAMETER/MODEL_STATE declaration cannot unfreeze.
                self.violations.append(
                    (
                        path,
                        _ROLE_FIELD_ON_TERMINAL_VALUE,
                        f"{_display(path)} is declared {field_role.value} but holds "
                        f"{type(node).__name__}, a terminal (always FIXED) node; "
                        "declare the field with fixed_field, or hold a trainable "
                        "component instead of a frozen one.",
                    )
                )
            role = field_role
        if terminal:
            # Rule 1: terminal nodes freeze their whole subtree.
            audited = isinstance(node, NonTrainableState) and not isinstance(
                node, ExplicitFreeze
            )
            return _Scope(role, owner, True, path if audited else None)
        return _Scope(role, owner or isinstance(node, ParameterOwner), False, None)

    def _record_silent_freeze(self, path: str, anchor: str, /) -> None:
        self.violations.append(
            (
                path,
                _PARAMETER_UNDER_FIXED_ANCESTOR,
                f"{_display(path)} is trainable but sits below the NonTrainableState "
                f"node {_display(anchor)}, which would freeze it silently; wrap it in "
                "an ExplicitFreeze holder (for example FrozenModel) to freeze it on "
                "purpose, or remove NonTrainableState from the ancestor and declare "
                "the ancestor's own arrays with fixed_field.",
            )
        )


def _leaf_role(leaf: Any, scope: _Scope, /) -> ArrayRole | None:
    if scope.frozen:
        return ArrayRole.FIXED
    match scope.role:
        case ArrayRole.PARAMETER:
            return ArrayRole.PARAMETER if eqx.is_inexact_array(leaf) else ArrayRole.FIXED
        case ArrayRole.MODEL_STATE:
            return ArrayRole.MODEL_STATE if eqx.is_array(leaf) else ArrayRole.FIXED
        case ArrayRole.FIXED:
            return ArrayRole.FIXED
        case None:
            if not eqx.is_inexact_array(leaf):
                return ArrayRole.FIXED
            return ArrayRole.PARAMETER if scope.owner else None
        case _:
            raise ValueError(f"Unknown array role {scope.role!r}.")


def _leaf_signature(leaf: Any, /) -> int:
    # Everything a leaf contributes to its own resolution; node types, static
    # fields and field metadata are already fixed by the treedef.
    return (
        _is_terminal(leaf)
        | isinstance(leaf, ParameterOwner) << 1
        | eqx.is_inexact_array(leaf) << 2
        | eqx.is_array(leaf) << 3
    )


_RESOLUTION_CACHE: OrderedDict[Any, RoleResolution] = OrderedDict()
_RESOLUTION_CACHE_SIZE = 512


def resolve_array_roles(tree: PyTree[Any], /) -> RoleResolution:
    """Resolve the array role of every leaf of `tree` without raising.

    Precedence, top-down:

    1. `Domain`, `NonTrainableState` and `ExplicitFreeze` nodes are terminal:
       every leaf below is FIXED.
    2. An explicit field role (`parameter_field`, `fixed_field`,
       `model_state_field`) applies to its value's subtree; the nearest
       declaration wins.
    3. A `parameter_field` or `model_state_field` whose value is a terminal node
       is a `role-field-on-terminal-value` violation (the value stays FIXED).
    4. Outside terminal ancestors, containers inherit their parent field's role
       and unannotated inexact leaves below a `ParameterOwner` are PARAMETER.
    5. Below a plain `NonTrainableState`, a non-terminal `ParameterOwner` or a
       `parameter_field` is a `parameter-under-fixed-ancestor` violation; an
       `ExplicitFreeze` node ends the audit for its subtree and a nested plain
       `NonTrainableState` continues it.
    6. Remaining inexact leaves are unclassified (role `None`).

    The walk is authority-free and cached per treedef and leaf kind.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    key = (treedef, tuple(_leaf_signature(leaf) for leaf in leaves))
    try:
        hash(key)
    except TypeError:
        # Unhashable static structure: resolve without caching.
        key = None
    if key is not None and key in _RESOLUTION_CACHE:
        _RESOLUTION_CACHE.move_to_end(key)
        return _RESOLUTION_CACHE[key]
    walk = _RoleWalk()
    walk.visit(tree, (), _Scope(None, False, False, None), None)
    resolution = RoleResolution(
        treedef, tuple(walk.paths), tuple(walk.roles), tuple(walk.violations)
    )
    if key is not None:
        _RESOLUTION_CACHE[key] = resolution
        if len(_RESOLUTION_CACHE) > _RESOLUTION_CACHE_SIZE:
            _RESOLUTION_CACHE.popitem(last=False)
    return resolution


def _declared_roles(tree: PyTree[Any], context: str, /) -> RoleResolution:
    """Resolve roles and raise `ValueError` on violations or unclassified leaves."""
    if not isinstance(context, str) or not context:
        raise TypeError("context must be a non-empty string.")
    resolution = resolve_array_roles(tree)
    unclassified = resolution.unclassified
    if not resolution.violations and not unclassified:
        return resolution
    lines = [f"{context}: array roles are not fully declared."]
    if resolution.violations:
        lines.append("Role violations:")
        lines.extend(
            f"  - {_display(path)} [{kind}]: {remedy}"
            for path, kind, remedy in resolution.violations
        )
    if unclassified:
        lines.append(f"Unclassified inexact array leaves ({_UNCLASSIFIED_REMEDY}):")
        lines.extend(f"  - {_display(path)}" for path in unclassified)
    raise ValueError("\n".join(lines))


def require_parameter_roles(tree: PyTree[Any], /, *, context: str) -> RoleResolution:
    """Training-boundary preflight of array roles and callable categories.

    Raises `ValueError` on role violations, unclassified inexact leaves, and
    callables or static fields that hide inexact arrays
    (`require_declared_callables`).
    """
    resolution = _declared_roles(tree, context)
    require_declared_callables(tree, context=context)
    return resolution


_HIDDEN_STATE_REMEDY = (
    "make the array a visible module field with a declared role, pass it as an "
    "explicit argument, or freeze its provider on purpose with an ExplicitFreeze "
    "holder such as FrozenModel"
)
# NumPy scalars are immutable host constants, like the Python numbers beside them.
_HIDDEN_STATE_ATOMS = (
    type,
    ModuleType,
    str,
    bytes,
    int,
    float,
    complex,
    enum.Enum,
    np.generic,
)
_GLOBAL_READ_OPCODES = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})


@functools.lru_cache(maxsize=4096)
def _code_global_names(code: CodeType, /) -> frozenset[str]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in _GLOBAL_READ_OPCODES
    }
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names |= _code_global_names(constant)
    return frozenset(names)


def _global_reads(function: FunctionType, /) -> tuple[tuple[str, Any], ...]:
    """Return the `(name, value)` module globals `function` reads, sorted by name.

    Nested code (lambdas, inner functions, comprehensions) shares the function's
    globals and is included. Names absent from the module namespace resolve to
    builtins and are skipped. Functions imported from another module are
    operations named by their import, not state of this module: their own
    module state belongs to their defining module and is not followed.
    """
    namespace = function.__globals__
    return tuple(
        (name, namespace[name])
        for name in sorted(_code_global_names(function.__code__))
        if name in namespace
        and not (
            isinstance(namespace[name], FunctionType)
            and namespace[name].__module__ != function.__module__
        )
    )


def _hidden_state(value: Any, /) -> Iterator[tuple[str, Any]]:
    """Yield the `(label, value)` state an opaque PyTree leaf hides from flattening."""
    if isinstance(value, FunctionType):
        for name, cell in zip(
            value.__code__.co_freevars, value.__closure__ or (), strict=True
        ):
            try:
                contents = cell.cell_contents
            except ValueError:  # An unfilled cell holds nothing yet.
                continue
            yield f"closure variable {name!r}", contents
        for index, default in enumerate(value.__defaults__ or ()):
            yield f"default argument {index}", default
        for name, default in (value.__kwdefaults__ or {}).items():
            yield f"keyword default {name!r}", default
        for name, referenced in _global_reads(value):
            yield f"global {name!r}", referenced
    elif isinstance(value, functools.partial):
        yield "partial function", value.func
        for index, argument in enumerate(value.args):
            yield f"partial argument {index}", argument
        for name, argument in value.keywords.items():
            yield f"partial keyword {name!r}", argument
    elif isinstance(value, MethodType):
        yield "bound instance", value.__self__
        yield "method function", value.__func__
    elif hasattr(value, "__wrapped__"):
        yield "wrapped callable", value.__wrapped__
    else:
        yield from _instance_attributes(value)


def _declared_slot_names(cls: type, /) -> Iterator[str]:
    """Yield the attribute names of every `__slots__` entry along the MRO."""
    for owner in cls.__mro__:
        slots = owner.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{owner.__name__.lstrip('_')}{name}"
            yield name


def _instance_attributes(value: Any, /) -> Iterator[tuple[str, Any]]:
    """Yield instance state: dataclass fields, declared slots, then `__dict__`."""
    seen: set[str] = set()
    names: list[str] = []
    if dataclasses.is_dataclass(value):
        names.extend(field.name for field in dataclasses.fields(value))
    names.extend(_declared_slot_names(type(value)))
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            attribute = getattr(value, name)
        except AttributeError:  # A declared slot that was never assigned.
            continue
        yield f"attribute {name!r}", attribute
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for name, attribute in attributes.items():
            if name not in seen:
                yield f"attribute {name!r}", attribute


class _HiddenStateWalk:
    """Mutable accumulator of arrays hidden from PyTree flattening."""

    def __init__(self, selected: Callable[[Any], bool], /) -> None:
        self.selected = selected
        self.found: list[tuple[str, tuple[str, ...]]] = []
        self._seen: set[tuple[int, bool]] = set()

    def visit(
        self, node: Any, path: str, trail: tuple[str, ...], hidden: bool, /
    ) -> None:
        if node is None or isinstance(node, _HIDDEN_STATE_ATOMS):
            return
        if eqx.is_array(node):
            if hidden and self.selected(node):
                self.found.append((path, trail))
            return
        if isinstance(node, jax_core.Literal) and not node.aval.shape:
            # Scalar jaxpr literals are code constants, like bytecode constants;
            # the arrays a closed jaxpr captures live in its consts.
            return
        if not hidden and isinstance(node, ExplicitFreeze):
            # A visible ExplicitFreeze holder freezes its whole subtree on
            # purpose. A plain NonTrainableState only declares its visible leaves
            # FIXED: the state its callables hide is still searched.
            return
        marker = (id(node), hidden)
        if marker in self._seen:
            return
        self._seen.add(marker)
        if hidden and isinstance(node, dict):
            # Hidden mappings (module registries, caches) may key by values that
            # PyTree flattening cannot sort; their entries are searched directly.
            for key, value in node.items():
                self.visit(value, path, (*trail, f"[{key!r}]"), True)
            return
        if dataclasses.is_dataclass(node):
            for field in dataclasses.fields(node):
                if field.metadata.get("static", False):
                    # Static fields are treedef structure: their arrays never
                    # reach a role lane.
                    value = getattr(node, field.name)
                    if hidden:
                        self.visit(value, path, (*trail, f".{field.name}"), True)
                    else:
                        self.visit(value, f"{path}.{field.name}", ("static field",), True)
        children = _children(node)
        if len(children) == 1 and children[0][0] == ():
            for label, value in _hidden_state(node):
                self.visit(value, path, (*trail, label), True)
            return
        for keys, child in children:
            key = jax.tree_util.keystr(keys)
            if hidden:
                self.visit(child, path, (*trail, key), True)
            else:
                self.visit(child, path + key, trail, False)


def _hidden_arrays(
    tree: PyTree[Any], selected: Callable[[Any], bool], /
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return `(path, capture trail)` of every `selected` array hidden in `tree`."""
    walk = _HiddenStateWalk(selected)
    walk.visit(tree, "", (), False)
    return tuple(walk.found)


def require_declared_callables(tree: PyTree[Any], /, *, context: str) -> None:
    """Raise `ValueError` when callables in `tree` hide inexact numeric state.

    A callable reachable at a training or artifact boundary must be one of:

    1. a stateless operation whose closure cells, defaults, bound arguments and
       read module globals capture no inexact array (plain module-level
       functions over modules, classes and Python or NumPy scalar constants are
       the canonical case, see `phydrax._identity`);
    2. a visible component: a callable module held in a dynamic field, whose
       arrays are PyTree leaves with declared roles;
    3. an explicit frozen provider below a visible `ExplicitFreeze` holder, the
       only node that authorizes hidden numeric state in its subtree;
    4. a host-only callback that captures no inexact array.

    Everything a PyTree flattening cannot see is searched, below plain
    `NonTrainableState` and `Domain` nodes too: closure cells, defaults, module
    globals a function's code reads (transitively), `functools.partial`
    arguments, bound instances, wrapped callables, attributes of opaque leaves
    (dataclass fields, declared `__slots__` and `__dict__`, so slotted frozen
    dataclasses are covered) and static fields. Each inexact array found there is
    reported with its path and capture route.
    """
    if not isinstance(context, str) or not context:
        raise TypeError("context must be a non-empty string.")
    found = _hidden_arrays(tree, eqx.is_inexact_array)
    if not found:
        return
    lines = [
        f"{context}: callables or static fields hide inexact arrays that are not "
        f"visible PyTree leaves ({_HIDDEN_STATE_REMEDY}):"
    ]
    lines.extend(f"  - {_display(path)}: {' -> '.join(trail)}" for path, trail in found)
    raise ValueError("\n".join(lines))


def _lane_tree(
    tree: PyTree[Any],
    keep: Iterable[bool],
    /,
    *,
    terminals: bool,
) -> PyTree[Any]:
    """Keep the marked leaves of `tree` and put `None` holes elsewhere.

    Terminal nodes are all-FIXED, so a lane holds each one whole (`terminals`) or
    as a single `None`; no lane contains a hollowed terminal node. Callers must
    never keep a leaf below a terminal node without keeping all of them.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    kept = jax.tree_util.tree_unflatten(
        treedef,
        [leaf if marked else None for leaf, marked in zip(leaves, keep, strict=True)],
    )
    return jax.tree_util.tree_map(
        lambda node, part: (node if terminals else None) if _is_terminal(node) else part,
        tree,
        kept,
        is_leaf=_is_terminal,
    )


def partition_parameters(
    tree: PyTree[Any], /
) -> tuple[PyTree[Any], PyTree[Any], PyTree[Any]]:
    """Split `tree` into `(parameters, model_state, fixed)` role lanes.

    It applies the role check of `require_parameter_roles` (cached per treedef,
    so it stays cheap inside training loops): role violations and unclassified
    inexact leaves raise `ValueError`. Training entries run the full preflight,
    including callable categories, once. The lanes follow `eqx.partition`
    conventions with `None` holes; terminal nodes appear whole in `fixed` and as
    `None` in the other two lanes. `combine_parameters` inverts it.
    """
    roles = _declared_roles(tree, "partition_parameters").roles
    return tuple(
        _lane_tree(
            tree,
            (role is lane for role in roles),
            terminals=lane is ArrayRole.FIXED,
        )
        for lane in (ArrayRole.PARAMETER, ArrayRole.MODEL_STATE, ArrayRole.FIXED)
    )


def combine_parameters(
    parameters: PyTree[Any],
    model_state: PyTree[Any],
    fixed: PyTree[Any],
    /,
) -> PyTree[Any]:
    """Recombine the role lanes produced by `partition_parameters`."""
    return eqx.combine(parameters, model_state, fixed)


LaneKind: TypeAlias = Literal["item", "case", "member"]


class LaneLayout(StrictModule):
    """Declared leading lane axis (axis 0) of selected leaves, independent of roles.

    A lane is a batch of items, cases or ensemble members stacked on axis 0 of the
    declared leaves; every other leaf is shared across the lane. FIXED data may be
    lane-mapped (for example per-member normalizers). `mapped_paths` is a set:
    it is stored sorted, so equal declarations have one identity whatever order
    the caller lists them in.
    """

    kind: LaneKind = eqx.field(static=True)
    mapped_paths: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, kind: LaneKind, mapped_paths: Iterable[str]):
        match kind:
            case "item" | "case" | "member":
                pass
            case _:
                raise ValueError(
                    f"LaneLayout kind must be 'item', 'case' or 'member'; got {kind!r}."
                )
        paths = tuple(mapped_paths)
        if any(not isinstance(path, str) for path in paths):
            raise TypeError("mapped_paths must be PyTree path strings.")
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("mapped_paths must contain distinct PyTree paths.")
        self.kind = kind
        self.mapped_paths = tuple(sorted(paths))

    @classmethod
    def from_predicate(
        cls,
        tree: PyTree[Any],
        predicate: Callable[[str], bool],
        /,
        *,
        kind: LaneKind,
    ) -> LaneLayout:
        """Map every array leaf of `tree` whose path satisfies `predicate`."""
        return cls(
            kind,
            (
                jax.tree_util.keystr(path)
                for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
                if eqx.is_array(leaf) and predicate(jax.tree_util.keystr(path))
            ),
        )

    def _mapped_leaves(
        self, tree: PyTree[Any], /
    ) -> tuple[list[Any], list[bool], Any, int]:
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
        leaves = [leaf for _, leaf in path_leaves]
        declared = frozenset(self.mapped_paths)
        mapped: list[bool] = []
        sizes: dict[str, int] = {}
        for keys, leaf in path_leaves:
            path = jax.tree_util.keystr(keys)
            is_mapped = path in declared
            if is_mapped:
                if not eqx.is_array(leaf) or leaf.ndim == 0:
                    raise ValueError(
                        f"{self.kind} lane leaf {path} must be an array with a "
                        "leading lane axis."
                    )
                sizes[path] = leaf.shape[0]
            mapped.append(is_mapped)
        missing = tuple(path for path in self.mapped_paths if path not in sizes)
        if missing:
            raise ValueError(f"Unknown {self.kind} lane leaf paths: {missing!r}.")
        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"{self.kind} lane leaves must share one lane size; got {sizes!r}."
            )
        return leaves, mapped, treedef, next(iter(sizes.values()))

    def lane_size(self, tree: PyTree[Any], /) -> int:
        """Validated common lane size of the mapped leaves of `tree`."""
        return self._mapped_leaves(tree)[3]

    def in_axes(self, tree: PyTree[Any], /) -> PyTree[int | None]:
        """`filter_vmap` in_axes for `tree`: 0 on mapped leaves, `None` elsewhere."""
        _, mapped, treedef, _ = self._mapped_leaves(tree)
        return jax.tree_util.tree_unflatten(
            treedef, [0 if is_mapped else None for is_mapped in mapped]
        )

    def take(self, tree: PyTree[Any], index: Any, /) -> PyTree[Any]:
        """Index the lane axis of the mapped leaves; shared leaves stay whole.

        `index` is any axis-0 array index: an integer selects one lane (dropping
        the lane axis), a slice or an integer array gathers lanes.
        """
        leaves, mapped, treedef, _ = self._mapped_leaves(tree)
        return jax.tree_util.tree_unflatten(
            treedef,
            [
                leaf[index] if is_mapped else leaf
                for leaf, is_mapped in zip(leaves, mapped, strict=True)
            ],
        )


__all__ = [
    "ArrayRole",
    "ExplicitFreeze",
    "LaneLayout",
    "NonTrainableState",
    "ParameterOwner",
    "RoleResolution",
    "combine_parameters",
    "fixed_field",
    "model_state_field",
    "parameter_field",
    "partition_parameters",
    "require_declared_callables",
    "require_parameter_roles",
    "resolve_array_roles",
]

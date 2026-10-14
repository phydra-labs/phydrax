#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed configuration resolution, literal overrides, and canonical records.

A configuration definition is a frozen dataclass that declares its canonical
kind on the class itself, ``__config_kind__: ClassVar[str]``. Resolution layers
inputs in increasing precedence: field defaults, declared environment bindings
(``__config_environment__``: field path to variable name), an explicit Python
value layer, and dotted literal overrides such as ``solver.tolerance=1e-8``.

A field annotated exactly as a nested record is addressable field by field;
every other field, including optional records, is replaced as a whole.
Override and environment text is data, never code: Python literals are
recognized syntactically and anything else is the verbatim text, which only a
text-accepting field admits.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import math
import os
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import (
    cast,
    ClassVar,
    Final,
    Generic,
    get_args,
    get_origin,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
)

from ._fingerprint import canonical_fingerprint, canonical_json
from ._frozendict import frozendict
from ._validation import canonical_identifier
from .typing import parse


ConfigurationLayer: TypeAlias = Literal["default", "environment", "value", "override"]


class ConfigurationDefinition(Protocol):
    """A frozen dataclass declaring its canonical configuration kind."""

    __config_kind__: ClassVar[str]


_C = TypeVar("_C", bound=ConfigurationDefinition)
_U = TypeVar("_U", bound=ConfigurationDefinition)

# Declaration order of the layer selector is its precedence order.
_PRECEDENCE: Final = {
    layer: rank for rank, layer in enumerate(get_args(ConfigurationLayer))
}
_MAXIMUM_TEXT: Final = 65536
_MAXIMUM_DEPTH: Final = 16
_MAXIMUM_ITEMS: Final = 4096
_RECORD_FIELDS: Final = frozenset({"configuration_id", "kind", "value", "origins"})

type _Path = tuple[str, ...]


class _NotLiteral:
    """Marker for text that is not a Python literal."""


_NOT_LITERAL: Final = _NotLiteral()


@dataclass(frozen=True, slots=True)
class _Scalar:
    kind: type[bool] | type[int] | type[float] | type[str]


@dataclass(frozen=True, slots=True)
class _Literal:
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _Enum:
    members: type[enum.Enum]


@dataclass(frozen=True, slots=True)
class _None:
    pass


@dataclass(frozen=True, slots=True)
class _Union:
    alternatives: tuple[_Form, ...]


@dataclass(frozen=True, slots=True)
class _Tuple:
    """A fixed tuple when `items` is given, otherwise a variadic tuple of `item`."""

    items: tuple[_Form, ...] | None
    item: _Form | None


@dataclass(frozen=True, slots=True)
class _Mapping:
    key: _Form
    value: _Form


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    form: _Form
    default: object
    factory: Callable[[], object] | None


@dataclass(frozen=True, slots=True)
class _Record:
    cls: type
    fields: tuple[_Field, ...]


type _Form = _Scalar | _Literal | _Enum | _None | _Union | _Tuple | _Mapping | _Record


@dataclass(frozen=True, slots=True)
class _Assignment:
    """One input at a path; `text` is set for unparsed environment/override text."""

    layer: ConfigurationLayer
    value: object
    text: str | None = None
    environment_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldOrigin:
    """The input layer and contributing environment keys for one terminal field."""

    path: str
    layer: ConfigurationLayer
    environment_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not all(
            part.isidentifier() for part in self.path.split(".")
        ):
            raise ValueError("Field origin paths are dotted identifiers.")
        parse(self.layer, ConfigurationLayer, "layer")
        if type(self.environment_keys) is not tuple:
            raise TypeError("Field origin environment keys must be a tuple.")
        for key in self.environment_keys:
            canonical_identifier(key, "environment key")
        if self.environment_keys != tuple(sorted(set(self.environment_keys))):
            raise ValueError("Field origin environment keys must be sorted and unique.")
        if bool(self.environment_keys) != (self.layer == "environment"):
            raise ValueError("Only environment origins require environment keys.")

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "layer": self.layer,
            "environment_keys": list(self.environment_keys),
        }

    @classmethod
    def from_record(cls, record: object, /) -> FieldOrigin:
        if not isinstance(record, Mapping):
            raise TypeError("A field origin record must be a mapping.")
        if set(record) != {"path", "layer", "environment_keys"}:
            raise ValueError("A field origin record must use its canonical fields.")
        path, layer, keys = record["path"], record["layer"], record["environment_keys"]
        if not isinstance(path, str) or type(keys) is not list:
            raise TypeError("Field origin path must be text and keys must be a list.")
        return cls(path, parse(layer, ConfigurationLayer, "layer"), tuple(keys))


@dataclass(frozen=True, slots=True)
class ResolvedConfig(Generic[_C]):
    """A resolved configuration value, its content ID, and its field origins.

    The ID covers the configuration kind and the canonical value; origins are
    provenance, so equal values from different input layers share one ID.
    """

    value: _C
    kind: str
    configuration_id: str
    origins: tuple[FieldOrigin, ...]

    def to_record(self) -> dict[str, object]:
        schema = _record(type(self.value), type(self.value).__qualname__, frozenset())
        if configuration_kind(type(self.value)) != self.kind:
            raise ValueError("Configuration kind does not match its value.")
        encoded = _encode(schema, self.value, "value")
        if configuration_id(self.kind, encoded) != self.configuration_id:
            raise ValueError("Configuration ID does not match its content.")
        _validate_origins(schema, self.value, self.origins)
        return {
            "configuration_id": self.configuration_id,
            "kind": self.kind,
            "value": encoded,
            "origins": [origin.to_record() for origin in self.origins],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @classmethod
    def from_record(
        cls, config_type: type[_U], record: Mapping[str, object], /
    ) -> ResolvedConfig[_U]:
        """Restore a record of `config_type`, verifying completeness and ID."""
        if not isinstance(record, Mapping):
            raise TypeError("A configuration record must be a mapping.")
        missing = _RECORD_FIELDS - set(record)
        unknown = set(record) - _RECORD_FIELDS
        if missing or unknown:
            raise ValueError(
                "Configuration records use their canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        kind = configuration_kind(config_type)
        if record["kind"] != kind:
            raise ValueError(
                f"Configuration record kind {record['kind']!r} is not {kind!r}."
            )
        schema = _record(config_type, config_type.__qualname__, frozenset())
        value = _decode(schema, record["value"], "value")
        encoded = _encode(schema, value, "value")
        if canonical_json(encoded) != canonical_json(record["value"]):
            raise ValueError("Configuration record value is not canonical.")
        identifier = configuration_id(kind, encoded)
        if identifier != record["configuration_id"]:
            raise ValueError("Configuration record ID does not match its content.")
        origins = record["origins"]
        if type(origins) is not list:
            raise TypeError("Configuration record origins must be a list.")
        restored_origins = tuple(FieldOrigin.from_record(origin) for origin in origins)
        _validate_origins(schema, value, restored_origins)
        if not isinstance(value, config_type) or type(value) is not config_type:
            raise TypeError("Configuration constructor returned another type.")
        return ResolvedConfig(value, kind, identifier, restored_origins)


def configuration_kind(config_type: type, /) -> str:
    """The canonical kind declared on `config_type` itself, never inherited."""
    kind = vars(config_type).get("__config_kind__")
    if kind is None:
        raise TypeError(
            f"{config_type.__qualname__} must declare __config_kind__: ClassVar[str]."
        )
    return canonical_identifier(kind, "__config_kind__")


def _compile(annotation: object, owner: str, active: frozenset[type], /) -> _Form:
    if annotation is None or annotation is types.NoneType:
        return _None()
    match annotation:
        case type() if annotation is bool:
            return _Scalar(bool)
        case type() if annotation is int:
            return _Scalar(int)
        case type() if annotation is float:
            return _Scalar(float)
        case type() if annotation is str:
            return _Scalar(str)
        case typing.TypeAliasType():
            return _compile(annotation.__value__, owner, active)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if not all(type(value) in (bool, int, str) for value in arguments):
            raise TypeError(f"{owner}: Literal members must be bool, int, or str.")
        return _Literal(arguments)
    if origin is Union or origin is types.UnionType:
        return _compile_union(arguments, owner, active)
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return _Tuple(None, _compile(arguments[0], owner, active))
        return _Tuple(tuple(_compile(item, owner, active) for item in arguments), None)
    if origin is Mapping or origin is frozendict:
        key = _compile(arguments[0], owner, active)
        if not _text_keys(key):
            raise TypeError(f"{owner}: configuration mapping keys must be text.")
        return _Mapping(key, _compile(arguments[1], owner, active))
    if isinstance(annotation, type):
        return _compile_nominal(annotation, owner, active)
    raise TypeError(f"{owner}: unsupported configuration annotation {annotation!r}.")


def _compile_nominal(
    annotation: type, owner: str, active: frozenset[type], /
) -> _Record | _Enum:
    if issubclass(annotation, enum.Enum):
        for member in annotation.__members__.values():
            _enum_value(member.value)
        return _Enum(annotation)
    if dataclasses.is_dataclass(annotation):
        return _record(annotation, owner, active)
    raise TypeError(f"{owner}: unsupported configuration annotation {annotation!r}.")


def _compile_union(
    arguments: tuple[object, ...], owner: str, active: frozenset[type], /
) -> _Union:
    alternatives = tuple(_compile(item, owner, active) for item in arguments)
    cases = tuple(_case(item) for item in alternatives)
    if len(set(cases)) != len(cases):
        raise ValueError(f"{owner}: union alternatives require distinct canonical kinds.")
    return _Union(alternatives)


def _text_keys(form: _Form, /) -> bool:
    match form:
        case _Scalar(kind=kind):
            return kind is str
        case _Literal(values=values):
            return all(type(value) is str for value in values)
        case _:
            return False


def _record(cls: type, owner: str, active: frozenset[type], /) -> _Record:
    if cls in active:
        raise TypeError(f"{owner}: configuration records cannot contain themselves.")
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{owner}: {cls.__qualname__} is not a dataclass.")
    if not object.__getattribute__(cls, "__dataclass_params__").frozen:
        raise TypeError(
            f"{owner}: configuration record {cls.__qualname__} must be frozen."
        )
    hints = typing.get_type_hints(cls)
    fields: list[_Field] = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        if field.name.startswith("_"):
            raise TypeError(f"{cls.__qualname__}.{field.name}: fields must be public.")
        form = _compile(
            hints[field.name], f"{cls.__qualname__}.{field.name}", active | {cls}
        )
        factory = (
            None
            if field.default_factory is dataclasses.MISSING
            else field.default_factory
        )
        fields.append(_Field(field.name, form, field.default, factory))
    return _Record(cls, tuple(fields))


def _dotted(path: _Path, /) -> str:
    return ".".join(path) if path else "<root>"


def _locate(root: _Record, path: _Path, /) -> _Form:
    form: _Form = root
    for depth, name in enumerate(path):
        if not isinstance(form, _Record):
            raise ValueError(
                f"{_dotted(path[:depth])} is replaced as a whole; "
                f"{_dotted(path)} is not addressable."
            )
        field = next((item for item in form.fields if item.name == name), None)
        if field is None:
            raise ValueError(
                f"Unknown configuration field {_dotted(path[: depth + 1])!r}."
            )
        form = field.form
    return form


def _record_items(form: _Record, value: object, path: _Path, /) -> dict[str, object]:
    names = tuple(field.name for field in form.fields)
    if type(value) is form.cls:
        return {name: object.__getattribute__(value, name) for name in names}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{_dotted(path)} must be a {form.cls.__qualname__} or a mapping of its fields."
        )
    unknown = set(value) - set(names)
    if unknown:
        raise ValueError(
            f"Unknown configuration fields under {_dotted(path)}: {sorted(map(str, unknown))}."
        )
    return {str(name): item for name, item in value.items()}


def _place(
    root: _Record,
    path: _Path,
    assignment: _Assignment,
    assignments: dict[_Path, _Assignment],
    /,
) -> None:
    form = _locate(root, path)
    if not isinstance(form, _Record):
        assignments[path] = assignment
        return
    value = assignment.value
    if assignment.text is not None:
        value = _literal(assignment.text, _dotted(path))
        if value is _NOT_LITERAL:
            raise ValueError(f"{_dotted(path)} requires a mapping literal of its fields.")
    for name, item in _record_items(form, value, path).items():
        _place(
            root,
            (*path, name),
            _Assignment(assignment.layer, item, None, assignment.environment_keys),
            assignments,
        )


def _place_defaults(
    root: _Record, record: _Record, path: _Path, assignments: dict[_Path, _Assignment], /
) -> None:
    for field in record.fields:
        child = (*path, field.name)
        targets = (
            _terminal_fields(field.form, child)
            if isinstance(field.form, _Record)
            else {child: False}
        )
        if all(target in assignments for target in targets):
            continue
        defaults: dict[_Path, _Assignment] = {}
        if field.factory is not None:
            _place(root, child, _Assignment("default", field.factory()), defaults)
        elif field.default is not dataclasses.MISSING:
            _place(root, child, _Assignment("default", field.default), defaults)
        elif isinstance(field.form, _Record):
            _place_defaults(root, field.form, child, assignments)
        for target, assignment in defaults.items():
            assignments.setdefault(target, assignment)


def _place_environment(
    config_type: type,
    root: _Record,
    environment: Mapping[str, str],
    assignments: dict[_Path, _Assignment],
    /,
) -> None:
    from . import _config_environment

    pending: dict[_Path, _Assignment] = {}

    def visit(record: _Record, prefix: _Path, /) -> None:
        if _config_environment.is_native_record(record.cls):
            for item in _config_environment.native_environment_inputs(
                record.cls, environment
            ):
                target = prefix + item.path
                if target in pending:
                    raise ValueError(
                        f"{_dotted(target)}: overlapping environment targets."
                    )
                pending[target] = _Assignment(
                    "environment", item.value, None, item.environment_keys
                )
        for field in record.fields:
            if isinstance(field.form, _Record):
                visit(field.form, (*prefix, field.name))

    visit(root, ())
    bindings = vars(config_type).get("__config_environment__", {})
    if not isinstance(bindings, Mapping):
        raise TypeError("__config_environment__ must map field paths to variable names.")
    for dotted, key in sorted(bindings.items()):
        path = _override_path(dotted)
        if isinstance(_locate(root, path), _Record):
            raise ValueError(
                f"Environment binding {dotted!r} must name a field, not a record."
            )
        canonical_identifier(key, "environment variable")
        if key in environment:
            text = environment[key]
            if path in pending:
                raise ValueError(f"{dotted}: overlapping environment targets.")
            pending[path] = _Assignment("environment", None, text, (key,))
    for path, assignment in pending.items():
        _place(root, path, assignment, assignments)


def _override_path(dotted: object, /) -> _Path:
    if not isinstance(dotted, str):
        raise TypeError("Configuration paths must be text.")
    path = tuple(dotted.strip().split("."))
    if not all(part.isidentifier() for part in path):
        raise ValueError(f"Configuration path {dotted!r} must be dotted identifiers.")
    return path


def _overrides(overrides: Sequence[str], /) -> tuple[tuple[_Path, str], ...]:
    if isinstance(overrides, str) or not isinstance(overrides, Sequence):
        raise TypeError("overrides must be a sequence of 'path=value' strings.")
    if len(overrides) > _MAXIMUM_ITEMS:
        raise ValueError(f"At most {_MAXIMUM_ITEMS} overrides are accepted.")
    parsed: list[tuple[_Path, str]] = []
    for item in overrides:
        if not isinstance(item, str):
            raise TypeError("overrides must be a sequence of 'path=value' strings.")
        name, separator, text = item.partition("=")
        if not separator:
            raise ValueError(f"Override {item!r} must have the form path=value.")
        parsed.append((_override_path(name), text))
    for index, (first, _) in enumerate(parsed):
        for second, _ in parsed[index + 1 :]:
            shorter = min(len(first), len(second))
            if first[:shorter] == second[:shorter]:
                raise ValueError(
                    f"Overrides {_dotted(first)!r} and {_dotted(second)!r} overlap."
                )
    return tuple(parsed)


def _literal(text: str, path: str, /) -> object:
    """Return the Python literal spelled by `text`, or `_NOT_LITERAL`."""
    if len(text) > _MAXIMUM_TEXT:
        raise ValueError(f"{path}: input text exceeds {_MAXIMUM_TEXT} characters.")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return _NOT_LITERAL
    return _literal_node(tree.body, path, 0, [_MAXIMUM_ITEMS])


def _literal_node(node: ast.expr, path: str, depth: int, budget: list[int], /) -> object:
    if depth > _MAXIMUM_DEPTH:
        raise ValueError(f"{path}: literal nesting exceeds {_MAXIMUM_DEPTH}.")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError(f"{path}: literal exceeds {_MAXIMUM_ITEMS} items.")
    match node:
        case ast.Constant(value=value) if value is None or type(value) in (
            bool,
            int,
            float,
            str,
        ):
            return value
        case ast.UnaryOp(
            op=ast.USub(), operand=ast.Constant(value=int() | float() as value)
        ) if type(value) is not bool:
            return -value
        case ast.Tuple(elts=items) | ast.List(elts=items):
            values = tuple(_literal_node(item, path, depth + 1, budget) for item in items)
            if any(value is _NOT_LITERAL for value in values):
                return _NOT_LITERAL
            return values if isinstance(node, ast.Tuple) else list(values)
        case ast.Dict(keys=keys, values=items):
            if any(key is None for key in keys):
                return _NOT_LITERAL
            pairs = [
                (
                    _literal_node(key, path, depth + 1, budget),
                    _literal_node(item, path, depth + 1, budget),
                )
                for key, item in zip(keys, items, strict=True)
                if key is not None
            ]
            if any(_NOT_LITERAL in pair for pair in pairs):
                return _NOT_LITERAL
            return dict(pairs)
        case _:
            return _NOT_LITERAL


def _accepts_text(form: _Form, /) -> bool:
    match form:
        case _Scalar(kind=kind):
            return kind is str
        case _Literal(values=values):
            return any(type(value) is str for value in values)
        case _Enum():
            return True
        case _Union(alternatives=alternatives):
            return any(_accepts_text(item) for item in alternatives)
        case _:
            return False


def _text_value(form: _Form, text: str, path: str, /) -> object:
    stripped = text.strip()
    literal = _literal(text, path)
    if stripped[:1] in ("'", '"'):
        if not isinstance(literal, str):
            raise ValueError(f"{path}: {text!r} is not a quoted string literal.")
        return _coerce(form, literal, path)
    if literal is _NOT_LITERAL:
        return _coerce(form, text, path)
    if not _accepts_text(form):
        return _coerce(form, literal, path)
    # A literal spelling is preferred; verbatim text is the fallback for fields
    # that admit text, such as `name=123` for a `str` field.
    try:
        return _coerce(form, literal, path)
    except (TypeError, ValueError):
        return _coerce(form, text, path)


def _coerce(form: _Form, value: object, path: str, /) -> object:
    match form:
        case _None():
            if value is not None:
                raise TypeError(f"{path}: expected None; got {value!r}.")
            return None
        case _Scalar(kind=kind):
            return _scalar(kind, value, path)
        case _Literal(values=values):
            for member in values:
                if type(value) is type(member) and value == member:
                    return member
            raise ValueError(f"{path}: expected one of {values!r}; got {value!r}.")
        case _Enum(members=members):
            return _coerce_enum(members, value, path)
        case _Union(alternatives=alternatives):
            return _union(alternatives, value, path)
        case _Tuple(items=items, item=item):
            return _tuple(items, item, value, path)
        case _Mapping(key=key, value=item):
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{path}: expected a mapping; got {type(value).__name__}."
                )
            if len(value) > _MAXIMUM_ITEMS:
                raise ValueError(f"{path}: mapping exceeds {_MAXIMUM_ITEMS} items.")
            return frozendict(
                {
                    _coerce(key, name, f"{path} key"): _coerce(
                        item, entry, f"{path}.{name}"
                    )
                    for name, entry in value.items()
                }
            )
        case _Record():
            items = _record_items(form, value, tuple(path.split(".")))
            if type(value) is form.cls:
                return form.cls(
                    **{
                        field.name: _coerce(
                            field.form, items[field.name], f"{path}.{field.name}"
                        )
                        for field in form.fields
                    }
                )
            return _resolved_record(form, items, path)
    raise TypeError(f"{path}: unsupported configuration form.")


def _scalar(
    kind: type[bool] | type[int] | type[float] | type[str], value: object, path: str, /
) -> object:
    if kind is bool:
        if type(value) is not bool:
            raise TypeError(f"{path}: expected a bool; got {value!r}.")
        return value
    if kind is str:
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected text; got {type(value).__name__}.")
        return str(value)
    if isinstance(value, bool):
        raise TypeError(f"{path}: expected a number; got a bool.")
    if kind is int:
        if not isinstance(value, Integral):
            raise TypeError(f"{path}: expected an int; got {value!r}.")
        return int(value)
    if not isinstance(value, Real):
        raise TypeError(f"{path}: expected a real number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: expected a finite number; got {value!r}.")
    return result


def _union(alternatives: tuple[_Form, ...], value: object, path: str, /) -> object:
    accepted: list[object] = []
    failures: list[TypeError | ValueError] = []
    for alternative in alternatives:
        try:
            accepted.append(_coerce(alternative, value, path))
        except (TypeError, ValueError) as error:
            failures.append(error)
    if not accepted:
        message = " | ".join(str(error) for error in failures)
        if all(isinstance(error, TypeError) for error in failures):
            raise TypeError(message)
        raise ValueError(message)
    if len(accepted) == 1:
        return accepted[0]
    exact = [result for result in accepted if type(result) is type(value)]
    if len(exact) == 1:
        return exact[0]
    raise ValueError(f"{path}: {value!r} is ambiguous for its union annotation.")


def _tuple(
    items: tuple[_Form, ...] | None,
    item: _Form | None,
    value: object,
    path: str,
    /,
) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{path}: expected a tuple; got {type(value).__name__}.")
    if len(value) > _MAXIMUM_ITEMS:
        raise ValueError(f"{path}: tuple exceeds {_MAXIMUM_ITEMS} items.")
    if items is None:
        if item is None:
            raise TypeError(f"{path}: variadic tuple form has no item form.")
        return tuple(
            _coerce(item, entry, f"{path}[{index}]") for index, entry in enumerate(value)
        )
    if len(value) != len(items):
        raise ValueError(f"{path}: expected {len(items)} items; got {len(value)}.")
    return tuple(
        _coerce(form, entry, f"{path}[{index}]")
        for index, (form, entry) in enumerate(zip(items, value, strict=True))
    )


def _assigned(form: _Form, assignment: _Assignment, path: str, /) -> object:
    if assignment.text is None:
        return _coerce(form, assignment.value, path)
    return _text_value(form, assignment.text, path)


def _require_complete(
    record: _Record, path: _Path, assignments: Mapping[_Path, _Assignment], /
) -> list[str]:
    missing: list[str] = []
    for field in record.fields:
        child = (*path, field.name)
        if isinstance(field.form, _Record):
            missing.extend(_require_complete(field.form, child, assignments))
        elif child not in assignments:
            missing.append(_dotted(child))
    return missing


def _construct(
    record: _Record,
    path: _Path,
    assignments: Mapping[_Path, _Assignment],
    origins: list[FieldOrigin],
    prefix: str,
    /,
) -> object:
    arguments: dict[str, object] = {}
    for field in record.fields:
        child = (*path, field.name)
        if isinstance(field.form, _Record):
            arguments[field.name] = _construct(
                field.form, child, assignments, origins, prefix
            )
            continue
        assignment = assignments[child]
        dotted = _dotted(child)
        arguments[field.name] = _assigned(field.form, assignment, prefix + dotted)
        origins.append(FieldOrigin(dotted, assignment.layer, assignment.environment_keys))
    return record.cls(**arguments)


def _assemble(
    record: _Record,
    assignments: Mapping[_Path, _Assignment],
    prefix: str,
    /,
) -> tuple[object, tuple[FieldOrigin, ...]]:
    missing = _require_complete(record, (), assignments)
    if missing:
        raise ValueError(
            f"Missing required configuration inputs: {', '.join(prefix + m for m in missing)}."
        )
    origins: list[FieldOrigin] = []
    value = _construct(record, (), assignments, origins, prefix)
    return value, tuple(sorted(origins, key=lambda origin: origin.path))


def _resolved_record(
    record: _Record, values: Mapping[str, object], path: str, /
) -> object:
    assignments: dict[_Path, _Assignment] = {}
    _place(record, (), _Assignment("value", values), assignments)
    _place_defaults(record, record, (), assignments)
    value, _ = _assemble(record, assignments, f"{path}.")
    return value


def resolve(
    config_type: type[_C],
    /,
    *,
    values: Mapping[str, object] | _C | None = None,
    overrides: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> ResolvedConfig[_C]:
    """Resolve `config_type` from its layered inputs without applying anything.

    `environment` defaults to the process environment and is read only through
    the type's declared bindings. Failures leave no state behind.
    """
    kind = configuration_kind(config_type)
    record = _record(config_type, config_type.__qualname__, frozenset())
    assignments: dict[_Path, _Assignment] = {}
    _place_environment(
        config_type,
        record,
        os.environ if environment is None else environment,
        assignments,
    )
    if values is not None:
        _place(record, (), _Assignment("value", values), assignments)
    for path, text in _overrides(overrides):
        _place(record, path, _Assignment("override", None, text), assignments)
    _place_defaults(record, record, (), assignments)
    value, origins = _assemble(record, assignments, "")
    if not isinstance(value, config_type) or type(value) is not config_type:
        raise TypeError(f"{config_type.__qualname__} constructed another type.")
    encoded = _encode(record, value, "value")
    _validate_origins(record, value, origins)
    return ResolvedConfig(value, kind, configuration_id(kind, encoded), origins)


def _enum_value(value: object, depth: int = 0, /) -> object:
    if depth > _MAXIMUM_DEPTH:
        raise ValueError("Enum value exceeds configuration depth.")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Enum values must be finite.")
        return value
    if type(value) is tuple:
        if len(value) > _MAXIMUM_ITEMS:
            raise ValueError("Enum value exceeds configuration item limit.")
        return [_enum_value(item, depth + 1) for item in value]
    raise TypeError("Enum values must be recursively immutable builtin data.")


def _case(form: _Form, /) -> str:
    match form:
        case _Scalar(kind=kind):
            return kind.__name__
        case _None():
            return "none"
        case _Record(cls=cls):
            return f"record:{configuration_kind(cls)}"
        case _Enum(members=members):
            return f"enum:{configuration_kind(members)}"
        case _Literal(values=values):
            return "literal:" + canonical_json(
                sorted(
                    ([type(value).__name__, value] for value in values),
                    key=canonical_json,
                )
            )
        case _Tuple(items=items, item=item):
            if items is not None:
                return "tuple:" + canonical_json([_case(entry) for entry in items])
            if item is None:
                raise TypeError("A variadic tuple requires an item form.")
            return "tuple*:" + _case(item)
        case _Mapping(key=key, value=value):
            return "mapping:" + canonical_json([_case(key), _case(value)])
        case _Union(alternatives=alternatives):
            return "union:" + canonical_json(sorted(_case(item) for item in alternatives))


def _union_encoding(
    alternatives: tuple[_Form, ...], value: object, path: str, depth: int, /
) -> dict[str, object]:
    matches: list[tuple[_Form, object]] = []
    for alternative in alternatives:
        try:
            encoded = _encode(alternative, value, path, depth + 1)
        except (TypeError, ValueError):
            continue
        matches.append((alternative, encoded))
    if len(matches) != 1:
        raise ValueError(f"{path}: stored union value requires exactly one strict match.")
    form, encoded = matches[0]
    return {"case": _case(form), "value": encoded}


def _encode(form: _Form, value: object, path: str, depth: int = 0, /) -> object:
    """Validate stored canonical state without conversion or user callbacks."""
    if depth > _MAXIMUM_DEPTH:
        raise ValueError(f"{path}: configuration exceeds maximum depth.")
    match form:
        case _None():
            if value is not None:
                raise TypeError(f"{path}: expected None.")
            return None
        case _Scalar(kind=kind):
            if type(value) is not kind:
                raise TypeError(f"{path}: expected exact {kind.__name__}.")
            if kind is float and not math.isfinite(cast(float, value)):
                raise ValueError(f"{path}: expected a finite float.")
            return value
        case _Literal(values=values):
            if not any(type(value) is type(item) and value == item for item in values):
                raise ValueError(f"{path}: invalid literal member.")
            return value
        case _Enum(members=members):
            if type(value) is not members:
                raise TypeError(f"{path}: expected exact {members.__qualname__}.")
            return {"name": value.name, "value": _enum_value(value.value)}
        case _Union(alternatives=alternatives):
            return _union_encoding(alternatives, value, path, depth)
        case _Record():
            return _encode_record(form, value, path, depth)
        case _Tuple():
            return _encode_tuple(form, value, path, depth)
        case _Mapping():
            return _encode_mapping(form, value, path, depth)


def _encode_record(
    form: _Record, value: object, path: str, depth: int, /
) -> dict[str, object]:
    if type(value) is not form.cls:
        raise TypeError(f"{path}: expected exact {form.cls.__qualname__}.")
    return {
        field.name: _encode(
            field.form,
            object.__getattribute__(value, field.name),
            f"{path}.{field.name}",
            depth + 1,
        )
        for field in form.fields
    }


def _tuple_forms(form: _Tuple, length: int, path: str, /) -> tuple[_Form, ...]:
    if length > _MAXIMUM_ITEMS or (form.items is not None and len(form.items) != length):
        raise ValueError(f"{path}: invalid tuple length.")
    if form.items is not None:
        return form.items
    if form.item is None:
        raise TypeError(f"{path}: missing tuple item form.")
    return (form.item,) * length


def _encode_tuple(form: _Tuple, value: object, path: str, depth: int, /) -> list[object]:
    if type(value) is not tuple:
        raise TypeError(f"{path}: stored sequence must be a tuple.")
    forms = _tuple_forms(form, len(value), path)
    return [
        _encode(entry, element, f"{path}[{index}]", depth + 1)
        for index, (entry, element) in enumerate(zip(forms, value, strict=True))
    ]


def _encode_mapping(
    form: _Mapping, value: object, path: str, depth: int, /
) -> dict[str, object]:
    if type(value) is not frozendict:
        raise TypeError(f"{path}: stored mapping must be a frozendict.")
    if len(value) > _MAXIMUM_ITEMS:
        raise ValueError(f"{path}: mapping exceeds item limit.")
    result: dict[str, object] = {}
    for name, element in value.items():
        encoded_key = _encode(form.key, name, f"{path} key", depth + 1)
        if not isinstance(encoded_key, str):
            raise TypeError(f"{path}: mapping keys must be text.")
        result[encoded_key] = _encode(form.value, element, f"{path}.{name}", depth + 1)
    return result


def _coerce_enum(members: type[enum.Enum], value: object, path: str, /) -> enum.Enum:
    if isinstance(value, members):
        return value
    for member in members:
        if type(value) is type(member.value) and value == member.value:
            return member
    raise ValueError(f"{path}: {value!r} is not a {members.__qualname__} value.")


def _decode_enum(members: type[enum.Enum], value: object, path: str, /) -> enum.Enum:
    if not isinstance(value, Mapping) or set(value) != {"name", "value"}:
        raise ValueError(f"{path}: enum requires exact name/value fields.")
    name = value["name"]
    if not isinstance(name, str) or name not in members.__members__:
        raise ValueError(f"{path}: unknown enum member.")
    member = members.__members__[name]
    if member.name != name or canonical_json(_enum_value(member.value)) != canonical_json(
        value["value"]
    ):
        raise ValueError(f"{path}: noncanonical enum member or changed value.")
    return member


def _decode(form: _Form, value: object, path: str, depth: int = 0, /) -> object:
    if depth > _MAXIMUM_DEPTH:
        raise ValueError(f"{path}: configuration exceeds maximum depth.")
    match form:
        case _Union(alternatives=alternatives):
            if not isinstance(value, Mapping) or set(value) != {"case", "value"}:
                raise ValueError(f"{path}: union requires exact case/value fields.")
            for alternative in alternatives:
                if value["case"] == _case(alternative):
                    return _decode(alternative, value["value"], path, depth + 1)
            raise ValueError(f"{path}: unknown union case.")
        case _Enum(members=members):
            return _decode_enum(members, value, path)
        case _Record(cls=cls, fields=fields):
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}: record value must be a mapping.")
            if set(value) != {field.name for field in fields}:
                raise ValueError(f"{path}: record requires its complete exact field set.")
            return cls(
                **{
                    field.name: _decode(
                        field.form, value[field.name], f"{path}.{field.name}", depth + 1
                    )
                    for field in fields
                }
            )
        case _Tuple():
            if type(value) is not list:
                raise TypeError(f"{path}: wire tuple must be a list.")
            forms = _tuple_forms(form, len(value), path)
            return tuple(
                _decode(entry, element, f"{path}[{index}]", depth + 1)
                for index, (entry, element) in enumerate(zip(forms, value, strict=True))
            )
        case _Mapping(key=key, value=item):
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}: wire mapping must be a mapping.")
            if len(value) > _MAXIMUM_ITEMS:
                raise ValueError(f"{path}: mapping exceeds item limit.")
            return frozendict(
                {
                    _decode(key, name, f"{path} key", depth + 1): _decode(
                        item, element, f"{path}.{name}", depth + 1
                    )
                    for name, element in value.items()
                }
            )
        case _:
            _encode(form, value, path, depth)
            return value


def _terminal_fields(
    record: _Record, path: _Path = (), defaultable: bool = False, /
) -> dict[_Path, bool]:
    result: dict[_Path, bool] = {}
    for field in record.fields:
        child = (*path, field.name)
        available = (
            defaultable
            or field.factory is not None
            or field.default is not dataclasses.MISSING
        )
        if isinstance(field.form, _Record):
            result.update(_terminal_fields(field.form, child, available))
        else:
            result[child] = available
    return result


def _validate_origins(
    root: _Record, value: object, origins: tuple[FieldOrigin, ...], /
) -> None:
    from . import _config_environment

    terminals = _terminal_fields(root)
    if type(origins) is not tuple or any(
        type(origin) is not FieldOrigin for origin in origins
    ):
        raise TypeError("Origins must be a tuple of exact FieldOrigin values.")
    if tuple(origin.path for origin in origins) != tuple(
        sorted(_dotted(path) for path in terminals)
    ):
        raise ValueError("Origins must cover each terminal field once in sorted order.")
    bindings = vars(root.cls).get("__config_environment__", {})
    for origin in origins:
        origin.__post_init__()
        path = tuple(origin.path.split("."))
        if origin.layer == "default" and not terminals[path]:
            raise ValueError(
                f"{origin.path}: required field cannot have a default origin."
            )
        if origin.layer != "environment":
            continue
        form = root
        current = value
        native: tuple[type, _Path] | None = None
        for index in range(len(path)):
            if _config_environment.is_native_record(form.cls):
                native = (form.cls, path[index:])
            field = next(field for field in form.fields if field.name == path[index])
            current = object.__getattribute__(current, field.name)
            if isinstance(field.form, _Record):
                form = field.form
        if native is not None:
            cls, relative = native
            _config_environment.validate_native_origin(
                cls, relative, origin.environment_keys, current
            )
        elif not isinstance(bindings, Mapping) or origin.environment_keys != (
            bindings.get(origin.path),
        ):
            raise ValueError(
                f"{origin.path}: environment origin does not match its binding."
            )


def configuration_id(kind: str, encoded: object, /) -> str:
    return canonical_fingerprint(
        {"kind": "phydrax-configuration", "configuration_kind": kind, "value": encoded}
    )


def encode_snapshot_value(value: object, /, path: str = "value") -> object:
    """Observational JSON-compatible projection of host metadata; never restored."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return encode_snapshot_value(value.value, path)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: configuration values must be finite.")
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, tuple):
        return [
            encode_snapshot_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        encoded: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: configuration mapping keys must be text.")
            encoded[key] = encode_snapshot_value(item, f"{path}.{key}")
        return encoded
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encode_snapshot_value(
                object.__getattribute__(value, field.name), f"{path}.{field.name}"
            )
            for field in dataclasses.fields(value)
            if field.init
        }
    raise TypeError(
        f"{path}: {type(value).__name__} is not a canonical configuration value."
    )


__all__ = [
    "ConfigurationDefinition",
    "ConfigurationLayer",
    "FieldOrigin",
    "ResolvedConfig",
    "configuration_id",
    "configuration_kind",
    "encode_snapshot_value",
    "resolve",
]

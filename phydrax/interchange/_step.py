#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native bounded ISO 10303-21 (STEP Part 21) B-Rep reader and writer.

The reader lexes and parses one exchange structure under explicit byte,
instance, parameter, nesting and reference-depth budgets, verifies that the
instance graph has no dangling or cyclic references, and maps the AP203/AP214/
AP242 geometry, topology, unit and assembly entities listed in
`STEP_ENTITY_COVERAGE` to an exact native `BRepModel`. Any other entity reached
from an admitted root is refused with its dependency chain. The writer emits
the same exact entity set with deterministic instance numbering and
shortest-round-trip real literals.

Canonicalization (documented, stable under repeated round trips): placements
are written as ``(location, axis, ref_direction)`` so a reread frame's second
axis is ``axis x ref_direction``; plane faces use orthonormal frames (their
p-curves are mapped exactly); lines are ``point + t * magnitude * unit
direction``; pole (degenerate) edges are not STEP entities and are re-created
after the regular edges; occurrences are listed in depth-first assembly order.

Native tuple identities use versionless ``phydrax:container:``,
``phydrax:occurrence:`` and ``phydrax:definition:`` opaque standard identifier
namespaces with bounded flat-string JSON paths. Literal foreign identifiers
remain literal. Membership comes only from actual product/usage relationships;
shared path components or placements never imply a container.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, TypeAlias

import jax.numpy as jnp
import numpy as np

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceReadError,
)
from .._fingerprint import canonical_fingerprint
from .._publication import PublicationMode, publish_bytes
from ..geometry.brep._model import (
    BRepAssemblyContainer,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
)
from ..geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from ..typing import parse
from ._cad import (
    _export_loss_waivers,
    approximation_export_qualifiers,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadImportResult,
    CadInterchangeError,
    CadStage,
    curve_parameter,
    explicit_carrier,
    length_factor,
    prepare_cad_export_geometry,
    refuse,
    resource_refusal,
    si_prefix_for,
    si_prefix_scale,
    StagedCoedge,
    transform_pcurve,
)
from ._cad_carriers import RigidPlacement, surface_tag
from ._report import (
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)


StepSchema: TypeAlias = Literal["AP203", "AP214", "AP242"]

_SCHEMA_PREFIXES: tuple[tuple[str, StepSchema], ...] = (
    ("CONFIG_CONTROL_DESIGN", "AP203"),
    ("AP203_CONFIGURATION_CONTROLLED_3D_DESIGN", "AP203"),
    ("AUTOMOTIVE_DESIGN", "AP214"),
    ("AP214", "AP214"),
    ("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING", "AP242"),
)

# Entities (external names) the reader maps exactly; everything else reached
# from an admitted root is refused with its dependency chain.
STEP_ENTITY_COVERAGE: tuple[str, ...] = tuple(
    sorted(
        (
            "ADVANCED_BREP_SHAPE_REPRESENTATION",
            "ADVANCED_FACE",
            "AXIS1_PLACEMENT",
            "AXIS2_PLACEMENT_2D",
            "AXIS2_PLACEMENT_3D",
            "BREP_WITH_VOIDS",
            "B_SPLINE_CURVE_WITH_KNOTS",
            "B_SPLINE_SURFACE_WITH_KNOTS",
            "CARTESIAN_POINT",
            "CIRCLE",
            "CLOSED_SHELL",
            "CONICAL_SURFACE",
            "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
            "CONVERSION_BASED_UNIT",
            "CYLINDRICAL_SURFACE",
            "DEFINITIONAL_REPRESENTATION",
            "DIRECTION",
            "EDGE_CURVE",
            "EDGE_LOOP",
            "ELLIPSE",
            "FACE_BOUND",
            "FACE_OUTER_BOUND",
            "FACE_SURFACE",
            "ITEM_DEFINED_TRANSFORMATION",
            "LINE",
            "MANIFOLD_SOLID_BREP",
            "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
            "OPEN_SHELL",
            "ORIENTED_CLOSED_SHELL",
            "ORIENTED_EDGE",
            "ORIENTED_FACE",
            "PCURVE",
            "PLANE",
            "PRODUCT_DEFINITION",
            "PRODUCT_DEFINITION_SHAPE",
            "RATIONAL_B_SPLINE_CURVE",
            "RATIONAL_B_SPLINE_SURFACE",
            "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
            "SEAM_CURVE",
            "SHAPE_DEFINITION_REPRESENTATION",
            "SHAPE_REPRESENTATION",
            "SHAPE_REPRESENTATION_RELATIONSHIP",
            "SHELL_BASED_SURFACE_MODEL",
            "SI_UNIT",
            "SPHERICAL_SURFACE",
            "SURFACE_CURVE",
            "SURFACE_OF_LINEAR_EXTRUSION",
            "SURFACE_OF_REVOLUTION",
            "TOROIDAL_SURFACE",
            "TRIMMED_CURVE",
            "VECTOR",
            "VERTEX_LOOP",
            "VERTEX_POINT",
        )
    )
)

_BSPLINE_CURVE_PARTS = frozenset(
    {
        "BOUNDED_CURVE",
        "B_SPLINE_CURVE",
        "B_SPLINE_CURVE_WITH_KNOTS",
        "CURVE",
        "GEOMETRIC_REPRESENTATION_ITEM",
        "RATIONAL_B_SPLINE_CURVE",
        "REPRESENTATION_ITEM",
    }
)
_BSPLINE_SURFACE_PARTS = frozenset(
    {
        "BOUNDED_SURFACE",
        "B_SPLINE_SURFACE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
        "GEOMETRIC_REPRESENTATION_ITEM",
        "RATIONAL_B_SPLINE_SURFACE",
        "REPRESENTATION_ITEM",
        "SURFACE",
    }
)
_REPRESENTATIONS = frozenset(
    {
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        "SHAPE_REPRESENTATION",
    }
)


# --------------------------------------------------------------------- lexing


def _path_identifier(kind: str, path: tuple[str, ...], /) -> str:
    """Opaque standard identifier carrying an exact native instance identity."""
    return f"phydrax:{kind}:" + json.dumps(
        list(path), ensure_ascii=True, separators=(",", ":")
    )


def _identity_path(
    value: Any, kind: str, maximum_depth: int, label: str, /
) -> tuple[str, ...] | None:
    """Admit only the reserved namespace's bounded flat JSON string array."""
    prefix = f"phydrax:{kind}:"
    if not isinstance(value, str) or not value.startswith(prefix):
        return None
    text = value[len(prefix) :]
    decoder = json.JSONDecoder()
    cursor = 0
    values: list[str] = []

    def skip_space(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    cursor = skip_space(cursor)
    if cursor >= len(text) or text[cursor] != "[":
        raise refuse("malformed", label, "Invalid native path identifier.")
    cursor = skip_space(cursor + 1)
    while cursor < len(text) and text[cursor] != "]":
        if len(values) >= maximum_depth:
            raise refuse("limit", label, "Native identity path exceeds its depth limit.")
        if text[cursor] != '"':
            raise refuse(
                "malformed", label, "Native identity paths contain strings only."
            )
        try:
            name, cursor = decoder.raw_decode(text, cursor)
        except ValueError as error:
            raise refuse("malformed", label, "Invalid native path string.") from error
        if not isinstance(name, str) or not name:
            raise refuse(
                "malformed", label, "Native path components must be nonempty strings."
            )
        values.append(name)
        cursor = skip_space(cursor)
        if cursor < len(text) and text[cursor] == ",":
            cursor = skip_space(cursor + 1)
            if cursor >= len(text) or text[cursor] == "]":
                raise refuse(
                    "malformed", label, "Invalid trailing native path separator."
                )
        elif cursor >= len(text) or text[cursor] != "]":
            raise refuse("malformed", label, "Invalid native path separator.")
    if cursor >= len(text) or skip_space(cursor + 1) != len(text) or not values:
        raise refuse("malformed", label, "Invalid native path closure.")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _Ref:
    id: int


@dataclass(frozen=True, slots=True)
class _Enum:
    value: str


@dataclass(frozen=True, slots=True)
class _Typed:
    name: str
    value: Any


@dataclass(frozen=True, slots=True)
class _Binary:
    value: str


class _Derived:
    """The derived-value marker ``*``."""


_DERIVED = _Derived()

_TOKEN = re.compile(
    r"""
    (?P<space>\s+|/\*.*?\*/)
    |(?P<ref>\#[0-9]+)
    |(?P<real>[+-]?[0-9]+\.[0-9]*(?:[Ee][+-]?[0-9]+)?)
    |(?P<int>[+-]?[0-9]+)
    |(?P<string>'(?:[^']|'')*')
    |(?P<binary>"[0-9A-Fa-f]*")
    |(?P<enum>\.[A-Za-z_][A-Za-z0-9_]*\.)
    |(?P<keyword>!?[A-Za-z_][A-Za-z0-9_\-]*)
    |(?P<punct>[(),;=$*])
    """,
    re.VERBOSE | re.DOTALL,
)

_Token: TypeAlias = tuple[str, str, int]


def _decode_string(raw: str, position: int, /) -> str:
    """Part 21 string literal body: quote doubling and ``\\`` control directives."""
    text = raw[1:-1].replace("''", "'")
    result: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        rest = text[index:]
        if rest.startswith("\\\\"):
            result.append("\\")
            index += 2
        elif rest.startswith("\\X2\\") or rest.startswith("\\X4\\"):
            width = 4 if rest[2] == "2" else 8
            end = rest.find("\\X0\\")
            digits = rest[4:end]
            if end < 0 or len(digits) % width:
                raise refuse(
                    "malformed",
                    f"offset {position}",
                    "Invalid \\X2\\/\\X4\\ string escape.",
                )
            result.extend(
                chr(int(digits[offset : offset + width], 16))
                for offset in range(0, len(digits), width)
            )
            index += end + 4
        elif rest.startswith("\\X\\") and len(rest) >= 5:
            result.append(chr(int(rest[3:5], 16)))
            index += 5
        elif rest.startswith("\\S\\") and len(rest) >= 4:
            result.append(chr(ord(rest[3]) + 128))
            index += 4
        elif re.match(r"\\P[A-I]\\", rest):
            index += 4
        else:
            raise refuse("malformed", f"offset {position}", "Invalid string escape.")
    return "".join(result)


def _tokens(text: str, /) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise refuse(
                "malformed",
                f"offset {position}",
                "Unrecognized Part 21 token or unterminated literal.",
            )
        kind = match.lastgroup
        if kind is None:
            raise RuntimeError("Every token alternative is named.")
        if kind != "space":
            tokens.append((kind, match.group(), position))
        position = match.end()
    return tokens


@dataclass(frozen=True, slots=True)
class _Record:
    name: str
    params: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _Instance:
    id: int
    records: tuple[_Record, ...]
    complex: bool

    @property
    def names(self) -> frozenset[str]:
        return frozenset(record.name for record in self.records)

    @property
    def label(self) -> str:
        return f"#{self.id} " + "+".join(record.name for record in self.records)

    def record(self, name: str, /) -> _Record:
        for record in self.records:
            if record.name == name:
                return record
        raise refuse(
            "malformed", self.label, f"Instance lacks the {name} partial entity."
        )


class _Parser:
    """Bounded recursive-descent parser of one exchange structure."""

    def __init__(self, tokens: list[_Token], policy: CadImportPolicy) -> None:
        self.tokens = tokens
        self.index = 0
        self.max_depth = policy.limits.max_depth
        self.max_parameters = policy.limits.max_attributes
        self.max_instances = policy.limits.max_nodes
        self.parameters = 0
        self.depth = 0

    def _peek(self) -> _Token:
        if self.index >= len(self.tokens):
            raise refuse(
                "malformed", "end of file", "Unexpected end of the exchange structure."
            )
        return self.tokens[self.index]

    def _take(self, value: str | None = None, kind: str | None = None) -> _Token:
        token = self._peek()
        if (value is not None and token[1] != value) or (
            kind is not None and token[0] != kind
        ):
            raise refuse(
                "malformed",
                f"offset {token[2]}",
                f"Expected {value or kind!s} but found {token[1]!r}.",
            )
        self.index += 1
        return token

    def _count(self) -> None:
        self.parameters += 1
        if self.parameters > self.max_parameters:
            raise refuse(
                "limit", "parameters", "Parameter count exceeds its resource limit."
            )

    def _value(self) -> Any:
        kind, text, position = self._peek()
        self._count()
        match kind:
            case "punct" if text == "(":
                return self._list()
            case "punct" if text == "$":
                self.index += 1
                return None
            case "punct" if text == "*":
                self.index += 1
                return _DERIVED
            case "ref":
                self.index += 1
                return _Ref(int(text[1:]))
            case "real":
                self.index += 1
                return float(text)
            case "int":
                self.index += 1
                return int(text)
            case "string":
                self.index += 1
                return _decode_string(text, position)
            case "binary":
                self.index += 1
                return _Binary(text[1:-1])
            case "enum":
                self.index += 1
                return _Enum(text[1:-1].upper())
            case "keyword":
                self.index += 1
                self._take("(")
                self._enter()
                value = self._value()
                self._leave()
                self._take(")")
                return _Typed(text.upper(), value)
            case _:
                raise refuse(
                    "malformed", f"offset {position}", f"Unexpected token {text!r}."
                )

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > self.max_depth:
            raise refuse("limit", "nesting", "Parameter nesting exceeds its depth limit.")

    def _leave(self) -> None:
        self.depth -= 1

    def _list(self) -> tuple[Any, ...]:
        self._take("(")
        self._enter()
        values: list[Any] = []
        if self._peek()[1] != ")":
            values.append(self._value())
            while self._peek()[1] == ",":
                self.index += 1
                values.append(self._value())
        self._take(")")
        self._leave()
        return tuple(values)

    def _record(self) -> _Record:
        name = self._take(kind="keyword")[1].upper()
        return _Record(name, self._list())

    def parse(self) -> tuple[list[_Record], dict[int, _Instance]]:
        self._take("ISO-10303-21")
        self._take(";")
        self._take("HEADER")
        self._take(";")
        header: list[_Record] = []
        while self._peek()[1] != "ENDSEC":
            header.append(self._record())
            self._take(";")
        self._take("ENDSEC")
        self._take(";")
        self._take("DATA")
        if self._peek()[1] == "(":
            self._list()
        self._take(";")
        instances: dict[int, _Instance] = {}
        while self._peek()[1] != "ENDSEC":
            reference = self._take(kind="ref")
            identifier = int(reference[1][1:])
            if identifier in instances:
                raise refuse("malformed", reference[1], "Duplicate instance name.")
            self._take("=")
            if self._peek()[1] == "(":
                self._take("(")
                records = []
                while self._peek()[1] != ")":
                    records.append(self._record())
                self._take(")")
                if not records:
                    raise refuse("malformed", reference[1], "Empty complex instance.")
                instance = _Instance(identifier, tuple(records), True)
            else:
                instance = _Instance(identifier, (self._record(),), False)
            self._take(";")
            instances[identifier] = instance
            if len(instances) > self.max_instances:
                raise refuse(
                    "limit", reference[1], "Instance count exceeds its resource limit."
                )
        self._take("ENDSEC")
        self._take(";")
        if self._peek()[1] == "DATA":
            raise refuse(
                "unsupported-entity", "DATA", "Multiple data sections are not supported."
            )
        self._take("END-ISO-10303-21")
        self._take(";")
        if self.index != len(self.tokens):
            raise refuse("malformed", "trailer", "Content follows END-ISO-10303-21.")
        return header, instances


def _references(value: Any, /) -> Iterator[int]:
    match value:
        case _Ref():
            yield value.id
        case tuple():
            for item in value:
                yield from _references(item)
        case _Typed():
            yield from _references(value.value)
        case _:
            return


def _check_graph(
    instances: dict[int, _Instance], max_depth: int, /
) -> tuple[int, dict[int, tuple[int, ...]]]:
    """Reject dangling and cyclic references; bound the reference depth.

    Returns the depth and each instance's direct references.
    """
    children = {
        identifier: tuple(
            dict.fromkeys(
                child
                for record in instance.records
                for child in _references(record.params)
            )
        )
        for identifier, instance in instances.items()
    }
    depth: dict[int, int] = {}
    active: set[int] = set()
    deepest = 0
    for root in sorted(instances):
        if root in depth:
            continue
        stack: list[tuple[int, int]] = [(root, 0)]
        active.add(root)
        while stack:
            node, position = stack[-1]
            successors = children[node]
            if position < len(successors):
                stack[-1] = (node, position + 1)
                child = successors[position]
                chain = tuple(instances[item].label for item, _ in stack)
                if child not in instances:
                    raise refuse(
                        "dangling-reference",
                        f"#{child}",
                        "Reference to an absent instance.",
                        chain,
                    )
                if child in active:
                    raise refuse(
                        "cyclic-reference",
                        instances[child].label,
                        "Cyclic instance reference.",
                        chain,
                    )
                if child not in depth:
                    if len(stack) >= max_depth:
                        raise refuse(
                            "limit",
                            instances[child].label,
                            "Reference depth exceeds its resource limit.",
                            chain,
                        )
                    active.add(child)
                    stack.append((child, 0))
                continue
            depth[node] = 1 + max((depth[child] for child in successors), default=0)
            deepest = max(deepest, depth[node])
            if depth[node] > max_depth:
                raise refuse(
                    "limit",
                    instances[node].label,
                    "Reference depth exceeds its resource limit.",
                )
            active.discard(node)
            stack.pop()
    return deepest, children


def _schema(header: list[_Record], /) -> StepSchema:
    for record in header:
        if record.name == "FILE_SCHEMA":
            names = record.params[0] if record.params else ()
            if isinstance(names, tuple) and names and isinstance(names[0], str):
                identifier = names[0].strip().upper()
                for prefix, schema in _SCHEMA_PREFIXES:
                    if identifier.startswith(prefix):
                        return schema
                raise refuse(
                    "unsupported-entity",
                    "FILE_SCHEMA",
                    f"Unsupported schema {names[0]!r}.",
                )
    raise refuse("malformed", "HEADER", "The header lacks FILE_SCHEMA.")


# ------------------------------------------------------------------- decoding


@dataclass(slots=True)
class _Edge:
    index: int
    flip: int
    pcurves: list[tuple[int, AbstractCurve]]
    parameter_factor: float


@dataclass(frozen=True, slots=True)
class _Units:
    length_meters: Fraction
    angle_radians: float
    scale: float


def _unit(vector: np.ndarray, label: str, /) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0 or not np.isfinite(norm):
        raise refuse("malformed", label, "A direction must be finite and nonzero.")
    return vector if norm == 1.0 else vector / norm


def _reverse(loop: list[StagedCoedge], /) -> list[StagedCoedge]:
    return [StagedCoedge(c.edge, -c.sense, c.pcurve, c.label) for c in reversed(loop)]


class _Decoder:
    """Map an admitted instance graph to a `CadStage`."""

    def __init__(self, instances: dict[int, _Instance], policy: CadImportPolicy) -> None:
        self.instances = instances
        self.policy = policy
        self.chain: list[str] = []
        self.units: _Units | None = None
        self.used: dict[str, int] = {}
        self.seen: set[int] = set()
        self.stage: CadStage | None = None
        self.vertices: dict[int, int] = {}
        self.edges: dict[int, _Edge] = {}
        self.faces: dict[int, tuple[int, int]] = {}
        self.solids: dict[int, int] = {}
        self.representations: dict[int, list[int]] = {}
        self.unit_cache: dict[int, _Units] = {}
        self.entity_units: dict[int, _Units] = {}
        self.children: dict[int, tuple[int, ...]] = {}
        self.curve_trims: dict[int, tuple[tuple[float, float], bool]] = {}

    # -------------------------------------------------------------- access

    def _refuse(
        self, reason: Any, instance: _Instance | str, message: str
    ) -> CadInterchangeError:
        label = instance if isinstance(instance, str) else instance.label
        return refuse(reason, label, message, tuple(self.chain))

    def _get(self, value: Any, /) -> _Instance:
        if not isinstance(value, _Ref):
            raise self._refuse(
                "malformed",
                self.chain[-1] if self.chain else "?",
                "Expected an instance reference.",
            )
        instance = self.instances[value.id]
        if instance.id not in self.seen:
            self.seen.add(instance.id)
            for record in instance.records:
                self.used[record.name] = self.used.get(record.name, 0) + 1
        return instance

    def _simple(self, instance: _Instance, /) -> _Record:
        if instance.complex:
            raise self._refuse(
                "unsupported-entity", instance, "Unexpected complex instance."
            )
        return instance.records[0]

    def _params(self, instance: _Instance, count: int, /) -> tuple[Any, ...]:
        params = self._simple(instance).params
        if len(params) != count:
            raise self._refuse("malformed", instance, f"Expected {count} parameters.")
        return params

    def _float(self, value: Any, instance: _Instance, /) -> float:
        if isinstance(value, _Typed):
            value = value.value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise self._refuse("malformed", instance, "Expected a finite real parameter.")
        return float(value)

    def _int(self, value: Any, instance: _Instance, /) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._refuse("malformed", instance, "Expected an integer parameter.")
        return value

    def _bool(self, value: Any, instance: _Instance, /) -> bool:
        if not isinstance(value, _Enum) or value.value not in ("T", "F"):
            raise self._refuse("malformed", instance, "Expected a boolean parameter.")
        return value.value == "T"

    def _list(self, value: Any, instance: _Instance, /) -> tuple[Any, ...]:
        if not isinstance(value, tuple):
            raise self._refuse("malformed", instance, "Expected a list parameter.")
        return value

    def _floats(self, value: Any, instance: _Instance, /) -> np.ndarray:
        return np.asarray(
            [self._float(item, instance) for item in self._list(value, instance)],
            dtype=np.float64,
        )

    def _enter(self, instance: _Instance, /) -> None:
        self.chain.append(instance.label)

    def _leave(self) -> None:
        self.chain.pop()

    @property
    def _scale(self) -> _Units:
        if self.units is None:
            raise RuntimeError("Geometry is decoded inside a unit context.")
        return self.units

    # --------------------------------------------------------------- units

    def _unit_value(self, reference: Any, kind: str) -> Fraction:
        instance = self._get(reference)
        self._enter(instance)
        names = instance.names
        if kind not in names:
            raise self._refuse("units", instance, f"Expected a {kind}.")
        if "SI_UNIT" in names:
            prefix, name = instance.record("SI_UNIT").params
            expected = "METRE" if kind == "LENGTH_UNIT" else "RADIAN"
            if not isinstance(name, _Enum) or name.value != expected:
                raise self._refuse("units", instance, f"Unsupported SI unit for {kind}.")
            value = si_prefix_scale(prefix.value if isinstance(prefix, _Enum) else None)
        elif "CONVERSION_BASED_UNIT" in names:
            factor = self._get(instance.record("CONVERSION_BASED_UNIT").params[1])
            self._enter(factor)
            magnitude, base = (
                factor.records[0].params[:2]
                if not factor.complex
                else factor.record("MEASURE_WITH_UNIT").params[:2]
            )
            amount = self._float(magnitude, factor)
            value = Fraction(repr(amount)) * self._unit_value(base, kind)
            self._leave()
        else:
            raise self._refuse("units", instance, "Unsupported unit definition.")
        self._leave()
        return value

    def _context_units(self, reference: Any) -> _Units:
        instance = self._get(reference)
        if instance.id in self.unit_cache:
            return self.unit_cache[instance.id]
        self._enter(instance)
        if "GLOBAL_UNIT_ASSIGNED_CONTEXT" not in instance.names:
            raise self._refuse(
                "units", instance, "The representation context assigns no units."
            )
        lengths: list[Fraction] = []
        angles: list[Fraction] = []
        for unit in self._list(
            instance.record("GLOBAL_UNIT_ASSIGNED_CONTEXT").params[0], instance
        ):
            names = (
                self.instances[unit.id].names if isinstance(unit, _Ref) else frozenset()
            )
            if "LENGTH_UNIT" in names:
                lengths.append(self._unit_value(unit, "LENGTH_UNIT"))
            elif "PLANE_ANGLE_UNIT" in names:
                angles.append(self._unit_value(unit, "PLANE_ANGLE_UNIT"))
            else:
                self._get(unit)
        if len(set(lengths)) != 1 or len(lengths) != 1:
            raise self._refuse(
                "units", instance, "The context must assign exactly one length unit."
            )
        if len(angles) != 1:
            raise self._refuse(
                "units", instance, "The context must assign exactly one plane angle unit."
            )
        units = _Units(
            lengths[0],
            float(angles[0]),
            length_factor(lengths[0], self.policy.coordinate_contract),
        )
        self._leave()
        self.unit_cache[instance.id] = units
        return units

    # ------------------------------------------------------------ geometry

    def _point(self, reference: Any, dimension: int, scaled: bool) -> np.ndarray:
        instance = self._get(reference)
        if self._simple(instance).name != "CARTESIAN_POINT":
            raise self._refuse(
                "unsupported-entity", instance, "Expected CARTESIAN_POINT."
            )
        values = self._floats(self._params(instance, 2)[1], instance)
        if values.shape != (dimension,):
            raise self._refuse("malformed", instance, f"Expected a {dimension}D point.")
        return (
            values * self._scale.scale if scaled and self._scale.scale != 1.0 else values
        )

    def _direction(self, reference: Any, dimension: int) -> np.ndarray:
        instance = self._get(reference)
        if self._simple(instance).name != "DIRECTION":
            raise self._refuse("unsupported-entity", instance, "Expected DIRECTION.")
        values = self._floats(self._params(instance, 2)[1], instance)
        if values.shape != (dimension,):
            raise self._refuse(
                "malformed", instance, f"Expected a {dimension}D direction."
            )
        return _unit(values, instance.label)

    def _vector(self, reference: Any, dimension: int, scaled: bool) -> np.ndarray:
        instance = self._get(reference)
        if self._simple(instance).name != "VECTOR":
            raise self._refuse("unsupported-entity", instance, "Expected VECTOR.")
        _, orientation, magnitude = self._params(instance, 3)
        length = self._float(magnitude, instance)
        if scaled:
            length *= self._scale.scale
        direction = self._direction(orientation, dimension)
        return direction if length == 1.0 else direction * length

    def _frame3(
        self, reference: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        instance = self._get(reference)
        if self._simple(instance).name != "AXIS2_PLACEMENT_3D":
            raise self._refuse(
                "unsupported-entity", instance, "Expected AXIS2_PLACEMENT_3D."
            )
        _, location, axis, ref = self._params(instance, 4)
        origin = self._point(location, 3, True)
        z = np.asarray((0.0, 0.0, 1.0)) if axis is None else self._direction(axis, 3)
        if ref is None:
            ref_ = np.eye(3)[0] if abs(z[0]) < 1.0 - 1.0e-12 else np.eye(3)[2]
        else:
            ref_ = self._direction(ref, 3)
        along = float(ref_ @ z)
        x = _unit(ref_ if along == 0.0 else ref_ - along * z, instance.label)
        return origin, x, np.cross(z, x), z

    def _frame2(self, reference: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        instance = self._get(reference)
        if self._simple(instance).name != "AXIS2_PLACEMENT_2D":
            raise self._refuse(
                "unsupported-entity", instance, "Expected AXIS2_PLACEMENT_2D."
            )
        _, location, ref = self._params(instance, 3)
        origin = self._point(location, 2, False)
        x = np.asarray((1.0, 0.0)) if ref is None else self._direction(ref, 2)
        return origin, x, np.asarray((-x[1], x[0]))

    def _knots(
        self,
        multiplicities: Any,
        knots: Any,
        count: int,
        degree: int,
        instance: _Instance,
    ) -> np.ndarray:
        values = self._floats(knots, instance)
        repeats = [
            self._int(item, instance) for item in self._list(multiplicities, instance)
        ]
        if len(repeats) != values.shape[0] or any(item < 1 for item in repeats):
            raise self._refuse(
                "malformed", instance, "Knot multiplicities do not match the knots."
            )
        expected = count + degree + 1
        if degree < 1 or degree >= count or sum(repeats) != expected:
            raise self._refuse(
                "malformed", instance, "B-spline knot count or degree is inconsistent."
            )
        # Validate expansion before allocating: a tiny record can otherwise
        # request an arbitrarily large repeat count.
        if expected > self.policy.limits.max_attributes:
            raise self._refuse("limit", instance, "Expanded knots exceed their limit.")
        if np.any(np.diff(values) <= 0.0):
            raise self._refuse("malformed", instance, "Distinct knots must increase.")
        expanded = np.repeat(values, repeats)
        return expanded

    def _spline_curve(self, instance: _Instance, dimension: int) -> BSplineCurve:
        names = instance.names
        if not names <= _BSPLINE_CURVE_PARTS:
            raise self._refuse(
                "unsupported-entity",
                instance,
                f"Unsupported B-spline curve form {sorted(names - _BSPLINE_CURVE_PARTS)}.",
            )
        if instance.complex:
            degree_, points_, *_ = instance.record("B_SPLINE_CURVE").params
            multiplicities, knots, _ = instance.record("B_SPLINE_CURVE_WITH_KNOTS").params
        else:
            _, degree_, points_, _, _, _, multiplicities, knots, _ = self._params(
                instance, 9
            )
        degree = self._int(degree_, instance)
        points = np.stack(
            [
                self._point(item, dimension, dimension == 3)
                for item in self._list(points_, instance)
            ]
        )
        weights = (
            self._floats(instance.record("RATIONAL_B_SPLINE_CURVE").params[0], instance)
            if "RATIONAL_B_SPLINE_CURVE" in names
            else np.ones(points.shape[0])
        )
        expanded = self._knots(multiplicities, knots, points.shape[0], degree, instance)
        try:
            return BSplineCurve(points, weights, expanded, degree)
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _curve(self, reference: Any, dimension: int) -> AbstractCurve:
        instance = self._get(reference)
        self._enter(instance)
        names = instance.names
        scaled = dimension == 3
        units = self._scale
        try:
            if "B_SPLINE_CURVE_WITH_KNOTS" in names:
                curve: AbstractCurve = self._spline_curve(instance, dimension)
            else:
                record = self._simple(instance)
                match record.name:
                    case "LINE":
                        _, point, vector = self._params(instance, 3)
                        curve = LineCurve(
                            self._point(point, dimension, scaled),
                            self._vector(vector, dimension, scaled),
                        )
                    case "CIRCLE" | "ELLIPSE":
                        if dimension == 3:
                            origin, x, y, _ = self._frame3(record.params[1])
                        else:
                            origin, x, y = self._frame2(record.params[1])
                        factor = units.scale if scaled else 1.0
                        radii = [
                            self._float(item, instance) * factor
                            for item in record.params[2:]
                        ]
                        if record.name == "CIRCLE" and len(radii) == 1:
                            curve = CircleCurve(origin, x, y, radii[0])
                        elif record.name == "ELLIPSE" and len(radii) == 2:
                            curve = EllipseCurve(origin, x, y, radii[0], radii[1])
                        else:
                            raise self._refuse(
                                "malformed", instance, "Invalid conic parameters."
                            )
                    case "TRIMMED_CURVE":
                        _, basis, trim1, trim2, sense_, preference = self._params(
                            instance, 6
                        )
                        curve = self._curve(basis, dimension)
                        if isinstance(basis, _Ref) and basis.id in self.curve_trims:
                            raise self._refuse(
                                "unsupported-entity",
                                instance,
                                "Nested trimmed curves are not admitted.",
                            )
                        sense = self._bool(sense_, instance)
                        if not isinstance(preference, _Enum) or preference.value not in (
                            "PARAMETER",
                            "CARTESIAN",
                            "UNSPECIFIED",
                        ):
                            raise self._refuse(
                                "malformed", instance, "Invalid trimming preference."
                            )
                        first = self._trim_parameter(
                            trim1, curve, dimension, preference.value, instance
                        )
                        last = self._trim_parameter(
                            trim2, curve, dimension, preference.value, instance
                        )
                        if not sense:
                            first, last = last, first
                        if curve.period is not None:
                            if last <= first:
                                last += curve.period * (
                                    np.floor((first - last) / curve.period) + 1
                                )
                        curve.validate_range(first, last)
                        self.curve_trims[instance.id] = ((first, last), sense)
                    case _:
                        raise self._refuse(
                            "unsupported-entity",
                            instance,
                            f"Curve entity {record.name} has no exact native carrier.",
                        )
        except ValueError as error:
            if isinstance(error, CadInterchangeError):
                raise
            raise self._refuse("malformed", instance, str(error)) from error
        self._leave()
        return curve

    def _trim_parameter(
        self,
        selectors: Any,
        curve: AbstractCurve,
        dimension: int,
        preference: str,
        instance: _Instance,
    ) -> float:
        parameters: list[float] = []
        points: list[float] = []
        for selector in self._list(selectors, instance):
            if isinstance(selector, _Typed) and selector.name == "PARAMETER_VALUE":
                parameters.append(
                    self._float(selector.value, instance) * self._parameter_factor(curve)
                )
            elif isinstance(selector, _Ref):
                parameter, gap = curve_parameter(
                    curve, self._point(selector, dimension, dimension == 3)
                )
                if gap > self._stage().tolerance:
                    raise self._refuse(
                        "inconsistent-geometry",
                        instance,
                        "Trim point is not on its basis curve.",
                    )
                points.append(parameter)
            else:
                raise self._refuse("malformed", instance, "Invalid trimming selector.")
        if len(parameters) > 1 or len(points) > 1 or not (parameters or points):
            raise self._refuse(
                "malformed", instance, "A trim requires one parameter and/or point."
            )
        if parameters and points:
            gap = abs(parameters[0] - points[0])
            if curve.period is not None:
                gap = abs((gap + 0.5 * curve.period) % curve.period - 0.5 * curve.period)
            if gap > 1.0e-9 * max(1.0, abs(parameters[0])):
                raise self._refuse(
                    "inconsistent-geometry", instance, "Trim selectors disagree."
                )
        return (
            points[0]
            if points and preference == "CARTESIAN"
            else (parameters or points)[0]
        )

    def _parameter_factor(self, curve: AbstractCurve) -> float:
        return (
            self._scale.angle_radians
            if isinstance(curve, (CircleCurve, EllipseCurve))
            else 1.0
        )

    def _spline_surface(self, instance: _Instance) -> BSplineSurfacePatch:
        names = instance.names
        if not names <= _BSPLINE_SURFACE_PARTS:
            raise self._refuse(
                "unsupported-entity",
                instance,
                f"Unsupported B-spline surface form {sorted(names - _BSPLINE_SURFACE_PARTS)}.",
            )
        if instance.complex:
            u_degree_, v_degree_, grid_, *_ = instance.record("B_SPLINE_SURFACE").params
            u_mults, v_mults, u_knots, v_knots, _ = instance.record(
                "B_SPLINE_SURFACE_WITH_KNOTS"
            ).params
        else:
            params = self._params(instance, 13)
            u_degree_, v_degree_, grid_ = params[1:4]
            u_mults, v_mults, u_knots, v_knots = params[8:12]
        rows = self._list(grid_, instance)
        points = np.stack(
            [
                np.stack(
                    [self._point(item, 3, True) for item in self._list(row, instance)]
                )
                for row in rows
            ]
        )
        if "RATIONAL_B_SPLINE_SURFACE" in names:
            weight_rows = self._list(
                instance.record("RATIONAL_B_SPLINE_SURFACE").params[0], instance
            )
            weights = np.stack([self._floats(row, instance) for row in weight_rows])
        else:
            weights = np.ones(points.shape[:2])
        u_degree, v_degree = (
            self._int(u_degree_, instance),
            self._int(v_degree_, instance),
        )
        try:
            return BSplineSurfacePatch(
                points,
                weights,
                self._knots(u_mults, u_knots, points.shape[0], u_degree, instance),
                self._knots(v_mults, v_knots, points.shape[1], v_degree, instance),
                u_degree,
                v_degree,
            )
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _surface(
        self, reference: Any
    ) -> tuple[AbstractSurfacePatch, tuple[float, float]]:
        instance = self._get(reference)
        self._enter(instance)
        units = self._scale
        angle, scale = units.angle_radians, units.scale
        patch: AbstractSurfacePatch
        try:
            if "B_SPLINE_SURFACE_WITH_KNOTS" in instance.names:
                patch, factors = self._spline_surface(instance), (1.0, 1.0)
            else:
                record = self._simple(instance)
                params = record.params
                match record.name:
                    case "PLANE":
                        origin, x, y, _ = self._frame3(params[1])
                        patch, factors = PlanePatch(origin, x, y), (scale, scale)
                    case "CYLINDRICAL_SURFACE":
                        origin, x, y, z = self._frame3(params[1])
                        radius = self._float(params[2], instance) * scale
                        patch, factors = (
                            CylinderPatch(origin, x, y, z, radius),
                            (angle, scale),
                        )
                    case "CONICAL_SURFACE":
                        origin, x, y, z = self._frame3(params[1])
                        radius = self._float(params[2], instance) * scale
                        semi = self._float(params[3], instance) * angle
                        patch, factors = (
                            ConePatch(origin, x, y, z, radius, semi),
                            (angle, scale * float(np.cos(semi))),
                        )
                    case "SPHERICAL_SURFACE":
                        origin, x, y, z = self._frame3(params[1])
                        radius = self._float(params[2], instance) * scale
                        patch, factors = (
                            SpherePatch(origin, x, y, z, radius),
                            (angle, angle),
                        )
                    case "TOROIDAL_SURFACE":
                        origin, x, y, z = self._frame3(params[1])
                        major = self._float(params[2], instance) * scale
                        minor = self._float(params[3], instance) * scale
                        patch = TorusPatch(origin, x, y, z, major, minor)
                        factors = (angle, angle)
                    case "SURFACE_OF_LINEAR_EXTRUSION":
                        curve = self._curve(params[1], 3)
                        patch = ExtrusionSurface(curve, self._vector(params[2], 3, True))
                        factors = (self._parameter_factor(curve), 1.0)
                    case "SURFACE_OF_REVOLUTION":
                        curve = self._curve(params[1], 3)
                        axis = self._get(params[2])
                        if self._simple(axis).name != "AXIS1_PLACEMENT":
                            raise self._refuse(
                                "unsupported-entity", axis, "Expected AXIS1_PLACEMENT."
                            )
                        _, location, direction = self._params(axis, 3)
                        patch = RevolutionSurface(
                            curve,
                            self._point(location, 3, True),
                            self._direction(direction, 3),
                        )
                        factors = (angle, self._parameter_factor(curve))
                    case "OFFSET_SURFACE":
                        _, basis, distance, self_intersect = self._params(instance, 4)
                        if not isinstance(
                            self_intersect, _Enum
                        ) or self_intersect.value not in ("T", "F", "U"):
                            raise self._refuse(
                                "malformed",
                                instance,
                                "Expected a logical self-intersection parameter.",
                            )
                        base, factors = self._surface(basis)
                        patch = OffsetSurface(
                            base, self._float(distance, instance) * scale
                        )
                    case _:
                        raise self._refuse(
                            "unsupported-entity",
                            instance,
                            f"Surface entity {record.name} has no exact native carrier.",
                        )
        except ValueError as error:
            if isinstance(error, CadInterchangeError):
                raise
            raise self._refuse("malformed", instance, str(error)) from error
        self._leave()
        return patch, factors

    # ------------------------------------------------------------ topology

    def _stage(self) -> CadStage:
        if self.stage is None:
            raise RuntimeError("Topology is decoded into a stage.")
        return self.stage

    def _check_units(self, instance: _Instance) -> None:
        previous = self.entity_units.setdefault(instance.id, self._scale)
        if previous != self._scale:
            raise self._refuse(
                "units", instance, "Topology is shared by contexts with different units."
            )

    def _vertex(self, reference: Any) -> int:
        instance = self._get(reference)
        if instance.id in self.vertices:
            self._check_units(instance)
            return self.vertices[instance.id]
        self._enter(instance)
        if self._simple(instance).name != "VERTEX_POINT":
            raise self._refuse("unsupported-entity", instance, "Expected VERTEX_POINT.")
        self._check_units(instance)
        index = self._stage().vertex(
            self._point(self._params(instance, 2)[1], 3, True), instance.label
        )
        self._leave()
        self.vertices[instance.id] = index
        return index

    def _pcurve(self, reference: Any) -> tuple[int, AbstractCurve] | None:
        instance = self._get(reference)
        if instance.complex or instance.records[0].name != "PCURVE":
            # SURFACE_CURVE also permits a direct basis-surface association.
            # Do not ignore arbitrary unsupported geometry in that slot.
            self._surface(reference)
            return None
        self._enter(instance)
        _, surface, definition = self._params(instance, 3)
        if not isinstance(surface, _Ref):
            raise self._refuse("malformed", instance, "PCURVE requires a basis surface.")
        representation = self._get(definition)
        if self._simple(representation).name != "DEFINITIONAL_REPRESENTATION":
            raise self._refuse(
                "unsupported-entity",
                representation,
                "Expected DEFINITIONAL_REPRESENTATION.",
            )
        items = self._list(self._params(representation, 3)[1], representation)
        if len(items) != 1:
            raise self._refuse(
                "unsupported-entity", representation, "A p-curve must hold one 2D curve."
            )
        self._enter(representation)
        curve = self._curve(items[0], 2)
        self._leave()
        self._leave()
        return surface.id, curve

    def _edge(self, reference: Any) -> _Edge:
        instance = self._get(reference)
        if instance.id in self.edges:
            self._check_units(instance)
            return self.edges[instance.id]
        self._enter(instance)
        if self._simple(instance).name != "EDGE_CURVE":
            raise self._refuse("unsupported-entity", instance, "Expected EDGE_CURVE.")
        self._check_units(instance)
        _, start_, end_, geometry_, same_sense_ = self._params(instance, 5)
        geometry = self._get(geometry_)
        pcurves: list[tuple[int, AbstractCurve]] = []
        name = geometry.records[0].name if not geometry.complex else ""
        if name in ("SURFACE_CURVE", "SEAM_CURVE"):
            self._enter(geometry)
            _, curve_, associated, _ = self._params(geometry, 4)
            curve = self._curve(curve_, 3)
            curve_reference = curve_
            for item in self._list(associated, geometry):
                recovered = self._pcurve(item)
                if recovered is not None:
                    pcurves.append(recovered)
            self._leave()
        elif name == "PCURVE":
            raise self._refuse(
                "unsupported-entity",
                geometry,
                "An edge defined only by a p-curve has no 3D carrier.",
            )
        else:
            curve = self._curve(geometry_, 3)
            curve_reference = geometry_
        start, end = self._vertex(start_), self._vertex(end_)
        same_sense = self._bool(same_sense_, instance)
        trim = (
            self.curve_trims.get(curve_reference.id)
            if isinstance(curve_reference, _Ref)
            else None
        )
        if trim is not None:
            same_sense = same_sense == trim[1]
        stage = self._stage()
        try:
            directed = (start, end) if same_sense else (end, start)
            parameter_range = None
            if trim is not None:
                lower, upper = trim[0]
                endpoints = np.asarray(curve.evaluate(jnp.asarray((lower, upper))))
                vertices = np.asarray([stage.vertices[index] for index in directed])
                gaps = np.linalg.norm(endpoints - vertices, axis=1)
                if np.all(gaps <= stage.tolerance):
                    parameter_range = (lower, upper)
                else:
                    first, first_gap = curve_parameter(curve, vertices[0])
                    last, last_gap = curve_parameter(curve, vertices[1])
                    if max(first_gap, last_gap) > stage.tolerance:
                        raise ValueError("An edge vertex is not on its trimmed carrier.")
                    period = curve.period
                    if period is not None:
                        first = lower + (first - lower) % period
                        last = lower + (last - lower) % period
                        if last <= first:
                            last += period
                    if gaps[0] <= stage.tolerance:
                        first = lower
                    if gaps[1] <= stage.tolerance:
                        last = upper
                    if first < lower or last > upper or last <= first:
                        raise ValueError(
                            "Edge vertices lie outside the declared curve trim."
                        )
                    parameter_range = (first, last)
            index = stage.edge(
                curve,
                *directed,
                instance.label,
                parameter_range=parameter_range,
            )
        except ValueError as error:
            raise self._refuse("inconsistent-geometry", instance, str(error)) from error
        self._leave()
        edge = _Edge(
            index, 1 if same_sense else -1, pcurves, self._parameter_factor(curve)
        )
        self.edges[instance.id] = edge
        return edge

    def _loop(
        self, reference: Any, surface: int, factors: tuple[float, float]
    ) -> list[StagedCoedge] | int:
        instance = self._get(reference)
        self._enter(instance)
        record = self._simple(instance)
        if record.name == "VERTEX_LOOP":
            vertex = self._vertex(self._params(instance, 2)[1])
            self._leave()
            return vertex
        if record.name != "EDGE_LOOP":
            raise self._refuse(
                "unsupported-entity",
                instance,
                f"Loop entity {record.name} is not supported.",
            )
        coedges: list[StagedCoedge] = []
        for item in self._list(self._params(instance, 2)[1], instance):
            oriented = self._get(item)
            self._enter(oriented)
            if self._simple(oriented).name != "ORIENTED_EDGE":
                raise self._refuse(
                    "unsupported-entity", oriented, "Expected ORIENTED_EDGE."
                )
            _, _, _, element, orientation_ = self._params(oriented, 5)
            orientation = self._bool(orientation_, oriented)
            edge = self._edge(element)
            candidates = [curve for owner, curve in edge.pcurves if owner == surface]
            chosen = None
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) == 2:
                chosen = candidates[0 if orientation else 1]
            pcurve = None
            if chosen is not None:
                try:
                    pcurve = transform_pcurve(chosen, factors, edge.parameter_factor)
                except ValueError as error:
                    raise self._refuse(
                        "unsupported-entity",
                        oriented,
                        f"The supplied p-curve parameterization cannot be retained: {error}",
                    ) from error
            sense = (1 if orientation else -1) * edge.flip
            coedges.append(StagedCoedge(edge.index, sense, pcurve, oriented.label))
            self._leave()
        self._leave()
        if not coedges:
            raise self._refuse("malformed", instance, "An edge loop requires edges.")
        return coedges

    def _face(self, reference: Any, sign: int) -> tuple[int, int]:
        instance = self._get(reference)
        if instance.id in self.faces:
            self._check_units(instance)
            index, initial_sign = self.faces[instance.id]
            return index, sign * initial_sign
        self._enter(instance)
        self._check_units(instance)
        record = self._simple(instance)
        if record.name not in ("ADVANCED_FACE", "FACE_SURFACE"):
            raise self._refuse(
                "unsupported-entity",
                instance,
                f"Face entity {record.name} is not supported.",
            )
        _, bounds, surface_, same_sense_ = self._params(instance, 4)
        patch, factors = self._surface(surface_)
        if not isinstance(surface_, _Ref):
            raise RuntimeError("Surfaces were resolved by reference.")
        same_sense = self._bool(same_sense_, instance)
        outer: list[list[StagedCoedge]] = []
        inner: list[list[StagedCoedge]] = []
        boundary_vertices: list[int] = []
        for item in self._list(bounds, instance):
            bound = self._get(item)
            self._enter(bound)
            name = self._simple(bound).name
            if name not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
                raise self._refuse(
                    "unsupported-entity", bound, f"Bound entity {name} is not supported."
                )
            _, loop_, orientation_ = self._params(bound, 3)
            loop = self._loop(loop_, surface_.id, factors)
            forward = self._bool(orientation_, bound)
            if isinstance(loop, int):
                boundary_vertices.append(loop)
            else:
                loop = loop if forward else _reverse(loop)
                loop = loop if same_sense else _reverse(loop)
                (outer if name == "FACE_OUTER_BOUND" else inner).append(loop)
            self._leave()
        stage = self._stage()
        if len(outer) > 1:
            raise self._refuse(
                "inconsistent-geometry", instance, "A face has several outer bounds."
            )
        loops = outer + inner
        outer_known = bool(outer)
        if not loops:
            loops = [
                stage.natural_loop(
                    patch,
                    instance.label,
                    boundary_vertices=tuple(boundary_vertices),
                )
            ]
            outer_known = True
        elif boundary_vertices:
            used_vertices = {
                vertex
                for loop in loops
                for coedge in loop
                for vertex in stage.edge_vertices[coedge.edge]
            }
            if any(vertex not in used_vertices for vertex in boundary_vertices):
                raise self._refuse(
                    "unsupported-entity",
                    instance,
                    "A point boundary alongside edge loops lacks source-owned incidence.",
                )
        orientation = (1 if same_sense else -1) * sign
        index = stage.face(
            patch,
            loops,
            orientation,
            surface_tag(patch),
            instance.label,
            outer_known=outer_known,
        )
        self._leave()
        self.faces[instance.id] = (index, sign)
        return index, 1

    def _shell(self, reference: Any, sign: int) -> int:
        instance = self._get(reference)
        self._enter(instance)
        record = self._simple(instance)
        if record.name == "ORIENTED_CLOSED_SHELL":
            _, _, element, orientation = self._params(instance, 4)
            index = self._shell(
                element, sign * (1 if self._bool(orientation, instance) else -1)
            )
            self._leave()
            return index
        if record.name not in ("CLOSED_SHELL", "OPEN_SHELL"):
            raise self._refuse(
                "unsupported-entity",
                instance,
                f"Shell entity {record.name} is not supported.",
            )
        faces = []
        orientations = []
        for item in self._list(self._params(instance, 2)[1], instance):
            face = self._get(item)
            if not face.complex and face.records[0].name == "ORIENTED_FACE":
                self._enter(face)
                _, _, element, orientation = self._params(face, 4)
                index, incidence = self._face(
                    element, sign * (1 if self._bool(orientation, face) else -1)
                )
                self._leave()
            else:
                index, incidence = self._face(item, sign)
            faces.append(index)
            orientations.append(incidence)
        stage = self._stage()
        stage.shells.append((faces, record.name == "CLOSED_SHELL"))
        stage.shell_orientations[len(stage.shells) - 1] = orientations
        self._leave()
        return len(stage.shells) - 1

    def _prestage(self, root: int) -> None:
        """Stage a shape's vertices, then edges, in instance order.

        Native entity order follows the external instance numbering rather
        than traversal order, so a canonical writer's order is reproduced.
        """
        reachable = {root}
        pending = [root]
        while pending:
            for child in self.children.get(pending.pop(), ()):
                if child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        for kind in ("VERTEX_POINT", "EDGE_CURVE"):
            for identifier in sorted(reachable):
                instance = self.instances[identifier]
                if not instance.complex and instance.records[0].name == kind:
                    if kind == "VERTEX_POINT":
                        self._vertex(_Ref(identifier))
                    else:
                        self._edge(_Ref(identifier))

    def _solid(self, instance: _Instance) -> int | None:
        """Stage one representation item; returns a solid index (None for shells)."""
        if instance.id in self.solids:
            self._check_units(instance)
            return self.solids[instance.id]
        self._enter(instance)
        self._check_units(instance)
        record = self._simple(instance)
        stage = self._stage()
        self._prestage(instance.id)
        match record.name:
            case "MANIFOLD_SOLID_BREP":
                shells = [self._shell(self._params(instance, 2)[1], 1)]
            case "BREP_WITH_VOIDS":
                _, outer, voids = self._params(instance, 3)
                shells = [self._shell(outer, 1)] + [
                    self._shell(item, 1) for item in self._list(voids, instance)
                ]
            case "SHELL_BASED_SURFACE_MODEL":
                for item in self._list(self._params(instance, 2)[1], instance):
                    self._shell(item, 1)
                self._leave()
                return None
            case _:
                raise self._refuse(
                    "unsupported-entity",
                    instance,
                    f"Representation item {record.name} is not supported.",
                )
        stage.solids.append(shells)
        self.solids[instance.id] = len(stage.solids) - 1
        self._leave()
        return self.solids[instance.id]

    # ------------------------------------------------------ representations

    def _representation(self, identifier: int) -> list[int]:
        """Stage the solids of one shape representation (memoized)."""
        if identifier in self.representations:
            return self.representations[identifier]
        instance = self._get(_Ref(identifier))
        self._enter(instance)
        record = next(
            record for record in instance.records if record.name in _REPRESENTATIONS
        )
        if len(record.params) != 3:
            raise self._refuse(
                "malformed", instance, "A representation has name, items and context."
            )
        previous = self.units
        self.units = self._context_units(record.params[2])
        solids: list[int] = []
        for item in self._list(record.params[1], instance):
            element = self._get(item)
            if not element.complex and element.records[0].name == "AXIS2_PLACEMENT_3D":
                self._frame3(item)
                continue
            if element.complex:
                raise self._refuse(
                    "unsupported-entity",
                    element,
                    "Unsupported complex representation item.",
                )
            solid = self._solid(element)
            if solid is not None:
                solids.append(solid)
        self.units = previous
        self._leave()
        self.representations[identifier] = solids
        return solids

    def _placement(self, reference: Any, context: Any) -> RigidPlacement:
        previous = self.units
        self.units = self._context_units(context)
        origin, x, y, z = self._frame3(reference)
        self.units = previous
        rotation = np.stack((x, y, z), axis=1)
        return RigidPlacement(rotation, origin)

    def _items_context(self, identifier: int) -> Any:
        instance = self.instances[identifier]
        record = next((r for r in instance.records if r.name in _REPRESENTATIONS), None)
        if record is None or len(record.params) != 3:
            raise self._refuse(
                "unsupported-entity", instance, "Expected a shape representation."
            )
        return record.params[2]


@dataclass(slots=True)
class _Assembly:
    """Product structure: shape representations and placed usages."""

    product_reps: dict[int, list[int]]
    identity_links: dict[int, set[int]]
    children: dict[int, list[tuple[int, int]]]
    placements: dict[int, tuple[int, int, int]]


def _instances_named(instances: dict[int, _Instance], name: str, /) -> list[_Instance]:
    return [
        instance for _, instance in sorted(instances.items()) if name in instance.names
    ]


def _product_structure(instances: dict[int, _Instance], /) -> _Assembly:
    """Index product definitions, representation links and assembly usages."""
    product_reps: dict[int, list[int]] = {}
    for instance in _instances_named(instances, "SHAPE_DEFINITION_REPRESENTATION"):
        definition, representation = instance.records[0].params[:2]
        if not isinstance(definition, _Ref) or not isinstance(representation, _Ref):
            raise refuse(
                "malformed", instance.label, "Invalid shape definition representation."
            )
        shape = instances[definition.id]
        target = shape.records[0].params[2] if len(shape.records[0].params) == 3 else None
        if (
            isinstance(target, _Ref)
            and "PRODUCT_DEFINITION" in instances[target.id].names
        ):
            product_reps.setdefault(target.id, []).append(representation.id)
    identity_links: dict[int, set[int]] = {}
    placements: dict[int, tuple[int, int, int]] = {}
    transforms: dict[int, tuple[int, int, int]] = {}
    for instance in _instances_named(instances, "SHAPE_REPRESENTATION_RELATIONSHIP"):
        record = (
            instance.record("REPRESENTATION_RELATIONSHIP")
            if instance.complex
            else instance.records[0]
        )
        first, second = record.params[2:4]
        if not isinstance(first, _Ref) or not isinstance(second, _Ref):
            raise refuse(
                "malformed", instance.label, "Invalid representation relationship."
            )
        if "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" in instance.names:
            operator = instance.record(
                "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"
            ).params[0]
            if not isinstance(operator, _Ref):
                raise refuse(
                    "malformed", instance.label, "Invalid transformation operator."
                )
            transforms[instance.id] = (first.id, second.id, operator.id)
        else:
            identity_links.setdefault(first.id, set()).add(second.id)
            identity_links.setdefault(second.id, set()).add(first.id)
    for instance in _instances_named(instances, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
        relation, shape = instance.records[0].params[:2]
        if (
            not isinstance(relation, _Ref)
            or relation.id not in transforms
            or not isinstance(shape, _Ref)
        ):
            raise refuse(
                "unsupported-entity",
                instance.label,
                "Assembly placement lacks a transformation.",
            )
        usage = instances[shape.id].records[0].params[2]
        if isinstance(usage, _Ref):
            placements[usage.id] = transforms[relation.id]
    children: dict[int, list[tuple[int, int]]] = {}
    for instance in _instances_named(instances, "NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
        params = instance.records[0].params
        if (
            len(params) != 6
            or not isinstance(params[3], _Ref)
            or not isinstance(params[4], _Ref)
        ):
            raise refuse(
                "malformed", instance.label, "Invalid assembly usage occurrence."
            )
        children.setdefault(params[3].id, []).append((instance.id, params[4].id))
    return _Assembly(product_reps, identity_links, children, placements)


def _closure(start: list[int], links: dict[int, set[int]], /) -> list[int]:
    seen = dict.fromkeys(start)
    pending = list(start)
    while pending:
        for linked in sorted(links.get(pending.pop(), ())):
            if linked not in seen:
                seen[linked] = None
                pending.append(linked)
    return sorted(seen)


class _Expansion:
    """Depth-first expansion of the product structure into native occurrences."""

    def __init__(self, decoder: _Decoder, assembly: _Assembly) -> None:
        self.decoder = decoder
        self.assembly = assembly
        self.count = 0

    def solids(self, product: int) -> tuple[list[int], list[int]]:
        representations = [
            identifier
            for identifier in _closure(
                self.assembly.product_reps.get(product, []), self.assembly.identity_links
            )
            if self.decoder.instances[identifier].names & _REPRESENTATIONS
        ]
        solids: list[int] = []
        for identifier in representations:
            solids.extend(
                solid
                for solid in self.decoder._representation(identifier)
                if solid not in solids
            )
        return representations, solids

    def root_identity(self, product: int) -> tuple[tuple[str, ...], str]:
        definition = self.decoder.instances[product]
        record = definition.record("PRODUCT_DEFINITION")
        formation = self.decoder._get(record.params[2])
        formation_record = next(
            record
            for record in formation.records
            if record.name
            in (
                "PRODUCT_DEFINITION_FORMATION",
                "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
            )
        )
        part = self.decoder._get(formation_record.params[2])
        product_record = part.record("PRODUCT")
        identifier = product_record.params[0]
        for kind in ("container", "occurrence", "definition"):
            path = _identity_path(
                identifier, kind, self.decoder.policy.limits.max_depth, part.label
            )
            if path is not None:
                return path, kind
        for name in product_record.params[:2]:
            if isinstance(name, str) and name:
                return (name,), "container"
        return (definition.label,), "container"

    def _usage_placement(self, usage: int, child_reps: list[int]) -> RigidPlacement:
        decoder = self.decoder
        label = decoder.instances[usage].label
        if usage not in self.assembly.placements:
            raise refuse(
                "unsupported-entity",
                label,
                "Assembly usage has no placement.",
                tuple(decoder.chain),
            )
        first, second, operator = self.assembly.placements[usage]
        child, parent = (first, second) if first in child_reps else (second, first)
        if child not in child_reps:
            raise refuse(
                "inconsistent-geometry",
                label,
                "Placement does not relate the used component.",
            )
        transform = decoder._get(_Ref(operator))
        if decoder._simple(transform).name != "ITEM_DEFINED_TRANSFORMATION":
            raise decoder._refuse(
                "unsupported-entity", transform, "Expected ITEM_DEFINED_TRANSFORMATION."
            )
        _, _, source, target = decoder._params(transform, 4)
        if child != first:
            source, target = target, source
        decoder._enter(transform)
        origin = decoder._placement(source, decoder._items_context(child))
        destination = decoder._placement(target, decoder._items_context(parent))
        decoder._leave()
        matrix = (
            destination if origin.is_identity else destination.compose(origin.inverse())
        )
        return RigidPlacement.from_matrix(matrix.matrix(), transform.label)

    def expand(
        self,
        product: int,
        placement: RigidPlacement,
        path: tuple[str, ...],
        active: tuple[int, ...],
        occurrences: list[BRepOccurrence],
        *,
        container_path: tuple[str, ...] | None = None,
    ) -> list[tuple[str, ...]]:
        decoder = self.decoder
        if product in active:
            raise refuse(
                "cyclic-reference",
                decoder.instances[product].label,
                "Cyclic assembly structure.",
                tuple(decoder.chain),
            )
        if len(active) >= decoder.policy.limits.max_depth:
            raise refuse(
                "limit",
                decoder.instances[product].label,
                "Assembly expansion exceeds its reference-depth limit.",
                tuple(decoder.chain),
            )
        _, solids = self.solids(product)
        members: list[tuple[str, ...]] = []
        child_paths: list[tuple[str, ...]] = []
        for local, solid in enumerate(solids):
            suffix = (
                ()
                if container_path is None and len(solids) == 1 and path
                else (f"solid{local}",)
            )
            member = path + suffix
            self.count += 1
            if self.count > decoder.policy.maximum_occurrences:
                raise refuse(
                    "limit",
                    decoder.instances[product].label,
                    "Assembly occurrences exceed their limit.",
                    tuple(decoder.chain),
                )
            occurrences.append(
                BRepOccurrence(
                    member,
                    solid,
                    np.asarray(placement.rotation),
                    np.asarray(placement.translation),
                )
            )
            members.append(member)
        usages = self.assembly.children.get(product, [])
        names = [decoder.instances[usage].records[0].params[0] for usage, _ in usages]
        for position, (usage, child) in enumerate(usages):
            name = names[position]
            if not isinstance(name, str) or not name or names.count(name) != 1:
                name = f"occurrence{position}"
            decoder.chain.append(decoder.instances[usage].label)
            child_reps, child_solids = self.solids(child)
            child_is_container = len(child_solids) != 1 or bool(
                self.assembly.children.get(child)
            )
            if not child_solids and not self.assembly.children.get(child):
                raise refuse(
                    "unsupported-entity",
                    decoder.instances[usage].label,
                    "Placed surface or empty components have no native solid occurrence carrier.",
                    tuple(decoder.chain),
                )
            local = self._usage_placement(usage, child_reps)
            combined = local if placement.is_identity else placement.compose(local)
            kind = "container" if child_is_container else "occurrence"
            declared_path = _identity_path(
                names[position],
                kind,
                decoder.policy.limits.max_depth,
                decoder.instances[usage].label,
            )
            child_path = path + (name,) if declared_path is None else declared_path
            direct_members = self.expand(
                child,
                combined,
                child_path,
                active + (product,),
                occurrences,
                container_path=child_path if child_is_container else None,
            )
            if child_is_container:
                child_paths.append(child_path)
            else:
                members.extend(direct_members)
            decoder.chain.pop()
        if container_path is not None and (members or child_paths):
            decoder._stage().assembly_containers.append(
                BRepAssemblyContainer(container_path, tuple(members), tuple(child_paths))
            )
        return members


def _decode(
    resource: BoundedResource, policy: CadImportPolicy, /
) -> tuple[BRepModel, StepSchema, _Units, dict[str, int], _Decoder, int, int, int]:
    try:
        text = resource.data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise refuse(
            "malformed", "bytes", "STEP files must be UTF-8/ASCII text."
        ) from error
    parser = _Parser(_tokens(text), policy)
    header, instances = parser.parse()
    schema = _schema(header)
    depth, children = _check_graph(instances, policy.limits.max_depth)
    decoder = _Decoder(instances, policy)
    decoder.children = children
    assembly = _product_structure(instances)
    products = set(assembly.product_reps) | set(assembly.children)
    products.update(child for usages in assembly.children.values() for _, child in usages)
    _check_graph(
        {
            product: _Instance(
                product,
                (
                    _Record(
                        "PRODUCT_DEFINITION",
                        tuple(
                            _Ref(child) for _, child in assembly.children.get(product, ())
                        ),
                    ),
                ),
                False,
            )
            for product in products
        },
        policy.limits.max_depth,
    )
    related = {child for usages in assembly.children.values() for _, child in usages}
    roots = [product for product in sorted(products) if product not in related]
    representations = [
        instance
        for instance in sorted(instances.values(), key=lambda item: item.id)
        if instance.names & _REPRESENTATIONS
    ]
    factors = [
        decoder._context_units(decoder._items_context(instance.id)).scale
        for instance in representations
    ]
    if not factors:
        raise refuse(
            "unsupported-entity", "DATA", "The file contains no shape representation."
        )
    # The stage tolerance is relative to the model extent, known before staging.
    extent = max(
        (
            abs(value)
            for instance in _instances_named(instances, "CARTESIAN_POINT")
            if not instance.complex and len(instance.records[0].params) == 2
            for value in _numbers(instance.records[0].params[1])
        ),
        default=1.0,
    )
    stage = CadStage(
        max(1.0, extent * max(factors)),
        policy.pcurve_fit,
        relative_geometric_tolerance=policy.relative_geometric_tolerance,
    )
    decoder.stage = stage
    # Definition numbering, not assembly traversal, determines native rows.
    # Reused placed parts reference the same already-decoded definition.
    for representation in representations:
        decoder._representation(representation.id)
    expansion = _Expansion(decoder, assembly)
    occurrences: list[BRepOccurrence] = []
    represented: set[int] = set()
    root_identities = [expansion.root_identity(root) for root in roots]
    for position, root in enumerate(roots):
        identity, role = root_identities[position]
        if role == "definition":
            continue
        if root_identities.count((identity, role)) != 1:
            identity = (decoder.instances[root].label,)
        is_container = role == "container"
        if not is_container:
            _, direct_solids = expansion.solids(root)
            usages = assembly.children.get(root, ())
            if not (
                (len(direct_solids) == 1 and not usages)
                or (not direct_solids and len(usages) == 1)
            ):
                raise refuse(
                    "malformed",
                    decoder.instances[root].label,
                    "An independent-instance product must describe exactly one body instance.",
                )
        prefix = identity if not is_container or len(roots) > 1 else ()
        expansion.expand(
            root,
            RigidPlacement.identity(),
            prefix,
            (),
            occurrences,
            container_path=identity if is_container else None,
        )
    for product in sorted(assembly.product_reps):
        represented.update(expansion.solids(product)[0])
    for instance in representations:
        if instance.id in represented:
            continue
        for solid in decoder._representation(instance.id):
            if all(occurrence.solid != solid for occurrence in occurrences):
                occurrences.append(BRepOccurrence((f"solid{solid}",), solid))
                if len(occurrences) > policy.maximum_occurrences:
                    raise refuse(
                        "limit",
                        instance.label,
                        "Assembly occurrences exceed their limit.",
                    )
    if not stage.faces and (
        stage.shells
        or stage.solids
        or not any(
            "SHAPE_REPRESENTATION" in instance.names for instance in representations
        )
    ):
        raise refuse(
            "unsupported-entity",
            "DATA",
            "The file contains no admitted B-Rep or empty shape.",
        )
    stage.occurrences = occurrences
    units = decoder._context_units(decoder._items_context(representations[0].id))
    try:
        model = stage.publish(
            coordinate_contract=policy.coordinate_contract,
            source_id=canonical_fingerprint(
                {"kind": "step-source", "sha256": resource.manifest.content_sha256}
            ),
            source_format="step",
            source_digest=resource.manifest.content_sha256,
            import_policy_id=policy.policy_id,
            tessellation=policy.tessellation,
            default_occurrences=False,
        )
    except ValueError as error:
        if isinstance(error, CadInterchangeError):
            raise
        raise refuse("inconsistent-geometry", "DATA", str(error)) from error
    return (
        model,
        schema,
        units,
        decoder.used,
        decoder,
        depth,
        len(instances),
        parser.parameters,
    )


def _numbers(value: Any, /) -> Iterator[float]:
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                yield float(item)


def _import_report(
    model: BRepModel, schema: StepSchema, units: _Units, stage: CadStage, digest: str, /
) -> AdapterReport:
    losses: tuple[AdapterLoss, ...] = ()
    if stage.pcurves_fitted:
        losses = (
            AdapterLoss(
                "geometry.pcurves",
                "import",
                "synthesized",
                f"{stage.pcurves_fitted} absent p-curves were fitted within relative "
                f"error {stage.maximum_fit_error:.3e}.",
                changes_interpretation=False,
            ),
        )
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "step",
        "phydrax-brep",
        source_id=canonical_fingerprint({"kind": "step-source", "sha256": digest}),
        target_id=model.model_id,
        stage="host-import",
        source_profile=AdapterFormatProfile(
            "step",
            qualifiers={
                "schema": schema,
                "length-unit-m": format(float(units.length_meters), ".17g"),
                "plane-angle-unit-rad": format(units.angle_radians, ".17g"),
            },
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-brep",
            qualifiers={"coordinate-contract": model.coordinate_contract.spatial_id},
        ),
        coordinate_mapping=(
            f"source length unit {float(units.length_meters):.17g} m -> "
            f"{model.coordinate_contract.length_unit.symbol}",
            "assembly placements -> native occurrences",
        ),
        preserved_fields=(
            "exact analytic and rational geometry",
            "oriented topology",
            "p-curves",
            "assembly occurrences",
            "units",
        ),
        assumptions=("external instance names are provenance only",),
        losses=losses,
    )


def decode_step_resource(
    resource: BoundedResource, policy: CadImportPolicy, /
) -> CadImportResult:
    """Decode one bounded STEP Part 21 resource into an exact native B-Rep."""
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    if resource.manifest.limits != policy.limits:
        raise ValueError("The bounded resource and import policy limits must match.")
    try:
        model, schema, units, used, decoder, depth, count, parameters = _decode(
            resource, policy
        )
        accounted = account_bounded_resource(
            resource, depth=depth, nodes=count, attributes=parameters, losses=0
        )
    except CadInterchangeError as error:
        error.resource_manifest = resource.manifest
        raise
    except ResourceReadError as error:
        resource_refusal(error)
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        failure = refuse("malformed", "DATA", f"Invalid STEP record: {error}")
        failure.resource_manifest = resource.manifest
        raise failure from error
    stage = decoder._stage()
    report = _import_report(
        model, schema, units, stage, accounted.manifest.content_sha256
    )
    return CadImportResult(
        model,
        "step",
        schema,
        accounted.manifest.content_sha256,
        float(units.length_meters),
        units.angle_radians,
        accounted.manifest,
        report,
        stage.coverage(used),
        tuple(stage.provenance),
    )


def decode_step_bytes(
    data: bytes, policy: CadImportPolicy, /, *, source_path: str | None = None
) -> CadImportResult:
    """Bound and decode exact in-memory STEP bytes."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = bounded_resource_from_bytes(
            data, limits=policy.limits, source_path=source_path
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_step_resource(resource, policy)


def read_step(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
) -> CadImportResult:
    """Descriptor-read and decode one bounded local STEP file."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=policy.limits
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_step_resource(resource, policy)


# -------------------------------------------------------------------- writing


_SCHEMA_IDENTIFIERS: dict[StepSchema, tuple[str, str, str, int]] = {
    "AP203": (
        "CONFIG_CONTROL_DESIGN",
        "config_control_design",
        "config_control_design",
        1994,
    ),
    "AP214": (
        "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }",
        "core data for automotive mechanical design processes",
        "automotive_design",
        2000,
    ),
    "AP242": (
        "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }",
        "managed model based 3d engineering",
        "ap242_managed_model_based_3d_engineering",
        2014,
    ),
}


def _real(value: float, /) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise refuse("inexact-export", "real", "Nonfinite values have no STEP literal.")
    text = repr(number)
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{exponent}"
    return text if "." in text else text + "."


def _string(value: str, /) -> str:
    out = []
    for character in value:
        if character == "'":
            out.append("''")
        elif character == "\\":
            out.append("\\\\")
        elif 32 <= ord(character) < 127:
            out.append(character)
        else:
            out.append(f"\\X2\\{ord(character):04X}\\X0\\")
    return "'" + "".join(out) + "'"


def _refs(values: Sequence[int], /) -> str:
    return "(" + ",".join(f"#{value}" for value in values) + ")"


def _reals(values: Sequence[float] | np.ndarray, /) -> str:
    return "(" + ",".join(_real(value) for value in values) + ")"


def _bool(value: bool, /) -> str:
    return ".T." if value else ".F."


def _knot_vector(knots: np.ndarray, /) -> tuple[list[int], list[float]]:
    values, counts = np.unique(np.asarray(knots, dtype=np.float64), return_counts=True)
    return [int(count) for count in counts], [float(value) for value in values]


def _mapped_pcurve(curve: AbstractCurve, matrix: np.ndarray, /) -> AbstractCurve | None:
    """Image of a native p-curve under the linear native-to-STEP parameter map."""
    if np.array_equal(matrix, np.eye(2)):
        return curve
    match curve:
        case LineCurve():
            return LineCurve(
                matrix @ np.asarray(curve.origin), matrix @ np.asarray(curve.direction)
            )
        case BSplineCurve():
            return BSplineCurve(
                np.asarray(curve.control_points) @ matrix.T,
                curve.weights,
                curve.knots,
                curve.degree,
            )
        case CircleCurve() if _is_rotation(matrix):
            return CircleCurve(
                matrix @ np.asarray(curve.center),
                matrix @ np.asarray(curve.first_axis),
                matrix @ np.asarray(curve.second_axis),
                curve.radius,
            )
        case EllipseCurve() if _is_rotation(matrix):
            return EllipseCurve(
                matrix @ np.asarray(curve.center),
                matrix @ np.asarray(curve.first_axis),
                matrix @ np.asarray(curve.second_axis),
                curve.first_radius,
                curve.second_radius,
            )
        case _:
            return None


def _step_conic(curve: AbstractCurve, /) -> bool:
    """Whether a 2D curve has a STEP parameterization (conics must turn counterclockwise)."""
    if not isinstance(curve, (CircleCurve, EllipseCurve)):
        return True
    first, second = np.asarray(curve.first_axis), np.asarray(curve.second_axis)
    return bool(np.max(np.abs(second - np.asarray((-first[1], first[0])))) <= 1.0e-12)


def _is_rotation(matrix: np.ndarray, /) -> bool:
    return bool(
        np.allclose(matrix.T @ matrix, np.eye(2), rtol=0.0, atol=1.0e-15)
        and np.linalg.det(matrix) > 0.0
    )


class _Writer:
    """Deterministic instance emission of one native B-Rep."""

    def __init__(
        self,
        model: BRepModel,
        geometry: BRepGeometry,
        schema: StepSchema,
        policy: CadExportPolicy,
    ) -> None:
        self.model = model
        self.geometry = geometry
        self.schema = schema
        self.policy = policy
        self.lines: list[str] = []
        self.counts: dict[str, int] = {}
        self.context2 = 0

    def add(self, body: str, /) -> int:
        self.lines.append(body)
        for name in re.findall(r"(?:^|\(|\s)([A-Z_][A-Z0-9_]*)\(", body):
            self.counts[name] = self.counts.get(name, 0) + 1
        if len(self.lines) > self.policy.maximum_entities:
            raise refuse("limit", "export", "Exported instance count exceeds its limit.")
        return len(self.lines)

    # ------------------------------------------------------------ geometry

    def point(self, value: np.ndarray, /) -> int:
        return self.add(f"CARTESIAN_POINT('',{_reals(np.asarray(value))})")

    def direction(self, value: np.ndarray, /) -> int:
        return self.add(f"DIRECTION('',{_reals(np.asarray(value))})")

    def vector(self, value: np.ndarray, /) -> int:
        vector = np.asarray(value, dtype=np.float64)
        magnitude = float(np.linalg.norm(vector))
        return self.add(
            f"VECTOR('',#{self.direction(_unit(vector, 'vector'))},{_real(magnitude)})"
        )

    def frame(self, origin: np.ndarray, axis: np.ndarray, ref: np.ndarray, /) -> int:
        return self.add(
            f"AXIS2_PLACEMENT_3D('',#{self.point(origin)},#{self.direction(axis)},#{self.direction(ref)})"
        )

    def frame2(self, origin: np.ndarray, ref: np.ndarray, /) -> int:
        return self.add(
            f"AXIS2_PLACEMENT_2D('',#{self.point(origin)},#{self.direction(ref)})"
        )

    def spline_curve(self, curve: BSplineCurve, /) -> int:
        points = [self.point(row) for row in np.asarray(curve.control_points)]
        multiplicities, knots = _knot_vector(np.asarray(curve.knots))
        weights = np.asarray(curve.weights)
        head = f"{curve.degree},{_refs(points)},.UNSPECIFIED.,.F.,.F."
        tail = (
            f"({','.join(str(m) for m in multiplicities)}),{_reals(knots)},.UNSPECIFIED."
        )
        if np.all(weights == 1.0):
            return self.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{head},{tail})")
        return self.add(
            f"( BOUNDED_CURVE() B_SPLINE_CURVE({head}) B_SPLINE_CURVE_WITH_KNOTS({tail}) CURVE() "
            f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({_reals(weights)}) REPRESENTATION_ITEM('') )"
        )

    def curve(self, curve: AbstractCurve, label: str, /) -> int:
        match curve:
            case LineCurve():
                return self.add(
                    f"LINE('',#{self.point(np.asarray(curve.origin))},#{self.vector(np.asarray(curve.direction))})"
                )
            case CircleCurve() | EllipseCurve():
                first, second = (
                    np.asarray(curve.first_axis),
                    np.asarray(curve.second_axis),
                )
                if curve.ambient_dimension == 3:
                    placement = self.frame(
                        np.asarray(curve.center),
                        _unit(np.cross(first, second), label),
                        first,
                    )
                else:
                    if (
                        np.max(np.abs(second - np.asarray((-first[1], first[0]))))
                        > 1.0e-12
                    ):
                        raise refuse(
                            "inexact-export",
                            label,
                            "Clockwise 2D conics have no STEP parameterization.",
                        )
                    placement = self.frame2(np.asarray(curve.center), first)
                if isinstance(curve, CircleCurve):
                    return self.add(
                        f"CIRCLE('',#{placement},{_real(float(curve.radius))})"
                    )
                return self.add(
                    f"ELLIPSE('',#{placement},{_real(float(curve.first_radius))},{_real(float(curve.second_radius))})"
                )
            case BSplineCurve():
                return self.spline_curve(curve)
            case _:
                raise refuse(
                    "inexact-export",
                    label,
                    f"{type(curve).__name__} has no exact STEP entity.",
                )

    def surface(
        self, patch: AbstractSurfacePatch, label: str, /
    ) -> tuple[int, np.ndarray]:
        """Surface instance and the linear map from native to STEP parameters."""
        identity = np.eye(2)
        match patch:
            case PlanePatch():
                first, second = (
                    np.asarray(patch.first_axis),
                    np.asarray(patch.second_axis),
                )
                x = _unit(first, label)
                z = _unit(np.cross(first, second), label)
                y = np.cross(z, x)
                matrix = np.asarray(((first @ x, second @ x), (first @ y, second @ y)))
                if np.max(np.abs(matrix - identity)) <= 1.0e-15:
                    matrix = identity
                return self.add(
                    f"PLANE('',#{self.frame(np.asarray(patch.origin), z, x)})"
                ), matrix
            case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
                origin = np.asarray(
                    patch.center
                    if isinstance(patch, (SpherePatch, TorusPatch))
                    else patch.origin
                )
                first, second, axis = (
                    np.asarray(patch.first_axis),
                    np.asarray(patch.second_axis),
                    np.asarray(patch.axis),
                )
                handed = 1.0 if float(np.cross(first, second) @ axis) > 0.0 else -1.0
                flip = 1.0
                if isinstance(patch, ConePatch) and float(patch.semi_angle) < 0.0:
                    flip = -1.0
                placement = self.frame(origin, flip * axis, first)
                matrix = np.diag((handed * flip, flip))
                match patch:
                    case CylinderPatch():
                        body = f"CYLINDRICAL_SURFACE('',#{placement},{_real(float(patch.radius))})"
                    case ConePatch():
                        # STEP measures the second parameter along a generator;
                        # the native cone uses axial height.
                        matrix[1, 1] /= float(np.cos(patch.semi_angle))
                        body = (
                            f"CONICAL_SURFACE('',#{placement},{_real(float(patch.reference_radius))},"
                            f"{_real(abs(float(patch.semi_angle)))})"
                        )
                    case SpherePatch():
                        body = f"SPHERICAL_SURFACE('',#{placement},{_real(float(patch.radius))})"
                    case _:
                        body = (
                            f"TOROIDAL_SURFACE('',#{placement},{_real(float(patch.major_radius))},"
                            f"{_real(float(patch.minor_radius))})"
                        )
                return self.add(body), matrix
            case BSplineSurfacePatch():
                grid = np.asarray(patch.control_points)
                rows = [
                    "(" + ",".join(f"#{self.point(point)}" for point in row) + ")"
                    for row in grid
                ]
                u_mult, u_knots = _knot_vector(np.asarray(patch.u_knots))
                v_mult, v_knots = _knot_vector(np.asarray(patch.v_knots))
                weights = np.asarray(patch.weights)
                head = f"{patch.u_degree},{patch.v_degree},({','.join(rows)}),.UNSPECIFIED.,.F.,.F.,.F."
                tail = (
                    f"({','.join(map(str, u_mult))}),({','.join(map(str, v_mult))}),"
                    f"{_reals(u_knots)},{_reals(v_knots)},.UNSPECIFIED."
                )
                if np.all(weights == 1.0):
                    return self.add(
                        f"B_SPLINE_SURFACE_WITH_KNOTS('',{head},{tail})"
                    ), identity
                weight_rows = "(" + ",".join(_reals(row) for row in weights) + ")"
                return (
                    self.add(
                        f"( BOUNDED_SURFACE() B_SPLINE_SURFACE({head}) B_SPLINE_SURFACE_WITH_KNOTS({tail}) "
                        f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({weight_rows}) "
                        "REPRESENTATION_ITEM('') SURFACE() )"
                    ),
                    identity,
                )
            case ExtrusionSurface():
                swept = self.curve(patch.curve, label)
                return self.add(
                    f"SURFACE_OF_LINEAR_EXTRUSION('',#{swept},#{self.vector(np.asarray(patch.direction))})"
                ), identity
            case RevolutionSurface():
                swept = self.curve(patch.curve, label)
                axis = self.add(
                    f"AXIS1_PLACEMENT('',#{self.point(np.asarray(patch.axis_origin))},"
                    f"#{self.direction(np.asarray(patch.axis_direction))})"
                )
                return self.add(f"SURFACE_OF_REVOLUTION('',#{swept},#{axis})"), identity
            case OffsetSurface():
                basis, matrix = self.surface(patch.base, label)
                # Reversing the parameter chart reverses its oriented normal.
                sense = -1.0 if float(np.linalg.det(matrix)) < 0.0 else 1.0
                return self.add(
                    f"OFFSET_SURFACE('',#{basis},{_real(sense * float(patch.distance))},.U.)"
                ), matrix
            case _:
                raise refuse(
                    "inexact-export",
                    label,
                    f"{type(patch).__name__} has no exact STEP surface.",
                )

    def pcurve(
        self,
        surface: int,
        curve: AbstractCurve,
        matrix: np.ndarray,
        planar: bool,
        label: str,
        /,
    ) -> int:
        """PCURVE of a coedge, or the plane itself when no STEP p-curve exists.

        A curve on a plane is recovered exactly by the reader from its 3D
        carrier, so a clockwise conic p-curve (not a STEP parameterization) is
        written as the bare surface reference; elsewhere it is refused.
        """
        mapped = _mapped_pcurve(curve, matrix)
        if mapped is None or not _step_conic(mapped):
            if planar:
                return surface
            raise refuse(
                "inexact-export", label, "The p-curve has no exact STEP parameterization."
            )
        item = self.curve(mapped, label)
        definition = self.add(
            f"DEFINITIONAL_REPRESENTATION('',(#{item}),#{self.context2})"
        )
        return self.add(f"PCURVE('',#{surface},#{definition})")

    # ----------------------------------------------------------- topology

    def units(self) -> int:
        contract = self.model.coordinate_contract
        scale = contract.length_unit.scale_to_reference
        prefix = si_prefix_for(scale)
        if prefix is not None:
            prefix_text = "$" if prefix == "" else f".{prefix}."
            length = self.add(
                f"( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT({prefix_text},.METRE.) )"
            )
        else:
            metre = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )")
            measure = self.add(
                f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({_real(float(scale))}),#{metre})"
            )
            exponents = self.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
            length = self.add(
                f"( CONVERSION_BASED_UNIT({_string(contract.length_unit.symbol)},#{measure}) "
                f"LENGTH_UNIT() NAMED_UNIT(#{exponents}) )"
            )
        angle = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid_angle = self.add(
            "( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )"
        )
        uncertainty = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length},'distance_accuracy_value','confusion accuracy')"
        )
        return self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) REPRESENTATION_CONTEXT('','') )"
        )

    def shapes(self) -> tuple[list[int], list[int], dict[int, bool]]:
        """Emit exact topology; returns solid items, free shells and shell closure."""
        geometry = self.geometry
        ranges = np.asarray(geometry.edge_ranges)
        model = self.model
        self.context2 = self.add(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() "
            "REPRESENTATION_CONTEXT('2D SPACE','') )"
        )
        vertices = [
            self.add(f"VERTEX_POINT('',#{self.point(point)})")
            for point in np.asarray(geometry.vertex_points)
        ]
        owners = geometry.coedge_faces
        surfaces = [
            self.surface(patch, f"face:{index}")
            for index, patch in enumerate(model.patches)
        ]
        void_shells = {shell for shells in geometry.solid_shells for shell in shells[1:]}
        void_faces = {
            face
            for shells in geometry.solid_shells
            for shell in shells[1:]
            for face in geometry.shell_faces[shell]
        }
        normal_sign = [
            (-1 if face in void_faces else 1) * int(np.asarray(model.orientation)[face])
            for face in range(len(model.patches))
        ]
        uses: dict[int, list[int]] = {}
        for coedge, edge in enumerate(geometry.coedge_edges):
            uses.setdefault(edge, []).append(coedge)
        edges: dict[int, int] = {}
        for edge, curve_index in enumerate(geometry.edge_curves):
            if curve_index == -1:
                continue
            label = f"edge:{edge}"
            coedges = uses.get(edge, [])
            if len(coedges) > 2:
                raise refuse(
                    "non-manifold",
                    label,
                    "STEP surface curves carry at most two p-curves.",
                )
            carrier_curve = geometry.curves[curve_index]
            if not isinstance(carrier_curve, AbstractCurve):
                raise refuse(
                    "unsupported-entity",
                    label,
                    "STEP requires an explicit curve after intersection substitution.",
                )
            curve = self.curve(explicit_carrier(carrier_curve), label)
            first, last = ranges[edge]
            curve = self.add(
                f"TRIMMED_CURVE('',#{curve},(PARAMETER_VALUE({_real(first)})),"
                f"(PARAMETER_VALUE({_real(last)})),.T.,.PARAMETER.)"
            )
            written = sorted(
                coedges,
                key=lambda c: -geometry.coedge_senses[c] * normal_sign[owners[c]],
            )
            pcurves = []
            for c in written:
                parameter_curve = geometry.pcurves[c]
                if not isinstance(parameter_curve, AbstractCurve):
                    raise refuse(
                        "unsupported-entity",
                        f"coedge:{c}",
                        "STEP requires an explicit p-curve after intersection substitution.",
                    )
                pcurves.append(
                    self.pcurve(
                        surfaces[owners[c]][0],
                        explicit_carrier(parameter_curve),
                        surfaces[owners[c]][1],
                        isinstance(model.patches[owners[c]], PlanePatch),
                        f"coedge:{c}",
                    )
                )
            seam = len(coedges) == 2 and owners[coedges[0]] == owners[coedges[1]]
            kind = "SEAM_CURVE" if seam else "SURFACE_CURVE"
            carrier = self.add(f"{kind}('',#{curve},{_refs(pcurves)},.CURVE_3D.)")
            start, end = geometry.edge_vertices[edge]
            edges[edge] = self.add(
                f"EDGE_CURVE('',#{vertices[start]},#{vertices[end]},#{carrier},.T.)"
            )
        faces: list[int] = []
        for face, loops in enumerate(geometry.face_loops):
            sign = normal_sign[face]
            bounds = []
            for position, loop in enumerate(loops):
                ordered = loop if sign > 0 else tuple(reversed(loop))
                oriented = [
                    self.add(
                        f"ORIENTED_EDGE('',*,*,#{edges[geometry.coedge_edges[c]]},"
                        f"{_bool(geometry.coedge_senses[c] * sign > 0)})"
                    )
                    for c in ordered
                    if geometry.edge_curves[geometry.coedge_edges[c]] != -1
                ]
                if not oriented:
                    continue
                loop_id = self.add(f"EDGE_LOOP('',{_refs(oriented)})")
                kind = "FACE_OUTER_BOUND" if position == 0 else "FACE_BOUND"
                bounds.append(self.add(f"{kind}('',#{loop_id},.T.)"))
            surface, matrix = surfaces[face]
            same_sense = sign * float(np.linalg.det(matrix)) > 0.0
            faces.append(
                self.add(
                    f"ADVANCED_FACE('',{_refs(bounds)},#{surface},{_bool(same_sense)})"
                )
            )
        closed: dict[int, bool] = {}
        shells: list[int] = []
        for shell, members in enumerate(geometry.shell_faces):
            member_edges = [
                geometry.coedge_edges[c]
                for face in members
                for loop in geometry.face_loops[face]
                for c in loop
            ]
            closed[shell] = all(
                member_edges.count(edge) == 2
                for edge in set(member_edges)
                if geometry.edge_curves[edge] != -1
            )
            kind = "CLOSED_SHELL" if closed[shell] else "OPEN_SHELL"
            oriented_faces = []
            for face, incidence in zip(
                members, geometry.shell_orientations[shell], strict=True
            ):
                # Void shell wrappers reverse the shell. Compensate any
                # shared face whose intrinsic emitted orientation was flipped
                # for a different (void/non-void) shell use.
                sign = incidence * (
                    -1 if (face in void_faces) != (shell in void_shells) else 1
                )
                oriented_faces.append(
                    faces[face]
                    if sign > 0
                    else self.add(f"ORIENTED_FACE('',*,#{faces[face]},.F.)")
                )
            shells.append(self.add(f"{kind}('',{_refs(oriented_faces)})"))
        shelled_faces = {face for members in geometry.shell_faces for face in members}
        for face in range(len(faces)):
            if face in shelled_faces:
                continue
            shell = len(shells)
            shells.append(self.add(f"OPEN_SHELL('',{_refs([faces[face]])})"))
            closed[shell] = False
        solids: list[int] = []
        in_solids = {shell for group in geometry.solid_shells for shell in group}
        for group in geometry.solid_shells:
            if any(not closed[shell] for shell in group):
                raise refuse(
                    "non-manifold", f"solid:{len(solids)}", "Solid shells must be closed."
                )
            if len(group) == 1:
                solids.append(self.add(f"MANIFOLD_SOLID_BREP('',#{shells[group[0]]})"))
                continue
            voids = [
                self.add(f"ORIENTED_CLOSED_SHELL('',*,#{shells[shell]},.F.)")
                for shell in group[1:]
            ]
            solids.append(
                self.add(f"BREP_WITH_VOIDS('',#{shells[group[0]]},{_refs(voids)})")
            )
        free = [shells[index] for index in range(len(shells)) if index not in in_solids]
        return solids, free, closed

    def product(
        self,
        name: str,
        product_context: int,
        definition_context: int,
        *,
        identity: str | None = None,
    ) -> tuple[int, int]:
        product = self.add(
            f"PRODUCT({_string(name if identity is None else identity)},{_string(name)},'',(#{product_context}))"
        )
        formation = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
        definition = self.add(
            f"PRODUCT_DEFINITION('design','',#{formation},#{definition_context})"
        )
        shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
        return definition, shape

    def encode(self) -> str:
        schema_id, application, protocol, year = _SCHEMA_IDENTIFIERS[self.schema]
        solids, free, _ = self.shapes()
        context = self.units()
        application_context = self.add(f"APPLICATION_CONTEXT({_string(application)})")
        self.add(
            f"APPLICATION_PROTOCOL_DEFINITION('international standard',{_string(protocol)},{year},#{application_context})"
        )
        product_context = self.add(
            f"PRODUCT_CONTEXT('',#{application_context},'mechanical')"
        )
        definition_context = self.add(
            f"PRODUCT_DEFINITION_CONTEXT('part definition',#{application_context},'design')"
        )
        origin = np.zeros(3)
        z, x = np.eye(3)[2], np.eye(3)[0]
        geometry = self.geometry
        default = tuple(
            BRepOccurrence((f"solid{i}",), i) for i in range(len(geometry.solid_shells))
        )
        if geometry.assembly_containers:
            self._assembly(solids, context, product_context, definition_context)
        elif geometry.occurrences == default:
            if solids:
                axis = self.frame(origin, z, x)
                self.add(
                    f"ADVANCED_BREP_SHAPE_REPRESENTATION('',{_refs([axis, *solids])},#{context})"
                )
            elif not free:
                axis = self.frame(origin, z, x)
                self.add(f"SHAPE_REPRESENTATION('',(#{axis}),#{context})")
        else:
            self._assembly(solids, context, product_context, definition_context)
        if free:
            model = self.add(f"SHELL_BASED_SURFACE_MODEL('',{_refs(free)})")
            axis = self.frame(origin, z, x)
            self.add(
                f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',{_refs([axis, model])},#{context})"
            )
        header = (
            "ISO-10303-21;\nHEADER;\n"
            "FILE_DESCRIPTION(('phydrax native B-Rep'),'2;1');\n"
            "FILE_NAME('phydrax-brep','1970-01-01T00:00:00',(''),(''),'phydrax','phydrax','');\n"
            f"FILE_SCHEMA(({_string(schema_id)}));\nENDSEC;\nDATA;\n"
        )
        body = "".join(
            f"#{index}={line};\n" for index, line in enumerate(self.lines, start=1)
        )
        return header + body + "ENDSEC;\nEND-ISO-10303-21;\n"

    def _assembly(
        self,
        solids: list[int],
        context: int,
        product_context: int,
        definition_context: int,
    ) -> None:
        """Write declared membership only; paths and placements never create groups."""
        geometry = self.geometry
        containers = {item.path: item for item in geometry.assembly_containers}
        occurrences = {item.path: item for item in geometry.occurrences}
        owned = {path for item in containers.values() for path in item.member_paths}
        origin, z, x = np.zeros(3), np.eye(3)[2], np.eye(3)[0]
        parts: dict[int, tuple[int, int, int]] = {}
        for index, solid in enumerate(solids):
            definition, shape = self.product(
                f"solid{index}",
                product_context,
                definition_context,
                identity=_path_identifier("definition", (f"solid{index}",)),
            )
            axis = self.frame(origin, z, x)
            representation = self.add(
                f"ADVANCED_BREP_SHAPE_REPRESENTATION('',{_refs([axis, solid])},#{context})"
            )
            self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})")
            parts[index] = (definition, representation, axis)
        for path in sorted(set(occurrences) - owned):
            occurrence = occurrences[path]
            definition, shape = self.product(
                path[-1],
                product_context,
                definition_context,
                identity=_path_identifier("occurrence", path),
            )
            base = self.frame(origin, z, x)
            rotation = np.asarray(occurrence.rotation)
            placement = self.frame(
                np.asarray(occurrence.translation), rotation[:, 2], rotation[:, 0]
            )
            parent_rep = self.add(
                f"SHAPE_REPRESENTATION('',{_refs([base, placement])},#{context})"
            )
            self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{parent_rep})")
            child_definition, child_rep, source_axis = parts[occurrence.solid]
            usage = self.add(
                f"NEXT_ASSEMBLY_USAGE_OCCURRENCE({_string(_path_identifier('occurrence', path))},"
                f"{_string(path[-1])},'',#{definition},#{child_definition},$)"
            )
            usage_shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{usage})")
            transform = self.add(
                f"ITEM_DEFINED_TRANSFORMATION('','',#{source_axis},#{placement})"
            )
            relation = self.add(
                f"( REPRESENTATION_RELATIONSHIP('','',#{child_rep},#{parent_rep}) "
                f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transform}) SHAPE_REPRESENTATION_RELATIONSHIP() )"
            )
            self.add(
                f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relation},#{usage_shape})"
            )
        nodes: dict[tuple[str, ...], tuple[int, int, int, list[int]]] = {}
        for path in sorted(containers):
            definition, shape = self.product(
                path[-1],
                product_context,
                definition_context,
                identity=_path_identifier("container", path),
            )
            axis = self.frame(origin, z, x)
            nodes[path] = (definition, shape, axis, [axis])
        pending: list[tuple[int | tuple[str, ...], int, int, tuple[str, ...]]] = []
        for path, container in sorted(containers.items()):
            parent = nodes[path]
            uses = []
            for member in sorted(container.member_paths):
                occurrence = occurrences[member]
                definition, representation, source_axis = parts[occurrence.solid]
                rotation = np.asarray(occurrence.rotation)
                placement = self.frame(
                    np.asarray(occurrence.translation), rotation[:, 2], rotation[:, 0]
                )
                uses.append(
                    (
                        member[-1],
                        _path_identifier("occurrence", member),
                        definition,
                        representation,
                        source_axis,
                        placement,
                    )
                )
            for child in sorted(container.child_paths):
                definition, _, source_axis, _ = nodes[child]
                placement = self.frame(origin, z, x)
                uses.append(
                    (
                        child[-1],
                        _path_identifier("container", child),
                        definition,
                        child,
                        source_axis,
                        placement,
                    )
                )
            for (
                name,
                identifier,
                definition,
                representation,
                source_axis,
                placement,
            ) in uses:
                parent[3].append(placement)
                usage = self.add(
                    f"NEXT_ASSEMBLY_USAGE_OCCURRENCE({_string(identifier)},{_string(name)},'',"
                    f"#{parent[0]},#{definition},$)"
                )
                shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{usage})")
                transform = self.add(
                    f"ITEM_DEFINED_TRANSFORMATION('','',#{source_axis},#{placement})"
                )
                pending.append((representation, transform, shape, path))
        representations = {}
        for path, (_, shape, _, items) in sorted(nodes.items()):
            representation = self.add(
                f"SHAPE_REPRESENTATION('',{_refs(items)},#{context})"
            )
            self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})")
            representations[path] = representation
        for child, transform, shape, parent in pending:
            child_rep = representations[child] if isinstance(child, tuple) else child
            relation = self.add(
                f"( REPRESENTATION_RELATIONSHIP('','',#{child_rep},#{representations[parent]}) "
                f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transform}) SHAPE_REPRESENTATION_RELATIONSHIP() )"
            )
            self.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relation},#{shape})")


def write_step(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    schema: StepSchema = "AP214",
    policy: CadExportPolicy | None = None,
    mode: PublicationMode = "exclusive",
) -> CadExportResult:
    """Atomically publish a native B-Rep as a deterministic STEP Part 21 file.

    Exact carriers are written unless explicit intersection approximation
    requests coupled B-splines. Substitution requires continuous deviation and
    surface-lift bounds plus embedded source/fit homotopy and face-trim evidence.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    schema_ = parse(schema, StepSchema, "schema")
    policy_ = CadExportPolicy() if policy is None else policy
    if not isinstance(policy_, CadExportPolicy):
        raise TypeError("policy must be a CadExportPolicy or None.")
    if model.geometry is None:
        raise refuse(
            "inexact-export",
            model.source_id,
            "The model has no exact topology; a derived tessellation is never exported as CAD.",
        )
    geometry, _, approximations = prepare_cad_export_geometry(
        model, model.geometry, policy_, format_name="step"
    )
    approximation_policy = policy_.intersection_approximation
    if approximations and approximation_policy is None:
        raise ValueError(
            "Intersection substitutions require an explicit approximation policy."
        )
    writer = _Writer(model, geometry, schema_, policy_)
    text = writer.encode()
    data = text.encode("ascii")
    receipt = publish_bytes(path, data, maximum_bytes=policy_.maximum_bytes, mode=mode)
    losses = tuple(
        AdapterLoss(
            f"geometry.intersection-curve.{approximation.branch_id}",
            "export",
            "transformed",
            f"New approximation {approximation.approximation_id}; continuous 3D "
            f"deviation bound {approximation.continuous_evidence.distance_bound:.17g}, "
            f"UV deviation bound {approximation.continuous_evidence.parameter_bound:.17g}, "
            f"surface-lift bounds ({approximation.continuous_evidence.first_correspondence_bound:.17g}, "
            f"{approximation.continuous_evidence.second_correspondence_bound:.17g}); "
            "continuous embedded source-fit homotopy and exported trim separation.",
            changes_interpretation=True,
        )
        for approximation in approximations
    )
    qualifiers = {
        "schema": schema_,
        "length-unit": model.coordinate_contract.length_unit.symbol,
        "plane-angle-unit": "rad",
    }
    qualifiers.update(approximation_export_qualifiers(approximations))
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "phydrax-brep",
        "step",
        source_id=model.model_id,
        target_id=receipt.content_sha256,
        stage="host-export",
        source_profile=AdapterFormatProfile(
            "phydrax-brep",
            qualifiers={"coordinate-contract": model.coordinate_contract.spatial_id},
        ),
        target_profile=AdapterFormatProfile(
            "step",
            qualifiers=qualifiers,
        ),
        preserved_fields=(
            "exact surface definitions",
            "oriented topology",
            "coedge parameter correspondence",
            "occurrences",
            "units",
        ),
        losses=losses,
        waivers=_export_loss_waivers(losses, policy_, "STEP"),
    )
    return CadExportResult(
        receipt,
        "step",
        schema_,
        report,
        tuple(sorted(writer.counts.items())),
        approximations,
    )


__all__ = [
    "STEP_ENTITY_COVERAGE",
    "StepSchema",
    "decode_step_bytes",
    "decode_step_resource",
    "read_step",
    "write_step",
]

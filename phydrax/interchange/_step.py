#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native ISO 10303-21 (STEP Part 21) B-Rep reader and writer.

The reader lexes and parses the supported AP203/AP214/AP242 geometry, topology,
unit and assembly entities into a native ``BRepModel``. Nested values and
reference graphs are traversed iteratively; dangling and cyclic references are
refused. Any other entity reached from an admitted root is refused with its
dependency chain. The writer emits the same entity set with deterministic
instance numbering and shortest-round-trip real literals.

Canonicalization (documented, stable under repeated round trips): placements
are written as ``(location, axis, ref_direction)`` so a reread frame's second
axis is ``axis x ref_direction``; plane faces use orthonormal frames (their
p-curves are mapped exactly); lines are ``point + t * magnitude * unit
direction``; pole (degenerate) edges are not STEP entities and are re-created
after the regular edges. 2D and 3D offset curves retain their basis
parameterization, with STEP's signed distance mapped to native right-normal
orientation. Assembly definitions and per-use placements remain a shared DAG;
leaf occurrences are enumerated only when a consumer requests them.

Native assembly identities derive from STEP entity references and their
declared product/usage relationships. Product and usage names remain
provenance labels only; display strings and placements never define identity.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._publication import PublicationMode, publish_bytes
from .._strict import StrictModule
from ..geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from ..geometry.brep._model import BRepGeometry, BRepModel
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
    OffsetCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from ..typing import checked, parse
from ._cad import (
    _cad_export_gap_surfaces,
    _certified_cad_export_tolerances,
    _validate_curve_parameter_range,
    build_cad_export_report,
    build_cad_import_tolerance_evidence,
    cad_export_assembly_identifier_losses,
    cad_import_synthesis_losses,
    CadCodecInvariantError,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadImportResult,
    CadImportToleranceEvidence,
    CadInterchangeError,
    CadRefusalReason,
    CadStage,
    curve_parameter,
    explicit_carrier,
    length_factor,
    prepare_cad_export_geometry,
    prepare_cad_export_surfaces,
    refuse,
    si_prefix_for,
    si_prefix_scale,
    StagedCoedge,
    transform_pcurve,
    unrepresentable_cad_tolerance_loss,
)
from ._cad_carriers import RigidPlacement, surface_tag
from ._cad_io import CadResource, decode_cad_bytes, read_cad_file
from ._cad_placement_evidence import certify_placed_model, source_placement_evidence
from ._report import (
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._step_metadata import StepEntityLabels, StepEntityMetadata
from ._step_repair import (
    add_cylinder_seam,
    audit_shells,
    repair_record,
    StepRepairPolicy,
    supporting_plane_sense,
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
            "CARTESIAN_TRANSFORMATION_OPERATOR_3D",
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
            "MAPPED_ITEM",
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
            "REPRESENTATION_MAP",
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
            "UNCERTAINTY_MEASURE_WITH_UNIT",
            "VECTOR",
            "VERTEX_LOOP",
            "VERTEX_POINT",
            "COLOUR_RGB",
            "FILL_AREA_STYLE",
            "FILL_AREA_STYLE_COLOUR",
            "NAME_ATTRIBUTE",
            "PRESENTATION_LAYER_ASSIGNMENT",
            "PRESENTATION_STYLE_ASSIGNMENT",
            "PRODUCT",
            "STYLED_ITEM",
            "SURFACE_SIDE_STYLE",
            "SURFACE_STYLE_FILL_AREA",
            "SURFACE_STYLE_USAGE",
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

# Metadata has no native geometric interpretation. Other unreached records are
# reported as discarded semantics rather than assumed harmless.
_METADATA_ENTITIES = frozenset(
    {
        "APPLICATION_CONTEXT",
        "APPLICATION_PROTOCOL_DEFINITION",
        "PRODUCT",
        "PRODUCT_CONTEXT",
        "PRODUCT_DEFINITION_CONTEXT",
        "PRODUCT_DEFINITION_FORMATION",
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
        "PRODUCT_RELATED_PRODUCT_CATEGORY",
        "PERSON",
        "ORGANIZATION",
        "PERSON_AND_ORGANIZATION",
        "DATE_AND_TIME",
        "CALENDAR_DATE",
        "LOCAL_TIME",
        "COORDINATED_UNIVERSAL_TIME_OFFSET",
        "APPROVAL",
        "APPROVAL_STATUS",
        "DESCRIPTIVE_REPRESENTATION_ITEM",
        "NAME_ATTRIBUTE",
        "PRESENTATION_LAYER_ASSIGNMENT",
        "PRESENTATION_STYLE_BY_CONTEXT",
        "SURFACE_STYLE_RENDERING",
        "SURFACE_STYLE_TRANSPARENT",
        "SURFACE_STYLE_REFLECTANCE_AMBIENT",
        "PRESENTATION_STYLE_ASSIGNMENT",
        "STYLED_ITEM",
        "COLOUR_RGB",
        "CURVE_STYLE",
        "SURFACE_STYLE_USAGE",
        "SURFACE_SIDE_STYLE",
        "SURFACE_STYLE_FILL_AREA",
        "FILL_AREA_STYLE",
        "FILL_AREA_STYLE_COLOUR",
        "DRAUGHTING_PRE_DEFINED_COLOUR",
        "DRAUGHTING_PRE_DEFINED_CURVE_FONT",
        "CURVE_STYLE_FONT",
        "CURVE_STYLE_FONT_PATTERN",
        "POINT_STYLE",
        "TEXT_STYLE",
        "TEXT_STYLE_FOR_DEFINED_FONT",
        "SURFACE_STYLE_BOUNDARY",
        "SURFACE_STYLE_PARAMETER_LINE",
        "SURFACE_STYLE_SILHOUETTE",
        "SURFACE_STYLE_CONTROL_GRID",
        "SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE",
        "SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR",
        "NAMED_UNIT",
        "LENGTH_UNIT",
        "PLANE_ANGLE_UNIT",
        "SOLID_ANGLE_UNIT",
        "MASS_UNIT",
        "TIME_UNIT",
        "THERMODYNAMIC_TEMPERATURE_UNIT",
        "SI_UNIT",
        "CONVERSION_BASED_UNIT",
        "DIMENSIONAL_EXPONENTS",
        "GEOMETRIC_REPRESENTATION_CONTEXT",
        "PARAMETRIC_REPRESENTATION_CONTEXT",
        "REPRESENTATION_CONTEXT",
        "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT",
    }
)
_ASSEMBLY_ENTITIES = frozenset(
    {
        "PRODUCT_DEFINITION",
        "PRODUCT_DEFINITION_SHAPE",
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        "SHAPE_DEFINITION_REPRESENTATION",
        "SHAPE_REPRESENTATION_RELATIONSHIP",
        "REPRESENTATION_RELATIONSHIP",
        "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
        "ITEM_DEFINED_TRANSFORMATION",
    }
)


# --------------------------------------------------------------------- lexing


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
    rb"""
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

type _Token = tuple[str, str, int]

_PAGE_ESCAPE = re.compile(r"\\P[A-I]\\")


def _decode_string(raw: str, position: int, /) -> str:
    """Part 21 string literal body: quote doubling and ``\\`` control directives.

    Directives are matched in place, so decoding is linear in the literal length.
    """
    text = raw[1:-1].replace("''", "'")
    result: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        if text.startswith("\\\\", index):
            result.append("\\")
            index += 2
        elif text.startswith("\\X2\\", index) or text.startswith("\\X4\\", index):
            width = 4 if text[index + 2] == "2" else 8
            end = text.find("\\X0\\", index)
            digits = text[index + 4 : end]
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
            index = end + 4
        elif text.startswith("\\X\\", index) and length - index >= 5:
            result.append(chr(int(text[index + 3 : index + 5], 16)))
            index += 5
        elif text.startswith("\\S\\", index) and length - index >= 4:
            result.append(chr(ord(text[index + 3]) + 128))
            index += 4
        elif _PAGE_ESCAPE.match(text, index):
            index += 4
        else:
            raise refuse("malformed", f"offset {position}", "Invalid string escape.")
    return "".join(result)


def _tokens(data: bytes | mmap.mmap, /) -> list[_Token]:
    """Index tokens from the caller-owned buffer without copying it whole."""
    tokens: list[_Token] = []
    position = 0
    while position < len(data):
        match = _TOKEN.match(data, position)
        if match is None:
            raise refuse(
                "malformed",
                f"offset {position}",
                "Unrecognized Part 21 token or unterminated literal.",
            )
        kind = match.lastgroup
        if kind is None:
            raise CadCodecInvariantError(
                "STEP.token.kind must name an alternative; got None"
            )
        if kind != "space":
            try:
                text = match.group().decode("ascii")
            except UnicodeDecodeError as error:
                raise refuse(
                    "malformed",
                    f"offset {position}",
                    "STEP tokens must use the Part 21 ASCII character repertoire.",
                ) from error
            tokens.append((kind, text, position))
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
    """Iterative value parser and indexed Part 21 instance graph builder."""

    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.index = 0

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

    def _value(self) -> Any:
        frames: list[tuple[str, str | None, list[Any] | None]] = []
        need_value = True
        value: Any = None
        while True:
            if need_value:
                kind, text, position = self._peek()
                match kind:
                    case "punct" if text == "(":
                        self.index += 1
                        if self._peek()[1] == ")":
                            self.index += 1
                            value = ()
                            need_value = False
                        else:
                            frames.append(("list", None, []))
                        continue
                    case "punct" if text == "$":
                        self.index += 1
                        value = None
                    case "punct" if text == "*":
                        self.index += 1
                        value = _DERIVED
                    case "ref":
                        self.index += 1
                        value = _Ref(int(text[1:]))
                    case "real":
                        self.index += 1
                        value = float(text)
                    case "int":
                        self.index += 1
                        value = int(text)
                    case "string":
                        self.index += 1
                        value = _decode_string(text, position)
                    case "binary":
                        self.index += 1
                        value = _Binary(text[1:-1])
                    case "enum":
                        self.index += 1
                        value = _Enum(text[1:-1].upper())
                    case "keyword":
                        self.index += 1
                        self._take("(")
                        frames.append(("typed", text.upper(), None))
                        continue
                    case _:
                        raise refuse(
                            "malformed",
                            f"offset {position}",
                            f"Unexpected token {text!r}.",
                        )
                need_value = False
                continue
            if not frames:
                return value
            frame_kind, frame_name, frame_values = frames[-1]
            if frame_kind == "typed":
                self._take(")")
                frames.pop()
                if frame_name is None:
                    raise CadCodecInvariantError("STEP.typed.name cannot be absent.")
                value = _Typed(frame_name, value)
                continue
            if frame_values is None:
                raise CadCodecInvariantError("STEP.list.values cannot be absent.")
            frame_values.append(value)
            if self._peek()[1] == ",":
                self.index += 1
                need_value = True
                continue
            self._take(")")
            frames.pop()
            value = tuple(frame_values)

    def _list(self) -> tuple[Any, ...]:
        value = self._value()
        if not isinstance(value, tuple):
            raise CadCodecInvariantError("STEP.record.parameters must be a list.")
        return value

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
                names: set[str] = set()
                repeated: set[str] = set()
                for record in records:
                    (repeated if record.name in names else names).add(record.name)
                if repeated:
                    raise refuse(
                        "malformed",
                        reference[1],
                        f"STEP.complex_instance.partials must name each entity once; got repeated {sorted(repeated)}.",
                    )
                instance = _Instance(identifier, tuple(records), True)
            else:
                instance = _Instance(identifier, (self._record(),), False)
            self._take(";")
            instances[identifier] = instance
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
    pending = [value]
    while pending:
        current = pending.pop()
        match current:
            case _Ref():
                yield current.id
            case tuple():
                pending.extend(reversed(current))
            case _Typed():
                pending.append(current.value)


def _check_graph(
    instances: dict[int, _Instance], /
) -> tuple[int, dict[int, tuple[int, ...]]]:
    """Reject dangling and cyclic references using an explicit DFS stack."""
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
                    active.add(child)
                    stack.append((child, 0))
                continue
            depth[node] = 1 + max((depth[child] for child in successors), default=0)
            deepest = max(deepest, depth[node])
            active.discard(node)
            stack.pop()
    return deepest, children


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


def _composes_without_rounding(
    outer: RigidPlacement, inner: RigidPlacement, composed: RigidPlacement, /
) -> bool:
    """Whether ``composed`` represents ``outer`` after ``inner`` in rationals."""
    first, second, result = outer.matrix(), inner.matrix(), composed.matrix()
    for row in range(3):
        for column in range(4):
            value = sum(
                (
                    Fraction(float(first[row, axis]))
                    * Fraction(float(second[axis, column]))
                    for axis in range(3)
                ),
                Fraction(0) if column < 3 else Fraction(float(first[row, 3])),
            )
            if value != Fraction(float(result[row, column])):
                return False
    return True


class _Decoder:
    """Map an admitted instance graph to a `CadStage`."""

    def __init__(
        self,
        instances: dict[int, _Instance],
        policy: CadImportPolicy,
        repair_policy: StepRepairPolicy,
    ) -> None:
        self.instances = instances
        self.policy = policy
        self.repair_policy = repair_policy
        self.repairs: list[AdapterLoss] = []
        self.shell_labels: list[str] = []
        self.orientation_assumptions: tuple[str, ...] = ()
        self.sense_vertices: tuple[np.ndarray, ...] = ()
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
        self.mapped_items: dict[int, list[tuple[int, int, RigidPlacement]]] = {}
        self.occurrence_counts: dict[int, int] = {}
        self.unit_cache: dict[int, _Units] = {}
        self.entity_units: dict[int, _Units] = {}
        self.children: dict[int, tuple[int, ...]] = {}
        self.curve_trims: dict[int, tuple[tuple[float, float], bool]] = {}
        self.source_uncertainties: dict[int, float] = {}
        self.raw_source_uncertainties: dict[str, float] = {}
        self.source_uncertainty_units: dict[int, float] = {}
        self.unrepresentable_uncertainties: dict[int, str] = {}
        self.ignored_units: set[int] = set()

    # -------------------------------------------------------------- access

    def _refuse(
        self, reason: CadRefusalReason, instance: _Instance | str, message: str
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
            raise CadCodecInvariantError(
                "STEP.decoder.units must contain the geometry context; got None"
            )
        return self.units

    # --------------------------------------------------------------- units

    def _unit_value(self, reference: Any, kind: str) -> Fraction:
        instance = self._get(reference)
        self._enter(instance)
        names = instance.names
        if kind not in names:
            raise self._refuse("units", instance, f"Expected a {kind}.")
        extras = names - {"NAMED_UNIT", "SI_UNIT", "CONVERSION_BASED_UNIT", kind}
        if extras or {"SI_UNIT", "CONVERSION_BASED_UNIT"} <= names:
            raise self._refuse(
                "units",
                instance,
                f"STEP.unit must contain supported unit partial entities; got {sorted(names)}.",
            )
        if "SI_UNIT" in names:
            prefix, name = instance.record("SI_UNIT").params
            expected = "METRE" if kind == "LENGTH_UNIT" else "RADIAN"
            if not isinstance(name, _Enum) or name.value != expected:
                raise self._refuse("units", instance, f"Unsupported SI unit for {kind}.")
            if prefix is not None and not isinstance(prefix, _Enum):
                raise self._refuse(
                    "units",
                    instance,
                    f"STEP.unit.prefix must be an SI enumeration or absent; got {prefix!r}.",
                )
            value = si_prefix_scale(prefix.value if prefix is not None else None)
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
        if value <= 0:
            raise self._refuse(
                "units",
                instance,
                f"STEP.unit.scale must be positive; got {value}.",
            )
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
                ignored = self._get(unit)
                self.ignored_units.add(ignored.id)
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
        if "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT" in instance.names:
            self._context_uncertainties(instance)
        self._leave()
        self.unit_cache[instance.id] = units
        return units

    def _context_uncertainties(self, context: _Instance) -> None:
        references = self._list(
            context.record("GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT").params[0], context
        )
        for reference in references:
            self._record_uncertainty(reference)

    def _record_uncertainty(self, reference: Any) -> None:
        uncertainty = self._get(reference)
        if f"#{uncertainty.id}" in self.raw_source_uncertainties:
            return
        self._enter(uncertainty)
        if self._simple(uncertainty).name != "UNCERTAINTY_MEASURE_WITH_UNIT":
            raise self._refuse(
                "unsupported-entity",
                uncertainty,
                "Expected UNCERTAINTY_MEASURE_WITH_UNIT.",
            )
        magnitude, unit, _, _ = self._params(uncertainty, 4)
        if not isinstance(magnitude, _Typed) or magnitude.name != "LENGTH_MEASURE":
            raise self._refuse(
                "units",
                uncertainty,
                "uncertainty.measure must be LENGTH_MEASURE; got incompatible measure",
            )
        value = self._float(magnitude, uncertainty)
        if value < 0.0:
            raise self._refuse(
                "units",
                uncertainty,
                f"uncertainty.value must be nonnegative; got {value}",
            )
        unit_meters = self._unit_value(unit, "LENGTH_UNIT")
        converted = value * length_factor(unit_meters, self.policy.coordinate_contract)
        if np.isfinite(converted):
            self.source_uncertainties[uncertainty.id] = converted
        else:
            self.unrepresentable_uncertainties[uncertainty.id] = self._get(unit).label
        self.raw_source_uncertainties[f"#{uncertainty.id}"] = value
        self.source_uncertainty_units[uncertainty.id] = float(unit_meters)
        self._leave()

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

    def _direction(
        self,
        reference: Any,
        dimension: int,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        instance = self._get(reference)
        if self._simple(instance).name != "DIRECTION":
            raise self._refuse("unsupported-entity", instance, "Expected DIRECTION.")
        values = self._floats(self._params(instance, 2)[1], instance)
        if values.shape != (dimension,):
            raise self._refuse(
                "malformed", instance, f"Expected a {dimension}D direction."
            )
        return _unit(values, instance.label) if normalize else values

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
        except CadInterchangeError:
            raise
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _curve_leaf(self, instance: _Instance, dimension: int, /) -> AbstractCurve:
        if "B_SPLINE_CURVE_WITH_KNOTS" in instance.names:
            return self._spline_curve(instance, dimension)
        record = self._simple(instance)
        scaled = dimension == 3
        factor = self._scale.scale if scaled else 1.0
        try:
            match record.name:
                case "LINE":
                    _, point, vector = self._params(instance, 3)
                    return LineCurve(
                        self._point(point, dimension, scaled),
                        self._vector(vector, dimension, scaled),
                    )
                case "CIRCLE" | "ELLIPSE":
                    if dimension == 3:
                        origin, x, y, _ = self._frame3(record.params[1])
                    else:
                        origin, x, y = self._frame2(record.params[1])
                    radii = [
                        self._float(item, instance) * factor for item in record.params[2:]
                    ]
                    if record.name == "CIRCLE" and len(radii) == 1:
                        return CircleCurve(origin, x, y, radii[0])
                    if record.name == "ELLIPSE" and len(radii) == 2:
                        return EllipseCurve(origin, x, y, radii[0], radii[1])
                    raise self._refuse("malformed", instance, "Invalid conic parameters.")
                case _:
                    raise self._refuse(
                        "unsupported-entity",
                        instance,
                        f"STEP.curve must have a native carrier; got {record.name}.",
                    )
        except CadInterchangeError:
            raise
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _curve(self, reference: Any, dimension: int) -> AbstractCurve:
        """Resolve nested STEP curve carriers without using Python recursion."""
        frames: list[tuple[_Instance, str, tuple[Any, ...]]] = []
        current = reference
        entered = 0
        try:
            while True:
                instance = self._get(current)
                self._enter(instance)
                entered += 1
                try:
                    if "B_SPLINE_CURVE_WITH_KNOTS" in instance.names:
                        curve = self._spline_curve(instance, dimension)
                        self._leave()
                        entered -= 1
                        break
                    record = self._simple(instance)
                    if record.name in ("OFFSET_CURVE_2D", "OFFSET_CURVE_3D"):
                        is_planar = record.name == "OFFSET_CURVE_2D"
                        if is_planar != (dimension == 2):
                            raise self._refuse(
                                "malformed",
                                instance,
                                f"{record.name} requires dimension {2 if is_planar else 3}.",
                            )
                        if is_planar:
                            _, basis, distance_, self_intersect = self._params(
                                instance, 4
                            )
                            direction = None
                        else:
                            (
                                _,
                                basis,
                                distance_,
                                self_intersect,
                                ref_direction,
                            ) = self._params(instance, 5)
                            direction = self._direction(ref_direction, 3)
                        if not isinstance(
                            self_intersect, _Enum
                        ) or self_intersect.value not in ("T", "F", "U"):
                            raise self._refuse(
                                "malformed",
                                instance,
                                "Invalid offset-curve self_intersect logical.",
                            )
                        factor = self._scale.scale if dimension == 3 else 1.0
                        distance = -self._float(distance_, instance) * factor
                        frames.append((instance, "offset", (distance, direction)))
                        current = basis
                        continue
                    if record.name == "TRIMMED_CURVE":
                        _, basis, trim1, trim2, sense_, preference = self._params(
                            instance, 6
                        )
                        frames.append(
                            (instance, "trim", (basis, trim1, trim2, sense_, preference))
                        )
                        current = basis
                        continue
                    curve = self._curve_leaf(instance, dimension)
                    self._leave()
                    entered -= 1
                    break
                except CadInterchangeError:
                    raise
                except ValueError as error:
                    raise self._refuse("malformed", instance, str(error)) from error

            while frames:
                instance, kind, values = frames.pop()
                try:
                    if kind == "offset":
                        distance, direction = values
                        curve = OffsetCurve(curve, distance, direction)
                    elif kind == "trim":
                        basis, trim1, trim2, sense_, preference = values
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
                        if curve.period is not None and last <= first:
                            last += curve.period * (
                                np.floor((first - last) / curve.period) + 1
                            )
                        _validate_curve_parameter_range(
                            curve, first, last, instance.label
                        )
                        self.curve_trims[instance.id] = ((first, last), sense)
                    else:
                        raise CadCodecInvariantError(
                            f"STEP.curve.wrapper must be offset or trim; got {kind!r}"
                        )
                except CadInterchangeError:
                    raise
                except ValueError as error:
                    raise self._refuse("malformed", instance, str(error)) from error
                self._leave()
                entered -= 1
            return curve
        finally:
            while entered:
                self._leave()
                entered -= 1

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
            match selector:
                case _Typed(name="PARAMETER_VALUE"):
                    parameters.append(
                        self._float(selector.value, instance)
                        * self._parameter_factor(curve)
                    )
                case _Ref():
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
                case _:
                    raise self._refuse(
                        "malformed", instance, "Invalid trimming selector."
                    )
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
        while isinstance(curve, OffsetCurve):
            curve = curve.base
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
        except CadInterchangeError:
            raise
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _surface_leaf(
        self, instance: _Instance, record: _Record, /
    ) -> tuple[AbstractSurfacePatch, tuple[float, float]]:
        angle, scale = self._scale.angle_radians, self._scale.scale
        params = record.params
        try:
            match record.name:
                case "PLANE":
                    origin, x, y, _ = self._frame3(params[1])
                    return PlanePatch(origin, x, y), (scale, scale)
                case "CYLINDRICAL_SURFACE":
                    origin, x, y, z = self._frame3(params[1])
                    radius = self._float(params[2], instance) * scale
                    return CylinderPatch(origin, x, y, z, radius), (angle, scale)
                case "CONICAL_SURFACE":
                    origin, x, y, z = self._frame3(params[1])
                    radius = self._float(params[2], instance) * scale
                    semi = self._float(params[3], instance) * angle
                    return (
                        ConePatch(origin, x, y, z, radius, semi),
                        (angle, scale * float(np.cos(semi))),
                    )
                case "SPHERICAL_SURFACE":
                    origin, x, y, z = self._frame3(params[1])
                    radius = self._float(params[2], instance) * scale
                    return SpherePatch(origin, x, y, z, radius), (angle, angle)
                case "TOROIDAL_SURFACE":
                    origin, x, y, z = self._frame3(params[1])
                    major = self._float(params[2], instance) * scale
                    minor = self._float(params[3], instance) * scale
                    return TorusPatch(origin, x, y, z, major, minor), (angle, angle)
                case "SURFACE_OF_LINEAR_EXTRUSION":
                    curve = self._curve(params[1], 3)
                    patch = ExtrusionSurface(curve, self._vector(params[2], 3, True))
                    return patch, (self._parameter_factor(curve), 1.0)
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
                    return patch, (angle, self._parameter_factor(curve))
                case _:
                    raise self._refuse(
                        "unsupported-entity",
                        instance,
                        f"STEP.surface must have a native carrier; got {record.name}.",
                    )
        except CadInterchangeError:
            raise
        except ValueError as error:
            raise self._refuse("malformed", instance, str(error)) from error

    def _surface(
        self, reference: Any
    ) -> tuple[AbstractSurfacePatch, tuple[float, float]]:
        """Resolve nested STEP surface carriers without using Python recursion."""
        frames: list[tuple[_Instance, Any]] = []
        current = reference
        entered = 0
        try:
            while True:
                instance = self._get(current)
                self._enter(instance)
                entered += 1
                try:
                    if "B_SPLINE_SURFACE_WITH_KNOTS" in instance.names:
                        patch, factors = self._spline_surface(instance), (1.0, 1.0)
                        self._leave()
                        entered -= 1
                        break
                    record = self._simple(instance)
                    if record.name == "OFFSET_SURFACE":
                        _, basis, distance, self_intersect = self._params(instance, 4)
                        if not isinstance(
                            self_intersect, _Enum
                        ) or self_intersect.value not in ("T", "F", "U"):
                            raise self._refuse(
                                "malformed",
                                instance,
                                "Expected a logical self-intersection parameter.",
                            )
                        frames.append((instance, distance))
                        current = basis
                        continue
                    patch, factors = self._surface_leaf(instance, record)
                    self._leave()
                    entered -= 1
                    break
                except CadInterchangeError:
                    raise
                except ValueError as error:
                    raise self._refuse("malformed", instance, str(error)) from error

            while frames:
                instance, distance = frames.pop()
                try:
                    patch = OffsetSurface(
                        patch, self._float(distance, instance) * self._scale.scale
                    )
                except CadInterchangeError:
                    raise
                except ValueError as error:
                    raise self._refuse("malformed", instance, str(error)) from error
                self._leave()
                entered -= 1
            return patch, factors
        finally:
            while entered:
                self._leave()
                entered -= 1

    # ------------------------------------------------------------ topology

    def _stage(self) -> CadStage:
        if self.stage is None:
            raise CadCodecInvariantError(
                "STEP.decoder.stage must contain topology staging; got None"
            )
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
                endpoints = np.asarray(
                    curve.evaluate(jnp.asarray((lower, upper), dtype=jnp.float64)),
                    dtype=np.float64,
                )
                vertices = np.asarray([stage.vertices[index] for index in directed])
                gaps = np.linalg.norm(endpoints - vertices, axis=1)
                if np.all(gaps <= stage.tolerance):
                    parameter_range = (lower, upper)
                else:
                    first, first_gap = curve_parameter(curve, vertices[0])
                    last, last_gap = curve_parameter(curve, vertices[1])
                    if max(first_gap, last_gap) > stage.tolerance:
                        raise ValueError(
                            f"STEP.edge.vertex_gap must not exceed {stage.tolerance}; got {max(first_gap, last_gap)}"
                        )
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
                            f"STEP.edge.range must lie within ({lower}, {upper}); got ({first}, {last})"
                        )
                    parameter_range = (first, last)
            index = stage.edge(
                curve,
                *directed,
                instance.label,
                parameter_range=parameter_range,
            )
        except CadInterchangeError:
            raise
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
                except CadInterchangeError:
                    raise
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
        if not coedges and not self.repair_policy.add_missing_seams:
            raise self._refuse("malformed", instance, "An edge loop requires edges.")
        return coedges

    def _repair_face_sense(
        self,
        patch: AbstractSurfacePatch,
        sign: int,
        same_sense: bool,
        instance: _Instance,
    ) -> bool:
        if not self.repair_policy.orient_face_senses:
            return same_sense
        if not isinstance(patch, PlanePatch):
            raise self._refuse(
                "unsupported-entity",
                instance,
                "Certified face-sense repair supports only closed convex linear shells.",
            )
        outward = supporting_plane_sense(patch, self.sense_vertices, instance.label)
        corrected = outward * sign > 0
        if corrected != same_sense:
            self.repairs.append(
                repair_record(
                    instance.label,
                    "Oriented face sense by a rational supporting-plane outward test; "
                    "all shell vertices lie on the interior side and at least one strictly so; "
                    "carriers and tolerance unchanged.",
                )
            )
        return corrected

    def _repair_face_seams(
        self,
        patch: AbstractSurfacePatch,
        loops: list[list[StagedCoedge]],
        outer_known: bool,
        instance: _Instance,
    ) -> tuple[list[list[StagedCoedge]], bool]:
        if not self.repair_policy.add_missing_seams:
            return loops, outer_known
        stage = self._stage()
        if (
            patch.periods[0] is not None
            and len(loops) == 2
            and all(len(loop) == 1 for loop in loops)
        ):
            if not isinstance(patch, CylinderPatch):
                raise self._refuse(
                    "unsupported-entity",
                    instance,
                    "Missing seam between two circular bounds is supported only on a native cylinder.",
                )
            loops = [add_cylinder_seam(stage, patch, loops, instance.label)]
            outer_known = True
            self.repairs.append(
                repair_record(
                    instance.label,
                    "Added a longitudinal periodic seam between two coaxial full-circle bounds; "
                    "paired native-period chart representatives share one unchanged 3D carrier; "
                    "no geometry or tolerance changed.",
                )
            )
        if loops == [[]]:
            if not isinstance(patch, (SpherePatch, TorusPatch)):
                raise self._refuse(
                    "unsupported-entity",
                    instance,
                    "Missing natural seam repair requires a certified sphere or torus periodic chart.",
                )
            loops = [stage.natural_loop(patch, instance.label)]
            outer_known = True
            self.repairs.append(
                repair_record(
                    instance.label,
                    "Added the missing natural periodic seam and its paired chart representatives "
                    "on the source sphere/torus; native symbolic periods prove coincident carriers; "
                    "no geometry or tolerance changed.",
                )
            )
        return loops, outer_known

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
        face_name, bounds, surface_, same_sense_ = self._params(instance, 4)
        patch, factors = self._surface(surface_)
        if not isinstance(surface_, _Ref):
            raise CadCodecInvariantError(
                f"STEP.face.surface must be an instance reference; got {surface_!r}"
            )
        face_tag = (
            face_name if isinstance(face_name, str) and face_name else surface_tag(patch)
        )
        same_sense = self._bool(same_sense_, instance)
        same_sense = self._repair_face_sense(patch, sign, same_sense, instance)
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
        loops, outer_known = self._repair_face_seams(patch, loops, outer_known, instance)
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
            face_tag,
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
        if self.repair_policy.orient_face_senses:
            reachable = {instance.id}
            pending = [instance.id]
            while pending:
                for child in self.children.get(pending.pop(), ()):
                    if child not in reachable:
                        reachable.add(child)
                        pending.append(child)
            stage = self._stage()
            if record.name != "CLOSED_SHELL" or any(
                not isinstance(stage.curves[stage.edge_curves[edge.index]], LineCurve)
                for identifier, edge in self.edges.items()
                if identifier in reachable
            ):
                raise self._refuse(
                    "unsupported-entity",
                    instance,
                    "Face-sense repair requires a closed convex shell with linear edges.",
                )
            self.sense_vertices = tuple(
                stage.vertices[index]
                for identifier, index in self.vertices.items()
                if identifier in reachable
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
        self.shell_labels.append(instance.label)
        self._leave()
        return len(stage.shells) - 1

    def _prestage(self, root: int) -> None:
        """Stage reachable vertices and edges in authored file order."""
        reachable = {root}
        pending = [root]
        while pending:
            for child in self.children.get(pending.pop(), ()):
                if child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        for kind in ("VERTEX_POINT", "EDGE_CURVE"):
            for identifier, instance in self.instances.items():
                if identifier not in reachable:
                    continue
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
        if self.repair_policy.orient_face_senses and record.name == "BREP_WITH_VOIDS":
            raise self._refuse(
                "unsupported-entity",
                instance,
                "Supporting-plane face-sense repair does not support cavity shells.",
            )
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
        """Stage representation definitions with an explicit dependency stack."""
        if identifier in self.representations:
            return self.representations[identifier]
        direct: dict[int, list[int]] = {}
        active: set[int] = set()
        pending: list[tuple[int, bool]] = [(identifier, False)]
        while pending:
            current, exiting = pending.pop()
            if exiting:
                self.representations[current] = direct[current]
                active.remove(current)
                continue
            if current in self.representations:
                continue
            if current in active:
                raise self._refuse(
                    "cyclic-reference",
                    self.instances[current],
                    "Mapped representation dependencies must be acyclic.",
                )
            instance = self._get(_Ref(current))
            self._enter(instance)
            previous = self.units
            solids: list[int] = []
            mapped_representations: list[int] = []
            try:
                record = next(
                    record
                    for record in instance.records
                    if record.name in _REPRESENTATIONS
                )
                if len(record.params) != 3:
                    raise self._refuse(
                        "malformed",
                        instance,
                        "A representation has name, items and context.",
                    )
                self.units = self._context_units(record.params[2])
                for item in self._list(record.params[1], instance):
                    element = self._get(item)
                    if (
                        not element.complex
                        and element.records[0].name == "AXIS2_PLACEMENT_3D"
                    ):
                        self._frame3(item)
                        continue
                    if element.records[0].name == "MAPPED_ITEM":
                        mapped_rep, placement = self._mapped_item(
                            element, record.params[2]
                        )
                        self.mapped_items.setdefault(current, []).append(
                            (element.id, mapped_rep, placement)
                        )
                        mapped_representations.append(mapped_rep)
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
            finally:
                self.units = previous
                self._leave()
            direct[current] = solids
            active.add(current)
            pending.append((current, True))
            for mapped_rep in reversed(mapped_representations):
                if mapped_rep in active:
                    raise self._refuse(
                        "cyclic-reference",
                        self.instances[mapped_rep],
                        "Mapped representation dependencies must be acyclic.",
                    )
                if mapped_rep not in self.representations:
                    pending.append((mapped_rep, False))
        return self.representations[identifier]

    def _mapped_item(
        self, instance: _Instance, context: Any
    ) -> tuple[int, RigidPlacement]:
        """Decode a unit-scale map with the declared rigid-placement certificate."""
        _, source, target = self._params(instance, 3)
        mapping = self._get(source)
        if self._simple(mapping).name != "REPRESENTATION_MAP":
            raise self._refuse(
                "unsupported-entity", mapping, "Expected REPRESENTATION_MAP."
            )
        origin, representation = self._params(mapping, 2)
        mapped = self._get(representation)
        source_placement = self._placement(origin, self._items_context(mapped.id))
        operator = self._get(target)
        if self._simple(operator).name != "CARTESIAN_TRANSFORMATION_OPERATOR_3D":
            raise self._refuse(
                "unsupported-entity",
                operator,
                "Only rigid 3D mapping operators are supported.",
            )
        _, axis1, axis2, location, scale, axis3 = self._params(operator, 6)
        if scale is not None and self._float(scale, operator) != 1.0:
            raise self._refuse(
                "unsupported-entity",
                operator,
                "A mapped occurrence requires exactly unit scale.",
            )
        axes = tuple(
            np.eye(3)[i] if axis is None else self._direction(axis, 3, normalize=False)
            for i, axis in enumerate((axis1, axis2, axis3))
        )
        if axis1 is None and axes[2][1] == 0.0 and axes[2][2] == 0.0:
            axes = (np.eye(3)[1], axes[1], axes[2])
        previous = self.units
        self.units = self._context_units(context)
        translation = self._point(location, 3, True)
        self.units = previous
        destination = self._source_frame(
            axes[0], axes[2], translation, operator.label, second_axis=axes[1]
        )
        return mapped.id, self._compose(
            destination,
            source_placement.inverse(),
            instance.label,
        )

    def _compose(
        self, outer: RigidPlacement, inner: RigidPlacement, label: str
    ) -> RigidPlacement:
        """Compose poses; a zero placement-deviation limit refuses rounding.

        Positive limits carry composition error in the retained enclosures.
        Zero-width binary64 frames are signed permutations, so their inverses
        are representable and only composition can round.
        """
        composed = outer.compose(inner)
        if self.policy.maximum_placement_deviation > 0.0 or _composes_without_rounding(
            outer, inner, composed
        ):
            return composed
        raise self._refuse(
            "unsupported-entity",
            label,
            "STEP.placement.composition must have zero deviation under a zero placement-deviation limit; got a rounded composed pose.",
        )

    def _placement(self, reference: Any, context: Any) -> RigidPlacement:
        """Certify an ``AXIS2_PLACEMENT_3D`` from its source direction ratios."""
        previous = self.units
        self.units = self._context_units(context)
        instance = self._get(reference)
        if self._simple(instance).name != "AXIS2_PLACEMENT_3D":
            raise self._refuse(
                "unsupported-entity",
                instance,
                "STEP.placement must be AXIS2_PLACEMENT_3D; got an unsupported source frame.",
            )
        _, location, axis, ref = self._params(instance, 4)
        origin = self._point(location, 3, True)
        z = np.eye(3)[2] if axis is None else self._direction(axis, 3, normalize=False)
        x = np.eye(3)[0] if ref is None else self._direction(ref, 3, normalize=False)
        if ref is None and z[1] == 0.0 and z[2] == 0.0:
            x = np.eye(3)[1]
        self.units = previous
        return self._source_frame(x, z, origin, instance.label)

    def _source_frame(
        self,
        first_axis: np.ndarray,
        third_axis: np.ndarray,
        translation: np.ndarray,
        label: str,
        *,
        second_axis: np.ndarray | None = None,
    ) -> RigidPlacement:
        """Certify a source frame and enforce its physical deviation request."""
        placement = RigidPlacement.from_source_directions(
            first_axis,
            third_axis,
            translation,
            label,
            second_axis=second_axis,
        )
        if self.policy.maximum_placement_deviation > 0.0:
            return placement
        if (
            placement.rotation_deviation_bound == 0.0
            and placement.translation_deviation_bound == 0.0
        ):
            return RigidPlacement(placement.rotation, placement.translation)
        raise self._refuse(
            "unsupported-entity",
            label,
            "STEP.placement must have zero deviation under a zero placement-deviation limit.",
        )

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
    identity_links: dict[int, list[int]]
    children: dict[int, list[tuple[int, int]]]
    placements: dict[int, tuple[int, int, int]]


def _instances_named(instances: dict[int, _Instance], name: str, /) -> list[_Instance]:
    return [instance for instance in instances.values() if name in instance.names]


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
    identity_links: dict[int, list[int]] = {}
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
            identity_links.setdefault(first.id, []).append(second.id)
            identity_links.setdefault(second.id, []).append(first.id)
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


def _closure(start: list[int], links: dict[int, list[int]], /) -> list[int]:
    seen = dict.fromkeys(start)
    pending = list(start)
    while pending:
        for linked in links.get(pending.pop(), ()):
            if linked not in seen:
                seen[linked] = None
                pending.append(linked)
    return list(seen)


def _assembly_affine(placement: RigidPlacement, /) -> BRepAffinePlacement:
    rows = tuple(
        tuple(Fraction(float(value)) for value in row) for row in placement.matrix()
    ) + ((Fraction(0), Fraction(0), Fraction(0), Fraction(1)),)
    return BRepAffinePlacement(rows)


def _product_representations(
    decoder: _Decoder, assembly: _Assembly, product: int, /
) -> tuple[int, ...]:
    return tuple(
        identifier
        for identifier in _closure(
            assembly.product_reps.get(product, []), assembly.identity_links
        )
        if decoder.instances[identifier].names & _REPRESENTATIONS
    )


def _product_usage_placement(
    decoder: _Decoder,
    assembly: _Assembly,
    usage: int,
    child_representations: tuple[int, ...],
    /,
) -> RigidPlacement:
    label = decoder.instances[usage].label
    if usage not in assembly.placements:
        raise refuse("unsupported-entity", label, "Assembly usage has no placement.")
    first, second, operator = assembly.placements[usage]
    child, parent = (first, second) if first in child_representations else (second, first)
    if child not in child_representations:
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
    if origin.is_identity:
        return destination
    return decoder._compose(
        destination,
        origin.inverse(),
        transform.label,
    )


def _step_assembly_graph(
    decoder: _Decoder,
    assembly: _Assembly,
    representations: Sequence[_Instance],
    /,
) -> tuple[BRepAssemblyGraph, tuple[tuple[str, str], ...]]:
    """Build native definitions and uses from STEP entity references."""
    products = set(assembly.product_reps) | set(assembly.children)
    products.update(child for usages in assembly.children.values() for _, child in usages)
    related = {child for usages in assembly.children.values() for _, child in usages}
    product_source_order = tuple(
        product for product in decoder.instances if product in products
    )
    product_roots = tuple(
        product for product in product_source_order if product not in related
    )
    product_order: list[int] = []
    seen_products: set[int] = set()
    pending_products = list(reversed(product_roots))
    while pending_products:
        product = pending_products.pop()
        if product in seen_products:
            continue
        seen_products.add(product)
        product_order.append(product)
        pending_products.extend(
            child for _, child in reversed(assembly.children.get(product, ()))
        )
    for product in product_source_order:
        if product not in seen_products:
            seen_products.add(product)
            product_order.append(product)
    product_representations = {
        product: _product_representations(decoder, assembly, product)
        for product in product_order
    }
    representation_order: list[int] = []
    seen_representations: set[int] = set()
    for product in product_order:
        for representation in product_representations[product]:
            if representation not in seen_representations:
                seen_representations.add(representation)
                representation_order.append(representation)
    for instance in representations:
        if instance.id not in seen_representations:
            seen_representations.add(instance.id)
            representation_order.append(instance.id)
    for items in decoder.mapped_items.values():
        for _, mapped, _ in items:
            if mapped not in seen_representations:
                seen_representations.add(mapped)
                representation_order.append(mapped)
                decoder._representation(mapped)
    representation_ids = {
        identifier: f"step-representation-#{identifier}"
        for identifier in representation_order
    }
    product_definition_ids = {
        product: f"step-product-definition-#{product}" for product in product_order
    }
    definitions: list[BRepAssemblyDefinition] = []
    provenance: list[tuple[str, str]] = []
    identity = BRepAffinePlacement.identity()
    for representation in representation_order:
        uses: list[BRepAssemblyUse] = []
        for solid_position, solid in enumerate(
            decoder.representations.get(representation, ())
        ):
            use_id = f"step-representation-use-#{representation}-solid-{solid_position}"
            uses.append(BRepAssemblyUse(use_id, identity, solid_index=solid))
            provenance.append(
                (f"assembly-use:{use_id}", f"representation:#{representation}")
            )
        for item_position, (item, mapped, placement) in enumerate(
            decoder.mapped_items.get(representation, ())
        ):
            use_id = f"step-representation-use-#{representation}-mapped-{item_position}-#{item}"
            uses.append(
                BRepAssemblyUse(
                    use_id,
                    _assembly_affine(placement),
                    definition_id=representation_ids[mapped],
                    placement_evidence=source_placement_evidence(placement),
                )
            )
            provenance.append((f"assembly-use:{use_id}", f"mapped-item:#{item}"))
        definition_id = representation_ids[representation]
        definitions.append(BRepAssemblyDefinition(definition_id, tuple(uses)))
        provenance.append(
            (
                f"assembly-definition:{definition_id}",
                decoder.instances[representation].label,
            )
        )
    for product in product_order:
        uses: list[BRepAssemblyUse] = []
        reps = product_representations[product]
        if len(reps) != len(set(reps)):
            raise refuse(
                "malformed",
                decoder.instances[product].label,
                "STEP product definition must not repeat one shape representation reference.",
            )
        for representation in reps:
            use_id = f"step-product-use-#{product}-representation-#{representation}"
            uses.append(
                BRepAssemblyUse(
                    use_id,
                    identity,
                    definition_id=representation_ids[representation],
                )
            )
            provenance.append(
                (f"assembly-use:{use_id}", f"product-representation:#{representation}")
            )
        for usage, child in assembly.children.get(product, ()):
            child_representations = product_representations[child]
            placement = _product_usage_placement(
                decoder, assembly, usage, child_representations
            )
            use_id = f"step-use-#{usage}"
            uses.append(
                BRepAssemblyUse(
                    use_id,
                    _assembly_affine(placement),
                    definition_id=product_definition_ids[child],
                    placement_evidence=source_placement_evidence(placement),
                )
            )
            provenance.append((f"assembly-use:{use_id}", decoder.instances[usage].label))
        definition_id = product_definition_ids[product]
        definitions.append(BRepAssemblyDefinition(definition_id, tuple(uses)))
        provenance.append(
            (f"assembly-definition:{definition_id}", decoder.instances[product].label)
        )
    root_uses: list[BRepAssemblyUse] = []
    for product in product_roots:
        use_id = f"step-root-product-#{product}"
        root_uses.append(
            BRepAssemblyUse(
                use_id, identity, definition_id=product_definition_ids[product]
            )
        )
        provenance.append((f"assembly-use:{use_id}", decoder.instances[product].label))
    represented = {
        representation
        for product in product_order
        for representation in product_representations[product]
    }
    represented.update(
        mapped for items in decoder.mapped_items.values() for _, mapped, _ in items
    )
    for representation in representation_order:
        if representation in represented:
            continue
        use_id = f"step-root-representation-#{representation}"
        root_uses.append(
            BRepAssemblyUse(
                use_id, identity, definition_id=representation_ids[representation]
            )
        )
        provenance.append(
            (f"assembly-use:{use_id}", decoder.instances[representation].label)
        )
    graph = BRepAssemblyGraph(
        tuple(root_uses), tuple(definitions), len(decoder._stage().solids)
    )
    return graph, tuple(provenance)


class _DecodedStep(StrictModule):
    model: BRepModel
    schema: StepSchema = eqx.field(static=True)
    units: _Units = eqx.field(static=True)
    decoder: _Decoder = eqx.field(static=True)


def _decode(
    resource: CadResource,
    policy: CadImportPolicy,
    /,
    *,
    repair_policy: StepRepairPolicy,
) -> _DecodedStep:
    parser = _Parser(_tokens(resource.data))
    header, instances = parser.parse()
    schema = _schema(header)
    _, children = _check_graph(instances)
    decoder = _Decoder(instances, policy, repair_policy)
    decoder.children = children
    for uncertainty in _instances_named(instances, "UNCERTAINTY_MEASURE_WITH_UNIT"):
        decoder._record_uncertainty(_Ref(uncertainty.id))
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
    )
    representations = [
        instance for instance in instances.values() if instance.names & _REPRESENTATIONS
    ]
    factors = [
        decoder._context_units(decoder._items_context(instance.id)).scale
        for instance in representations
    ]
    if not factors:
        raise refuse(
            "unsupported-entity", "DATA", "The file contains no shape representation."
        )
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
    for representation in representations:
        decoder._representation(representation.id)
    assembly_graph, assembly_provenance = _step_assembly_graph(
        decoder, assembly, representations
    )
    stage.assembly_graph = assembly_graph
    stage.provenance.extend(assembly_provenance)
    if not stage.faces and (
        stage.shells
        or stage.solids
        or not any(
            "SHAPE_REPRESENTATION" in instance.names for instance in representations
        )
        or any(
            name.startswith(("TESSELLATED_", "TRIANGULATED_", "COMPLEX_TRIANGULATED_"))
            or name == "CARTESIAN_POINT_LIST_3D"
            for instance in instances.values()
            for name in instance.names
        )
    ):
        raise refuse(
            "unsupported-entity",
            "DATA",
            "The file contains no admitted B-Rep or empty shape.",
        )
    units = decoder._context_units(decoder._items_context(representations[0].id))
    digest = resource.digest
    identity = resource.source_path or f"step-{digest[:16]}"
    try:
        model = stage.publish(
            coordinate_contract=policy.coordinate_contract,
            source_id=identity,
            source_format="step",
            source_digest=digest,
            import_policy_id=policy.policy_id,
            tessellation=(
                None if policy.maximum_placement_deviation > 0.0 else policy.tessellation
            ),
        )
        model, repairs, decoder.orientation_assumptions = audit_shells(
            model, repair_policy, tuple(decoder.shell_labels)
        )
        decoder.repairs.extend(repairs)
        if policy.maximum_placement_deviation > 0.0:
            model = certify_placed_model(model, policy)
    except CadInterchangeError:
        raise
    except ValueError as error:
        raise refuse("inconsistent-geometry", "DATA", str(error)) from error
    return _DecodedStep(model=model, schema=schema, units=units, decoder=decoder)


def _numbers(value: Any, /) -> Iterator[float]:
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                yield float(item)


# Each supported surface-fill style attribute: parameter count, attribute
# index, and the only entity that attribute's ISO select admits here.
_PRESENTATION_CHILDREN = {
    "PRESENTATION_STYLE_ASSIGNMENT": (1, 0, "SURFACE_STYLE_USAGE"),
    "SURFACE_STYLE_USAGE": (2, 1, "SURFACE_SIDE_STYLE"),
    "SURFACE_SIDE_STYLE": (2, 1, "SURFACE_STYLE_FILL_AREA"),
    "SURFACE_STYLE_FILL_AREA": (1, 0, "FILL_AREA_STYLE"),
    "FILL_AREA_STYLE": (2, 1, "FILL_AREA_STYLE_COLOUR"),
    "FILL_AREA_STYLE_COLOUR": (2, 1, "COLOUR_RGB"),
}

type _Colors = tuple[tuple[float, float, float], ...]


def _presentation_colors(
    decoder: _Decoder,
    reference: Any,
    expected: str,
    decoded: dict[int, _Colors | None],
    /,
) -> _Colors | None:
    """Decode only the explicit ISO surface-fill RGB style chain.

    Every reference must name the entity its parent attribute admits, which
    fixes the chain depth; ``decoded`` visits each shared style instance once.
    """
    if not isinstance(reference, _Ref):
        return None
    instance = decoder.instances[reference.id]
    if instance.complex or instance.records[0].name != expected:
        return None
    if instance.id in decoded:
        return decoded[instance.id]
    record = instance.records[0]
    colors: _Colors | None = None
    if record.name == "COLOUR_RGB":
        colors = _rgb_color(record.params)
    else:
        count, index, child_name = _PRESENTATION_CHILDREN[record.name]
        if len(record.params) == count:
            children = record.params[index]
            references = children if isinstance(children, tuple) else (children,)
            found: set[tuple[float, float, float]] = set()
            for child in references:
                child_colors = _presentation_colors(decoder, child, child_name, decoded)
                if child_colors is None:
                    break
                found.update(child_colors)
            else:
                colors = tuple(sorted(found))
    if colors is not None:
        decoder.seen.add(instance.id)
    decoded[instance.id] = colors
    return colors


def _rgb_color(
    params: tuple[Any, ...], /
) -> tuple[tuple[float, float, float], ...] | None:
    if len(params) != 4:
        return None
    values = params[1:]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None
    color = tuple(float(v) for v in values)
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in color):
        return None
    return ((color[0], color[1], color[2]),)


def _metadata_names(decoder: _Decoder, names: dict[int, list[str]], /) -> None:
    for key in names:
        record = decoder.instances[key].records[0]
        if record.params and isinstance(record.params[0], str) and record.params[0]:
            names[key].append(record.params[0])
    for instance in _instances_named(decoder.instances, "NAME_ATTRIBUTE"):
        params = instance.records[0].params
        if (
            len(params) == 2
            and isinstance(params[0], str)
            and params[0]
            and isinstance(params[1], _Ref)
            and params[1].id in names
        ):
            names[params[1].id].append(params[0])
            decoder.seen.add(instance.id)


def _metadata_product_label(decoder: _Decoder, definition: int, /) -> str | None:
    product_definition = decoder.instances[definition].records[0]
    if len(product_definition.params) != 4:
        return None
    formation_ref = product_definition.params[2]
    if not isinstance(formation_ref, _Ref):
        return None
    formation = decoder.instances[formation_ref.id].records[0]
    expected_fields = {
        "PRODUCT_DEFINITION_FORMATION": 3,
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE": 4,
    }.get(formation.name)
    if len(formation.params) != expected_fields:
        return None
    product_ref = formation.params[2]
    if not isinstance(product_ref, _Ref):
        return None
    product = decoder.instances[product_ref.id].records[0]
    if (
        product.name != "PRODUCT"
        or len(product.params) != 4
        or not isinstance(product.params[1], str)
    ):
        return None
    decoder.seen.update((definition, formation_ref.id, product_ref.id))
    return product.params[1]


def _metadata_product_names(decoder: _Decoder, names: dict[int, list[str]], /) -> None:
    assembly = _product_structure(decoder.instances)
    for definition, representations in assembly.product_reps.items():
        name = _metadata_product_label(decoder, definition)
        if name is None:
            continue
        for representation in representations:
            solids = decoder.representations.get(representation, [])
            for key, index in decoder.solids.items():
                if index in solids and name:
                    names[key].append(name)


def _metadata_colors(
    decoder: _Decoder, colors: dict[int, list[tuple[float, float, float]]], /
) -> None:
    decoded: dict[int, _Colors | None] = {}
    for instance in _instances_named(decoder.instances, "STYLED_ITEM"):
        params = instance.records[0].params
        if (
            len(params) != 3
            or not isinstance(params[2], _Ref)
            or params[2].id not in colors
        ):
            continue
        if not isinstance(params[1], tuple):
            continue
        found = [
            _presentation_colors(decoder, style, "PRESENTATION_STYLE_ASSIGNMENT", decoded)
            for style in params[1]
        ]
        if any(value is None for value in found):
            continue
        for value in found:
            if value is not None:
                colors[params[2].id].extend(value)
        decoder.seen.add(instance.id)


def _metadata_layers(decoder: _Decoder, layers: dict[int, list[str]], /) -> None:
    for instance in _instances_named(decoder.instances, "PRESENTATION_LAYER_ASSIGNMENT"):
        params = instance.records[0].params
        if (
            len(params) != 3
            or not isinstance(params[0], str)
            or not params[0]
            or not isinstance(params[2], tuple)
        ):
            continue
        for reference in params[2]:
            if isinstance(reference, _Ref) and reference.id in layers:
                layers[reference.id].append(params[0])
        decoder.seen.add(instance.id)


def _entity_metadata(model: BRepModel, decoder: _Decoder, /) -> StepEntityMetadata:
    """Bind presentation only through explicit source entity relationships."""
    targets = {
        **{key: model.vertex_ids[index] for key, index in decoder.vertices.items()},
        **{key: model.edge_ids[edge.index] for key, edge in decoder.edges.items()},
        **{key: model.face_ids[index] for key, (index, _) in decoder.faces.items()},
        **{key: model.solid_ids[index] for key, index in decoder.solids.items()},
    }
    names: dict[int, list[str]] = {key: [] for key in targets}
    colors: dict[int, list[tuple[float, float, float]]] = {key: [] for key in targets}
    layers: dict[int, list[str]] = {key: [] for key in targets}
    _metadata_names(decoder, names)
    _metadata_product_names(decoder, names)
    _metadata_colors(decoder, colors)
    _metadata_layers(decoder, layers)
    return StepEntityMetadata(
        model,
        tuple(
            StepEntityLabels(
                entity,
                names=tuple(names[key]),
                colors=tuple(colors[key]),
                layers=tuple(layers[key]),
            )
            for key, entity in targets.items()
            if names[key] or colors[key] or layers[key]
        ),
    )


def _coverage_classification(
    decoder: _Decoder, /
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    consumed: list[str] = []
    ignored: list[str] = []
    discarded: list[str] = []
    for identifier, instance in sorted(decoder.instances.items()):
        if identifier in decoder.ignored_units:
            ignored.append(instance.label)
        elif identifier in decoder.seen or instance.names & _ASSEMBLY_ENTITIES:
            consumed.append(instance.label)
        else:
            ignored.append(instance.label)
            if not instance.names <= _METADATA_ENTITIES:
                discarded.append(instance.label)
    return tuple(consumed), tuple(ignored), tuple(discarded)


def _import_report(
    model: BRepModel,
    schema: StepSchema,
    units: _Units,
    stage: CadStage,
    digest: str,
    decoder: _Decoder,
    tolerance_evidence: CadImportToleranceEvidence,
    /,
) -> AdapterReport:
    losses: tuple[AdapterLoss, ...] = ()
    consumed, ignored, discarded = _coverage_classification(decoder)
    if discarded:
        losses = (
            AdapterLoss(
                "geometry.ignored-step-entities",
                "import",
                "dropped",
                "Ignored geometry-bearing STEP entities: " + ", ".join(discarded),
                changes_interpretation=False,
            ),
        )
    losses += tolerance_evidence.losses
    losses += tuple(decoder.repairs)
    losses += cad_import_synthesis_losses(stage)
    uncertainty = tolerance_evidence.uncertainty
    qualifiers = {
        "schema": schema,
        "artifact-sha256": digest,
        "length-unit-m": format(float(units.length_meters), ".17g"),
        "plane-angle-unit-rad": format(units.angle_radians, ".17g"),
        "declared-tolerance-limit": format(tolerance_evidence.declared_limit, ".17g"),
        "carried-uncertainty": (
            "absent" if uncertainty is None else format(uncertainty, ".17g")
        ),
        "p-curves-honored": str(stage.pcurves_honored),
        "p-curves-recovered-exact": str(stage.pcurves_exact),
        "coverage-consumed": json.dumps(consumed, separators=(",", ":")),
        "coverage-ignored-by-design": json.dumps(ignored, separators=(",", ":")),
        "source-uncertainties": json.dumps(
            tuple(
                (identifier, decoder.source_uncertainties.get(identifier))
                for identifier in sorted(decoder.source_uncertainty_units)
            ),
            separators=(",", ":"),
        ),
        "source-uncertainty-length-units-m": json.dumps(
            tuple(sorted(decoder.source_uncertainty_units.items())),
            separators=(",", ":"),
        ),
    }
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "step",
        "phydrax-brep",
        source_id=model.source_id,
        target_id=model.model_id,
        stage="host-import",
        source_profile=AdapterFormatProfile("step", qualifiers=qualifiers),
        target_profile=AdapterFormatProfile(
            "phydrax-brep",
            qualifiers={"coordinate-contract": model.coordinate_contract.spatial_id},
        ),
        coordinate_mapping=(
            f"source length unit {float(units.length_meters):.17g} m -> "
            f"{model.coordinate_contract.length_unit.symbol}",
            "assembly definition/use placements -> native assembly graph",
        ),
        preserved_fields=(
            "exact analytic and rational geometry",
            "oriented topology",
            "p-curves",
            "assembly definitions",
            "assembly uses",
            "units",
            "authored product and representation-item names",
            "RGB surface-fill colors",
            "presentation layers",
        ),
        assumptions=("external instance numbers are provenance only",)
        + decoder.orientation_assumptions,
        losses=losses,
    )


@checked
def decode_step_resource(
    resource: CadResource,
    policy: CadImportPolicy,
    /,
    *,
    repair_policy: StepRepairPolicy | None = None,
) -> CadImportResult:
    """Decode one caller-owned or mapped STEP Part 21 source."""
    repair_policy = StepRepairPolicy() if repair_policy is None else repair_policy
    try:
        decoded = _decode(resource, policy, repair_policy=repair_policy)
    except CadInterchangeError:
        raise
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise refuse("malformed", "DATA", f"Invalid STEP record: {error}") from error
    model, schema, units, decoder = (
        decoded.model,
        decoded.schema,
        decoded.units,
        decoded.decoder,
    )
    stage = decoder._stage()
    tolerance_evidence = build_cad_import_tolerance_evidence(
        model,
        tuple((None, value) for _, value in sorted(decoder.source_uncertainties.items())),
        unrepresentable_losses=tuple(
            unrepresentable_cad_tolerance_loss(
                f"global (#{identifier})",
                decoder.raw_source_uncertainties[f"#{identifier}"],
                source_unit,
                model.report.curve_surface_tolerance,
            )
            for identifier, source_unit in sorted(
                decoder.unrepresentable_uncertainties.items()
            )
        ),
    )
    metadata = _entity_metadata(model, decoder)
    report = _import_report(
        model,
        schema,
        units,
        stage,
        resource.digest,
        decoder,
        tolerance_evidence,
    )
    return CadImportResult(
        model=model,
        format="step",
        schema=schema,
        source_digest=resource.digest,
        source_length_unit_meters=float(units.length_meters),
        source_angle_unit_radians=units.angle_radians,
        report=report,
        coverage=stage.coverage(decoder.used),
        provenance=tuple(stage.provenance),
        entity_metadata=metadata,
        source_tolerance=max(decoder.source_uncertainties.values(), default=None),
        source_tolerances=tuple(sorted(decoder.raw_source_uncertainties.items())),
        tolerance_evidence=tolerance_evidence,
    )


@checked
def decode_step_bytes(
    data: bytes,
    policy: CadImportPolicy,
    /,
    *,
    source_path: str | None = None,
    repair_policy: StepRepairPolicy | None = None,
) -> CadImportResult:
    """Decode caller-owned STEP bytes."""
    return decode_cad_bytes(
        data,
        policy,
        decoder=partial(
            decode_step_resource,
            repair_policy=repair_policy,
        ),
        source_path=source_path,
    )


@checked
def read_step(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
    repair_policy: StepRepairPolicy | None = None,
) -> CadImportResult:
    """Decode one descriptor-owned local STEP source."""
    return read_cad_file(
        path,
        policy,
        decoder=partial(
            decode_step_resource,
            repair_policy=repair_policy,
        ),
        trusted_root=trusted_root,
    )


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


def _real(value: float, /, *, certified_precision: bool = False) -> str:
    number = float(value)
    if not np.isfinite(number):
        raise refuse("inexact-export", "real", "Nonfinite values have no STEP literal.")
    text = format(number, ".17g") if certified_precision else repr(number)
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
        elif ord(character) <= 0xFFFF:
            out.append(f"\\X2\\{ord(character):04X}\\X0\\")
        else:
            out.append(f"\\X4\\{ord(character):08X}\\X0\\")
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


class _WrittenShells(StrictModule):
    shells: list[int] = eqx.field(static=True)
    closed: dict[int, bool] = eqx.field(static=True)


class _WrittenShapes(StrictModule):
    solids: list[int] = eqx.field(static=True)
    free: list[int] = eqx.field(static=True)


class _Writer:
    """Deterministic instance emission of one native B-Rep."""

    def __init__(
        self,
        model: BRepModel,
        geometry: BRepGeometry,
        schema: StepSchema,
        policy: CadExportPolicy,
        uncertainty: float,
        metadata: StepEntityMetadata | None,
        surface_carriers: tuple[AbstractSurfacePatch, ...],
        surface_losses: tuple[AdapterLoss, ...],
    ) -> None:
        self.model = model
        self.geometry = geometry
        self.schema = schema
        self.policy = policy
        self.uncertainty = uncertainty
        self.metadata = metadata
        self.surface_carriers = surface_carriers
        self.surface_losses = surface_losses
        self.entity_names = {
            (row.entity.kind, row.entity.index): row.names[0]
            for row in (() if metadata is None else metadata.rows)
            if row.names
        }
        self.lines: list[str] = []
        self.counts: dict[str, int] = {}
        self.context2 = 0
        self.entity_instances: dict[tuple[str, int], int] = {}
        self.assembly_losses: list[AdapterLoss] = []
        self.assembly_provenance: list[tuple[str, str]] = []

    def add(self, body: str, /) -> int:
        self.lines.append(body)
        for name in re.findall(r"(?:^|\(|\s)([A-Z_][A-Z0-9_]*)\(", body):
            self.counts[name] = self.counts.get(name, 0) + 1
        return len(self.lines)

    def provenance(self, /) -> tuple[tuple[str, str], ...]:
        geometry_rows = tuple(
            (f"{kind}:{index}", f"#{instance}")
            for (kind, index), instance in sorted(self.entity_instances.items())
        )
        return tuple(sorted((*geometry_rows, *self.assembly_provenance)))

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
        curve = explicit_carrier(curve)
        match curve:
            case LineCurve():
                return self.add(
                    f"LINE('',#{self.point(np.asarray(curve.origin))},#{self.vector(np.asarray(curve.direction))})"
                )
            case OffsetCurve():
                basis = self.curve(curve.base, label)
                distance = -float(curve.distance_exact)
                if curve.direction is None:
                    return self.add(f"OFFSET_CURVE_2D('',#{basis},{_real(distance)},.U.)")
                direction = self.direction(np.asarray(curve.direction))
                return self.add(
                    f"OFFSET_CURVE_3D('',#{basis},{_real(distance)},.U.,#{direction})"
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
                    f"STEP.curve must have a supported entity; got {type(curve).__name__}.",
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
                    f"STEP.surface must have a supported entity; got {type(patch).__name__}.",
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
                "inexact-export",
                label,
                "STEP.pcurve must have a supported parameterization; got unrepresentable curve.",
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
        description = "confusion accuracy"
        name = "distance_accuracy_value"
        uncertainty = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({_real(self.uncertainty, certified_precision=True)}),#{length},{_string(name)},{_string(description)})"
        )
        return self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) REPRESENTATION_CONTEXT('','') )"
        )

    def shapes(self) -> _WrittenShapes:
        """Emit vertices, curves, faces, shells and solid definitions in order."""
        geometry = self.geometry
        model = self.model
        self.context2 = self.add(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() "
            "REPRESENTATION_CONTEXT('2D SPACE','') )"
        )
        vertices = [
            self.add(f"VERTEX_POINT('',#{self.point(point)})")
            for point in np.asarray(geometry.vertex_points)
        ]
        self.entity_instances.update(
            {("vertex", i): instance for i, instance in enumerate(vertices)}
        )
        surfaces = [
            self.surface(patch, f"face:{index}")
            for index, patch in enumerate(self.surface_carriers)
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
        edges = self._shape_edges(vertices, surfaces, normal_sign)
        faces = self._shape_faces(edges, surfaces, normal_sign)
        shell_records = self._shape_shells(faces, void_faces, void_shells)
        shells, closed = shell_records.shells, shell_records.closed
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
        self.entity_instances.update(
            {("solid", i): instance for i, instance in enumerate(solids)}
        )
        self.entity_instances.update(
            {("face", i): instance for i, instance in enumerate(faces)}
        )
        self.entity_instances.update(
            {("edge", i): instance for i, instance in edges.items()}
        )
        self._write_metadata()
        free = [shells[index] for index in range(len(shells)) if index not in in_solids]
        return _WrittenShapes(solids=solids, free=free)

    def _shape_shells(
        self,
        faces: list[int],
        void_faces: set[int],
        void_shells: set[int],
        /,
    ) -> _WrittenShells:
        geometry = self.geometry
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
        return _WrittenShells(shells=shells, closed=closed)

    def _shape_edges(
        self,
        vertices: list[int],
        surfaces: list[tuple[int, np.ndarray]],
        normal_sign: list[int],
        /,
    ) -> dict[int, int]:
        geometry = self.geometry
        model = self.model
        ranges = np.asarray(geometry.edge_ranges)
        owners = [-1] * len(geometry.coedge_edges)
        for face, loops in enumerate(geometry.face_loops):
            for loop in loops:
                for coedge in loop:
                    if not 0 <= coedge < len(owners) or owners[coedge] != -1:
                        raise refuse(
                            "inconsistent-geometry",
                            f"face:{face}",
                            "STEP face-loop incidence must assign each coedge to exactly one face.",
                        )
                    owners[coedge] = face
        if any(owner < 0 for owner in owners):
            coedge = owners.index(-1)
            raise refuse(
                "inconsistent-geometry",
                f"coedge:{coedge}",
                "STEP face-loop incidence must assign each coedge to exactly one face.",
            )
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
        return edges

    def _shape_faces(
        self,
        edges: dict[int, int],
        surfaces: list[tuple[int, np.ndarray]],
        normal_sign: list[int],
        /,
    ) -> list[int]:
        geometry = self.geometry
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
                    f"ADVANCED_FACE({_string(self.model.physical_tags[face])},{_refs(bounds)},#{surface},{_bool(same_sense)})"
                )
            )
        return faces

    def _write_metadata(self) -> None:
        """Emit authored labels without altering any carrier or topology."""
        if self.metadata is None:
            return
        for row in self.metadata.rows:
            key = (row.entity.kind, row.entity.index)
            if key not in self.entity_instances:
                raise refuse(
                    "inexact-export",
                    str(key),
                    "Metadata refers to a nonrepresentable STEP entity.",
                )
            instance = self.entity_instances[key]
            if row.names:
                if row.entity.kind == "face":
                    for name in row.names:
                        self.add(f"NAME_ATTRIBUTE({_string(name)},#{instance})")
                else:
                    body = self.lines[instance - 1]
                    prefix, suffix = body.split("('',", 1)
                    self.lines[instance - 1] = (
                        prefix + "(" + _string(row.names[0]) + "," + suffix
                    )
                    for name in row.names[1:]:
                        self.add(f"NAME_ATTRIBUTE({_string(name)},#{instance})")
            for color in row.colors:
                rgb = self.add(
                    f"COLOUR_RGB('',{','.join(_real(value) for value in color)})"
                )
                fill_color = self.add(f"FILL_AREA_STYLE_COLOUR('',#{rgb})")
                fill = self.add(f"FILL_AREA_STYLE('',(#{fill_color}))")
                surface_fill = self.add(f"SURFACE_STYLE_FILL_AREA(#{fill})")
                side = self.add(f"SURFACE_SIDE_STYLE('',(#{surface_fill}))")
                usage = self.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{side})")
                assignment = self.add(f"PRESENTATION_STYLE_ASSIGNMENT((#{usage}))")
                self.add(f"STYLED_ITEM('',(#{assignment}),#{instance})")
            for layer in row.layers:
                self.add(
                    f"PRESENTATION_LAYER_ASSIGNMENT({_string(layer)},'',(#{instance}))"
                )

    def product(
        self,
        name: str,
        product_context: int,
        definition_context: int,
        /,
    ) -> tuple[int, int]:
        product = self.add(f"PRODUCT('',{_string(name)},'',(#{product_context}))")
        formation = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
        definition = self.add(
            f"PRODUCT_DEFINITION('design','',#{formation},#{definition_context})"
        )
        shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
        return definition, shape

    def encode(self) -> str:
        schema_id, application, protocol, year = _SCHEMA_IDENTIFIERS[self.schema]
        shapes = self.shapes()
        solids, free = shapes.solids, shapes.free
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
        graph = geometry.assembly_graph
        if graph is None:
            graph = BRepAssemblyGraph.single_solids(len(geometry.solid_shells))
        if graph.solid_count != len(solids):
            raise refuse(
                "inconsistent-geometry",
                "assembly",
                "Assembly graph solid count must match native B-Rep topology.",
            )
        identity = BRepAffinePlacement.identity()
        default = (
            not graph.definitions
            and len(graph.roots) == len(solids)
            and all(
                use.use_id == f"solid-{index}"
                and use.solid_index == index
                and use.placement == identity
                for index, use in enumerate(graph.roots)
            )
        )
        if default:
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
        """Serialize native assembly definitions and uses without expansion."""
        graph = self.geometry.assembly_graph
        if graph is None:
            graph = BRepAssemblyGraph.single_solids(len(solids))
        if graph.solid_count != len(solids):
            raise refuse(
                "inconsistent-geometry",
                "assembly",
                "Assembly graph solid count must match native B-Rep topology.",
            )
        origin = np.zeros(3)
        z, x = np.eye(3)[2], np.eye(3)[0]
        parts: dict[int, tuple[int, int, int]] = {}
        for index, solid in enumerate(solids):
            definition, shape = self.product(
                f"solid{index}"
                if self.metadata is None
                else self.entity_names.get(("solid", index), ""),
                product_context,
                definition_context,
            )
            axis = self.frame(origin, z, x)
            representation = self.add(
                f"ADVANCED_BREP_SHAPE_REPRESENTATION('',{_refs([axis, solid])},#{context})"
            )
            self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})")
            parts[index] = (definition, representation, axis)
        nodes: dict[str, tuple[int, int, int, list[int]]] = {}
        for assembly_definition in graph.definitions:
            identifier = assembly_definition.definition_id
            definition, shape = self.product(
                f"assembly{len(nodes)}", product_context, definition_context
            )
            axis = self.frame(origin, z, x)
            nodes[identifier] = (definition, shape, axis, [axis])
            self.assembly_provenance.append(
                (f"assembly-definition:{identifier}", f"#{definition}")
            )
        root_definition, root_shape = self.product(
            "model",
            product_context,
            definition_context,
        )
        self.assembly_provenance.append(
            ("assembly-container:root", f"#{root_definition}")
        )
        root_node = (
            root_definition,
            root_shape,
            self.frame(origin, z, x),
            [],
        )
        root_node[3].append(root_node[2])
        pending: list[tuple[int | str, int, int, str]] = []

        def placement_frame(placement: BRepAffinePlacement, label: str, /) -> int:
            matrix = placement.matrix
            rotation = tuple(
                tuple(matrix[row][column] for column in range(3)) for row in range(3)
            )
            for first in range(3):
                for second in range(3):
                    value = sum(
                        (
                            rotation[row][first] * rotation[row][second]
                            for row in range(3)
                        ),
                        Fraction(0),
                    )
                    if value != Fraction(first == second):
                        raise refuse(
                            "inexact-export",
                            label,
                            "STEP assembly uses require exact proper-rigid placement coefficients.",
                        )
            a, b, c = rotation[0]
            d, e, f = rotation[1]
            g, h, i = rotation[2]
            determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            if determinant != 1:
                raise refuse(
                    "inexact-export",
                    label,
                    "STEP assembly uses require exact proper-rigid placement coefficients.",
                )
            represented = tuple(
                tuple(float(value) for value in row[:4]) for row in matrix[:3]
            )
            if any(
                Fraction(value) != exact
                for row, exact_row in zip(represented, matrix[:3], strict=True)
                for value, exact in zip(row, exact_row, strict=True)
            ):
                raise refuse(
                    "inexact-export",
                    label,
                    "An exact STEP assembly placement coefficient is not representable in the exchange real format.",
                )
            translation = np.asarray([row[3] for row in represented])
            x_axis = np.asarray([row[0] for row in represented])
            z_axis = np.asarray([row[2] for row in represented])
            return self.frame(translation, z_axis, x_axis)

        def emit_use(
            use: BRepAssemblyUse,
            parent_definition: int,
            parent_key: str,
            parent_items: list[int],
        ) -> None:
            if use.solid_index is not None:
                child_definition, child_representation, source_axis = parts[
                    use.solid_index
                ]
            else:
                if use.definition_id is None:
                    raise RuntimeError("Validated assembly use lost its target.")
                child_definition, _, source_axis, _ = nodes[use.definition_id]
                child_representation = use.definition_id
            target_axis = placement_frame(use.placement, f"assembly-use:{use.use_id}")
            parent_items.append(target_axis)
            usage = self.add(
                "NEXT_ASSEMBLY_USAGE_OCCURRENCE('','','',"
                f"#{parent_definition},#{child_definition},$)"
            )
            self.assembly_provenance.append((f"assembly-use:{use.use_id}", f"#{usage}"))
            usage_shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{usage})")
            transform = self.add(
                f"ITEM_DEFINED_TRANSFORMATION('','',#{source_axis},#{target_axis})"
            )
            pending.append((child_representation, transform, usage_shape, parent_key))

        for assembly_definition in graph.definitions:
            identifier = assembly_definition.definition_id
            parent_definition, _, _, items = nodes[identifier]
            for use in assembly_definition.uses:
                emit_use(use, parent_definition, identifier, items)
        for use in graph.roots:
            emit_use(use, root_definition, "root", root_node[3])
        self.assembly_losses.append(
            AdapterLoss(
                "geometry.assembly-root",
                "export",
                "transformed",
                "STEP serializes the native root-use table beneath one product-definition root.",
                changes_interpretation=True,
            )
        )
        self.assembly_losses.append(
            AdapterLoss(
                "geometry.assembly-representation-wrappers",
                "export",
                "transformed",
                "STEP product definitions and shape representations introduce standard assembly wrapper nodes; definition/use sharing and placements are retained.",
                changes_interpretation=True,
            )
        )
        representations: dict[str, int] = {}
        for identifier, (_, shape, _, items) in nodes.items():
            representation = self.add(
                f"SHAPE_REPRESENTATION('',{_refs(items)},#{context})"
            )
            self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})")
            representations[identifier] = representation
        root_representation = self.add(
            f"SHAPE_REPRESENTATION('',{_refs(root_node[3])},#{context})"
        )
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{root_shape},#{root_representation})")
        representations["root"] = root_representation
        for child_representation, transform, shape, parent in pending:
            if isinstance(child_representation, str):
                child_representation = representations[child_representation]
            relation = self.add(
                f"( REPRESENTATION_RELATIONSHIP('','',#{child_representation},#{representations[parent]}) "
                f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transform}) SHAPE_REPRESENTATION_RELATIONSHIP() )"
            )
            self.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relation},#{shape})")


@checked
def write_step(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    schema: StepSchema = "AP214",
    policy: CadExportPolicy | None = None,
    metadata: StepEntityMetadata | None = None,
    mode: PublicationMode = "exclusive",
) -> CadExportResult:
    """Publish exact native carriers or explicitly certified intersection fits.

    When requested, branch fits retain the source parameter domain and require
    whole-domain coupled error, topology, and used-trim certificates.
    Optional metadata emits authored names, RGB colors, and layers; it is excluded
    from geometric identity.
    """
    schema_ = parse(schema, StepSchema, "schema")
    policy_ = CadExportPolicy() if policy is None else policy
    if metadata is not None and metadata.model_id != model.model_id:
        raise ValueError(
            f"StepEntityMetadata.model_id must equal the exported model ID; got {metadata.model_id!r}, expected={model.model_id!r}"
        )
    if model.geometry is None:
        raise refuse(
            "inexact-export",
            model.source_id,
            "STEP.model.geometry must contain B-Rep topology; got only a derived tessellation.",
        )
    geometry, _, approximation_losses = prepare_cad_export_geometry(
        model, model.geometry, policy_, format_name="step"
    )
    surface_carriers, surface_losses = prepare_cad_export_surfaces(model.patches, "STEP")
    uncertainty = _certified_cad_export_tolerances(
        model, _cad_export_gap_surfaces(surface_carriers)
    ).uncertainty
    writer = _Writer(
        model,
        geometry,
        schema_,
        policy_,
        uncertainty,
        metadata,
        surface_carriers,
        surface_losses,
    )
    text = writer.encode()
    data = text.encode("ascii")
    digest = hashlib.sha256(data).hexdigest()
    publish_bytes(path, data, maximum_bytes=len(data), mode=mode)
    qualifiers = {
        "schema": schema_,
        "length-unit": model.coordinate_contract.length_unit.symbol,
        "plane-angle-unit": "rad",
    }
    report = build_cad_export_report(
        model,
        digest,
        policy_,
        format_name="step",
        qualifiers=qualifiers,
        preserved_fields=(
            "exact surface definitions",
            "oriented topology",
            "physical tags",
            "coedge parameter correspondence",
            "assembly definitions",
            "assembly uses",
            "units",
        )
        + (
            ("authored names", "RGB colors", "presentation layers")
            if metadata is not None
            else ()
        ),
        source_profile=AdapterFormatProfile(
            "phydrax-brep",
            qualifiers={"coordinate-contract": model.coordinate_contract.spatial_id},
        ),
        target_id=digest,
        extra_losses=(
            *approximation_losses,
            *writer.assembly_losses,
            *writer.surface_losses,
            *cad_export_assembly_identifier_losses(geometry.assembly_graph, "STEP"),
        ),
    )
    return CadExportResult(
        Path(path).expanduser().resolve(),
        "step",
        schema_,
        report,
        tuple(sorted(writer.counts.items())),
        writer.provenance(),
    )


__all__ = [
    "STEP_ENTITY_COVERAGE",
    "StepSchema",
    "decode_step_bytes",
    "decode_step_resource",
    "read_step",
    "write_step",
]

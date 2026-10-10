#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native bounded IGES 5.3 reader and exact B-Rep writer.

Reading parses the fixed-format Start/Global/Directory/Parameter/Terminate
sections, the global parameter/record delimiters, Hollerith strings, the
directory/parameter linkage and the global units, and lowers the entity graph
to one native exact `BRepModel` through the shared `CadStage`.

Coverage (read):

* Geometry: 100 circular arc, 102 composite curve (single segment as an edge
  carrier; segments become edges in trimming boundaries), 104 conic (ellipse),
  110 line, 116 point, 123 direction, 124 transformation (proper rigid), 126
  rational B-spline curve, 108 plane, 120 surface of revolution, 122 tabulated
  cylinder, 128 rational B-spline surface, and the IGES 5.3 analytic surfaces
  190 plane, 192 cylinder, 194 cone, 196 sphere and 198 torus.
* Trimmed surfaces: 144 with 142 curves on surface (model-space boundary
  curves) become free faces; IGES trimmed surfaces carry no shared topology.
* B-Rep topology: 186 manifold solid (with voids), 514 shell, 510 face, 508
  loop, 504 edge list, 502 vertex list.

Supplied parameter-space carriers are retained and affinely reparameterized
when their domains differ from the edge. Inconsistent supplied carriers are
refused, never discarded. Missing p-curves are recovered algebraically for
admitted carrier pairs; sampled fitting requires an explicit import policy
and does not certify a continuous distance bound. Subfigure definitions and
instances (308/408) become rigid native assembly occurrences. Unsupported
required geometry is refused with its dependency chain.

Writing emits exact B-Rep topology, parameter-space carriers and rigid
subfigure occurrences. Carriers without an exact IGES entity are refused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from math import cos, degrees, isfinite, radians, sin

import jax
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
    _axis_frame,
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
from ..typing import ConvertibleToArray
from ._cad import (
    _export_loss_waivers,
    _natural_box,
    approximation_export_qualifiers,
    CadCurveApproximation,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadImportResult,
    CadInterchangeError,
    CadStage,
    explicit_carrier,
    length_factor,
    prepare_cad_export_geometry,
    refuse,
    resource_refusal,
    StagedCoedge,
    transform_pcurve,
)
from ._cad_carriers import (
    orthonormal_frame,
    place_curve,
    place_surface,
    RigidPlacement,
    surface_tag,
)
from ._report import AdapterFormatProfile, AdapterLoss, AdapterReport, AdapterStatus


_FORMAT = "iges"
_SCHEMA = "iges-5.3"
_SECTIONS = "SGDPT"

# IGES global unit flags: (canonical name, exact meters per unit).
_UNITS: dict[int, tuple[str, Fraction]] = {
    1: ("INCH", Fraction(254, 10_000)),
    2: ("MM", Fraction(1, 1_000)),
    4: ("FT", Fraction(3_048, 10_000)),
    5: ("MI", Fraction(1_609_344, 1_000)),
    6: ("M", Fraction(1)),
    7: ("KM", Fraction(1_000)),
    8: ("MIL", Fraction(254, 10_000_000)),
    9: ("UM", Fraction(1, 1_000_000)),
    10: ("CM", Fraction(1, 100)),
    11: ("UIN", Fraction(254, 10_000_000_000)),
}
_UNIT_NAMES: dict[str, int] = {
    "IN": 1,
    "INCH": 1,
    "MM": 2,
    "FT": 4,
    "MI": 5,
    "M": 6,
    "KM": 7,
    "MIL": 8,
    "UM": 9,
    "MICRON": 9,
    "CM": 10,
    "UIN": 11,
}
# Entity types that never carry B-Rep geometry (structure, annotation, views).
_DROPPABLE = frozenset(
    (
        0,
        202,
        206,
        208,
        210,
        212,
        214,
        216,
        218,
        220,
        222,
        228,
        230,
        302,
        304,
        306,
        308,
        310,
        312,
        314,
        316,
        320,
        322,
        402,
        404,
        406,
        408,
        410,
        412,
        414,
        416,
        418,
        420,
        422,
        430,
    )
)


def _profile() -> AdapterFormatProfile:
    return AdapterFormatProfile(_FORMAT, qualifiers={"profile": _SCHEMA})


# ------------------------------------------------------------------ sections


@dataclass(frozen=True, slots=True)
class _Hollerith:
    text: str


type _Parameter = str | _Hollerith | None


@dataclass(frozen=True, slots=True)
class _Entry:
    """One directory entry and its decoded parameter list."""

    number: int
    entity_type: int
    transform: int
    form: int
    label: str
    parameters: tuple[_Parameter, ...]

    @property
    def name(self) -> str:
        return f"D{self.number} E{self.entity_type}"


def _split_parameters(
    text: str, parameter_delimiter: str, record_delimiter: str, label: str, limit: int
) -> tuple[list[_Parameter], int]:
    """Free-format parameter tokens up to the record delimiter."""
    values: list[_Parameter] = []
    position = 0
    length = len(text)
    while True:
        while position < length and text[position] == " ":
            position += 1
        start = position
        while position < length and text[position].isdigit():
            position += 1
        if position < length and position > start and text[position] == "H":
            count = int(text[start:position])
            value: _Parameter = _Hollerith(text[position + 1 : position + 1 + count])
            position += 1 + count
            if position > length:
                raise refuse("malformed", label, "A Hollerith string is truncated.")
            while position < length and text[position] == " ":
                position += 1
        else:
            position = start
            while position < length and text[position] not in (
                parameter_delimiter,
                record_delimiter,
            ):
                position += 1
            token = text[start:position].strip()
            value = token if token else None
        values.append(value)
        if len(values) > limit:
            raise refuse("limit", label, "The entity exceeds the attribute limit.")
        if position >= length:
            raise refuse("malformed", label, "A parameter record has no terminator.")
        delimiter = text[position]
        position += 1
        if delimiter == record_delimiter:
            return values, position
        if delimiter != parameter_delimiter:
            raise refuse("malformed", label, f"Unexpected delimiter {delimiter!r}.")


def _integer(value: _Parameter, label: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise refuse("malformed", label, "A required integer parameter is missing.")
        return default
    if isinstance(value, _Hollerith):
        raise refuse("malformed", label, "Expected an integer, found a string.")
    try:
        return int(value)
    except ValueError as error:
        raise refuse("malformed", label, f"Malformed integer {value!r}.") from error


def _real(value: _Parameter, label: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise refuse("malformed", label, "A required real parameter is missing.")
        return default
    if isinstance(value, _Hollerith):
        raise refuse("malformed", label, "Expected a real, found a string.")
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise refuse("malformed", label, f"Malformed real {value!r}.") from error
    if not isfinite(number):
        raise refuse("malformed", label, "A real parameter is not finite.")
    return number


def _string(value: _Parameter) -> str:
    return value.text if isinstance(value, _Hollerith) else (value or "")


@dataclass(slots=True)
class _File:
    unit_flag: int
    unit_meters: Fraction
    entries: dict[int, _Entry]
    attributes: int


def _lines(data: bytes, label: str) -> dict[str, list[str]]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise refuse("malformed", label, "IGES files are ASCII text.") from error
    sections: dict[str, list[str]] = {letter: [] for letter in _SECTIONS}
    order = ""
    for index, raw in enumerate(text.splitlines()):
        if not raw.strip():
            continue
        line = raw.ljust(80)
        if len(line) != 80:
            raise refuse("malformed", f"line {index + 1}", "IGES records are 80 columns.")
        letter = line[72]
        if letter in ("B", "C"):
            raise refuse(
                "unsupported-entity",
                f"line {index + 1}",
                "Binary and compressed IGES forms are not admitted.",
            )
        if letter not in sections:
            raise refuse("malformed", f"line {index + 1}", f"Unknown section {letter!r}.")
        if not order or order[-1] != letter:
            if letter in order or (
                order and _SECTIONS.index(letter) < _SECTIONS.index(order[-1])
            ):
                raise refuse(
                    "malformed", f"line {index + 1}", "IGES sections are out of order."
                )
            order += letter
        sequence = line[73:80].strip()
        if not sequence.isdigit() or int(sequence) != len(sections[letter]) + 1:
            raise refuse(
                "malformed",
                f"line {index + 1}",
                f"Section {letter} sequence numbers are broken.",
            )
        sections[letter].append(line)
    if order[:1] != "S" or "G" not in order or "D" not in order or "P" not in order:
        raise refuse("malformed", label, "The file lacks its S/G/D/P sections.")
    if order[-1] != "T" or len(sections["T"]) != 1:
        raise refuse("malformed", label, "The file lacks its single Terminate record.")
    terminate = sections["T"][0]
    counts = [terminate[8 * i + 1 : 8 * i + 8].strip() for i in range(4)]
    expected = [len(sections[letter]) for letter in "SGDP"]
    if [int(value) if value.isdigit() else -1 for value in counts] != expected:
        raise refuse("malformed", "T", "The Terminate section counts do not match.")
    return sections


def _global(lines: list[str], limit: int) -> tuple[str, str, list[_Parameter]]:
    text = "".join(line[:72] for line in lines)
    parameter_delimiter, record_delimiter = ",", ";"
    position = 0
    if text.startswith("1H"):
        parameter_delimiter = text[2]
        position = 3
    if text[position : position + 1] != parameter_delimiter:
        raise refuse("malformed", "G", "Malformed global parameter delimiter.")
    position += 1
    if text.startswith("1H", position):
        record_delimiter = text[position + 2]
        position += 3
    if text[position : position + 1] != parameter_delimiter:
        raise refuse("malformed", "G", "Malformed global record delimiter.")
    values, _ = _split_parameters(
        text[position + 1 :], parameter_delimiter, record_delimiter, "G", limit
    )
    return parameter_delimiter, record_delimiter, [None, None, *values]


def _units(values: list[_Parameter]) -> tuple[int, Fraction]:
    padded = values + [None] * max(0, 16 - len(values))
    scale = _real(padded[12], "G13", 1.0)
    if scale != 1.0:
        raise refuse("units", "G13", "A model space scale other than 1 is not admitted.")
    flag = _integer(padded[13], "G14", 1)
    name = _string(padded[14]).strip().upper()
    if flag == 3:
        if name not in _UNIT_NAMES:
            raise refuse("units", "G15", f"Unknown IGES unit name {name!r}.")
        flag = _UNIT_NAMES[name]
    if flag not in _UNITS:
        raise refuse("units", "G14", f"Unknown IGES unit flag {flag}.")
    if name and _UNIT_NAMES.get(name, flag) != flag:
        raise refuse(
            "units",
            "G15",
            f"The unit name {name!r} contradicts the unit flag {flag}.",
        )
    return flag, _UNITS[flag][1]


def _decode(data: bytes, policy: CadImportPolicy) -> _File:
    sections = _lines(data, "file")
    limits = policy.limits
    parameter_delimiter, record_delimiter, global_values = _global(
        sections["G"], limits.max_attributes
    )
    unit_flag, unit_meters = _units(global_values)
    directory = sections["D"]
    if len(directory) % 2:
        raise refuse("malformed", "D", "The directory has an odd number of records.")
    if len(directory) // 2 > limits.max_nodes:
        raise refuse("limit", "D", "The file exceeds the entity limit.")
    parameters = sections["P"]
    entries: dict[int, _Entry] = {}
    attributes = len(global_values)
    for index in range(0, len(directory), 2):
        number = index + 1
        label = f"D{number}"
        first = [directory[index][8 * k : 8 * k + 8].strip() for k in range(9)]
        second = [directory[index + 1][8 * k : 8 * k + 8] for k in range(9)]
        entity_type = _integer(first[0] or None, label)
        if _integer(second[0].strip() or None, label) != entity_type:
            raise refuse("malformed", label, "Directory entry type fields disagree.")
        pointer = _integer(first[1] or None, label)
        count = _integer(second[3].strip() or None, label)
        if pointer < 1 or count < 1 or pointer + count - 1 > len(parameters):
            raise refuse(
                "dangling-reference", label, "Parameter data lies outside the file."
            )
        records = parameters[pointer - 1 : pointer - 1 + count]
        if any(
            not record[64:72].strip().isdigit() or int(record[64:72]) != number
            for record in records
        ):
            raise refuse("malformed", label, "Parameter records do not point back to it.")
        text = "".join(record[:64] for record in records)
        values, _ = _split_parameters(
            text,
            parameter_delimiter,
            record_delimiter,
            label,
            limits.max_attributes - attributes,
        )
        attributes += len(values)
        if _integer(values[0], label) != entity_type:
            raise refuse("malformed", label, "Parameter data names another entity type.")
        entries[number] = _Entry(
            number,
            entity_type,
            _integer(first[6] or None, label, 0),
            _integer(second[4].strip() or None, label, 0),
            second[7].strip(),
            tuple(values[1:]),
        )
    return _File(unit_flag, unit_meters, entries, attributes)


# ------------------------------------------------------------------ resolution


class _Resolver:
    """Lower the IGES entity graph to one exact native stage."""

    def __init__(self, file: _File, policy: CadImportPolicy, factor: float) -> None:
        self.file = file
        self.policy = policy
        self.factor = factor
        self.stage = CadStage(
            1.0,
            policy.pcurve_fit,
            relative_geometric_tolerance=policy.relative_geometric_tolerance,
        )
        self.active: list[int] = []
        self.used: set[int] = set()
        self.counts: dict[str, int] = {}
        self.placements: dict[tuple[int, float], RigidPlacement] = {}
        self.vertices: dict[tuple[int, int], int] = {}
        self.edges: dict[tuple[int, int], int] = {}
        self.solid_indices: dict[int, int] = {}
        self.occurrence_paths: set[tuple[str, ...]] = set()
        self.native_names: dict[int, tuple[str, ...] | None] = {}
        self.definition_uses: dict[int, int] = {}
        for entry in file.entries.values():
            if entry.entity_type == 408:
                definition = self.integer(entry, 0)
                self.definition_uses[definition] = (
                    self.definition_uses.get(definition, 0) + 1
                )
        self.maximum_depth = 0

    # ---------------------------------------------------------- graph access

    def entry(
        self, pointer: int, chain: tuple[str, ...], allowed: tuple[int, ...]
    ) -> _Entry:
        if pointer not in self.file.entries:
            raise refuse(
                "dangling-reference",
                f"D{pointer}",
                "The pointer does not name a directory entry.",
                chain,
            )
        entry = self.file.entries[pointer]
        if entry.entity_type not in allowed:
            raise refuse(
                "unsupported-entity",
                entry.name,
                f"Entity type {entry.entity_type} (form {entry.form}) is not admitted "
                f"here; expected one of {allowed}.",
                chain,
            )
        if pointer in self.active:
            raise refuse(
                "cyclic-reference", entry.name, "The entity graph is cyclic.", chain
            )
        if len(chain) > self.policy.limits.max_depth:
            raise refuse(
                "limit", entry.name, "The reference depth exceeds its limit.", chain
            )
        stage_size = (
            len(self.stage.vertices)
            + len(self.stage.curves)
            + len(self.stage.edge_curves)
            + len(self.stage.faces)
            + len(self.stage.shells)
        )
        if stage_size >= self.policy.limits.max_attributes:
            raise refuse(
                "limit",
                entry.name,
                "Expanded native topology exceeds its allocation budget.",
                chain,
            )
        self.maximum_depth = max(self.maximum_depth, len(chain))
        key = str(entry.entity_type)
        if pointer not in self.used:
            self.counts[key] = self.counts.get(key, 0) + 1
        self.used.add(pointer)
        return entry

    def _parameter(self, entry: _Entry, index: int) -> _Parameter:
        return entry.parameters[index] if index < len(entry.parameters) else None

    def integer(self, entry: _Entry, index: int, default: int | None = None) -> int:
        return _integer(
            self._parameter(entry, index), f"{entry.name} P{index + 1}", default
        )

    def count(self, entry: _Entry, index: int) -> int:
        value = self.integer(entry, index)
        if value < 0:
            raise refuse("malformed", entry.name, "An entity count is negative.")
        if value > self.policy.limits.max_attributes or value > len(entry.parameters):
            raise refuse(
                "limit", entry.name, "An entity count exceeds the bounded parameter data."
            )
        return value

    def real(self, entry: _Entry, index: int, default: float | None = None) -> float:
        return _real(self._parameter(entry, index), f"{entry.name} P{index + 1}", default)

    def reals(self, entry: _Entry, start: int, count: int) -> np.ndarray:
        return np.asarray([self.real(entry, start + k) for k in range(count)])

    def point(self, entry: _Entry, start: int) -> np.ndarray:
        return self.reals(entry, start, 3) * self.factor

    def placement(self, entry: _Entry, chain: tuple[str, ...]) -> RigidPlacement:
        if entry.transform == 0:
            return RigidPlacement.identity()
        key = (entry.transform, self.factor)
        if key in self.placements:
            return self.placements[key]
        if not chain or chain[-1] != entry.name:
            chain = (*chain, entry.name)
        matrix = self.entry(entry.transform, chain, (124,))
        if matrix.form not in (0, 1):
            raise refuse(
                "unsupported-entity",
                matrix.name,
                "Only rigid (form 0/1) transformation matrices are admitted.",
                chain,
            )
        values = self.reals(matrix, 0, 12).reshape(3, 4)
        values[:, 3] *= self.factor
        self.active.append(matrix.number)
        outer = self.placement(matrix, (*chain, matrix.name))
        self.active.pop()
        placed = outer.compose(
            RigidPlacement.from_matrix(values, matrix.name, reflection=True)
        )
        self.placements[key] = placed
        return placed

    # -------------------------------------------------------------- geometry

    def curve(self, pointer: int, chain: tuple[str, ...]) -> AbstractCurve:
        entry = self.entry(pointer, chain, (100, 102, 104, 110, 126))
        chain = (*chain, entry.name)
        self.active.append(pointer)
        placement = self.placement(entry, chain)
        match entry.entity_type:
            case 110:
                start, end = self.point(entry, 0), self.point(entry, 3)
                length = float(np.linalg.norm(end - start))
                if not length > 0.0:
                    raise refuse(
                        "inconsistent-geometry",
                        entry.name,
                        "A line has zero length.",
                        chain,
                    )
                curve: AbstractCurve = LineCurve(start, (end - start) / length)
            case 100:
                height = self.real(entry, 0) * self.factor
                center = (
                    np.asarray((self.real(entry, 1), self.real(entry, 2))) * self.factor
                )
                start = (
                    np.asarray((self.real(entry, 3), self.real(entry, 4))) * self.factor
                )
                radius = float(np.linalg.norm(start - center))
                curve = CircleCurve(
                    (center[0], center[1], height),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    radius,
                )
            case 104:
                curve = self._conic(entry, chain)
            case 126:
                try:
                    curve = self._spline_curve(entry, 3)
                except (ValueError, TypeError) as error:
                    if isinstance(error, CadInterchangeError):
                        raise refuse(
                            error.refusal.reason,
                            error.refusal.entity,
                            error.refusal.message,
                            chain,
                        ) from error
                    raise refuse("malformed", entry.name, str(error), chain) from error
            case 102:
                count = self.integer(entry, 0)
                if count != 1:
                    raise refuse(
                        "unsupported-entity",
                        entry.name,
                        "A composite curve with several segments is not one edge carrier.",
                        chain,
                    )
                curve = self.curve(self.integer(entry, 1), chain)
            case _:
                raise RuntimeError("entry() restricts curve types.")
        self.active.pop()
        return place_curve(curve, placement, entry.name)

    def _conic(self, entry: _Entry, chain: tuple[str, ...]) -> CircleCurve | EllipseCurve:
        a, b, c, d, e, f = (self.real(entry, k) for k in range(6))
        height = self.real(entry, 6) * self.factor
        if entry.form not in (0, 1):
            raise refuse(
                "unsupported-entity",
                entry.name,
                "Only ellipse conics are admitted.",
                chain,
            )
        quadratic = np.asarray(((a, 0.5 * b), (0.5 * b, c)))
        linear = np.asarray((d, e))
        eigenvalues, axes = np.linalg.eigh(quadratic)
        if eigenvalues[0] * eigenvalues[1] <= 0.0:
            raise refuse(
                "unsupported-entity",
                entry.name,
                "Parabolic and hyperbolic conics have no exact native carrier.",
                chain,
            )
        center = -0.5 * np.linalg.solve(quadratic, linear)
        right = -(float(center @ quadratic @ center + linear @ center) + f)
        squared = right / eigenvalues
        if np.any(squared <= 0.0):
            raise refuse(
                "inconsistent-geometry",
                entry.name,
                "An ellipse conic has no real positive radii.",
                chain,
            )
        if b == d == e == 0.0:
            axes = np.eye(2)
            squared = np.asarray((-f / a, -f / c))
        if np.linalg.det(axes) < 0.0:
            axes[:, 1] *= -1.0
        first, second = np.sqrt(squared) * self.factor
        origin = np.asarray((center[0] * self.factor, center[1] * self.factor, height))
        first_axis, second_axis = np.pad(axes.T, ((0, 0), (0, 1)))
        if first == second:
            return CircleCurve(origin, first_axis, second_axis, float(first))
        return EllipseCurve(origin, first_axis, second_axis, float(first), float(second))

    def _spline_curve(self, entry: _Entry, dimension: int) -> BSplineCurve:
        upper, degree = self.integer(entry, 0), self.integer(entry, 1)
        if (
            upper < degree
            or degree < 1
            or upper + degree + 2 > self.policy.limits.max_attributes
        ):
            raise refuse("malformed", entry.name, "Inconsistent B-spline curve sizes.")
        if any(self.integer(entry, index) not in (0, 1) for index in range(2, 6)):
            raise refuse(
                "malformed", entry.name, "B-spline curve properties must be binary."
            )
        count = upper + 1
        start = 6
        knots = self.reals(entry, start, count + degree + 1)
        start += count + degree + 1
        weights = self.reals(entry, start, count)
        start += count
        points = self.reals(entry, start, 3 * count).reshape(count, 3) * self.factor
        domain = self.reals(entry, start + 3 * count, 2)
        if not knots[degree] <= domain[0] < domain[1] <= knots[-degree - 1]:
            raise refuse(
                "malformed",
                entry.name,
                "Curve domain lies outside its native knot support.",
            )
        return BSplineCurve(points[:, :dimension], weights, knots, degree)

    def direction(self, pointer: int, chain: tuple[str, ...]) -> np.ndarray:
        entry = self.entry(pointer, chain, (123,))
        vector = self.reals(entry, 0, 3)
        norm = float(np.linalg.norm(vector))
        if not norm > 0.0:
            raise refuse(
                "inconsistent-geometry", entry.name, "A direction is zero.", chain
            )
        return self.placement(entry, chain).vector(vector / norm)

    def location(self, pointer: int, chain: tuple[str, ...]) -> np.ndarray:
        entry = self.entry(pointer, chain, (116,))
        return self.placement(entry, chain).point(self.point(entry, 0))

    def surface(self, pointer: int, chain: tuple[str, ...]) -> AbstractSurfacePatch:
        entry = self.entry(
            pointer, chain, (108, 120, 122, 128, 140, 190, 192, 194, 196, 198)
        )
        chain = (*chain, entry.name)
        self.active.append(pointer)
        placement = self.placement(entry, chain)
        patch = self._surface(entry, chain)
        self.active.pop()
        return place_surface(patch, placement, entry.name)

    def _frame(
        self,
        entry: _Entry,
        axis: np.ndarray,
        reference: int | None,
        chain: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        if reference:
            first = self.direction(reference, chain)
            first = first - (first @ axis) * axis
            first = first / np.linalg.norm(first)
        else:
            first, _ = _axis_frame(axis)
        return first, np.cross(axis, first)

    def _surface(self, entry: _Entry, chain: tuple[str, ...]) -> AbstractSurfacePatch:
        match entry.entity_type:
            case 108:
                normal = self.reals(entry, 0, 3)
                norm = float(np.linalg.norm(normal))
                if not norm > 0.0:
                    raise refuse(
                        "inconsistent-geometry",
                        entry.name,
                        "A plane normal is zero.",
                        chain,
                    )
                axis = normal / norm
                origin = axis * self.real(entry, 3) / norm * self.factor
                first, second = self._frame(entry, axis, None, chain)
                return PlanePatch(origin, first, second)
            case 190:
                origin = self.location(self.integer(entry, 0), chain)
                axis = self.direction(self.integer(entry, 1), chain)
                first, second = self._frame(entry, axis, self.integer(entry, 2, 0), chain)
                return PlanePatch(origin, first, second)
            case 192:
                origin = self.location(self.integer(entry, 0), chain)
                axis = self.direction(self.integer(entry, 1), chain)
                radius = self.real(entry, 2) * self.factor
                first, second = self._frame(entry, axis, self.integer(entry, 3, 0), chain)
                return CylinderPatch(origin, first, second, axis, radius)
            case 194:
                origin = self.location(self.integer(entry, 0), chain)
                axis = self.direction(self.integer(entry, 1), chain)
                radius = self.real(entry, 2) * self.factor
                angle = radians(self.real(entry, 3))
                first, second = self._frame(entry, axis, self.integer(entry, 4, 0), chain)
                return ConePatch(origin, first, second, axis, radius, angle)
            case 196:
                center = self.location(self.integer(entry, 0), chain)
                radius = self.real(entry, 1) * self.factor
                axis_pointer = self.integer(entry, 2, 0)
                axis = (
                    self.direction(axis_pointer, chain)
                    if axis_pointer
                    else np.asarray((0.0, 0.0, 1.0))
                )
                first, second = self._frame(entry, axis, self.integer(entry, 3, 0), chain)
                return SpherePatch(center, first, second, axis, radius)
            case 198:
                center = self.location(self.integer(entry, 0), chain)
                axis = self.direction(self.integer(entry, 1), chain)
                major, minor = (
                    self.real(entry, 2) * self.factor,
                    self.real(entry, 3) * self.factor,
                )
                first, second = self._frame(entry, axis, self.integer(entry, 4, 0), chain)
                return TorusPatch(center, first, second, axis, major, minor)
            case 120:
                axis = self.curve(self.integer(entry, 0), chain)
                if not isinstance(axis, LineCurve):
                    raise refuse(
                        "malformed",
                        entry.name,
                        "A revolution axis must be a line.",
                        chain,
                    )
                generatrix_pointer = self.integer(entry, 1)
                generatrix = self.curve(generatrix_pointer, chain)
                if isinstance(generatrix, LineCurve):
                    # IGES parameterizes a line on [0, 1], not by its length.
                    line = self.file.entries[generatrix_pointer]
                    while line.entity_type == 102:
                        line = self.file.entries[self.integer(line, 1)]
                    length = self._range(line, generatrix)[1]
                    generatrix = LineCurve(
                        generatrix.origin, np.asarray(generatrix.direction) * length
                    )
                # IGES 120 is S(t, theta) = R(theta) C(t) with normal
                # S_t x S_theta. Revolving about the reversed axis with
                # u = 2 pi - theta, v = t keeps that normal and loop orientation.
                return RevolutionSurface(generatrix, axis.origin, -axis.direction)
            case 122:
                directrix_pointer = self.integer(entry, 0)
                directrix = self.curve(directrix_pointer, chain)
                end = self.point(entry, 1)
                start = self._curve_start(directrix)
                return ExtrusionSurface(directrix, end - start)
            case 128:
                try:
                    return self._spline_surface(entry)
                except (ValueError, TypeError) as error:
                    if isinstance(error, CadInterchangeError):
                        raise refuse(
                            error.refusal.reason,
                            error.refusal.entity,
                            error.refusal.message,
                            chain,
                        ) from error
                    raise refuse("malformed", entry.name, str(error), chain) from error
            case 140:
                if entry.form != 0:
                    raise refuse(
                        "unsupported-entity",
                        entry.name,
                        "Offset surfaces require form 0.",
                        chain,
                    )
                # IGES5.3 section4.30 defines the indicator as advisory; it
                # does not participate in S + d (S_u cross S_v)/|S_u cross S_v|.
                self.reals(entry, 0, 3)
                distance = self.real(entry, 3) * self.factor
                if distance == 0.0:
                    raise refuse(
                        "malformed",
                        entry.name,
                        "An IGES normal-offset distance must be nonzero.",
                        chain,
                    )
                return OffsetSurface(
                    self.surface(self.integer(entry, 4), chain), distance
                )
            case _:
                raise RuntimeError("entry() restricts surface types.")

    @staticmethod
    def _curve_start(curve: AbstractCurve) -> np.ndarray:
        match curve:
            case LineCurve():
                return np.asarray(curve.origin)
            case BSplineCurve():
                domain = curve.parameter_domain
                first = 0.0 if domain is None else domain[0]
                return np.asarray(curve.evaluate(jnp.asarray(first, dtype=jnp.float64)))
            case _:
                return np.asarray(curve.evaluate(jnp.asarray(0.0, dtype=jnp.float64)))

    def _spline_surface(self, entry: _Entry) -> BSplineSurfacePatch:
        upper_u, upper_v = self.integer(entry, 0), self.integer(entry, 1)
        degree_u, degree_v = self.integer(entry, 2), self.integer(entry, 3)
        count_u, count_v = upper_u + 1, upper_v + 1
        if (
            upper_u < degree_u
            or upper_v < degree_v
            or min(degree_u, degree_v) < 1
            or 4 * count_u * count_v > self.policy.limits.max_attributes
        ):
            raise refuse("malformed", entry.name, "Inconsistent B-spline surface sizes.")
        if any(self.integer(entry, index) not in (0, 1) for index in range(4, 9)):
            raise refuse(
                "malformed", entry.name, "B-spline surface properties must be binary."
            )
        start = 9
        u_knots = self.reals(entry, start, count_u + degree_u + 1)
        start += count_u + degree_u + 1
        v_knots = self.reals(entry, start, count_v + degree_v + 1)
        start += count_v + degree_v + 1
        weights = self.reals(entry, start, count_u * count_v)
        start += count_u * count_v
        points = self.reals(entry, start, 3 * count_u * count_v) * self.factor
        domain = self.reals(entry, start + 3 * count_u * count_v, 4)
        if not (
            u_knots[degree_u] <= domain[0] < domain[1] <= u_knots[-degree_u - 1]
            and v_knots[degree_v] <= domain[2] < domain[3] <= v_knots[-degree_v - 1]
        ):
            raise refuse(
                "malformed",
                entry.name,
                "Surface domain lies outside its native knot support.",
            )
        # IGES orders controls with the first (u) index varying fastest.
        return BSplineSurfacePatch(
            points.reshape(count_v, count_u, 3).transpose(1, 0, 2),
            weights.reshape(count_v, count_u).T,
            u_knots,
            v_knots,
            degree_u,
            degree_v,
        )

    # -------------------------------------------------------------- topology

    def vertex(self, pointer: int, index: int, chain: tuple[str, ...]) -> int:
        key = (pointer, index)
        if key not in self.vertices:
            entry = self.entry(pointer, chain, (502,))
            count = self.count(entry, 0)
            if not 1 <= index <= count:
                raise refuse(
                    "dangling-reference", entry.name, f"Vertex {index} is absent.", chain
                )
            point = self.placement(entry, chain).point(
                self.point(entry, 1 + 3 * (index - 1))
            )
            self.vertices[key] = self.stage.vertex(point, f"{entry.name}[{index}]")
        return self.vertices[key]

    def edge(self, pointer: int, index: int, chain: tuple[str, ...]) -> int:
        key = (pointer, index)
        if key not in self.edges:
            entry = self.entry(pointer, chain, (504,))
            chain = (*chain, entry.name)
            count = self.count(entry, 0)
            if not 1 <= index <= count:
                raise refuse(
                    "dangling-reference", entry.name, f"Edge {index} is absent.", chain
                )
            base = 1 + 5 * (index - 1)
            self.active.append(pointer)
            curve_pointer = self.integer(entry, base)
            curve = self.curve(curve_pointer, chain)
            start = self.vertex(
                self.integer(entry, base + 1), self.integer(entry, base + 2), chain
            )
            end = self.vertex(
                self.integer(entry, base + 3), self.integer(entry, base + 4), chain
            )
            self.active.pop()
            label = f"{entry.name}[{index}]"
            try:
                self.edges[key] = self.stage.edge(
                    curve,
                    start,
                    end,
                    label,
                    parameter_range=self._range(self.file.entries[curve_pointer], curve),
                )
            except ValueError as error:
                raise refuse("inconsistent-geometry", label, str(error), chain) from error
        return self.edges[key]

    def parameter_curve(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> AbstractCurve:
        """Decode a UV carrier without treating dimensionless parameters as lengths."""
        entry = self.entry(pointer, chain, (100, 102, 104, 110, 126))
        if entry.entity_type == 102:
            pieces = self.parameter_segments(pointer, patch, chain)
            if len(pieces) != 1:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "A piecewise UV carrier cannot be represented as one native coedge.",
                    (*chain, entry.name),
                )
            return pieces[0]
        factor = self.factor
        self.factor = 1.0
        try:
            curve = self.curve(pointer, chain)
            domain = self._range(self.file.entries[pointer], curve)
        finally:
            self.factor = factor
        match curve:
            case LineCurve():
                points = np.asarray(
                    curve.evaluate(jnp.asarray(domain, dtype=jnp.float64))
                )[:, :2]
                result: AbstractCurve = BSplineCurve(
                    points, np.ones(2), [domain[0], domain[0], domain[1], domain[1]], 1
                )
            case BSplineCurve():
                result = BSplineCurve(
                    np.asarray(curve.control_points)[:, :2],
                    curve.weights,
                    curve.knots,
                    curve.degree,
                )
            case CircleCurve():
                result = CircleCurve(
                    np.asarray(curve.center)[:2],
                    np.asarray(curve.first_axis)[:2],
                    np.asarray(curve.second_axis)[:2],
                    curve.radius,
                )
            case EllipseCurve():
                result = EllipseCurve(
                    np.asarray(curve.center)[:2],
                    np.asarray(curve.first_axis)[:2],
                    np.asarray(curve.second_axis)[:2],
                    curve.first_radius,
                    curve.second_radius,
                )
            case _:
                raise refuse(
                    "unsupported-entity",
                    f"D{pointer}",
                    "No exact UV carrier for this parameter-space curve.",
                    chain,
                )
        while isinstance(patch, OffsetSurface):
            patch = patch.base
        angular = radians(1.0)
        factors = (
            (factor, factor)
            if isinstance(patch, PlanePatch)
            else (angular, factor)
            if isinstance(patch, (CylinderPatch, ConePatch))
            else (angular, angular)
            if isinstance(patch, (SpherePatch, TorusPatch))
            else (-1.0, 1.0)
            if isinstance(patch, RevolutionSurface)
            else (1.0, factor)
            if isinstance(patch, ExtrusionSurface)
            else (1.0, 1.0)
        )
        revolution = isinstance(patch, RevolutionSurface)
        if revolution or isinstance(patch, TorusPatch):
            # IGES 198 orders tube angle before azimuth and IGES 120 orders the
            # generatrix parameter before the angle; native patches put the
            # azimuth first.
            match result:
                case BSplineCurve():
                    result = BSplineCurve(
                        np.asarray(result.control_points)[:, ::-1],
                        result.weights,
                        result.knots,
                        result.degree,
                    )
                case CircleCurve():
                    result = CircleCurve(
                        np.asarray(result.center)[::-1],
                        np.asarray(result.first_axis)[::-1],
                        np.asarray(result.second_axis)[::-1],
                        result.radius,
                    )
                case EllipseCurve():
                    result = EllipseCurve(
                        np.asarray(result.center)[::-1],
                        np.asarray(result.first_axis)[::-1],
                        np.asarray(result.second_axis)[::-1],
                        result.first_radius,
                        result.second_radius,
                    )
                case _:
                    raise RuntimeError("UV decoding restricts exact carrier types.")
        try:
            return transform_pcurve(
                result, factors, 1.0, (2.0 * np.pi, 0.0) if revolution else (0.0, 0.0)
            )
        except ValueError as error:
            raise refuse(
                "unsupported-entity", f"D{pointer}", str(error), chain
            ) from error

    def align_parameter_curve(
        self, curve: AbstractCurve, edge: int, patch: AbstractSurfacePatch
    ) -> AbstractCurve:
        """Affine reparameterization of a file UV carrier to the staged edge range."""
        first, last = self.stage.edge_ranges[edge]
        if isinstance(curve, BSplineCurve):
            domain = curve.parameter_domain
            if domain is None:
                raise refuse(
                    "inconsistent-geometry",
                    f"edge:{edge}",
                    "A B-spline p-curve requires a finite parameter domain.",
                )
            lower, upper = domain
            # Closed edges have identical endpoint vertices; interior carrier
            # correspondence, not endpoint coincidence, owns their direction.
            samples = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)
            world = np.asarray(
                patch.evaluate(
                    curve.evaluate(jnp.asarray(lower + samples * (upper - lower)))
                )
            )
            carrier_index = self.stage.edge_curves[edge]
            carrier = self.stage.curves[carrier_index]
            reference = np.asarray(
                carrier.evaluate(jnp.asarray(first + samples * (last - first)))
            )
            forward_error = float(np.max(np.linalg.norm(world - reference, axis=1)))
            reverse_error = float(np.max(np.linalg.norm(world[::-1] - reference, axis=1)))
            reverse = reverse_error < forward_error
            if reverse:
                points, weights = (
                    np.asarray(curve.control_points)[::-1],
                    np.asarray(curve.weights)[::-1],
                )
                knots = lower + upper - np.asarray(curve.knots)[::-1]
            else:
                points, weights, knots = (
                    curve.control_points,
                    curve.weights,
                    np.asarray(curve.knots),
                )
            if reverse or (lower, upper) != (first, last):
                return BSplineCurve(
                    points,
                    weights,
                    first + (knots - lower) * ((last - first) / (upper - lower)),
                    curve.degree,
                )
        return curve

    def loop(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> list[StagedCoedge]:
        entry = self.entry(pointer, chain, (508,))
        chain = (*chain, entry.name)
        self.active.append(pointer)
        count = self.count(entry, 0)
        position = 1
        coedges = []
        for _ in range(count):
            kind = self.integer(entry, position)
            list_pointer = self.integer(entry, position + 1)
            index = self.integer(entry, position + 2)
            agrees = self.integer(entry, position + 3)
            curves = self.integer(entry, position + 4)
            if curves < 0 or agrees not in (0, 1) or kind not in (0, 1):
                raise refuse(
                    "malformed", entry.name, "Malformed loop edge record.", chain
                )
            if curves > 1:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "A multi-segment parameter-space edge requires a piecewise carrier.",
                    chain,
                )
            pcurve = (
                self.parameter_curve(self.integer(entry, position + 6), patch, chain)
                if curves
                else None
            )
            position += 5 + 2 * curves
            if kind == 1:
                vertex = self.vertex(list_pointer, index, chain)
                if pcurve is None or pcurve.parameter_domain is None:
                    raise refuse(
                        "unsupported-entity",
                        entry.name,
                        "A collapsed vertex use requires a bounded parameter-space curve.",
                        chain,
                    )
                domain = pcurve.parameter_domain
                if (
                    isinstance(pcurve, BSplineCurve)
                    and pcurve.degree == 1
                    and len(pcurve.control_points) == 2
                ):
                    points = np.asarray(pcurve.control_points)
                    delta = points[1] - points[0]
                    moving = int(np.argmax(np.abs(delta)))
                    period = patch.periods[moving]
                    if (
                        period is not None
                        and abs(float(delta[moving])) == period
                        and delta[1 - moving] == 0.0
                    ):
                        lower, upper = domain
                        # A vertex use has no 3D parameter to preserve. Choose
                        # its authored angular extent as the exact native phase.
                        pcurve = BSplineCurve(
                            points,
                            pcurve.weights,
                            (np.asarray(pcurve.knots) - lower)
                            * (period / (upper - lower)),
                            pcurve.degree,
                        )
                        domain = (0.0, period)
                edge = self.stage.degenerate(vertex, *domain)
                # A vertex has no model-space direction. Its parameter-space
                # curve is already directed in loop order (IGES 508).
                agrees = 1
            else:
                edge = self.edge(list_pointer, index, chain)
                if pcurve is not None:
                    pcurve = self.align_parameter_curve(pcurve, edge, patch)
            coedges.append(
                StagedCoedge(
                    edge, 1 if agrees else -1, pcurve, f"{entry.name}[{len(coedges)}]"
                )
            )
        self.active.pop()
        if not coedges:
            raise refuse("malformed", entry.name, "A loop has no edges.", chain)
        return coedges

    def face(self, pointer: int, orientation: int, chain: tuple[str, ...]) -> int:
        entry = self.entry(pointer, chain, (510,))
        chain = (*chain, entry.name)
        self.active.append(pointer)
        patch = self.surface(self.integer(entry, 0), chain)
        count = self.count(entry, 1)
        outer = self.integer(entry, 2)
        loops = [
            self.loop(self.integer(entry, 3 + k), patch, chain) for k in range(count)
        ]
        self.active.pop()
        if not loops:
            raise refuse("unsupported-entity", entry.name, "A face has no loops.", chain)
        face = self.stage.face(
            patch,
            loops,
            orientation,
            surface_tag(patch),
            entry.name,
            outer_known=outer == 1,
        )
        self.place_faces([face], self.placement(entry, chain), entry.name)
        return face

    def shell(self, pointer: int, sign: int, closed: bool, chain: tuple[str, ...]) -> int:
        entry = self.entry(pointer, chain, (514,))
        chain = (*chain, entry.name)
        self.active.append(pointer)
        count = self.count(entry, 0)
        faces = []
        for k in range(count):
            agrees = self.integer(entry, 2 + 2 * k)
            faces.append(
                self.face(
                    self.integer(entry, 1 + 2 * k), sign * (1 if agrees else -1), chain
                )
            )
        self.active.pop()
        if not faces:
            raise refuse("malformed", entry.name, "A shell has no faces.", chain)
        self.place_faces(faces, self.placement(entry, chain), entry.name)
        self.stage.shells.append((faces, closed))
        return len(self.stage.shells) - 1

    def place_faces(
        self, faces: list[int], placement: RigidPlacement, label: str
    ) -> None:
        """Clone shared incidences before placing a face/shell's model-space geometry."""
        if placement.is_identity:
            return
        if not placement.proper:
            raise refuse(
                "unsupported-entity",
                label,
                "Reflected subshape placements are not admitted.",
                (label,),
            )
        vertices: dict[int, int] = {}
        edges: dict[int, int] = {}
        for face in faces:
            staged = self.stage.faces[face]
            staged.patch = place_surface(staged.patch, placement, label)
            for loop in staged.loops:
                for coedge in loop:
                    original = coedge.edge
                    if original not in edges:
                        if (
                            len(self.stage.edge_curves) + len(self.stage.vertices) + 3
                            >= self.policy.limits.max_attributes
                        ):
                            raise refuse(
                                "limit",
                                label,
                                "Placed topology exceeds its allocation budget.",
                                (label,),
                            )
                        ends = self.stage.edge_vertices[original]
                        for vertex in ends:
                            if vertex not in vertices:
                                vertices[vertex] = self.stage.vertex(
                                    placement.point(self.stage.vertices[vertex]), label
                                )
                        first, last = self.stage.edge_ranges[original]
                        curve = self.stage.edge_curves[original]
                        if curve < 0:
                            edges[original] = self.stage.degenerate(
                                vertices[ends[0]], first, last
                            )
                        else:
                            edges[original] = self.stage.edge(
                                place_curve(self.stage.curves[curve], placement, label),
                                vertices[ends[0]],
                                vertices[ends[1]],
                                label,
                                parameter_range=(first, last),
                            )
                    coedge.edge = edges[original]

    def solid(self, entry: _Entry, chain: tuple[str, ...] = ()) -> int:
        if entry.number in self.solid_indices:
            return self.solid_indices[entry.number]
        entry = self.entry(entry.number, chain, (186,))
        chain = (*chain, entry.name)
        self.active.append(entry.number)
        outer_sign = 1 if self.integer(entry, 1) else -1
        shells = [self.shell(self.integer(entry, 0), outer_sign, True, chain)]
        voids = self.count(entry, 2)
        for k in range(voids):
            sign = 1 if self.integer(entry, 4 + 2 * k) else -1
            shells.append(self.shell(self.integer(entry, 3 + 2 * k), sign, True, chain))
        self.active.pop()
        index = len(self.stage.solids)
        self.stage.solids.append(shells)
        self.solid_indices[entry.number] = index
        return index

    def occurrence(
        self,
        pointer: int,
        outer: RigidPlacement,
        path: tuple[str, ...],
        chain: tuple[str, ...] = (),
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
        """Expand actual subfigure membership, never grouping placements or prefixes."""
        entry = self.entry(pointer, chain, (186, 308, 408))
        chain = (*chain, entry.name)
        if entry.entity_type == 186:
            solid = self.solid(entry, chain[:-1])
            placement = outer.compose(self.placement(entry, chain))
            if not placement.proper:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "Reflected solid occurrences are not admitted.",
                    chain,
                )
            if len(self.stage.occurrences) >= self.policy.maximum_occurrences:
                raise refuse(
                    "limit", entry.name, "Expanded occurrences exceed their limit.", chain
                )
            source_path = path or (f"solid{solid}",)
            if source_path in self.occurrence_paths:
                raise refuse(
                    "malformed",
                    entry.name,
                    "Distinct occurrences require distinct scientific instance paths.",
                    chain,
                )
            self.occurrence_paths.add(source_path)
            self.stage.occurrences.append(
                BRepOccurrence(
                    source_path, solid, placement.rotation, placement.translation
                )
            )
            return (source_path,), ()
        self.active.append(pointer)
        placement = outer.compose(self.placement(entry, chain))
        if entry.entity_type == 408:
            scale = self.real(entry, 4, 1.0)
            if scale != 1.0:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "Scaled subfigure instances have no rigid native representation.",
                    chain,
                )
            definition = self.integer(entry, 0)
            translation = self.reals(entry, 1, 3) * self.factor
            placement = placement.compose(RigidPlacement(np.eye(3), translation))
            if self.definition_uses.get(definition, 0) > 1:
                path = (*path, entry.label or f"instance{entry.number}")
            result = self.occurrence(definition, placement, path, chain)
            self.active.pop()
            return result
        name = _string(self._parameter(entry, 1)) or entry.label or entry.name
        if pointer not in self.native_names:
            self.native_names[pointer] = _decode_path_name(
                name, entry.name, self.policy.limits.max_attributes
            )
        native_path = self.native_names[pointer]
        count = self.count(entry, 2)
        if count > self.policy.limits.max_nodes:
            raise refuse(
                "limit", entry.name, "Subfigure member count exceeds its limit.", chain
            )
        children = [self.integer(entry, 3 + k) for k in range(count)]
        depth = self.integer(entry, 0)
        if depth < 0:
            raise refuse(
                "malformed", entry.name, "Subfigure hierarchy depth is negative.", chain
            )
        single_part = (
            count == 1
            and depth == 0
            and self.file.entries.get(children[0], None) is not None
            and self.file.entries[children[0]].entity_type == 186
        )
        scientific_path = native_path if native_path is not None else (*path, name)
        if single_part:
            result = self.occurrence(children[0], placement, scientific_path, chain)
            self.active.pop()
            return result
        members: list[tuple[str, ...]] = []
        containers: list[tuple[str, ...]] = []
        for child in children:
            target = self.file.entries.get(child)
            child_path = scientific_path
            if target is not None and target.entity_type == 186:
                child_path = (*scientific_path, target.label or target.name)
            child_members, child_containers = self.occurrence(
                child, placement, child_path, chain
            )
            members.extend(child_members)
            containers.extend(child_containers)
        self.active.pop()
        if not members and not containers:
            return (), ()
        self.stage.assembly_containers.append(
            BRepAssemblyContainer(scientific_path, tuple(members), tuple(containers))
        )
        return (), (scientific_path,)

    def trimmed_face(self, entry: _Entry) -> None:
        chain = (entry.name,)
        self.active.append(entry.number)
        patch = self.surface(self.integer(entry, 0), chain)
        if self.integer(entry, 1) == 0:
            loops = [self.stage.natural_loop(patch, entry.name)]
        else:
            loops = [self.boundary(self.integer(entry, 3), patch, chain)]
        loops.extend(
            self.boundary(self.integer(entry, 4 + k), patch, chain)
            for k in range(self.count(entry, 2))
        )
        self.active.pop()
        self.stage.face(
            patch,
            loops,
            1,
            surface_tag(patch),
            entry.name,
            outer_known=True,
            loops_by_containment=True,
        )

    def boundary(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> list[StagedCoedge]:
        """Both model-space and parameter-space curves of an IGES 142 boundary."""
        entry = self.entry(pointer, chain, (142,))
        chain = (*chain, entry.name)
        self.active.append(pointer)
        model_curve = self.integer(entry, 3, 0)
        if model_curve == 0:
            raise refuse(
                "unsupported-entity",
                entry.name,
                "Curves on surface without a model-space curve are not admitted.",
                chain,
            )
        segments = self._segments(model_curve, chain)
        parameter_pointer = self.integer(entry, 2, 0)
        parameters = (
            self.parameter_segments(parameter_pointer, patch, chain)
            if parameter_pointer
            else [None] * len(segments)
        )
        if len(parameters) != len(segments):
            raise refuse(
                "unsupported-entity",
                entry.name,
                "Model-space and parameter-space boundary segmentation differs.",
                chain,
            )
        self.active.pop()
        coedges = []
        points = []
        for curve, first, last in segments:
            ends = np.asarray(
                curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64))
            )
            points.append(ends)
        tolerance = self.stage.tolerance
        vertices = []
        for index, ends in enumerate(points):
            start = vertices[-1] if vertices else self.stage.vertex(ends[0], entry.name)
            closing = index == len(points) - 1
            if closing:
                end = vertices[0] if vertices else start
                reference = points[0][0]
                if np.linalg.norm(ends[1] - reference) > tolerance:
                    raise refuse(
                        "inconsistent-geometry",
                        entry.name,
                        "A boundary is not closed.",
                        chain,
                    )
            else:
                if np.linalg.norm(ends[1] - points[index + 1][0]) > tolerance:
                    raise refuse(
                        "inconsistent-geometry",
                        entry.name,
                        "Boundary segments do not connect.",
                        chain,
                    )
                end = self.stage.vertex(ends[1], entry.name)
            if not vertices:
                vertices.append(start)
            vertices.append(end)
            curve = segments[index][0]
            label = f"{entry.name}[{index}]"
            try:
                edge = self.stage.edge(
                    curve, start, end, label, parameter_range=segments[index][1:]
                )
            except ValueError as error:
                raise refuse("inconsistent-geometry", label, str(error), chain) from error
            pcurve = parameters[index]
            if pcurve is not None:
                pcurve = self.align_parameter_curve(pcurve, edge, patch)
            coedges.append(StagedCoedge(edge, 1, pcurve, label))
        return coedges

    def parameter_segments(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> list[AbstractCurve]:
        entry = self.entry(pointer, chain, (100, 102, 104, 110, 126))
        if entry.entity_type != 102:
            return [self.parameter_curve(pointer, patch, chain)]
        self.active.append(pointer)
        chain = (*chain, entry.name)
        if entry.transform:
            raise refuse(
                "unsupported-entity",
                entry.name,
                "A transformed UV composite has no single parameter frame.",
                chain,
            )
        result = []
        for k in range(self.count(entry, 0)):
            result.extend(
                self.parameter_segments(self.integer(entry, 1 + k), patch, chain)
            )
        self.active.pop()
        return result

    def _segments(
        self, pointer: int, chain: tuple[str, ...]
    ) -> list[tuple[AbstractCurve, float, float]]:
        entry = self.entry(pointer, chain, (100, 102, 104, 110, 126))
        if entry.entity_type == 102:
            chain = (*chain, entry.name)
            self.active.append(pointer)
            placement = self.placement(entry, chain)
            if not placement.is_identity:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "Transformed composite curves are not admitted.",
                    chain,
                )
            segments = []
            for k in range(self.count(entry, 0)):
                segments.extend(self._segments(self.integer(entry, 1 + k), chain))
            self.active.pop()
            return segments
        curve = self.curve(pointer, chain)
        return [(curve, *self._range(entry, curve))]

    def _range(self, entry: _Entry, curve: AbstractCurve) -> tuple[float, float]:
        """Parameter range of a stand-alone IGES curve on its native carrier."""
        match entry.entity_type:
            case 110:
                if not isinstance(curve, LineCurve):
                    raise RuntimeError("Lines decode to line carriers.")
                length = float(
                    np.linalg.norm(self.point(entry, 3) - self.point(entry, 0))
                )
                return 0.0, length
            case 100 | 104:
                if entry.entity_type == 100:
                    center = np.asarray((self.real(entry, 1), self.real(entry, 2)))
                    start, end = self.reals(entry, 3, 2), self.reals(entry, 5, 2)
                    scale = np.ones(2)
                else:
                    local = self._conic(entry, (entry.name,))
                    center = np.asarray(local.center)[:2] / self.factor
                    start, end = self.reals(entry, 7, 2), self.reals(entry, 9, 2)
                    radius1 = (
                        float(local.radius)
                        if isinstance(local, CircleCurve)
                        else float(local.first_radius)
                    )
                    radius2 = (
                        float(local.radius)
                        if isinstance(local, CircleCurve)
                        else float(local.second_radius)
                    )
                    frame = np.stack(
                        (
                            np.asarray(local.first_axis)[:2],
                            np.asarray(local.second_axis)[:2],
                        )
                    )
                    start, end = frame @ (start - center), frame @ (end - center)
                    scale = np.asarray((radius1, radius2)) / self.factor
                    center = np.zeros(2)
                start, end = (start - center) / scale, (end - center) / scale
                first = float(np.arctan2(start[1], start[0]))
                last = float(np.arctan2(end[1], end[0]))
                while last <= first:
                    last += 2.0 * np.pi
                return first, last
            case _:
                upper, degree = self.integer(entry, 0), self.integer(entry, 1)
                index = 6 + (upper + degree + 2) + 4 * (upper + 1)
                return self.real(entry, index), self.real(entry, index + 1)

    def compact_placed_topology(self) -> None:
        """Remove private placement templates, retaining source-labelled incidences."""
        stage = self.stage
        edges = sorted(
            {
                coedge.edge
                for face in stage.faces
                for loop in face.loops
                for coedge in loop
            }
        )
        vertices = sorted(
            {vertex for edge in edges for vertex in stage.edge_vertices[edge]}
        )
        curves = sorted(
            {stage.edge_curves[edge] for edge in edges if stage.edge_curves[edge] >= 0}
        )
        edge_map = {value: index for index, value in enumerate(edges)}
        vertex_map = {value: index for index, value in enumerate(vertices)}
        curve_map = {value: index for index, value in enumerate(curves)}
        for face in stage.faces:
            for loop in face.loops:
                for coedge in loop:
                    coedge.edge = edge_map[coedge.edge]
        stage.edge_vertices = [
            (
                vertex_map[stage.edge_vertices[edge][0]],
                vertex_map[stage.edge_vertices[edge][1]],
            )
            for edge in edges
        ]
        stage.edge_ranges = [stage.edge_ranges[edge] for edge in edges]
        stage.edge_curves = [
            -1 if stage.edge_curves[edge] < 0 else curve_map[stage.edge_curves[edge]]
            for edge in edges
        ]
        stage.curves = [stage.curves[curve] for curve in curves]
        stage.vertices = [stage.vertices[vertex] for vertex in vertices]
        provenance = []
        for native, source in stage.provenance:
            kind, _, number = native.partition(":")
            mapping = (
                vertex_map if kind == "vertex" else edge_map if kind == "edge" else None
            )
            if mapping is None:
                provenance.append((native, source))
            elif int(number) in mapping:
                provenance.append((f"{kind}:{mapping[int(number)]}", source))
        stage.provenance = provenance
        stage.degenerate_edges = sum(value < 0 for value in stage.edge_curves)


# ------------------------------------------------------------------ reader


def _import(resource: BoundedResource, policy: CadImportPolicy, /) -> CadImportResult:
    file = _decode(resource.data, policy)
    factor = length_factor(file.unit_meters, policy.coordinate_contract)
    resolver = _Resolver(file, policy, factor)
    vertex_lists = [e for e in file.entries.values() if e.entity_type == 502]
    points = [
        resolver.reals(e, 1 + 3 * k, 3) * factor
        for e in vertex_lists
        for k in range(resolver.count(e, 0))
    ]
    resolver.stage.scale = max(1.0, float(np.max(np.abs(points)))) if points else 1.0
    by_type: dict[int, list[_Entry]] = {}
    for entry in file.entries.values():
        by_type.setdefault(entry.entity_type, []).append(entry)
    referenced_subfigures = {resolver.integer(entry, 0) for entry in by_type.get(408, [])}
    referenced_subfigures.update(
        resolver.integer(entry, 3 + k)
        for entry in by_type.get(308, [])
        for k in range(resolver.count(entry, 2))
    )
    for entry in (*by_type.get(408, []), *by_type.get(308, [])):
        if entry.number not in referenced_subfigures:
            resolver.occurrence(entry.number, RigidPlacement.identity(), ())
    for entry in by_type.get(308, []):
        if entry.number not in resolver.used:
            resolver.occurrence(entry.number, RigidPlacement.identity(), ())
    for entry in by_type.get(186, []):
        if entry.number not in resolver.used:
            resolver.occurrence(entry.number, RigidPlacement.identity(), ())
    for entry in by_type.get(514, []):
        if entry.number not in resolver.used:
            resolver.shell(entry.number, 1, entry.form == 1, ())
            if entry.form == 1:
                resolver.stage.shells[-1] = (resolver.stage.shells[-1][0], False)
    for entry in by_type.get(510, []):
        if entry.number not in resolver.used:
            resolver.face(entry.number, 1, ())
    for entry in by_type.get(144, []):
        if entry.number not in resolver.used:
            resolver.used.add(entry.number)
            resolver.trimmed_face(entry)
    if not resolver.stage.faces and any(
        entry.entity_type not in _DROPPABLE and entry.entity_type not in (124, 308, 408)
        for entry in file.entries.values()
    ):
        raise refuse(
            "unsupported-entity",
            "file",
            "A nonempty geometric file contains no admitted B-Rep faces.",
        )
    dropped: dict[int, int] = {}
    for entry in file.entries.values():
        if entry.number not in resolver.used:
            if entry.entity_type not in _DROPPABLE and entry.entity_type not in (
                100,
                102,
                104,
                110,
                116,
                123,
                124,
                126,
            ):
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "An unconsumed geometric entity cannot be discarded as annotation.",
                    (entry.name,),
                )
            dropped[entry.entity_type] = dropped.get(entry.entity_type, 0) + 1
    digest = resource.manifest.content_sha256
    identity = resource.manifest.source_path or f"iges-{digest[:16]}"
    resolver.compact_placed_topology()
    model = resolver.stage.publish(
        coordinate_contract=policy.coordinate_contract,
        source_id=identity,
        source_format=_FORMAT,
        source_digest=digest,
        import_policy_id=policy.policy_id,
        tessellation=policy.tessellation,
        default_occurrences=False,
    )
    losses = [
        AdapterLoss(
            f"E{entity_type}",
            "import",
            "dropped",
            f"{count} unreferenced type-{entity_type} entities carry no B-Rep "
            + ("semantics." if entity_type in _DROPPABLE else "boundary (wireframe)."),
            changes_interpretation=False,
        )
        for entity_type, count in sorted(dropped.items())
    ]
    stage = resolver.stage
    if stage.pcurves_fitted:
        losses.append(
            AdapterLoss(
                "p-curves",
                "import",
                "synthesized",
                f"{stage.pcurves_fitted} p-curves fitted within the policy "
                f"(max relative error {stage.maximum_fit_error:.3e}).",
                changes_interpretation=False,
            )
        )
    resource = account_bounded_resource(
        resource,
        depth=resolver.maximum_depth,
        nodes=len(file.entries),
        attributes=file.attributes,
        losses=len(losses),
    )
    unit_name = _UNITS[file.unit_flag][0]
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        _FORMAT,
        "phydrax-brep",
        source_id=identity,
        target_id=model.model_id,
        source_profile=_profile(),
        coordinate_mapping=(
            f"IGES unit {unit_name} -> {policy.coordinate_contract.length_unit.symbol} "
            f"(factor {factor!r})",
        ),
        preserved_fields=(
            "exact-carriers",
            "oriented-topology",
            "voids",
            "p-curves",
            "assembly-occurrences",
        ),
        assumptions=(
            "supplied p-curves retained; missing p-curves require algebraic recovery or an explicit sampled-fit policy",
        ),
        losses=losses,
    )
    return CadImportResult(
        model=model,
        format=_FORMAT,
        schema=_SCHEMA,
        source_digest=digest,
        source_length_unit_meters=float(file.unit_meters),
        source_angle_unit_radians=1.0,
        resource_manifest=resource.manifest,
        report=report,
        coverage=stage.coverage(dict(resolver.counts)),
        provenance=tuple(stage.provenance),
    )


def decode_iges_resource(
    resource: BoundedResource, policy: CadImportPolicy, /
) -> CadImportResult:
    """Decode one bounded IGES 5.3 resource into one native exact `BRepModel`.

    Lengths are converted exactly from the global-section unit to
    ``policy.coordinate_contract``. Manifold solids (186) become solids;
    unreferenced shells, faces and trimmed surfaces become free shells and
    faces. Unreferenced non-geometric and wireframe entities are dropped with
    a declared loss.
    """
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    if resource.manifest.limits != policy.limits:
        raise ValueError("The bounded resource and import policy limits must match.")
    try:
        return _import(resource, policy)
    except CadInterchangeError as error:
        error.resource_manifest = resource.manifest
        raise
    except ResourceReadError as error:
        resource_refusal(error)
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as error:
        failure = refuse("malformed", "file", str(error), ("file",))
        failure.resource_manifest = resource.manifest
        raise failure from error


def decode_iges_bytes(
    data: bytes, policy: CadImportPolicy, /, *, source_path: str | None = None
) -> CadImportResult:
    """Bound and decode exact in-memory IGES bytes."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = bounded_resource_from_bytes(
            data, limits=policy.limits, source_path=source_path
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_iges_resource(resource, policy)


def read_iges(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
) -> CadImportResult:
    """Descriptor-read and decode one bounded local IGES file."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=policy.limits
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_iges_resource(resource, policy)


# ------------------------------------------------------------------ writer


def _real_text(value: float, /) -> str:
    number = float(value)
    if not isfinite(number):
        raise refuse("inexact-export", "number", "Non-finite values cannot be exported.")
    text = repr(number).upper()
    mantissa, _, exponent = text.partition("E")
    if "." not in mantissa:
        mantissa += "."
    return mantissa + ("E" + exponent if exponent else "")


def _hollerith(text: str, /) -> str:
    if any(ord(character) < 32 or ord(character) > 126 for character in text):
        raise refuse(
            "inexact-export",
            "text",
            "IGES text fields require printable ASCII; no transliteration is performed.",
        )
    return f"{len(text)}H{text}"


def _path_name(path: tuple[str, ...]) -> str:
    """Store a scientific tuple path in the external standard NAME text field."""
    return _hollerith(
        "PhydraxPath:" + json.dumps(path, ensure_ascii=True, separators=(",", ":"))
    )


def _decode_path_name(
    text: str, label: str, maximum_components: int
) -> tuple[str, ...] | None:
    prefix = "PhydraxPath:"
    if not text.startswith(prefix):
        return None
    payload = text[len(prefix) :]
    # The namespace admits a flat string array only; reject recursive data before JSON decoding.
    quoted = escaped = False
    depth = 0
    for character in payload:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "[":
            depth += 1
            if depth > 1:
                raise refuse(
                    "malformed",
                    label,
                    "Scientific path metadata must be a flat string array.",
                )
        elif character == "]":
            depth -= 1
        elif character in "{}":
            raise refuse(
                "malformed", label, "Scientific path metadata cannot contain records."
            )
    try:
        path = json.loads(payload)
    except (ValueError, RecursionError) as error:
        raise refuse("malformed", label, "Malformed scientific path metadata.") from error
    if (
        not isinstance(path, list)
        or not path
        or any(not isinstance(item, str) or not item for item in path)
    ):
        raise refuse(
            "malformed", label, "Scientific paths require nonempty string components."
        )
    if len(path) > maximum_components:
        raise refuse(
            "limit", label, "Scientific path components exceed the attribute budget."
        )
    return tuple(path)


@dataclass(slots=True)
class _Out:
    entity_type: int
    parameters: list[str]
    form: int = 0
    transform: int = 0
    root: bool = False


@dataclass(slots=True)
class _IgesWriter:
    model: BRepModel
    policy: CadExportPolicy
    entities: list[_Out] = field(default_factory=list)
    byte_count: int = 1024
    approximations: tuple[CadCurveApproximation, ...] = ()

    def add(
        self, entity_type: int, parameters: list[str], form: int = 0, transform: int = 0
    ) -> int:
        if len(self.entities) >= min(self.policy.maximum_entities, 4_999_999):
            raise refuse("limit", "export", "The export exceeds the entity limit.")
        length = len(str(entity_type)) + 1 + sum(len(value) + 1 for value in parameters)
        cost = 162 + 81 * ((length + 63) // 64)
        if self.byte_count + cost > self.policy.maximum_bytes:
            raise refuse("limit", "export", "The export exceeds the byte limit.")
        self.byte_count += cost
        self.entities.append(_Out(entity_type, parameters, form, transform))
        return 2 * len(self.entities) - 1

    def reals(self, values: object) -> list[str]:
        return [_real_text(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]

    def transform(
        self,
        origin: ConvertibleToArray,
        first: ConvertibleToArray,
        second: ConvertibleToArray,
        label: str,
    ) -> int:
        first_, second_, normal = orthonormal_frame(first, second, label)
        rotation = np.stack((first_, second_, normal), axis=1)
        matrix = np.concatenate((rotation, np.asarray(origin)[:, None]), axis=1)
        return self.add(124, self.reals(matrix))

    def point(self, value: object) -> int:
        return self.add(116, self.reals(value))

    def direction(self, value: object) -> int:
        return self.add(123, self.reals(value))

    def curve(self, curve: AbstractCurve, first: float, last: float, label: str) -> int:
        evaluate = curve.evaluate
        match curve:
            case LineCurve():
                ends = np.asarray(evaluate(jnp.asarray((first, last), dtype=jnp.float64)))
                return self.add(110, self.reals(ends))
            case CircleCurve():
                transform = self.transform(
                    np.asarray(curve.center), curve.first_axis, curve.second_axis, label
                )
                radius = float(curve.radius)
                start = radius * np.asarray((cos(first), sin(first)))
                end = radius * np.asarray((cos(last), sin(last)))
                if last - first == 2.0 * np.pi:
                    end = start
                return self.add(
                    100, self.reals((0.0, 0.0, 0.0, *start, *end)), transform=transform
                )
            case EllipseCurve():
                transform = self.transform(
                    np.asarray(curve.center), curve.first_axis, curve.second_axis, label
                )
                a, b = float(curve.first_radius), float(curve.second_radius)
                start = np.asarray((a * cos(first), b * sin(first)))
                end = np.asarray((a * cos(last), b * sin(last)))
                if last - first == 2.0 * np.pi:
                    end = start
                return self.add(
                    104,
                    self.reals(
                        (b * b, 0.0, a * a, 0.0, 0.0, -a * a * b * b, 0.0, *start, *end)
                    ),
                    form=1,
                    transform=transform,
                )
            case BSplineCurve():
                points = np.asarray(curve.control_points)
                weights = np.asarray(curve.weights)
                polynomial = int(bool(np.all(weights == 1.0)))
                header = [
                    str(points.shape[0] - 1),
                    str(curve.degree),
                    "0",
                    "0",
                    str(polynomial),
                    "0",
                ]
                return self.add(
                    126,
                    header
                    + self.reals(curve.knots)
                    + self.reals(weights)
                    + self.reals(points)
                    + self.reals((first, last, 0.0, 0.0, 0.0)),
                )
            case _:
                raise refuse(
                    "inexact-export",
                    label,
                    f"{type(curve).__name__} carriers have no exact IGES entity.",
                )

    def surface(self, patch: AbstractSurfacePatch, box: np.ndarray, label: str) -> int:
        match patch:
            case PlanePatch():
                first_raw = np.asarray(patch.first_axis)
                second_raw = np.asarray(patch.second_axis)
                first, _, normal = orthonormal_frame(
                    first_raw / np.linalg.norm(first_raw),
                    second_raw / np.linalg.norm(second_raw),
                    label,
                )
                return self.add(
                    190,
                    [
                        str(self.point(patch.origin)),
                        str(self.direction(normal)),
                        str(self.direction(first)),
                    ],
                    form=1,
                )
            case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
                first, _, _ = orthonormal_frame(
                    patch.first_axis, patch.second_axis, label
                )
                axis = np.asarray(patch.axis)
                reference = str(self.direction(first))
                match patch:
                    case CylinderPatch():
                        return self.add(
                            192,
                            [
                                str(self.point(patch.origin)),
                                str(self.direction(axis)),
                                _real_text(float(patch.radius)),
                                reference,
                            ],
                            form=1,
                        )
                    case ConePatch():
                        return self.add(
                            194,
                            [
                                str(self.point(patch.origin)),
                                str(self.direction(axis)),
                                _real_text(float(patch.reference_radius)),
                                _real_text(degrees(float(patch.semi_angle))),
                                reference,
                            ],
                            form=1,
                        )
                    case SpherePatch():
                        return self.add(
                            196,
                            [
                                str(self.point(patch.center)),
                                _real_text(float(patch.radius)),
                                str(self.direction(axis)),
                                reference,
                            ],
                            form=1,
                        )
                    case _:
                        return self.add(
                            198,
                            [
                                str(self.point(patch.center)),
                                str(self.direction(axis)),
                                _real_text(float(patch.major_radius)),
                                _real_text(float(patch.minor_radius)),
                                reference,
                            ],
                            form=1,
                        )
            case BSplineSurfacePatch():
                points = np.asarray(patch.control_points)
                weights = np.asarray(patch.weights)
                polynomial = int(bool(np.all(weights == 1.0)))
                u_knots, v_knots = np.asarray(patch.u_knots), np.asarray(patch.v_knots)
                header = [
                    str(points.shape[0] - 1),
                    str(points.shape[1] - 1),
                    str(patch.u_degree),
                    str(patch.v_degree),
                    "0",
                    "0",
                    str(polynomial),
                    "0",
                    "0",
                ]
                domain = (
                    u_knots[patch.u_degree],
                    u_knots[-patch.u_degree - 1],
                    v_knots[patch.v_degree],
                    v_knots[-patch.v_degree - 1],
                )
                return self.add(
                    128,
                    header
                    + self.reals(u_knots)
                    + self.reals(v_knots)
                    + self.reals(weights.T)
                    + self.reals(points.transpose(1, 0, 2))
                    + self.reals(domain),
                )
            case RevolutionSurface():
                # IGES 120 is S(t, theta) = R(theta) C(t) with normal
                # S_t x S_theta. Revolving about the reversed axis with
                # theta = 2 pi - u, t = v keeps the native normal and loops.
                origin = np.asarray(patch.axis_origin)
                axis = self.add(
                    110,
                    self.reals((*origin, *(origin - np.asarray(patch.axis_direction)))),
                )
                generatrix = self.curve(
                    patch.curve, float(box[0, 1]), float(box[1, 1]), label
                )
                turn = 2.0 * np.pi
                return self.add(
                    120,
                    [
                        str(axis),
                        str(generatrix),
                        *self.reals((turn - box[1, 0], turn - box[0, 0])),
                    ],
                )
            case ExtrusionSurface():
                first, last = float(box[0, 0]), float(box[1, 0])
                directrix = self.curve(patch.curve, first, last, label)
                if isinstance(patch.curve, LineCurve):
                    start_parameter = first
                elif isinstance(patch.curve, BSplineCurve):
                    domain = patch.curve.parameter_domain
                    if domain is None:
                        raise refuse(
                            "inexact-export",
                            label,
                            "A spline directrix requires a finite parameter domain.",
                        )
                    start_parameter = domain[0]
                else:
                    start_parameter = 0.0
                end = np.asarray(
                    patch.curve.evaluate(jnp.asarray(start_parameter, dtype=jnp.float64))
                ) + np.asarray(patch.direction)
                return self.add(122, [str(directrix), *self.reals(end)])
            case OffsetSurface():
                if float(patch.distance) == 0.0:
                    raise refuse(
                        "inexact-export",
                        label,
                        "IGES offset surfaces require a nonzero distance.",
                    )
                basis = self.surface(patch.base, box, label)
                domain = _natural_box(patch.base)
                midpoint = (
                    np.mean(domain, axis=0)
                    if np.all(np.isfinite(domain))
                    else np.zeros(2, dtype=np.float64)
                )
                frame = np.asarray(
                    jax.jacfwd(patch.base.evaluate)(
                        jnp.asarray(midpoint, dtype=jnp.float64)
                    )
                )
                normal = np.cross(frame[:, 0], frame[:, 1])
                magnitude = float(np.linalg.norm(normal))
                if not isfinite(magnitude) or magnitude == 0.0:
                    raise refuse(
                        "inexact-export",
                        label,
                        "The offset-indicator reference has no regular source normal.",
                    )
                return self.add(
                    140,
                    [
                        *self.reals(normal / magnitude),
                        _real_text(float(patch.distance)),
                        str(basis),
                    ],
                )
            case _:
                raise refuse(
                    "inexact-export",
                    label,
                    f"{type(patch).__name__} surfaces are not exported to IGES.",
                )

    def parameter_curve(
        self,
        curve: AbstractCurve,
        patch: AbstractSurfacePatch,
        box: np.ndarray,
        edge_curve: AbstractCurve | None,
        first: float,
        last: float,
        label: str,
    ) -> int:
        """Write a UV carrier in the emitted surface and edge parameterization."""
        while isinstance(patch, OffsetSurface):
            patch = patch.base
        matrix, shift = np.eye(2), np.zeros(2)
        if isinstance(patch, PlanePatch):
            first_raw = np.asarray(patch.first_axis)
            second_raw = np.asarray(patch.second_axis)
            axis, second, _ = orthonormal_frame(
                first_raw / np.linalg.norm(first_raw),
                second_raw / np.linalg.norm(second_raw),
                label,
            )
            matrix = np.stack((axis, second)) @ np.stack((first_raw, second_raw), axis=1)
        elif isinstance(patch, (CylinderPatch, ConePatch, SpherePatch, TorusPatch)):
            first_axis, second_axis, _ = orthonormal_frame(
                patch.first_axis, patch.second_axis, label
            )
            external_second = np.cross(np.asarray(patch.axis), first_axis)
            handedness = float(external_second @ second_axis)
            if abs(abs(handedness) - 1.0) > 1.0e-9:
                raise refuse(
                    "inexact-export",
                    label,
                    "The analytic surface frame is not aligned with its axis.",
                )
            matrix[0, 0] = (1.0 if handedness > 0.0 else -1.0) * degrees(1.0)
            if isinstance(patch, (SpherePatch, TorusPatch)):
                matrix[1, 1] = degrees(1.0)
            if isinstance(patch, TorusPatch):
                matrix = matrix[::-1]
        elif isinstance(patch, ExtrusionSurface) and isinstance(patch.curve, LineCurve):
            matrix[0, 0] = float(np.linalg.norm(np.asarray(patch.curve.direction)))
            shift[0] = -float(box[0, 0]) * matrix[0, 0]
        elif isinstance(patch, RevolutionSurface):
            # (u, v) -> (t, theta) = (v, 2 pi - u); IGES parameterizes a line
            # generatrix on [0, 1].
            scale = (
                1.0 / float(box[1, 1] - box[0, 1])
                if isinstance(patch.curve, LineCurve)
                else 1.0
            )
            matrix = np.asarray(((0.0, scale), (-1.0, 0.0)))
            shift = np.asarray((-float(box[0, 1]) * scale, 2.0 * np.pi))
        parameter_first, parameter_last = first, last
        if isinstance(edge_curve, LineCurve):
            parameter_first = 0.0
            parameter_last = (last - first) * float(
                np.linalg.norm(np.asarray(edge_curve.direction))
            )
        match curve:
            case LineCurve():
                uv = (
                    np.asarray(
                        curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64))
                    )
                    @ matrix.T
                    + shift
                )
                points = np.pad(uv, ((0, 0), (0, 1)))
                output: AbstractCurve = BSplineCurve(
                    points,
                    np.ones(2),
                    [parameter_first, parameter_first, parameter_last, parameter_last],
                    1,
                )
            case BSplineCurve():
                points = np.asarray(curve.control_points) @ matrix.T + shift
                knots = parameter_first + (np.asarray(curve.knots) - first) * (
                    (parameter_last - parameter_first) / (last - first)
                )
                output = BSplineCurve(
                    np.pad(points, ((0, 0), (0, 1))), curve.weights, knots, curve.degree
                )
            case CircleCurve() | EllipseCurve():
                if (parameter_first, parameter_last) != (first, last):
                    raise refuse(
                        "inexact-export",
                        label,
                        "A conic UV curve does not admit affine angular reparameterization.",
                        (label,),
                    )
                center = matrix @ np.asarray(curve.center) + shift
                axis1, axis2 = (
                    matrix @ np.asarray(curve.first_axis),
                    matrix @ np.asarray(curve.second_axis),
                )
                norm1, norm2 = float(np.linalg.norm(axis1)), float(np.linalg.norm(axis2))
                if abs(float(axis1 @ axis2)) > 1.0e-12 * norm1 * norm2:
                    raise refuse(
                        "inexact-export",
                        label,
                        "A skew conic UV image has no parameter-preserving native carrier.",
                        (label,),
                    )
                radius1 = (
                    float(curve.radius)
                    if isinstance(curve, CircleCurve)
                    else float(curve.first_radius)
                )
                radius2 = (
                    float(curve.radius)
                    if isinstance(curve, CircleCurve)
                    else float(curve.second_radius)
                )
                output = EllipseCurve(
                    np.pad(center, (0, 1)),
                    np.pad(axis1 / norm1, (0, 1)),
                    np.pad(axis2 / norm2, (0, 1)),
                    norm1 * radius1,
                    norm2 * radius2,
                )
            case _:
                raise refuse(
                    "inexact-export",
                    label,
                    f"No exact IGES UV carrier for {type(curve).__name__}.",
                    (label,),
                )
        return self.curve(output, parameter_first, parameter_last, label)

    def encode(self) -> bytes:
        model = self.model
        geometry = model.geometry
        if geometry is None:
            raise refuse(
                "inexact-export",
                model.source_id,
                "Only models with exact native geometry can be exported.",
            )
        geometry, _, self.approximations = prepare_cad_export_geometry(
            model, geometry, self.policy, format_name=_FORMAT
        )
        if (
            not model.patches
            and not geometry.edge_curves
            and not len(geometry.vertex_points)
        ):
            self.add(308, ["0", _hollerith(model.source_id or "empty"), "0"])
            self.entities[-1].root = True
            return self._text(0.0)
        points = np.asarray(geometry.vertex_points)
        vertex_list = self.add(502, [str(points.shape[0]), *self.reals(points)], form=1)
        ranges = np.asarray(geometry.edge_ranges)
        edge_rows: list[str] = []
        edge_index: dict[int, int] = {}
        for edge, (curve_index, (start, end)) in enumerate(
            zip(geometry.edge_curves, geometry.edge_vertices, strict=True)
        ):
            if curve_index == -1:
                continue
            carrier = geometry.curves[curve_index]
            if not isinstance(carrier, AbstractCurve):
                raise refuse(
                    "inexact-export",
                    f"edge:{edge}",
                    "The prepared export edge has no explicit IGES carrier.",
                    (f"edge:{edge}",),
                )
            curve = self.curve(
                explicit_carrier(carrier),
                float(ranges[edge, 0]),
                float(ranges[edge, 1]),
                f"edge:{edge}",
            )
            edge_rows.extend(
                (
                    str(curve),
                    str(vertex_list),
                    str(start + 1),
                    str(vertex_list),
                    str(end + 1),
                )
            )
            edge_index[edge] = len(edge_index) + 1
        edge_list = self.add(504, [str(len(edge_index)), *edge_rows], form=1)
        orientation = np.asarray(model.orientation)
        faces = []
        frame_signs: list[int] = []
        bounds = np.asarray(model.parameter_bounds)
        for face, loops in enumerate(geometry.face_loops):
            surface = self.surface(model.patches[face], bounds[face], f"face:{face}")
            patch = model.patches[face]
            while isinstance(patch, OffsetSurface):
                patch = patch.base
            frame_sign = 1
            if isinstance(patch, (CylinderPatch, ConePatch, SpherePatch, TorusPatch)):
                handedness = float(
                    np.cross(np.asarray(patch.axis), np.asarray(patch.first_axis))
                    @ np.asarray(patch.second_axis)
                )
                frame_sign = 1 if handedness > 0.0 else -1
            frame_signs.append(frame_sign)
            loop_pointers = []
            for loop in loops:
                rows = []
                for coedge in loop if frame_sign > 0 else reversed(loop):
                    edge = geometry.coedge_edges[coedge]
                    agrees = (
                        "1" if frame_sign * geometry.coedge_senses[coedge] > 0 else "0"
                    )
                    first, last = (float(value) for value in ranges[edge])
                    pcurve = geometry.pcurves[coedge]
                    if not isinstance(pcurve, AbstractCurve):
                        raise refuse(
                            "inexact-export",
                            f"coedge:{coedge}",
                            "The prepared export p-curve has no explicit IGES carrier.",
                            (f"face:{face}", f"coedge:{coedge}"),
                        )
                    if edge in edge_index:
                        edge_curve = geometry.curves[geometry.edge_curves[edge]]
                        if not isinstance(edge_curve, AbstractCurve):
                            raise refuse(
                                "inexact-export",
                                f"edge:{edge}",
                                "The prepared export edge has no explicit IGES carrier.",
                                (f"face:{face}", f"edge:{edge}"),
                            )
                    else:
                        edge_curve = None
                        if agrees == "0":
                            pcurve = explicit_carrier(pcurve)
                            match pcurve:
                                case LineCurve():
                                    pcurve = LineCurve(
                                        np.asarray(pcurve.origin)
                                        + (first + last) * np.asarray(pcurve.direction),
                                        -np.asarray(pcurve.direction),
                                    )
                                case BSplineCurve():
                                    pcurve = BSplineCurve(
                                        np.asarray(pcurve.control_points)[::-1],
                                        np.asarray(pcurve.weights)[::-1],
                                        first + last - np.asarray(pcurve.knots)[::-1],
                                        pcurve.degree,
                                    )
                                case _:
                                    raise refuse(
                                        "inexact-export",
                                        f"coedge:{coedge}",
                                        "A reversed vertex-use p-curve has no exact IGES carrier.",
                                        (f"face:{face}", f"coedge:{coedge}"),
                                    )
                        agrees = "1"
                    pointer = self.parameter_curve(
                        explicit_carrier(pcurve),
                        model.patches[face],
                        bounds[face],
                        None if edge_curve is None else explicit_carrier(edge_curve),
                        first,
                        last,
                        f"coedge:{coedge}",
                    )
                    if edge in edge_index:
                        rows.extend(
                            (
                                "0",
                                str(edge_list),
                                str(edge_index[edge]),
                                agrees,
                                "1",
                                "0",
                                str(pointer),
                            )
                        )
                    else:
                        vertex = geometry.edge_vertices[edge][0] + 1
                        rows.extend(
                            (
                                "1",
                                str(vertex_list),
                                str(vertex),
                                agrees,
                                "1",
                                "0",
                                str(pointer),
                            )
                        )
                loop_pointers.append(str(self.add(508, [str(len(loop)), *rows], form=1)))
            faces.append(
                self.add(
                    510, [str(surface), str(len(loops)), "1", *loop_pointers], form=1
                )
            )
        shells = []
        for members, signs in zip(
            geometry.shell_faces, geometry.shell_orientations, strict=True
        ):
            rows = []
            for face, sign in zip(members, signs, strict=True):
                rows.extend(
                    (
                        str(faces[face]),
                        "1" if sign * orientation[face] * frame_signs[face] > 0 else "0",
                    )
                )
            closed = any(
                members[0] in geometry.shell_faces[s]
                for group in geometry.solid_shells
                for s in group
            )
            shells.append(
                self.add(514, [str(len(members)), *rows], form=1 if closed else 2)
            )
        in_solid = {shell for group in geometry.solid_shells for shell in group}
        solids = []
        for group in geometry.solid_shells:
            voids = []
            for shell in group[1:]:
                voids.extend((str(shells[shell]), "1"))
            solids.append(
                self.add(186, [str(shells[group[0]]), "1", str(len(group) - 1), *voids])
            )
        self.assembly(solids, geometry)
        for index, shell in enumerate(shells):
            if index not in in_solid:
                self.entities[(shell - 1) // 2].root = True
        in_shell = {face for members in geometry.shell_faces for face in members}
        free = [face for face in range(len(faces)) if face not in in_shell]
        if free:
            rows = []
            for face in free:
                rows.extend(
                    (
                        str(faces[face]),
                        "1" if orientation[face] * frame_signs[face] > 0 else "0",
                    )
                )
            self.add(514, [str(len(free)), *rows], form=2)
            self.entities[-1].root = True
        return self._text(float(np.max(np.abs(points))))

    def assembly(self, solids: list[int], geometry: BRepGeometry) -> None:
        occurrences = {occurrence.path: occurrence for occurrence in geometry.occurrences}
        containers = {
            container.path: container for container in geometry.assembly_containers
        }
        declared_members = {
            path for container in containers.values() for path in container.member_paths
        }
        child_paths = {
            path for container in containers.values() for path in container.child_paths
        }
        prepared: dict[tuple[str, ...], tuple[int, int]] = {}

        def member(path: tuple[str, ...]) -> int:
            occurrence = occurrences[path]
            definition = self.add(
                308, ["0", _path_name(path), "1", str(solids[occurrence.solid])]
            )
            matrix = np.concatenate(
                (
                    np.asarray(occurrence.rotation),
                    np.asarray(occurrence.translation)[:, None],
                ),
                axis=1,
            )
            transform = self.add(124, self.reals(matrix))
            return self.add(
                408, [str(definition), "0.", "0.", "0.", "1."], transform=transform
            )

        def container(path: tuple[str, ...]) -> tuple[int, int]:
            if path in prepared:
                return prepared[path]
            record = containers[path]
            children = [container(child) for child in record.child_paths]
            pointers = [member(path_) for path_ in record.member_paths]
            pointers.extend(
                self.add(408, [str(pointer), "0.", "0.", "0.", "1."])
                for pointer, _ in children
            )
            depth = 1 + max((level for _, level in children), default=0)
            definition = self.add(
                308,
                [
                    str(depth),
                    _path_name(path),
                    str(len(pointers)),
                    *(str(pointer) for pointer in pointers),
                ],
            )
            prepared[path] = definition, depth
            return definition, depth

        for path in containers:
            if path not in child_paths:
                definition, _ = container(path)
                self.add(408, [str(definition), "0.", "0.", "0.", "1."])
                self.entities[-1].root = True
        for path in occurrences:
            if path not in declared_members:
                pointer = member(path)
                self.entities[(pointer - 1) // 2].root = True

    def _text(self, extent: float) -> bytes:
        unit = self.model.coordinate_contract.length_unit
        meters = Fraction(unit.scale_to_reference)
        flags = [flag for flag, (_, value) in _UNITS.items() if value == meters]
        if not flags:
            raise refuse(
                "units",
                unit.symbol,
                "The model length unit has no IGES unit flag.",
            )
        flag = flags[0]
        timestamp = "19700101.000000"
        global_values = [
            _hollerith(","),
            _hollerith(";"),
            _hollerith("Phydrax"),
            _hollerith(self.model.source_id or "model"),
            _hollerith("Phydrax native CAD"),
            _hollerith("Phydrax IGES 5.3 writer"),
            "32",
            "308",
            "15",
            "308",
            "15",
            _hollerith("Phydrax"),
            "1.0",
            str(flag),
            _hollerith(_UNITS[flag][0]),
            "1",
            "1.0",
            _hollerith(timestamp),
            "1.0E-07",
            _real_text(max(extent, 1.0)),
            _hollerith(""),
            _hollerith(""),
            "11",
            "0",
            _hollerith(timestamp),
        ]
        global_length = sum(len(value) + 1 for value in global_values)
        global_bytes = 81 * ((global_length + 71) // 72) + 162
        if self.byte_count - 1024 + global_bytes > self.policy.maximum_bytes:
            raise refuse(
                "limit",
                "export",
                "Global metadata exceeds the remaining output byte budget.",
            )
        label = (
            "Phydrax native CAD export with declared approximations"
            if self.approximations
            else "Phydrax native exact B-Rep export"
        )
        start = [label.ljust(72) + "S0000001"]
        glob = _wrap(global_values, ",", ";", 72)
        global_lines = [text.ljust(72) + f"G{i + 1:7d}" for i, text in enumerate(glob)]
        directory: list[str] = []
        parameter_lines: list[str] = []
        for index, entity in enumerate(self.entities):
            number = 2 * index + 1
            chunks = _wrap([str(entity.entity_type), *entity.parameters], ",", ";", 64)
            pointer = len(parameter_lines) + 1
            for chunk in chunks:
                parameter_lines.append(
                    f"{chunk.ljust(64)} {number:7d}P{len(parameter_lines) + 1:7d}"
                )
            status = "00000000" if entity.root else "00010000"
            first = [entity.entity_type, pointer, 0, 0, 0, 0, entity.transform, 0]
            second = [entity.entity_type, 0, 0, len(chunks), entity.form]
            directory.append(
                "".join(f"{value:8d}" for value in first) + status + f"D{number:7d}"
            )
            directory.append(
                "".join(f"{value:8d}" for value in second)
                + " " * 16
                + " " * 8
                + f"{0:8d}"
                + f"D{number + 1:7d}"
            )
        terminate = (
            f"S{len(start):7d}G{len(global_lines):7d}D{len(directory):7d}P{len(parameter_lines):7d}"
        ).ljust(72) + "T0000001"
        text = (
            "\n".join((*start, *global_lines, *directory, *parameter_lines, terminate))
            + "\n"
        )
        return text.encode("ascii")


def _wrap(tokens: list[str], delimiter: str, terminator: str, width: int) -> list[str]:
    """Pack the continuous IGES stream, including Hollerith strings spanning records."""
    text = delimiter.join(tokens) + terminator
    return [text[start : start + width] for start in range(0, len(text), width)]


def write_iges(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    policy: CadExportPolicy | None = None,
    mode: PublicationMode = "exclusive",
) -> CadExportResult:
    """Publish IGES carriers exactly unless explicit branch approximation is requested.

    Solids become 186 manifold solids (inner shells as voids); free shells and
    faces become open 514 shells. Coordinates are written in the model length
    unit (declared in the global section). The global timestamps are fixed so
    identical models produce identical bytes.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    policy_ = CadExportPolicy() if policy is None else policy
    if not isinstance(policy_, CadExportPolicy):
        raise TypeError("policy must be a CadExportPolicy or None.")
    writer = _IgesWriter(model, policy_)
    data = writer.encode()
    receipt = publish_bytes(path, data, maximum_bytes=policy_.maximum_bytes, mode=mode)
    counts: dict[str, int] = {}
    for entity in writer.entities:
        counts[str(entity.entity_type)] = counts.get(str(entity.entity_type), 0) + 1
    losses = tuple(
        AdapterLoss(
            f"intersection-curve {item.branch_id}",
            "export",
            "transformed",
            f"New approximation {item.approximation_id}; continuous 3D "
            f"deviation bound {item.continuous_evidence.distance_bound:.17g}, "
            f"UV deviation bound {item.continuous_evidence.parameter_bound:.17g}, "
            f"surface-lift bounds ({item.continuous_evidence.first_correspondence_bound:.17g}, "
            f"{item.continuous_evidence.second_correspondence_bound:.17g}); "
            "continuous embedded source-fit homotopy and exported trim separation.",
            changes_interpretation=True,
        )
        for item in writer.approximations
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if writer.approximations else AdapterStatus.LOSSLESS,
        "phydrax-brep",
        _FORMAT,
        source_id=model.model_id,
        target_id=receipt.receipt_id,
        target_profile=AdapterFormatProfile(
            _FORMAT,
            qualifiers={
                "profile": _SCHEMA,
                **approximation_export_qualifiers(writer.approximations),
            },
        ),
        coordinate_mapping=(
            f"IGES unit of {model.coordinate_contract.length_unit.symbol}",
        ),
        preserved_fields=(
            "represented-carriers" if writer.approximations else "exact-carriers",
            "oriented-topology",
            "voids",
            "p-curves",
            "assembly-occurrences",
            "assembly-containers",
        ),
        assumptions=(
            canonical_fingerprint({"kind": "iges-export", "policy": policy_.policy_id}),
        ),
        losses=losses,
        waivers=_export_loss_waivers(losses, policy_, "IGES"),
    )
    return CadExportResult(
        receipt=receipt,
        format=_FORMAT,
        schema=_SCHEMA,
        report=report,
        entity_counts=tuple(sorted(counts.items())),
        approximations=writer.approximations,
    )


__all__ = ["decode_iges_bytes", "decode_iges_resource", "read_iges", "write_iges"]

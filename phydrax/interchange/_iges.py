#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Read bounded IGES 5.3 resources and write native B-Rep models.

Reading parses the fixed-format Start/Global/Directory/Parameter/Terminate
sections, the global parameter/record delimiters, Hollerith strings, the
directory/parameter linkage and the global units, and lowers the entity graph
to one native `BRepModel` through the shared `CadStage`.

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
refused, never discarded. Missing p-curves are recovered only from exact
algebraic source maps; unsupported maps are refused. Subfigure definitions and
instances (308/408) become a shared native assembly DAG with exact placements.
Unsupported required geometry is refused with its dependency chain.

Writing emits B-Rep topology, parameter-space carriers and assembly definitions
and uses without occurrence expansion. Carriers without an exact supported
IGES entity remain refused unless an explicit intersection-fit policy obtains
continuous coupled error, branch-topology and used-trim certificates. The Global
minimum resolution remains coordinate discernibility (IGES 5.3 §2.2.4.3.19),
never a generic edge/surface gap allowance.
"""

from __future__ import annotations

import hashlib
import mmap
import os
from dataclasses import dataclass, field
from fractions import Fraction
from math import cos, degrees, isfinite, radians, sin
from pathlib import Path
from typing import assert_never, final

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
from ..geometry.brep._constructors import _axis_frame
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
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    surface_differential,
    TorusPatch,
)
from ..typing import checked, ConvertibleToArray
from ._cad import (
    assembly_use_label,
    build_cad_export_report,
    build_cad_import_tolerance_evidence,
    cad_export_assembly_identifier_losses,
    cad_export_pole_identity_losses,
    cad_import_stage_losses,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadImportResult,
    CadInterchangeError,
    CadStage,
    explicit_carrier,
    length_factor,
    natural_box,
    prepare_cad_export_geometry,
    prepare_cad_export_surfaces,
    refuse,
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
from ._cad_io import CadResource, decode_cad_bytes, read_cad_file
from ._report import AdapterFormatProfile, AdapterLoss, AdapterReport, AdapterStatus


_FORMAT = "iges"
_SCHEMA = "iges-5.3"
_SECTIONS = "SGDPT"

# IGES global unit flags: (standard name, rational meters per unit).
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
    parameters: tuple[_Parameter, ...]
    entity_label: str = ""

    @property
    def name(self) -> str:
        return f"D{self.number} E{self.entity_type}"


def _split_parameters(
    text: str, parameter_delimiter: str, record_delimiter: str, label: str, /
) -> tuple[list[_Parameter], int]:
    """Parse free-format parameters, with Hollerith extents checked against input."""
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
            value_start = position + 1
            if count > length - value_start:
                raise refuse(
                    "malformed",
                    label,
                    "iges.parameters.string must fit its declared length; got truncated text",
                )
            value = _Hollerith(text[value_start : value_start + count])
            position = value_start + count
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
        if position >= length:
            raise refuse(
                "malformed",
                label,
                "iges.parameters.record must end with a record delimiter; got unterminated text",
            )
        delimiter = text[position]
        position += 1
        if delimiter == record_delimiter:
            return values, position
        if delimiter != parameter_delimiter:
            raise refuse(
                "malformed",
                label,
                f"iges.parameters.delimiter must match the declared delimiters; got {delimiter!r}",
            )


def _integer(value: _Parameter, label: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise refuse(
                "malformed",
                label,
                "iges.parameters.integer must be present; got a missing parameter",
            )
        return default
    if isinstance(value, _Hollerith):
        raise refuse(
            "malformed",
            label,
            "iges.parameters.integer must contain integer text; got a Hollerith string",
        )
    try:
        return int(value)
    except ValueError as error:
        raise refuse(
            "malformed",
            label,
            f"iges.parameters.integer must contain integer text; got {value!r}",
        ) from error


def _real(value: _Parameter, label: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise refuse(
                "malformed",
                label,
                "iges.parameters.real must be present; got a missing parameter",
            )
        return default
    if isinstance(value, _Hollerith):
        raise refuse(
            "malformed",
            label,
            "iges.parameters.real must contain real-number text; got a Hollerith string",
        )
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise refuse(
            "malformed",
            label,
            f"iges.parameters.real must contain real-number text; got {value!r}",
        ) from error
    if not isfinite(number):
        raise refuse(
            "malformed", label, f"iges.parameters.real must be finite; got {number!r}"
        )
    return number


def _string(value: _Parameter) -> str:
    return value.text if isinstance(value, _Hollerith) else (value or "")


@dataclass(slots=True)
class _File:
    unit_flag: int
    unit_meters: Fraction
    entries: dict[int, _Entry]
    minimum_resolution: _Parameter


def _lines(data: bytes | mmap.mmap, label: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {letter: [] for letter in _SECTIONS}
    order = ""
    offset = 0
    index = 0
    length = len(data)
    while offset < length:
        end = data.find(b"\n", offset)
        next_offset = length if end < 0 else end + 1
        raw_bytes = data[offset : length if end < 0 else end]
        offset = next_offset
        index += 1
        if raw_bytes.endswith(b"\r"):
            raw_bytes = raw_bytes[:-1]
        try:
            raw = raw_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise refuse(
                "malformed",
                f"line {index}",
                "iges.file.encoding must be ASCII; got non-ASCII bytes",
            ) from error
        if not raw.strip():
            continue
        if len(raw) != 80:
            raise refuse(
                "malformed",
                f"line {index}",
                f"iges.file.record must contain exactly 80 columns; got {len(raw)}",
            )
        line = raw
        letter = line[72]
        if letter in ("B", "C"):
            raise refuse(
                "unsupported-entity",
                f"line {index}",
                f"iges.file.section must use uncompressed ASCII records; got {letter!r}",
            )
        if letter not in sections:
            raise refuse(
                "malformed",
                f"line {index}",
                f"iges.file.section must name an S/G/D/P/T section; got {letter!r}",
            )
        if not order or order[-1] != letter:
            if letter in order or (
                order and _SECTIONS.index(letter) < _SECTIONS.index(order[-1])
            ):
                raise refuse(
                    "malformed",
                    f"line {index}",
                    f"iges.file.sections must follow S/G/D/P/T order; got {order + letter!r}",
                )
            order += letter
        sequence = line[73:80].strip()
        if not sequence.isdigit() or int(sequence) != len(sections[letter]) + 1:
            raise refuse(
                "malformed",
                f"line {index}",
                f"iges.file.sequence must be consecutive within section {letter}; got {sequence!r}",
            )
        sections[letter].append(line)
    if order[:1] != "S" or "G" not in order or "D" not in order or "P" not in order:
        raise refuse(
            "malformed", label, f"iges.file.sections must include S/G/D/P; got {order!r}"
        )
    if order[-1] != "T" or len(sections["T"]) != 1:
        raise refuse(
            "malformed",
            label,
            "iges.file.termination must contain one final T record; got missing or repeated termination",
        )
    terminate = sections["T"][0]
    counts = [terminate[8 * i + 1 : 8 * i + 8].strip() for i in range(4)]
    expected = [len(sections[letter]) for letter in "SGDP"]
    if [int(value) if value.isdigit() else -1 for value in counts] != expected:
        raise refuse(
            "malformed",
            "T",
            f"iges.file.termination_counts must match {expected!r}; got {counts!r}",
        )
    return sections


def _global(lines: list[str]) -> tuple[str, str, list[_Parameter]]:
    text = "".join(line[:72] for line in lines)
    parameter_delimiter, record_delimiter = ",", ";"
    position = 0
    if text.startswith("1H"):
        parameter_delimiter = text[2]
        position = 3
    if text[position : position + 1] != parameter_delimiter:
        raise refuse(
            "malformed",
            "G",
            "iges.global.parameter_delimiter must separate its declaration; got malformed delimiter text",
        )
    position += 1
    if text.startswith("1H", position):
        record_delimiter = text[position + 2]
        position += 3
    if text[position : position + 1] != parameter_delimiter:
        raise refuse(
            "malformed",
            "G",
            "iges.global.record_delimiter must have a separated declaration; got malformed delimiter text",
        )
    values, _ = _split_parameters(
        text[position + 1 :], parameter_delimiter, record_delimiter, "G"
    )
    return parameter_delimiter, record_delimiter, [None, None, *values]


def _units(values: list[_Parameter]) -> tuple[int, Fraction]:
    padded = values + [None] * max(0, 16 - len(values))
    scale = _real(padded[12], "G13", 1.0)
    if scale != 1.0:
        raise refuse(
            "units", "G13", f"iges.global.model_scale must equal 1; got {scale!r}"
        )
    flag = _integer(padded[13], "G14", 1)
    name = _string(padded[14]).strip().upper()
    if flag == 3:
        if name not in _UNIT_NAMES:
            raise refuse(
                "units",
                "G15",
                f"iges.global.unit_name must name a supported length unit; got {name!r}",
            )
        flag = _UNIT_NAMES[name]
    if flag not in _UNITS:
        raise refuse(
            "units",
            "G14",
            f"iges.global.unit_flag must name a supported length unit; got {flag}",
        )
    if name and _UNIT_NAMES.get(name, flag) != flag:
        raise refuse(
            "units",
            "G15",
            f"iges.global.unit_name must agree with unit flag {flag}; got {name!r}",
        )
    return flag, _UNITS[flag][1]


def _decode(data: bytes | mmap.mmap) -> _File:
    sections = _lines(data, "file")
    parameter_delimiter, record_delimiter, global_values = _global(sections["G"])
    unit_flag, unit_meters = _units(global_values)
    directory = sections["D"]
    if len(directory) % 2:
        raise refuse(
            "malformed",
            "D",
            f"iges.directory.records must occur in pairs; got {len(directory)} records",
        )
    parameters = sections["P"]
    entries: dict[int, _Entry] = {}
    for index in range(0, len(directory), 2):
        number = index + 1
        label = f"D{number}"
        first = [directory[index][8 * k : 8 * k + 8].strip() for k in range(9)]
        second = [directory[index + 1][8 * k : 8 * k + 8] for k in range(9)]
        entity_type = _integer(first[0] or None, label)
        if _integer(second[0].strip() or None, label) != entity_type:
            raise refuse(
                "malformed",
                label,
                "iges.directory.entity_type must agree across its two records; got conflicting type fields",
            )
        pointer = _integer(first[1] or None, label)
        count = _integer(second[3].strip() or None, label)
        if pointer < 1 or count < 1 or pointer + count - 1 > len(parameters):
            raise refuse(
                "dangling-reference",
                label,
                f"iges.directory.parameter_range must lie within the P section; got pointer {pointer}, count {count}",
            )
        records = parameters[pointer - 1 : pointer - 1 + count]
        if any(
            not record[64:72].strip().isdigit() or int(record[64:72]) != number
            for record in records
        ):
            raise refuse(
                "malformed",
                label,
                "iges.parameters.directory_pointer must reference its directory entry; got a conflicting pointer",
            )
        text = "".join(record[:64] for record in records)
        values, _ = _split_parameters(
            text,
            parameter_delimiter,
            record_delimiter,
            label,
        )
        if not values or _integer(values[0], label) != entity_type:
            raise refuse(
                "malformed",
                label,
                "iges.parameters.entity_type must match its directory entry; got a conflicting type",
            )
        entries[number] = _Entry(
            number,
            entity_type,
            _integer(first[6] or None, label, 0),
            _integer(second[4].strip() or None, label, 0),
            tuple(values[1:]),
            second[7].strip(),
        )
    return _File(
        unit_flag,
        unit_meters,
        entries,
        global_values[18] if len(global_values) > 18 else None,
    )


# ------------------------------------------------------------------ resolution


class _Resolver:
    """Lower the IGES entity graph to one native stage."""

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
        self.exact_placements: dict[int, BRepAffinePlacement] = {}
        self.exact_length_factor = file.unit_meters / Fraction(
            policy.coordinate_contract.length_unit.scale_to_reference
        )

    # ---------------------------------------------------------- graph access

    def entry(
        self, pointer: int, chain: tuple[str, ...], allowed: tuple[int, ...]
    ) -> _Entry:
        if pointer not in self.file.entries:
            raise refuse(
                "dangling-reference",
                f"D{pointer}",
                f"iges.reference.pointer must name a directory entry; got {pointer}",
                chain,
            )
        entry = self.file.entries[pointer]
        if entry.entity_type not in allowed:
            raise refuse(
                "unsupported-entity",
                entry.name,
                f"iges.reference.entity_type must be one of {allowed}; got type {entry.entity_type}, form {entry.form}",
                chain,
            )
        if pointer in self.active:
            raise refuse(
                "cyclic-reference",
                entry.name,
                "iges.reference.graph must be acyclic; got a repeated active reference",
                chain,
            )
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
            raise refuse(
                "malformed",
                entry.name,
                f"iges.parameters.count must be nonnegative; got {value}",
            )
        if value > len(entry.parameters):
            raise refuse(
                "malformed",
                entry.name,
                f"iges.parameters.count must fit its source payload; got {value}",
            )
        return value

    def real(self, entry: _Entry, index: int, default: float | None = None) -> float:
        return _real(self._parameter(entry, index), f"{entry.name} P{index + 1}", default)

    def reals(self, entry: _Entry, start: int, count: int) -> np.ndarray:
        return np.asarray([self.real(entry, start + k) for k in range(count)])

    def point(self, entry: _Entry, start: int) -> np.ndarray:
        return self.reals(entry, start, 3) * self.factor

    def placement(self, entry: _Entry, chain: tuple[str, ...]) -> RigidPlacement:
        """Resolve a chained IGES transformation without recursive traversal."""
        if entry.transform == 0:
            return RigidPlacement.identity()
        pending: list[tuple[int, RigidPlacement]] = []
        visited: set[int] = set()
        pointer = entry.transform
        result = RigidPlacement.identity()
        while pointer:
            key = (pointer, self.factor)
            cached = self.placements.get(key)
            if cached is not None:
                result = cached
                break
            if pointer in visited:
                raise refuse(
                    "cyclic-reference",
                    f"D{pointer}",
                    "iges.transformation.graph must be acyclic; got a repeated transformation",
                    chain,
                )
            visited.add(pointer)
            matrix = self.entry(pointer, chain, (124,))
            if matrix.form not in (0, 1):
                raise refuse(
                    "unsupported-entity",
                    matrix.name,
                    f"iges.transform.form must be 0 or 1; got {matrix.form}",
                    chain,
                )
            values = self.reals(matrix, 0, 12).reshape(3, 4)
            values[:, 3] *= self.factor
            pending.append(
                (
                    pointer,
                    RigidPlacement.from_matrix(values, matrix.name, reflection=True),
                )
            )
            pointer = matrix.transform
        for transform_id, local in reversed(pending):
            result = result.compose(local)
            self.placements[(transform_id, self.factor)] = result
        return result

    def _exact_real(
        self, entry: _Entry, index: int, default: Fraction | None = None
    ) -> Fraction:
        value = self._parameter(entry, index)
        if value is None:
            if default is not None:
                return default
            raise refuse(
                "malformed",
                entry.name,
                f"iges.parameters.real must be present; got missing parameter P{index + 1}",
            )
        if isinstance(value, _Hollerith):
            raise refuse(
                "malformed",
                entry.name,
                f"iges.parameters.real must contain numeric text; got Hollerith P{index + 1}",
            )
        self.real(entry, index)
        try:
            return Fraction(value.replace("D", "E").replace("d", "e"))
        except (ValueError, ZeroDivisionError) as error:
            raise refuse(
                "malformed",
                entry.name,
                f"iges.parameters.real must contain exact decimal text; got P{index + 1}={value!r}",
            ) from error

    def _exact_transform(
        self, pointer: int, chain: tuple[str, ...], /
    ) -> BRepAffinePlacement:
        identity = BRepAffinePlacement.identity()
        if pointer == 0:
            return identity
        cached = self.exact_placements.get(pointer)
        if cached is not None:
            return cached
        pending: list[tuple[int, BRepAffinePlacement]] = []
        visited: set[int] = set()
        current = pointer
        result = identity
        while current:
            cached = self.exact_placements.get(current)
            if cached is not None:
                result = cached
                break
            if current in visited:
                raise refuse(
                    "cyclic-reference",
                    f"D{current}",
                    "iges.transformation.graph must be acyclic; got a repeated transformation",
                    chain,
                )
            visited.add(current)
            matrix = self.entry(current, chain, (124,))
            if matrix.form not in (0, 1):
                raise refuse(
                    "unsupported-entity",
                    matrix.name,
                    f"iges.transform.form must be 0 or 1; got {matrix.form}",
                    chain,
                )
            values = tuple(self._exact_real(matrix, index) for index in range(12))
            translation = tuple(
                values[4 * row + 3] * self.exact_length_factor for row in range(3)
            )
            rows = tuple(
                tuple(values[4 * row + column] for column in range(3))
                + (translation[row],)
                for row in range(3)
            ) + ((Fraction(0), Fraction(0), Fraction(0), Fraction(1)),)
            numeric = np.asarray(
                [[float(rows[row][column]) for column in range(4)] for row in range(3)],
                dtype=np.float64,
            )
            RigidPlacement.from_matrix(numeric, matrix.name, reflection=True)
            pending.append((current, BRepAffinePlacement.from_matrix(rows)))
            current = matrix.transform
        for transform_id, local in reversed(pending):
            result = result.compose(local)
            self.exact_placements[transform_id] = result
        return result

    def _exact_entry_placement(
        self, entry: _Entry, chain: tuple[str, ...]
    ) -> BRepAffinePlacement:
        self.placement(entry, chain)
        return self._exact_transform(entry.transform, chain)

    def _exact_instance_placement(
        self, entry: _Entry, chain: tuple[str, ...]
    ) -> BRepAffinePlacement:
        scale = self._exact_real(entry, 4, Fraction(1))
        if scale != 1:
            raise refuse(
                "unsupported-entity",
                entry.name,
                "iges.subfigure.scale must equal one; got a scaled instance",
                chain,
            )
        translation = tuple(
            self._exact_real(entry, 1 + axis) * self.exact_length_factor
            for axis in range(3)
        )
        offset = BRepAffinePlacement.from_matrix(
            (
                (Fraction(1), Fraction(0), Fraction(0), translation[0]),
                (Fraction(0), Fraction(1), Fraction(0), translation[1]),
                (Fraction(0), Fraction(0), Fraction(1), translation[2]),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
            )
        )
        return self._exact_entry_placement(entry, chain).compose(offset)

    # -------------------------------------------------------------- geometry

    def curve(self, pointer: int, chain: tuple[str, ...]) -> AbstractCurve:
        """Resolve nested single-segment composites with an explicit stack."""
        active_base = len(self.active)
        frames: list[tuple[_Entry, RigidPlacement]] = []
        current_pointer = pointer
        current_chain = chain
        try:
            while True:
                entry = self.entry(
                    current_pointer, current_chain, (100, 102, 104, 110, 126)
                )
                current_chain = (*current_chain, entry.name)
                self.active.append(current_pointer)
                placement = self.placement(entry, current_chain)
                if entry.entity_type == 102:
                    count = self.integer(entry, 0)
                    if count != 1:
                        raise refuse(
                            "unsupported-entity",
                            entry.name,
                            f"iges.composite_curve.segments must contain one edge carrier; got {count} segments",
                            current_chain,
                        )
                    frames.append((entry, placement))
                    current_pointer = self.integer(entry, 1)
                    continue
                curve = self._curve_leaf(entry, current_chain)
                curve = place_curve(curve, placement, entry.name)
                self.active.pop()
                while frames:
                    composite, composite_placement = frames.pop()
                    self.active.pop()
                    curve = place_curve(curve, composite_placement, composite.name)
                return curve
        finally:
            while len(self.active) > active_base:
                self.active.pop()

    def _curve_leaf(self, entry: _Entry, chain: tuple[str, ...], /) -> AbstractCurve:
        try:
            match entry.entity_type:
                case 110:
                    start, end = self.point(entry, 0), self.point(entry, 3)
                    length = float(np.linalg.norm(end - start))
                    if not length > 0.0:
                        raise refuse(
                            "inconsistent-geometry",
                            entry.name,
                            "iges.line.length must be positive; got zero",
                            chain,
                        )
                    return LineCurve(start, (end - start) / length)
                case 100:
                    height = self.real(entry, 0) * self.factor
                    center = (
                        np.asarray((self.real(entry, 1), self.real(entry, 2)))
                        * self.factor
                    )
                    start = (
                        np.asarray((self.real(entry, 3), self.real(entry, 4)))
                        * self.factor
                    )
                    radius = float(np.linalg.norm(start - center))
                    return CircleCurve(
                        (center[0], center[1], height),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        radius,
                    )
                case 104:
                    return self._conic(entry, chain)
                case 126:
                    return self._spline_curve(entry, 3)
                case _:
                    raise RuntimeError(
                        "IGES curve leaf reached an unsupported entity type "
                        f"{entry.entity_type}."
                    )
        except CadInterchangeError as error:
            raise refuse(
                error.refusal.reason,
                error.refusal.entity,
                error.refusal.message,
                chain,
            ) from error
        except (ValueError, TypeError) as error:
            raise refuse("malformed", entry.name, str(error), chain) from error

    def _conic(self, entry: _Entry, chain: tuple[str, ...]) -> CircleCurve | EllipseCurve:
        a, b, c, d, e, f = (self.real(entry, k) for k in range(6))
        height = self.real(entry, 6) * self.factor
        if entry.form not in (0, 1):
            raise refuse(
                "unsupported-entity",
                entry.name,
                "iges.conic.form must describe an ellipse; got a different conic form",
                chain,
            )
        # This fixed two-dimensional eigensystem and solve decode one external
        # conic into immutable host geometry; they are not numerical runtime kernels.
        quadratic = np.asarray(((a, 0.5 * b), (0.5 * b, c)))
        linear = np.asarray((d, e))
        eigenvalues, axes = np.linalg.eigh(quadratic)
        if eigenvalues[0] * eigenvalues[1] <= 0.0:
            raise refuse(
                "unsupported-entity",
                entry.name,
                "iges.conic.quadratic must be definite; got a parabolic or hyperbolic conic",
                chain,
            )
        center = -0.5 * np.linalg.solve(quadratic, linear)
        right = -(float(center @ quadratic @ center + linear @ center) + f)
        squared = right / eigenvalues
        if np.any(squared <= 0.0):
            raise refuse(
                "inconsistent-geometry",
                entry.name,
                "iges.conic.squared_radii must be positive; got nonpositive radii",
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
        if upper < degree or degree < 1 or upper + degree + 2 > len(entry.parameters):
            raise refuse(
                "malformed",
                entry.name,
                "iges.spline_curve.sizes must fit its source payload and define a valid control and knot layout; got inconsistent sizes",
            )
        if any(self.integer(entry, index) not in (0, 1) for index in range(2, 6)):
            raise refuse(
                "malformed",
                entry.name,
                "iges.spline_curve.properties must be binary; got a nonbinary property",
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
                "iges.spline_curve.domain must lie within its knot support; got an out-of-support domain",
            )
        return BSplineCurve(points[:, :dimension], weights, knots, degree)

    def direction(self, pointer: int, chain: tuple[str, ...]) -> np.ndarray:
        entry = self.entry(pointer, chain, (123,))
        vector = self.reals(entry, 0, 3)
        norm = float(np.linalg.norm(vector))
        if not norm > 0.0:
            raise refuse(
                "inconsistent-geometry",
                entry.name,
                "iges.direction.norm must be positive; got zero",
                chain,
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
                        "iges.plane.normal_norm must be positive; got zero",
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
                        f"iges.revolution.axis must be a line; got {type(axis).__name__}",
                        chain,
                    )
                generatrix_pointer = self.integer(entry, 1)
                generatrix = self.curve(generatrix_pointer, chain)
                if isinstance(generatrix, LineCurve):
                    # IGES parameterizes a line on [0, 1], not by its length.
                    length = self._range(self.file.entries[generatrix_pointer], chain)[1]
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
                except CadInterchangeError as error:
                    raise refuse(
                        error.refusal.reason,
                        error.refusal.entity,
                        error.refusal.message,
                        chain,
                    ) from error
                except (ValueError, TypeError) as error:
                    raise refuse("malformed", entry.name, str(error), chain) from error
            case 140:
                if entry.form != 0:
                    raise refuse(
                        "unsupported-entity",
                        entry.name,
                        f"iges.offset_surface.form must equal 0; got {entry.form}",
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
                        "iges.offset_surface.distance must be nonzero; got zero",
                        chain,
                    )
                return OffsetSurface(
                    self.surface(self.integer(entry, 4), chain), distance
                )
            case _:
                raise RuntimeError(
                    f"iges.surface.entity_type must name a supported surface; got {entry.entity_type}"
                )

    @staticmethod
    def _curve_start(curve: AbstractCurve) -> np.ndarray:
        match curve:
            case LineCurve():
                return np.asarray(curve.origin)
            case BSplineCurve():
                domain = curve.parameter_domain
                first = 0.0 if domain is None else domain[0]
                return np.asarray(
                    curve.evaluate(jnp.asarray(first, dtype=jnp.float64)),
                    dtype=np.float64,
                )
            case _:
                return np.asarray(
                    curve.evaluate(jnp.asarray(0.0, dtype=jnp.float64)),
                    dtype=np.float64,
                )

    def _spline_surface(self, entry: _Entry) -> BSplineSurfacePatch:
        upper_u, upper_v = self.integer(entry, 0), self.integer(entry, 1)
        degree_u, degree_v = self.integer(entry, 2), self.integer(entry, 3)
        count_u, count_v = upper_u + 1, upper_v + 1
        if (
            upper_u < degree_u
            or upper_v < degree_v
            or min(degree_u, degree_v) < 1
            or 4 * count_u * count_v > len(entry.parameters)
        ):
            raise refuse(
                "malformed",
                entry.name,
                "iges.spline_surface.sizes must fit its source payload and define a valid control and knot layout; got inconsistent sizes",
            )
        if any(self.integer(entry, index) not in (0, 1) for index in range(4, 9)):
            raise refuse(
                "malformed",
                entry.name,
                "iges.spline_surface.properties must be binary; got a nonbinary property",
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
                "iges.spline_surface.domain must lie within its knot support; got an out-of-support domain",
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
                    "dangling-reference",
                    entry.name,
                    f"iges.vertex_list.index must name a listed vertex; got {index}",
                    chain,
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
                    "dangling-reference",
                    entry.name,
                    f"iges.edge_list.index must name a listed edge; got {index}",
                    chain,
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
                    parameter_range=self._range(self.file.entries[curve_pointer], chain),
                )
            except ValueError as error:
                raise refuse("inconsistent-geometry", label, str(error), chain) from error
        return self.edges[key]

    def parameter_curve(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> tuple[AbstractCurve, tuple[float, float]]:
        """Decode a UV carrier and its active file interval without length-scaling parameters."""
        entry = self.entry(pointer, chain, (100, 102, 104, 110, 126))
        if entry.entity_type == 102:
            self._admit_composite(pointer, chain)
            pieces = self.parameter_segments(pointer, patch, chain)
            if len(pieces) != 1:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "iges.pcurve.segments must contain one coedge carrier; got multiple segments",
                    (*chain, entry.name),
                )
            return pieces[0]
        factor = self.factor
        self.factor = 1.0
        try:
            curve = self.curve(pointer, chain)
            domain = self._range(self.file.entries[pointer], chain)
        finally:
            self.factor = factor
        match curve:
            case LineCurve():
                points = np.asarray(
                    curve.evaluate(jnp.asarray(domain, dtype=jnp.float64)),
                    dtype=np.float64,
                )[:, :2]
                result: BSplineCurve | CircleCurve | EllipseCurve = BSplineCurve(
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
                    f"iges.pcurve.carrier must have a supported UV representation; got {type(curve).__name__}",
                    chain,
                )
        while isinstance(patch, OffsetSurface):
            patch = patch.base
        angular = radians(1.0)
        match patch:
            case PlanePatch():
                factors = (factor, factor)
            case CylinderPatch() | ConePatch():
                factors = (angular, factor)
            case SpherePatch() | TorusPatch():
                factors = (angular, angular)
            case RevolutionSurface():
                factors = (-1.0, 1.0)
            case ExtrusionSurface():
                factors = (1.0, factor)
            case _:
                factors = (1.0, 1.0)
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
                    assert_never(result)
        try:
            return (
                transform_pcurve(
                    result,
                    factors,
                    1.0,
                    (2.0 * np.pi, 0.0) if revolution else (0.0, 0.0),
                ),
                domain,
            )
        except ValueError as error:
            raise refuse(
                "unsupported-entity", f"D{pointer}", str(error), chain
            ) from error

    def align_parameter_curve(
        self,
        curve: AbstractCurve,
        domain: tuple[float, float],
        edge: int,
        patch: AbstractSurfacePatch,
    ) -> AbstractCurve:
        """Map the active file interval, not the full spline knot support, to the edge."""
        first, last = self.stage.edge_ranges[edge]
        if isinstance(curve, BSplineCurve):
            lower, upper = domain
            # IGES 126's final V(0), V(1) parameters can select a strict
            # subinterval of the knot domain. Keep that trim when determining
            # direction and affinely mapping every knot onto the edge phase.
            # Closed edges have identical endpoint vertices; interior carrier
            # correspondence, not endpoint coincidence, owns their direction.
            samples = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)
            world = np.asarray(
                patch.evaluate(
                    curve.evaluate(
                        jnp.asarray(lower + samples * (upper - lower), dtype=jnp.float64)
                    )
                ),
                dtype=np.float64,
            )
            carrier_index = self.stage.edge_curves[edge]
            carrier = self.stage.curves[carrier_index]
            reference = np.asarray(
                carrier.evaluate(
                    jnp.asarray(first + samples * (last - first), dtype=jnp.float64)
                ),
                dtype=np.float64,
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
                    "malformed",
                    entry.name,
                    "iges.loop.edge_record must contain a valid edge use; got malformed parameters",
                    chain,
                )
            if curves > 1:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "iges.loop.pcurve must contain one segment; got a multi-segment carrier",
                    chain,
                )
            parameter = (
                self.parameter_curve(self.integer(entry, position + 6), patch, chain)
                if curves
                else None
            )
            pcurve, domain = parameter if parameter is not None else (None, None)
            position += 5 + 2 * curves
            if kind == 1:
                vertex = self.vertex(list_pointer, index, chain)
                if pcurve is None or domain is None:
                    raise refuse(
                        "unsupported-entity",
                        entry.name,
                        "iges.loop.vertex_use must have a bounded p-curve; got an unbounded carrier",
                        chain,
                    )
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
                        # its authored angular extent as the native phase.
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
                    assert domain is not None
                    pcurve = self.align_parameter_curve(pcurve, domain, edge, patch)
            coedges.append(
                StagedCoedge(
                    edge,
                    1 if agrees else -1,
                    pcurve,
                    f"{entry.name}[{len(coedges)}]",
                    origin="absent" if pcurve is None else "file",
                )
            )
        self.active.pop()
        if not coedges:
            raise refuse(
                "malformed",
                entry.name,
                "iges.loop.edges must be nonempty; got no edges",
                chain,
            )
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
            raise refuse(
                "unsupported-entity",
                entry.name,
                "iges.face.loops must be nonempty; got no loops",
                chain,
            )
        face = self.stage.face(
            patch,
            loops,
            orientation,
            entry.entity_label or surface_tag(patch),
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
            raise refuse(
                "malformed",
                entry.name,
                "iges.shell.faces must be nonempty; got no faces",
                chain,
            )
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
                "iges.subshape.placement must be proper rigid; got a reflection",
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

    def assembly_graph(self) -> BRepAssemblyGraph:
        """Build a validated DAG whose identities derive only from D references."""
        definition_entries = tuple(
            sorted(
                (
                    entry
                    for entry in self.file.entries.values()
                    if entry.entity_type == 308
                ),
                key=lambda entry: entry.number,
            )
        )
        instance_entries = tuple(
            sorted(
                (
                    entry
                    for entry in self.file.entries.values()
                    if entry.entity_type == 408
                ),
                key=lambda entry: entry.number,
            )
        )
        if not definition_entries and not instance_entries:
            graph = BRepAssemblyGraph.single_solids(len(self.stage.solids))
            self.stage.assembly_graph = graph
            return graph

        definition_ids = {
            entry.number: f"iges-definition-D{entry.number}"
            for entry in definition_entries
        }
        children: dict[int, tuple[_Entry, ...]] = {}
        links: dict[int, tuple[int, ...]] = {}
        referenced_entities: set[int] = set()
        referenced_definitions: set[int] = set()
        instance_placements: dict[int, BRepAffinePlacement] = {}
        definition_placements: dict[int, BRepAffinePlacement] = {}

        for instance in instance_entries:
            chain = (instance.name,)
            self.entry(instance.number, (), (408,))
            target = self.integer(instance, 0)
            self.entry(target, chain, (308,))
            referenced_definitions.add(target)
            instance_placements[instance.number] = self._exact_instance_placement(
                instance, chain
            )

        for definition in definition_entries:
            chain = (definition.name,)
            self.entry(definition.number, (), (308,))
            declared_depth = self.integer(definition, 0)
            if declared_depth < 0:
                raise refuse(
                    "malformed",
                    definition.name,
                    f"iges.subfigure.depth must be nonnegative; got {declared_depth}",
                    chain,
                )
            definition_placements[definition.number] = self._exact_entry_placement(
                definition, chain
            )
            count = self.count(definition, 2)
            child_entries = tuple(
                self.entry(
                    self.integer(definition, 3 + index),
                    chain,
                    (186, 308, 408),
                )
                for index in range(count)
            )
            children[definition.number] = child_entries
            definition_links: list[int] = []
            for child in child_entries:
                referenced_entities.add(child.number)
                if child.entity_type == 308:
                    definition_links.append(child.number)
                    referenced_definitions.add(child.number)
                elif child.entity_type == 408:
                    target = self.integer(child, 0)
                    self.entry(target, (*chain, child.name), (308,))
                    instance_placements[child.number] = self._exact_instance_placement(
                        child, (*chain, child.name)
                    )
                    definition_links.append(target)
                    referenced_definitions.add(target)
            links[definition.number] = tuple(definition_links)

        state: dict[int, int] = {}
        for root_number in definition_ids:
            if state.get(root_number, 0) == 2:
                continue
            stack: list[tuple[int, bool]] = [(root_number, False)]
            while stack:
                number, exiting = stack.pop()
                if exiting:
                    state[number] = 2
                    continue
                current_state = state.get(number, 0)
                if current_state == 2:
                    continue
                if current_state == 1:
                    raise refuse(
                        "cyclic-reference",
                        f"D{number}",
                        "iges.subfigure.graph must be acyclic; got a repeated definition",
                    )
                state[number] = 1
                stack.append((number, True))
                for child in reversed(links[number]):
                    child_state = state.get(child, 0)
                    if child_state == 1:
                        raise refuse(
                            "cyclic-reference",
                            f"D{child}",
                            "iges.subfigure.graph must be acyclic; got a cyclic reference",
                            (f"D{number}",),
                        )
                    if child_state == 0:
                        stack.append((child, False))

        definitions: list[BRepAssemblyDefinition] = []
        identity = BRepAffinePlacement.identity()
        solid_placements: dict[int, BRepAffinePlacement] = {}
        for definition in definition_entries:
            definition_id = definition_ids[definition.number]
            parent_placement = definition_placements[definition.number]
            uses: list[BRepAssemblyUse] = []
            for index, child in enumerate(children[definition.number]):
                chain = (definition.name, child.name)
                if child.entity_type == 186:
                    solid = self.solid_indices.get(child.number)
                    if solid is None:
                        raise refuse(
                            "inconsistent-geometry",
                            child.name,
                            "An IGES assembly use refers to a solid that was not staged.",
                            chain,
                        )
                    placement = solid_placements.get(child.number)
                    if placement is None:
                        placement = self._exact_entry_placement(child, chain)
                        solid_placements[child.number] = placement
                    use = BRepAssemblyUse(
                        f"iges-use-D{definition.number}-child-{index}-D{child.number}",
                        parent_placement.compose(placement),
                        solid_index=solid,
                    )
                elif child.entity_type == 308:
                    use = BRepAssemblyUse(
                        f"iges-use-D{definition.number}-child-{index}-D{child.number}",
                        parent_placement,
                        definition_id=definition_ids[child.number],
                    )
                elif child.entity_type == 408:
                    target = self.integer(child, 0)
                    use = BRepAssemblyUse(
                        f"iges-use-D{child.number}-from-D{definition.number}-child-{index}",
                        parent_placement.compose(instance_placements[child.number]),
                        definition_id=definition_ids[target],
                    )
                else:
                    raise RuntimeError("Validated IGES subfigure child changed type.")
                uses.append(use)
                self.stage.provenance.append(
                    (
                        assembly_use_label(definition_id, use.use_id),
                        f"{definition.name} -> {child.name}",
                    )
                )
            definitions.append(BRepAssemblyDefinition(definition_id, tuple(uses)))
            self.stage.provenance.append(
                (f"assembly-definition:{definition_id}", definition.name)
            )

        root_rows: list[tuple[int, BRepAssemblyUse]] = []
        for instance in instance_entries:
            if instance.number in referenced_entities:
                continue
            target = self.integer(instance, 0)
            use = BRepAssemblyUse(
                f"iges-root-D{instance.number}",
                instance_placements[instance.number],
                definition_id=definition_ids[target],
            )
            root_rows.append((instance.number, use))
            self.stage.provenance.append(
                (assembly_use_label(None, use.use_id), f"root instance {instance.name}")
            )

        for definition in definition_entries:
            if (
                definition.number in referenced_definitions
                or definition.number in referenced_entities
            ):
                continue
            use_id = f"iges-root-D{definition.number}"
            use = BRepAssemblyUse(
                use_id, identity, definition_id=definition_ids[definition.number]
            )
            root_rows.append((definition.number, use))
            self.stage.provenance.append(
                (assembly_use_label(None, use_id), f"root definition {definition.name}")
            )

        for entry in sorted(
            (value for value in self.file.entries.values() if value.entity_type == 186),
            key=lambda value: value.number,
        ):
            if entry.number in referenced_entities:
                continue
            solid = self.solid_indices.get(entry.number)
            if solid is None:
                raise refuse(
                    "inconsistent-geometry",
                    entry.name,
                    "An IGES root solid was not staged.",
                    (entry.name,),
                )
            placement = solid_placements.get(entry.number)
            if placement is None:
                placement = self._exact_entry_placement(entry, (entry.name,))
                solid_placements[entry.number] = placement
            use_id = f"iges-root-D{entry.number}"
            use = BRepAssemblyUse(use_id, placement, solid_index=solid)
            root_rows.append((entry.number, use))
            self.stage.provenance.append(
                (assembly_use_label(None, use_id), f"root solid {entry.name}")
            )

        graph = BRepAssemblyGraph(
            tuple(use for _, use in sorted(root_rows, key=lambda row: row[0])),
            tuple(definitions),
            len(self.stage.solids),
        )

        def orientation(placement: BRepAffinePlacement, /) -> int:
            a, b, c = placement.matrix[0][:3]
            d, e, f = placement.matrix[1][:3]
            g, h, i = placement.matrix[2][:3]
            determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            return 1 if determinant > 0 else -1

        definition_map = {value.definition_id: value for value in graph.definitions}
        orientation_states: dict[str, set[int]] = {}
        pending: list[tuple[str, int, tuple[str, ...]]] = []
        for use in reversed(graph.roots):
            sign = orientation(use.placement)
            if use.solid_index is not None:
                if sign < 0:
                    raise refuse(
                        "unsupported-entity",
                        use.use_id,
                        "iges.occurrence.placement must be proper rigid; got a reflection",
                        (use.use_id,),
                    )
            elif use.definition_id is not None:
                pending.append((use.definition_id, sign, (use.use_id,)))
        while pending:
            definition_id, outer_sign, chain = pending.pop()
            seen_signs = orientation_states.setdefault(definition_id, set())
            if outer_sign in seen_signs:
                continue
            seen_signs.add(outer_sign)
            for use in reversed(definition_map[definition_id].uses):
                sign = outer_sign * orientation(use.placement)
                if use.solid_index is not None:
                    if sign < 0:
                        raise refuse(
                            "unsupported-entity",
                            use.use_id,
                            "iges.occurrence.placement must be proper rigid; got a reflection",
                            (*chain, use.use_id),
                        )
                elif use.definition_id is not None:
                    pending.append((use.definition_id, sign, (*chain, use.use_id)))
        self.stage.assembly_graph = graph
        return graph

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
                "iges.boundary.model_curve must be present; got no model-space curve",
                chain,
            )
        self._admit_composite(model_curve, chain)
        segments = self._segments(model_curve, chain)
        parameter_pointer = self.integer(entry, 2, 0)
        if parameter_pointer:
            self._admit_composite(parameter_pointer, chain)
        parameters = (
            self.parameter_segments(parameter_pointer, patch, chain)
            if parameter_pointer
            else [None] * len(segments)
        )
        if len(parameters) != len(segments):
            raise refuse(
                "unsupported-entity",
                entry.name,
                "iges.boundary.segmentation must agree in model and parameter space; got differing segments",
                chain,
            )
        self.active.pop()
        coedges = []
        points = []
        for curve, first, last in segments:
            ends = np.asarray(
                curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)),
                dtype=np.float64,
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
                        "iges.boundary.endpoints must close the loop; got an open boundary",
                        chain,
                    )
            else:
                if np.linalg.norm(ends[1] - points[index + 1][0]) > tolerance:
                    raise refuse(
                        "inconsistent-geometry",
                        entry.name,
                        "iges.boundary.segments must connect at their endpoints; got disconnected segments",
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
            parameter = parameters[index]
            if parameter is None:
                pcurve = None
            else:
                pcurve, domain = parameter
                pcurve = self.align_parameter_curve(pcurve, domain, edge, patch)
            coedges.append(
                StagedCoedge(
                    edge,
                    1,
                    pcurve,
                    label,
                    origin="absent" if pcurve is None else "file",
                )
            )
        return coedges

    def _admit_composite(self, pointer: int, chain: tuple[str, ...]) -> None:
        """Reject cyclic composite-curve references with an iterative DFS."""
        active: set[int] = set()
        complete: set[int] = set()
        pending: list[tuple[int, tuple[str, ...], bool]] = [(pointer, chain, False)]
        allowed = (100, 102, 104, 110, 126)
        while pending:
            current, current_chain, exiting = pending.pop()
            entry = self.entry(current, current_chain, allowed)
            if entry.entity_type != 102:
                continue
            if exiting:
                active.remove(current)
                complete.add(current)
                continue
            if current in active:
                raise refuse(
                    "cyclic-reference",
                    entry.name,
                    "iges.composite_curve references must be acyclic.",
                    current_chain,
                )
            if current in complete:
                continue
            active.add(current)
            pending.append((current, current_chain, True))
            child_chain = (*current_chain, entry.name)
            children = [
                self.integer(entry, 1 + index) for index in range(self.count(entry, 0))
            ]
            pending.extend((child, child_chain, False) for child in reversed(children))

    def parameter_segments(
        self, pointer: int, patch: AbstractSurfacePatch, chain: tuple[str, ...]
    ) -> list[tuple[AbstractCurve, tuple[float, float]]]:
        self._admit_composite(pointer, chain)
        result: list[tuple[AbstractCurve, tuple[float, float]]] = []
        pending = [(pointer, chain)]
        while pending:
            current, current_chain = pending.pop()
            entry = self.entry(current, current_chain, (100, 102, 104, 110, 126))
            if entry.entity_type != 102:
                result.append(self.parameter_curve(current, patch, current_chain))
                continue
            next_chain = (*current_chain, entry.name)
            if entry.transform:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "iges.pcurve.composite_transform must preserve one parameter frame; got a transformed composite",
                    next_chain,
                )
            children = [
                self.integer(entry, 1 + index) for index in range(self.count(entry, 0))
            ]
            pending.extend((child, next_chain) for child in reversed(children))
        return result

    def _segments(
        self, pointer: int, chain: tuple[str, ...]
    ) -> list[tuple[AbstractCurve, float, float]]:
        self._admit_composite(pointer, chain)
        result: list[tuple[AbstractCurve, float, float]] = []
        pending = [(pointer, chain)]
        while pending:
            current, current_chain = pending.pop()
            entry = self.entry(current, current_chain, (100, 102, 104, 110, 126))
            if entry.entity_type != 102:
                curve = self.curve(current, current_chain)
                result.append((curve, *self._range(entry, current_chain)))
                continue
            next_chain = (*current_chain, entry.name)
            if not self.placement(entry, next_chain).is_identity:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    "iges.composite_curve.transform must be absent; got a transformed composite",
                    next_chain,
                )
            children = [
                self.integer(entry, 1 + index) for index in range(self.count(entry, 0))
            ]
            pending.extend((child, next_chain) for child in reversed(children))
        return result

    def _range(self, entry: _Entry, chain: tuple[str, ...] = ()) -> tuple[float, float]:
        """Parameter range of a native curve or a single-use composite chain."""
        seen: set[int] = set()
        current_chain = chain
        while entry.entity_type == 102:
            if entry.number in seen:
                raise refuse(
                    "cyclic-reference",
                    entry.name,
                    "iges.composite_curve.references must be acyclic.",
                    current_chain,
                )
            seen.add(entry.number)
            current_chain = (*current_chain, entry.name)
            count = self.count(entry, 0)
            if count != 1:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    f"iges.composite_curve.segments must contain one edge carrier; got {count} segments",
                    current_chain,
                )
            pointer = self.integer(entry, 1)
            if pointer in seen:
                raise refuse(
                    "cyclic-reference",
                    entry.name,
                    "iges.composite_curve.references must be acyclic.",
                    current_chain,
                )
            entry = self.entry(pointer, current_chain, (100, 102, 104, 110, 126))
        current_chain = (*current_chain, entry.name)
        match entry.entity_type:
            case 110:
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
                    local = self._conic(entry, current_chain)
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
            case 126:
                upper, degree = self.integer(entry, 0), self.integer(entry, 1)
                index = 6 + (upper + degree + 2) + 4 * (upper + 1)
                return self.real(entry, index), self.real(entry, index + 1)
            case _:
                raise refuse(
                    "unsupported-entity",
                    entry.name,
                    f"iges.curve.entity_type must define a supported parameter range; got {entry.entity_type}",
                    current_chain,
                )

    def compact_placed_topology(self) -> None:
        """Remove placement templates, retaining source-labeled incidences."""
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


def _import(resource: CadResource, policy: CadImportPolicy, /) -> CadImportResult:
    file = _decode(resource.data)
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
    for entry in by_type.get(186, []):
        resolver.solid(entry)
    resolver.assembly_graph()
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
            "iges.file.geometry must contain supported B-Rep faces; got nonempty geometry without faces",
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
                    "iges.file.entity must be consumed or classified as non-geometric; got unconsumed geometry",
                    (entry.name,),
                )
            dropped[entry.entity_type] = dropped.get(entry.entity_type, 0) + 1
    digest = resource.digest
    identity = resource.source_path or f"iges-{digest[:16]}"
    resolver.compact_placed_topology()
    model = resolver.stage.publish(
        coordinate_contract=policy.coordinate_contract,
        source_id=identity,
        source_format=_FORMAT,
        source_digest=digest,
        import_policy_id=policy.policy_id,
        tessellation=policy.tessellation,
    )
    stage = resolver.stage
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
    losses.extend(cad_import_stage_losses(stage))
    unit_name = _UNITS[file.unit_flag][0]
    resolution_qualifiers: dict[str, str] = {}
    if file.minimum_resolution is not None:
        # Retain G19 as descriptive coordinate-discernibility metadata only.
        # Its physical unit conversion does not enter any gap admission path.
        resolution = _real(file.minimum_resolution, "G19")
        resolution_qualifiers = {
            "minimum-coordinate-discernibility-source-units": _string(
                file.minimum_resolution
            ),
            "minimum-coordinate-discernibility-target-units": format(
                resolution * factor, ".17g"
            ),
        }
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        _FORMAT,
        "phydrax-brep",
        source_id=identity,
        target_id=model.model_id,
        source_profile=AdapterFormatProfile(
            _FORMAT,
            qualifiers={
                "profile": _SCHEMA,
                "artifact-sha256": digest,
            }
            | resolution_qualifiers,
        ),
        coordinate_mapping=(
            f"IGES unit {unit_name} -> {policy.coordinate_contract.length_unit.symbol} "
            f"(factor {factor!r})",
        ),
        preserved_fields=(
            "exact-carriers",
            "oriented-topology",
            "voids",
            "p-curves",
            "assembly-definitions",
            "assembly-uses",
        ),
        assumptions=(
            "supplied p-curves retained; absent p-curves require an exact algebraic source map or import refuses",
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
        report=report,
        coverage=stage.coverage(dict(resolver.counts)),
        provenance=tuple(stage.provenance),
        entity_metadata=None,
        # IGES declares no gap tolerance (G19 is coordinate discernibility), so
        # the evidence is the model's certified gaps against its own limit.
        tolerance_evidence=build_cad_import_tolerance_evidence(model, ()),
    )


@checked
def decode_iges_resource(
    resource: CadResource, policy: CadImportPolicy, /
) -> CadImportResult:
    """Decode one mapped or caller-owned IGES 5.3 resource."""
    try:
        return _import(resource, policy)
    except CadInterchangeError:
        raise
    except (
        ValueError,
        TypeError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as error:
        failure = refuse("malformed", "file", str(error), ("file",))
        raise failure from error


@checked
def decode_iges_bytes(
    data: bytes, policy: CadImportPolicy, /, *, source_path: str | None = None
) -> CadImportResult:
    """Decode caller-owned IGES bytes."""
    return decode_cad_bytes(
        data,
        policy,
        decoder=decode_iges_resource,
        source_path=source_path,
    )


@checked
def read_iges(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
) -> CadImportResult:
    """Decode one descriptor-owned, indexed local IGES file."""
    return read_cad_file(
        path,
        policy,
        decoder=decode_iges_resource,
        trusted_root=trusted_root,
    )


# ------------------------------------------------------------------ writer


def _real_text(value: float, /) -> str:
    number = float(value)
    if not isfinite(number):
        raise refuse(
            "inexact-export",
            "number",
            f"iges.export.number must be finite; got {number!r}",
        )
    text = repr(number).upper()
    mantissa, _, exponent = text.partition("E")
    if "." not in mantissa:
        mantissa += "."
    return mantissa + ("E" + exponent if exponent else "")


def _fraction_text(value: Fraction, label: str, /) -> str:
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise refuse(
            "inexact-export",
            label,
            "IGES real fields must exactly represent rational placement coefficients; got a nonterminating decimal.",
        )
    places = max(twos, fives)
    numerator = value.numerator * (2 ** (places - twos)) * (5 ** (places - fives))
    sign = "-" if numerator < 0 else ""
    digits = str(abs(numerator))
    if places == 0:
        return f"{sign}{digits}."
    if len(digits) <= places:
        digits = "0" * (places + 1 - len(digits)) + digits
    text = f"{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")
    return f"{sign}{text}"


def _hollerith(text: str, /) -> str:
    if any(ord(character) < 32 or ord(character) > 126 for character in text):
        raise refuse(
            "inexact-export",
            "text",
            "iges.export.text must contain printable ASCII; got a nonprintable or non-ASCII character",
        )
    return f"{len(text)}H{text}"


# IGES records contain 80 ASCII columns and one newline. Directory entries
# occupy two records; P records reserve 64 columns for parameter text.
_DIRECTORY_RECORDS = 2
_PARAMETER_COLUMNS = 64
_GLOBAL_COLUMNS = 72
_MAXIMUM_RECORD_SEQUENCE = 10**7 - 1
_MAXIMUM_DIRECTORY_ENTITIES = (10**7 - 1) // _DIRECTORY_RECORDS


def _iges_directory_label(value: str, /) -> str | None:
    if (
        not value
        or len(value) > 8
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
    ):
        return None
    return value


@dataclass(slots=True)
class _Out:
    number: int
    entity_type: int
    parameters: list[str]
    form: int = 0
    transform: _Out | None = None
    root: bool = False
    entity_label: str = ""

    def __str__(self) -> str:
        """Return the IGES directory reference used in parameter records."""
        return str(self.number)


@final
class _EdgeEntities(StrictModule):
    vertex_list: _Out
    edge_list: _Out
    edge_index: dict[int, int]


@final
class _FaceEntities(StrictModule):
    faces: tuple[_Out, ...]
    frame_signs: tuple[int, ...]


@final
class _RootEntities(StrictModule):
    shells: tuple[_Out, ...]
    solids: tuple[_Out, ...]


@final
class _ParameterFrame(StrictModule):
    matrix: np.ndarray
    shift: np.ndarray


@dataclass(slots=True)
class _IgesWriter:
    model: BRepModel
    policy: CadExportPolicy
    entities: list[_Out] = field(default_factory=list)
    assembly_provenance: list[tuple[str, str]] = field(default_factory=list)
    surface_carriers: tuple[AbstractSurfacePatch, ...] = field(init=False)
    surface_losses: tuple[AdapterLoss, ...] = field(init=False)
    approximation_losses: tuple[AdapterLoss, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.surface_carriers, self.surface_losses = prepare_cad_export_surfaces(
            self.model.patches, "IGES"
        )

    def add(
        self,
        entity_type: int,
        parameters: list[str],
        form: int = 0,
        transform: _Out | None = None,
        *,
        entity_label: str = "",
    ) -> _Out:
        if len(self.entities) >= _MAXIMUM_DIRECTORY_ENTITIES:
            raise refuse(
                "inexact-export",
                "export",
                "IGES seven-digit directory pointers cannot represent another entity.",
            )
        entity = _Out(
            _DIRECTORY_RECORDS * len(self.entities) + 1,
            entity_type,
            parameters,
            form,
            transform,
            entity_label=entity_label,
        )
        self.entities.append(entity)
        return entity

    def provenance(self, /) -> tuple[tuple[str, str], ...]:
        return tuple(self.assembly_provenance)

    def reals(self, values: ConvertibleToArray) -> list[str]:
        return [_real_text(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]

    def transform(
        self,
        origin: ConvertibleToArray,
        first: ConvertibleToArray,
        second: ConvertibleToArray,
        label: str,
    ) -> _Out:
        first_values = np.asarray(first, dtype=np.float64)
        second_values = np.asarray(second, dtype=np.float64)
        first_, second_, normal = orthonormal_frame(first_values, second_values, label)
        rotation = np.stack((first_, second_, normal), axis=1)
        matrix = np.concatenate((rotation, np.asarray(origin)[:, None]), axis=1)
        return self.add(124, self.reals(matrix))

    def point(self, value: ConvertibleToArray) -> _Out:
        return self.add(116, self.reals(value))

    def direction(self, value: ConvertibleToArray) -> _Out:
        return self.add(123, self.reals(value))

    def curve(self, curve: AbstractCurve, first: float, last: float, label: str) -> _Out:
        curve = explicit_carrier(curve)
        match curve:
            case LineCurve():
                ends = np.asarray(
                    curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)),
                    dtype=np.float64,
                )
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
                    f"iges.export.curve must have a supported IGES entity; got {type(curve).__name__}",
                )

    # IGES 5.3 headers and control ordering are external layout contracts.
    # Keep their straight-line parameter lists beside the fail-closed carrier dispatch.
    def surface(self, patch: AbstractSurfacePatch, box: np.ndarray, label: str) -> _Out:
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
                    case TorusPatch():
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
                    case _:
                        assert_never(patch)
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
                curve = explicit_carrier(patch.curve)
                directrix = self.curve(curve, first, last, label)
                match curve:
                    case LineCurve():
                        start_parameter = first
                    case BSplineCurve():
                        domain = curve.parameter_domain
                        if domain is None:
                            raise refuse(
                                "inexact-export",
                                label,
                                "iges.export.directrix_domain must be finite; got an unbounded spline domain",
                            )
                        start_parameter = domain[0]
                    case _:
                        start_parameter = 0.0
                end = np.asarray(
                    curve.evaluate(jnp.asarray(start_parameter, dtype=jnp.float64)),
                    dtype=np.float64,
                ) + np.asarray(patch.direction)
                return self.add(122, [str(directrix), *self.reals(end)])
            case OffsetSurface():
                if float(patch.distance) == 0.0:
                    raise refuse(
                        "inexact-export",
                        label,
                        "iges.export.offset_distance must be nonzero; got zero",
                    )
                basis = self.surface(patch.base, box, label)
                domain = natural_box(patch.base)
                midpoint = (
                    np.mean(domain, axis=0)
                    if np.all(np.isfinite(domain))
                    else np.zeros(2, dtype=np.float64)
                )
                frame = np.asarray(
                    surface_differential(
                        patch.base, np.asarray(midpoint, dtype=np.float64)
                    ),
                    dtype=np.float64,
                )
                normal = np.cross(frame[:, 0], frame[:, 1])
                magnitude = float(np.linalg.norm(normal))
                if not isfinite(magnitude) or magnitude == 0.0:
                    raise refuse(
                        "inexact-export",
                        label,
                        "iges.export.offset_normal must be regular; got a zero or nonfinite normal",
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
                    f"iges.export.surface must have a supported IGES entity; got {type(patch).__name__}",
                )

    def _parameter_frame(
        self, patch: AbstractSurfacePatch, box: np.ndarray, label: str
    ) -> _ParameterFrame:
        """Map native UV coordinates to the emitted IGES surface parameter frame."""
        while isinstance(patch, OffsetSurface):
            patch = patch.base
        matrix, shift = np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
        match patch:
            case PlanePatch():
                first_raw = np.asarray(patch.first_axis)
                second_raw = np.asarray(patch.second_axis)
                axis, second, _ = orthonormal_frame(
                    first_raw / np.linalg.norm(first_raw),
                    second_raw / np.linalg.norm(second_raw),
                    label,
                )
                matrix = np.stack((axis, second)) @ np.stack(
                    (first_raw, second_raw), axis=1
                )
            case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
                first_axis, second_axis, _ = orthonormal_frame(
                    patch.first_axis, patch.second_axis, label
                )
                external_second = np.cross(np.asarray(patch.axis), first_axis)
                handedness = float(external_second @ second_axis)
                if abs(abs(handedness) - 1.0) > 1.0e-9:
                    raise refuse(
                        "inexact-export",
                        label,
                        "iges.export.surface_frame must align with its axis; got a misaligned frame",
                    )
                matrix[0, 0] = (1.0 if handedness > 0.0 else -1.0) * degrees(1.0)
                match patch:
                    case SpherePatch():
                        matrix[1, 1] = degrees(1.0)
                    case TorusPatch():
                        matrix[1, 1] = degrees(1.0)
                        matrix = matrix[::-1]
                    case _:
                        pass
            case ExtrusionSurface(curve=LineCurve() as directrix):
                matrix[0, 0] = float(np.linalg.norm(np.asarray(directrix.direction)))
                shift[0] = -float(box[0, 0]) * matrix[0, 0]
            case RevolutionSurface():
                # IGES orders the generatrix parameter before the angle.
                # Reversing the angle preserves the native surface normal.
                scale = (
                    1.0 / float(box[1, 1] - box[0, 1])
                    if isinstance(patch.curve, LineCurve)
                    else 1.0
                )
                matrix = np.asarray(((0.0, scale), (-1.0, 0.0)))
                shift = np.asarray((-float(box[0, 1]) * scale, 2.0 * np.pi))
            case _:
                pass
        return _ParameterFrame(matrix, shift)

    def parameter_curve(
        self,
        curve: AbstractCurve,
        patch: AbstractSurfacePatch,
        box: np.ndarray,
        edge_curve: AbstractCurve | None,
        first: float,
        last: float,
        label: str,
    ) -> _Out:
        """Write a UV carrier in the emitted surface and edge parameterization."""
        frame = self._parameter_frame(patch, box, label)
        matrix, shift = frame.matrix, frame.shift
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
                        curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)),
                        dtype=np.float64,
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
                        "iges.export.conic_parameters must retain their angular interval; got affine reparameterization",
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
                        "iges.export.conic_axes must be orthogonal; got a skew UV image",
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
                    f"iges.export.pcurve must have a supported IGES entity; got {type(curve).__name__}",
                    (label,),
                )
        return self.curve(output, parameter_first, parameter_last, label)

    def _edge_entities(self, geometry: BRepGeometry) -> _EdgeEntities:
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
            match carrier:
                case AbstractCurve():
                    curve = self.curve(
                        explicit_carrier(carrier),
                        float(ranges[edge, 0]),
                        float(ranges[edge, 1]),
                        f"edge:{edge}",
                    )
                case _:
                    raise refuse(
                        "inexact-export",
                        f"edge:{edge}",
                        "iges.export.edge must have an explicit curve carrier; got an implicit carrier",
                        (f"edge:{edge}",),
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
        return _EdgeEntities(vertex_list, edge_list, edge_index)

    def _coedge_row(
        self,
        geometry: BRepGeometry,
        edges: _EdgeEntities,
        face: int,
        coedge: int,
        frame_sign: int,
        box: np.ndarray,
        interval: np.ndarray,
    ) -> list[str]:
        edge = geometry.coedge_edges[coedge]
        agrees = "1" if frame_sign * geometry.coedge_senses[coedge] > 0 else "0"
        first, last = (float(value) for value in interval)
        pcurve = geometry.pcurves[coedge]
        match pcurve:
            case AbstractCurve():
                pass
            case _:
                raise refuse(
                    "inexact-export",
                    f"coedge:{coedge}",
                    "iges.export.pcurve must have an explicit curve carrier; got an implicit carrier",
                    (f"face:{face}", f"coedge:{coedge}"),
                )
        if edge in edges.edge_index:
            edge_curve = geometry.curves[geometry.edge_curves[edge]]
            match edge_curve:
                case AbstractCurve():
                    pass
                case _:
                    raise refuse(
                        "inexact-export",
                        f"edge:{edge}",
                        "iges.export.edge must have an explicit curve carrier; got an implicit carrier",
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
                            "iges.export.vertex_pcurve must support reversed parameterization; got an unsupported carrier",
                            (f"face:{face}", f"coedge:{coedge}"),
                        )
            agrees = "1"
        pointer = self.parameter_curve(
            explicit_carrier(pcurve),
            self.surface_carriers[face],
            box,
            None if edge_curve is None else explicit_carrier(edge_curve),
            first,
            last,
            f"coedge:{coedge}",
        )
        if edge in edges.edge_index:
            return [
                "0",
                str(edges.edge_list),
                str(edges.edge_index[edge]),
                agrees,
                "1",
                "0",
                str(pointer),
            ]
        return [
            "1",
            str(edges.vertex_list),
            str(geometry.edge_vertices[edge][0] + 1),
            agrees,
            "1",
            "0",
            str(pointer),
        ]

    def _face_entities(
        self, geometry: BRepGeometry, edges: _EdgeEntities
    ) -> _FaceEntities:
        faces: list[_Out] = []
        frame_signs: list[int] = []
        bounds = np.asarray(self.model.parameter_bounds)
        ranges = np.asarray(geometry.edge_ranges)
        for face, loops in enumerate(geometry.face_loops):
            surface = self.surface(
                self.surface_carriers[face], bounds[face], f"face:{face}"
            )
            patch = self.surface_carriers[face]
            while isinstance(patch, OffsetSurface):
                patch = patch.base
            frame_sign = 1
            match patch:
                case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
                    handedness = float(
                        np.cross(np.asarray(patch.axis), np.asarray(patch.first_axis))
                        @ np.asarray(patch.second_axis)
                    )
                    frame_sign = 1 if handedness > 0.0 else -1
                case _:
                    pass
            frame_signs.append(frame_sign)
            loop_pointers: list[str] = []
            for loop in loops:
                rows: list[str] = []
                for coedge in loop if frame_sign > 0 else reversed(loop):
                    rows.extend(
                        self._coedge_row(
                            geometry,
                            edges,
                            face,
                            coedge,
                            frame_sign,
                            bounds[face],
                            ranges[geometry.coedge_edges[coedge]],
                        )
                    )
                loop_pointers.append(str(self.add(508, [str(len(loop)), *rows], form=1)))
            faces.append(
                self.add(
                    510,
                    [str(surface), str(len(loops)), "1", *loop_pointers],
                    form=1,
                    entity_label=_iges_directory_label(self.model.physical_tags[face])
                    or "",
                )
            )
        return _FaceEntities(tuple(faces), tuple(frame_signs))

    def _root_entities(
        self, geometry: BRepGeometry, faces: _FaceEntities
    ) -> _RootEntities:
        orientation = np.asarray(self.model.orientation)
        shells: list[_Out] = []
        for members, signs in zip(
            geometry.shell_faces, geometry.shell_orientations, strict=True
        ):
            rows: list[str] = []
            for face, sign in zip(members, signs, strict=True):
                sense = sign * orientation[face] * faces.frame_signs[face]
                rows.extend((str(faces.faces[face]), "1" if sense > 0 else "0"))
            closed = any(
                members[0] in geometry.shell_faces[s]
                for group in geometry.solid_shells
                for s in group
            )
            shells.append(
                self.add(514, [str(len(members)), *rows], form=1 if closed else 2)
            )
        solids: list[_Out] = []
        for group in geometry.solid_shells:
            voids: list[str] = []
            for shell in group[1:]:
                voids.extend((str(shells[shell]), "1"))
            solids.append(
                self.add(186, [str(shells[group[0]]), "1", str(len(group) - 1), *voids])
            )
        return _RootEntities(tuple(shells), tuple(solids))

    def _mark_free_roots(
        self, geometry: BRepGeometry, faces: _FaceEntities, roots: _RootEntities
    ) -> None:
        in_solid = {shell for group in geometry.solid_shells for shell in group}
        for index, shell in enumerate(roots.shells):
            if index not in in_solid:
                shell.root = True
        in_shell = {face for members in geometry.shell_faces for face in members}
        free = [face for face in range(len(faces.faces)) if face not in in_shell]
        if free:
            orientation = np.asarray(self.model.orientation)
            rows: list[str] = []
            for face in free:
                sense = orientation[face] * faces.frame_signs[face]
                rows.extend((str(faces.faces[face]), "1" if sense > 0 else "0"))
            self.add(514, [str(len(free)), *rows], form=2).root = True

    def encode(self) -> bytes:
        model = self.model
        geometry = model.geometry
        if geometry is None:
            raise refuse(
                "inexact-export",
                model.source_id,
                "iges.export.model_geometry must be present; got no B-Rep geometry",
            )
        geometry, _, self.approximation_losses = prepare_cad_export_geometry(
            model, geometry, self.policy, format_name=_FORMAT
        )
        graph = geometry.assembly_graph
        if (
            not model.patches
            and not geometry.edge_curves
            and not len(geometry.vertex_points)
        ):
            if graph is not None and (graph.roots or graph.definitions):
                self.assembly((), geometry)
            else:
                self.add(308, ["0", _hollerith(""), "0"]).root = True
            return self._text(0.0)
        edges = self._edge_entities(geometry)
        faces = self._face_entities(geometry, edges)
        roots = self._root_entities(geometry, faces)
        self.assembly(roots.solids, geometry)
        self._mark_free_roots(geometry, faces, roots)
        extent = (
            float(np.max(np.abs(np.asarray(geometry.vertex_points))))
            if len(geometry.vertex_points)
            else 0.0
        )
        return self._text(extent)

    def _assembly_transform(
        self, placement: BRepAffinePlacement, label: str, /
    ) -> _Out | None:
        matrix = placement.matrix
        identity = BRepAffinePlacement.identity().matrix
        if matrix == identity:
            return None
        a, b, c = matrix[0][:3]
        d, e, f = matrix[1][:3]
        g, h, i = matrix[2][:3]
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if determinant <= 0:
            raise refuse(
                "inexact-export",
                label,
                "iges.export.assembly_placement must be proper; got a reflection.",
            )
        for first in range(3):
            for second in range(3):
                dot = sum(
                    (matrix[row][first] * matrix[row][second] for row in range(3)),
                    Fraction(0),
                )
                expected = Fraction(1) if first == second else Fraction(0)
                if dot != expected:
                    raise refuse(
                        "inexact-export",
                        label,
                        "iges.export.assembly_placement must be exactly orthogonal; got scale or shear.",
                    )
        values = [
            _fraction_text(matrix[row][column], label)
            for row in range(3)
            for column in range(4)
        ]
        return self.add(124, values)

    def assembly(self, solids: tuple[_Out, ...], geometry: BRepGeometry) -> None:
        graph = geometry.assembly_graph
        if graph is None:
            graph = BRepAssemblyGraph.single_solids(len(solids))
        if graph.solid_count != len(solids):
            raise refuse(
                "inconsistent-geometry",
                "assembly",
                "iges.export.assembly_graph must reference the emitted solid bank.",
            )
        definitions = {
            definition.definition_id: definition for definition in graph.definitions
        }
        depths: dict[str, int] = {}
        state: dict[str, int] = {}
        for root in definitions:
            if state.get(root, 0) == 2:
                continue
            stack: list[tuple[str, bool]] = [(root, False)]
            while stack:
                definition_id, exiting = stack.pop()
                definition = definitions[definition_id]
                if exiting:
                    child_depths = tuple(
                        depths[use.definition_id]
                        for use in definition.uses
                        if use.definition_id is not None
                    )
                    depths[definition_id] = (
                        0 if not definition.uses else 1 + max(child_depths, default=0)
                    )
                    state[definition_id] = 2
                    continue
                current_state = state.get(definition_id, 0)
                if current_state == 2:
                    continue
                if current_state == 1:
                    raise refuse(
                        "cyclic-reference",
                        definition_id,
                        "iges.export.assembly_graph must be acyclic.",
                    )
                state[definition_id] = 1
                stack.append((definition_id, True))
                for use in reversed(definition.uses):
                    child = use.definition_id
                    if child is None:
                        continue
                    child_state = state.get(child, 0)
                    if child_state == 1:
                        raise refuse(
                            "cyclic-reference",
                            child,
                            "iges.export.assembly_graph must be acyclic.",
                            (definition_id,),
                        )
                    if child_state == 0:
                        stack.append((child, False))

        definition_records = {
            definition_id: self.add(308, ["0", _hollerith(""), "0"])
            for definition_id in definitions
        }
        for definition_id, record in definition_records.items():
            self.assembly_provenance.append(
                (f"assembly-definition:{definition_id}", f"D{record.number}")
            )
        solid_definitions = {
            index: self.add(308, ["0", _hollerith(""), "1", str(solid)])
            for index, solid in enumerate(solids)
        }
        for index, record in solid_definitions.items():
            self.assembly_provenance.append(
                (f"assembly-solid-definition:{index}", f"D{record.number}")
            )
        for index, solid in enumerate(solids):
            self.assembly_provenance.append((f"solid:{index}", f"D{solid.number}"))

        def use_instance(use: BRepAssemblyUse, owner: str | None, /) -> _Out:
            if use.solid_index is not None:
                target = solid_definitions[use.solid_index]
            else:
                definition_id = use.definition_id
                if definition_id is None:
                    raise RuntimeError("A validated IGES assembly use lost its target.")
                target = definition_records[definition_id]
            label = assembly_use_label(owner, use.use_id)
            transform = self._assembly_transform(use.placement, label)
            instance = self.add(
                408,
                [str(target), "0.", "0.", "0.", "1."],
                transform=transform,
            )
            self.assembly_provenance.append((label, f"D{instance.number}"))
            return instance

        for definition_id, definition in definitions.items():
            children = tuple(use_instance(use, definition_id) for use in definition.uses)
            definition_records[definition_id].parameters = [
                str(depths[definition_id]),
                _hollerith(""),
                str(len(children)),
                *(str(child) for child in children),
            ]

        used_solid_indices = {
            use.solid_index for use in graph.roots if use.solid_index is not None
        }
        used_solid_indices.update(
            use.solid_index
            for definition in graph.definitions
            for use in definition.uses
            if use.solid_index is not None
        )
        for index, record in solid_definitions.items():
            if index not in used_solid_indices:
                record.root = True
        for use in graph.roots:
            use_instance(use, None).root = True

    def _text(self, extent: float) -> bytes:
        unit = self.model.coordinate_contract.length_unit
        meters = Fraction(unit.scale_to_reference)
        flags = [flag for flag, (_, value) in _UNITS.items() if value == meters]
        if not flags:
            raise refuse(
                "units",
                unit.symbol,
                f"iges.export.length_unit must have an IGES unit flag; got {unit.symbol!r}",
            )
        flag = flags[0]
        timestamp = "19700101.000000"
        global_values = [
            _hollerith(","),
            _hollerith(";"),
            _hollerith("Phydrax"),
            _hollerith("phydrax-model.iges"),
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
            # IGES 5.3 §2.2.4.3.19: minimum discernible coordinate distance.
            # Preserve this resolution independently of the bounded gap law.
            "1.0E-07",
            _real_text(max(extent, 1.0)),
            _hollerith(""),
            _hollerith(""),
            "11",
            "0",
            _hollerith(timestamp),
        ]
        global_length = sum(len(value) + 1 for value in global_values)
        global_records = (global_length + _GLOBAL_COLUMNS - 1) // _GLOBAL_COLUMNS
        label = "Phydrax native exact B-Rep export"
        start_text = [label]
        start = [
            text.ljust(72) + f"S{index + 1:7d}" for index, text in enumerate(start_text)
        ]
        if (
            len(start) > _MAXIMUM_RECORD_SEQUENCE
            or global_records > _MAXIMUM_RECORD_SEQUENCE
        ):
            raise refuse(
                "inexact-export",
                "export",
                "IGES seven-digit S/G record sequences cannot represent this payload.",
            )
        glob = _wrap(global_values, ",", ";", 72)
        global_lines = [text.ljust(72) + f"G{i + 1:7d}" for i, text in enumerate(glob)]
        directory: list[str] = []
        parameter_lines: list[str] = []
        for entity in self.entities:
            number = entity.number
            chunks = _wrap(
                [str(entity.entity_type), *entity.parameters],
                ",",
                ";",
                _PARAMETER_COLUMNS,
            )
            pointer = len(parameter_lines) + 1
            for chunk in chunks:
                record_sequence = len(parameter_lines) + 1
                if record_sequence > _MAXIMUM_RECORD_SEQUENCE:
                    raise refuse(
                        "inexact-export",
                        "export",
                        "IGES seven-digit P record sequence cannot represent this payload.",
                    )
                parameter_lines.append(
                    f"{chunk.ljust(64)} {number:7d}P{record_sequence:7d}"
                )
            status = "00000000" if entity.root else "00010000"
            transform = 0 if entity.transform is None else entity.transform.number
            first = [entity.entity_type, pointer, 0, 0, 0, 0, transform, 0]
            second = [entity.entity_type, 0, 0, len(chunks), entity.form]
            directory.append(
                "".join(f"{value:8d}" for value in first) + status + f"D{number:7d}"
            )
            directory.append(
                "".join(f"{value:8d}" for value in second)
                + " " * 16
                + entity.entity_label.ljust(8)
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


def _iges_line_pcurve_losses(geometry: BRepGeometry, /) -> tuple[AdapterLoss, ...]:
    return tuple(
        AdapterLoss(
            f"coedge:{coedge}.pcurve-carrier",
            "export",
            "transformed",
            "IGES encodes a line-valued p-curve as a degree-one B-spline; its locus and edge-parameter map are retained, but its primitive carrier identity is not.",
            changes_interpretation=True,
        )
        for coedge, pcurve in enumerate(geometry.pcurves)
        if isinstance(explicit_carrier(pcurve), LineCurve)
    )


def _iges_physical_tag_losses(model: BRepModel, /) -> tuple[AdapterLoss, ...]:
    return tuple(
        AdapterLoss(
            f"face:{face}.physical-tag",
            "export",
            "dropped",
            "IGES D-record labels preserve at most eight printable ASCII characters; this physical tag cannot be encoded and re-import derives the carrier-family tag.",
            changes_interpretation=True,
        )
        for face, tag in enumerate(model.physical_tags)
        if _iges_directory_label(tag) is None
    )


@checked
def write_iges(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    policy: CadExportPolicy | None = None,
    mode: PublicationMode = "exclusive",
) -> CadExportResult:
    """Publish exact IGES carriers; unsupported source structures are refused.

    Solids become 186 manifold solids (inner shells as voids); free shells and
    faces become open 514 shells. Coordinates use the model length unit declared
    in the Global section. Assembly definitions and uses remain a shared DAG.
    Fixed timestamps make identical native models produce identical bytes.
    """
    policy_ = CadExportPolicy() if policy is None else policy
    writer = _IgesWriter(model, policy_)
    data = writer.encode()
    digest = hashlib.sha256(data).hexdigest()
    publish_bytes(path, data, maximum_bytes=len(data), mode=mode)
    counts: dict[str, int] = {}
    for entity in writer.entities:
        counts[str(entity.entity_type)] = counts.get(str(entity.entity_type), 0) + 1
    report = build_cad_export_report(
        model,
        digest,
        policy_,
        format_name=_FORMAT,
        qualifiers={
            "profile": _SCHEMA,
            "artifact-sha256": digest,
        },
        target_id=digest,
        stage="adapter",
        coordinate_mapping=(
            f"IGES unit of {model.coordinate_contract.length_unit.symbol}",
        ),
        extra_losses=(
            *writer.surface_losses,
            *cad_export_pole_identity_losses(
                writer.surface_carriers, "IGES", "undeclared"
            ),
            *writer.approximation_losses,
            *(() if model.geometry is None else _iges_line_pcurve_losses(model.geometry)),
            *_iges_physical_tag_losses(model),
            *cad_export_assembly_identifier_losses(
                None if model.geometry is None else model.geometry.assembly_graph,
                _FORMAT,
            ),
        ),
        preserved_fields=(
            "exact-carriers",
            "oriented-topology",
            "voids",
            "physical-tags",
            "p-curves",
            "assembly-definitions",
            "assembly-uses",
        ),
    )
    return CadExportResult(
        Path(path).expanduser().resolve(),
        _FORMAT,
        _SCHEMA,
        report,
        tuple(sorted(counts.items())),
        writer.provenance(),
    )


__all__ = ["decode_iges_bytes", "decode_iges_resource", "read_iges", "write_iges"]

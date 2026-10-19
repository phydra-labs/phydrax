# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded ASCII DXF profiles preserving source curves and explicit units.

No fit-point interpolation, endpoint snapping, or inferred drawing units are
performed. Angular source uses retain analytic carriers and certify their
junction displacement against the caller's declared bound.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import assert_never, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._external_resource import (
    bounded_resource_from_bytes,
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
from .._fingerprint import canonical_fingerprint
from .._publication import PublicationMode, PublicationReceipt, publish_bytes
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_real_scalar, positive_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry.brep._carrier_pieces import carrier_payload
from ..geometry.brep._conic_gaps import endpoint_gap_bounds
from ..geometry.brep._constructors import (
    PlanarProfile,
    ProfileArc,
    ProfileCurveUse,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
    ProfileSegment,
)
from ..geometry.brep._patches import BSplineCurve, CircleCurve, EllipseCurve
from ..typing import checked, parse
from ..units import (
    CENTIMETER,
    FOOT,
    INCH,
    KILOMETER,
    LENGTH,
    METER,
    MICROMETER,
    MILLIMETER,
    UnitDefinition,
)
from ._report import (
    _AdapterDirection,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
)


type _Point = tuple[float, float]
type _Groups = tuple[tuple[int, str], ...]
type _Edge = tuple[_Point, _Point, ProfileSegment, str]

_UNITS = {
    1: INCH,
    2: FOOT,
    4: MILLIMETER,
    5: CENTIMETER,
    6: METER,
    7: KILOMETER,
    13: MICROMETER,
}
_ANNOTATIONS = frozenset(("TEXT", "MTEXT", "DIMENSION", "POINT", "VIEWPORT"))


@final
class DxfImportPolicy(StrictModule, NonTrainableState):
    """Declare parser limits and the maximum admitted source junction gap.

    The gap declaration never changes an authored coordinate or analytic curve.
    Zero requires coincident source junctions; a positive declaration is reported.
    """

    maximum_bytes: int = eqx.field(static=True)
    maximum_groups: int = eqx.field(static=True)
    maximum_entities: int = eqx.field(static=True)
    maximum_junction_gap: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_bytes: int = 4_194_304,
        maximum_groups: int = 100_000,
        maximum_entities: int = 10_000,
        maximum_junction_gap: float = 0.0,
    ) -> None:
        bytes_ = positive_integer(maximum_bytes, "maximum_bytes")
        groups = positive_integer(maximum_groups, "maximum_groups")
        entities = positive_integer(maximum_entities, "maximum_entities")
        gap = nonnegative_real_scalar(maximum_junction_gap, "maximum_junction_gap")
        self.maximum_bytes = bytes_
        self.maximum_groups = groups
        self.maximum_entities = entities
        self.maximum_junction_gap = gap


@final
class DxfImportResult(StrictModule, NonTrainableState):
    """One certified planar region, its declared coordinate unit and coverage."""

    profile: PlanarProfile
    coordinate_unit: UnitDefinition = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)

    @checked
    def __init__(
        self,
        profile: PlanarProfile,
        coordinate_unit: UnitDefinition,
        report: AdapterReport,
    ) -> None:
        if coordinate_unit.dimension != LENGTH:
            raise ValueError(
                "DxfImportResult.coordinate_unit must declare length; got a nonlength unit."
            )
        if not report.valid or report.target_format != "phydrax-planar-profile":
            raise ValueError(
                "DxfImportResult.report must describe a successful planar import; got invalid conversion evidence."
            )
        self.profile = profile
        self.coordinate_unit = coordinate_unit
        self.report = report


@final
class DxfExportResult(StrictModule, NonTrainableState):
    """Published byte identity and external semantic accounting."""

    receipt: PublicationReceipt = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)

    @checked
    def __init__(self, receipt: PublicationReceipt, report: AdapterReport) -> None:
        if not report.valid or report.target_format != "dxf":
            raise ValueError(
                "DxfExportResult.report must describe a successful DXF export; got invalid conversion evidence."
            )
        self.receipt = receipt
        self.report = report


class DxfImportError(ValueError):
    """Refused source with adapter coverage and any unjoined endpoints."""

    @checked
    def __init__(
        self,
        message: str,
        report: AdapterReport,
        *,
        open_endpoints: tuple[_Point, ...] = (),
    ) -> None:
        super().__init__(message)
        self.report = report
        self.open_endpoints = open_endpoints


class _SourceRefusal(ValueError):
    def __init__(
        self,
        message: str,
        *,
        endpoints: tuple[_Point, ...] = (),
        gaps: tuple[tuple[float, float], ...] = (),
    ) -> None:
        super().__init__(message)
        self.endpoints = endpoints
        self.gaps = gaps


def _junction_waivers(
    losses: tuple[AdapterLoss, ...],
    gaps: Sequence[tuple[float, float]],
    declared: float | None,
) -> tuple[AdapterWaiver, ...]:
    if not losses or declared is None or declared <= 0.0:
        return ()
    if any(max(row) > declared for row in gaps):
        return ()
    return tuple(
        AdapterWaiver(
            loss,
            f"Caller declared maximum_junction_gap={declared:.17g}; every recomputed bound is within that fixed declaration.",
        )
        for loss in losses
    )


def _report(
    digest: str,
    *,
    consumed: Sequence[str],
    refused: Sequence[str],
    ignored: Sequence[str],
    unit: UnitDefinition | None,
    failure: str | None = None,
    gaps: Sequence[tuple[float, float]] = (),
    declared_junction_gap: float | None = None,
    direction: _AdapterDirection = "import",
) -> AdapterReport:
    direction = parse(direction, _AdapterDirection, "direction")
    qualifiers = {
        "coverage-consumed": json.dumps(tuple(consumed), separators=(",", ":")),
        "coverage-refused": json.dumps(tuple(refused), separators=(",", ":")),
        "coverage-ignored-by-design": json.dumps(tuple(ignored), separators=(",", ":")),
    }
    if unit is not None:
        qualifiers["coordinate-unit"] = unit.unit_id
    if declared_junction_gap is not None:
        qualifiers["declared-maximum-junction-gap"] = format(
            declared_junction_gap, ".17g"
        )
    if failure is not None:
        qualifiers["refusal"] = failure
    if gaps:
        qualifiers["junction-gap-bounds"] = json.dumps(tuple(gaps), separators=(",", ":"))
    losses = ()
    if failure is None and any(max(row) > 0.0 for row in gaps):
        losses = (
            AdapterLoss(
                "geometry.junction-gaps",
                direction,
                "transformed",
                "Authored vertices and analytic endpoints retain certified nonzero displacement; no geometry moves.",
                changes_interpretation=True,
            ),
        )
    status = (
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        if failure is not None
        else AdapterStatus.DECLARED_LOSS
        if losses
        else AdapterStatus.LOSSLESS
    )
    match direction:
        case "import":
            source, target = "dxf", "phydrax-planar-profile"
        case "export":
            source, target = "phydrax-planar-profile", "dxf"
        case _:
            assert_never(direction)
    return AdapterReport(
        status,
        source,
        target,
        source_id=canonical_fingerprint({"kind": source, "source": digest}),
        target_id=canonical_fingerprint({"kind": target, "source": digest}),
        source_profile=AdapterFormatProfile(source, qualifiers=qualifiers),
        preserved_fields=(
            ("source-curves", "declared-units", "certified-loop-topology")
            if failure is None
            else ()
        ),
        losses=losses,
        waivers=_junction_waivers(losses, gaps, declared_junction_gap),
    )


def _parse(
    data: bytes, policy: DxfImportPolicy, budget: CoordinateEnclosureBudget
) -> tuple[_Groups, tuple[tuple[str, _Groups], ...], tuple[str, ...]]:
    if len(data) > policy.maximum_bytes:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF maximum_bytes exhausted before decoding."
        )
    budget.reserve(len(data), 8 * len(data) + 256 * policy.maximum_groups)
    text = data.decode("ascii")
    lines = text.splitlines()
    if len(lines) % 2 or len(lines) // 2 > policy.maximum_groups:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF requires paired group-code lines within maximum_groups."
        )
    groups = tuple(
        (int(lines[i].strip()), lines[i + 1].strip()) for i in range(0, len(lines), 2)
    )
    if not groups or groups[-1] != (0, "EOF"):
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF requires a terminal EOF group."
        )
    return _sections(groups, policy)


def _sections(
    groups: _Groups, policy: DxfImportPolicy
) -> tuple[_Groups, tuple[tuple[str, _Groups], ...], tuple[str, ...]]:
    header: _Groups = ()
    header_seen = False
    entities: list[tuple[str, _Groups]] = []
    ignored: list[str] = []
    index = 0
    while index < len(groups) - 1:
        if (
            groups[index] != (0, "SECTION")
            or index + 1 >= len(groups)
            or groups[index + 1][0] != 2
        ):
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF requires named sections."
            )
        name = groups[index + 1][1]
        start = index + 2
        index = start
        while index < len(groups) and groups[index] != (0, "ENDSEC"):
            index += 1
        if index == len(groups):
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF section lacks ENDSEC."
            )
        section = groups[start:index]
        if name == "HEADER":
            if header_seen:
                raise _SourceRefusal(
                    "DxfSource.geometry must satisfy its declared planar source contract; got "
                    + "DXF has duplicate HEADER sections."
                )
            header_seen = True
            header = section
        elif name == "ENTITIES":
            entities.extend(_entities(section, policy.maximum_entities - len(entities)))
        elif name in ("TABLES", "BLOCKS", "CLASSES", "OBJECTS"):
            ignored.append("section:" + name)
        else:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "Unsupported DXF section: "
                + name
            )
        index += 1
    return header, tuple(entities), tuple(ignored)


def _entities(groups: _Groups, remaining: int) -> list[tuple[str, _Groups]]:
    entities: list[tuple[str, _Groups]] = []
    index = 0
    while index < len(groups):
        code, name = groups[index]
        if code != 0:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF entity must begin with group code 0."
            )
        start = index + 1
        index = start
        while index < len(groups) and groups[index][0] != 0:
            index += 1
        if len(entities) >= remaining:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF maximum_entities exhausted before entity allocation."
            )
        entities.append((name, groups[start:index]))
    return entities


def _values(groups: _Groups, code: int) -> tuple[str, ...]:
    return tuple(value for key, value in groups if key == code)


def _number(groups: _Groups, code: int, default: float | None = None) -> float:
    values = _values(groups, code)
    if not values and default is not None:
        return default
    if len(values) != 1:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + f"DXF group {code} requires exactly one value."
        )
    value = float(values[0])
    if not math.isfinite(value):
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + f"DXF group {code} must be finite."
        )
    return value


def _integer(groups: _Groups, code: int, default: int | None = None) -> int:
    values = _values(groups, code)
    if not values and default is not None:
        return default
    if len(values) != 1:
        raise _SourceRefusal(
            f"DxfSource.group must supply one integer for code {code}; got {len(values)} values."
        )
    return int(values[0])


def _point(groups: _Groups, code: int) -> _Point:
    if _number(groups, code + 20, 0.0) != 0.0:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF profiles require zero out-of-plane coordinates."
        )
    return _number(groups, code), _number(groups, code + 10)


def _header_unit(header: _Groups) -> UnitDefinition | None:
    codes = [index for index, pair in enumerate(header) if pair == (9, "$INSUNITS")]
    if len(codes) > 1:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF has duplicate $INSUNITS declarations."
        )
    file_unit = None
    if codes:
        index = codes[0] + 1
        if index >= len(header) or header[index][0] != 70:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF $INSUNITS requires group 70."
            )
        code = int(header[index][1])
        if code != 0:
            file_unit = _UNITS.get(code)
            if file_unit is None:
                raise _SourceRefusal(
                    "DxfSource.geometry must satisfy its declared planar source contract; got "
                    + f"Unsupported DXF $INSUNITS code {code}."
                )
    return file_unit


def _unit(header: _Groups, declared: UnitDefinition | None) -> UnitDefinition:
    file_unit = _header_unit(header)
    if declared is not None and declared.dimension != LENGTH:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF caller unit must have length dimension."
        )
    if (
        declared is not None
        and file_unit is not None
        and declared.unit_id != file_unit.unit_id
    ):
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF file and caller units disagree; implicit conversion is forbidden."
        )
    unit = file_unit if file_unit is not None else declared
    if unit is None:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF drawing unit must be declared by $INSUNITS or the caller."
        )
    return unit


def _binary(value: Fraction) -> float:
    if abs(value) > Fraction(float(np.finfo(np.float64).max)):
        raise _SourceRefusal(
            "DxfSource.coefficient must fit finite binary64 storage; got rational coefficient overflow."
        )
    result = float(value)
    if not math.isfinite(result) or Fraction(result) != value:
        raise _SourceRefusal(
            "DxfSource.geometry must preserve rational source coefficients; got a coefficient not representable in binary64."
        )
    return result


def _bulge(start: _Point, end: _Point, bulge: float) -> ProfileLine | ProfileArc:
    if bulge == 0.0:
        return ProfileLine()
    x, y = (Fraction(value) for value in start)
    u, v = (Fraction(value) for value in end)
    b = Fraction(bulge)
    factor = (1 - b * b) / (4 * b)
    center = (
        _binary((x + u) / 2 - (v - y) * factor),
        _binary((y + v) / 2 + (u - x) * factor),
    )
    return ProfileArc(center, counterclockwise=bulge > 0.0)


def _polyline(groups: _Groups, label: str) -> list[_Edge]:
    if _number(groups, 38, 0.0) != 0.0:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF polyline elevation must be zero."
        )
    vertices: list[_Point] = []
    bulges: list[float] = []
    index = 0
    while index < len(groups):
        if groups[index][0] != 10:
            index += 1
            continue
        start = index
        index += 1
        while index < len(groups) and groups[index][0] != 10:
            index += 1
        bank = groups[start:index]
        vertices.append(_point(bank, 10))
        bulges.append(_number(bank, 42, 0.0))
        if _number(bank, 40, 0.0) != 0.0 or _number(bank, 41, 0.0) != 0.0:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF nonzero polyline widths are unsupported required geometry."
            )
    count = _integer(groups, 90)
    flags = _integer(groups, 70, 0)
    if count != len(vertices) or count < 2 or flags & ~129:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF polyline count or flags are invalid."
        )
    if _number(groups, 43, 0.0) != 0.0:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF nonzero constant width is unsupported required geometry."
        )
    edges = count if flags & 1 else count - 1
    return [
        (
            vertices[i],
            vertices[(i + 1) % count],
            _bulge(vertices[i], vertices[(i + 1) % count], bulges[i]),
            label,
        )
        for i in range(edges)
    ]


def _spline(groups: _Groups, label: str) -> list[_Edge]:
    xs, ys = _values(groups, 10), _values(groups, 20)
    if not xs:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF fit-point-only SPLINE refuses: interpolation would invent geometry."
        )
    degree = _integer(groups, 71)
    knots = tuple(float(value) for value in _values(groups, 40))
    weights = tuple(float(value) for value in _values(groups, 41))
    zs = _values(groups, 30)
    if len(xs) != len(ys) or (
        zs and (len(zs) != len(xs) or any(float(z) != 0.0 for z in zs))
    ):
        raise _SourceRefusal(
            "DxfSource.controls must have planar rows with matching coordinates; got nonplanar or mismatched spline controls."
        )
    if _integer(groups, 72) != len(knots) or _integer(groups, 73) != len(xs):
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF spline declared counts disagree with source coefficients."
        )
    if _integer(groups, 70, 0) & 2:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF periodic unclamped spline is unsupported by PlanarProfile."
        )
    points = tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True))
    curve = BSplineCurve(
        points, weights if weights else (1.0,) * len(points), knots, degree
    )
    return [(points[0], points[-1], curve, label)]


def _quadrant(angle: float, *, degrees: bool) -> tuple[float, float] | None:
    step = 90.0 if degrees else math.pi / 2
    quotient = angle / step
    if quotient != int(quotient):
        return None
    return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(quotient) % 4]


def _ellipse_quarters(
    center: _Point, major: _Point, ratio: float, first: int, quarters: int
) -> BSplineCurve:
    cx, cy = map(Fraction, center)
    ux, uy = map(Fraction, major)
    vx, vy = -uy * Fraction(ratio), ux * Fraction(ratio)
    axes = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
    )
    controls: list[_Point] = []
    weights: list[float] = []
    for i in range(quarters):
        a, b = axes[(first + i) % 4], axes[(first + i + 1) % 4]
        raw = (a, (a[0] + b[0], a[1] + b[1]), b)
        mapped = [
            (_binary(cx + x * ux + y * vx), _binary(cy + x * uy + y * vy)) for x, y in raw
        ]
        factor = float(2**i)
        if not controls:
            controls.append(mapped[0])
            weights.append(factor)
        controls.extend(mapped[1:])
        weights.extend((factor, 2 * factor))
    knots = (
        [0.0] * 3
        + [float(i) for i in range(1, quarters) for _ in range(2)]
        + [float(quarters)] * 3
    )
    return BSplineCurve(controls, weights, knots, 2)


def _conic_angles(
    groups: _Groups, degrees: bool
) -> tuple[float, float, _Point | None, _Point | None]:
    first = _number(groups, 50 if degrees else 41)
    last = _number(groups, 51 if degrees else 42)
    period = 360.0 if degrees else 2 * math.pi
    if not 0.0 <= first < period or not 0.0 <= last <= period:
        raise _SourceRefusal(
            "DxfSource.angles must lie in one declared period; got out-of-period conic angles."
        )
    if last <= first:
        last += period
    a, b = _quadrant(first, degrees=degrees), _quadrant(last, degrees=degrees)
    return first, last, a, b


def _conic(name: str, groups: _Groups, label: str) -> list[_Edge]:
    center = _point(groups, 10)
    if name == "CIRCLE":
        radius = _number(groups, 40)
        if radius <= 0.0:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF circle radius must be positive."
            )
        start = (_binary(Fraction(center[0]) + Fraction(radius)), center[1])
        return [(start, start, ProfileArc(center), label)]
    degrees = name == "ARC"
    first, last, a, b = _conic_angles(groups, degrees)
    if degrees:
        radius = _number(groups, 40)
        if radius <= 0.0:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF arc radius must be positive."
            )
        if a is not None and b is not None:
            points = [
                (
                    _binary(Fraction(center[0]) + Fraction(radius) * Fraction(x)),
                    _binary(Fraction(center[1]) + Fraction(radius) * Fraction(y)),
                )
                for x, y in (a, b)
            ]
            return [(points[0], points[1], ProfileArc(center), label)]
        curve = CircleCurve(center, (1.0, 0.0), (0.0, 1.0), radius)
        first, last = math.radians(first), math.radians(last)
    else:
        major = _point(groups, 11)
        ratio = _number(groups, 40)
        length = math.hypot(*major)
        if not length > 0.0 or not 0.0 < ratio <= 1.0:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF ellipse requires nonzero major axis and 0 < ratio <= 1."
            )
        if a is not None and b is not None:
            spline = _ellipse_quarters(
                center,
                major,
                ratio,
                int(first / (math.pi / 2)),
                int((last - first) / (math.pi / 2)),
            )
            points = np.asarray(spline.control_points)
            return [(tuple(points[0]), tuple(points[-1]), spline, label)]
        axis = (major[0] / length, major[1] / length)
        curve = EllipseCurve(center, axis, (-axis[1], axis[0]), length, ratio * length)
    points = np.asarray(curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)))
    return [
        (tuple(points[0]), tuple(points[1]), ProfileCurveUse(curve, first, last), label)
    ]


def _decode_entity(name: str, groups: _Groups, label: str) -> list[_Edge]:
    if _number(groups, 39, 0.0) != 0.0:
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF entity thickness is unsupported required geometry."
        )
    extrusion = (
        _number(groups, 210, 0.0),
        _number(groups, 220, 0.0),
        _number(groups, 230, 1.0),
    )
    if extrusion != (0.0, 0.0, 1.0):
        raise _SourceRefusal(
            "DxfSource.geometry must satisfy its declared planar source contract; got "
            + "DXF profile entities require the declared XY extrusion direction."
        )
    match name:
        case "LINE":
            return [(_point(groups, 10), _point(groups, 11), ProfileLine(), label)]
        case "LWPOLYLINE":
            return _polyline(groups, label)
        case "SPLINE":
            return _spline(groups, label)
        case "ARC" | "CIRCLE" | "ELLIPSE":
            return _conic(name, groups, label)
        case _:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "Unsupported required DXF entity: "
                + name
            )


def _classify_entities(
    entities: tuple[tuple[str, _Groups], ...],
    refused: list[str],
    ignored: list[str],
) -> tuple[tuple[str, _Groups, str], ...]:
    labelled = tuple(
        (name, groups, f"{name}:{handles[0] if handles else index}")
        for index, (name, groups) in enumerate(entities)
        for handles in (_values(groups, 5),)
    )
    refused.extend(label for name, _, label in labelled if name not in _ANNOTATIONS)
    ignored.extend(label for name, _, label in labelled if name in _ANNOTATIONS)
    return labelled


def _reverse(edge: _Edge) -> _Edge:
    start, end, segment, label = edge
    match segment:
        case ProfileLine():
            reversed_segment = segment
        case ProfileArc():
            reversed_segment = ProfileArc(segment.center, not segment.counterclockwise)
        case BSplineCurve():
            knots = np.asarray(segment.knots)
            reversed_segment = BSplineCurve(
                np.asarray(segment.control_points)[::-1],
                np.asarray(segment.weights)[::-1],
                knots[0] + knots[-1] - knots[::-1],
                segment.degree,
            )
        case ProfileCurveUse():
            reversed_segment = ProfileCurveUse(segment.curve, segment.last, segment.first)
        case _:
            raise TypeError(
                "DxfExport.segment must have a supported profile kind; got "
                + "Unsupported DXF profile segment."
            )
    return end, start, reversed_segment, label


def _endpoint_bound(
    edge: _Edge, vertex: _Point, endpoint: int, budget: CoordinateEnclosureBudget
) -> float:
    segment = edge[2]
    if not isinstance(segment, ProfileCurveUse):
        return 0.0 if edge[endpoint] == vertex else math.inf
    parameters = (segment.first, segment.last)
    vertices = np.asarray((vertex, vertex), dtype=np.float64)
    return endpoint_gap_bounds(segment.curve, parameters, vertices, budget=budget)[
        endpoint
    ]


def _join(
    left: _Edge,
    right: _Edge,
    limit: float,
    budget: CoordinateEnclosureBudget,
    failed_bounds: list[tuple[float, float]] | None = None,
) -> _Point | None:
    vertex = right[0] if isinstance(left[2], ProfileCurveUse) else left[1]
    first = _endpoint_bound(left, vertex, 1, budget)
    second = _endpoint_bound(right, vertex, 0, budget)
    bound = max(first, second)
    if left[1] == right[0] and bound > limit and math.isfinite(bound):
        raise _SourceRefusal(
            "DxfSource.junctions must satisfy declared loop closure; got "
            + f"DXF analytic junction gap {bound:.17g} exceeds declared maximum_junction_gap {limit:.17g}.",
            endpoints=(left[1], right[0]),
            gaps=((first, second),),
        )
    if failed_bounds is not None and math.isfinite(bound) and bound > limit:
        failed_bounds.append((first, second))
    return vertex if bound <= limit else None


def _continuations(
    left: _Edge, remaining: list[_Edge], limit: float, budget: CoordinateEnclosureBudget
) -> tuple[list[tuple[int, _Edge, _Point]], list[tuple[float, float]]]:
    budget.reserve(2 * len(remaining))
    candidates: list[tuple[int, _Edge, _Point]] = []
    failed_bounds: list[tuple[float, float]] = []
    for index, edge in enumerate(remaining):
        for oriented in (edge, _reverse(edge)):
            vertex = _join(left, oriented, limit, budget, failed_bounds)
            if vertex is not None:
                candidates.append((index, oriented, vertex))
    return candidates, failed_bounds


def _open_endpoint_context(
    edges: list[_Edge], budget: CoordinateEnclosureBudget
) -> tuple[_Point, ...]:
    budget.reserve(2 * len(edges), 128 * len(edges))
    counts = Counter(point for edge in edges for point in edge[:2])
    unmatched = tuple(point for point, count in counts.items() if count == 1)
    return unmatched if unmatched else tuple(counts)


def _loops(
    edges: list[_Edge], policy: DxfImportPolicy, budget: CoordinateEnclosureBudget
) -> tuple[ProfileLoop, ...]:
    remaining = list(edges)
    loops: list[ProfileLoop] = []
    while remaining:
        chain = [remaining.pop(0)]
        joins: list[_Point] = []
        while True:
            closing = _join(chain[-1], chain[0], policy.maximum_junction_gap, budget)
            if closing is not None:
                vertices = (closing, *joins)
                loops.append(
                    ProfileLoop(tuple(vertices), tuple(edge[2] for edge in chain))
                )
                break
            candidates, failed_bounds = _continuations(
                chain[-1], remaining, policy.maximum_junction_gap, budget
            )
            if len(candidates) != 1:
                endpoints = _open_endpoint_context(chain + remaining, budget)
                raise _SourceRefusal(
                    "DxfSource.junctions must satisfy declared loop closure; got open or ambiguous endpoints and refused source-coordinate or declared-gap joining.",
                    endpoints=endpoints,
                    gaps=tuple(failed_bounds),
                )
            index, edge, vertex = candidates[0]
            remaining.pop(index)
            joins.append(vertex)
            chain.append(edge)
    return tuple(loops)


def _profile(
    loops: tuple[ProfileLoop, ...], budget: CoordinateEnclosureBudget
) -> PlanarProfile:
    rows = tuple(
        tuple(
            endpoint_gap_bounds(
                segment.curve,
                (segment.first, segment.last),
                np.asarray(
                    (loop.vertices[i], loop.vertices[(i + 1) % len(loop.vertices)]),
                    dtype=np.float64,
                ),
                budget=budget,
            )
            if isinstance(segment, ProfileCurveUse)
            else (0.0, 0.0)
            for i, segment in enumerate(loop.segments)
        )
        for loop in loops
    )
    gap_rows = rows if any(max(row) > 0.0 for loop in rows for row in loop) else None
    failures: list[str] = []
    for index, outer in enumerate(loops):
        ordered = (
            None
            if gap_rows is None
            else (gap_rows[index], *gap_rows[:index], *gap_rows[index + 1 :])
        )
        try:
            return PlanarProfile(
                ProfilePlane(),
                outer,
                loops[:index] + loops[index + 1 :],
                endpoint_gap_bounds=ordered,
                coordinate_budget=budget,
            )
        except ValueError as error:
            failures.append(str(error))
    raise _SourceRefusal(
        "DxfSource.geometry must satisfy its declared planar source contract; got "
        + "DXF loops do not form one certified planar region: "
        + "; ".join(failures)
    )


def _admit_bytes(
    data: bytes, policy: DxfImportPolicy, budget: CoordinateEnclosureBudget
) -> tuple[bytes, str]:
    budget.reserve(min(len(data), policy.maximum_bytes))
    limits = ResourceLimits(
        max_bytes=policy.maximum_bytes,
        max_depth=64,
        max_nodes=policy.maximum_entities,
        max_attributes=policy.maximum_groups,
        max_losses=0,
    )
    try:
        resource = bounded_resource_from_bytes(data, limits=limits)
    except ResourceReadError as error:
        raise _SourceRefusal(
            f"DxfSource.maximum_bytes must admit the source; got maximum_bytes {policy.maximum_bytes}: {error}"
        ) from error
    return resource.data, resource.manifest.content_sha256


@checked
def decode_dxf_bytes(
    data: bytes,
    policy: DxfImportPolicy,
    /,
    *,
    source_length_unit: UnitDefinition | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> DxfImportResult:
    """Decode bounded ASCII DXF to one source-faithful planar region."""
    if coordinate_budget is None:
        memory = 16 * policy.maximum_bytes + 512 * policy.maximum_groups
        work = 64 * policy.maximum_groups + 2 * policy.maximum_bytes
        with HostBatchTicket(work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            with budget.activate():
                return decode_dxf_bytes(
                    data,
                    policy,
                    source_length_unit=source_length_unit,
                    coordinate_budget=budget,
                )
    digest = "unadmitted"
    consumed: list[str] = []
    refused: list[str] = []
    ignored: list[str] = []
    unit = None
    try:
        data, digest = _admit_bytes(data, policy, coordinate_budget)
        header, entities, sections = _parse(data, policy, coordinate_budget)
        ignored.extend(sections)
        labelled = _classify_entities(entities, refused, ignored)
        unit = _unit(header, source_length_unit)
        edges: list[_Edge] = []
        for name, groups, label in labelled:
            if name in _ANNOTATIONS:
                continue
            edges.extend(_decode_entity(name, groups, label))
            refused.remove(label)
            consumed.append(label)
        if not edges:
            raise _SourceRefusal(
                "DxfSource.geometry must satisfy its declared planar source contract; got "
                + "DXF contains no supported profile geometry."
            )
        loops = _loops(edges, policy, coordinate_budget)
        profile = _profile(loops, coordinate_budget)
    except ValueError as error:
        endpoints = error.endpoints if isinstance(error, _SourceRefusal) else ()
        report = _report(
            digest,
            consumed=consumed,
            refused=refused if refused else ("source-contract",),
            ignored=ignored,
            unit=unit,
            failure=str(error),
            gaps=error.gaps if isinstance(error, _SourceRefusal) else (),
            declared_junction_gap=policy.maximum_junction_gap,
        )
        raise DxfImportError(str(error), report, open_endpoints=endpoints) from error
    gaps = tuple(row for loop in profile.endpoint_gap_bounds or () for row in loop)
    report = _report(
        digest,
        consumed=consumed,
        refused=(),
        ignored=ignored,
        unit=unit,
        gaps=gaps,
        declared_junction_gap=policy.maximum_junction_gap,
    )
    return DxfImportResult(profile, unit, report)


@checked
def read_dxf(
    path: str | os.PathLike[str],
    policy: DxfImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
    source_length_unit: UnitDefinition | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> DxfImportResult:
    """Descriptor-read one bounded regular DXF file beneath a trusted root."""
    if coordinate_budget is None:
        memory = 18 * policy.maximum_bytes + 512 * policy.maximum_groups
        work = 64 * policy.maximum_groups + 3 * policy.maximum_bytes
        with HostBatchTicket(work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            with budget.activate():
                return read_dxf(
                    path,
                    policy,
                    trusted_root=trusted_root,
                    source_length_unit=source_length_unit,
                    coordinate_budget=budget,
                )
    coordinate_budget.reserve(policy.maximum_bytes, policy.maximum_bytes)
    limits = ResourceLimits(
        max_bytes=policy.maximum_bytes,
        max_depth=64,
        max_nodes=policy.maximum_entities,
        max_attributes=policy.maximum_groups,
        max_losses=0,
    )
    try:
        resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    except ResourceReadError as error:
        report = _report(
            "unreadable",
            consumed=(),
            refused=("file",),
            ignored=(),
            unit=source_length_unit,
            failure=str(error),
        )
        raise DxfImportError(str(error), report) from error
    return decode_dxf_bytes(
        resource.data,
        policy,
        source_length_unit=source_length_unit,
        coordinate_budget=coordinate_budget,
    )


def _pairs(groups: Sequence[tuple[int, str | float | int]]) -> str:
    return "".join(
        f"{code}\n{value if isinstance(value, str) else format(value, '.17g')}\n"
        for code, value in groups
    )


def _write_spline(segment: BSplineCurve) -> str:
    controls = np.asarray(segment.control_points)
    knots, weights = np.asarray(segment.knots), np.asarray(segment.weights)
    groups: list[tuple[int, str | float | int]] = [
        (0, "SPLINE"),
        (70, 4),
        (71, segment.degree),
        (72, knots.size),
        (73, controls.shape[0]),
        (74, 0),
    ]
    groups.extend((40, float(value)) for value in knots)
    groups.extend((41, float(value)) for value in weights)
    for x, y in controls:
        groups.extend(((10, float(x)), (20, float(y)), (30, 0)))
    return _pairs(groups)


def _rational_sqrt(value: Fraction) -> Fraction:
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        raise ValueError(
            "DxfExport.coefficient must have a rational square root; got an unrepresentable conic coefficient."
        )
    return Fraction(numerator, denominator)


def _write_arc(start: _Point, end: _Point, segment: ProfileArc) -> str:
    if start == end:
        squared = sum(
            (
                (Fraction(a) - Fraction(b)) ** 2
                for a, b in zip(start, segment.center, strict=True)
            ),
            Fraction(),
        )
        radius = _binary(_rational_sqrt(squared))
        return _pairs(
            (
                (0, "CIRCLE"),
                (10, segment.center[0]),
                (20, segment.center[1]),
                (40, radius),
            )
        )
    cx, cy = map(Fraction, segment.center)
    ax, ay = Fraction(start[0]) - cx, Fraction(start[1]) - cy
    bx, by = Fraction(end[0]) - cx, Fraction(end[1]) - cy
    radius_squared = ax * ax + ay * ay
    if (ax == 0 or ay == 0) and (bx == 0 or by == 0):
        radius = _binary(_rational_sqrt(radius_squared))
        first = math.degrees(math.atan2(float(ay), float(ax))) % 360
        last = math.degrees(math.atan2(float(by), float(bx))) % 360
        if not segment.counterclockwise:
            first, last = last, first
        return _pairs(
            (
                (0, "ARC"),
                (10, segment.center[0]),
                (20, segment.center[1]),
                (40, radius),
                (50, first),
                (51, last),
            )
        )
    dx, dy = bx - ax, by - ay
    chord_squared = dx * dx + dy * dy
    factor = ((ax + bx) * dy - (ay + by) * dx) / (2 * chord_squared)
    root = _rational_sqrt(4 * factor * factor + 1)
    bulge = _binary(-2 * factor + (root if segment.counterclockwise else -root))
    candidate = _bulge(start, end, bulge)
    if not isinstance(candidate, ProfileArc) or candidate.center != segment.center:
        raise ValueError(
            "DxfExport.geometry must satisfy representability and resource limits; got "
            + "DXF bulge cannot exactly reproduce the source center; export refuses."
        )
    return _pairs(
        (
            (0, "LWPOLYLINE"),
            (90, 2),
            (70, 0),
            (10, start[0]),
            (20, start[1]),
            (42, bulge),
            (10, end[0]),
            (20, end[1]),
        )
    )


def _export_preflight(
    profile: PlanarProfile,
    coordinate_unit: UnitDefinition,
    policy: DxfImportPolicy,
    coordinate_budget: CoordinateEnclosureBudget,
) -> int:
    groups_upper = 10
    entities = sum(len(loop.segments) for loop in profile.loops)
    if entities > policy.maximum_entities:
        raise ValueError(
            "DxfExport.geometry must satisfy representability and resource limits; got "
            + "DXF export maximum_entities exhausted before allocation."
        )
    for loop in profile.loops:
        for segment in loop.segments:
            groups_upper += (
                6 + segment.knots.size + 4 * segment.weights.size
                if isinstance(segment, BSplineCurve)
                else 16
            )
    if groups_upper > policy.maximum_groups:
        raise ValueError(
            "DxfExport.geometry must satisfy representability and resource limits; got "
            + "DXF export maximum_groups exhausted before allocation."
        )
    coordinate_budget.reserve(groups_upper, groups_upper * 512)
    unit_codes = [
        code for code, unit in _UNITS.items() if unit.unit_id == coordinate_unit.unit_id
    ]
    if not unit_codes:
        raise ValueError(
            "DxfExport.geometry must satisfy representability and resource limits; got "
            + "DXF export requires a supported explicit drawing unit."
        )
    if profile.plane != ProfilePlane():
        raise ValueError(
            "DxfExport.plane must be the XY frame; got a translated or rotated source frame."
        )
    return unit_codes[0]


@checked
def encode_dxf(
    profile: PlanarProfile,
    /,
    *,
    coordinate_unit: UnitDefinition,
    policy: DxfImportPolicy,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> bytes:
    """Encode source controls and analytic profiles without refitting."""
    if coordinate_budget is None:
        memory = 16 * policy.maximum_bytes + 512 * policy.maximum_groups
        work = 64 * policy.maximum_groups + policy.maximum_bytes
        with HostBatchTicket(work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            with budget.activate():
                return encode_dxf(
                    profile,
                    coordinate_unit=coordinate_unit,
                    policy=policy,
                    coordinate_budget=budget,
                )
    unit_code = _export_preflight(profile, coordinate_unit, policy, coordinate_budget)
    chunks = [
        _pairs(
            (
                (0, "SECTION"),
                (2, "HEADER"),
                (9, "$INSUNITS"),
                (70, unit_code),
                (0, "ENDSEC"),
                (0, "SECTION"),
                (2, "ENTITIES"),
            )
        )
    ]
    count = 0
    for loop in profile.loops:
        for index, segment in enumerate(loop.segments):
            start, end = (
                loop.vertices[index],
                loop.vertices[(index + 1) % len(loop.vertices)],
            )
            match segment:
                case ProfileLine():
                    chunk = _pairs(
                        (
                            (0, "LINE"),
                            (10, start[0]),
                            (20, start[1]),
                            (11, end[0]),
                            (21, end[1]),
                        )
                    )
                case ProfileArc():
                    chunk = _write_arc(start, end, segment)
                case BSplineCurve():
                    chunk = _write_spline(segment)
                case ProfileCurveUse():
                    chunk = _write_conic_use(segment, coordinate_budget)
                case _:
                    raise TypeError(
                        "DxfExport.segment must have a supported profile kind; got "
                        + "DXF export requires a supported profile segment."
                    )
            count += 1
            if count > policy.maximum_entities:
                raise ValueError(
                    "DxfExport.geometry must satisfy representability and resource limits; got "
                    + "DXF export maximum_entities exhausted."
                )
            chunks.append(chunk)
    chunks.append(_pairs(((0, "ENDSEC"), (0, "EOF"))))
    data = "".join(chunks).encode("ascii")
    if len(data) > policy.maximum_bytes or data.count(b"\n") // 2 > policy.maximum_groups:
        raise ValueError(
            "DxfExport.geometry must satisfy representability and resource limits; got "
            + "DXF export exceeds its declared size budget."
        )
    return data


def _neighbor_values(value: float) -> tuple[float, float, float]:
    return value, float(np.nextafter(value, -np.inf)), float(np.nextafter(value, np.inf))


def _ellipse_encoding(
    curve: EllipseCurve, budget: CoordinateEnclosureBudget
) -> tuple[_Point, float]:
    """Invert factorized ellipse coefficients only after binary identity replay."""
    budget.reserve(27, 512)
    radius, minor = float(curve.first_radius), float(curve.second_radius)
    axes = tuple(float(value) for value in np.asarray(curve.first_axis))
    if tuple(float(value) for value in np.asarray(curve.second_axis)) != (
        -axes[1],
        axes[0],
    ):
        raise ValueError(
            "DxfExport.axes must encode a perpendicular ellipse frame; got incompatible axes."
        )
    # Adjacent dyadic values can invert a rounded axis-radius product.
    # Finite candidates are admitted by coefficient equality, never proximity.
    for x, y in product(
        _neighbor_values(radius * axes[0]), _neighbor_values(radius * axes[1])
    ):
        length = math.hypot(x, y)
        if length != radius or (x / length, y / length) != axes:
            continue
        for ratio in _neighbor_values(minor / radius):
            if 0.0 < ratio <= 1.0 and ratio * radius == minor:
                return (x, y), ratio
    raise ValueError(
        "DxfExport.ellipse must retain its coefficient identity; got a nonrepresentable DXF axis or ratio."
    )


def _write_conic_use(segment: ProfileCurveUse, budget: CoordinateEnclosureBudget) -> str:
    curve = segment.curve
    first, last = segment.first, segment.last
    if last < first:
        first, last = last, first
    center = np.asarray(curve.center)
    groups: list[tuple[int, str | float | int]]
    if isinstance(curve, CircleCurve):
        groups = [
            (0, "ARC"),
            (10, float(center[0])),
            (20, float(center[1])),
            (40, float(curve.radius)),
            (50, math.degrees(first) % 360),
            (51, math.degrees(last) % 360),
        ]
    else:
        major, ratio = _ellipse_encoding(curve, budget)
        groups = [
            (0, "ELLIPSE"),
            (10, float(center[0])),
            (20, float(center[1])),
            (11, float(major[0])),
            (21, float(major[1])),
            (40, ratio),
            (41, first % (2 * math.pi)),
            (42, last % (2 * math.pi)),
        ]
    entity_name = groups[0][1]
    if not isinstance(entity_name, str):
        raise TypeError(
            "DxfExport.entity must have a DXF entity name; got a numerical name."
        )
    parsed = _conic(
        entity_name, tuple((code, str(value)) for code, value in groups[1:]), "export"
    )[0][2]
    if (
        not isinstance(parsed, ProfileCurveUse)
        or parsed.first != first
        or parsed.last != last
        or carrier_payload(parsed.curve) != carrier_payload(curve)
    ):
        raise ValueError(
            "DxfExport.conic must retain its source parameters and coefficient identity; got a nonrepresentable DXF angle or axis encoding."
        )
    return _pairs(groups)


@checked
def write_dxf(
    profile: PlanarProfile,
    path: str | os.PathLike[str],
    /,
    *,
    coordinate_unit: UnitDefinition,
    policy: DxfImportPolicy,
    mode: PublicationMode = "exclusive",
) -> DxfExportResult:
    """Atomically publish one bounded profile with durable byte evidence."""
    data = encode_dxf(profile, coordinate_unit=coordinate_unit, policy=policy)
    receipt = publish_bytes(path, data, maximum_bytes=policy.maximum_bytes, mode=mode)
    report = _report(
        receipt.content_sha256,
        consumed=("profile",),
        refused=(),
        ignored=(),
        unit=coordinate_unit,
        direction="export",
    )
    return DxfExportResult(receipt, report)

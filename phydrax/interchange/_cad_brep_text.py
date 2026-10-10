#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native reader/writer of the external OCCT BRep text format (``.brep``/``.brp``).

This implements the ``CASCADE Topology`` V1/V2/V3 text profiles written by
``BRepTools::Write`` (optionally headed by ``DBRep_DrawableShape``) as a file
format only; no OCCT algorithm is used. The external format identity is kept:
the profile names its OCCT topology revision.

Coverage (read):

* Locations: elementary 3x4 matrices (proper rigid only) and powered products.
* Curve2ds/Curves: line, circle, ellipse, parabola, hyperbola, exact right-normal
  offsets, Bezier, rational B-splines (periodic splines unrolled exactly over
  one finite period) and trimmed curves lowered to their basis carrier with
  enforced trim ranges. Operation trees retain the original basis parameter.
* Surfaces: plane, cylinder, cone (OCCT's slant parameter rescaled exactly to
  the native axial parameter), sphere, torus, linear extrusion, revolution,
  Bezier, rational B-spline and rectangular trimmed surfaces. Periodic spline
  axes are unrolled exactly; offsets retain their exact normal-operation trees.
  Nested operation bases are validated as curves or surfaces before construction;
  export preparation must likewise produce concrete curve and p-curve carriers.
* Polygon3D, PolygonOnTriangulations and Triangulations are derived data:
  parsed completely (so malformed data is refused) and ignored with a declared
  ``dropped`` loss.
* TShapes Ve, Ed, Wi, Fa, Sh, So, CS, Co with FORWARD/REVERSED orientations
  and locations, shared face incidences with per-shell senses, and exact empty
  compounds. Actual compound membership and nested child containers are explicit;
  placements or path prefixes never manufacture assembly relationships.
  Free edges and vertices retain exact geometry and empty face incidence.
  INTERNAL/EXTERNAL uses, free wires, reversed compound/free-entity uses,
  non-same-parameter edges and unsupported boundaryless surfaces refuse.

The format carries no length unit or scientific occurrence/container names.
Readers declare units; writers declare path-name normalization losses. Exact
carriers are exported by default, with plane/line/cone/extrusion parameter maps
applied to p-curves. An explicit intersection-approximation policy substitutes
coupled splines and reports sampled residuals, never continuous tube guarantees.
Authoritative trims must match the exported coedges; unsupported required
geometry refuses rather than disappearing. A forest of independent solid roots
cannot be encoded by inventing membership under this single-root file format.

Import/export provenance maps native inventory indices to actual ``TShapes#N``
declarations. Qualified import labels retain nonzero source location-reference
chains; occurrence/root entries retain the raw ``orientation+number location``
reference syntax. Synthesized natural-boundary entities have no source
declaration and no identity mapping; coordinates and geometry digests never
stand in for unavailable external entity references.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import cos, isfinite, pi
from typing import Literal, TypeAlias

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
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from ..typing import parse
from ..units import UnitDefinition
from ._cad import (
    _export_loss_waivers,
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
    recover_pcurve,
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


BRepTextVersion: TypeAlias = Literal["V1", "V2", "V3"]

_HEADERS: dict[str, BRepTextVersion] = {
    "CASCADE Topology V1, (c) Matra-Datavision": "V1",
    "CASCADE Topology V2, (c) Matra-Datavision": "V2",
    "CASCADE Topology V3, (c) Open Cascade": "V3",
}
_DRAWABLE = "DBRep_DrawableShape"
_FORMAT = "occt-brep-text"
_KINDS = ("Ve", "Ed", "Wi", "Fa", "Sh", "So", "CS", "Co")
_NATIVE_TSHAPE_KINDS = {
    "vertex": "Ve",
    "edge": "Ed",
    "face": "Fa",
    "shell": "Sh",
    "solid": "So",
}
_REGULARITIES = ("C0", "G1", "C1", "G2", "C2", "C3", "CN")
_INTEGER = re.compile(rb"\s*([+-]?\d+)")
_REAL = re.compile(rb"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_WORD = re.compile(rb"\s*(\S+)")
_TWO_PI = 2.0 * pi


def _profile(version: BRepTextVersion, /) -> AdapterFormatProfile:
    return AdapterFormatProfile(
        _FORMAT, qualifiers={"profile": f"cascade-topology-{version}"}
    )


# ------------------------------------------------------------------ scanning


class _Scanner:
    """Bounded token scanner over the exact file bytes (OCCT stream semantics)."""

    def __init__(self, data: bytes, policy: CadImportPolicy) -> None:
        self.data = data
        self.position = 0
        self.attributes = 0
        self.nodes = 0
        self.limits = policy.limits
        self.depth = 0
        self.label = "header"

    def _fail(self, message: str) -> Exception:
        return refuse("malformed", self.label, f"{message} at byte {self.position}.")

    def _match(self, pattern: re.Pattern[bytes], what: str) -> bytes:
        found = pattern.match(self.data, self.position)
        if found is None:
            raise self._fail(f"Expected {what}")
        self.position = found.end()
        self.attributes += 1
        if self.attributes > self.limits.max_attributes:
            raise refuse("limit", self.label, "The file exceeds the attribute limit.")
        return found.group(1)

    def integer(self) -> int:
        return int(self._match(_INTEGER, "an integer"))

    def real(self) -> float:
        value = float(self._match(_REAL, "a real number"))
        if not isfinite(value):
            raise self._fail("A real number is not finite")
        return value

    def reals(self, count: int) -> np.ndarray:
        return np.asarray([self.real() for _ in range(count)], dtype=np.float64)

    def word(self) -> str:
        return self._match(_WORD, "a token").decode("ascii", "replace")

    def keyword(self, expected: str) -> None:
        found = self.word()
        if found != expected:
            raise self._fail(f"Expected {expected!r}, found {found!r}")

    def count(self, name: str) -> int:
        self.keyword(name)
        value = self.integer()
        if value < 0:
            raise self._fail(f"Negative {name} count")
        self.nodes += value
        if self.nodes > self.limits.max_nodes:
            raise refuse("limit", name, "The file exceeds the entity limit.")
        return value

    def boolean(self) -> bool:
        value = self.integer()
        if value not in (0, 1):
            raise self._fail("Expected a 0/1 flag")
        return value == 1

    def regularity(self) -> str:
        for token in _REGULARITIES:
            end = self.position
            while end < len(self.data) and self.data[end : end + 1].isspace():
                end += 1
            if self.data.startswith(token.encode(), end):
                self.position = end + len(token)
                return token
        raise self._fail("Expected a regularity (C0..CN)")

    def line(self) -> str:
        end = self.data.find(b"\n", self.position)
        end = len(self.data) if end < 0 else end
        text = self.data[self.position : end].decode("ascii", "replace").strip()
        self.position = end + 1
        return text

    def peek_word(self) -> str:
        found = _WORD.match(self.data, self.position)
        return "" if found is None else found.group(1).decode("ascii", "replace")


# ---------------------------------------------------------------- geometry


@dataclass(frozen=True, slots=True)
class _Carrier:
    """Decoded curve/surface: a native carrier or a deferred refusal.

    ``trim`` bounds admissible curve parameters. Surface and curve parameter
    factors preserve file p-curves under length-unit conversion; ``surface_trim``
    retains rectangular surface trims instead of silently dropping them.
    """

    label: str
    value: AbstractCurve | AbstractSurfacePatch | None
    refusal: str | None = None
    trim: tuple[float, float] | None = None
    v_factor: float = 1.0
    parameter_factor: float = 1.0
    u_factor: float = 1.0
    surface_trim: tuple[float, float, float, float] | None = None


def _unsupported(label: str, reason: str) -> _Carrier:
    return _Carrier(label, None, reason)


def _expanded_knots(values: np.ndarray, multiplicities: list[int]) -> np.ndarray:
    return np.repeat(values, multiplicities)


class _GeometryReader:
    def __init__(self, scanner: _Scanner, factor: float) -> None:
        self.scanner = scanner
        self.factor = factor

    def _point(self, dimension: int) -> np.ndarray:
        return self.scanner.reals(dimension) * self.factor

    def _direction(self, dimension: int) -> np.ndarray:
        vector = self.scanner.reals(dimension)
        norm = float(np.linalg.norm(vector))
        if not norm > 0.0:
            raise self.scanner._fail("A direction is zero")
        # OCCT normalizes directions; values already unit within rounding keep
        # their exact bits so exported native carriers re-read identically.
        return vector if _unit_speed(vector) == 1.0 else vector / norm

    def _spline_poles(
        self, count: int, dimension: int, rational: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        points, weights = [], []
        for _ in range(count):
            points.append(self._point(dimension))
            weights.append(self.scanner.real() if rational else 1.0)
            if weights[-1] <= 0.0:
                raise self.scanner._fail("A rational spline weight must be positive")
        return np.asarray(points), np.asarray(weights)

    def _knots(self, count: int) -> tuple[np.ndarray, list[int]]:
        values, multiplicities = [], []
        for _ in range(count):
            values.append(self.scanner.real())
            multiplicities.append(self.scanner.integer())
        if any(value < 1 for value in multiplicities):
            raise self.scanner._fail("A knot multiplicity is not positive")
        self._budget(sum(multiplicities), self.scanner.label)
        return np.asarray(values), multiplicities

    def _spline_axis(
        self,
        values: np.ndarray,
        multiplicities: list[int],
        degree: int,
        poles: int,
        periodic: bool,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OCCT's exact finite-period knot and cyclic-pole expansion.

        BSplCLib::KnotSequence/Unperiodize append cyclic poles without rotating
        their origin. No fitting, sampling or OCCT runtime algorithm is used.
        https://github.com/Open-Cascade-SAS/OCCT/blob/V7_8_1/src/BSplCLib/BSplCLib.cxx
        """
        if np.any(np.diff(values) <= 0.0):
            raise self.scanner._fail("Distinct spline knots must strictly increase")
        if not periodic:
            if sum(multiplicities) != poles + degree + 1:
                raise self.scanner._fail(
                    "Spline multiplicities disagree with its pole count"
                )
            return _expanded_knots(values, multiplicities), np.arange(poles)
        seam = multiplicities[0]
        if (
            seam != multiplicities[-1]
            or max(multiplicities) > degree
            or sum(multiplicities[:-1]) != poles
        ):
            raise self.scanner._fail(
                "Inconsistent periodic spline multiplicities or pole count"
            )
        padding = degree + 1 - seam
        total_knots = sum(multiplicities) + 2 * padding
        expanded_poles = total_knots - degree - 1
        self._budget(total_knots + expanded_poles, label)
        base = _expanded_knots(values, multiplicities)
        period = float(values[-1] - values[0])
        before = base[:-seam][-padding:] - period
        after = base[seam:][:padding] + period
        expanded = np.concatenate((before, base, after))
        if not np.all(np.isfinite(expanded)):
            raise self.scanner._fail(
                "Periodic knot expansion overflows finite parameters"
            )
        return expanded, np.arange(expanded_poles) % poles

    def curve(self, dimension: int, label: str, depth: int) -> _Carrier:
        scanner = self.scanner
        scanner.depth = max(scanner.depth, depth + 1)
        if depth + 1 > scanner.limits.max_depth:
            raise refuse("limit", label, "Nested basis curves exceed the depth limit.")
        kind = scanner.integer()
        match kind:
            case 1:
                origin = self._point(dimension)
                return _Carrier(
                    label,
                    LineCurve(origin, self._direction(dimension)),
                    parameter_factor=self.factor,
                )
            case 2 | 3 | 4 | 5:
                center = self._point(dimension)
                if dimension == 3:
                    normal = self._direction(3)
                first, second = self._direction(dimension), self._direction(dimension)
                if (
                    dimension == 3
                    and np.max(np.abs(np.cross(first, second) - normal)) > 1.0e-9
                ):
                    raise scanner._fail("A conic frame has inconsistent handedness")
                if kind == 2:
                    radius = scanner.real() * self.factor
                    return _Carrier(label, CircleCurve(center, first, second, radius))
                if kind == 4:
                    focal = scanner.real() * self.factor
                    return _Carrier(
                        label,
                        ParabolaCurve(center, first, second, focal),
                        parameter_factor=self.factor,
                    )
                major, minor = scanner.real() * self.factor, scanner.real() * self.factor
                conic = (
                    EllipseCurve(center, first, second, major, minor)
                    if kind == 3
                    else HyperbolaCurve(center, first, second, major, minor)
                )
                return _Carrier(label, conic)
            case 6:
                rational = scanner.boolean()
                degree = scanner.integer()
                if not 1 <= degree <= 25:
                    raise scanner._fail("Bezier degree out of range")
                points, weights = self._spline_poles(degree + 1, dimension, rational)
                return _Carrier(label, BSplineCurve.bezier(points, weights))
            case 7:
                rational, periodic = scanner.boolean(), scanner.boolean()
                degree, poles, knots = (
                    scanner.integer(),
                    scanner.integer(),
                    scanner.integer(),
                )
                if not (1 <= degree <= 25 and poles > degree and knots >= 2):
                    raise scanner._fail("Inconsistent B-spline curve header")
                self._budget(poles * (dimension + 1) + 2 * knots, label)
                points, weights = self._spline_poles(poles, dimension, rational)
                values, multiplicities = self._knots(knots)
                expanded, indices = self._spline_axis(
                    values,
                    multiplicities,
                    degree,
                    poles,
                    periodic,
                    label,
                )
                self._budget(indices.size * (dimension + 1) + expanded.size, label)
                return _Carrier(
                    label,
                    BSplineCurve(points[indices], weights[indices], expanded, degree),
                )
            case 8:
                first, last = scanner.real(), scanner.real()
                basis = self.curve(dimension, label, depth + 1)
                bounds = (first * basis.parameter_factor, last * basis.parameter_factor)
                if not bounds[1] > bounds[0]:
                    raise scanner._fail("A trimmed curve range must increase")
                if basis.trim is not None:
                    bounds = (
                        max(bounds[0], basis.trim[0]),
                        min(bounds[1], basis.trim[1]),
                    )
                    if not bounds[1] > bounds[0]:
                        raise scanner._fail("Nested curve trims do not overlap")
                return _Carrier(
                    label,
                    basis.value,
                    basis.refusal,
                    bounds,
                    parameter_factor=basis.parameter_factor,
                )
            case 9:
                distance = scanner.real() * self.factor
                direction = self._direction(3) if dimension == 3 else None
                basis = self.curve(dimension, label, depth + 1)
                if basis.refusal is not None:
                    return _Carrier(
                        label,
                        None,
                        basis.refusal,
                        basis.trim,
                        parameter_factor=basis.parameter_factor,
                    )
                if not isinstance(basis.value, AbstractCurve):
                    raise scanner._fail("An offset curve basis must be a curve")
                return _Carrier(
                    label,
                    OffsetCurve(basis.value, distance, direction),
                    trim=basis.trim,
                    parameter_factor=basis.parameter_factor,
                )
            case _:
                raise scanner._fail(f"Unknown curve type {kind}")

    def _budget(self, count: int, label: str) -> None:
        if count > self.scanner.limits.max_attributes:
            raise refuse("limit", label, "A spline exceeds the attribute limit.")

    def surface(self, label: str, depth: int) -> _Carrier:
        scanner = self.scanner
        scanner.depth = max(scanner.depth, depth + 1)
        if depth + 1 > scanner.limits.max_depth:
            raise refuse("limit", label, "Nested basis surfaces exceed the depth limit.")
        kind = scanner.integer()
        if 1 <= kind <= 5:
            origin = self._point(3)
            axis, first, second = (self._direction(3) for _ in range(3))
            cross = np.cross(first, second)
            if min(np.max(np.abs(cross - axis)), np.max(np.abs(cross + axis))) > 1.0e-9:
                raise scanner._fail("A surface frame has inconsistent orthogonality")
        match kind:
            case 1:
                return _Carrier(
                    label,
                    PlanePatch(origin, first, second),
                    u_factor=self.factor,
                    v_factor=self.factor,
                )
            case 2:
                radius = scanner.real() * self.factor
                return _Carrier(
                    label,
                    CylinderPatch(origin, first, second, axis, radius),
                    v_factor=self.factor,
                )
            case 3:
                radius, angle = scanner.real() * self.factor, scanner.real()
                return _Carrier(
                    label,
                    ConePatch(origin, first, second, axis, radius, angle),
                    v_factor=self.factor * cos(angle),
                )
            case 4:
                radius = scanner.real() * self.factor
                return _Carrier(label, SpherePatch(origin, first, second, axis, radius))
            case 5:
                major, minor = scanner.real() * self.factor, scanner.real() * self.factor
                return _Carrier(
                    label, TorusPatch(origin, first, second, axis, major, minor)
                )
            case 6:
                direction = self._direction(3)
                basis = self.curve(3, label, depth + 1)
                if basis.value is None:
                    return _unsupported(label, f"Extrusion basis: {basis.refusal}")
                if not isinstance(basis.value, AbstractCurve):
                    raise scanner._fail("An extrusion basis must be a curve")
                return _Carrier(
                    label,
                    ExtrusionSurface(basis.value, direction),
                    u_factor=basis.parameter_factor,
                    v_factor=self.factor,
                    surface_trim=None
                    if basis.trim is None
                    else (
                        basis.trim[0],
                        basis.trim[1],
                        -np.inf,
                        np.inf,
                    ),
                )
            case 7:
                origin, direction = self._point(3), self._direction(3)
                basis = self.curve(3, label, depth + 1)
                if basis.value is None:
                    return _unsupported(label, f"Revolution basis: {basis.refusal}")
                if not isinstance(basis.value, AbstractCurve):
                    raise scanner._fail("A revolution basis must be a curve")
                return _Carrier(
                    label,
                    RevolutionSurface(basis.value, origin, direction),
                    v_factor=basis.parameter_factor,
                    surface_trim=None
                    if basis.trim is None
                    else (
                        -np.inf,
                        np.inf,
                        basis.trim[0],
                        basis.trim[1],
                    ),
                )
            case 8:
                u_rational, v_rational = scanner.boolean(), scanner.boolean()
                u_degree, v_degree = scanner.integer(), scanner.integer()
                if not (1 <= u_degree <= 25 and 1 <= v_degree <= 25):
                    raise scanner._fail("Bezier surface degree out of range")
                count = (u_degree + 1) * (v_degree + 1)
                points, weights = self._spline_poles(count, 3, u_rational or v_rational)
                u_knots = np.repeat((0.0, 1.0), u_degree + 1)
                v_knots = np.repeat((0.0, 1.0), v_degree + 1)
                return _Carrier(
                    label,
                    BSplineSurfacePatch(
                        points.reshape(u_degree + 1, v_degree + 1, 3),
                        weights.reshape(u_degree + 1, v_degree + 1),
                        u_knots,
                        v_knots,
                        u_degree,
                        v_degree,
                    ),
                )
            case 9:
                u_rational, v_rational = scanner.boolean(), scanner.boolean()
                u_periodic, v_periodic = scanner.boolean(), scanner.boolean()
                u_degree, v_degree = scanner.integer(), scanner.integer()
                u_poles, v_poles = scanner.integer(), scanner.integer()
                u_count, v_count = scanner.integer(), scanner.integer()
                if not (
                    1 <= u_degree <= 25
                    and 1 <= v_degree <= 25
                    and u_poles > u_degree
                    and v_poles > v_degree
                    and u_count >= 2
                    and v_count >= 2
                ):
                    raise scanner._fail("Inconsistent B-spline surface header")
                self._budget(u_poles * v_poles * 4 + 2 * (u_count + v_count), label)
                points, weights = self._spline_poles(
                    u_poles * v_poles, 3, u_rational or v_rational
                )
                u_values, u_multiplicities = self._knots(u_count)
                v_values, v_multiplicities = self._knots(v_count)
                u_expanded, u_indices = self._spline_axis(
                    u_values,
                    u_multiplicities,
                    u_degree,
                    u_poles,
                    u_periodic,
                    label,
                )
                v_expanded, v_indices = self._spline_axis(
                    v_values,
                    v_multiplicities,
                    v_degree,
                    v_poles,
                    v_periodic,
                    label,
                )
                self._budget(
                    u_indices.size * v_indices.size * 4
                    + u_expanded.size
                    + v_expanded.size,
                    label,
                )
                grid = points.reshape(u_poles, v_poles, 3)
                weight_grid = weights.reshape(u_poles, v_poles)
                return _Carrier(
                    label,
                    BSplineSurfacePatch(
                        grid[u_indices[:, None], v_indices[None, :]],
                        weight_grid[u_indices[:, None], v_indices[None, :]],
                        u_expanded,
                        v_expanded,
                        u_degree,
                        v_degree,
                    ),
                )
            case 10:
                u0, u1, v0, v1 = scanner.reals(4)
                basis = self.surface(label, depth + 1)
                bounds = (
                    float(u0 * basis.u_factor),
                    float(u1 * basis.u_factor),
                    float(v0 * basis.v_factor),
                    float(v1 * basis.v_factor),
                )
                if basis.surface_trim is not None:
                    inner = basis.surface_trim
                    bounds = (
                        max(bounds[0], inner[0]),
                        min(bounds[1], inner[1]),
                        max(bounds[2], inner[2]),
                        min(bounds[3], inner[3]),
                    )
                if not bounds[1] > bounds[0] or not bounds[3] > bounds[2]:
                    raise scanner._fail("A trimmed surface has an empty parameter domain")
                return _Carrier(
                    label,
                    basis.value,
                    basis.refusal,
                    v_factor=basis.v_factor,
                    u_factor=basis.u_factor,
                    surface_trim=bounds,
                )
            case 11:
                distance = scanner.real() * self.factor
                basis = self.surface(label, depth + 1)
                if basis.value is None:
                    return _unsupported(label, f"Offset basis: {basis.refusal}")
                if not isinstance(basis.value, AbstractSurfacePatch):
                    raise scanner._fail("An offset basis must be a surface")
                return _Carrier(
                    label,
                    OffsetSurface(basis.value, distance),
                    u_factor=basis.u_factor,
                    v_factor=basis.v_factor,
                    surface_trim=basis.surface_trim,
                )
            case _:
                raise scanner._fail(f"Unknown surface type {kind}")


# ---------------------------------------------------------------- topology


@dataclass(frozen=True, slots=True)
class _Reference:
    orientation: str
    number: int
    location: int


@dataclass(frozen=True, slots=True)
class _CurveRepresentation:
    kind: int
    curve: int
    second: int
    surface: int
    location: int
    first: float
    last: float


@dataclass(slots=True)
class _TShape:
    kind: str
    number: int
    point: np.ndarray | None = None
    same_parameter: bool = True
    same_range: bool = True
    degenerated: bool = False
    representations: list[_CurveRepresentation] = field(default_factory=list)
    surface: int = 0
    surface_location: int = 0
    children: tuple[_Reference, ...] = ()

    @property
    def label(self) -> str:
        return f"TShapes#{self.number} {self.kind}"


@dataclass(slots=True)
class _Decoded:
    version: BRepTextVersion
    locations: list[RigidPlacement]
    curves2d: list[_Carrier]
    curves: list[_Carrier]
    surfaces: list[_Carrier]
    shapes: dict[int, _TShape]
    root: _Reference
    derived: dict[str, int]


def _read_locations(scanner: _Scanner, factor: float) -> list[RigidPlacement]:
    locations = [RigidPlacement.identity()]
    depths = [0]
    for index in range(1, scanner.count("Locations") + 1):
        scanner.label = f"Locations#{index}"
        kind = scanner.integer()
        if kind == 1:
            matrix = scanner.reals(12).reshape(3, 4)
            matrix[:, 3] *= factor
            depths.append(1)
            locations.append(RigidPlacement.from_matrix(matrix, scanner.label))
        elif kind == 2:
            placement = RigidPlacement.identity()
            depth = 1
            factor_index = scanner.integer()
            while factor_index != 0:
                power = scanner.integer()
                if not 0 < factor_index < index:
                    raise refuse(
                        "dangling-reference",
                        scanner.label,
                        f"Location factor {factor_index} is not defined before use.",
                    )
                depth = max(depth, depths[factor_index] + 1)
                if depth > scanner.limits.max_depth:
                    raise refuse(
                        "limit",
                        scanner.label,
                        "Location references exceed the depth limit.",
                    )
                if abs(power) > 64:
                    raise refuse("limit", scanner.label, "A location power exceeds 64.")
                placement = locations[factor_index].power(power).compose(placement)
                factor_index = scanner.integer()
            locations.append(placement)
            depths.append(depth)
        else:
            raise scanner._fail(f"Unknown location type {kind}")
        scanner.depth = max(scanner.depth, depths[-1])
    return locations


def _skip_derived(scanner: _Scanner) -> dict[str, int]:
    derived = {}
    count = scanner.count("Polygon3D")
    for index in range(count):
        scanner.label = f"Polygon3D#{index + 1}"
        nodes, has_parameters = scanner.integer(), scanner.boolean()
        if nodes < 0:
            raise scanner._fail("Negative polygon size")
        scanner.real()
        scanner.reals(3 * nodes + (nodes if has_parameters else 0))
    derived["Polygon3D"] = count
    count = scanner.count("PolygonOnTriangulations")
    for index in range(count):
        scanner.label = f"PolygonOnTriangulations#{index + 1}"
        nodes = scanner.integer()
        if nodes < 0:
            raise scanner._fail("Negative polygon size")
        for _ in range(nodes):
            scanner.integer()
        scanner.keyword("p")
        scanner.real()
        if scanner.boolean():
            scanner.reals(nodes)
    derived["PolygonOnTriangulations"] = count
    return derived


def _skip_triangulations(scanner: _Scanner, version: BRepTextVersion) -> int:
    count = scanner.count("Triangulations")
    for index in range(count):
        scanner.label = f"Triangulations#{index + 1}"
        nodes, triangles, has_uv = scanner.integer(), scanner.integer(), scanner.boolean()
        has_normals = scanner.boolean() if version == "V3" else False
        if nodes < 0 or triangles < 0:
            raise scanner._fail("Negative triangulation size")
        scanner.real()
        scanner.reals(3 * nodes + (2 * nodes if has_uv else 0))
        for _ in range(3 * triangles):
            scanner.integer()
        if has_normals:
            scanner.reals(3 * nodes)
    return count


def _read_edge_geometry(
    scanner: _Scanner, shape: _TShape, version: BRepTextVersion
) -> None:
    scanner.real()
    shape.same_parameter = scanner.boolean()
    shape.same_range = scanner.boolean()
    shape.degenerated = scanner.boolean()
    while True:
        kind = scanner.integer()
        match kind:
            case 0:
                return
            case 1:
                curve, location = scanner.integer(), scanner.integer()
                first, last = scanner.real(), scanner.real()
                shape.representations.append(
                    _CurveRepresentation(1, curve, 0, 0, location, first, last)
                )
            case 2 | 3:
                curve = scanner.integer()
                second = 0
                if kind == 3:
                    second = scanner.integer()
                    scanner.regularity()
                surface, location = scanner.integer(), scanner.integer()
                first, last = scanner.real(), scanner.real()
                if version == "V2":
                    scanner.reals(4)
                shape.representations.append(
                    _CurveRepresentation(
                        kind, curve, second, surface, location, first, last
                    )
                )
            case 4:
                scanner.regularity()
                scanner.integer(), scanner.integer(), scanner.integer(), scanner.integer()
            case 5:
                scanner.integer(), scanner.integer()
            case 6 | 7:
                scanner.integer()
                if kind == 7:
                    scanner.integer()
                scanner.integer(), scanner.integer()
            case _:
                raise scanner._fail(f"Unknown edge representation {kind}")


def _read_shape_geometry(
    scanner: _Scanner, shape: _TShape, version: BRepTextVersion, factor: float
) -> None:
    match shape.kind:
        case "Ve":
            scanner.real()
            shape.point = scanner.reals(3) * factor
            while True:
                scanner.real()
                kind = scanner.integer()
                match kind:
                    case 0:
                        break
                    case 1:
                        scanner.integer()
                    case 2:
                        scanner.integer(), scanner.integer()
                    case 3:
                        scanner.real()
                        scanner.integer()
                    case _:
                        raise scanner._fail(f"Unknown vertex representation {kind}")
                scanner.integer()
        case "Ed":
            _read_edge_geometry(scanner, shape, version)
        case "Fa":
            scanner.boolean()
            scanner.real()
            shape.surface, shape.surface_location = scanner.integer(), scanner.integer()
            if scanner.peek_word() == "2":
                scanner.integer(), scanner.integer()
        case _:
            pass


def _read_shapes(
    scanner: _Scanner, version: BRepTextVersion, factor: float
) -> dict[int, _TShape]:
    count = scanner.count("TShapes")
    shapes: dict[int, _TShape] = {}
    depths: dict[int, int] = {}
    for position in range(count):
        number = count - position
        scanner.label = f"TShapes#{number}"
        kind = scanner.word()
        if kind not in _KINDS:
            raise scanner._fail(f"Unknown shape type {kind!r}")
        shape = _TShape(kind, number)
        scanner.label = shape.label
        _read_shape_geometry(scanner, shape, version, factor)
        flags = scanner.word()
        if len(flags) != 7 or set(flags) - {"0", "1"}:
            raise scanner._fail(f"Malformed shape flags {flags!r}")
        children = []
        while scanner.peek_word() != "*":
            children.append(_read_reference(scanner, count, number))
        scanner.keyword("*")
        shape.children = tuple(children)
        depth = 1 + max((depths[child.number] for child in children), default=0)
        if depth > scanner.limits.max_depth:
            raise refuse("limit", shape.label, "Shape references exceed the depth limit.")
        depths[number] = depth
        scanner.depth = max(scanner.depth, depth)
        shapes[number] = shape
    return shapes


def _read_reference(scanner: _Scanner, count: int, owner: int | None) -> _Reference:
    token = scanner.word()
    orientation, digits = token[:1], token[1:]
    if orientation not in ("+", "-", "i", "e") or not digits.isdigit():
        raise scanner._fail(f"Malformed shape reference {token!r}")
    number = int(digits)
    location = scanner.integer()
    if not 0 < number <= count:
        raise refuse(
            "dangling-reference", scanner.label, f"Shape reference {number} is undefined."
        )
    if owner is not None and number <= owner:
        raise refuse(
            "cyclic-reference",
            scanner.label,
            f"Shape reference {number} is not defined before its owner.",
        )
    return _Reference(orientation, number, location)


def _decode(
    data: bytes, policy: CadImportPolicy, factor: float
) -> tuple[_Decoded, _Scanner]:
    scanner = _Scanner(data, policy)
    header = scanner.line()
    while not header and scanner.position < len(data):
        header = scanner.line()
    if header == _DRAWABLE:
        header = scanner.line()
        while not header and scanner.position < len(data):
            header = scanner.line()
    if header not in _HEADERS:
        raise refuse("malformed", "header", f"Unrecognized BRep text header {header!r}.")
    version = _HEADERS[header]
    locations = _read_locations(scanner, factor)
    geometry = _GeometryReader(scanner, factor)
    curves2d = []
    reader_2d = _GeometryReader(scanner, 1.0)
    for index in range(scanner.count("Curve2ds")):
        scanner.label = f"Curve2ds#{index + 1}"
        curves2d.append(reader_2d.curve(2, scanner.label, 0))
    curves = []
    for index in range(scanner.count("Curves")):
        scanner.label = f"Curves#{index + 1}"
        curves.append(geometry.curve(3, scanner.label, 0))
    derived = _skip_derived(scanner)
    surfaces = []
    for index in range(scanner.count("Surfaces")):
        scanner.label = f"Surfaces#{index + 1}"
        surfaces.append(geometry.surface(scanner.label, 0))
    derived["Triangulations"] = _skip_triangulations(scanner, version)
    shapes = _read_shapes(scanner, version, factor)
    scanner.label = "root"
    root = _read_reference(scanner, len(shapes), None)
    if scanner.peek_word():
        raise scanner._fail("Unexpected trailing content")
    return (
        _Decoded(version, locations, curves2d, curves, surfaces, shapes, root, derived),
        scanner,
    )


# ------------------------------------------------------------ resolution


def _sign(orientation: str, label: str, /) -> int:
    match orientation:
        case "+":
            return 1
        case "-":
            return -1
        case _:
            raise refuse(
                "unsupported-entity",
                label,
                "INTERNAL/EXTERNAL sub-shapes have no native boundary semantics.",
            )


class _Resolver:
    """Lower the located TShape graph to one exact native stage."""

    def __init__(self, decoded: _Decoded, policy: CadImportPolicy, factor: float) -> None:
        self.decoded = decoded
        self.policy = policy
        points = [s.point for s in decoded.shapes.values() if s.point is not None]
        scale = max(1.0, float(np.max(np.abs(points)))) if points else 1.0
        self.stage = CadStage(
            scale,
            policy.pcurve_fit,
            relative_geometric_tolerance=policy.relative_geometric_tolerance,
        )
        self.vertices: dict[tuple[int, bytes], int] = {}
        self.edges: dict[tuple[int, bytes], int] = {}
        self.faces: dict[tuple[int, bytes], int] = {}
        self.solids: dict[int, int] = {}
        self.solid_uses: list[tuple[int, RigidPlacement]] = []
        self.solid_reference_uses: list[str] = []
        self.occurrence_source_references: dict[tuple[str, ...], str] = {}
        self.container_uses: list[tuple[tuple[str, ...], list[int], list[int]]] = []
        self.container_source_references: dict[tuple[str, ...], str] = {}
        self.container_stack: list[int] = []
        self.counts: dict[str, int] = {}
        # Vertex/edge uses in first-use order: (kind, TShape number, arguments).
        self.created: list[
            tuple[str, int, _Reference, RigidPlacement, tuple[str, ...], tuple[int, ...]]
        ] = []

    @staticmethod
    def _reference_label(shape: _TShape, location_path: tuple[int, ...]) -> str:
        """Qualify a declaration with actual source location references, not coordinates."""
        locations = tuple(location for location in location_path if location != 0)
        suffix = (
            "" if not locations else f" [location refs:{','.join(map(str, locations))}]"
        )
        return shape.label + suffix

    def _location(self, index: int, label: str) -> RigidPlacement:
        if not 0 <= index < len(self.decoded.locations):
            raise refuse("dangling-reference", label, f"Location {index} is undefined.")
        return self.decoded.locations[index]

    def _shape(self, reference: _Reference, kind: str, chain: tuple[str, ...]) -> _TShape:
        shape = self.decoded.shapes[reference.number]
        if shape.kind != kind:
            raise refuse(
                "malformed",
                shape.label,
                f"Expected a {kind} sub-shape, found {shape.kind}.",
                chain,
            )
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if sum(self.counts.values()) > self.policy.limits.max_nodes:
            raise refuse(
                "limit",
                shape.label,
                "Expanded shape uses exceed the entity limit.",
                chain,
            )
        return shape

    def _carrier(
        self, table: list[_Carrier], index: int, chain: tuple[str, ...], what: str
    ) -> _Carrier:
        if not 1 <= index <= len(table):
            raise refuse(
                "dangling-reference", chain[-1], f"{what} {index} is undefined.", chain
            )
        carrier = table[index - 1]
        if carrier.value is None:
            raise refuse("unsupported-entity", carrier.label, str(carrier.refusal), chain)
        return carrier

    def vertex(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
    ) -> int:
        shape = self._shape(reference, "Ve", chain)
        located = placement.compose(self._location(reference.location, shape.label))
        key = (reference.number, located.key())
        if key not in self.vertices:
            if shape.point is None:
                raise RuntimeError("Decoded vertices carry a point.")
            self.created.append(
                ("Ve", reference.number, reference, placement, chain, location_path)
            )
            self.vertices[key] = self.stage.vertex(
                located.point(shape.point),
                self._reference_label(shape, (*location_path, reference.location)),
            )
        return self.vertices[key]

    def edge(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
    ) -> tuple[int, _TShape, RigidPlacement]:
        shape = self._shape(reference, "Ed", chain)
        chain = (*chain, shape.label)
        located = placement.compose(self._location(reference.location, shape.label))
        key = (reference.number, located.key())
        if key in self.edges:
            return self.edges[key], shape, located
        if not shape.same_parameter:
            raise refuse(
                "unsupported-entity",
                shape.label,
                "Edges whose p-curves are not same-parameter are not admitted.",
                chain,
            )
        starts = [child for child in shape.children if child.orientation == "+"]
        ends = [child for child in shape.children if child.orientation == "-"]
        if len(starts) != 1 or len(ends) != 1 or len(shape.children) != 2:
            raise refuse(
                "unsupported-entity",
                shape.label,
                "An edge needs exactly one FORWARD and one REVERSED vertex.",
                chain,
            )
        uses = (*location_path, reference.location)
        start = self.vertex(starts[0], located, chain, location_path=uses)
        end = self.vertex(ends[0], located, chain, location_path=uses)
        curves = [r for r in shape.representations if r.kind == 1]
        if shape.degenerated:
            if start != end:
                raise refuse(
                    "malformed",
                    shape.label,
                    "A degenerated edge has two vertices.",
                    chain,
                )
            surfaces = [r for r in shape.representations if r.kind in (2, 3)]
            if not surfaces:
                raise refuse(
                    "malformed", shape.label, "A degenerated edge has no p-curve.", chain
                )
            index = self.stage.degenerate(start, surfaces[0].first, surfaces[0].last)
            self.stage.provenance.append(
                (f"edge:{index}", self._reference_label(shape, uses))
            )
            # The file represents this pole edge; it is not synthesized.
            self.stage.degenerate_edges -= 1
        else:
            if len(curves) != 1:
                raise refuse(
                    "unsupported-entity",
                    shape.label,
                    "An edge needs exactly one 3D curve representation.",
                    chain,
                )
            representation = curves[0]
            carrier = self._carrier(
                self.decoded.curves, representation.curve, (*chain, "Curves"), "Curve"
            )
            if not isinstance(carrier.value, AbstractCurve):
                raise RuntimeError("Curve tables hold curves.")
            curve = place_curve(
                carrier.value,
                located.compose(self._location(representation.location, shape.label)),
                carrier.label,
            )
            first, last = representation.first, representation.last
            first *= carrier.parameter_factor
            last *= carrier.parameter_factor
            if carrier.trim is not None and (
                first < min(carrier.trim) - 1.0e-9 or last > max(carrier.trim) + 1.0e-9
            ):
                raise refuse(
                    "inconsistent-geometry",
                    carrier.label,
                    "An edge range leaves its trimmed curve.",
                    chain,
                )
            try:
                index = self.stage.edge(
                    curve,
                    start,
                    end,
                    self._reference_label(shape, uses),
                    parameter_range=(first, last),
                )
            except ValueError as error:
                raise refuse(
                    "inconsistent-geometry", shape.label, str(error), chain
                ) from error
        self.created.append(
            ("Ed", reference.number, reference, placement, chain, location_path)
        )
        self.edges[key] = index
        return index, shape, located

    def coedge_pcurve(
        self,
        shape: _TShape,
        located: RigidPlacement,
        face: _TShape,
        surface: RigidPlacement,
        sense: int,
        chain: tuple[str, ...],
    ) -> tuple[AbstractCurve | None, float, float]:
        """The declared p-curve/range, or the 3D range when the file omits it."""
        target = surface.key()
        for representation in shape.representations:
            if (
                representation.kind not in (2, 3)
                or representation.surface != face.surface
            ):
                continue
            placed = located.compose(self._location(representation.location, shape.label))
            if placed.key() != target:
                continue
            index = representation.curve
            if representation.kind == 3 and sense < 0:
                index = representation.second
            carrier = self._carrier(
                self.decoded.curves2d, index, (*chain, shape.label, "Curve2ds"), "Curve2d"
            )
            edge_curves = [r for r in shape.representations if r.kind == 1]
            if edge_curves and (
                edge_curves[0].first != representation.first
                or edge_curves[0].last != representation.last
            ):
                raise refuse(
                    "unsupported-entity",
                    shape.label,
                    "P-curve ranges differing from the 3D range are not admitted.",
                    chain,
                )
            if carrier.trim is not None and (
                representation.first < carrier.trim[0]
                or representation.last > carrier.trim[1]
            ):
                raise refuse(
                    "inconsistent-geometry",
                    carrier.label,
                    "A p-curve range leaves its trim.",
                    (*chain, shape.label, "Curve2ds"),
                )
            pcurve = carrier.value
            if not isinstance(pcurve, AbstractCurve):
                raise RuntimeError("Curve2d tables hold curves.")
            return pcurve, representation.first, representation.last
        spatial = [r for r in shape.representations if r.kind == 1]
        if len(spatial) == 1 and not shape.degenerated:
            return None, spatial[0].first, spatial[0].last
        raise refuse(
            "unsupported-entity",
            shape.label,
            "An edge without a p-curve requires a nondegenerate explicit 3D carrier.",
            chain,
        )

    def _vertex_key(
        self, shape: _TShape, located: RigidPlacement, orientation: str
    ) -> tuple[int, bytes]:
        for child in shape.children:
            if child.orientation == orientation:
                placed = located.compose(self._location(child.location, shape.label))
                return child.number, placed.key()
        raise refuse(
            "unsupported-entity",
            shape.label,
            "An edge needs exactly one FORWARD and one REVERSED vertex.",
        )

    def wire(
        self,
        wire: _TShape,
        reference: _Reference,
        located: RigidPlacement,
        face: _TShape,
        surface: RigidPlacement,
        surface_factors: tuple[float, float],
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
        face_patch: AbstractSurfacePatch,
    ) -> list[StagedCoedge]:
        """Coedges of a wire in traversal order about the parametric normal.

        OCCT stores wires relative to the face TShape (the face's orientation
        in its shell is not composed in): outer bounds run counterclockwise
        about the parametric normal, which is the native convention.
        Stored wires need not list edges head to tail; uses are chained by
        shared vertices, disambiguated by p-curve continuity (seams and poles
        meet the same vertex several times).
        """
        wire_located = located.compose(self._location(reference.location, wire.label))
        wire_sign = _sign(reference.orientation, wire.label)
        uses = []
        for edge_reference in wire.children:
            sense = wire_sign * _sign(edge_reference.orientation, wire.label)
            stored = sense
            shape = self._shape(edge_reference, "Ed", (*chain, wire.label))
            edge_located = wire_located.compose(
                self._location(edge_reference.location, shape.label)
            )
            pcurve, first, last = self.coedge_pcurve(
                shape, edge_located, face, surface, stored, (*chain, wire.label)
            )
            if pcurve is None:
                edge, _, _ = self.edge(
                    edge_reference,
                    wire_located,
                    (*chain, wire.label),
                    location_path=(*location_path, reference.location),
                )
                curve_index = self.stage.edge_curves[edge]
                if curve_index < 0:
                    raise refuse(
                        "unsupported-entity",
                        shape.label,
                        "A collapsed edge requires a declared p-curve.",
                        (*chain, wire.label),
                    )
                native_first, native_last = self.stage.edge_ranges[edge]
                try:
                    recovered = recover_pcurve(
                        face_patch,
                        self.stage.curves[curve_index],
                        native_first,
                        native_last,
                        self.stage.scale,
                        None,
                    )
                except ValueError as error:
                    raise refuse(
                        "unsupported-entity",
                        shape.label,
                        str(error),
                        (*chain, wire.label),
                    ) from error
                ends = np.asarray(
                    recovered.curve.evaluate(jnp.asarray((native_first, native_last)))
                )
            else:
                ends = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
            keys = (
                self._vertex_key(shape, edge_located, "+"),
                self._vertex_key(shape, edge_located, "-"),
            )
            if sense < 0:
                ends, keys = ends[::-1], keys[::-1]
            uses.append((edge_reference, shape, sense, pcurve, ends, keys))
        if not uses:
            raise refuse("malformed", wire.label, "A wire has no edges.", chain)
        order = [0]
        remaining = set(range(1, len(uses)))
        while remaining:
            _, _, _, _, ends, keys = uses[order[-1]]
            candidates = [i for i in remaining if uses[i][5][0] == keys[1]]
            scale = 1.0e-7 * max(1.0, float(np.max(np.abs(ends))))
            continuous = [
                i for i in candidates if np.max(np.abs(uses[i][4][0] - ends[1])) <= scale
            ]
            chosen = continuous if continuous else candidates
            if len(chosen) != 1:
                raise refuse(
                    "inconsistent-geometry",
                    wire.label,
                    "The wire does not form one unambiguous connected loop.",
                    chain,
                )
            order.append(chosen[0])
            remaining.discard(chosen[0])
        loop = []
        for index in order:
            edge_reference, shape, sense, pcurve, _, _ = uses[index]
            edge, _, _ = self.edge(
                edge_reference,
                wire_located,
                (*chain, wire.label),
                location_path=(*location_path, reference.location),
            )
            parameter_factor = 1.0
            if not shape.degenerated:
                representation = next(r for r in shape.representations if r.kind == 1)
                parameter_factor = self.decoded.curves[
                    representation.curve - 1
                ].parameter_factor
            if pcurve is None:
                honored = None
            else:
                try:
                    honored = transform_pcurve(pcurve, surface_factors, parameter_factor)
                except ValueError as error:
                    raise refuse(
                        "unsupported-entity",
                        shape.label,
                        str(error),
                        (*chain, wire.label, "Curve2ds"),
                    ) from error
            loop.append(StagedCoedge(edge, sense, honored, shape.label))
        return loop

    def face(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        orientation: int,
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
    ) -> int:
        shape = self._shape(reference, "Fa", chain)
        chain = (*chain, shape.label)
        located = placement.compose(self._location(reference.location, shape.label))
        key = (reference.number, located.key())
        if key in self.faces:
            return self.faces[key]
        carrier = self._carrier(
            self.decoded.surfaces, shape.surface, (*chain, "Surfaces"), "Surface"
        )
        surface = located.compose(self._location(shape.surface_location, shape.label))
        raw = carrier.value
        if not isinstance(raw, AbstractSurfacePatch):
            raise RuntimeError("Surface tables hold surfaces.")
        patch = place_surface(raw, surface, carrier.label)
        loops = []
        for wire_reference in shape.children:
            wire = self._shape(wire_reference, "Wi", chain)
            loops.append(
                self.wire(
                    wire,
                    wire_reference,
                    located,
                    shape,
                    surface,
                    (carrier.u_factor, carrier.v_factor),
                    chain,
                    location_path=(*location_path, reference.location),
                    face_patch=patch,
                )
            )
        if not loops:
            loops = [self.stage.natural_loop(patch, shape.label)]
        if carrier.surface_trim is not None:
            u0, u1, v0, v1 = carrier.surface_trim
            lower, upper = np.asarray((u0, v0)), np.asarray((u1, v1))
            for loop in loops:
                for coedge in loop:
                    first, last = self.stage.edge_ranges[coedge.edge]
                    if coedge.pcurve is None:
                        raise refuse(
                            "inconsistent-geometry",
                            shape.label,
                            "A rectangular surface trim requires an explicit p-curve.",
                            chain,
                        )
                    box = np.asarray(coedge.pcurve.bounding_box(first, last))
                    if np.any(box[0] < lower - 1.0e-9) or np.any(box[1] > upper + 1.0e-9):
                        raise refuse(
                            "inconsistent-geometry",
                            shape.label,
                            "A face boundary leaves its rectangular surface trim.",
                            chain,
                        )
        index = self.stage.face(
            patch,
            loops,
            orientation,
            surface_tag(patch),
            self._reference_label(shape, (*location_path, reference.location)),
            outer_known=False,
        )
        self.faces[key] = index
        return index

    def shell(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
    ) -> int:
        shape = self._shape(reference, "Sh", chain)
        chain = (*chain, shape.label)
        located = placement.compose(self._location(reference.location, shape.label))
        sign = _sign(reference.orientation, shape.label)
        faces, orientations = [], []
        for child in shape.children:
            use = sign * _sign(child.orientation, shape.label)
            face = self.face(
                child,
                located,
                use,
                chain,
                location_path=(*location_path, reference.location),
            )
            faces.append(face)
            orientations.append(use * self.stage.faces[face].orientation)
        if not faces:
            raise refuse("malformed", shape.label, "A shell has no faces.", chain)
        self.stage.shells.append((faces, True))
        self.stage.shell_orientations[len(self.stage.shells) - 1] = orientations
        self.stage.provenance.append(
            (
                f"shell:{len(self.stage.shells) - 1}",
                self._reference_label(shape, (*location_path, reference.location)),
            )
        )
        return len(self.stage.shells) - 1

    def solid(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        chain: tuple[str, ...],
        *,
        location_path: tuple[int, ...] = (),
    ) -> None:
        shape = self._shape(reference, "So", chain)
        if _sign(reference.orientation, shape.label) < 0:
            raise refuse(
                "unsupported-entity",
                shape.label,
                "A REVERSED solid use is not admitted.",
                chain,
            )
        located = placement.compose(self._location(reference.location, shape.label))
        if reference.number not in self.solids:
            identity = RigidPlacement.identity()
            shells = [
                self.shell(child, identity, (*chain, shape.label))
                for child in shape.children
            ]
            if not shells:
                raise refuse("malformed", shape.label, "A solid has no shells.", chain)
            shells.sort(key=lambda shell: -self._shell_extent(shell))
            self.stage.solids.append(shells)
            self.solids[reference.number] = len(self.stage.solids) - 1
            self.stage.provenance.append(
                (f"solid:{self.solids[reference.number]}", shape.label)
            )
        self.solid_uses.append((self.solids[reference.number], located))
        parents = tuple(location for location in location_path if location != 0)
        suffix = (
            ""
            if not parents
            else f" [parent location refs:{','.join(map(str, parents))}]"
        )
        self.solid_reference_uses.append(
            f"{reference.orientation}{reference.number} {reference.location}{suffix}"
        )
        if self.container_stack:
            self.container_uses[self.container_stack[-1]][1].append(
                len(self.solid_uses) - 1
            )
        if len(self.solid_uses) > self.policy.maximum_occurrences:
            raise refuse(
                "limit", shape.label, "Solid occurrences exceed the policy limit."
            )

    def _shell_extent(self, shell: int) -> float:
        points = [
            self.stage.vertices[vertex]
            for face in self.stage.shells[shell][0]
            for loop in self.stage.faces[face].loops
            for coedge in loop
            for vertex in self.stage.edge_vertices[coedge.edge]
        ]
        values = np.asarray(points)
        return float(np.linalg.norm(np.max(values, 0) - np.min(values, 0)))

    def walk(
        self,
        reference: _Reference,
        placement: RigidPlacement,
        chain: tuple[str, ...],
        depth: int,
        *,
        location_path: tuple[int, ...] = (),
    ) -> None:
        shape = self.decoded.shapes[reference.number]
        if depth > self.policy.limits.max_depth:
            raise refuse(
                "limit", shape.label, "Compound nesting exceeds the depth limit.", chain
            )
        match shape.kind:
            case "Co" | "CS":
                self._shape(reference, shape.kind, chain)
                if _sign(reference.orientation, shape.label) < 0:
                    raise refuse(
                        "unsupported-entity",
                        shape.label,
                        "Reversed compound uses have no native occurrence orientation.",
                        chain,
                    )
                located = placement.compose(
                    self._location(reference.location, shape.label)
                )
                parent = self.container_stack[-1] if self.container_stack else None
                path = (
                    ("model",)
                    if parent is None
                    else (
                        *self.container_uses[parent][0],
                        f"container{len(self.container_uses[parent][2])}",
                    )
                )
                container = len(self.container_uses)
                self.container_uses.append((path, [], []))
                self.container_source_references[path] = self._reference_label(
                    shape,
                    (*location_path, reference.location),
                )
                if parent is not None:
                    self.container_uses[parent][2].append(container)
                self.container_stack.append(container)
                try:
                    for child in shape.children:
                        self.walk(
                            child,
                            located,
                            (*chain, shape.label),
                            depth + 1,
                            location_path=(*location_path, reference.location),
                        )
                finally:
                    self.container_stack.pop()
            case "So":
                self.solid(reference, placement, chain, location_path=location_path)
            case "Sh":
                shell = self.shell(
                    reference, placement, chain, location_path=location_path
                )
                self.stage.shells[shell] = (self.stage.shells[shell][0], False)
            case "Fa":
                self.face(
                    reference,
                    placement,
                    _sign(reference.orientation, shape.label),
                    chain,
                    location_path=location_path,
                )
            case "Ed" | "Ve":
                if _sign(reference.orientation, shape.label) < 0:
                    raise refuse(
                        "unsupported-entity",
                        shape.label,
                        "A reversed free entity has no native use-orientation field.",
                        chain,
                    )
                if shape.kind == "Ed":
                    self.edge(reference, placement, chain, location_path=location_path)
                else:
                    self.vertex(reference, placement, chain, location_path=location_path)
            case _:
                raise refuse(
                    "unsupported-entity",
                    shape.label,
                    f"Free {shape.kind} shapes have no native boundary representation.",
                    chain,
                )

    def occurrences(self) -> bool:
        """Stage explicit source occurrences and actual compound relationships."""
        uses: dict[int, list[RigidPlacement]] = {}
        for solid, placement in self.solid_uses:
            uses.setdefault(solid, []).append(placement)
        seen: dict[int, int] = {}
        paths: list[tuple[str, ...]] = []
        for solid, placement in self.solid_uses:
            path = (f"solid{solid}",)
            if len(uses[solid]) > 1:
                path = (*path, f"instance{seen.get(solid, 0)}")
                seen[solid] = seen.get(solid, 0) + 1
            paths.append(path)
            self.occurrence_source_references[path] = self.solid_reference_uses[
                len(paths) - 1
            ]
            self.stage.occurrences.append(
                BRepOccurrence(
                    path,
                    solid,
                    placement.rotation,
                    placement.translation,
                )
            )
        retained: set[int] = set()
        for index in range(len(self.container_uses) - 1, -1, -1):
            path, members, children = self.container_uses[index]
            child_paths = tuple(
                self.container_uses[child][0] for child in children if child in retained
            )
            if members or child_paths:
                self.stage.assembly_containers.append(
                    BRepAssemblyContainer(
                        path,
                        tuple(paths[member] for member in members),
                        child_paths,
                    )
                )
                retained.add(index)
        return False


# ------------------------------------------------------------------ reader


def _import(
    resource: BoundedResource, policy: CadImportPolicy, unit: UnitDefinition, /
) -> CadImportResult:
    unit_meters = Fraction(unit.scale_to_reference)
    factor = length_factor(unit_meters, policy.coordinate_contract)
    decoded, scanner = _decode(resource.data, policy, factor)
    # Stage vertices and edges in file (TShape) order, not traversal order, so
    # exported native models re-read with their exact entity numbering.
    survey = _Resolver(decoded, policy, factor)
    survey.walk(decoded.root, RigidPlacement.identity(), (), 0)
    resolver = _Resolver(decoded, policy, factor)
    for kind, _, reference, placement, chain, location_path in sorted(
        survey.created, key=lambda item: (item[0] != "Ve", -item[1])
    ):
        if kind == "Ve":
            resolver.vertex(reference, placement, chain, location_path=location_path)
        else:
            resolver.edge(reference, placement, chain, location_path=location_path)
    resolver.walk(decoded.root, RigidPlacement.identity(), (), 0)
    default_occurrences = resolver.occurrences()
    digest = resource.manifest.content_sha256
    identity = resource.manifest.source_path or f"brep-text-{digest[:16]}"
    model = resolver.stage.publish(
        coordinate_contract=policy.coordinate_contract,
        source_id=identity,
        source_format=_FORMAT,
        source_digest=digest,
        import_policy_id=policy.policy_id,
        tessellation=policy.tessellation,
        default_occurrences=default_occurrences,
    )
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("Publishing an exact CAD stage must produce geometry.")
    dropped = {name: count for name, count in decoded.derived.items() if count}
    losses = [
        AdapterLoss(
            name,
            "import",
            "dropped",
            f"{count} derived {name} records ignored; native tessellation is re-derived.",
            changes_interpretation=False,
        )
        for name, count in sorted(dropped.items())
    ]
    resource = account_bounded_resource(
        resource,
        depth=scanner.depth,
        nodes=scanner.nodes,
        attributes=scanner.attributes,
        losses=len(losses),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        _FORMAT,
        "phydrax-brep",
        source_id=identity,
        target_id=model.model_id,
        source_profile=_profile(decoded.version),
        coordinate_mapping=(
            f"source length unit {unit.symbol} -> "
            f"{policy.coordinate_contract.length_unit.symbol} (factor {factor!r})",
        ),
        preserved_fields=(
            "exact-carriers",
            "oriented-topology",
            "p-curves",
            "occurrence-placements",
            "assembly-membership",
        ),
        losses=losses,
    )
    counts = {
        "Curve2ds": len(decoded.curves2d),
        "Curves": len(decoded.curves),
        "Surfaces": len(decoded.surfaces),
    }
    for shape in decoded.shapes.values():
        counts[shape.kind] = counts.get(shape.kind, 0) + 1
    return CadImportResult(
        model=model,
        format=_FORMAT,
        schema=f"cascade-topology-{decoded.version}",
        source_digest=digest,
        source_length_unit_meters=float(unit_meters),
        source_angle_unit_radians=1.0,
        resource_manifest=resource.manifest,
        report=report,
        coverage=resolver.stage.coverage(counts),
        provenance=(
            tuple(
                (label, external)
                for label, external in resolver.stage.provenance
                if external.startswith("TShapes#")
                and external.split(" ", 2)[1]
                == _NATIVE_TSHAPE_KINDS.get(label.split(":", 1)[0])
            )
            + tuple(
                (
                    f"container:{index}",
                    resolver.container_source_references[container.path],
                )
                for index, container in enumerate(geometry.assembly_containers)
            )
            + tuple(
                (
                    f"occurrence:{index}",
                    resolver.occurrence_source_references[occurrence.path],
                )
                for index, occurrence in enumerate(geometry.occurrences)
            )
            + (
                (
                    "root",
                    f"{decoded.root.orientation}{decoded.root.number} {decoded.root.location}",
                ),
            )
        ),
    )


def decode_brep_text_resource(
    resource: BoundedResource,
    policy: CadImportPolicy,
    /,
    *,
    source_length_unit: UnitDefinition,
) -> CadImportResult:
    """Decode one bounded OCCT BRep text resource into a native exact `BRepModel`.

    ``source_length_unit`` declares the unitless file's length unit; values are
    converted exactly to ``policy.coordinate_contract``. File same-parameter
    ranges and p-curves are preserved by exact unit/parameter maps, including
    degenerate pole edges; unsupported maps are refused rather than discarded.
    """
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    if not isinstance(source_length_unit, UnitDefinition):
        raise TypeError("source_length_unit must be a UnitDefinition.")
    if resource.manifest.limits != policy.limits:
        raise ValueError("The bounded resource and import policy limits must match.")
    try:
        return _import(resource, policy, source_length_unit)
    except CadInterchangeError as error:
        error.resource_manifest = resource.manifest
        raise
    except ResourceReadError as error:
        resource_refusal(error)
    except (ValueError, TypeError, OverflowError) as error:
        malformed = refuse("malformed", "resource", str(error))
        malformed.resource_manifest = resource.manifest
        raise malformed from error


def decode_brep_text_bytes(
    data: bytes,
    policy: CadImportPolicy,
    /,
    *,
    source_length_unit: UnitDefinition,
    source_path: str | None = None,
) -> CadImportResult:
    """Bound and decode exact in-memory OCCT BRep text bytes."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = bounded_resource_from_bytes(
            data, limits=policy.limits, source_path=source_path
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_brep_text_resource(
        resource, policy, source_length_unit=source_length_unit
    )


def read_brep_text(
    path: str | os.PathLike[str],
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | os.PathLike[str],
    source_length_unit: UnitDefinition,
) -> CadImportResult:
    """Descriptor-read and decode one bounded local OCCT BRep text file."""
    if not isinstance(policy, CadImportPolicy):
        raise TypeError("policy must be a CadImportPolicy.")
    try:
        resource = read_bounded_resource(
            path, trusted_root=trusted_root, limits=policy.limits
        )
    except ResourceReadError as error:
        resource_refusal(error)
    return decode_brep_text_resource(
        resource, policy, source_length_unit=source_length_unit
    )


# ------------------------------------------------------------------ writer


def _number(value: float, /) -> str:
    value_ = float(value)
    if not isfinite(value_):
        raise refuse("inexact-export", "number", "Non-finite values cannot be exported.")
    return repr(value_)


def _numbers(values: object, /) -> str:
    return " ".join(_number(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))


def _unit_speed(vector: np.ndarray, /) -> float:
    """Line speed, treating one within four ulps of unity as exactly unit."""
    speed = float(np.linalg.norm(vector))
    return 1.0 if abs(speed - 1.0) <= 4.0 * np.finfo(np.float64).eps else speed


def _knot_table(knots: np.ndarray, /) -> tuple[np.ndarray, list[int]]:
    values, counts = np.unique(knots, return_counts=True)
    return values, [int(c) for c in counts]


def _curve_record(curve: AbstractCurve, label: str, /) -> str:
    dimension = curve.ambient_dimension
    match curve:
        case LineCurve():
            direction = np.asarray(curve.direction)
            if _unit_speed(direction) != 1.0:
                raise RuntimeError("Lines are normalized before encoding.")
            return f"1 {_numbers(curve.origin)} {_numbers(direction)}"
        case CircleCurve() | EllipseCurve() | ParabolaCurve() | HyperbolaCurve():
            first, second, normal = orthonormal_frame(
                curve.first_axis, curve.second_axis, label
            )
            if dimension == 3:
                frame = f"{_numbers(normal)} {_numbers(first)} {_numbers(second)}"
            else:
                frame = f"{_numbers(first)} {_numbers(second)}"
            if isinstance(curve, CircleCurve):
                return (
                    f"2 {_numbers(curve.center)} {frame} {_number(float(curve.radius))}"
                )
            if isinstance(curve, ParabolaCurve):
                return f"4 {_numbers(curve.vertex)} {frame} {_number(float(curve.focal_length))}"
            if isinstance(curve, HyperbolaCurve):
                return f"5 {_numbers(curve.center)} {frame} {_number(float(curve.first_radius))} {_number(float(curve.second_radius))}"
            if float(curve.first_radius) < float(curve.second_radius):
                raise refuse(
                    "inexact-export",
                    label,
                    "OCCT ellipses require the first radius to be the major radius.",
                )
            return (
                f"3 {_numbers(curve.center)} {frame} {_number(float(curve.first_radius))} "
                f"{_number(float(curve.second_radius))}"
            )
        case OffsetCurve():
            direction = "" if curve.direction is None else f" {_numbers(curve.direction)}"
            return f"9 {_number(float(curve.distance))}{direction}\n{_curve_record(curve.base, label)}"
        case BSplineCurve():
            weights = np.asarray(curve.weights)
            rational = bool(np.any(weights != 1.0))
            values, multiplicities = _knot_table(np.asarray(curve.knots))
            points = np.asarray(curve.control_points)
            poles = " ".join(
                _numbers(point) + (f" {_number(weight)}" if rational else "")
                for point, weight in zip(points, weights, strict=True)
            )
            knots = " ".join(
                f"{_number(value)} {count}"
                for value, count in zip(values, multiplicities, strict=True)
            )
            return (
                f"7 {int(rational)} 0 {curve.degree} {points.shape[0]} {values.shape[0]} "
                f"{poles} {knots}"
            )
        case _:
            raise refuse(
                "inexact-export",
                label,
                f"{type(curve).__name__} carriers have no exact OCCT entity.",
            )


def _surface_record(patch: AbstractSurfacePatch, label: str, /) -> tuple[str, np.ndarray]:
    """OCCT surface record and its exact native-to-external parameter map."""
    match patch:
        case OffsetSurface():
            basis, parameter_map = _surface_record(patch.base, label)
            return f"11 {_number(float(patch.distance))}\n{basis}", parameter_map
        case PlanePatch():
            axes = np.stack((np.asarray(patch.first_axis), np.asarray(patch.second_axis)))
            if np.array_equal(axes @ axes.T, np.eye(2)):
                first, second = axes
                normal = np.cross(first, second)
                parameter_map = np.eye(2)
            else:
                first = axes[0] / np.linalg.norm(axes[0])
                normal = np.cross(axes[0], axes[1])
                normal /= np.linalg.norm(normal)
                second = np.cross(normal, first)
                parameter_map = np.stack((first, second)) @ axes.T
            return (
                f"1 {_numbers(patch.origin)} {_numbers(normal)} {_numbers(first)} {_numbers(second)}",
                parameter_map,
            )
        case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
            first, second, normal = orthonormal_frame(
                patch.first_axis, patch.second_axis, label
            )
            axis = np.asarray(patch.axis)
            if min(np.max(np.abs(normal - axis)), np.max(np.abs(normal + axis))) > 1.0e-9:
                raise refuse(
                    "inexact-export",
                    label,
                    "The surface axis must be orthogonal to its chart.",
                )
            # OCCT Ax3 supports direct and indirect orthonormal frames.
            normal = axis
            origin = (
                patch.center
                if isinstance(patch, (SpherePatch, TorusPatch))
                else patch.origin
            )
            frame = f"{_numbers(origin)} {_numbers(normal)} {_numbers(first)} {_numbers(second)}"
            match patch:
                case CylinderPatch():
                    return f"2 {frame} {_number(float(patch.radius))}", np.eye(2)
                case ConePatch():
                    angle = float(patch.semi_angle)
                    return (
                        f"3 {frame} {_number(float(patch.reference_radius))} {_number(angle)}",
                        np.diag((1.0, 1.0 / cos(angle))),
                    )
                case SpherePatch():
                    return f"4 {frame} {_number(float(patch.radius))}", np.eye(2)
                case _:
                    return (
                        f"5 {frame} {_number(float(patch.major_radius))} {_number(float(patch.minor_radius))}",
                        np.eye(2),
                    )
        case ExtrusionSurface():
            direction = np.asarray(patch.direction)
            speed = float(np.linalg.norm(direction))
            if (
                isinstance(patch.curve, LineCurve)
                and _unit_speed(np.asarray(patch.curve.direction)) != 1.0
            ):
                raise refuse(
                    "inexact-export",
                    label,
                    "An extrusion of a non-unit-speed line is not exported.",
                )
            basis = _curve_record(patch.curve, label)
            return f"6 {_numbers(direction / speed)}\n{basis}", np.diag((1.0, speed))
        case RevolutionSurface():
            if (
                isinstance(patch.curve, LineCurve)
                and _unit_speed(np.asarray(patch.curve.direction)) != 1.0
            ):
                raise refuse(
                    "inexact-export",
                    label,
                    "A revolution of a non-unit-speed line is not exported.",
                )
            basis = _curve_record(patch.curve, label)
            return (
                f"7 {_numbers(patch.axis_origin)} {_numbers(patch.axis_direction)}\n{basis}",
                np.eye(2),
            )
        case BSplineSurfacePatch():
            weights = np.asarray(patch.weights)
            rational = bool(np.any(weights != 1.0))
            points = np.asarray(patch.control_points)
            u_values, u_mult = _knot_table(np.asarray(patch.u_knots))
            v_values, v_mult = _knot_table(np.asarray(patch.v_knots))
            poles = " ".join(
                _numbers(points[i, j])
                + (f" {_number(weights[i, j])}" if rational else "")
                for i in range(points.shape[0])
                for j in range(points.shape[1])
            )
            knots = " ".join(
                f"{_number(v)} {m}"
                for v, m in (
                    *zip(u_values, u_mult, strict=True),
                    *zip(v_values, v_mult, strict=True),
                )
            )
            return (
                f"9 {int(rational)} {int(rational)} 0 0 {patch.u_degree} {patch.v_degree} "
                f"{points.shape[0]} {points.shape[1]} {u_values.shape[0]} {v_values.shape[0]} "
                f"{poles} {knots}",
                np.eye(2),
            )
        case _:
            raise refuse(
                "inexact-export",
                label,
                f"{type(patch).__name__} surfaces have no exact OCCT entity.",
            )


def _normalized_line(curve: AbstractCurve, label: str, /) -> tuple[AbstractCurve, float]:
    """Unit-speed OCCT form of a carrier and its parameter factor ``s = factor t``."""
    if isinstance(curve, OffsetCurve):
        basis, factor = _normalized_line(curve.base, label)
        return OffsetCurve(basis, curve.distance, curve.direction), factor
    if isinstance(curve, LineCurve):
        speed = _unit_speed(np.asarray(curve.direction))
        if speed != 1.0:
            return LineCurve(curve.origin, np.asarray(curve.direction) / speed), speed
    del label
    return curve, 1.0


def _export_pcurve(
    curve: AbstractCurve,
    parameter_map: np.ndarray,
    parameter_factor: float,
    label: str,
    /,
) -> AbstractCurve:
    """Apply a surface chart map without fitting or changing the edge parameter."""
    curve = explicit_carrier(curve)
    if parameter_map[0, 1] == 0.0 and parameter_map[1, 0] == 0.0:
        try:
            return transform_pcurve(
                curve,
                (float(parameter_map[0, 0]), float(parameter_map[1, 1])),
                parameter_factor,
            )
        except ValueError as error:
            raise refuse(
                "inexact-export", label, str(error), ("surface-parameter-map",)
            ) from error
    match curve:
        case LineCurve():
            return LineCurve(
                parameter_map @ np.asarray(curve.origin),
                parameter_map @ np.asarray(curve.direction) / parameter_factor,
            )
        case BSplineCurve():
            return BSplineCurve(
                np.asarray(curve.control_points) @ parameter_map.T,
                curve.weights,
                np.asarray(curve.knots) * parameter_factor,
                curve.degree,
            )
        case _:
            raise refuse(
                "inexact-export",
                label,
                "This p-curve has no exact native carrier under the oblique plane chart map.",
                ("surface-parameter-map",),
            )


def _pcurve_record(curve: AbstractCurve, first: float, last: float, label: str, /) -> str:
    if isinstance(curve, LineCurve) and _unit_speed(np.asarray(curve.direction)) != 1.0:
        origin, direction = np.asarray(curve.origin), np.asarray(curve.direction)
        start, end = origin + first * direction, origin + last * direction
        return f"7 0 0 1 2 2 {_numbers(start)} {_numbers(end)} {_number(first)} 2 {_number(last)} 2"
    return _curve_record(curve, label)


@dataclass(slots=True)
class _Writer:
    model: BRepModel
    version: BRepTextVersion
    policy: CadExportPolicy
    curves2d: list[str] = field(default_factory=list)
    curves: list[str] = field(default_factory=list)
    surfaces: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    records: list[tuple[str, str, list[tuple[str, int, int]]]] = field(
        default_factory=list
    )
    path_normalizations: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list
    )
    approximations: tuple[CadCurveApproximation, ...] = ()
    entity_records: list[tuple[str, int]] = field(default_factory=list)
    reference_records: list[tuple[str, str, int, int]] = field(default_factory=list)

    def provenance(self) -> tuple[tuple[str, str], ...]:
        """Native inventory indices to actual emitted declarations/reference tokens."""
        count = len(self.records)
        definitions = tuple(
            (label, f"TShapes#{count - record} {self.records[record][0]}")
            for label, record in self.entity_records
        )
        uses = tuple(
            (label, f"{orientation}{count - record} {location}")
            for label, orientation, record, location in self.reference_records
        )
        return (*definitions, *uses)

    def _assembly_root(
        self,
        geometry: BRepGeometry,
        solids: list[int],
        loose: list[tuple[str, int, int]],
    ) -> tuple[str, int, int]:
        """Emit only declared container relationships, never name/placement groups."""
        containers = {
            container.path: container for container in geometry.assembly_containers
        }
        children = {
            path for container in containers.values() for path in container.child_paths
        }
        roots = [
            container.path
            for container in geometry.assembly_containers
            if container.path not in children
        ]
        members = {
            path for container in containers.values() for path in container.member_paths
        }
        independent = [
            occurrence
            for occurrence in geometry.occurrences
            if occurrence.path not in members
        ]
        if containers and (len(roots) != 1 or independent):
            raise refuse(
                "inexact-export",
                "assembly",
                "BRep text has one root: disconnected containers or independent occurrence "
                "roots require an explicitly declared parent container, never an inferred one.",
            )
        if not containers and (len(independent) > 1 or (independent and loose)):
            raise refuse(
                "inexact-export",
                "assembly",
                "Multiple independent roots cannot be exported without inventing container membership.",
            )
        used_solids = {occurrence.solid for occurrence in geometry.occurrences}
        if used_solids != set(range(len(solids))):
            raise refuse(
                "inexact-export",
                "assembly",
                "Uninstantiated solid definitions cannot be silently activated as root occurrences.",
            )
        references: dict[tuple[str, ...], tuple[str, int, int]] = {}
        by_path = {occurrence.path: occurrence for occurrence in geometry.occurrences}
        for index, occurrence in enumerate(geometry.occurrences):
            placement = RigidPlacement(
                np.asarray(occurrence.rotation, dtype=np.float64),
                np.asarray(occurrence.translation, dtype=np.float64),
            )
            location = 0
            if not placement.is_identity:
                self.locations.append(
                    "1\n" + "\n".join(_numbers(row) for row in placement.matrix())
                )
                location = len(self.locations)
            references[occurrence.path] = ("+", solids[occurrence.solid], location)
            self.reference_records.append(
                (
                    f"occurrence:{index}",
                    "+",
                    solids[occurrence.solid],
                    location,
                )
            )
        order: list[BRepOccurrence] = []
        if containers:
            pending: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
                (roots[0], ("model",))
            ]
            while pending:
                path, canonical = pending.pop()
                container = containers[path]
                if path != canonical:
                    self.path_normalizations.append(("container", path, canonical))
                order.extend(by_path[member] for member in container.member_paths)
                pending.extend(
                    (child, (*canonical, f"container{index}"))
                    for index, child in reversed(tuple(enumerate(container.child_paths)))
                )
            records: dict[tuple[str, ...], int] = {}
            pending_records = [(roots[0], False)]
            while pending_records:
                path, ready = pending_records.pop()
                container = containers[path]
                if not ready:
                    pending_records.append((path, True))
                    pending_records.extend(
                        (child, False) for child in reversed(container.child_paths)
                    )
                    continue
                uses = [references[member] for member in container.member_paths]
                uses.extend(("+", records[child], 0) for child in container.child_paths)
                if path == roots[0]:
                    uses.extend(loose)
                records[path] = self.record("Co", "", uses)
            self.entity_records.extend(
                (f"container:{index}", records[container.path])
                for index, container in enumerate(geometry.assembly_containers)
            )
            root = ("+", records[roots[0]], 0)
        elif independent:
            order = independent
            root = references[independent[0].path]
        elif len(loose) == 1:
            root = loose[0]
        else:
            root = ("+", self.record("Co", "", loose), 0)
        totals: dict[int, int] = {}
        for occurrence in order:
            totals[occurrence.solid] = totals.get(occurrence.solid, 0) + 1
        definitions: dict[int, int] = {}
        seen: dict[int, int] = {}
        for occurrence in order:
            definition = definitions.setdefault(occurrence.solid, len(definitions))
            canonical: tuple[str, ...] = (f"solid{definition}",)
            if totals[occurrence.solid] > 1:
                canonical = (*canonical, f"instance{seen.get(occurrence.solid, 0)}")
                seen[occurrence.solid] = seen.get(occurrence.solid, 0) + 1
            if occurrence.path != canonical:
                self.path_normalizations.append(
                    ("occurrence", occurrence.path, canonical)
                )
        return root

    def record(
        self, kind: str, geometry: str, children: list[tuple[str, int, int]]
    ) -> int:
        self.records.append((kind, geometry, children))
        if (
            len(self.records)
            + len(self.curves2d)
            + len(self.curves)
            + len(self.surfaces)
            + len(self.locations)
        ) > self.policy.maximum_entities:
            raise refuse("limit", kind, "The export exceeds the total entity limit.")
        return len(self.records) - 1

    def encode(self) -> bytes:
        geometry = self.model.geometry
        if geometry is None:
            raise refuse(
                "inexact-export",
                self.model.source_id,
                "Only models with exact native geometry can be exported.",
            )
        carrier_count = (
            len(geometry.curves)
            + len(geometry.pcurves)
            + len(self.model.patches)
            + len(geometry.assembly_containers)
        )
        if carrier_count > self.policy.maximum_entities:
            raise refuse(
                "limit", "geometry", "The export carriers exceed the entity limit."
            )
        geometry, _, self.approximations = prepare_cad_export_geometry(
            self.model,
            geometry,
            self.policy,
            format_name=_FORMAT,
        )
        normalized: list[tuple[AbstractCurve, float]] = []
        for index, curve in enumerate(geometry.curves):
            if not isinstance(curve, AbstractCurve):
                raise refuse(
                    "inexact-export",
                    f"curve:{index}",
                    "Export preparation must produce a concrete curve carrier.",
                )
            normalized.append(_normalized_line(curve, f"curve:{index}"))
        for curve, _ in normalized:
            self.curves.append(_curve_record(curve, "curve"))
        surface_factors = []
        for face, patch in enumerate(self.model.patches):
            text, factor = _surface_record(patch, f"face:{face}")
            self.surfaces.append(text)
            surface_factors.append(factor)
        vertices = [
            self.record("Ve", f"1e-07\n{_numbers(point)}\n0 0", [])
            for point in np.asarray(geometry.vertex_points)
        ]
        ranges = np.asarray(geometry.edge_ranges)
        orientation = np.asarray(self.model.orientation)
        # Native loops run about the parametric normal, exactly like OCCT's
        # stored (TShape-level) wires; face orientations go on shell uses.
        uses: dict[int, list[tuple[int, int]]] = {}
        for face, loops in enumerate(geometry.face_loops):
            for loop in loops:
                for coedge in loop:
                    uses.setdefault(geometry.coedge_edges[coedge], []).append(
                        (face, coedge)
                    )
        edges = []
        for edge, (curve_index, (start, end)) in enumerate(
            zip(geometry.edge_curves, geometry.edge_vertices, strict=True)
        ):
            factor = 1.0 if curve_index == -1 else normalized[curve_index][1]
            first, last = float(ranges[edge, 0]) * factor, float(ranges[edge, 1]) * factor
            lines = []
            if curve_index != -1:
                lines.append(f"1 {curve_index + 1} 0 {_number(first)} {_number(last)}")
            by_face: dict[int, list[int]] = {}
            for face, coedge in uses.get(edge, []):
                by_face.setdefault(face, []).append(coedge)
            for face, coedges in sorted(by_face.items()):
                indices = {}
                for coedge in coedges:
                    raw_pcurve = geometry.pcurves[coedge]
                    if not isinstance(raw_pcurve, AbstractCurve):
                        raise refuse(
                            "inexact-export",
                            f"coedge:{coedge}",
                            "Export preparation must produce a concrete parameter curve.",
                        )
                    pcurve = _export_pcurve(
                        raw_pcurve,
                        surface_factors[face],
                        factor,
                        f"coedge:{coedge}",
                    )
                    self.curves2d.append(
                        _pcurve_record(pcurve, first, last, f"coedge:{coedge}")
                    )
                    indices[geometry.coedge_senses[coedge]] = len(self.curves2d)
                    points = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
                if len(coedges) == 1:
                    head = f"2 {len(self.curves2d)}"
                elif len(coedges) == 2 and set(indices) == {-1, 1}:
                    head = f"3 {indices[1]} {indices[-1]} CN"
                else:
                    raise refuse(
                        "non-manifold",
                        f"edge:{edge}",
                        "An edge is used more than twice or twice in one sense by a face.",
                    )
                lines.append(f"{head} {face + 1} 0 {_number(first)} {_number(last)}")
                if self.version == "V2":
                    lines.append(_numbers(points))
            lines.append("0")
            flag = int(curve_index == -1)
            edges.append(
                self.record(
                    "Ed",
                    f" 1e-07 1 1 {flag}\n" + "\n".join(lines),
                    [("+", vertices[start], 0), ("-", vertices[end], 0)],
                )
            )
        faces = []
        for face, loops in enumerate(geometry.face_loops):
            wires = [
                self.record(
                    "Wi",
                    "",
                    [
                        (
                            "+" if geometry.coedge_senses[c] > 0 else "-",
                            edges[geometry.coedge_edges[c]],
                            0,
                        )
                        for c in loop
                    ],
                )
                for loop in loops
            ]
            faces.append(
                self.record(
                    "Fa", f"0 1e-07 {face + 1} 0", [("+", wire, 0) for wire in wires]
                )
            )
        shells = []
        in_shell = set()
        for members, signs in zip(
            geometry.shell_faces, geometry.shell_orientations, strict=True
        ):
            in_shell.update(members)
            shells.append(
                self.record(
                    "Sh",
                    "",
                    [
                        ("+" if sign * orientation[face] > 0 else "-", faces[face], 0)
                        for face, sign in zip(members, signs, strict=True)
                    ],
                )
            )
        solids = [
            self.record("So", "", [("+", shells[shell], 0) for shell in group])
            for group in geometry.solid_shells
        ]
        for kind, records_ in (
            ("vertex", vertices),
            ("edge", edges),
            ("face", faces),
            ("shell", shells),
            ("solid", solids),
        ):
            self.entity_records.extend(
                (f"{kind}:{index}", record) for index, record in enumerate(records_)
            )
        in_solid = {shell for group in geometry.solid_shells for shell in group}
        roots: list[tuple[str, int, int]] = []
        roots.extend(
            ("+", shell, 0) for index, shell in enumerate(shells) if index not in in_solid
        )
        roots.extend(
            ("+" if orientation[face] > 0 else "-", faces[face], 0)
            for face in range(len(faces))
            if face not in in_shell
        )
        bound_edges = set(geometry.coedge_edges)
        roots.extend(
            ("+", edge, 0) for index, edge in enumerate(edges) if index not in bound_edges
        )
        bound_vertices = {vertex for pair in geometry.edge_vertices for vertex in pair}
        roots.extend(
            ("+", vertex, 0)
            for index, vertex in enumerate(vertices)
            if index not in bound_vertices
        )
        orientation_, root, location = self._assembly_root(geometry, solids, roots)
        self.reference_records.append(("root", orientation_, root, location))
        return self._text(root, orientation_, location)

    def _text(self, root: int, orientation: str, location: int) -> bytes:
        count = len(self.records)
        header = {
            "V1": "CASCADE Topology V1, (c) Matra-Datavision",
            "V2": "CASCADE Topology V2, (c) Matra-Datavision",
            "V3": "CASCADE Topology V3, (c) Open Cascade",
        }[self.version]
        flags = {
            "Ve": "0101101",
            "Ed": "0101000",
            "Wi": "0101100",
            "Fa": "0101000",
            "Sh": "0101100",
            "So": "0100000",
            "Co": "1100000",
        }
        parts = [_DRAWABLE, "", header]
        parts.append(f"Locations {len(self.locations)}")
        parts.extend(self.locations)
        parts.append(f"Curve2ds {len(self.curves2d)}")
        parts.extend(self.curves2d)
        parts.append(f"Curves {len(self.curves)}")
        parts.extend(self.curves)
        parts.extend(("Polygon3D 0", "PolygonOnTriangulations 0"))
        parts.append(f"Surfaces {len(self.surfaces)}")
        parts.extend(self.surfaces)
        parts.extend(("Triangulations 0", "", f"TShapes {count}"))
        for kind, geometry, children in self.records:
            references = " ".join(
                f"{orientation}{count - index} {location}"
                for orientation, index, location in children
            )
            parts.extend((kind, geometry, "", flags[kind], f"{references} *".lstrip()))
        parts.extend(("", f"{orientation}{count - root} {location}", ""))
        if sum(map(len, parts)) + len(parts) - 1 > self.policy.maximum_bytes:
            raise refuse("limit", "resource", "The export exceeds the byte limit.")
        return "\n".join(parts).encode("ascii")


def write_brep_text(
    model: BRepModel,
    path: str | os.PathLike[str],
    /,
    *,
    policy: CadExportPolicy | None = None,
    version: BRepTextVersion = "V3",
    mode: PublicationMode = "exclusive",
) -> CadExportResult:
    """Atomically publish a native exact model as an OCCT BRep text file.

    Coordinates are written in the model's length unit (the format has none).
    Plane charts, lines, cones and extrusions use OCCT's parameterization, with
    their p-curves mapped exactly. Only declared container relationships are
    emitted; scientific path names normalize with declared loss. Intersections
    require explicit coupled approximation policy and sampled-error evidence.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    policy_ = CadExportPolicy() if policy is None else policy
    if not isinstance(policy_, CadExportPolicy):
        raise TypeError("policy must be a CadExportPolicy or None.")
    version_ = parse(version, BRepTextVersion, "version")
    writer = _Writer(model, version_, policy_)
    try:
        data = writer.encode()
    except CadInterchangeError as error:
        if error.refusal.chain:
            raise
        raise refuse(
            error.refusal.reason,
            error.refusal.entity,
            error.refusal.message,
            (f"model:{model.model_id}", error.refusal.entity),
        ) from error
    receipt = publish_bytes(path, data, maximum_bytes=policy_.maximum_bytes, mode=mode)
    counts = {kind: 0 for kind in ("Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co")}
    for kind, _, _ in writer.records:
        counts[kind] += 1
    losses = tuple(
        AdapterLoss(
            f"{kind}-path:{'/'.join(source)}",
            "export",
            "transformed",
            f"BRep text stores topology, not names: {source!r} -> {target!r}.",
            changes_interpretation=True,
        )
        for kind, source, target in writer.path_normalizations
    ) + tuple(
        AdapterLoss(
            f"intersection:{approximation.branch_id}",
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
        for approximation in writer.approximations
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "phydrax-brep",
        _FORMAT,
        source_id=model.model_id,
        target_id=receipt.receipt_id,
        target_profile=AdapterFormatProfile(
            _FORMAT,
            qualifiers={
                "profile": f"cascade-topology-{version_}",
                **approximation_export_qualifiers(writer.approximations),
            },
        ),
        coordinate_mapping=(
            f"length unit {model.coordinate_contract.length_unit.symbol} (not recorded by the format)",
        ),
        preserved_fields=(
            (
                ("unapproximated-carriers",)
                if writer.approximations
                else ("exact-carriers", "p-curves")
            )
            + ("oriented-topology", "occurrence-placements", "assembly-membership")
        ),
        losses=losses,
        waivers=_export_loss_waivers(losses, policy_, "OCCT BRep text"),
        assumptions=(
            "planes, lines, cones and extrusions are written in OCCT's parameterization",
            canonical_fingerprint(
                {"kind": "brep-text-export", "policy": policy_.policy_id}
            ),
        ),
    )
    return CadExportResult(
        receipt=receipt,
        format=_FORMAT,
        schema=f"cascade-topology-{version_}",
        report=report,
        entity_counts=tuple(sorted(counts.items())),
        approximations=writer.approximations,
        provenance=writer.provenance(),
    )


__all__ = [
    "BRepTextVersion",
    "decode_brep_text_bytes",
    "decode_brep_text_resource",
    "read_brep_text",
    "write_brep_text",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import atan2, ceil, cos, floor, isfinite, pi, sin
from numbers import Integral, Real
from typing import overload, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._validation import positive_finite_float
from ...discretization._exact_algebraic_geometry import ExactScalar, rational_coefficient
from ...typing import ConvertibleToArray
from .._atlas import TrimDomain
from ._assembly import BRepAffinePlacement, BRepAssemblyGraph
from ._intersection import (
    ConicParameterEndpoint,
    NativePeriodEndpoint,
    RootEndpoint,
)
from ._model import BRepGeometry, BRepImportReport, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BRepCurve,
    BSplineCurve,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
)
from ._profile_certificate import (
    arc_pieces,
    bspline_pieces,
    bspline_turn_terms,
    certify_profile_contacts,
    line_piece,
)
from ._root_bindings import BRepConicVertex, BRepVertexRoot


if TYPE_CHECKING:
    from ._offset_result import BRepOffsetResult


_TWO_PI = 2.0 * pi
_NATIVE_CONSTRUCTION_POLICY_ID = canonical_fingerprint(
    {"kind": "native-brep-construction"}
)


class BRepTessellationPolicy(StrictModule):
    """Request physical chord and oriented-normal deviation bounds for triangles.

    ``linear_deflection`` is a source-length bound. ``angular_deflection`` is
    the maximum continuous angle between oriented source and target-map
    normals, in radians. ``trim_samples_per_edge`` is a minimum scheduling
    density and never a complexity limit.
    """

    linear_deflection: float = eqx.field(static=True)
    angular_deflection: float = eqx.field(static=True)
    trim_samples_per_edge: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_deflection: float = 1.0e-3,
        angular_deflection: float = 0.1,
        trim_samples_per_edge: int = 33,
    ) -> None:
        linear = positive_finite_float(linear_deflection, "linear_deflection")
        angular = positive_finite_float(angular_deflection, "angular_deflection")
        if angular >= pi:
            raise ValueError("angular_deflection must be below pi.")
        if isinstance(trim_samples_per_edge, bool) or not isinstance(
            trim_samples_per_edge, Integral
        ):
            raise TypeError("trim_samples_per_edge must be an integer.")
        subdivisions = int(trim_samples_per_edge)
        if subdivisions < 3:
            raise ValueError("trim_samples_per_edge must be at least three.")
        self.linear_deflection = linear
        self.angular_deflection = angular
        self.trim_samples_per_edge = subdivisions
        self.policy_id = canonical_fingerprint(
            {
                "kind": "native-brep-tessellation-policy",
                "linear_deflection": linear,
                "angular_deflection": angular,
                "trim_samples_per_edge": subdivisions,
            }
        )


def _point2(value: ConvertibleToArray, name: str, /) -> tuple[float, float]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 2D point.")
    return float(point[0]), float(point[1])


def _point3(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite 3D vector.")
    return point


def _unit(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    vector = _point3(value, name)
    length = float(np.linalg.norm(vector))
    if not length > 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / length


@dataclass(frozen=True, slots=True)
class ProfileLine:
    """Straight profile segment from its declared vertex to the next."""


@dataclass(frozen=True, slots=True)
class ProfileArc:
    """Circular profile arc about a declared planar center."""

    center: tuple[float, float]
    counterclockwise: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point2(self.center, "center"))
        if not isinstance(self.counterclockwise, bool):
            raise TypeError("counterclockwise must be a boolean.")


@dataclass(frozen=True, slots=True)
class ProfileCurveUse:
    """Directed use of an unchanged planar circle or ellipse."""

    curve: CircleCurve | EllipseCurve
    first: float
    last: float

    def __post_init__(self) -> None:
        if not isinstance(self.curve, (CircleCurve, EllipseCurve)):
            raise TypeError("curve must be a native planar circle or ellipse.")
        if self.curve.ambient_dimension != 2:
            raise ValueError("Profile conics must be two-dimensional.")
        if (
            isinstance(self.first, bool)
            or isinstance(self.last, bool)
            or not isfinite(self.first)
            or not isfinite(self.last)
            or not self.first < self.last
            or self.last - self.first > _TWO_PI
        ):
            raise ValueError(
                "A conic use requires a finite increasing range of at most one turn."
            )


ProfileSegment = ProfileLine | ProfileArc | ProfileCurveUse | BSplineCurve


@dataclass(frozen=True, slots=True)
class ProfileLoop:
    """Closed line/conic boundary; each segment joins vertex i to i+1."""

    vertices: tuple[tuple[float, float], ...]
    segments: tuple[ProfileSegment, ...]

    def __post_init__(self) -> None:
        vertices = tuple(_point2(point, "profile vertex") for point in self.vertices)
        segments = tuple(self.segments)
        if not vertices or len(vertices) != len(segments):
            raise ValueError("A profile loop requires one segment per declared vertex.")
        if any(
            not isinstance(
                segment, (ProfileLine, ProfileArc, ProfileCurveUse, BSplineCurve)
            )
            for segment in segments
        ):
            raise TypeError(
                "Profile loops support line, conic and native B-spline segments."
            )
        if len(vertices) > 1 and len(set(vertices)) != len(vertices):
            raise ValueError(
                "Profile loop vertices must be distinct except for a one-vertex conic."
            )
        if len(vertices) == 1 and isinstance(segments[0], ProfileLine):
            raise ValueError("A one-vertex profile loop must be a closed conic.")
        if len(vertices) == 2 and all(
            isinstance(segment, ProfileLine) for segment in segments
        ):
            raise ValueError("A two-vertex profile loop must contain a conic segment.")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "segments", segments)

    @classmethod
    def polygon(cls, vertices: ConvertibleToArray, /) -> ProfileLoop:
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("polygon vertices must have shape (num_vertices, 2).")
        return cls(
            tuple((float(point[0]), float(point[1])) for point in points),
            tuple(ProfileLine() for _ in range(points.shape[0])),
        )

    @classmethod
    def circle(cls, center: ConvertibleToArray, radius: float, /) -> ProfileLoop:
        center_ = _point2(center, "center")
        radius_ = positive_finite_float(radius, "radius")
        return cls(
            ((center_[0] + radius_, center_[1]),),
            (ProfileArc(center_),),
        )


@dataclass(frozen=True, slots=True)
class ProfilePlane:
    """Orthonormal frame mapping profile coordinates into physical space."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    y_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        origin = _point3(self.origin, "origin")
        x_axis = _point3(self.x_axis, "x_axis")
        y_axis = _point3(self.y_axis, "y_axis")
        tolerance = 128.0 * np.finfo(np.float64).eps
        if (
            abs(float(np.linalg.norm(x_axis)) - 1.0) > tolerance
            or abs(float(np.linalg.norm(y_axis)) - 1.0) > tolerance
            or abs(float(x_axis @ y_axis)) > tolerance
        ):
            raise ValueError("ProfilePlane axes must be orthonormal.")
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(self, "x_axis", tuple(float(value) for value in x_axis))
        object.__setattr__(self, "y_axis", tuple(float(value) for value in y_axis))

    @property
    def normal(self) -> np.ndarray:
        return np.cross(np.asarray(self.x_axis), np.asarray(self.y_axis))

    def point(self, coordinates: ConvertibleToArray, /) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        return (
            np.asarray(self.origin)
            + values[..., 0:1] * np.asarray(self.x_axis)
            + values[..., 1:2] * np.asarray(self.y_axis)
        )

    def vector(self, coordinates: ConvertibleToArray, /) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        return values[..., 0:1] * np.asarray(self.x_axis) + values[..., 1:2] * np.asarray(
            self.y_axis
        )


@overload
def _profile_point_exact(
    plane: ProfilePlane, coordinates: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]: ...


@overload
def _profile_point_exact(
    plane: ProfilePlane, coordinates: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]: ...


def _profile_point_exact(
    plane: ProfilePlane, coordinates: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    if len(coordinates) != 2:
        raise ValueError("Profile points must have two exact coordinates.")
    origin = tuple(Fraction.from_float(value) for value in plane.origin)
    x_axis = tuple(Fraction.from_float(value) for value in plane.x_axis)
    y_axis = tuple(Fraction.from_float(value) for value in plane.y_axis)
    values = tuple(
        origin[axis] + coordinates[0] * x_axis[axis] + coordinates[1] * y_axis[axis]
        for axis in range(3)
    )
    return values[0], values[1], values[2]


@overload
def _profile_vector_exact(
    plane: ProfilePlane, coordinates: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]: ...


@overload
def _profile_vector_exact(
    plane: ProfilePlane, coordinates: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]: ...


def _profile_vector_exact(
    plane: ProfilePlane, coordinates: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    if len(coordinates) != 2:
        raise ValueError("Profile vectors must have two exact coordinates.")
    x_axis = tuple(Fraction.from_float(value) for value in plane.x_axis)
    y_axis = tuple(Fraction.from_float(value) for value in plane.y_axis)
    values = tuple(
        coordinates[0] * x_axis[axis] + coordinates[1] * y_axis[axis] for axis in range(3)
    )
    return values[0], values[1], values[2]


def _profile_vertex_exact(
    plane: ProfilePlane, coordinates: ConvertibleToArray, /
) -> tuple[Fraction, Fraction, Fraction]:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("Profile vertex coordinates must be finite 2D points.")
    exact = tuple(Fraction.from_float(float(value)) for value in values)
    return _profile_point_exact(plane, exact)


@dataclass(frozen=True, slots=True)
class PlanarProfile:
    """Validated profile loops oriented with material on their left side."""

    plane: ProfilePlane
    outer: ProfileLoop
    holes: tuple[ProfileLoop, ...] = ()
    endpoint_gap_bounds: tuple[tuple[tuple[float, float], ...], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plane, ProfilePlane):
            raise TypeError("plane must be a ProfilePlane.")
        original_loops = (self.outer, *tuple(self.holes))
        if any(not isinstance(loop, ProfileLoop) for loop in original_loops):
            raise TypeError("outer and holes must be ProfileLoop values.")
        gap_rows = _profile_endpoint_gap_rows(original_loops, self.endpoint_gap_bounds)
        normalized = tuple(
            _oriented_loop(loop, positive=index == 0)
            for index, loop in enumerate(original_loops)
        )
        if all(_profile_loop_is_polygon(loop) for loop in normalized):
            _validate_polygon_loops(normalized)
        elif len(normalized) > 1:
            _validate_profile_loop_relations(normalized)
        if gap_rows is not None:
            gap_rows = tuple(
                rows
                if normalized[index] is original_loops[index]
                else tuple(
                    (
                        rows[(len(rows) - edge - 2) % len(rows)][1],
                        rows[(len(rows) - edge - 2) % len(rows)][0],
                    )
                    for edge in range(len(rows))
                )
                for index, rows in enumerate(gap_rows)
            )
            object.__setattr__(self, "endpoint_gap_bounds", gap_rows)
        object.__setattr__(self, "outer", normalized[0])
        object.__setattr__(self, "holes", normalized[1:])

    @property
    def loops(self) -> tuple[ProfileLoop, ...]:
        return (self.outer, *self.holes)


@dataclass(slots=True)
class _Face:
    patch: AbstractSurfacePatch
    bounds: np.ndarray
    orientation: int
    tag: str
    loops: list[list[int]]


@dataclass(frozen=True, slots=True)
class _ExtrusionEdgeSpec:
    curve2d: AbstractCurve
    first: float
    last: float
    base_curve: AbstractCurve
    top_curve: AbstractCurve
    patch: AbstractSurfacePatch
    base_roots: tuple[ConicParameterRoot | None, ConicParameterRoot | None]
    top_roots: tuple[ConicParameterRoot | None, ConicParameterRoot | None]
    start_pcurve: AbstractCurve
    end_pcurve: AbstractCurve
    wall_box: np.ndarray


@dataclass(slots=True)
class _Builder:
    vertices: list[np.ndarray] = field(default_factory=list)
    vertex_exact_points: list[tuple[Fraction, Fraction, Fraction] | None] = field(
        default_factory=list
    )
    vertex_roots: list[BRepVertexRoot | None] = field(default_factory=list)
    edge_endpoint_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = field(
        default_factory=list
    )
    coedge_endpoint_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = field(
        default_factory=list
    )
    curves: list[BRepCurve] = field(default_factory=list)
    edge_curves: list[int] = field(default_factory=list)
    edge_ranges: list[tuple[float, float]] = field(default_factory=list)
    edge_vertices: list[tuple[int, int]] = field(default_factory=list)
    pcurves: list[BRepCurve] = field(default_factory=list)
    coedge_edges: list[int] = field(default_factory=list)
    coedge_senses: list[int] = field(default_factory=list)
    faces: list[_Face] = field(default_factory=list)

    def vertex(
        self,
        point: ConvertibleToArray,
        /,
        *,
        exact_point: tuple[Fraction, Fraction, Fraction] | None = None,
        root: BRepVertexRoot | None = None,
    ) -> int:
        point_ = _point3(point, "vertex")
        if exact_point is not None:
            if len(exact_point) != 3 or any(
                not isinstance(value, Fraction) for value in exact_point
            ):
                raise TypeError("exact_point must contain three Fraction coordinates.")
            rounded = np.asarray(tuple(float(value) for value in exact_point))
            if not np.array_equal(rounded, point_):
                raise ValueError(
                    "exact_point must round to the supplied vertex realization."
                )
        self.vertices.append(point_)
        self.vertex_exact_points.append(exact_point)
        self.vertex_roots.append(root)
        return len(self.vertices) - 1

    def edge(
        self,
        curve: BRepCurve,
        first: float,
        last: float,
        start: int,
        end: int,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None),
    ) -> int:
        if not first < last:
            raise ValueError("A nondegenerate edge range must be increasing.")
        if curve.ambient_dimension != 3:
            raise ValueError("A B-Rep edge curve must be three-dimensional.")
        if not isinstance(endpoint_roots, tuple) or len(endpoint_roots) != 2:
            raise TypeError("endpoint_roots must contain start and end source roots.")
        self.curves.append(curve)
        self.edge_curves.append(len(self.curves) - 1)
        self.edge_ranges.append((float(first), float(last)))
        self.edge_vertices.append((start, end))
        self.edge_endpoint_roots.append(endpoint_roots)
        return len(self.edge_curves) - 1

    def degenerate(
        self,
        vertex: int,
        first: float,
        last: float,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None),
    ) -> int:
        if not first < last:
            raise ValueError("A degenerate edge range must be increasing.")
        if not isinstance(endpoint_roots, tuple) or len(endpoint_roots) != 2:
            raise TypeError("endpoint_roots must contain start and end source roots.")
        self.edge_curves.append(-1)
        self.edge_ranges.append((float(first), float(last)))
        self.edge_vertices.append((vertex, vertex))
        self.edge_endpoint_roots.append(endpoint_roots)
        return len(self.edge_curves) - 1

    def coedge(
        self,
        edge: int,
        sense: int,
        pcurve: BRepCurve,
        /,
        *,
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None),
    ) -> int:
        if sense not in (-1, 1) or pcurve.ambient_dimension != 2:
            raise ValueError("A coedge requires a 2D p-curve and orientation sign.")
        if not isinstance(endpoint_roots, tuple) or len(endpoint_roots) != 2:
            raise TypeError("endpoint_roots must contain start and end source roots.")
        self.coedge_edges.append(edge)
        self.coedge_senses.append(sense)
        self.pcurves.append(pcurve)
        self.coedge_endpoint_roots.append(endpoint_roots)
        return len(self.coedge_edges) - 1

    def native_phase(
        self,
        edge: int,
        patch: AbstractSurfacePatch,
        axis: int,
        /,
        *,
        coedge: int,
        rational: Fraction = Fraction(0),
    ) -> None:
        """Bind one full native period to its edge and p-curve source carriers."""
        from ._intersection import NativePeriodEndpoint

        if not 0 <= edge < len(self.edge_curves):
            raise ValueError("native_phase.edge must reference a staged edge.")
        if not 0 <= coedge < len(self.coedge_edges) or self.coedge_edges[coedge] != edge:
            raise ValueError("native_phase.coedge must use the staged edge.")
        if patch.native_period_symbols()[axis] != "two_pi":
            raise ValueError("native_phase requires a source-declared patch period.")
        curve_index = self.edge_curves[edge]
        carrier = self.pcurves[coedge] if curve_index < 0 else self.curves[curve_index]
        self.edge_endpoint_roots[edge] = (
            NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=0),
            NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=1),
        )
        pcurve = self.pcurves[coedge]
        self.coedge_endpoint_roots[coedge] = (
            NativePeriodEndpoint(pcurve, patch, axis, rational=rational, turns=0),
            NativePeriodEndpoint(pcurve, patch, axis, rational=rational, turns=1),
        )

    def native_curve_phase(
        self,
        edge: int,
        /,
        *,
        coedge: int | None = None,
        rational: Fraction = Fraction(0),
    ) -> None:
        """Declare one exact native curve turn without consulting numeric range width."""
        from ._intersection import NativePeriodEndpoint

        if not 0 <= edge < len(self.edge_curves) or self.edge_curves[edge] < 0:
            raise ValueError("native_curve_phase requires a nondegenerate source edge.")
        curve = self.curves[self.edge_curves[edge]]
        if curve.native_period_symbol() != "two_pi":
            raise ValueError(
                "native_curve_phase requires a source-declared curve period."
            )
        self.edge_endpoint_roots[edge] = (
            NativePeriodEndpoint(curve, rational=rational, turns=0),
            NativePeriodEndpoint(curve, rational=rational, turns=1),
        )
        if coedge is None:
            return
        if not 0 <= coedge < len(self.coedge_edges) or self.coedge_edges[coedge] != edge:
            raise ValueError("native_curve_phase.coedge must use the staged edge.")
        pcurve = self.pcurves[coedge]
        if pcurve.native_period_symbol() != "two_pi":
            raise ValueError(
                "The source p-curve does not declare the same native period."
            )
        self.coedge_endpoint_roots[coedge] = (
            NativePeriodEndpoint(pcurve, rational=rational, turns=0),
            NativePeriodEndpoint(pcurve, rational=rational, turns=1),
        )

    def face(
        self,
        patch: AbstractSurfacePatch,
        bounds: ConvertibleToArray,
        orientation: int,
        tag: str,
        loops: list[list[int]],
        /,
    ) -> int:
        box = np.asarray(bounds, dtype=np.float64)
        if box.shape != (2, 2) or np.any(box[1] <= box[0]):
            raise ValueError("A source face requires a finite increasing 2D chart box.")
        if (
            orientation not in (-1, 1)
            or not tag
            or not loops
            or any(not loop for loop in loops)
        ):
            raise ValueError("A source face requires orientation, tag, and closed loops.")
        self.faces.append(_Face(patch, box, orientation, tag, loops))
        return len(self.faces) - 1

    def geometry(
        self,
        *,
        solid: bool,
        shell_faces: tuple[tuple[int, ...], ...] | None = None,
        shell_orientations: tuple[tuple[int, ...], ...] | None = None,
        solid_shells: tuple[tuple[int, ...], ...] | None = None,
        assembly_graph: BRepAssemblyGraph | None = None,
        edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = None,
        edge_face_gap_bounds: tuple[float, ...] | None = None,
    ) -> tuple[
        BRepGeometry,
        tuple[AbstractSurfacePatch, ...],
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        tuple[TrimDomain | None, ...],
    ]:
        face_count = len(self.faces)
        if shell_faces is None and shell_orientations is None and solid_shells is None:
            shell_faces_ = (tuple(range(face_count)),) if solid and face_count else ()
            shell_orientations_ = ((1,) * face_count,) if solid and face_count else ()
            solid_shells_ = ((0,),) if solid and face_count else ()
        elif shell_faces is None or shell_orientations is None or solid_shells is None:
            raise ValueError(
                "A custom shell partition requires all three incidence banks."
            )
        else:
            shell_faces_ = tuple(tuple(values) for values in shell_faces)
            shell_orientations_ = tuple(tuple(values) for values in shell_orientations)
            solid_shells_ = tuple(tuple(values) for values in solid_shells)
        geometry = BRepGeometry(
            vertex_points=np.asarray(self.vertices, dtype=np.float64).reshape((-1, 3)),
            curves=tuple(self.curves),
            edge_curves=tuple(self.edge_curves),
            edge_ranges=np.asarray(self.edge_ranges, dtype=np.float64).reshape((-1, 2)),
            edge_vertices=tuple(self.edge_vertices),
            pcurves=tuple(self.pcurves),
            vertex_exact_points=tuple(self.vertex_exact_points),
            vertex_roots=tuple(self.vertex_roots),
            edge_endpoint_roots=tuple(self.edge_endpoint_roots),
            coedge_endpoint_roots=tuple(self.coedge_endpoint_roots),
            edge_vertex_gap_bounds=edge_vertex_gap_bounds,
            edge_face_gap_bounds=edge_face_gap_bounds,
            coedge_edges=tuple(self.coedge_edges),
            coedge_senses=tuple(self.coedge_senses),
            face_loops=tuple(
                tuple(tuple(loop) for loop in face.loops) for face in self.faces
            ),
            shell_faces=shell_faces_,
            shell_orientations=shell_orientations_,
            solid_shells=solid_shells_,
            assembly_graph=assembly_graph,
        )
        patches = tuple(face.patch for face in self.faces)
        bounds = np.asarray(
            [face.bounds for face in self.faces], dtype=np.float64
        ).reshape((-1, 2, 2))
        orientations = np.asarray(
            [face.orientation for face in self.faces], dtype=np.float64
        ).reshape((-1,))
        tags = tuple(face.tag for face in self.faces)
        trims = tuple(
            _face_trim_domain(geometry, index, face.bounds)
            for index, face in enumerate(self.faces)
        )
        return geometry, patches, bounds, orientations, tags, trims


def _fraction(value: float, /) -> Fraction:
    return Fraction.from_float(float(value))


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], /
) -> Fraction:
    ax, ay = _fraction(a[0]), _fraction(a[1])
    bx, by = _fraction(b[0]), _fraction(b[1])
    cx, cy = _fraction(c[0]), _fraction(c[1])
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], p: tuple[float, float], /
) -> bool:
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
        and _orientation(a, b, p) == 0
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    /,
) -> bool:
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if (ab_c > 0 > ab_d or ab_d > 0 > ab_c) and (cd_a > 0 > cd_b or cd_b > 0 > cd_a):
        return True
    return (
        (ab_c == 0 and _on_segment(a, b, c))
        or (ab_d == 0 and _on_segment(a, b, d))
        or (cd_a == 0 and _on_segment(c, d, a))
        or (cd_b == 0 and _on_segment(c, d, b))
    )


def _polygon_area(loop: ProfileLoop, /) -> Fraction:
    return sum(
        (
            _fraction(loop.vertices[index][0])
            * _fraction(loop.vertices[(index + 1) % len(loop.vertices)][1])
            - _fraction(loop.vertices[(index + 1) % len(loop.vertices)][0])
            * _fraction(loop.vertices[index][1])
            for index in range(len(loop.vertices))
        ),
        Fraction(0),
    )


def _profile_bspline_controls(
    curve: BSplineCurve,
    start: tuple[float, float],
    end: tuple[float, float],
    /,
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    """Exact controls and endpoint chord of one clamped profile B-spline."""
    if curve.ambient_dimension != 2:
        raise ValueError(
            "Planar profile B-splines must have two-dimensional control points."
        )
    control_points = np.asarray(curve.control_points, dtype=np.float64)
    knots = np.asarray(curve.knots, dtype=np.float64)
    first, last = curve.parameter_domain
    if not np.all(knots[: curve.degree + 1] == first) or not np.all(
        knots[-curve.degree - 1 :] == last
    ):
        raise ValueError(
            "Planar profile B-splines require a full clamped parameter domain."
        )
    exact_start = tuple(_fraction(value) for value in start)
    exact_end = tuple(_fraction(value) for value in end)
    controls = tuple(
        tuple(_fraction(value) for value in point) for point in control_points
    )
    if controls[0] != exact_start or controls[-1] != exact_end:
        raise ValueError(
            "Profile B-spline boundary controls must match declared vertices exactly."
        )
    delta = tuple(b - a for a, b in zip(exact_start, exact_end, strict=True))
    if delta[0] == 0 and delta[1] == 0:
        raise ValueError("Profile B-spline endpoints must be distinct.")
    return controls, exact_start, delta


def _profile_bspline_is_straight(
    curve: BSplineCurve,
    start: tuple[float, float],
    end: tuple[float, float],
    /,
) -> bool:
    """Whether every control lies exactly on the endpoint chord."""
    controls, exact_start, delta = _profile_bspline_controls(curve, start, end)
    return all(
        (point[0] - exact_start[0]) * delta[1] - (point[1] - exact_start[1]) * delta[0]
        == 0
        for point in controls
    )


def _profile_bspline_domain(
    curve: BSplineCurve,
    start: tuple[float, float],
    end: tuple[float, float],
    /,
) -> tuple[float, float]:
    controls, exact_start, delta = _profile_bspline_controls(curve, start, end)
    axis = max(range(2), key=lambda index: abs(delta[index]))
    previous = Fraction(0)
    for point in controls:
        offset = tuple(point[i] - exact_start[i] for i in range(2))
        if offset[0] * delta[1] - offset[1] * delta[0] != 0:
            raise ValueError(
                "Profile B-spline controls must lie on their exact endpoint line."
            )
        fraction = offset[axis] / delta[axis]
        if fraction < previous or not 0 <= fraction <= 1:
            raise ValueError(
                "Profile B-spline controls must be ordered along their endpoint line."
            )
        previous = fraction
    return curve.parameter_domain


def _profile_loop_is_polygon(loop: ProfileLoop, /) -> bool:
    """Classify loops whose every segment is an exact chord.

    A curved B-spline makes the loop non-polygonal; it is not a refusal here.
    A straight B-spline must also be monotone along its chord to be a polygon
    edge, which ``_profile_bspline_domain`` certifies.
    """
    return all(
        isinstance(segment, ProfileLine)
        or (
            isinstance(segment, BSplineCurve)
            and _profile_bspline_is_straight(
                segment,
                loop.vertices[index],
                loop.vertices[(index + 1) % len(loop.vertices)],
            )
            and bool(
                _profile_bspline_domain(
                    segment,
                    loop.vertices[index],
                    loop.vertices[(index + 1) % len(loop.vertices)],
                )
            )
        )
        for index, segment in enumerate(loop.segments)
    )


def _reverse_loop(loop: ProfileLoop, /) -> ProfileLoop:
    count = len(loop.vertices)
    segments = []
    for index in range(count):
        # Reversed vertex k is source vertex n-1-k, so reversed segment k
        # (vertex k to k+1) is source segment n-2-k traversed backward.
        source = loop.segments[(count - index - 2) % count]
        if isinstance(source, ProfileLine):
            segments.append(ProfileLine())
        elif isinstance(source, ProfileArc):
            segments.append(ProfileArc(source.center, not source.counterclockwise))
        elif isinstance(source, BSplineCurve):
            first, last = source.parameter_domain
            segments.append(
                BSplineCurve(
                    np.asarray(source.control_points)[::-1],
                    np.asarray(source.weights)[::-1],
                    first + last - np.asarray(source.knots)[::-1],
                    source.degree,
                )
            )
        else:
            curve = source.curve
            if isinstance(curve, CircleCurve):
                reversed_curve = CircleCurve(
                    curve.center,
                    curve.first_axis,
                    -curve.second_axis,
                    curve.radius,
                    center_exact=curve.center_exact,
                    first_axis_exact=curve.first_axis_exact,
                    second_axis_exact=tuple(-value for value in curve.second_axis_exact),
                    radius_exact=curve.radius_exact,
                )
            else:
                reversed_curve = EllipseCurve(
                    curve.center,
                    curve.first_axis,
                    -curve.second_axis,
                    curve.first_radius,
                    curve.second_radius,
                    center_exact=curve.center_exact,
                    first_axis_exact=curve.first_axis_exact,
                    second_axis_exact=tuple(-value for value in curve.second_axis_exact),
                    first_radius_exact=curve.first_radius_exact,
                    second_radius_exact=curve.second_radius_exact,
                )
            segments.append(ProfileCurveUse(reversed_curve, -source.last, -source.first))
    return ProfileLoop(tuple(reversed(loop.vertices)), tuple(segments))


def _oriented_loop(loop: ProfileLoop, *, positive: bool) -> ProfileLoop:
    if _profile_loop_is_polygon(loop):
        area = _polygon_area(loop)
        if area == 0:
            raise ValueError("A polygon profile loop must enclose nonzero area.")
        return loop if (area > 0) == positive else _reverse_loop(loop)
    direction = _validate_supported_conic_loop(loop)
    if (direction > 0) == positive:
        return loop
    return _reverse_loop(loop)


def _unit_tangent(curve: AbstractCurve, parameter: float, /) -> np.ndarray:
    derivative = np.asarray(
        jax.jacfwd(curve.evaluate)(jnp.asarray(parameter, dtype=jnp.float64)),
        dtype=np.float64,
    )
    scale = float(np.max(np.abs(derivative)))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("Profile segment tangents must be finite and nonzero.")
    scaled = derivative / scale
    length = float(np.linalg.norm(scaled))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("Profile segment tangents must be finite and nonzero.")
    return scaled / length


def _conic_turning_number(loop: ProfileLoop, /) -> int:
    tangent_pairs = []
    intrinsic_turns = []
    for index in range(len(loop.segments)):
        curve, first, last = _segment_curve(loop, index)
        tangent_pairs.append((_unit_tangent(curve, first), _unit_tangent(curve, last)))
        if isinstance(curve, LineCurve):
            intrinsic_turns.append(0.0)
            continue
        if isinstance(curve, BSplineCurve):
            # Exact hodograph turns; a straight spline contributes zero.
            intrinsic_turns.extend(bspline_turn_terms(curve))
            continue
        first_axis = np.asarray(curve.first_axis, dtype=np.float64)
        second_axis = np.asarray(curve.second_axis, dtype=np.float64)
        determinant = float(
            first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0]
        )
        if determinant == 0.0:
            raise ValueError("Conic profile axes must be independent.")
        intrinsic_turns.append((last - first) * (1.0 if determinant > 0.0 else -1.0))
    corner_turns = []
    for index, (_, end_tangent) in enumerate(tangent_pairs):
        start_tangent = tangent_pairs[(index + 1) % len(tangent_pairs)][0]
        cross = float(
            end_tangent[0] * start_tangent[1] - end_tangent[1] * start_tangent[0]
        )
        corner_turns.append(atan2(cross, float(end_tangent @ start_tangent)))
    terms = (*intrinsic_turns, *corner_turns)
    total = float(sum(terms))
    tolerance = 2048.0 * np.finfo(np.float64).eps * max(1, len(loop.segments))
    if abs(abs(total) - _TWO_PI) > tolerance:
        raise NotImplementedError(
            "Curved profile loops must have one non-self-intersecting turn."
        )
    direction = 1 if total > 0.0 else -1
    if any(
        direction * value < -tolerance for value in terms
    ) and not _curved_loop_is_simple(loop):
        raise NotImplementedError(
            "Curved profile loops must have one non-self-intersecting turn; "
            "nonconvex loops require an exact line/circle simplicity proof."
        )
    return direction


def _validate_polygon_loops(loops: tuple[ProfileLoop, ...], /) -> None:
    for loop in loops:
        count = len(loop.vertices)
        if count < 3:
            raise ValueError("A polygon profile requires at least three vertices.")
        for first in range(count):
            a, b = loop.vertices[first], loop.vertices[(first + 1) % count]
            if a == b:
                raise ValueError("Polygon profile edges must have positive length.")
            for second in range(first + 1, count):
                if (
                    second in (first, (first + 1) % count)
                    or first == (second + 1) % count
                ):
                    continue
                c, d = loop.vertices[second], loop.vertices[(second + 1) % count]
                if _segments_intersect(a, b, c, d):
                    raise ValueError("Polygon profile loops must be simple.")
    if len(loops) > 1:
        outer = loops[0].vertices
        for hole in loops[1:]:
            if not _inside_polygon(hole.vertices[0], outer):
                raise ValueError(
                    "Every polygon profile hole must lie inside the outer loop."
                )
            for first in range(len(hole.vertices)):
                a, b = (
                    hole.vertices[first],
                    hole.vertices[(first + 1) % len(hole.vertices)],
                )
                for second in range(len(outer)):
                    c, d = outer[second], outer[(second + 1) % len(outer)]
                    if _segments_intersect(a, b, c, d):
                        raise ValueError(
                            "Polygon profile holes must not touch the outer loop."
                        )
        for first_index, first_hole in enumerate(loops[1:]):
            for second_hole in loops[first_index + 2 :]:
                if _inside_polygon(
                    first_hole.vertices[0], second_hole.vertices
                ) or _inside_polygon(second_hole.vertices[0], first_hole.vertices):
                    raise ValueError("Polygon profile holes must be disjoint.")
                for first_edge in range(len(first_hole.vertices)):
                    a = first_hole.vertices[first_edge]
                    b = first_hole.vertices[(first_edge + 1) % len(first_hole.vertices)]
                    for second_edge in range(len(second_hole.vertices)):
                        c = second_hole.vertices[second_edge]
                        d = second_hole.vertices[
                            (second_edge + 1) % len(second_hole.vertices)
                        ]
                        if _segments_intersect(a, b, c, d):
                            raise ValueError("Polygon profile holes must be disjoint.")


def _inside_polygon(
    point: tuple[float, float], loop: tuple[tuple[float, float], ...], /
) -> bool:
    px, py = _fraction(point[0]), _fraction(point[1])
    inside = False
    for index, start in enumerate(loop):
        end = loop[(index + 1) % len(loop)]
        x0, y0 = _fraction(start[0]), _fraction(start[1])
        x1, y1 = _fraction(end[0]), _fraction(end[1])
        if (y0 > py) != (y1 > py):
            crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if px < crossing_x:
                inside = not inside
    return inside


def _arc_range(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    counterclockwise: bool,
    /,
) -> tuple[float, float]:
    first = atan2(start[1] - center[1], start[0] - center[0])
    last = atan2(end[1] - center[1], end[0] - center[0])
    if start == end:
        return first, first + (_TWO_PI if counterclockwise else -_TWO_PI)
    if counterclockwise:
        while last <= first:
            last += _TWO_PI
    else:
        while last >= first:
            last -= _TWO_PI
    if abs(last - first) > pi + 128.0 * np.finfo(np.float64).eps:
        raise ValueError("ProfileArc sweeps are limited to a half turn.")
    return first, last


def _validate_supported_conic_loop(loop: ProfileLoop, /) -> int:
    return _conic_turning_number(loop)


def _segment_curve(
    loop: ProfileLoop, index: int, /
) -> tuple[LineCurve | BSplineCurve | CircleCurve | EllipseCurve, float, float]:
    start = loop.vertices[index]
    end = loop.vertices[(index + 1) % len(loop.vertices)]
    segment = loop.segments[index]
    if isinstance(segment, ProfileLine):
        first_exact = tuple(Fraction.from_float(float(value)) for value in start)
        last_exact = tuple(Fraction.from_float(float(value)) for value in end)
        direction_exact = tuple(
            last - first for first, last in zip(first_exact, last_exact, strict=True)
        )
        return (
            LineCurve(
                tuple(float(value) for value in first_exact),
                tuple(float(value) for value in direction_exact),
                origin_exact=first_exact,
                direction_exact=direction_exact,
            ),
            0.0,
            1.0,
        )
    if isinstance(segment, BSplineCurve):
        if _profile_bspline_is_straight(segment, start, end):
            first, last = _profile_bspline_domain(segment, start, end)
        else:
            first, last = segment.parameter_domain
        return segment, first, last
    if isinstance(segment, ProfileArc):
        first, last = _arc_range(start, end, segment.center, segment.counterclockwise)
        start_radius = float(np.linalg.norm(np.subtract(start, segment.center)))
        end_radius = float(np.linalg.norm(np.subtract(end, segment.center)))
        scale = max(start_radius, end_radius)
        if scale == 0.0 or abs(start_radius - end_radius) > (
            128.0 * np.finfo(np.float64).eps * scale
        ):
            raise ValueError(
                "ProfileArc endpoints must lie on one positive-radius circle."
            )
        if segment.counterclockwise:
            return (
                CircleCurve(segment.center, (1.0, 0.0), (0.0, 1.0), start_radius),
                first,
                last,
            )
        # A clockwise arc runs forward on the mirrored chart angle, so every
        # traversed arc keeps an increasing edge parameter.
        return (
            CircleCurve(segment.center, (1.0, 0.0), (0.0, -1.0), start_radius),
            -first,
            -last,
        )
    curve = segment.curve
    return curve, segment.first, segment.last


def _extrusion_patch(curve: AbstractCurve, vector: np.ndarray, /) -> AbstractSurfacePatch:
    if isinstance(curve, LineCurve):
        origin_shadow = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(curve.origin)
        )
        direction_shadow = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(curve.direction)
        )
        if (
            curve.origin_exact != origin_shadow
            or curve.direction_exact != direction_shadow
        ):
            return ExtrusionSurface(curve, vector)
        return PlanePatch(curve.origin, curve.direction, vector)
    if isinstance(curve, ExactLineCurve):
        origin = np.asarray(curve.first, dtype=np.float64)
        direction = np.asarray(curve.last, dtype=np.float64) - origin
        origin_shadow = tuple(Fraction.from_float(float(value)) for value in origin)
        direction_shadow = tuple(Fraction.from_float(float(value)) for value in direction)
        exact_direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
        if curve.first_exact != origin_shadow or exact_direction != direction_shadow:
            return ExtrusionSurface(curve, vector)
        return PlanePatch(origin, direction, vector)
    if not isinstance(curve, CircleCurve):
        return ExtrusionSurface(curve, vector)
    first = curve.first_axis_exact
    second = curve.second_axis_exact
    axis = tuple(Fraction.from_float(float(value)) for value in vector)

    def dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    if dot(first, first) != 1 or dot(second, second) != 1 or dot(first, second) != 0:
        return ExtrusionSurface(curve, vector)
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    cross = (
        normal[1] * axis[2] - normal[2] * axis[1],
        normal[2] * axis[0] - normal[0] * axis[2],
        normal[0] * axis[1] - normal[1] * axis[0],
    )
    if any(value != 0 for value in cross):
        return ExtrusionSurface(curve, vector)
    return CylinderPatch(
        curve.center,
        curve.first_axis,
        curve.second_axis,
        vector,
        curve.radius,
        origin_exact=curve.center_exact,
        first_axis_exact=curve.first_axis_exact,
        second_axis_exact=curve.second_axis_exact,
        axis_exact=axis,
        radius_exact=curve.radius_exact,
    )


def _space_curve(curve: AbstractCurve, plane: ProfilePlane, /) -> AbstractCurve:
    if isinstance(curve, BSplineCurve):
        exact_controls = tuple(
            _profile_point_exact(plane, point) for point in curve.exact_control_points
        )
        spatial_points = np.asarray(
            tuple(tuple(float(value) for value in row) for row in exact_controls),
            dtype=np.float64,
        )
        return BSplineCurve(
            spatial_points,
            np.asarray(curve.weights),
            np.asarray(curve.knots),
            curve.degree,
            exact_control_points=exact_controls,
            exact_weights=curve.exact_weights,
            exact_knots=curve.exact_knots,
        )
    match curve:
        case LineCurve():
            origin_exact = _profile_point_exact(plane, curve.origin_exact)
            direction_exact = _profile_vector_exact(plane, curve.direction_exact)
            return LineCurve(
                tuple(float(value) for value in origin_exact),
                tuple(float(value) for value in direction_exact),
                origin_exact=origin_exact,
                direction_exact=direction_exact,
            )
        case CircleCurve():
            center_exact = _profile_point_exact(plane, curve.center_exact)
            first_exact = _profile_vector_exact(plane, curve.first_axis_exact)
            second_exact = _profile_vector_exact(plane, curve.second_axis_exact)
            return CircleCurve(
                tuple(float(value) for value in center_exact),
                tuple(float(value) for value in first_exact),
                tuple(float(value) for value in second_exact),
                float(curve.radius_exact),
                center_exact=center_exact,
                first_axis_exact=first_exact,
                second_axis_exact=second_exact,
                radius_exact=curve.radius_exact,
            )
        case EllipseCurve():
            center_exact = _profile_point_exact(plane, curve.center_exact)
            first_exact = _profile_vector_exact(plane, curve.first_axis_exact)
            second_exact = _profile_vector_exact(plane, curve.second_axis_exact)
            return EllipseCurve(
                tuple(float(value) for value in center_exact),
                tuple(float(value) for value in first_exact),
                tuple(float(value) for value in second_exact),
                float(curve.first_radius_exact),
                float(curve.second_radius_exact),
                center_exact=center_exact,
                first_axis_exact=first_exact,
                second_axis_exact=second_exact,
                first_radius_exact=curve.first_radius_exact,
                second_radius_exact=curve.second_radius_exact,
            )
        case ParabolaCurve(
            vertex=vertex, first_axis=first, second_axis=second, focal_length=focal
        ):
            return ParabolaCurve(
                plane.point(np.asarray(vertex)),
                plane.vector(np.asarray(first)),
                plane.vector(np.asarray(second)),
                focal,
            )
        case HyperbolaCurve(
            center=center,
            first_axis=first,
            second_axis=second,
            first_radius=ra,
            second_radius=rb,
        ):
            return HyperbolaCurve(
                plane.point(np.asarray(center)),
                plane.vector(np.asarray(first)),
                plane.vector(np.asarray(second)),
                ra,
                rb,
            )
        case _:
            raise TypeError(f"Unsupported profile curve carrier: {type(curve).__name__}.")


def _translated_curve(curve: AbstractCurve, vector: np.ndarray, /) -> AbstractCurve:
    if isinstance(curve, BSplineCurve):
        translation = tuple(Fraction.from_float(float(value)) for value in vector)
        exact_controls = tuple(
            tuple(value + delta for value, delta in zip(row, translation, strict=True))
            for row in curve.exact_control_points
        )
        return BSplineCurve(
            np.asarray(
                tuple(tuple(float(value) for value in row) for row in exact_controls),
                dtype=np.float64,
            ),
            np.asarray(curve.weights),
            np.asarray(curve.knots),
            curve.degree,
            exact_control_points=exact_controls,
            exact_weights=curve.exact_weights,
            exact_knots=curve.exact_knots,
        )
    match curve:
        case LineCurve():
            translation = tuple(Fraction.from_float(float(value)) for value in vector)
            origin_exact = tuple(
                value + delta
                for value, delta in zip(curve.origin_exact, translation, strict=True)
            )
            return LineCurve(
                tuple(float(value) for value in origin_exact),
                tuple(float(value) for value in curve.direction_exact),
                origin_exact=origin_exact,
                direction_exact=curve.direction_exact,
            )
        case CircleCurve():
            translation = tuple(Fraction.from_float(float(value)) for value in vector)
            center_exact = tuple(
                value + delta
                for value, delta in zip(curve.center_exact, translation, strict=True)
            )
            return CircleCurve(
                tuple(float(value) for value in center_exact),
                curve.first_axis,
                curve.second_axis,
                float(curve.radius_exact),
                center_exact=center_exact,
                first_axis_exact=curve.first_axis_exact,
                second_axis_exact=curve.second_axis_exact,
                radius_exact=curve.radius_exact,
            )
        case EllipseCurve():
            translation = tuple(Fraction.from_float(float(value)) for value in vector)
            center_exact = tuple(
                value + delta
                for value, delta in zip(curve.center_exact, translation, strict=True)
            )
            return EllipseCurve(
                tuple(float(value) for value in center_exact),
                curve.first_axis,
                curve.second_axis,
                float(curve.first_radius_exact),
                float(curve.second_radius_exact),
                center_exact=center_exact,
                first_axis_exact=curve.first_axis_exact,
                second_axis_exact=curve.second_axis_exact,
                first_radius_exact=curve.first_radius_exact,
                second_radius_exact=curve.second_radius_exact,
            )
        case ParabolaCurve(
            vertex=vertex, first_axis=first, second_axis=second, focal_length=focal
        ):
            return ParabolaCurve(np.asarray(vertex) + vector, first, second, focal)
        case HyperbolaCurve(
            center=center,
            first_axis=first,
            second_axis=second,
            first_radius=ra,
            second_radius=rb,
        ):
            return HyperbolaCurve(np.asarray(center) + vector, first, second, ra, rb)
        case _:
            raise TypeError(
                f"Unsupported translated curve carrier: {type(curve).__name__}."
            )


def _loop_edges(
    builder: _Builder, loop: ProfileLoop, plane: ProfilePlane, /
) -> tuple[tuple[int, float, float, AbstractCurve], ...]:
    exact_vertices = tuple(_profile_vertex_exact(plane, point) for point in loop.vertices)
    vertices = tuple(
        builder.vertex(
            tuple(float(value) for value in point),
            exact_point=point,
        )
        for point in exact_vertices
    )
    result = []
    for index in range(len(loop.vertices)):
        curve2d, first, last = _segment_curve(loop, index)
        curve3d = _space_curve(curve2d, plane)
        endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None)
        if len(vertices) > 1 and isinstance(curve3d, (CircleCurve, EllipseCurve)):
            next_index = (index + 1) % len(vertices)
            points = (exact_vertices[index], exact_vertices[next_index])
            first_root = ConicParameterRoot.from_exact_point(curve3d, points[0], first)
            last_root = ConicParameterRoot.from_exact_point(curve3d, points[1], last)
            if (
                first_root is not None
                and last_root is not None
                and first_root.exact_point() == points[0]
                and last_root.exact_point() == points[1]
                and first_root.parameter_lower <= first <= first_root.parameter_upper
                and last_root.parameter_lower <= last <= last_root.parameter_upper
            ):
                endpoint_roots = (
                    ConicParameterEndpoint(first_root, curve3d, None, points[0]),
                    ConicParameterEndpoint(last_root, curve3d, None, points[1]),
                )
                for vertex, root in (
                    (vertices[index], first_root),
                    (vertices[next_index], last_root),
                ):
                    if builder.vertex_roots[vertex] is None:
                        builder.vertex_roots[vertex] = BRepVertexRoot(
                            BRepConicVertex(root)
                        )
        edge = builder.edge(
            curve3d,
            first,
            last,
            vertices[index],
            vertices[(index + 1) % len(vertices)],
            endpoint_roots=endpoint_roots,
        )
        result.append((edge, first, last, curve2d))
    # A rooted vertex must bind every incident edge, not only its conic.
    # Prove the other carrier's exact parameter point before attaching it.
    from ._intersection_curve import exact_curve_parameter_point

    for edge, first, last, _ in result:
        curve = builder.curves[builder.edge_curves[edge]]
        roots = list(builder.edge_endpoint_roots[edge])
        for endpoint, (vertex, parameter) in enumerate(
            zip(builder.edge_vertices[edge], (first, last), strict=True)
        ):
            binding = builder.vertex_roots[vertex]
            if roots[endpoint] is not None or binding is None:
                continue
            if not isinstance(binding.primary, BRepConicVertex):
                raise ValueError(
                    "A rooted profile vertex requires its conic source witness."
                )
            exact_parameter = Fraction.from_float(parameter)
            point = exact_curve_parameter_point(curve, exact_parameter)
            if point is None or tuple(point) != binding.primary.point_exact:
                builder.vertex_roots[vertex] = None
            else:
                roots[endpoint] = ConicParameterEndpoint(
                    binding.primary.root,
                    curve,
                    exact_parameter,
                    binding.primary.point_exact,
                )
        builder.edge_endpoint_roots[edge] = (roots[0], roots[1])
    for edge, _, _, _ in result:
        roots = builder.edge_endpoint_roots[edge]
        first_vertex, last_vertex = builder.edge_vertices[edge]
        builder.edge_endpoint_roots[edge] = (
            roots[0] if builder.vertex_roots[first_vertex] is not None else None,
            roots[1] if builder.vertex_roots[last_vertex] is not None else None,
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _ProfileEdge:
    pcurve: AbstractCurve
    first: float
    last: float
    start: tuple[float, float]
    end: tuple[float, float]
    turn: float
    radius: float


def _profile_edge(
    start: tuple[float, float], end: tuple[float, float], segment: ProfileSegment, /
) -> _ProfileEdge:
    if isinstance(segment, ProfileLine):
        return _ProfileEdge(
            LineCurve(start, np.subtract(end, start)), 0.0, 1.0, start, end, 0.0, 0.0
        )
    if isinstance(segment, BSplineCurve):
        if not _profile_bspline_is_straight(segment, start, end):
            first, last = segment.parameter_domain
            return _ProfileEdge(segment, first, last, start, end, 0.0, 0.0)
        _profile_bspline_domain(segment, start, end)
        return _ProfileEdge(
            LineCurve(start, np.subtract(end, start)),
            0.0,
            1.0,
            start,
            end,
            0.0,
            0.0,
        )
    if isinstance(segment, ProfileArc):
        first, last = _arc_range(start, end, segment.center, segment.counterclockwise)
        radius = float(np.linalg.norm(np.subtract(start, segment.center)))
        return _ProfileEdge(
            CircleCurve(segment.center, (1.0, 0.0), (0.0, 1.0), radius),
            first,
            last,
            start,
            end,
            1.0 if segment.counterclockwise else -1.0,
            radius,
        )
    if isinstance(segment, ProfileCurveUse):
        first_axis = np.asarray(segment.curve.first_axis, dtype=np.float64)
        second_axis = np.asarray(segment.curve.second_axis, dtype=np.float64)
        turn = float(
            np.sign(first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0])
        )
        radius = (
            float(segment.curve.radius) if isinstance(segment.curve, CircleCurve) else 0.0
        )
        return _ProfileEdge(
            segment.curve,
            segment.first,
            segment.last,
            start,
            end,
            turn,
            radius,
        )
    raise TypeError(f"Unsupported profile edge segment: {type(segment).__name__}.")


def _profile_radical_sign(
    a: ExactScalar, b: ExactScalar, radicand: ExactScalar, /
) -> int:
    """Sign of a + b sqrt(radicand), with exact algebraic coefficients."""
    if radicand < 0:
        raise ValueError("A profile contact radicand must be nonnegative.")
    if b == 0 or radicand == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if (a > 0) == (b > 0):
        return 1 if a > 0 else -1
    comparison = a * a - b * b * radicand
    return ((comparison > 0) - (comparison < 0)) * (1 if a > 0 else -1)


def _line_arc_contact(
    line: _ProfileEdge,
    arc: _ProfileEdge,
    allowed: frozenset[tuple[Fraction, Fraction]],
    /,
) -> bool:
    radical_sign = _profile_radical_sign
    if not isinstance(line.pcurve, LineCurve) or not isinstance(arc.pcurve, CircleCurve):
        return False
    start = tuple(Fraction(float(value)) for value in line.start)
    end = tuple(Fraction(float(value)) for value in line.end)
    direction = tuple(b - a for a, b in zip(start, end, strict=True))
    center = arc.pcurve.center_exact
    radius_squared = arc.pcurve.radius_exact * arc.pcurve.radius_exact
    relative = tuple(a - b for a, b in zip(start, center, strict=True))
    quadratic = sum((value * value for value in direction), Fraction(0))
    if quadratic == 0:
        return False
    linear = 2 * sum(
        (a * b for a, b in zip(relative, direction, strict=True)), Fraction(0)
    )
    constant = sum((value * value for value in relative), Fraction(0)) - radius_squared
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < 0:
        return False
    roots = (
        ((-linear / (2 * quadratic), Fraction(0)),)
        if discriminant == 0
        else (
            (-linear / (2 * quadratic), -1 / (2 * quadratic)),
            (-linear / (2 * quadratic), 1 / (2 * quadratic)),
        )
    )

    arc_start = tuple(Fraction(float(value)) for value in arc.start)
    arc_end = tuple(Fraction(float(value)) for value in arc.end)
    start_radius = tuple(a - b for a, b in zip(arc_start, center, strict=True))
    end_radius = tuple(a - b for a, b in zip(arc_end, center, strict=True))
    direction_sign = 1 if arc.turn > 0 else -1
    minor = (
        direction_sign
        * (start_radius[0] * end_radius[1] - start_radius[1] * end_radius[0])
        >= 0
    )
    for parameter_base, parameter_radical in roots:
        if (
            radical_sign(parameter_base, parameter_radical, discriminant) < 0
            or radical_sign(parameter_base - 1, parameter_radical, discriminant) > 0
        ):
            continue
        base = tuple(
            start[axis] + parameter_base * direction[axis] - center[axis]
            for axis in range(2)
        )
        radical = tuple(parameter_radical * value for value in direction)
        if arc.start != arc.end:
            cross_start_base = start_radius[0] * base[1] - start_radius[1] * base[0]
            cross_start_radical = (
                start_radius[0] * radical[1] - start_radius[1] * radical[0]
            )
            cross_base_end = base[0] * end_radius[1] - base[1] * end_radius[0]
            cross_radical_end = radical[0] * end_radius[1] - radical[1] * end_radius[0]
            before = direction_sign * radical_sign(
                cross_start_base, cross_start_radical, discriminant
            )
            after = direction_sign * radical_sign(
                cross_base_end, cross_radical_end, discriminant
            )
            in_arc = before >= 0 and after >= 0 if minor else before >= 0 or after >= 0
            if not in_arc:
                continue
        is_allowed_vertex = any(
            all(
                radical_sign(
                    center[axis] + base[axis] - vertex[axis],
                    radical[axis],
                    discriminant,
                )
                == 0
                for axis in range(2)
            )
            for vertex in allowed
        )
        if not is_allowed_vertex:
            return True
    return False


def _arc_circle(
    edge: _ProfileEdge, /
) -> tuple[tuple[ExactScalar, ExactScalar], ExactScalar]:
    if not isinstance(edge.pcurve, CircleCurve):
        raise NotImplementedError(
            "Curved profile relations currently require exact circles."
        )
    center = edge.pcurve.center_exact
    radius = edge.pcurve.radius_exact
    return (center[0], center[1]), radius * radius


def _radical_is_vertex(
    point: tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]],
    radicand: ExactScalar,
    vertices: frozenset[tuple[Fraction, Fraction]],
    /,
) -> bool:
    radical_sign = _profile_radical_sign

    base, radical = point
    return any(
        all(
            radical_sign(base[axis] - vertex[axis], radical[axis], radicand) == 0
            for axis in range(2)
        )
        for vertex in vertices
    )


def _arc_contains_radical(
    edge: _ProfileEdge,
    point: tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]],
    radicand: ExactScalar,
    /,
) -> bool:
    radical_sign = _profile_radical_sign

    if edge.start == edge.end:
        return True
    center, _ = _arc_circle(edge)
    start = tuple(
        Fraction(float(value)) - center[axis] for axis, value in enumerate(edge.start)
    )
    end = tuple(
        Fraction(float(value)) - center[axis] for axis, value in enumerate(edge.end)
    )
    base, radical = point
    relative = tuple(base[axis] - center[axis] for axis in range(2))
    turn = 1 if edge.turn > 0 else -1
    cross_start_relative = start[0] * relative[1] - start[1] * relative[0]
    cross_start_radical = start[0] * radical[1] - start[1] * radical[0]
    cross_relative_end = relative[0] * end[1] - relative[1] * end[0]
    cross_radical_end = radical[0] * end[1] - radical[1] * end[0]
    before = turn * radical_sign(cross_start_relative, cross_start_radical, radicand)
    after = turn * radical_sign(cross_relative_end, cross_radical_end, radicand)
    minor = turn * (start[0] * end[1] - start[1] * end[0]) >= 0
    return before >= 0 and after >= 0 if minor else before >= 0 or after >= 0


def _arc_arc_contact(
    first: _ProfileEdge,
    second: _ProfileEdge,
    allowed: frozenset[tuple[Fraction, Fraction]],
    /,
) -> bool:
    a, radius_a = _arc_circle(first)
    b, radius_b = _arc_circle(second)
    delta = (b[0] - a[0], b[1] - a[1])
    distance_squared = delta[0] ** 2 + delta[1] ** 2
    if distance_squared == 0:
        if radius_a != radius_b:
            return False
        if first.start == first.end or second.start == second.end:
            return True
        for edge, other in ((first, second), (second, first)):
            for vertex in (edge.start, edge.end):
                point = (Fraction(float(vertex[0])), Fraction(float(vertex[1])))
                if point not in allowed and _arc_contains_radical(
                    other, (point, (Fraction(0), Fraction(0))), Fraction(0)
                ):
                    return True
        return False
    along = (radius_a - radius_b + distance_squared) / (2 * distance_squared)
    radicand = radius_a / distance_squared - along * along
    if radicand < 0:
        return False
    base = (a[0] + along * delta[0], a[1] + along * delta[1])
    for sign in (1,) if radicand == 0 else (-1, 1):
        point = (base, (-sign * delta[1], sign * delta[0]))
        if (
            _arc_contains_radical(first, point, radicand)
            and _arc_contains_radical(second, point, radicand)
            and not _radical_is_vertex(point, radicand, allowed)
        ):
            return True
    return False


def _profile_circle(loop: ProfileLoop, /) -> _ProfileEdge | None:
    if len(loop.vertices) != 1:
        return None
    segment = loop.segments[0]
    if isinstance(segment, ProfileArc):
        return _profile_edge(loop.vertices[0], loop.vertices[0], segment)
    if isinstance(segment, ProfileCurveUse) and isinstance(segment.curve, CircleCurve):
        return _profile_edge(loop.vertices[0], loop.vertices[0], segment)
    return None


def _profile_loop_edges(loop: ProfileLoop, /) -> tuple[_ProfileEdge, ...]:
    return tuple(
        _profile_edge(
            loop.vertices[index],
            loop.vertices[(index + 1) % len(loop.vertices)],
            segment,
        )
        for index, segment in enumerate(loop.segments)
    )


def _profile_loop_contains(loop: ProfileLoop, point: tuple[float, float], /) -> bool:
    circle = _profile_circle(loop)
    if circle is not None:
        center, radius_squared = _arc_circle(circle)
        delta = tuple(Fraction(float(point[axis])) - center[axis] for axis in range(2))
        return delta[0] * delta[0] + delta[1] * delta[1] < radius_squared
    if _profile_loop_is_polygon(loop):
        return _inside_polygon(point, loop.vertices)
    raise NotImplementedError(
        "Curved profile containment requires a circle or exact polygon loop."
    )


def _profile_loops_intersect(first: ProfileLoop, second: ProfileLoop, /) -> bool:
    first_circle, second_circle = _profile_circle(first), _profile_circle(second)
    if first_circle is not None and second_circle is not None:
        return _arc_arc_contact(first_circle, second_circle, frozenset())
    if first_circle is not None:
        first_edges, second_edges = (first_circle,), _profile_loop_edges(second)
    elif second_circle is not None:
        first_edges, second_edges = _profile_loop_edges(first), (second_circle,)
    else:
        first_edges, second_edges = (
            _profile_loop_edges(first),
            _profile_loop_edges(second),
        )
    for first_edge in first_edges:
        for second_edge in second_edges:
            first_arc = isinstance(first_edge.pcurve, CircleCurve)
            second_arc = isinstance(second_edge.pcurve, CircleCurve)
            if first_arc and second_arc:
                if _arc_arc_contact(first_edge, second_edge, frozenset()):
                    return True
            elif first_arc and isinstance(second_edge.pcurve, LineCurve):
                if _line_arc_contact(second_edge, first_edge, frozenset()):
                    return True
            elif second_arc and isinstance(first_edge.pcurve, LineCurve):
                if _line_arc_contact(first_edge, second_edge, frozenset()):
                    return True
            elif isinstance(first_edge.pcurve, LineCurve) and isinstance(
                second_edge.pcurve, LineCurve
            ):
                if _segments_intersect(
                    first_edge.start, first_edge.end, second_edge.start, second_edge.end
                ):
                    return True
            else:
                raise NotImplementedError(
                    "Profile relations require exact line/circle carriers."
                )
    return False


def _validate_profile_loop_relations(loops: tuple[ProfileLoop, ...], /) -> None:
    for loop in loops:
        if _profile_loop_is_polygon(loop):
            _validate_polygon_loops((loop,))
        elif _profile_circle(loop) is None:
            raise NotImplementedError(
                "Curved profile holes require a single exact circular loop."
            )
    outer = loops[0]
    for hole in loops[1:]:
        if _profile_loops_intersect(outer, hole):
            raise ValueError("Profile holes and outer loops must not touch or cross.")
        circle = _profile_circle(hole)
        if not _profile_loop_contains(
            outer,
            circle.start if circle is not None else hole.vertices[0],
        ):
            raise ValueError(
                "Every profile hole must lie strictly inside the outer loop."
            )
    for index, first_hole in enumerate(loops[1:]):
        for second_hole in loops[index + 2 :]:
            if _profile_loops_intersect(first_hole, second_hole):
                raise ValueError("Profile holes must not touch or cross each other.")
            first_circle = _profile_circle(first_hole)
            second_circle = _profile_circle(second_hole)
            first_point = (
                first_circle.start if first_circle is not None else first_hole.vertices[0]
            )
            second_point = (
                second_circle.start
                if second_circle is not None
                else second_hole.vertices[0]
            )
            if _profile_loop_contains(first_hole, second_point) or _profile_loop_contains(
                second_hole, first_point
            ):
                raise ValueError("Profile holes must not be nested.")


def _curved_loop_is_simple(loop: ProfileLoop, /) -> bool:
    """Prove that one line/circle profile loop meets itself only at its junctions.

    Exact rational line/circle contacts decide every segment pair; adjacent
    segments may meet only at their shared profile vertices.
    """
    edges = _profile_loop_edges(loop)
    vertices = tuple(
        (_fraction(point[0]), _fraction(point[1])) for point in loop.vertices
    )
    count = len(edges)
    splines = any(isinstance(edge.pcurve, BSplineCurve) for edge in edges)
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            first, second = edges[first_index], edges[second_index]
            if isinstance(first.pcurve, BSplineCurve) or isinstance(
                second.pcurve, BSplineCurve
            ):
                continue
            shared = frozenset(
                vertices[index]
                for index in (first_index, (first_index + 1) % count)
                if index in (second_index, (second_index + 1) % count)
            )
            first_arc = isinstance(first.pcurve, CircleCurve)
            second_arc = isinstance(second.pcurve, CircleCurve)
            first_line = isinstance(first.pcurve, LineCurve)
            second_line = isinstance(second.pcurve, LineCurve)
            if first_arc and second_arc:
                contact = _arc_arc_contact(first, second, shared)
            elif first_arc and second_line:
                contact = _line_arc_contact(second, first, shared)
            elif first_line and second_arc:
                contact = _line_arc_contact(first, second, shared)
            elif first_line and second_line:
                contact = _line_pair_contact(first, second, shared)
            else:
                return False
            if contact:
                return False
    if splines:
        pieces = []
        for edge in edges:
            if isinstance(edge.pcurve, BSplineCurve):
                pieces.append(bspline_pieces(edge.pcurve))
            elif isinstance(edge.pcurve, LineCurve):
                pieces.append(line_piece(edge.start, edge.end))
            elif isinstance(edge.pcurve, CircleCurve):
                pieces.append(
                    arc_pieces(
                        tuple(float(value) for value in np.asarray(edge.pcurve.center)),
                        float(edge.pcurve.radius),
                        edge.start,
                        edge.end,
                        edge.turn > 0,
                        edge.last - edge.first,
                    )
                )
            else:
                return False
        # Pairs with a B-spline piece are decided or refused here.
        certify_profile_contacts(pieces)
    return True


def _line_pair_contact(
    first: _ProfileEdge,
    second: _ProfileEdge,
    shared: frozenset[tuple[Fraction, Fraction]],
    /,
) -> bool:
    if not shared:
        return _segments_intersect(first.start, first.end, second.start, second.end)
    # Two segments sharing an endpoint meet elsewhere only by folding back
    # along one common line.
    a = tuple(_fraction(value) for value in first.start)
    b = tuple(_fraction(value) for value in first.end)
    c = tuple(_fraction(value) for value in second.start)
    d = tuple(_fraction(value) for value in second.end)
    u = (b[0] - a[0], b[1] - a[1])
    v = (d[0] - c[0], d[1] - c[1])
    if u[0] * v[1] - u[1] * v[0] != 0:
        return len(shared) > 1
    if len(shared) > 1:
        return True
    pivot = next(iter(shared))
    outward_first = u if a == pivot else (-u[0], -u[1])
    outward_second = v if c == pivot else (-v[0], -v[1])
    return outward_first[0] * outward_second[0] + outward_first[1] * outward_second[1] > 0


def _profile_endpoint_gap_rows(
    loops: tuple[ProfileLoop, ...],
    declared: tuple[tuple[tuple[float, float], ...], ...] | None,
    /,
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    from ._conic_gaps import endpoint_gap_bounds
    from ._intersection import NativePeriodEndpoint

    if declared is not None and (
        len(declared) != len(loops)
        or any(
            len(rows) != len(loop.segments)
            for rows, loop in zip(declared, loops, strict=True)
        )
    ):
        raise ValueError(
            "PlanarProfile.endpoint_gap_bounds must align with every loop edge."
        )
    certified = []

    def exact_profile_endpoint_root(
        curve: AbstractCurve, point: tuple[float, float], parameter: float, /
    ) -> ConicParameterRoot | None:
        if not isinstance(curve, (CircleCurve, EllipseCurve)):
            return None
        exact_point = tuple(Fraction.from_float(float(value)) for value in point)
        if len(exact_point) != 2:
            return None
        first_axis = curve.first_axis_exact
        second_axis = curve.second_axis_exact

        def dot(
            first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
        ) -> ExactScalar:
            return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

        if (
            dot(first_axis, first_axis) != 1
            or dot(second_axis, second_axis) != 1
            or dot(first_axis, second_axis) != 0
        ):
            return None
        delta = tuple(
            value - center
            for value, center in zip(exact_point, curve.center_exact, strict=True)
        )
        first_radius, second_radius = (
            (curve.radius_exact, curve.radius_exact)
            if isinstance(curve, CircleCurve)
            else (curve.first_radius_exact, curve.second_radius_exact)
        )
        cosine = dot(delta, first_axis) / first_radius
        sine = dot(delta, second_axis) / second_radius
        principal = atan2(float(sine), float(cosine))
        if principal < 0.0:
            principal += _TWO_PI
        turns = int(round((parameter - principal) / _TWO_PI))
        try:
            root = ConicParameterRoot(curve, cosine, sine, turns=turns)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if (
            root.exact_point() != exact_point
            or not root.parameter_lower <= parameter <= root.parameter_upper
        ):
            return None
        return root

    for loop_index, loop in enumerate(loops):
        rows = []
        for edge, segment in enumerate(loop.segments):
            curve, first, last = _segment_curve(loop, edge)
            start = loop.vertices[edge]
            end = loop.vertices[(edge + 1) % len(loop.vertices)]
            profile_roots = (
                exact_profile_endpoint_root(curve, start, first),
                exact_profile_endpoint_root(curve, end, last),
            )
            roots: tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None] = (
                None,
                None,
            )
            if len(loop.vertices) == 1 and curve.native_period_symbol() == "two_pi":
                rational = Fraction.from_float(float(first))
                period_roots = (
                    NativePeriodEndpoint(curve, rational=rational, turns=0),
                    NativePeriodEndpoint(curve, rational=rational, turns=1),
                )
                lower, upper = period_roots[1].parameter_enclosure()
                if not lower <= last <= upper:
                    raise ValueError(
                        "A closed conic profile must realize its exact native full turn."
                    )
                first_root, last_root = profile_roots
                rooted_turn = (
                    first_root is not None
                    and last_root is not None
                    and first_root.cosine_exact == last_root.cosine_exact
                    and first_root.sine_exact == last_root.sine_exact
                    and last_root.turns - first_root.turns == 1
                )
                if not rooted_turn:
                    roots = period_roots
            actual_values = list(
                endpoint_gap_bounds(
                    curve,
                    (first, last),
                    np.asarray((start, end), dtype=np.float64),
                    endpoint_roots=roots,
                )
            )
            for endpoint, root in enumerate(profile_roots):
                if root is not None:
                    actual_values[endpoint] = 0.0
            actual = actual_values[0], actual_values[1]
            if declared is not None:
                limits = declared[loop_index][edge]
                if len(limits) != 2 or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not isfinite(float(value))
                    or value < 0.0
                    for value in limits
                ):
                    raise ValueError(
                        "Profile endpoint gap limits must be finite nonnegative pairs."
                    )
                if any(
                    value > limit for value, limit in zip(actual, limits, strict=True)
                ):
                    raise ValueError(
                        "Profile endpoint gap limits do not cover the recomputed source gaps."
                    )
            rows.append(actual)
        certified.append(tuple(rows))
    result = tuple(certified)
    return (
        result if any(bound for rows in result for row in rows for bound in row) else None
    )


def _profile_trim(profile: PlanarProfile, /) -> TrimDomain | None:
    if any(
        not all(isinstance(segment, ProfileLine) for segment in loop.segments)
        for loop in profile.loops
    ):
        return None
    outer = np.asarray(profile.outer.vertices, dtype=np.float64)
    holes = tuple(np.asarray(loop.vertices, dtype=np.float64) for loop in profile.holes)
    lower = np.min(np.concatenate((outer, *holes), axis=0), axis=0)
    upper = np.max(np.concatenate((outer, *holes), axis=0), axis=0)
    extent = upper - lower
    return TrimDomain(
        (outer - lower) / extent, tuple((hole - lower) / extent for hole in holes)
    )


def _is_chart_line(curve: AbstractCurve, /) -> bool:
    """Accept chart lines, including lines on an integer native-period sheet."""
    from ._boolean_core import PeriodicPCurve

    while isinstance(curve, PeriodicPCurve):
        curve = curve.source_curve
    return isinstance(curve, (LineCurve, ExactLineCurve))


def _line_trim_domain(
    geometry: BRepGeometry, face: int, bounds: np.ndarray, /
) -> TrimDomain | None:
    loops: list[np.ndarray] = []
    lower, extent = bounds[0], bounds[1] - bounds[0]
    for loop in geometry.face_loops[face]:
        points = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if edge < 0 or edge >= len(geometry.edge_curves):
                raise ValueError("A face coedge references an absent source edge.")
            curve = geometry.pcurves[coedge]
            if not _is_chart_line(curve):
                return None
            first, last = np.asarray(geometry.edge_ranges)[edge]
            parameters = (
                (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)
            )
            values = np.asarray(curve.evaluate(jnp.asarray(parameters)), dtype=np.float64)
            if not points:
                points.append(values[0])
            points.append(values[1])
        if len(points) > 1 and np.array_equal(points[0], points[-1]):
            points.pop()
        loops.append((np.asarray(points) - lower) / extent)
    if not loops or any(loop.shape[0] < 3 for loop in loops):
        return None
    return TrimDomain(loops[0], tuple(loops[1:]))


def _face_trim_domain(
    geometry: BRepGeometry, face: int, bounds: np.ndarray, /
) -> TrimDomain | None:
    loops = geometry.face_loops[face]
    if all(
        isinstance(geometry.pcurves[coedge], (LineCurve, ExactLineCurve))
        for loop in loops
        for coedge in loop
    ):
        return _line_trim_domain(geometry, face, bounds)
    from ._intersection_curve import CurveTrimSegment

    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    curve_loops = []
    for loop in loops:
        segments = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first, last = (float(value) for value in ranges[edge])
            first_root, last_root = geometry.coedge_endpoint_roots[coedge]
            segments.append(
                CurveTrimSegment(
                    geometry.pcurves[coedge],
                    first,
                    last,
                    reversed=geometry.coedge_senses[coedge] < 0,
                    first_root=first_root,
                    last_root=last_root,
                )
            )
        curve_loops.append(tuple(segments))
    return TrimDomain.from_curve_loops(tuple(curve_loops))


def _check_solid_edge_balance(geometry: BRepGeometry, orientation: np.ndarray, /) -> None:
    edge_count = geometry.topology.num_edges
    for solid, shell_indices in enumerate(geometry.solid_shells):
        balance = [0] * edge_count
        uses = [0] * edge_count
        for shell in shell_indices:
            for face, shell_sign in zip(
                geometry.shell_faces[shell],
                geometry.shell_orientations[shell],
                strict=True,
            ):
                face_sign = int(orientation[face]) * shell_sign
                for loop in geometry.face_loops[face]:
                    for coedge in loop:
                        edge = geometry.coedge_edges[coedge]
                        balance[edge] += geometry.coedge_senses[coedge] * face_sign
                        uses[edge] += 1
        for edge, curve_index in enumerate(geometry.edge_curves):
            if (
                curve_index >= 0
                and uses[edge] != 0
                and (uses[edge] != 2 or balance[edge] != 0)
            ):
                raise ValueError(
                    f"Solid {solid} has an open or inconsistently oriented source shell at edge {edge}."
                )


def assemble_brep_model(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: ConvertibleToArray,
    orientation: ConvertibleToArray,
    physical_tags: tuple[str, ...],
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str,
    source_format: str = "native",
    source_digest: str | None = None,
    import_policy_id: str = _NATIVE_CONSTRUCTION_POLICY_ID,
    converted_surface_count: int = 0,
    curve_surface_tolerance: float = 0.0,
    curve_surface_scale: float = 1.0,
    tessellation: BRepTessellationPolicy | None = None,
    trim_domains: tuple[TrimDomain | None, ...] | None = None,
) -> BRepModel:
    """Publish native source carriers and realize triangles only when requested."""
    if not isinstance(geometry, BRepGeometry):
        raise TypeError("geometry must be a BRepGeometry.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be nonempty.")
    if tessellation is not None and not isinstance(tessellation, BRepTessellationPolicy):
        raise TypeError("tessellation must be a BRepTessellationPolicy or None.")
    bounds = np.asarray(parameter_bounds, dtype=np.float64)
    signs = np.asarray(orientation, dtype=np.float64)
    if signs.shape != (len(patches),):
        raise ValueError("orientation must contain one sign per face.")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("orientation must contain only -1 and 1.")
    if geometry.solid_shells:
        _check_solid_edge_balance(geometry, signs)
    provisional_digest = source_digest or canonical_fingerprint(
        {
            "kind": "native-brep-origin",
            "geometry": geometry.geometry_id,
            "patches": tuple(type(patch).__qualname__ for patch in patches),
            "bounds": bounds,
            "orientation": signs,
            "tags": physical_tags,
        }
    )
    report = BRepImportReport(
        source_id=source_id,
        source_digest=provisional_digest,
        source_format=source_format,
        coordinate_contract=coordinate_contract,
        import_policy_id=import_policy_id,
        num_solids=geometry.topology.num_solids,
        num_faces=geometry.topology.num_faces,
        num_edges=geometry.topology.num_edges,
        num_vertices=geometry.topology.num_vertices,
        converted_surface_count=converted_surface_count,
        curve_surface_tolerance=curve_surface_tolerance,
        curve_surface_scale=curve_surface_scale,
    )
    trims = (
        tuple(
            _face_trim_domain(geometry, face, bounds[face])
            for face in range(len(patches))
        )
        if trim_domains is None
        else trim_domains
    )
    model = BRepModel(
        geometry=geometry,
        patches=patches,
        parameter_bounds=bounds,
        orientation=signs,
        trim_domains=trims,
        physical_tags=physical_tags,
        coordinate_contract=coordinate_contract,
        report=report,
    )
    if tessellation is None or not patches:
        return model
    from ._tessellation import tessellate_brep_model

    realization, certificate = tessellate_brep_model(model, tessellation)
    return BRepModel(
        geometry=geometry,
        patches=patches,
        parameter_bounds=bounds,
        orientation=signs,
        trim_domains=trims,
        physical_tags=physical_tags,
        coordinate_contract=coordinate_contract,
        report=report,
        tessellation=realization,
        tessellation_certificate=certificate,
        binding_proof=model.binding_proof,
    )


def brep_trim_domain(
    geometry: BRepGeometry,
    face: int,
    patch: AbstractSurfacePatch,
    /,
    *,
    tolerance: float,
    parameter_bounds: ConvertibleToArray | None = None,
) -> TrimDomain:
    """Return an exact polygonal chart trim where every source coedge is linear."""
    if not isinstance(geometry, BRepGeometry):
        raise TypeError("geometry must be a BRepGeometry.")
    if isinstance(face, bool) or not isinstance(face, Integral):
        raise TypeError("face must be an integer.")
    face_ = int(face)
    if not 0 <= face_ < len(geometry.face_loops):
        raise ValueError("face references an absent B-Rep face.")
    if not isinstance(patch, AbstractSurfacePatch):
        raise TypeError("patch must be an AbstractSurfacePatch.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    del tolerance_
    bounds = (
        np.asarray(parameter_bounds, dtype=np.float64)
        if parameter_bounds is not None
        else np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )
    trim = _line_trim_domain(geometry, face_, bounds)
    if trim is None:
        raise NotImplementedError("Curved trim domains require a certified chart cover.")
    return trim


def _profile_cap(
    builder: _Builder,
    profile: PlanarProfile,
    loops: tuple[tuple[tuple[int, float, float, AbstractCurve], ...], ...],
    patch: PlanePatch,
    orientation: int,
    tag: str,
    /,
) -> int:
    coedge_loops = []
    for edges in loops:
        coedge_loop = []
        for edge, first, last, curve in edges:
            edge_roots = builder.edge_endpoint_roots[edge]

            def coedge_root(root: RootEndpoint | None) -> ConicParameterEndpoint | None:
                if isinstance(root, ConicParameterEndpoint):
                    return ConicParameterEndpoint(
                        root.root, curve, root.parameter_exact, root.point_exact
                    )
                return None

            coedge_roots = (coedge_root(edge_roots[0]), coedge_root(edge_roots[1]))
            coedge = builder.coedge(edge, 1, curve, endpoint_roots=coedge_roots)
            coedge_loop.append(coedge)
            if (
                builder.edge_vertices[edge][0] == builder.edge_vertices[edge][1]
                and curve.native_period_symbol() == "two_pi"
            ):
                builder.native_curve_phase(
                    edge,
                    coedge=coedge,
                    rational=Fraction.from_float(first),
                )
        coedge_loops.append(coedge_loop)
    bounds = _profile_bounds(profile)
    return builder.face(patch, bounds, orientation, tag, coedge_loops)


def _profile_bounds(profile: PlanarProfile, /) -> np.ndarray:
    points = [
        np.asarray(point, dtype=np.float64)
        for loop in profile.loops
        for point in loop.vertices
    ]
    for loop in profile.loops:
        for index, segment in enumerate(loop.segments):
            if isinstance(segment, BSplineCurve):
                points.extend(np.asarray(segment.control_points, dtype=np.float64))
            if isinstance(segment, ProfileArc):
                radius = float(
                    np.linalg.norm(np.subtract(loop.vertices[index], segment.center))
                )
                points.extend(
                    (
                        np.asarray((segment.center[0] - radius, segment.center[1])),
                        np.asarray((segment.center[0] + radius, segment.center[1])),
                        np.asarray((segment.center[0], segment.center[1] - radius)),
                        np.asarray((segment.center[0], segment.center[1] + radius)),
                    )
                )
            elif isinstance(segment, ProfileCurveUse):
                center = np.asarray(segment.curve.center, dtype=np.float64)
                first_axis = np.asarray(segment.curve.first_axis, dtype=np.float64)
                second_axis = np.asarray(segment.curve.second_axis, dtype=np.float64)
                if isinstance(segment.curve, CircleCurve):
                    first_radius = second_radius = float(segment.curve.radius)
                else:
                    first_radius = float(segment.curve.first_radius)
                    second_radius = float(segment.curve.second_radius)
                extents = np.asarray(
                    (
                        np.hypot(
                            first_radius * first_axis[0],
                            second_radius * second_axis[0],
                        ),
                        np.hypot(
                            first_radius * first_axis[1],
                            second_radius * second_axis[1],
                        ),
                    ),
                    dtype=np.float64,
                )
                points.extend((center - extents, center + extents))
    values = np.stack(points)
    lower, upper = np.min(values, axis=0), np.max(values, axis=0)
    if np.any(upper <= lower):
        raise ValueError(
            "Profile bounds must have positive extent in both chart coordinates."
        )
    return np.stack((lower, upper))


def brep_planar_face(
    profile: PlanarProfile,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-planar-face",
) -> BRepModel:
    if not isinstance(profile, PlanarProfile):
        raise TypeError("profile must be a PlanarProfile.")
    builder = _Builder()
    loop_edges = tuple(
        _loop_edges(builder, loop, profile.plane) for loop in profile.loops
    )
    patch = PlanePatch(profile.plane.origin, profile.plane.x_axis, profile.plane.y_axis)
    _profile_cap(builder, profile, loop_edges, patch, 1, "profile")
    geometry, patches, bounds, orientations, tags, trims = builder.geometry(solid=False)
    profile_trim = _profile_trim(profile)
    face_trims = (profile_trim,)
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=face_trims if profile_trim is not None else trims,
    )


def brep_extrusion(
    profile: PlanarProfile,
    vector: ConvertibleToArray,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-extrusion",
) -> BRepModel:
    """Construct an oriented solid by translating a line/conic planar profile."""
    vector_ = _point3(vector, "vector")
    normal_component = float(vector_ @ profile.plane.normal)
    if normal_component == 0.0:
        raise ValueError("An extrusion vector must leave the source profile plane.")
    from .._rational import rational_vector
    from ._correspondence import exact_source_lift
    from ._intersection_curve import SurfaceRegion

    builder = _Builder()
    loops = profile.loops
    base_patch = PlanePatch(
        profile.plane.origin, profile.plane.x_axis, profile.plane.y_axis
    )
    top_origin = np.asarray(profile.plane.origin) + vector_
    top_patch = PlanePatch(top_origin, profile.plane.x_axis, profile.plane.y_axis)
    profile_bounds = _profile_bounds(profile)
    base_region = SurfaceRegion(base_patch, profile_bounds)
    top_region = SurfaceRegion(top_patch, profile_bounds)
    vector_exact = rational_vector(vector_)

    edge_specs: list[list[_ExtrusionEdgeSpec]] = []
    base_candidates: list[list[list[ConicParameterRoot | None]]] = []
    top_candidates: list[list[list[ConicParameterRoot | None]]] = []
    base_points_exact = tuple(
        tuple(_profile_vertex_exact(profile.plane, point) for point in loop.vertices)
        for loop in loops
    )
    top_points_exact = tuple(
        tuple(
            (
                point[0] + vector_exact[0],
                point[1] + vector_exact[1],
                point[2] + vector_exact[2],
            )
            for point in points
        )
        for points in base_points_exact
    )
    base_points = tuple(
        tuple(tuple(float(value) for value in point) for point in points)
        for points in base_points_exact
    )
    top_points = tuple(
        tuple(tuple(float(value) for value in point) for point in points)
        for points in top_points_exact
    )

    def generator_pcurve(
        root: ConicParameterRoot | None, angle: float, /
    ) -> AbstractCurve:
        return (
            ConicRootLinePCurve(root, 0, 1, angle_realization=angle)
            if root is not None
            else LineCurve((angle, 0.0), (0.0, 1.0))
        )

    for loop_index, loop in enumerate(loops):
        specs: list[_ExtrusionEdgeSpec] = []
        base_loop_candidates: list[list[ConicParameterRoot | None]] = [
            [] for _ in loop.vertices
        ]
        top_loop_candidates: list[list[ConicParameterRoot | None]] = [
            [] for _ in loop.vertices
        ]
        for edge_index in range(len(loop.vertices)):
            curve2d, first, last = _segment_curve(loop, edge_index)
            curve3d = _space_curve(curve2d, profile.plane)
            top_curve3d = _translated_curve(curve3d, vector_)
            next_index = (edge_index + 1) % len(loop.vertices)
            base_start, base_end = (
                base_points[loop_index][edge_index],
                base_points[loop_index][next_index],
            )
            base_start_exact, base_end_exact = (
                base_points_exact[loop_index][edge_index],
                base_points_exact[loop_index][next_index],
            )
            top_start_exact, top_end_exact = (
                top_points_exact[loop_index][edge_index],
                top_points_exact[loop_index][next_index],
            )
            conic_carrier = isinstance(curve3d, (CircleCurve, EllipseCurve))
            base_roots = (
                (
                    ConicParameterRoot.from_exact_point(curve3d, base_start_exact, first),
                    ConicParameterRoot.from_exact_point(curve3d, base_end_exact, last),
                )
                if conic_carrier
                else (None, None)
            )
            top_roots = (
                (
                    ConicParameterRoot.from_exact_point(
                        top_curve3d, top_start_exact, first
                    ),
                    ConicParameterRoot.from_exact_point(top_curve3d, top_end_exact, last),
                )
                if isinstance(top_curve3d, (CircleCurve, EllipseCurve))
                else (None, None)
            )
            patch = _extrusion_patch(curve3d, vector_)
            start_pcurve = generator_pcurve(base_roots[0], first)
            end_pcurve = generator_pcurve(base_roots[1], last)
            base_chart = LineCurve((0.0, 0.0), (1.0, 0.0))
            top_chart = LineCurve((0.0, 1.0), (1.0, 0.0))
            chart_first, chart_last = first, last
            for root in base_roots:
                if root is not None:
                    chart_first = min(chart_first, root.parameter_lower)
                    chart_last = max(chart_last, root.parameter_upper)
            wall_box = np.asarray(
                ((chart_first, 0.0), (chart_last, 1.0)), dtype=np.float64
            )
            wall_region = SurfaceRegion(patch, wall_box)

            def endpoint_matches(
                root: ConicParameterRoot | None,
                point: tuple[ExactScalar, ExactScalar, ExactScalar],
                /,
            ) -> bool:
                return root is not None and root.exact_point() == point

            exact_roots = all(root is not None for root in (*base_roots, *top_roots))
            if exact_roots:
                base_exact = tuple(
                    root.exact_point() for root in base_roots if root is not None
                )
                top_exact = tuple(
                    root.exact_point() for root in top_roots if root is not None
                )
                exact_roots = (
                    all(
                        endpoint_matches(root, point)
                        for root, point in zip(
                            base_roots,
                            (base_start_exact, base_end_exact),
                            strict=True,
                        )
                    )
                    and all(
                        endpoint_matches(root, point)
                        for root, point in zip(
                            top_roots,
                            (top_start_exact, top_end_exact),
                            strict=True,
                        )
                    )
                    and all(
                        tuple(
                            base_exact[index][axis] + vector_exact[axis]
                            for axis in range(3)
                        )
                        == top_exact[index]
                        for index in range(2)
                    )
                    and exact_source_lift(curve3d, curve2d, base_region, first, last)
                    and exact_source_lift(top_curve3d, curve2d, top_region, first, last)
                    and exact_source_lift(curve3d, base_chart, wall_region, first, last)
                    and exact_source_lift(
                        top_curve3d, top_chart, wall_region, first, last
                    )
                    and exact_source_lift(
                        LineCurve(
                            base_start,
                            vector_,
                            origin_exact=base_start_exact,
                            direction_exact=vector_exact,
                        ),
                        start_pcurve,
                        wall_region,
                        0.0,
                        1.0,
                    )
                    and exact_source_lift(
                        LineCurve(
                            base_end,
                            vector_,
                            origin_exact=base_end_exact,
                            direction_exact=vector_exact,
                        ),
                        end_pcurve,
                        wall_region,
                        0.0,
                        1.0,
                    )
                )
            supported_base = base_roots if exact_roots else (None, None)
            supported_top = top_roots if exact_roots else (None, None)
            base_loop_candidates[edge_index].append(supported_base[0])
            base_loop_candidates[next_index].append(supported_base[1])
            top_loop_candidates[edge_index].append(supported_top[0])
            top_loop_candidates[next_index].append(supported_top[1])
            specs.append(
                _ExtrusionEdgeSpec(
                    curve2d,
                    float(first),
                    float(last),
                    curve3d,
                    top_curve3d,
                    patch,
                    supported_base,
                    supported_top,
                    start_pcurve,
                    end_pcurve,
                    wall_box,
                )
            )
        edge_specs.append(specs)
        base_candidates.append(base_loop_candidates)
        top_candidates.append(top_loop_candidates)

    def vertex_binding(
        candidates: list[ConicParameterRoot | None],
        point_exact: tuple[ExactScalar, ExactScalar, ExactScalar],
        /,
    ) -> tuple[tuple[Fraction, Fraction, Fraction] | None, BRepVertexRoot | None]:
        first, second, third = point_exact
        exact_fraction = (
            (first, second, third)
            if isinstance(first, Fraction)
            and isinstance(second, Fraction)
            and isinstance(third, Fraction)
            else None
        )
        if not candidates or any(root is None for root in candidates):
            return exact_fraction, None
        roots = tuple(root for root in candidates if root is not None)
        if any(root.exact_point() != point_exact for root in roots):
            return exact_fraction, None
        return exact_fraction, BRepVertexRoot(BRepConicVertex(roots[0]))

    base_bindings = tuple(
        tuple(
            vertex_binding(candidates, point)
            for candidates, point in zip(root_rows, points, strict=True)
        )
        for root_rows, points in zip(base_candidates, base_points_exact, strict=True)
    )
    top_bindings = tuple(
        tuple(
            vertex_binding(candidates, point)
            for candidates, point in zip(root_rows, points, strict=True)
        )
        for root_rows, points in zip(top_candidates, top_points_exact, strict=True)
    )
    base_vertices = tuple(
        tuple(
            builder.vertex(point, exact_point=binding[0], root=binding[1])
            for point, binding in zip(points, bindings, strict=True)
        )
        for points, bindings in zip(base_points, base_bindings, strict=True)
    )
    top_vertices = tuple(
        tuple(
            builder.vertex(point, exact_point=binding[0], root=binding[1])
            for point, binding in zip(points, bindings, strict=True)
        )
        for points, bindings in zip(top_points, top_bindings, strict=True)
    )

    def endpoint_root(
        root: ConicParameterRoot | None,
        carrier: AbstractCurve,
        exact_parameter: Fraction | None,
        point: tuple[ExactScalar, ...] | None,
        vertex_root: BRepVertexRoot | None,
        /,
    ) -> ConicParameterEndpoint | None:
        if root is None or point is None or vertex_root is None:
            return None
        if len(point) != 3:
            raise ValueError(
                "A spatial conic endpoint must have three exact coordinates."
            )
        return ConicParameterEndpoint(
            root, carrier, exact_parameter, (point[0], point[1], point[2])
        )

    base_edges: list[list[tuple[int, _ExtrusionEdgeSpec]]] = []
    top_edges: list[list[tuple[int, _ExtrusionEdgeSpec]]] = []
    vertical_edges: list[tuple[int, ...]] = []
    for loop_index, specs in enumerate(edge_specs):
        base_loop: list[tuple[int, _ExtrusionEdgeSpec]] = []
        top_loop: list[tuple[int, _ExtrusionEdgeSpec]] = []
        for edge_index, spec in enumerate(specs):
            curve2d = spec.curve2d
            first, last = spec.first, spec.last
            curve3d, top_curve3d = spec.base_curve, spec.top_curve
            base_roots, top_roots = spec.base_roots, spec.top_roots
            next_index = (edge_index + 1) % len(loops[loop_index].vertices)
            base_start_exact = (
                None if base_roots[0] is None else base_roots[0].exact_point()
            )
            base_end_exact = (
                None if base_roots[1] is None else base_roots[1].exact_point()
            )
            top_start_exact = None if top_roots[0] is None else top_roots[0].exact_point()
            top_end_exact = None if top_roots[1] is None else top_roots[1].exact_point()
            base_edge = builder.edge(
                curve3d,
                first,
                last,
                base_vertices[loop_index][edge_index],
                base_vertices[loop_index][next_index],
                endpoint_roots=(
                    endpoint_root(
                        base_roots[0],
                        curve3d,
                        None,
                        base_start_exact,
                        base_bindings[loop_index][edge_index][1],
                    ),
                    endpoint_root(
                        base_roots[1],
                        curve3d,
                        None,
                        base_end_exact,
                        base_bindings[loop_index][next_index][1],
                    ),
                ),
            )
            top_edge = builder.edge(
                top_curve3d,
                first,
                last,
                top_vertices[loop_index][edge_index],
                top_vertices[loop_index][next_index],
                endpoint_roots=(
                    endpoint_root(
                        top_roots[0],
                        top_curve3d,
                        None,
                        top_start_exact,
                        top_bindings[loop_index][edge_index][1],
                    ),
                    endpoint_root(
                        top_roots[1],
                        top_curve3d,
                        None,
                        top_end_exact,
                        top_bindings[loop_index][next_index][1],
                    ),
                ),
            )
            base_loop.append((base_edge, spec))
            top_loop.append((top_edge, spec))
        base_edges.append(base_loop)
        top_edges.append(top_loop)

        vertical_loop = []
        for vertex_index, point in enumerate(base_points[loop_index]):
            top_vertex = top_vertices[loop_index][vertex_index]
            base_vertex = base_vertices[loop_index][vertex_index]
            base_binding = base_bindings[loop_index][vertex_index]
            top_binding = top_bindings[loop_index][vertex_index]
            root = (
                base_binding[1].primary.root
                if base_binding[1] is not None
                and isinstance(base_binding[1].primary, BRepConicVertex)
                and top_binding[1] is not None
                else None
            )
            line = LineCurve(
                point,
                vector_,
                origin_exact=base_points_exact[loop_index][vertex_index],
                direction_exact=vector_exact,
            )
            vertical_loop.append(
                builder.edge(
                    line,
                    0.0,
                    1.0,
                    base_vertex,
                    top_vertex,
                    endpoint_roots=(
                        endpoint_root(
                            root, line, Fraction(0), base_binding[0], base_binding[1]
                        ),
                        endpoint_root(
                            root, line, Fraction(1), top_binding[0], top_binding[1]
                        ),
                    ),
                )
            )
        vertical_edges.append(tuple(vertical_loop))

    normal_sign = 1 if normal_component > 0.0 else -1
    bounds = profile_bounds
    base_loops = []
    top_loops = []
    for loop_index, (base_loop, top_loop) in enumerate(
        zip(base_edges, top_edges, strict=True)
    ):
        base_coedges = []
        top_coedges = []
        for edge_index, ((base_edge, base_spec), (top_edge, top_spec)) in enumerate(
            zip(base_loop, top_loop, strict=True)
        ):
            curve2d = base_spec.curve2d
            base_roots = base_spec.base_roots
            top_roots = top_spec.top_roots
            next_index = (edge_index + 1) % len(loops[loop_index].vertices)
            base_coedges.append(
                builder.coedge(
                    base_edge,
                    1,
                    curve2d,
                    endpoint_roots=(
                        endpoint_root(
                            base_roots[0],
                            curve2d,
                            None,
                            base_roots[0].exact_point()
                            if base_roots[0] is not None
                            else None,
                            base_bindings[loop_index][edge_index][1],
                        ),
                        endpoint_root(
                            base_roots[1],
                            curve2d,
                            None,
                            base_roots[1].exact_point()
                            if base_roots[1] is not None
                            else None,
                            base_bindings[loop_index][next_index][1],
                        ),
                    ),
                )
            )
            top_coedges.append(
                builder.coedge(
                    top_edge,
                    1,
                    curve2d,
                    endpoint_roots=(
                        endpoint_root(
                            top_roots[0],
                            curve2d,
                            None,
                            top_roots[0].exact_point()
                            if top_roots[0] is not None
                            else None,
                            top_bindings[loop_index][edge_index][1],
                        ),
                        endpoint_root(
                            top_roots[1],
                            curve2d,
                            None,
                            top_roots[1].exact_point()
                            if top_roots[1] is not None
                            else None,
                            top_bindings[loop_index][next_index][1],
                        ),
                    ),
                )
            )
        base_loops.append(base_coedges)
        top_loops.append(top_coedges)

    builder.face(base_patch, bounds, -normal_sign, "cap:bottom", base_loops)
    builder.face(top_patch, bounds, normal_sign, "cap:top", top_loops)
    for loop_index, edges in enumerate(base_edges):
        loop = loops[loop_index]
        for edge_index, (base_edge, spec) in enumerate(edges):
            first, last = spec.first, spec.last
            curve3d, patch = spec.base_curve, spec.patch
            base_roots, top_roots = spec.base_roots, spec.top_roots
            start_pcurve, end_pcurve, box = (
                spec.start_pcurve,
                spec.end_pcurve,
                spec.wall_box,
            )
            top_edge = top_edges[loop_index][edge_index][0]
            next_index = (edge_index + 1) % len(loop.vertices)
            start_edge = vertical_edges[loop_index][edge_index]
            end_edge = vertical_edges[loop_index][next_index]
            base_chart = LineCurve((0.0, 0.0), (1.0, 0.0))
            top_chart = LineCurve((0.0, 1.0), (1.0, 0.0))
            end_root = base_roots[1]
            start_root = base_roots[0]
            end_point = base_bindings[loop_index][next_index][0]
            start_point = base_bindings[loop_index][edge_index][0]
            top_end_point = top_bindings[loop_index][next_index][0]
            top_start_point = top_bindings[loop_index][edge_index][0]
            end_roots = (
                endpoint_root(
                    end_root,
                    end_pcurve,
                    Fraction(0),
                    end_point,
                    base_bindings[loop_index][next_index][1],
                ),
                endpoint_root(
                    end_root,
                    end_pcurve,
                    Fraction(1),
                    top_end_point,
                    top_bindings[loop_index][next_index][1],
                ),
            )
            start_roots = (
                endpoint_root(
                    start_root,
                    start_pcurve,
                    Fraction(0),
                    start_point,
                    base_bindings[loop_index][edge_index][1],
                ),
                endpoint_root(
                    start_root,
                    start_pcurve,
                    Fraction(1),
                    top_start_point,
                    top_bindings[loop_index][edge_index][1],
                ),
            )
            coedges = [
                builder.coedge(
                    base_edge,
                    1,
                    base_chart,
                    endpoint_roots=(
                        endpoint_root(
                            base_roots[0],
                            base_chart,
                            None,
                            base_roots[0].exact_point()
                            if base_roots[0] is not None
                            else None,
                            base_bindings[loop_index][edge_index][1],
                        ),
                        endpoint_root(
                            base_roots[1],
                            base_chart,
                            None,
                            base_roots[1].exact_point()
                            if base_roots[1] is not None
                            else None,
                            base_bindings[loop_index][next_index][1],
                        ),
                    ),
                ),
                builder.coedge(end_edge, 1, end_pcurve, endpoint_roots=end_roots),
                builder.coedge(
                    top_edge,
                    -1,
                    top_chart,
                    endpoint_roots=(
                        endpoint_root(
                            top_roots[0],
                            top_chart,
                            None,
                            top_roots[0].exact_point()
                            if top_roots[0] is not None
                            else None,
                            top_bindings[loop_index][edge_index][1],
                        ),
                        endpoint_root(
                            top_roots[1],
                            top_chart,
                            None,
                            top_roots[1].exact_point()
                            if top_roots[1] is not None
                            else None,
                            top_bindings[loop_index][next_index][1],
                        ),
                    ),
                ),
                builder.coedge(start_edge, -1, start_pcurve, endpoint_roots=start_roots),
            ]
            builder.face(
                patch,
                box,
                normal_sign,
                f"wall:{loop_index}:{edge_index}",
                [coedges],
            )
            if (
                builder.edge_vertices[base_edge][0] == builder.edge_vertices[base_edge][1]
                and curve3d.native_period_symbol() == "two_pi"
            ):
                for coedge in (coedges[0], coedges[2]):
                    builder.native_phase(
                        builder.coedge_edges[coedge],
                        patch,
                        0,
                        coedge=coedge,
                        rational=Fraction.from_float(first),
                    )
    geometry, patches, bounds, orientations, tags, trims = builder.geometry(solid=True)
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
    )


def _axis_frame(axis: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, basis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def _curve_axis_distance_bounds(
    curve: AbstractCurve,
    first: float,
    last: float,
    axis_origin: np.ndarray,
    radial_direction: np.ndarray,
    /,
) -> tuple[float, float]:
    if isinstance(curve, LineCurve):
        values = np.asarray(curve.evaluate(jnp.asarray((first, last))), dtype=np.float64)
        signed = (values - axis_origin) @ radial_direction
        return float(np.min(signed)), float(np.max(signed))
    if isinstance(curve, (CircleCurve, EllipseCurve)):
        center = np.asarray(curve.center) - axis_origin
        center_radius = float(center @ radial_direction)
        if isinstance(curve, CircleCurve):
            first_scale = second_scale = float(curve.radius)
        else:
            first_scale = float(curve.first_radius)
            second_scale = float(curve.second_radius)
        first_coefficient = first_scale * float(
            np.asarray(curve.first_axis) @ radial_direction
        )
        second_coefficient = second_scale * float(
            np.asarray(curve.second_axis) @ radial_direction
        )
        lower, upper = min(first, last), max(first, last)
        angles = [first, last]
        phase = atan2(second_coefficient, first_coefficient)
        for turn in range(
            floor((lower - phase) / pi),
            ceil((upper - phase) / pi) + 1,
        ):
            candidate = phase + turn * pi
            if lower <= candidate <= upper:
                angles.append(candidate)
        values = tuple(
            center_radius
            + first_coefficient * cos(angle)
            + second_coefficient * sin(angle)
            for angle in angles
        )
        return min(values), max(values)
    raise NotImplementedError(
        "Revolution requires line, circle, or ellipse profile carriers."
    )


def _rotational_edge(
    builder: _Builder,
    point: np.ndarray,
    vertex: int,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    /,
) -> int:
    relative = point - axis_origin
    axial = float(relative @ axis_direction)
    center = axis_origin + axial * axis_direction
    radial = relative - axial * axis_direction
    radius = float(np.linalg.norm(radial))
    if radius == 0.0:
        return builder.degenerate(vertex, 0.0, _TWO_PI)
    first = radial / radius
    second = np.cross(axis_direction, first)
    curve = CircleCurve(center, first, second, radius)
    return builder.edge(curve, 0.0, _TWO_PI, vertex, vertex)


def _revolution_orientation(
    patch: RevolutionSurface,
    profile_plane: ProfilePlane,
    first: float,
    last: float,
    /,
) -> int:
    middle = 0.5 * (first + last)
    point = np.asarray(patch.curve.evaluate(jnp.asarray(middle)), dtype=np.float64)
    tangent = np.asarray(
        jax.jacfwd(patch.curve.evaluate)(jnp.asarray(middle)), dtype=np.float64
    )
    axis = np.asarray(patch.axis_direction)
    axis_origin = np.asarray(patch.axis_origin)
    radial = point - axis_origin - float((point - axis_origin) @ axis) * axis
    angular = np.cross(axis, radial)
    candidate = np.cross(angular, tangent)
    planar_tangent = np.asarray(profile_plane.x_axis) * float(
        tangent @ np.asarray(profile_plane.x_axis)
    ) + np.asarray(profile_plane.y_axis) * float(
        tangent @ np.asarray(profile_plane.y_axis)
    )
    outward = np.cross(planar_tangent, profile_plane.normal)
    return 1 if float(candidate @ outward) > 0.0 else -1


def brep_revolution(
    profile: PlanarProfile,
    axis_origin: ConvertibleToArray,
    axis_direction: ConvertibleToArray,
    /,
    *,
    angle: float | None = None,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-revolution",
) -> BRepModel:
    """Construct a full-turn solid from a profile in a plane containing its axis."""
    if not isinstance(profile, PlanarProfile):
        raise TypeError("profile must be a PlanarProfile.")
    if profile.holes:
        raise NotImplementedError(
            "Revolution profiles with holes require a certified shell arrangement."
        )
    angle_ = _TWO_PI if angle is None else positive_finite_float(angle, "angle")
    if angle_ != _TWO_PI:
        raise NotImplementedError("Only complete 2π revolutions are currently supported.")
    origin2 = _point2(axis_origin, "axis_origin")
    direction2 = np.asarray(axis_direction, dtype=np.float64)
    if direction2.shape != (2,) or not np.all(np.isfinite(direction2)):
        raise ValueError("axis_direction must be a finite 2D vector.")
    length = float(np.linalg.norm(direction2))
    if length == 0.0:
        raise ValueError("axis_direction must be nonzero.")
    direction2 /= length
    origin3 = profile.plane.point(origin2)
    axis3 = _unit(profile.plane.vector(direction2), "axis_direction")
    builder = _Builder()
    loop = profile.outer
    points2 = tuple(np.asarray(point, dtype=np.float64) for point in loop.vertices)
    points3 = tuple(profile.plane.point(point) for point in loop.vertices)
    exact_points3 = tuple(
        _profile_vertex_exact(profile.plane, point) for point in loop.vertices
    )
    vertex_ids = tuple(
        builder.vertex(
            point,
            exact_point=(
                exact_point
                if np.array_equal(
                    np.asarray(tuple(float(value) for value in exact_point)),
                    point,
                )
                else None
            ),
        )
        for point, exact_point in zip(points3, exact_points3, strict=True)
    )
    profile_edges: list[tuple[int, float, float, AbstractCurve] | None] = []
    for index in range(len(loop.vertices)):
        curve2, first, last = _segment_curve(loop, index)
        curve3 = _space_curve(curve2, profile.plane)
        radial_direction = np.cross(axis3, profile.plane.normal)
        radial_direction /= np.linalg.norm(radial_direction)
        minimum_radius, _ = _curve_axis_distance_bounds(
            curve3, first, last, origin3, radial_direction
        )
        endpoints = np.asarray(
            curve3.evaluate(jnp.asarray((first, last))), dtype=np.float64
        )
        endpoint_radii = (endpoints - origin3) @ radial_direction
        if minimum_radius < 0.0:
            raise ValueError(
                "Revolution profile must stay in one closed half-plane of its axis."
            )
        is_axis_line = isinstance(curve3, LineCurve) and np.all(endpoint_radii == 0.0)
        if (
            minimum_radius == 0.0
            and not is_axis_line
            and not np.any(endpoint_radii == 0.0)
        ):
            raise ValueError(
                "A revolution conic may touch its axis only at an authored endpoint."
            )
        if is_axis_line:
            profile_edges.append(None)
            continue
        edge = builder.edge(
            curve3,
            first,
            last,
            vertex_ids[index],
            vertex_ids[(index + 1) % len(loop.vertices)],
        )
        profile_edges.append((edge, first, last, curve3))
    rotational_edges = tuple(
        _rotational_edge(builder, points3[index], vertex_ids[index], origin3, axis3)
        for index in range(len(points3))
    )
    from ._boolean_core import AffinePCurve, PeriodicPCurve

    def rooted_ring_pcurve(
        parameter: float,
        root: ConicParameterRoot | None,
        rotational_edge: int,
        /,
    ) -> AbstractCurve:
        curve_index = builder.edge_curves[rotational_edge]
        if root is None:
            return LineCurve((0.0, parameter), (1.0, 0.0))
        if curve_index < 0:
            # A pole latitude is the conic root itself, not the binary64
            # realization of its generally irrational parameter.
            return AffinePCurve(
                ConicRootLinePCurve(root, 0, 1, parameter_range=None),
                ((0, 1), (1, 0)),
                (0, 0),
            )
        ring_curve = builder.curves[curve_index]
        if not isinstance(ring_curve, CircleCurve):
            return LineCurve((0.0, parameter), (1.0, 0.0))
        ring_point = tuple(
            ring_curve.center_exact[index]
            + ring_curve.radius_exact * ring_curve.first_axis_exact[index]
            for index in range(3)
        )
        if ring_point != root.exact_point():
            return LineCurve((0.0, parameter), (1.0, 0.0))
        return AffinePCurve(
            ConicRootLinePCurve(root, 0, 1, parameter_range=(0.0, _TWO_PI)),
            ((0, 1), (1, 0)),
            (0, 0),
        )

    for index, item in enumerate(profile_edges):
        if item is None:
            continue
        edge, first, last, curve = item
        patch = RevolutionSurface(curve, origin3, axis3)
        bounds = np.asarray(((0.0, first), (_TWO_PI, last)), dtype=np.float64)
        next_index = (index + 1) % len(profile_edges)
        start_vertex, end_vertex = builder.edge_vertices[edge]
        first_point_exact = builder.vertex_exact_points[start_vertex]
        last_point_exact = builder.vertex_exact_points[end_vertex]
        profile_first_root: ConicParameterRoot | None = None
        if (
            isinstance(curve, (CircleCurve, EllipseCurve))
            and first_point_exact is not None
        ):
            profile_first_root = ConicParameterRoot.from_exact_point(
                curve, first_point_exact, first
            )
            if (
                profile_first_root is not None
                and profile_first_root.exact_point() != first_point_exact
            ):
                profile_first_root = None
        profile_last_root: ConicParameterRoot | None = None
        if (
            isinstance(curve, (CircleCurve, EllipseCurve))
            and last_point_exact is not None
        ):
            profile_last_root = ConicParameterRoot.from_exact_point(
                curve, last_point_exact, last
            )
            if (
                profile_last_root is not None
                and profile_last_root.exact_point() != last_point_exact
            ):
                profile_last_root = None
        first_ring_pcurve = rooted_ring_pcurve(
            first, profile_first_root, rotational_edges[index]
        )
        last_ring_pcurve = rooted_ring_pcurve(
            last, profile_last_root, rotational_edges[next_index]
        )
        profile_turn_root = (
            start_vertex == end_vertex
            and curve.native_period_symbol() == "two_pi"
            and curve.period is not None
            and profile_first_root is not None
            and profile_last_root is not None
            and profile_first_root.cosine_exact == profile_last_root.cosine_exact
            and profile_first_root.sine_exact == profile_last_root.sine_exact
            and profile_last_root.turns - profile_first_root.turns == 1
        )
        if profile_turn_root:
            last_ring_pcurve = PeriodicPCurve(first_ring_pcurve, patch, (0, 1))
        else:
            if profile_first_root is not None:
                bounds[0, 1] = min(bounds[0, 1], profile_first_root.parameter_lower)
            if profile_last_root is not None:
                bounds[1, 1] = max(bounds[1, 1], profile_last_root.parameter_upper)
        # The seam is traversed on both chart sheets: the forward use closes the
        # authored loop one native turn later, the reverse use on the base sheet.
        seam_pcurve = LineCurve((0.0, 0.0), (0.0, 1.0))
        coedges = [
            builder.coedge(
                rotational_edges[index],
                1,
                first_ring_pcurve,
            ),
            builder.coedge(
                edge,
                1,
                PeriodicPCurve(seam_pcurve, patch, (1, 0)),
            ),
            builder.coedge(
                rotational_edges[next_index],
                -1,
                last_ring_pcurve,
            ),
            builder.coedge(
                edge,
                -1,
                seam_pcurve,
            ),
        ]
        orientation = _revolution_orientation(patch, profile.plane, first, last)
        builder.face(patch, bounds, orientation, f"revolved:{index}", [coedges])
        for coedge in coedges:
            edge_id = builder.coedge_edges[coedge]
            if edge_id in rotational_edges:
                builder.native_phase(edge_id, patch, 0, coedge=coedge)
    if not builder.faces:
        raise ValueError(
            "A revolution profile must generate at least one regular surface face."
        )

    def profile_endpoint_root(
        curve: AbstractCurve,
        point_exact: tuple[ExactScalar, ExactScalar, ExactScalar],
        parameter: float,
        primary_root: ConicParameterRoot | None,
        /,
    ) -> ConicParameterEndpoint | None:
        exact_parameter = Fraction.from_float(float(parameter))
        if isinstance(curve, (CircleCurve, EllipseCurve)):
            root = ConicParameterRoot.from_exact_point(
                curve, point_exact, float(parameter)
            )
            if root is None or root.exact_point() != point_exact:
                return None
        elif isinstance(curve, ExactLineCurve):
            if primary_root is None:
                return None
            root = primary_root
            source_point = tuple(
                first + exact_parameter * (last - first)
                for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
            )
            if source_point != point_exact:
                return None
        elif isinstance(curve, LineCurve):
            if primary_root is None:
                return None
            root = primary_root
            source_point = tuple(
                origin + exact_parameter * direction
                for origin, direction in zip(
                    curve.origin_exact, curve.direction_exact, strict=True
                )
            )
            if source_point != point_exact:
                return None
        else:
            return None
        parameter_exact = (
            None if isinstance(curve, (CircleCurve, EllipseCurve)) else exact_parameter
        )
        return ConicParameterEndpoint(root, curve, parameter_exact, point_exact)

    incident_profile_endpoints: list[list[tuple[AbstractCurve, float]]] = [
        [] for _ in points2
    ]
    for index, item in enumerate(profile_edges):
        if item is None:
            continue
        _, first, last, curve = item
        incident_profile_endpoints[index].append((curve, first))
        incident_profile_endpoints[(index + 1) % len(points2)].append((curve, last))
    for vertex, point_exact in enumerate(exact_points3):
        if not np.array_equal(
            np.asarray(tuple(float(value) for value in point_exact)),
            points3[vertex],
        ):
            continue
        root_candidates: list[ConicParameterRoot] = []
        rotational_edge = rotational_edges[vertex]
        curve_index = builder.edge_curves[rotational_edge]
        if curve_index >= 0:
            rotational_curve = builder.curves[curve_index]
            if not isinstance(rotational_curve, (CircleCurve, EllipseCurve)):
                continue
            rotational_root = ConicParameterRoot.from_exact_point(
                rotational_curve, point_exact, 0.0
            )
            if rotational_root is None or rotational_root.exact_point() != point_exact:
                continue
            root_candidates.append(rotational_root)
        for curve, parameter in incident_profile_endpoints[vertex]:
            if isinstance(curve, (CircleCurve, EllipseCurve)):
                profile_root = ConicParameterRoot.from_exact_point(
                    curve, point_exact, parameter
                )
                if profile_root is not None and profile_root.exact_point() == point_exact:
                    root_candidates.append(profile_root)
        for root in root_candidates:
            if all(
                profile_endpoint_root(curve, point_exact, parameter, root) is not None
                for curve, parameter in incident_profile_endpoints[vertex]
            ):
                builder.vertex_roots[vertex] = BRepVertexRoot(BRepConicVertex(root))
                break
    for index, item in enumerate(profile_edges):
        if item is None:
            continue
        edge, first, last, curve = item
        next_index = (index + 1) % len(points2)
        bindings = []
        for vertex, parameter in ((index, first), (next_index, last)):
            vertex_root = builder.vertex_roots[vertex]
            primary_root = (
                vertex_root.primary.root
                if vertex_root is not None
                and isinstance(vertex_root.primary, BRepConicVertex)
                else None
            )
            bindings.append(
                None
                if primary_root is None
                else profile_endpoint_root(
                    curve, exact_points3[vertex], parameter, primary_root
                )
            )
        if any(
            isinstance(root, NativePeriodEndpoint)
            for root in builder.edge_endpoint_roots[edge]
        ):
            continue
        if any(binding is not None for binding in bindings):
            builder.edge_endpoint_roots[edge] = (bindings[0], bindings[1])
    for coedge, edge in enumerate(builder.coedge_edges):
        edge_roots = builder.edge_endpoint_roots[edge]
        current_roots = builder.coedge_endpoint_roots[coedge]
        if any(isinstance(root, NativePeriodEndpoint) for root in current_roots):
            continue
        if not any(isinstance(root, ConicParameterEndpoint) for root in edge_roots):
            continue
        pcurve = builder.pcurves[coedge]
        first_root, last_root = edge_roots
        first_existing, last_existing = current_roots
        builder.coedge_endpoint_roots[coedge] = (
            ConicParameterEndpoint(
                first_root.root,
                pcurve,
                first_root.parameter_exact,
                first_root.point_exact,
            )
            if isinstance(first_root, ConicParameterEndpoint)
            else first_existing,
            ConicParameterEndpoint(
                last_root.root,
                pcurve,
                last_root.parameter_exact,
                last_root.point_exact,
            )
            if isinstance(last_root, ConicParameterEndpoint)
            else last_existing,
        )
    geometry, patches, bounds, orientations, tags, trims = builder.geometry(solid=True)
    from ._conic_gaps import native_face_gap_squared
    from ._correspondence import exact_source_lift
    from ._exact import square_root_interval
    from ._intersection_curve import SurfaceRegion

    coedge_faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                if coedge_faces[coedge] not in (-1, face):
                    raise ValueError("A revolution coedge must belong to one face.")
                coedge_faces[coedge] = face
    gap_bounds: list[float] = []
    edge_ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for coedge, edge in enumerate(geometry.coedge_edges):
        face = coedge_faces[coedge]
        if face < 0:
            raise ValueError("A revolution coedge must belong to a source face.")
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            gap_bounds.append(0.0)
            continue
        curve = geometry.curves[curve_index]
        pcurve = geometry.pcurves[coedge]
        source_region = SurfaceRegion(
            patches[face], np.asarray(bounds[face], dtype=np.float64)
        )
        first_parameter, last_parameter = (float(value) for value in edge_ranges[edge])
        squared = (
            Fraction(0)
            if exact_source_lift(
                curve, pcurve, source_region, first_parameter, last_parameter
            )
            else native_face_gap_squared(
                curve, pcurve, source_region, first_parameter, last_parameter
            )
        )
        if squared is None:
            raise ValueError(
                f"A revolution coedge {coedge} has no exact source face-gap proof."
            )
        gap_bounds.append(float(square_root_interval(squared)[1]))
    if any(value > 0.0 for value in gap_bounds):
        geometry, patches, bounds, orientations, tags, trims = builder.geometry(
            solid=True,
            edge_face_gap_bounds=tuple(gap_bounds),
        )
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
    )


def _axial_plane(center: ConvertibleToArray, axis: ConvertibleToArray, /) -> ProfilePlane:
    origin = _point3(center, "center")
    direction = _unit(axis, "axis")
    first, _ = _axis_frame(direction)
    return ProfilePlane(tuple(origin), tuple(first), tuple(direction))


def brep_box(
    lower: ConvertibleToArray,
    upper: ConvertibleToArray,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-box",
) -> BRepModel:
    lower_, upper_ = _point3(lower, "lower"), _point3(upper, "upper")
    extent = upper_ - lower_
    if np.any(extent <= 0.0):
        raise ValueError("upper must exceed lower in every coordinate.")
    profile = PlanarProfile(
        ProfilePlane(tuple(lower_)),
        ProfileLoop.polygon(
            ((0.0, 0.0), (extent[0], 0.0), (extent[0], extent[1]), (0.0, extent[1]))
        ),
    )
    return brep_extrusion(
        profile,
        (0.0, 0.0, extent[2]),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


def brep_cylinder(
    radius: float,
    height: float,
    /,
    *,
    base_center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-cylinder",
) -> BRepModel:
    return brep_cone(
        radius,
        radius,
        height,
        base_center=base_center,
        axis=axis,
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


def brep_cone(
    bottom_radius: float,
    top_radius: float,
    height: float,
    /,
    *,
    base_center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-cone",
) -> BRepModel:
    radii = (float(bottom_radius), float(top_radius))
    height_ = positive_finite_float(height, "height")
    if any(not isfinite(value) or value < 0.0 for value in radii) or max(radii) <= 0.0:
        raise ValueError("Cone radii must be finite, nonnegative, and not both zero.")
    corners = [(0.0, 0.0), (radii[0], 0.0), (radii[1], height_), (0.0, height_)]
    corners = [
        corner for index, corner in enumerate(corners) if corner not in corners[:index]
    ]
    profile = PlanarProfile(_axial_plane(base_center, axis), ProfileLoop.polygon(corners))
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


def brep_sphere(
    radius: float,
    /,
    *,
    center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-sphere",
) -> BRepModel:
    radius_ = positive_finite_float(radius, "radius")
    profile = PlanarProfile(
        _axial_plane(center, axis),
        ProfileLoop(
            ((0.0, -radius_), (0.0, radius_)), (ProfileArc((0.0, 0.0)), ProfileLine())
        ),
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


def brep_torus(
    major_radius: float,
    minor_radius: float,
    /,
    *,
    center: ConvertibleToArray = (0.0, 0.0, 0.0),
    axis: ConvertibleToArray = (0.0, 0.0, 1.0),
    coordinate_contract: SpatialCoordinateContract,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-torus",
) -> BRepModel:
    major = positive_finite_float(major_radius, "major_radius")
    minor = positive_finite_float(minor_radius, "minor_radius")
    if minor >= major:
        raise ValueError("A ring torus requires minor_radius < major_radius.")
    profile = PlanarProfile(
        _axial_plane(center, axis),
        ProfileLoop.circle((major, 0.0), minor),
    )
    return brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
    )


def _offset_period_roots(
    roots: tuple[object | None, object | None],
    carrier: AbstractCurve,
    patch: AbstractSurfacePatch,
    /,
) -> tuple[NativePeriodEndpoint, NativePeriodEndpoint]:
    rebound: list[NativePeriodEndpoint] = []
    for root in roots:
        if not isinstance(root, NativePeriodEndpoint):
            raise ValueError("A normal offset requires native period endpoint witnesses.")
        rebound.append(
            NativePeriodEndpoint(
                carrier,
                patch,
                root.axis,
                rational=root.rational,
                turns=root.turns,
            )
        )
    return rebound[0], rebound[1]


def _offset_chart_pcurve(
    pcurve: AbstractCurve,
    source_patch: AbstractSurfacePatch,
    offset_patch: AbstractSurfacePatch,
    /,
) -> AbstractCurve:
    """Rebind native-period sheet gauges from a source chart to its normal offset.

    A normal offset shares the source chart and declared periods, so the same
    integer gauge names the same sheet of the offset patch.
    """
    from ._boolean_core import PeriodicPCurve

    if not isinstance(pcurve, PeriodicPCurve) or pcurve.patch is not source_patch:
        return pcurve
    return PeriodicPCurve(
        _offset_chart_pcurve(pcurve.source_curve, source_patch, offset_patch),
        offset_patch,
        pcurve.period_shifts,
    )


def _offset_native_turn_roots(
    carrier: AbstractCurve,
    patch: AbstractSurfacePatch,
    axis: int,
    /,
) -> tuple[NativePeriodEndpoint, NativePeriodEndpoint]:
    """Bind one whole native turn of a zero-phase offset trim line."""
    return (
        NativePeriodEndpoint(carrier, patch, axis, rational=0, turns=0),
        NativePeriodEndpoint(carrier, patch, axis, rational=0, turns=1),
    )


def _single_chart_torus_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    include_parameter_range: bool,
) -> BRepOffsetResult | None:
    from ._boolean_core import AffinePCurve, PeriodicPCurve
    from ._carrier_pieces import carrier_payload
    from ._intersection import NativePeriodEndpoint
    from ._offset_result import exact_operation_offset_result

    geometry = model.geometry
    occurrences = tuple(geometry.assembly_graph.iter_occurrences())
    identity = BRepAffinePlacement.identity().identity_payload()
    if (
        len(occurrences) != 1
        or occurrences[0].solid_index != 0
        or occurrences[0].placement.identity_payload() != identity
    ):
        return None
    if (
        len(model.patches) != 1
        or geometry.topology.num_solids != 1
        or geometry.topology.num_faces != 1
        or geometry.topology.num_edges != 2
        or geometry.topology.num_vertices != 1
        or len(geometry.coedge_edges) != 4
        or len(geometry.face_loops) != 1
        or len(geometry.face_loops[0]) != 1
        or len(geometry.face_loops[0][0]) != 4
    ):
        return None
    patch = model.patches[0]
    if not isinstance(patch, RevolutionSurface) or not isinstance(
        patch.curve, (CircleCurve, EllipseCurve)
    ):
        return None
    curve = patch.curve

    def exact_vector(value: ConvertibleToArray, /) -> tuple[Fraction, ...]:
        return tuple(Fraction.from_float(float(item)) for item in np.asarray(value))

    def exact_dot(
        first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
    ) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    axis = exact_vector(patch.axis_direction)
    curve_center = exact_vector(curve.center)
    axis_origin = exact_vector(patch.axis_origin)
    center = tuple(
        value - base for value, base in zip(curve_center, axis_origin, strict=True)
    )
    first_axis, second_axis = (
        exact_vector(curve.first_axis),
        exact_vector(curve.second_axis),
    )
    frame_normal = (
        first_axis[1] * second_axis[2] - first_axis[2] * second_axis[1],
        first_axis[2] * second_axis[0] - first_axis[0] * second_axis[2],
        first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0],
    )
    if (
        exact_dot(axis, axis) != 1
        or exact_dot(first_axis, first_axis) != 1
        or exact_dot(second_axis, second_axis) != 1
        or exact_dot(first_axis, second_axis) != 0
        or exact_dot(axis, frame_normal) != 0
        or exact_dot(axis, center) != 0
    ):
        return None
    major_squared = exact_dot(center, center)
    if isinstance(curve, CircleCurve):
        radial_radius = Fraction.from_float(float(curve.radius))
        if radial_radius <= 0 or major_squared <= radial_radius * radial_radius:
            return None
        curvature_radius = radial_radius
    else:
        first_radius = Fraction.from_float(float(curve.first_radius))
        second_radius = Fraction.from_float(float(curve.second_radius))
        first_axis_dot = exact_dot(axis, first_axis)
        second_axis_dot = exact_dot(axis, second_axis)
        if first_axis_dot in (-1, 1) and second_axis_dot == 0:
            radial_axis, radial_radius = second_axis, second_radius
            axial_radius = first_radius
        elif second_axis_dot in (-1, 1) and first_axis_dot == 0:
            radial_axis, radial_radius = first_axis, first_radius
            axial_radius = second_radius
        else:
            return None
        radial_center = exact_dot(center, radial_axis)
        if (
            tuple(radial_center * value for value in radial_axis) != center
            or major_squared <= radial_radius * radial_radius
        ):
            return None
        curvature_radius = min(
            radial_radius * radial_radius / axial_radius,
            axial_radius * axial_radius / radial_radius,
        )
    shell_signs = geometry.topology.solid_face_orientations[0]
    if len(shell_signs) != 1:
        return None
    frame_sign = int(np.asarray(model.orientation, dtype=np.float64)[0]) * shell_signs[0]
    if frame_sign not in (-1, 1):
        return None
    effective_distance = frame_sign * Fraction.from_float(distance)
    if (
        major_squared <= (radial_radius + abs(effective_distance)) ** 2
        or effective_distance < 0
        and -effective_distance >= curvature_radius
    ):
        raise ValueError(
            "brep_offset.revolution must retain a regular embedded normal offset."
        )
    if patch.native_period_symbols() != ("two_pi", "two_pi"):
        return None
    first_period, second_period = patch.periods
    if first_period is None or second_period is None:
        return None
    if any(
        not isfinite(period) or period <= 0.0 for period in (first_period, second_period)
    ):
        return None
    period_exact = (
        Fraction.from_float(first_period),
        Fraction.from_float(second_period),
    )
    bounds = np.asarray(model.parameter_bounds[0], dtype=np.float64)
    if any(
        Fraction.from_float(float(bounds[0, axis_index])) != 0
        or Fraction.from_float(float(bounds[1, axis_index]))
        - Fraction.from_float(float(bounds[0, axis_index]))
        != period_exact[axis_index]
        for axis_index in range(2)
    ):
        return None
    uses: list[list[int]] = [[] for _ in geometry.edge_curves]
    for coedge, edge in enumerate(geometry.coedge_edges):
        uses[edge].append(coedge)
    if any(
        geometry.edge_vertices[edge] != (0, 0) or len(uses[edge]) != 2
        for edge in range(geometry.topology.num_edges)
    ):
        return None

    # A rooted one-vertex profile starts at its exact zero phase; any other
    # rooted start has no rational chart coordinate for the offset trim lines.
    vertex_root = geometry.vertex_roots[0]
    profile_point = geometry.vertex_exact_points[0]
    if vertex_root is not None:
        primary = vertex_root.primary
        if (
            not isinstance(primary, BRepConicVertex)
            or vertex_root.aliases
            or any(primary.offset_exact)
            or primary.point_exact != profile_point
        ):
            return None

    def zero_phase_profile_root(root: ConicParameterRoot, turns: int, /) -> bool:
        return (
            profile_point is not None
            and carrier_payload(root.conic) == carrier_payload(curve)
            and (root.cosine_exact, root.sine_exact, root.turns) == (1, 0, turns)
            and root.exact_point() == profile_point
        )

    def native_pair(
        roots: tuple[object | None, object | None],
        carrier: AbstractCurve,
        axis_index: int,
        /,
    ) -> bool:
        first_root, last_root = roots
        return (
            isinstance(first_root, NativePeriodEndpoint)
            and isinstance(last_root, NativePeriodEndpoint)
            and carrier_payload(first_root.curve) == carrier_payload(carrier)
            and carrier_payload(last_root.curve) == carrier_payload(carrier)
            and first_root.patch is not None
            and last_root.patch is not None
            and carrier_payload(first_root.patch) == carrier_payload(patch)
            and carrier_payload(last_root.patch) == carrier_payload(patch)
            and first_root.axis == last_root.axis == axis_index
            and first_root.rational == last_root.rational == 0
            and first_root.turns == 0
            and last_root.turns == 1
        )

    def rooted_turn_pair(roots: tuple[object | None, object | None], /) -> bool:
        first_root, last_root = roots
        return (
            isinstance(first_root, ConicParameterEndpoint)
            and isinstance(last_root, ConicParameterEndpoint)
            and first_root.parameter_exact is None
            and last_root.parameter_exact is None
            and first_root.point_exact == profile_point
            and last_root.point_exact == profile_point
            and zero_phase_profile_root(first_root.root, 0)
            and zero_phase_profile_root(last_root.root, 1)
        )

    def chart_line(
        pcurve: AbstractCurve, /
    ) -> (
        tuple[
            tuple[ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar],
            tuple[int, int],
        ]
        | None
    ):
        """Return the exact chart line and native sheet of one source use."""
        shifts = (0, 0)
        while isinstance(pcurve, PeriodicPCurve):
            if pcurve.patch is not patch:
                return None
            shifts = (
                shifts[0] + pcurve.period_shifts[0],
                shifts[1] + pcurve.period_shifts[1],
            )
            pcurve = pcurve.source_curve
        if isinstance(pcurve, AffinePCurve):
            inner = pcurve.curve
            if (
                not isinstance(inner, ConicRootLinePCurve)
                or not zero_phase_profile_root(inner.root, 0)
                or inner.phase_shift_exact != 0
                or pcurve.matrix != ((0, 1), (1, 0))
            ):
                return None
            # The rooted ring chart is (t, phase) at the exact zero phase.
            return (
                (pcurve.offset[0] + inner.height_origin_exact, pcurve.offset[1]),
                (inner.height_direction_exact, Fraction(0)),
                shifts,
            )
        if not isinstance(pcurve, LineCurve):
            return None
        return (
            (pcurve.origin_exact[0], pcurve.origin_exact[1]),
            (pcurve.direction_exact[0], pcurve.direction_exact[1]),
            shifts,
        )

    edge_axes: list[int] = []
    coedge_lines: dict[int, tuple[int, tuple[int, int]]] = {}
    for edge, coedges in enumerate(uses):
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            return None
        edge_curve = geometry.curves[curve_index]
        if (
            not isinstance(edge_curve, (CircleCurve, EllipseCurve))
            or edge_curve.native_period_symbol() != "two_pi"
        ):
            return None
        edge_pair = geometry.edge_endpoint_roots[edge]
        if native_pair(edge_pair, edge_curve, 0):
            axis_index = 0
        elif native_pair(edge_pair, edge_curve, 1) or rooted_turn_pair(edge_pair):
            axis_index = 1
        else:
            return None
        edge_axes.append(axis_index)
        edge_range = tuple(
            Fraction.from_float(float(value))
            for value in np.asarray(geometry.edge_ranges[edge])
        )
        if edge_range != (0, period_exact[axis_index]):
            return None
        fixed_axis = 1 - axis_index
        for coedge in coedges:
            line = chart_line(geometry.pcurves[coedge])
            if line is None:
                return None
            origin, direction, shifts = line
            if (
                direction[axis_index] != 1
                or direction[fixed_axis] != 0
                or origin != (0, 0)
                or shifts[axis_index]
                or shifts[fixed_axis] not in (0, 1)
            ):
                return None
            coedge_roots = geometry.coedge_endpoint_roots[coedge]
            if not (
                native_pair(coedge_roots, geometry.pcurves[coedge], axis_index)
                or (axis_index == 1 and rooted_turn_pair(coedge_roots))
            ):
                return None
            coedge_lines[coedge] = (axis_index, shifts)
        # Each closed edge is used once on the base sheet and once one turn later.
        if sorted(coedge_lines[coedge][1][fixed_axis] for coedge in coedges) != [0, 1]:
            return None
    if sorted(edge_axes) != [0, 1]:
        return None

    offset_patch = OffsetSurface(
        patch,
        Fraction.from_float(distance),
        frame_sign=frame_sign,
    )
    builder = _Builder()
    vertex_point = np.asarray(
        offset_patch.evaluate(jnp.zeros((2,), dtype=jnp.float64)),
        dtype=np.float64,
    )
    builder.vertex(vertex_point)
    for edge, axis_index in enumerate(edge_axes):
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
        offset_curve = SurfaceIsoparametricCurve(
            offset_patch, 1 - axis_index, 0.0, parameter_range=(first, last)
        )
        builder.edge(
            offset_curve,
            first,
            last,
            0,
            0,
            endpoint_roots=_offset_native_turn_roots(
                offset_curve, offset_patch, axis_index
            ),
        )
    for coedge, edge in enumerate(geometry.coedge_edges):
        axis_index, shifts = coedge_lines[coedge]
        direction = (
            (Fraction(1), Fraction(0)) if axis_index == 0 else (Fraction(0), Fraction(1))
        )
        line = LineCurve(
            (0.0, 0.0),
            tuple(float(value) for value in direction),
            origin_exact=(Fraction(0), Fraction(0)),
            direction_exact=direction,
        )
        pcurve = PeriodicPCurve(line, offset_patch, shifts) if any(shifts) else line
        builder.coedge(
            edge,
            geometry.coedge_senses[coedge],
            pcurve,
            endpoint_roots=_offset_native_turn_roots(pcurve, offset_patch, axis_index),
        )
    builder.face(
        offset_patch,
        bounds,
        int(np.asarray(model.orientation, dtype=np.float64)[0]),
        model.physical_tags[0],
        [list(loop) for loop in geometry.face_loops[0]],
    )
    result_geometry, patches, parameter_bounds, orientations, tags, trims = (
        builder.geometry(
            solid=True,
            shell_faces=geometry.shell_faces,
            shell_orientations=geometry.shell_orientations,
            solid_shells=geometry.solid_shells,
            assembly_graph=geometry.assembly_graph,
        )
    )
    result_model = assemble_brep_model(
        result_geometry,
        patches,
        parameter_bounds,
        orientations,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
    )
    return exact_operation_offset_result(
        model,
        result_model,
        include_parameter_range=include_parameter_range,
    )


def _single_chart_sphere_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    include_parameter_range: bool,
) -> BRepOffsetResult | None:
    from ._boolean_core import AffinePCurve, PeriodicPCurve
    from ._carrier_pieces import carrier_payload
    from ._intersection import NativePeriodEndpoint
    from ._offset_result import exact_operation_offset_result

    geometry = model.geometry
    occurrences = tuple(geometry.assembly_graph.iter_occurrences())
    identity = BRepAffinePlacement.identity().identity_payload()
    if (
        len(occurrences) != 1
        or occurrences[0].solid_index != 0
        or occurrences[0].placement.identity_payload() != identity
    ):
        return None
    if (
        len(model.patches) != 1
        or geometry.topology.num_solids != 1
        or geometry.topology.num_faces != 1
        or geometry.topology.num_edges != 3
        or geometry.topology.num_vertices != 2
        or len(geometry.coedge_edges) != 4
        or len(geometry.face_loops) != 1
        or len(geometry.face_loops[0]) != 1
        or len(geometry.face_loops[0][0]) != 4
    ):
        return None
    patch = model.patches[0]
    has_source_pole_roots = any(root is not None for root in geometry.vertex_roots)
    if has_source_pole_roots and not isinstance(patch, RevolutionSurface):
        return None

    def exact_vector(value: ConvertibleToArray, /) -> tuple[Fraction, ...]:
        return tuple(Fraction.from_float(float(item)) for item in np.asarray(value))

    def exact_dot(
        first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
    ) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    sphere_center_exact: tuple[ExactScalar, ...]
    sphere_radius_exact: ExactScalar
    sphere_axis_exact: tuple[Fraction, Fraction, Fraction]
    if isinstance(patch, SpherePatch):
        axes = tuple(
            exact_vector(value)
            for value in (patch.first_axis, patch.second_axis, patch.axis)
        )
        gram = tuple(
            tuple(exact_dot(axes[first], axes[second]) for second in range(3))
            for first in range(3)
        )
        if gram != tuple(
            tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
        ):
            return None
        rows = tuple(tuple(axes[column][row] for column in range(3)) for row in range(3))
        determinant = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        if determinant not in (-1, 1):
            return None
        sphere_center_exact = exact_vector(patch.center)
        sphere_radius_exact = Fraction.from_float(float(patch.radius))
        sphere_axis_exact = (axes[2][0], axes[2][1], axes[2][2])
        sphere_frame_orientation = int(determinant)
    elif isinstance(patch, RevolutionSurface):
        sphere_witness = patch.exact_sphere_revolution_witness()
        if sphere_witness is None:
            return None
        (
            sphere_center_exact,
            sphere_radius_exact,
            sphere_axis_exact,
            sphere_frame_orientation,
        ) = sphere_witness
    else:
        return None

    shell_signs = geometry.topology.solid_face_orientations[0]
    if len(shell_signs) != 1:
        return None
    material_sign = (
        int(np.asarray(model.orientation, dtype=np.float64)[0]) * shell_signs[0]
    )
    frame_sign = sphere_frame_orientation * material_sign
    if frame_sign not in (-1, 1):
        return None
    if sphere_radius_exact <= 0:
        return None
    offset_radius = sphere_radius_exact + frame_sign * Fraction.from_float(distance)
    if offset_radius <= 0:
        raise ValueError("brep_offset.sphere must retain a positive radius after offset.")

    bounds = np.asarray(model.parameter_bounds[0], dtype=np.float64)
    expected_symbols = (
        ("two_pi", None) if isinstance(patch, SpherePatch) else ("two_pi", "two_pi")
    )
    if patch.native_period_symbols() != expected_symbols:
        return None
    period = patch.periods[0]
    if period is None or not isfinite(float(period)) or period <= 0.0:
        return None
    period_exact = Fraction.from_float(float(period))
    profile_bounds = (
        Fraction.from_float(float(bounds[0, 1])),
        Fraction.from_float(float(bounds[1, 1])),
    )
    if (
        Fraction.from_float(float(bounds[0, 0])) != 0
        or Fraction.from_float(float(bounds[1, 0])) != period_exact
    ):
        return None
    if isinstance(patch, SpherePatch):
        profile_limits = (
            Fraction.from_float(-0.5 * pi),
            Fraction.from_float(0.5 * pi),
        )
        if profile_bounds != profile_limits:
            return None
    else:
        profile = patch.curve
        profile_period = patch.periods[1]
        if (
            not isinstance(profile, CircleCurve)
            or profile_period is None
            or not isfinite(float(profile_period))
            or profile_period <= 0.0
            or profile.native_period_symbol() != "two_pi"
        ):
            return None
        quarter_turn = Fraction.from_float(0.25 * float(profile_period))
        profile_limits = (-quarter_turn, quarter_turn)
        # Rooted pole vertices widen the chart to their parameter enclosures.
        if not (
            profile_bounds[0] <= profile_limits[0]
            and profile_limits[1] <= profile_bounds[1]
        ):
            return None
        profile_axis = tuple(profile.second_axis_exact)
        if (
            profile_axis != sphere_axis_exact
            and tuple(-value for value in profile_axis) != sphere_axis_exact
        ):
            return None

    uses: list[list[int]] = [[] for _ in geometry.edge_curves]
    for coedge, edge in enumerate(geometry.coedge_edges):
        uses[edge].append(coedge)
    degenerate_edges = tuple(
        edge for edge, curve_index in enumerate(geometry.edge_curves) if curve_index < 0
    )
    profile_edges = tuple(
        edge for edge, curve_index in enumerate(geometry.edge_curves) if curve_index >= 0
    )
    if len(degenerate_edges) != 2 or len(profile_edges) != 1:
        return None

    def line_data(pcurve: AbstractCurve, /) -> tuple[int, int, Fraction] | None:
        # A forward seam use lives on its closing native-period sheet.
        shifts = (0, 0)
        if isinstance(pcurve, PeriodicPCurve):
            shifts, pcurve = pcurve.period_shifts, pcurve.source_curve
        if not isinstance(pcurve, LineCurve):
            return None
        origin_exact, direction_exact = pcurve.origin_exact, pcurve.direction_exact
        if len(origin_exact) != 2 or len(direction_exact) != 2:
            return None
        first, second = origin_exact
        first_direction, second_direction = direction_exact
        if (
            not isinstance(first, Fraction)
            or not isinstance(second, Fraction)
            or not isinstance(first_direction, Fraction)
            or not isinstance(second_direction, Fraction)
        ):
            return None
        origin = (first, second)
        direction = (first_direction, second_direction)
        variable = tuple(
            axis for axis in range(2) if direction[axis] == 1 and direction[1 - axis] == 0
        )
        if len(variable) != 1 or shifts[variable[0]]:
            return None
        variable_axis = variable[0]
        fixed_axis = 1 - variable_axis
        if shifts[fixed_axis] and fixed_axis != 0:
            return None
        return (
            variable_axis,
            fixed_axis,
            origin[fixed_axis] + shifts[fixed_axis] * period_exact,
        )

    def rooted_pole(pcurve: AbstractCurve, vertex: int, /) -> bool:
        root = geometry.vertex_roots[vertex]
        if (
            not isinstance(patch, RevolutionSurface)
            or not isinstance(root, BRepVertexRoot)
            or not isinstance(root.primary, BRepConicVertex)
            or not isinstance(pcurve, AffinePCurve)
            or not isinstance(pcurve.curve, ConicRootLinePCurve)
        ):
            return False
        line = pcurve.curve
        source = root.primary
        return (
            pcurve.matrix == ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
            and pcurve.offset == (Fraction(0), Fraction(0))
            and line.height_origin_exact == 0
            and line.height_direction_exact == 1
            and line.phase_shift_exact == 0
            and line.root.root_id == source.root.root_id
            and carrier_payload(line.root.conic) == carrier_payload(patch.curve)
            and source.root.exact_point() == source.point_exact
            and not any(source.offset_exact)
        )

    degenerate_vertices: set[int] = set()
    for edge in degenerate_edges:
        first, last = (
            Fraction.from_float(float(value))
            for value in np.asarray(geometry.edge_ranges[edge])
        )
        endpoints = geometry.edge_vertices[edge]
        coedges = uses[edge]
        if (
            endpoints[0] != endpoints[1]
            or len(coedges) != 1
            or first != 0
            or last != period_exact
        ):
            return None
        pcurve = geometry.pcurves[coedges[0]]
        if has_source_pole_roots:
            if not rooted_pole(pcurve, endpoints[0]):
                return None
        else:
            info = line_data(pcurve)
            if info is None or info[0] != 0 or info[2] not in profile_limits:
                return None
        roots = geometry.edge_endpoint_roots[edge]
        start_root, end_root = roots
        if not isinstance(start_root, NativePeriodEndpoint) or not isinstance(
            end_root, NativePeriodEndpoint
        ):
            return None
        if not (
            carrier_payload(start_root.curve) == carrier_payload(pcurve)
            and carrier_payload(end_root.curve) == carrier_payload(pcurve)
            and start_root.patch is not None
            and end_root.patch is not None
            and carrier_payload(start_root.patch) == carrier_payload(patch)
            and carrier_payload(end_root.patch) == carrier_payload(patch)
            and start_root.axis == end_root.axis == 0
            and start_root.rational == end_root.rational == 0
            and start_root.turns == 0
            and end_root.turns == 1
        ):
            return None
        coedge_roots = geometry.coedge_endpoint_roots[coedges[0]]
        if any(root is not None for root in coedge_roots):
            coedge_start, coedge_end = coedge_roots
            if not isinstance(coedge_start, NativePeriodEndpoint) or not isinstance(
                coedge_end, NativePeriodEndpoint
            ):
                return None
            if not (
                carrier_payload(coedge_start.curve) == carrier_payload(pcurve)
                and carrier_payload(coedge_end.curve) == carrier_payload(pcurve)
                and coedge_start.patch is not None
                and coedge_end.patch is not None
                and carrier_payload(coedge_start.patch) == carrier_payload(patch)
                and carrier_payload(coedge_end.patch) == carrier_payload(patch)
                and coedge_start.axis == coedge_end.axis == 0
                and coedge_start.rational == coedge_end.rational == 0
                and coedge_start.turns == 0
                and coedge_end.turns == 1
            ):
                return None
        degenerate_vertices.add(endpoints[0])

    profile_edge = profile_edges[0]
    profile_range = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(geometry.edge_ranges[profile_edge])
    )
    profile_uses = uses[profile_edge]
    profile_curve = geometry.curves[geometry.edge_curves[profile_edge]]
    if (
        len(profile_uses) != 2
        or not isinstance(profile_curve, (CircleCurve, SurfaceIsoparametricCurve))
        or profile_range != profile_limits
    ):
        return None
    source_pole_roots: dict[int, BRepConicVertex] = {}
    if isinstance(patch, RevolutionSurface):
        if not isinstance(profile_curve, CircleCurve) or carrier_payload(
            profile_curve
        ) != carrier_payload(patch.curve):
            return None
        profile_vertices = geometry.edge_vertices[profile_edge]
        if len(set(profile_vertices)) != 2:
            return None
        actual_poles = tuple(
            tuple(
                Fraction.from_float(float(value))
                for value in geometry.vertex_points[vertex]
            )
            for vertex in profile_vertices
        )
        expected_poles = (
            tuple(
                sphere_center_exact[axis] - sphere_radius_exact * sphere_axis_exact[axis]
                for axis in range(3)
            ),
            tuple(
                sphere_center_exact[axis] + sphere_radius_exact * sphere_axis_exact[axis]
                for axis in range(3)
            ),
        )
        if actual_poles not in (expected_poles, expected_poles[::-1]):
            return None
        if has_source_pole_roots:
            if any(root is None for root in geometry.vertex_roots):
                return None
            for vertex, actual_pole in zip(profile_vertices, actual_poles, strict=True):
                root = geometry.vertex_roots[vertex]
                if not isinstance(root, BRepVertexRoot) or not isinstance(
                    root.primary, BRepConicVertex
                ):
                    return None
                primary = root.primary
                if (
                    carrier_payload(primary.root.conic) != carrier_payload(profile_curve)
                    or any(primary.offset_exact)
                    or primary.point_exact != actual_pole
                    or primary.root.exact_point() != primary.point_exact
                ):
                    return None
                source_pole_roots[vertex] = primary
    elif has_source_pole_roots:
        return None

    profile_edge_roots = geometry.edge_endpoint_roots[profile_edge]
    if has_source_pole_roots:
        for endpoint, root in enumerate(profile_edge_roots):
            vertex = geometry.edge_vertices[profile_edge][endpoint]
            primary = source_pole_roots.get(vertex)
            if (
                not isinstance(root, ConicParameterEndpoint)
                or primary is None
                or carrier_payload(root.carrier) != carrier_payload(profile_curve)
                or root.root.root_id != primary.root.root_id
                or root.point_exact != primary.point_exact
                or (
                    root.parameter_exact is not None
                    and root.parameter_exact != profile_range[endpoint]
                )
            ):
                return None
    elif any(root is not None for root in profile_edge_roots):
        return None
    fixed_longitudes: list[Fraction] = []
    for coedge in profile_uses:
        info = line_data(geometry.pcurves[coedge])
        if info is None or info[0] != 1:
            return None
        fixed_longitudes.append(info[2])
        if has_source_pole_roots:
            for endpoint, root in enumerate(geometry.coedge_endpoint_roots[coedge]):
                vertex = geometry.edge_vertices[profile_edge][endpoint]
                primary = source_pole_roots.get(vertex)
                if (
                    not isinstance(root, ConicParameterEndpoint)
                    or primary is None
                    or carrier_payload(root.carrier)
                    != carrier_payload(geometry.pcurves[coedge])
                    or root.root.root_id != primary.root.root_id
                    or root.point_exact != primary.point_exact
                    or (
                        root.parameter_exact is not None
                        and root.parameter_exact != profile_range[endpoint]
                    )
                ):
                    return None
        elif any(root is not None for root in geometry.coedge_endpoint_roots[coedge]):
            return None
    if (
        len(fixed_longitudes) != 2
        or ((fixed_longitudes[1] - fixed_longitudes[0]) / period_exact).denominator != 1
    ):
        return None
    if (
        set(geometry.edge_vertices[profile_edge]) != degenerate_vertices
        or len(degenerate_vertices) != 2
    ):
        return None

    offset_patch = OffsetSurface(
        patch,
        Fraction.from_float(distance),
        frame_sign=frame_sign,
    )
    target_vertex_roots: list[BRepVertexRoot | None] = [
        None
    ] * geometry.topology.num_vertices
    target_conic_roots: dict[int, ConicParameterRoot] = {}
    if has_source_pole_roots:
        if not isinstance(patch, RevolutionSurface) or not isinstance(
            profile_curve, CircleCurve
        ):
            return None
        target_profile = CircleCurve(
            tuple(float(value) for value in profile_curve.center_exact),
            tuple(float(value) for value in profile_curve.first_axis_exact),
            tuple(float(value) for value in profile_curve.second_axis_exact),
            float(offset_radius),
            center_exact=profile_curve.center_exact,
            first_axis_exact=profile_curve.first_axis_exact,
            second_axis_exact=profile_curve.second_axis_exact,
            radius_exact=offset_radius,
        )
        distance_exact = Fraction.from_float(distance)
        for vertex, source_primary in source_pole_roots.items():
            source_direction = tuple(
                (value - center) / sphere_radius_exact
                for value, center in zip(
                    source_primary.point_exact, sphere_center_exact, strict=True
                )
            )
            offset_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = (
                frame_sign * distance_exact * source_direction[0],
                frame_sign * distance_exact * source_direction[1],
                frame_sign * distance_exact * source_direction[2],
            )
            target_primary = BRepConicVertex(
                source_primary.root, offset_exact=offset_exact
            )
            expected_point = tuple(
                center + offset_radius * direction
                for center, direction in zip(
                    sphere_center_exact, source_direction, strict=True
                )
            )
            if target_primary.point_exact != expected_point:
                return None
            target_vertex_roots[vertex] = BRepVertexRoot(target_primary)
            target_conic_root = ConicParameterRoot(
                target_profile,
                source_primary.root.cosine_exact,
                source_primary.root.sine_exact,
                turns=source_primary.root.turns,
            )
            if target_conic_root.exact_point() != expected_point:
                return None
            target_conic_roots[vertex] = target_conic_root

    def offset_pole_endpoint(
        vertex: int,
        carrier: AbstractCurve,
        parameter_exact: Fraction | None,
        /,
    ) -> ConicParameterEndpoint | None:
        target_root = target_conic_roots.get(vertex)
        vertex_root = target_vertex_roots[vertex]
        if (
            target_root is None
            or vertex_root is None
            or not isinstance(vertex_root.primary, BRepConicVertex)
        ):
            return None
        return ConicParameterEndpoint(
            target_root,
            carrier,
            parameter_exact,
            vertex_root.primary.point_exact,
        )

    vertex_parameters: list[np.ndarray | None] = [None] * geometry.topology.num_vertices
    for edge, coedges in enumerate(uses):
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
        for coedge in coedges:
            pcurve = geometry.pcurves[coedge]
            for endpoint, parameter in enumerate((first, last)):
                vertex = geometry.edge_vertices[edge][endpoint]
                if vertex_parameters[vertex] is None:
                    vertex_parameters[vertex] = np.asarray(
                        pcurve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)),
                        dtype=np.float64,
                    )
    if any(parameters is None for parameters in vertex_parameters):
        return None
    builder = _Builder()
    vertex_map: list[int] = []
    for vertex, parameters in enumerate(vertex_parameters):
        if parameters is None:
            return None
        root = target_vertex_roots[vertex]
        point = (
            np.asarray(
                tuple(float(value) for value in root.primary.point_exact),
                dtype=np.float64,
            )
            if root is not None and isinstance(root.primary, BRepConicVertex)
            else np.asarray(
                offset_patch.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
                dtype=np.float64,
            )
        )
        vertex_map.append(builder.vertex(point, root=root))
    edge_map: list[int] = []
    for edge, curve_index in enumerate(geometry.edge_curves):
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
        if curve_index < 0:
            vertex = geometry.edge_vertices[edge][0]
            edge_map.append(
                builder.degenerate(
                    vertex_map[vertex],
                    first,
                    last,
                    endpoint_roots=_offset_period_roots(
                        geometry.edge_endpoint_roots[edge],
                        geometry.pcurves[uses[edge][0]],
                        offset_patch,
                    ),
                )
            )
        else:
            # The edge carrier follows its base-sheet use, not the closing sheet.
            pcurve = min(
                (geometry.pcurves[coedge] for coedge in uses[edge]),
                key=lambda value: isinstance(value, PeriodicPCurve),
            )
            info = line_data(pcurve)
            if info is None:
                raise ValueError(
                    "Every admitted sphere offset edge needs its line chart."
                )
            # The carrier spans the chart so rooted pole enclosures stay in domain.
            curve = SurfaceIsoparametricCurve(
                offset_patch,
                info[1],
                float(info[2]),
                parameter_range=(
                    float(min(first, bounds[0, info[0]])),
                    float(max(last, bounds[1, info[0]])),
                ),
            )
            start, end = geometry.edge_vertices[edge]
            endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (
                None,
                None,
            )
            if edge == profile_edge and has_source_pole_roots:
                source_roots = geometry.edge_endpoint_roots[edge]
                first_root = source_roots[0]
                last_root = source_roots[1]
                if not isinstance(first_root, ConicParameterEndpoint) or not isinstance(
                    last_root, ConicParameterEndpoint
                ):
                    return None
                target_first = offset_pole_endpoint(
                    start, curve, first_root.parameter_exact
                )
                target_last = offset_pole_endpoint(end, curve, last_root.parameter_exact)
                if target_first is None or target_last is None:
                    return None
                endpoint_roots = target_first, target_last
            edge_map.append(
                builder.edge(
                    curve,
                    first,
                    last,
                    vertex_map[start],
                    vertex_map[end],
                    endpoint_roots=endpoint_roots,
                )
            )
    for coedge, edge in enumerate(geometry.coedge_edges):
        pcurve = _offset_chart_pcurve(geometry.pcurves[coedge], patch, offset_patch)
        roots = geometry.coedge_endpoint_roots[coedge]
        rebound_roots: list[RootEndpoint | None] = []
        for endpoint, root in enumerate(roots):
            if isinstance(root, NativePeriodEndpoint):
                rebound_roots.append(
                    NativePeriodEndpoint(
                        pcurve,
                        offset_patch,
                        root.axis,
                        rational=root.rational,
                        turns=root.turns,
                    )
                )
            elif (
                isinstance(root, ConicParameterEndpoint)
                and edge == profile_edge
                and has_source_pole_roots
            ):
                vertex = geometry.edge_vertices[edge][endpoint]
                rebound = offset_pole_endpoint(vertex, pcurve, root.parameter_exact)
                if rebound is None:
                    return None
                rebound_roots.append(rebound)
            else:
                rebound_roots.append(root)
        builder.coedge(
            edge_map[edge],
            geometry.coedge_senses[coedge],
            pcurve,
            endpoint_roots=(rebound_roots[0], rebound_roots[1]),
        )
    builder.face(
        offset_patch,
        bounds,
        int(np.asarray(model.orientation, dtype=np.float64)[0]),
        model.physical_tags[0],
        [list(loop) for loop in geometry.face_loops[0]],
    )
    result_geometry, patches, parameter_bounds, orientations, tags, trims = (
        builder.geometry(
            solid=True,
            shell_faces=geometry.shell_faces,
            shell_orientations=geometry.shell_orientations,
            solid_shells=geometry.solid_shells,
            assembly_graph=geometry.assembly_graph,
        )
    )
    result_model = assemble_brep_model(
        result_geometry,
        patches,
        parameter_bounds,
        orientations,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
    )
    return exact_operation_offset_result(
        model,
        result_model,
        include_parameter_range=include_parameter_range,
    )


def _single_chart_revolution_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    include_parameter_range: bool,
) -> BRepOffsetResult | None:
    from ._intersection import ConicParameterEndpoint
    from ._offset_result import (
        certify_revolution_offset_embedding,
        exact_operation_offset_result,
    )
    from ._root_bindings import revolution_offset_conic_frame

    if len(model.patches) != 1 or not isinstance(model.patches[0], RevolutionSurface):
        return None
    patch = model.patches[0]
    if not isinstance(patch.curve, BSplineCurve):
        return None
    geometry = model.geometry
    if (
        not geometry.topology.solid_face_orientations
        or len(geometry.topology.solid_face_orientations[0]) != 1
    ):
        return None
    frame_sign = (
        int(np.asarray(model.orientation, dtype=np.float64)[0])
        * geometry.topology.solid_face_orientations[0][0]
    )
    if frame_sign not in (-1, 1):
        return None
    offset_patch = OffsetSurface(
        patch,
        Fraction.from_float(distance),
        frame_sign=frame_sign,
    )
    proof = certify_revolution_offset_embedding(model, offset_patch)
    if proof.status == "unresolved":
        raise ValueError(
            f"brep_offset revolution embedding is unresolved: {proof.reason}"
        )
    if proof.status != "certified":
        return None

    profile_edges = tuple(
        edge
        for edge, curve_index in enumerate(geometry.edge_curves)
        if curve_index >= 0 and isinstance(geometry.curves[curve_index], BSplineCurve)
    )
    turn_edges = tuple(
        edge
        for edge, curve_index in enumerate(geometry.edge_curves)
        if curve_index >= 0 and isinstance(geometry.curves[curve_index], CircleCurve)
    )
    if len(profile_edges) != 1 or len(turn_edges) != 1:
        return None
    profile_edge, turn_edge = profile_edges[0], turn_edges[0]
    profile_coedges = tuple(
        coedge
        for coedge, edge in enumerate(geometry.coedge_edges)
        if edge == profile_edge
    )
    turn_coedges = tuple(
        coedge for coedge, edge in enumerate(geometry.coedge_edges) if edge == turn_edge
    )
    profile_coedge = next(
        (
            coedge
            for coedge in profile_coedges
            if isinstance(pcurve := geometry.pcurves[coedge], LineCurve)
            and pcurve.origin_exact == (Fraction(0), Fraction(0))
            and pcurve.direction_exact == (Fraction(0), Fraction(1))
        ),
        None,
    )
    if profile_coedge is None or not turn_coedges:
        return None

    control_curve = patch.curve
    parameter_lower = rational_coefficient(
        control_curve.exact_knots[control_curve.degree], role="profile knot"
    )
    parameter_upper = rational_coefficient(
        control_curve.exact_knots[len(control_curve.exact_control_points)],
        role="profile knot",
    )
    first_frame = revolution_offset_conic_frame(offset_patch, parameter_lower)
    last_frame = revolution_offset_conic_frame(offset_patch, parameter_upper)
    if first_frame is None or last_frame is None or first_frame != last_frame:
        return None
    point_exact, center_exact, first_axis_exact, second_axis_exact, radius_exact = (
        first_frame
    )
    circle = CircleCurve(
        tuple(float(value) for value in center_exact),
        tuple(float(value) for value in first_axis_exact),
        tuple(float(value) for value in second_axis_exact),
        float(radius_exact),
        center_exact=center_exact,
        first_axis_exact=first_axis_exact,
        second_axis_exact=second_axis_exact,
        radius_exact=radius_exact,
    )
    root = ConicParameterRoot(circle, Fraction(1), Fraction(0))
    vertex_root = BRepVertexRoot(BRepConicVertex(root))
    builder = _Builder()
    vertex = builder.vertex(
        tuple(float(value) for value in point_exact),
        root=vertex_root,
    )

    face_bounds = np.asarray(model.parameter_bounds[0], dtype=np.float64)
    u_first, v_first = (float(value) for value in face_bounds[0])
    u_last, v_last = (float(value) for value in face_bounds[1])
    profile_pcurve = geometry.pcurves[profile_coedge]
    profile_curve = SurfaceMappedCurve(offset_patch, profile_pcurve)
    turn_curve = SurfaceIsoparametricCurve(
        offset_patch,
        1,
        v_first,
        parameter_range=(u_first, u_last),
    )
    edge_map: dict[int, int] = {}
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for edge in range(len(geometry.edge_curves)):
        first, last = (float(value) for value in ranges[edge])
        if edge == profile_edge:
            curve = profile_curve
            endpoint_roots = (
                ConicParameterEndpoint(root, curve, parameter_lower, point_exact),
                ConicParameterEndpoint(root, curve, parameter_upper, point_exact),
            )
        elif edge == turn_edge:
            curve = turn_curve
            endpoint_roots = (None, None)
        else:
            return None
        edge_map[edge] = builder.edge(
            curve,
            first,
            last,
            vertex,
            vertex,
            endpoint_roots=endpoint_roots,
        )

    coedge_map: dict[int, int] = {}
    for coedge, edge in enumerate(geometry.coedge_edges):
        pcurve = geometry.pcurves[coedge]
        if edge == profile_edge:
            roots = (
                ConicParameterEndpoint(root, pcurve, parameter_lower, point_exact),
                ConicParameterEndpoint(root, pcurve, parameter_upper, point_exact),
            )
        else:
            roots = (None, None)
        coedge_map[coedge] = builder.coedge(
            edge_map[edge],
            geometry.coedge_senses[coedge],
            pcurve,
            endpoint_roots=roots,
        )
        if edge == turn_edge:
            builder.native_phase(
                edge_map[edge], offset_patch, 0, coedge=coedge_map[coedge]
            )
    builder.face(
        offset_patch,
        face_bounds,
        int(np.asarray(model.orientation, dtype=np.float64)[0]),
        model.physical_tags[0],
        [[coedge_map[coedge] for coedge in loop] for loop in geometry.face_loops[0]],
    )
    result_geometry, patches, parameter_bounds, orientations, tags, trims = (
        builder.geometry(
            solid=True,
            shell_faces=geometry.shell_faces,
            shell_orientations=geometry.shell_orientations,
            solid_shells=geometry.solid_shells,
            assembly_graph=geometry.assembly_graph,
        )
    )
    result_model = assemble_brep_model(
        result_geometry,
        patches,
        parameter_bounds,
        orientations,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        source_digest=canonical_fingerprint(
            {
                "kind": "native-spline-revolution-offset",
                "source_revision": model.source_revision,
                "offset_surface": offset_patch.identity_payload(),
            }
        ),
        tessellation=tessellation,
        trim_domains=trims,
    )
    return exact_operation_offset_result(
        model,
        result_model,
        include_parameter_range=include_parameter_range,
    )


def brep_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-offset",
    include_parameter_range: bool = False,
) -> "BRepOffsetResult":
    """Offset source-proven planar solids, regular analytic patches, and certified
    closed B-spline meridian revolutions."""
    if not isinstance(model, BRepModel):
        raise TypeError("brep_offset.model must be a BRepModel.")
    if isinstance(distance, bool) or not isinstance(distance, (Integral, Real)):
        raise TypeError("brep_offset.distance must be a finite nonzero real scalar.")
    distance_ = float(distance)
    if not isfinite(distance_) or distance_ == 0.0:
        raise ValueError("brep_offset.distance must be finite and nonzero.")
    from ._planar_offset import planar_carrier, planar_offset

    if model.patches and all(
        planar_carrier(patch) is not None for patch in model.patches
    ):
        return planar_offset(
            model,
            distance_,
            tessellation=tessellation,
            source_id=source_id,
            include_parameter_range=include_parameter_range,
        )
    if (
        len(model.patches) == 1
        and isinstance(model.patches[0], RevolutionSurface)
        and isinstance(model.patches[0].curve, BSplineCurve)
    ):
        result = _single_chart_revolution_offset(
            model,
            distance_,
            tessellation=tessellation,
            source_id=source_id,
            include_parameter_range=include_parameter_range,
        )
        if result is None:
            raise NotImplementedError(
                "The B-spline revolution offset lacks a complete source endpoint frame."
            )
        return result
    for offsetter in (_single_chart_torus_offset, _single_chart_sphere_offset):
        result = offsetter(
            model,
            distance_,
            tessellation=tessellation,
            source_id=source_id,
            include_parameter_range=include_parameter_range,
        )
        if result is not None:
            return result
    raise NotImplementedError(
        "brep_offset supports source-proven planar solids, regular single-chart "
        "spheres, ring tori, and aligned elliptic revolutions."
    )


__all__ = [
    "BRepTessellationPolicy",
    "PlanarProfile",
    "ProfileArc",
    "ProfileCurveUse",
    "ProfileLine",
    "ProfileLoop",
    "ProfilePlane",
    "assemble_brep_model",
    "brep_box",
    "brep_cone",
    "brep_cylinder",
    "brep_extrusion",
    "brep_offset",
    "brep_planar_face",
    "brep_revolution",
    "brep_sphere",
    "brep_torus",
    "brep_trim_domain",
]

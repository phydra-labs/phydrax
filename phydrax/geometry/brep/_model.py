#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import assert_never, final, Literal, Self, TypeAlias, TypedDict, Unpack

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.geometry.brep._correspondence as correspondence_owner

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, ConvertibleToArray, parse
from .._atlas import AbstractBoundaryMap, BoundaryAtlas, TrimDomain
from .._chart_restriction import ExactRationalChartRestriction
from .._rational import rational_vector
from . import _carrier_pieces as carrier_owner
from ._conic_gaps import endpoint_gap_bounds, native_face_gap_squared
from ._coordinate_request import coordinate_source_request
from ._exact import exact_determinant, gram_deviation_bound, square_root_interval
from ._intersection import (
    BranchRootEndpoint,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import (
    affine_curve_coefficients,
    encode_geometry,
    IntersectionCurve,
    IntersectionPCurve,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    curve_evaluate_host,
    LineCurve,
    PlanePatch,
)
from ._placed import PlacedSurface
from ._request_preparations import TrimRequestPreparations
from ._root_bindings import BRepPlacedVertex, BRepVertexRoot
from ._source_correspondence import norm_upper, placed_deviation_bound


type BRepCurve = AbstractCurve | IntersectionCurve
type BRepPCurve = AbstractCurve | IntersectionPCurve


def _carrier_points_host(
    curve: BRepCurve | BRepPCurve,
    parameters: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> np.ndarray:
    """Evaluate host preparation through the carrier's nominal numerical plan."""
    match curve:
        case AbstractCurve():
            return curve_evaluate_host(curve, parameters, budget=budget)
        case IntersectionCurve():
            result = curve.evaluate(
                jnp.asarray(parameters, dtype=jnp.float64),
                budget=budget,
                preparations=preparations,
            )
            if not curve.fully_certified or not np.all(
                np.isfinite(np.asarray(result.parameter_bound))
            ):
                raise ValueError(
                    "BRepGeometry.curve must have resolved continuation evidence; got an unresolved intersection evaluation"
                )
            return np.asarray(result.point)
        case IntersectionPCurve():
            return np.asarray(
                curve.evaluate(
                    jnp.asarray(parameters, dtype=jnp.float64),
                    budget=budget,
                    preparations=preparations,
                )
            )
        case _:
            assert_never(curve)


def _native_gap_endpoint_roots(
    roots: tuple[RootEndpoint | None, RootEndpoint | None],
    /,
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    """Admit native periodic atoms, not unresolved intersection endpoints."""
    if len(roots) != 2:
        raise ValueError(
            "BRepGeometry.gap_endpoint_roots must contain two bindings; got invalid rows"
        )
    admitted: list[NativePeriodEndpoint | None] = []
    for root in roots:
        if root is not None and not isinstance(root, NativePeriodEndpoint):
            raise ValueError(
                "BRepGeometry.gap_endpoint_roots must use authored scalar or native periodic endpoints; got unsupported root"
            )
        admitted.append(root)
    return admitted[0], admitted[1]


def _certify_endpoint_gaps(
    curve: BRepCurve,
    points: np.ndarray,
    vertices: tuple[int, int],
    parameters: tuple[float, float],
    bounds: tuple[float, float],
    vertex_roots: tuple[BRepVertexRoot | None, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (None, None),
) -> None:
    """Recompute native endpoint bounds without trusting authored evidence rows."""
    if not isinstance(curve, AbstractCurve) or any(
        vertex_roots[vertex] is not None for vertex in vertices
    ):
        raise ValueError(
            "BRepGeometry.edge_vertex_gap_bounds must use authored native endpoints; got unsupported carrier or root"
        )
    native_roots = _native_gap_endpoint_roots(endpoint_roots)
    if not isinstance(curve, LineCurve) or any(root is not None for root in native_roots):
        certified = endpoint_gap_bounds(
            curve,
            parameters,
            points[np.asarray(vertices, dtype=np.int64)],
            budget=budget,
            endpoint_roots=native_roots,
        )
        if any(
            actual > declared for actual, declared in zip(certified, bounds, strict=True)
        ):
            raise ValueError(
                "BRepGeometry.edge_vertex_gap_bounds must contain certified endpoint distances; got exceeded bound"
            )
        return
    # Keep rational squared comparisons for the established affine line family.
    budget.reserve(64)
    for parameter, vertex, bound in zip(parameters, vertices, bounds, strict=True):
        squared = sum(
            (
                (
                    Fraction(float(origin))
                    + Fraction(parameter) * Fraction(float(direction))
                    - Fraction(float(point))
                )
                ** 2
                for origin, direction, point in zip(
                    np.asarray(curve.origin),
                    np.asarray(curve.direction),
                    points[vertex],
                    strict=True,
                )
            ),
            Fraction(0),
        )
        if squared > Fraction(bound) ** 2:
            raise ValueError(
                "BRepGeometry.edge_vertex_gap_bounds must contain certified endpoint distances; got exceeded bound"
            )


def _edge_face_gap_squared(
    curve: BRepCurve,
    pcurve: BRepPCurve,
    patch: AbstractSurfacePatch,
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Recompute whole-range affine or native conic source lift discrepancies."""
    if not isinstance(curve, AbstractCurve) or not isinstance(pcurve, AbstractCurve):
        return None
    if not isinstance(patch, PlanePatch):
        return native_face_gap_squared(curve, pcurve, patch, first, last, budget=budget)
    source = affine_curve_coefficients(
        curve, np.zeros((curve.ambient_dimension,), dtype=np.float64)
    )
    uv = affine_curve_coefficients(pcurve, np.zeros((2,), dtype=np.float64))
    if source is None or uv is None:
        return native_face_gap_squared(curve, pcurve, patch, first, last, budget=budget)
    origin = tuple(Fraction(float(value)) for value in np.asarray(patch.origin))
    axes = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis)
    )
    constant = tuple(
        origin[dimension]
        - source[0][dimension]
        + sum((axes[axis][dimension] * uv[0][axis] for axis in range(2)), Fraction(0))
        for dimension in range(3)
    )
    slope = tuple(
        sum((axes[axis][dimension] * uv[1][axis] for axis in range(2)), Fraction(0))
        - source[1][dimension]
        for dimension in range(3)
    )
    # Squared norm of an affine vector is convex, so the endpoint maximum
    # encloses every point, not merely the two evaluated parameter stations.
    return max(
        sum(
            ((a + Fraction(t) * b) ** 2 for a, b in zip(constant, slope, strict=True)),
            Fraction(0),
        )
        for t in (first, last)
    )


def _certify_generated_zero_face_gap(
    curve: BRepCurve,
    coedge: int,
    face: int,
    edge_ranges: np.ndarray,
    pcurves: tuple[BRepPCurve, ...],
    coedge_edges: tuple[int, ...],
    coedge_faces: tuple[int, ...],
    patch: AbstractSurfacePatch,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Require an independent zero-deviation source proof before generating zero."""
    preparations = TrimRequestPreparations()
    with budget.temporary_scope():
        budget.reserve(128 + 32 * len(pcurves), 8192 + 128 * len(pcurves))
        boxes = tuple(
            pcurve.bounding_box(
                float(edge_ranges[coedge_edges[index], 0]),
                float(edge_ranges[coedge_edges[index], 1]),
                preparations=preparations,
                budget=budget,
            )
            for index, pcurve in enumerate(pcurves)
            if coedge_faces[index] == face
        )
        enclosure = np.asarray(boxes, dtype=np.float64)
        box = np.stack((np.min(enclosure[:, 0], axis=0), np.max(enclosure[:, 1], axis=0)))
        edge = coedge_edges[coedge]
        witness = correspondence_owner.certify_curve_surface(
            curve,
            pcurves[coedge],
            patch,
            box,
            float(edge_ranges[edge, 0]),
            float(edge_ranges[edge, 1]),
            point=np.zeros((3,), dtype=np.float64),
            tolerance=0.0,
            preparations=preparations,
            budget=budget,
        )
        if not witness.complete or witness.deviation_bound != 0.0:
            raise ValueError(
                "BRepGeometry.edge_face_gap_bounds must independently prove zero-deviation source correspondence; got unsupported or nonzero gap"
            )


def _gap_coedge_parameter_range(
    edge: int,
    coedge: int,
    ranges: np.ndarray,
    edge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
    coedge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Include periodic source atoms as well as their numerical representatives."""
    first, last = float(ranges[edge, 0]), float(ranges[edge, 1])
    for roots in (
        (None, None) if edge_roots is None else edge_roots[edge],
        (None, None) if coedge_roots is None else coedge_roots[coedge],
    ):
        for endpoint, root in enumerate(_native_gap_endpoint_roots(roots)):
            if root is not None:
                lower, upper = root.parameter_enclosure(budget=budget)
                if endpoint == 0:
                    first = min(first, lower)
                else:
                    last = max(last, upper)
    return first, last


def _certify_edge_face_gap_bounds(
    curves: tuple[BRepCurve, ...],
    edge_curves: tuple[int, ...],
    edge_ranges: np.ndarray,
    pcurves: tuple[BRepPCurve, ...],
    coedge_edges: tuple[int, ...],
    coedge_faces: tuple[int, ...],
    patches: tuple[AbstractSurfacePatch, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    declared_bounds: tuple[float, ...] | None = None,
    exact_unsupported: bool = False,
    edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    | None = None,
    coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    | None = None,
) -> tuple[float, ...] | None:
    """Certify coedge bounds; generation may independently prove exact zero rows."""
    budget.reserve(len(coedge_edges), 32 * len(coedge_edges))
    result: list[float] = []
    for coedge, (edge, face) in enumerate(zip(coedge_edges, coedge_faces, strict=True)):
        with budget.temporary_scope():
            budget.reserve(128, 8192)
            curve_index = edge_curves[edge]
            first, last = _gap_coedge_parameter_range(
                edge,
                coedge,
                edge_ranges,
                edge_endpoint_roots,
                coedge_endpoint_roots,
                budget=budget,
            )
            squared = (
                None
                if curve_index == -1
                else _edge_face_gap_squared(
                    curves[curve_index],
                    pcurves[coedge],
                    patches[face],
                    first,
                    last,
                    budget=budget,
                )
            )
        if squared is None:
            if declared_bounds is None and exact_unsupported and curve_index != -1:
                _certify_generated_zero_face_gap(
                    curves[curve_index],
                    coedge,
                    face,
                    edge_ranges,
                    pcurves,
                    coedge_edges,
                    coedge_faces,
                    patches[face],
                    budget=budget,
                )
            elif declared_bounds is None or declared_bounds[coedge] != 0.0:
                raise ValueError(
                    "BRepGeometry.edge_face_gap_bounds must use a certified carrier family; got unsupported nonzero gap"
                )
            # The normal correspondence validation independently proves zero
            # for these rows; no unsupported positive gap is admitted here.
            result.append(0.0)
            continue
        if (
            declared_bounds is not None
            and squared > Fraction(declared_bounds[coedge]) ** 2
        ):
            raise ValueError(
                "BRepGeometry.edge_face_gap_bounds must contain certified carrier distances; got exceeded bound"
            )
        result.append(square_root_interval(squared)[1])
    return tuple(result) if any(result) else None


def _surface_intervals_valid(bounds: np.ndarray, /) -> bool:
    """Check finite, increasing surface intervals after the caller checks shape."""
    return bool(
        np.all(np.isfinite(bounds)) and not np.any(bounds[:, 1, :] <= bounds[:, 0, :])
    )


BRepEntityKind: TypeAlias = Literal["solid", "shell", "face", "coedge", "edge", "vertex"]


@dataclass(frozen=True, slots=True, order=True)
class BRepEntityId:
    """Stable source-revision-scoped identity for one B-Rep entity."""

    source_revision: str
    kind: str
    index: int
    occurrence_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_revision, str):
            raise TypeError(
                f"BRepEntityId.source_revision must be a string; got {type(self.source_revision).__name__}"
            )
        if not self.source_revision:
            raise ValueError(
                f"BRepEntityId.source_revision must be nonempty; got {self.source_revision!r}"
            )
        if not isinstance(self.kind, str):
            raise TypeError(
                f"BRepEntityId.kind must be a string; got {type(self.kind).__name__}"
            )
        object.__setattr__(
            self, "kind", parse(self.kind, BRepEntityKind, "BRepEntityId.kind")
        )
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError(
                f"BRepEntityId.index must be an integer; got {type(self.index).__name__}"
            )
        if self.index < 0:
            raise ValueError(f"BRepEntityId.index must be nonnegative; got {self.index}")
        if not isinstance(self.occurrence_path, tuple):
            raise TypeError(
                f"BRepEntityId.occurrence_path must be a tuple of names; got {type(self.occurrence_path).__name__}"
            )
        if any(not isinstance(name, str) or not name for name in self.occurrence_path):
            raise ValueError(
                f"BRepEntityId.occurrence_path must contain nonempty strings; got {self.occurrence_path!r}"
            )


@dataclass(frozen=True, slots=True)
class BRepImportReport:
    """Host-side physical provenance and approximation limits for one import."""

    source_id: str
    source_digest: str
    source_format: str
    coordinate_contract: SpatialCoordinateContract
    import_policy_id: str
    num_solids: int
    num_faces: int
    num_edges: int
    num_vertices: int
    num_triangles: int
    linear_deflection: float
    angular_deflection: float
    trim_samples_per_edge: int
    converted_surface_count: int
    curve_surface_tolerance: float = 1.0e-8
    curve_surface_scale: float = 1.0
    source_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.curve_surface_tolerance, bool)
            or not np.isfinite(self.curve_surface_tolerance)
            or self.curve_surface_tolerance < 0.0
        ):
            raise ValueError(
                f"BRepImportReport.curve_surface_tolerance must be a finite nonnegative physical length; got {self.curve_surface_tolerance!r}"
            )
        if (
            isinstance(self.curve_surface_scale, bool)
            or not np.isfinite(self.curve_surface_scale)
            or self.curve_surface_scale < 0.0
        ):
            raise ValueError(
                f"BRepImportReport.curve_surface_scale must be a finite nonnegative physical length; got {self.curve_surface_scale!r}"
            )
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError(
                f"BRepImportReport.source_id must be a nonempty string; got {self.source_id!r}"
            )
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or self.source_digest != self.source_digest.lower()
            or any(
                character not in "0123456789abcdef" for character in self.source_digest
            )
        ):
            raise ValueError(
                f"BRepImportReport.source_digest must be a lowercase SHA-256 digest; got {self.source_digest!r}"
            )
        if not isinstance(self.source_format, str) or not self.source_format:
            raise ValueError(
                f"BRepImportReport.source_format must be a nonempty string; got {self.source_format!r}"
            )
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError(
                f"BRepImportReport.coordinate_contract must be a SpatialCoordinateContract; got {type(self.coordinate_contract).__name__}"
            )
        if (
            not isinstance(self.import_policy_id, str)
            or len(self.import_policy_id) != 64
            or self.import_policy_id != self.import_policy_id.lower()
            or any(
                character not in "0123456789abcdef" for character in self.import_policy_id
            )
        ):
            raise ValueError(
                f"BRepImportReport.import_policy_id must be a lowercase SHA-256 digest; got {self.import_policy_id!r}"
            )
        counts = (
            self.num_solids,
            self.num_faces,
            self.num_edges,
            self.num_vertices,
            self.num_triangles,
            self.converted_surface_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError(f"BRepImportReport.counts must be integers; got {counts!r}")
        if any(value < 0 for value in counts):
            raise ValueError(
                f"BRepImportReport.counts must be nonnegative; got {counts!r}"
            )
        if self.num_faces == 0 and (self.num_solids != 0 or self.num_triangles != 0):
            raise ValueError(
                f"BRepImportReport.num_faces must be positive when solids or surface triangles are present; got faces={self.num_faces}, solids={self.num_solids}, triangles={self.num_triangles}"
            )
        if self.converted_surface_count > self.num_faces:
            raise ValueError(
                f"BRepImportReport.converted_surface_count must not exceed num_faces; got converted={self.converted_surface_count}, faces={self.num_faces}"
            )
        if (
            not np.isfinite(self.linear_deflection)
            or not np.isfinite(self.angular_deflection)
            or self.linear_deflection <= 0.0
            or self.angular_deflection <= 0.0
        ):
            raise ValueError(
                f"BRepImportReport.deflections must be finite and positive; got linear={self.linear_deflection!r}, angular={self.angular_deflection!r}"
            )
        if isinstance(self.trim_samples_per_edge, bool) or not isinstance(
            self.trim_samples_per_edge, int
        ):
            raise TypeError(
                f"BRepImportReport.trim_samples_per_edge must be an integer; got {type(self.trim_samples_per_edge).__name__}"
            )
        if self.trim_samples_per_edge < 3:
            raise ValueError(
                f"BRepImportReport.trim_samples_per_edge must be at least three; got {self.trim_samples_per_edge}"
            )
        object.__setattr__(
            self,
            "source_revision",
            canonical_fingerprint(
                {
                    "kind": "brep-source-revision",
                    "source_digest": self.source_digest,
                    "spatial_id": self.coordinate_contract.spatial_id,
                }
            ),
        )


class BRepTopology(StrictModule):
    """Immutable incidence relations with stable local entity ordering."""

    face_edges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    face_wires: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    solid_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_face_orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    face_solids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_edges: tuple[tuple[int, ...], ...],
        edge_faces: tuple[tuple[int, ...], ...],
        face_wires: tuple[tuple[tuple[int, ...], ...], ...],
        solid_faces: tuple[tuple[int, ...], ...],
        solid_face_orientations: tuple[tuple[int, ...], ...],
        num_vertices: int,
    ) -> None:
        face_count = len(face_edges)
        if len(face_wires) != face_count:
            raise ValueError(
                f"BRepTopology.face_wires must contain one entry per face; got wires={len(face_wires)}, faces={face_count}"
            )
        if len(solid_faces) != len(solid_face_orientations):
            raise ValueError(
                f"BRepTopology.solid_face_orientations must contain one entry per solid; got orientations={len(solid_face_orientations)}, solids={len(solid_faces)}"
            )
        if int(num_vertices) < 0:
            raise ValueError(
                f"BRepTopology.num_vertices must be nonnegative; got {num_vertices!r}"
            )
        face_solids: list[list[int]] = [[] for _ in range(face_count)]
        for solid_index, (indices, orientations) in enumerate(
            zip(solid_faces, solid_face_orientations, strict=True)
        ):
            if len(indices) != len(orientations):
                raise ValueError(
                    f"BRepTopology.solid_face_orientations must align with solid_faces; got solid={solid_index}, orientations={len(orientations)}, faces={len(indices)}"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"BRepTopology.solid_faces must not repeat a face; got solid={solid_index}, faces={indices!r}"
                )
            if any(index < 0 or index >= face_count for index in indices):
                raise ValueError(
                    f"BRepTopology.solid_faces must reference existing faces; got solid={solid_index}, faces={indices!r}, face_count={face_count}"
                )
            if any(orientation not in (-1, 1) for orientation in orientations):
                raise ValueError(
                    f"BRepTopology.solid_face_orientations must contain only -1 or 1; got solid={solid_index}, orientations={orientations!r}"
                )
            for face_index in indices:
                face_solids[face_index].append(solid_index)
        self.face_edges = face_edges
        self.edge_faces = edge_faces
        self.face_wires = face_wires
        self.solid_faces = solid_faces
        self.solid_face_orientations = solid_face_orientations
        self.face_solids = tuple(tuple(indices) for indices in face_solids)
        self.num_vertices = int(num_vertices)

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: BRepTopology, /) -> Self:
        """Reconstruct stored inputs through the validated construction phases.

        The structure-recipe boundary admits the registered type and field set.
        This owner requires its node type before revalidating constructor inputs;
        derived incidence, bounds, and identities are recomputed by their phases.
        """
        if type(value) is not cls:
            raise TypeError(
                f"BRepTopology.stored_fields must have the declared node type; got {type(value).__name__}"
            )
        return cls(
            face_edges=value.face_edges,
            edge_faces=value.edge_faces,
            face_wires=value.face_wires,
            solid_faces=value.solid_faces,
            solid_face_orientations=value.solid_face_orientations,
            num_vertices=value.num_vertices,
        )

    @property
    def num_faces(self) -> int:
        return len(self.face_edges)

    @property
    def num_edges(self) -> int:
        return len(self.edge_faces)

    @property
    def num_solids(self) -> int:
        return len(self.solid_faces)


@final
class BRepPlacementEvidence(StrictModule):
    """Certified source-frame and polar-factor displacement in length coordinates."""

    rotation_deviation_bound: float = eqx.field(static=True)
    model_extent: float = eqx.field(static=True)
    point_deviation_bound: float = eqx.field(static=True)
    translation_deviation_bound: float = eqx.field(static=True, default=0.0)
    entrywise_deviation_bounds: tuple[tuple[float, ...], ...] | None = eqx.field(
        static=True,
        default=None,
    )

    def __init__(
        self,
        rotation_deviation_bound: float,
        model_extent: float,
        point_deviation_bound: float,
        translation_deviation_bound: float = 0.0,
        *,
        entrywise_deviation_bounds: tuple[tuple[float, ...], ...] | None = None,
    ) -> None:
        values = (
            rotation_deviation_bound,
            model_extent,
            point_deviation_bound,
            translation_deviation_bound,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "BRepPlacementEvidence.bounds must be finite and nonnegative; got invalid bounds."
            )
        if Fraction(point_deviation_bound) < (
            Fraction(rotation_deviation_bound) * Fraction(model_extent)
            + Fraction(translation_deviation_bound)
        ):
            raise ValueError(
                "BRepPlacementEvidence.point_deviation_bound must enclose rotation and translation displacement; got an insufficient bound."
            )
        if entrywise_deviation_bounds is not None:
            _validate_placement_entry_bounds(
                entrywise_deviation_bounds,
                rotation_deviation_bound,
                translation_deviation_bound,
            )
        self.rotation_deviation_bound = rotation_deviation_bound
        self.model_extent = model_extent
        self.point_deviation_bound = point_deviation_bound
        self.translation_deviation_bound = translation_deviation_bound
        self.entrywise_deviation_bounds = entrywise_deviation_bounds

    def payload(self) -> dict[str, float | tuple[tuple[float, ...], ...]]:
        """Encode placement bounds in deterministic evidence identity order."""
        return {
            "rotation_deviation_bound": self.rotation_deviation_bound,
            "model_extent": self.model_extent,
            "point_deviation_bound": self.point_deviation_bound,
            **(
                {"translation_deviation_bound": self.translation_deviation_bound}
                if self.translation_deviation_bound > 0.0
                else {}
            ),
            **(
                {"entrywise_deviation_bounds": self.entrywise_deviation_bounds}
                if self.entrywise_deviation_bounds is not None
                else {}
            ),
        }


def _validate_placement_entry_bounds(
    bounds: tuple[tuple[float, ...], ...],
    rotation_bound: float,
    translation_bound: float,
    /,
) -> None:
    """Validate entrywise affine errors and their scalar norm implications."""
    if len(bounds) != 3 or any(
        len(row) != 4 or any(not np.isfinite(value) or value < 0.0 for value in row)
        for row in bounds
    ):
        raise ValueError(
            "BRepPlacementEvidence.entrywise_deviation_bounds must be finite nonnegative (3,4) rows; got invalid bounds."
        )
    rotation_squared = sum(
        (Fraction(value) ** 2 for row in bounds for value in row[:3]),
        Fraction(0),
    )
    translation_squared = sum(
        (Fraction(row[3]) ** 2 for row in bounds),
        Fraction(0),
    )
    if Fraction(rotation_bound) ** 2 < rotation_squared or (
        Fraction(translation_bound) ** 2 < translation_squared
    ):
        raise ValueError(
            "BRepPlacementEvidence.norm_bounds must enclose entrywise errors; got insufficient scalar bounds."
        )


def _validated_occurrence_pose(
    rotation: ConvertibleToArray,
    translation: ConvertibleToArray,
    placement_evidence: BRepPlacementEvidence | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite poses and replay their represented Gram-defect evidence."""
    rotation_host = np.asarray(rotation, dtype=np.float64)
    translation_host = np.asarray(translation, dtype=np.float64)
    if rotation_host.shape != (3, 3) or translation_host.shape != (3,):
        raise ValueError(
            f"BRepOccurrence.rotation must have shape (3, 3) with translation shape (3,); got rotation={rotation_host.shape}, translation={translation_host.shape}"
        )
    if not np.all(np.isfinite(rotation_host)) or not np.all(
        np.isfinite(translation_host)
    ):
        raise ValueError(
            f"BRepOccurrence.rotation must be finite with finite translation; got rotation={rotation_host!r}, translation={translation_host!r}"
        )
    if (
        np.max(np.abs(rotation_host.T @ rotation_host - np.eye(3))) > 1.0e-12
        or np.linalg.det(rotation_host) == 0.0
    ):
        raise ValueError(
            f"BRepOccurrence.rotation must be orthogonal and nonsingular; got {rotation_host!r}"
        )
    if placement_evidence is not None and (
        exact_determinant(rotation_host) <= 0
        or gram_deviation_bound(rotation_host)
        > placement_evidence.rotation_deviation_bound
    ):
        raise ValueError(
            "BRepOccurrence.placement_evidence must enclose a certified proper frame; got invalid rotation evidence."
        )
    return rotation_host, translation_host


@final
class BRepOccurrence(StrictModule):
    """One placed assembly instance of a model solid.

    ``path`` is the occurrence path from the assembly root. ``rotation`` (a
    orthogonal matrix, with either coorientation) and ``translation`` map definition coordinates of
    the solid into the model's physical coordinate frame.
    """

    path: tuple[str, ...] = eqx.field(static=True)
    solid: int = eqx.field(static=True)
    rotation: tuple[tuple[float, float, float], ...] = eqx.field(static=True)
    translation: tuple[float, float, float] = eqx.field(static=True)
    placement_evidence: BRepPlacementEvidence | None = None

    @checked
    def __init__(
        self,
        path: tuple[str, ...],
        solid: int,
        rotation: ConvertibleToArray = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation: ConvertibleToArray = (0.0, 0.0, 0.0),
        *,
        placement_evidence: BRepPlacementEvidence | None = None,
    ) -> None:
        if (
            not isinstance(path, tuple)
            or not path
            or any(not isinstance(name, str) or not name for name in path)
        ):
            raise ValueError(
                f"BRepOccurrence.path must be a nonempty tuple of nonempty names; got {path!r}"
            )
        if isinstance(solid, bool) or not isinstance(solid, int):
            raise TypeError(
                f"BRepOccurrence.solid must be an integer index; got {type(solid).__name__}"
            )
        if solid < 0:
            raise ValueError(f"BRepOccurrence.solid must be nonnegative; got {solid}")
        rotation_host, translation_host = _validated_occurrence_pose(
            rotation,
            translation,
            placement_evidence,
        )
        self.path = path
        self.solid = solid
        self.rotation = tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in rotation_host
        )
        self.translation = (
            float(translation_host[0]),
            float(translation_host[1]),
            float(translation_host[2]),
        )
        self.placement_evidence = placement_evidence

    def place(self, points: ArrayLike, /) -> np.ndarray:
        """Map definition coordinates of the solid into model coordinates."""
        values = np.asarray(points, dtype=np.float64)
        return values @ np.asarray(self.rotation, dtype=np.float64).T + np.asarray(
            self.translation, dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class BRepAssemblyContainer:
    """Declared occurrence members and child relationships of a container instance."""

    path: tuple[str, ...]
    member_paths: tuple[tuple[str, ...], ...]
    child_paths: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, tuple)
            or not self.path
            or any(not isinstance(name, str) or not name for name in self.path)
        ):
            raise ValueError(
                f"BRepAssemblyContainer.path must be a nonempty tuple of nonempty names; got {self.path!r}"
            )
        if (
            not isinstance(self.member_paths, tuple)
            or not isinstance(self.child_paths, tuple)
            or not (self.member_paths or self.child_paths)
        ):
            raise ValueError(
                f"BRepAssemblyContainer.member_paths must be a tuple with declared members or child_paths; got members={self.member_paths!r}, children={self.child_paths!r}"
            )
        for paths in (self.member_paths, self.child_paths):
            if any(
                not isinstance(path, tuple)
                or not path
                or any(not isinstance(name, str) or not name for name in path)
                for path in paths
            ):
                raise ValueError(
                    f"BRepAssemblyContainer.member_paths must contain nonempty tuple paths, including child_paths; got {paths!r}"
                )
            if len(set(paths)) != len(paths):
                raise ValueError(
                    f"BRepAssemblyContainer.member_paths must not repeat an instance, including child_paths; got {paths!r}"
                )


@dataclass(frozen=True, slots=True, order=True)
class BRepQualifiedIncidence:
    """Occurrence-qualified incidence; coedge uses retain their source index."""

    container: BRepEntityId
    member: BRepEntityId
    orientation: int
    use_index: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.container, BRepEntityId) or not isinstance(
            self.member, BRepEntityId
        ):
            raise TypeError(
                f"BRepQualifiedIncidence.container must be a BRepEntityId with a BRepEntityId member; got container={type(self.container).__name__}, member={type(self.member).__name__}"
            )
        if self.container.source_revision != self.member.source_revision:
            raise ValueError(
                f"BRepQualifiedIncidence.container must share its source_revision with member; got container={self.container.source_revision!r}, member={self.member.source_revision!r}"
            )
        if self.orientation not in (-1, 1) or self.use_index < -1:
            raise ValueError(
                f"BRepQualifiedIncidence.orientation must be -1 or 1 with use_index at least -1; got orientation={self.orientation}, use_index={self.use_index}"
            )


def vertex_root_payload(root: BRepVertexRoot, /) -> dict[str, object]:
    return {
        "root_id": root.root_id,
        "primary": root.primary.root_id,
        "aliases": tuple(sorted(alias.root_id for alias in root.aliases)),
        "spatial_root": None if root.spatial_root is None else root.spatial_root.root_id,
        "joint_root": None if root.joint_root is None else root.joint_root.root_id,
        "source_edge_lifts": tuple(
            sorted(lift.lift_id for lift in root.source_edge_lifts)
        ),
    }


def endpoint_root_payload(root: RootEndpoint, /) -> dict[str, object]:
    if isinstance(root, NativePeriodEndpoint):
        return {
            "kind": "native-period-endpoint",
            "source_id": root.root_id,
            "carrier": carrier_owner.carrier_payload(root.carrier),
            "patch": None
            if root.patch is None
            else carrier_owner.carrier_payload(root.patch),
            "axis": root.axis,
            "rational": (root.rational.numerator, root.rational.denominator),
            "turns": (root.turns.numerator, root.turns.denominator),
        }
    if isinstance(root, BranchRootEndpoint):
        return {
            "kind": "branch-root-endpoint",
            "root_id": root.root.root_id,
            "curve_id": root.curve.branch_id,
            "chart": root.chart,
            "source_pcurve": None
            if root.source_pcurve is None
            else carrier_owner.carrier_payload(root.source_pcurve),
            "source_side": root.source_side,
            "source_first": root.source_first,
            "source_last": root.source_last,
            "periodic_shifts": root.periodic_shifts,
            "affine_transforms": root.affine_transforms,
        }
    return {
        "kind": "trim-root-endpoint",
        "root_id": root.root.root_id,
        "operand": root.operand,
        "affine_parameter_offset": root.affine_parameter_offset,
        "affine_parameter_scale": root.affine_parameter_scale,
        "affine_transforms": root.affine_transforms,
        "source_pcurve": None
        if root.source_pcurve is None
        else carrier_owner.carrier_payload(root.source_pcurve),
        "source_surface": None
        if root.source_surface is None
        else encode_geometry(root.source_surface),
    }


type BRepValidationCheck = Literal[
    "coordinate-contract",
    "topology",
    "parameter-bounds",
    "orientation",
    "face-metadata",
    "mesh-vertices",
    "mesh-faces",
    "triangle-face-ids",
    "triangle-parameters",
    "tessellation-bounds",
    "vertex-lineage",
    "report-counts",
    "occurrence-lineage",
    "root-binding",
    "trim-containment",
    "curve-surface",
    "trim-closure",
    "edge-incidence",
    "shell-incidence",
    "container-incidence",
    "chart-restriction",
]


class BRepValidationFinding(StrictModule):
    """One failed check with its entity, measurement, and declared limit.

    Structural measurements count failed predicates against a zero limit.
    Curve/surface measurements and limits are physical lengths; trim closure
    measurements and limits are distances in the face parameter plane.
    """

    code: BRepValidationCheck = eqx.field(static=True)
    entity: BRepEntityId | None = eqx.field(static=True)
    measured: float = eqx.field(static=True)
    limit: float = eqx.field(static=True)
    message: str = eqx.field(static=True)
    coedge: int | None = eqx.field(static=True)
    unresolved: tuple[tuple[float, float], ...] = eqx.field(static=True)
    intervals_processed: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        code: BRepValidationCheck,
        entity: BRepEntityId | None,
        measured: float,
        limit: float,
        message: str,
        *,
        coedge: int | None = None,
        unresolved: tuple[tuple[float, float], ...] = (),
        intervals_processed: int = 0,
    ) -> None:
        self.code = parse(code, BRepValidationCheck, "BRepValidationFinding.code")
        self.entity = entity
        self.measured = measured
        self.limit = limit
        self.message = message
        self.coedge = coedge
        self.unresolved = unresolved
        self.intervals_processed = intervals_processed


class _CoedgeFailurePhase(StrictModule):
    """Typed primary coedge diagnostic selected from a construction report."""

    face: int = eqx.field(static=True)
    coedge: int = eqx.field(static=True)
    unresolved: tuple[tuple[float, float], ...] = eqx.field(static=True)
    intervals_processed: int = eqx.field(static=True)


class BRepValidationReport(StrictModule):
    """Collect independent failures in validation order; findings never soft-pass."""

    findings: tuple[BRepValidationFinding, ...] = eqx.field(static=True)

    @property
    def valid(self) -> bool:
        return not self.findings

    def require_valid(self) -> None:
        if self.findings:
            raise BRepValidationError(self)

    def require_coedge_failure(self) -> _CoedgeFailurePhase:
        """Select the first coedge failure, preserving primary refusal ordering.

        Other construction failures retain their report rather than acquiring a
        curve/surface classification. Every secondary finding remains available.
        """
        if not self.findings:
            raise ValueError(
                "BRepValidationReport.findings must contain a coedge failure; got no findings"
            )
        finding = self.findings[0]
        if finding.code != "curve-surface":
            raise BRepValidationError(self)
        entity, coedge = finding.entity, finding.coedge
        if entity is None or entity.kind != "face" or coedge is None or coedge < 0:
            raise ValueError(
                f"BRepValidationFinding.coedge must identify a face use; got entity={entity!r}, coedge={coedge!r}"
            )
        return _CoedgeFailurePhase(
            face=entity.index,
            coedge=coedge,
            unresolved=finding.unresolved,
            intervals_processed=finding.intervals_processed,
        )


class BRepValidationError(ValueError):
    """A refused construction with all independently evaluable failed checks."""

    @checked
    def __init__(self, report: BRepValidationReport, /) -> None:
        self.report = report
        super().__init__(report.findings[0].message)


class _CoedgeValidationPhase(StrictModule):
    """Certified correspondence and ordered endpoint coordinates of one coedge."""

    deviation: float = eqx.field(static=True)
    uv: np.ndarray


class _ModelIdentityPhase(StrictModule):
    """Scientific and derived identities after all construction phases validate."""

    model_id: str = eqx.field(static=True)
    tessellation_id: str = eqx.field(static=True)


def _coedge_vertices(
    edge_vertices: tuple[tuple[int, int], ...], edge: int, sense: int, /
) -> tuple[int, int]:
    start, end = edge_vertices[edge]
    return (start, end) if sense > 0 else (end, start)


class _GeometrySourcePhase(StrictModule):
    """Validated host inputs from one geometry construction phase."""

    points: np.ndarray
    ranges: np.ndarray
    vertex_definitions: tuple[BRepVertexRoot | None, ...]
    edge_definitions: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    coedge_definitions: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    vertex_errors: np.ndarray


class _GeometryLoopPhase(StrictModule):
    """Validated host inputs from one geometry construction phase."""

    coedge_count: int = eqx.field(static=True)


class _GeometryAssemblyPhase(StrictModule):
    """Validated host inputs from one geometry construction phase."""

    occurrences_: tuple[BRepOccurrence, ...] = eqx.field(static=True)
    containers: tuple[BRepAssemblyContainer, ...] = eqx.field(static=True)


class _IncidenceRowsPhase(StrictModule):
    """Incidence rows and newly visited edges from one face traversal."""

    rows: tuple[BRepQualifiedIncidence, ...] = eqx.field(static=True)
    edges: tuple[BRepEntityId, ...] = eqx.field(static=True)


class BRepGeometry(StrictModule):
    """Curve and trim geometry with oriented B-Rep topology.

    Edges carry a 3D ``curve`` over ``edge_ranges`` (``edge_curves[e] == -1``
    marks a degenerate edge collapsed to its single vertex, such as a pole).
    Every coedge is one oriented use of an edge by a face loop: ``sense`` is
    +1 when the loop traverses the edge in increasing parameter and its 2D
    ``pcurve`` maps the same edge parameter into the face parameter plane.
    Face loops list coedge indices; the first loop is the outer boundary,
    counterclockwise in the face parameter plane, and holes are clockwise.
    Shells collect oriented faces; solids list shells (outer first).
    """

    vertex_points: Array
    curves: tuple[BRepCurve, ...]
    edge_ranges: Array
    pcurves: tuple[BRepPCurve, ...]
    vertex_roots: tuple[BRepVertexRoot | None, ...]
    vertex_evaluation_bounds: Array = eqx.field(init=False)
    edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    edge_curves: tuple[int, ...] = eqx.field(static=True)
    edge_vertices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = eqx.field(
        static=True
    )
    edge_face_gap_bounds: tuple[float, ...] | None = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    coedge_senses: tuple[int, ...] = eqx.field(static=True)
    face_loops: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    shell_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shell_orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_shells: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occurrences: tuple[BRepOccurrence, ...] = eqx.field(static=True)
    assembly_containers: tuple[BRepAssemblyContainer, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True, init=False)

    @checked
    def __init__(
        self,
        *,
        vertex_points: ArrayLike,
        curves: tuple[BRepCurve, ...],
        edge_curves: tuple[int, ...],
        edge_ranges: ArrayLike,
        edge_vertices: tuple[tuple[int, int], ...],
        pcurves: tuple[BRepPCurve, ...],
        coedge_edges: tuple[int, ...],
        coedge_senses: tuple[int, ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        occurrences: tuple[BRepOccurrence, ...] | None = None,
        assembly_containers: tuple[BRepAssemblyContainer, ...] | None = None,
        vertex_roots: tuple[BRepVertexRoot | None, ...] | None = None,
        edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = None,
        edge_face_gap_bounds: tuple[float, ...] | None = None,
        edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                vertex_points,
                curves,
                edge_curves,
                edge_ranges,
                edge_vertices,
                pcurves,
                coedge_edges,
                coedge_senses,
                face_loops,
                shell_faces,
                shell_orientations,
                solid_shells,
                occurrences,
                assembly_containers,
                vertex_roots,
                edge_endpoint_roots,
                coedge_endpoint_roots,
            ),
            budget=budget,
        ) as budget:
            findings: list[BRepValidationFinding] = []
            if edge_vertex_gap_bounds is not None and (
                len(edge_vertex_gap_bounds) != len(edge_curves)
                or any(
                    len(row) != 2 or any(not np.isfinite(x) or x < 0.0 for x in row)
                    for row in edge_vertex_gap_bounds
                )
            ):
                raise ValueError(
                    "BRepGeometry.edge_vertex_gap_bounds must align with finite nonnegative endpoint bounds; got invalid rows"
                )
            if edge_face_gap_bounds is not None and (
                len(edge_face_gap_bounds) != len(coedge_edges)
                or any(
                    not np.isfinite(bound) or bound < 0.0
                    for bound in edge_face_gap_bounds
                )
            ):
                raise ValueError(
                    "BRepGeometry.edge_face_gap_bounds must align with finite nonnegative coedge bounds; got invalid rows"
                )
            sources = self._prepare_geometry_sources(
                budget=budget,
                coedge_edges=coedge_edges,
                coedge_endpoint_roots=coedge_endpoint_roots,
                curves=curves,
                edge_curves=edge_curves,
                edge_endpoint_roots=edge_endpoint_roots,
                edge_ranges=edge_ranges,
                edge_vertices=edge_vertices,
                pcurves=pcurves,
                preparations=preparations,
                vertex_points=vertex_points,
                vertex_roots=vertex_roots,
                findings=findings,
                edge_vertex_gap_bounds=edge_vertex_gap_bounds,
            )
            points = sources.points
            ranges = sources.ranges
            vertex_definitions = sources.vertex_definitions
            edge_definitions = sources.edge_definitions
            coedge_definitions = sources.coedge_definitions
            vertex_errors = sources.vertex_errors
            edge_count = len(edge_curves)
            self._validate_geometry_loops(
                coedge_edges=coedge_edges,
                coedge_senses=coedge_senses,
                edge_count=edge_count,
                edge_vertices=edge_vertices,
                face_loops=face_loops,
                pcurves=pcurves,
                shell_faces=shell_faces,
                shell_orientations=shell_orientations,
                solid_shells=solid_shells,
                findings=findings,
            )
            assembly = self._prepare_geometry_assembly(
                assembly_containers=assembly_containers,
                coedge_edges=coedge_edges,
                edge_vertices=edge_vertices,
                face_loops=face_loops,
                occurrences=occurrences,
                shell_faces=shell_faces,
                solid_shells=solid_shells,
                findings=findings,
            )
            occurrences_ = assembly.occurrences_
            containers = assembly.containers
            BRepValidationReport(findings=tuple(findings)).require_valid()
            self.vertex_points = jnp.asarray(points)
            self.curves = tuple(curves)
            self.edge_ranges = jnp.asarray(ranges)
            self.pcurves = tuple(pcurves)
            self.vertex_roots = vertex_definitions
            self.vertex_evaluation_bounds = jnp.asarray(vertex_errors, dtype=jnp.float64)
            self.edge_endpoint_roots = edge_definitions
            self.coedge_endpoint_roots = coedge_definitions
            self.edge_curves = tuple(int(index) for index in edge_curves)
            self.edge_vertices = tuple((int(a), int(b)) for a, b in edge_vertices)
            self.edge_vertex_gap_bounds = edge_vertex_gap_bounds
            self.edge_face_gap_bounds = edge_face_gap_bounds
            self.coedge_edges = tuple(int(edge) for edge in coedge_edges)
            self.coedge_senses = tuple(int(sense) for sense in coedge_senses)
            self.face_loops = tuple(
                tuple(tuple(int(c) for c in loop) for loop in loops)
                for loops in face_loops
            )
            self.shell_faces = tuple(
                tuple(int(f) for f in faces) for faces in shell_faces
            )
            self.shell_orientations = tuple(
                tuple(int(sign) for sign in signs) for signs in shell_orientations
            )
            self.solid_shells = tuple(
                tuple(int(shell) for shell in shells) for shells in solid_shells
            )
            self.occurrences = occurrences_
            self.assembly_containers = tuple(
                sorted(containers, key=lambda container: container.path)
            )
            self.geometry_id = self._prepare_geometry_identity(points, ranges)

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: BRepGeometry, /) -> Self:
        """Reconstruct stored inputs through the validated construction phases.

        The structure-recipe boundary admits the registered type and field set.
        This owner requires its node type before revalidating constructor inputs;
        derived incidence, bounds, and identities are recomputed by their phases.
        """
        if type(value) is not cls:
            raise TypeError(
                f"BRepGeometry.stored_fields must have the declared node type; got {type(value).__name__}"
            )
        return cls(
            vertex_points=value.vertex_points,
            curves=value.curves,
            edge_curves=value.edge_curves,
            edge_ranges=value.edge_ranges,
            edge_vertices=value.edge_vertices,
            pcurves=value.pcurves,
            coedge_edges=value.coedge_edges,
            coedge_senses=value.coedge_senses,
            face_loops=value.face_loops,
            shell_faces=value.shell_faces,
            shell_orientations=value.shell_orientations,
            solid_shells=value.solid_shells,
            occurrences=value.occurrences,
            assembly_containers=value.assembly_containers,
            vertex_roots=value.vertex_roots,
            edge_endpoint_roots=value.edge_endpoint_roots,
            coedge_endpoint_roots=value.coedge_endpoint_roots,
            edge_vertex_gap_bounds=value.edge_vertex_gap_bounds,
            edge_face_gap_bounds=value.edge_face_gap_bounds,
        )

    def _prepare_geometry_sources(
        self,
        budget: CoordinateEnclosureBudget,
        coedge_edges: tuple[int, ...],
        coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None,
        curves: tuple[BRepCurve, ...],
        edge_curves: tuple[int, ...],
        edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None,
        edge_ranges: ArrayLike,
        edge_vertices: tuple[tuple[int, int], ...],
        pcurves: tuple[BRepPCurve, ...],
        preparations: TrimRequestPreparations | None,
        vertex_points: ArrayLike,
        vertex_roots: tuple[BRepVertexRoot | None, ...] | None,
        findings: list[BRepValidationFinding],
        edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None,
    ) -> _GeometrySourcePhase:
        points = np.asarray(vertex_points, dtype=np.float64)
        ranges = np.asarray(edge_ranges, dtype=np.float64)
        edge_count = len(edge_curves)
        if points.ndim != 2 or points.shape[1] != 3 or (not np.all(np.isfinite(points))):
            findings.append(
                BRepValidationFinding(
                    "edge-incidence",
                    None,
                    1.0,
                    0.0,
                    "vertex_points must be finite with shape (V, 3).",
                )
            )
        if len(edge_vertices) != edge_count or ranges.shape != (edge_count, 2):
            findings.append(
                BRepValidationFinding(
                    "edge-incidence",
                    None,
                    1.0,
                    0.0,
                    "Edge curves, ranges, and vertices must align.",
                )
            )
        if any((curve.ambient_dimension != 3 for curve in curves)):
            findings.append(
                BRepValidationFinding(
                    "edge-incidence",
                    None,
                    1.0,
                    0.0,
                    "Edge curves must be three-dimensional.",
                )
            )
        if any((curve.ambient_dimension != 2 for curve in pcurves)):
            findings.append(
                BRepValidationFinding(
                    "edge-incidence", None, 1.0, 0.0, "p-curves must be two-dimensional."
                )
            )
        vertex_definitions = (
            (None,) * (points.shape[0] if points.ndim else 0)
            if vertex_roots is None
            else tuple(vertex_roots)
        )
        edge_definitions = (
            ((None, None),) * edge_count
            if edge_endpoint_roots is None
            else tuple(edge_endpoint_roots)
        )
        coedge_definitions = (
            ((None, None),) * len(coedge_edges)
            if coedge_endpoint_roots is None
            else tuple(coedge_endpoint_roots)
        )
        if not findings:
            vertex_errors = self._validate_root_definitions(
                points,
                ranges,
                edge_vertices,
                coedge_edges,
                vertex_definitions,
                edge_definitions,
                coedge_definitions,
                budget=budget,
                preparations=preparations,
                findings=findings,
            )
            if not findings:
                self._validate_edges(
                    points,
                    curves,
                    edge_curves,
                    ranges,
                    edge_vertices,
                    vertex_definitions,
                    findings=findings,
                    budget=budget,
                    preparations=preparations,
                    edge_vertex_gap_bounds=edge_vertex_gap_bounds,
                    edge_endpoint_roots=edge_definitions,
                )
        else:
            vertex_errors = np.zeros(
                (points.shape[0] if points.ndim else 0,), dtype=np.float64
            )
        return _GeometrySourcePhase(
            points=points,
            ranges=ranges,
            vertex_definitions=vertex_definitions,
            edge_definitions=edge_definitions,
            coedge_definitions=coedge_definitions,
            vertex_errors=vertex_errors,
        )

    def _validate_geometry_loops(
        self,
        coedge_edges: tuple[int, ...],
        coedge_senses: tuple[int, ...],
        edge_count: int,
        edge_vertices: tuple[tuple[int, int], ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        pcurves: tuple[BRepPCurve, ...],
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        findings: list[BRepValidationFinding],
    ) -> _GeometryLoopPhase:
        coedge_count = len(coedge_edges)
        if len(coedge_senses) != coedge_count or len(pcurves) != coedge_count:
            findings.append(
                BRepValidationFinding(
                    "topology",
                    None,
                    1.0,
                    0.0,
                    "Coedge edges, senses, and p-curves must align.",
                )
            )
        if any((edge < 0 or edge >= edge_count for edge in coedge_edges)):
            findings.append(
                BRepValidationFinding(
                    "topology", None, 1.0, 0.0, "A coedge references an absent edge."
                )
            )
        if any((sense not in (-1, 1) for sense in coedge_senses)):
            findings.append(
                BRepValidationFinding(
                    "topology", None, 1.0, 0.0, "Coedge senses must be -1 or 1."
                )
            )
        used = [coedge for loops in face_loops for loop in loops for coedge in loop]
        if sorted(used) != list(range(coedge_count)):
            findings.append(
                BRepValidationFinding(
                    "topology",
                    None,
                    1.0,
                    0.0,
                    "Every coedge must belong to exactly one face loop.",
                )
            )
        for loops in face_loops:
            if not loops or any((not loop for loop in loops)):
                findings.append(
                    BRepValidationFinding(
                        "topology",
                        None,
                        1.0,
                        0.0,
                        "Every face needs a nonempty outer loop.",
                    )
                )
            for loop in loops:
                if (
                    not loop
                    or len(coedge_senses) != coedge_count
                    or any((c < 0 or c >= coedge_count for c in loop))
                    or any(
                        (edge < 0 or edge >= len(edge_vertices) for edge in coedge_edges)
                    )
                ):
                    continue
                ends = [
                    _coedge_vertices(edge_vertices, coedge_edges[c], coedge_senses[c])
                    for c in loop
                ]
                if any(
                    (
                        ends[index][1] != ends[(index + 1) % len(ends)][0]
                        for index in range(len(ends))
                    )
                ):
                    findings.append(
                        BRepValidationFinding(
                            "topology",
                            None,
                            1.0,
                            0.0,
                            "Face loop coedges must connect head to tail.",
                        )
                    )
        self._validate_shells(
            len(face_loops),
            shell_faces,
            shell_orientations,
            solid_shells,
            findings=findings,
        )
        return _GeometryLoopPhase(coedge_count=coedge_count)

    def _prepare_geometry_assembly(
        self,
        assembly_containers: tuple[BRepAssemblyContainer, ...] | None,
        coedge_edges: tuple[int, ...],
        edge_vertices: tuple[tuple[int, int], ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        occurrences: tuple[BRepOccurrence, ...] | None,
        shell_faces: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        findings: list[BRepValidationFinding],
    ) -> _GeometryAssemblyPhase:
        occurrences_ = (
            tuple(
                (
                    BRepOccurrence((f"solid{index}",), index)
                    for index in range(len(solid_shells))
                )
            )
            if occurrences is None
            else tuple(occurrences)
        )
        if any((occurrence.solid >= len(solid_shells) for occurrence in occurrences_)):
            findings.append(
                BRepValidationFinding(
                    "occurrence-lineage",
                    None,
                    1.0,
                    0.0,
                    "Occurrences must reference model solids.",
                )
            )
        if len({occurrence.path for occurrence in occurrences_}) != len(occurrences_):
            findings.append(
                BRepValidationFinding(
                    "occurrence-lineage",
                    None,
                    1.0,
                    0.0,
                    "Occurrence paths must be unique.",
                )
            )
        occurrences_ = tuple(sorted(occurrences_, key=lambda occurrence: occurrence.path))
        containers = (
            (
                BRepAssemblyContainer(
                    ("model",), tuple((occurrence.path for occurrence in occurrences_))
                ),
            )
            if assembly_containers is None and occurrences is None and occurrences_
            else ()
            if assembly_containers is None
            else tuple(assembly_containers)
        )
        self._validate_containers(
            containers,
            occurrences_,
            shell_faces,
            solid_shells,
            face_loops,
            coedge_edges,
            edge_vertices,
            findings=findings,
        )
        return _GeometryAssemblyPhase(occurrences_=occurrences_, containers=containers)

    def _prepare_geometry_identity(self, points: np.ndarray, ranges: np.ndarray) -> str:
        vertex_definitions = self.vertex_roots
        edge_definitions = self.edge_endpoint_roots
        coedge_definitions = self.coedge_endpoint_roots
        occurrences_ = self.occurrences
        return canonical_fingerprint(
            {
                "kind": "brep-exact-geometry",
                "vertex_points": [
                    points[index] if root is None else None
                    for index, root in enumerate(vertex_definitions)
                ],
                "curves": [carrier_owner.carrier_payload(curve) for curve in self.curves],
                "edge_curves": list(self.edge_curves),
                "edge_ranges": [
                    [
                        float(ranges[edge, end]) if root is None else None
                        for end, root in enumerate(endpoints)
                    ]
                    for edge, endpoints in enumerate(edge_definitions)
                ],
                "edge_vertices": [list(pair) for pair in self.edge_vertices],
                **(
                    {"edge_vertex_gap_bounds": self.edge_vertex_gap_bounds}
                    if self.edge_vertex_gap_bounds is not None
                    else {}
                ),
                **(
                    {"edge_face_gap_bounds": self.edge_face_gap_bounds}
                    if self.edge_face_gap_bounds is not None
                    else {}
                ),
                "pcurves": [
                    carrier_owner.carrier_payload(curve) for curve in self.pcurves
                ],
                "coedge_edges": list(self.coedge_edges),
                "coedge_senses": list(self.coedge_senses),
                "face_loops": [
                    [list(loop) for loop in loops] for loops in self.face_loops
                ],
                "shell_faces": [list(faces) for faces in self.shell_faces],
                "shell_orientations": [list(s) for s in self.shell_orientations],
                "solid_shells": [list(shells) for shells in self.solid_shells],
                "occurrences": [
                    {
                        "path": list(occurrence.path),
                        "solid": occurrence.solid,
                        "rotation": [list(row) for row in occurrence.rotation],
                        "translation": list(occurrence.translation),
                        **(
                            {
                                "placement_evidence": occurrence.placement_evidence.payload()
                            }
                            if occurrence.placement_evidence is not None
                            else {}
                        ),
                    }
                    for occurrence in occurrences_
                ],
                "assembly_containers": [
                    {
                        "path": container.path,
                        "member_paths": tuple(sorted(container.member_paths)),
                        "child_paths": tuple(sorted(container.child_paths)),
                    }
                    for container in self.assembly_containers
                ],
                "vertex_roots": [
                    None if root is None else vertex_root_payload(root)
                    for root in vertex_definitions
                ],
                "edge_endpoint_roots": [
                    [
                        None if root is None else endpoint_root_payload(root)
                        for root in endpoints
                    ]
                    for endpoints in edge_definitions
                ],
                "coedge_endpoint_roots": [
                    [
                        None if root is None else endpoint_root_payload(root)
                        for root in endpoints
                    ]
                    for endpoints in coedge_definitions
                ],
            }
        )

    @staticmethod
    def _validate_root_definitions(
        points: np.ndarray,
        ranges: np.ndarray,
        edge_vertices: tuple[tuple[int, int], ...],
        coedge_edges: tuple[int, ...],
        vertices: tuple[BRepVertexRoot | None, ...],
        edges: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
        coedges: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
        findings: list[BRepValidationFinding],
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> np.ndarray:
        errors = np.zeros((points.shape[0],), dtype=np.float64)
        if (
            len(vertices) != points.shape[0]
            or len(edges) != len(edge_vertices)
            or len(coedges) != len(coedge_edges)
        ):
            findings.append(
                BRepValidationFinding(
                    "root-binding",
                    None,
                    1.0,
                    0.0,
                    "Root definitions must align with vertices, edges and coedges.",
                )
            )
        if any(
            (
                start < 0
                or end < 0
                or start >= points.shape[0]
                or (end >= points.shape[0])
                for start, end in edge_vertices
            )
        ):
            findings.append(
                BRepValidationFinding(
                    "root-binding", None, 1.0, 0.0, "An edge references an absent vertex."
                )
            )
        if any((edge < 0 or edge >= len(edge_vertices) for edge in coedge_edges)):
            findings.append(
                BRepValidationFinding(
                    "root-binding", None, 1.0, 0.0, "A coedge references an absent edge."
                )
            )
        if findings:
            return errors
        for vertex, root in enumerate(vertices):
            if root is not None:
                box = root.point_enclosure(preparations=preparations, budget=budget)
                errors[vertex] = norm_upper(
                    np.max(np.nextafter(np.abs(box - points[vertex]), np.inf), axis=0)
                )
        for endpoints, edge in [
            (endpoints, edge) for edge, endpoints in enumerate(edges)
        ] + [
            (endpoints, edge)
            for edge, endpoints in zip(coedge_edges, coedges, strict=True)
        ]:
            if not isinstance(endpoints, tuple) or len(endpoints) != 2:
                raise TypeError(
                    f"BRepGeometry.endpoint_roots must be pairs; got {endpoints!r}"
                )
            for endpoint, root in enumerate(endpoints):
                vertex = edge_vertices[edge][endpoint]
                if root is None:
                    vertex_definition = vertices[vertex]
                    if vertex_definition is not None and (
                        not (
                            isinstance(vertex_definition.primary, BRepPlacedVertex)
                            and vertex_definition.primary.source_root is None
                        )
                    ):
                        findings.append(
                            BRepValidationFinding(
                                "root-binding",
                                None,
                                1.0,
                                0.0,
                                "A root-valued vertex requires every incident endpoint's source binding.",
                            )
                        )
                        continue
                    continue
                if isinstance(root, NativePeriodEndpoint):
                    lower, upper = root.parameter_enclosure(
                        preparations=preparations, budget=budget
                    )
                    if not lower <= ranges[edge, endpoint] <= upper:
                        findings.append(
                            BRepValidationFinding(
                                "root-binding",
                                None,
                                1.0,
                                0.0,
                                "A numerical edge endpoint leaves its authored native scalar enclosure.",
                            )
                        )
                        continue
                    continue
                if vertices[vertex] is None:
                    raise TypeError(
                        f"BRepGeometry.endpoint_roots must have a vertex source definition; got edge={edge}, endpoint={endpoint}, vertex={vertex}"
                    )
                vertex_root = vertices[vertex]
                if vertex_root is None or not vertex_root.supports_endpoint(
                    root, preparations=preparations, budget=budget
                ):
                    findings.append(
                        BRepValidationFinding(
                            "root-binding",
                            None,
                            1.0,
                            0.0,
                            "BRepGeometry.endpoint_root must belong to its vertex root; got an unsupported endpoint expression.",
                        )
                    )
                    continue
                lower, upper = root.parameter_enclosure(
                    preparations=preparations, budget=budget
                )
                if not lower <= ranges[edge, endpoint] <= upper:
                    findings.append(
                        BRepValidationFinding(
                            "root-binding",
                            None,
                            1.0,
                            0.0,
                            "A numerical edge endpoint leaves its exact root expression enclosure.",
                        )
                    )
                    continue
        return errors

    @staticmethod
    def _validate_containers(
        containers: tuple[BRepAssemblyContainer, ...],
        occurrences: tuple[BRepOccurrence, ...],
        shell_faces: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        coedge_edges: tuple[int, ...],
        edge_vertices: tuple[tuple[int, int], ...],
        findings: list[BRepValidationFinding],
    ) -> None:
        if len({container.path for container in containers}) != len(containers):
            findings.append(
                BRepValidationFinding(
                    "container-incidence",
                    None,
                    1.0,
                    0.0,
                    "Container instance paths must be unique.",
                )
            )
        by_path = {occurrence.path: occurrence for occurrence in occurrences}
        members = [path for container in containers for path in container.member_paths]
        if len(set(members)) != len(members) or any(
            (path not in by_path for path in members)
        ):
            findings.append(
                BRepValidationFinding(
                    "container-incidence",
                    None,
                    1.0,
                    0.0,
                    "Every declared container member must belong to exactly one known instance.",
                )
            )
        by_container = {container.path: container for container in containers}
        children = [path for container in containers for path in container.child_paths]
        if len(set(children)) != len(children) or any(
            (path not in by_container for path in children)
        ):
            findings.append(
                BRepValidationFinding(
                    "container-incidence",
                    None,
                    1.0,
                    0.0,
                    "Every child container must reference one known container instance.",
                )
            )
        for container in containers:
            pending = list(container.child_paths)
            visited: set[tuple[str, ...]] = set()
            while pending:
                path = pending.pop()
                if path not in by_container:
                    continue
                if path == container.path or path in visited:
                    findings.append(
                        BRepValidationFinding(
                            "container-incidence",
                            None,
                            1.0,
                            0.0,
                            "Container membership must form an acyclic instance hierarchy.",
                        )
                    )
                    break
                visited.add(path)
                pending.extend(by_container[path].child_paths)
        if not findings:
            BRepGeometry._validate_container_placements(
                containers,
                by_path,
                shell_faces,
                solid_shells,
                face_loops,
                coedge_edges,
                edge_vertices,
                findings=findings,
            )

    @staticmethod
    def _validate_container_placements(
        containers: tuple[BRepAssemblyContainer, ...],
        by_path: dict[tuple[str, ...], BRepOccurrence],
        shell_faces: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        coedge_edges: tuple[int, ...],
        edge_vertices: tuple[tuple[int, int], ...],
        findings: list[BRepValidationFinding],
    ) -> None:
        for container in containers:
            placed = [by_path[path] for path in container.member_paths]
            owners: dict[tuple[str, int], list[BRepOccurrence]] = {}
            for occurrence in placed:
                faces = {
                    face
                    for shell in solid_shells[occurrence.solid]
                    for face in shell_faces[shell]
                }
                edges = {
                    coedge_edges[coedge]
                    for face in faces
                    for loop in face_loops[face]
                    for coedge in loop
                }
                vertices = {vertex for edge in edges for vertex in edge_vertices[edge]}
                for kind, indices in (
                    ("face", faces),
                    ("edge", edges),
                    ("vertex", vertices),
                ):
                    for index in indices:
                        owners.setdefault((kind, index), []).append(occurrence)
            for uses in owners.values():
                if len(uses) > 1 and len({use.solid for use in uses}) == len(uses):
                    first = uses[0]
                    if any(
                        (
                            first.rotation != other.rotation
                            or first.translation != other.translation
                            for other in uses[1:]
                        )
                    ):
                        findings.append(
                            BRepValidationFinding(
                                "container-incidence",
                                None,
                                1.0,
                                0.0,
                                "A declared shared container stratum requires one consistent definition placement.",
                            )
                        )

    @staticmethod
    def _validate_edges(
        points: np.ndarray,
        curves: tuple[BRepCurve, ...],
        edge_curves: tuple[int, ...],
        ranges: np.ndarray,
        edge_vertices: tuple[tuple[int, int], ...],
        vertex_roots: tuple[BRepVertexRoot | None, ...],
        findings: list[BRepValidationFinding],
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None,
        edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None,
        edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    ) -> None:
        vertex_count = points.shape[0]
        scale = max(1.0, float(np.max(np.abs(points), initial=0.0)))
        tolerance = 1e-09 * scale
        for edge, (curve_index, (start, end)) in enumerate(
            zip(edge_curves, edge_vertices, strict=True)
        ):
            if not (0 <= start < vertex_count and 0 <= end < vertex_count):
                findings.append(
                    BRepValidationFinding(
                        "edge-incidence",
                        None,
                        1.0,
                        0.0,
                        "An edge references an absent vertex.",
                    )
                )
                continue
            if curve_index == -1:
                if start != end:
                    findings.append(
                        BRepValidationFinding(
                            "edge-incidence",
                            None,
                            1.0,
                            0.0,
                            "A degenerate edge must start and end at one vertex.",
                        )
                    )
                    continue
                continue
            if not 0 <= curve_index < len(curves):
                findings.append(
                    BRepValidationFinding(
                        "edge-incidence",
                        None,
                        1.0,
                        0.0,
                        "An edge references an absent curve.",
                    )
                )
                continue
            curve = curves[curve_index]
            first, last = curve.validate_range(ranges[edge, 0], ranges[edge, 1])
            ends = _carrier_points_host(
                curve,
                np.asarray((first, last), dtype=np.float64),
                budget=budget,
                preparations=preparations,
            )
            certified_gap = edge_vertex_gap_bounds is not None
            if edge_vertex_gap_bounds is not None:
                _certify_endpoint_gaps(
                    curve,
                    points,
                    (start, end),
                    (first, last),
                    edge_vertex_gap_bounds[edge],
                    vertex_roots,
                    budget=budget,
                    endpoint_roots=edge_endpoint_roots[edge],
                )
            if not certified_gap and (
                vertex_roots[start] is None
                and np.linalg.norm(ends[0] - points[start]) > tolerance
                or (
                    vertex_roots[end] is None
                    and np.linalg.norm(ends[1] - points[end]) > tolerance
                )
            ):
                findings.append(
                    BRepValidationFinding(
                        "edge-incidence",
                        None,
                        1.0,
                        0.0,
                        f"Edge {edge} curve end points do not meet its vertices.",
                    )
                )
                continue

    @staticmethod
    def _validate_shells(
        face_count: int,
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        findings: list[BRepValidationFinding],
    ) -> None:
        if len(shell_faces) != len(shell_orientations):
            findings.append(
                BRepValidationFinding(
                    "shell-incidence",
                    None,
                    1.0,
                    0.0,
                    "Shell faces and orientations must align.",
                )
            )
            return
        members = [face for faces in shell_faces for face in faces]
        if any((face < 0 or face >= face_count for face in members)):
            findings.append(
                BRepValidationFinding(
                    "shell-incidence",
                    None,
                    1.0,
                    0.0,
                    "A shell references an absent face.",
                )
            )
        incidences: dict[int, list[tuple[int, int]]] = {}
        for faces, signs in zip(shell_faces, shell_orientations, strict=True):
            if not faces or len(faces) != len(signs):
                findings.append(
                    BRepValidationFinding(
                        "shell-incidence",
                        None,
                        1.0,
                        0.0,
                        "Every shell needs faces with one orientation each.",
                    )
                )
            if any((sign not in (-1, 1) for sign in signs)):
                findings.append(
                    BRepValidationFinding(
                        "shell-incidence",
                        None,
                        1.0,
                        0.0,
                        "Shell face orientations must be -1 or 1.",
                    )
                )
            if len(set(faces)) != len(faces):
                findings.append(
                    BRepValidationFinding(
                        "shell-incidence", None, 1.0, 0.0, "A shell cannot repeat a face."
                    )
                )
        for shell, (faces, signs) in enumerate(
            zip(shell_faces, shell_orientations, strict=True)
        ):
            if len(faces) != len(signs):
                continue
            for face, sign in zip(faces, signs, strict=True):
                incidences.setdefault(face, []).append((shell, sign))
        shell_owner = {
            shell: solid for solid, group in enumerate(solid_shells) for shell in group
        }
        for uses in incidences.values():
            if len(uses) > 2:
                findings.append(
                    BRepValidationFinding(
                        "shell-incidence",
                        None,
                        1.0,
                        0.0,
                        "A face has at most two region-side incidences.",
                    )
                )
            if len(uses) == 2:
                (first_shell, first_sign), (second_shell, second_sign) = uses
                if (
                    first_sign == second_sign
                    or first_shell not in shell_owner
                    or second_shell not in shell_owner
                    or (shell_owner[first_shell] == shell_owner[second_shell])
                ):
                    findings.append(
                        BRepValidationFinding(
                            "shell-incidence",
                            None,
                            1.0,
                            0.0,
                            "A shared face must separate distinct solids with opposite incidences.",
                        )
                    )
        shells = [shell for group in solid_shells for shell in group]
        if any((shell < 0 or shell >= len(shell_faces) for shell in shells)):
            findings.append(
                BRepValidationFinding(
                    "shell-incidence",
                    None,
                    1.0,
                    0.0,
                    "A solid references an absent shell.",
                )
            )
        if len(set(shells)) != len(shells) or any((not group for group in solid_shells)):
            findings.append(
                BRepValidationFinding(
                    "shell-incidence",
                    None,
                    1.0,
                    0.0,
                    "Each solid owns a nonempty set of distinct shells.",
                )
            )

    @property
    def coedge_faces(self) -> tuple[int, ...]:
        owners = [0] * len(self.coedge_edges)
        for face, loops in enumerate(self.face_loops):
            for loop in loops:
                for coedge in loop:
                    owners[coedge] = face
        return tuple(owners)

    @property
    def degenerate_edges(self) -> tuple[bool, ...]:
        return tuple(index == -1 for index in self.edge_curves)

    def topology(self) -> BRepTopology:
        """Return incidence relations for the oriented B-Rep topology."""
        face_wires = tuple(
            tuple(
                tuple(self.coedge_senses[c] * (self.coedge_edges[c] + 1) for c in loop)
                for loop in loops
            )
            for loops in self.face_loops
        )
        face_edges = tuple(
            tuple(dict.fromkeys(abs(e) - 1 for wire in wires for e in wire))
            for wires in face_wires
        )
        edge_faces: list[list[int]] = [[] for _ in self.edge_curves]
        for face, edges in enumerate(face_edges):
            for edge in edges:
                edge_faces[edge].append(face)
        solid_faces = tuple(
            tuple(face for shell in shells for face in self.shell_faces[shell])
            for shells in self.solid_shells
        )
        solid_orientations = tuple(
            tuple(sign for shell in shells for sign in self.shell_orientations[shell])
            for shells in self.solid_shells
        )
        return BRepTopology(
            face_edges=face_edges,
            edge_faces=tuple(tuple(faces) for faces in edge_faces),
            face_wires=face_wires,
            solid_faces=solid_faces,
            solid_face_orientations=solid_orientations,
            num_vertices=self.vertex_points.shape[0],
        )

    def edge_parameter_enclosure(
        self,
        edge: int,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Rows enclose each true endpoint parameter; literals have zero width."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    edge,
                )
            ) as budget:
                return self.edge_parameter_enclosure(
                    edge, preparations=preparations, budget=budget
                )
        if not 0 <= edge < len(self.edge_curves):
            raise ValueError(
                "BRepGeometry.edge must index a declared edge; got an absent edge."
            )
        values = np.asarray(self.edge_ranges)[edge]
        return np.asarray(
            [
                (float(values[endpoint]), float(values[endpoint]))
                if root is None
                else root.parameter_enclosure(preparations=preparations, budget=budget)
                for endpoint, root in enumerate(self.edge_endpoint_roots[edge])
            ],
            dtype=np.float64,
        )

    def euler_characteristic(self) -> int:
        """``V - E + F - (L - F)`` over non-degenerate edges (Euler-Poincare form).

        For closed manifold shells this equals ``2 * (shells - genus)``.
        """
        edges = sum(1 for index in self.edge_curves if index != -1)
        faces = len(self.face_loops)
        loops = sum(len(loops) for loops in self.face_loops)
        return self.vertex_points.shape[0] - edges + faces - (loops - faces)

    def edge_use_balance(
        self, face_orientation: ArrayLike, /, *, solid: int | None = None
    ) -> tuple[int, ...]:
        """Per edge: signed count of oriented uses by shell faces.

        Each coedge of a shelled face contributes ``sense * face_orientation *
        shell_orientation`` (loops are counterclockwise about the parametric
        normal, so the oriented face traverses them in that product's
        direction). A consistently oriented closed shell uses every
        non-degenerate edge once in each direction: every entry is zero.
        """
        signs = np.asarray(face_orientation, dtype=np.float64).reshape(-1)
        if signs.shape != (len(self.face_loops),):
            raise ValueError(
                f"BRepGeometry.face_orientation must contain one sign per face; got shape={signs.shape}, faces={len(self.face_loops)}"
            )
        if solid is not None and not 0 <= solid < len(self.solid_shells):
            raise ValueError(
                f"BRepGeometry.solid must index an existing solid; got index={solid}, solids={len(self.solid_shells)}"
            )
        shells = (
            range(len(self.shell_faces)) if solid is None else self.solid_shells[solid]
        )
        balance = [0] * len(self.edge_curves)
        for shell in shells:
            for face, sign in zip(
                self.shell_faces[shell], self.shell_orientations[shell], strict=True
            ):
                orientation = sign * int(signs[face])
                for loop in self.face_loops[face]:
                    for coedge in loop:
                        balance[self.coedge_edges[coedge]] += (
                            self.coedge_senses[coedge] * orientation
                        )
        return tuple(balance)

    def qualified_entity_incidence(
        self, source_revision: str, /
    ) -> tuple[BRepQualifiedIncidence, ...]:
        """Publish authored instance closure without coordinate or prefix merging."""
        topology = self.topology()
        qualification = self._prepare_incidence_qualification(source_revision, topology)

        records: set[BRepQualifiedIncidence] = set()
        seen_faces: set[BRepEntityId] = set()
        seen_edges: set[BRepEntityId] = set()
        instances = [
            (index, topology.solid_faces[occurrence.solid])
            for index, occurrence in enumerate(self.occurrences)
        ]
        unshelled = tuple(
            face for face, owners in enumerate(topology.face_solids) if not owners
        )
        if unshelled:
            instances.append((-1, unshelled))
        for occurrence_index, faces in instances:
            if occurrence_index >= 0:
                occurrence = self.occurrences[occurrence_index]
                solid = qualification.entity(occurrence_index, "solid", occurrence.solid)
                records.add(BRepQualifiedIncidence(solid, solid, 1))
                for face, sign in zip(
                    faces, topology.solid_face_orientations[occurrence.solid], strict=True
                ):
                    records.add(
                        BRepQualifiedIncidence(
                            solid,
                            qualification.entity(occurrence_index, "face", face),
                            sign,
                        )
                    )
            for face in faces:
                face_id = qualification.entity(occurrence_index, "face", face)
                phase = self._collect_face_incidence(
                    face,
                    occurrence_index,
                    topology,
                    qualification,
                    face_id not in seen_faces,
                    seen_edges,
                )
                records.update(phase.rows)
                seen_faces.add(face_id)
                seen_edges.update(phase.edges)
        for edge, owners in enumerate(topology.edge_faces):
            if not owners:
                edge_id = BRepEntityId(source_revision, "edge", edge)
                records.add(BRepQualifiedIncidence(edge_id, edge_id, 1))
                for endpoint, vertex in enumerate(self.edge_vertices[edge]):
                    vertex_id = BRepEntityId(source_revision, "vertex", vertex)
                    records.add(
                        BRepQualifiedIncidence(
                            edge_id, vertex_id, -1 if endpoint == 0 else 1, endpoint
                        )
                    )
                    records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
        incident_vertices = {
            vertex for endpoints in self.edge_vertices for vertex in endpoints
        }
        for vertex in range(topology.num_vertices):
            if vertex not in incident_vertices:
                vertex_id = BRepEntityId(source_revision, "vertex", vertex)
                records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
        return tuple(sorted(records))

    def _collect_face_incidence(
        self,
        face: int,
        occurrence_index: int,
        topology: BRepTopology,
        qualification: _IncidenceQualification,
        first_use: bool,
        seen_edges: set[BRepEntityId],
    ) -> _IncidenceRowsPhase:
        records: set[BRepQualifiedIncidence] = set()
        new_edges: list[BRepEntityId] = []
        face_id = qualification.entity(occurrence_index, "face", face)
        records.add(BRepQualifiedIncidence(face_id, face_id, 1))
        if first_use:
            for loop in self.face_loops[face]:
                for coedge in loop:
                    edge = self.coedge_edges[coedge]
                    edge_id = qualification.entity(occurrence_index, "edge", edge)
                    records.add(
                        BRepQualifiedIncidence(
                            face_id, edge_id, self.coedge_senses[coedge], coedge
                        )
                    )
        for edge in topology.face_edges[face]:
            edge_id = qualification.entity(occurrence_index, "edge", edge)
            records.add(BRepQualifiedIncidence(edge_id, edge_id, 1))
            if edge_id not in seen_edges:
                new_edges.append(edge_id)
                for endpoint, vertex in enumerate(self.edge_vertices[edge]):
                    vertex_id = qualification.entity(occurrence_index, "vertex", vertex)
                    records.add(
                        BRepQualifiedIncidence(
                            edge_id,
                            vertex_id,
                            -1 if endpoint == 0 else 1,
                            endpoint,
                        )
                    )
                    records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
        return _IncidenceRowsPhase(rows=tuple(sorted(records)), edges=tuple(new_edges))

    def _prepare_incidence_qualification(
        self, source_revision: str, topology: BRepTopology
    ) -> _IncidenceQualification:
        membership = {
            path: container.path
            for container in self.assembly_containers
            for path in container.member_paths
        }
        grouped: dict[tuple[tuple[str, ...], str, int], list[int]] = {}
        for occurrence_index, occurrence in enumerate(self.occurrences):
            faces = topology.solid_faces[occurrence.solid]
            edges = tuple(
                sorted({edge for face in faces for edge in topology.face_edges[face]})
            )
            vertices = tuple(
                sorted({vertex for edge in edges for vertex in self.edge_vertices[edge]})
            )
            for kind, indices in (("face", faces), ("edge", edges), ("vertex", vertices)):
                if occurrence.path in membership:
                    for index in indices:
                        grouped.setdefault(
                            (membership[occurrence.path], kind, index), []
                        ).append(occurrence_index)
        return _IncidenceQualification(
            geometry=self,
            source_revision=source_revision,
            membership=membership,
            grouped={key: tuple(owners) for key, owners in grouped.items()},
        )


class _IncidenceQualification(StrictModule):
    """Prepared occurrence membership and shared-stratum identity rules."""

    geometry: BRepGeometry
    source_revision: str = eqx.field(static=True)
    membership: dict[tuple[str, ...], tuple[str, ...]] = eqx.field(static=True)
    grouped: dict[tuple[tuple[str, ...], str, int], tuple[int, ...]] = eqx.field(
        static=True
    )

    def entity(self, occurrence_index: int, kind: str, index: int, /) -> BRepEntityId:
        if occurrence_index < 0:
            return BRepEntityId(self.source_revision, kind, index)
        occurrence = self.geometry.occurrences[occurrence_index]
        container = self.membership.get(occurrence.path)
        path = occurrence.path
        owners = (
            () if container is None else self.grouped.get((container, kind, index), ())
        )
        if (
            kind != "solid"
            and container is not None
            and len(owners) > 1
            and len({self.geometry.occurrences[owner].solid for owner in owners})
            == len(owners)
        ):
            path = container
        return BRepEntityId(self.source_revision, kind, index, path)


@checked
def brep_physical_scale(
    geometry: BRepGeometry,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> float:
    """Return a translation-independent length scale from edge and vertex enclosures."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((geometry,)) as budget:
            return brep_physical_scale(geometry, preparations=preparations, budget=budget)
    points = np.asarray(geometry.vertex_points)
    boxes = []
    literal = [
        points[index] for index, root in enumerate(geometry.vertex_roots) if root is None
    ]
    if literal:
        values = np.asarray(literal, dtype=np.float64)
        boxes.append(np.stack((np.min(values, axis=0), np.max(values, axis=0))))
    boxes.extend(
        root.point_enclosure(preparations=preparations, budget=budget)
        for root in geometry.vertex_roots
        if root is not None
    )
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index >= 0:
            endpoints = geometry.edge_parameter_enclosure(
                edge, preparations=preparations, budget=budget
            )
            curve = geometry.curves[curve_index]
            box = (
                curve.bounding_box(
                    float(endpoints[0, 0]),
                    float(endpoints[1, 1]),
                    endpoint_roots=geometry.edge_endpoint_roots[edge],
                    preparations=preparations,
                    budget=budget,
                )
                if isinstance(curve, IntersectionCurve)
                else curve.bounding_box(
                    float(endpoints[0, 0]),
                    float(endpoints[1, 1]),
                    preparations=preparations,
                    budget=budget,
                )
            )
            boxes.append(box)
    if not boxes:
        return 0.0
    source_boxes = np.asarray(boxes, dtype=np.float64)
    if not np.all(np.isfinite(source_boxes)):
        raise ValueError(
            f"BRepGeometry.source_enclosures must be finite to determine physical scale; got {source_boxes!r}"
        )
    extent = np.max(source_boxes[:, 1], axis=0) - np.min(source_boxes[:, 0], axis=0)
    maximum = float(np.max(extent))
    return 0.0 if maximum == 0.0 else float(np.nextafter(maximum, np.inf))


class BRepBoundaryMap(AbstractBoundaryMap):
    """Dispatch heterogeneous JAX surface patches over normalized face charts."""

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array

    def __init__(
        self,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: ArrayLike,
    ) -> None:
        bounds = np.asarray(parameter_bounds, dtype=np.float64)
        if not patches:
            raise ValueError(
                f"BRepBoundaryMap.patches must contain at least one patch; got count={len(patches)}"
            )
        if bounds.shape != (len(patches), 2, 2):
            raise ValueError(
                f"BRepBoundaryMap.parameter_bounds must have shape (num_faces, 2, 2); got shape={bounds.shape}, faces={len(patches)}"
            )
        if not _surface_intervals_valid(bounds):
            raise ValueError(
                f"BRepBoundaryMap.parameter_bounds must contain finite nonempty intervals; got {bounds!r}"
            )
        self.patches = patches
        self.parameter_bounds = jnp.asarray(bounds, dtype=jnp.float64)

    @property
    def num_charts(self) -> int:
        return len(self.patches)

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _map_one(self, chart_index: Array, reference: Array) -> Array:
        bounds = self.parameter_bounds[chart_index]
        parameters = bounds[0] + reference * (bounds[1] - bounds[0])
        branches = tuple(
            lambda coordinate, patch=patch: patch.evaluate(coordinate)
            for patch in self.patches
        )
        return jax.lax.switch(chart_index, branches, parameters)

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=self.parameter_bounds.dtype)
        leading = indices.shape
        values = jax.vmap(self._map_one)(
            indices.reshape((-1,)), reference_.reshape((-1, 2))
        )
        return values.reshape((*leading, 3))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=self.parameter_bounds.dtype)
        leading = indices.shape
        differential = jax.vmap(
            lambda index, coordinate: jax.jacfwd(
                lambda value: self._map_one(index, value)
            )(coordinate)
        )(indices.reshape((-1,)), reference_.reshape((-1, 2)))
        jacobian = jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )
        return jacobian.reshape(leading)


class BRepPlacedBoundaryMap(AbstractBoundaryMap):
    """Occurrence-qualified charts with definition face identity kept separate."""

    base: BRepBoundaryMap
    definition_faces: Array
    rotations: Array
    translations: Array
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    @property
    def num_charts(self) -> int:
        return self.definition_faces.shape[0]

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        definitions = self.definition_faces[chart_indices]
        points = self.base.map(definitions, reference)
        return (self.rotations[chart_indices] @ points[..., None])[
            ..., 0
        ] + self.translations[chart_indices]

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(self.definition_faces[chart_indices], reference)


def _trim_uv_root(endpoint: RootEndpoint, /) -> TrimIntersectionRoot | None:
    """Return the endpoint's UV trim root; scalar period endpoints have no spatial root."""
    match endpoint:
        case TrimRootEndpoint() | BranchRootEndpoint():
            return (
                endpoint.root if isinstance(endpoint.root, TrimIntersectionRoot) else None
            )
        case NativePeriodEndpoint():
            return None
        case _:
            assert_never(endpoint)


def _pcurve_inside_parameter_box(
    pcurve: BRepPCurve,
    first: float,
    last: float,
    parameter_box: np.ndarray,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None],
    slack: float,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove trim containment by bounded subdivision of source enclosures."""
    pending = [(float(first), float(last), 0)]
    processed = 0
    while pending:
        lower, upper, depth = pending.pop()
        roots = (
            endpoint_roots[0] if lower == first else None,
            endpoint_roots[1] if upper == last else None,
        )
        enclosure = (
            pcurve.bounding_box(
                lower,
                upper,
                endpoint_roots=roots,
                preparations=preparations,
                budget=budget,
            )
            if isinstance(pcurve, IntersectionPCurve)
            else pcurve.bounding_box(
                lower, upper, preparations=preparations, budget=budget
            )
        )
        if np.all(enclosure[0] >= parameter_box[0] - slack) and np.all(
            enclosure[1] <= parameter_box[1] + slack
        ):
            continue
        middle = lower + 0.5 * (upper - lower)
        points = _carrier_points_host(
            pcurve,
            np.asarray((lower, middle, upper), dtype=np.float64),
            budget=budget,
            preparations=preparations,
        )
        if np.any(points < parameter_box[0] - slack) or np.any(
            points > parameter_box[1] + slack
        ):
            return False
        processed += 1
        if depth == 32 or processed > 65_536 or middle in (lower, upper):
            return False
        pending.extend(((middle, upper, depth + 1), (lower, middle, depth + 1)))
    return True


def _validate_coedge_roots(
    geometry: BRepGeometry,
    patch: AbstractSurfacePatch,
    coedge: int,
    points: np.ndarray,
    findings: list[BRepValidationFinding],
    entity: BRepEntityId,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
    box: np.ndarray,
) -> bool:
    edge = geometry.coedge_edges[coedge]
    for endpoint, definition in enumerate(geometry.coedge_endpoint_roots[coedge]):
        if definition is None:
            continue
        if isinstance(definition, NativePeriodEndpoint):
            pcurve = geometry.pcurves[coedge]
            if canonical_fingerprint(
                carrier_owner.carrier_payload(definition.carrier)
            ) != canonical_fingerprint(carrier_owner.carrier_payload(pcurve)):
                findings.append(
                    BRepValidationFinding(
                        "root-binding",
                        entity,
                        1.0,
                        0.0,
                        "BRepModel.coedge_root must retain its source p-curve; got a different carrier.",
                    )
                )
                return False
            continue
        vertex_root = geometry.vertex_roots[geometry.edge_vertices[edge][endpoint]]
        if vertex_root is None:
            findings.append(
                BRepValidationFinding(
                    "root-binding",
                    entity,
                    1.0,
                    0.0,
                    "A coedge root lost its vertex definition.",
                )
            )
            return False
        carrier = definition.carrier
        pcurve = geometry.pcurves[coedge]
        same = canonical_fingerprint(
            carrier_owner.carrier_payload(carrier)
        ) == canonical_fingerprint(carrier_owner.carrier_payload(pcurve))
        if isinstance(carrier, IntersectionPCurve) and isinstance(
            pcurve, IntersectionPCurve
        ):
            same = (
                carrier.curve.branch_id == pcurve.curve.branch_id
                and carrier.side == pcurve.side
                and (carrier.reversed == pcurve.reversed)
            )
        if isinstance(carrier, IntersectionCurve) and isinstance(
            pcurve, IntersectionPCurve
        ):
            same = carrier.branch_id == pcurve.curve.branch_id and (not pcurve.reversed)
        if (
            not same
            and isinstance(carrier, (AbstractCurve, IntersectionCurve))
            and (carrier.ambient_dimension == 3)
        ):
            source_first, source_last = definition.parameter_enclosure(
                preparations=preparations, budget=budget
            )
            supporting = patch
            source_point = points[geometry.edge_vertices[edge][endpoint]]
            if isinstance(vertex_root.primary, BRepPlacedVertex):
                if not isinstance(
                    patch, PlacedSurface
                ) or not vertex_root.primary.matches_pose(patch):
                    findings.append(
                        BRepValidationFinding(
                            "root-binding",
                            entity,
                            1.0,
                            0.0,
                            "BRepModel.coedge_root must retain the supporting source pose; got a different pose.",
                        )
                    )
                    return False
                supporting = patch.definition
                source_point = np.asarray(vertex_root.primary.source_point)
            binding = correspondence_owner.certify_curve_surface(
                carrier,
                pcurve,
                supporting,
                box,
                source_first,
                source_last,
                point=source_point,
                tolerance=0.0,
                endpoint_roots=(definition, definition),
                preparations=preparations,
                budget=budget,
            )
            same = binding.complete and binding.deviation_bound == 0.0
        identity = not definition.affine_transforms
        if isinstance(definition, TrimRootEndpoint):
            identity &= (
                definition.affine_parameter_scale == 1.0
                and definition.affine_parameter_offset == 0.0
            )
        if not same or not identity:
            findings.append(
                BRepValidationFinding(
                    "root-binding",
                    entity,
                    1.0,
                    0.0,
                    "BRepModel.coedge_root must bind its source carrier parameter expression; got a changed carrier or parameter transform.",
                )
            )
            return False
        if isinstance(definition.root, TrimIntersectionRoot) and (
            not vertex_root.supports_uv(patch, definition.root)
        ):
            findings.append(
                BRepValidationFinding(
                    "root-binding",
                    entity,
                    1.0,
                    0.0,
                    "BRepModel.coedge_root must have verified UV support on its face; got unsupported UV root.",
                )
            )
            return False
    return True


def _certify_coedge(
    geometry: BRepGeometry,
    patch: AbstractSurfacePatch,
    coedge: int,
    points: np.ndarray,
    findings: list[BRepValidationFinding],
    entity: BRepEntityId,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
    box: np.ndarray,
    parameters: np.ndarray,
    parameter_scale: float,
    curve_surface_tolerance: float,
) -> _CoedgeValidationPhase | None:
    edge = geometry.coedge_edges[coedge]
    endpoint_boxes = geometry.edge_parameter_enclosure(
        edge, preparations=preparations, budget=budget
    )
    for endpoint, definition in enumerate(geometry.coedge_endpoint_roots[coedge]):
        if definition is not None:
            lower, upper = definition.parameter_enclosure(
                preparations=preparations, budget=budget
            )
            endpoint_boxes[endpoint, 0] = max(endpoint_boxes[endpoint, 0], lower)
            endpoint_boxes[endpoint, 1] = min(endpoint_boxes[endpoint, 1], upper)
    if np.any(endpoint_boxes[:, 0] > endpoint_boxes[:, 1]):
        findings.append(
            BRepValidationFinding(
                "trim-containment",
                entity,
                1.0,
                0.0,
                "Source-proven endpoint scalar enclosures have no common value.",
            )
        )
        return None
    proof_first, proof_last = (float(endpoint_boxes[0, 0]), float(endpoint_boxes[1, 1]))
    uv = _carrier_points_host(
        geometry.pcurves[coedge],
        parameters,
        budget=budget,
        preparations=preparations,
    )
    pcurve = geometry.pcurves[coedge]
    slack = 1e-09 * parameter_scale
    if not _pcurve_inside_parameter_box(
        pcurve,
        proof_first,
        proof_last,
        box,
        geometry.coedge_endpoint_roots[coedge],
        slack,
        preparations=preparations,
        budget=budget,
    ):
        findings.append(
            BRepValidationFinding(
                "trim-containment",
                entity,
                1.0,
                0.0,
                f"Coedge {coedge} p-curve leaves its face parameter box.",
            )
        )
        return None
    curve_index = geometry.edge_curves[edge]
    source_curve = None if curve_index == -1 else geometry.curves[curve_index]
    supporting_patch = patch
    source_point = points[geometry.edge_vertices[edge][0]]
    placed_vertex = geometry.vertex_roots[geometry.edge_vertices[edge][0]]
    if (
        source_curve is None
        and isinstance(patch, PlacedSurface)
        and (placed_vertex is not None)
        and isinstance(placed_vertex.primary, BRepPlacedVertex)
        and placed_vertex.primary.matches_pose(patch)
    ):
        supporting_patch = patch.definition
        source_point = np.asarray(placed_vertex.primary.source_point)
    stored_gaps = geometry.edge_face_gap_bounds
    if stored_gaps is not None and source_curve is not None:
        with budget.temporary_scope():
            budget.reserve(128, 8192)
            certified_gap = _edge_face_gap_squared(
                source_curve,
                pcurve,
                supporting_patch,
                proof_first,
                proof_last,
                budget=budget,
            )
        if certified_gap is not None:
            if certified_gap > Fraction(stored_gaps[coedge]) ** 2:
                raise ValueError(
                    "BRepGeometry.edge_face_gap_bounds must enclose full endpoint ranges; got exceeded bound"
                )
            return _CoedgeValidationPhase(
                deviation=square_root_interval(certified_gap)[1], uv=uv
            )
    correspondence_limit = curve_surface_tolerance if stored_gaps is None else 0.0
    witness = correspondence_owner.certify_curve_surface(
        source_curve,
        geometry.pcurves[coedge],
        supporting_patch,
        box,
        proof_first,
        proof_last,
        point=source_point,
        tolerance=correspondence_limit,
        endpoint_roots=geometry.coedge_endpoint_roots[coedge],
        preparations=preparations,
        budget=budget,
    )
    deviation = witness.deviation_bound
    match patch:
        case PlacedSurface() if supporting_patch is not patch:
            deviation = placed_deviation_bound(np.asarray(patch.rotation), deviation)
        case _:
            pass

    if not witness.complete or deviation > correspondence_limit:
        findings.append(
            BRepValidationFinding(
                "curve-surface",
                entity,
                deviation,
                correspondence_limit,
                f"BRepModel.coedge[{coedge}] must meet its certified carrier limit; got bound={deviation}, limit={correspondence_limit}",
                coedge=coedge,
                unresolved=witness.unresolved,
                intervals_processed=witness.intervals_processed,
            )
        )
    return _CoedgeValidationPhase(deviation=deviation, uv=uv)


def _validate_trim_junctions(
    geometry: BRepGeometry,
    patch: AbstractSurfacePatch,
    loop: tuple[int, ...],
    heads: list[np.ndarray],
    tails: list[np.ndarray],
    parameter_scale: float,
    findings: list[BRepValidationFinding],
    entity: BRepEntityId,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> None:
    gaps = np.linalg.norm(np.asarray(tails) - np.roll(heads, -1, axis=0), axis=1)
    for position, coedge in enumerate(loop):
        following = loop[(position + 1) % len(loop)]
        end = 1 if geometry.coedge_senses[coedge] > 0 else 0
        start = 0 if geometry.coedge_senses[following] > 0 else 1
        vertex = geometry.edge_vertices[geometry.coedge_edges[coedge]][end]
        root = geometry.vertex_roots[vertex]
        if root is None or (
            isinstance(root.primary, BRepPlacedVertex)
            and root.primary.source_root is None
        ):
            if gaps[position] > 1e-09 * parameter_scale:
                findings.append(
                    BRepValidationFinding(
                        "trim-closure",
                        entity,
                        float(gaps[position]),
                        1.0e-9 * parameter_scale,
                        "Face loops must be closed in the parameter plane.",
                    )
                )
                continue
            continue
        first = geometry.coedge_endpoint_roots[coedge][end]
        second = geometry.coedge_endpoint_roots[following][start]
        if first is None or second is None:
            findings.append(
                BRepValidationFinding(
                    "trim-closure",
                    entity,
                    1.0,
                    0.0,
                    "A root-valued trim junction lost an endpoint binding.",
                )
            )
            continue
        first_uv, second_uv = (_trim_uv_root(first), _trim_uv_root(second))
        same_uv_root = (
            first_uv is not None
            and second_uv is not None
            and (first_uv.root_id == second_uv.root_id)
        )
        if not same_uv_root and (
            not root.same_uv_endpoint(
                patch,
                geometry.pcurves[coedge],
                first,
                geometry.pcurves[following],
                second,
                preparations=preparations,
                budget=budget,
            )
        ):
            findings.append(
                BRepValidationFinding(
                    "trim-closure",
                    entity,
                    1.0,
                    0.0,
                    "The trim junction has no common source-root and injective chart certificate.",
                )
            )
            continue


def _validate_exact_geometry(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    bounds: np.ndarray,
    topology: BRepTopology,
    curve_surface_tolerance: float,
    source_revision: str,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
    findings: list[BRepValidationFinding],
) -> np.ndarray:
    """Bind incidence and whole-edge source correspondence, retaining its bounds."""

    derived = geometry.topology()
    if (
        derived.face_edges,
        derived.edge_faces,
        derived.face_wires,
        derived.solid_faces,
        derived.solid_face_orientations,
        derived.num_vertices,
    ) != (
        topology.face_edges,
        topology.edge_faces,
        topology.face_wires,
        topology.solid_faces,
        topology.solid_face_orientations,
        topology.num_vertices,
    ):
        findings.append(
            BRepValidationFinding(
                "topology",
                None,
                1.0,
                0.0,
                "topology must be the incidence view of the exact geometry.",
            )
        )
        return np.full((len(geometry.coedge_edges),), np.inf, dtype=np.float64)
    if geometry.edge_face_gap_bounds is not None:
        _certify_edge_face_gap_bounds(
            geometry.curves,
            geometry.edge_curves,
            np.asarray(geometry.edge_ranges),
            geometry.pcurves,
            geometry.coedge_edges,
            geometry.coedge_faces,
            patches,
            budget=budget,
            declared_bounds=geometry.edge_face_gap_bounds,
            edge_endpoint_roots=geometry.edge_endpoint_roots,
            coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        )
    points = np.asarray(geometry.vertex_points)
    ranges = np.asarray(geometry.edge_ranges)
    witnesses = np.full((len(geometry.coedge_edges),), np.inf, dtype=np.float64)
    for face, loops in enumerate(geometry.face_loops):
        patch = patches[face]
        box = patch.validate_parameter_box(bounds[face], budget=budget)
        parameter_scale = max(1.0, float(np.max(np.abs(box))))
        entity = BRepEntityId(source_revision, "face", face)
        for loop in loops:
            heads: list[np.ndarray] = []
            tails: list[np.ndarray] = []
            for coedge in loop:
                if not _validate_coedge_roots(
                    geometry,
                    patch,
                    coedge,
                    points,
                    findings,
                    entity,
                    preparations=preparations,
                    budget=budget,
                    box=box,
                ):
                    continue
                phase = _certify_coedge(
                    geometry,
                    patch,
                    coedge,
                    points,
                    findings,
                    entity,
                    preparations=preparations,
                    budget=budget,
                    box=box,
                    parameters=ranges[geometry.coedge_edges[coedge]],
                    parameter_scale=parameter_scale,
                    curve_surface_tolerance=curve_surface_tolerance,
                )
                if phase is None:
                    continue
                witnesses[coedge] = phase.deviation
                ordered = (
                    phase.uv if geometry.coedge_senses[coedge] > 0 else phase.uv[::-1]
                )
                heads.append(ordered[0])
                tails.append(ordered[-1])
            if len(heads) == len(loop):
                _validate_trim_junctions(
                    geometry,
                    patch,
                    loop,
                    heads,
                    tails,
                    parameter_scale,
                    findings,
                    entity,
                    preparations=preparations,
                    budget=budget,
                )
    return witnesses


def _validate_tessellation_chart_restrictions(
    source_revision: str,
    vertices: np.ndarray,
    edges: np.ndarray,
    endpoint_parameters: np.ndarray,
    ratios: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    triangle_parameters: np.ndarray,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    source_parameters: np.ndarray,
    occurrence_ids: np.ndarray,
    triangle_occurrence_ids: np.ndarray,
    /,
    *,
    findings: list[BRepValidationFinding],
) -> tuple[str, ...]:
    """Validate each persisted rational UV restriction against its source incidence."""
    count = vertices.size
    if (
        vertices.dtype != np.int64
        or vertices.shape != (count,)
        or edges.dtype != np.int64
        or (edges.shape != (count, 2))
        or (endpoint_parameters.dtype != np.float64)
        or (endpoint_parameters.shape != (count, 2, 2))
        or (ratios.dtype != np.int64)
        or (ratios.shape != (count, 2))
    ):
        raise TypeError(
            f"BRepModel.chart_restrictions must have aligned int64 index banks and float64 parameter banks; got vertices=({vertices.shape}, {vertices.dtype}), edges=({edges.shape}, {edges.dtype}), parameters=({endpoint_parameters.shape}, {endpoint_parameters.dtype}), ratios=({ratios.shape}, {ratios.dtype})"
        )
    if count == 0:
        return ()
    if (
        not np.array_equal(vertices, np.unique(vertices))
        or np.any(vertices < 0)
        or np.any(vertices >= source_dimensions.size)
        or np.any(edges < 0)
        or np.any(edges >= source_dimensions.size)
        or (not np.all(np.isfinite(endpoint_parameters)))
    ):
        findings.append(
            BRepValidationFinding(
                "chart-restriction",
                None,
                1.0,
                0.0,
                "Tessellation chart restrictions have invalid vertex incidence.",
            )
        )
        return ()
    exact_by_vertex: dict[int, tuple[Fraction, Fraction]] = {}
    identifiers = []
    for row, vertex_ in enumerate(vertices.tolist()):
        starting_findings = len(findings)
        vertex = int(vertex_)
        first, second = (int(value) for value in edges[row])
        numerator, denominator = (int(value) for value in ratios[row])
        if (
            max(first, second) >= vertex
            or first == second
            or denominator <= 0
            or (not 0 < numerator < denominator)
        ):
            findings.append(
                BRepValidationFinding(
                    "chart-restriction",
                    None,
                    1.0,
                    0.0,
                    "Tessellation chart restrictions are not ordered open-edge roots.",
                )
            )
            continue
        if source_dimensions[vertex] != 2:
            findings.append(
                BRepValidationFinding(
                    "chart-restriction",
                    None,
                    1.0,
                    0.0,
                    "An exact chart restriction must retain a source-face association.",
                )
            )
            continue
        if np.any(occurrence_ids[[first, second]] != occurrence_ids[vertex]):
            findings.append(
                BRepValidationFinding(
                    "chart-restriction",
                    None,
                    1.0,
                    0.0,
                    "An exact chart restriction cannot cross occurrence ownership.",
                )
            )
            continue
        face = int(source_indices[vertex])
        endpoint_exact = []
        for endpoint_row, endpoint in enumerate((first, second)):
            represented = endpoint_parameters[row, endpoint_row]
            if endpoint in exact_by_vertex:
                exact = exact_by_vertex[endpoint]
            else:
                exact = rational_vector((float(represented[0]), float(represented[1])))
            endpoint_exact.append(exact)
            incident = np.argwhere(
                (faces == endpoint)
                & (face_ids[:, None] == face)
                & (triangle_occurrence_ids[:, None] == occurrence_ids[vertex])
            )
            matched = any(
                (
                    np.array_equal(
                        triangle_parameters[int(cell), int(corner)], represented
                    )
                    for cell, corner in incident
                )
            )
            # A zero-area pole child may retire its UV copy while another
            # pole copy remains incident to this face; the endpoint bank
            # retains the retired UV coordinate.
            if not matched and (source_dimensions[endpoint] != 0 or not incident.size):
                findings.append(
                    BRepValidationFinding(
                        "chart-restriction",
                        None,
                        1.0,
                        0.0,
                        "A chart restriction endpoint is not bound to its face incidence.",
                    )
                )
                continue
        execution = source_parameters[vertex]
        incident = np.argwhere(
            (faces == vertex)
            & (face_ids[:, None] == face)
            & (triangle_occurrence_ids[:, None] == occurrence_ids[vertex])
        )
        if not incident.size or any(
            (
                not np.array_equal(triangle_parameters[int(cell), int(corner)], execution)
                for cell, corner in incident
            )
        ):
            findings.append(
                BRepValidationFinding(
                    "chart-restriction",
                    None,
                    1.0,
                    0.0,
                    "A rounded chart restriction is not bound to every incident face corner.",
                )
            )
            continue
        if len(findings) != starting_findings:
            continue
        restriction = ExactRationalChartRestriction(
            source_revision,
            face,
            vertex,
            (first, second),
            (endpoint_exact[0], endpoint_exact[1]),
            Fraction(numerator, denominator),
            execution,
            root_coefficients=(-numerator, denominator),
            isolation=(Fraction(0), Fraction(1)),
        )
        exact_by_vertex[vertex] = restriction.exact_coordinate
        identifiers.append(restriction.restriction_id)
    return tuple(identifiers)


class _ModelCarrierPhase(StrictModule):
    """Validated host arrays from one model construction lifecycle phase."""

    bounds: np.ndarray
    orientation_: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    face_ids: np.ndarray
    parameters: np.ndarray


class _ModelLineagePhase(StrictModule):
    """Validated host arrays from one model construction lifecycle phase."""

    deviation: np.ndarray
    normal: np.ndarray
    source_dimensions: np.ndarray
    source_indices: np.ndarray
    source_parameters: np.ndarray


class _ModelCorrespondencePhase(StrictModule):
    """Validated host arrays from one model construction lifecycle phase."""

    correspondence: np.ndarray


class _ModelOccurrencePhase(StrictModule):
    """Validated host arrays from one model construction lifecycle phase."""

    triangle_instances: np.ndarray
    vertex_instances: np.ndarray


class _ModelRestrictionPhase(StrictModule):
    """Validated host arrays from one model construction lifecycle phase."""

    restriction_vertices: np.ndarray
    restriction_edges: np.ndarray
    restriction_endpoint_parameters: np.ndarray
    restriction_parameters: np.ndarray
    restriction_ids: tuple[str, ...] = eqx.field(static=True)


def _collect_model_faces(
    carrier: _ModelCarrierPhase,
    face_count: int,
    topology: BRepTopology,
    trim_domains: tuple[TrimDomain | None, ...],
    physical_tags: tuple[str, ...],
) -> BRepValidationReport:
    findings: list[BRepValidationFinding] = []
    bounds = carrier.bounds
    orientation_ = carrier.orientation_
    if topology.num_faces != face_count:
        findings.append(
            BRepValidationFinding(
                "topology", None, 1.0, 0.0, "topology must contain one entry per face."
            )
        )
    if bounds.shape != (face_count, 2, 2):
        findings.append(
            BRepValidationFinding(
                "parameter-bounds",
                None,
                1.0,
                0.0,
                "parameter_bounds must contain one 2D box per face.",
            )
        )
    if bounds.shape == (face_count, 2, 2) and not _surface_intervals_valid(bounds):
        findings.append(
            BRepValidationFinding(
                "parameter-bounds",
                None,
                1.0,
                0.0,
                "Every surface parameter interval must be finite and nonempty.",
            )
        )
    if orientation_.shape != (face_count,):
        findings.append(
            BRepValidationFinding(
                "orientation",
                None,
                1.0,
                0.0,
                "orientation must contain one sign per face.",
            )
        )
    if not np.all(np.isin(orientation_, (-1.0, 1.0))):
        findings.append(
            BRepValidationFinding(
                "orientation", None, 1.0, 0.0, "orientation entries must be -1 or 1."
            )
        )
    if len(trim_domains) != face_count or len(physical_tags) != face_count:
        findings.append(
            BRepValidationFinding(
                "face-metadata",
                None,
                1.0,
                0.0,
                "Trim domains and physical tags must align with faces.",
            )
        )
    return BRepValidationReport(findings=tuple(findings))


def _collect_mesh_connectivity(carrier: _ModelCarrierPhase) -> BRepValidationReport:
    findings: list[BRepValidationFinding] = []
    faces = carrier.faces
    vertices = carrier.vertices
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        findings.append(
            BRepValidationFinding(
                "mesh-vertices",
                None,
                1.0,
                0.0,
                "mesh_vertices must have shape (num_vertices, 3).",
            )
        )
    if not np.all(np.isfinite(vertices)):
        findings.append(
            BRepValidationFinding(
                "mesh-vertices", None, 1.0, 0.0, "mesh_vertices must be finite."
            )
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        findings.append(
            BRepValidationFinding(
                "mesh-faces",
                None,
                1.0,
                0.0,
                "mesh_faces must have shape (num_triangles, 3).",
            )
        )
    if faces.size and (
        np.min(faces) < 0 or np.max(faces) >= (vertices.shape[0] if vertices.ndim else 0)
    ):
        findings.append(
            BRepValidationFinding(
                "mesh-faces",
                None,
                1.0,
                0.0,
                "mesh_faces reference an absent mesh vertex.",
            )
        )
    return BRepValidationReport(findings=tuple(findings))


def _collect_mesh_parameters(
    carrier: _ModelCarrierPhase, face_count: int
) -> BRepValidationReport:
    findings: list[BRepValidationFinding] = []
    face_ids = carrier.face_ids
    faces = carrier.faces
    parameters = carrier.parameters
    if face_ids.shape != (faces.shape[0] if faces.ndim else 0,):
        findings.append(
            BRepValidationFinding(
                "triangle-face-ids",
                None,
                1.0,
                0.0,
                "triangle_face_ids must align with mesh faces.",
            )
        )
    if face_ids.size and (np.min(face_ids) < 0 or np.max(face_ids) >= face_count):
        findings.append(
            BRepValidationFinding(
                "triangle-face-ids",
                None,
                1.0,
                0.0,
                "triangle_face_ids reference an absent B-Rep face.",
            )
        )
    if parameters.shape != (faces.shape[0] if faces.ndim else 0, 3, 2):
        findings.append(
            BRepValidationFinding(
                "triangle-parameters",
                None,
                1.0,
                0.0,
                "triangle_parameters must have shape (num_triangles, 3, 2).",
            )
        )
    if not np.all(np.isfinite(parameters)):
        findings.append(
            BRepValidationFinding(
                "triangle-parameters",
                None,
                1.0,
                0.0,
                "triangle_parameters must be finite.",
            )
        )
    return BRepValidationReport(findings=tuple(findings))


def _prepare_model_carrier(
    coordinate_contract: SpatialCoordinateContract,
    mesh_faces: ArrayLike,
    mesh_vertices: ArrayLike,
    orientation: ArrayLike,
    parameter_bounds: ArrayLike,
    patches: tuple[AbstractSurfacePatch, ...],
    physical_tags: tuple[str, ...],
    report: BRepImportReport,
    topology: BRepTopology,
    triangle_face_ids: ArrayLike,
    triangle_parameters: ArrayLike,
    trim_domains: tuple[TrimDomain | None, ...],
    findings: list[BRepValidationFinding],
) -> _ModelCarrierPhase:
    if coordinate_contract.spatial_id != report.coordinate_contract.spatial_id:
        findings.append(
            BRepValidationFinding(
                "coordinate-contract",
                None,
                1.0,
                0.0,
                "The model and import report coordinate contracts must match.",
            )
        )
    face_count = len(patches)
    bounds = np.asarray(parameter_bounds, dtype=np.float64)
    orientation_ = np.asarray(orientation, dtype=np.float64).reshape((-1,))
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int32)
    face_ids = np.asarray(triangle_face_ids, dtype=np.int32).reshape((-1,))
    parameters = np.asarray(triangle_parameters, dtype=np.float64)
    carrier = _ModelCarrierPhase(
        bounds=bounds,
        orientation_=orientation_,
        vertices=vertices,
        faces=faces,
        face_ids=face_ids,
        parameters=parameters,
    )
    findings.extend(
        _collect_model_faces(
            carrier, face_count, topology, trim_domains, physical_tags
        ).findings
    )
    findings.extend(_collect_mesh_connectivity(carrier).findings)
    findings.extend(_collect_mesh_parameters(carrier, face_count).findings)
    if (
        bounds.shape != (face_count, 2, 2)
        or orientation_.shape != (face_count,)
        or vertices.ndim != 2
        or (vertices.shape[-1] != 3)
        or (faces.ndim != 2)
        or (faces.shape[-1] != 3)
        or (face_ids.shape != (faces.shape[0],))
        or (parameters.shape != (faces.shape[0], 3, 2))
    ):
        BRepValidationReport(findings=tuple(findings)).require_valid()
    return carrier


def _prepare_model_lineage(
    faces: np.ndarray,
    mesh_vertex_parameters: ArrayLike | None,
    mesh_vertex_source_dimensions: ArrayLike | None,
    mesh_vertex_source_indices: ArrayLike | None,
    tessellation_deviation_bounds: ArrayLike | None,
    tessellation_normal_bounds: ArrayLike | None,
    topology: BRepTopology,
    vertices: np.ndarray,
    findings: list[BRepValidationFinding],
) -> _ModelLineagePhase:
    deviation = (
        np.full((faces.shape[0],), np.inf, dtype=np.float64)
        if tessellation_deviation_bounds is None
        else np.asarray(tessellation_deviation_bounds, dtype=np.float64)
    )
    normal = (
        np.full((faces.shape[0],), np.inf, dtype=np.float64)
        if tessellation_normal_bounds is None
        else np.asarray(tessellation_normal_bounds, dtype=np.float64)
    )
    for name, evidence in (
        ("tessellation_deviation_bounds", deviation),
        ("tessellation_normal_bounds", normal),
    ):
        host = np.asarray(evidence)
        if (
            host.shape != (faces.shape[0],)
            or np.any(np.isnan(host))
            or np.any(host < 0.0)
        ):
            findings.append(
                BRepValidationFinding(
                    "tessellation-bounds",
                    None,
                    1.0,
                    0.0,
                    f"{name} must be nonnegative triangle-aligned bounds; infinity means unresolved.",
                )
            )
    vertex_count = vertices.shape[0]
    for name, value in (
        ("mesh_vertex_source_dimensions", mesh_vertex_source_dimensions),
        ("mesh_vertex_source_indices", mesh_vertex_source_indices),
    ):
        if value is not None and (not np.issubdtype(np.asarray(value).dtype, np.integer)):
            raise TypeError(
                f"{'BRepModel.' + name} must contain integer entity metadata; got dtype={np.asarray(value).dtype}"
            )
    source_dimensions = (
        np.full((vertex_count,), -1, dtype=np.int32)
        if mesh_vertex_source_dimensions is None
        else np.asarray(mesh_vertex_source_dimensions, dtype=np.int32)
    )
    source_indices = (
        np.full((vertex_count,), -1, dtype=np.int64)
        if mesh_vertex_source_indices is None
        else np.asarray(mesh_vertex_source_indices, dtype=np.int64)
    )
    source_parameters = (
        np.full((vertex_count, 2), np.nan, dtype=np.float64)
        if mesh_vertex_parameters is None
        else np.asarray(mesh_vertex_parameters, dtype=np.float64)
    )
    dimensions_host, indices_host, source_parameters_host = (
        np.asarray(source_dimensions),
        np.asarray(source_indices),
        np.asarray(source_parameters),
    )
    if (
        source_dimensions.shape != (vertex_count,)
        or source_indices.shape != (vertex_count,)
        or source_parameters.shape != (vertex_count, 2)
    ):
        findings.append(
            BRepValidationFinding(
                "vertex-lineage",
                None,
                1.0,
                0.0,
                "Tessellation vertex lineage must align with mesh vertices.",
            )
        )
        BRepValidationReport(findings=tuple(findings)).require_valid()
    if not np.all(np.isin(dimensions_host, (-1, 0, 1, 2))):
        findings.append(
            BRepValidationFinding(
                "vertex-lineage",
                None,
                1.0,
                0.0,
                "Tessellation vertex source dimensions must be unknown, vertex, edge, or face.",
            )
        )
    if np.any(indices_host[dimensions_host == -1] != -1):
        findings.append(
            BRepValidationFinding(
                "vertex-lineage",
                None,
                1.0,
                0.0,
                "An unknown tessellation source dimension requires an unknown source index.",
            )
        )
    for dimension, capacity in enumerate(
        (topology.num_vertices, topology.num_edges, topology.num_faces)
    ):
        selected = dimensions_host == dimension
        if np.any(indices_host[selected] < 0) or np.any(
            indices_host[selected] >= capacity
        ):
            findings.append(
                BRepValidationFinding(
                    "vertex-lineage",
                    None,
                    1.0,
                    0.0,
                    "Tessellation vertex lineage references an absent exact entity.",
                )
            )
        if dimension > 0 and (
            not np.all(np.isfinite(source_parameters_host[selected, :dimension]))
        ):
            findings.append(
                BRepValidationFinding(
                    "vertex-lineage",
                    None,
                    1.0,
                    0.0,
                    "Curve/face tessellation lineage requires finite native parameters.",
                )
            )
    return _ModelLineagePhase(
        deviation=deviation,
        normal=normal,
        source_dimensions=source_dimensions,
        source_indices=source_indices,
        source_parameters=source_parameters,
    )


def _prepare_model_correspondence(
    bounds_host: np.ndarray,
    budget: CoordinateEnclosureBudget,
    coedge_deviation_bounds: ArrayLike | None,
    face_count: int,
    faces: np.ndarray,
    geometry: BRepGeometry | None,
    patches: tuple[AbstractSurfacePatch, ...],
    preparations: TrimRequestPreparations | None,
    report: BRepImportReport,
    topology: BRepTopology,
    findings: list[BRepValidationFinding],
) -> _ModelCorrespondencePhase:
    expected_report_counts = (
        topology.num_solids,
        face_count,
        topology.num_edges,
        topology.num_vertices,
        faces.shape[0],
    )
    report_counts = (
        report.num_solids,
        report.num_faces,
        report.num_edges,
        report.num_vertices,
        report.num_triangles,
    )
    if report_counts != expected_report_counts:
        findings.append(
            BRepValidationFinding(
                "report-counts",
                None,
                1.0,
                0.0,
                "The import report entity counts must match the B-Rep model.",
            )
        )
    correspondence = (
        np.empty((0,), dtype=np.float64)
        if geometry is None or findings
        else _validate_exact_geometry(
            geometry,
            patches,
            bounds_host,
            topology,
            report.curve_surface_tolerance,
            report.source_revision,
            budget=budget,
            preparations=preparations,
            findings=findings,
        )
    )
    if coedge_deviation_bounds is not None and (not findings):
        retained = np.asarray(coedge_deviation_bounds, dtype=np.float64)
        if (
            retained.shape != correspondence.shape
            or np.any(np.isnan(retained))
            or np.any(retained < correspondence)
        ):
            findings.append(
                BRepValidationFinding(
                    "tessellation-bounds",
                    None,
                    1.0,
                    0.0,
                    "Retained coedge bounds must contain the independently recomputed source bounds.",
                )
            )
        correspondence = retained
    return _ModelCorrespondencePhase(correspondence=correspondence)


def _prepare_model_occurrences(
    face_ids_host: np.ndarray,
    faces: np.ndarray,
    faces_host: np.ndarray,
    geometry: BRepGeometry | None,
    topology: BRepTopology,
    triangle_occurrence_ids: ArrayLike | None,
    vertex_count: int,
    vertex_occurrence_ids: ArrayLike | None,
    findings: list[BRepValidationFinding],
) -> _ModelOccurrencePhase:
    for name, value in (
        ("triangle_occurrence_ids", triangle_occurrence_ids),
        ("vertex_occurrence_ids", vertex_occurrence_ids),
    ):
        if value is not None and (not np.issubdtype(np.asarray(value).dtype, np.integer)):
            raise TypeError(
                f"{'BRepModel.' + name} must contain integer occurrence-table indices; got dtype={np.asarray(value).dtype}"
            )
    triangle_instances = (
        np.full((faces.shape[0],), -1, dtype=np.int32)
        if triangle_occurrence_ids is None
        else np.asarray(triangle_occurrence_ids, dtype=np.int32)
    )
    vertex_instances = (
        np.full((vertex_count,), -1, dtype=np.int32)
        if vertex_occurrence_ids is None
        else np.asarray(vertex_occurrence_ids, dtype=np.int32)
    )
    triangle_instances_host, vertex_instances_host = (
        np.asarray(triangle_instances),
        np.asarray(vertex_instances),
    )
    occurrence_count = 0 if geometry is None else len(geometry.occurrences)
    if triangle_instances.shape != (faces.shape[0],) or vertex_instances.shape != (
        vertex_count,
    ):
        findings.append(
            BRepValidationFinding(
                "occurrence-lineage",
                None,
                1.0,
                0.0,
                "Derived occurrence IDs must align with triangles and vertices.",
            )
        )
        BRepValidationReport(findings=tuple(findings)).require_valid()
    if (
        np.any(triangle_instances_host < -1)
        or np.any(triangle_instances_host >= occurrence_count)
        or np.any(vertex_instances_host < -1)
        or np.any(vertex_instances_host >= occurrence_count)
    ):
        findings.append(
            BRepValidationFinding(
                "occurrence-lineage",
                None,
                1.0,
                0.0,
                "BRepModel.occurrence_ids must reference the occurrence table or unplaced definition; got an absent occurrence.",
            )
        )
    if (
        faces_host.size
        and np.all((faces_host >= 0) & (faces_host < vertex_count))
        and not np.all(
            vertex_instances_host[faces_host] == triangle_instances_host[:, None]
        )
    ):
        findings.append(
            BRepValidationFinding(
                "occurrence-lineage",
                None,
                1.0,
                0.0,
                "A derived triangle cannot weld independent occurrence vertices.",
            )
        )
    if geometry is not None and len(topology.solid_faces) == len(geometry.solid_shells):
        for occurrence_index, occurrence in enumerate(geometry.occurrences):
            selected = triangle_instances_host == occurrence_index
            if not np.all(
                np.isin(face_ids_host[selected], topology.solid_faces[occurrence.solid])
            ):
                findings.append(
                    BRepValidationFinding(
                        "occurrence-lineage",
                        None,
                        1.0,
                        0.0,
                        "A placed triangle must belong to its declared solid occurrence.",
                    )
                )
    return _ModelOccurrencePhase(
        triangle_instances=triangle_instances, vertex_instances=vertex_instances
    )


def _prepare_model_restrictions(
    dimensions_host: np.ndarray,
    face_ids_host: np.ndarray,
    faces_host: np.ndarray,
    indices_host: np.ndarray,
    mesh_chart_restriction_edges: ArrayLike | None,
    mesh_chart_restriction_endpoint_parameters: ArrayLike | None,
    mesh_chart_restriction_parameters: ArrayLike | None,
    mesh_chart_restriction_vertices: ArrayLike | None,
    parameters_host: np.ndarray,
    report: BRepImportReport,
    source_parameters_host: np.ndarray,
    triangle_instances_host: np.ndarray,
    vertex_instances_host: np.ndarray,
) -> _ModelRestrictionPhase:
    findings: list[BRepValidationFinding] = []
    restriction_count = (
        0
        if mesh_chart_restriction_vertices is None
        else np.asarray(mesh_chart_restriction_vertices).size
    )
    for name, value in (
        ("mesh_chart_restriction_vertices", mesh_chart_restriction_vertices),
        ("mesh_chart_restriction_edges", mesh_chart_restriction_edges),
        ("mesh_chart_restriction_parameters", mesh_chart_restriction_parameters),
    ):
        if value is not None and not np.issubdtype(np.asarray(value).dtype, np.integer):
            raise TypeError(
                f"{'BRepModel.' + name} must contain integer metadata; got dtype={np.asarray(value).dtype}"
            )
    restriction_vertices = np.asarray(
        np.empty((0,), dtype=np.int64)
        if mesh_chart_restriction_vertices is None
        else mesh_chart_restriction_vertices,
        dtype=np.int64,
    )
    restriction_edges = np.asarray(
        np.empty((restriction_count, 2), dtype=np.int64)
        if mesh_chart_restriction_edges is None
        else mesh_chart_restriction_edges,
        dtype=np.int64,
    )
    restriction_endpoint_parameters = np.asarray(
        np.empty((restriction_count, 2, 2), dtype=np.float64)
        if mesh_chart_restriction_endpoint_parameters is None
        else mesh_chart_restriction_endpoint_parameters,
        dtype=np.float64,
    )
    restriction_parameters = np.asarray(
        np.empty((restriction_count, 2), dtype=np.int64)
        if mesh_chart_restriction_parameters is None
        else mesh_chart_restriction_parameters,
        dtype=np.int64,
    )
    restriction_ids = _validate_tessellation_chart_restrictions(
        report.source_revision,
        np.asarray(restriction_vertices),
        np.asarray(restriction_edges),
        np.asarray(restriction_endpoint_parameters),
        np.asarray(restriction_parameters),
        faces_host,
        face_ids_host,
        parameters_host,
        dimensions_host,
        indices_host,
        source_parameters_host,
        vertex_instances_host,
        triangle_instances_host,
        findings=findings,
    )
    BRepValidationReport(findings=tuple(findings)).require_valid()
    return _ModelRestrictionPhase(
        restriction_vertices=restriction_vertices,
        restriction_edges=restriction_edges,
        restriction_endpoint_parameters=restriction_endpoint_parameters,
        restriction_parameters=restriction_parameters,
        restriction_ids=restriction_ids,
    )


class _ModelChanges(TypedDict, total=False):
    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: ArrayLike
    orientation: ArrayLike
    trim_domains: tuple[TrimDomain | None, ...]
    topology: BRepTopology
    coordinate_contract: SpatialCoordinateContract
    mesh_vertices: ArrayLike
    mesh_faces: ArrayLike
    triangle_face_ids: ArrayLike
    triangle_parameters: ArrayLike
    physical_tags: tuple[str, ...]
    report: BRepImportReport
    geometry: BRepGeometry | None
    tessellation_deviation_bounds: ArrayLike | None
    tessellation_normal_bounds: ArrayLike | None
    mesh_vertex_source_dimensions: ArrayLike | None
    mesh_vertex_source_indices: ArrayLike | None
    mesh_vertex_parameters: ArrayLike | None
    mesh_chart_restriction_vertices: ArrayLike | None
    mesh_chart_restriction_edges: ArrayLike | None
    mesh_chart_restriction_endpoint_parameters: ArrayLike | None
    mesh_chart_restriction_parameters: ArrayLike | None
    coedge_deviation_bounds: ArrayLike | None
    triangle_occurrence_ids: ArrayLike | None
    vertex_occurrence_ids: ArrayLike | None


class BRepModel(StrictModule):
    """Represented B-Rep with a separately identified query tessellation.

    ``model_id`` identifies the represented geometry and topology
    (patches, parameter domains, orientation, incidence and, when present, the
    ``geometry`` of curves, p-curves, loops, shells and occurrences).
    ``tessellation_id`` additionally identifies derived mesh/UV lineage,
    continuous deviation and normal-turn bounds, trim chord covers, and
    rational chart restrictions for vertices whose source UV
    coordinate has no binary64 representative. Rounded UV rows remain execution
    values only when that source-root evidence is present. Changing resolution
    never changes ``model_id`` or ``source_revision``. Infinite tessellation
    bounds and unknown vertex lineage explicitly mark an external comparison
    artifact without native geometric qualification.
    """

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array
    orientation: Array
    trim_domains: tuple[TrimDomain | None, ...]
    topology: BRepTopology
    geometry: BRepGeometry | None
    mesh_vertices: Array
    mesh_faces: Array
    triangle_face_ids: Array
    triangle_parameters: Array
    tessellation_deviation_bounds: Array
    tessellation_normal_bounds: Array
    mesh_vertex_source_dimensions: Array
    mesh_vertex_source_indices: Array
    mesh_vertex_parameters: Array
    mesh_chart_restriction_vertices: Array
    mesh_chart_restriction_edges: Array
    mesh_chart_restriction_endpoint_parameters: Array
    mesh_chart_restriction_parameters: Array
    coedge_deviation_bounds: Array
    triangle_occurrence_ids: Array
    vertex_occurrence_ids: Array
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    report: BRepImportReport = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    tessellation_id: str = eqx.field(static=True)
    chart_restriction_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: ArrayLike,
        orientation: ArrayLike,
        trim_domains: tuple[TrimDomain | None, ...],
        topology: BRepTopology,
        coordinate_contract: SpatialCoordinateContract,
        mesh_vertices: ArrayLike,
        mesh_faces: ArrayLike,
        triangle_face_ids: ArrayLike,
        triangle_parameters: ArrayLike,
        physical_tags: tuple[str, ...],
        report: BRepImportReport,
        geometry: BRepGeometry | None = None,
        tessellation_deviation_bounds: ArrayLike | None = None,
        tessellation_normal_bounds: ArrayLike | None = None,
        mesh_vertex_source_dimensions: ArrayLike | None = None,
        mesh_vertex_source_indices: ArrayLike | None = None,
        mesh_vertex_parameters: ArrayLike | None = None,
        mesh_chart_restriction_vertices: ArrayLike | None = None,
        mesh_chart_restriction_edges: ArrayLike | None = None,
        mesh_chart_restriction_endpoint_parameters: ArrayLike | None = None,
        mesh_chart_restriction_parameters: ArrayLike | None = None,
        coedge_deviation_bounds: ArrayLike | None = None,
        triangle_occurrence_ids: ArrayLike | None = None,
        vertex_occurrence_ids: ArrayLike | None = None,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                patches,
                parameter_bounds,
                orientation,
                trim_domains,
                topology,
                coordinate_contract,
                mesh_vertices,
                mesh_faces,
                triangle_face_ids,
                triangle_parameters,
                physical_tags,
                report,
                geometry,
                tessellation_deviation_bounds,
                tessellation_normal_bounds,
                mesh_vertex_source_dimensions,
                mesh_vertex_source_indices,
                mesh_vertex_parameters,
                mesh_chart_restriction_vertices,
                mesh_chart_restriction_edges,
                mesh_chart_restriction_endpoint_parameters,
                mesh_chart_restriction_parameters,
                coedge_deviation_bounds,
                triangle_occurrence_ids,
                vertex_occurrence_ids,
            ),
            budget=budget,
        ) as budget:
            findings: list[BRepValidationFinding] = []
            carrier = _prepare_model_carrier(
                coordinate_contract=coordinate_contract,
                mesh_faces=mesh_faces,
                mesh_vertices=mesh_vertices,
                orientation=orientation,
                parameter_bounds=parameter_bounds,
                patches=patches,
                physical_tags=physical_tags,
                report=report,
                topology=topology,
                triangle_face_ids=triangle_face_ids,
                triangle_parameters=triangle_parameters,
                trim_domains=trim_domains,
                findings=findings,
            )
            bounds = carrier.bounds
            vertices = carrier.vertices
            faces = carrier.faces
            face_ids = carrier.face_ids
            parameters = carrier.parameters
            face_count = len(patches)
            vertex_count = vertices.shape[0]
            bounds_host, faces_host, face_ids_host, parameters_host = (
                bounds,
                faces,
                face_ids,
                parameters,
            )
            lineage = _prepare_model_lineage(
                faces=faces,
                mesh_vertex_parameters=mesh_vertex_parameters,
                mesh_vertex_source_dimensions=mesh_vertex_source_dimensions,
                mesh_vertex_source_indices=mesh_vertex_source_indices,
                tessellation_deviation_bounds=tessellation_deviation_bounds,
                tessellation_normal_bounds=tessellation_normal_bounds,
                topology=topology,
                vertices=vertices,
                findings=findings,
            )
            source_dimensions = lineage.source_dimensions
            source_indices = lineage.source_indices
            source_parameters = lineage.source_parameters
            dimensions_host, indices_host, source_parameters_host = (
                source_dimensions,
                source_indices,
                source_parameters,
            )
            correspondence_phase = _prepare_model_correspondence(
                bounds_host=bounds_host,
                budget=budget,
                coedge_deviation_bounds=coedge_deviation_bounds,
                face_count=face_count,
                faces=faces,
                geometry=geometry,
                patches=patches,
                preparations=preparations,
                report=report,
                topology=topology,
                findings=findings,
            )
            occurrences = _prepare_model_occurrences(
                face_ids_host=face_ids_host,
                faces=faces,
                faces_host=faces_host,
                geometry=geometry,
                topology=topology,
                triangle_occurrence_ids=triangle_occurrence_ids,
                vertex_count=vertex_count,
                vertex_occurrence_ids=vertex_occurrence_ids,
                findings=findings,
            )
            triangle_instances = occurrences.triangle_instances
            vertex_instances = occurrences.vertex_instances
            triangle_instances_host, vertex_instances_host = (
                triangle_instances,
                vertex_instances,
            )
            BRepValidationReport(findings=tuple(findings)).require_valid()
            restrictions = _prepare_model_restrictions(
                dimensions_host=dimensions_host,
                face_ids_host=face_ids_host,
                faces_host=faces_host,
                indices_host=indices_host,
                mesh_chart_restriction_edges=mesh_chart_restriction_edges,
                mesh_chart_restriction_endpoint_parameters=mesh_chart_restriction_endpoint_parameters,
                mesh_chart_restriction_parameters=mesh_chart_restriction_parameters,
                mesh_chart_restriction_vertices=mesh_chart_restriction_vertices,
                parameters_host=parameters_host,
                report=report,
                source_parameters_host=source_parameters_host,
                triangle_instances_host=triangle_instances_host,
                vertex_instances_host=vertex_instances_host,
            )
            identity = self._commit_model(
                patches,
                trim_domains,
                topology,
                geometry,
                physical_tags,
                report,
                carrier,
                lineage,
                correspondence_phase,
                occurrences,
                restrictions,
            )
            self.model_id = identity.model_id
            self.tessellation_id = identity.tessellation_id

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: BRepModel, /) -> Self:
        """Reconstruct stored inputs through the validated construction phases.

        The structure-recipe boundary admits the registered type and field set.
        This owner requires its node type before revalidating constructor inputs;
        derived incidence, bounds, and identities are recomputed by their phases.
        """
        if type(value) is not cls:
            raise TypeError(
                f"BRepModel.stored_fields must have the declared node type; got {type(value).__name__}"
            )
        return cls(
            patches=value.patches,
            parameter_bounds=value.parameter_bounds,
            orientation=value.orientation,
            trim_domains=value.trim_domains,
            topology=value.topology,
            coordinate_contract=value.report.coordinate_contract,
            mesh_vertices=value.mesh_vertices,
            mesh_faces=value.mesh_faces,
            triangle_face_ids=value.triangle_face_ids,
            triangle_parameters=value.triangle_parameters,
            physical_tags=value.physical_tags,
            report=value.report,
            geometry=value.geometry,
            tessellation_deviation_bounds=value.tessellation_deviation_bounds,
            tessellation_normal_bounds=value.tessellation_normal_bounds,
            mesh_vertex_source_dimensions=value.mesh_vertex_source_dimensions,
            mesh_vertex_source_indices=value.mesh_vertex_source_indices,
            mesh_vertex_parameters=value.mesh_vertex_parameters,
            mesh_chart_restriction_vertices=value.mesh_chart_restriction_vertices,
            mesh_chart_restriction_edges=value.mesh_chart_restriction_edges,
            mesh_chart_restriction_endpoint_parameters=value.mesh_chart_restriction_endpoint_parameters,
            mesh_chart_restriction_parameters=value.mesh_chart_restriction_parameters,
            coedge_deviation_bounds=value.coedge_deviation_bounds,
            triangle_occurrence_ids=value.triangle_occurrence_ids,
            vertex_occurrence_ids=value.vertex_occurrence_ids,
        )

    def _commit_model(
        self,
        patches: tuple[AbstractSurfacePatch, ...],
        trim_domains: tuple[TrimDomain | None, ...],
        topology: BRepTopology,
        geometry: BRepGeometry | None,
        physical_tags: tuple[str, ...],
        report: BRepImportReport,
        carrier: _ModelCarrierPhase,
        lineage: _ModelLineagePhase,
        correspondence_phase: _ModelCorrespondencePhase,
        occurrences: _ModelOccurrencePhase,
        restrictions: _ModelRestrictionPhase,
    ) -> _ModelIdentityPhase:
        """Commit validated host phases to numerical leaves and identity payloads."""
        self.patches = patches
        self.parameter_bounds = jnp.asarray(carrier.bounds)
        self.orientation = jnp.asarray(carrier.orientation_)
        self.trim_domains = trim_domains
        self.topology = topology
        self.geometry = geometry
        self.mesh_vertices = jnp.asarray(carrier.vertices)
        self.mesh_faces = jnp.asarray(carrier.faces)
        self.triangle_face_ids = jnp.asarray(carrier.face_ids)
        self.triangle_parameters = jnp.asarray(carrier.parameters)
        self.tessellation_deviation_bounds = jnp.asarray(lineage.deviation)
        self.tessellation_normal_bounds = jnp.asarray(lineage.normal)
        self.mesh_vertex_source_dimensions = jnp.asarray(lineage.source_dimensions)
        self.mesh_vertex_source_indices = jnp.asarray(lineage.source_indices)
        self.mesh_vertex_parameters = jnp.asarray(lineage.source_parameters)
        self.mesh_chart_restriction_vertices = jnp.asarray(
            restrictions.restriction_vertices
        )
        self.mesh_chart_restriction_edges = jnp.asarray(restrictions.restriction_edges)
        self.mesh_chart_restriction_endpoint_parameters = jnp.asarray(
            restrictions.restriction_endpoint_parameters
        )
        self.mesh_chart_restriction_parameters = jnp.asarray(
            restrictions.restriction_parameters
        )
        self.coedge_deviation_bounds = jnp.asarray(
            correspondence_phase.correspondence, dtype=jnp.float64
        )
        self.triangle_occurrence_ids = jnp.asarray(occurrences.triangle_instances)
        self.vertex_occurrence_ids = jnp.asarray(occurrences.vertex_instances)
        self.physical_tags = physical_tags
        self.report = report
        self.chart_restriction_ids = restrictions.restriction_ids
        identity = self._prepare_identity()
        return identity

    def require_geometry(self) -> BRepGeometry:
        """Return the curve/trim geometry or refuse an external comparison model."""
        geometry = self.geometry
        if geometry is None:
            raise ValueError("BRepModel.geometry must provide curves and trims; got None")
        return geometry

    def evolve(
        self,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        **changes: Unpack[_ModelChanges],
    ) -> BRepModel:
        """Construct a changed model through the same validation and identity phases."""
        return BRepModel(
            patches=changes.get("patches", self.patches),
            parameter_bounds=changes.get("parameter_bounds", self.parameter_bounds),
            orientation=changes.get("orientation", self.orientation),
            trim_domains=changes.get("trim_domains", self.trim_domains),
            topology=changes.get("topology", self.topology),
            coordinate_contract=changes.get(
                "coordinate_contract", self.coordinate_contract
            ),
            mesh_vertices=changes.get("mesh_vertices", self.mesh_vertices),
            mesh_faces=changes.get("mesh_faces", self.mesh_faces),
            triangle_face_ids=changes.get("triangle_face_ids", self.triangle_face_ids),
            triangle_parameters=changes.get(
                "triangle_parameters", self.triangle_parameters
            ),
            physical_tags=changes.get("physical_tags", self.physical_tags),
            report=changes.get("report", self.report),
            geometry=changes.get("geometry", self.geometry),
            tessellation_deviation_bounds=changes.get(
                "tessellation_deviation_bounds", self.tessellation_deviation_bounds
            ),
            tessellation_normal_bounds=changes.get(
                "tessellation_normal_bounds", self.tessellation_normal_bounds
            ),
            mesh_vertex_source_dimensions=changes.get(
                "mesh_vertex_source_dimensions", self.mesh_vertex_source_dimensions
            ),
            mesh_vertex_source_indices=changes.get(
                "mesh_vertex_source_indices", self.mesh_vertex_source_indices
            ),
            mesh_vertex_parameters=changes.get(
                "mesh_vertex_parameters", self.mesh_vertex_parameters
            ),
            mesh_chart_restriction_vertices=changes.get(
                "mesh_chart_restriction_vertices", self.mesh_chart_restriction_vertices
            ),
            mesh_chart_restriction_edges=changes.get(
                "mesh_chart_restriction_edges", self.mesh_chart_restriction_edges
            ),
            mesh_chart_restriction_endpoint_parameters=changes.get(
                "mesh_chart_restriction_endpoint_parameters",
                self.mesh_chart_restriction_endpoint_parameters,
            ),
            mesh_chart_restriction_parameters=changes.get(
                "mesh_chart_restriction_parameters",
                self.mesh_chart_restriction_parameters,
            ),
            coedge_deviation_bounds=changes.get(
                "coedge_deviation_bounds", self.coedge_deviation_bounds
            ),
            triangle_occurrence_ids=changes.get(
                "triangle_occurrence_ids", self.triangle_occurrence_ids
            ),
            vertex_occurrence_ids=changes.get(
                "vertex_occurrence_ids", self.vertex_occurrence_ids
            ),
            budget=budget,
            preparations=preparations,
        )

    def certified_coedge_bounds(
        self,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        """Recompute spatial curve/trim bounds without retained tessellation limits."""
        geometry = self.require_geometry()
        preparations = TrimRequestPreparations() if preparations is None else preparations
        findings: list[BRepValidationFinding] = []
        with coordinate_source_request(
            (self,), budget=budget, rows=len(geometry.coedge_edges)
        ) as admitted:
            bounds = _validate_exact_geometry(
                geometry,
                self.patches,
                np.asarray(self.parameter_bounds),
                self.topology,
                self.report.curve_surface_tolerance,
                self.source_revision,
                budget=admitted,
                preparations=preparations,
                findings=findings,
            )
            BRepValidationReport(findings=tuple(findings)).require_valid()
            return bounds

    def _prepare_identity(self) -> _ModelIdentityPhase:
        patches = self.patches
        bounds = self.parameter_bounds
        orientation_ = self.orientation
        trim_domains = self.trim_domains
        topology = self.topology
        geometry = self.geometry
        physical_tags = self.physical_tags
        report = self.report
        vertices = self.mesh_vertices
        faces = self.mesh_faces
        face_ids = self.triangle_face_ids
        parameters = self.triangle_parameters
        deviation = self.tessellation_deviation_bounds
        normal = self.tessellation_normal_bounds
        source_dimensions = self.mesh_vertex_source_dimensions
        source_indices = self.mesh_vertex_source_indices
        source_parameters = self.mesh_vertex_parameters
        restriction_vertices = self.mesh_chart_restriction_vertices
        restriction_edges = self.mesh_chart_restriction_edges
        restriction_endpoint_parameters = self.mesh_chart_restriction_endpoint_parameters
        restriction_parameters = self.mesh_chart_restriction_parameters
        restriction_ids = self.chart_restriction_ids
        triangle_instances = self.triangle_occurrence_ids
        vertex_instances = self.vertex_occurrence_ids
        correspondence = self.coedge_deviation_bounds
        model_id = canonical_fingerprint(
            {
                "kind": "brep-model",
                "source_revision": report.source_revision,
                "patches": [carrier_owner.carrier_payload(patch) for patch in patches],
                "parameter_bounds": bounds,
                "orientation": orientation_,
                "topology_representation": {
                    "face_edges": topology.face_edges,
                    "edge_faces": topology.edge_faces,
                    "face_wires": topology.face_wires,
                    "solid_faces": topology.solid_faces,
                    "solid_face_orientations": topology.solid_face_orientations,
                    "num_vertices": topology.num_vertices,
                },
                "exact_geometry": None if geometry is None else geometry.geometry_id,
                "physical_tags": physical_tags,
            }
        )
        tessellation_id = canonical_fingerprint(
            {
                "kind": "brep-query-tessellation",
                "model_id": model_id,
                "import_policy_id": report.import_policy_id,
                "linear_deflection": report.linear_deflection,
                "angular_deflection": report.angular_deflection,
                "trim_samples_per_edge": report.trim_samples_per_edge,
                "trim_domains": tuple(
                    None if domain is None else domain.trim_id for domain in trim_domains
                ),
                "mesh_vertices": vertices,
                "mesh_faces": faces,
                "triangle_face_ids": face_ids,
                "triangle_parameters": parameters,
                "deviation_bounds": deviation,
                "normal_bounds": normal,
                "vertex_source_dimensions": source_dimensions,
                "vertex_source_indices": source_indices,
                "vertex_parameters": source_parameters,
                "chart_restriction_vertices": restriction_vertices,
                "chart_restriction_edges": restriction_edges,
                "chart_restriction_endpoint_parameters": (
                    restriction_endpoint_parameters
                ),
                "chart_restriction_parameters": restriction_parameters,
                "chart_restriction_ids": restriction_ids,
                "triangle_occurrence_ids": triangle_instances,
                "vertex_occurrence_ids": vertex_instances,
                "coedge_deviation_bounds": correspondence,
                "curve_surface_tolerance": report.curve_surface_tolerance,
                "curve_surface_scale": report.curve_surface_scale,
            }
        )
        return _ModelIdentityPhase(model_id=model_id, tessellation_id=tessellation_id)

    @property
    def source_id(self) -> str:
        return self.report.source_id

    @property
    def source_digest(self) -> str:
        return self.report.source_digest

    @property
    def source_revision(self) -> str:
        return self.report.source_revision

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.report.coordinate_contract

    @property
    def import_policy_id(self) -> str:
        return self.report.import_policy_id

    @property
    def qualified_face_ids(self) -> tuple[BRepEntityId, ...]:
        """Return physical face IDs sorted by revision, kind, index, and path."""
        if self.geometry is None:
            return self.face_ids
        return tuple(
            sorted(
                {
                    record.member
                    for record in self.geometry.qualified_entity_incidence(
                        self.source_revision
                    )
                    if record.member.kind == "face"
                }
            )
        )

    @property
    def face_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "face", index)
            for index in range(len(self.patches))
        )

    @property
    def edge_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "edge", index)
            for index in range(self.topology.num_edges)
        )

    @property
    def vertex_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "vertex", index)
            for index in range(self.topology.num_vertices)
        )

    @property
    def solid_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "solid", index)
            for index in range(self.topology.num_solids)
        )

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        base = BRepBoundaryMap(self.patches, self.parameter_bounds)
        geometry = self.geometry
        if geometry is None or not geometry.occurrences:
            return BoundaryAtlas(
                base,
                source_entity_ids=jnp.arange(len(self.patches), dtype=jnp.int32),
                source_id=self.source_id,
                physical_tags=self.physical_tags,
                orientation=self.orientation,
                trim_domains=self.trim_domains,
            )
        entries: list[tuple[int, int, BRepOccurrence | None]] = [
            (face, sign, occurrence)
            for occurrence in geometry.occurrences
            for face, sign in zip(
                self.topology.solid_faces[occurrence.solid],
                self.topology.solid_face_orientations[occurrence.solid],
                strict=True,
            )
        ]
        entries.extend(
            (face, 1, None)
            for face, owners in enumerate(self.topology.face_solids)
            if not owners
        )
        face_identity = {
            (record.container.occurrence_path, record.member.index): record.member
            for record in geometry.qualified_entity_incidence(self.source_revision)
            if record.container.kind == "solid" and record.member.kind == "face"
        }

        def face_key(face: int, occurrence: BRepOccurrence | None, /) -> BRepEntityId:
            if occurrence is None:
                return BRepEntityId(self.source_revision, "face", face)
            return face_identity[(occurrence.path, face)]

        sides: dict[BRepEntityId, list[int]] = {}
        for face, sign, occurrence in entries:
            sides.setdefault(face_key(face, occurrence), []).append(sign)
        entries = [
            (face, sign, occurrence)
            for face, sign, occurrence in entries
            if sorted(sides[face_key(face, occurrence)]) != [-1, 1]
        ]
        faces = jnp.asarray([face for face, _, _ in entries], dtype=jnp.int32)
        mapping = BRepPlacedBoundaryMap(
            base,
            faces,
            jnp.asarray(
                [
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                    if occurrence is None
                    else occurrence.rotation
                    for _, _, occurrence in entries
                ],
                dtype=jnp.float64,
            ).reshape((-1, 3, 3)),
            jnp.asarray(
                [
                    (0.0, 0.0, 0.0) if occurrence is None else occurrence.translation
                    for _, _, occurrence in entries
                ],
                dtype=jnp.float64,
            ).reshape((-1, 3)),
            tuple(
                () if occurrence is None else occurrence.path
                for _, _, occurrence in entries
            ),
        )
        qualified = {
            entity: index for index, entity in enumerate(self.qualified_face_ids)
        }
        source_rows = jnp.asarray(
            [qualified[face_key(face, occurrence)] for face, _, occurrence in entries],
            dtype=jnp.int32,
        )
        return BoundaryAtlas(
            mapping,
            source_entity_ids=source_rows,
            source_id=self.source_id,
            physical_tags=tuple(self.physical_tags[face] for face, _, _ in entries),
            orientation=self.orientation[faces]
            * jnp.asarray([sign for _, sign, _ in entries], dtype=jnp.float64),
            trim_domains=tuple(self.trim_domains[face] for face, _, _ in entries),
        )


__all__ = [
    "BRepBoundaryMap",
    "BRepEntityId",
    "BRepGeometry",
    "BRepImportReport",
    "BRepModel",
    "BRepOccurrence",
    "BRepTopology",
    "BRepEntityKind",
    "BRepValidationCheck",
    "BRepValidationError",
    "BRepValidationFinding",
    "BRepValidationReport",
]

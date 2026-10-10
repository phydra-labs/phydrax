#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import assert_never, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...typing import checked, ConvertibleToArray
from .._atlas import AbstractBoundaryMap, BoundaryAtlas, TrimDomain
from .._chart_restriction import ExactRationalChartRestriction
from ._intersection import (
    BranchRootEndpoint,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import encode_geometry, IntersectionCurve, IntersectionPCurve
from ._patches import AbstractCurve, AbstractSurfacePatch
from ._placed import PlacedSurface
from ._root_bindings import BRepPlacedVertex, BRepVertexRoot


type BRepCurve = AbstractCurve | IntersectionCurve
type BRepPCurve = AbstractCurve | IntersectionPCurve


class NativeCurveCapabilities(Protocol):
    """Common exact carrier enclosure contract; evaluation retains family evidence."""

    @property
    def ambient_dimension(self) -> int: ...

    @property
    def parameter_domain(self) -> tuple[float, float] | None: ...

    @property
    def period(self) -> float | None: ...

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]: ...

    def bounding_box(self, first: float, last: float, /) -> np.ndarray: ...


def curve_points(curve: BRepCurve, parameters: Array, /) -> Array:
    """Evaluate an exact carrier without discarding continuation failures."""
    if isinstance(curve, IntersectionCurve):
        result = curve.evaluate(parameters)
        if not curve.fully_certified or not np.all(
            np.isfinite(np.asarray(result.parameter_bound))
        ):
            raise ValueError("An intersection curve evaluation is unresolved.")
        return result.point
    return curve.evaluate(parameters)


@dataclass(frozen=True, slots=True, order=True)
class BRepEntityId:
    """Stable source-revision-scoped identity for one B-Rep entity."""

    source_revision: str
    kind: str
    index: int
    occurrence_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_revision, str):
            raise TypeError("source_revision must be a string.")
        if not self.source_revision:
            raise ValueError("source_revision must be non-empty.")
        if not isinstance(self.kind, str):
            raise TypeError("kind must be a string.")
        if self.kind not in ("solid", "face", "edge", "vertex"):
            raise ValueError("kind must be solid, face, edge, or vertex.")
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("index must be an integer.")
        if self.index < 0:
            raise ValueError("index must be non-negative.")
        if not isinstance(self.occurrence_path, tuple):
            raise TypeError("occurrence_path must be a tuple of names.")
        if any(not isinstance(name, str) or not name for name in self.occurrence_path):
            raise ValueError("Occurrence path components must be nonempty strings.")


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
                "curve_surface_tolerance must be a finite nonnegative physical length."
            )
        if (
            isinstance(self.curve_surface_scale, bool)
            or not np.isfinite(self.curve_surface_scale)
            or self.curve_surface_scale < 0.0
        ):
            raise ValueError(
                "curve_surface_scale must be a finite nonnegative physical length."
            )
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or self.source_digest != self.source_digest.lower()
            or any(
                character not in "0123456789abcdef" for character in self.source_digest
            )
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest.")
        if not isinstance(self.source_format, str) or not self.source_format:
            raise ValueError("source_format must be a non-empty string.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if (
            not isinstance(self.import_policy_id, str)
            or len(self.import_policy_id) != 64
            or self.import_policy_id != self.import_policy_id.lower()
            or any(
                character not in "0123456789abcdef" for character in self.import_policy_id
            )
        ):
            raise ValueError("import_policy_id must be a lowercase SHA-256 digest.")
        counts = (
            self.num_solids,
            self.num_faces,
            self.num_edges,
            self.num_vertices,
            self.num_triangles,
            self.converted_surface_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("B-Rep import counts must be integers.")
        if any(value < 0 for value in counts):
            raise ValueError("B-Rep import counts must be non-negative.")
        if self.num_faces == 0 and (self.num_solids != 0 or self.num_triangles != 0):
            raise ValueError(
                "A B-Rep without surfaces cannot contain solids or surface triangles."
            )
        if self.converted_surface_count > self.num_faces:
            raise ValueError("converted_surface_count cannot exceed num_faces.")
        if (
            not np.isfinite(self.linear_deflection)
            or not np.isfinite(self.angular_deflection)
            or self.linear_deflection <= 0.0
            or self.angular_deflection <= 0.0
        ):
            raise ValueError("Meshing deflections must be finite and positive.")
        if isinstance(self.trim_samples_per_edge, bool) or not isinstance(
            self.trim_samples_per_edge, int
        ):
            raise TypeError("trim_samples_per_edge must be an integer.")
        if self.trim_samples_per_edge < 3:
            raise ValueError("trim_samples_per_edge must be at least three.")
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
            raise ValueError("face_wires must contain one entry per face.")
        if len(solid_faces) != len(solid_face_orientations):
            raise ValueError("solid_face_orientations must contain one entry per solid.")
        if int(num_vertices) < 0:
            raise ValueError("num_vertices must be non-negative.")
        face_solids: list[list[int]] = [[] for _ in range(face_count)]
        for solid_index, (indices, orientations) in enumerate(
            zip(solid_faces, solid_face_orientations, strict=True)
        ):
            if len(indices) != len(orientations):
                raise ValueError(
                    "Each solid's face orientations must align with its faces."
                )
            if len(set(indices)) != len(indices):
                raise ValueError("A solid cannot contain the same face more than once.")
            if any(index < 0 or index >= face_count for index in indices):
                raise ValueError("A solid references an absent global face.")
            if any(orientation not in (-1, 1) for orientation in orientations):
                raise ValueError("Solid face orientations must be -1 or 1.")
            for face_index in indices:
                face_solids[face_index].append(solid_index)
        self.face_edges = face_edges
        self.edge_faces = edge_faces
        self.face_wires = face_wires
        self.solid_faces = solid_faces
        self.solid_face_orientations = solid_face_orientations
        self.face_solids = tuple(tuple(indices) for indices in face_solids)
        self.num_vertices = int(num_vertices)

    @property
    def num_faces(self) -> int:
        return len(self.face_edges)

    @property
    def num_edges(self) -> int:
        return len(self.edge_faces)

    @property
    def num_solids(self) -> int:
        return len(self.solid_faces)


@dataclass(frozen=True, slots=True)
class BRepOccurrence:
    """One placed assembly instance of a model solid.

    ``path`` is the occurrence path from the assembly root. ``rotation`` (a
    orthogonal matrix, with either coorientation) and ``translation`` map definition coordinates of
    the solid into the model's physical coordinate frame.
    """

    path: tuple[str, ...]
    solid: int
    rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

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
    ) -> None:
        if (
            not isinstance(path, tuple)
            or not path
            or any(not isinstance(name, str) or not name for name in path)
        ):
            raise ValueError("An occurrence path must be a nonempty tuple of names.")
        if isinstance(solid, bool) or not isinstance(solid, int):
            raise TypeError("An occurrence solid must be an integer index.")
        if solid < 0:
            raise ValueError("An occurrence solid index must be non-negative.")
        rotation_host = np.asarray(rotation, dtype=np.float64)
        translation_host = np.asarray(translation, dtype=np.float64)
        if rotation_host.shape != (3, 3) or translation_host.shape != (3,):
            raise ValueError("An occurrence placement is a 3x3 rotation and 3-vector.")
        if not np.all(np.isfinite(rotation_host)) or not np.all(
            np.isfinite(translation_host)
        ):
            raise ValueError("An occurrence placement must be finite.")
        if (
            np.max(np.abs(rotation_host.T @ rotation_host - np.eye(3))) > 1.0e-12
            or np.linalg.det(rotation_host) == 0.0
        ):
            raise ValueError("An occurrence rotation must be orthogonal and nonsingular.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "solid", solid)
        object.__setattr__(
            self,
            "rotation",
            tuple((float(row[0]), float(row[1]), float(row[2])) for row in rotation_host),
        )
        object.__setattr__(
            self,
            "translation",
            (
                float(translation_host[0]),
                float(translation_host[1]),
                float(translation_host[2]),
            ),
        )

    def place(self, points: ArrayLike, /) -> np.ndarray:
        """Map definition coordinates of the solid into model coordinates."""
        values = np.asarray(points, dtype=np.float64)
        return values @ np.asarray(self.rotation, dtype=np.float64).T + np.asarray(
            self.translation, dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class BRepAssemblyContainer:
    """Declared members of one exact container instance, never a spatial grouping."""

    path: tuple[str, ...]
    member_paths: tuple[tuple[str, ...], ...]
    child_paths: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, tuple)
            or not self.path
            or any(not isinstance(name, str) or not name for name in self.path)
        ):
            raise ValueError("A container instance needs an explicit nonempty path.")
        if (
            not isinstance(self.member_paths, tuple)
            or not isinstance(self.child_paths, tuple)
            or not (self.member_paths or self.child_paths)
        ):
            raise ValueError(
                "A container instance needs explicitly declared members or child containers."
            )
        for paths in (self.member_paths, self.child_paths):
            if any(
                not isinstance(path, tuple)
                or not path
                or any(not isinstance(name, str) or not name for name in path)
                for path in paths
            ):
                raise ValueError(
                    "Container relationships must have exact nonempty paths."
                )
            if len(set(paths)) != len(paths):
                raise ValueError("Container relationships cannot repeat an instance.")


@dataclass(frozen=True, slots=True, order=True)
class BRepQualifiedIncidence:
    """Exact occurrence-qualified incidence; coedge uses retain their source index."""

    container: BRepEntityId
    member: BRepEntityId
    orientation: int
    use_index: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.container, BRepEntityId) or not isinstance(
            self.member, BRepEntityId
        ):
            raise TypeError("Qualified incidence endpoints must be BRepEntityId values.")
        if self.container.source_revision != self.member.source_revision:
            raise ValueError("Qualified incidence cannot cross source revisions.")
        if self.orientation not in (-1, 1) or self.use_index < -1:
            raise ValueError(
                "Qualified incidence needs a signed orientation and valid use index."
            )


def _carrier_payload(carrier: StrictModule, /) -> dict[str, object]:
    """Exact identity of one curve/surface carrier: family, statics and arrays."""
    return {
        "family": type(carrier).__name__,
        "structure": str(jax.tree_util.tree_structure(carrier)),
        "arrays": array_tree_fingerprint(carrier),
    }


def _vertex_root_payload(root: BRepVertexRoot, /) -> dict[str, object]:
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


def _endpoint_root_payload(root: RootEndpoint, /) -> dict[str, object]:
    if isinstance(root, NativePeriodEndpoint):
        return {
            "kind": "native-period-endpoint",
            "source_id": root.root_id,
            "carrier": _carrier_payload(root.carrier),
            "patch": None if root.patch is None else _carrier_payload(root.patch),
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
            else _carrier_payload(root.source_pcurve),
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
        else _carrier_payload(root.source_pcurve),
        "source_surface": None
        if root.source_surface is None
        else encode_geometry(root.source_surface),
    }


def _coedge_vertices(
    edge_vertices: tuple[tuple[int, int], ...], edge: int, sense: int, /
) -> tuple[int, int]:
    start, end = edge_vertices[edge]
    return (start, end) if sense > 0 else (end, start)


class BRepGeometry(StrictModule):
    """Exact authoritative B-Rep geometry and oriented topology.

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
    vertex_evaluation_bounds: Array
    edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    edge_curves: tuple[int, ...] = eqx.field(static=True)
    edge_vertices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    coedge_senses: tuple[int, ...] = eqx.field(static=True)
    face_loops: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    shell_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shell_orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_shells: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occurrences: tuple[BRepOccurrence, ...] = eqx.field(static=True)
    assembly_containers: tuple[BRepAssemblyContainer, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

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
        edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
    ) -> None:
        points = np.asarray(vertex_points, dtype=np.float64)
        ranges = np.asarray(edge_ranges, dtype=np.float64)
        edge_count = len(edge_curves)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("vertex_points must be finite with shape (V, 3).")
        if len(edge_vertices) != edge_count or ranges.shape != (edge_count, 2):
            raise ValueError("Edge curves, ranges, and vertices must align.")
        if not all(
            isinstance(curve, (AbstractCurve, IntersectionCurve)) for curve in curves
        ):
            raise TypeError("Edges require native exact curve carriers.")
        if not all(
            isinstance(curve, (AbstractCurve, IntersectionPCurve)) for curve in pcurves
        ):
            raise TypeError("Coedges require native exact p-curve carriers.")
        if any(curve.ambient_dimension != 3 for curve in curves):
            raise ValueError("Edge curves must be three-dimensional.")
        if any(curve.ambient_dimension != 2 for curve in pcurves):
            raise ValueError("p-curves must be two-dimensional.")
        vertex_definitions = (
            (None,) * points.shape[0] if vertex_roots is None else tuple(vertex_roots)
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
        vertex_errors = self._validate_root_definitions(
            points,
            ranges,
            edge_vertices,
            coedge_edges,
            vertex_definitions,
            edge_definitions,
            coedge_definitions,
        )
        self._validate_edges(
            points, curves, edge_curves, ranges, edge_vertices, vertex_definitions
        )
        coedge_count = len(coedge_edges)
        if len(coedge_senses) != coedge_count or len(pcurves) != coedge_count:
            raise ValueError("Coedge edges, senses, and p-curves must align.")
        if any(edge < 0 or edge >= edge_count for edge in coedge_edges):
            raise ValueError("A coedge references an absent edge.")
        if any(sense not in (-1, 1) for sense in coedge_senses):
            raise ValueError("Coedge senses must be -1 or 1.")
        used = [coedge for loops in face_loops for loop in loops for coedge in loop]
        if sorted(used) != list(range(coedge_count)):
            raise ValueError("Every coedge must belong to exactly one face loop.")
        for loops in face_loops:
            if not loops or any(not loop for loop in loops):
                raise ValueError("Every face needs a nonempty outer loop.")
            for loop in loops:
                ends = [
                    _coedge_vertices(edge_vertices, coedge_edges[c], coedge_senses[c])
                    for c in loop
                ]
                if any(
                    ends[index][1] != ends[(index + 1) % len(ends)][0]
                    for index in range(len(ends))
                ):
                    raise ValueError("Face loop coedges must connect head to tail.")
        self._validate_shells(
            len(face_loops), shell_faces, shell_orientations, solid_shells
        )
        occurrences_ = (
            tuple(
                BRepOccurrence((f"solid{index}",), index)
                for index in range(len(solid_shells))
            )
            if occurrences is None
            else tuple(occurrences)
        )
        if any(
            not isinstance(occurrence, BRepOccurrence)
            or occurrence.solid >= len(solid_shells)
            for occurrence in occurrences_
        ):
            raise ValueError("Occurrences must reference model solids.")
        if len({occurrence.path for occurrence in occurrences_}) != len(occurrences_):
            raise ValueError("Occurrence paths must be unique.")
        occurrences_ = tuple(sorted(occurrences_, key=lambda occurrence: occurrence.path))
        containers = (
            (
                BRepAssemblyContainer(
                    ("model",), tuple(occurrence.path for occurrence in occurrences_)
                ),
            )
            if assembly_containers is None and occurrences is None and occurrences_
            else (() if assembly_containers is None else tuple(assembly_containers))
        )
        self._validate_containers(
            containers,
            occurrences_,
            shell_faces,
            solid_shells,
            face_loops,
            coedge_edges,
            edge_vertices,
        )
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
        self.coedge_edges = tuple(int(edge) for edge in coedge_edges)
        self.coedge_senses = tuple(int(sense) for sense in coedge_senses)
        self.face_loops = tuple(
            tuple(tuple(int(c) for c in loop) for loop in loops) for loops in face_loops
        )
        self.shell_faces = tuple(tuple(int(f) for f in faces) for faces in shell_faces)
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
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "brep-exact-geometry",
                "vertex_points": [
                    points[index] if root is None else None
                    for index, root in enumerate(vertex_definitions)
                ],
                "curves": [_carrier_payload(curve) for curve in self.curves],
                "edge_curves": list(self.edge_curves),
                "edge_ranges": [
                    [
                        float(ranges[edge, end]) if root is None else None
                        for end, root in enumerate(endpoints)
                    ]
                    for edge, endpoints in enumerate(edge_definitions)
                ],
                "edge_vertices": [list(pair) for pair in self.edge_vertices],
                "pcurves": [_carrier_payload(curve) for curve in self.pcurves],
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
                    None if root is None else _vertex_root_payload(root)
                    for root in vertex_definitions
                ],
                "edge_endpoint_roots": [
                    [
                        None if root is None else _endpoint_root_payload(root)
                        for root in endpoints
                    ]
                    for endpoints in edge_definitions
                ],
                "coedge_endpoint_roots": [
                    [
                        None if root is None else _endpoint_root_payload(root)
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
    ) -> np.ndarray:
        from ._correspondence import _norm_upper

        errors = np.zeros((points.shape[0],), dtype=np.float64)
        if (
            len(vertices) != points.shape[0]
            or len(edges) != len(edge_vertices)
            or len(coedges) != len(coedge_edges)
        ):
            raise ValueError(
                "Root definitions must align with vertices, edges and coedges."
            )
        if any(
            start < 0 or end < 0 or start >= points.shape[0] or end >= points.shape[0]
            for start, end in edge_vertices
        ):
            raise ValueError("An edge references an absent vertex.")
        if any(edge < 0 or edge >= len(edge_vertices) for edge in coedge_edges):
            raise ValueError("A coedge references an absent edge.")
        for vertex, root in enumerate(vertices):
            if root is not None:
                if not isinstance(root, BRepVertexRoot):
                    raise TypeError("Vertex roots must be BRepVertexRoot values or None.")
                box = root.point_enclosure()
                errors[vertex] = _norm_upper(
                    np.max(np.nextafter(np.abs(box - points[vertex]), np.inf), axis=0)
                )
        for endpoints, edge in [
            (endpoints, edge) for edge, endpoints in enumerate(edges)
        ] + [
            (endpoints, edge)
            for edge, endpoints in zip(coedge_edges, coedges, strict=True)
        ]:
            if not isinstance(endpoints, tuple) or len(endpoints) != 2:
                raise TypeError("Endpoint root bindings must be pairs.")
            for endpoint, root in enumerate(endpoints):
                vertex = edge_vertices[edge][endpoint]
                if root is None:
                    vertex_definition = vertices[vertex]
                    if vertex_definition is not None and not (
                        isinstance(vertex_definition.primary, BRepPlacedVertex)
                        and vertex_definition.primary.source_root is None
                    ):
                        raise ValueError(
                            "A root-valued vertex requires every incident endpoint's source binding."
                        )
                    continue
                if isinstance(root, NativePeriodEndpoint):
                    lower, upper = root.parameter_enclosure()
                    if not lower <= ranges[edge, endpoint] <= upper:
                        raise ValueError(
                            "A numerical edge endpoint leaves its authored native scalar enclosure."
                        )
                    continue
                if (
                    not isinstance(root, (TrimRootEndpoint, BranchRootEndpoint))
                    or vertices[vertex] is None
                ):
                    raise TypeError(
                        "Endpoint roots require canonical expressions and a vertex source definition."
                    )
                vertex_root = vertices[vertex]
                if vertex_root is None or not vertex_root.supports_endpoint(root):
                    raise ValueError(
                        "An endpoint expression does not belong to its authoritative vertex root."
                    )
                lower, upper = root.parameter_enclosure()
                if not lower <= ranges[edge, endpoint] <= upper:
                    raise ValueError(
                        "A numerical edge endpoint leaves its exact root expression enclosure."
                    )
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
    ) -> None:
        if any(
            not isinstance(container, BRepAssemblyContainer) for container in containers
        ):
            raise TypeError(
                "assembly_containers must contain BRepAssemblyContainer values."
            )
        if len({container.path for container in containers}) != len(containers):
            raise ValueError("Container instance paths must be unique.")
        by_path = {occurrence.path: occurrence for occurrence in occurrences}
        members = [path for container in containers for path in container.member_paths]
        if len(set(members)) != len(members) or any(
            path not in by_path for path in members
        ):
            raise ValueError(
                "Every declared container member must belong to exactly one known instance."
            )
        by_container = {container.path: container for container in containers}
        children = [path for container in containers for path in container.child_paths]
        if len(set(children)) != len(children) or any(
            path not in by_container for path in children
        ):
            raise ValueError(
                "Every child container must reference one known container instance."
            )
        for container in containers:
            pending = list(container.child_paths)
            visited: set[tuple[str, ...]] = set()
            while pending:
                path = pending.pop()
                if path == container.path or path in visited:
                    raise ValueError(
                        "Container membership must form an acyclic instance hierarchy."
                    )
                visited.add(path)
                pending.extend(by_container[path].child_paths)
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
                        first.rotation != other.rotation
                        or first.translation != other.translation
                        for other in uses[1:]
                    ):
                        raise ValueError(
                            "A declared shared container stratum requires one consistent definition placement."
                        )

    @staticmethod
    def _validate_edges(
        points: np.ndarray,
        curves: tuple[BRepCurve, ...],
        edge_curves: tuple[int, ...],
        ranges: np.ndarray,
        edge_vertices: tuple[tuple[int, int], ...],
        vertex_roots: tuple[BRepVertexRoot | None, ...],
    ) -> None:
        vertex_count = points.shape[0]
        scale = max(1.0, float(np.max(np.abs(points), initial=0.0)))
        tolerance = 1.0e-9 * scale
        for edge, (curve_index, (start, end)) in enumerate(
            zip(edge_curves, edge_vertices, strict=True)
        ):
            if not (0 <= start < vertex_count and 0 <= end < vertex_count):
                raise ValueError("An edge references an absent vertex.")
            if curve_index == -1:
                if start != end:
                    raise ValueError(
                        "A degenerate edge must start and end at one vertex."
                    )
                continue
            if not 0 <= curve_index < len(curves):
                raise ValueError("An edge references an absent curve.")
            curve = curves[curve_index]
            first, last = curve.validate_range(ranges[edge, 0], ranges[edge, 1])
            ends = np.asarray(
                curve_points(curve, jnp.asarray((first, last), dtype=jnp.float64))
            )
            if (
                vertex_roots[start] is None
                and np.linalg.norm(ends[0] - points[start]) > tolerance
            ) or (
                vertex_roots[end] is None
                and np.linalg.norm(ends[1] - points[end]) > tolerance
            ):
                raise ValueError(
                    f"Edge {edge} curve end points do not meet its vertices."
                )

    @staticmethod
    def _validate_shells(
        face_count: int,
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
    ) -> None:
        if len(shell_faces) != len(shell_orientations):
            raise ValueError("Shell faces and orientations must align.")
        members = [face for faces in shell_faces for face in faces]
        if any(face < 0 or face >= face_count for face in members):
            raise ValueError("A shell references an absent face.")
        incidences: dict[int, list[tuple[int, int]]] = {}
        for faces, signs in zip(shell_faces, shell_orientations, strict=True):
            if not faces or len(faces) != len(signs):
                raise ValueError("Every shell needs faces with one orientation each.")
            if any(sign not in (-1, 1) for sign in signs):
                raise ValueError("Shell face orientations must be -1 or 1.")
            if len(set(faces)) != len(faces):
                raise ValueError("A shell cannot repeat a face.")
        for shell, (faces, signs) in enumerate(
            zip(shell_faces, shell_orientations, strict=True)
        ):
            for face, sign in zip(faces, signs, strict=True):
                incidences.setdefault(face, []).append((shell, sign))
        shell_owner = {
            shell: solid for solid, group in enumerate(solid_shells) for shell in group
        }
        for uses in incidences.values():
            if len(uses) > 2:
                raise ValueError("A face has at most two region-side incidences.")
            if len(uses) == 2:
                (first_shell, first_sign), (second_shell, second_sign) = uses
                if (
                    first_sign == second_sign
                    or first_shell not in shell_owner
                    or second_shell not in shell_owner
                    or shell_owner[first_shell] == shell_owner[second_shell]
                ):
                    raise ValueError(
                        "A shared face must separate distinct solids with opposite incidences."
                    )
        shells = [shell for group in solid_shells for shell in group]
        if any(shell < 0 or shell >= len(shell_faces) for shell in shells):
            raise ValueError("A solid references an absent shell.")
        if len(set(shells)) != len(shells) or any(not group for group in solid_shells):
            raise ValueError("Each solid owns a nonempty set of distinct shells.")

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
        """Incidence view of the oriented exact topology."""
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

    def edge_parameter_enclosure(self, edge: int, /) -> np.ndarray:
        """Rows enclose each true endpoint parameter; literals have zero width."""
        if not 0 <= edge < len(self.edge_curves):
            raise ValueError("edge must index an authoritative edge.")
        values = np.asarray(self.edge_ranges)[edge]
        return np.asarray(
            [
                (float(values[endpoint]), float(values[endpoint]))
                if root is None
                else root.parameter_enclosure()
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
            raise ValueError("face_orientation must contain one sign per face.")
        if solid is not None and not 0 <= solid < len(self.solid_shells):
            raise ValueError("solid must index an existing solid.")
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

        def qualified(occurrence_index: int, kind: str, index: int, /) -> BRepEntityId:
            if occurrence_index < 0:
                return BRepEntityId(source_revision, kind, index)
            occurrence = self.occurrences[occurrence_index]
            container = membership.get(occurrence.path)
            path = occurrence.path
            owners = (
                () if container is None else grouped.get((container, kind, index), ())
            )
            if (
                kind != "solid"
                and container is not None
                and len(owners) > 1
                and len({self.occurrences[owner].solid for owner in owners})
                == len(owners)
            ):
                path = container
            return BRepEntityId(source_revision, kind, index, path)

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
                solid = qualified(occurrence_index, "solid", occurrence.solid)
                records.add(BRepQualifiedIncidence(solid, solid, 1))
                for face, sign in zip(
                    faces, topology.solid_face_orientations[occurrence.solid], strict=True
                ):
                    records.add(
                        BRepQualifiedIncidence(
                            solid, qualified(occurrence_index, "face", face), sign
                        )
                    )
            for face in faces:
                face_id = qualified(occurrence_index, "face", face)
                records.add(BRepQualifiedIncidence(face_id, face_id, 1))
                if face_id not in seen_faces:
                    seen_faces.add(face_id)
                    for loop in self.face_loops[face]:
                        for coedge in loop:
                            edge = self.coedge_edges[coedge]
                            edge_id = qualified(occurrence_index, "edge", edge)
                            records.add(
                                BRepQualifiedIncidence(
                                    face_id, edge_id, self.coedge_senses[coedge], coedge
                                )
                            )
                for edge in topology.face_edges[face]:
                    edge_id = qualified(occurrence_index, "edge", edge)
                    records.add(BRepQualifiedIncidence(edge_id, edge_id, 1))
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)
                        for endpoint, vertex in enumerate(self.edge_vertices[edge]):
                            vertex_id = qualified(occurrence_index, "vertex", vertex)
                            records.add(
                                BRepQualifiedIncidence(
                                    edge_id,
                                    vertex_id,
                                    -1 if endpoint == 0 else 1,
                                    endpoint,
                                )
                            )
                            records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
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


def brep_physical_scale(geometry: BRepGeometry, /) -> float:
    """Translation-independent declared length scale from exact edge/vertex bounds."""
    points = np.asarray(geometry.vertex_points)
    boxes = []
    literal = [
        points[index] for index, root in enumerate(geometry.vertex_roots) if root is None
    ]
    if literal:
        values = np.asarray(literal, dtype=np.float64)
        boxes.append(np.stack((np.min(values, axis=0), np.max(values, axis=0))))
    boxes.extend(
        root.point_enclosure() for root in geometry.vertex_roots if root is not None
    )
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index >= 0:
            endpoints = geometry.edge_parameter_enclosure(edge)
            curve = geometry.curves[curve_index]
            box = (
                curve.bounding_box(
                    float(endpoints[0, 0]),
                    float(endpoints[1, 1]),
                    endpoint_roots=geometry.edge_endpoint_roots[edge],
                )
                if isinstance(curve, IntersectionCurve)
                else curve.bounding_box(float(endpoints[0, 0]), float(endpoints[1, 1]))
            )
            boxes.append(box)
    if not boxes:
        return 0.0
    source_boxes = np.asarray(boxes, dtype=np.float64)
    if not np.all(np.isfinite(source_boxes)):
        raise ValueError(
            "The native B-Rep physical scale requires finite source enclosures."
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
        parameter_bounds: Array,
    ) -> None:
        bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        if not patches:
            raise ValueError("A BRepBoundaryMap requires at least one patch.")
        if bounds.shape != (len(patches), 2, 2):
            raise ValueError("parameter_bounds must have shape (num_faces, 2, 2).")
        bounds_host = np.asarray(bounds)
        if not np.all(np.isfinite(bounds_host)) or np.any(
            bounds_host[:, 1, :] <= bounds_host[:, 0, :]
        ):
            raise ValueError(
                "Every surface parameter interval must be finite and nonempty."
            )
        self.patches = patches
        self.parameter_bounds = bounds

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
    """The endpoint's UV trim root; scalar period endpoints have no spatial root."""
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
            pcurve.bounding_box(lower, upper, endpoint_roots=roots)
            if isinstance(pcurve, IntersectionPCurve)
            else pcurve.bounding_box(lower, upper)
        )
        if np.all(enclosure[0] >= parameter_box[0] - slack) and np.all(
            enclosure[1] <= parameter_box[1] + slack
        ):
            continue
        middle = lower + 0.5 * (upper - lower)
        points = np.asarray(
            pcurve.evaluate(jnp.asarray((lower, middle, upper), dtype=jnp.float64))
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


def _validate_exact_geometry(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    bounds: np.ndarray,
    topology: BRepTopology,
    curve_surface_tolerance: float,
) -> np.ndarray:
    """Bind incidence and whole-edge source correspondence, retaining its bounds."""
    from ._correspondence import certify_curve_surface, CurveSurfaceCorrespondenceError

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
        raise ValueError("topology must be the incidence view of the exact geometry.")
    points = np.asarray(geometry.vertex_points)
    ranges = np.asarray(geometry.edge_ranges)
    witnesses = np.full((len(geometry.coedge_edges),), np.inf, dtype=np.float64)
    for face, loops in enumerate(geometry.face_loops):
        patch = patches[face]
        box = patch.validate_parameter_box(bounds[face])
        parameter_scale = max(1.0, float(np.max(np.abs(box))))
        for loop in loops:
            heads, tails = [], []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                parameters = ranges[edge]
                for endpoint, definition in enumerate(
                    geometry.coedge_endpoint_roots[coedge]
                ):
                    if definition is None:
                        continue
                    if isinstance(definition, NativePeriodEndpoint):
                        pcurve = geometry.pcurves[coedge]
                        if canonical_fingerprint(
                            _carrier_payload(definition.carrier)
                        ) != canonical_fingerprint(_carrier_payload(pcurve)):
                            raise ValueError(
                                "A native scalar coedge endpoint must retain its actual original pcurve."
                            )
                        continue
                    vertex_root = geometry.vertex_roots[
                        geometry.edge_vertices[edge][endpoint]
                    ]
                    if vertex_root is None:
                        raise ValueError("A coedge root lost its vertex definition.")
                    carrier = definition.carrier
                    pcurve = geometry.pcurves[coedge]
                    same = canonical_fingerprint(
                        _carrier_payload(carrier)
                    ) == canonical_fingerprint(_carrier_payload(pcurve))
                    if isinstance(carrier, IntersectionPCurve) and isinstance(
                        pcurve, IntersectionPCurve
                    ):
                        same = (
                            carrier.curve.branch_id == pcurve.curve.branch_id
                            and carrier.side == pcurve.side
                            and carrier.reversed == pcurve.reversed
                        )
                    if isinstance(carrier, IntersectionCurve) and isinstance(
                        pcurve, IntersectionPCurve
                    ):
                        same = (
                            carrier.branch_id == pcurve.curve.branch_id
                            and not pcurve.reversed
                        )
                    if (
                        not same
                        and isinstance(carrier, (AbstractCurve, IntersectionCurve))
                        and carrier.ambient_dimension == 3
                    ):
                        source_first, source_last = definition.parameter_enclosure()
                        supporting = patch
                        source_point = points[geometry.edge_vertices[edge][endpoint]]
                        if isinstance(vertex_root.primary, BRepPlacedVertex):
                            if not isinstance(
                                patch, PlacedSurface
                            ) or not vertex_root.primary.matches_pose(patch):
                                raise ValueError(
                                    "A placed endpoint requires the actual identical supporting source pose."
                                )
                            supporting = patch.definition
                            source_point = np.asarray(vertex_root.primary.source_point)
                        binding = certify_curve_surface(
                            carrier,
                            pcurve,
                            supporting,
                            box,
                            source_first,
                            source_last,
                            point=source_point,
                            tolerance=0.0,
                            endpoint_roots=(definition, definition),
                        )
                        same = binding.complete and binding.deviation_bound == 0.0
                    identity = not definition.affine_transforms
                    if isinstance(definition, TrimRootEndpoint):
                        identity &= (
                            definition.affine_parameter_scale == 1.0
                            and definition.affine_parameter_offset == 0.0
                        )
                    if not same or not identity:
                        raise ValueError(
                            "A coedge root must bind its own source-proven original carrier parameter expression."
                        )
                    if isinstance(
                        definition.root, TrimIntersectionRoot
                    ) and not vertex_root.supports_uv(patch, definition.root):
                        raise ValueError(
                            "A coedge UV root has no verified support on its actual face."
                        )
                endpoint_boxes = geometry.edge_parameter_enclosure(edge)
                # Both declared supports were proved above to bind the same
                # original scalar endpoint. Intersect their certified images,
                # rather than query one support with the other's uncertainty.
                for endpoint, definition in enumerate(
                    geometry.coedge_endpoint_roots[coedge]
                ):
                    if definition is not None:
                        lower, upper = definition.parameter_enclosure()
                        endpoint_boxes[endpoint, 0] = max(
                            endpoint_boxes[endpoint, 0], lower
                        )
                        endpoint_boxes[endpoint, 1] = min(
                            endpoint_boxes[endpoint, 1], upper
                        )
                if np.any(endpoint_boxes[:, 0] > endpoint_boxes[:, 1]):
                    raise ValueError(
                        "Source-proven endpoint scalar enclosures have no common value."
                    )
                proof_first, proof_last = (
                    float(endpoint_boxes[0, 0]),
                    float(endpoint_boxes[1, 1]),
                )
                uv = np.asarray(
                    geometry.pcurves[coedge].evaluate(jnp.asarray(parameters))
                )
                pcurve = geometry.pcurves[coedge]
                slack = 1.0e-9 * parameter_scale
                if not _pcurve_inside_parameter_box(
                    pcurve,
                    proof_first,
                    proof_last,
                    box,
                    geometry.coedge_endpoint_roots[coedge],
                    slack,
                ):
                    raise ValueError(
                        f"Coedge {coedge} p-curve leaves its face parameter box."
                    )
                curve_index = geometry.edge_curves[edge]
                source_curve = None if curve_index == -1 else geometry.curves[curve_index]
                supporting_patch = patch
                source_point = points[geometry.edge_vertices[edge][0]]
                placed_vertex = geometry.vertex_roots[geometry.edge_vertices[edge][0]]
                if (
                    source_curve is None
                    and isinstance(patch, PlacedSurface)
                    and placed_vertex is not None
                    and isinstance(placed_vertex.primary, BRepPlacedVertex)
                    and placed_vertex.primary.matches_pose(patch)
                ):
                    supporting_patch = patch.definition
                    source_point = np.asarray(placed_vertex.primary.source_point)
                witness = certify_curve_surface(
                    source_curve,
                    geometry.pcurves[coedge],
                    supporting_patch,
                    box,
                    proof_first,
                    proof_last,
                    point=source_point,
                    tolerance=curve_surface_tolerance,
                    endpoint_roots=geometry.coedge_endpoint_roots[coedge],
                )
                deviation = witness.deviation_bound
                if supporting_patch is not patch:
                    if not isinstance(patch, PlacedSurface):
                        raise RuntimeError(
                            "A pulled-back source support requires its owning PlacedSurface."
                        )
                    from ._correspondence import placed_deviation_bound

                    deviation = placed_deviation_bound(
                        np.asarray(patch.rotation), deviation
                    )
                witnesses[coedge] = deviation
                if not witness.complete or deviation > curve_surface_tolerance:
                    raise CurveSurfaceCorrespondenceError(
                        face, coedge, witness, curve_surface_tolerance
                    )
                ordered = uv if geometry.coedge_senses[coedge] > 0 else uv[::-1]
                heads.append(ordered[0])
                tails.append(ordered[-1])
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
                    if gaps[position] > 1.0e-9 * parameter_scale:
                        raise ValueError(
                            "Face loops must be closed in the parameter plane."
                        )
                    continue
                first = geometry.coedge_endpoint_roots[coedge][end]
                second = geometry.coedge_endpoint_roots[following][start]
                if first is None or second is None:
                    raise ValueError(
                        "A root-valued trim junction lost an endpoint binding."
                    )
                first_uv, second_uv = _trim_uv_root(first), _trim_uv_root(second)
                same_uv_root = (
                    first_uv is not None
                    and second_uv is not None
                    and first_uv.root_id == second_uv.root_id
                )
                if not same_uv_root and not root.same_uv_endpoint(
                    patch,
                    geometry.pcurves[coedge],
                    first,
                    geometry.pcurves[following],
                    second,
                ):
                    raise ValueError(
                        "The trim junction has no common source-root and injective chart certificate."
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
) -> tuple[str, ...]:
    """Rebuild every exact UV restriction from persisted source authority."""
    count = vertices.size
    if (
        vertices.dtype != np.int64
        or vertices.shape != (count,)
        or edges.dtype != np.int64
        or edges.shape != (count, 2)
        or endpoint_parameters.dtype != np.float64
        or endpoint_parameters.shape != (count, 2, 2)
        or ratios.dtype != np.int64
        or ratios.shape != (count, 2)
    ):
        raise TypeError("Tessellation chart restrictions use canonical typed banks.")
    if count == 0:
        return ()
    if (
        not np.array_equal(vertices, np.unique(vertices))
        or np.any(vertices < 0)
        or np.any(vertices >= source_dimensions.size)
        or np.any(edges < 0)
        or np.any(edges >= source_dimensions.size)
        or not np.all(np.isfinite(endpoint_parameters))
    ):
        raise ValueError("Tessellation chart restrictions have invalid vertex incidence.")
    exact_by_vertex: dict[int, tuple[Fraction, Fraction]] = {}
    identifiers = []
    for row, vertex_ in enumerate(vertices.tolist()):
        vertex = int(vertex_)
        first, second = (int(value) for value in edges[row])
        numerator, denominator = (int(value) for value in ratios[row])
        if (
            max(first, second) >= vertex
            or first == second
            or denominator <= 0
            or not 0 < numerator < denominator
        ):
            raise ValueError(
                "Tessellation chart restrictions are not ordered open-edge roots."
            )
        if source_dimensions[vertex] != 2:
            raise ValueError(
                "An exact chart restriction must retain a source-face association."
            )
        if np.any(occurrence_ids[[first, second]] != occurrence_ids[vertex]):
            raise ValueError(
                "An exact chart restriction cannot cross occurrence ownership."
            )
        face = int(source_indices[vertex])
        endpoint_exact = []
        for endpoint_row, endpoint in enumerate((first, second)):
            represented = endpoint_parameters[row, endpoint_row]
            if endpoint in exact_by_vertex:
                exact = exact_by_vertex[endpoint]
            else:
                exact = (
                    Fraction(float(represented[0])),
                    Fraction(float(represented[1])),
                )
            endpoint_exact.append(exact)
            incident = np.argwhere(
                (faces == endpoint)
                & (face_ids[:, None] == face)
                & (triangle_occurrence_ids[:, None] == occurrence_ids[vertex])
            )
            matched = any(
                np.array_equal(triangle_parameters[int(cell), int(corner)], represented)
                for cell, corner in incident
            )
            # A zero-area pole child is intentionally absent from the physical
            # tessellation. Its UV copy can therefore retire while the same
            # authored corner remains incident to this face through another
            # pole copy; the restriction endpoint bank retains the retired UV.
            if not matched and (source_dimensions[endpoint] != 0 or not incident.size):
                raise ValueError(
                    "A chart restriction endpoint is not bound to its face incidence."
                )
        execution = source_parameters[vertex]
        incident = np.argwhere(
            (faces == vertex)
            & (face_ids[:, None] == face)
            & (triangle_occurrence_ids[:, None] == occurrence_ids[vertex])
        )
        if not incident.size or any(
            not np.array_equal(triangle_parameters[int(cell), int(corner)], execution)
            for cell, corner in incident
        ):
            raise ValueError(
                "A rounded chart restriction is not bound to every incident face corner."
            )
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


class BRepModel(StrictModule):
    """Authoritative B-Rep with a separately identified derived query tessellation.

    ``model_id`` identifies the exact represented geometry and topology
    (patches, parameter domains, orientation, incidence and, when present, the
    exact ``geometry`` of curves, p-curves, loops, shells and occurrences).
    ``tessellation_id`` additionally identifies derived mesh/UV lineage,
    continuous deviation and normal-turn bounds, exact trim chord covers, and
    exact rational chart restrictions for vertices whose authoritative UV
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
    ) -> None:
        if coordinate_contract.spatial_id != report.coordinate_contract.spatial_id:
            raise ValueError(
                "The model and import report coordinate contracts must match."
            )
        face_count = len(patches)
        bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        orientation_ = jnp.asarray(orientation, dtype=jnp.float64).reshape((-1,))
        vertices = jnp.asarray(mesh_vertices, dtype=jnp.float64)
        faces = jnp.asarray(mesh_faces, dtype=jnp.int32)
        face_ids = jnp.asarray(triangle_face_ids, dtype=jnp.int32).reshape((-1,))
        parameters = jnp.asarray(triangle_parameters, dtype=jnp.float64)
        bounds_host = np.asarray(bounds)
        orientation_host = np.asarray(orientation_)
        vertices_host = np.asarray(vertices)
        faces_host = np.asarray(faces)
        face_ids_host = np.asarray(face_ids)
        parameters_host = np.asarray(parameters)
        if topology.num_faces != face_count:
            raise ValueError("topology must contain one entry per face.")
        if bounds.shape != (face_count, 2, 2):
            raise ValueError("parameter_bounds must contain one 2D box per face.")
        if not np.all(np.isfinite(bounds_host)) or np.any(
            bounds_host[:, 1, :] <= bounds_host[:, 0, :]
        ):
            raise ValueError(
                "Every surface parameter interval must be finite and nonempty."
            )
        if orientation_.shape != (face_count,):
            raise ValueError("orientation must contain one sign per face.")
        if not np.all(np.isin(orientation_host, (-1.0, 1.0))):
            raise ValueError("orientation entries must be -1 or 1.")
        if len(trim_domains) != face_count or len(physical_tags) != face_count:
            raise ValueError("Trim domains and physical tags must align with faces.")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh_vertices must have shape (num_vertices, 3).")
        if not np.all(np.isfinite(vertices_host)):
            raise ValueError("mesh_vertices must be finite.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("mesh_faces must have shape (num_triangles, 3).")
        if faces_host.size and (
            np.min(faces_host) < 0 or np.max(faces_host) >= vertices.shape[0]
        ):
            raise ValueError("mesh_faces reference an absent mesh vertex.")
        if face_ids.shape != (faces.shape[0],):
            raise ValueError("triangle_face_ids must align with mesh faces.")
        if face_ids_host.size and (
            np.min(face_ids_host) < 0 or np.max(face_ids_host) >= face_count
        ):
            raise ValueError("triangle_face_ids reference an absent B-Rep face.")
        if parameters.shape != (faces.shape[0], 3, 2):
            raise ValueError("triangle_parameters must have shape (num_triangles, 3, 2).")
        if not np.all(np.isfinite(parameters_host)):
            raise ValueError("triangle_parameters must be finite.")
        deviation = (
            jnp.full((faces.shape[0],), jnp.inf, dtype=jnp.float64)
            if tessellation_deviation_bounds is None
            else jnp.asarray(tessellation_deviation_bounds, dtype=jnp.float64)
        )
        normal = (
            jnp.full((faces.shape[0],), jnp.inf, dtype=jnp.float64)
            if tessellation_normal_bounds is None
            else jnp.asarray(tessellation_normal_bounds, dtype=jnp.float64)
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
                raise ValueError(
                    f"{name} must be nonnegative triangle-aligned bounds; infinity means unresolved."
                )
        vertex_count = vertices.shape[0]
        for name, value in (
            ("mesh_vertex_source_dimensions", mesh_vertex_source_dimensions),
            ("mesh_vertex_source_indices", mesh_vertex_source_indices),
        ):
            if value is not None and not np.issubdtype(
                np.asarray(value).dtype, np.integer
            ):
                raise TypeError(f"{name} must contain integer entity metadata.")
        source_dimensions = (
            jnp.full((vertex_count,), -1, dtype=jnp.int32)
            if mesh_vertex_source_dimensions is None
            else jnp.asarray(mesh_vertex_source_dimensions, dtype=jnp.int32)
        )
        source_indices = (
            jnp.full((vertex_count,), -1, dtype=jnp.int64)
            if mesh_vertex_source_indices is None
            else jnp.asarray(mesh_vertex_source_indices, dtype=jnp.int64)
        )
        source_parameters = (
            jnp.full((vertex_count, 2), jnp.nan, dtype=jnp.float64)
            if mesh_vertex_parameters is None
            else jnp.asarray(mesh_vertex_parameters, dtype=jnp.float64)
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
            raise ValueError("Tessellation vertex lineage must align with mesh vertices.")
        if not np.all(np.isin(dimensions_host, (-1, 0, 1, 2))):
            raise ValueError(
                "Tessellation vertex source dimensions must be unknown, vertex, edge, or face."
            )
        if np.any(indices_host[dimensions_host == -1] != -1):
            raise ValueError(
                "An unknown tessellation source dimension requires an unknown source index."
            )
        for dimension, capacity in enumerate(
            (topology.num_vertices, topology.num_edges, topology.num_faces)
        ):
            selected = dimensions_host == dimension
            if np.any(indices_host[selected] < 0) or np.any(
                indices_host[selected] >= capacity
            ):
                raise ValueError(
                    "Tessellation vertex lineage references an absent exact entity."
                )
            if dimension > 0 and not np.all(
                np.isfinite(source_parameters_host[selected, :dimension])
            ):
                raise ValueError(
                    "Curve/face tessellation lineage requires finite native parameters."
                )
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
            raise ValueError(
                "The import report entity counts must match the B-Rep model."
            )
        correspondence = (
            np.empty((0,), dtype=np.float64)
            if geometry is None
            else _validate_exact_geometry(
                geometry,
                patches,
                bounds_host,
                topology,
                report.curve_surface_tolerance,
            )
        )
        if coedge_deviation_bounds is not None:
            retained = np.asarray(coedge_deviation_bounds, dtype=np.float64)
            if (
                retained.shape != correspondence.shape
                or np.any(np.isnan(retained))
                or np.any(retained < correspondence)
            ):
                raise ValueError(
                    "Retained coedge bounds must contain the independently recomputed source bounds."
                )
            correspondence = retained
        for name, value in (
            ("triangle_occurrence_ids", triangle_occurrence_ids),
            ("vertex_occurrence_ids", vertex_occurrence_ids),
        ):
            if value is not None and not np.issubdtype(
                np.asarray(value).dtype, np.integer
            ):
                raise TypeError(f"{name} must contain integer occurrence-table indices.")
        triangle_instances = (
            jnp.full((faces.shape[0],), -1, dtype=jnp.int32)
            if triangle_occurrence_ids is None
            else jnp.asarray(triangle_occurrence_ids, dtype=jnp.int32)
        )
        vertex_instances = (
            jnp.full((vertex_count,), -1, dtype=jnp.int32)
            if vertex_occurrence_ids is None
            else jnp.asarray(vertex_occurrence_ids, dtype=jnp.int32)
        )
        triangle_instances_host, vertex_instances_host = (
            np.asarray(triangle_instances),
            np.asarray(vertex_instances),
        )
        occurrence_count = 0 if geometry is None else len(geometry.occurrences)
        if triangle_instances.shape != (faces.shape[0],) or vertex_instances.shape != (
            vertex_count,
        ):
            raise ValueError(
                "Derived occurrence IDs must align with triangles and vertices."
            )
        if (
            np.any(triangle_instances_host < -1)
            or np.any(triangle_instances_host >= occurrence_count)
            or np.any(vertex_instances_host < -1)
            or np.any(vertex_instances_host >= occurrence_count)
        ):
            raise ValueError(
                "Derived occurrence IDs must reference the authoritative table or unplaced definition."
            )
        if faces_host.size and not np.all(
            vertex_instances_host[faces_host] == triangle_instances_host[:, None]
        ):
            raise ValueError(
                "A derived triangle cannot weld independent occurrence vertices."
            )
        if geometry is not None:
            for occurrence_index, occurrence in enumerate(geometry.occurrences):
                selected = triangle_instances_host == occurrence_index
                if not np.all(
                    np.isin(
                        face_ids_host[selected], topology.solid_faces[occurrence.solid]
                    )
                ):
                    raise ValueError(
                        "A placed triangle must belong to its declared solid occurrence."
                    )
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
            if value is not None and not np.issubdtype(
                np.asarray(value).dtype, np.integer
            ):
                raise TypeError(f"{name} must contain exact integer metadata.")
        restriction_vertices = jnp.asarray(
            np.empty((0,), dtype=np.int64)
            if mesh_chart_restriction_vertices is None
            else mesh_chart_restriction_vertices,
            dtype=jnp.int64,
        )
        restriction_edges = jnp.asarray(
            np.empty((restriction_count, 2), dtype=np.int64)
            if mesh_chart_restriction_edges is None
            else mesh_chart_restriction_edges,
            dtype=jnp.int64,
        )
        restriction_endpoint_parameters = jnp.asarray(
            np.empty((restriction_count, 2, 2), dtype=np.float64)
            if mesh_chart_restriction_endpoint_parameters is None
            else mesh_chart_restriction_endpoint_parameters,
            dtype=jnp.float64,
        )
        restriction_parameters = jnp.asarray(
            np.empty((restriction_count, 2), dtype=np.int64)
            if mesh_chart_restriction_parameters is None
            else mesh_chart_restriction_parameters,
            dtype=jnp.int64,
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
        )
        self.patches = patches
        self.parameter_bounds = bounds
        self.orientation = orientation_
        self.trim_domains = trim_domains
        self.topology = topology
        self.geometry = geometry
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.triangle_face_ids = face_ids
        self.triangle_parameters = parameters
        self.tessellation_deviation_bounds = deviation
        self.tessellation_normal_bounds = normal
        self.mesh_vertex_source_dimensions = source_dimensions
        self.mesh_vertex_source_indices = source_indices
        self.mesh_vertex_parameters = source_parameters
        self.mesh_chart_restriction_vertices = restriction_vertices
        self.mesh_chart_restriction_edges = restriction_edges
        self.mesh_chart_restriction_endpoint_parameters = restriction_endpoint_parameters
        self.mesh_chart_restriction_parameters = restriction_parameters
        self.coedge_deviation_bounds = jnp.asarray(correspondence, dtype=jnp.float64)
        self.triangle_occurrence_ids = triangle_instances
        self.vertex_occurrence_ids = vertex_instances
        self.physical_tags = physical_tags
        self.report = report
        self.chart_restriction_ids = restriction_ids
        self.model_id = canonical_fingerprint(
            {
                "kind": "brep-model",
                "source_revision": report.source_revision,
                "patches": [_carrier_payload(patch) for patch in patches],
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
        self.tessellation_id = canonical_fingerprint(
            {
                "kind": "brep-query-tessellation",
                "model_id": self.model_id,
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
        """Canonical physical face row inventory, retaining authored occurrence paths."""
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
]

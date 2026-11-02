#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from math import hypot
from numbers import Integral, Real
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import ExactScalar
from ...discretization._surface_chart_records import SurfaceChartWitness
from ...typing import checked, ConvertibleToArray
from .._atlas import AbstractBoundaryMap, BoundaryAtlas, TrimDomain
from .._source_fidelity_certificate import SourceFidelityCertificate
from ._assembly import (
    BRepAffinePlacement,
    BRepAssemblyGraph,
    BRepAssemblyOccurrence,
    BRepAssemblyUse,
)
from ._intersection import ConicRootLineEndpointWitness, RootEndpoint
from ._patches import AbstractSurfacePatch, BRepCurve, ExactLineCurve, PlanePatch
from ._root_bindings import BRepVertexRoot


if TYPE_CHECKING:
    from .._meshing_domain import MeshingDomain
    from ..multiregion_surface import MultiRegionSurfaceTopology


@dataclass(frozen=True, slots=True, order=True)
class BRepEntityId:
    """Stable source-owner- and revision-scoped identity for one B-Rep entity."""

    source_id: str
    source_revision: str
    kind: str
    index: int
    occurrence_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TypeError("source_id must be a string.")
        if not self.source_id:
            raise ValueError("source_id must be non-empty.")
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
        if not isinstance(self.occurrence_path, tuple) or any(
            not isinstance(part, str) or not part for part in self.occurrence_path
        ):
            raise ValueError("occurrence_path must contain nonempty string identifiers.")


@dataclass(frozen=True, slots=True)
class BRepAssemblyContainer:
    """Source assembly namespace and its direct occurrence/child members."""

    path: tuple[str, ...]
    member_paths: tuple[tuple[str, ...], ...]
    child_paths: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.path or any(
            not isinstance(part, str) or not part for part in self.path
        ):
            raise ValueError("Assembly container paths must be nonempty tuples of names.")
        if not isinstance(self.member_paths, tuple) or not isinstance(
            self.child_paths, tuple
        ):
            raise TypeError("Assembly container members and children must be tuples.")
        for paths in (self.member_paths, self.child_paths):
            if any(
                not path or any(not isinstance(part, str) or not part for part in path)
                for path in paths
            ):
                raise ValueError(
                    "Assembly member paths must be nonempty tuples of names."
                )
            if len(set(paths)) != len(paths):
                raise ValueError("Assembly container paths must not repeat members.")
        if not self.member_paths and not self.child_paths:
            raise ValueError("An assembly container must declare a member or child.")


@dataclass(frozen=True, slots=True, order=True)
class BRepQualifiedIncidence:
    """Occurrence-qualified entity incidence with directed use identity."""

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
            raise ValueError(
                "Qualified incidence endpoints must share a source revision."
            )
        if self.orientation not in (-1, 1) or self.use_index < -1:
            raise ValueError(
                "Qualified incidence requires orientation ±1 and use_index ≥ -1."
            )


@dataclass(frozen=True, slots=True)
class BRepImportReport:
    """Import provenance and source-fidelity metadata, independent of realization."""

    source_id: str
    source_digest: str
    source_format: str
    coordinate_contract: SpatialCoordinateContract
    import_policy_id: str
    num_solids: int
    num_faces: int
    num_edges: int
    num_vertices: int
    converted_surface_count: int = 0
    curve_surface_tolerance: float = 0.0
    curve_surface_scale: float = 1.0
    origin_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        for name, value in (
            ("source_digest", self.source_digest),
            ("import_policy_id", self.import_policy_id),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        if not isinstance(self.source_format, str) or not self.source_format:
            raise ValueError("source_format must be a non-empty string.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        counts = (
            self.num_solids,
            self.num_faces,
            self.num_edges,
            self.num_vertices,
            self.converted_surface_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in counts
        ):
            raise TypeError("B-Rep source counts must be integers.")
        if any(value < 0 for value in counts):
            raise ValueError("B-Rep source counts must be non-negative.")
        # Free vertices and edges need no face; a solid always bounds faces.
        if self.num_faces == 0 and self.num_solids != 0:
            raise ValueError("A face-free B-Rep report cannot describe solids.")
        if self.converted_surface_count > self.num_faces:
            raise ValueError("converted_surface_count cannot exceed num_faces.")
        if (
            isinstance(self.curve_surface_tolerance, bool)
            or not np.isfinite(self.curve_surface_tolerance)
            or self.curve_surface_tolerance < 0.0
        ):
            raise ValueError("curve_surface_tolerance must be finite and non-negative.")
        if (
            isinstance(self.curve_surface_scale, bool)
            or not np.isfinite(self.curve_surface_scale)
            or self.curve_surface_scale < 0.0
        ):
            raise ValueError("curve_surface_scale must be finite and non-negative.")
        object.__setattr__(
            self,
            "origin_revision",
            canonical_fingerprint(
                {
                    "kind": "brep-import-origin",
                    "source_digest": self.source_digest,
                    "source_format": self.source_format,
                    "spatial_id": self.coordinate_contract.spatial_id,
                    "import_policy_id": self.import_policy_id,
                }
            ),
        )


class BRepTessellation(StrictModule):
    """Derived triangles with source ancestry and continuous fidelity bounds.

    ``normal_bounds`` measures the continuous oriented source-normal deviation
    from the represented target coordinate-map normal, not source curvature or
    triangle-shape quality. Target vertex/cell IDs and ``source_chart_witness``
    preserve exact root, extraction-piece, face, and occurrence ancestry for proof renewal.
    """

    vertices: Array
    triangles: Array
    face_ids: Array
    parameters: Array
    deviation_bounds: Array
    normal_bounds: Array
    vertex_source_dimensions: Array
    vertex_source_indices: Array
    vertex_parameters: Array
    triangle_occurrence_ids: Array
    vertex_occurrence_ids: Array
    vertex_global_ids: Array
    triangle_global_ids: Array
    source_chart_witness: SurfaceChartWitness
    linear_deflection: float = eqx.field(static=True)
    angular_deflection: float = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    tessellation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        linear_deflection: float,
        angular_deflection: float,
        source_revision: str,
        policy_id: str,
        vertices: ArrayLike,
        triangles: ArrayLike,
        face_ids: ArrayLike,
        parameters: ArrayLike,
        deviation_bounds: ArrayLike,
        normal_bounds: ArrayLike,
        vertex_source_dimensions: ArrayLike,
        vertex_source_indices: ArrayLike,
        vertex_parameters: ArrayLike,
        vertex_global_ids: ArrayLike,
        triangle_global_ids: ArrayLike,
        source_chart_witness: SurfaceChartWitness,
        triangle_occurrence_ids: ArrayLike | None = None,
        vertex_occurrence_ids: ArrayLike | None = None,
    ) -> None:
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("source_revision must be a non-empty string.")
        if (
            not isinstance(policy_id, str)
            or len(policy_id) != 64
            or policy_id != policy_id.lower()
            or any(character not in "0123456789abcdef" for character in policy_id)
        ):
            raise ValueError("policy_id must be a lowercase SHA-256 digest.")
        if isinstance(linear_deflection, bool) or not isinstance(linear_deflection, Real):
            raise TypeError("linear_deflection must be a real scalar.")
        if isinstance(angular_deflection, bool) or not isinstance(
            angular_deflection, Real
        ):
            raise TypeError("angular_deflection must be a real scalar.")
        linear_ = float(linear_deflection)
        angular_ = float(angular_deflection)
        if not np.isfinite(linear_) or linear_ <= 0.0:
            raise ValueError("linear_deflection must be finite and positive.")
        if not np.isfinite(angular_) or not 0.0 < angular_ < np.pi:
            raise ValueError("angular_deflection must lie strictly between zero and pi.")
        vertex_values = np.asarray(vertices, dtype=np.float64)
        raw_triangles = np.asarray(triangles)
        raw_face_ids = np.asarray(face_ids)
        parameter_values = np.asarray(parameters, dtype=np.float64)
        deviation_values = np.asarray(deviation_bounds, dtype=np.float64)
        normal_values = np.asarray(normal_bounds, dtype=np.float64)
        raw_dimensions = np.asarray(vertex_source_dimensions)
        raw_indices = np.asarray(vertex_source_indices)
        vertex_parameter_values = np.asarray(vertex_parameters, dtype=np.float64)
        raw_vertex_ids = np.asarray(vertex_global_ids)
        raw_triangle_ids = np.asarray(triangle_global_ids)
        if vertex_values.ndim != 2 or vertex_values.shape[1] != 3:
            raise ValueError("vertices must have shape (num_vertices, 3).")
        if vertex_values.shape[0] == 0 or not np.all(np.isfinite(vertex_values)):
            raise ValueError("A tessellation requires finite, nonempty vertices.")
        if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (num_triangles, 3).")
        if raw_triangles.shape[0] == 0 or not np.issubdtype(
            raw_triangles.dtype, np.integer
        ):
            raise ValueError("A tessellation requires nonempty integer triangles.")
        if raw_face_ids.ndim != 1 or not np.issubdtype(raw_face_ids.dtype, np.integer):
            raise ValueError("face_ids must be a one-dimensional integer array.")
        if raw_face_ids.shape != (raw_triangles.shape[0],):
            raise ValueError("face_ids must align with triangles.")
        if raw_face_ids.size and (
            np.min(raw_face_ids) < 0 or np.max(raw_face_ids) > np.iinfo(np.int32).max
        ):
            raise ValueError("face_ids must be nonnegative int32-representable IDs.")
        if raw_triangles.size and (
            np.min(raw_triangles) < 0
            or np.max(raw_triangles) >= vertex_values.shape[0]
            or np.max(raw_triangles) > np.iinfo(np.int32).max
        ):
            raise ValueError("triangles reference an absent or unrepresentable vertex.")
        if (
            raw_vertex_ids.shape != (vertex_values.shape[0],)
            or not np.issubdtype(raw_vertex_ids.dtype, np.integer)
            or np.any(raw_vertex_ids < 0)
            or np.max(raw_vertex_ids) > np.iinfo(np.int64).max
            or np.unique(raw_vertex_ids).size != raw_vertex_ids.size
        ):
            raise ValueError("vertex_global_ids must be unique nonnegative int64 IDs.")
        if (
            raw_triangle_ids.shape != (raw_triangles.shape[0],)
            or not np.issubdtype(raw_triangle_ids.dtype, np.integer)
            or np.any(raw_triangle_ids < 0)
            or np.max(raw_triangle_ids) > np.iinfo(np.int64).max
            or np.unique(raw_triangle_ids).size != raw_triangle_ids.size
        ):
            raise ValueError("triangle_global_ids must be unique nonnegative int64 IDs.")

        witness_ids = np.asarray(source_chart_witness.cell_global_ids, dtype=np.int64)
        if not np.array_equal(
            np.sort(witness_ids), np.sort(raw_triangle_ids.astype(np.int64))
        ):
            raise ValueError(
                "Source chart witness IDs must cover each tessellation triangle."
            )
        if parameter_values.shape != (raw_triangles.shape[0], 3, 2):
            raise ValueError("parameters must have shape (num_triangles, 3, 2).")
        if not np.all(np.isfinite(parameter_values)):
            raise ValueError("parameters must be finite.")
        if deviation_values.shape != (raw_triangles.shape[0],) or np.any(
            ~np.isfinite(deviation_values) | (deviation_values < 0.0)
        ):
            raise ValueError("deviation_bounds must be finite nonnegative per triangle.")
        if normal_values.shape != (raw_triangles.shape[0],) or np.any(
            ~np.isfinite(normal_values) | (normal_values < 0.0)
        ):
            raise ValueError("normal_bounds must be finite nonnegative per triangle.")
        if (
            raw_dimensions.shape != (vertex_values.shape[0],)
            or not np.issubdtype(raw_dimensions.dtype, np.integer)
            or np.any((raw_dimensions < 0) | (raw_dimensions > 2))
        ):
            raise ValueError(
                "vertex_source_dimensions must contain 0, 1, or 2 per vertex."
            )
        if (
            raw_indices.shape != (vertex_values.shape[0],)
            or not np.issubdtype(raw_indices.dtype, np.integer)
            or np.any(raw_indices < 0)
        ):
            raise ValueError("vertex_source_indices must contain nonnegative IDs.")
        if raw_indices.size and np.max(raw_indices) > np.iinfo(np.int64).max:
            raise ValueError("vertex_source_indices exceed the int64 identity range.")
        if vertex_parameter_values.shape != (vertex_values.shape[0], 2) or not np.all(
            np.isfinite(vertex_parameter_values)
        ):
            raise ValueError(
                "vertex_parameters must be finite with shape (num_vertices, 2)."
            )
        used = np.unique(raw_triangles)
        if used.size != vertex_values.shape[0]:
            raise ValueError("A tessellation cannot contain unused vertices.")
        triangle_occurrences = _occurrence_ids(
            triangle_occurrence_ids, raw_triangles.shape[0], "triangle_occurrence_ids"
        )
        vertex_occurrences = _occurrence_ids(
            vertex_occurrence_ids, vertex_values.shape[0], "vertex_occurrence_ids"
        )
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("source_revision must be a non-empty string.")
        self.vertices = jnp.asarray(vertex_values, dtype=jnp.float64)
        self.triangles = jnp.asarray(raw_triangles, dtype=jnp.int32)
        self.face_ids = jnp.asarray(raw_face_ids, dtype=jnp.int32)
        self.parameters = jnp.asarray(parameter_values, dtype=jnp.float64)
        self.deviation_bounds = jnp.asarray(deviation_values, dtype=jnp.float64)
        self.normal_bounds = jnp.asarray(normal_values, dtype=jnp.float64)
        self.vertex_source_dimensions = jnp.asarray(raw_dimensions, dtype=jnp.int32)
        self.vertex_source_indices = jnp.asarray(raw_indices, dtype=jnp.int64)
        self.vertex_parameters = jnp.asarray(vertex_parameter_values, dtype=jnp.float64)
        self.triangle_occurrence_ids = jnp.asarray(triangle_occurrences, dtype=jnp.int32)
        self.vertex_occurrence_ids = jnp.asarray(vertex_occurrences, dtype=jnp.int32)
        self.vertex_global_ids = jnp.asarray(raw_vertex_ids, dtype=jnp.int64)
        self.triangle_global_ids = jnp.asarray(raw_triangle_ids, dtype=jnp.int64)
        self.source_chart_witness = source_chart_witness
        self.linear_deflection = linear_
        self.angular_deflection = angular_
        self.source_revision = source_revision
        self.policy_id = policy_id
        self.tessellation_id = canonical_fingerprint(
            {
                "kind": "brep-tessellation",
                "source_revision": source_revision,
                "policy_id": policy_id,
                "linear_deflection": linear_,
                "angular_deflection": angular_,
                "vertices": array_tree_fingerprint(self.vertices),
                "triangles": array_tree_fingerprint(self.triangles),
                "face_ids": array_tree_fingerprint(self.face_ids),
                "parameters": array_tree_fingerprint(self.parameters),
                "deviation_bounds": array_tree_fingerprint(self.deviation_bounds),
                "normal_bounds": array_tree_fingerprint(self.normal_bounds),
                "vertex_source_dimensions": array_tree_fingerprint(
                    self.vertex_source_dimensions
                ),
                "vertex_source_indices": array_tree_fingerprint(
                    self.vertex_source_indices
                ),
                "vertex_parameters": array_tree_fingerprint(self.vertex_parameters),
                "triangle_occurrence_ids": array_tree_fingerprint(
                    self.triangle_occurrence_ids
                ),
                "vertex_occurrence_ids": array_tree_fingerprint(
                    self.vertex_occurrence_ids
                ),
                "vertex_global_ids": array_tree_fingerprint(self.vertex_global_ids),
                "triangle_global_ids": array_tree_fingerprint(self.triangle_global_ids),
                "source_chart_witness": source_chart_witness.witness_id,
            }
        )


def _occurrence_ids(values: ArrayLike | None, count: int, name: str, /) -> np.ndarray:
    if values is None:
        return np.full((count,), -1, dtype=np.int32)
    identifiers = np.asarray(values)
    if (
        identifiers.shape != (count,)
        or not np.issubdtype(identifiers.dtype, np.integer)
        or np.any(identifiers < -1)
        or np.any(identifiers > np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name} must contain an occurrence ID or -1 per entry.")
    return identifiers.astype(np.int32, copy=False)


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


def _curve_identity(curve: BRepCurve, /) -> dict[str, object]:
    source_identity = getattr(curve, "identity_payload", None)
    return {
        "type": f"{type(curve).__module__}.{type(curve).__qualname__}",
        "structure": str(jax.tree_util.tree_structure(curve)),
        "arrays": array_tree_fingerprint(curve),
        "source": source_identity() if callable(source_identity) else None,
    }


def _index(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _root_pair(
    pair: tuple[RootEndpoint | None, RootEndpoint | None], /
) -> tuple[RootEndpoint | None, RootEndpoint | None]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("A source endpoint root pair must contain exactly two entries.")
    return pair[0], pair[1]


def _root_binding_payloads(
    vertices: tuple[BRepVertexRoot | None, ...],
    edges: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    coedges: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    /,
) -> tuple[object, object, object]:
    if (
        not any(root is not None for root in vertices)
        and not any(root is not None for pair in edges for root in pair)
        and not any(root is not None for pair in coedges for root in pair)
    ):
        return vertices, edges, coedges
    from ._intersection import endpoint_root_payload
    from ._root_bindings import vertex_root_payload

    def execution_payload(root: object, source: object, /) -> dict[str, object]:
        return {
            "source": source,
            "structure": str(jax.tree_util.tree_structure(root)),
            "arrays": array_tree_fingerprint(root),
        }

    return (
        tuple(
            None if root is None else execution_payload(root, vertex_root_payload(root))
            for root in vertices
        ),
        tuple(
            tuple(
                None
                if root is None
                else execution_payload(root, endpoint_root_payload(root))
                for root in pair
            )
            for pair in edges
        ),
        tuple(
            tuple(
                None
                if root is None
                else execution_payload(root, endpoint_root_payload(root))
                for root in pair
            )
            for pair in coedges
        ),
    )


def _same_recomputed_evidence(source: object, rebuilt: object, name: str, /) -> None:
    if (
        type(source) is not type(rebuilt)
        or jax.tree_util.tree_structure(source) != jax.tree_util.tree_structure(rebuilt)
        or array_tree_fingerprint(source) != array_tree_fingerprint(rebuilt)
    ):
        raise ValueError(f"{name} no longer matches its recomputed source evidence.")


def _validate_surface_carrier_source(patch: Any, cache: dict[int, Any], /) -> None:
    key = id(patch)
    if key not in cache:
        from ._patches import _validate_surface_source_shadows

        _validate_surface_source_shadows(patch)
        cache[key] = patch


def _validate_curve_carrier_source(curve: Any, cache: dict[int, Any], /) -> None:
    key = id(curve)
    if key not in cache:
        from ._patches import _validate_curve_source_shadows

        _validate_curve_source_shadows(curve)
        cache[key] = curve


def _rebuild_surface_region(region: Any, cache: dict[int, Any] | None = None, /) -> Any:
    cache = {} if cache is None else cache
    key = id(region)
    if key in cache:
        return cache[key]
    from ._intersection_curve import SurfaceRegion

    if not isinstance(region, SurfaceRegion):
        raise TypeError("A source root must retain its SurfaceRegion.")
    _validate_surface_carrier_source(region.patch, cache)
    rebuilt = SurfaceRegion(region.patch, region.parameter_box)
    _same_recomputed_evidence(region, rebuilt, "SurfaceRegion")
    cache[key] = rebuilt
    return rebuilt


# The key under which a validation cache keeps its proved branches by id.
_PROVED_BRANCHES = object()


def _rebuild_intersection_curve(
    curve: Any, cache: dict[int, Any] | None = None, /
) -> Any:
    cache = {} if cache is None else cache
    key = id(curve)
    if key in cache:
        return cache[key]
    from ._intersection_curve import IntersectionCurve

    if not isinstance(curve, IntersectionCurve):
        raise TypeError("A source branch witness must retain an IntersectionCurve.")
    # A copy of a branch proved earlier in this validation is checked against
    # that proof instead of being proved again.
    proved = cache.setdefault(id(_PROVED_BRANCHES), {}).get(curve.branch_id)
    if proved is not None:
        _same_recomputed_evidence(curve, proved, "IntersectionCurve")
        cache[key] = proved
        return proved
    first = _rebuild_surface_region(curve.first, cache)
    second = _rebuild_surface_region(curve.second, cache)
    rebuilt = IntersectionCurve(
        first,
        second,
        chart_axes=curve.chart_axes,
        chart_start=curve.chart_start,
        chart_end=curve.chart_end,
        chart_lower=curve.chart_lower,
        chart_upper=curve.chart_upper,
        node_axes=curve.node_axes,
        node_values=curve.node_values,
        node_lower=curve.node_lower,
        node_upper=curve.node_upper,
        node_period_shifts=curve.node_period_shifts,
        closed=curve.closed,
        start_kind=curve.start_kind,
        end_kind=curve.end_kind,
    )
    _same_recomputed_evidence(curve, rebuilt, "IntersectionCurve")
    cache[id(_PROVED_BRANCHES)][curve.branch_id] = rebuilt
    cache[key] = rebuilt
    return rebuilt


def _rebuild_curve_carrier(curve: Any, cache: dict[int, Any] | None = None, /) -> Any:
    cache = {} if cache is None else cache
    key = id(curve)
    if key in cache:
        return cache[key]
    from ._intersection_curve import IntersectionCurve, IntersectionPCurve

    if isinstance(curve, IntersectionPCurve):
        branch = _rebuild_intersection_curve(curve.curve, cache)
        rebuilt = IntersectionPCurve(
            branch,
            curve.side,
            first=curve.first,
            last=curve.last,
            reversed=curve.reversed,
        )
        _same_recomputed_evidence(curve, rebuilt, "IntersectionPCurve")
        cache[key] = rebuilt
        return rebuilt
    if isinstance(curve, IntersectionCurve):
        return _rebuild_intersection_curve(curve, cache)
    _validate_curve_carrier_source(curve, cache)
    return curve


def _rebuild_conic_root_line_endpoint(
    witness: ConicRootLineEndpointWitness, cache: dict[int, Any], /
) -> ConicRootLineEndpointWitness:
    key = id(witness)
    if key in cache:
        return cache[key]
    rebuilt = ConicRootLineEndpointWitness(
        _rebuild_curve_carrier(witness.source_curve, cache),
        _rebuild_curve_carrier(witness.source_pcurve, cache),
        _rebuild_surface_region(witness.source_region, cache),
        witness.parameter_exact,
        witness.source_interval,
        _rebuild_curve_carrier(witness.trim_curve, cache),
        witness.trim_interval,
    )
    _same_recomputed_evidence(witness, rebuilt, "ConicRootLineEndpointWitness")
    cache[key] = rebuilt
    return rebuilt


def _rebuild_intersection_root(root: Any, cache: dict[int, Any] | None = None, /) -> Any:
    cache = {} if cache is None else cache
    key = id(root)
    if key in cache:
        return cache[key]
    from ._intersection import (
        CurveSurfaceIntersectionRoot,
        ExactLineCylinderTangency,
        IntersectionCurvePointRoot,
        TrimIntersectionRoot,
        TripleSurfaceIntersectionRoot,
    )
    from ._patches import ConicParameterRoot

    if isinstance(root, TrimIntersectionRoot):
        first = _rebuild_curve_carrier(root.first, cache)
        second = _rebuild_curve_carrier(root.second, cache)
        endpoint_witness = (
            None
            if root.endpoint_witness is None
            else _rebuild_conic_root_line_endpoint(root.endpoint_witness, cache)
        )
        rebuilt = TrimIntersectionRoot(
            first,
            second,
            parameter_lower=root.parameter_lower,
            parameter_upper=root.parameter_upper,
            endpoint_witness=endpoint_witness,
        )
    elif isinstance(root, CurveSurfaceIntersectionRoot):
        curve = _rebuild_curve_carrier(root.curve, cache)
        surface = _rebuild_surface_region(root.surface, cache)
        rebuilt = CurveSurfaceIntersectionRoot(
            curve,
            surface,
            parameter_lower=root.parameter_lower,
            parameter_upper=root.parameter_upper,
            endpoint_witness=(
                None
                if root.endpoint_witness is None
                else _rebuild_conic_root_line_endpoint(root.endpoint_witness, cache)
            ),
            tangent_witness=(
                None
                if root.tangent_witness is None
                else ExactLineCylinderTangency(curve, surface)
            ),
        )
    elif isinstance(root, TripleSurfaceIntersectionRoot):
        first = _rebuild_surface_region(root.first, cache)
        second = _rebuild_surface_region(root.second, cache)
        third = _rebuild_surface_region(root.third, cache)
        rebuilt = TripleSurfaceIntersectionRoot(
            first,
            second,
            third,
            parameter_lower=root.parameter_lower,
            parameter_upper=root.parameter_upper,
        )
    elif isinstance(root, IntersectionCurvePointRoot):
        curve = _rebuild_intersection_curve(root.curve, cache)
        rebuilt = IntersectionCurvePointRoot(curve, root.parameter)
    elif isinstance(root, ConicParameterRoot):
        conic = _rebuild_curve_carrier(root.conic, cache)
        rebuilt = ConicParameterRoot(
            conic, root.cosine_exact, root.sine_exact, turns=root.turns
        )
    else:
        raise TypeError(f"Unsupported source-root record: {type(root)!r}.")
    _same_recomputed_evidence(root, rebuilt, type(root).__name__)
    cache[key] = rebuilt
    return rebuilt


def _rebuild_root_support(root: Any, cache: dict[int, Any] | None = None, /) -> Any:
    cache = {} if cache is None else cache
    key = id(root)
    if key in cache:
        return cache[key]
    from ._root_bindings import BRepRootSupport

    if not isinstance(root, BRepRootSupport):
        raise TypeError("A source support witness must be a BRepRootSupport.")
    _validate_surface_carrier_source(root.patch, cache)
    rebuilt = BRepRootSupport(
        root.patch,
        _rebuild_intersection_root(root.root, cache),
    )
    _same_recomputed_evidence(root, rebuilt, "BRepRootSupport")
    cache[key] = rebuilt
    return rebuilt


def _rebuild_vertex_root(
    root: BRepVertexRoot | None, cache: dict[int, Any] | None = None, /
) -> BRepVertexRoot | None:
    if root is None:
        return None
    cache = {} if cache is None else cache
    key = id(root)
    if key in cache:
        return cache[key]
    from ._intersection_curve import SurfaceRegion
    from ._root_bindings import (
        BRepBranchNodeRoots,
        BRepConicVertex,
        BRepCurveSurfaceLift,
        BRepPlacedVertex,
        BRepRootSupport,
        BRepVertexRoot,
    )

    def primary(value: Any, /) -> Any:
        value_key = id(value)
        if value_key in cache:
            return cache[value_key]
        if isinstance(value, BRepRootSupport):
            return _rebuild_root_support(value, cache)
        if isinstance(value, BRepConicVertex):
            rebuilt_primary = BRepConicVertex(
                _rebuild_intersection_root(value.root, cache),
                offset_exact=value.offset_exact,
            )
        elif isinstance(value, BRepPlacedVertex):
            from ._patches import _validate_placed_source_shadows

            _validate_placed_source_shadows(
                value.rotation, value.translation, value.affine
            )
            nested = _rebuild_vertex_root(value.source_root, cache)
            rebuilt_primary = BRepPlacedVertex(
                value.source_point,
                value.rotation,
                value.translation,
                value.source_entity_id,
                source_root=nested,
                affine=value.affine,
            )
        else:
            return _rebuild_intersection_root(value, cache)
        _same_recomputed_evidence(value, rebuilt_primary, type(value).__name__)
        cache[value_key] = rebuilt_primary
        return rebuilt_primary

    aliases = tuple(_rebuild_root_support(alias, cache) for alias in root.aliases)
    spatial = (
        None
        if root.spatial_root is None
        else _rebuild_intersection_root(root.spatial_root, cache)
    )
    joint = (
        None
        if root.joint_root is None
        else _rebuild_intersection_root(root.joint_root, cache)
    )
    lifts = []
    for lift in root.source_edge_lifts:
        lift_key = id(lift)
        if lift_key in cache:
            lifts.append(cache[lift_key])
            continue
        curve = _rebuild_curve_carrier(lift.curve, cache)
        _validate_surface_carrier_source(lift.surface, cache)
        pcurve = _rebuild_curve_carrier(lift.pcurve, cache)
        region = SurfaceRegion(
            lift.surface, np.asarray(lift.parameter_box, dtype=np.float64)
        )
        rebuilt_lift = BRepCurveSurfaceLift(
            curve,
            lift.surface,
            pcurve,
            lift.first,
            lift.last,
            source_region=region,
        )
        _same_recomputed_evidence(lift, rebuilt_lift, "BRepCurveSurfaceLift")
        cache[lift_key] = rebuilt_lift
        lifts.append(rebuilt_lift)
    branch_node = None
    if root.branch_node is not None:
        from ._intersection import BranchRootEndpoint

        endpoints = []
        for endpoint in root.branch_node.endpoints:
            rebuilt_endpoint = _rebuild_endpoint_root(endpoint, cache)
            if not isinstance(rebuilt_endpoint, BranchRootEndpoint):
                raise TypeError("A branch-node witness must keep branch endpoints.")
            endpoints.append(rebuilt_endpoint)
        branch_node = BRepBranchNodeRoots(
            _rebuild_intersection_root(root.branch_node.node, cache), tuple(endpoints)
        )
        _same_recomputed_evidence(root.branch_node, branch_node, "BRepBranchNodeRoots")
    rebuilt = BRepVertexRoot(
        primary(root.primary),
        aliases=aliases,
        spatial_root=spatial,
        joint_root=joint,
        source_edge_lifts=tuple(lifts),
        branch_node=branch_node,
    )
    _same_recomputed_evidence(root, rebuilt, "BRepVertexRoot")
    cache[key] = rebuilt
    return rebuilt


def _rebuild_endpoint_root(
    root: RootEndpoint | None, cache: dict[int, Any] | None = None, /
) -> RootEndpoint | None:
    if root is None:
        return None
    cache = {} if cache is None else cache
    key = id(root)
    if key in cache:
        return cache[key]
    from ._intersection import (
        BranchRootEndpoint,
        ConicParameterEndpoint,
        NativePeriodEndpoint,
        TrimRootEndpoint,
    )

    if isinstance(root, TrimRootEndpoint):
        root_definition = _rebuild_intersection_root(root.root, cache)
        source_pcurve = (
            None
            if root.source_pcurve is None
            else _rebuild_curve_carrier(root.source_pcurve, cache)
        )
        source_surface = (
            None
            if root.source_surface is None
            else _rebuild_surface_region(root.source_surface, cache)
        )
        rebuilt = TrimRootEndpoint(
            root_definition,
            root.operand,
            affine_parameter_offset=root.affine_parameter_offset,
            affine_parameter_scale=root.affine_parameter_scale,
            affine_transforms=root.affine_transforms,
            source_pcurve=source_pcurve,
            source_surface=source_surface,
        )
    elif isinstance(root, BranchRootEndpoint):
        root_definition = _rebuild_intersection_root(root.root, cache)
        curve = _rebuild_intersection_curve(root.curve, cache)
        source_pcurve = (
            None
            if root.source_pcurve is None
            else _rebuild_curve_carrier(root.source_pcurve, cache)
        )
        rebuilt = BranchRootEndpoint(
            root_definition,
            curve,
            root.chart,
            source_pcurve=source_pcurve,
            source_side=root.source_side,
            source_first=root.source_first,
            source_last=root.source_last,
            periodic_shifts=root.periodic_shifts,
            affine_transforms=root.affine_transforms,
        )
    elif isinstance(root, NativePeriodEndpoint):
        curve = _rebuild_curve_carrier(root.curve, cache)
        patch = root.patch
        if patch is not None:
            _validate_surface_carrier_source(patch, cache)
        rebuilt = NativePeriodEndpoint(
            curve,
            patch,
            root.axis,
            rational=root.rational,
            turns=root.turns,
        )
    elif isinstance(root, ConicParameterEndpoint):
        conic_root = _rebuild_intersection_root(root.root, cache)
        carrier = _rebuild_curve_carrier(root.carrier, cache)
        rebuilt = ConicParameterEndpoint(
            conic_root,
            carrier,
            root.parameter_exact,
            root.point_exact,
        )
    else:
        raise TypeError(f"Unsupported source endpoint root: {type(root)!r}.")
    _same_recomputed_evidence(root, rebuilt, type(root).__name__)
    cache[key] = rebuilt
    return rebuilt


def _validate_geometry_source_carriers(
    curves: tuple[BRepCurve, ...],
    pcurves: tuple[BRepCurve, ...],
    vertex_roots: tuple[BRepVertexRoot | None, ...],
    edge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    coedge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    /,
) -> None:
    cache: dict[int, Any] = {}
    for curve in (*curves, *pcurves):
        _rebuild_curve_carrier(curve, cache)
    for root in vertex_roots:
        _rebuild_vertex_root(root, cache)
    for pair in (*edge_roots, *coedge_roots):
        for root in pair:
            _rebuild_endpoint_root(root, cache)


def _brep_geometry_identity(geometry: BRepGeometry, /) -> str:
    root_payloads = _root_binding_payloads(
        geometry.vertex_roots,
        geometry.edge_endpoint_roots,
        geometry.coedge_endpoint_roots,
    )
    return canonical_fingerprint(
        {
            "kind": "brep-source-geometry",
            "vertex_points": array_tree_fingerprint(geometry.vertex_points),
            "vertex_exact_points": tuple(
                None
                if point is None
                else tuple((value.numerator, value.denominator) for value in point)
                for point in geometry.vertex_exact_points
            ),
            "vertex_evaluation_bounds": array_tree_fingerprint(
                geometry.vertex_evaluation_bounds
            ),
            "vertex_roots": root_payloads[0],
            "edge_endpoint_roots": root_payloads[1],
            "coedge_endpoint_roots": root_payloads[2],
            "edge_vertex_gap_bounds": geometry.edge_vertex_gap_bounds,
            "edge_face_gap_bounds": geometry.edge_face_gap_bounds,
            "curves": tuple(_curve_identity(curve) for curve in geometry.curves),
            "edge_ranges": array_tree_fingerprint(geometry.edge_ranges),
            "edge_curves": geometry.edge_curves,
            "edge_vertices": geometry.edge_vertices,
            "pcurves": tuple(_curve_identity(curve) for curve in geometry.pcurves),
            "coedge_edges": geometry.coedge_edges,
            "coedge_senses": geometry.coedge_senses,
            "face_loops": geometry.face_loops,
            "shell_faces": geometry.shell_faces,
            "shell_orientations": geometry.shell_orientations,
            "solid_shells": geometry.solid_shells,
            "topology": {
                "face_edges": geometry.topology.face_edges,
                "edge_faces": geometry.topology.edge_faces,
                "face_wires": geometry.topology.face_wires,
                "solid_faces": geometry.topology.solid_faces,
                "solid_face_orientations": geometry.topology.solid_face_orientations,
                "face_solids": geometry.topology.face_solids,
                "num_vertices": geometry.topology.num_vertices,
            },
            "assembly": {
                "source": geometry.assembly_graph.identity_payload(),
                "structure": str(jax.tree_util.tree_structure(geometry.assembly_graph)),
                "arrays": array_tree_fingerprint(geometry.assembly_graph),
            },
        }
    )


def _validate_root_bindings(
    points: np.ndarray,
    exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
    ranges: np.ndarray,
    curves: tuple[BRepCurve, ...],
    edge_curves: tuple[int, ...],
    pcurves: tuple[BRepCurve, ...],
    edge_vertices: tuple[tuple[int, int], ...],
    coedge_edges: tuple[int, ...],
    vertex_roots: tuple[BRepVertexRoot | None, ...],
    edge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    coedge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    /,
) -> np.ndarray:
    errors = np.zeros((points.shape[0],), dtype=np.float64)
    periodic_closed_edges = tuple(
        edge
        for edge, curve_index in enumerate(edge_curves)
        if edge_vertices[edge][0] == edge_vertices[edge][1]
        and curve_index >= 0
        and curves[curve_index].native_period_symbol() == "two_pi"
    )
    if (
        not any(root is not None for root in vertex_roots)
        and not any(root is not None for pair in edge_roots for root in pair)
        and not any(root is not None for pair in coedge_roots for root in pair)
        and not periodic_closed_edges
    ):
        return errors
    from ._carrier_pieces import carrier_payload
    from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint, RootEndpoint
    from ._patches import CircleCurve, EllipseCurve
    from ._root_bindings import BRepPlacedVertex, BRepVertexRoot

    for vertex, root in enumerate(vertex_roots):
        if root is None:
            continue
        if not isinstance(root, BRepVertexRoot):
            raise TypeError("vertex_roots must contain BRepVertexRoot values or None.")
        enclosure = np.asarray(root.point_enclosure(), dtype=np.float64)
        if (
            enclosure.shape != (2, 3)
            or not np.all(np.isfinite(enclosure))
            or np.any(enclosure[1] < enclosure[0])
        ):
            raise ValueError(
                "A vertex root must provide a finite ordered 3D point enclosure."
            )
        coordinate_error = np.max(
            np.nextafter(np.abs(enclosure - points[vertex]), np.inf), axis=0
        )
        error = float(np.nextafter(hypot(*coordinate_error), np.inf))
        if not np.isfinite(error):
            raise ValueError("A vertex root has no finite numeric realization bound.")
        errors[vertex] = error

    def literal_placed_vertex(root: BRepVertexRoot | None, /) -> bool:
        return bool(
            root is not None
            and isinstance(root.primary, BRepPlacedVertex)
            and root.primary.source_root is None
        )

    def conic_endpoint_curve_point(
        root: ConicParameterEndpoint,
        curve: BRepCurve,
        edge_parameter_exact: Fraction | None = None,
        /,
    ) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
        from math import comb

        from ..._interpolation import rational_bspline_curve_spans
        from .._rational import rational_vector
        from ._intersection_curve import exact_curve_parameter_point
        from ._patches import (
            BSplineCurve,
            CircleCurve,
            ConicRootLinePCurve,
            CylinderPatch,
            ExactLineCurve,
            ExtrusionSurface,
            LineCurve,
            OffsetSurface,
            RevolutionSurface,
            SurfaceIsoparametricCurve,
            SurfaceMappedCurve,
        )
        from ._placed import PlacedCurve

        if isinstance(curve, PlacedCurve):
            source_point = conic_endpoint_curve_point(
                root, curve.definition, edge_parameter_exact
            )
            return None if source_point is None else curve.affine.apply(source_point)

        if (
            isinstance(curve, SurfaceMappedCurve)
            and isinstance(curve.surface, OffsetSurface)
            and isinstance(curve.surface.base, RevolutionSurface)
            and isinstance(curve.pcurve, LineCurve)
            and root.parameter_exact is not None
        ):
            if curve.pcurve.origin_exact != (
                Fraction(0),
                Fraction(0),
            ) or curve.pcurve.direction_exact != (Fraction(0), Fraction(1)):
                return None
            from ._root_bindings import revolution_offset_conic_frame

            frame = revolution_offset_conic_frame(curve.surface, root.parameter_exact)
            if frame is None:
                return None
            point, center, first_axis, second_axis, radius = frame
            conic = root.root.conic
            if (
                not isinstance(conic, CircleCurve)
                or conic.center_exact != center
                or conic.first_axis_exact != first_axis
                or conic.second_axis_exact != second_axis
                or conic.radius_exact != radius
                or root.root.cosine_exact != 1
                or root.root.sine_exact != 0
                or root.point_exact != point
            ):
                return None
            return point
        if (
            isinstance(curve, SurfaceIsoparametricCurve)
            and isinstance(curve.surface, OffsetSurface)
            and isinstance(curve.surface.base, RevolutionSurface)
            and isinstance(curve.surface.base.curve, CircleCurve)
            and curve.fixed_axis == 0
        ):
            surface = curve.surface
            if not isinstance(surface, OffsetSurface):
                return None
            base = surface.base
            if not isinstance(base, RevolutionSurface) or not isinstance(
                base.curve, CircleCurve
            ):
                return None
            profile = base.curve
            witness = surface.exact_sphere_revolution_witness()
            if witness is None:
                return None
            center, radius, axis, frame_sign = witness
            period = surface.periods[0]
            if (
                len(center) != 3
                or len(axis) != 3
                or frame_sign not in (-1, 1)
                or surface.native_period_symbols()[0] != "two_pi"
                or period is None
                or curve.fixed_value not in (0.0, float(period))
                or profile.center_exact != center
                or profile.second_axis_exact != axis
                and tuple(-value for value in profile.second_axis_exact) != axis
            ):
                return None
            offset_radius = radius + frame_sign * surface.distance_exact
            conic = root.root.conic
            if (
                offset_radius <= 0
                or not isinstance(conic, CircleCurve)
                or conic.center_exact != center
                or conic.first_axis_exact != profile.first_axis_exact
                or conic.second_axis_exact != profile.second_axis_exact
                or conic.radius_exact != offset_radius
                or root.root.cosine_exact != 0
                or root.root.sine_exact not in (-1, 1)
                or root.root.exact_point() != root.point_exact
            ):
                return None
            expected_point: tuple[ExactScalar, ExactScalar, ExactScalar] = (
                center[0]
                + offset_radius * root.root.sine_exact * profile.second_axis_exact[0],
                center[1]
                + offset_radius * root.root.sine_exact * profile.second_axis_exact[1],
                center[2]
                + offset_radius * root.root.sine_exact * profile.second_axis_exact[2],
            )
            if expected_point != root.point_exact:
                return None
            return expected_point
        if carrier_payload(root.root.conic) == carrier_payload(curve):
            first, second, third = root.root.exact_point()
            return first, second, third
        if isinstance(curve, CircleCurve):
            point = root.point_exact
            if curve.radius_exact == 0:
                return None
            first = curve.first_axis_exact
            second = curve.second_axis_exact
            delta = tuple(point[axis] - curve.center_exact[axis] for axis in range(3))

            def dot(
                left: tuple[ExactScalar, ...],
                right: tuple[ExactScalar, ...],
                /,
            ) -> ExactScalar:
                return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

            g00, g01, g11 = dot(first, first), dot(first, second), dot(second, second)
            determinant = g00 * g11 - g01 * g01
            if determinant <= 0:
                return None
            first_coordinate = dot(delta, first)
            second_coordinate = dot(delta, second)
            cosine = (first_coordinate * g11 - second_coordinate * g01) / (
                determinant * curve.radius_exact
            )
            sine = (second_coordinate * g00 - first_coordinate * g01) / (
                determinant * curve.radius_exact
            )
            reconstructed = tuple(
                curve.radius_exact * (cosine * first[axis] + sine * second[axis])
                for axis in range(3)
            )
            if cosine * cosine + sine * sine != 1 or reconstructed != delta:
                return None
            return point[0], point[1], point[2]
        parameter_exact = (
            root.parameter_exact
            if root.parameter_exact is not None
            else edge_parameter_exact
        )
        if parameter_exact is not None:
            source_point = exact_curve_parameter_point(curve, parameter_exact)
            if source_point is not None and tuple(source_point) == root.point_exact:
                return source_point[0], source_point[1], source_point[2]
        if not isinstance(curve, SurfaceMappedCurve):
            return None
        pcurve = curve.pcurve
        if (
            not isinstance(pcurve, ConicRootLinePCurve)
            or pcurve.phase_shift_exact != 0
            or pcurve.root.identity_payload() != root.root.identity_payload()
            or root.parameter_exact is None
        ):
            return None
        height = (
            pcurve.height_origin_exact
            + pcurve.height_direction_exact * root.parameter_exact
        )
        conic = root.root.conic

        def cylinder_lift(
            patch: CylinderPatch, exact_radius: ExactScalar, /
        ) -> tuple[ExactScalar, ExactScalar, ExactScalar] | None:
            if not isinstance(conic, CircleCurve) or (
                conic.center_exact != patch.origin_exact
                or conic.first_axis_exact != patch.first_axis_exact
                or conic.second_axis_exact != patch.second_axis_exact
                or conic.radius_exact != exact_radius
            ):
                return None
            first, second, third = (
                value + height * axis
                for value, axis in zip(
                    root.root.exact_point(), patch.axis_exact, strict=True
                )
            )
            return first, second, third

        if isinstance(curve.surface, CylinderPatch):
            return cylinder_lift(curve.surface, curve.surface.radius_exact)
        if isinstance(curve.surface, OffsetSurface) and isinstance(
            curve.surface.base, CylinderPatch
        ):
            base = curve.surface.base
            return cylinder_lift(
                base,
                base.radius_exact
                + curve.surface.frame_sign * curve.surface.distance_exact,
            )
        if isinstance(curve.surface, ExtrusionSurface):
            if carrier_payload(conic) != carrier_payload(curve.surface.curve):
                return None
            direction = rational_vector(np.asarray(curve.surface.direction))
            first, second, third = (
                value + height * axis
                for value, axis in zip(root.root.exact_point(), direction, strict=True)
            )
            return first, second, third
        if not isinstance(curve.surface, RevolutionSurface):
            return None

        profile = curve.surface.curve
        if isinstance(profile, ExactLineCurve):
            profile_point = profile.evaluate_exact(height)
        elif isinstance(profile, LineCurve):
            origin = profile.origin_exact
            direction = profile.direction_exact
            profile_point = tuple(
                first + height * delta
                for first, delta in zip(origin, direction, strict=True)
            )
        elif isinstance(profile, BSplineCurve):
            try:
                spans = rational_bspline_curve_spans(
                    profile.exact_control_points,
                    profile.exact_weights,
                    profile.exact_knots,
                    profile.degree,
                )
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            selected = next(
                (
                    span
                    for span in spans
                    if span.parameter_bounds[0] <= height < span.parameter_bounds[1]
                ),
                spans[-1] if spans and height == spans[-1].parameter_bounds[1] else None,
            )
            if selected is None:
                return None
            first, last = selected.parameter_bounds
            relative = (height - first) / (last - first)
            degree = selected.degree
            coefficients = selected.homogeneous_bernstein_coefficients
            homogeneous = tuple(
                sum(
                    (
                        row[axis]
                        * comb(degree, index)
                        * relative**index
                        * (1 - relative) ** (degree - index)
                        for index, row in enumerate(coefficients)
                    ),
                    Fraction(0),
                )
                for axis in range(len(coefficients[0]))
            )
            if homogeneous[-1] <= 0:
                return None
            profile_point = tuple(value / homogeneous[-1] for value in homogeneous[:-1])
        else:
            return None

        axis_origin = rational_vector(np.asarray(curve.surface.axis_origin))
        axis = rational_vector(np.asarray(curve.surface.axis_direction))
        if sum((value * value for value in axis), Fraction(0)) != 1:
            return None
        relative = tuple(
            point - origin
            for point, origin in zip(profile_point, axis_origin, strict=True)
        )
        axial = sum(
            (value * direction for value, direction in zip(relative, axis, strict=True)),
            Fraction(0),
        )
        center = tuple(
            origin + axial * direction
            for origin, direction in zip(axis_origin, axis, strict=True)
        )
        radial = tuple(
            value - axial * direction
            for value, direction in zip(relative, axis, strict=True)
        )
        if not isinstance(conic, CircleCurve) or conic.center_exact != center:
            return None
        first_radial = tuple(
            conic.radius_exact * value for value in conic.first_axis_exact
        )
        second_radial = tuple(
            conic.radius_exact * value for value in conic.second_axis_exact
        )
        expected_second = (
            axis[1] * radial[2] - axis[2] * radial[1],
            axis[2] * radial[0] - axis[0] * radial[2],
            axis[0] * radial[1] - axis[1] * radial[0],
        )
        if first_radial != radial or second_radial != expected_second:
            return None
        first, second, third = root.root.exact_point()
        return first, second, third

    def full_period_pair(
        roots: tuple[object | None, object | None], carrier: BRepCurve, /
    ) -> bool:
        if all(isinstance(root, NativePeriodEndpoint) for root in roots):
            first, last = roots
            assert isinstance(first, NativePeriodEndpoint)
            assert isinstance(last, NativePeriodEndpoint)
            same_patch = (first.patch is None and last.patch is None) or (
                first.patch is not None
                and last.patch is not None
                and carrier_payload(first.patch) == carrier_payload(last.patch)
            )
            return bool(
                carrier_payload(first.curve) == carrier_payload(carrier)
                and carrier_payload(last.curve) == carrier_payload(carrier)
                and same_patch
                and first.axis == last.axis
                and first.rational == last.rational
                and last.turns - first.turns == 1
            )
        if not all(isinstance(root, ConicParameterEndpoint) for root in roots):
            return False
        first, last = roots
        assert isinstance(first, ConicParameterEndpoint)
        assert isinstance(last, ConicParameterEndpoint)
        first_root, last_root = first.root, last.root
        return bool(
            isinstance(carrier, (CircleCurve, EllipseCurve))
            and first.parameter_exact is None
            and last.parameter_exact is None
            and carrier_payload(first.carrier) == carrier_payload(carrier)
            and carrier_payload(last.carrier) == carrier_payload(carrier)
            and carrier_payload(first_root.conic) == carrier_payload(carrier)
            and carrier_payload(last_root.conic) == carrier_payload(carrier)
            and first_root.cosine_exact == last_root.cosine_exact
            and first_root.sine_exact == last_root.sine_exact
            and last_root.turns - first_root.turns == 1
            and first.point_exact == last.point_exact
            and first.point_exact == first_root.exact_point()
            and last.point_exact == last_root.exact_point()
        )

    def validate_endpoints(
        edge: int,
        roots: tuple[object | None, object | None],
        /,
        *,
        coedge: int | None = None,
    ) -> None:
        if not isinstance(roots, tuple) or len(roots) != 2:
            raise TypeError("Endpoint root bindings must be ordered start/end pairs.")
        for endpoint, root in enumerate(roots):
            vertex = edge_vertices[edge][endpoint]
            vertex_root = vertex_roots[vertex]
            if root is None:
                if vertex_root is not None and not literal_placed_vertex(vertex_root):
                    raise ValueError(
                        "A source-root vertex requires a binding on every incident edge and coedge endpoint."
                    )
                continue
            if isinstance(root, NativePeriodEndpoint):
                if coedge is not None:
                    carriers = (pcurves[coedge],)
                elif edge_curves[edge] >= 0:
                    carriers = (curves[edge_curves[edge]],)
                else:
                    carriers = tuple(
                        pcurves[index]
                        for index, used_edge in enumerate(coedge_edges)
                        if used_edge == edge
                    )
                if not any(
                    carrier_payload(root.curve) == carrier_payload(carrier)
                    for carrier in carriers
                ):
                    raise ValueError(
                        "A native-period endpoint must bind its exact source carrier."
                    )
            if isinstance(root, ConicParameterEndpoint):
                if coedge is None:
                    if edge_curves[edge] < 0:
                        raise ValueError(
                            "A conic source endpoint requires a nondegenerate edge carrier."
                        )
                    carrier = curves[edge_curves[edge]]
                else:
                    carrier = pcurves[coedge]
                if carrier_payload(root.carrier) != carrier_payload(carrier):
                    raise ValueError(
                        "A conic endpoint must bind its exact edge or coedge carrier."
                    )
                if (
                    exact_points[vertex] is not None
                    and exact_points[vertex] != root.point_exact
                ):
                    raise ValueError(
                        "A conic endpoint must retain the exact point bound to its vertex."
                    )
                if (
                    root.parameter_exact is not None
                    and Fraction.from_float(float(ranges[edge, endpoint]))
                    != root.parameter_exact
                ):
                    raise ValueError(
                        "A conic endpoint exact parameter must equal its source edge endpoint."
                    )
                if (
                    coedge is None
                    and conic_endpoint_curve_point(
                        root,
                        carrier,
                        Fraction.from_float(float(ranges[edge, endpoint])),
                    )
                    != root.point_exact
                ):
                    raise ValueError(
                        "A conic edge endpoint must be proved by its exact source curve."
                    )
            if not isinstance(root, RootEndpoint):
                raise TypeError("Endpoint roots must be RootEndpoint values or None.")
            if not isinstance(root, NativePeriodEndpoint):
                if vertex_root is None or not vertex_root.supports_endpoint(root):
                    raise ValueError(
                        "An endpoint root must be supported by its vertex source root."
                    )
            lower, upper = root.parameter_enclosure()
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower > upper
                or not lower <= ranges[edge, endpoint] <= upper
            ):
                raise ValueError(
                    "A numerical edge endpoint must lie in its exact source-root enclosure."
                )

    for edge, roots in enumerate(edge_roots):
        validate_endpoints(edge, roots)
    for coedge, roots in enumerate(coedge_roots):
        validate_endpoints(coedge_edges[coedge], roots, coedge=coedge)
    for edge in periodic_closed_edges:
        carrier = curves[edge_curves[edge]]
        if not full_period_pair(edge_roots[edge], carrier):
            raise ValueError(
                "A closed periodic source edge requires an exact conic or native-period turn pair."
            )
    return errors


def _bspline_revolution_box(
    patch: AbstractSurfacePatch,
    lower_uv: tuple[float, float],
    upper_uv: tuple[float, float],
    /,
    *,
    source_curve: BRepCurve | None = None,
    root: object | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Enclose a B-spline revolution or its analytic normal offset by spans."""
    from ..._interpolation import rational_bspline_curve_spans
    from .._interval_enclosure import Interval, interval_cos, interval_sin
    from .._rational import fraction_down, fraction_up, rational_vector
    from ._carrier_pieces import carrier_payload
    from ._exact import square_root_interval
    from ._intersection import NativePeriodEndpoint
    from ._patches import (
        BSplineCurve,
        OffsetSurface,
        RevolutionSurface,
        SurfaceIsoparametricCurve,
    )

    if isinstance(patch, OffsetSurface):
        offset = patch
        revolution = patch.base
    elif isinstance(patch, RevolutionSurface):
        offset = None
        revolution = patch
    else:
        return None
    if not isinstance(revolution, RevolutionSurface) or not isinstance(
        revolution.curve, BSplineCurve
    ):
        return None
    profile = revolution.curve
    axis = rational_vector(np.asarray(revolution.axis_direction))
    origin = rational_vector(np.asarray(revolution.axis_origin))

    def vector_dot(
        first_: tuple[Fraction, ...], second_: tuple[tuple[Fraction, Fraction], ...]
    ) -> tuple[Fraction, Fraction]:
        lower = Fraction(0)
        upper = Fraction(0)
        for coefficient, (lo, hi) in zip(first_, second_, strict=True):
            if coefficient >= 0:
                lower += coefficient * lo
                upper += coefficient * hi
            else:
                lower += coefficient * hi
                upper += coefficient * lo
        return lower, upper

    def interval_add(
        first_: tuple[Fraction, Fraction], second_: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        return first_[0] + second_[0], first_[1] + second_[1]

    def interval_neg(value: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
        return -value[1], -value[0]

    def interval_subtract(
        first_: tuple[Fraction, Fraction], second_: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        return interval_add(first_, interval_neg(second_))

    def interval_multiply(
        first_: tuple[Fraction, Fraction], second_: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        products = tuple(a * b for a in first_ for b in second_)
        return min(products), max(products)

    def interval_scale(
        coefficient: Fraction, value: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        return (
            (coefficient * value[0], coefficient * value[1])
            if coefficient >= 0
            else (coefficient * value[1], coefficient * value[0])
        )

    def vector_cross(
        first_: tuple[tuple[Fraction, Fraction], ...],
        second_: tuple[tuple[Fraction, Fraction], ...],
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple(
            interval_subtract(
                interval_multiply(first_[left], second_[right]),
                interval_multiply(first_[right], second_[left]),
            )
            for left, right in ((1, 2), (2, 0), (0, 1))
        )

    def vector_rotate(
        value: tuple[tuple[Fraction, Fraction], ...],
        cosine: tuple[Fraction, Fraction],
        sine: tuple[Fraction, Fraction],
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        axial = vector_dot(axis, value)
        cross = vector_cross(tuple((item, item) for item in axis), value)
        one_minus_cosine = interval_subtract((Fraction(1), Fraction(1)), cosine)
        return tuple(
            interval_add(
                interval_add(
                    interval_multiply(component, cosine),
                    interval_multiply(cross[index], sine),
                ),
                interval_multiply(
                    (axis[index] * axial[0], axis[index] * axial[1]),
                    one_minus_cosine,
                ),
            )
            for index, component in enumerate(value)
        )

    if isinstance(root, NativePeriodEndpoint):
        if (
            not isinstance(source_curve, SurfaceIsoparametricCurve)
            or source_curve.fixed_axis != 1
            or carrier_payload(root.curve) != carrier_payload(source_curve)
            or root.patch is None
            and (root.axis is not None or source_curve.native_period_symbol() != "two_pi")
            or root.patch is not None
            and (root.axis != 0 or carrier_payload(root.patch) != carrier_payload(patch))
        ):
            return None
        if root.turns.denominator == 1:
            angle_lower = fraction_down(root.rational)
            angle_upper = fraction_up(root.rational)
        else:
            angle_lower, angle_upper = root.parameter_enclosure()
            angle_lower = float(np.nextafter(angle_lower, -np.inf))
            angle_upper = float(np.nextafter(angle_upper, np.inf))
    else:
        angle_lower = float(np.nextafter(lower_uv[0], -np.inf))
        angle_upper = float(np.nextafter(upper_uv[0], np.inf))
    if (
        not np.isfinite(angle_lower)
        or not np.isfinite(angle_upper)
        or angle_lower > angle_upper
    ):
        return None

    def trig_bounds(
        function: Callable[[Interval], Interval],
    ) -> tuple[Fraction, Fraction]:
        lower, upper = function(
            (
                np.asarray(float(angle_lower), dtype=np.float64),
                np.asarray(float(angle_upper), dtype=np.float64),
            )
        )
        return Fraction.from_float(float(lower)), Fraction.from_float(float(upper))

    cosine = trig_bounds(interval_cos)
    sine = trig_bounds(interval_sin)
    try:
        spans = rational_bspline_curve_spans(
            profile.exact_control_points,
            profile.exact_weights,
            profile.exact_knots,
            profile.degree,
        )
        source_profile_lower = Fraction.from_float(float(lower_uv[1]))
        source_profile_upper = Fraction.from_float(float(upper_uv[1]))
        domain_lower, domain_upper = (
            spans[0].parameter_bounds[0],
            spans[-1].parameter_bounds[1],
        )
        if (
            source_profile_lower > source_profile_upper
            or source_profile_lower < domain_lower
            or source_profile_upper > domain_upper
        ):
            return None
        profile_lower = max(
            Fraction.from_float(float(np.nextafter(lower_uv[1], -np.inf))),
            domain_lower,
        )
        profile_upper = min(
            Fraction.from_float(float(np.nextafter(upper_uv[1], np.inf))),
            domain_upper,
        )
        if profile_lower > profile_upper:
            return None
        selected_list = []
        for span in spans:
            first_profile = max(profile_lower, span.parameter_bounds[0])
            last_profile = min(profile_upper, span.parameter_bounds[1])
            if first_profile <= last_profile:
                selected_list.append(span.enclose((first_profile, last_profile)))
        selected = tuple(selected_list)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not selected or any(
        not enclosure.denominator_bounds[0] > 0 for enclosure in selected
    ):
        return None
    value_bounds = tuple(
        (
            min(enclosure.value_bounds[axis_index][0] for enclosure in selected),
            max(enclosure.value_bounds[axis_index][1] for enclosure in selected),
        )
        for axis_index in range(3)
    )
    derivative_bounds = tuple(
        (
            min(
                enclosure.first_derivative_bounds[axis_index][0] for enclosure in selected
            ),
            max(
                enclosure.first_derivative_bounds[axis_index][1] for enclosure in selected
            ),
        )
        for axis_index in range(3)
    )

    relative = tuple(
        (value_bounds[index][0] - origin[index], value_bounds[index][1] - origin[index])
        for index in range(3)
    )
    relative_axis = vector_dot(axis, relative)
    rotated_relative = vector_rotate(relative, cosine, sine)
    base_point = tuple(
        interval_add((origin[index], origin[index]), rotated_relative[index])
        for index in range(3)
    )
    if offset is None:
        lower = np.asarray(
            tuple(fraction_down(component[0]) for component in base_point),
            dtype=np.float64,
        )
        upper = np.asarray(
            tuple(fraction_up(component[1]) for component in base_point),
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            return None
        return lower, upper
    derivative = tuple(derivative_bounds[index] for index in range(3))
    axis_cross_relative = vector_cross(tuple((item, item) for item in axis), relative)
    angle_tangent = tuple(
        interval_add(
            interval_add(
                interval_neg(interval_multiply(relative[index], sine)),
                interval_multiply(axis_cross_relative[index], cosine),
            ),
            interval_multiply(
                (axis[index] * relative_axis[0], axis[index] * relative_axis[1]), sine
            ),
        )
        for index in range(3)
    )
    normal = vector_cross(angle_tangent, vector_rotate(derivative, cosine, sine))

    def square_bounds(value: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
        lower = (
            Fraction(0)
            if value[0] <= 0 <= value[1]
            else min(value[0] ** 2, value[1] ** 2)
        )
        return lower, max(value[0] ** 2, value[1] ** 2)

    norm_squared = tuple(square_bounds(component) for component in normal)
    norm_lower_squared = sum((item[0] for item in norm_squared), Fraction(0))
    norm_upper_squared = sum((item[1] for item in norm_squared), Fraction(0))
    if norm_lower_squared <= 0 or norm_upper_squared <= 0:
        return None
    norm_lower = Fraction.from_float(square_root_interval(norm_lower_squared)[0])
    norm_upper = Fraction.from_float(square_root_interval(norm_upper_squared)[1])
    if norm_lower <= 0 or norm_upper < norm_lower:
        return None

    def divide_positive(
        numerator: tuple[Fraction, Fraction], denominator: tuple[Fraction, Fraction]
    ) -> tuple[Fraction, Fraction]:
        return interval_multiply(
            numerator, (Fraction(1, 1) / denominator[1], Fraction(1, 1) / denominator[0])
        )

    unit_normal = tuple(
        divide_positive(component, (norm_lower, norm_upper)) for component in normal
    )
    rotated_normal = vector_rotate(unit_normal, cosine, sine)
    distance = offset.distance_exact * (offset.frame_sign if offset.frame_sign else 1)
    image = tuple(
        interval_add(base_point[index], interval_scale(distance, rotated_normal[index]))
        for index in range(3)
    )
    lower = np.asarray(
        tuple(fraction_down(component[0]) for component in image), dtype=np.float64
    )
    upper = np.asarray(
        tuple(fraction_up(component[1]) for component in image), dtype=np.float64
    )
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper < lower)
    ):
        return None
    return lower, upper


def _certify_edge_vertex_gap_bounds(
    points: np.ndarray,
    exact_vertices: tuple[tuple[Fraction, Fraction, Fraction] | None, ...],
    vertex_roots: tuple[BRepVertexRoot | None, ...],
    curves: tuple[BRepCurve, ...],
    edge_curves: tuple[int, ...],
    ranges: np.ndarray,
    edge_vertices: tuple[tuple[int, int], ...],
    edge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    /,
    *,
    declared_bounds: tuple[tuple[float, float], ...] | None = None,
) -> tuple[tuple[float, float], ...] | None:
    """Recompute edge/vertex source gaps; caller values are limits, never evidence."""
    if declared_bounds is not None and len(declared_bounds) != len(edge_curves):
        raise ValueError("Declared edge-vertex gap limits must align with all edges.")
    from .._rational import fraction_down, fraction_up
    from ._conic_gaps import _squared_gap_upper, endpoint_gap_bounds
    from ._exact import square_root_interval
    from ._intersection import NativePeriodEndpoint
    from ._intersection_curve import exact_curve_parameter_point

    def vertex_box(vertex: int, /) -> tuple[np.ndarray, np.ndarray]:
        root = vertex_roots[vertex]
        if root is not None:
            box = np.asarray(root.point_enclosure(), dtype=np.float64)
            if (
                box.shape != (2, 3)
                or not np.all(np.isfinite(box))
                or np.any(box[1] < box[0])
            ):
                raise ValueError(
                    "A vertex source root has no finite ordered point enclosure."
                )
            return box[0], box[1]
        exact = exact_vertices[vertex]
        if exact is not None:
            return (
                np.asarray(tuple(fraction_down(value) for value in exact)),
                np.asarray(tuple(fraction_up(value) for value in exact)),
            )
        value = points[vertex]
        return value.copy(), value.copy()

    def endpoint_bound(
        edge: int,
        endpoint: int,
        curve: BRepCurve,
        /,
    ) -> float:
        vertex = edge_vertices[edge][endpoint]
        root = edge_roots[edge][endpoint]
        exact_vertex = exact_vertices[vertex]
        from ._carrier_pieces import carrier_payload
        from ._intersection import (
            BranchRootEndpoint,
            ConicParameterEndpoint,
            IntersectionCurvePointRoot,
        )
        from ._intersection_curve import IntersectionCurve
        from ._patches import CircleCurve, LineCurve
        from ._root_bindings import BRepConicVertex

        vertex_root = vertex_roots[vertex]
        if (
            isinstance(curve, IntersectionCurve)
            and isinstance(root, BranchRootEndpoint)
            and vertex_root is not None
            and (
                root.curve is curve
                or carrier_payload(root.curve) == carrier_payload(curve)
            )
            and vertex_root.supports_endpoint(root)
            and root.parameter_enclosure()
            == (float(ranges[edge, endpoint]), float(ranges[edge, endpoint]))
            and root.certifies_point_root(
                IntersectionCurvePointRoot(curve, float(ranges[edge, endpoint]))
            )
        ):
            return 0.0
        if (
            isinstance(root, ConicParameterEndpoint)
            and vertex_root is not None
            and vertex_root.supports_endpoint(root)
            and carrier_payload(root.carrier) == carrier_payload(curve)
        ):
            if (
                isinstance(curve, CircleCurve)
                and root.parameter_exact is None
                and carrier_payload(root.root.conic) == carrier_payload(curve)
                and root.root.exact_point() == root.point_exact
            ):
                return 0.0
            if isinstance(curve, LineCurve) and root.parameter_exact is not None:
                image = exact_curve_parameter_point(curve, root.parameter_exact)
                if image == root.point_exact:
                    return 0.0
        if (
            isinstance(root, NativePeriodEndpoint)
            and root.patch is None
            and root.rational == 0
            and root.turns.denominator == 1
            and isinstance(curve, CircleCurve)
            and carrier_payload(root.curve) == carrier_payload(curve)
            and vertex_root is not None
            and isinstance(vertex_root.primary, BRepConicVertex)
            and vertex_root.primary.point_exact
            == tuple(
                center + curve.radius_exact * axis
                for center, axis in zip(
                    curve.center_exact, curve.first_axis_exact, strict=True
                )
            )
        ):
            return 0.0
        if vertex_roots[vertex] is None and exact_vertex is not None:
            exact_parameter = (
                root.exact_parameter
                if isinstance(root, NativePeriodEndpoint)
                else Fraction.from_float(float(ranges[edge, endpoint]))
            )
            exact_image = exact_curve_parameter_point(curve, exact_parameter)
            if exact_image is not None:
                return square_root_interval(
                    _squared_gap_upper(exact_image, exact_vertex)
                )[1]
        source_lower, source_upper = (
            (float(ranges[edge, endpoint]), float(ranges[edge, endpoint]))
            if root is None
            else root.parameter_enclosure()
        )
        from ._patches import (
            BSplineCurve,
            OffsetSurface,
            RevolutionSurface,
            SurfaceIsoparametricCurve,
        )

        source_box = None
        source_surface = (
            curve.surface if isinstance(curve, SurfaceIsoparametricCurve) else None
        )
        revolution = (
            source_surface.base
            if isinstance(source_surface, OffsetSurface)
            else source_surface
        )
        if (
            isinstance(curve, SurfaceIsoparametricCurve)
            and isinstance(source_surface, (OffsetSurface, RevolutionSurface))
            and isinstance(revolution, RevolutionSurface)
            and isinstance(revolution.curve, BSplineCurve)
        ):
            if curve.fixed_axis == 1:
                lower_uv = (source_lower, curve.fixed_value)
                upper_uv = (source_upper, curve.fixed_value)
                proof_root = root
            else:
                lower_uv = (curve.fixed_value, source_lower)
                upper_uv = (curve.fixed_value, source_upper)
                proof_root = None
            source_box = _bspline_revolution_box(
                source_surface,
                lower_uv,
                upper_uv,
                source_curve=curve,
                root=proof_root,
            )
        if source_box is not None:
            image_lower, image_upper = source_box
            vertex_lower, vertex_upper = vertex_box(vertex)
            squared = Fraction(0)
            for axis in range(3):
                low = Fraction(float(image_lower[axis])) - Fraction(
                    float(vertex_upper[axis])
                )
                high = Fraction(float(image_upper[axis])) - Fraction(
                    float(vertex_lower[axis])
                )
                squared += max(abs(low), abs(high)) ** 2
            return square_root_interval(squared)[1]
        if (
            vertex_roots[vertex] is None
            and exact_vertices[vertex] is None
            and (root is None or isinstance(root, NativePeriodEndpoint))
        ):
            first_root, last_root = edge_roots[edge]
            if (first_root is None or isinstance(first_root, NativePeriodEndpoint)) and (
                last_root is None or isinstance(last_root, NativePeriodEndpoint)
            ):
                row = endpoint_gap_bounds(
                    curve,
                    (float(ranges[edge, 0]), float(ranges[edge, 1])),
                    points[np.asarray(edge_vertices[edge])],
                    endpoint_roots=(first_root, last_root),
                )
                return row[endpoint]
        lower_parameter, upper_parameter = (
            (float(ranges[edge, endpoint]), float(ranges[edge, endpoint]))
            if root is None
            else root.parameter_enclosure()
        )
        if (
            not np.isfinite(lower_parameter)
            or not np.isfinite(upper_parameter)
            or lower_parameter > upper_parameter
        ):
            raise ValueError("A source endpoint has no finite parameter enclosure.")
        image = np.asarray(
            curve.bounding_box(lower_parameter, upper_parameter), dtype=np.float64
        )
        if image.shape != (2, 3):
            raise ValueError("A source curve bounding box must have shape (2,3).")
        image_lower, image_upper = image[0], image[1]
        if (
            image_lower.shape != (3,)
            or image_upper.shape != (3,)
            or not np.all(np.isfinite(image_lower))
            or not np.all(np.isfinite(image_upper))
            or np.any(image_upper < image_lower)
        ):
            raise ValueError(
                "A source curve endpoint has no finite ordered image enclosure."
            )
        vertex_lower, vertex_upper = vertex_box(vertex)
        squared = Fraction(0)
        for axis in range(3):
            low = Fraction(float(image_lower[axis])) - Fraction(float(vertex_upper[axis]))
            high = Fraction(float(image_upper[axis])) - Fraction(
                float(vertex_lower[axis])
            )
            squared += max(abs(low), abs(high)) ** 2
        return square_root_interval(squared)[1]

    rows: list[tuple[float, float]] = []
    for edge, curve_index in enumerate(edge_curves):
        if curve_index < 0:
            row = (0.0, 0.0)
        else:
            curve = curves[curve_index]
            row = tuple(endpoint_bound(edge, endpoint, curve) for endpoint in range(2))
        if declared_bounds is not None:
            limits = declared_bounds[edge]
            if len(limits) != 2 or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0.0
                for value in limits
            ):
                raise ValueError(
                    "Declared edge-vertex gap limits must be finite nonnegative endpoint pairs."
                )
            if any(value > limit for value, limit in zip(row, limits, strict=True)):
                raise ValueError(
                    "A declared edge-vertex gap limit does not cover its source gap."
                )
        rows.append((float(row[0]), float(row[1])))
    return tuple(rows) if any(bound for row in rows for bound in row) else None


def _certify_edge_face_gap_bounds(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    /,
    *,
    parameter_bounds: ArrayLike | None = None,
    declared_bounds: tuple[float, ...] | None = None,
    exact_unsupported: bool = False,
    certified_degenerate_coedges: bool = False,
) -> tuple[float, ...] | None:
    """Recompute source gaps; degenerate coedges need a separate pole proof."""
    if len(patches) != len(geometry.face_loops):
        raise ValueError("Face-gap certification requires one patch per source face.")
    bounds_array = (
        None
        if parameter_bounds is None
        else np.asarray(parameter_bounds, dtype=np.float64)
    )
    if bounds_array is not None and bounds_array.shape != (len(patches), 2, 2):
        raise ValueError("parameter_bounds must contain one 2D box per source face.")
    if declared_bounds is not None and len(declared_bounds) != len(geometry.coedge_edges):
        raise ValueError("Declared face-gap bounds must align with every source coedge.")
    from ._conic_gaps import native_face_gap_squared
    from ._correspondence import exact_source_lift
    from ._exact import square_root_interval
    from ._intersection_curve import SurfaceRegion

    coedge_faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                if coedge_faces[coedge] not in (-1, face):
                    raise ValueError(
                        "Each source coedge must belong to exactly one face."
                    )
                coedge_faces[coedge] = face
    bounds: list[float] = []
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for coedge, edge in enumerate(geometry.coedge_edges):
        face = coedge_faces[coedge]
        if face < 0:
            raise ValueError("Every source coedge must belong to a face.")
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            if not certified_degenerate_coedges:
                if exact_unsupported:
                    raise ValueError(
                        "A degenerate coedge has no supported exact face-gap proof."
                    )
                return None
            squared = Fraction(0)
        else:
            curve, pcurve, patch = (
                geometry.curves[curve_index],
                geometry.pcurves[coedge],
                patches[face],
            )
            region_box = (
                patch.parameter_domain if bounds_array is None else bounds_array[face]
            )
            if region_box is None:
                region_box = ((0.0, 0.0), (1.0, 1.0))
            region = SurfaceRegion(patch, np.asarray(region_box, dtype=np.float64))
            first, last = (float(value) for value in ranges[edge])
            if exact_source_lift(curve, pcurve, region, first, last):
                squared = Fraction(0)
            else:
                squared = native_face_gap_squared(curve, pcurve, region, first, last)
            if squared is None:
                if exact_unsupported:
                    raise ValueError(
                        f"No exact source face-gap proof supports coedge {coedge}."
                    )
                return None
        bound = square_root_interval(squared)[1]
        if declared_bounds is not None:
            limit = declared_bounds[coedge]
            if (
                isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, Real)
                or not np.isfinite(limit)
                or limit < 0.0
            ):
                raise ValueError(
                    "Declared face-gap limits must be finite nonnegative real scalars."
                )
            if bound > float(limit):
                raise ValueError(
                    f"Declared face-gap limit for coedge {coedge} does not cover its recomputed source gap."
                )
        bounds.append(bound)
    return tuple(bounds)


class BRepGeometry(StrictModule):
    """Source curves, p-curves, oriented topology and shared assembly structure."""

    vertex_roots: tuple[BRepVertexRoot | None, ...]
    vertex_evaluation_bounds: Array

    vertex_points: Array
    vertex_exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...] = (
        eqx.field(static=True)
    )
    edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
    edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = eqx.field(
        static=True
    )
    edge_face_gap_bounds: tuple[float, ...] | None = eqx.field(static=True)
    curves: tuple[BRepCurve, ...]
    edge_ranges: Array
    pcurves: tuple[BRepCurve, ...]
    edge_curves: tuple[int, ...] = eqx.field(static=True)
    edge_vertices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    coedge_senses: tuple[int, ...] = eqx.field(static=True)
    face_loops: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    shell_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shell_orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_shells: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    assembly_graph: BRepAssemblyGraph = eqx.field(static=True)
    topology: BRepTopology = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        vertex_points: ArrayLike,
        vertex_exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...]
        | None = None,
        vertex_roots: tuple[BRepVertexRoot | None, ...] | None = None,
        edge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        coedge_endpoint_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...]
        | None = None,
        edge_vertex_gap_bounds: tuple[tuple[float, float], ...] | None = None,
        edge_face_gap_bounds: tuple[float, ...] | None = None,
        curves: tuple[BRepCurve, ...],
        edge_curves: tuple[int, ...],
        edge_ranges: ArrayLike,
        edge_vertices: tuple[tuple[int, int], ...],
        pcurves: tuple[BRepCurve, ...],
        coedge_edges: tuple[int, ...],
        coedge_senses: tuple[int, ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        assembly_graph: BRepAssemblyGraph | None = None,
    ) -> None:
        points = jnp.asarray(vertex_points, dtype=jnp.float64)
        ranges = jnp.asarray(edge_ranges, dtype=jnp.float64)
        edge_curves_ = tuple(_index(value, "edge_curves entry") for value in edge_curves)
        raw_edge_vertices = tuple(
            tuple(_index(value, "edge vertex index") for value in pair)
            for pair in edge_vertices
        )
        coedge_edges_ = tuple(
            _index(value, "coedge edge index") for value in coedge_edges
        )
        coedge_senses_ = tuple(_index(value, "coedge sense") for value in coedge_senses)
        face_loops_ = tuple(
            tuple(
                tuple(_index(value, "face coedge index") for value in loop)
                for loop in loops
            )
            for loops in face_loops
        )
        shell_faces_ = tuple(
            tuple(_index(value, "shell face index") for value in faces)
            for faces in shell_faces
        )
        shell_orientations_ = tuple(
            tuple(_index(value, "shell face orientation") for value in signs)
            for signs in shell_orientations
        )
        solid_shells_ = tuple(
            tuple(_index(value, "solid shell index") for value in shells)
            for shells in solid_shells
        )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("vertex_points must have shape (num_vertices, 3).")
        points_host = np.asarray(points, dtype=np.float64)
        if not np.all(np.isfinite(points_host)):
            raise ValueError("vertex_points must be finite.")
        if vertex_exact_points is None:
            exact_vertices = (None,) * points.shape[0]
        else:
            exact_vertices = tuple(vertex_exact_points)
            if len(exact_vertices) != points.shape[0]:
                raise ValueError("vertex_exact_points must align with vertex_points.")
            for index, exact in enumerate(exact_vertices):
                if exact is None:
                    continue
                if len(exact) != 3 or any(
                    not isinstance(value, Fraction) for value in exact
                ):
                    raise TypeError(
                        "Each exact vertex binding must be a 3D Fraction tuple or None."
                    )
                rounded = np.asarray(
                    tuple(float(value) for value in exact), dtype=np.float64
                )
                if not np.all(np.isfinite(rounded)) or not np.array_equal(
                    rounded, points_host[index]
                ):
                    raise ValueError(
                        "Rounded vertex coordinates must realize their exact source binding."
                    )
        vertex_roots_ = (
            (None,) * points.shape[0] if vertex_roots is None else tuple(vertex_roots)
        )
        edge_endpoint_roots_ = (
            ((None, None),) * len(edge_curves_)
            if edge_endpoint_roots is None
            else tuple(_root_pair(pair) for pair in edge_endpoint_roots)
        )
        coedge_endpoint_roots_ = (
            ((None, None),) * len(coedge_edges_)
            if coedge_endpoint_roots is None
            else tuple(_root_pair(pair) for pair in coedge_endpoint_roots)
        )
        if len(vertex_roots_) != points.shape[0]:
            raise ValueError("vertex_roots must align with vertex_points.")
        if len(edge_endpoint_roots_) != len(edge_curves_):
            raise ValueError("edge_endpoint_roots must align with edges.")
        if any(len(pair) != 2 for pair in edge_endpoint_roots_):
            raise ValueError("Each edge_endpoint_roots entry must contain two roots.")
        if len(coedge_endpoint_roots_) != len(coedge_edges_):
            raise ValueError("coedge_endpoint_roots must align with coedges.")
        if any(len(pair) != 2 for pair in coedge_endpoint_roots_):
            raise ValueError("Each coedge_endpoint_roots entry must contain two roots.")
        _validate_geometry_source_carriers(
            tuple(curves),
            tuple(pcurves),
            vertex_roots_,
            edge_endpoint_roots_,
            coedge_endpoint_roots_,
        )
        if edge_vertex_gap_bounds is None:
            edge_vertex_gap_bounds_ = None
        else:
            raw_vertex_gaps = tuple(tuple(row) for row in edge_vertex_gap_bounds)
            if len(raw_vertex_gaps) != len(edge_curves_) or any(
                len(row) != 2
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0.0
                    for value in row
                )
                for row in raw_vertex_gaps
            ):
                raise ValueError(
                    "edge_vertex_gap_bounds must contain finite nonnegative endpoint pairs per edge."
                )
            edge_vertex_gap_bounds_ = tuple(
                (float(row[0]), float(row[1])) for row in raw_vertex_gaps
            )
        if edge_face_gap_bounds is None:
            edge_face_gap_bounds_ = None
        else:
            raw_face_gaps = tuple(edge_face_gap_bounds)
            if len(raw_face_gaps) != len(coedge_edges_) or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0.0
                for value in raw_face_gaps
            ):
                raise ValueError(
                    "edge_face_gap_bounds must contain finite nonnegative bounds per coedge."
                )
            edge_face_gap_bounds_ = tuple(float(value) for value in raw_face_gaps)
        if ranges.shape != (len(edge_curves_), 2):
            raise ValueError("edge_ranges must contain one interval per edge.")
        ranges_host = np.asarray(ranges)
        if not np.all(np.isfinite(ranges_host)):
            raise ValueError("edge_ranges must be finite.")
        if len(raw_edge_vertices) != len(edge_curves_):
            raise ValueError("edge_vertices must align with edge_curves.")
        if any(len(pair) != 2 for pair in raw_edge_vertices):
            raise ValueError("Each edge_vertices entry must have two endpoint indices.")
        edge_vertices_ = tuple((pair[0], pair[1]) for pair in raw_edge_vertices)
        if any(index < -1 or index >= len(curves) for index in edge_curves_):
            raise ValueError("edge_curves references an absent curve.")
        for edge, (first, last) in enumerate(ranges_host):
            if first > last or (edge_curves_[edge] >= 0 and not first < last):
                raise ValueError("Nondegenerate edge ranges must be increasing.")
        if any(curve.ambient_dimension != 3 for curve in curves):
            raise ValueError("BRep edge curves must be three-dimensional.")
        if any(curve.ambient_dimension != 2 for curve in pcurves):
            raise ValueError("BRep p-curves must be two-dimensional.")
        if any(
            index < 0 or index >= points.shape[0]
            for pair in edge_vertices_
            for index in pair
        ):
            raise ValueError("edge_vertices references an absent vertex.")
        for edge, curve_index in enumerate(edge_curves_):
            if curve_index < 0:
                continue
            curve = curves[curve_index]
            if not isinstance(curve, ExactLineCurve):
                continue
            first, last = ranges_host[edge]
            if first < 0.0 or last > 1.0:
                raise ValueError("ExactLineCurve edge ranges must lie in [0, 1].")
            start, end = edge_vertices_[edge]

            def source_vertex(index: int, /) -> tuple[Fraction, Fraction, Fraction]:
                exact = exact_vertices[index]
                if exact is not None:
                    return exact
                values = tuple(
                    Fraction.from_float(float(value)) for value in points_host[index]
                )
                return values[0], values[1], values[2]

            if curve.evaluate_exact(Fraction.from_float(float(first))) != source_vertex(
                start
            ) or curve.evaluate_exact(Fraction.from_float(float(last))) != source_vertex(
                end
            ):
                raise ValueError(
                    "ExactLineCurve endpoints must match their exact vertex bindings."
                )
        from ._patches import LineCurve
        from ._root_bindings import BRepConicVertex, BRepVertexRoot

        for edge, curve_index in enumerate(edge_curves_):
            if curve_index < 0:
                continue
            curve = curves[curve_index]
            if not isinstance(curve, LineCurve):
                continue
            for endpoint, vertex in enumerate(edge_vertices_[edge]):
                exact_point = exact_vertices[vertex]
                vertex_root = vertex_roots_[vertex]
                if exact_point is None and isinstance(vertex_root, BRepVertexRoot):
                    if isinstance(vertex_root.primary, BRepConicVertex):
                        exact_point = vertex_root.primary.point_exact
                if exact_point is None:
                    continue
                parameter = Fraction.from_float(float(ranges_host[edge, endpoint]))
                if curve.evaluate_exact(parameter) != exact_point:
                    raise ValueError(
                        "LineCurve endpoint must match its exact vertex source binding."
                    )
        for edge, curve_index in enumerate(edge_curves_):
            start, end = edge_vertices_[edge]
            if curve_index == -1 and start != end:
                raise ValueError("A degenerate edge must collapse to one vertex.")
        coedge_count = len(coedge_edges_)
        if len(coedge_senses_) != coedge_count or len(pcurves) != coedge_count:
            raise ValueError("Coedge edges, senses and p-curves must align.")
        if any(edge < 0 or edge >= len(edge_curves_) for edge in coedge_edges_):
            raise ValueError("coedge_edges references an absent edge.")
        if any(sense not in (-1, 1) for sense in coedge_senses_):
            raise ValueError("coedge_senses must contain only -1 or 1.")
        root_evaluation_bounds = _validate_root_bindings(
            points_host,
            exact_vertices,
            ranges_host,
            tuple(curves),
            edge_curves_,
            tuple(pcurves),
            edge_vertices_,
            coedge_edges_,
            vertex_roots_,
            edge_endpoint_roots_,
            coedge_endpoint_roots_,
        )
        from ._exact import square_root_interval

        for vertex, exact in enumerate(exact_vertices):
            if exact is None:
                continue
            numeric = tuple(
                Fraction.from_float(float(value)) for value in points_host[vertex]
            )
            squared = sum(
                (
                    (value - target) ** 2
                    for value, target in zip(numeric, exact, strict=True)
                ),
                Fraction(0),
            )
            root = vertex_roots_[vertex]
            if root is not None:
                enclosure = np.asarray(root.point_enclosure(), dtype=np.float64)
                if any(
                    Fraction.from_float(float(enclosure[0, axis])) > value
                    or Fraction.from_float(float(enclosure[1, axis])) < value
                    for axis, value in enumerate(exact)
                ):
                    raise ValueError(
                        "A vertex root enclosure must contain its exact rational source point."
                    )
            root_evaluation_bounds[vertex] = max(
                root_evaluation_bounds[vertex], square_root_interval(squared)[1]
            )
        edge_vertex_gap_bounds_ = _certify_edge_vertex_gap_bounds(
            points_host,
            exact_vertices,
            vertex_roots_,
            tuple(curves),
            edge_curves_,
            ranges_host,
            edge_vertices_,
            edge_endpoint_roots_,
            declared_bounds=edge_vertex_gap_bounds_,
        )
        # Free vertices and edges are source entities in their own right (wire
        # and point roots), so a faceless geometry may carry them. Coedges, shells
        # and solids still require faces through the incidence checks below.
        if any(not loops or any(not loop for loop in loops) for loops in face_loops_):
            raise ValueError("Every face requires nonempty coedge loops.")
        used_coedges = tuple(
            coedge for loops in face_loops_ for loop in loops for coedge in loop
        )
        if sorted(used_coedges) != list(range(coedge_count)):
            raise ValueError("Each coedge must belong to exactly one face loop.")
        for loops in face_loops_:
            for loop in loops:
                if any(coedge < 0 or coedge >= coedge_count for coedge in loop):
                    raise ValueError("A face loop references an absent coedge.")
                ends = tuple(
                    self._oriented_edge_vertices(
                        edge_vertices_[coedge_edges_[coedge]],
                        coedge_senses_[coedge],
                    )
                    for coedge in loop
                )
                if any(
                    ends[index][1] != ends[(index + 1) % len(ends)][0]
                    for index in range(len(ends))
                ):
                    raise ValueError("Oriented coedges in each face loop must connect.")
        if len(shell_faces_) != len(shell_orientations_):
            raise ValueError("shell_faces and shell_orientations must align.")
        for faces, signs in zip(shell_faces_, shell_orientations_, strict=True):
            if len(faces) != len(signs):
                raise ValueError("Each shell face needs one orientation sign.")
            if any(face < 0 or face >= len(face_loops_) for face in faces):
                raise ValueError("A shell references an absent face.")
            if any(sign not in (-1, 1) for sign in signs):
                raise ValueError("Shell face orientations must be -1 or 1.")
        if any(
            shell < 0 or shell >= len(shell_faces_)
            for shells in solid_shells_
            for shell in shells
        ):
            raise ValueError("A solid references an absent shell.")
        topology = self._topology(
            edge_curves_,
            coedge_edges_,
            face_loops_,
            shell_faces_,
            shell_orientations_,
            solid_shells_,
            points.shape[0],
        )
        assembly = (
            BRepAssemblyGraph.single_solids(len(solid_shells_))
            if assembly_graph is None
            else assembly_graph
        )
        if assembly.solid_count != len(solid_shells_):
            raise ValueError("assembly_graph must reference the model's solid bank.")
        self.vertex_points = points
        self.vertex_exact_points = exact_vertices
        self.vertex_roots = vertex_roots_
        self.vertex_evaluation_bounds = jnp.asarray(
            root_evaluation_bounds, dtype=jnp.float64
        )
        self.edge_endpoint_roots = edge_endpoint_roots_
        self.coedge_endpoint_roots = coedge_endpoint_roots_
        self.edge_vertex_gap_bounds = edge_vertex_gap_bounds_
        self.edge_face_gap_bounds = edge_face_gap_bounds_
        self.curves = tuple(curves)
        self.edge_ranges = ranges
        self.pcurves = tuple(pcurves)
        self.edge_curves = edge_curves_
        self.edge_vertices = edge_vertices_
        self.coedge_edges = coedge_edges_
        self.coedge_senses = coedge_senses_
        self.face_loops = face_loops_
        self.shell_faces = shell_faces_
        self.shell_orientations = shell_orientations_
        self.solid_shells = solid_shells_
        self.assembly_graph = assembly
        self.topology = topology
        self.geometry_id = _brep_geometry_identity(self)

    def require_current(self, /) -> None:
        """Refuse geometry whose content changed after its identity was recorded.

        Construction validated the content this identity names, so an unchanged
        identity keeps that validation; :meth:`revalidate` re-runs it.
        """
        if _brep_geometry_identity(self) != self.geometry_id:
            raise ValueError(
                "BRepGeometry content has changed since its source identity was recorded."
            )

    def revalidate(self, /) -> None:
        """Re-run every constructor invariant on this content and require its identity."""
        # Stale content is diagnosed by identity before the rebuild re-runs
        # constructor invariants that edited content may also violate.
        self.require_current()
        rebuilt = BRepGeometry(
            vertex_points=self.vertex_points,
            vertex_exact_points=self.vertex_exact_points,
            vertex_roots=self.vertex_roots,
            edge_endpoint_roots=self.edge_endpoint_roots,
            coedge_endpoint_roots=self.coedge_endpoint_roots,
            edge_vertex_gap_bounds=self.edge_vertex_gap_bounds,
            edge_face_gap_bounds=self.edge_face_gap_bounds,
            curves=self.curves,
            edge_curves=self.edge_curves,
            edge_ranges=self.edge_ranges,
            edge_vertices=self.edge_vertices,
            pcurves=self.pcurves,
            coedge_edges=self.coedge_edges,
            coedge_senses=self.coedge_senses,
            face_loops=self.face_loops,
            shell_faces=self.shell_faces,
            shell_orientations=self.shell_orientations,
            solid_shells=self.solid_shells,
            assembly_graph=self.assembly_graph,
        )
        if rebuilt.geometry_id != self.geometry_id:
            raise ValueError(
                "BRepGeometry content has changed since its source identity was recorded."
            )

    def iter_occurrences(self, /) -> Iterator[BRepAssemblyOccurrence]:
        """Yield exact placed solid leaves from the source assembly DAG."""
        yield from self.assembly_graph.iter_occurrences()

    def iter_assembly_containers(self, /) -> Iterator[BRepAssemblyContainer]:
        """Yield requested instance containers without storing expanded paths."""
        definitions = {
            definition.definition_id: definition
            for definition in self.assembly_graph.definitions
        }
        stack: list[tuple[BRepAssemblyUse, tuple[str, ...]]] = [
            (use, ()) for use in reversed(self.assembly_graph.roots)
        ]
        while stack:
            use, parent_path = stack.pop()
            path = (*parent_path, use.use_id)
            if use.definition_id is None:
                continue
            definition = definitions[use.definition_id]
            member_paths = tuple(
                (*path, child.use_id)
                for child in definition.uses
                if child.solid_index is not None
            )
            child_paths = tuple(
                (*path, child.use_id)
                for child in definition.uses
                if child.definition_id is not None
            )
            if member_paths or child_paths:
                yield BRepAssemblyContainer(path, member_paths, child_paths)
            stack.extend(
                (child, path)
                for child in reversed(definition.uses)
                if child.definition_id is not None
            )

    def qualified_entity_incidence(
        self, source_id: str, source_revision: str, /
    ) -> tuple[BRepQualifiedIncidence, ...]:
        """Publish exact occurrence-qualified topology without coordinate merging."""
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a nonempty string.")
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("source_revision must be a nonempty string.")
        records: set[BRepQualifiedIncidence] = set()

        def entity(kind: str, index: int, path: tuple[str, ...] = ()) -> BRepEntityId:
            return BRepEntityId(source_id, source_revision, kind, index, path)

        def add_face(
            face: int, path: tuple[str, ...], container: BRepEntityId | None
        ) -> None:
            face_id = entity("face", face, path)
            records.add(BRepQualifiedIncidence(face_id, face_id, 1))
            if container is not None:
                sign = 1
                records.add(BRepQualifiedIncidence(container, face_id, sign))
            for loop in self.face_loops[face]:
                for coedge in loop:
                    edge = self.coedge_edges[coedge]
                    edge_id = entity("edge", edge, path)
                    records.add(BRepQualifiedIncidence(edge_id, edge_id, 1))
                    records.add(
                        BRepQualifiedIncidence(
                            face_id,
                            edge_id,
                            self.coedge_senses[coedge],
                            coedge,
                        )
                    )
                    for endpoint, vertex in enumerate(self.edge_vertices[edge]):
                        vertex_id = entity("vertex", vertex, path)
                        records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
                        records.add(
                            BRepQualifiedIncidence(
                                edge_id,
                                vertex_id,
                                -1 if endpoint == 0 else 1,
                                endpoint,
                            )
                        )

        for occurrence in self.assembly_graph.iter_occurrences():
            path = occurrence.path
            solid = occurrence.solid_index
            solid_id = entity("solid", solid, path)
            records.add(BRepQualifiedIncidence(solid_id, solid_id, 1))
            for face, orientation in zip(
                self.topology.solid_faces[solid],
                self.topology.solid_face_orientations[solid],
                strict=True,
            ):
                face_id = entity("face", face, path)
                records.add(BRepQualifiedIncidence(solid_id, face_id, orientation))
                add_face(face, path, None)

        for face, owners in enumerate(self.topology.face_solids):
            if not owners:
                add_face(face, (), None)
        incident_edges: set[int] = set()
        incident_vertices: set[int] = set()
        for edge, owners in enumerate(self.topology.edge_faces):
            if owners:
                incident_edges.add(edge)
                incident_vertices.update(self.edge_vertices[edge])
                continue
            edge_id = entity("edge", edge)
            records.add(BRepQualifiedIncidence(edge_id, edge_id, 1))
            for endpoint, vertex in enumerate(self.edge_vertices[edge]):
                vertex_id = entity("vertex", vertex)
                records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
                records.add(
                    BRepQualifiedIncidence(
                        edge_id, vertex_id, -1 if endpoint == 0 else 1, endpoint
                    )
                )
                incident_vertices.add(vertex)
        for vertex in range(self.topology.num_vertices):
            if vertex not in incident_vertices:
                vertex_id = entity("vertex", vertex)
                records.add(BRepQualifiedIncidence(vertex_id, vertex_id, 1))
        return tuple(sorted(records))

    @staticmethod
    def _oriented_edge_vertices(
        vertices: tuple[int, int], sense: int, /
    ) -> tuple[int, int]:
        return vertices if sense > 0 else (vertices[1], vertices[0])

    @staticmethod
    def _topology(
        edge_curves: tuple[int, ...],
        coedge_edges: tuple[int, ...],
        face_loops: tuple[tuple[tuple[int, ...], ...], ...],
        shell_faces: tuple[tuple[int, ...], ...],
        shell_orientations: tuple[tuple[int, ...], ...],
        solid_shells: tuple[tuple[int, ...], ...],
        num_vertices: int,
        /,
    ) -> BRepTopology:
        face_wires = tuple(
            tuple(tuple(coedge_edges[index] for index in loop) for loop in loops)
            for loops in face_loops
        )
        face_edges = tuple(
            tuple(dict.fromkeys(edge for wire in wires for edge in wire))
            for wires in face_wires
        )
        edge_faces: list[list[int]] = [[] for _ in edge_curves]
        for face, edges in enumerate(face_edges):
            for edge in edges:
                edge_faces[edge].append(face)
        solid_faces: list[tuple[int, ...]] = []
        solid_face_orientations: list[tuple[int, ...]] = []
        for shells in solid_shells:
            faces: list[int] = []
            orientations: list[int] = []
            for shell in shells:
                faces.extend(shell_faces[shell])
                orientations.extend(shell_orientations[shell])
            if len(set(faces)) != len(faces):
                raise ValueError("A solid cannot repeat a face across its shells.")
            solid_faces.append(tuple(faces))
            solid_face_orientations.append(tuple(orientations))
        return BRepTopology(
            face_edges=face_edges,
            edge_faces=tuple(tuple(faces) for faces in edge_faces),
            face_wires=face_wires,
            solid_faces=tuple(solid_faces),
            solid_face_orientations=tuple(solid_face_orientations),
            num_vertices=num_vertices,
        )

    def source_vertex_point_bounds(
        self, vertex_index: int, /
    ) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
        """Return outward rational bounds for one exact source vertex point."""
        index = _index(vertex_index, "vertex_index")
        if index < 0 or index >= self.topology.num_vertices:
            raise IndexError("vertex_index does not name a source vertex.")
        exact = self.vertex_exact_points[index]
        if exact is not None:
            return exact, exact
        root = self.vertex_roots[index]
        if root is not None:
            primary = getattr(root, "primary", root)
            exact_point = getattr(primary, "point_exact", None)
            if exact_point is not None:
                from ...discretization._exact_algebraic_geometry import (
                    coefficient_interval,
                )

                intervals = tuple(
                    coefficient_interval(value, bits=128) for value in exact_point
                )
                return (
                    (intervals[0][0], intervals[1][0], intervals[2][0]),
                    (intervals[0][1], intervals[1][1], intervals[2][1]),
                )
            enclosure = np.asarray(root.point_enclosure(), dtype=np.float64)
            if (
                enclosure.shape != (2, 3)
                or not np.all(np.isfinite(enclosure))
                or np.any(enclosure[0] > enclosure[1])
            ):
                raise ArithmeticError(
                    "A source vertex root has no finite outward enclosure."
                )
            lower_x, lower_y, lower_z = (
                Fraction.from_float(float(value)) for value in enclosure[0]
            )
            upper_x, upper_y, upper_z = (
                Fraction.from_float(float(value)) for value in enclosure[1]
            )
            return (lower_x, lower_y, lower_z), (upper_x, upper_y, upper_z)
        point = np.asarray(self.vertex_points[index], dtype=np.float64)
        x, y, z = (Fraction.from_float(float(value)) for value in point)
        return (x, y, z), (x, y, z)

    def source_curve_parameter_domain(self, edge_index: int, /) -> tuple[float, float]:
        """Return the authored finite parameter interval of a source edge curve."""
        index = _index(edge_index, "edge_index")
        if index < 0 or index >= self.topology.num_edges:
            raise IndexError("edge_index does not name a source edge.")
        if self.edge_curves[index] < 0:
            raise NotImplementedError("The source edge has no parametric curve carrier.")
        lower, upper = np.asarray(self.edge_ranges, dtype=np.float64)[index]
        return float(lower), float(upper)

    def source_curve_point_bounds(
        self, edge_index: int, lower: float, upper: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return outward 3D bounds for one exact source-curve parameter interval."""
        index = _index(edge_index, "edge_index")
        first, last = float(lower), float(upper)
        if not np.isfinite(first) or not np.isfinite(last) or first > last:
            raise ValueError("Source curve bounds must be finite and ordered.")
        domain = self.source_curve_parameter_domain(index)
        if first < domain[0] or last > domain[1]:
            raise ValueError(
                "Source curve bounds must lie inside its authored edge interval."
            )
        from ._root_bindings import _curve_box

        curve_index = self.edge_curves[index]
        box = np.asarray(
            _curve_box(self.curves[curve_index], first, last), dtype=np.float64
        )
        if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1]):
            raise ArithmeticError("A source curve has no finite outward 3D enclosure.")
        return box[0], box[1]


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
        if bounds.shape != (len(patches), 2, 2):
            raise ValueError("parameter_bounds must have shape (num_faces, 2, 2).")
        if isinstance(bounds, jax.core.Tracer):
            bounds = eqx.error_if(
                bounds,
                jnp.any(~jnp.isfinite(bounds))
                | jnp.any(bounds[:, 1, :] <= bounds[:, 0, :]),
                "Every surface parameter interval must be finite and nonempty.",
            )
        else:
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
        if indices.size == 0:
            return jnp.empty((*leading, 3), dtype=reference_.dtype)
        if not self.patches:
            raise IndexError("An empty BRepBoundaryMap has no charts.")
        values = jax.vmap(self._map_one)(
            indices.reshape((-1,)), reference_.reshape((-1, 2))
        )
        return values.reshape((*leading, 3))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=self.parameter_bounds.dtype)
        leading = indices.shape
        if indices.size == 0:
            return jnp.empty(leading, dtype=reference_.dtype)
        if not self.patches:
            raise IndexError("An empty BRepBoundaryMap has no charts.")
        differential = jax.vmap(
            lambda index, coordinate: jax.jacfwd(
                lambda value: self._map_one(index, value)
            )(coordinate)
        )(indices.reshape((-1,)), reference_.reshape((-1, 2)))
        jacobian = jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )
        return jacobian.reshape(leading)


def _surface_patch_identity(patch: AbstractSurfacePatch, /) -> dict[str, object]:
    from ._tspline import TSplineSurfacePatch

    if isinstance(patch, TSplineSurfacePatch):
        patch.validate_source_bank()
    result: dict[str, object] = {
        "type": f"{type(patch).__module__}.{type(patch).__qualname__}",
        "structure": str(jax.tree_util.tree_structure(patch)),
        "arrays": array_tree_fingerprint(patch),
    }
    identity = getattr(patch, "identity_payload", None)
    if callable(identity):
        result["source"] = identity()
    return result


def _brep_model_source_revision(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    orientation: np.ndarray,
    trim_domains: tuple[TrimDomain | None, ...],
    physical_tags: tuple[str, ...],
    coordinate_contract: SpatialCoordinateContract,
    report: BRepImportReport,
    /,
) -> str:
    trim_identity = tuple(
        None if domain is None else domain.identity_payload() for domain in trim_domains
    )
    return canonical_fingerprint(
        {
            "kind": "brep-source-revision",
            "coordinate_contract": coordinate_contract.spatial_id,
            "geometry": geometry.geometry_id,
            "source_accuracy": report.curve_surface_tolerance,
            "face_table": tuple(
                {
                    "patch": _surface_patch_identity(patch),
                    "parameter_bounds": parameter_bounds[index],
                    "orientation": orientation[index],
                    "trim_domain": trim_identity[index],
                    "physical_tag": physical_tags[index],
                }
                for index, patch in enumerate(patches)
            ),
        }
    )


def _validate_exact_line_face_bindings(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    /,
) -> None:
    edge_ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for face_index, loops in enumerate(geometry.face_loops):
        patch = patches[face_index]
        if not isinstance(patch, PlanePatch):
            continue
        origin = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.origin)
        )
        first_axis = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.first_axis)
        )
        second_axis = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.second_axis)
        )
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                curve_index = geometry.edge_curves[edge]
                if curve_index < 0:
                    continue
                curve = geometry.curves[curve_index]
                pcurve = geometry.pcurves[coedge]
                if not isinstance(curve, ExactLineCurve) or not isinstance(
                    pcurve, ExactLineCurve
                ):
                    continue
                for parameter in edge_ranges[edge]:
                    exact_parameter = Fraction.from_float(float(parameter))
                    uv = pcurve.evaluate_exact(exact_parameter)
                    surface_point = tuple(
                        origin[axis]
                        + uv[0] * first_axis[axis]
                        + uv[1] * second_axis[axis]
                        for axis in range(3)
                    )
                    if surface_point != curve.evaluate_exact(exact_parameter):
                        raise ValueError(
                            "ExactLineCurve coedges must map exactly from their p-curves to the source face."
                        )


def _validate_conic_endpoint_lifts(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    /,
) -> None:
    from ._carrier_pieces import carrier_payload
    from ._correspondence import exact_source_lift
    from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
    from ._intersection_curve import SurfaceRegion
    from ._patches import ConicRootLinePCurve
    from ._placed import PlacedCurve

    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    coedge_faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                coedge_faces[coedge] = face
    for coedge, roots in enumerate(geometry.coedge_endpoint_roots):
        if not any(isinstance(root, ConicParameterEndpoint) for root in roots):
            continue
        edge = geometry.coedge_edges[coedge]
        curve_index = geometry.edge_curves[edge]
        face = coedge_faces[coedge]
        if curve_index < 0 or face < 0:
            raise ValueError(
                "A conic endpoint lift requires a nondegenerate edge and owning face."
            )
        curve, pcurve = geometry.curves[curve_index], geometry.pcurves[coedge]
        first, last = (float(value) for value in ranges[edge])
        region = SurfaceRegion(patches[face], parameter_bounds[face])
        if not exact_source_lift(curve, pcurve, region, first, last):
            raise ValueError(
                "A conic endpoint requires an exact full-range source p-curve lift."
            )
        for endpoint, root in enumerate(roots):
            if not isinstance(root, ConicParameterEndpoint):
                continue
            edge_root = geometry.edge_endpoint_roots[edge][endpoint]
            if isinstance(edge_root, ConicParameterEndpoint):
                if edge_root.point_exact != root.point_exact:
                    raise ValueError(
                        "A conic coedge endpoint must share its exact edge endpoint source point."
                    )
            elif isinstance(edge_root, NativePeriodEndpoint):
                source_curve = (
                    curve.definition if isinstance(curve, PlacedCurve) else curve
                )
                source_point = root.root.exact_point()
                if isinstance(curve, PlacedCurve):
                    source_point = curve.affine.apply(source_point)
                if (
                    carrier_payload(edge_root.curve) != carrier_payload(curve)
                    or carrier_payload(root.root.conic) != carrier_payload(source_curve)
                    or root.point_exact != source_point
                ):
                    raise ValueError(
                        "A periodic conic coedge root must bind its exact source seam vertex."
                    )
            else:
                raise ValueError(
                    "A conic coedge endpoint must share an exact edge endpoint root."
                )
            if isinstance(pcurve, ConicRootLinePCurve) and (
                pcurve.root.identity_payload() != root.root.identity_payload()
            ):
                raise ValueError(
                    "A conic generator p-curve endpoint must retain its exact radial root."
                )


def _validate_conic_phase_endpoint_lifts(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    /,
) -> None:
    from ._correspondence import exact_source_lift
    from ._intersection import TrimIntersectionRoot, TrimRootEndpoint
    from ._intersection_curve import SurfaceRegion

    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for face, loops in enumerate(geometry.face_loops):
        for coedge in (index for loop in loops for index in loop):
            bindings = tuple(
                (endpoint, root, root.root.endpoint_witness)
                for endpoint, root in enumerate(geometry.coedge_endpoint_roots[coedge])
                if isinstance(root, TrimRootEndpoint)
                and isinstance(root.root, TrimIntersectionRoot)
                and root.root.endpoint_witness is not None
            )
            if not bindings:
                continue
            edge = geometry.coedge_edges[coedge]
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0:
                raise ValueError(
                    "A conic-phase endpoint requires a nondegenerate source edge."
                )
            curve = geometry.curves[curve_index]
            pcurve = geometry.pcurves[coedge]
            first, last = (float(value) for value in ranges[edge])
            region = SurfaceRegion(patches[face], parameter_bounds[face])
            if not exact_source_lift(curve, pcurve, region, first, last):
                raise ValueError(
                    "A conic-phase endpoint requires exact current-segment incidence."
                )
            for endpoint, binding, witness in bindings:
                vertex = geometry.edge_vertices[edge][endpoint]
                vertex_root = geometry.vertex_roots[vertex]
                if vertex_root is None:
                    raise ValueError(
                        "A conic-phase endpoint requires retained vertex source authority."
                    )
                witness.require_source_binding(
                    curve,
                    pcurve,
                    region,
                    Fraction.from_float(float(ranges[edge, endpoint])),
                    binding,
                    source_lifts=vertex_root.source_edge_lifts,
                )


def _validate_curve_trim_domains(
    geometry: BRepGeometry,
    trim_domains: tuple[TrimDomain | None, ...],
    /,
) -> None:
    from ._carrier_pieces import carrier_payload
    from ._intersection import endpoint_root_payload
    from ._intersection_curve import CurveTrimSegment

    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for face, domain in enumerate(trim_domains):
        if domain is None or not domain.curve_loops:
            continue
        if domain.outer is not None or domain.holes:
            raise ValueError(
                "A source curve trim cannot also claim a polygonal surrogate."
            )
        source_loops = geometry.face_loops[face]
        if len(domain.curve_loops) != len(source_loops):
            raise ValueError(
                "Every source curve trim loop must align with one face source loop."
            )
        for loop_index, (segments, coedges) in enumerate(
            zip(domain.curve_loops, source_loops, strict=True)
        ):
            if len(segments) != len(coedges):
                raise ValueError(
                    "A source curve trim loop must contain one segment per coedge."
                )
            for segment, coedge in zip(segments, coedges, strict=True):
                if not isinstance(segment, CurveTrimSegment):
                    raise TypeError(
                        "Source curve trim loops must contain CurveTrimSegment values."
                    )
                edge = geometry.coedge_edges[coedge]
                if carrier_payload(segment.curve) != carrier_payload(
                    geometry.pcurves[coedge]
                ):
                    raise ValueError(
                        "A source trim segment must retain its exact coedge p-curve carrier."
                    )
                first, last = (float(value) for value in ranges[edge])
                if segment.first != first or segment.last != last:
                    raise ValueError(
                        "A source trim segment must retain its exact edge parameter interval."
                    )
                if segment.reversed != (geometry.coedge_senses[coedge] < 0):
                    raise ValueError(
                        "A source trim segment orientation must match its coedge sense."
                    )
                expected_roots = geometry.coedge_endpoint_roots[coedge]
                for actual, expected in zip(
                    (segment.first_root, segment.last_root), expected_roots, strict=True
                ):
                    if (actual is None) != (expected is None):
                        raise ValueError(
                            "A source trim segment must retain its endpoint root bindings."
                        )
                    if (
                        actual is not None
                        and expected is not None
                        and endpoint_root_payload(actual)
                        != endpoint_root_payload(expected)
                    ):
                        raise ValueError(
                            "A source trim segment endpoint must retain its coedge root definition."
                        )


def _validate_native_period_patch_bindings(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    /,
) -> None:
    from ._carrier_pieces import carrier_payload
    from ._intersection import NativePeriodEndpoint

    coedge_faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                coedge_faces[coedge] = face

    def validate(root: object | None, patch: AbstractSurfacePatch, /) -> None:
        if not isinstance(root, NativePeriodEndpoint) or root.patch is None:
            return
        if carrier_payload(root.patch) != carrier_payload(patch):
            raise ValueError(
                "A native period endpoint must bind its incident source patch."
            )
        if root.axis is None or patch.native_period_symbols()[root.axis] != "two_pi":
            raise ValueError(
                "A native period endpoint requires its declared source-period axis."
            )

    for coedge, roots in enumerate(geometry.coedge_endpoint_roots):
        face = coedge_faces[coedge]
        if face < 0:
            raise ValueError("A source coedge has no owning face.")
        for root in roots:
            validate(root, patches[face])
    for edge, roots in enumerate(geometry.edge_endpoint_roots):
        faces = geometry.topology.edge_faces[edge]
        for root in roots:
            if isinstance(root, NativePeriodEndpoint) and root.patch is not None:
                if not any(
                    carrier_payload(root.patch) == carrier_payload(patches[face])
                    and root.axis is not None
                    and patches[face].native_period_symbols()[root.axis] == "two_pi"
                    for face in faces
                ):
                    raise ValueError(
                        "An edge period endpoint must bind one of its incident source patches."
                    )


def _validate_degenerate_coedge_poles(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    /,
) -> None:
    from ._correspondence import _exact_brep_pole_collapse

    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    used_edges: set[int] = set()
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if geometry.edge_curves[edge] >= 0:
                    continue
                used_edges.add(edge)
                vertex = geometry.edge_vertices[edge][0]
                if not _exact_brep_pole_collapse(
                    None,
                    patches[face],
                    geometry.pcurves[coedge],
                    float(ranges[edge, 0]),
                    float(ranges[edge, 1]),
                    geometry.vertex_exact_points[vertex],
                    geometry.vertex_roots[vertex],
                    geometry.coedge_endpoint_roots[coedge],
                ):
                    raise ValueError(
                        "A degenerate coedge requires an exact full-interval source pole proof."
                    )
    if any(
        curve_index < 0 and edge not in used_edges
        for edge, curve_index in enumerate(geometry.edge_curves)
    ):
        raise ValueError("A degenerate source edge requires an owning pole coedge.")


def _validate_model_source_bindings(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    trim_domains: tuple[TrimDomain | None, ...],
    /,
) -> None:
    for patch, box in zip(patches, parameter_bounds, strict=True):
        patch.validate_parameter_box(box)
    _validate_exact_line_face_bindings(geometry, patches)
    _validate_conic_endpoint_lifts(geometry, patches, parameter_bounds)
    _validate_conic_phase_endpoint_lifts(geometry, patches, parameter_bounds)
    _validate_native_period_patch_bindings(geometry, patches)
    _validate_degenerate_coedge_poles(geometry, patches)
    _validate_curve_trim_domains(geometry, trim_domains)
    face_gap_bounds = _certify_edge_face_gap_bounds(
        geometry,
        patches,
        parameter_bounds=parameter_bounds,
        declared_bounds=geometry.edge_face_gap_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    if face_gap_bounds is None:
        raise ValueError("Source coedge face gaps lack a replayable exact proof.")
    if geometry.edge_face_gap_bounds is None and any(
        bound != 0.0 for bound in face_gap_bounds
    ):
        raise ValueError(
            "An undeclared source coedge face gap must have an exact zero proof."
        )


@final
@dataclass(frozen=True, slots=True)
class BRepSourceBindingProof:
    """Evidence that the source bindings of one source revision were proved.

    `BRepModel.binding_proof` issues it for a constructed model, whose
    construction replayed the binding proof of exactly that revision. A model
    rebuilt from the same source content passes it back instead of replaying
    the proof; the constructor checks it against the revision it computes.
    """

    source_revision: str


def _brep_model_id(source_revision: str, tessellation: BRepTessellation | None, /) -> str:
    """Identify a model as its physical source plus its query representation.

    A model without a tessellation is its physical source; a tessellated model is
    a distinct query representation of that source.
    """
    if tessellation is None:
        return source_revision
    return canonical_fingerprint(
        {
            "kind": "brep-model",
            "source_revision": source_revision,
            "query_representation": tessellation.tessellation_id,
        }
    )


def _tessellation_multiregion_surface(
    domain: MeshingDomain, tessellation: BRepTessellation, /
) -> MultiRegionSurfaceTopology | None:
    """Label tessellation triangles by the domain regions on their patch sides."""
    witness = tessellation.source_chart_witness
    witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(witness.cell_global_ids, dtype=np.int64).tolist()
        )
    }
    triangle_ids = np.asarray(tessellation.triangle_global_ids, dtype=np.int64)
    triangle_patches = np.asarray(witness.patches, dtype=np.int64)[
        np.asarray(
            [witness_rows[identifier] for identifier in triangle_ids.tolist()],
            dtype=np.int64,
        )
    ]
    return domain.multiregion_surface(
        np.asarray(tessellation.vertices, dtype=np.float64),
        np.asarray(tessellation.triangles, dtype=np.int64),
        triangle_patches,
        vertex_global_ids=np.asarray(tessellation.vertex_global_ids, dtype=np.int64),
        cell_global_ids=triangle_ids,
    )


class BRepModel(StrictModule):
    """Source-authoritative B-Rep with an optional derived triangle realization."""

    geometry: BRepGeometry
    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array
    orientation: Array
    trim_domains: tuple[TrimDomain | None, ...]
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)
    report: BRepImportReport = eqx.field(static=True)
    tessellation: BRepTessellation | None
    source_revision: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        geometry: BRepGeometry,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: ArrayLike,
        orientation: ArrayLike,
        physical_tags: tuple[str, ...],
        coordinate_contract: SpatialCoordinateContract,
        report: BRepImportReport,
        trim_domains: tuple[TrimDomain | None, ...] | None = None,
        tessellation: BRepTessellation | None = None,
        tessellation_certificate: SourceFidelityCertificate | None = None,
        binding_proof: BRepSourceBindingProof | None = None,
    ) -> None:
        """Validate the source model and, when given, its derived tessellation.

        ``tessellation_certificate`` is the source-fidelity proof that
        ``tessellate_brep_model`` returned with ``tessellation``. It must bind
        exactly the tessellation's mesh, coordinate map, source domain, witness
        and deflection policy; the model then reuses it instead of recomputing
        the proof. Without it the proof is recomputed. ``binding_proof`` is the
        `binding_proof` of a model of the same source revision; it must name
        exactly the revision computed here, and the source binding proof is
        then not replayed.
        """
        if coordinate_contract.spatial_id != report.coordinate_contract.spatial_id:
            raise ValueError("The model and report coordinate contracts must match.")
        patches_ = tuple(patches)
        if any(not isinstance(patch, AbstractSurfacePatch) for patch in patches_):
            raise TypeError("patches must contain only native surface patches.")
        from ._patches import _validate_surface_source_shadows

        for patch in patches_:
            _validate_surface_source_shadows(patch)
        # A model accepts geometry it did not construct: renew its invariants.
        geometry.revalidate()
        # Free vertex and edge roots are faceless sources; only solids need faces.
        if not patches_ and geometry.topology.num_solids:
            raise ValueError("A faceless BRepModel cannot contain solids.")
        face_count = len(patches_)
        bounds_host = np.asarray(parameter_bounds, dtype=np.float64)
        orientation_host = np.asarray(orientation, dtype=np.float64)
        if (
            len(geometry.face_loops) != face_count
            or geometry.topology.num_faces != face_count
        ):
            raise ValueError("The face patch table must align with source geometry.")
        if bounds_host.shape != (face_count, 2, 2):
            raise ValueError("parameter_bounds must have shape (num_faces, 2, 2).")
        if not np.all(np.isfinite(bounds_host)) or np.any(
            bounds_host[:, 1, :] <= bounds_host[:, 0, :]
        ):
            raise ValueError(
                "Every surface parameter interval must be finite and nonempty."
            )
        if orientation_host.shape != (face_count,):
            raise ValueError("orientation must contain one sign per face.")
        if not np.all(np.isin(orientation_host, (-1.0, 1.0))):
            raise ValueError("orientation entries must be -1 or 1.")
        trims = (None,) * face_count if trim_domains is None else tuple(trim_domains)
        if len(trims) != face_count or any(
            domain is not None and not isinstance(domain, TrimDomain) for domain in trims
        ):
            raise ValueError("trim_domains must contain one TrimDomain or None per face.")
        tags = tuple(physical_tags)
        if len(tags) != face_count or any(
            not isinstance(tag, str) or not tag for tag in tags
        ):
            raise ValueError("physical_tags must contain one non-empty string per face.")
        topology = geometry.topology
        source_counts = (
            topology.num_solids,
            face_count,
            topology.num_edges,
            topology.num_vertices,
        )
        report_counts = (
            report.num_solids,
            report.num_faces,
            report.num_edges,
            report.num_vertices,
        )
        if source_counts != report_counts:
            raise ValueError("The import report source counts must match the B-Rep.")
        bounds = jnp.asarray(bounds_host, dtype=jnp.float64)
        signs = jnp.asarray(orientation_host, dtype=jnp.float64)
        source_revision = _brep_model_source_revision(
            geometry,
            patches_,
            bounds_host,
            orientation_host,
            trims,
            tags,
            coordinate_contract,
            report,
        )
        if binding_proof is None:
            _validate_model_source_bindings(geometry, patches_, bounds_host, trims)
        elif (
            not isinstance(binding_proof, BRepSourceBindingProof)
            or binding_proof.source_revision != source_revision
        ):
            raise ValueError(
                "binding_proof must prove the source bindings of exactly this source revision."
            )
        if tessellation is not None and not isinstance(tessellation, BRepTessellation):
            raise TypeError("tessellation must be a BRepTessellation or None.")
        self.geometry = geometry
        self.patches = patches_
        self.parameter_bounds = bounds
        self.orientation = signs
        self.trim_domains = trims
        self.physical_tags = tags
        self.coordinate_contract = coordinate_contract
        self.report = report
        self.tessellation = tessellation
        self.source_revision = source_revision
        self.model_id = _brep_model_id(source_revision, tessellation)
        if tessellation is not None:
            self._validate_tessellation(
                tessellation, source_revision, geometry, patches_, bounds_host, signs
            )
            self._validate_tessellation_source_witness(
                tessellation, tessellation_certificate
            )
        elif tessellation_certificate is not None:
            raise ValueError("A tessellation certificate requires its tessellation.")

    @property
    def binding_proof(self, /) -> BRepSourceBindingProof:
        """Proof that this model's source revision has validated source bindings."""
        return BRepSourceBindingProof(self.source_revision)

    def require_current(self, /) -> None:
        """Refuse a model whose source content changed after construction.

        Construction proved the source bindings of the recorded revision, so an
        unchanged revision keeps that proof.
        """
        self.geometry.require_current()
        from ._patches import _validate_surface_source_shadows

        for patch in self.patches:
            _validate_surface_source_shadows(patch)
        bounds = np.asarray(self.parameter_bounds, dtype=np.float64)
        current_revision = _brep_model_source_revision(
            self.geometry,
            self.patches,
            bounds,
            np.asarray(self.orientation, dtype=np.float64),
            self.trim_domains,
            self.physical_tags,
            self.coordinate_contract,
            self.report,
        )
        if current_revision != self.source_revision or self.model_id != _brep_model_id(
            self.source_revision, self.tessellation
        ):
            raise ValueError(
                "BRepModel content has changed since its source revision was recorded."
            )

    @staticmethod
    def _validate_tessellation(
        tessellation: BRepTessellation,
        source_revision: str,
        geometry: BRepGeometry,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: np.ndarray,
        orientation: Array,
        /,
    ) -> None:
        if tessellation.source_revision != source_revision:
            raise ValueError("Tessellation source_revision does not match its B-Rep.")
        vertices = np.asarray(tessellation.vertices)
        triangles = np.asarray(tessellation.triangles)
        face_ids = np.asarray(tessellation.face_ids)
        parameters = np.asarray(tessellation.parameters)
        dimensions = np.asarray(tessellation.vertex_source_dimensions)
        source_indices = np.asarray(tessellation.vertex_source_indices)
        vertex_parameters = np.asarray(tessellation.vertex_parameters)
        triangle_occurrences = np.asarray(tessellation.triangle_occurrence_ids)
        vertex_occurrences = np.asarray(tessellation.vertex_occurrence_ids)
        if np.max(face_ids) >= len(patches):
            raise ValueError("Tessellation references an absent B-Rep face.")

        face_edges = tuple(
            frozenset(geometry.coedge_edges[coedge] for loop in loops for coedge in loop)
            for loops in geometry.face_loops
        )
        face_vertices = tuple(
            frozenset(vertex for edge in edges for vertex in geometry.edge_vertices[edge])
            for edges in face_edges
        )
        occurrences = tuple(geometry.assembly_graph.iter_occurrences())
        occurrence_faces = tuple(
            frozenset(geometry.topology.solid_faces[value.solid_index])
            for value in occurrences
        )
        # A face shared by occurrences with one exact placement is one placed copy,
        # tessellated once under the first such occurrence (`MeshingDomain` slots).
        from .._meshing_domain import _placed_copy_keys

        copy_keys = _placed_copy_keys(occurrences)
        copy_index: dict[tuple[tuple[tuple[Fraction, ...], ...], int], int] = {}
        occurrence_copies = np.asarray(
            [
                copy_index.setdefault(copy_keys[value.path], index)
                for index, value in enumerate(occurrences)
            ]
            + [-1],
            dtype=np.int64,
        )
        copy_owner: dict[tuple[int, int], int] = {}
        for occurrence, faces in enumerate(occurrence_faces):
            for face in faces:
                copy_owner.setdefault(
                    (face, int(occurrence_copies[occurrence])), occurrence
                )
        expected_pairs = set(
            (face, occurrence) for (face, _), occurrence in copy_owner.items()
        )
        instantiated_faces = {face for face, _ in expected_pairs}
        expected_pairs.update(
            (face, -1) for face in range(len(patches)) if face not in instantiated_faces
        )
        observed_pairs = set(
            zip(face_ids.tolist(), triangle_occurrences.tolist(), strict=True)
        )
        if observed_pairs != expected_pairs:
            raise ValueError(
                "Tessellation must cover every source face occurrence exactly by its definition and use."
            )
        for row, (face, occurrence) in enumerate(
            zip(face_ids.tolist(), triangle_occurrences.tolist(), strict=True)
        ):
            if occurrence >= 0 and face not in occurrence_faces[occurrence]:
                raise ValueError(
                    "Tessellation face is not a member of its source occurrence."
                )
            # Index -1 (unplaced) maps to the trailing -1 copy.
            if np.any(
                occurrence_copies[vertex_occurrences[triangles[row]]]
                != occurrence_copies[occurrence]
            ):
                raise ValueError(
                    "Tessellation triangles cannot weld vertices across placed source copies."
                )
        for dimension, count in (
            (0, geometry.topology.num_vertices),
            (1, geometry.topology.num_edges),
            (2, len(patches)),
        ):
            rows = dimensions == dimension
            if np.any(source_indices[rows] >= count):
                raise ValueError(
                    "Tessellation vertex lineage references an absent entity."
                )
        if np.any(vertex_occurrences >= len(occurrences)):
            raise ValueError("Tessellation references an absent source occurrence.")
        for row in np.flatnonzero(dimensions == 1):
            edge = int(source_indices[row])
            lower, upper = np.asarray(geometry.edge_ranges)[edge]
            parameter = vertex_parameters[row, 0]
            if parameter < lower or parameter > upper:
                raise ValueError(
                    "Tessellation edge parameter lies outside its source range."
                )
        face_vertex_rows = np.flatnonzero(dimensions == 2)
        if face_vertex_rows.size:
            face_indices = source_indices[face_vertex_rows].astype(np.intp)
            values = vertex_parameters[face_vertex_rows]
            if np.any(values < parameter_bounds[face_indices, 0]) or np.any(
                values > parameter_bounds[face_indices, 1]
            ):
                raise ValueError("Tessellation face parameters lie outside their charts.")
        triangle_bounds = parameter_bounds[face_ids]
        if np.any(parameters < triangle_bounds[:, None, 0, :]) or np.any(
            parameters > triangle_bounds[:, None, 1, :]
        ):
            raise ValueError("Tessellation triangle parameters lie outside their charts.")

        expected_vertices = vertices[triangles]
        placement_rows = {
            index: value.placement for index, value in enumerate(occurrences)
        }
        group_rows: dict[tuple[int, int], list[int]] = {}
        for row, (face, occurrence) in enumerate(
            zip(face_ids.tolist(), triangle_occurrences.tolist(), strict=True)
        ):
            group_rows.setdefault((int(face), int(occurrence)), []).append(row)
        for (face_index, occurrence), row_values in group_rows.items():
            rows = np.asarray(row_values, dtype=np.int64)
            patch = patches[face_index]
            chart = parameters[rows]
            corners = expected_vertices[rows]
            placement = None if occurrence < 0 else placement_rows[occurrence]
            evaluated = np.asarray(
                patch.evaluate(jnp.asarray(chart.reshape((-1, 2)), dtype=jnp.float64)),
                dtype=np.float64,
            ).reshape((-1, 3))
            differential = np.asarray(
                jax.vmap(jax.jacfwd(patch.evaluate))(
                    jnp.asarray(np.mean(chart, axis=1), dtype=jnp.float64)
                ),
                dtype=np.float64,
            )
            face_sign = int(np.asarray(orientation)[face_index])
            if placement is not None:
                matrix = np.asarray(
                    [
                        [float(placement.matrix[row][column]) for column in range(3)]
                        for row in range(3)
                    ],
                    dtype=np.float64,
                )
                offset = np.asarray(
                    [float(placement.matrix[row][3]) for row in range(3)],
                    dtype=np.float64,
                )
                evaluated = evaluated @ matrix.T + offset
                differential = np.einsum("ij,njk->nik", matrix, differential)
                linear = placement.matrix
                determinant = (
                    linear[0][0]
                    * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
                    - linear[0][1]
                    * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
                    + linear[0][2]
                    * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
                )
                if determinant < 0:
                    face_sign = -face_sign
            scale = max(1.0, float(np.max(np.abs(corners), initial=0.0)))
            if not np.allclose(
                evaluated,
                corners.reshape((-1, 3)),
                rtol=0.0,
                atol=128.0 * np.finfo(np.float64).eps * scale,
            ):
                raise ValueError(
                    "Tessellation source parameters do not reproduce their placed vertices."
                )
            uv_area = (chart[:, 1, 0] - chart[:, 0, 0]) * (
                chart[:, 2, 1] - chart[:, 0, 1]
            ) - (chart[:, 1, 1] - chart[:, 0, 1]) * (chart[:, 2, 0] - chart[:, 0, 0])
            cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            if (
                not np.all(np.isfinite(cross))
                or np.any(uv_area == 0.0)
                or np.any(np.linalg.norm(cross, axis=1) == 0.0)
            ):
                raise ValueError(
                    "Tessellation triangles must be finite and nondegenerate."
                )
            source_normals = (
                np.cross(differential[:, :, 0], differential[:, :, 1]) * face_sign
            )
            alignment = np.einsum("ij,ij->i", cross, source_normals)
            if not np.all(np.isfinite(source_normals)) or np.any(alignment <= 0.0):
                raise ValueError(
                    "Tessellation triangle orientation disagrees with its placed face."
                )
            for global_row in rows.tolist():
                triangle = triangles[global_row]
                for vertex in triangle.tolist():
                    dimension = int(dimensions[vertex])
                    source_index = int(source_indices[vertex])
                    if dimension == 0 and source_index not in face_vertices[face_index]:
                        raise ValueError(
                            "Tessellation vertex lineage is not incident to its source face."
                        )
                    if dimension == 1 and source_index not in face_edges[face_index]:
                        raise ValueError(
                            "Tessellation edge lineage is not incident to its source face."
                        )
                    if dimension == 2 and source_index != face_index:
                        raise ValueError(
                            "Tessellation face-interior lineage disagrees with its triangle face."
                        )

        deviation = np.asarray(tessellation.deviation_bounds)
        normal = np.asarray(tessellation.normal_bounds)
        if np.any(deviation > tessellation.linear_deflection) or np.any(
            normal > tessellation.angular_deflection
        ):
            raise ValueError("Tessellation bounds exceed their physical policy.")

    def _validate_tessellation_source_witness(
        self,
        tessellation: BRepTessellation,
        certificate: SourceFidelityCertificate | None,
        /,
    ) -> None:
        from ...discretization._cell_geometry import CellGeometrySpec
        from ...discretization._cell_geometry_validity import cell_geometry_id
        from ...discretization._cell_mesh import CellMesh
        from ...discretization._surface_chart_records import SurfaceRootMeshCellBinding
        from .._domain_boundary import MeshingDomainBoundarySource
        from .._meshing_domain import MeshingDomain
        from .._source_fidelity import certify_source_fidelity

        witness = tessellation.source_chart_witness
        if not isinstance(witness, SurfaceChartWitness):
            raise TypeError(
                "Tessellation source_chart_witness must be a SurfaceChartWitness."
            )
        domain = MeshingDomain.from_brep(self)
        if witness.domain_id != domain.domain_id:
            raise ValueError(
                "Tessellation chart witness belongs to a different source domain."
            )
        triangle_ids = np.asarray(tessellation.triangle_global_ids, dtype=np.int64)
        triangle_rows = {
            int(identifier): row for row, identifier in enumerate(triangle_ids)
        }
        occurrences = tuple(self.geometry.assembly_graph.iter_occurrences())
        occurrence_rows = {value.path: row for row, value in enumerate(occurrences)}
        witness_patches = np.asarray(witness.patches, dtype=np.int64)
        witness_charts = np.asarray(witness.charts, dtype=np.float64)
        witness_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
        face_ids = np.asarray(tessellation.face_ids, dtype=np.int64)
        occurrence_ids = np.asarray(tessellation.triangle_occurrence_ids, dtype=np.int64)
        parameters = np.asarray(tessellation.parameters, dtype=np.float64)
        for witness_row, (cell_id, patch) in enumerate(
            zip(witness_ids.tolist(), witness_patches.tolist(), strict=True)
        ):
            if patch < 0 or patch >= len(domain.source_indices[2]):
                raise ValueError(
                    "Tessellation chart witness names an absent source patch."
                )
            row = triangle_rows[int(cell_id)]
            face = int(domain.source_indices[2][patch])
            path = domain.source_occurrences[2][patch]
            occurrence = -1 if not path else occurrence_rows.get(path, -2)
            if occurrence == -2:
                raise ValueError(
                    "Tessellation chart witness names an absent source occurrence."
                )
            chart = witness_charts[witness_row]
            try:
                c1 = self.patches[face].is_c1_on(
                    np.stack((np.min(chart, axis=0), np.max(chart, axis=0)))
                )
            except (NotImplementedError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Tessellation source-chart C1 continuity is unresolved: {error}"
                ) from error
            if not c1:
                raise ValueError(
                    "Tessellation source chart has no established C1 continuity."
                )
            if face_ids[row] != face or occurrence_ids[row] != occurrence:
                raise ValueError(
                    "Tessellation face/occurrence ancestry disagrees with its exact chart witness."
                )
            if not np.array_equal(parameters[row], witness_charts[witness_row]):
                raise ValueError(
                    "Tessellation UV ancestry differs from its exact source chart witness."
                )

        mesh = CellMesh.from_triangles(
            np.asarray(tessellation.vertices),
            np.asarray(tessellation.triangles),
            vertex_global_ids=np.asarray(tessellation.vertex_global_ids, dtype=np.int64),
            cell_global_ids=triangle_ids,
            numeric_version=self.source_revision,
            multiregion_surface=_tessellation_multiregion_surface(domain, tessellation),
        )
        coordinate_geometry = CellGeometrySpec.affine(mesh)
        root_cell_ids = {
            int(value)
            for block in mesh.blocks
            for value in np.asarray(block.global_ids, dtype=np.int64)
        }
        if root_cell_ids != set(witness_ids.tolist()):
            raise ValueError(
                "Tessellation witness root-cell IDs do not match "
                "the retained source mesh."
            )
        root_geometry_id = cell_geometry_id(coordinate_geometry)
        root_topology_id = mesh.topology_id
        for witness_row, cell_id in enumerate(witness_ids.tolist()):
            expected_binding = SurfaceRootMeshCellBinding(
                root_geometry_id,
                root_topology_id,
                int(cell_id),
            )
            if witness.root_mesh_cell_bindings[witness_row] != expected_binding:
                raise ValueError(
                    "Tessellation witness root-cell binding does not match "
                    "its actual source mesh."
                )
        boundary = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
        if certificate is None:
            certificate = certify_source_fidelity(
                mesh,
                coordinate_geometry,
                boundary,
                tolerance=tessellation.linear_deflection,
                maximum_normal_deviation=tessellation.angular_deflection,
                witness=witness,
            )
            # The proof was just derived from exactly these inputs; a full replay
            # would recompute the same certificate. Only its identity binding is checked.
            certificate.binding.require(mesh, coordinate_geometry, source=boundary)
        else:
            # The tessellation's own proof answers this exact question when it
            # binds these inputs; it is reused, never recomputed.
            certificate.require_inputs(
                mesh,
                coordinate_geometry,
                boundary,
                tolerance=tessellation.linear_deflection,
                maximum_normal_deviation=tessellation.angular_deflection,
                witness=witness,
            )
        if certificate.status != "certified":
            details = "; ".join(finding.detail for finding in certificate.findings)
            raise ValueError(
                "Tessellation source-fidelity witness is not certified: " + details
            )
        proof_rows = {
            int(identifier): row
            for row, identifier in enumerate(certificate.source_cell_global_ids)
        }
        if set(proof_rows) != set(triangle_ids.tolist()):
            raise ValueError(
                "Canonical source-fidelity proof omits a tessellation triangle."
            )
        deviation = np.asarray(tessellation.deviation_bounds)
        normal = np.asarray(tessellation.normal_bounds)
        if certificate.normal_deviation_bounds is None:
            raise ValueError(
                "Canonical source-fidelity proof omitted oriented-normal bounds."
            )
        proof_deviation = np.asarray(certificate.deviation_bounds, dtype=np.float64)
        proof_normal = np.asarray(certificate.normal_deviation_bounds, dtype=np.float64)
        for row, identifier in enumerate(triangle_ids.tolist()):
            proof_row = proof_rows[int(identifier)]
            if (
                deviation[row] < proof_deviation[proof_row, 1]
                or normal[row] < proof_normal[proof_row]
            ):
                raise ValueError(
                    "Stored tessellation bounds do not enclose canonical continuous source fidelity."
                )

    def evaluate_face_carrier(
        self, face_ids: ConvertibleToArray, parameters: ConvertibleToArray, /
    ) -> Array:
        """Evaluate source face carriers at raw UV values without tessellation.

        This evaluates the rectangular carrier chart, not trim membership,
        projection, or solid validity.
        """
        indices = jnp.asarray(face_ids)
        values = jnp.asarray(parameters, dtype=jnp.float64)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("face_ids must have an integer dtype.")
        if values.shape != (*indices.shape, 2):
            raise ValueError("parameters must have shape (*face_ids.shape, 2).")
        indices = eqx.error_if(
            indices,
            jnp.any((indices < 0) | (indices >= len(self.patches))),
            "face_ids reference an absent source face.",
        )
        values = eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), "parameters must be finite."
        )
        safe_indices = jnp.clip(indices, 0, len(self.patches) - 1)
        bounds = self.parameter_bounds[safe_indices]
        values = eqx.error_if(
            values,
            jnp.any(values < bounds[..., 0, :]) | jnp.any(values > bounds[..., 1, :]),
            "parameters lie outside a source face chart.",
        )
        reference = (values - bounds[..., 0, :]) / (bounds[..., 1, :] - bounds[..., 0, :])
        return BRepBoundaryMap(self.patches, self.parameter_bounds).map(
            safe_indices, reference
        )

    def require_tessellation(self) -> BRepTessellation:
        if self.tessellation is None:
            raise ValueError(
                "This B-Rep is source-only; request native tessellation explicitly."
            )
        return self.tessellation

    @property
    def tessellation_id(self) -> str | None:
        return None if self.tessellation is None else self.tessellation.tessellation_id

    @property
    def mesh_vertices(self) -> Array:
        return self.require_tessellation().vertices

    @property
    def mesh_faces(self) -> Array:
        return self.require_tessellation().triangles

    @property
    def triangle_face_ids(self) -> Array:
        return self.require_tessellation().face_ids

    @property
    def triangle_parameters(self) -> Array:
        return self.require_tessellation().parameters

    @property
    def source_id(self) -> str:
        return self.report.source_id

    @property
    def source_digest(self) -> str:
        return self.report.source_digest

    @property
    def import_policy_id(self) -> str:
        return self.report.import_policy_id

    @property
    def topology(self) -> BRepTopology:
        return self.geometry.topology

    def _occurrence_for_source_entity(
        self, kind: str, index: int, occurrence_path: tuple[str, ...], /
    ) -> BRepAffinePlacement | None:
        if not isinstance(occurrence_path, tuple) or any(
            not isinstance(part, str) or not part for part in occurrence_path
        ):
            raise TypeError("occurrence_path must be a tuple of nonempty strings.")
        if not occurrence_path:
            return None
        if kind not in ("edge", "vertex"):
            raise ValueError(
                "Occurrence source bounds support only edge and vertex entities."
            )
        matches = tuple(
            occurrence
            for occurrence in self.geometry.assembly_graph.iter_occurrences()
            if occurrence.path == occurrence_path
        )
        if len(matches) != 1:
            raise ValueError(
                "occurrence_path must identify exactly one source assembly instance."
            )
        occurrence = matches[0]
        faces = self.topology.solid_faces[occurrence.solid_index]
        edges = {
            edge
            for face in faces
            for wire in self.topology.face_wires[face]
            for edge in wire
        }
        vertices = {
            vertex for edge in edges for vertex in self.geometry.edge_vertices[edge]
        }
        if (kind == "edge" and index not in edges) or (
            kind == "vertex" and index not in vertices
        ):
            raise ValueError(
                "Source entity is not incident to the named solid occurrence."
            )
        return occurrence.placement

    @staticmethod
    def _place_exact_bounds(
        placement: BRepAffinePlacement | None,
        lower: tuple[Fraction, Fraction, Fraction],
        upper: tuple[Fraction, Fraction, Fraction],
        /,
    ) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
        if placement is None:
            return lower, upper
        matrix = placement.matrix
        result_lower = []
        result_upper = []
        for row in range(3):
            low = high = matrix[row][3]
            for column in range(3):
                coefficient = matrix[row][column]
                if coefficient >= 0:
                    low += coefficient * lower[column]
                    high += coefficient * upper[column]
                else:
                    low += coefficient * upper[column]
                    high += coefficient * lower[column]
            result_lower.append(low)
            result_upper.append(high)
        return (
            (result_lower[0], result_lower[1], result_lower[2]),
            (result_upper[0], result_upper[1], result_upper[2]),
        )

    def source_vertex_point_bounds(
        self, vertex_index: int, /, *, occurrence_path: tuple[str, ...] = ()
    ) -> tuple[tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction]]:
        """Return outward rational bounds for one source or placed vertex."""
        index = _index(vertex_index, "vertex_index")
        lower, upper = self.geometry.source_vertex_point_bounds(index)
        placement = self._occurrence_for_source_entity("vertex", index, occurrence_path)
        return self._place_exact_bounds(placement, lower, upper)

    def source_curve_parameter_domain(
        self, edge_index: int, /, *, occurrence_path: tuple[str, ...] = ()
    ) -> tuple[float, float]:
        """Return one source edge's authored finite curve parameter interval."""
        index = _index(edge_index, "edge_index")
        self._occurrence_for_source_entity("edge", index, occurrence_path)
        return self.geometry.source_curve_parameter_domain(index)

    def source_curve_point_bounds(
        self,
        edge_index: int,
        lower: float,
        upper: float,
        /,
        *,
        occurrence_path: tuple[str, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return outward 3D bounds for one source edge interval and occurrence."""
        index = _index(edge_index, "edge_index")
        lower_values, upper_values = self.geometry.source_curve_point_bounds(
            index, lower, upper
        )
        placement = self._occurrence_for_source_entity("edge", index, occurrence_path)
        exact_lower = tuple(Fraction.from_float(float(value)) for value in lower_values)
        exact_upper = tuple(Fraction.from_float(float(value)) for value in upper_values)
        placed_lower, placed_upper = self._place_exact_bounds(
            placement,
            (exact_lower[0], exact_lower[1], exact_lower[2]),
            (exact_upper[0], exact_upper[1], exact_upper[2]),
        )
        from .._rational import fraction_down, fraction_up

        return (
            np.asarray(
                tuple(fraction_down(value) for value in placed_lower), dtype=np.float64
            ),
            np.asarray(
                tuple(fraction_up(value) for value in placed_upper), dtype=np.float64
            ),
        )

    @property
    def face_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_id, self.source_revision, "face", index)
            for index in range(len(self.patches))
        )

    @property
    def edge_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_id, self.source_revision, "edge", index)
            for index in range(self.topology.num_edges)
        )

    @property
    def vertex_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_id, self.source_revision, "vertex", index)
            for index in range(self.topology.num_vertices)
        )

    @property
    def solid_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_id, self.source_revision, "solid", index)
            for index in range(self.topology.num_solids)
        )

    def tessellation_multiregion_surface(self, /) -> MultiRegionSurfaceTopology | None:
        """Region-labeled topology of this model's tessellation.

        Each triangle carries the solid occurrences on both sides of its source
        face: a face shared by two solids is one triangulation with two sides,
        and every solid's closed boundary is its side selection. ``None`` when no
        face separates two solid occurrences.
        """
        from .._meshing_domain import MeshingDomain

        if self.tessellation is None:
            raise ValueError("The B-Rep model has no tessellation.")
        return _tessellation_multiregion_surface(
            MeshingDomain.from_brep(self), self.tessellation
        )

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        return BoundaryAtlas(
            BRepBoundaryMap(self.patches, self.parameter_bounds),
            source_entity_ids=jnp.arange(len(self.patches), dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=self.physical_tags,
            orientation=self.orientation,
            trim_domains=self.trim_domains,
        )


__all__ = [
    "BRepBoundaryMap",
    "BRepAssemblyContainer",
    "BRepEntityId",
    "BRepTessellation",
    "BRepGeometry",
    "BRepQualifiedIncidence",
    "BRepImportReport",
    "BRepModel",
    "BRepSourceBindingProof",
    "BRepTopology",
]

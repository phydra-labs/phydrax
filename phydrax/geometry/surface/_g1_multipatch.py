#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable biquintic multipatch topology and linear G1 gluing plans.

The supported G1 construction is the parametric-C1 specialization of geometric
G1: corresponding traces agree and the two outward cross-boundary derivatives
sum to zero.  This is deliberately narrower than general polynomial gluing data;
extraordinary vertices are classified but fail closed for G1 basis preparation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum, IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import bspline_jet_stencil, TensorBSplineJetPlan
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import (
    FactorizationPolicy,
    factorize,
    LinearSubspace,
    RankPolicy,
)
from ...sparse import EdgeRelation, SparseLinearMap


_BIQUINTIC_DEGREE = 5
_CONTROL_COUNT = _BIQUINTIC_DEGREE + 1
_CONTROL_COUNT_PER_PATCH = _CONTROL_COUNT**2
_OPEN_BIQUINTIC_KNOTS = (0.0,) * _CONTROL_COUNT + (1.0,) * _CONTROL_COUNT


class PatchEdge(IntEnum):
    """Counter-clockwise local half-edges of a tensor-product patch."""

    V_MIN = 0
    U_MAX = 1
    V_MAX = 2
    U_MIN = 3


class PatchVertexKind(str, Enum):
    """Quad-mesh vertex class used by the supported gluing path."""

    REGULAR_INTERIOR = "regular_interior"
    REGULAR_BOUNDARY = "regular_boundary"
    REGULAR_CORNER = "regular_corner"
    EXTRAORDINARY_INTERIOR = "extraordinary_interior"
    EXTRAORDINARY_BOUNDARY = "extraordinary_boundary"


class G1PatchStatus(str, Enum):
    """Preparation status without an unqualified G1-support claim."""

    SUPPORTED = "supported_parametric_c1"
    NO_SHARED_EDGES_UNSUPPORTED = "no_shared_edges_unsupported"
    EXTRAORDINARY_INTERIOR_UNSUPPORTED = "extraordinary_interior_unsupported"
    EXTRAORDINARY_BOUNDARY_UNSUPPORTED = "extraordinary_boundary_unsupported"
    EXTRAORDINARY_MIXED_UNSUPPORTED = "extraordinary_mixed_unsupported"
    MALFORMED_TOPOLOGY = "malformed_topology"


class GluingContinuity(str, Enum):
    """Linear trace order assembled for every shared patch edge."""

    G0 = "g0"
    G1 = "g1_parametric_c1"


class PatchTopologyError(ValueError):
    """Malformed half-edge input rejected before any numerical preparation."""

    def __init__(self, message: str, /):
        message_ = str(message)
        if not message_:
            raise ValueError("Patch topology errors require a message.")
        self.status = G1PatchStatus.MALFORMED_TOPOLOGY
        super().__init__(message_)


class UnsupportedG1TopologyError(ValueError):
    """A classified topology outside the supported regular G1 path."""

    def __init__(
        self,
        status: G1PatchStatus,
        unsupported_vertex_labels: Sequence[str],
        /,
    ):
        if status is G1PatchStatus.SUPPORTED:
            raise ValueError("A supported topology cannot raise an unsupported error.")
        self.status = status
        self.unsupported_vertex_labels = tuple(unsupported_vertex_labels)
        labels = ", ".join(self.unsupported_vertex_labels) or "none"
        super().__init__(
            f"G1 basis preparation status is {status.value}; classified vertices: {labels}."
        )


def _nonempty_labels(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    labels = tuple(str(value) for value in values)
    if not labels or any(not label for label in labels):
        raise PatchTopologyError(f"{name} must contain non-empty labels.")
    if len(set(labels)) != len(labels):
        raise PatchTopologyError(f"{name} must be unique.")
    return labels


def _patch_status(
    kinds: tuple[PatchVertexKind, ...],
    shared_edge_count: int,
    /,
) -> G1PatchStatus:
    interior = any(kind is PatchVertexKind.EXTRAORDINARY_INTERIOR for kind in kinds)
    boundary = any(kind is PatchVertexKind.EXTRAORDINARY_BOUNDARY for kind in kinds)
    if interior and boundary:
        return G1PatchStatus.EXTRAORDINARY_MIXED_UNSUPPORTED
    if interior:
        return G1PatchStatus.EXTRAORDINARY_INTERIOR_UNSUPPORTED
    if boundary:
        return G1PatchStatus.EXTRAORDINARY_BOUNDARY_UNSUPPORTED
    if shared_edge_count == 0:
        return G1PatchStatus.NO_SHARED_EDGES_UNSUPPORTED
    return G1PatchStatus.SUPPORTED


class PreparedHalfEdgePatchTopology(StrictModule, NonTrainableState):
    """Canonical immutable half-edge topology for oriented quadrilateral patches.

    ``patch_vertices`` uses the tensor order ``(canonical, canonical, canonical, canonical)``. Patch and
    vertex labels are canonicalized lexicographically. Local half-edges traverse
    the patch boundary counter-clockwise; every shared edge must therefore occur
    once in each direction. ``halfedge_orientation`` is +1 when a half-edge runs
    from the lexicographically smaller endpoint to the larger endpoint and -1
    otherwise.
    """

    patch_labels: tuple[str, ...] = eqx.field(static=True)
    vertex_labels: tuple[str, ...] = eqx.field(static=True)
    patch_vertices: Array
    halfedge_patch_indices: Array
    halfedge_local_edges: Array
    halfedge_origin_vertices: Array
    halfedge_destination_vertices: Array
    halfedge_twins: Array
    halfedge_canonical_edges: Array
    halfedge_orientation: Array
    canonical_edges: Array
    shared_halfedges: Array
    boundary_halfedges: Array
    vertex_valence: Array
    vertex_kinds: tuple[PatchVertexKind, ...] = eqx.field(static=True)
    unsupported_vertex_labels: tuple[str, ...] = eqx.field(static=True)
    status: G1PatchStatus = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_labels: Sequence[str],
        patch_vertices: Sequence[Sequence[str]],
        /,
    ):
        labels = _nonempty_labels(patch_labels, "patch_labels")
        corners = tuple(
            tuple(str(vertex) for vertex in patch) for patch in patch_vertices
        )
        if len(corners) != len(labels):
            raise PatchTopologyError(
                "patch_vertices must contain one four-corner record per patch."
            )
        if any(len(patch) != 4 for patch in corners):
            raise PatchTopologyError(
                "Each patch must provide vertices in (canonical, canonical, canonical, canonical) order."
            )
        if any(any(not vertex for vertex in patch) for patch in corners):
            raise PatchTopologyError("Patch vertex labels must be non-empty.")
        if any(len(set(patch)) != 4 for patch in corners):
            raise PatchTopologyError("A patch must contain four distinct vertices.")

        records = tuple(
            sorted(zip(labels, corners, strict=True), key=lambda item: item[0])
        )
        canonical_patch_labels = tuple(label for label, _ in records)
        canonical_corners = tuple(patch for _, patch in records)
        vertex_labels = tuple(
            sorted({vertex for patch in canonical_corners for vertex in patch})
        )
        vertex_index = {label: index for index, label in enumerate(vertex_labels)}
        patch_vertex_indices = np.asarray(
            [[vertex_index[vertex] for vertex in patch] for patch in canonical_corners],
            dtype=np.int32,
        )

        patch_count = len(canonical_patch_labels)
        halfedge_count = 4 * patch_count
        halfedge_patch = np.repeat(np.arange(patch_count, dtype=np.int32), 4)
        halfedge_local = np.tile(np.arange(4, dtype=np.int32), patch_count)
        origin = np.empty((halfedge_count,), dtype=np.int32)
        destination = np.empty((halfedge_count,), dtype=np.int32)
        for patch_index, vertices in enumerate(patch_vertex_indices):
            start = 4 * patch_index
            origin[start : start + 4] = vertices
            destination[start : start + 4] = vertices[
                np.asarray((1, 2, 3, 0), dtype=np.int32)
            ]

        edge_routes: dict[tuple[int, int], list[int]] = {}
        for halfedge in range(halfedge_count):
            endpoints = (int(origin[halfedge]), int(destination[halfedge]))
            key = (min(endpoints), max(endpoints))
            edge_routes.setdefault(key, []).append(halfedge)
        if any(len(routes) > 2 for routes in edge_routes.values()):
            raise PatchTopologyError(
                "An undirected patch edge may belong to at most two patches."
            )

        edge_keys = tuple(sorted(edge_routes))
        edge_index = {key: index for index, key in enumerate(edge_keys)}
        twins = np.full((halfedge_count,), -1, dtype=np.int32)
        canonical_edge = np.empty((halfedge_count,), dtype=np.int32)
        orientation = np.empty((halfedge_count,), dtype=np.int32)
        shared: list[tuple[int, int]] = []
        boundary: list[int] = []
        for key in edge_keys:
            routes = edge_routes[key]
            for halfedge in routes:
                canonical_edge[halfedge] = edge_index[key]
                orientation[halfedge] = 1 if int(origin[halfedge]) == key[0] else -1
            if len(routes) == 1:
                boundary.append(routes[0])
                continue
            left, right = routes
            if not (
                origin[left] == destination[right] and destination[left] == origin[right]
            ):
                raise PatchTopologyError(
                    "Shared half-edges must have opposite patch-boundary orientations."
                )
            twins[left], twins[right] = right, left
            ordered = tuple(
                sorted(
                    (left, right),
                    key=lambda value: (
                        canonical_patch_labels[int(halfedge_patch[value])],
                        int(halfedge_local[value]),
                    ),
                )
            )
            shared.append((ordered[0], ordered[1]))
        shared.sort(key=lambda pair: int(canonical_edge[pair[0]]))
        boundary.sort(key=lambda value: int(canonical_edge[value]))

        boundary_degree = np.zeros((len(vertex_labels),), dtype=np.int32)
        for halfedge in boundary:
            boundary_degree[int(origin[halfedge])] += 1
            boundary_degree[int(destination[halfedge])] += 1
        if np.any((boundary_degree != 0) & (boundary_degree != 2)):
            raise PatchTopologyError(
                "A manifold quad boundary must meet each boundary vertex in two edges."
            )

        vertex_patch_sets = [set() for _ in vertex_labels]
        for patch_index, vertices in enumerate(patch_vertex_indices):
            for vertex in vertices:
                vertex_patch_sets[int(vertex)].add(patch_index)
        valence = np.asarray(
            [len(incident) for incident in vertex_patch_sets], dtype=np.int32
        )
        kinds: list[PatchVertexKind] = []
        for vertex in range(len(vertex_labels)):
            if boundary_degree[vertex] == 0:
                kind = (
                    PatchVertexKind.REGULAR_INTERIOR
                    if valence[vertex] == 4
                    else PatchVertexKind.EXTRAORDINARY_INTERIOR
                )
            elif valence[vertex] == 1:
                kind = PatchVertexKind.REGULAR_CORNER
            elif valence[vertex] == 2:
                kind = PatchVertexKind.REGULAR_BOUNDARY
            else:
                kind = PatchVertexKind.EXTRAORDINARY_BOUNDARY
            kinds.append(kind)
        vertex_kinds = tuple(kinds)
        unsupported = tuple(
            vertex_labels[index]
            for index, kind in enumerate(vertex_kinds)
            if kind
            in (
                PatchVertexKind.EXTRAORDINARY_INTERIOR,
                PatchVertexKind.EXTRAORDINARY_BOUNDARY,
            )
        )
        status = _patch_status(vertex_kinds, len(shared))

        canonical_edges = np.asarray(edge_keys, dtype=np.int32).reshape((-1, 2))
        shared_array = np.asarray(shared, dtype=np.int32).reshape((-1, 2))
        boundary_array = np.asarray(boundary, dtype=np.int32)
        topology_id = canonical_fingerprint(
            {
                "kind": "prepared-biquintic-halfedge-topology",
                "patch_labels": canonical_patch_labels,
                "vertex_labels": vertex_labels,
                "patch_vertices": array_tree_fingerprint(patch_vertex_indices),
                "halfedge_twins": array_tree_fingerprint(twins),
                "canonical_edges": array_tree_fingerprint(canonical_edges),
                "shared_halfedges": array_tree_fingerprint(shared_array),
                "vertex_kinds": tuple(kind.value for kind in vertex_kinds),
                "status": status.value,
            }
        )

        self.patch_labels = canonical_patch_labels
        self.vertex_labels = vertex_labels
        self.patch_vertices = jnp.asarray(patch_vertex_indices)
        self.halfedge_patch_indices = jnp.asarray(halfedge_patch)
        self.halfedge_local_edges = jnp.asarray(halfedge_local)
        self.halfedge_origin_vertices = jnp.asarray(origin)
        self.halfedge_destination_vertices = jnp.asarray(destination)
        self.halfedge_twins = jnp.asarray(twins)
        self.halfedge_canonical_edges = jnp.asarray(canonical_edge)
        self.halfedge_orientation = jnp.asarray(orientation)
        self.canonical_edges = jnp.asarray(canonical_edges)
        self.shared_halfedges = jnp.asarray(shared_array)
        self.boundary_halfedges = jnp.asarray(boundary_array)
        self.vertex_valence = jnp.asarray(valence)
        self.vertex_kinds = vertex_kinds
        self.unsupported_vertex_labels = unsupported
        self.status = status
        self.topology_id = topology_id

    @property
    def patch_count(self) -> int:
        return len(self.patch_labels)

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_labels)

    @property
    def halfedge_count(self) -> int:
        return 4 * self.patch_count

    @property
    def shared_edge_count(self) -> int:
        return self.shared_halfedges.shape[0]

    def vertex_kind(self, label: str, /) -> PatchVertexKind:
        label_ = str(label)
        if label_ not in self.vertex_labels:
            raise KeyError(f"Unknown patch vertex {label_!r}.")
        return self.vertex_kinds[self.vertex_labels.index(label_)]

    def halfedge_index(self, patch_label: str, edge: PatchEdge, /) -> int:
        label = str(patch_label)
        if label not in self.patch_labels:
            raise KeyError(f"Unknown patch {label!r}.")
        if not isinstance(edge, PatchEdge):
            raise TypeError("edge must be a PatchEdge.")
        return 4 * self.patch_labels.index(label) + int(edge)


def _edge_control_routes(
    patch_index: int,
    edge: PatchEdge,
    orientation: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(_CONTROL_COUNT, dtype=np.int32)
    reverse = local[::-1]
    if edge is PatchEdge.V_MIN:
        u, v = local, np.zeros_like(local)
        inner_u, inner_v = local, np.ones_like(local)
    elif edge is PatchEdge.U_MAX:
        u, v = np.full_like(local, _BIQUINTIC_DEGREE), local
        inner_u, inner_v = np.full_like(local, _BIQUINTIC_DEGREE - 1), local
    elif edge is PatchEdge.V_MAX:
        u, v = reverse, np.full_like(local, _BIQUINTIC_DEGREE)
        inner_u, inner_v = reverse, np.full_like(local, _BIQUINTIC_DEGREE - 1)
    else:
        u, v = np.zeros_like(local), reverse
        inner_u, inner_v = np.ones_like(local), reverse
    boundary = patch_index * _CONTROL_COUNT_PER_PATCH + u * _CONTROL_COUNT + v
    interior = patch_index * _CONTROL_COUNT_PER_PATCH + inner_u * _CONTROL_COUNT + inner_v
    if orientation == -1:
        boundary = boundary[::-1]
        interior = interior[::-1]
    return boundary, interior


def _constraint_status(
    topology: PreparedHalfEdgePatchTopology,
    continuity: GluingContinuity,
    /,
) -> G1PatchStatus:
    if topology.shared_edge_count == 0:
        return G1PatchStatus.NO_SHARED_EDGES_UNSUPPORTED
    if continuity is GluingContinuity.G0:
        return G1PatchStatus.SUPPORTED
    return topology.status


def _assemble_constraint_routes(
    topology: PreparedHalfEdgePatchTopology,
    continuity: GluingContinuity,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows_per_seam = _CONTROL_COUNT * (2 if continuity is GluingContinuity.G1 else 1)
    source: list[int] = []
    target: list[int] = []
    values: list[float] = []
    halfedge_patch = np.asarray(topology.halfedge_patch_indices)
    halfedge_edge = np.asarray(topology.halfedge_local_edges)
    halfedge_orientation = np.asarray(topology.halfedge_orientation)
    for seam_index, pair in enumerate(np.asarray(topology.shared_halfedges)):
        routes: list[tuple[np.ndarray, np.ndarray]] = []
        for halfedge in pair:
            routes.append(
                _edge_control_routes(
                    int(halfedge_patch[halfedge]),
                    PatchEdge(int(halfedge_edge[halfedge])),
                    int(halfedge_orientation[halfedge]),
                )
            )
        (left_boundary, left_interior), (right_boundary, right_interior) = routes
        for control in range(_CONTROL_COUNT):
            row = seam_index * rows_per_seam + control
            source.extend((int(left_boundary[control]), int(right_boundary[control])))
            target.extend((row, row))
            values.extend((1.0, -1.0))
        if continuity is GluingContinuity.G1:
            for control in range(_CONTROL_COUNT):
                row = seam_index * rows_per_seam + _CONTROL_COUNT + control
                source.extend(
                    (
                        int(left_boundary[control]),
                        int(left_interior[control]),
                        int(right_boundary[control]),
                        int(right_interior[control]),
                    )
                )
                target.extend((row, row, row, row))
                values.extend(
                    (
                        float(_BIQUINTIC_DEGREE),
                        float(-_BIQUINTIC_DEGREE),
                        float(_BIQUINTIC_DEGREE),
                        float(-_BIQUINTIC_DEGREE),
                    )
                )
    row_count = rows_per_seam * topology.shared_edge_count
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(target, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
        row_count,
    )


def _coefficient_value(
    coefficients: ArrayLike,
    topology: PreparedHalfEdgePatchTopology,
    /,
) -> Array:
    value = jnp.asarray(coefficients)
    expected = (topology.patch_count, _CONTROL_COUNT, _CONTROL_COUNT)
    if value.ndim < 3 or tuple(value.shape[:3]) != expected:
        raise ValueError(f"Biquintic coefficients must begin with shape {expected}.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype("float64")
    return value


class BiquinticGluingConstraints(StrictModule, NonTrainableState):
    """Canonical sparse G0 or parametric-C1/G1 coefficient constraints."""

    topology: PreparedHalfEdgePatchTopology
    operator: SparseLinearMap
    continuity: GluingContinuity = eqx.field(static=True)
    status: G1PatchStatus = eqx.field(static=True)
    rows_per_seam: int = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: PreparedHalfEdgePatchTopology,
        continuity: GluingContinuity = GluingContinuity.G1,
        /,
    ):
        if not isinstance(topology, PreparedHalfEdgePatchTopology):
            raise TypeError("topology must be a PreparedHalfEdgePatchTopology.")
        if not isinstance(continuity, GluingContinuity):
            raise TypeError("continuity must be a GluingContinuity.")
        source, target, values, row_count = _assemble_constraint_routes(
            topology, continuity
        )
        relation = EdgeRelation(
            jnp.asarray(source),
            jnp.asarray(target),
            source_size=topology.patch_count * _CONTROL_COUNT_PER_PATCH,
            target_size=row_count,
        )
        coefficient_values = jnp.asarray(values)
        status = _constraint_status(topology, continuity)
        rows_per_seam = _CONTROL_COUNT * (2 if continuity is GluingContinuity.G1 else 1)
        identifier = canonical_fingerprint(
            {
                "kind": "biquintic-linear-gluing-constraints",
                "topology": topology.topology_id,
                "continuity": continuity.value,
                "source_indices": array_tree_fingerprint(source),
                "target_indices": array_tree_fingerprint(target),
                "values": array_tree_fingerprint(coefficient_values),
                "row_count": row_count,
                "status": status.value,
            }
        )
        operator = SparseLinearMap(
            relation,
            coefficient_values,
            operator_id=identifier,
        )
        self.topology = topology
        self.operator = operator
        self.continuity = continuity
        self.status = status
        self.rows_per_seam = rows_per_seam
        self.constraint_id = identifier

    @property
    def coefficient_count(self) -> int:
        return self.topology.patch_count * _CONTROL_COUNT_PER_PATCH

    @property
    def constraint_count(self) -> int:
        return self.operator.output_size

    def residual(self, coefficients: ArrayLike, /) -> Array:
        value = _coefficient_value(coefficients, self.topology)
        payload = tuple(value.shape[3:])
        flattened = value.reshape((self.coefficient_count, *payload))
        return self.operator.mv(flattened)

    def seam_evidence(
        self,
        coefficients: ArrayLike,
        parameters: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-7,
    ) -> "SeamContinuityEvidence":
        return _seam_evidence(self, coefficients, parameters, tolerance=tolerance)


class ProjectedBiquinticCoefficients(StrictModule):
    """Orthogonal coefficient projection with explicit algebraic residuals."""

    coefficients: Array
    residual_before: Array
    residual_after: Array
    correction_norm: Array
    finite: Array
    basis_id: str = eqx.field(static=True)


class BiquinticGluingBasis(StrictModule, NonTrainableState):
    """Phydrax-linalg right-nullspace basis of one supported gluing operator."""

    constraints: BiquinticGluingConstraints
    subspace: LinearSubspace
    nullspace_residual: Array
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)
    relative_rank_tolerance: float = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraints: BiquinticGluingConstraints,
        /,
        *,
        relative_rank_tolerance: float = 1.0e-7,
    ):
        if not isinstance(constraints, BiquinticGluingConstraints):
            raise TypeError("constraints must be BiquinticGluingConstraints.")
        tolerance = float(relative_rank_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("relative_rank_tolerance must be finite and non-negative.")
        if constraints.status is not G1PatchStatus.SUPPORTED:
            raise UnsupportedG1TopologyError(
                constraints.status,
                constraints.topology.unsupported_vertex_labels,
            )
        decomposition = factorize(
            constraints.operator,
            FactorizationPolicy(
                "svd",
                rank=RankPolicy(relative_cutoff=tolerance),
            ),
        )
        subspace = decomposition.right_nullspace()
        rank = int(np.asarray(decomposition.rank()))
        nullity = int(np.asarray(subspace.dimension))
        residual = constraints.operator.mv(subspace.basis)
        residual_norm = jnp.max(jnp.abs(residual))
        identifier = canonical_fingerprint(
            {
                "kind": "biquintic-gluing-nullspace-basis",
                "constraints": constraints.constraint_id,
                "relative_rank_tolerance": tolerance,
                "rank": rank,
                "nullity": nullity,
            }
        )
        self.constraints = constraints
        self.subspace = subspace
        self.nullspace_residual = residual_norm
        self.rank = rank
        self.nullity = nullity
        self.relative_rank_tolerance = tolerance
        self.basis_id = identifier

    @property
    def active_basis(self) -> Array:
        return self.subspace.basis[:, : self.nullity]

    def reconstruct(self, free_coefficients: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(free_coefficients)
        if coordinates.ndim < 1 or coordinates.shape[0] != self.nullity:
            raise ValueError(f"Free coefficients must begin with nullity {self.nullity}.")
        if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
            coordinates = coordinates.astype(self.active_basis.dtype)
        flattened = contract(
            "dn,n...->d...",
            self.active_basis.astype(coordinates.dtype),
            coordinates,
            backend="jax",
        )
        payload = tuple(coordinates.shape[1:])
        shape = (
            self.constraints.topology.patch_count,
            _CONTROL_COUNT,
            _CONTROL_COUNT,
            *payload,
        )
        return flattened.reshape(shape)

    def coordinates(self, coefficients: ArrayLike, /) -> Array:
        value = _coefficient_value(coefficients, self.constraints.topology)
        payload = tuple(value.shape[3:])
        flattened = value.reshape((self.constraints.coefficient_count, *payload))
        basis = self.active_basis.astype(flattened.dtype)
        return contract(
            "dn,d...->n...",
            jnp.conj(basis),
            flattened,
            backend="jax",
        )

    def project(self, coefficients: ArrayLike, /) -> ProjectedBiquinticCoefficients:
        value = _coefficient_value(coefficients, self.constraints.topology)
        projected = self.reconstruct(self.coordinates(value))
        residual_before = self.constraints.residual(value)
        residual_after = self.constraints.residual(projected)
        correction = projected - value
        correction_norm = jnp.sqrt(jnp.sum(jnp.abs(correction) ** 2))
        finite = (
            jnp.all(jnp.isfinite(projected))
            & jnp.all(jnp.isfinite(residual_before))
            & jnp.all(jnp.isfinite(residual_after))
        )
        return ProjectedBiquinticCoefficients(
            coefficients=projected,
            residual_before=residual_before,
            residual_after=residual_after,
            correction_norm=correction_norm,
            finite=finite,
            basis_id=self.basis_id,
        )


class SeamContinuityEvidence(StrictModule, NonTrainableState):
    """Sampled seam evidence; it is numerical evidence, not certification.

    ``preparation_status`` reports whether the selected construction path is
    supported. The ``g0_geometry_accepted`` and ``g1_geometry_accepted`` fields
    are separate physical-geometry checks over the supplied samples.
    """

    parameters: Array
    algebraic_residual: Array
    value_residual: Array
    tangent_residual: Array
    transverse_residual: Array
    maximum_value_residual: Array
    maximum_first_derivative_residual: Array
    finite: Array
    g0_geometry_accepted: Array
    g1_geometry_accepted: Array
    preparation_status: G1PatchStatus = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def path_supported(self) -> bool:
        return self.preparation_status is G1PatchStatus.SUPPORTED


def _edge_jet(
    coefficients: Array,
    edge: PatchEdge,
    orientation: int,
    parameters: Array,
    /,
) -> tuple[Array, Array, Array]:
    halfedge_parameter = parameters if orientation == 1 else 1.0 - parameters
    zero = jnp.asarray(0.0, dtype=parameters.dtype)
    one = jnp.asarray(1.0, dtype=parameters.dtype)
    if edge is PatchEdge.V_MIN:
        u, v = halfedge_parameter, zero
    elif edge is PatchEdge.U_MAX:
        u, v = one, halfedge_parameter
    elif edge is PatchEdge.V_MAX:
        u, v = 1.0 - halfedge_parameter, one
    else:
        u, v = zero, 1.0 - halfedge_parameter
    knots = jnp.asarray(_OPEN_BIQUINTIC_KNOTS, dtype=parameters.dtype)
    stencils = tuple(
        bspline_jet_stencil(
            knots,
            coordinate,
            degree=_BIQUINTIC_DEGREE,
            maximum_order=1,
        )
        for coordinate in (u, v)
    )
    plan = TensorBSplineJetPlan(stencils, maximum_order=1)
    value = plan.value(coefficients)
    gradient = plan.gradient(coefficients)
    du, dv = gradient[..., 0], gradient[..., 1]
    if edge is PatchEdge.V_MIN:
        halfedge_tangent, outward = du, -dv
    elif edge is PatchEdge.U_MAX:
        halfedge_tangent, outward = dv, du
    elif edge is PatchEdge.V_MAX:
        halfedge_tangent, outward = -du, dv
    else:
        halfedge_tangent, outward = -dv, -du
    return value, float(orientation) * halfedge_tangent, outward


def _seam_evidence(
    constraints: BiquinticGluingConstraints,
    coefficients: ArrayLike,
    parameters: ArrayLike,
    /,
    *,
    tolerance: float,
) -> SeamContinuityEvidence:
    if constraints.topology.shared_edge_count == 0:
        raise UnsupportedG1TopologyError(
            G1PatchStatus.NO_SHARED_EDGES_UNSUPPORTED,
            (),
        )
    value = _coefficient_value(coefficients, constraints.topology)
    if value.ndim != 4 or value.shape[-1] < 1:
        raise ValueError(
            "Seam geometry coefficients must have shape (patches, 6, 6, dimension)."
        )
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("Seam geometry coefficients must be real-valued.")
    sample_host = np.asarray(parameters, dtype=np.float64)
    if sample_host.ndim != 1 or sample_host.size == 0:
        raise ValueError("Seam parameters must be one non-empty rank-one array.")
    if not np.all(np.isfinite(sample_host)) or np.any(
        (sample_host < 0.0) | (sample_host > 1.0)
    ):
        raise ValueError("Seam parameters must be finite and lie in [0, 1].")
    tolerance_ = float(tolerance)
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("Seam tolerance must be finite and non-negative.")
    samples = jnp.asarray(sample_host, dtype=value.dtype)

    topology = constraints.topology
    patch_indices = np.asarray(topology.halfedge_patch_indices)
    local_edges = np.asarray(topology.halfedge_local_edges)
    orientations = np.asarray(topology.halfedge_orientation)
    value_residuals: list[Array] = []
    tangent_residuals: list[Array] = []
    transverse_residuals: list[Array] = []
    for pair in np.asarray(topology.shared_halfedges):
        jets: list[tuple[Array, Array, Array]] = []
        for halfedge in pair:
            patch_index = int(patch_indices[halfedge])
            jets.append(
                _edge_jet(
                    value[patch_index],
                    PatchEdge(int(local_edges[halfedge])),
                    int(orientations[halfedge]),
                    samples,
                )
            )
        (
            (left_value, left_tangent, left_outward),
            (
                right_value,
                right_tangent,
                right_outward,
            ),
        ) = jets
        value_residuals.append(left_value - right_value)
        tangent_residuals.append(left_tangent - right_tangent)
        transverse_residuals.append(left_outward + right_outward)
    value_residual = jnp.stack(value_residuals)
    tangent_residual = jnp.stack(tangent_residuals)
    transverse_residual = jnp.stack(transverse_residuals)
    maximum_value = jnp.max(jnp.abs(value_residual))
    maximum_first = jnp.maximum(
        jnp.max(jnp.abs(tangent_residual)),
        jnp.max(jnp.abs(transverse_residual)),
    )
    algebraic = constraints.residual(value)
    finite = (
        jnp.all(jnp.isfinite(value_residual))
        & jnp.all(jnp.isfinite(tangent_residual))
        & jnp.all(jnp.isfinite(transverse_residual))
        & jnp.all(jnp.isfinite(algebraic))
    )
    g0_accepted = finite & (maximum_value <= tolerance_)
    g1_accepted = g0_accepted & (maximum_first <= tolerance_)
    evidence_id = canonical_fingerprint(
        {
            "kind": "sampled-biquintic-seam-evidence",
            "constraint": constraints.constraint_id,
            "parameters": array_tree_fingerprint(sample_host),
            "tolerance": tolerance_,
            "claim": "sampled-numerical-evidence-not-certification",
        }
    )
    return SeamContinuityEvidence(
        parameters=samples,
        algebraic_residual=algebraic,
        value_residual=value_residual,
        tangent_residual=tangent_residual,
        transverse_residual=transverse_residual,
        maximum_value_residual=maximum_value,
        maximum_first_derivative_residual=maximum_first,
        finite=finite,
        g0_geometry_accepted=g0_accepted,
        g1_geometry_accepted=g1_accepted,
        preparation_status=constraints.status,
        tolerance=tolerance_,
        constraint_id=constraints.constraint_id,
        evidence_id=evidence_id,
    )


__all__ = [
    "BiquinticGluingBasis",
    "BiquinticGluingConstraints",
    "G1PatchStatus",
    "GluingContinuity",
    "PatchEdge",
    "PatchTopologyError",
    "PatchVertexKind",
    "PreparedHalfEdgePatchTopology",
    "ProjectedBiquinticCoefficients",
    "SeamContinuityEvidence",
    "UnsupportedG1TopologyError",
]

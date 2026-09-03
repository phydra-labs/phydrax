#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared fixed-shape cardiovascular observation sampling operators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import prod
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._interpolation import apply_gather_stencil, GatherStencil, MaskMode
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._metadata import SpatialAffine, TimeBase


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return normalized


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    result = tuple(int(size) for size in value)
    if not result or any(size <= 0 for size in result):
        raise ValueError(f"{name} must contain positive dimensions.")
    return result


def _host_float_array(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    array = np.array(
        value,
        dtype=np.result_type(original.dtype, np.float64),
        copy=True,
    )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a real numerical dtype.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


def _host_integer_array(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    if not np.issubdtype(original.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    array = np.array(original, dtype=np.int32, copy=True)
    array.setflags(write=False)
    return array


class ObservationSamplingEvidence(StrictModule):
    """Coverage and fail-closed numerical evidence from one operator action."""

    support: Array
    covered_count: Array
    query_count: Array
    coverage_fraction: Array
    complete_coverage: Array
    finite: Array
    successful: Array


class ObservationCandidate(StrictModule):
    """Fixed-shape sampled values and their observation evidence."""

    values: Array
    evidence: ObservationSamplingEvidence
    prepared_id: str = eqx.field(static=True)


class ObservationJVPResult(StrictModule):
    """Primal observation candidate and exact linear tangent action."""

    primal: ObservationCandidate
    tangent: Array


class ObservationSamplingPlan(StrictModule, NonTrainableState):
    """Explicit fixed-width sparse sampling plan.

    The final axis of ``indices`` and ``weights`` is route width.  All preceding
    axes are observation query axes.  Source payload axes follow ``source_shape``
    at evaluation time and are preserved by forward, transpose, and JVP actions.
    """

    indices: Array
    weights: Array
    valid: Array
    support: Array
    source_shape: tuple[int, ...] = eqx.field(static=True)
    operator_kind: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    require_complete_coverage: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        weights: ArrayLike,
        source_shape: Sequence[int],
        /,
        *,
        operator_kind: str,
        source_geometry_id: str,
        valid: ArrayLike | None = None,
        support: ArrayLike | None = None,
        require_complete_coverage: bool = False,
    ):
        shape = _shape(source_shape, "source_shape")
        indices_ = jax.lax.stop_gradient(jnp.asarray(indices))
        if not jnp.issubdtype(indices_.dtype, jnp.integer):
            raise TypeError("indices must have an integer dtype.")
        indices_ = indices_.astype(jnp.int32)
        weights_ = jax.lax.stop_gradient(jnp.asarray(weights))
        if not jnp.issubdtype(weights_.dtype, jnp.inexact):
            weights_ = weights_.astype(float)
        if indices_.ndim < 1 or indices_.shape != weights_.shape:
            raise ValueError(
                "indices and weights must have one identical route-width shape."
            )
        if int(indices_.shape[-1]) < 1:
            raise ValueError("Observation route width must be positive.")
        valid_ = (
            jnp.ones(indices_.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != indices_.shape:
            raise ValueError("valid must match indices shape.")
        support_ = (
            jnp.any(valid_, axis=-1)
            if support is None
            else jnp.asarray(support, dtype=bool)
        )
        if support_.shape != indices_.shape[:-1]:
            raise ValueError("support must match the observation query shape.")
        source_size = prod(shape)
        indices_ = eqx.error_if(
            indices_,
            jnp.any(valid_ & ((indices_ < 0) | (indices_ >= source_size))),
            "A valid observation route index lies outside the source shape.",
        )
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)),
            "Observation route weights must be finite.",
        )
        if not isinstance(require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        kind = _identifier(operator_kind, "operator_kind")
        geometry_id = _identifier(source_geometry_id, "source_geometry_id")
        self.indices = indices_
        self.weights = weights_
        self.valid = valid_
        self.support = support_
        self.source_shape = shape
        self.operator_kind = kind
        self.source_geometry_id = geometry_id
        self.require_complete_coverage = require_complete_coverage
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-observation-sampling-plan",
                "operator_kind": kind,
                "source_geometry_id": geometry_id,
                "source_shape": list(shape),
                "require_complete_coverage": require_complete_coverage,
                "routes": array_tree_fingerprint(
                    {
                        "indices": indices_,
                        "weights": weights_,
                        "valid": valid_,
                        "support": support_,
                    }
                ),
            }
        )

    def prepare(self) -> "PreparedObservationOperator":
        stencil = GatherStencil(
            indices=self.indices,
            weights=self.weights,
            source_size=prod(self.source_shape),
            valid=self.valid,
            support=self.support,
        )
        return PreparedObservationOperator(
            stencil,
            self.source_shape,
            self.operator_kind,
            self.source_geometry_id,
            self.require_complete_coverage,
            self.plan_id,
        )


class PreparedObservationOperator(StrictModule, NonTrainableState):
    """Prepared sparse gather with exact transpose and JVP actions."""

    stencil: GatherStencil
    source_shape: tuple[int, ...] = eqx.field(static=True)
    operator_kind: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    require_complete_coverage: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        stencil: GatherStencil,
        source_shape: tuple[int, ...],
        operator_kind: str,
        source_geometry_id: str,
        require_complete_coverage: bool,
        plan_id: str,
        /,
    ):
        if not isinstance(stencil, GatherStencil):
            raise TypeError("stencil must be a GatherStencil.")
        shape = _shape(source_shape, "source_shape")
        if stencil.source_size != prod(shape):
            raise ValueError("Stencil source size does not match source_shape.")
        self.stencil = stencil
        self.source_shape = shape
        self.operator_kind = _identifier(operator_kind, "operator_kind")
        self.source_geometry_id = _identifier(source_geometry_id, "source_geometry_id")
        self.require_complete_coverage = bool(require_complete_coverage)
        self.plan_id = _identifier(plan_id, "plan_id")
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-observation-operator",
                "plan_id": self.plan_id,
                "source_shape": list(shape),
            }
        )

    @property
    def query_shape(self) -> tuple[int, ...]:
        return self.stencil.relation.output_shape

    @property
    def source_size(self) -> int:
        return prod(self.source_shape)

    def _flatten_source(self, values: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(values)
        if (
            array.ndim < len(self.source_shape)
            or tuple(array.shape[: len(self.source_shape)]) != self.source_shape
        ):
            raise ValueError(
                f"{name} must begin with source shape {self.source_shape}; got {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            array = array.astype(float)
        payload_shape = tuple(array.shape[len(self.source_shape) :])
        return array.reshape((self.source_size,) + payload_shape)

    def _flatten_mask(self, source_mask: ArrayLike | None, /) -> Array | None:
        if source_mask is None:
            return None
        mask = jnp.asarray(source_mask, dtype=bool)
        if mask.shape != self.source_shape:
            raise ValueError(
                f"source_mask must have shape {self.source_shape}; got {mask.shape}."
            )
        return mask.reshape((self.source_size,))

    def _values(
        self,
        values: ArrayLike,
        source_mask: ArrayLike | None,
        mask_mode: MaskMode,
        /,
    ) -> tuple[Array, Array]:
        flattened = self._flatten_source(values, "values")
        mask = self._flatten_mask(source_mask)
        result = apply_gather_stencil(
            flattened,
            self.stencil,
            source_mask=mask,
            mask_mode=mask_mode,
        )
        return result.values, result.support

    def apply(
        self,
        values: ArrayLike,
        /,
        *,
        source_mask: ArrayLike | None = None,
        mask_mode: MaskMode = "strict",
    ) -> ObservationCandidate:
        flattened = self._flatten_source(values, "values")
        mask = self._flatten_mask(source_mask)
        interpolation = apply_gather_stencil(
            flattened,
            self.stencil,
            source_mask=mask,
            mask_mode=mask_mode,
        )
        if mask is None:
            finite_input = jnp.all(jnp.isfinite(flattened))
        else:
            expanded = mask.reshape(mask.shape + (1,) * (flattened.ndim - 1))
            finite_input = jnp.all(jnp.where(expanded, jnp.isfinite(flattened), True))
        finite = finite_input & jnp.all(jnp.isfinite(interpolation.values))
        covered_count = jnp.sum(interpolation.support, dtype=jnp.int32)
        query_count = jnp.asarray(interpolation.support.size, dtype=jnp.int32)
        coverage = covered_count.astype(interpolation.values.real.dtype) / jnp.maximum(
            query_count, 1
        ).astype(interpolation.values.real.dtype)
        complete = jnp.all(interpolation.support)
        required_coverage = (
            complete if self.require_complete_coverage else covered_count > 0
        )
        successful = finite & required_coverage
        evidence = ObservationSamplingEvidence(
            interpolation.support,
            covered_count,
            query_count,
            coverage,
            complete,
            finite,
            successful,
        )
        return ObservationCandidate(interpolation.values, evidence, self.prepared_id)

    def transpose(
        self,
        cotangent: ArrayLike,
        /,
        *,
        source_mask: ArrayLike | None = None,
        mask_mode: MaskMode = "strict",
    ) -> Array:
        messages = jnp.asarray(cotangent)
        if not jnp.issubdtype(messages.dtype, jnp.inexact):
            messages = messages.astype(float)
        query_shape = self.query_shape
        if (
            messages.ndim < len(query_shape)
            or tuple(messages.shape[: len(query_shape)]) != query_shape
        ):
            raise ValueError(
                f"cotangent must begin with query shape {query_shape}; got {messages.shape}."
            )
        payload_shape = tuple(messages.shape[len(query_shape) :])
        zero = jnp.zeros(self.source_shape + payload_shape, dtype=messages.dtype)
        action = lambda source: self._values(source, source_mask, mask_mode)[0]
        return jax.linear_transpose(action, zero)(messages)[0]

    def jvp(
        self,
        values: ArrayLike,
        tangent: ArrayLike,
        /,
        *,
        source_mask: ArrayLike | None = None,
        mask_mode: MaskMode = "strict",
    ) -> ObservationJVPResult:
        primal_values = jnp.asarray(values)
        tangent_values = jnp.asarray(tangent)
        if tangent_values.shape != primal_values.shape:
            raise ValueError("tangent must have the same shape as values.")
        dtype = jnp.result_type(primal_values.dtype, tangent_values.dtype, jnp.float32)
        primal_values = primal_values.astype(dtype)
        tangent_values = tangent_values.astype(dtype)
        action = lambda source: self._values(source, source_mask, mask_mode)[0]
        _, tangent_output = jax.jvp(action, (primal_values,), (tangent_values,))
        return ObservationJVPResult(
            self.apply(primal_values, source_mask=source_mask, mask_mode=mask_mode),
            tangent_output,
        )


@dataclass(frozen=True, slots=True)
class VoxelObservationPlan:
    """Host preparation of trilinear voxel samples at patient-space points."""

    image_shape: tuple[int, int, int]
    spatial_affine: SpatialAffine
    world_points_mm: np.ndarray
    require_complete_coverage: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        shape = _shape(self.image_shape, "image_shape")
        if len(shape) != 3:
            raise ValueError("image_shape must contain exactly three spatial dimensions.")
        if not isinstance(self.spatial_affine, SpatialAffine):
            raise TypeError("spatial_affine must be a SpatialAffine.")
        points = _host_float_array(self.world_points_mm, "world_points_mm")
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError(
                "world_points_mm must end with a coordinate axis of length three."
            )
        if not isinstance(self.require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "world_points_mm", points)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-voxel-observation-plan",
                    "image_shape": list(shape),
                    "spatial_affine_id": self.spatial_affine.affine_id,
                    "points": array_tree_fingerprint(points),
                    "require_complete_coverage": self.require_complete_coverage,
                }
            ),
        )

    def prepare(self) -> PreparedObservationOperator:
        coordinates = self.spatial_affine.world_to_index(self.world_points_mm)
        query_shape = coordinates.shape[:-1]
        flat = coordinates.reshape((-1, 3))
        shape = np.asarray(self.image_shape, dtype=np.int64)
        tolerance = (
            64.0 * np.finfo(coordinates.dtype).eps * max(1.0, float(np.max(shape)))
        )
        support_flat = np.all(
            (flat >= -tolerance) & (flat <= shape[None, :] - 1.0 + tolerance), axis=-1
        )
        clipped = np.clip(flat, 0.0, shape[None, :] - 1.0)
        lower = np.floor(clipped).astype(np.int64)
        fraction = clipped - lower
        corners = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.int64,
        )
        corner_indices = np.minimum(
            lower[:, None, :] + corners[None, :, :], shape[None, None, :] - 1
        )
        factors = np.where(
            corners[None, :, :] == 1, fraction[:, None, :], 1.0 - fraction[:, None, :]
        )
        weights = np.prod(factors, axis=-1)
        indices = np.ravel_multi_index(
            tuple(corner_indices[..., axis].reshape(-1) for axis in range(3)),
            self.image_shape,
        ).reshape((flat.shape[0], 8))
        support = support_flat.reshape(query_shape)
        valid = np.broadcast_to(support[..., None], query_shape + (8,))
        return ObservationSamplingPlan(
            indices.reshape(query_shape + (8,)),
            weights.reshape(query_shape + (8,)),
            self.image_shape,
            operator_kind="voxel-trilinear",
            source_geometry_id=self.spatial_affine.affine_id,
            valid=valid,
            support=support,
            require_complete_coverage=self.require_complete_coverage,
        ).prepare()


def _tetrahedron_weights(vertices: np.ndarray, point: np.ndarray, /) -> np.ndarray | None:
    matrix = np.concatenate((vertices.T, np.ones((1, 4), dtype=vertices.dtype)), axis=0)
    if abs(float(np.linalg.det(matrix))) <= 64.0 * np.finfo(vertices.dtype).eps:
        return None
    return np.linalg.solve(
        matrix, np.concatenate((point, np.ones((1,), dtype=point.dtype)))
    )


@dataclass(frozen=True, slots=True)
class P1ObservationPlan:
    """Host-prepared tetrahedral P1 interpolation at fixed world points."""

    node_coordinates_mm: np.ndarray
    tetrahedra: np.ndarray
    query_points_mm: np.ndarray
    mesh_id: str
    cell_indices: np.ndarray | None = None
    containment_tolerance: float = 1.0e-9
    require_complete_coverage: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        nodes = _host_float_array(self.node_coordinates_mm, "node_coordinates_mm")
        cells = _host_integer_array(self.tetrahedra, "tetrahedra")
        points = _host_float_array(self.query_points_mm, "query_points_mm")
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("node_coordinates_mm must have shape (num_nodes, 3).")
        if cells.ndim != 2 or cells.shape[1] != 4 or cells.shape[0] == 0:
            raise ValueError("tetrahedra must have shape (num_cells, 4).")
        if np.any((cells < 0) | (cells >= nodes.shape[0])):
            raise ValueError("tetrahedra contains an out-of-bounds node index.")
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError(
                "query_points_mm must end with a coordinate axis of length three."
            )
        cell_indices = None
        if self.cell_indices is not None:
            cell_indices = _host_integer_array(self.cell_indices, "cell_indices")
            if cell_indices.shape != points.shape[:-1]:
                raise ValueError("cell_indices must match the query point shape.")
            if np.any((cell_indices < -1) | (cell_indices >= cells.shape[0])):
                raise ValueError("cell_indices must be -1 or a valid tetrahedron index.")
        tolerance = float(self.containment_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("containment_tolerance must be finite and non-negative.")
        if not isinstance(self.require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        object.__setattr__(self, "node_coordinates_mm", nodes)
        object.__setattr__(self, "tetrahedra", cells)
        object.__setattr__(self, "query_points_mm", points)
        object.__setattr__(self, "mesh_id", _identifier(self.mesh_id, "mesh_id"))
        object.__setattr__(self, "cell_indices", cell_indices)
        object.__setattr__(self, "containment_tolerance", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-p1-observation-plan",
                    "mesh_id": self.mesh_id,
                    "nodes": array_tree_fingerprint(nodes),
                    "tetrahedra": array_tree_fingerprint(cells),
                    "queries": array_tree_fingerprint(points),
                    "cell_indices": None
                    if cell_indices is None
                    else array_tree_fingerprint(cell_indices),
                    "containment_tolerance": tolerance,
                    "require_complete_coverage": self.require_complete_coverage,
                }
            ),
        )

    def prepare(self) -> PreparedObservationOperator:
        points = self.query_points_mm.reshape((-1, 3))
        supplied = None if self.cell_indices is None else self.cell_indices.reshape((-1,))
        indices = np.zeros((points.shape[0], 4), dtype=np.int32)
        weights = np.zeros((points.shape[0], 4), dtype=float)
        support = np.zeros((points.shape[0],), dtype=bool)
        for query_index, point in enumerate(points):
            candidate_cells = (
                range(self.tetrahedra.shape[0])
                if supplied is None
                else (int(supplied[query_index]),)
            )
            for cell_index in candidate_cells:
                if cell_index < 0:
                    continue
                cell = self.tetrahedra[cell_index]
                barycentric = _tetrahedron_weights(self.node_coordinates_mm[cell], point)
                if barycentric is None:
                    continue
                inside = np.all(barycentric >= -self.containment_tolerance) and np.all(
                    barycentric <= 1.0 + self.containment_tolerance
                )
                if inside:
                    indices[query_index] = cell
                    weights[query_index] = barycentric
                    support[query_index] = True
                    break
        query_shape = self.query_points_mm.shape[:-1]
        return ObservationSamplingPlan(
            indices.reshape(query_shape + (4,)),
            weights.reshape(query_shape + (4,)),
            (self.node_coordinates_mm.shape[0],),
            operator_kind="tetrahedral-p1",
            source_geometry_id=self.mesh_id,
            valid=np.broadcast_to(
                support.reshape(query_shape)[..., None], query_shape + (4,)
            ),
            support=support.reshape(query_shape),
            require_complete_coverage=self.require_complete_coverage,
        ).prepare()


@dataclass(frozen=True, slots=True)
class ElectrodeObservationPlan:
    """Fixed electrode or lead map over a declared source potential space."""

    source_indices: np.ndarray
    weights: np.ndarray
    source_size: int
    electrode_ids: tuple[str, ...]
    source_geometry_id: str
    valid: np.ndarray | None = None
    require_complete_coverage: bool = True
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        indices = _host_integer_array(self.source_indices, "source_indices")
        weights = _host_float_array(self.weights, "weights")
        source_size = int(self.source_size)
        electrode_ids = tuple(
            _identifier(value, "electrode_id") for value in self.electrode_ids
        )
        if indices.ndim != 2 or indices.shape != weights.shape:
            raise ValueError(
                "source_indices and weights must have shape (num_electrodes, route_width)."
            )
        if source_size <= 0 or len(electrode_ids) != indices.shape[0]:
            raise ValueError(
                "source_size and electrode_ids must match the electrode map."
            )
        if len(set(electrode_ids)) != len(electrode_ids):
            raise ValueError("electrode_ids must be unique.")
        if not isinstance(self.require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        valid = (
            np.ones(indices.shape, dtype=bool)
            if self.valid is None
            else np.array(self.valid, dtype=bool, copy=True)
        )
        if valid.shape != indices.shape:
            raise ValueError("valid must match source_indices shape.")
        if np.any(valid & ((indices < 0) | (indices >= source_size))):
            raise ValueError("A valid electrode route index lies outside source_size.")
        valid.setflags(write=False)
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "electrode_ids", electrode_ids)
        object.__setattr__(
            self,
            "source_geometry_id",
            _identifier(self.source_geometry_id, "source_geometry_id"),
        )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-electrode-observation-plan",
                    "source_geometry_id": self.source_geometry_id,
                    "source_size": source_size,
                    "electrode_ids": list(electrode_ids),
                    "require_complete_coverage": self.require_complete_coverage,
                    "routes": array_tree_fingerprint(
                        {"indices": indices, "weights": weights, "valid": valid}
                    ),
                }
            ),
        )

    def prepare(self) -> PreparedObservationOperator:
        support = np.any(self.valid, axis=-1)
        return ObservationSamplingPlan(
            self.source_indices,
            self.weights,
            (self.source_size,),
            operator_kind="electrode-linear-map",
            source_geometry_id=self.source_geometry_id,
            valid=self.valid,
            support=support,
            require_complete_coverage=self.require_complete_coverage,
        ).prepare()


def _triangle_weights(
    vertices: np.ndarray, point: np.ndarray, tolerance: float, /
) -> np.ndarray | None:
    first = vertices[1] - vertices[0]
    second = vertices[2] - vertices[0]
    delta = point - vertices[0]
    gram = np.asarray(
        [
            [np.dot(first, first), np.dot(first, second)],
            [np.dot(first, second), np.dot(second, second)],
        ]
    )
    determinant = float(gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0])
    if abs(determinant) <= 64.0 * np.finfo(vertices.dtype).eps:
        return None
    coordinates = np.linalg.solve(
        gram, np.asarray([np.dot(first, delta), np.dot(second, delta)])
    )
    projection = vertices[0] + coordinates[0] * first + coordinates[1] * second
    if float(np.linalg.norm(point - projection)) > tolerance:
        return None
    return np.asarray([1.0 - coordinates.sum(), coordinates[0], coordinates[1]])


@dataclass(frozen=True, slots=True)
class SurfaceObservationPlan:
    """Host-prepared triangular-surface P1 interpolation."""

    node_coordinates_mm: np.ndarray
    triangles: np.ndarray
    query_points_mm: np.ndarray
    surface_id: str
    face_indices: np.ndarray | None = None
    containment_tolerance_mm: float = 1.0e-6
    require_complete_coverage: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        nodes = _host_float_array(self.node_coordinates_mm, "node_coordinates_mm")
        faces = _host_integer_array(self.triangles, "triangles")
        points = _host_float_array(self.query_points_mm, "query_points_mm")
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("node_coordinates_mm must have shape (num_nodes, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            raise ValueError("triangles must have shape (num_faces, 3).")
        if np.any((faces < 0) | (faces >= nodes.shape[0])):
            raise ValueError("triangles contains an out-of-bounds node index.")
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError(
                "query_points_mm must end with a coordinate axis of length three."
            )
        face_indices = None
        if self.face_indices is not None:
            face_indices = _host_integer_array(self.face_indices, "face_indices")
            if face_indices.shape != points.shape[:-1]:
                raise ValueError("face_indices must match the query point shape.")
            if np.any((face_indices < -1) | (face_indices >= faces.shape[0])):
                raise ValueError("face_indices must be -1 or a valid triangle index.")
        tolerance = float(self.containment_tolerance_mm)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("containment_tolerance_mm must be finite and non-negative.")
        if not isinstance(self.require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        object.__setattr__(self, "node_coordinates_mm", nodes)
        object.__setattr__(self, "triangles", faces)
        object.__setattr__(self, "query_points_mm", points)
        object.__setattr__(self, "surface_id", _identifier(self.surface_id, "surface_id"))
        object.__setattr__(self, "face_indices", face_indices)
        object.__setattr__(self, "containment_tolerance_mm", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-surface-observation-plan",
                    "surface_id": self.surface_id,
                    "nodes": array_tree_fingerprint(nodes),
                    "triangles": array_tree_fingerprint(faces),
                    "queries": array_tree_fingerprint(points),
                    "face_indices": None
                    if face_indices is None
                    else array_tree_fingerprint(face_indices),
                    "tolerance_mm": tolerance,
                    "require_complete_coverage": self.require_complete_coverage,
                }
            ),
        )

    def prepare(self) -> PreparedObservationOperator:
        points = self.query_points_mm.reshape((-1, 3))
        supplied = None if self.face_indices is None else self.face_indices.reshape((-1,))
        indices = np.zeros((points.shape[0], 3), dtype=np.int32)
        weights = np.zeros((points.shape[0], 3), dtype=float)
        support = np.zeros((points.shape[0],), dtype=bool)
        for query_index, point in enumerate(points):
            candidates = (
                range(self.triangles.shape[0])
                if supplied is None
                else (int(supplied[query_index]),)
            )
            for face_index in candidates:
                if face_index < 0:
                    continue
                face = self.triangles[face_index]
                barycentric = _triangle_weights(
                    self.node_coordinates_mm[face], point, self.containment_tolerance_mm
                )
                if barycentric is None:
                    continue
                barycentric_tolerance = 64.0 * np.finfo(barycentric.dtype).eps
                if np.all(barycentric >= -barycentric_tolerance) and np.all(
                    barycentric <= 1.0 + barycentric_tolerance
                ):
                    indices[query_index] = face
                    weights[query_index] = barycentric
                    support[query_index] = True
                    break
        query_shape = self.query_points_mm.shape[:-1]
        return ObservationSamplingPlan(
            indices.reshape(query_shape + (3,)),
            weights.reshape(query_shape + (3,)),
            (self.node_coordinates_mm.shape[0],),
            operator_kind="surface-p1",
            source_geometry_id=self.surface_id,
            valid=np.broadcast_to(
                support.reshape(query_shape)[..., None], query_shape + (3,)
            ),
            support=support.reshape(query_shape),
            require_complete_coverage=self.require_complete_coverage,
        ).prepare()


@dataclass(frozen=True, slots=True)
class TimeObservationPlan:
    """Host preparation of piecewise-linear samples on an explicit timebase."""

    source_timebase: TimeBase
    query_times_ms: np.ndarray
    require_complete_coverage: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_timebase, TimeBase):
            raise TypeError("source_timebase must be a TimeBase.")
        times = _host_float_array(self.query_times_ms, "query_times_ms")
        if times.size == 0:
            raise ValueError("query_times_ms must be non-empty.")
        if not isinstance(self.require_complete_coverage, bool):
            raise TypeError("require_complete_coverage must be boolean.")
        object.__setattr__(self, "query_times_ms", times)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-time-observation-plan",
                    "source_timebase_id": self.source_timebase.timebase_id,
                    "query_times_ms": array_tree_fingerprint(times),
                    "require_complete_coverage": self.require_complete_coverage,
                }
            ),
        )

    def prepare(self) -> PreparedObservationOperator:
        source = self.source_timebase.sample_times_ms
        query = self.query_times_ms.reshape((-1,))
        tolerance = (
            64.0 * np.finfo(source.dtype).eps * max(1.0, float(np.max(np.abs(source))))
        )
        support = (query >= source[0] - tolerance) & (query <= source[-1] + tolerance)
        if source.size == 1:
            indices = np.zeros((query.size, 1), dtype=np.int32)
            weights = np.ones((query.size, 1), dtype=float)
            support &= np.abs(query - source[0]) <= tolerance
        else:
            upper = np.searchsorted(source, query, side="right")
            lower = np.clip(upper - 1, 0, source.size - 2)
            upper = lower + 1
            fraction = (np.clip(query, source[0], source[-1]) - source[lower]) / (
                source[upper] - source[lower]
            )
            indices = np.stack((lower, upper), axis=-1).astype(np.int32)
            weights = np.stack((1.0 - fraction, fraction), axis=-1)
        query_shape = self.query_times_ms.shape
        width = indices.shape[-1]
        support = support.reshape(query_shape)
        return ObservationSamplingPlan(
            indices.reshape(query_shape + (width,)),
            weights.reshape(query_shape + (width,)),
            (self.source_timebase.sample_count,),
            operator_kind="time-linear",
            source_geometry_id=self.source_timebase.timebase_id,
            valid=np.broadcast_to(support[..., None], query_shape + (width,)),
            support=support,
            require_complete_coverage=self.require_complete_coverage,
        ).prepare()


__all__ = [
    "ElectrodeObservationPlan",
    "ObservationCandidate",
    "ObservationJVPResult",
    "ObservationSamplingEvidence",
    "ObservationSamplingPlan",
    "P1ObservationPlan",
    "PreparedObservationOperator",
    "SurfaceObservationPlan",
    "TimeObservationPlan",
    "VoxelObservationPlan",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._spaces import DiscreteFieldSpace


if TYPE_CHECKING:
    from ._unstructured import UnstructuredFiniteVolumeDiscretization

_INT32_INFO = np.iinfo(np.int32)


def _canonical_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical stripped string.")
    return value


def _normalized_int32_route(
    values: np.ndarray,
    name: str,
    *,
    minimum: int,
) -> np.ndarray:
    smallest = int(np.min(values))
    largest = int(np.max(values))
    if smallest < minimum or largest > _INT32_INFO.max:
        raise ValueError(f"{name} entries must be representable as int32.")
    return values.astype(np.int32, copy=False)


def _validated_boundary_policy_count(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("boundary_policy_count must be a nonnegative integer.")
    return int(value)


def _validate_boundary_policy_routes(
    neighbours: np.ndarray,
    active: np.ndarray,
    policies: np.ndarray,
    boundary_policy_count: int,
) -> None:
    active_boundary = active & (neighbours < 0)
    active_interior = active & (neighbours >= 0)
    if np.any(active_boundary & ((policies < 0) | (policies >= boundary_policy_count))):
        raise ValueError(
            "Active boundary face routes require an in-range nonnegative "
            "boundary policy ID."
        )
    if np.any(active_interior & (policies != -1)):
        raise ValueError("Active interior face routes must use boundary policy ID -1.")
    if np.any((~active) & (policies != -1)):
        raise ValueError("Inactive face routes must use boundary policy ID -1.")


def _validated_static_shape(
    value: tuple[int, ...],
    name: str,
    *,
    rank: int,
) -> tuple[int, ...]:
    try:
        shape = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a rank-{rank} shape.") from error
    if len(shape) != rank or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or int(item) <= 0
        for item in shape
    ):
        raise ValueError(f"{name} must be a rank-{rank} shape with positive extents.")
    return tuple(int(item) for item in shape)


def _dynamic_int32_scalar(
    value: ArrayLike,
    name: str,
    *,
    minimum: int,
) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or scalar.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a scalar integer.")
    scalar = eqx.error_if(
        scalar,
        (scalar < minimum) | (scalar > _INT32_INFO.max),
        f"{name} must be at least {minimum} and representable as int32.",
    )
    return scalar.astype(jnp.int32)


def _safe_defect_tolerance_ratio(defect: Array, tolerance: Array, /) -> Array:
    """Return a nonnegative defect ratio without producing a zero-denominator NaN."""

    dtype = jnp.result_type(defect, tolerance, jnp.asarray(1.0))
    defect_ = defect.astype(dtype)
    tolerance_ = tolerance.astype(dtype)
    positive_tolerance = tolerance_ > 0.0
    safe_tolerance = jnp.where(
        positive_tolerance,
        tolerance_,
        jnp.ones((), dtype=dtype),
    )
    finite_ratio = defect_ / safe_tolerance
    zero = jnp.zeros((), dtype=dtype)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    return jnp.where(
        positive_tolerance,
        finite_ratio,
        jnp.where(defect_ == 0.0, zero, infinity),
    )


class FiniteVolumeFaceBlock(StrictModule, NonTrainableState):
    """One owner-oriented homogeneous explicit face batch."""

    face_ids: Array
    owner_cells: Array
    neighbour_cells: Array
    boundary_patch_ids: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    active_mask: Array
    block_id: str = eqx.field(static=True)


class FiniteVolumeStageFaceLayout(StrictModule, NonTrainableState):
    """Host-validated immutable routing and shape contract for one face batch."""

    face_ids: Array
    owner_cells: Array
    neighbour_cells: Array
    active_mask: Array
    boundary_policy_ids: Array
    boundary_policy_count: int = eqx.field(static=True)
    spatial_shape: tuple[int, int] = eqx.field(static=True)
    quadrature_shape: tuple[int, int] = eqx.field(static=True)
    block_id: str = eqx.field(static=True)
    block_kind: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_ids: ArrayLike,
        owner_cells: ArrayLike,
        neighbour_cells: ArrayLike,
        active_mask: ArrayLike,
        boundary_policy_ids: ArrayLike | None = None,
        boundary_policy_count: int,
        spatial_shape: tuple[int, int],
        quadrature_shape: tuple[int, int],
        block_id: str,
        block_kind: str = "physical",
    ):
        identifier = _canonical_identifier(block_id, "block_id")
        policy_count = _validated_boundary_policy_count(boundary_policy_count)
        ids = np.asarray(face_ids)
        owners = np.asarray(owner_cells)
        neighbours = np.asarray(neighbour_cells)
        policies = (
            np.full(ids.shape, -1, dtype=np.int32)
            if boundary_policy_ids is None
            else np.asarray(boundary_policy_ids)
        )
        active = np.asarray(active_mask)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("face_ids must be a non-empty rank-1 array.")
        face_count = int(ids.size)
        route_shape = (face_count,)
        if (
            owners.shape != route_shape
            or neighbours.shape != route_shape
            or active.shape != route_shape
            or policies.shape != route_shape
        ):
            raise ValueError(
                "Owner, neighbour, boundary-policy, and active routes must have "
                "one entry per face."
            )
        if (
            ids.dtype.kind not in "iu"
            or owners.dtype.kind not in "iu"
            or neighbours.dtype.kind not in "iu"
            or policies.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Face IDs, cell routes, and boundary policies must be integer arrays."
            )
        if active.dtype.kind != "b":
            raise ValueError("active_mask must be a boolean array.")
        ids = _normalized_int32_route(ids, "face_ids", minimum=0)
        owners = _normalized_int32_route(owners, "owner_cells", minimum=0)
        neighbours = _normalized_int32_route(
            neighbours,
            "neighbour_cells",
            minimum=-1,
        )
        policies = np.where(active, policies, -1)
        policies = _normalized_int32_route(
            policies,
            "boundary_policy_ids",
            minimum=-1,
        )
        if (
            np.unique(ids).size != face_count
            or np.any(owners < 0)
            or np.any(neighbours < -1)
            or np.any(policies < -1)
        ):
            raise ValueError(
                "Face IDs must be unique and nonnegative; owner/neighbour routes "
                "must use nonnegative cells or -1 for a boundary neighbour; "
                "boundary policy IDs must be at least -1."
            )
        _validate_boundary_policy_routes(
            neighbours,
            active,
            policies,
            policy_count,
        )
        spatial = _validated_static_shape(
            spatial_shape,
            "spatial_shape",
            rank=2,
        )
        quadrature = _validated_static_shape(
            quadrature_shape,
            "quadrature_shape",
            rank=2,
        )
        if spatial[0] != face_count or quadrature[0] != face_count:
            raise ValueError(
                "Spatial and quadrature shapes must have one leading entry per face."
            )

        kind = _canonical_identifier(block_kind, "block_kind")
        if kind not in ("physical", "cut"):
            raise ValueError("block_kind must be 'physical' or 'cut'.")
        self.face_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.owner_cells = jnp.asarray(owners, dtype=jnp.int32)
        self.neighbour_cells = jnp.asarray(neighbours, dtype=jnp.int32)
        self.active_mask = jnp.asarray(active, dtype=bool)
        self.boundary_policy_ids = jnp.asarray(policies, dtype=jnp.int32)
        self.boundary_policy_count = policy_count
        self.spatial_shape = (spatial[0], spatial[1])
        self.quadrature_shape = (quadrature[0], quadrature[1])
        self.block_id = identifier
        self.block_kind = kind

    def validate_boundary_policy_count(
        self,
        boundary_policy_count: int,
        /,
    ) -> None:
        """Revalidate routes against the exact policy set bound by dynamics."""

        expected = _validated_boundary_policy_count(boundary_policy_count)
        if self.boundary_policy_count != expected:
            raise ValueError(
                "Stage face layout boundary-policy count does not match "
                "the bound boundary set."
            )
        _validate_boundary_policy_routes(
            np.asarray(self.neighbour_cells),
            np.asarray(self.active_mask),
            np.asarray(self.boundary_policy_ids),
            expected,
        )

    @property
    def face_count(self) -> int:
        return int(self.face_ids.size)

    @property
    def spatial_dimension(self) -> int:
        return self.spatial_shape[1]

    @property
    def quadrature_count(self) -> int:
        return self.quadrature_shape[1]


class FiniteVolumeStageFaceBlock(StrictModule, NonTrainableState):
    """Dynamic face geometry bound to one immutable stage-face layout."""

    layout: FiniteVolumeStageFaceLayout
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    quadrature_points: Array
    quadrature_weights: Array
    quadrature_grid_normal_velocity: Array

    def __init__(
        self,
        *,
        layout: FiniteVolumeStageFaceLayout,
        face_centers: ArrayLike,
        area_vectors: ArrayLike,
        face_measures: ArrayLike,
        quadrature_points: ArrayLike,
        quadrature_weights: ArrayLike,
        quadrature_grid_normal_velocity: ArrayLike,
    ):
        if not isinstance(layout, FiniteVolumeStageFaceLayout):
            raise TypeError("layout must be FiniteVolumeStageFaceLayout.")

        centers = jnp.asarray(face_centers)
        vectors = jnp.asarray(area_vectors)
        measures = jnp.asarray(face_measures)
        points = jnp.asarray(quadrature_points)
        weights = jnp.asarray(quadrature_weights)
        grid_velocity = jnp.asarray(quadrature_grid_normal_velocity)
        if centers.shape != layout.spatial_shape:
            raise ValueError("face_centers must match the prepared layout spatial_shape.")
        if vectors.shape != layout.spatial_shape:
            raise ValueError("area_vectors must have the same shape as face_centers.")
        face_shape = (layout.face_count,)
        if measures.shape != face_shape:
            raise ValueError("face_measures must have shape (face_count,).")
        point_shape = (*layout.quadrature_shape, layout.spatial_dimension)
        if points.shape != point_shape:
            raise ValueError(
                "quadrature_points must match the prepared layout quadrature and "
                "spatial shapes."
            )
        if (
            weights.shape != layout.quadrature_shape
            or grid_velocity.shape != layout.quadrature_shape
        ):
            raise ValueError(
                "Quadrature weights and grid-normal velocity must match the "
                "prepared layout quadrature_shape."
            )

        centers = eqx.error_if(
            centers,
            jnp.any(~jnp.isfinite(centers))
            | jnp.any(~jnp.isfinite(vectors))
            | jnp.any(~jnp.isfinite(measures))
            | jnp.any(~jnp.isfinite(points))
            | jnp.any(~jnp.isfinite(weights))
            | jnp.any(~jnp.isfinite(grid_velocity)),
            "Stage face geometry must be finite.",
        )
        active = jnp.asarray(layout.active_mask, dtype=bool)
        measures = eqx.error_if(
            measures,
            jnp.any(measures < 0.0),
            "face_measures must be nonnegative.",
        )
        measures = eqx.error_if(
            measures,
            jnp.any(active & (measures <= 0.0)),
            "Active face measures must be strictly positive.",
        )
        measures = eqx.error_if(
            measures,
            jnp.any(~active & (measures != 0.0)),
            "Inactive face measures must be zero.",
        )
        weights = eqx.error_if(
            weights,
            jnp.any(weights < 0.0),
            "Face quadrature weights must be nonnegative.",
        )
        accounting_dtype = jnp.result_type(weights.dtype, measures.dtype)
        accounting_tolerance = (
            64.0 * jnp.finfo(accounting_dtype).eps
            if jnp.issubdtype(accounting_dtype, jnp.inexact)
            else 1.0e-12
        )
        weights = eqx.error_if(
            weights,
            ~jnp.allclose(
                jnp.sum(weights, axis=1),
                measures,
                rtol=accounting_tolerance,
                atol=accounting_tolerance,
            ),
            "Face quadrature weights must sum to the corresponding face measure.",
        )

        self.layout = layout
        self.face_centers = centers
        self.area_vectors = vectors
        self.face_measures = measures
        self.quadrature_points = points
        self.quadrature_weights = weights
        self.quadrature_grid_normal_velocity = grid_velocity

    @property
    def grid_normal_velocity(self) -> Array:
        """Quadrature-weighted face-average grid-normal velocity."""

        weighted_velocity = jnp.sum(
            self.quadrature_weights * self.quadrature_grid_normal_velocity, axis=1
        )
        safe_measures = jnp.where(
            self.face_measures > 0.0,
            self.face_measures,
            jnp.ones_like(self.face_measures),
        )
        return weighted_velocity / safe_measures


class FiniteVolumeGeometryStatus(IntEnum):
    """Portable validity status for one finite-volume stage geometry."""

    SUCCESS = 0
    FAILED = 1


class FiniteVolumeStageGeometryEvidence(StrictModule, NonTrainableState):
    """Dynamic defects and version bound to one static certification policy."""

    coordinate_effective_volume_defect: Array
    coordinate_effective_volume_tolerance: Array
    face_closure_defect: Array
    face_closure_tolerance: Array
    gcl_identity_defect: Array
    gcl_identity_tolerance: Array
    expected_order: Array
    proposed_reduction_factor: Array
    passed: Array
    status: Array
    evidence_version: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_effective_volume_defect: ArrayLike,
        coordinate_effective_volume_tolerance: ArrayLike,
        face_closure_defect: ArrayLike,
        face_closure_tolerance: ArrayLike,
        gcl_identity_defect: ArrayLike,
        gcl_identity_tolerance: ArrayLike,
        expected_order: ArrayLike,
        proposed_reduction_factor: ArrayLike,
        passed: ArrayLike,
        status: ArrayLike,
        evidence_version: ArrayLike,
        policy_id: str,
    ):
        policy = _canonical_identifier(policy_id, "policy_id")
        coordinate_defect = jnp.asarray(coordinate_effective_volume_defect)
        coordinate_tolerance = jnp.asarray(coordinate_effective_volume_tolerance)
        closure_defect = jnp.asarray(face_closure_defect)
        closure_tolerance = jnp.asarray(face_closure_tolerance)
        gcl_defect = jnp.asarray(gcl_identity_defect)
        gcl_tolerance = jnp.asarray(gcl_identity_tolerance)
        if coordinate_defect.ndim != 1 or coordinate_defect.size == 0:
            raise ValueError(
                "coordinate_effective_volume_defect must be a non-empty rank-1 array."
            )
        cell_shape = coordinate_defect.shape
        if (
            coordinate_tolerance.shape != cell_shape
            or closure_defect.shape != cell_shape
            or closure_tolerance.shape != cell_shape
            or gcl_defect.shape != cell_shape
            or gcl_tolerance.shape != cell_shape
        ):
            raise ValueError(
                "Geometry evidence defects and tolerances must have one entry per cell."
            )
        coordinate_defect = eqx.error_if(
            coordinate_defect,
            jnp.any(~jnp.isfinite(coordinate_defect))
            | jnp.any(~jnp.isfinite(coordinate_tolerance))
            | jnp.any(~jnp.isfinite(closure_defect))
            | jnp.any(~jnp.isfinite(closure_tolerance))
            | jnp.any(~jnp.isfinite(gcl_defect))
            | jnp.any(~jnp.isfinite(gcl_tolerance)),
            "Geometry evidence defects and tolerances must be finite.",
        )
        coordinate_defect = eqx.error_if(
            coordinate_defect,
            jnp.any(coordinate_defect < 0.0)
            | jnp.any(coordinate_tolerance < 0.0)
            | jnp.any(closure_defect < 0.0)
            | jnp.any(closure_tolerance < 0.0)
            | jnp.any(gcl_defect < 0.0)
            | jnp.any(gcl_tolerance < 0.0),
            "Geometry evidence defects and tolerances must be nonnegative.",
        )

        order = _dynamic_int32_scalar(
            expected_order,
            "expected_order",
            minimum=0,
        )
        version = _dynamic_int32_scalar(
            evidence_version,
            "evidence_version",
            minimum=0,
        )
        reduction = jnp.asarray(proposed_reduction_factor)
        if reduction.shape != ():
            raise ValueError("proposed_reduction_factor must be a scalar.")
        reduction = eqx.error_if(
            reduction,
            ~jnp.isfinite(reduction) | (reduction <= 0.0) | (reduction > 1.0),
            "proposed_reduction_factor must be a finite scalar in (0, 1].",
        )
        passed_value = jnp.asarray(passed)
        if passed_value.shape != () or passed_value.dtype.kind != "b":
            raise ValueError("passed must be a scalar boolean.")
        status_value = jnp.asarray(status)
        if status_value.shape != () or status_value.dtype.kind not in "iu":
            raise ValueError("status must be a scalar FiniteVolumeGeometryStatus.")
        status_value = eqx.error_if(
            status_value,
            (status_value != int(FiniteVolumeGeometryStatus.SUCCESS))
            & (status_value != int(FiniteVolumeGeometryStatus.FAILED)),
            "status must be a valid FiniteVolumeGeometryStatus.",
        ).astype(jnp.int32)
        computed_pass = (
            jnp.all(coordinate_defect <= coordinate_tolerance)
            & jnp.all(closure_defect <= closure_tolerance)
            & jnp.all(gcl_defect <= gcl_tolerance)
        )
        passed_value = eqx.error_if(
            passed_value,
            passed_value != computed_pass,
            "passed must equal the outcome implied by all geometry evidence defect "
            "tolerances.",
        )
        computed_status = jnp.where(
            computed_pass,
            int(FiniteVolumeGeometryStatus.SUCCESS),
            int(FiniteVolumeGeometryStatus.FAILED),
        )
        status_value = eqx.error_if(
            status_value,
            status_value != computed_status,
            "status must equal the outcome implied by all geometry evidence defect "
            "tolerances.",
        )

        self.coordinate_effective_volume_defect = coordinate_defect
        self.coordinate_effective_volume_tolerance = coordinate_tolerance
        self.face_closure_defect = closure_defect
        self.face_closure_tolerance = closure_tolerance
        self.gcl_identity_defect = gcl_defect
        self.gcl_identity_tolerance = gcl_tolerance
        self.expected_order = order
        self.proposed_reduction_factor = reduction
        self.passed = passed_value
        self.status = status_value
        self.evidence_version = version
        self.policy_id = policy

    @property
    def cell_count(self) -> int:
        return int(self.coordinate_effective_volume_defect.size)


class ALEGeometryConsistencyPolicy(StrictModule, NonTrainableState):
    """Mixed-tolerance certification and retry policy for SSPRK ALE geometry."""

    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    reduction_safety_factor: float = eqx.field(static=True)
    minimum_reduction_factor: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        reduction_safety_factor: float = 0.9,
        minimum_reduction_factor: float = 0.1,
    ):
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        safety = float(reduction_safety_factor)
        minimum = float(minimum_reduction_factor)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
            or (absolute == 0.0 and relative == 0.0)
        ):
            raise ValueError(
                "ALE absolute and relative tolerances must be finite, nonnegative, "
                "and not both zero."
            )
        if not np.isfinite(safety) or safety <= 0.0 or safety > 1.0:
            raise ValueError("reduction_safety_factor must be a finite value in (0, 1].")
        if not np.isfinite(minimum) or minimum <= 0.0 or minimum > safety:
            raise ValueError(
                "minimum_reduction_factor must be finite, positive, and no larger "
                "than reduction_safety_factor."
            )

        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.reduction_safety_factor = safety
        self.minimum_reduction_factor = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "ale-geometry-consistency-policy",
                "mixed_tolerance_rule": "absolute-plus-relative-reference",
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "reduction_rule": "safety-times-worst-ratio-to-negative-inverse-order",
                "zero_tolerance_ratio_rule": {
                    "zero_defect": 0.0,
                    "positive_defect": "infinity",
                },
                "reduction_safety_factor": safety,
                "minimum_reduction_factor": minimum,
                "ssprk33_stage_expected_order": 2,
                "ssprk33_final_expected_order": 4,
            }
        )

    def evidence(
        self,
        *,
        coordinate_effective_volume_defect: ArrayLike,
        coordinate_effective_volume_reference: ArrayLike,
        face_closure_defect: ArrayLike,
        face_closure_reference: ArrayLike,
        gcl_identity_defect: ArrayLike,
        gcl_identity_reference: ArrayLike,
        expected_order: int,
        evidence_version: ArrayLike,
    ) -> FiniteVolumeStageGeometryEvidence:
        """Certify one stage with an order-aware proposed retry factor."""

        order = int(expected_order)
        if order <= 0:
            raise ValueError("expected_order must be positive.")
        coordinate_defect = jnp.asarray(coordinate_effective_volume_defect)
        closure_defect = jnp.asarray(face_closure_defect)
        gcl_defect = jnp.asarray(gcl_identity_defect)
        coordinate_reference = jnp.asarray(coordinate_effective_volume_reference)
        closure_reference = jnp.asarray(face_closure_reference)
        gcl_reference = jnp.asarray(gcl_identity_reference)
        if not (
            coordinate_reference.shape
            == closure_reference.shape
            == gcl_reference.shape
            == coordinate_defect.shape
            == closure_defect.shape
            == gcl_defect.shape
        ):
            raise ValueError(
                "ALE defects and mixed-tolerance references must share one cell shape."
            )
        absolute = jnp.asarray(self.absolute_tolerance, dtype=coordinate_defect.dtype)
        relative = jnp.asarray(self.relative_tolerance, dtype=coordinate_defect.dtype)
        coordinate_tolerance = absolute + relative * jnp.abs(coordinate_reference)
        closure_tolerance = absolute + relative * jnp.abs(closure_reference)
        gcl_tolerance = absolute + relative * jnp.abs(gcl_reference)
        passed = (
            jnp.all(coordinate_defect <= coordinate_tolerance)
            & jnp.all(closure_defect <= closure_tolerance)
            & jnp.all(gcl_defect <= gcl_tolerance)
        )
        coordinate_ratio = _safe_defect_tolerance_ratio(
            coordinate_defect,
            coordinate_tolerance,
        )
        closure_ratio = _safe_defect_tolerance_ratio(
            closure_defect,
            closure_tolerance,
        )
        gcl_ratio = _safe_defect_tolerance_ratio(
            gcl_defect,
            gcl_tolerance,
        )
        worst_ratio = jnp.max(
            jnp.concatenate((coordinate_ratio, closure_ratio, gcl_ratio))
        )
        dtype = jnp.result_type(
            coordinate_defect.dtype,
            closure_defect.dtype,
            gcl_defect.dtype,
        )
        reduction = jnp.maximum(
            jnp.asarray(self.minimum_reduction_factor, dtype=dtype),
            jnp.asarray(self.reduction_safety_factor, dtype=dtype)
            * jnp.maximum(worst_ratio, jnp.asarray(1.0, dtype=dtype)) ** (-1.0 / order),
        )
        reduction = jnp.where(passed, jnp.asarray(1.0, dtype=dtype), reduction)
        status = jnp.where(
            passed,
            int(FiniteVolumeGeometryStatus.SUCCESS),
            int(FiniteVolumeGeometryStatus.FAILED),
        )
        return FiniteVolumeStageGeometryEvidence(
            coordinate_effective_volume_defect=coordinate_defect,
            coordinate_effective_volume_tolerance=coordinate_tolerance,
            face_closure_defect=closure_defect,
            face_closure_tolerance=closure_tolerance,
            gcl_identity_defect=gcl_defect,
            gcl_identity_tolerance=gcl_tolerance,
            expected_order=order,
            proposed_reduction_factor=reduction,
            passed=passed,
            status=status,
            evidence_version=evidence_version,
            policy_id=self.policy_id,
        )


class FiniteVolumeStageMetrics(StrictModule, NonTrainableState):
    """Immutable effective cell and face geometry bound to one dynamic version."""

    topology_epoch_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_version: Array
    time: Array
    effective_cell_volumes: Array
    coordinate_effective_cell_volumes: Array
    mesh_volume_rate: Array
    cell_centers: Array
    active_cell_mask: Array
    face_blocks: tuple[FiniteVolumeStageFaceBlock, ...]
    evidence: FiniteVolumeStageGeometryEvidence

    def __init__(
        self,
        *,
        topology_epoch_id: str,
        geometry_family_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        time: ArrayLike,
        effective_cell_volumes: ArrayLike,
        coordinate_effective_cell_volumes: ArrayLike,
        mesh_volume_rate: ArrayLike,
        cell_centers: ArrayLike,
        active_cell_mask: ArrayLike,
        face_blocks: tuple[FiniteVolumeStageFaceBlock, ...],
        evidence: FiniteVolumeStageGeometryEvidence,
    ):
        epoch = _canonical_identifier(topology_epoch_id, "topology_epoch_id")
        family_id = _canonical_identifier(
            geometry_family_id,
            "geometry_family_id",
        )
        layout_id = _canonical_identifier(geometry_layout_id, "geometry_layout_id")
        version = _dynamic_int32_scalar(
            geometry_version,
            "geometry_version",
            minimum=0,
        )
        time_value = jnp.asarray(time)
        if time_value.shape != ():
            raise ValueError("time must be a scalar.")
        time_value = eqx.error_if(
            time_value,
            ~jnp.isfinite(time_value),
            "time must be a finite scalar.",
        )
        volumes = jnp.asarray(effective_cell_volumes)
        if volumes.ndim != 1 or volumes.size == 0:
            raise ValueError("effective_cell_volumes must be a non-empty rank-1 array.")
        cell_count = int(volumes.size)
        coordinate = jnp.asarray(coordinate_effective_cell_volumes)
        volume_rate = jnp.asarray(mesh_volume_rate)
        centers = jnp.asarray(cell_centers)
        active = jnp.asarray(active_cell_mask)
        if centers.ndim != 2 or centers.shape[0] != cell_count or centers.shape[1] == 0:
            raise ValueError(
                "cell_centers must have shape (cell_count, spatial_dimension)."
            )
        if active.shape != (cell_count,) or active.dtype.kind != "b":
            raise ValueError(
                "active_cell_mask must be a boolean array with shape (cell_count,)."
            )
        volumes = eqx.error_if(
            volumes,
            jnp.any(~jnp.isfinite(volumes)) | jnp.any(~jnp.isfinite(centers)),
            "Effective cell volumes and cell centers must be finite.",
        )
        volumes = eqx.error_if(
            volumes,
            jnp.any(volumes < 0.0),
            "Effective cell volumes must be nonnegative.",
        )
        volumes = eqx.error_if(
            volumes,
            jnp.any(active & (volumes <= 0.0)),
            "Active effective cell volumes must be strictly positive.",
        )
        volumes = eqx.error_if(
            volumes,
            jnp.any(~active & (volumes != 0.0)),
            "Inactive effective cell volumes must be exactly zero.",
        )
        if coordinate.shape != (cell_count,):
            raise ValueError(
                "coordinate_effective_cell_volumes must have shape (cell_count,)."
            )
        coordinate = eqx.error_if(
            coordinate,
            jnp.any(~jnp.isfinite(coordinate)) | jnp.any(coordinate < 0.0),
            "coordinate_effective_cell_volumes must be finite and nonnegative "
            "with shape (cell_count,).",
        )
        coordinate = eqx.error_if(
            coordinate,
            jnp.any(active & (coordinate <= 0.0)),
            "Active coordinate-effective cell volumes must be strictly positive.",
        )
        coordinate = eqx.error_if(
            coordinate,
            jnp.any(~active & (coordinate != 0.0)),
            "Inactive coordinate-effective cell volumes must be exactly zero.",
        )
        if volume_rate.shape != (cell_count,):
            raise ValueError("mesh_volume_rate must be finite with shape (cell_count,).")
        volume_rate = eqx.error_if(
            volume_rate,
            jnp.any(~jnp.isfinite(volume_rate)),
            "mesh_volume_rate must be finite with shape (cell_count,).",
        )
        volume_rate = eqx.error_if(
            volume_rate,
            jnp.any(~active & (volume_rate != 0.0)),
            "Inactive mesh_volume_rate entries must be exactly zero.",
        )
        if not isinstance(evidence, FiniteVolumeStageGeometryEvidence):
            raise TypeError("evidence must be FiniteVolumeStageGeometryEvidence.")
        if evidence.cell_count != cell_count:
            raise ValueError("Geometry evidence must have one entry per stage cell.")

        blocks = tuple(face_blocks)
        if not all(isinstance(block, FiniteVolumeStageFaceBlock) for block in blocks):
            raise TypeError(
                "face_blocks must contain only FiniteVolumeStageFaceBlock instances."
            )
        block_ids = tuple(block.layout.block_id for block in blocks)
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Stage face block IDs must be unique.")
        if blocks:
            all_face_ids = jnp.concatenate(
                tuple(block.layout.face_ids for block in blocks)
            )
            sorted_face_ids = jnp.sort(all_face_ids)
            active = eqx.error_if(
                active,
                jnp.any(sorted_face_ids[1:] == sorted_face_ids[:-1]),
                "Stage face IDs must be unique across face blocks.",
            )
        spatial_dimension = int(centers.shape[1])
        for block in blocks:
            face_layout = block.layout
            if face_layout.spatial_dimension != spatial_dimension:
                raise ValueError(
                    "Face blocks and cell centers must have one spatial dimension."
                )
            owners = face_layout.owner_cells
            neighbours = face_layout.neighbour_cells
            face_active = face_layout.active_mask
            active = eqx.error_if(
                active,
                jnp.any(owners >= cell_count) | jnp.any(neighbours >= cell_count),
                "Face block routes index outside the stage cell set.",
            )
            active = eqx.error_if(
                active,
                jnp.any(face_active & ~active[owners]),
                "Every active face owner must be an active cell.",
            )
            internal_active = face_active & (neighbours >= 0)
            safe_neighbours = jnp.where(internal_active, neighbours, 0)
            active = eqx.error_if(
                active,
                jnp.any(internal_active & ~active[safe_neighbours]),
                "Every active internal face neighbour must be an active cell.",
            )

        self.topology_epoch_id = epoch
        self.geometry_family_id = family_id
        self.geometry_layout_id = layout_id
        self.geometry_version = version
        self.time = time_value
        self.effective_cell_volumes = volumes
        self.coordinate_effective_cell_volumes = coordinate
        self.mesh_volume_rate = volume_rate
        self.cell_centers = centers
        self.active_cell_mask = active
        self.face_blocks = blocks
        self.evidence = evidence

    @property
    def cell_count(self) -> int:
        return int(self.effective_cell_volumes.size)


def lower_static_unstructured_stage_metrics(
    discretization: UnstructuredFiniteVolumeDiscretization,
    /,
    *,
    time: ArrayLike = 0.0,
    topology_epoch_id: str | None = None,
) -> FiniteVolumeStageMetrics:
    """Lower static unstructured FV geometry without motion-specific dependencies."""

    from ._dyadic import DyadicFiniteVolumeDiscretization
    from ._unstructured import UnstructuredFiniteVolumeDiscretization

    if not isinstance(
        discretization,
        (UnstructuredFiniteVolumeDiscretization, DyadicFiniteVolumeDiscretization),
    ):
        raise TypeError("discretization must be explicit-face finite-volume geometry.")
    epoch = _canonical_identifier(
        (discretization.topology_id if topology_epoch_id is None else topology_epoch_id),
        "topology_epoch_id",
    )
    stage_blocks = tuple(
        FiniteVolumeStageFaceBlock(
            layout=FiniteVolumeStageFaceLayout(
                face_ids=block.face_ids,
                owner_cells=block.owner_cells,
                neighbour_cells=block.neighbour_cells,
                boundary_policy_ids=block.boundary_patch_ids,
                boundary_policy_count=len(discretization.boundary_patch_names),
                active_mask=block.active_mask,
                block_kind="physical",
                spatial_shape=tuple(block.face_centers.shape),
                quadrature_shape=tuple(
                    discretization.face_quadrature_weights[block.face_ids].shape
                ),
                block_id=canonical_fingerprint(
                    {
                        "kind": "static-unstructured-stage-face-route",
                        "topology_epoch": epoch,
                        "face_ids": array_tree_fingerprint(block.face_ids),
                        "owner_cells": array_tree_fingerprint(block.owner_cells),
                        "neighbour_cells": array_tree_fingerprint(block.neighbour_cells),
                        "boundary_policy_ids": array_tree_fingerprint(
                            block.boundary_patch_ids
                        ),
                        "boundary_policy_count": len(discretization.boundary_patch_names),
                        "block_kind": "physical",
                        "active_mask": array_tree_fingerprint(block.active_mask),
                        "spatial_shape": tuple(block.face_centers.shape),
                        "quadrature_shape": tuple(
                            discretization.face_quadrature_weights[block.face_ids].shape
                        ),
                    }
                ),
            ),
            face_centers=block.face_centers,
            area_vectors=block.area_vectors,
            face_measures=block.face_measures,
            quadrature_points=discretization.face_quadrature_points[block.face_ids],
            quadrature_weights=discretization.face_quadrature_weights[block.face_ids],
            quadrature_grid_normal_velocity=jnp.zeros_like(
                discretization.face_quadrature_weights[block.face_ids]
            ),
        )
        for block in discretization.face_blocks
    )
    geometry_layout_id = canonical_fingerprint(
        {
            "kind": "static-unstructured-stage-geometry-layout",
            "topology": discretization.topology_id,
            "cell_shape": tuple(discretization.cell_volumes.shape),
            "cell_dtype": str(discretization.cell_volumes.dtype),
            "center_shape": tuple(discretization.cell_centers.shape),
            "quadrature_layouts": [
                {
                    "face_ids": array_tree_fingerprint(block.face_ids),
                    "owner_cells": array_tree_fingerprint(block.owner_cells),
                    "neighbour_cells": array_tree_fingerprint(block.neighbour_cells),
                    "active_mask": array_tree_fingerprint(block.active_mask),
                    "boundary_policy_ids": array_tree_fingerprint(
                        block.boundary_patch_ids
                    ),
                    "boundary_policy_count": len(discretization.boundary_patch_names),
                    "block_kind": "physical",
                    "spatial_shape": tuple(block.face_centers.shape),
                    "shape": tuple(
                        discretization.face_quadrature_weights[block.face_ids].shape
                    ),
                    "dtype": str(
                        discretization.face_quadrature_weights[block.face_ids].dtype
                    ),
                }
                for block in discretization.face_blocks
            ],
        }
    )
    roundoff_multiplier = 32.0
    roundoff_tolerance = (
        roundoff_multiplier
        * jnp.finfo(discretization.cell_volumes.dtype).eps
        * jnp.maximum(jnp.abs(discretization.cell_volumes), 1.0)
    )
    evidence_policy_id = canonical_fingerprint(
        {
            "kind": "static-unstructured-stage-geometry-evidence-policy",
            "expected_order": 0,
            "proposed_reduction_factor": 1.0,
            "tolerance_policy": {
                "fields": (
                    "coordinate_effective_volume_tolerance",
                    "face_closure_tolerance",
                    "gcl_identity_tolerance",
                ),
                "rule": "scaled-dtype-roundoff",
                "roundoff_multiplier": roundoff_multiplier,
                "reference_floor": 1.0,
            },
            "numeric_version": discretization.numeric_version,
        }
    )
    zero_cell_defect = jnp.zeros_like(discretization.cell_volumes)
    evidence = FiniteVolumeStageGeometryEvidence(
        coordinate_effective_volume_defect=zero_cell_defect,
        coordinate_effective_volume_tolerance=roundoff_tolerance,
        face_closure_defect=zero_cell_defect,
        face_closure_tolerance=roundoff_tolerance,
        gcl_identity_defect=zero_cell_defect,
        gcl_identity_tolerance=roundoff_tolerance,
        expected_order=0,
        proposed_reduction_factor=1.0,
        passed=True,
        status=FiniteVolumeGeometryStatus.SUCCESS,
        evidence_version=0,
        policy_id=evidence_policy_id,
    )
    return FiniteVolumeStageMetrics(
        topology_epoch_id=epoch,
        geometry_family_id=discretization.geometry_id,
        geometry_layout_id=geometry_layout_id,
        geometry_version=0,
        time=time,
        effective_cell_volumes=discretization.cell_volumes,
        coordinate_effective_cell_volumes=discretization.cell_volumes,
        mesh_volume_rate=jnp.zeros_like(discretization.cell_volumes),
        cell_centers=discretization.cell_centers,
        active_cell_mask=jnp.ones((discretization.cell_count,), dtype=bool),
        face_blocks=stage_blocks,
        evidence=evidence,
    )


@runtime_checkable
class PreparedFiniteVolumeGeometry(Protocol):
    """Minimal FV geometry contract shared without explicit connectivity."""

    component_names: tuple[str, ...]
    cell_space: DiscreteFieldSpace
    cell_volumes: Array
    cell_centers: Array
    prepared_id: str
    numeric_version: str

    @property
    def component_count(self) -> int: ...

    @property
    def state_shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class ExplicitFaceBlockGeometry(PreparedFiniteVolumeGeometry, Protocol):
    """FV geometry exposing owner/neighbour face blocks."""

    face_blocks: tuple[FiniteVolumeFaceBlock, ...]

    @property
    def cell_count(self) -> int: ...


__all__ = [
    "ALEGeometryConsistencyPolicy",
    "ExplicitFaceBlockGeometry",
    "FiniteVolumeFaceBlock",
    "FiniteVolumeStageFaceBlock",
    "FiniteVolumeStageFaceLayout",
    "FiniteVolumeStageMetrics",
    "PreparedFiniteVolumeGeometry",
    "lower_static_unstructured_stage_metrics",
]

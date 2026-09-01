#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    CollisionSurfacePlan,
    ContactPairPolicy,
    ContactPrecisionPolicy,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from ...linalg import ArraySpace, SmallLinearSolvePlan, solve_small_linear


RodDimension: TypeAlias = Literal[2, 3]
RodEndpoint: TypeAlias = Literal["start", "end"]
RodIntegrator: TypeAlias = Literal["symplectic"]


def _require_real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_psd(name: str, matrices: np.ndarray, /) -> None:
    if matrices.shape[0] == 0:
        return
    scale = max(1.0, float(np.max(np.abs(matrices))))
    tolerance = 100.0 * np.finfo(matrices.dtype).eps * scale
    if not np.allclose(
        matrices,
        np.swapaxes(matrices, -1, -2),
        rtol=100.0 * np.finfo(matrices.dtype).eps,
        atol=tolerance,
    ):
        raise ValueError(f"{name} must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(matrices))) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")


def _validate_spd(name: str, matrices: np.ndarray, /) -> None:
    _validate_psd(name, matrices)
    scale = max(1.0, float(np.max(np.abs(matrices))))
    tolerance = 100.0 * np.finfo(matrices.dtype).eps * scale
    if float(np.min(np.linalg.eigvalsh(matrices))) <= tolerance:
        raise ValueError(f"{name} must be positive definite.")


def _frames_to_quaternions(frames: np.ndarray, /) -> np.ndarray:
    quaternions = np.empty((frames.shape[0], 4), dtype=frames.dtype)
    for index, matrix in enumerate(frames):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.asarray(
                (
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ),
                dtype=frames.dtype,
            )
        elif matrix[0, 0] >= matrix[1, 1] and matrix[0, 0] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ),
                dtype=frames.dtype,
            )
        elif matrix[1, 1] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            )
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ),
                dtype=frames.dtype,
            )
        else:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            )
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=frames.dtype,
            )
        quaternion = quaternion / np.linalg.norm(quaternion)
        quaternions[index] = np.where(quaternion[0] < 0.0, -quaternion, quaternion)
    return quaternions


def _quaternion_conjugate(quaternion: Array, /) -> Array:
    return jnp.concatenate((quaternion[..., :1], -quaternion[..., 1:]), axis=-1)


def _quaternion_multiply(left: Array, right: Array, /) -> Array:
    left_scalar = left[..., 0]
    left_vector = left[..., 1:]
    right_scalar = right[..., 0]
    right_vector = right[..., 1:]
    scalar = left_scalar * right_scalar - jnp.sum(left_vector * right_vector, axis=-1)
    vector = (
        left_scalar[..., None] * right_vector
        + right_scalar[..., None] * left_vector
        + jnp.cross(left_vector, right_vector)
    )
    return jnp.concatenate((scalar[..., None], vector), axis=-1)


def _safe_unit_quaternion(quaternion: Array, /) -> tuple[Array, Array]:
    norm = jnp.sqrt(jnp.sum(quaternion * quaternion, axis=-1))
    tiny = jnp.sqrt(jnp.finfo(quaternion.dtype).tiny)
    safe_norm = jnp.maximum(norm, tiny)
    normalized = quaternion / safe_norm[..., None]
    identity = jnp.zeros_like(normalized).at[..., 0].set(1.0)
    normalized = jnp.where((norm > tiny)[..., None], normalized, identity)
    return normalized, norm


def _quaternion_rotation_matrix(quaternion: Array, /) -> Array:
    scalar = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]
    return jnp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - scalar * z),
            2.0 * (x * z + scalar * y),
            2.0 * (x * y + scalar * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - scalar * x),
            2.0 * (x * z - scalar * y),
            2.0 * (y * z + scalar * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _planar_rotation_matrix(angle: Array, /) -> Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(angle.shape + (2, 2))


def _quaternion_rotation_vector(quaternion: Array, /) -> Array:
    canonical = jnp.where((quaternion[..., :1] < 0.0), -quaternion, quaternion)
    scalar = jnp.clip(canonical[..., 0], 0.0, 1.0)
    vector = canonical[..., 1:]
    squared_norm = jnp.sum(vector * vector, axis=-1)
    threshold = jnp.finfo(quaternion.dtype).eps
    safe_norm = jnp.sqrt(jnp.maximum(squared_norm, threshold))
    angle = 2.0 * jnp.arctan2(safe_norm, scalar)
    regular_scale = angle / safe_norm
    limiting_scale = 2.0 / jnp.maximum(scalar, jnp.sqrt(threshold)) + squared_norm / 3.0
    scale = jnp.where(squared_norm > threshold, regular_scale, limiting_scale)
    return scale[..., None] * vector


class RodPlan(StrictModule, NonTrainableState):
    """Fixed ordered centerline topology and constitutive data for one Cosserat rod."""

    segment_node_ids: Array
    rest_positions: Array
    rest_frames: Array
    rest_lengths: Array
    node_masses: Array
    segment_inertias: Array
    stretch_shear_stiffness: Array
    bend_twist_stiffness: Array
    dimension: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    inextensible: bool = eqx.field(static=True)
    minimum_segment_length: float = eqx.field(static=True)
    chart_margin: float = eqx.field(static=True)
    orientation_norm_tolerance: float = eqx.field(static=True)
    inextensibility_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_node_ids: ArrayLike,
        rest_positions: ArrayLike,
        rest_frames: ArrayLike,
        node_masses: ArrayLike,
        segment_inertias: ArrayLike,
        stretch_shear_stiffness: ArrayLike,
        bend_twist_stiffness: ArrayLike,
        /,
        *,
        rest_lengths: ArrayLike | None = None,
        inextensible: bool = False,
        minimum_segment_length: float = 1.0e-9,
        chart_margin: float = 1.0e-5,
        orientation_norm_tolerance: float = 1.0e-4,
        inextensibility_tolerance: float = 1.0e-5,
        plan_id: str | None = None,
    ):
        segments = np.asarray(segment_node_ids)
        if segments.ndim != 2 or segments.shape[1:] != (2,) or segments.shape[0] < 1:
            raise ValueError("segment_node_ids must have shape (segment_count, 2).")
        if not np.issubdtype(segments.dtype, np.integer):
            raise TypeError("segment_node_ids must have an integer dtype.")
        positions = _require_real_array("rest_positions", rest_positions, 2)
        dimension = int(positions.shape[1])
        if dimension not in (2, 3):
            raise ValueError("Cosserat rods require ambient dimension 2 or 3.")
        segment_count = int(segments.shape[0])
        node_count = int(positions.shape[0])
        if node_count != segment_count + 1:
            raise ValueError("An ordered rod must contain one more node than segments.")
        expected_ids = np.arange(node_count)
        flattened = segments.reshape((-1,))
        if (
            np.any(segments < 0)
            or np.any(segments >= node_count)
            or np.any(segments[:, 0] == segments[:, 1])
            or not np.array_equal(segments[:-1, 1], segments[1:, 0])
            or not np.array_equal(np.sort(np.unique(flattened)), expected_ids)
        ):
            raise ValueError(
                "segment_node_ids must describe one ordered, non-self-looping path "
                "through every contiguous node ID exactly once."
            )
        frames = _require_real_array("rest_frames", rest_frames, 3)
        if frames.shape != (segment_count, dimension, dimension):
            raise ValueError("rest_frames must contain one square frame per segment.")
        identity = np.eye(dimension, dtype=frames.dtype)
        orthogonality = np.matmul(np.swapaxes(frames, -1, -2), frames)
        frame_tolerance = 500.0 * np.finfo(frames.dtype).eps
        if not np.allclose(
            orthogonality, identity, rtol=frame_tolerance, atol=frame_tolerance
        ) or not np.allclose(
            np.linalg.det(frames), 1.0, rtol=frame_tolerance, atol=frame_tolerance
        ):
            raise ValueError("rest_frames must be right-handed orthonormal frames.")
        masses = _require_real_array("node_masses", node_masses, 1)
        if masses.shape != (node_count,) or np.any(masses <= 0.0):
            raise ValueError("node_masses must be positive with one value per node.")
        inertia_rank = 1 if dimension == 2 else 3
        inertias = _require_real_array("segment_inertias", segment_inertias, inertia_rank)
        if dimension == 2:
            if inertias.shape != (segment_count,) or np.any(inertias <= 0.0):
                raise ValueError(
                    "Planar segment_inertias must be positive with one value per segment."
                )
        else:
            if inertias.shape != (segment_count, 3, 3):
                raise ValueError(
                    "Spatial segment_inertias must have shape (segments, 3, 3)."
                )
            _validate_spd("segment_inertias", inertias)
        stretch = _require_real_array(
            "stretch_shear_stiffness", stretch_shear_stiffness, 3
        )
        if stretch.shape != (segment_count, dimension, dimension):
            raise ValueError(
                "stretch_shear_stiffness must contain one material matrix per segment."
            )
        _validate_psd("stretch_shear_stiffness", stretch)
        rotation_dimension = 1 if dimension == 2 else 3
        bend = _require_real_array("bend_twist_stiffness", bend_twist_stiffness, 3)
        if bend.shape != (
            max(segment_count - 1, 0),
            rotation_dimension,
            rotation_dimension,
        ):
            raise ValueError(
                "bend_twist_stiffness must contain one material matrix per segment junction."
            )
        _validate_psd("bend_twist_stiffness", bend)
        minimum = float(minimum_segment_length)
        chart = float(chart_margin)
        norm_tolerance = float(orientation_norm_tolerance)
        constraint_tolerance = float(inextensibility_tolerance)
        if (
            not isfinite(minimum)
            or minimum <= 0.0
            or not isfinite(chart)
            or chart <= 0.0
            or chart >= np.pi
            or not isfinite(norm_tolerance)
            or norm_tolerance <= 0.0
            or not isfinite(constraint_tolerance)
            or constraint_tolerance <= 0.0
        ):
            raise ValueError("Rod geometric and chart tolerances are invalid.")
        geometric_lengths = np.linalg.norm(
            positions[segments[:, 1]] - positions[segments[:, 0]], axis=-1
        )
        if rest_lengths is None:
            lengths = geometric_lengths
        else:
            lengths = _require_real_array("rest_lengths", rest_lengths, 1)
            if lengths.shape != (segment_count,):
                raise ValueError("rest_lengths must contain one value per segment.")
            length_scale = max(1.0, float(np.max(geometric_lengths)))
            if not np.allclose(
                lengths,
                geometric_lengths,
                rtol=500.0 * np.finfo(positions.dtype).eps,
                atol=500.0 * np.finfo(positions.dtype).eps * length_scale,
            ):
                raise ValueError(
                    "rest_lengths must agree with the supplied rest centerline geometry."
                )
        if np.any(lengths <= minimum):
            raise ValueError("Every rest segment must exceed minimum_segment_length.")
        dtype = positions.dtype
        arrays = {
            "segments": segments.astype(np.int32, copy=False),
            "rest_positions": positions,
            "rest_frames": frames.astype(dtype, copy=False),
            "rest_lengths": lengths.astype(dtype, copy=False),
            "node_masses": masses.astype(dtype, copy=False),
            "segment_inertias": inertias.astype(dtype, copy=False),
            "stretch_shear_stiffness": stretch.astype(dtype, copy=False),
            "bend_twist_stiffness": bend.astype(dtype, copy=False),
        }
        generated = canonical_fingerprint(
            {
                "kind": "fixed-topology-cosserat-rod-plan",
                "dimension": dimension,
                "inextensible": bool(inextensible),
                "minimum_segment_length": minimum,
                "chart_margin": chart,
                "orientation_norm_tolerance": norm_tolerance,
                "inextensibility_tolerance": constraint_tolerance,
                "values": array_tree_fingerprint(arrays),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.segment_node_ids = jnp.asarray(arrays["segments"])
        self.rest_positions = jnp.asarray(arrays["rest_positions"])
        self.rest_frames = jnp.asarray(arrays["rest_frames"])
        self.rest_lengths = jnp.asarray(arrays["rest_lengths"])
        self.node_masses = jnp.asarray(arrays["node_masses"])
        self.segment_inertias = jnp.asarray(arrays["segment_inertias"])
        self.stretch_shear_stiffness = jnp.asarray(arrays["stretch_shear_stiffness"])
        self.bend_twist_stiffness = jnp.asarray(arrays["bend_twist_stiffness"])
        self.dimension = dimension
        self.segment_count = segment_count
        self.node_count = node_count
        self.inextensible = bool(inextensible)
        self.minimum_segment_length = minimum
        self.chart_margin = chart
        self.orientation_norm_tolerance = norm_tolerance
        self.inextensibility_tolerance = constraint_tolerance
        self.plan_id = identifier


class RodState(StrictModule):
    """Centerline and material-orientation phase state of one discrete rod."""

    positions: Array
    velocities: Array
    orientations: Array
    angular_velocities: Array

    def __init__(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        orientations: ArrayLike,
        angular_velocities: ArrayLike,
        /,
    ):
        positions_ = jnp.asarray(positions)
        velocities_ = jnp.asarray(velocities)
        orientations_ = jnp.asarray(orientations)
        angular_ = jnp.asarray(angular_velocities)
        if positions_.ndim != 2 or positions_.shape[-1] not in (2, 3):
            raise ValueError("Rod positions must have shape (nodes, 2|3).")
        if velocities_.shape != positions_.shape:
            raise ValueError("Rod velocities must match positions.")
        segment_count = positions_.shape[0] - 1
        if positions_.shape[-1] == 2:
            if orientations_.shape != (segment_count,) or angular_.shape != (
                segment_count,
            ):
                raise ValueError(
                    "Planar orientations and angular velocities must have shape (segments,)."
                )
        else:
            if orientations_.shape != (segment_count, 4) or angular_.shape != (
                segment_count,
                3,
            ):
                raise ValueError(
                    "Spatial orientations/angular velocities must have shapes "
                    "(segments, 4) and (segments, 3)."
                )
        for name, value in (
            ("positions", positions_),
            ("velocities", velocities_),
            ("orientations", orientations_),
            ("angular_velocities", angular_),
        ):
            if not jnp.issubdtype(value.dtype, jnp.inexact) or jnp.iscomplexobj(value):
                raise TypeError(f"Rod {name} must be a real inexact array.")
        self.positions = positions_
        self.velocities = velocities_
        self.orientations = orientations_
        self.angular_velocities = angular_


class PreparedRod(StrictModule, NonTrainableState):
    """Prepared objective strain maps for one fixed rod topology."""

    plan: RodPlan
    rest_orientations: Array
    rest_stretch_shear: Array
    rest_relative_orientations: Array
    dual_lengths: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: RodPlan, /):
        if not isinstance(plan, RodPlan):
            raise TypeError("plan must be a RodPlan.")
        segment_vectors = (
            plan.rest_positions[plan.segment_node_ids[:, 1]]
            - plan.rest_positions[plan.segment_node_ids[:, 0]]
        )
        rest_stretch_shear = oe.contract(
            "sji,sj->si",
            plan.rest_frames,
            segment_vectors / plan.rest_lengths[:, None],
        )
        if plan.dimension == 2:
            orientations = jnp.arctan2(
                plan.rest_frames[:, 1, 0], plan.rest_frames[:, 0, 0]
            )
            relative = orientations[1:] - orientations[:-1]
        else:
            quaternion_values = _frames_to_quaternions(np.asarray(plan.rest_frames))
            orientations = jnp.asarray(quaternion_values, dtype=plan.rest_positions.dtype)
            relative = _quaternion_multiply(
                _quaternion_conjugate(orientations[:-1]), orientations[1:]
            )
        dual_lengths = 0.5 * (plan.rest_lengths[:-1] + plan.rest_lengths[1:])
        self.plan = plan
        self.rest_orientations = orientations
        self.rest_stretch_shear = rest_stretch_shear
        self.rest_relative_orientations = relative
        self.dual_lengths = dual_lengths
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-topology-cosserat-rod",
                "plan": plan.plan_id,
                "rest_kinematics": array_tree_fingerprint(
                    {
                        "orientation": np.asarray(orientations),
                        "stretch_shear": np.asarray(rest_stretch_shear),
                        "relative": np.asarray(relative),
                        "dual_lengths": np.asarray(dual_lengths),
                    }
                ),
            }
        )

    def initialize_state(self) -> RodState:
        return RodState(
            self.plan.rest_positions,
            jnp.zeros_like(self.plan.rest_positions),
            self.rest_orientations,
            jnp.zeros(
                (
                    (self.plan.segment_count,)
                    if self.plan.dimension == 2
                    else (self.plan.segment_count, 3)
                ),
                dtype=self.plan.rest_positions.dtype,
            ),
        )

    def collision_surface(
        self,
        /,
        *,
        vertex_ids: ArrayLike | None = None,
        body_id: int = 0,
        patch_id: int = 0,
        minimum_separation: float = 0.0,
    ) -> PreparedCollisionSurface:
        """Expose the rod centerline through the shared collision interface."""
        identifiers = (
            np.arange(self.plan.node_count, dtype=np.int64)
            if vertex_ids is None
            else np.asarray(vertex_ids)
        )
        if identifiers.shape != (self.plan.node_count,) or not np.issubdtype(
            identifiers.dtype, np.integer
        ):
            raise TypeError("vertex_ids must be one integer ID per rod node.")
        pair_policy = ContactPairPolicy(
            self.plan.node_count,
            body_ids=np.full((self.plan.node_count,), int(body_id), dtype=np.int64),
            patch_ids=np.full((self.plan.node_count,), int(patch_id), dtype=np.int64),
        )
        topology = CollisionSurfacePlan(
            identifiers,
            ambient_dimension=self.plan.dimension,
            edges=self.plan.segment_node_ids,
            pair_policy=pair_policy,
            minimum_separation=minimum_separation,
        )
        dtype = np.dtype(self.plan.rest_positions.dtype)
        source = ArraySpace((self.plan.node_count, self.plan.dimension), dtype=dtype)
        precision = ContactPrecisionPolicy(
            geometry_dtype=dtype,
            accumulation_dtype=np.float64,
            certification_dtype=np.float64,
            output_dtype=dtype,
        )
        return PreparedCollisionSurface(
            topology,
            self.plan.rest_positions,
            selection_collision_operator(
                source, np.arange(self.plan.node_count, dtype=np.int32)
            ),
            precision=precision,
        )

    def evaluate(self, state: RodState, /) -> RodEvaluation:
        return evaluate_rod(self, state)


def prepare_rod(plan: RodPlan, /) -> PreparedRod:
    """Prepare immutable rest strains and orientation charts for a rod plan."""
    return PreparedRod(plan)


def _validate_state_contract(prepared: PreparedRod, state: RodState, /) -> None:
    if not isinstance(state, RodState):
        raise TypeError("state must be a RodState.")
    plan = prepared.plan
    if (
        state.positions.shape != (plan.node_count, plan.dimension)
        or state.velocities.shape != state.positions.shape
    ):
        raise ValueError("Rod state centerline shape does not match the prepared rod.")
    if plan.dimension == 2:
        valid_orientation_shape = state.orientations.shape == (plan.segment_count,)
        valid_angular_shape = state.angular_velocities.shape == (plan.segment_count,)
    else:
        valid_orientation_shape = state.orientations.shape == (plan.segment_count, 4)
        valid_angular_shape = state.angular_velocities.shape == (plan.segment_count, 3)
    if not valid_orientation_shape or not valid_angular_shape:
        raise ValueError("Rod state orientation shape does not match the prepared rod.")


def _rod_strains(
    prepared: PreparedRod,
    positions: Array,
    orientations: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    plan = prepared.plan
    vectors = (
        positions[plan.segment_node_ids[:, 1]] - positions[plan.segment_node_ids[:, 0]]
    )
    lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    if plan.dimension == 2:
        frames = _planar_rotation_matrix(orientations)
        orientation_norms = jnp.ones_like(orientations)
        relative_error = (
            orientations[1:] - orientations[:-1] - prepared.rest_relative_orientations
        )
        wrapped_error = jnp.arctan2(jnp.sin(relative_error), jnp.cos(relative_error))
        bend_twist = wrapped_error[:, None] / prepared.dual_lengths[:, None]
        chart_valid = jnp.abs(wrapped_error) < (jnp.pi - plan.chart_margin)
    else:
        normalized, orientation_norms = _safe_unit_quaternion(orientations)
        frames = _quaternion_rotation_matrix(normalized)
        current_relative = _quaternion_multiply(
            _quaternion_conjugate(normalized[:-1]), normalized[1:]
        )
        relative_error = _quaternion_multiply(
            _quaternion_conjugate(prepared.rest_relative_orientations),
            current_relative,
        )
        relative_error, _ = _safe_unit_quaternion(relative_error)
        bend_twist = (
            _quaternion_rotation_vector(relative_error) / prepared.dual_lengths[:, None]
        )
        chart_threshold = jnp.sin(0.5 * plan.chart_margin)
        chart_valid = jnp.abs(relative_error[:, 0]) > chart_threshold
    stretch_shear = (
        oe.contract("sji,sj->si", frames, vectors / plan.rest_lengths[:, None])
        - prepared.rest_stretch_shear
    )
    if plan.inextensible:
        axial_component = jnp.sum(
            stretch_shear * prepared.rest_stretch_shear, axis=-1, keepdims=True
        )
        constitutive_stretch_shear = (
            stretch_shear - axial_component * prepared.rest_stretch_shear
        )
    else:
        constitutive_stretch_shear = stretch_shear
    return (
        stretch_shear,
        constitutive_stretch_shear,
        bend_twist,
        lengths,
        orientation_norms,
        chart_valid,
    )


def rod_potential_energy(
    prepared: PreparedRod,
    positions: ArrayLike,
    orientations: ArrayLike,
    /,
) -> Array:
    """Return objective discrete stretch/shear and bend/twist energy."""
    positions_ = jnp.asarray(positions)
    orientations_ = jnp.asarray(orientations)
    _, constitutive, bend_twist, _, _, _ = _rod_strains(
        prepared, positions_, orientations_
    )
    stretch_density = 0.5 * oe.contract(
        "si,sij,sj->s",
        constitutive,
        prepared.plan.stretch_shear_stiffness,
        constitutive,
    )
    bend_density = 0.5 * oe.contract(
        "si,sij,sj->s",
        bend_twist,
        prepared.plan.bend_twist_stiffness,
        bend_twist,
    )
    return jnp.sum(prepared.plan.rest_lengths * stretch_density) + jnp.sum(
        prepared.dual_lengths * bend_density
    )


class RodEvaluation(StrictModule):
    """Energy-gradient loads and complete finite-domain evidence for one rod state."""

    stretch_shear_strain: Array
    constitutive_stretch_shear_strain: Array
    bend_twist_strain: Array
    segment_lengths: Array
    internal_forces: Array
    internal_moments: Array
    potential_energy: Array
    kinetic_energy: Array
    total_energy: Array
    minimum_segment_length: Array
    maximum_orientation_norm_error: Array
    inextensibility_error: Array
    resultant_force_residual: Array
    finite: Array
    chart_valid: Array
    orientation_valid: Array
    nondegenerate: Array
    inextensibility_valid: Array
    valid: Array
    rod_id: str = eqx.field(static=True)


def evaluate_rod(prepared: PreparedRod, state: RodState, /) -> RodEvaluation:
    """Evaluate strains, energies, and conservative generalized rod loads."""
    if not isinstance(prepared, PreparedRod):
        raise TypeError("prepared must be a PreparedRod.")
    _validate_state_contract(prepared, state)
    plan = prepared.plan
    (
        stretch_shear,
        constitutive,
        bend_twist,
        lengths,
        orientation_norms,
        chart_evidence,
    ) = _rod_strains(prepared, state.positions, state.orientations)
    potential, gradients = jax.value_and_grad(
        lambda positions, orientations: rod_potential_energy(
            prepared, positions, orientations
        ),
        argnums=(0, 1),
    )(state.positions, state.orientations)
    position_gradient, orientation_gradient = gradients
    internal_forces = -position_gradient
    if plan.dimension == 2:
        internal_moments = -orientation_gradient
        rotational_kinetic = 0.5 * jnp.sum(
            plan.segment_inertias * state.angular_velocities**2
        )
        maximum_norm_error = jnp.asarray(0.0, dtype=potential.dtype)
        orientation_valid = jnp.all(jnp.isfinite(state.orientations))
    else:
        normalized, _ = _safe_unit_quaternion(state.orientations)
        material_gradient = _quaternion_multiply(
            _quaternion_conjugate(normalized), orientation_gradient
        )
        internal_moments = -0.5 * material_gradient[:, 1:]
        rotational_kinetic = 0.5 * oe.contract(
            "si,sij,sj->",
            state.angular_velocities,
            plan.segment_inertias,
            state.angular_velocities,
        )
        maximum_norm_error = jnp.max(jnp.abs(orientation_norms - 1.0))
        orientation_valid = jnp.all(jnp.isfinite(state.orientations)) & (
            maximum_norm_error <= plan.orientation_norm_tolerance
        )
    translational_kinetic = 0.5 * jnp.sum(plan.node_masses[:, None] * state.velocities**2)
    kinetic = translational_kinetic + rotational_kinetic
    total = potential + kinetic
    minimum_length = jnp.min(lengths)
    nondegenerate = jnp.all(jnp.isfinite(lengths)) & (
        minimum_length > plan.minimum_segment_length
    )
    chart_valid = jnp.all(chart_evidence)
    inextensibility_error = jnp.max(jnp.abs(lengths - plan.rest_lengths))
    inextensibility_valid = (~jnp.asarray(plan.inextensible)) | (
        inextensibility_error <= plan.inextensibility_tolerance
    )
    resultant_residual = jnp.sqrt(jnp.sum(jnp.sum(internal_forces, axis=0) ** 2))
    finite = (
        jnp.all(jnp.isfinite(stretch_shear))
        & jnp.all(jnp.isfinite(constitutive))
        & jnp.all(jnp.isfinite(bend_twist))
        & jnp.all(jnp.isfinite(internal_forces))
        & jnp.all(jnp.isfinite(internal_moments))
        & jnp.isfinite(potential)
        & jnp.isfinite(kinetic)
        & jnp.isfinite(total)
        & jnp.isfinite(resultant_residual)
    )
    valid = (
        finite & chart_valid & orientation_valid & nondegenerate & inextensibility_valid
    )
    return RodEvaluation(
        stretch_shear,
        constitutive,
        bend_twist,
        lengths,
        internal_forces,
        internal_moments,
        potential,
        kinetic,
        total,
        minimum_length,
        maximum_norm_error,
        inextensibility_error,
        resultant_residual,
        finite,
        chart_valid,
        orientation_valid,
        nondegenerate,
        inextensibility_valid,
        valid,
        prepared.prepared_id,
    )


class RodEndpointAttachment(StrictModule, NonTrainableState):
    """One rod endpoint coincident with an offset point of a rigid body."""

    endpoint: RodEndpoint = eqx.field(static=True)
    rigid_body_id: int = eqx.field(static=True)
    local_offset: Array
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint: RodEndpoint,
        rigid_body_id: int,
        local_offset: ArrayLike,
        /,
        *,
        attachment_id: str | None = None,
    ):
        if endpoint not in ("start", "end"):
            raise ValueError("endpoint must be 'start' or 'end'.")
        body_id = int(rigid_body_id)
        if body_id < 0:
            raise ValueError("rigid_body_id must be nonnegative.")
        offset = _require_real_array("local_offset", local_offset, 1)
        if offset.shape not in ((2,), (3,)):
            raise ValueError("local_offset must be a planar or spatial vector.")
        generated = canonical_fingerprint(
            {
                "kind": "cosserat-rod-endpoint-rigid-attachment",
                "endpoint": endpoint,
                "rigid_body_id": body_id,
                "offset": array_tree_fingerprint(offset),
            }
        )
        identifier = generated if attachment_id is None else str(attachment_id)
        if not identifier:
            raise ValueError("attachment_id must be nonempty.")
        self.endpoint = endpoint
        self.rigid_body_id = body_id
        self.local_offset = jnp.asarray(offset)
        self.attachment_id = identifier


class RodEndpointResponse(StrictModule):
    """Explicit equal-and-opposite endpoint wrench transfer evidence."""

    endpoint_position: Array
    force_on_rod: Array
    moment_on_rod: Array
    force_on_rigid: Array
    moment_on_rigid: Array
    force_balance: Array
    moment_balance: Array
    finite: Array
    valid: Array
    attachment_id: str = eqx.field(static=True)
    rod_id: str = eqx.field(static=True)


def evaluate_endpoint_attachment(
    prepared: PreparedRod,
    state: RodState,
    evaluation: RodEvaluation,
    attachment: RodEndpointAttachment,
    rigid_orientation: ArrayLike,
    /,
) -> RodEndpointResponse:
    """Transfer a conservative endpoint wrench to an attached rigid-body frame."""
    if not isinstance(prepared, PreparedRod):
        raise TypeError("prepared must be a PreparedRod.")
    _validate_state_contract(prepared, state)
    if not isinstance(evaluation, RodEvaluation):
        raise TypeError("evaluation must be a RodEvaluation.")
    if evaluation.rod_id != prepared.prepared_id:
        raise ValueError("evaluation was produced by a different prepared rod.")
    if not isinstance(attachment, RodEndpointAttachment):
        raise TypeError("attachment must be a RodEndpointAttachment.")
    plan = prepared.plan
    if attachment.local_offset.shape != (plan.dimension,):
        raise ValueError("Attachment dimension does not match the rod dimension.")
    body_orientation = jnp.asarray(rigid_orientation)
    node_index = 0 if attachment.endpoint == "start" else plan.node_count - 1
    segment_index = 0 if attachment.endpoint == "start" else plan.segment_count - 1
    elastic_force = evaluation.internal_forces[node_index]
    if plan.dimension == 2:
        if body_orientation.shape != ():
            raise ValueError("Planar rigid_orientation must be a scalar angle.")
        body_frame = _planar_rotation_matrix(body_orientation)
        lever = oe.contract("ij,j->i", body_frame, attachment.local_offset)
        elastic_moment = evaluation.internal_moments[segment_index]
        force_on_rod = -elastic_force
        moment_on_rod = -elastic_moment
        force_on_rigid = elastic_force
        moment_on_rigid = (
            elastic_moment + lever[0] * force_on_rigid[1] - lever[1] * force_on_rigid[0]
        )
        moment_on_rod_about_center = (
            moment_on_rod + lever[0] * force_on_rod[1] - lever[1] * force_on_rod[0]
        )
        orientation_valid = jnp.isfinite(body_orientation)
    else:
        if body_orientation.shape != (4,):
            raise ValueError("Spatial rigid_orientation must be a quaternion.")
        body_quaternion, body_norm = _safe_unit_quaternion(body_orientation)
        body_frame = _quaternion_rotation_matrix(body_quaternion)
        lever = oe.contract("ij,j->i", body_frame, attachment.local_offset)
        rod_quaternion, _ = _safe_unit_quaternion(state.orientations[segment_index])
        rod_frame = _quaternion_rotation_matrix(rod_quaternion)
        elastic_moment = oe.contract(
            "ij,j->i", rod_frame, evaluation.internal_moments[segment_index]
        )
        force_on_rod = -elastic_force
        moment_on_rod = -elastic_moment
        force_on_rigid = elastic_force
        moment_on_rigid = elastic_moment + jnp.cross(lever, force_on_rigid)
        moment_on_rod_about_center = moment_on_rod + jnp.cross(lever, force_on_rod)
        orientation_valid = jnp.isfinite(body_norm) & (
            jnp.abs(body_norm - 1.0) <= plan.orientation_norm_tolerance
        )
    force_balance = force_on_rod + force_on_rigid
    moment_balance = moment_on_rod_about_center + moment_on_rigid
    finite = (
        jnp.all(jnp.isfinite(force_on_rod))
        & jnp.all(jnp.isfinite(moment_on_rod))
        & jnp.all(jnp.isfinite(force_on_rigid))
        & jnp.all(jnp.isfinite(moment_on_rigid))
        & jnp.all(jnp.isfinite(force_balance))
        & jnp.all(jnp.isfinite(moment_balance))
    )
    return RodEndpointResponse(
        state.positions[node_index],
        force_on_rod,
        moment_on_rod,
        force_on_rigid,
        moment_on_rigid,
        force_balance,
        moment_balance,
        finite,
        finite & orientation_valid & evaluation.valid,
        attachment.attachment_id,
        prepared.prepared_id,
    )


class RodDynamicsPlan(StrictModule, NonTrainableState):
    """Static bounds and projection work budget for symplectic rod dynamics."""

    integrator: RodIntegrator = eqx.field(static=True)
    maximum_time_step: float = eqx.field(static=True)
    maximum_nodal_displacement: float = eqx.field(static=True)
    maximum_angular_increment: float = eqx.field(static=True)
    projection_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        integrator: RodIntegrator = "symplectic",
        maximum_time_step: float = 1.0,
        maximum_nodal_displacement: float = 1.0e3,
        maximum_angular_increment: float = np.pi,
        projection_iterations: int = 8,
        plan_id: str | None = None,
    ):
        if integrator != "symplectic":
            raise ValueError("The fixed-topology rod supports the symplectic integrator.")
        time_bound = float(maximum_time_step)
        displacement_bound = float(maximum_nodal_displacement)
        angular_bound = float(maximum_angular_increment)
        iterations = int(projection_iterations)
        if (
            not isfinite(time_bound)
            or time_bound <= 0.0
            or not isfinite(displacement_bound)
            or displacement_bound <= 0.0
            or not isfinite(angular_bound)
            or angular_bound <= 0.0
            or iterations < 1
        ):
            raise ValueError("Rod dynamics bounds and projection budget are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "bounded-symplectic-cosserat-rod-dynamics",
                "integrator": integrator,
                "maximum_time_step": time_bound,
                "maximum_nodal_displacement": displacement_bound,
                "maximum_angular_increment": angular_bound,
                "projection_iterations": iterations,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.integrator = integrator
        self.maximum_time_step = time_bound
        self.maximum_nodal_displacement = displacement_bound
        self.maximum_angular_increment = angular_bound
        self.projection_iterations = iterations
        self.plan_id = identifier


class RodStepEvidence(StrictModule):
    finite: Array
    current_valid: Array
    candidate_valid: Array
    time_step_valid: Array
    displacement_bounded: Array
    rotation_bounded: Array
    inertia_solve_valid: Array
    successful: Array
    dynamics_id: str = eqx.field(static=True)


class RodStepResult(StrictModule):
    candidate_state: RodState
    accepted_state: RodState
    current_evaluation: RodEvaluation
    candidate_evaluation: RodEvaluation
    accepted_evaluation: RodEvaluation
    evidence: RodStepEvidence
    successful: Array
    dynamics_id: str = eqx.field(static=True)


def _project_inextensible_positions(
    rod: PreparedRod,
    positions: Array,
    iterations: int,
    /,
) -> Array:
    plan = rod.plan
    inverse_mass = 1.0 / plan.node_masses

    def project_segment(segment_index, current):
        left = plan.segment_node_ids[segment_index, 0]
        right = plan.segment_node_ids[segment_index, 1]
        vector = current[right] - current[left]
        length = jnp.sqrt(jnp.sum(vector * vector))
        safe_length = jnp.maximum(length, jnp.sqrt(jnp.finfo(current.dtype).tiny))
        correction = ((length - plan.rest_lengths[segment_index]) / safe_length) * vector
        weight_sum = inverse_mass[left] + inverse_mass[right]
        left_update = (inverse_mass[left] / weight_sum) * correction
        right_update = (inverse_mass[right] / weight_sum) * correction
        current = current.at[left].add(left_update)
        return current.at[right].add(-right_update)

    def project_sweep(_, current):
        return jax.lax.fori_loop(0, plan.segment_count, project_segment, current)

    return jax.lax.fori_loop(0, iterations, project_sweep, positions)


def _project_inextensible_velocities(
    rod: PreparedRod,
    positions: Array,
    velocities: Array,
    iterations: int,
    /,
) -> Array:
    plan = rod.plan
    inverse_mass = 1.0 / plan.node_masses

    def project_segment(segment_index, current):
        left = plan.segment_node_ids[segment_index, 0]
        right = plan.segment_node_ids[segment_index, 1]
        vector = positions[right] - positions[left]
        length = jnp.sqrt(jnp.sum(vector * vector))
        direction = vector / jnp.maximum(
            length, jnp.sqrt(jnp.finfo(positions.dtype).tiny)
        )
        relative_speed = jnp.sum((current[right] - current[left]) * direction)
        weight_sum = inverse_mass[left] + inverse_mass[right]
        impulse = relative_speed * direction
        current = current.at[left].add((inverse_mass[left] / weight_sum) * impulse)
        return current.at[right].add(-(inverse_mass[right] / weight_sum) * impulse)

    def project_sweep(_, current):
        return jax.lax.fori_loop(0, plan.segment_count, project_segment, current)

    return jax.lax.fori_loop(0, iterations, project_sweep, velocities)


def _integrate_quaternion(
    quaternion: Array, angular_velocity: Array, dt: Array, /
) -> Array:
    angular_norm = jnp.sqrt(jnp.sum(angular_velocity * angular_velocity, axis=-1))
    half_angle = 0.5 * dt * angular_norm
    threshold = jnp.sqrt(jnp.finfo(quaternion.dtype).eps)
    safe_norm = jnp.maximum(angular_norm, threshold)
    regular_scale = jnp.sin(half_angle) / safe_norm
    limiting_scale = 0.5 * dt * jnp.ones_like(regular_scale)
    scale = jnp.where(angular_norm > threshold, regular_scale, limiting_scale)
    increment = jnp.concatenate(
        (jnp.cos(half_angle)[:, None], scale[:, None] * angular_velocity), axis=-1
    )
    candidate = _quaternion_multiply(quaternion, increment)
    normalized, _ = _safe_unit_quaternion(candidate)
    return normalized


class PreparedRodDynamics(StrictModule, NonTrainableState):
    """Pure bounded symplectic update with static inextensibility projection work."""

    rod: PreparedRod
    plan: RodDynamicsPlan
    inertia_solve: SmallLinearSolvePlan | None
    prepared_id: str = eqx.field(static=True)

    def __init__(self, rod: PreparedRod, plan: RodDynamicsPlan, /):
        if not isinstance(rod, PreparedRod):
            raise TypeError("rod must be a PreparedRod.")
        if not isinstance(plan, RodDynamicsPlan):
            raise TypeError("plan must be a RodDynamicsPlan.")
        inertia_solve = SmallLinearSolvePlan(3) if rod.plan.dimension == 3 else None
        self.rod = rod
        self.plan = plan
        self.inertia_solve = inertia_solve
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-bounded-symplectic-cosserat-rod-dynamics",
                "rod": rod.prepared_id,
                "plan": plan.plan_id,
                "inertia_solve": None if inertia_solve is None else inertia_solve.plan_id,
            }
        )

    def step(
        self,
        state: RodState,
        dt: ArrayLike,
        /,
        *,
        external_forces: ArrayLike | None = None,
        external_moments: ArrayLike | None = None,
    ) -> RodStepResult:
        _validate_state_contract(self.rod, state)
        rod_plan = self.rod.plan
        time_step = jnp.asarray(dt, dtype=state.positions.dtype)
        if time_step.shape != ():
            raise ValueError("dt must be a scalar.")
        applied_forces = (
            jnp.zeros_like(state.positions)
            if external_forces is None
            else jnp.asarray(external_forces, dtype=state.positions.dtype)
        )
        applied_moments = (
            jnp.zeros_like(state.angular_velocities)
            if external_moments is None
            else jnp.asarray(external_moments, dtype=state.angular_velocities.dtype)
        )
        if applied_forces.shape != state.positions.shape:
            raise ValueError("external_forces must match rod positions.")
        if applied_moments.shape != state.angular_velocities.shape:
            raise ValueError("external_moments must match rod angular velocities.")
        current_evaluation = evaluate_rod(self.rod, state)
        force = current_evaluation.internal_forces + applied_forces
        velocity = state.velocities + time_step * force / rod_plan.node_masses[:, None]
        position = state.positions + time_step * velocity
        moment = current_evaluation.internal_moments + applied_moments
        if rod_plan.dimension == 2:
            angular_velocity = state.angular_velocities + (
                time_step * moment / rod_plan.segment_inertias
            )
            orientation = state.orientations + time_step * angular_velocity
            inertia_valid = jnp.asarray(True)
            angular_increment = jnp.max(jnp.abs(time_step * angular_velocity))
        else:
            angular_momentum = oe.contract(
                "sij,sj->si", rod_plan.segment_inertias, state.angular_velocities
            )
            gyroscopic = jnp.cross(state.angular_velocities, angular_momentum)
            inertia_result = solve_small_linear(
                self.inertia_solve,
                rod_plan.segment_inertias,
                moment - gyroscopic,
            )
            angular_velocity = state.angular_velocities + time_step * inertia_result.value
            orientation, _ = _safe_unit_quaternion(state.orientations)
            orientation = _integrate_quaternion(orientation, angular_velocity, time_step)
            inertia_valid = jnp.all(inertia_result.successful)
            angular_increment = jnp.max(
                jnp.abs(time_step)
                * jnp.sqrt(jnp.sum(angular_velocity * angular_velocity, axis=-1))
            )
        if rod_plan.inextensible:
            position = _project_inextensible_positions(
                self.rod, position, self.plan.projection_iterations
            )
            velocity = _project_inextensible_velocities(
                self.rod, position, velocity, self.plan.projection_iterations
            )
        candidate_state = RodState(position, velocity, orientation, angular_velocity)
        candidate_evaluation = evaluate_rod(self.rod, candidate_state)
        nodal_displacement = jnp.max(
            jnp.sqrt(jnp.sum((position - state.positions) ** 2, axis=-1))
        )
        time_step_valid = (
            jnp.isfinite(time_step)
            & (time_step > 0.0)
            & (time_step <= self.plan.maximum_time_step)
        )
        displacement_bounded = jnp.isfinite(nodal_displacement) & (
            nodal_displacement <= self.plan.maximum_nodal_displacement
        )
        rotation_bounded = jnp.isfinite(angular_increment) & (
            angular_increment <= self.plan.maximum_angular_increment
        )
        finite = (
            candidate_evaluation.finite
            & jnp.all(jnp.isfinite(applied_forces))
            & jnp.all(jnp.isfinite(applied_moments))
        )
        successful = (
            current_evaluation.valid
            & candidate_evaluation.valid
            & finite
            & time_step_valid
            & displacement_bounded
            & rotation_bounded
            & inertia_valid
        )
        accepted_state = RodState(
            jnp.where(successful, candidate_state.positions, state.positions),
            jnp.where(successful, candidate_state.velocities, state.velocities),
            jnp.where(successful, candidate_state.orientations, state.orientations),
            jnp.where(
                successful,
                candidate_state.angular_velocities,
                state.angular_velocities,
            ),
        )
        accepted_evaluation = evaluate_rod(self.rod, accepted_state)
        evidence = RodStepEvidence(
            finite,
            current_evaluation.valid,
            candidate_evaluation.valid,
            time_step_valid,
            displacement_bounded,
            rotation_bounded,
            inertia_valid,
            successful,
            self.prepared_id,
        )
        return RodStepResult(
            candidate_state,
            accepted_state,
            current_evaluation,
            candidate_evaluation,
            accepted_evaluation,
            evidence,
            successful,
            self.prepared_id,
        )


def prepare_rod_dynamics(
    rod: PreparedRod,
    plan: RodDynamicsPlan | None = None,
    /,
) -> PreparedRodDynamics:
    """Prepare a bounded symplectic rod update with a static work budget."""
    return PreparedRodDynamics(rod, RodDynamicsPlan() if plan is None else plan)


__all__ = [
    "PreparedRod",
    "PreparedRodDynamics",
    "RodDimension",
    "RodDynamicsPlan",
    "RodEndpoint",
    "RodEndpointAttachment",
    "RodEndpointResponse",
    "RodEvaluation",
    "RodIntegrator",
    "RodPlan",
    "RodState",
    "RodStepEvidence",
    "RodStepResult",
    "evaluate_endpoint_attachment",
    "evaluate_rod",
    "prepare_rod",
    "prepare_rod_dynamics",
    "rod_potential_energy",
]

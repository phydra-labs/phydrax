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
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    dual_transpose,
    FunctionLinearOperator,
)
from ...metrix import QuaternionPoseStateGeometry
from ._rod_reduction import (
    PreparedReducedRod,
    ReducedRodEvaluation,
    ReducedRodState,
)


RodReconstructionMethod: TypeAlias = Literal["auto", "pcs", "gvs"]

_POSE_GEOMETRY = QuaternionPoseStateGeometry(convention="body", tolerance=1.0e-9)
_FRAME_CONVENTION = "material-body/world-origin/world-frame"
_OBSERVATIONAL_ROLE = "observational/reference; native-discrete lift is authoritative"


def _real_vector(name: str, value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 array.")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a real floating dtype.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_finite(name: str, value: float, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _method(value: RodReconstructionMethod, /) -> RodReconstructionMethod:
    if value not in ("auto", "pcs", "gvs"):
        raise ValueError("method must be 'auto', 'pcs', or 'gvs'.")
    return value


def _unit_quaternion(value: ArrayLike, name: str, /) -> Array:
    quaternion = jnp.asarray(value)
    if quaternion.shape != (4,):
        raise ValueError(f"{name} must have shape (4,).")
    norm = jnp.linalg.norm(quaternion)
    quaternion = eqx.error_if(
        quaternion,
        (~jnp.isfinite(norm)) | (norm <= jnp.finfo(quaternion.dtype).tiny),
        f"{name} must be finite and nonzero.",
    )
    return quaternion / norm


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


def _quaternion_rotation_vector(quaternion: Array, /) -> Array:
    canonical = jnp.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = canonical[..., 1:]
    vector_norm = jnp.linalg.norm(vector, axis=-1)
    safe_norm = jnp.where(vector_norm < 1.0e-7, 1.0, vector_norm)
    angle = 2.0 * jnp.arctan2(vector_norm, canonical[..., 0])
    scale = jnp.where(
        vector_norm < 1.0e-7,
        2.0 + vector_norm * vector_norm / 3.0,
        angle / safe_norm,
    )
    return scale[..., None] * vector


def _quaternion_angle(left: Array, right: Array, /) -> Array:
    relative = _quaternion_multiply(_quaternion_conjugate(left), right)
    return jnp.linalg.norm(_quaternion_rotation_vector(relative), axis=-1)


def _rotate(quaternion: Array, vector: Array, /) -> Array:
    imaginary = quaternion[..., 1:]
    doubled_cross = 2.0 * jnp.cross(imaginary, vector)
    return (
        vector + quaternion[..., :1] * doubled_cross + jnp.cross(imaginary, doubled_cross)
    )


def _maximum_norm(value: Array, /) -> Array:
    if value.shape[0] == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.linalg.norm(value, axis=-1))


def _maximum_absolute(value: Array, dtype, /) -> Array:
    if value.size == 0:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.max(jnp.abs(value))


def _routes(points: np.ndarray, breakpoints: np.ndarray, /) -> np.ndarray:
    route = np.searchsorted(breakpoints, points, side="right") - 1
    return np.where(points == breakpoints[-1], breakpoints.size - 2, route).astype(
        np.int32
    )


def _indices_in_union(points: np.ndarray, union: np.ndarray, name: str, /) -> np.ndarray:
    indices = np.searchsorted(union, points)
    if np.any(indices >= union.size) or not np.array_equal(union[indices], points):
        raise RuntimeError(f"{name} were not retained in the prepared arc-length union.")
    return indices.astype(np.int32)


class RodFrameQueryPlan(StrictModule, NonTrainableState):
    """Immutable physical arc-length frame queries for one reconstruction."""

    arc_lengths: Array
    query_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, arc_lengths: ArrayLike, /):
        points = _real_vector("arc_lengths", arc_lengths)
        if points.size < 1:
            raise ValueError("arc_lengths must contain at least one physical query.")
        self.arc_lengths = jnp.asarray(points)
        self.query_count = int(points.size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rod-frame-physical-arc-length-query-plan",
                "arc_lengths": array_tree_fingerprint(points),
            }
        )


class RodReconstructionPlan(StrictModule, NonTrainableState):
    """Fixed-work observational PCS/GVS reconstruction policy."""

    queries: RodFrameQueryPlan
    method: RodReconstructionMethod = eqx.field(static=True)
    refinement: int = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    chart_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        queries: RodFrameQueryPlan,
        /,
        *,
        method: RodReconstructionMethod = "auto",
        refinement: int = 1,
        quadrature_tolerance: float = 1.0e-6,
        chart_margin: float = 1.0e-5,
    ):
        if not isinstance(queries, RodFrameQueryPlan):
            raise TypeError("queries must be a RodFrameQueryPlan.")
        method_ = _method(method)
        refinement_ = int(refinement)
        if refinement_ < 1 or refinement_ != refinement:
            raise ValueError("refinement must be a positive integer.")
        tolerance = _positive_finite("quadrature_tolerance", quadrature_tolerance)
        margin = _positive_finite("chart_margin", chart_margin)
        if margin >= np.pi:
            raise ValueError("chart_margin must be smaller than pi.")
        self.queries = queries
        self.method = method_
        self.refinement = refinement_
        self.quadrature_tolerance = tolerance
        self.chart_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-work-reduced-rod-reconstruction-plan",
                "queries": queries.plan_id,
                "method": method_,
                "refinement": refinement_,
                "quadrature_tolerance": tolerance,
                "chart_margin": margin,
            }
        )


class RodReconstructionDomainEvidence(StrictModule):
    """Deterministic half-open query routing with closed final endpoint."""

    arc_lengths: Array
    route_indices: Array
    in_domain: Array
    half_open_routing: Array
    final_endpoint_closed: Array
    valid: Array


class RodReconstructionChartEvidence(StrictModule):
    """Rotation-increment chart margin over the fixed integration workset."""

    maximum_increment_angle: Array
    minimum_chart_margin: Array
    finite: Array
    valid: Array


class RodReconstructionQuadratureEvidence(StrictModule):
    """Fixed-work PCS exactness or embedded CF4/Zanna local discrepancy."""

    maximum_scaled_local_error: Array
    tolerance: Array
    panel_count: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    finite: Array
    valid: Array


class RodNativeDiscretizationDiscrepancy(StrictModule):
    """Observational reconstruction discrepancy from the authoritative native lift."""

    maximum_node_position_error: Array
    maximum_frame_orientation_error: Array
    maximum_stretch_shear_error: Array
    maximum_bend_twist_error: Array
    maximum_node_velocity_error: Array
    maximum_frame_angular_velocity_error: Array
    continuous_strain_energy: Array
    native_discrete_energy: Array
    native_energy_error: Array
    finite: Array


class RodReconstructionEvaluation(StrictModule):
    """Observational continuum samples; native-discrete mechanics remains authoritative."""

    arc_lengths: Array
    poses: Array
    positions: Array
    orientations: Array
    strains: Array
    reduced_strains: Array
    body_twists: Array
    world_origin_velocities: Array
    frame_velocities: Array
    native_evaluation: ReducedRodEvaluation
    domain_evidence: RodReconstructionDomainEvidence
    chart_evidence: RodReconstructionChartEvidence
    quadrature_evidence: RodReconstructionQuadratureEvidence
    native_discrepancy: RodNativeDiscretizationDiscrepancy
    finite: Array
    valid: Array
    characteristic_length: float = eqx.field(static=True)
    maximum_panel_length: float = eqx.field(static=True)
    refinement: int = eqx.field(static=True)
    frame_convention: str = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    query_plan_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)


class RodDiscretizationErrors(StrictModule):
    """Adjacent-resolution errors at identical physical frame queries."""

    scaled_se3_log: Array
    scaled_strain: Array
    scaled_body_twist: Array
    scaled_frame_jvp: Array
    native_energy: Array

    @property
    def maximum_scaled_se3_log(self) -> Array:
        return jnp.max(self.scaled_se3_log)

    @property
    def maximum_scaled_strain(self) -> Array:
        return jnp.max(self.scaled_strain)

    @property
    def maximum_scaled_body_twist(self) -> Array:
        return jnp.max(self.scaled_body_twist)

    @property
    def maximum_scaled_frame_jvp(self) -> Array:
        return jnp.max(self.scaled_frame_jvp)


class RodObservedOrder(StrictModule):
    scaled_se3_log: Array
    scaled_strain: Array
    scaled_body_twist: Array
    scaled_frame_jvp: Array
    native_energy: Array


class RodDiscretizationComparisonEvidence(StrictModule):
    query_plan_matches: Array
    physical_queries_match: Array
    reduction_matches: Array
    frame_matches: Array
    route_matches: Array
    observed_order_supported: Array
    refinement_valid: Array
    finite: Array
    valid: Array


class RodDiscretizationComparison(StrictModule):
    coarse_medium: RodDiscretizationErrors
    medium_fine: RodDiscretizationErrors | None
    observed_order: RodObservedOrder
    evidence: RodDiscretizationComparisonEvidence
    comparison_id: str = eqx.field(static=True)


class PreparedRodReconstruction(StrictModule, NonTrainableState):
    """One fixed physical-query PCS/GVS view of a fixed-base reduced rod.

    This is an observational/reference reconstruction. It never replaces the
    native-discrete lift as the mechanics or energy authority.
    """

    reduced: PreparedReducedRod
    plan: RodReconstructionPlan
    union_arc_lengths: Array
    integration_arc_lengths: Array
    node_arc_lengths: Array
    stretch_arc_lengths: Array
    bend_arc_lengths: Array
    query_union_indices: Array
    node_union_indices: Array
    stretch_union_indices: Array
    bend_union_indices: Array
    query_route_indices: Array
    frame_space: ArraySpace
    method: str = eqx.field(static=True)
    panel_count: int = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    maximum_panel_length: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduced: PreparedReducedRod,
        plan: RodReconstructionPlan,
        /,
    ):
        if not isinstance(reduced, PreparedReducedRod):
            raise TypeError("reduced must be a PreparedReducedRod.")
        if not isinstance(plan, RodReconstructionPlan):
            raise TypeError("plan must be a RodReconstructionPlan.")
        if reduced.rod.plan.dimension != 3:
            raise ValueError("Rod reconstruction requires a spatial PreparedReducedRod.")
        if reduced.plan.base_policy not in ("reference", "fixed"):
            raise ValueError("Rod reconstruction supports fixed-base reductions only.")

        basis = reduced.basis
        breakpoints = np.asarray(basis.breakpoints)
        if breakpoints.ndim != 1 or breakpoints.size < 2:
            raise ValueError("The prepared strain basis must expose finite breakpoints.")
        if not np.all(np.isfinite(breakpoints)) or np.any(np.diff(breakpoints) <= 0.0):
            raise ValueError("Strain-basis breakpoints must be finite and increasing.")
        domain_start = float(basis.domain_start)
        domain_end = float(basis.domain_end)
        rod_length = float(np.sum(np.asarray(reduced.rod.plan.rest_lengths)))
        scale = max(1.0, rod_length)
        tolerance = 500.0 * np.finfo(breakpoints.dtype).eps * scale
        if (
            abs(domain_start) > tolerance
            or abs(domain_end - rod_length) > tolerance
            or abs(float(breakpoints[0]) - domain_start) > tolerance
            or abs(float(breakpoints[-1]) - domain_end) > tolerance
        ):
            raise ValueError(
                "The strain-basis physical domain must equal the native rod arc-length domain."
            )

        queries = np.asarray(plan.queries.arc_lengths, dtype=breakpoints.dtype)
        in_domain = (queries >= domain_start) & (queries <= domain_end)
        if not np.all(in_domain):
            raise ValueError(
                "Every physical arc-length query must lie in the rod domain."
            )
        stretch_points = np.asarray(basis.stretch_arc_lengths, dtype=breakpoints.dtype)
        bend_points = np.asarray(basis.bend_arc_lengths, dtype=breakpoints.dtype)
        if (
            stretch_points.shape != (reduced.rod.plan.segment_count,)
            or bend_points.shape != (reduced.rod.plan.segment_count - 1,)
            or np.any(stretch_points < domain_start)
            or np.any(stretch_points > domain_end)
            or np.any(bend_points < domain_start)
            or np.any(bend_points > domain_end)
        ):
            raise ValueError(
                "Prepared strain-basis native sample points are incompatible."
            )
        node_points = np.concatenate(
            (
                np.asarray((domain_start,), dtype=breakpoints.dtype),
                np.cumsum(np.asarray(reduced.rod.plan.rest_lengths)),
            )
        )
        union = np.unique(
            np.concatenate(
                (breakpoints, node_points, stretch_points, bend_points, queries)
            )
        )
        integration_values = [float(union[0])]
        for left, right in zip(union[:-1], union[1:], strict=True):
            integration_values.extend(
                np.linspace(
                    left,
                    right,
                    plan.refinement + 1,
                    dtype=breakpoints.dtype,
                )[1:].tolist()
            )
        integration = np.asarray(integration_values, dtype=breakpoints.dtype)
        maximum_panel_length = float(np.max(np.diff(integration)))
        query_routes = _routes(queries, breakpoints)
        route_id = canonical_fingerprint(
            {
                "kind": "rod-reconstruction-half-open-query-routes",
                "basis": basis.basis_id,
                "queries": array_tree_fingerprint(queries),
                "breakpoints": array_tree_fingerprint(breakpoints),
                "routes": array_tree_fingerprint(query_routes),
                "final_endpoint": "closed",
            }
        )
        resolved_method = (
            "pcs"
            if plan.method == "auto" and basis.method == "piecewise_constant"
            else "gvs"
            if plan.method == "auto"
            else plan.method
        )
        if resolved_method == "pcs" and basis.method != "piecewise_constant":
            raise ValueError("PCS reconstruction requires a piecewise-constant basis.")
        dtype = np.dtype(reduced.rod.plan.rest_positions.dtype)
        frame_space = ArraySpace(
            (plan.queries.query_count, 6),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "rod-reconstruction-frame-velocity-space",
                    "queries": plan.queries.plan_id,
                    "reduction": reduced.prepared_id,
                    "frame": _FRAME_CONVENTION,
                }
            ),
        )
        reconstruction_id = canonical_fingerprint(
            {
                "kind": "prepared-observational-reduced-rod-reconstruction",
                "reduction": reduced.prepared_id,
                "basis": basis.basis_id,
                "plan": plan.plan_id,
                "routes": route_id,
                "method": resolved_method,
                "union": array_tree_fingerprint(union),
                "integration": array_tree_fingerprint(integration),
            }
        )

        self.reduced = reduced
        self.plan = plan
        self.union_arc_lengths = jnp.asarray(union)
        self.integration_arc_lengths = jnp.asarray(integration)
        self.node_arc_lengths = jnp.asarray(node_points)
        self.stretch_arc_lengths = jnp.asarray(stretch_points)
        self.bend_arc_lengths = jnp.asarray(bend_points)
        self.query_union_indices = jnp.asarray(
            _indices_in_union(queries, integration, "Query points")
        )
        self.node_union_indices = jnp.asarray(
            _indices_in_union(node_points, integration, "Native nodes")
        )
        self.stretch_union_indices = jnp.asarray(
            _indices_in_union(stretch_points, integration, "Stretch/shear sites")
        )
        self.bend_union_indices = jnp.asarray(
            _indices_in_union(bend_points, integration, "Bend/twist sites")
        )
        self.query_route_indices = jnp.asarray(query_routes)
        self.frame_space = frame_space
        self.method = resolved_method
        self.panel_count = int(integration.size - 1)
        self.characteristic_length = rod_length
        self.maximum_panel_length = maximum_panel_length
        self.route_id = route_id
        self.reconstruction_id = reconstruction_id

    def pose(self, coefficients: ArrayLike, /) -> Array:
        values = self.reduced.coefficient_space.validate(coefficients)
        poses, _, _, valid = _integrate(self, values)
        queried = poses[self.query_union_indices]
        return eqx.error_if(
            queried,
            ~valid,
            "Rod pose reconstruction failed chart or fixed-quadrature evidence.",
        )

    def strain(self, coefficients: ArrayLike, /) -> Array:
        values = self.reduced.coefficient_space.validate(coefficients)
        result = _total_strain(self, values, self.plan.queries.arc_lengths)
        return eqx.error_if(
            result,
            ~jnp.all(jnp.isfinite(result)),
            "Rod strain reconstruction produced nonfinite values.",
        )

    def body_twist(self, state: ReducedRodState, /) -> Array:
        coefficients, rates = _validated_state(self, state)
        poses, body_twists = _pose_body_jvp(self, coefficients, rates)
        del poses
        return body_twists[self.query_union_indices]

    def world_velocity(self, state: ReducedRodState, /) -> Array:
        coefficients, rates = _validated_state(self, state)
        poses, body_twists = _pose_body_jvp(self, coefficients, rates)
        world, _ = _world_and_frame_velocities(poses, body_twists)
        return world[self.query_union_indices]

    def frame_velocity_operator(
        self, coefficients: ArrayLike, /
    ) -> AbstractLinearOperator:
        values = self.reduced.coefficient_space.validate(coefficients)

        def action(rates):
            poses, body_twists = _pose_body_jvp(self, values, rates)
            _, frame = _world_and_frame_velocities(poses, body_twists)
            return frame[self.query_union_indices]

        def transpose_action(efforts):
            return jax.linear_transpose(action, jnp.zeros_like(values))(efforts)[0]

        return FunctionLinearOperator(
            action,
            source=self.reduced.coefficient_space,
            target=self.frame_space,
            transpose_action=transpose_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "rod-reconstruction-frame-velocity-jvp",
                    "reconstruction": self.reconstruction_id,
                }
            ),
        )

    def frame_effort_pullback(self, coefficients: ArrayLike, /) -> AbstractLinearOperator:
        """Map world-frame wrenches at query-frame origins to reduced duals."""
        return dual_transpose(self.frame_velocity_operator(coefficients))

    def evaluate(self, state: ReducedRodState, /) -> RodReconstructionEvaluation:
        return evaluate_rod_reconstruction(self, state)


def _validated_state(
    prepared: PreparedRodReconstruction,
    state: ReducedRodState,
    /,
) -> tuple[Array, Array]:
    if not isinstance(state, ReducedRodState):
        raise TypeError("state must be a ReducedRodState.")
    coefficients = prepared.reduced.coefficient_space.validate(state.coefficients)
    rates = prepared.reduced.coefficient_space.validate(state.coefficient_velocities)
    return coefficients, rates


def _reference_strain(
    prepared: PreparedRodReconstruction, arc_lengths: Array, /
) -> Array:
    points = jnp.asarray(arc_lengths)
    flattened = points.reshape((-1,))
    rod = prepared.reduced.rod
    node_points = prepared.node_arc_lengths
    segment_routes = jnp.searchsorted(node_points, flattened, side="right") - 1
    segment_routes = jnp.where(
        flattened == node_points[-1], rod.plan.segment_count - 1, segment_routes
    )
    segment_routes = jnp.clip(segment_routes, 0, rod.plan.segment_count - 1)
    stretch = rod.rest_stretch_shear[segment_routes]

    if rod.plan.segment_count == 1:
        bend = jnp.zeros((flattened.shape[0], 3), dtype=stretch.dtype)
    else:
        bend_routes = (
            jnp.searchsorted(prepared.stretch_arc_lengths, flattened, side="right") - 1
        )
        in_bend_support = (bend_routes >= 0) & (bend_routes < rod.plan.segment_count - 1)
        safe_routes = jnp.clip(bend_routes, 0, rod.plan.segment_count - 2)
        rest_rotation = _quaternion_rotation_vector(rod.rest_relative_orientations)
        routed = rest_rotation[safe_routes] / rod.dual_lengths[safe_routes, None]
        bend = jnp.where(in_bend_support[:, None], routed, 0.0)
    result = jnp.concatenate((stretch, bend), axis=-1)
    return result.reshape(points.shape + (6,))


def _total_strain(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    arc_lengths: Array,
    /,
) -> Array:
    reduced = prepared.reduced.basis.strain(coefficients, arc_lengths)
    return _reference_strain(prepared, arc_lengths) + reduced


def _base_pose(prepared: PreparedRodReconstruction, /) -> Array:
    quaternion = _unit_quaternion(
        prepared.reduced.base_orientation, "Reduced-rod base orientation"
    )
    position = jnp.asarray(prepared.reduced.base_position)
    if position.shape != (3,):
        raise ValueError("Reduced-rod base position must have shape (3,).")
    return jnp.concatenate((quaternion, position))


def _pcs_step(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    pose: Array,
    start: Array,
    length: Array,
    /,
) -> tuple[Array, Array]:
    strain = _total_strain(prepared, coefficients, start + 0.5 * length)
    increment = length * strain
    return _POSE_GEOMETRY.retract(pose, increment), jnp.linalg.norm(increment[3:])


def _cf4_step(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    pose: Array,
    start: Array,
    length: Array,
    /,
) -> tuple[Array, Array]:
    root_three = jnp.sqrt(jnp.asarray(3.0, dtype=length.dtype))
    first_node = start + length * (0.5 - root_three / 6.0)
    second_node = start + length * (0.5 + root_three / 6.0)
    first_strain = _total_strain(prepared, coefficients, first_node)
    second_strain = _total_strain(prepared, coefficients, second_node)
    first_weight = (3.0 - 2.0 * root_three) / 12.0
    second_weight = (3.0 + 2.0 * root_three) / 12.0
    early_increment = length * (
        second_weight * first_strain + first_weight * second_strain
    )
    late_increment = length * (
        first_weight * first_strain + second_weight * second_strain
    )
    intermediate = _POSE_GEOMETRY.retract(pose, early_increment)
    result = _POSE_GEOMETRY.retract(intermediate, late_increment)
    maximum_angle = jnp.maximum(
        jnp.linalg.norm(early_increment[3:]),
        jnp.linalg.norm(late_increment[3:]),
    )
    return result, maximum_angle


def _integrate(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    starts = prepared.integration_arc_lengths[:-1]
    lengths = jnp.diff(prepared.integration_arc_lengths)
    length_scale = jnp.asarray(
        (prepared.characteristic_length,) * 3 + (1.0,) * 3,
        dtype=coefficients.dtype,
    )

    if prepared.method == "pcs":

        def panel_step(pose, panel):
            start, length = panel
            next_pose, angle = _pcs_step(prepared, coefficients, pose, start, length)
            return next_pose, (
                next_pose,
                jnp.asarray(0.0, dtype=coefficients.dtype),
                angle,
            )

    else:

        def panel_step(pose, panel):
            start, length = panel
            next_pose, full_angle = _cf4_step(prepared, coefficients, pose, start, length)
            half_length = 0.5 * length
            midpoint_pose, first_half_angle = _cf4_step(
                prepared, coefficients, pose, start, half_length
            )
            refined_pose, second_half_angle = _cf4_step(
                prepared,
                coefficients,
                midpoint_pose,
                start + half_length,
                half_length,
            )
            local_log = _POSE_GEOMETRY.inverse_retract(next_pose, refined_pose)
            local_error = jnp.linalg.norm(local_log / length_scale)
            maximum_angle = jnp.maximum(
                full_angle, jnp.maximum(first_half_angle, second_half_angle)
            )
            return next_pose, (next_pose, local_error, maximum_angle)

    base = _base_pose(prepared)
    _, (endpoint_poses, local_errors, increment_angles) = jax.lax.scan(
        panel_step, base, (starts, lengths)
    )
    poses = jnp.concatenate((base[None, :], endpoint_poses), axis=0)
    maximum_error = jnp.max(local_errors)
    maximum_angle = jnp.max(increment_angles)
    finite = (
        jnp.all(jnp.isfinite(poses))
        & jnp.isfinite(maximum_error)
        & jnp.isfinite(maximum_angle)
    )
    chart_valid = maximum_angle < (jnp.pi - prepared.plan.chart_margin)
    quadrature_valid = (prepared.method == "pcs") | (
        maximum_error <= prepared.plan.quadrature_tolerance
    )
    return poses, maximum_error, maximum_angle, finite & chart_valid & quadrature_valid


def _pose_body_jvp(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    rates: Array,
    /,
) -> tuple[Array, Array]:
    def pose_function(values):
        poses, _, _, valid = _integrate(prepared, values)
        return eqx.error_if(
            poses,
            ~valid,
            "Rod pose reconstruction failed chart or fixed-quadrature evidence.",
        )

    poses, ambient_tangents = jax.jvp(pose_function, (coefficients,), (rates,))
    body_twists = jax.vmap(_POSE_GEOMETRY.project_tangent)(poses, ambient_tangents)
    return poses, body_twists


def _world_and_frame_velocities(
    poses: Array,
    body_twists: Array,
    /,
) -> tuple[Array, Array]:
    quaternions = poses[:, :4]
    positions = poses[:, 4:]
    frame_linear = jax.vmap(_rotate)(quaternions, body_twists[:, :3])
    world_angular = jax.vmap(_rotate)(quaternions, body_twists[:, 3:])
    world_origin_linear = frame_linear + jnp.cross(positions, world_angular)
    world_origin = jnp.concatenate((world_origin_linear, world_angular), axis=-1)
    frame = jnp.concatenate((frame_linear, world_angular), axis=-1)
    return world_origin, frame


def _domain_evidence(
    prepared: PreparedRodReconstruction, /
) -> RodReconstructionDomainEvidence:
    queries = prepared.plan.queries.arc_lengths
    start = prepared.reduced.basis.domain_start
    end = prepared.reduced.basis.domain_end
    in_domain = (queries >= start) & (queries <= end) & jnp.isfinite(queries)
    routes = prepared.query_route_indices
    breakpoints = prepared.reduced.basis.breakpoints
    expected = jnp.searchsorted(breakpoints, queries, side="right") - 1
    expected = jnp.where(queries == end, breakpoints.shape[0] - 2, expected)
    half_open = jnp.all(routes == expected)
    final_closed = jnp.all(
        jnp.where(queries == end, routes == breakpoints.shape[0] - 2, True)
    )
    valid = jnp.all(in_domain) & half_open & final_closed
    return RodReconstructionDomainEvidence(
        queries, routes, in_domain, half_open, final_closed, valid
    )


def _continuous_strain_energy(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    /,
) -> Array:
    basis = prepared.reduced.basis
    points = basis.quadrature_arc_lengths
    weights = basis.quadrature_weights
    increments = basis.strain(coefficients, points)
    rod = prepared.reduced.rod
    segment_routes = jnp.searchsorted(prepared.node_arc_lengths, points, side="right") - 1
    segment_routes = jnp.where(
        points == prepared.node_arc_lengths[-1],
        rod.plan.segment_count - 1,
        segment_routes,
    )
    segment_routes = jnp.clip(segment_routes, 0, rod.plan.segment_count - 1)
    stretch_stiffness = rod.plan.stretch_shear_stiffness[segment_routes]
    stretch_density = 0.5 * contract(
        "qi,qij,qj->q",
        increments[:, :3],
        stretch_stiffness,
        increments[:, :3],
    )
    if rod.plan.segment_count == 1:
        bend_density = jnp.zeros_like(stretch_density)
    else:
        bend_routes = (
            jnp.searchsorted(prepared.stretch_arc_lengths, points, side="right") - 1
        )
        in_support = (bend_routes >= 0) & (bend_routes < rod.plan.segment_count - 1)
        safe_routes = jnp.clip(bend_routes, 0, rod.plan.segment_count - 2)
        bend_stiffness = rod.plan.bend_twist_stiffness[safe_routes]
        routed_density = 0.5 * contract(
            "qi,qij,qj->q",
            increments[:, 3:],
            bend_stiffness,
            increments[:, 3:],
        )
        bend_density = jnp.where(in_support, routed_density, 0.0)
    return jnp.sum(weights * (stretch_density + bend_density))


def _native_discrepancy(
    prepared: PreparedRodReconstruction,
    coefficients: Array,
    all_poses: Array,
    all_body_twists: Array,
    all_frame_velocities: Array,
    native_evaluation,
    /,
) -> RodNativeDiscretizationDiscrepancy:
    native_state = native_evaluation.native_state
    rod_evaluation = native_evaluation.native_evaluation
    path_nodes = prepared.reduced.path_node_ids
    node_position_error = _maximum_norm(
        all_poses[prepared.node_union_indices, 4:] - native_state.positions[path_nodes]
    )
    frame_orientation_error = jnp.max(
        _quaternion_angle(
            all_poses[prepared.stretch_union_indices, :4],
            native_state.orientations,
        )
    )
    stretch_increment = prepared.reduced.basis.strain(
        coefficients, prepared.stretch_arc_lengths
    )[:, :3]
    bend_increment = prepared.reduced.basis.strain(
        coefficients, prepared.bend_arc_lengths
    )[:, 3:]
    stretch_error = _maximum_absolute(
        stretch_increment - rod_evaluation.stretch_shear_strain,
        coefficients.dtype,
    )
    bend_error = _maximum_absolute(
        bend_increment - rod_evaluation.bend_twist_strain,
        coefficients.dtype,
    )
    node_velocity_error = _maximum_norm(
        all_frame_velocities[prepared.node_union_indices, :3]
        - native_state.velocities[path_nodes]
    )
    angular_velocity_error = _maximum_norm(
        all_body_twists[prepared.stretch_union_indices, 3:]
        - native_state.angular_velocities
    )
    continuous_energy = _continuous_strain_energy(prepared, coefficients)
    native_energy = native_evaluation.potential_energy
    energy_error = jnp.abs(continuous_energy - native_energy)
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    node_position_error,
                    frame_orientation_error,
                    stretch_error,
                    bend_error,
                    node_velocity_error,
                    angular_velocity_error,
                    continuous_energy,
                    native_energy,
                    energy_error,
                )
            )
        )
    )
    return RodNativeDiscretizationDiscrepancy(
        node_position_error,
        frame_orientation_error,
        stretch_error,
        bend_error,
        node_velocity_error,
        angular_velocity_error,
        continuous_energy,
        native_energy,
        energy_error,
        finite,
    )


def prepare_rod_reconstruction(
    reduced: PreparedReducedRod,
    plan: RodReconstructionPlan,
    /,
) -> PreparedRodReconstruction:
    """Bind fixed physical queries and fixed integration work to one reduction."""
    return PreparedRodReconstruction(reduced, plan)


def evaluate_rod_reconstruction(
    prepared: PreparedRodReconstruction,
    state: ReducedRodState,
    /,
) -> RodReconstructionEvaluation:
    """Evaluate an observational continuum reconstruction and native discrepancy."""
    if not isinstance(prepared, PreparedRodReconstruction):
        raise TypeError("prepared must be a PreparedRodReconstruction.")
    coefficients, rates = _validated_state(prepared, state)
    all_poses, quadrature_error, maximum_angle, reconstruction_valid = _integrate(
        prepared, coefficients
    )
    all_poses = eqx.error_if(
        all_poses,
        ~reconstruction_valid,
        "Rod reconstruction failed chart or fixed-quadrature evidence.",
    )
    _, all_body_twists = _pose_body_jvp(prepared, coefficients, rates)
    all_world_velocities, all_frame_velocities = _world_and_frame_velocities(
        all_poses, all_body_twists
    )
    query_poses = all_poses[prepared.query_union_indices]
    query_body = all_body_twists[prepared.query_union_indices]
    query_world = all_world_velocities[prepared.query_union_indices]
    query_frame = all_frame_velocities[prepared.query_union_indices]
    reduced_strains = prepared.reduced.basis.strain(
        coefficients, prepared.plan.queries.arc_lengths
    )
    strains = (
        _reference_strain(prepared, prepared.plan.queries.arc_lengths) + reduced_strains
    )
    native_evaluation = prepared.reduced.evaluate(state)
    discrepancy = _native_discrepancy(
        prepared,
        coefficients,
        all_poses,
        all_body_twists,
        all_frame_velocities,
        native_evaluation,
    )
    domain = _domain_evidence(prepared)
    chart_finite = jnp.isfinite(maximum_angle)
    chart_valid = chart_finite & (maximum_angle < (jnp.pi - prepared.plan.chart_margin))
    chart = RodReconstructionChartEvidence(
        maximum_angle,
        jnp.pi - prepared.plan.chart_margin - maximum_angle,
        chart_finite,
        chart_valid,
    )
    quadrature_finite = jnp.isfinite(quadrature_error)
    quadrature_valid = quadrature_finite & (
        (prepared.method == "pcs")
        | (quadrature_error <= prepared.plan.quadrature_tolerance)
    )
    quadrature = RodReconstructionQuadratureEvidence(
        quadrature_error,
        jnp.asarray(prepared.plan.quadrature_tolerance, dtype=quadrature_error.dtype),
        prepared.panel_count,
        prepared.method,
        quadrature_finite,
        quadrature_valid,
    )
    finite = (
        jnp.all(jnp.isfinite(query_poses))
        & jnp.all(jnp.isfinite(strains))
        & jnp.all(jnp.isfinite(reduced_strains))
        & jnp.all(jnp.isfinite(query_body))
        & jnp.all(jnp.isfinite(query_world))
        & jnp.all(jnp.isfinite(query_frame))
        & discrepancy.finite
    )
    valid = (
        finite & domain.valid & chart.valid & quadrature.valid & native_evaluation.valid
    )
    return RodReconstructionEvaluation(
        prepared.plan.queries.arc_lengths,
        query_poses,
        query_poses[:, 4:],
        query_poses[:, :4],
        strains,
        reduced_strains,
        query_body,
        query_world,
        query_frame,
        native_evaluation,
        domain,
        chart,
        quadrature,
        discrepancy,
        finite,
        valid,
        prepared.characteristic_length,
        prepared.maximum_panel_length,
        prepared.plan.refinement,
        _FRAME_CONVENTION,
        _OBSERVATIONAL_ROLE,
        prepared.plan.queries.plan_id,
        prepared.route_id,
        prepared.reduced.prepared_id,
        prepared.reconstruction_id,
    )


def _comparison_errors(
    left: RodReconstructionEvaluation,
    right: RodReconstructionEvaluation,
    /,
) -> RodDiscretizationErrors:
    length_scale = jnp.asarray(
        (left.characteristic_length,) * 3 + (1.0,) * 3,
        dtype=left.poses.dtype,
    )
    pose_logs = jax.vmap(_POSE_GEOMETRY.inverse_retract)(left.poses, right.poses)
    pose_error = jnp.linalg.norm(pose_logs / length_scale, axis=-1)
    strain_scale = jnp.asarray(
        (1.0,) * 3 + (left.characteristic_length,) * 3,
        dtype=left.strains.dtype,
    )
    strain_error = jnp.linalg.norm((left.strains - right.strains) * strain_scale, axis=-1)
    twist_error = jnp.linalg.norm(
        (left.body_twists - right.body_twists) / length_scale, axis=-1
    )
    jvp_error = jnp.linalg.norm(
        (left.frame_velocities - right.frame_velocities) / length_scale,
        axis=-1,
    )
    left_energy = left.native_evaluation.potential_energy
    right_energy = right.native_evaluation.potential_energy
    energy_error = jnp.abs(left_energy - right_energy) / jnp.maximum(
        jnp.asarray(1.0, dtype=left_energy.dtype), jnp.abs(right_energy)
    )
    return RodDiscretizationErrors(
        pose_error, strain_error, twist_error, jvp_error, energy_error
    )


def _order(first: Array, second: Array, ratio: Array, /) -> Array:
    valid = (first > 0.0) & (second > 0.0) & (ratio > 1.0)
    safe_first = jnp.where(valid, first, 1.0)
    safe_second = jnp.where(valid, second, 1.0)
    value = jnp.log(safe_first / safe_second) / jnp.log(jnp.where(valid, ratio, 2.0))
    return jnp.where(valid, value, jnp.nan)


def _observed_order(
    first: RodDiscretizationErrors,
    second: RodDiscretizationErrors,
    ratio: Array,
    /,
) -> RodObservedOrder:
    return RodObservedOrder(
        _order(
            first.maximum_scaled_se3_log,
            second.maximum_scaled_se3_log,
            ratio,
        ),
        _order(
            first.maximum_scaled_strain,
            second.maximum_scaled_strain,
            ratio,
        ),
        _order(
            first.maximum_scaled_body_twist,
            second.maximum_scaled_body_twist,
            ratio,
        ),
        _order(
            first.maximum_scaled_frame_jvp,
            second.maximum_scaled_frame_jvp,
            ratio,
        ),
        _order(first.native_energy, second.native_energy, ratio),
    )


def _nan_order(dtype, /) -> RodObservedOrder:
    value = jnp.asarray(jnp.nan, dtype=dtype)
    return RodObservedOrder(value, value, value, value, value)


def _require_comparable(
    left: RodReconstructionEvaluation,
    right: RodReconstructionEvaluation,
    /,
) -> None:
    if not isinstance(left, RodReconstructionEvaluation) or not isinstance(
        right, RodReconstructionEvaluation
    ):
        raise TypeError("Discretization comparison requires reconstruction evaluations.")
    if left.query_plan_id != right.query_plan_id:
        raise ValueError("Rod reconstruction query-plan identities do not match.")
    if not np.array_equal(np.asarray(left.arc_lengths), np.asarray(right.arc_lengths)):
        raise ValueError("Rod reconstruction physical query coordinates do not match.")
    if left.reduction_id != right.reduction_id:
        raise ValueError("Rod reconstruction reduction identities do not match.")
    if left.frame_convention != right.frame_convention:
        raise ValueError("Rod reconstruction frame conventions do not match.")
    if left.route_id != right.route_id:
        raise ValueError("Rod reconstruction route identities do not match.")


def compare_reduced_rod_discretizations(
    coarse: RodReconstructionEvaluation,
    medium: RodReconstructionEvaluation,
    fine: RodReconstructionEvaluation | None = None,
    /,
) -> RodDiscretizationComparison:
    """Compare two or three fixed refinements at common physical query frames.

    Two levels report adjacent errors and explicitly unsupported (NaN) observed
    order. Three levels report the adjacent errors and the measured order when
    the refinement ratios form one decreasing, geometrically consistent sequence.
    """
    _require_comparable(coarse, medium)
    if fine is not None:
        _require_comparable(medium, fine)
    coarse_medium = _comparison_errors(coarse, medium)
    query_match = jnp.asarray(coarse.query_plan_id == medium.query_plan_id)
    physical_match = jnp.asarray(
        np.array_equal(np.asarray(coarse.arc_lengths), np.asarray(medium.arc_lengths))
    )
    reduction_match = jnp.asarray(coarse.reduction_id == medium.reduction_id)
    frame_match = jnp.asarray(coarse.frame_convention == medium.frame_convention)
    route_match = jnp.asarray(coarse.route_id == medium.route_id)

    if fine is None:
        medium_fine = None
        observed = _nan_order(coarse.poses.dtype)
        supported = jnp.asarray(False)
        refinement_valid = jnp.asarray(False)
    else:
        medium_fine = _comparison_errors(medium, fine)
        coarse_ratio = coarse.maximum_panel_length / medium.maximum_panel_length
        medium_ratio = medium.maximum_panel_length / fine.maximum_panel_length
        ratio_tolerance = 500.0 * np.finfo(np.dtype(coarse.poses.dtype)).eps
        refinement_valid = jnp.asarray(
            (coarse_ratio > 1.0)
            and (medium_ratio > 1.0)
            and np.isclose(
                coarse_ratio,
                medium_ratio,
                rtol=ratio_tolerance,
                atol=ratio_tolerance,
            )
        )
        ratio = jnp.asarray(coarse_ratio, dtype=coarse.poses.dtype)
        observed = _observed_order(coarse_medium, medium_fine, ratio)
        supported = refinement_valid

    error_values = (
        coarse_medium.scaled_se3_log,
        coarse_medium.scaled_strain,
        coarse_medium.scaled_body_twist,
        coarse_medium.scaled_frame_jvp,
        coarse_medium.native_energy,
    )
    if medium_fine is not None:
        error_values = error_values + (
            medium_fine.scaled_se3_log,
            medium_fine.scaled_strain,
            medium_fine.scaled_body_twist,
            medium_fine.scaled_frame_jvp,
            medium_fine.native_energy,
        )
    finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(v)) for v in error_values)))
    valid = (
        query_match
        & physical_match
        & reduction_match
        & frame_match
        & route_match
        & finite
        & ((~supported) | refinement_valid)
    )
    evidence = RodDiscretizationComparisonEvidence(
        query_match,
        physical_match,
        reduction_match,
        frame_match,
        route_match,
        supported,
        refinement_valid,
        finite,
        valid,
    )
    comparison_id = canonical_fingerprint(
        {
            "kind": "reduced-rod-observational-discretization-comparison",
            "coarse": coarse.reconstruction_id,
            "medium": medium.reconstruction_id,
            "fine": None if fine is None else fine.reconstruction_id,
            "queries": coarse.query_plan_id,
            "routes": coarse.route_id,
        }
    )
    return RodDiscretizationComparison(
        coarse_medium, medium_fine, observed, evidence, comparison_id
    )


__all__ = [
    "PreparedRodReconstruction",
    "RodDiscretizationComparison",
    "RodDiscretizationComparisonEvidence",
    "RodDiscretizationErrors",
    "RodFrameQueryPlan",
    "RodNativeDiscretizationDiscrepancy",
    "RodObservedOrder",
    "RodReconstructionChartEvidence",
    "RodReconstructionDomainEvidence",
    "RodReconstructionEvaluation",
    "RodReconstructionPlan",
    "RodReconstructionQuadratureEvidence",
    "compare_reduced_rod_discretizations",
    "evaluate_rod_reconstruction",
    "prepare_rod_reconstruction",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_allfinite, tree_where
from ...nonlinear import (
    NonlinearTermination,
    SemismoothNewton,
    VariationalInequalityProblem,
)
from ._rigid_body import (
    _rigid_body_world_inertia,
    PreparedRigidBodySet,
    RigidBodyKinematics,
)
from ._rigid_joints import PreparedRigidJointGraph


class FixedCapacityUnilateralPlan(StrictModule, NonTrainableState):
    """Static row topology for a nonnegative-impulse variational inequality."""

    route_keys: Array
    valid: Array
    method: SemismoothNewton
    termination: NonlinearTermination
    complementarity_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        route_keys: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        method: SemismoothNewton | None = None,
        termination: NonlinearTermination | None = None,
        complementarity_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ):
        keys = np.asarray(route_keys)
        if keys.ndim != 1 or keys.size == 0 or not np.issubdtype(keys.dtype, np.integer):
            raise TypeError("route_keys must be a nonempty rank-1 integer array.")
        valid_ = (
            np.ones(keys.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != keys.shape:
            raise ValueError("valid must have the unilateral row-capacity shape.")
        if np.unique(keys[valid_]).size != np.count_nonzero(valid_):
            raise ValueError("Valid unilateral route keys must be unique.")
        method_ = (
            SemismoothNewton(
                formulation="natural",
                feasibility="preserve-box",
                certification_tolerance=complementarity_tolerance,
            )
            if method is None
            else method
        )
        termination_ = (
            NonlinearTermination(
                absolute_residual=1.0e-11,
                relative_residual=1.0e-9,
                absolute_step=1.0e-12,
                relative_step=1.0e-10,
                maximum_steps=48,
                maximum_evaluations=192,
                maximum_linear_iterations=512,
            )
            if termination is None
            else termination
        )
        tolerance = float(complementarity_tolerance)
        if not isinstance(method_, SemismoothNewton):
            raise TypeError("method must be SemismoothNewton or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("complementarity_tolerance must be positive and finite.")
        generated = canonical_fingerprint(
            {
                "kind": "fixed-capacity-unilateral-plan",
                "topology": array_tree_fingerprint({"route_keys": keys, "valid": valid_}),
                "formulation": method_.formulation,
                "feasibility": method_.feasibility,
                "tolerance": tolerance,
                "maximum_steps": termination_.maximum_steps,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.route_keys = jnp.asarray(keys, dtype=jnp.int32)
        self.valid = jnp.asarray(valid_)
        self.method = method_
        self.termination = termination_
        self.complementarity_tolerance = tolerance
        self.plan_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.route_keys.shape[0])

    def prepare(self, /, *, prepared_scope_id: str) -> PreparedUnilateralRows:
        return PreparedUnilateralRows(self, prepared_scope_id=prepared_scope_id)


class UnilateralState(StrictModule):
    impulses: Array
    active: Array
    numeric_version: Array


class UnilateralCertificate(StrictModule):
    primal_violation: Array
    dual_violation: Array
    complementarity_residual: Array
    natural_residual: Array
    row_primal_violation: Array
    row_dual_violation: Array
    row_complementarity: Array
    finite: Array
    certified: Array


class UnilateralEvaluation(StrictModule):
    impulses: Array
    operator_value: Array
    active: Array
    lower_active: Array
    release_margin: Array
    branch_margin: Array
    certificate: UnilateralCertificate
    solver_successful: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class UnilateralStepResult(StrictModule):
    candidate_state: UnilateralState
    accepted_state: UnilateralState
    evaluation: UnilateralEvaluation
    successful: Array


class PreparedUnilateralRows(StrictModule, NonTrainableState):
    plan: FixedCapacityUnilateralPlan
    route_keys: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FixedCapacityUnilateralPlan,
        /,
        *,
        prepared_scope_id: str,
    ):
        if not isinstance(plan, FixedCapacityUnilateralPlan):
            raise TypeError("plan must be a FixedCapacityUnilateralPlan.")
        scope = str(prepared_scope_id)
        if not scope:
            raise ValueError("prepared_scope_id must be nonempty.")
        self.plan = plan
        self.route_keys = plan.route_keys
        self.valid = plan.valid
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unilateral-rows",
                "plan": plan.plan_id,
                "scope": scope,
            }
        )

    @property
    def capacity(self) -> int:
        return self.plan.capacity

    def initial_state(self, reference: ArrayLike, /) -> UnilateralState:
        value = jnp.asarray(reference)
        if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("reference must be an inexact scalar.")
        return UnilateralState(
            jnp.zeros((self.capacity,), dtype=value.dtype),
            jnp.zeros((self.capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def evaluate(
        self,
        state: UnilateralState,
        delassus: ArrayLike,
        free_residual: ArrayLike,
        active: ArrayLike,
        /,
    ) -> UnilateralStepResult:
        if not isinstance(state, UnilateralState):
            raise TypeError("state must be UnilateralState.")
        matrix = jnp.asarray(delassus)
        free = jnp.asarray(free_residual, dtype=matrix.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        expected = (self.capacity,)
        if (
            matrix.shape != (self.capacity, self.capacity)
            or free.shape != expected
            or active_.shape != expected
            or state.impulses.shape != expected
            or state.active.shape != expected
            or state.numeric_version.ndim != 0
        ):
            raise ValueError(
                "Unilateral solve arrays do not match the prepared capacity."
            )
        enabled = self.valid & active_
        enabled_float = enabled.astype(matrix.dtype)
        effective_matrix = matrix * enabled_float[:, None] * enabled_float[
            None, :
        ] + jnp.diag((~enabled).astype(matrix.dtype))
        effective_free = jnp.where(enabled, free, 0.0)
        warm = jnp.where(enabled, jnp.maximum(state.impulses, 0.0), 0.0)

        def operator(impulses, arguments):
            del arguments
            return contract("ij,j->i", effective_matrix, impulses) + effective_free

        problem = VariationalInequalityProblem(
            operator,
            Bounds(0.0, jnp.inf),
            problem_id=f"{self.prepared_id}/nonnegative-impulse",
        )
        solve = self.plan.method.solve(
            problem,
            warm,
            termination=self.plan.termination,
        )
        impulses = jnp.where(enabled, jnp.maximum(solve.state, 0.0), 0.0)
        operator_value = operator(impulses, None)
        row_primal = jnp.where(enabled, jnp.maximum(-operator_value, 0.0), 0.0)
        row_dual = jnp.where(enabled, jnp.maximum(-impulses, 0.0), 0.0)
        row_product = jnp.where(enabled, jnp.abs(impulses * operator_value), 0.0)
        natural = jnp.where(
            enabled,
            jnp.abs(impulses - jnp.maximum(impulses - operator_value, 0.0)),
            0.0,
        )
        primal_violation = jnp.max(row_primal, initial=0.0)
        dual_violation = jnp.max(row_dual, initial=0.0)
        complementarity = jnp.max(row_product, initial=0.0)
        natural_residual = jnp.max(natural, initial=0.0)
        finite = tree_allfinite((matrix, free, impulses, operator_value)) & jnp.all(
            jnp.isfinite(matrix)
        )
        tolerance = jnp.asarray(self.plan.complementarity_tolerance, dtype=matrix.dtype)
        certified = (
            finite
            & (primal_violation <= tolerance)
            & (dual_violation <= tolerance)
            & (complementarity <= tolerance)
            & (natural_residual <= tolerance)
        )
        successful = solve.successful & certified
        candidate = UnilateralState(
            impulses,
            enabled,
            state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted = accept_unilateral_candidate(state, candidate, successful)
        lower_active = enabled & (impulses > tolerance)
        release_margin = jnp.where(enabled, operator_value, jnp.inf)
        branch_margin = jnp.where(
            lower_active,
            impulses,
            jnp.where(enabled, operator_value, jnp.inf),
        )
        certificate = UnilateralCertificate(
            primal_violation,
            dual_violation,
            complementarity,
            natural_residual,
            row_primal,
            row_dual,
            row_product,
            finite,
            certified,
        )
        evaluation = UnilateralEvaluation(
            impulses,
            operator_value,
            enabled,
            lower_active,
            release_margin,
            branch_margin,
            certificate,
            solve.successful,
            successful,
            self.prepared_id,
        )
        return UnilateralStepResult(candidate, accepted, evaluation, successful)


def unilateral_candidate(
    state: UnilateralState,
    impulses: ArrayLike,
    active: ArrayLike,
    /,
) -> UnilateralState:
    """Build a versioned warm-start candidate without committing it."""

    if not isinstance(state, UnilateralState):
        raise TypeError("state must be UnilateralState.")
    impulses_ = jnp.asarray(impulses, dtype=state.impulses.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    if impulses_.shape != state.impulses.shape or active_.shape != state.active.shape:
        raise ValueError("Candidate unilateral arrays must preserve state capacity.")
    return UnilateralState(
        impulses_,
        active_,
        state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def accept_unilateral_candidate(
    current: UnilateralState,
    candidate: UnilateralState,
    accepted: ArrayLike,
    /,
) -> UnilateralState:
    """Atomically commit or roll back all unilateral warm-start leaves."""

    if not isinstance(current, UnilateralState) or not isinstance(
        candidate, UnilateralState
    ):
        raise TypeError("current and candidate must be UnilateralState values.")
    predicate = jnp.asarray(accepted, dtype=bool)
    if predicate.ndim != 0:
        raise ValueError("accepted must be scalar.")
    return tree_where(predicate, candidate, current)


class JointLimitPlan(StrictModule, NonTrainableState):
    """Fixed-capacity lower/upper limits for unwrapped hinge coordinates."""

    hinge_ids: Array
    lower_limits: Array
    upper_limits: Array
    valid: Array
    rows: FixedCapacityUnilateralPlan
    activation_distance: float = eqx.field(static=True)
    release_velocity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hinge_ids: ArrayLike,
        lower_limits: ArrayLike,
        upper_limits: ArrayLike,
        /,
        *,
        capacity: int | None = None,
        activation_distance: float = 1.0e-8,
        release_velocity: float = 1.0e-10,
        complementarity_tolerance: float = 1.0e-8,
        plan_id: str | None = None,
    ):
        identifiers = np.asarray(hinge_ids)
        lower = np.asarray(lower_limits)
        upper = np.asarray(upper_limits)
        if (
            identifiers.ndim != 1
            or identifiers.size == 0
            or not np.issubdtype(identifiers.dtype, np.integer)
        ):
            raise TypeError("hinge_ids must be a nonempty rank-1 integer array.")
        if lower.shape != identifiers.shape or upper.shape != identifiers.shape:
            raise ValueError("Joint limit bounds must match hinge_ids.")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("Joint lower limits must be finite and below upper limits.")
        if np.unique(identifiers).size != identifiers.size:
            raise ValueError("Joint-limit hinge IDs must be unique.")
        capacity_ = identifiers.size if capacity is None else int(capacity)
        if capacity_ < identifiers.size or capacity_ <= 0:
            raise ValueError("capacity must contain every configured joint limit.")
        activation = float(activation_distance)
        release = float(release_velocity)
        if (
            not isfinite(activation)
            or activation < 0.0
            or not isfinite(release)
            or release < 0.0
        ):
            raise ValueError("Joint-limit activation/release scales must be finite.")
        padded_ids = np.full((capacity_,), -1, dtype=np.int64)
        padded_lower = np.zeros((capacity_,), dtype=lower.dtype)
        padded_upper = np.ones((capacity_,), dtype=upper.dtype)
        valid = np.zeros((capacity_,), dtype=bool)
        count = identifiers.size
        padded_ids[:count] = identifiers
        padded_lower[:count] = lower
        padded_upper[:count] = upper
        valid[:count] = True
        unsigned_ids = padded_ids.astype(np.uint64)
        route_keys = np.stack(
            (
                (2 * unsigned_ids).astype(np.uint32),
                (2 * unsigned_ids + 1).astype(np.uint32),
            ),
            axis=-1,
        ).reshape((-1,))
        row_valid = np.repeat(valid[:, None], 2, axis=1).reshape((-1,))
        rows = FixedCapacityUnilateralPlan(
            route_keys,
            valid=row_valid,
            complementarity_tolerance=complementarity_tolerance,
            plan_id=canonical_fingerprint(
                {
                    "kind": "joint-limit-unilateral-rows",
                    "ids": identifiers.tolist(),
                    "capacity": capacity_,
                }
            ),
        )
        generated = canonical_fingerprint(
            {
                "kind": "joint-limit-plan",
                "values": array_tree_fingerprint(
                    {
                        "hinge_ids": padded_ids,
                        "lower": padded_lower,
                        "upper": padded_upper,
                        "valid": valid,
                    }
                ),
                "rows": rows.plan_id,
                "activation_distance": activation,
                "release_velocity": release,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.hinge_ids = jnp.asarray(padded_ids, dtype=jnp.int32)
        self.lower_limits = jnp.asarray(padded_lower)
        self.upper_limits = jnp.asarray(padded_upper)
        self.valid = jnp.asarray(valid)
        self.rows = rows
        self.activation_distance = activation
        self.release_velocity = release
        self.plan_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.hinge_ids.shape[0])

    def prepare(
        self,
        graph: PreparedRigidJointGraph,
        /,
    ) -> PreparedJointLimits:
        return PreparedJointLimits(self, graph)


class JointLimitState(StrictModule):
    coordinate: Array
    lower_impulse: Array
    upper_impulse: Array
    lower_active: Array
    upper_active: Array
    numeric_version: Array


class JointLimitCertificate(StrictModule):
    lower_gap_violation: Array
    upper_gap_violation: Array
    position_complementarity: Array
    velocity_primal_violation: Array
    velocity_complementarity: Array
    finite: Array
    certified: Array


class JointLimitEvaluation(StrictModule):
    predicted_coordinate: Array
    coordinate: Array
    relative_speed_before: Array
    relative_speed_after: Array
    lower_gap: Array
    upper_gap: Array
    lower_impulse: Array
    upper_impulse: Array
    lower_active: Array
    upper_active: Array
    released_lower: Array
    released_upper: Array
    branch_margin: Array
    certificate: JointLimitCertificate
    unilateral: UnilateralEvaluation
    corrected_kinematics: RigidBodyKinematics
    successful: Array
    prepared_id: str = eqx.field(static=True)


class JointLimitStepResult(StrictModule):
    candidate_state: JointLimitState
    accepted_state: JointLimitState
    evaluation: JointLimitEvaluation
    successful: Array


class PreparedJointLimits(StrictModule, NonTrainableState):
    plan: JointLimitPlan
    graph: PreparedRigidJointGraph
    bodies: PreparedRigidBodySet
    hinge_indices: Array
    left: Array
    right: Array
    rows: PreparedUnilateralRows
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: JointLimitPlan, graph: PreparedRigidJointGraph, /):
        if not isinstance(plan, JointLimitPlan):
            raise TypeError("plan must be a JointLimitPlan.")
        if not isinstance(graph, PreparedRigidJointGraph):
            raise TypeError("graph must be a PreparedRigidJointGraph.")
        if graph.plan.hinge is None:
            raise ValueError("Joint limits require a prepared hinge graph.")
        if graph.bodies.ambient_dimension != 3:
            raise ValueError("Hinge joint limits require three-dimensional rigid bodies.")
        graph_ids = np.asarray(graph.plan.hinge.joint_ids)
        order = np.argsort(graph_ids)
        sorted_ids = graph_ids[order]
        requested = np.asarray(plan.hinge_ids)
        valid = np.asarray(plan.valid)
        ranks = np.searchsorted(sorted_ids, requested[valid])
        found = (ranks < sorted_ids.size) & (
            sorted_ids[np.minimum(ranks, sorted_ids.size - 1)] == requested[valid]
        )
        if not np.all(found):
            raise ValueError(
                "Every configured joint-limit ID must name a prepared hinge."
            )
        indices = np.zeros((plan.capacity,), dtype=np.int32)
        indices[valid] = order[ranks]
        left = np.asarray(graph.hinge_left)[indices]
        right = np.asarray(graph.hinge_right)[indices]
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-joint-limits",
                "plan": plan.plan_id,
                "graph": graph.prepared_id,
            }
        )
        self.plan = plan
        self.graph = graph
        self.bodies = graph.bodies
        self.hinge_indices = jnp.asarray(indices)
        self.left = jnp.asarray(left)
        self.right = jnp.asarray(right)
        self.rows = plan.rows.prepare(prepared_scope_id=prepared_id)
        self.prepared_id = prepared_id

    @property
    def capacity(self) -> int:
        return self.plan.capacity

    def initial_state(
        self,
        coordinate: ArrayLike | None = None,
        /,
    ) -> JointLimitState:
        dtype = self.bodies.particles.safe_masses.dtype
        value = (
            jnp.zeros((self.capacity,), dtype=dtype)
            if coordinate is None
            else jnp.asarray(coordinate, dtype=dtype)
        )
        if value.shape != (self.capacity,):
            raise ValueError("coordinate must have joint-limit capacity shape.")
        zero = jnp.zeros((self.capacity,), dtype=dtype)
        inactive = jnp.zeros((self.capacity,), dtype=bool)
        return JointLimitState(
            jnp.where(self.plan.valid, value, 0.0),
            zero,
            zero,
            inactive,
            inactive,
            jnp.asarray(0, dtype=jnp.int32),
        )

    def evaluate(
        self,
        state: JointLimitState,
        kinematics: RigidBodyKinematics,
        time_step: ArrayLike,
        /,
    ) -> JointLimitStepResult:
        if not isinstance(state, JointLimitState):
            raise TypeError("state must be JointLimitState.")
        if not isinstance(kinematics, RigidBodyKinematics):
            raise TypeError("kinematics must be RigidBodyKinematics.")
        expected = (self.capacity,)
        if (
            state.coordinate.shape != expected
            or state.lower_impulse.shape != expected
            or state.upper_impulse.shape != expected
            or state.lower_active.shape != expected
            or state.upper_active.shape != expected
        ):
            raise ValueError("Joint-limit state does not match prepared capacity.")
        if (
            kinematics.position.shape != (self.bodies.capacity, 3)
            or kinematics.velocity.shape != (self.bodies.capacity, 3)
            or kinematics.orientation.shape != (self.bodies.capacity, 4)
            or kinematics.angular_velocity.shape != (self.bodies.capacity, 3)
        ):
            raise ValueError("Joint-limit kinematics do not match prepared bodies.")
        dt = jnp.asarray(time_step, dtype=kinematics.position.dtype)
        if dt.ndim != 0:
            raise ValueError("time_step must be scalar.")
        rotation = self._rotation(kinematics)
        left_axis = contract(
            "cij,cj->ci",
            rotation[self.left],
            self.graph.hinge_axis_left[self.hinge_indices],
        )
        right_axis = contract(
            "cij,cj->ci",
            rotation[self.right],
            self.graph.hinge_axis_right[self.hinge_indices],
        )
        axis_sum = left_axis + right_axis
        axis_norm = jnp.sqrt(jnp.sum(axis_sum * axis_sum, axis=-1))
        safe_norm = jnp.where(axis_norm > 0.0, axis_norm, 1.0)
        axis = axis_sum / safe_norm[:, None]
        relative_angular = (
            kinematics.angular_velocity[self.right]
            - kinematics.angular_velocity[self.left]
        )
        relative_speed = jnp.sum(relative_angular * axis, axis=-1)
        predicted = state.coordinate + dt * relative_speed
        lower_gap_predicted = predicted - self.plan.lower_limits
        upper_gap_predicted = self.plan.upper_limits - predicted
        activation = jnp.asarray(self.plan.activation_distance, dtype=predicted.dtype)
        release = jnp.asarray(self.plan.release_velocity, dtype=predicted.dtype)
        valid = self.plan.valid
        lower_active = (
            valid
            & (lower_gap_predicted <= activation)
            & (
                (relative_speed < release)
                | (lower_gap_predicted < 0.0)
                | (state.lower_impulse > 0.0)
            )
        )
        upper_active = (
            valid
            & (upper_gap_predicted <= activation)
            & (
                (relative_speed > -release)
                | (upper_gap_predicted < 0.0)
                | (state.upper_impulse > 0.0)
            )
        )
        both = lower_active & upper_active
        lower_active = lower_active & ~both
        upper_active = upper_active & ~both
        row_active = jnp.stack((lower_active, upper_active), axis=-1).reshape((-1,))
        signs = jnp.broadcast_to(
            jnp.asarray((1.0, -1.0), dtype=predicted.dtype),
            (self.capacity, 2),
        ).reshape((-1,))
        row_speed = (signs.reshape((self.capacity, 2)) * relative_speed[:, None]).reshape(
            (-1,)
        )
        predicted_gaps = jnp.stack(
            (lower_gap_predicted, upper_gap_predicted), axis=-1
        ).reshape((-1,))
        separation_bias = jnp.where(
            row_active,
            jnp.maximum(-predicted_gaps, 0.0) / jnp.maximum(dt, jnp.finfo(dt.dtype).tiny),
            0.0,
        )
        free_residual = row_speed - separation_bias
        delassus = self._delassus(kinematics, axis, signs)
        warm = UnilateralState(
            jnp.stack((state.lower_impulse, state.upper_impulse), axis=-1).reshape((-1,)),
            jnp.stack((state.lower_active, state.upper_active), axis=-1).reshape((-1,)),
            state.numeric_version,
        )
        unilateral = self.rows.evaluate(warm, delassus, free_residual, row_active)
        impulse_rows = unilateral.candidate_state.impulses.reshape((self.capacity, 2))
        lower_impulse = impulse_rows[:, 0]
        upper_impulse = impulse_rows[:, 1]
        signed_impulse = lower_impulse - upper_impulse
        corrected_angular = self._apply_impulses(kinematics, axis, signed_impulse)
        corrected = RigidBodyKinematics(
            kinematics.position,
            kinematics.velocity,
            kinematics.orientation,
            corrected_angular,
        )
        post_relative = jnp.sum(
            (corrected_angular[self.right] - corrected_angular[self.left]) * axis,
            axis=-1,
        )
        coordinate = jnp.where(
            valid,
            jnp.clip(predicted, self.plan.lower_limits, self.plan.upper_limits),
            0.0,
        )
        lower_gap = jnp.where(valid, coordinate - self.plan.lower_limits, 0.0)
        upper_gap = jnp.where(valid, self.plan.upper_limits - coordinate, 0.0)
        position_product = jnp.maximum(
            jnp.abs(lower_gap * lower_impulse),
            jnp.abs(upper_gap * upper_impulse),
        )
        lower_violation = jnp.max(
            jnp.where(valid, jnp.maximum(-lower_gap, 0.0), 0.0), initial=0.0
        )
        upper_violation = jnp.max(
            jnp.where(valid, jnp.maximum(-upper_gap, 0.0), 0.0), initial=0.0
        )
        position_complementarity = jnp.max(position_product, initial=0.0)
        row_post = (
            contract("ij,j->i", delassus, unilateral.evaluation.impulses) + free_residual
        )
        velocity_primal = jnp.max(
            jnp.where(row_active, jnp.maximum(-row_post, 0.0), 0.0), initial=0.0
        )
        velocity_product = jnp.max(
            jnp.where(
                row_active,
                jnp.abs(unilateral.evaluation.impulses * row_post),
                0.0,
            ),
            initial=0.0,
        )
        finite = tree_allfinite(
            (
                coordinate,
                corrected,
                lower_impulse,
                upper_impulse,
                lower_gap,
                upper_gap,
                delassus,
            )
        ) & jnp.isfinite(dt)
        tolerance = jnp.asarray(
            self.plan.rows.complementarity_tolerance, dtype=coordinate.dtype
        )
        certified = (
            finite
            & (dt > 0.0)
            & (lower_violation <= tolerance)
            & (upper_violation <= tolerance)
            & (position_complementarity <= tolerance)
            & (velocity_primal <= tolerance)
            & (velocity_product <= tolerance)
        )
        successful = unilateral.successful & certified & jnp.all(axis_norm > 0.0)
        candidate = JointLimitState(
            coordinate,
            lower_impulse,
            upper_impulse,
            lower_active,
            upper_active,
            state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        )
        accepted = accept_joint_limit_candidate(state, candidate, successful)
        released_lower = state.lower_active & ~lower_active
        released_upper = state.upper_active & ~upper_active
        branch_margin = jnp.minimum(lower_gap_predicted, upper_gap_predicted)
        certificate = JointLimitCertificate(
            lower_violation,
            upper_violation,
            position_complementarity,
            velocity_primal,
            velocity_product,
            finite,
            certified,
        )
        evaluation = JointLimitEvaluation(
            predicted,
            coordinate,
            relative_speed,
            post_relative,
            lower_gap,
            upper_gap,
            lower_impulse,
            upper_impulse,
            lower_active,
            upper_active,
            released_lower,
            released_upper,
            branch_margin,
            certificate,
            unilateral.evaluation,
            corrected,
            successful,
            self.prepared_id,
        )
        return JointLimitStepResult(candidate, accepted, evaluation, successful)

    @staticmethod
    def _rotation(kinematics: RigidBodyKinematics, /) -> Array:
        from ._rigid_body import quaternion_rotation_matrix

        return quaternion_rotation_matrix(kinematics.orientation)

    def _delassus(
        self,
        kinematics: RigidBodyKinematics,
        axis: Array,
        signs: Array,
        /,
    ) -> Array:
        _, inverse_world = _rigid_body_world_inertia(self.bodies, kinematics.orientation)
        row_axis = (signs.reshape((self.capacity, 2, 1)) * axis[:, None, :]).reshape(
            (-1, 3)
        )
        row_left = jnp.repeat(self.left, 2)
        row_right = jnp.repeat(self.right, 2)
        row_valid = jnp.repeat(self.plan.valid, 2)
        row_count = 2 * self.capacity

        def response(impulse):
            torque = jnp.zeros((self.bodies.capacity, 3), dtype=kinematics.position.dtype)
            applied = row_axis * impulse[:, None] * row_valid[:, None]
            torque = torque.at[row_right].add(applied)
            torque = torque.at[row_left].add(-applied)
            delta_angular = contract("bij,bj->bi", inverse_world, torque)
            relative = delta_angular[row_right] - delta_angular[row_left]
            return jnp.sum(relative * row_axis, axis=-1)

        basis = jnp.eye(row_count, dtype=kinematics.position.dtype)
        return jax.vmap(response)(basis).T

    def _apply_impulses(
        self,
        kinematics: RigidBodyKinematics,
        axis: Array,
        signed_impulse: Array,
        /,
    ) -> Array:
        _, inverse_world = _rigid_body_world_inertia(self.bodies, kinematics.orientation)
        torque = jnp.zeros((self.bodies.capacity, 3), dtype=kinematics.position.dtype)
        applied = axis * signed_impulse[:, None] * self.plan.valid[:, None]
        torque = torque.at[self.right].add(applied)
        torque = torque.at[self.left].add(-applied)
        delta = contract("bij,bj->bi", inverse_world, torque)
        return kinematics.angular_velocity + delta


def joint_limit_candidate(
    state: JointLimitState,
    coordinate: ArrayLike,
    lower_impulse: ArrayLike,
    upper_impulse: ArrayLike,
    lower_active: ArrayLike,
    upper_active: ArrayLike,
    /,
) -> JointLimitState:
    """Build a complete versioned joint-limit candidate without committing it."""

    if not isinstance(state, JointLimitState):
        raise TypeError("state must be JointLimitState.")
    coordinate_ = jnp.asarray(coordinate, dtype=state.coordinate.dtype)
    lower_impulse_ = jnp.asarray(lower_impulse, dtype=state.lower_impulse.dtype)
    upper_impulse_ = jnp.asarray(upper_impulse, dtype=state.upper_impulse.dtype)
    lower_active_ = jnp.asarray(lower_active, dtype=bool)
    upper_active_ = jnp.asarray(upper_active, dtype=bool)
    expected = state.coordinate.shape
    if (
        coordinate_.shape != expected
        or lower_impulse_.shape != expected
        or upper_impulse_.shape != expected
        or lower_active_.shape != expected
        or upper_active_.shape != expected
    ):
        raise ValueError("Joint-limit candidate must preserve state capacity.")
    return JointLimitState(
        coordinate_,
        lower_impulse_,
        upper_impulse_,
        lower_active_,
        upper_active_,
        state.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def accept_joint_limit_candidate(
    current: JointLimitState,
    candidate: JointLimitState,
    accepted: ArrayLike,
    /,
) -> JointLimitState:
    """Atomically commit or roll back joint coordinate and both warm impulses."""

    if not isinstance(current, JointLimitState) or not isinstance(
        candidate, JointLimitState
    ):
        raise TypeError("current and candidate must be JointLimitState values.")
    predicate = jnp.asarray(accepted, dtype=bool)
    if predicate.ndim != 0:
        raise ValueError("accepted must be scalar.")
    return tree_where(predicate, candidate, current)


__all__ = [
    "FixedCapacityUnilateralPlan",
    "JointLimitCertificate",
    "JointLimitEvaluation",
    "JointLimitPlan",
    "JointLimitState",
    "JointLimitStepResult",
    "PreparedJointLimits",
    "PreparedUnilateralRows",
    "UnilateralCertificate",
    "UnilateralEvaluation",
    "UnilateralState",
    "UnilateralStepResult",
    "accept_joint_limit_candidate",
    "joint_limit_candidate",
    "accept_unilateral_candidate",
    "unilateral_candidate",
]

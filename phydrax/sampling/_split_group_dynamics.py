#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from ..metrix import (
    AbstractStateGeometry,
    FlatTorusStateGeometry,
    LieGroupStateGeometry,
    PointwiseStateGeometry,
    SpecialUnitaryGroup,
    UnitaryGroup,
)


_MOMENTUM_ADDRESS = SampleAddress(
    "markov", "split-group-dynamics", target="partial-momentum", role="transition"
)
_ACCEPT_ADDRESS = SampleAddress(
    "markov", "split-group-dynamics", target="exact-correction", role="transition"
)
_NUTS_ADDRESS = SampleAddress(
    "markov", "transported-group-nuts", target="finite-tree", role="transition"
)

IntegratorKind = Literal["leapfrog", "omelyan"]
DynamicsKind = Literal["ghmc", "nuts-reference"]


class SplitGroupDynamicsStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_OR_DIVERGENT_TRAJECTORY = 2
    GROUP_MEMBERSHIP_FAILURE = 3
    INVALID_INPUT_STATE = 4


class SplitGroupTarget(StrictModule):
    """Authoritative compact target plus separately differentiable force terms."""

    log_target: Callable[[Array], Array]
    force_terms: tuple[Callable[[Array], Array], ...]
    geometry: AbstractStateGeometry
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_target: Callable[[Array], Array],
        force_terms: tuple[Callable[[Array], Array], ...],
        geometry: AbstractStateGeometry,
        /,
        *,
        configuration_shape: tuple[int, ...],
        local_coordinate_shape: tuple[int, ...],
        reference_measure: str,
        target_id: str,
    ):
        if not callable(log_target) or any(not callable(term) for term in force_terms):
            raise TypeError("log_target and every force term must be callable.")
        if not force_terms or len(force_terms) > 64:
            raise ValueError("force_terms must contain between one and 64 terms.")
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("geometry must implement AbstractStateGeometry.")
        point_shape = tuple(int(size) for size in configuration_shape)
        local_shape = tuple(int(size) for size in local_coordinate_shape)
        if (
            not point_shape
            or not local_shape
            or any(size <= 0 for size in point_shape + local_shape)
        ):
            raise ValueError("Target shapes must contain only positive dimensions.")
        if reference_measure not in ("flat-torus", "product-haar"):
            raise ValueError(
                "Split group dynamics requires flat-torus or product-Haar measure."
            )
        identifier = str(target_id)
        if not identifier:
            raise ValueError("target_id must be nonempty.")
        self.log_target = log_target
        self.force_terms = tuple(force_terms)
        self.geometry = geometry
        self.configuration_shape = point_shape
        self.local_coordinate_shape = local_shape
        self.reference_measure = reference_measure
        self.target_id = identifier

    def __call__(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(self.log_target(jnp.asarray(position)))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError("log_target must return one real scalar.")
        return value


class SplitGroupDynamicsPlan(StrictModule):
    """Resource-bounded integration and generalized-momentum policy."""

    step_size: float = eqx.field(static=True)
    trajectory_steps: int = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    momentum_persistence: float = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    omelyan_lambda: float = eqx.field(static=True)
    integrator: IntegratorKind = eqx.field(static=True)
    dynamics: DynamicsKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        step_size: float,
        trajectory_steps: int = 8,
        maximum_tree_depth: int = 6,
        momentum_persistence: float = 0.0,
        divergence_threshold: float = 1000.0,
        integrator: IntegratorKind = "omelyan",
        dynamics: DynamicsKind = "ghmc",
        omelyan_lambda: float = 0.1931833275037836,
    ):
        size = float(step_size)
        steps = int(trajectory_steps)
        depth = int(maximum_tree_depth)
        persistence = float(momentum_persistence)
        threshold = float(divergence_threshold)
        coefficient = float(omelyan_lambda)
        if integrator not in ("leapfrog", "omelyan"):
            raise ValueError("integrator must be 'leapfrog' or 'omelyan'.")
        if dynamics not in ("ghmc", "nuts-reference"):
            raise ValueError("Unsupported split group dynamics kind.")
        if (
            not np.isfinite(size)
            or size <= 0.0
            or steps <= 0
            or steps > 1_000_000
            or depth <= 0
            or depth > 20
        ):
            raise ValueError(
                "Step size must be positive, trajectory_steps at most 1000000, "
                "and maximum_tree_depth in [1, 20]."
            )
        if not 0.0 <= persistence < 1.0 or not np.isfinite(persistence):
            raise ValueError("momentum_persistence must lie in [0, 1).")
        if dynamics == "nuts-reference" and persistence != 0.0:
            raise ValueError(
                "Transported group NUTS requires full independent momentum refresh."
            )
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("divergence_threshold must be finite and positive.")
        if not np.isfinite(coefficient) or not 0.0 < coefficient < 0.5:
            raise ValueError(
                "omelyan_lambda must lie strictly between zero and one half."
            )
        self.step_size = size
        self.trajectory_steps = steps
        self.maximum_tree_depth = depth
        self.momentum_persistence = persistence
        self.divergence_threshold = threshold
        self.omelyan_lambda = coefficient
        self.integrator = integrator
        self.dynamics = dynamics
        self.plan_id = canonical_fingerprint(
            {
                "kind": "split-compact-group-dynamics",
                "step_size": size,
                "trajectory_steps": steps,
                "maximum_tree_depth": depth,
                "momentum_persistence": persistence,
                "divergence_threshold": threshold,
                "integrator": integrator,
                "dynamics": dynamics,
                "omelyan_lambda": coefficient,
            }
        )


class SplitMetricAdaptationPlan(StrictModule):
    warmup_steps: int = eqx.field(static=True)
    minimum_inverse_mass: float = eqx.field(static=True)
    maximum_inverse_mass: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        warmup_steps: int,
        /,
        *,
        minimum_inverse_mass: float = 1e-3,
        maximum_inverse_mass: float = 1e3,
        regularization: float = 1e-3,
    ):
        steps = int(warmup_steps)
        lower = float(minimum_inverse_mass)
        upper = float(maximum_inverse_mass)
        ridge = float(regularization)
        if (
            steps < 2
            or steps > 1_000_000
            or not 0.0 < lower < upper
            or not np.isfinite(ridge)
            or ridge <= 0.0
        ):
            raise ValueError(
                "Metric adaptation requires 2 to 1000000 warmup steps and "
                "finite positive ordered regularization bounds."
            )
        self.warmup_steps = steps
        self.minimum_inverse_mass = lower
        self.maximum_inverse_mass = upper
        self.regularization = ridge
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-diagonal-group-metric-adaptation",
                "warmup_steps": steps,
                "minimum_inverse_mass": lower,
                "maximum_inverse_mass": upper,
                "regularization": ridge,
            }
        )


class PreparedSplitGroupDynamics(StrictModule):
    target: SplitGroupTarget
    inverse_mass: Array
    kick_coefficients: Array
    drift_coefficients: Array
    step_size: Array
    valid: Array
    trajectory_steps: int = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    maximum_integration_steps: int = eqx.field(static=True)
    momentum_persistence: float = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    integrator: IntegratorKind = eqx.field(static=True)
    dynamics: DynamicsKind = eqx.field(static=True)
    geometry_kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)


class SplitGroupDynamicsState(StrictModule):
    position: Array
    momentum: Array
    log_target: Array
    force_gradients: Array
    step_index: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)


class SplitGroupTransitionEvidence(StrictModule):
    accepted: Array
    acceptance_probability: Array
    energy_error: Array
    log_target_ratio: Array
    log_kinetic_ratio: Array
    log_acceptance_ratio: Array
    divergent: Array
    integration_steps: Array
    force_evaluations: Array
    membership_preserved: Array
    exact_target_correction: Array
    momentum_refresh_correlation: Array
    u_turn_detected: Array
    maximum_depth_reached: Array
    detailed_balance_residual: Array
    status: Array


class SplitGroupTransitionResult(StrictModule):
    state: SplitGroupDynamicsState
    evidence: SplitGroupTransitionEvidence
    root_key: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class SplitMetricAdaptationResult(StrictModule):
    prepared: PreparedSplitGroupDynamics
    state: SplitGroupDynamicsState
    inverse_mass_history: Array
    valid: Array
    frozen: bool = eqx.field(static=True)
    adaptation_steps: int = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class TransportedUTurnEvidence(StrictModule):
    displacement: Array
    left_velocity: Array
    right_velocity_at_left: Array
    left_inner_product: Array
    right_inner_product: Array
    turning: Array
    finite: Array


def _geometry_kind(target: SplitGroupTarget, /) -> str:
    geometry = target.geometry
    if isinstance(geometry, FlatTorusStateGeometry):
        if target.configuration_shape != target.local_coordinate_shape:
            raise ValueError("Flat-torus point and local-coordinate shapes must agree.")
        if target.reference_measure != "flat-torus":
            raise ValueError("Flat-torus dynamics requires flat-torus reference measure.")
        return "flat-torus"
    if isinstance(geometry, PointwiseStateGeometry) and isinstance(
        geometry.geometry, LieGroupStateGeometry
    ):
        group = geometry.geometry.group
        if not isinstance(group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Transported group dynamics supports only U(N) and SU(N).")
        if target.reference_measure != "product-haar":
            raise ValueError(
                "Matrix group dynamics requires product-Haar reference measure."
            )
        if target.local_coordinate_shape[-1:] != group.algebra_shape:
            raise ValueError("Local coordinate shape disagrees with the Lie algebra.")
        return "pointwise-left-lie-group"
    raise TypeError("Split group dynamics supports flat tori and pointwise U(N)/SU(N).")


def _composition(plan: SplitGroupDynamicsPlan, dtype) -> tuple[Array, Array]:
    if plan.integrator == "leapfrog":
        return jnp.asarray((0.5, 0.5), dtype=dtype), jnp.asarray((1.0,), dtype=dtype)
    coefficient = plan.omelyan_lambda
    return (
        jnp.asarray((coefficient, 1.0 - 2.0 * coefficient, coefficient), dtype=dtype),
        jnp.asarray((0.5, 0.5), dtype=dtype),
    )


def prepare_split_group_dynamics(
    target: SplitGroupTarget,
    plan: SplitGroupDynamicsPlan,
    /,
    *,
    inverse_mass: ArrayLike | None = None,
) -> PreparedSplitGroupDynamics:
    """Prepare fixed-shape integration resources and freeze production policy."""
    if not isinstance(target, SplitGroupTarget):
        raise TypeError("target must be SplitGroupTarget.")
    if not isinstance(plan, SplitGroupDynamicsPlan):
        raise TypeError("plan must be SplitGroupDynamicsPlan.")
    geometry_kind = _geometry_kind(target)
    metric = (
        jnp.ones(target.local_coordinate_shape)
        if inverse_mass is None
        else jnp.asarray(inverse_mass, dtype=float)
    )
    if metric.shape != target.local_coordinate_shape or jnp.iscomplexobj(metric):
        raise ValueError("inverse_mass must match the target local-coordinate shape.")
    valid = jnp.all(jnp.isfinite(metric) & (metric > 0.0))
    kick, drift = _composition(plan, metric.dtype)
    maximum_steps = (
        2**plan.maximum_tree_depth - 1
        if plan.dynamics == "nuts-reference"
        else plan.trajectory_steps
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-split-group-dynamics",
            "target": target.target_id,
            "geometry": target.geometry.geometry_id,
            "plan": plan.plan_id,
            "inverse_mass": np.asarray(metric),
            "kick_coefficients": np.asarray(kick),
            "drift_coefficients": np.asarray(drift),
            "maximum_integration_steps": maximum_steps,
        }
    )
    return PreparedSplitGroupDynamics(
        target,
        metric,
        kick,
        drift,
        jnp.asarray(plan.step_size, dtype=metric.dtype),
        valid,
        plan.trajectory_steps,
        plan.maximum_tree_depth,
        maximum_steps,
        plan.momentum_persistence,
        plan.divergence_threshold,
        plan.integrator,
        plan.dynamics,
        geometry_kind,
        plan.plan_id,
        prepared_id,
        True,
    )


def _value_and_forces(
    prepared: PreparedSplitGroupDynamics, position: Array, /
) -> tuple[Array, Array]:
    zero = jnp.zeros(
        prepared.target.local_coordinate_shape, dtype=jnp.real(position).dtype
    )

    def gradient(term):
        return jax.grad(
            lambda local: term(prepared.target.geometry.retract(position, local))
        )(zero)

    forces = jnp.stack(tuple(gradient(term) for term in prepared.target.force_terms))
    return prepared.target(position), forces


def initialize_split_group_dynamics_state(
    prepared: PreparedSplitGroupDynamics,
    position: ArrayLike,
    /,
    *,
    momentum: ArrayLike | None = None,
) -> SplitGroupDynamicsState:
    if not isinstance(prepared, PreparedSplitGroupDynamics):
        raise TypeError("prepared must be PreparedSplitGroupDynamics.")
    point = jnp.asarray(position)
    if point.shape != prepared.target.configuration_shape:
        raise ValueError("position shape disagrees with the prepared target.")
    impulse = (
        jnp.zeros(prepared.target.local_coordinate_shape, dtype=jnp.real(point).dtype)
        if momentum is None
        else jnp.asarray(momentum, dtype=jnp.real(point).dtype)
    )
    if impulse.shape != prepared.target.local_coordinate_shape:
        raise ValueError("momentum shape disagrees with target local coordinates.")
    value, forces = _value_and_forces(prepared, point)
    valid = (
        prepared.valid
        & prepared.target.geometry.contains(point)
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(point))
        & jnp.all(jnp.isfinite(impulse))
        & jnp.all(jnp.isfinite(forces))
    )
    point = eqx.error_if(
        point,
        ~valid,
        "Initial split group state must be finite and on the target support.",
    )
    return SplitGroupDynamicsState(
        point,
        impulse,
        value,
        forces,
        jnp.asarray(0, dtype=jnp.uint32),
        valid,
        prepared.prepared_id,
    )


def _velocity(prepared, momentum):
    return prepared.inverse_mass * momentum


def _kinetic(prepared, momentum):
    return (
        0.5
        * jnp.asarray(
            ein.contract("...,...->", momentum, _velocity(prepared, momentum))
        ).real
    )


def _sample_momentum(prepared, key):
    normal = jr.normal(
        key, prepared.target.local_coordinate_shape, dtype=prepared.inverse_mass.dtype
    )
    return normal / jnp.sqrt(prepared.inverse_mass)


def _kick(momentum, force_gradients, amount):
    return momentum + amount * jnp.sum(force_gradients, axis=0)


def _integrator_step(prepared, position, momentum, force_gradients, direction=1.0):
    epsilon = direction * prepared.step_size
    force_evaluations = jnp.asarray(0, dtype=jnp.int32)
    if prepared.integrator == "leapfrog":
        momentum = _kick(
            momentum, force_gradients, epsilon * prepared.kick_coefficients[0]
        )
        position = prepared.target.geometry.retract(
            position,
            epsilon * prepared.drift_coefficients[0] * _velocity(prepared, momentum),
        )
        value, force_gradients = _value_and_forces(prepared, position)
        momentum = _kick(
            momentum, force_gradients, epsilon * prepared.kick_coefficients[1]
        )
        force_evaluations = jnp.asarray(len(prepared.target.force_terms), dtype=jnp.int32)
    else:
        momentum = _kick(
            momentum, force_gradients, epsilon * prepared.kick_coefficients[0]
        )
        position = prepared.target.geometry.retract(
            position,
            epsilon * prepared.drift_coefficients[0] * _velocity(prepared, momentum),
        )
        value, force_gradients = _value_and_forces(prepared, position)
        momentum = _kick(
            momentum, force_gradients, epsilon * prepared.kick_coefficients[1]
        )
        position = prepared.target.geometry.retract(
            position,
            epsilon * prepared.drift_coefficients[1] * _velocity(prepared, momentum),
        )
        value, force_gradients = _value_and_forces(prepared, position)
        momentum = _kick(
            momentum, force_gradients, epsilon * prepared.kick_coefficients[2]
        )
        force_evaluations = jnp.asarray(
            2 * len(prepared.target.force_terms), dtype=jnp.int32
        )
    member = prepared.target.geometry.contains(position)
    finite = (
        member
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(position))
        & jnp.all(jnp.isfinite(momentum))
        & jnp.all(jnp.isfinite(force_gradients))
    )
    return position, momentum, value, force_gradients, finite, force_evaluations


def split_integrator_trajectory(
    prepared: PreparedSplitGroupDynamics,
    position: Array,
    momentum: Array,
    force_gradients: Array,
    /,
    *,
    steps: int | None = None,
    direction: int = 1,
):
    """Apply an exposed reversible palindromic trajectory for audit and reuse."""
    count = prepared.trajectory_steps if steps is None else int(steps)
    if (
        count < 0
        or count > prepared.maximum_integration_steps
        or direction not in (-1, 1)
    ):
        raise ValueError("Trajectory count or direction exceeds prepared resources.")
    q, p, value, forces = position, momentum, prepared.target(position), force_gradients
    active = jnp.asarray(True)
    evaluations = jnp.asarray(0, dtype=jnp.int32)
    used = jnp.asarray(0, dtype=jnp.int32)
    for _ in range(count):
        next_q, next_p, next_value, next_forces, finite, calls = _integrator_step(
            prepared, q, p, forces, float(direction)
        )
        commit = active & finite
        q = jnp.where(commit, next_q, q)
        p = jnp.where(commit, next_p, p)
        value = jnp.where(commit, next_value, value)
        forces = jnp.where(commit, next_forces, forces)
        used = used + active.astype(jnp.int32)
        evaluations = evaluations + jnp.where(active, calls, 0)
        active = commit
    return q, p, value, forces, active, used, evaluations


def transported_group_u_turn(
    prepared: PreparedSplitGroupDynamics,
    left_position: ArrayLike,
    right_position: ArrayLike,
    left_momentum: ArrayLike,
    right_momentum: ArrayLike,
    /,
) -> TransportedUTurnEvidence:
    """Evaluate the generalized NUTS criterion in the left tangent space."""
    if not isinstance(prepared, PreparedSplitGroupDynamics):
        raise TypeError("prepared must be PreparedSplitGroupDynamics.")
    left = jnp.asarray(left_position)
    right = jnp.asarray(right_position)
    left_p = jnp.asarray(left_momentum)
    right_p = jnp.asarray(right_momentum)
    displacement = prepared.target.geometry.inverse_retract(left, right)
    left_velocity = _velocity(prepared, left_p)
    right_velocity = prepared.target.geometry.transport_tangent(
        right, left, _velocity(prepared, right_p)
    )
    left_inner = jnp.asarray(ein.contract("...,...->", displacement, left_velocity)).real
    right_inner = jnp.asarray(
        ein.contract("...,...->", displacement, right_velocity)
    ).real
    finite = (
        jnp.all(jnp.isfinite(displacement))
        & jnp.isfinite(left_inner)
        & jnp.isfinite(right_inner)
    )
    turning = finite & ((left_inner <= 0.0) | (right_inner <= 0.0))
    return TransportedUTurnEvidence(
        displacement,
        left_velocity,
        right_velocity,
        left_inner,
        right_inner,
        turning,
        finite,
    )


def _ghmc_transition(prepared, state, key):
    momentum_key = derive_key(key, _MOMENTUM_ADDRESS, state.step_index)
    accept_key = derive_key(key, _ACCEPT_ADDRESS, state.step_index)
    fresh = _sample_momentum(prepared, momentum_key)
    rho = jnp.asarray(prepared.momentum_persistence, dtype=fresh.dtype)
    refreshed = rho * state.momentum + jnp.sqrt(1.0 - rho**2) * fresh
    q, p, value, forces, finite, used, evaluations = split_integrator_trajectory(
        prepared, state.position, refreshed, state.force_gradients
    )
    proposal_momentum = -p
    error = (-value + _kinetic(prepared, proposal_momentum)) - (
        -state.log_target + _kinetic(prepared, refreshed)
    )
    log_target_ratio = value - state.log_target
    log_kinetic_ratio = _kinetic(prepared, proposal_momentum) - _kinetic(
        prepared, refreshed
    )
    log_acceptance_ratio = log_target_ratio - log_kinetic_ratio
    divergent = (
        ~finite | ~jnp.isfinite(error) | (jnp.abs(error) > prepared.divergence_threshold)
    )
    log_accept = jnp.minimum(log_acceptance_ratio, 0.0)
    accepted = state.valid & ~divergent & (jnp.log(jr.uniform(accept_key)) < log_accept)
    position = jnp.where(accepted, q, state.position)
    momentum = jnp.where(accepted, proposal_momentum, -refreshed)
    log_target = jnp.where(accepted, value, state.log_target)
    next_forces = jnp.where(accepted, forces, state.force_gradients)
    probability = jnp.where(
        ~divergent & jnp.isfinite(log_accept), jnp.exp(log_accept), 0.0
    )
    membership = prepared.target.geometry.contains(position)
    status = jnp.where(
        ~membership,
        SplitGroupDynamicsStatus.GROUP_MEMBERSHIP_FAILURE,
        jnp.where(
            divergent,
            SplitGroupDynamicsStatus.NONFINITE_OR_DIVERGENT_TRAJECTORY,
            jnp.where(
                ~state.valid,
                SplitGroupDynamicsStatus.INVALID_INPUT_STATE,
                SplitGroupDynamicsStatus.SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return (
        position,
        momentum,
        log_target,
        next_forces,
        (
            accepted,
            probability,
            error,
            log_target_ratio,
            log_kinetic_ratio,
            log_acceptance_ratio,
            divergent,
            used,
            evaluations,
            membership,
            ~divergent,
            rho,
            jnp.asarray(False),
            jnp.asarray(False),
            log_acceptance_ratio - (log_target_ratio - log_kinetic_ratio),
            status,
        ),
    )


def _nuts_reference_transition(prepared, state, key):
    """Finite slice-weighted transported NUTS reference trajectory."""
    momentum_key = derive_key(key, _MOMENTUM_ADDRESS, state.step_index)
    tree_key = derive_key(key, _NUTS_ADDRESS, state.step_index)
    momentum = _sample_momentum(prepared, momentum_key)
    initial_joint = state.log_target - _kinetic(prepared, momentum)
    slice_level = initial_joint + jnp.log(
        jr.uniform(derive_key(tree_key, _NUTS_ADDRESS, 0))
    )
    left_q = right_q = candidate_q = state.position
    left_p = right_p = candidate_p = momentum
    left_f = right_f = candidate_f = state.force_gradients
    candidate_value = state.log_target
    valid_count = jnp.asarray(1, dtype=jnp.int32)
    active = jnp.asarray(True)
    turning = jnp.asarray(False)
    divergent = jnp.asarray(False)
    used = jnp.asarray(0, dtype=jnp.int32)
    evaluations = jnp.asarray(0, dtype=jnp.int32)
    for depth in range(prepared.maximum_tree_depth):
        direction_key = derive_key(tree_key, _NUTS_ADDRESS, 1, depth)
        direction = jnp.where(jr.bernoulli(direction_key), 1, -1)
        for offset in range(2**depth):
            start_q = jnp.where(direction > 0, right_q, left_q)
            start_p = jnp.where(direction > 0, right_p, left_p)
            start_f = jnp.where(direction > 0, right_f, left_f)
            (
                next_q,
                next_p,
                next_value,
                next_f,
                finite,
                calls,
            ) = jax.lax.cond(
                direction > 0,
                lambda _: _integrator_step(prepared, start_q, start_p, start_f, 1.0),
                lambda _: _integrator_step(prepared, start_q, start_p, start_f, -1.0),
                operand=None,
            )
            joint = next_value - _kinetic(prepared, next_p)
            leaf_divergent = (
                ~finite
                | ~jnp.isfinite(joint)
                | (initial_joint - joint > prepared.divergence_threshold)
            )
            leaf_valid = active & ~leaf_divergent & (joint >= slice_level)
            next_count = valid_count + leaf_valid.astype(jnp.int32)
            choose_key = derive_key(tree_key, _NUTS_ADDRESS, 2, depth, offset)
            choose = leaf_valid & (
                jr.uniform(choose_key) < 1.0 / next_count.astype(float)
            )
            candidate_q = jnp.where(choose, next_q, candidate_q)
            candidate_p = jnp.where(choose, next_p, candidate_p)
            candidate_value = jnp.where(choose, next_value, candidate_value)
            candidate_f = jnp.where(choose, next_f, candidate_f)
            valid_count = next_count
            extend = active & ~leaf_divergent
            left_q = jnp.where(extend & (direction < 0), next_q, left_q)
            left_p = jnp.where(extend & (direction < 0), next_p, left_p)
            left_f = jnp.where(extend & (direction < 0), next_f, left_f)
            right_q = jnp.where(extend & (direction > 0), next_q, right_q)
            right_p = jnp.where(extend & (direction > 0), next_p, right_p)
            right_f = jnp.where(extend & (direction > 0), next_f, right_f)
            turn = transported_group_u_turn(
                prepared, left_q, right_q, left_p, right_p
            ).turning
            turning = turning | (extend & turn)
            divergent = divergent | (active & leaf_divergent)
            used = used + active.astype(jnp.int32)
            evaluations = evaluations + jnp.where(active, calls, 0)
            active = extend & ~turn
    accepted = state.valid & (valid_count > 1) & ~divergent
    position = jnp.where(accepted, candidate_q, state.position)
    next_momentum = jnp.where(accepted, -candidate_p, -momentum)
    value = jnp.where(accepted, candidate_value, state.log_target)
    forces = jnp.where(accepted, candidate_f, state.force_gradients)
    probability = jnp.where(accepted, 1.0, 0.0)
    energy_error = (-(candidate_value - _kinetic(prepared, candidate_p))) - (
        -initial_joint
    )
    log_target_ratio = candidate_value - state.log_target
    log_kinetic_ratio = _kinetic(prepared, candidate_p) - _kinetic(prepared, momentum)
    log_acceptance_ratio = log_target_ratio - log_kinetic_ratio
    maximum = used >= prepared.maximum_integration_steps
    membership = prepared.target.geometry.contains(position)
    status = jnp.where(
        ~membership,
        SplitGroupDynamicsStatus.GROUP_MEMBERSHIP_FAILURE,
        jnp.where(
            divergent,
            SplitGroupDynamicsStatus.NONFINITE_OR_DIVERGENT_TRAJECTORY,
            jnp.where(
                ~state.valid,
                SplitGroupDynamicsStatus.INVALID_INPUT_STATE,
                SplitGroupDynamicsStatus.SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return (
        position,
        next_momentum,
        value,
        forces,
        (
            accepted,
            probability,
            energy_error,
            log_target_ratio,
            log_kinetic_ratio,
            log_acceptance_ratio,
            divergent,
            used,
            evaluations,
            membership,
            jnp.asarray(True),
            jnp.asarray(0.0),
            turning,
            maximum,
            log_acceptance_ratio - (log_target_ratio - log_kinetic_ratio),
            status,
        ),
    )


def split_group_transition(
    prepared: PreparedSplitGroupDynamics,
    state: SplitGroupDynamicsState,
    /,
    *,
    key: Key[Array, ""],
) -> SplitGroupTransitionResult:
    """Run one frozen GHMC or finite transported-NUTS transition."""
    if not isinstance(prepared, PreparedSplitGroupDynamics):
        raise TypeError("prepared must be PreparedSplitGroupDynamics.")
    if not isinstance(state, SplitGroupDynamicsState):
        raise TypeError("state must be SplitGroupDynamicsState.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("Dynamics state belongs to another prepared policy.")
    if prepared.dynamics == "ghmc":
        position, momentum, value, forces, raw = _ghmc_transition(prepared, state, key)
        claim = "exact-corrected-generalized-compact-group-hmc"
    else:
        position, momentum, value, forces, raw = _nuts_reference_transition(
            prepared, state, key
        )
        claim = "finite-transported-group-nuts-reference"
    valid = (
        state.valid
        & prepared.valid
        & raw[9]
        & jnp.isfinite(value)
        & jnp.all(jnp.isfinite(forces))
    )
    successor = SplitGroupDynamicsState(
        position,
        momentum,
        value,
        forces,
        state.step_index + jnp.asarray(1, dtype=jnp.uint32),
        valid,
        state.prepared_id,
    )
    return SplitGroupTransitionResult(
        successor,
        SplitGroupTransitionEvidence(*raw),
        jnp.asarray(key),
        prepared.plan_id,
        prepared.prepared_id,
        prepared.target.reference_measure,
        claim,
    )


def adapt_split_group_metric(
    prepared: PreparedSplitGroupDynamics,
    state: SplitGroupDynamicsState,
    plan: SplitMetricAdaptationPlan,
    /,
    *,
    key: Key[Array, ""],
) -> SplitMetricAdaptationResult:
    """Run finite warmup, estimate a diagonal metric, then freeze a new executable."""
    if not isinstance(prepared, PreparedSplitGroupDynamics):
        raise TypeError("prepared must be PreparedSplitGroupDynamics.")
    if not isinstance(state, SplitGroupDynamicsState):
        raise TypeError("state must be SplitGroupDynamicsState.")
    if not isinstance(plan, SplitMetricAdaptationPlan):
        raise TypeError("plan must be SplitMetricAdaptationPlan.")
    if prepared.dynamics != "ghmc":
        raise ValueError("Finite metric adaptation is defined only for GHMC warmup.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError("Dynamics state belongs to another prepared policy.")
    current = state
    anchor = state.position
    count = jnp.asarray(0, dtype=jnp.int32)
    mean = jnp.zeros(
        prepared.target.local_coordinate_shape, dtype=prepared.inverse_mass.dtype
    )
    second = jnp.zeros_like(mean)
    history = []
    for warmup_index in range(plan.warmup_steps):
        transition = split_group_transition(
            prepared,
            current,
            key=derive_key(key, _MOMENTUM_ADDRESS, warmup_index),
        )
        current = transition.state
        coordinates = prepared.target.geometry.inverse_retract(anchor, current.position)
        count = count + 1
        delta = coordinates - mean
        mean = mean + delta / count.astype(mean.dtype)
        second = second + delta * (coordinates - mean)
        variance = second / jnp.maximum(count - 1, 1).astype(mean.dtype)
        estimate = jnp.clip(
            variance + plan.regularization,
            plan.minimum_inverse_mass,
            plan.maximum_inverse_mass,
        )
        history.append(estimate)
    inverse_mass_history = jnp.stack(tuple(history))
    frozen_metric = inverse_mass_history[-1]
    source_plan = SplitGroupDynamicsPlan(
        step_size=float(prepared.step_size),
        trajectory_steps=prepared.trajectory_steps,
        maximum_tree_depth=prepared.maximum_tree_depth,
        momentum_persistence=prepared.momentum_persistence,
        divergence_threshold=prepared.divergence_threshold,
        integrator=prepared.integrator,
        dynamics=prepared.dynamics,
        omelyan_lambda=float(prepared.kick_coefficients[0])
        if prepared.integrator == "omelyan"
        else 0.1931833275037836,
    )
    frozen = prepare_split_group_dynamics(
        prepared.target, source_plan, inverse_mass=frozen_metric
    )
    rebound_momentum = current.momentum * jnp.sqrt(
        prepared.inverse_mass / frozen.inverse_mass
    )
    valid = (
        current.valid
        & frozen.valid
        & jnp.all(jnp.isfinite(inverse_mass_history))
        & jnp.all(jnp.isfinite(rebound_momentum))
    )
    rebound = SplitGroupDynamicsState(
        current.position,
        rebound_momentum,
        current.log_target,
        current.force_gradients,
        current.step_index,
        valid,
        frozen.prepared_id,
    )
    return SplitMetricAdaptationResult(
        frozen,
        rebound,
        inverse_mass_history,
        valid,
        True,
        plan.warmup_steps,
        "finite-warmup-diagonal-metric-production-frozen",
    )


__all__ = [
    "PreparedSplitGroupDynamics",
    "SplitGroupDynamicsPlan",
    "SplitGroupDynamicsState",
    "SplitGroupTarget",
    "SplitGroupTransitionEvidence",
    "SplitGroupTransitionResult",
    "SplitGroupDynamicsStatus",
    "SplitMetricAdaptationPlan",
    "SplitMetricAdaptationResult",
    "TransportedUTurnEvidence",
    "adapt_split_group_metric",
    "initialize_split_group_dynamics_state",
    "prepare_split_group_dynamics",
    "split_group_transition",
    "split_integrator_trajectory",
    "transported_group_u_turn",
]

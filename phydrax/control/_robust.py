#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Robust synthesis, tube/chance constraints, safety filters, and estimation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
import scipy.stats as scipy_stats
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._lqr import discrete_lqr


@dataclass(frozen=True, slots=True)
class RobustStateFeedbackResult:
    gain: Array
    riccati: Array
    closed_loop_eigenvalues: Array
    performance: Array
    stable: bool
    certified: bool
    synthesis_id: str


def h2_state_feedback(
    dynamics: ArrayLike,
    control: ArrayLike,
    state_cost: ArrayLike,
    control_cost: ArrayLike,
    disturbance: ArrayLike,
    /,
) -> RobustStateFeedbackResult:
    """Discrete H2 state feedback using the native infinite-horizon LQR owner."""

    result = discrete_lqr(dynamics, control, state_cost, control_cost)
    gain = jnp.asarray(result.feedback_gain)
    riccati = jnp.asarray(result.value_matrix)
    closed = jnp.asarray(dynamics) + jnp.asarray(control) @ gain
    eigenvalues = jnp.linalg.eigvals(closed)
    disturbance_ = jnp.asarray(disturbance)
    performance = jnp.real(jnp.trace(disturbance_.T @ riccati @ disturbance_))
    stable = bool(np.all(np.abs(np.asarray(eigenvalues)) < 1.0))
    payload = {
        "kind": "h2-state-feedback",
        "dynamics_shape": tuple(closed.shape),
        "control_shape": tuple(gain.shape),
    }
    return RobustStateFeedbackResult(
        gain,
        riccati,
        eigenvalues,
        performance,
        stable,
        bool(stable and result.valid),
        canonical_fingerprint(payload),
    )


def hinfinity_state_feedback(
    dynamics: ArrayLike,
    control: ArrayLike,
    disturbance: ArrayLike,
    state_cost: ArrayLike,
    control_cost: ArrayLike,
    gamma: float,
    /,
) -> RobustStateFeedbackResult:
    """Continuous-time bounded-real state-feedback synthesis for fixed gamma."""

    a = np.asarray(dynamics, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    w = np.asarray(disturbance, dtype=np.float64)
    q = np.asarray(state_cost, dtype=np.float64)
    r = np.asarray(control_cost, dtype=np.float64)
    gamma_ = float(gamma)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("H-infinity dynamics must be square.")
    if b.shape[0] != a.shape[0] or w.shape[0] != a.shape[0]:
        raise ValueError("H-infinity input matrices must share the state dimension.")
    if not np.isfinite(gamma_) or gamma_ <= 0.0:
        raise ValueError("gamma must be finite and positive.")
    combined = np.concatenate((b, w), axis=1)
    game_metric = scipy_linalg.block_diag(r, -(gamma_**2) * np.eye(w.shape[1]))
    riccati = scipy_linalg.solve_continuous_are(a, combined, q, game_metric)
    gain = np.linalg.solve(r, b.T @ riccati)
    closed = a - b @ gain
    eigenvalues = np.linalg.eigvals(closed)
    bounded_real = (
        a.T @ riccati
        + riccati @ a
        + q
        - riccati @ b @ np.linalg.solve(r, b.T) @ riccati
        + riccati @ w @ w.T @ riccati / gamma_**2
    )
    residual = np.linalg.norm(bounded_real)
    stable = bool(np.all(np.real(eigenvalues) < 0.0))
    tolerance = 1.0e3 * np.finfo(np.float64).eps * max(1.0, np.linalg.norm(q))
    payload = {
        "kind": "hinfinity-state-feedback",
        "gamma": gamma_,
        "state_dimension": a.shape[0],
        "control_dimension": b.shape[1],
        "disturbance_dimension": w.shape[1],
    }
    return RobustStateFeedbackResult(
        jnp.asarray(gain),
        jnp.asarray(riccati),
        jnp.asarray(eigenvalues),
        jnp.asarray(gamma_),
        stable,
        bool(stable and np.isfinite(residual) and residual <= tolerance),
        canonical_fingerprint(payload),
    )


@dataclass(frozen=True, slots=True)
class TubeMPCPlan:
    feedback_gain: Array
    error_radius: Array
    tightened_state_lower: Array
    tightened_state_upper: Array
    tightened_control_lower: Array
    tightened_control_upper: Array
    plan_id: str

    def control(self, nominal_control: ArrayLike, deviation: ArrayLike, /) -> Array:
        value = jnp.asarray(nominal_control) - self.feedback_gain @ jnp.asarray(deviation)
        return jnp.clip(value, self.tightened_control_lower, self.tightened_control_upper)


def prepare_tube_mpc(
    dynamics: ArrayLike,
    control: ArrayLike,
    feedback_gain: ArrayLike,
    disturbance_radius: ArrayLike,
    state_lower: ArrayLike,
    state_upper: ArrayLike,
    control_lower: ArrayLike,
    control_upper: ArrayLike,
    /,
    *,
    maximum_terms: int = 256,
    tolerance: float = 1.0e-12,
) -> TubeMPCPlan:
    a = np.asarray(dynamics, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    gain = np.asarray(feedback_gain, dtype=np.float64)
    radius = np.asarray(disturbance_radius, dtype=np.float64)
    closed = a - b @ gain
    if np.any(radius < 0.0) or not np.all(np.isfinite(radius)):
        raise ValueError("disturbance_radius must be finite and non-negative.")
    accumulated = np.zeros_like(radius)
    propagated = radius.copy()
    for _ in range(int(maximum_terms)):
        accumulated += propagated
        propagated = np.abs(closed) @ propagated
        if np.max(propagated) <= tolerance:
            break
    else:
        raise ValueError("Tube invariant-radius series did not converge.")
    control_radius = np.abs(gain) @ accumulated
    state_lower_ = np.asarray(state_lower, dtype=np.float64) + accumulated
    state_upper_ = np.asarray(state_upper, dtype=np.float64) - accumulated
    control_lower_ = np.asarray(control_lower, dtype=np.float64) + control_radius
    control_upper_ = np.asarray(control_upper, dtype=np.float64) - control_radius
    if np.any(state_lower_ > state_upper_) or np.any(control_lower_ > control_upper_):
        raise ValueError("Robust tube tightening makes the constraints infeasible.")
    payload = {
        "kind": "tube-mpc-plan",
        "closed_loop": closed.tolist(),
        "error_radius": accumulated.tolist(),
    }
    return TubeMPCPlan(
        jnp.asarray(gain),
        jnp.asarray(accumulated),
        jnp.asarray(state_lower_),
        jnp.asarray(state_upper_),
        jnp.asarray(control_lower_),
        jnp.asarray(control_upper_),
        canonical_fingerprint(payload),
    )


@dataclass(frozen=True, slots=True)
class GaussianChanceConstraint:
    coefficients: Array
    bound: Array
    probability: float
    quantile: float

    def margin(self, mean: ArrayLike, covariance: ArrayLike, /) -> Array:
        mean_ = jnp.asarray(mean)
        covariance_ = jnp.asarray(covariance)
        variance = self.coefficients @ covariance_ @ self.coefficients
        return (
            self.bound
            - self.coefficients @ mean_
            - self.quantile * jnp.sqrt(jnp.maximum(variance, 0.0))
        )


def gaussian_chance_constraint(
    coefficients: ArrayLike,
    bound: ArrayLike,
    probability: float,
    /,
) -> GaussianChanceConstraint:
    probability_ = float(probability)
    if not 0.5 < probability_ < 1.0:
        raise ValueError("One-sided chance probability must lie in (0.5, 1).")
    return GaussianChanceConstraint(
        jnp.asarray(coefficients),
        jnp.asarray(bound),
        probability_,
        float(scipy_stats.norm.ppf(probability_)),
    )


@dataclass(frozen=True, slots=True)
class MovingHorizonEstimate:
    states: Array
    residual_norm: Array
    normal_condition: Array
    successful: Array


def linear_moving_horizon_estimate(
    dynamics: ArrayLike,
    observation: ArrayLike,
    measurements: ArrayLike,
    prior_mean: ArrayLike,
    prior_precision: ArrayLike,
    process_precision: ArrayLike,
    observation_precision: ArrayLike,
    /,
) -> MovingHorizonEstimate:
    """Solve one linear-Gaussian moving-horizon problem by its block normal system."""

    a = jnp.asarray(dynamics)
    c = jnp.asarray(observation)
    values = jnp.asarray(measurements)
    prior = jnp.asarray(prior_mean)
    horizon = values.shape[0]
    state_size = a.shape[0]
    if a.shape != (state_size, state_size) or c.shape[1] != state_size:
        raise ValueError("Moving-horizon matrices have incompatible state dimensions.")
    rows = []
    targets = []
    prior_factor = jnp.linalg.cholesky(jnp.asarray(prior_precision))
    first = (
        jnp.zeros((state_size, horizon * state_size)).at[:, :state_size].set(prior_factor)
    )
    rows.append(first)
    targets.append(prior_factor @ prior)
    process_factor = jnp.linalg.cholesky(jnp.asarray(process_precision))
    for index in range(1, horizon):
        row = jnp.zeros((state_size, horizon * state_size))
        row = row.at[:, (index - 1) * state_size : index * state_size].set(
            -process_factor @ a
        )
        row = row.at[:, index * state_size : (index + 1) * state_size].set(process_factor)
        rows.append(row)
        targets.append(jnp.zeros((state_size,), dtype=values.dtype))
    observation_factor = jnp.linalg.cholesky(jnp.asarray(observation_precision))
    for index in range(horizon):
        row = jnp.zeros((c.shape[0], horizon * state_size))
        row = row.at[:, index * state_size : (index + 1) * state_size].set(
            observation_factor @ c
        )
        rows.append(row)
        targets.append(observation_factor @ values[index])
    design = jnp.concatenate(rows, axis=0)
    target = jnp.concatenate(targets)
    normal = design.T @ design
    right = design.T @ target
    solution = jnp.linalg.solve(normal, right)
    residual = design @ solution - target
    condition = jnp.linalg.cond(normal)
    successful = jnp.all(jnp.isfinite(solution)) & jnp.isfinite(condition)
    return MovingHorizonEstimate(
        solution.reshape((horizon, state_size)),
        jnp.linalg.norm(residual),
        condition,
        successful,
    )


@dataclass(frozen=True, slots=True)
class SafetyFilterResult:
    control: Array
    maximum_violation: Array
    iterations: int
    successful: Array


def project_control_halfspaces(
    nominal_control: ArrayLike,
    coefficients: ArrayLike,
    bounds: ArrayLike,
    /,
    *,
    iterations: int = 32,
) -> SafetyFilterResult:
    """Dykstra projection onto C u <= d for CBF/CLF control constraints."""

    nominal = jnp.asarray(nominal_control)
    matrix = jnp.asarray(coefficients)
    right = jnp.asarray(bounds)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != nominal.size
        or right.shape != (matrix.shape[0],)
    ):
        raise ValueError("Safety halfspaces do not match the control dimension.")
    corrections = jnp.zeros_like(matrix)
    control = nominal
    for _ in range(int(iterations)):
        for index in range(matrix.shape[0]):
            normal = matrix[index]
            shifted = control + corrections[index]
            violation = normal @ shifted - right[index]
            scale = jnp.vdot(normal, normal)
            adjustment = (
                jnp.maximum(violation, 0.0) * normal / jnp.maximum(scale, 1.0e-30)
            )
            projected = shifted - adjustment
            corrections = corrections.at[index].set(shifted - projected)
            control = projected
    maximum = jnp.max(jnp.maximum(matrix @ control - right, 0.0), initial=0.0)
    return SafetyFilterResult(
        control,
        maximum,
        int(iterations),
        jnp.isfinite(maximum) & (maximum <= 1.0e-8),
    )


@dataclass(frozen=True, slots=True)
class LinearReachableBox:
    center: Array
    radius: Array


def propagate_linear_reachable_box(
    dynamics: ArrayLike,
    control: ArrayLike,
    state: LinearReachableBox,
    control_box: LinearReachableBox,
    disturbance_radius: ArrayLike,
    /,
) -> LinearReachableBox:
    a = jnp.asarray(dynamics)
    b = jnp.asarray(control)
    disturbance = jnp.asarray(disturbance_radius)
    return LinearReachableBox(
        a @ state.center + b @ control_box.center,
        jnp.abs(a) @ state.radius + jnp.abs(b) @ control_box.radius + disturbance,
    )


__all__ = [
    "GaussianChanceConstraint",
    "LinearReachableBox",
    "MovingHorizonEstimate",
    "RobustStateFeedbackResult",
    "SafetyFilterResult",
    "TubeMPCPlan",
    "gaussian_chance_constraint",
    "h2_state_feedback",
    "hinfinity_state_feedback",
    "linear_moving_horizon_estimate",
    "prepare_tube_mpc",
    "project_control_halfspaces",
    "propagate_linear_reachable_box",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded complex Langevin execution with gauge cooling and drift-tail gates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ... import ein
from ..._fingerprint import canonical_fingerprint
from ..._sampling._addressing import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_NOISE_ADDRESS = SampleAddress(
    "sign-problem", "complex-langevin", target="real-noise", role="transition"
)


class ComplexLangevinStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_ACTION_OR_DRIFT = 1
    STATE_NORM_EXCEEDED = 2
    GAUGE_COOLING_FAILED = 3
    DRIFT_TAIL_REJECTED = 4


class GaugeCoolingPlan(StrictModule, NonTrainableState):
    """Fixed gauge-orbit descent budget and monotonicity contract."""

    iterations: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    monotonicity_tolerance: float = eqx.field(static=True)
    maximum_state_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        iterations: int,
        step_size: float,
        monotonicity_tolerance: float = 1e-10,
        maximum_state_size: int = 1_000_000,
    ):
        iterations_ = int(iterations)
        step = float(step_size)
        tolerance = float(monotonicity_tolerance)
        maximum = int(maximum_state_size)
        if iterations_ <= 0:
            raise ValueError("Gauge cooling iterations must be positive.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("Gauge cooling step_size must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("monotonicity_tolerance must be finite and non-negative.")
        if maximum <= 0:
            raise ValueError("maximum_state_size must be positive.")
        self.iterations = iterations_
        self.step_size = step
        self.monotonicity_tolerance = tolerance
        self.maximum_state_size = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gauge-cooling-plan",
                "iterations": iterations_,
                "step_size": step,
                "monotonicity_tolerance": tolerance,
                "maximum_state_size": maximum,
            }
        )


class PreparedGaugeCooling(StrictModule, NonTrainableState):
    """Gauge-orbit gradient, retraction, and norm frozen before execution."""

    gauge_gradient: Callable[[Array], Array] = eqx.field(static=True)
    gauge_retract: Callable[[Array, Array], Array] = eqx.field(static=True)
    unitarity_norm: Callable[[Array], Array] = eqx.field(static=True)
    plan: GaugeCoolingPlan
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    cooling_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class GaugeCoolingEvidence(StrictModule):
    initial_norm: Array
    final_norm: Array
    accepted_updates: Array
    rejected_updates: Array
    valid: Array


def prepare_gauge_cooling(
    plan: GaugeCoolingPlan,
    gauge_gradient: Callable[[Array], Array],
    gauge_retract: Callable[[Array, Array], Array],
    unitarity_norm: Callable[[Array], Array],
    /,
    *,
    configuration_shape: Sequence[int],
    cooling_id: str,
) -> PreparedGaugeCooling:
    """Bind a genuine gauge-orbit retraction to a finite cooling plan."""
    if not isinstance(plan, GaugeCoolingPlan):
        raise TypeError("plan must be GaugeCoolingPlan.")
    if not all(
        callable(item) for item in (gauge_gradient, gauge_retract, unitarity_norm)
    ):
        raise TypeError("Gauge cooling operations must be callable.")
    shape = tuple(configuration_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("configuration_shape must contain positive dimensions.")
    if prod(shape) > plan.maximum_state_size:
        raise ValueError("Gauge cooling state exceeds maximum_state_size.")
    identifier = str(cooling_id)
    if not identifier:
        raise ValueError("cooling_id must be non-empty.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-gauge-cooling",
            "plan": plan.plan_id,
            "configuration_shape": list(shape),
            "cooling": identifier,
        }
    )
    return PreparedGaugeCooling(
        gauge_gradient=gauge_gradient,
        gauge_retract=gauge_retract,
        unitarity_norm=unitarity_norm,
        plan=plan,
        configuration_shape=shape,
        cooling_id=identifier,
        prepared_id=prepared_id,
    )


class ComplexLangevinPlan(StrictModule, NonTrainableState):
    """Static finite trajectory, output, state, and drift-tail capacities."""

    num_steps: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    thinning: int = eqx.field(static=True)
    tail_window: int = eqx.field(static=True)
    drift_tail_threshold: float = eqx.field(static=True)
    maximum_tail_probability: float = eqx.field(static=True)
    maximum_state_norm: float = eqx.field(static=True)
    maximum_state_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int,
        step_size: float,
        burn_in: int = 0,
        thinning: int = 1,
        tail_window: int = 128,
        drift_tail_threshold: float = 100.0,
        maximum_tail_probability: float = 0.01,
        maximum_state_norm: float = 1e6,
        maximum_state_size: int = 1_000_000,
    ):
        steps = int(num_steps)
        step = float(step_size)
        burn = int(burn_in)
        thin = int(thinning)
        window = int(tail_window)
        threshold = float(drift_tail_threshold)
        tail_probability = float(maximum_tail_probability)
        state_norm = float(maximum_state_norm)
        state_size = int(maximum_state_size)
        if steps <= 0 or burn < 0 or burn >= steps or thin <= 0:
            raise ValueError("Complex Langevin step/burn-in/thinning counts are invalid.")
        if window <= 0 or window > steps:
            raise ValueError("tail_window must lie between one and num_steps.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("drift_tail_threshold must be finite and positive.")
        if not np.isfinite(tail_probability) or not 0.0 <= tail_probability <= 1.0:
            raise ValueError("maximum_tail_probability must lie in [0, 1].")
        if not np.isfinite(state_norm) or state_norm <= 0.0 or state_size <= 0:
            raise ValueError("Complex Langevin state resource limits are invalid.")
        self.num_steps = steps
        self.step_size = step
        self.burn_in = burn
        self.thinning = thin
        self.tail_window = window
        self.drift_tail_threshold = threshold
        self.maximum_tail_probability = tail_probability
        self.maximum_state_norm = state_norm
        self.maximum_state_size = state_size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "complex-langevin-plan",
                "num_steps": steps,
                "step_size": step,
                "burn_in": burn,
                "thinning": thin,
                "tail_window": window,
                "drift_tail_threshold": threshold,
                "maximum_tail_probability": tail_probability,
                "maximum_state_norm": state_norm,
                "maximum_state_size": state_size,
            }
        )

    @property
    def output_count(self) -> int:
        return len(range(self.burn_in, self.num_steps, self.thinning))


class PreparedComplexLangevin(StrictModule, NonTrainableState):
    action: Callable[[Array], Array] = eqx.field(static=True)
    plan: ComplexLangevinPlan
    cooling: PreparedGaugeCooling | None
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ComplexLangevinDiagnostics(StrictModule):
    status: Array
    maximum_drift_norm: Array
    drift_tail_probability: Array
    tail_mean_drift_norm: Array
    first_invalid_step: Array
    cooling_initial_norm: Array
    cooling_final_norm: Array
    cooling_accepted_updates: Array
    cooling_rejected_updates: Array
    finite_steps: Array


class ComplexLangevinResult(StrictModule):
    samples: Array
    action_values: Array
    drift_norms: Array
    sample_step_indices: Array
    final_state: Array
    diagnostics: ComplexLangevinDiagnostics
    successful: Array
    root_key: Array
    action_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class ComplexLangevinControlResult(StrictModule):
    orders: Array
    schwinger_dyson_residuals: Array
    maximum_residual: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_complex_langevin(
    plan: ComplexLangevinPlan,
    action: Callable[[Array], Array],
    /,
    *,
    configuration_shape: Sequence[int],
    action_id: str,
    cooling: PreparedGaugeCooling | None = None,
) -> PreparedComplexLangevin:
    """Freeze the holomorphic action and optional gauge-cooling runtime."""
    if not isinstance(plan, ComplexLangevinPlan):
        raise TypeError("plan must be ComplexLangevinPlan.")
    if not callable(action):
        raise TypeError("action must be callable.")
    shape = tuple(configuration_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("configuration_shape must contain positive dimensions.")
    if prod(shape) > plan.maximum_state_size:
        raise ValueError("Complex Langevin state exceeds maximum_state_size.")
    identifier = str(action_id)
    if not identifier:
        raise ValueError("action_id must be non-empty.")
    if cooling is not None:
        if not isinstance(cooling, PreparedGaugeCooling):
            raise TypeError("cooling must be PreparedGaugeCooling or None.")
        if cooling.configuration_shape != shape:
            raise ValueError(
                "Gauge cooling configuration shape does not match the action."
            )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-complex-langevin",
            "plan": plan.plan_id,
            "action": identifier,
            "configuration_shape": list(shape),
            "cooling": None if cooling is None else cooling.prepared_id,
        }
    )
    return PreparedComplexLangevin(
        action=action,
        plan=plan,
        cooling=cooling,
        configuration_shape=shape,
        action_id=identifier,
        prepared_id=prepared_id,
    )


def _action_value(runtime: PreparedComplexLangevin, state: Array, /) -> Array:
    value = jnp.asarray(runtime.action(state))
    if value.shape != () or not jnp.iscomplexobj(value):
        raise TypeError("Complex Langevin action must return one complex scalar.")
    return value


def _action_drift(runtime: PreparedComplexLangevin, state: Array, /) -> Array:
    return jax.grad(lambda value: _action_value(runtime, value), holomorphic=True)(state)


def _cool_state(
    cooling: PreparedGaugeCooling,
    state: Array,
    /,
) -> tuple[Array, GaugeCoolingEvidence]:
    initial_norm = jnp.asarray(cooling.unitarity_norm(state))
    if initial_norm.shape != () or jnp.iscomplexobj(initial_norm):
        raise TypeError("unitarity_norm must return one real scalar.")

    def iteration(carry, _):
        current, current_norm, accepted_count, rejected_count, valid = carry
        gradient = jnp.asarray(cooling.gauge_gradient(current))
        if gradient.shape != cooling.configuration_shape:
            raise ValueError("gauge_gradient must match configuration_shape.")
        candidate = jnp.asarray(
            cooling.gauge_retract(current, -cooling.plan.step_size * gradient)
        )
        if candidate.shape != cooling.configuration_shape:
            raise ValueError("gauge_retract must preserve configuration_shape.")
        candidate_norm = jnp.asarray(cooling.unitarity_norm(candidate))
        if candidate_norm.shape != () or jnp.iscomplexobj(candidate_norm):
            raise TypeError("unitarity_norm must return one real scalar.")
        finite = (
            jnp.all(jnp.isfinite(jnp.real(gradient)))
            & jnp.all(jnp.isfinite(jnp.imag(gradient)))
            & jnp.all(jnp.isfinite(jnp.real(candidate)))
            & jnp.all(jnp.isfinite(jnp.imag(candidate)))
            & jnp.isfinite(candidate_norm)
        )
        accept = (
            valid
            & finite
            & (candidate_norm <= current_norm + cooling.plan.monotonicity_tolerance)
        )
        return (
            jnp.where(accept, candidate, current),
            jnp.where(accept, candidate_norm, current_norm),
            accepted_count + accept.astype(jnp.int32),
            rejected_count + (valid & ~accept).astype(jnp.int32),
            valid & finite,
        ), None

    (cooled, final_norm, accepted, rejected, valid), _ = jax.lax.scan(
        iteration,
        (
            state,
            initial_norm,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.isfinite(initial_norm),
        ),
        None,
        length=cooling.plan.iterations,
    )
    return cooled, GaugeCoolingEvidence(
        initial_norm=initial_norm,
        final_norm=final_norm,
        accepted_updates=accepted,
        rejected_updates=rejected,
        valid=valid,
    )


def sample_complex_langevin(
    runtime: PreparedComplexLangevin,
    initial_state: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
) -> ComplexLangevinResult:
    """Run one fixed-length real-noise complex Langevin trajectory."""
    if not isinstance(runtime, PreparedComplexLangevin):
        raise TypeError("runtime must be PreparedComplexLangevin.")
    state = jnp.asarray(initial_state)
    if state.shape != runtime.configuration_shape:
        raise ValueError("initial_state does not match configuration_shape.")
    if not jnp.iscomplexobj(state):
        raise TypeError("initial_state must use a complex dtype.")
    initial_action = _action_value(runtime, state)
    initial_drift = _action_drift(runtime, state)
    initial_finite = (
        jnp.isfinite(jnp.real(initial_action))
        & jnp.isfinite(jnp.imag(initial_action))
        & jnp.all(jnp.isfinite(jnp.real(initial_drift)))
        & jnp.all(jnp.isfinite(jnp.imag(initial_drift)))
        & (jnp.linalg.norm(state.reshape((-1,))) <= runtime.plan.maximum_state_norm)
    )

    def step(carry, step_index):
        current, active, first_invalid = carry
        drift = _action_drift(runtime, current)
        drift_norm = jnp.linalg.norm(drift.reshape((-1,)))
        noise_key = derive_key(key, _NOISE_ADDRESS, step_index)
        noise = jr.normal(
            noise_key, runtime.configuration_shape, dtype=jnp.real(current).dtype
        )
        candidate = (
            current
            - runtime.plan.step_size * drift
            + jnp.sqrt(2.0 * runtime.plan.step_size) * noise
        )
        if runtime.cooling is None:
            cooled = candidate
            cooling_evidence = GaugeCoolingEvidence(
                initial_norm=jnp.asarray(jnp.nan),
                final_norm=jnp.asarray(jnp.nan),
                accepted_updates=jnp.asarray(0, dtype=jnp.int32),
                rejected_updates=jnp.asarray(0, dtype=jnp.int32),
                valid=jnp.asarray(True),
            )
        else:
            cooled, cooling_evidence = _cool_state(runtime.cooling, candidate)
        action_value = _action_value(runtime, cooled)
        finite = (
            jnp.all(jnp.isfinite(jnp.real(drift)))
            & jnp.all(jnp.isfinite(jnp.imag(drift)))
            & jnp.isfinite(drift_norm)
            & jnp.all(jnp.isfinite(jnp.real(cooled)))
            & jnp.all(jnp.isfinite(jnp.imag(cooled)))
            & jnp.isfinite(jnp.real(action_value))
            & jnp.isfinite(jnp.imag(action_value))
            & cooling_evidence.valid
        )
        bounded = (
            jnp.linalg.norm(cooled.reshape((-1,))) <= runtime.plan.maximum_state_norm
        )
        commit = active & finite & bounded
        next_state = jnp.where(commit, cooled, current)
        invalid_now = active & ~(finite & bounded)
        next_first_invalid = jnp.where(
            (first_invalid < 0) & invalid_now,
            step_index.astype(jnp.int32),
            first_invalid,
        )
        return (next_state, commit, next_first_invalid), (
            next_state,
            jnp.where(commit, action_value, _action_value(runtime, current)),
            drift_norm,
            finite,
            bounded,
            cooling_evidence.initial_norm,
            cooling_evidence.final_norm,
            cooling_evidence.accepted_updates,
            cooling_evidence.rejected_updates,
            cooling_evidence.valid,
        )

    (final_state, active, first_invalid), history = jax.lax.scan(
        step,
        (
            state,
            initial_finite,
            jnp.where(initial_finite, -1, 0).astype(jnp.int32),
        ),
        jnp.arange(runtime.plan.num_steps, dtype=jnp.uint32),
    )
    (
        state_history,
        action_history,
        drift_history,
        finite_history,
        bounded_history,
        cooling_initial_history,
        cooling_final_history,
        cooling_accepted_history,
        cooling_rejected_history,
        cooling_valid_history,
    ) = history
    sample_indices = jnp.arange(runtime.plan.num_steps, dtype=jnp.int32)[
        runtime.plan.burn_in :: runtime.plan.thinning
    ]
    samples = state_history[runtime.plan.burn_in :: runtime.plan.thinning]
    actions = action_history[runtime.plan.burn_in :: runtime.plan.thinning]
    tail = drift_history[-runtime.plan.tail_window :]
    tail_probability = jnp.mean(
        (tail > runtime.plan.drift_tail_threshold).astype("float64")
    )
    tail_mean = jnp.mean(tail)
    maximum_drift = jnp.max(drift_history)
    cooling_failed = ~jnp.all(cooling_valid_history)
    nonfinite_failed = ~jnp.all(finite_history)
    bound_failed = ~jnp.all(bounded_history)
    tail_failed = tail_probability > runtime.plan.maximum_tail_probability
    status = jnp.where(
        nonfinite_failed | ~initial_finite,
        int(ComplexLangevinStatus.NONFINITE_ACTION_OR_DRIFT),
        jnp.where(
            bound_failed,
            int(ComplexLangevinStatus.STATE_NORM_EXCEEDED),
            jnp.where(
                cooling_failed,
                int(ComplexLangevinStatus.GAUGE_COOLING_FAILED),
                jnp.where(
                    tail_failed,
                    int(ComplexLangevinStatus.DRIFT_TAIL_REJECTED),
                    int(ComplexLangevinStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = ComplexLangevinDiagnostics(
        status=status,
        maximum_drift_norm=maximum_drift,
        drift_tail_probability=tail_probability,
        tail_mean_drift_norm=tail_mean,
        first_invalid_step=first_invalid,
        cooling_initial_norm=cooling_initial_history,
        cooling_final_norm=cooling_final_history,
        cooling_accepted_updates=jnp.sum(cooling_accepted_history, dtype=jnp.int32),
        cooling_rejected_updates=jnp.sum(cooling_rejected_history, dtype=jnp.int32),
        finite_steps=jnp.sum(finite_history & bounded_history, dtype=jnp.int32),
    )
    return ComplexLangevinResult(
        samples=samples,
        action_values=actions,
        drift_norms=drift_history,
        sample_step_indices=sample_indices,
        final_state=final_state,
        diagnostics=diagnostics,
        successful=active & (status == int(ComplexLangevinStatus.SUCCESS)),
        root_key=jnp.asarray(key),
        action_id=runtime.action_id,
        prepared_id=runtime.prepared_id,
        claim="finite-complex-langevin-with-drift-tail-and-gauge-cooling-evidence",
    )


def complex_langevin_one_variable_controls(
    runtime: PreparedComplexLangevin,
    result: ComplexLangevinResult,
    /,
    *,
    maximum_order: int = 4,
) -> ComplexLangevinControlResult:
    """Evaluate exact one-variable Schwinger-Dyson identities through a fixed order."""
    if not isinstance(runtime, PreparedComplexLangevin):
        raise TypeError("runtime must be PreparedComplexLangevin.")
    if not isinstance(result, ComplexLangevinResult):
        raise TypeError("result must be ComplexLangevinResult.")
    if result.prepared_id != runtime.prepared_id:
        raise ValueError("Complex Langevin result belongs to another preparation.")
    if runtime.configuration_shape != (1,):
        raise ValueError("One-variable controls require configuration_shape == (1,).")
    order_limit = int(maximum_order)
    if order_limit <= 0 or order_limit > 32:
        raise ValueError("maximum_order must lie between one and 32.")
    z = result.samples[:, 0]
    drift = jax.vmap(lambda value: _action_drift(runtime, value[None])[0])(z)
    orders = jnp.arange(1, order_limit + 1)
    powers = z[:, None] ** orders[None, :]
    lower_powers = z[:, None] ** (orders[None, :] - 1)
    residuals = ein.contract(
        "no,n->o",
        orders[None, :] * lower_powers - powers * drift[:, None],
        jnp.full((z.shape[0],), 1.0 / z.shape[0]),
    )
    maximum = jnp.max(jnp.abs(residuals))
    finite = (
        result.successful
        & jnp.all(jnp.isfinite(jnp.real(residuals)))
        & jnp.all(jnp.isfinite(jnp.imag(residuals)))
    )
    return ComplexLangevinControlResult(
        orders=orders,
        schwinger_dyson_residuals=residuals,
        maximum_residual=maximum,
        finite=finite,
        prepared_id=runtime.prepared_id,
        claim="one-variable-holomorphic-schwinger-dyson-controls",
    )


__all__ = [
    "ComplexLangevinControlResult",
    "ComplexLangevinDiagnostics",
    "ComplexLangevinPlan",
    "ComplexLangevinResult",
    "ComplexLangevinStatus",
    "GaugeCoolingEvidence",
    "GaugeCoolingPlan",
    "PreparedComplexLangevin",
    "PreparedGaugeCooling",
    "complex_langevin_one_variable_controls",
    "prepare_complex_langevin",
    "prepare_gauge_cooling",
    "sample_complex_langevin",
]

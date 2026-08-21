#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..linalg._causal_linear import (
    causal_linearized_residual,
    solve_causal_least_squares,
)
from ._causal_adjoint import attach_causal_implicit_derivative
from ._types import NonlinearStatus, NonlinearTermination


CausalLinearizationMode: TypeAlias = Literal[
    "dense-exact",
    "diagonal-exact",
    "diagonal-hutchinson",
    "fixed-block",
]
CausalProbeDistribution: TypeAlias = Literal["rademacher", "normal"]


def _tree_all_finite(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves])
    )


def _flat_maximum_norm(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value))


class CausalRecurrenceProblem(StrictModule):
    """One explicit first-order causal recurrence over a fixed driver sequence."""

    transition_function: Callable[[Any, PyTree[Any], PyTree[Any]], PyTree[Any]] = (
        eqx.field(static=True)
    )
    initial_state: PyTree[Array]
    drivers: PyTree[Array]
    parameters: Any
    unravel_state: Callable[[Array], PyTree[Array]] = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: Callable[[Any, PyTree[Any], PyTree[Any]], PyTree[Any]],
        initial_state: PyTree[Any],
        drivers: PyTree[Any],
        /,
        *,
        parameters: Any = None,
        problem_id: str = "causal-recurrence",
    ):
        if not callable(transition):
            raise TypeError("transition must be callable.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        state = jax.tree.map(jnp.asarray, initial_state)
        state_leaves = jax.tree.leaves(state)
        if not state_leaves:
            raise ValueError("initial_state must contain at least one array leaf.")
        if any(not jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in state_leaves):
            raise TypeError("Causal recurrence states must be real floating arrays.")
        flat_state, unravel = ravel_pytree(state)
        if flat_state.size == 0:
            raise ValueError("initial_state must contain at least one scalar coordinate.")

        driver_tree = jax.tree.map(jnp.asarray, drivers)
        driver_leaves = jax.tree.leaves(driver_tree)
        if not driver_leaves:
            raise ValueError("drivers must contain at least one array leaf.")
        if any(leaf.ndim < 1 for leaf in driver_leaves):
            raise ValueError("Every driver leaf must have a leading temporal axis.")
        steps = int(driver_leaves[0].shape[0])
        if steps < 1 or any(int(leaf.shape[0]) != steps for leaf in driver_leaves):
            raise ValueError("Every driver leaf must share one nonempty temporal axis.")

        first_driver = jax.tree.map(lambda leaf: leaf[0], driver_tree)
        first_output = jax.tree.map(
            jnp.asarray, transition(parameters, state, first_driver)
        )
        if jax.tree.structure(first_output) != jax.tree.structure(state) or any(
            output.shape != expected.shape
            for output, expected in zip(
                jax.tree.leaves(first_output),
                state_leaves,
                strict=True,
            )
        ):
            raise ValueError(
                "transition outputs must match the initial-state PyTree and leaf shapes."
            )
        output_flat, _ = ravel_pytree(first_output)
        if not jnp.issubdtype(output_flat.dtype, jnp.floating):
            raise TypeError(
                "Causal recurrence transitions must return real floating arrays."
            )

        self.transition_function = transition
        self.initial_state = state
        self.drivers = driver_tree
        self.parameters = parameters
        self.unravel_state = unravel
        self.num_steps = steps
        self.state_size = int(flat_state.size)
        self.problem_id = identifier

    @property
    def flat_initial_state(self) -> Array:
        return ravel_pytree(self.initial_state)[0]

    def transition_flat(
        self,
        previous_state: Array,
        driver: PyTree[Any],
        /,
    ) -> Array:
        previous = self.unravel_state(previous_state)
        output = self.transition_function(self.parameters, previous, driver)
        return ravel_pytree(output)[0]

    def evaluate_flat(self, trajectory: Array, /) -> tuple[Array, Array]:
        values = jnp.asarray(trajectory)
        expected = (self.num_steps, self.state_size)
        if values.shape != expected:
            raise ValueError(f"trajectory must have shape {expected}.")
        predecessors = jnp.concatenate(
            (self.flat_initial_state[None, :], values[:-1]), axis=0
        )
        proposed = jax.vmap(self.transition_flat)(predecessors, self.drivers)
        return values - proposed, proposed

    def unravel_trajectory(self, trajectory: Array, /) -> PyTree[Array]:
        return jax.vmap(self.unravel_state)(trajectory)


class CausalLinearizationPolicy(StrictModule):
    """Per-step Jacobian representation for one causal nonlinear solve."""

    block_builder: Callable[[Any, PyTree[Any], PyTree[Any]], Array] | None = eqx.field(
        static=True
    )
    mode: CausalLinearizationMode = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    probe_distribution: CausalProbeDistribution = eqx.field(static=True)
    linearization_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: CausalLinearizationMode = "dense-exact",
        /,
        *,
        probe_count: int = 1,
        probe_distribution: CausalProbeDistribution = "rademacher",
        block_builder: Callable[[Any, PyTree[Any], PyTree[Any]], Array] | None = None,
        linearization_id: str | None = None,
    ):
        if mode not in (
            "dense-exact",
            "diagonal-exact",
            "diagonal-hutchinson",
            "fixed-block",
        ):
            raise ValueError("Unknown causal linearization mode.")
        probes = int(probe_count)
        if probes < 1:
            raise ValueError("probe_count must be positive.")
        if probe_distribution not in ("rademacher", "normal"):
            raise ValueError("Unknown causal probe distribution.")
        if mode == "fixed-block" and not callable(block_builder):
            raise TypeError("fixed-block mode requires block_builder.")
        if mode != "fixed-block" and block_builder is not None:
            raise ValueError("block_builder is only valid for fixed-block mode.")
        identifier = mode if linearization_id is None else str(linearization_id)
        if not identifier:
            raise ValueError("linearization_id must be non-empty.")
        self.mode = mode
        self.probe_count = probes
        self.probe_distribution = probe_distribution
        self.block_builder = block_builder
        self.linearization_id = identifier

    @property
    def exact(self) -> bool:
        return self.mode == "dense-exact"


class CausalNewton(StrictModule):
    """Undamped exact or quasi-Newton causal recurrence evaluation."""

    linearization: CausalLinearizationPolicy

    def __init__(self, *, linearization: CausalLinearizationPolicy | None = None):
        policy = CausalLinearizationPolicy() if linearization is None else linearization
        if not isinstance(policy, CausalLinearizationPolicy):
            raise TypeError("linearization must be CausalLinearizationPolicy or None.")
        self.linearization = policy

    @property
    def method_id(self) -> str:
        return "causal-newton"


class CausalLevenbergMarquardt(StrictModule):
    """ELK-style damped causal least squares with ratio-based acceptance."""

    linearization: CausalLinearizationPolicy
    initial_damping: float = eqx.field(static=True)
    minimum_damping: float = eqx.field(static=True)
    maximum_damping: float = eqx.field(static=True)
    damping_increase: float = eqx.field(static=True)
    damping_decrease: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    decrease_ratio: float = eqx.field(static=True)
    increase_ratio: float = eqx.field(static=True)
    maximum_trials: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        linearization: CausalLinearizationPolicy | None = None,
        initial_damping: float = 1e-3,
        minimum_damping: float = 1e-12,
        maximum_damping: float = 1e12,
        damping_increase: float = 2.0,
        damping_decrease: float = 0.5,
        acceptance_ratio: float = 1e-4,
        decrease_ratio: float = 0.75,
        increase_ratio: float = 0.25,
        maximum_trials: int = 12,
    ):
        policy = CausalLinearizationPolicy() if linearization is None else linearization
        if not isinstance(policy, CausalLinearizationPolicy):
            raise TypeError("linearization must be CausalLinearizationPolicy or None.")
        damping_values = (
            float(initial_damping),
            float(minimum_damping),
            float(maximum_damping),
        )
        if any(not isfinite(value) or value <= 0.0 for value in damping_values):
            raise ValueError("Causal LM damping values must be positive and finite.")
        if not damping_values[1] <= damping_values[0] <= damping_values[2]:
            raise ValueError("initial_damping must lie inside the damping bounds.")
        increase = float(damping_increase)
        decrease = float(damping_decrease)
        if not isfinite(increase) or increase <= 1.0:
            raise ValueError("damping_increase must exceed one.")
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError("damping_decrease must lie strictly inside (0, 1).")
        ratios = (
            float(acceptance_ratio),
            float(decrease_ratio),
            float(increase_ratio),
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("Causal LM ratios must lie in [0, 1].")
        if not ratios[0] <= ratios[2] < ratios[1]:
            raise ValueError(
                "Ratios must satisfy acceptance_ratio <= increase_ratio < decrease_ratio."
            )
        trials = int(maximum_trials)
        if trials < 1:
            raise ValueError("maximum_trials must be positive.")
        self.linearization = policy
        self.initial_damping, self.minimum_damping, self.maximum_damping = damping_values
        self.damping_increase = increase
        self.damping_decrease = decrease
        self.acceptance_ratio, self.decrease_ratio, self.increase_ratio = ratios
        self.maximum_trials = trials

    @property
    def method_id(self) -> str:
        return "causal-levenberg-marquardt"


CausalMethod: TypeAlias = CausalNewton | CausalLevenbergMarquardt


class CausalRecurrenceDiagnostics(StrictModule):
    """Fixed-shape nonlinear histories and exact temporal work counts."""

    residual_norm: Array
    step_norm: Array
    damping: Array
    actual_reduction: Array
    predicted_reduction: Array
    reduction_ratio: Array
    accepted: Array
    finite: Array
    iteration_count: Array
    accepted_steps: Array
    rejected_steps: Array
    transition_evaluations: Array
    jacobian_evaluations: Array
    jvp_evaluations: Array


class CausalRecurrenceResult(StrictModule):
    """Certified trajectory from one causal nonlinear recurrence solve."""

    problem: CausalRecurrenceProblem
    states: PyTree[Array]
    residuals: PyTree[Array]
    flat_states: Array
    flat_residuals: Array
    final_state: PyTree[Array]
    status: Array
    diagnostics: CausalRecurrenceDiagnostics
    method_id: str = eqx.field(static=True)
    linearization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(NonlinearStatus.SUCCESS)


def _initial_flat_trajectory(
    problem: CausalRecurrenceProblem,
    initial_trajectory: PyTree[Any] | None,
    /,
) -> Array:
    if initial_trajectory is None:
        return jnp.broadcast_to(
            problem.flat_initial_state,
            (problem.num_steps, problem.state_size),
        )
    trajectory = jax.tree.map(jnp.asarray, initial_trajectory)
    leaves = jax.tree.leaves(trajectory)
    if not leaves or any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("initial_trajectory leaves need a leading temporal axis.")
    if any(int(leaf.shape[0]) != problem.num_steps for leaf in leaves):
        raise ValueError("initial_trajectory must match the driver sequence length.")
    flat = jax.vmap(lambda state: ravel_pytree(state)[0])(trajectory)
    if flat.shape != (problem.num_steps, problem.state_size):
        raise ValueError("initial_trajectory does not match the recurrence state layout.")
    return flat


def _probe_bank(
    policy: CausalLinearizationPolicy,
    key: Array | None,
    *,
    num_steps: int,
    state_size: int,
    dtype: Any,
) -> Array:
    if policy.mode != "diagonal-hutchinson":
        return jnp.empty((0, num_steps, state_size), dtype=dtype)
    if key is None:
        raise ValueError("diagonal-hutchinson linearization requires probe_key.")
    shape = (policy.probe_count, num_steps, state_size)
    if policy.probe_distribution == "rademacher":
        return jr.rademacher(key, shape, dtype=dtype)
    return jr.normal(key, shape, dtype=dtype)


def _linearize_transitions(
    problem: CausalRecurrenceProblem,
    trajectory: Array,
    policy: CausalLinearizationPolicy,
    probes: Array,
    /,
) -> tuple[Array, Array, Array]:
    predecessors = jnp.concatenate(
        (problem.flat_initial_state[None, :], trajectory[:-1]),
        axis=0,
    )
    if policy.mode in ("dense-exact", "diagonal-exact"):
        dense = jax.vmap(jax.jacfwd(problem.transition_flat))(
            predecessors,
            problem.drivers,
        )
        if policy.mode == "dense-exact":
            matrices = dense
        else:
            matrices = jax.vmap(jnp.diag)(jax.vmap(jnp.diag)(dense))
        matrices = matrices.at[0].set(jnp.zeros_like(matrices[0]))
        return (
            matrices,
            jnp.asarray(problem.num_steps, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    if policy.mode == "fixed-block":
        block_builder = policy.block_builder
        if block_builder is None:
            raise RuntimeError("fixed-block policy lost its block builder.")

        def build(previous, driver):
            state = problem.unravel_state(previous)
            return jnp.asarray(block_builder(problem.parameters, state, driver))

        matrices = jax.vmap(build)(predecessors, problem.drivers)
        expected = (problem.num_steps, problem.state_size, problem.state_size)
        if matrices.shape != expected:
            raise ValueError(
                f"block_builder must return local matrices with shape {expected}."
            )
        matrices = matrices.at[0].set(jnp.zeros_like(matrices[0]))
        return matrices, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)

    def probe_action(probe):
        def one(previous, driver, direction):
            _, tangent = jax.jvp(
                lambda candidate: problem.transition_flat(candidate, driver),
                (previous,),
                (direction,),
            )
            return tangent

        return jax.vmap(one)(predecessors, problem.drivers, probe)

    actions = jax.vmap(probe_action)(probes)
    diagonal = jnp.mean(probes * actions, axis=0)
    matrices = jax.vmap(jnp.diag)(diagonal)
    matrices = matrices.at[0].set(jnp.zeros_like(matrices[0]))
    jvp_count = policy.probe_count * problem.num_steps
    return (
        matrices,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jvp_count, dtype=jnp.int32),
    )


def _empty_histories(steps: int, dtype: Any, /) -> tuple[Array, ...]:
    nan = jnp.full((steps,), jnp.nan, dtype=dtype)
    return (
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        jnp.zeros((steps,), dtype=bool),
        jnp.zeros((steps,), dtype=bool),
    )


def _solve_causal_forward(
    problem: CausalRecurrenceProblem,
    initial_trajectory: PyTree[Any] | None,
    method: CausalMethod,
    termination: NonlinearTermination,
    probe_key: Array | None,
    /,
) -> CausalRecurrenceResult:
    trajectory = _initial_flat_trajectory(problem, initial_trajectory)
    residual, _ = problem.evaluate_flat(trajectory)
    initial_norm = _flat_maximum_norm(residual)
    threshold = termination.residual_threshold(initial_norm)
    dtype = trajectory.dtype
    histories = _empty_histories(termination.maximum_steps, dtype)
    converged = jnp.isfinite(initial_norm) & (initial_norm <= threshold)
    failed = ~jnp.isfinite(initial_norm)
    status = jnp.where(
        converged,
        int(NonlinearStatus.SUCCESS),
        jnp.where(
            failed,
            int(NonlinearStatus.NONFINITE_EVALUATION),
            int(NonlinearStatus.ITERATING),
        ),
    ).astype(jnp.int32)
    damping = jnp.asarray(
        method.initial_damping if isinstance(method, CausalLevenbergMarquardt) else 0.0,
        dtype=dtype,
    )
    probes = _probe_bank(
        method.linearization,
        probe_key,
        num_steps=problem.num_steps,
        state_size=problem.state_size,
        dtype=dtype,
    )

    carry = (
        trajectory,
        residual,
        converged,
        failed,
        status,
        damping,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(problem.num_steps, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        *histories,
    )

    def iteration(index, state):
        (
            current,
            current_residual,
            already_converged,
            already_failed,
            current_status,
            current_damping,
            accepted_count,
            rejected_count,
            transition_evaluations,
            jacobian_evaluations,
            jvp_evaluations,
            residual_history,
            step_history,
            damping_history,
            actual_history,
            predicted_history,
            ratio_history,
            accepted_history,
            finite_history,
        ) = state
        active = ~(already_converged | already_failed)

        def active_iteration(_):
            matrices, jacobian_increment, jvp_increment = _linearize_transitions(
                problem,
                current,
                method.linearization,
                probes,
            )
            current_objective = 0.5 * jnp.sum(jnp.square(current_residual))

            if isinstance(method, CausalNewton):
                step = solve_causal_least_squares(
                    matrices,
                    current_residual,
                    jnp.asarray(0.0, dtype=dtype),
                )
                candidate = current + step
                candidate_residual, _ = problem.evaluate_flat(candidate)
                linear_residual = causal_linearized_residual(
                    matrices,
                    current_residual,
                    step,
                )
                candidate_objective = 0.5 * jnp.sum(jnp.square(candidate_residual))
                predicted = current_objective - 0.5 * jnp.sum(jnp.square(linear_residual))
                actual = current_objective - candidate_objective
                finite = (
                    jnp.all(jnp.isfinite(step))
                    & jnp.all(jnp.isfinite(candidate))
                    & jnp.all(jnp.isfinite(candidate_residual))
                )
                accepted = finite
                ratio = jnp.where(predicted > 0.0, actual / predicted, -jnp.inf)
                next_damping = current_damping
                trial_count = jnp.asarray(1, dtype=jnp.int32)
            else:
                trial_initial = (
                    jnp.asarray(0, dtype=jnp.int32),
                    current_damping,
                    jnp.asarray(False),
                    current,
                    current_residual,
                    jnp.zeros_like(current),
                    jnp.asarray(-jnp.inf, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                )

                def trial_body(_, trial_state):
                    (
                        trial_number,
                        trial_damping,
                        trial_accepted,
                        saved_candidate,
                        saved_residual,
                        saved_step,
                        saved_ratio,
                        saved_actual,
                        saved_predicted,
                        finite_seen,
                    ) = trial_state

                    def evaluate_trial(_):
                        direction = solve_causal_least_squares(
                            matrices,
                            current_residual,
                            trial_damping,
                        )
                        proposed = current + direction
                        proposed_residual, _ = problem.evaluate_flat(proposed)
                        linear_residual = causal_linearized_residual(
                            matrices,
                            current_residual,
                            direction,
                        )
                        proposed_objective = 0.5 * jnp.sum(jnp.square(proposed_residual))
                        predicted_reduction = current_objective - 0.5 * jnp.sum(
                            jnp.square(linear_residual)
                        )
                        actual_reduction = current_objective - proposed_objective
                        finite_trial = (
                            jnp.all(jnp.isfinite(direction))
                            & jnp.all(jnp.isfinite(proposed))
                            & jnp.all(jnp.isfinite(proposed_residual))
                            & jnp.isfinite(predicted_reduction)
                            & jnp.isfinite(actual_reduction)
                        )
                        reduction_ratio = jnp.where(
                            finite_trial & (predicted_reduction > 0.0),
                            actual_reduction / predicted_reduction,
                            -jnp.inf,
                        )
                        use = finite_trial & (reduction_ratio >= method.acceptance_ratio)
                        adjusted_damping = jnp.where(
                            reduction_ratio > method.decrease_ratio,
                            jnp.maximum(
                                method.minimum_damping,
                                trial_damping * method.damping_decrease,
                            ),
                            jnp.where(
                                reduction_ratio < method.increase_ratio,
                                jnp.minimum(
                                    method.maximum_damping,
                                    trial_damping * method.damping_increase,
                                ),
                                trial_damping,
                            ),
                        )
                        return (
                            trial_number + 1,
                            adjusted_damping,
                            use,
                            jnp.where(use, proposed, saved_candidate),
                            jnp.where(use, proposed_residual, saved_residual),
                            jnp.where(use, direction, saved_step),
                            reduction_ratio,
                            actual_reduction,
                            predicted_reduction,
                            finite_seen | finite_trial,
                        )

                    return jax.lax.cond(
                        trial_accepted,
                        lambda _: trial_state,
                        evaluate_trial,
                        operand=None,
                    )

                trial_final = jax.lax.fori_loop(
                    0,
                    method.maximum_trials,
                    trial_body,
                    trial_initial,
                )
                (
                    trial_count,
                    next_damping,
                    accepted,
                    candidate,
                    candidate_residual,
                    step,
                    ratio,
                    actual,
                    predicted,
                    finite,
                ) = trial_final

            next_residual = jnp.where(accepted, candidate_residual, current_residual)
            next_trajectory = jnp.where(accepted, candidate, current)
            residual_norm = _flat_maximum_norm(next_residual)
            step_norm = _flat_maximum_norm(step)
            now_converged = (
                accepted & jnp.isfinite(residual_norm) & (residual_norm <= threshold)
            )
            stagnated = (
                accepted
                & ~now_converged
                & (
                    step_norm
                    <= termination.step_threshold(_flat_maximum_norm(next_trajectory))
                )
            )
            diverged = jnp.isfinite(residual_norm) & (
                residual_norm
                > termination.divergence_factor * jnp.maximum(initial_norm, 1.0)
            )
            failed_trial = ~accepted
            now_failed = (
                failed_trial | stagnated | diverged | (~jnp.isfinite(residual_norm))
            )
            next_status = jnp.where(
                now_converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    ~jnp.isfinite(residual_norm),
                    int(NonlinearStatus.NONFINITE_EVALUATION),
                    jnp.where(
                        failed_trial,
                        (
                            int(NonlinearStatus.TRUST_REGION_FAILED)
                            if isinstance(method, CausalLevenbergMarquardt)
                            else int(NonlinearStatus.NONFINITE_EVALUATION)
                        ),
                        jnp.where(
                            stagnated,
                            int(NonlinearStatus.RESIDUAL_STAGNATION),
                            jnp.where(
                                diverged,
                                int(NonlinearStatus.DIVERGENCE),
                                int(NonlinearStatus.ITERATING),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            transition_increment = problem.num_steps * trial_count
            return (
                next_trajectory,
                next_residual,
                now_converged,
                now_failed,
                next_status,
                next_damping,
                accepted_count + accepted.astype(jnp.int32),
                rejected_count + (trial_count - accepted.astype(jnp.int32)),
                transition_evaluations + transition_increment,
                jacobian_evaluations + jacobian_increment,
                jvp_evaluations + jvp_increment,
                residual_history.at[index].set(residual_norm),
                step_history.at[index].set(step_norm),
                damping_history.at[index].set(current_damping),
                actual_history.at[index].set(actual),
                predicted_history.at[index].set(predicted),
                ratio_history.at[index].set(ratio),
                accepted_history.at[index].set(accepted),
                finite_history.at[index].set(finite),
            )

        return jax.lax.cond(active, active_iteration, lambda _: state, operand=None)

    final = jax.lax.fori_loop(0, termination.maximum_steps, iteration, carry)
    (
        final_trajectory,
        final_residual,
        final_converged,
        final_failed,
        final_status,
        _,
        accepted_steps,
        rejected_steps,
        transition_evaluations,
        jacobian_evaluations,
        jvp_evaluations,
        residual_history,
        step_history,
        damping_history,
        actual_history,
        predicted_history,
        ratio_history,
        accepted_history,
        finite_history,
    ) = final
    final_status = jnp.where(
        final_converged,
        int(NonlinearStatus.SUCCESS),
        jnp.where(
            final_failed,
            final_status,
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
        ),
    ).astype(jnp.int32)
    iteration_count = jnp.sum(~jnp.isnan(residual_history), dtype=jnp.int32)
    states = problem.unravel_trajectory(final_trajectory)
    residuals = problem.unravel_trajectory(final_residual)
    final_state = jax.tree.map(lambda leaf: leaf[-1], states)
    diagnostics = CausalRecurrenceDiagnostics(
        residual_norm=residual_history,
        step_norm=step_history,
        damping=damping_history,
        actual_reduction=actual_history,
        predicted_reduction=predicted_history,
        reduction_ratio=ratio_history,
        accepted=accepted_history,
        finite=finite_history,
        iteration_count=iteration_count,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        transition_evaluations=transition_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        jvp_evaluations=jvp_evaluations,
    )
    approximation = (
        "exact-dense-causal-root"
        if method.linearization.exact
        else f"quasi-causal-root/{method.linearization.linearization_id}"
    )
    return CausalRecurrenceResult(
        problem=problem,
        states=states,
        residuals=residuals,
        flat_states=final_trajectory,
        flat_residuals=final_residual,
        final_state=final_state,
        status=final_status,
        diagnostics=diagnostics,
        method_id=method.method_id,
        linearization_id=method.linearization.linearization_id,
        approximation_id=approximation,
    )


def solve_causal_recurrence(
    problem: CausalRecurrenceProblem,
    /,
    *,
    initial_trajectory: PyTree[Any] | None = None,
    method: CausalMethod | None = None,
    termination: NonlinearTermination | None = None,
    probe_key: Array | None = None,
) -> CausalRecurrenceResult:
    """Solve and certify one causal recurrence from its direct temporal residual."""

    if not isinstance(problem, CausalRecurrenceProblem):
        raise TypeError("problem must be a CausalRecurrenceProblem.")
    method_ = CausalLevenbergMarquardt() if method is None else method
    if not isinstance(method_, (CausalNewton, CausalLevenbergMarquardt)):
        raise TypeError("method must be a causal recurrence method or None.")
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    result = _solve_causal_forward(
        problem,
        initial_trajectory,
        method_,
        termination_,
        probe_key,
    )
    return attach_causal_implicit_derivative(problem, result)


__all__ = [
    "CausalLevenbergMarquardt",
    "CausalLinearizationMode",
    "CausalLinearizationPolicy",
    "CausalNewton",
    "CausalProbeDistribution",
    "CausalRecurrenceDiagnostics",
    "CausalRecurrenceProblem",
    "CausalRecurrenceResult",
    "solve_causal_recurrence",
]

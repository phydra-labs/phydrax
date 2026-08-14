#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from ..._strict import StrictModule
from ...stochastic._state_space import StateSpaceStepContext
from .._status import TransportStatus
from ._problem import SchrodingerBridgeProblem


class BridgeProvenance(StrictModule):
    """Static identity of an exact finite-state bridge solve."""

    method: str = eqx.field(static=True)
    reference_process: str = eqx.field(static=True)
    initial: str = eqx.field(static=True)
    terminal: str = eqx.field(static=True)
    time_grid: str = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)


class SchrodingerBridgeDiagnostics(StrictModule):
    """Fixed-structure convergence and path-law diagnostics."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    endpoint_residual: Array
    initial_residual: Array
    terminal_residual: Array
    physical_endpoint_residual: Array
    path_kl: Array
    reference_row_residual: Array
    endpoint_residual_history: Array
    path_kl_history: Array
    controlled_row_valid: Array
    feasible: Array


class SchrodingerBridgeResult(StrictModule):
    """Exact finite-state bridge, messages, Doob kernel, and diagnostics."""

    problem: SchrodingerBridgeProblem
    initial_log_scaling: Array
    terminal_log_scaling: Array
    forward_log_potentials: Array
    backward_log_potentials: Array
    reference_log_transitions: Array
    controlled_transition_probabilities: Array
    marginal_probabilities: Array
    endpoint_coupling: Array
    diagnostics: SchrodingerBridgeDiagnostics
    provenance: BridgeProvenance
    approximate: bool = eqx.field(static=True)

    @property
    def converged(self) -> Array:
        """Whether each declared physical case met the endpoint contract."""
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    @property
    def controlled_row_valid(self) -> Array:
        """Rows on which the Doob potential is positive and uniquely defined."""
        return self.diagnostics.controlled_row_valid

    def marginal_weights(self) -> Array:
        """Physical path marginals with shape ``case + (time, state)``."""
        return self.problem.mass[..., None, None] * self.marginal_probabilities

    def initial_marginal(self) -> Array:
        return self.problem.mass[..., None] * self.marginal_probabilities[..., 0, :]

    def terminal_marginal(self) -> Array:
        return self.problem.mass[..., None] * self.marginal_probabilities[..., -1, :]

    def physical_endpoint_coupling(self) -> Array:
        return self.problem.mass[..., None, None] * self.endpoint_coupling

    def controlled_kernel(self):
        """Return the exact Doob-transformed stochastic transition contract."""
        from ._kernel import ControlledTransitionKernel

        return ControlledTransitionKernel(self)

    def sample_state_indices(self, key, sample_shape=()):
        """Sample stable finite-state index paths from the controlled law."""
        from ._kernel import sample_bridge_state_indices

        return sample_bridge_state_indices(key, self, sample_shape=sample_shape)

    def sample_paths(self, key, sample_shape=()):
        """Sample stable state-value paths from the controlled law."""
        from ._kernel import sample_bridge_paths

        return sample_bridge_paths(key, self, sample_shape=sample_shape)

    def path_log_prob(self, paths: Array) -> Array:
        """Evaluate normalized controlled path log probabilities."""
        from ._kernel import bridge_path_log_prob

        return bridge_path_log_prob(self, paths)

    def reference_path_log_prob(self, paths: Array) -> Array:
        """Evaluate the endpoint-initialized reference path log probabilities."""
        from ._kernel import reference_path_log_prob

        return reference_path_log_prob(self, paths)


class SchrodingerBridgeSolver(StrictModule):
    """Log-coordinate iterative proportional fitting for finite Markov bridges."""

    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(self, *, max_iterations: int = 500, tolerance: float = 1e-9):
        iterations = int(max_iterations)
        tolerance = float(tolerance)
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.max_iterations = iterations
        self.tolerance = tolerance

    def solve(self, problem: SchrodingerBridgeProblem, /) -> SchrodingerBridgeResult:
        if not isinstance(problem, SchrodingerBridgeProblem):
            raise TypeError("problem must be a SchrodingerBridgeProblem.")
        log_transitions, reference_row_residual = _reference_log_transitions(problem)
        return _solve_bridge(
            problem,
            log_transitions,
            reference_row_residual,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

    def __call__(self, problem: SchrodingerBridgeProblem, /) -> SchrodingerBridgeResult:
        return self.solve(problem)


def _step_context(
    base: StateSpaceStepContext, case_index: Array, step_index: Array, /
) -> StateSpaceStepContext:
    return eqx.tree_at(
        lambda context: (context.case_index, context.step_index),
        base,
        (case_index.astype(jnp.int32), step_index.astype(jnp.int32)),
    )


def _case_transition_matrix(
    problem: SchrodingerBridgeProblem,
    support: Array,
    case_index: Array,
    step_index: Array,
    /,
) -> Array:
    context = _step_context(problem.context, case_index, step_index)
    t0 = problem.times[step_index]
    t1 = problem.times[step_index + 1]

    def from_state(state: Array) -> Array:
        return jax.vmap(
            lambda next_state: problem.reference.log_prob(
                next_state, state, t0, t1, context
            )
        )(support)

    matrix = jax.vmap(from_state)(support)
    expected = (problem.num_states, problem.num_states)
    if matrix.shape != expected:
        raise ValueError(
            "reference.log_prob must return one scalar per finite state pair; "
            f"expected {expected}, got {matrix.shape}."
        )
    return matrix


def _reference_log_transitions(
    problem: SchrodingerBridgeProblem, /
) -> tuple[Array, Array]:
    count = problem.num_cases
    support = problem.state_support.reshape(
        (count, problem.num_states) + problem.state_shape
    )
    case_indices = jnp.arange(count, dtype=jnp.int32)
    matrices = []
    for step in range(problem.num_steps):
        step_index = jnp.asarray(step, dtype=jnp.int32)
        matrices.append(
            jax.vmap(
                lambda state_support, case_index: _case_transition_matrix(
                    problem, state_support, case_index, step_index
                )
            )(support, case_indices)
        )
    log_transitions = jnp.stack(matrices, axis=1)
    admissible = jnp.isfinite(log_transitions) | jnp.isneginf(log_transitions)
    log_transitions = eqx.error_if(
        log_transitions,
        jnp.any(~admissible),
        "Reference transition log densities must be finite or negative infinity.",
    )
    row_log_mass = jsp.special.logsumexp(log_transitions, axis=-1)
    row_residual = jnp.max(jnp.abs(jnp.exp(row_log_mass) - 1.0), axis=(-2, -1))
    log_transitions = eqx.error_if(
        log_transitions,
        jnp.any(row_residual > problem.transition_tolerance),
        "Reference transition density is not normalized on the declared finite support.",
    )
    return log_transitions, row_residual


def _log_matrix_product(left: Array, right: Array, /) -> Array:
    return jsp.special.logsumexp(left[:, :, None] + right[None, :, :], axis=1)


def _endpoint_reference(log_transitions: Array, initial_probabilities: Array, /) -> Array:
    size = int(log_transitions.shape[-1])
    identity = jnp.where(jnp.eye(size, dtype=bool), 0.0, -jnp.inf)
    product = identity
    for step in range(int(log_transitions.shape[0])):
        product = _log_matrix_product(product, log_transitions[step])
    return _safe_log(initial_probabilities)[:, None] + product


def _safe_log(probabilities: Array, /) -> Array:
    return jnp.where(probabilities > 0.0, jnp.log(probabilities), -jnp.inf)


def _coupling_statistics(
    log_reference: Array, log_a: Array, log_b: Array, mu0: Array, mu1: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    log_coupling = log_reference + log_a[:, None] + log_b[None, :]
    coupling = jnp.exp(log_coupling)
    initial = jnp.sum(coupling, axis=-1)
    terminal = jnp.sum(coupling, axis=-2)
    initial_residual = jnp.sum(jnp.abs(initial - mu0))
    terminal_residual = jnp.sum(jnp.abs(terminal - mu1))
    residual = jnp.maximum(initial_residual, terminal_residual)
    positive = coupling > 0.0
    safe_log_coupling = jnp.where(positive, log_coupling, 0.0)
    safe_log_reference = jnp.where(positive, log_reference, 0.0)
    path_kl = jnp.sum(coupling * (safe_log_coupling - safe_log_reference))
    return coupling, initial_residual, terminal_residual, residual, path_kl, log_coupling


def _ipf_case(
    mu0: Array,
    mu1: Array,
    log_reference: Array,
    /,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    log_mu0 = _safe_log(mu0)
    log_mu1 = _safe_log(mu1)
    source_reachable = jnp.any(
        jnp.isfinite(log_reference) & (mu1[None, :] > 0.0), axis=-1
    )
    target_reachable = jnp.any(
        jnp.isfinite(log_reference) & (mu0[:, None] > 0.0), axis=-2
    )
    feasible = jnp.all((mu0 <= 0.0) | source_reachable) & jnp.all(
        (mu1 <= 0.0) | target_reachable
    )
    initial_a = jnp.zeros_like(mu0)
    initial_b = jnp.zeros_like(mu1)
    _, _, _, initial_residual, initial_kl, _ = _coupling_statistics(
        log_reference, initial_a, initial_b, mu0, mu1
    )

    def update(carry, _):
        log_a, log_b = carry
        source_denominator = jsp.special.logsumexp(
            log_reference + log_b[None, :], axis=-1
        )
        proposed_a = jnp.where(
            mu0 > 0.0,
            jnp.where(
                jnp.isfinite(source_denominator),
                log_mu0 - source_denominator,
                0.0,
            ),
            -jnp.inf,
        )
        target_denominator = jsp.special.logsumexp(
            log_reference + proposed_a[:, None], axis=-2
        )
        proposed_b = jnp.where(
            mu1 > 0.0,
            jnp.where(
                jnp.isfinite(target_denominator),
                log_mu1 - target_denominator,
                0.0,
            ),
            -jnp.inf,
        )
        log_a = jnp.where(feasible, proposed_a, log_a)
        log_b = jnp.where(feasible, proposed_b, log_b)
        _, initial_residual, terminal_residual, residual, path_kl, _ = (
            _coupling_statistics(log_reference, log_a, log_b, mu0, mu1)
        )
        return (log_a, log_b), (
            residual,
            path_kl,
            initial_residual,
            terminal_residual,
        )

    (log_a, log_b), histories = jax.lax.scan(
        update,
        (initial_a, initial_b),
        xs=None,
        length=max_iterations,
    )
    residual_history = jnp.concatenate((initial_residual[None], histories[0]), axis=0)
    kl_history = jnp.concatenate((initial_kl[None], histories[1]), axis=0)
    initial_residual = histories[2][-1]
    terminal_residual = histories[3][-1]
    endpoint_residual = histories[0][-1]
    coupling, _, _, _, path_kl, _ = _coupling_statistics(
        log_reference, log_a, log_b, mu0, mu1
    )
    iterate_admissible = jnp.all(jnp.isfinite(log_a) | jnp.isneginf(log_a)) & jnp.all(
        jnp.isfinite(log_b) | jnp.isneginf(log_b)
    )
    convergence = (residual_history <= tolerance) & feasible & iterate_admissible
    any_converged = jnp.any(convergence)
    first = jnp.where(any_converged, jnp.argmax(convergence), -1).astype(jnp.int32)
    num_iterations = jnp.where(any_converged, first, max_iterations).astype(jnp.int32)
    status = jnp.where(
        ~feasible,
        int(TransportStatus.INFEASIBLE_SUPPORT),
        jnp.where(
            ~iterate_admissible,
            int(TransportStatus.NONFINITE_ITERATE),
            jnp.where(
                any_converged,
                int(TransportStatus.CONVERGED),
                int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
            ),
        ),
    ).astype(jnp.int32)
    return (
        log_a,
        log_b,
        coupling,
        endpoint_residual,
        initial_residual,
        terminal_residual,
        path_kl,
        residual_history,
        kl_history,
        status,
        num_iterations,
        first,
        feasible,
    )


def _messages(
    log_transitions: Array,
    log_a: Array,
    log_b: Array,
    reference_initial: Array,
    /,
) -> tuple[Array, Array]:
    forward = [_safe_log(reference_initial) + log_a]
    for step in range(int(log_transitions.shape[0])):
        forward.append(
            jsp.special.logsumexp(forward[-1][:, None] + log_transitions[step], axis=-2)
        )
    backward = [log_b]
    for step in range(int(log_transitions.shape[0]) - 1, -1, -1):
        backward.append(
            jsp.special.logsumexp(log_transitions[step] + backward[-1][None, :], axis=-1)
        )
    return jnp.stack(forward, axis=0), jnp.stack(backward[::-1], axis=0)


def _doob_and_marginals(
    log_transitions: Array, forward: Array, backward: Array, /
) -> tuple[Array, Array, Array]:
    log_normalizer = jsp.special.logsumexp(forward[0] + backward[0])
    marginals = jnp.exp(forward + backward - log_normalizer)
    probabilities = []
    valid_rows = []
    for step in range(int(log_transitions.shape[0])):
        denominator = backward[step]
        row_valid = jnp.isfinite(denominator)
        log_controlled = (
            log_transitions[step] + backward[step + 1][None, :] - denominator[:, None]
        )
        doob = jnp.exp(jnp.where(row_valid[:, None], log_controlled, -jnp.inf))
        # The reference row is the canonical normalized extension on polar states.
        extended = jnp.where(row_valid[:, None], doob, jnp.exp(log_transitions[step]))
        probabilities.append(extended)
        valid_rows.append(row_valid)
    return (
        jnp.stack(probabilities, axis=0),
        marginals,
        jnp.stack(valid_rows, axis=0),
    )


def _solve_bridge(
    problem: SchrodingerBridgeProblem,
    log_transitions: Array,
    reference_row_residual: Array,
    /,
    *,
    max_iterations: int,
    tolerance: float,
) -> SchrodingerBridgeResult:
    count = problem.num_cases
    mu0 = problem.initial_probabilities.reshape((count, problem.num_states))
    mu1 = problem.terminal_probabilities.reshape((count, problem.num_states))
    endpoint_reference = jax.vmap(_endpoint_reference)(log_transitions, mu0)
    solved = jax.vmap(
        lambda start, end, reference: _ipf_case(
            start,
            end,
            reference,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    )(mu0, mu1, endpoint_reference)
    (
        log_a,
        log_b,
        coupling,
        endpoint_residual,
        initial_residual,
        terminal_residual,
        path_kl,
        residual_history,
        path_kl_history,
        status,
        num_iterations,
        first_converged,
        feasible,
    ) = solved
    forward, backward = jax.vmap(_messages)(log_transitions, log_a, log_b, mu0)
    controlled, marginals, controlled_row_valid = jax.vmap(_doob_and_marginals)(
        log_transitions, forward, backward
    )
    case = problem.case_shape
    reshape_case = lambda value, trailing: value.reshape(case + trailing)
    diagnostics = SchrodingerBridgeDiagnostics(
        status=reshape_case(status, ()),
        num_iterations=reshape_case(num_iterations, ()),
        first_converged_iteration=reshape_case(first_converged, ()),
        endpoint_residual=reshape_case(endpoint_residual, ()),
        initial_residual=reshape_case(initial_residual, ()),
        terminal_residual=reshape_case(terminal_residual, ()),
        physical_endpoint_residual=reshape_case(
            endpoint_residual * problem.mass.reshape((count,)), ()
        ),
        path_kl=reshape_case(path_kl, ()),
        reference_row_residual=reshape_case(reference_row_residual, ()),
        endpoint_residual_history=reshape_case(residual_history, (max_iterations + 1,)),
        path_kl_history=reshape_case(path_kl_history, (max_iterations + 1,)),
        controlled_row_valid=reshape_case(
            controlled_row_valid,
            (problem.num_steps, problem.num_states),
        ),
        feasible=reshape_case(feasible, ()),
    )
    return SchrodingerBridgeResult(
        problem=problem,
        initial_log_scaling=reshape_case(log_a, (problem.num_states,)),
        terminal_log_scaling=reshape_case(log_b, (problem.num_states,)),
        forward_log_potentials=reshape_case(
            forward, (problem.num_steps + 1, problem.num_states)
        ),
        backward_log_potentials=reshape_case(
            backward, (problem.num_steps + 1, problem.num_states)
        ),
        reference_log_transitions=reshape_case(
            log_transitions,
            (problem.num_steps, problem.num_states, problem.num_states),
        ),
        controlled_transition_probabilities=reshape_case(
            controlled,
            (problem.num_steps, problem.num_states, problem.num_states),
        ),
        marginal_probabilities=reshape_case(
            marginals, (problem.num_steps + 1, problem.num_states)
        ),
        endpoint_coupling=reshape_case(
            coupling, (problem.num_states, problem.num_states)
        ),
        diagnostics=diagnostics,
        provenance=BridgeProvenance(
            method="log-ipf",
            reference_process=problem.reference.process_id,
            initial=problem.provenance.initial,
            terminal=problem.provenance.terminal,
            time_grid=problem.time_id,
            execution="dense-exact-finite-state",
            approximation="exact",
        ),
        approximate=False,
    )


def solve_schrodinger_bridge(
    problem: SchrodingerBridgeProblem,
    solver: SchrodingerBridgeSolver | None = None,
    /,
) -> SchrodingerBridgeResult:
    """Solve an exact finite-state Schrödinger bridge."""
    method = SchrodingerBridgeSolver() if solver is None else solver
    if not isinstance(method, SchrodingerBridgeSolver):
        raise TypeError("solver must be a SchrodingerBridgeSolver or None.")
    return method.solve(problem)


def require_converged_bridge(
    result: SchrodingerBridgeResult, /
) -> SchrodingerBridgeResult:
    """Reject nonconverged bridge results at scientific integration boundaries."""
    if not isinstance(result, SchrodingerBridgeResult):
        raise TypeError("result must be a SchrodingerBridgeResult.")
    checked = eqx.error_if(
        result.forward_log_potentials,
        jnp.any(~result.converged),
        "Exact finite-state Schrödinger bridge did not converge for every case.",
    )
    return eqx.tree_at(
        lambda value: value.forward_log_potentials,
        result,
        checked,
    )


__all__ = [
    "BridgeProvenance",
    "SchrodingerBridgeDiagnostics",
    "SchrodingerBridgeResult",
    "SchrodingerBridgeSolver",
    "require_converged_bridge",
    "solve_schrodinger_bridge",
]

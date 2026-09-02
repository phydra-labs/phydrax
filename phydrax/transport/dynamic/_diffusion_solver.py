#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._diffusion_problem import (
    _proposal_arrays,
    DiffusionBridgePlan,
    DiffusionBridgeProblem,
    PreparedDiffusionBridge,
)
from ._solver import _doob_and_marginals, _endpoint_reference, _ipf_case, _messages


class DiffusionBridgeDiagnostics(StrictModule):
    effective_sample_sizes: Array
    row_normalizer_residuals: Array
    uncovered_transition_mass: Array
    endpoint_residual: Array
    initial_residual: Array
    terminal_residual: Array
    path_kl: Array
    num_iterations: Array
    ipf_status: Array
    support_valid: Array
    valid: Array
    evidence_kind: str = eqx.field(static=True)


class DiffusionBridgeResult(StrictModule):
    """Bridge exact on one prepared finite chain, approximate for the diffusion."""

    prepared: PreparedDiffusionBridge
    doob_transitions: Array
    physical_marginals: Array
    forward_messages: Array
    backward_messages: Array
    endpoint_coupling: Array
    diagnostics: DiffusionBridgeDiagnostics
    valid: Array
    status: Array
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def _normalized_log_weights(log_weights: Array, mask: Array, /) -> tuple[Array, Array]:
    selected = jnp.where(mask, log_weights, -jnp.inf)
    normalizer = jax.scipy.special.logsumexp(selected)
    valid = jnp.isfinite(normalizer) & jnp.any(mask)
    return jnp.where(mask, selected - normalizer, -jnp.inf), valid


def prepare_diffusion_bridge(
    problem: DiffusionBridgeProblem,
    plan: DiffusionBridgePlan,
    /,
    *,
    key: Key[Array, ""],
) -> PreparedDiffusionBridge:
    """Lower finite proposals and normalized transition factors to one chain."""
    del key
    if not isinstance(problem, DiffusionBridgeProblem):
        raise TypeError("problem must be a DiffusionBridgeProblem.")
    if not isinstance(plan, DiffusionBridgePlan):
        raise TypeError("plan must be a DiffusionBridgePlan.")
    time_count = int(problem.time_grid.times.size)
    if len(plan.proposal_realizations) != time_count:
        raise ValueError("proposal_realizations must contain one support per time node.")
    support_rows = []
    weight_rows = []
    mask_rows = []
    proposal_valid = []
    for realization in plan.proposal_realizations:
        points, log_weights, mask = _proposal_arrays(realization)
        if points.shape[0] != plan.support_capacity:
            raise ValueError("every proposal must use exactly support_capacity slots.")
        if tuple(points.shape[1:]) != problem.initial_law.event_shape:
            raise ValueError("proposal event shape differs from endpoint laws.")
        normalized, valid = _normalized_log_weights(log_weights, mask)
        support_rows.append(points)
        weight_rows.append(normalized)
        mask_rows.append(mask)
        proposal_valid.append(valid)
    supports = jnp.stack(support_rows)
    log_weights = jnp.stack(weight_rows)
    masks = jnp.stack(mask_rows)

    def transition_matrix(step: int) -> Array:
        start = problem.time_grid.times[step]
        end = problem.time_grid.times[step + 1]
        context = problem.contexts[step]
        source = supports[step]
        target = supports[step + 1]

        def one_source(state):
            return jax.vmap(
                lambda next_state: problem.reference.log_prob(
                    next_state,
                    state,
                    start,
                    end,
                    context,
                )
            )(target)

        raw = jax.vmap(one_source)(source) + log_weights[step + 1][None, :]
        active = masks[step][:, None] & masks[step + 1][None, :]
        return jnp.where(active, raw, -jnp.inf)

    raw_transitions = jnp.stack(
        tuple(transition_matrix(step) for step in range(time_count - 1))
    )
    log_normalizers = jax.scipy.special.logsumexp(raw_transitions, axis=-1)
    row_valid = jnp.isfinite(log_normalizers) & masks[:-1]
    log_transitions = jnp.where(
        row_valid[..., None],
        raw_transitions - log_normalizers[..., None],
        -jnp.inf,
    )
    row_residuals = jnp.where(
        masks[:-1],
        jnp.abs(jnp.exp(log_normalizers) - 1.0),
        0.0,
    )

    def endpoint_probabilities(law, index):
        values = jax.vmap(law.log_prob)(supports[index]) + log_weights[index]
        values = jnp.where(masks[index], values, -jnp.inf)
        return jax.nn.softmax(values)

    initial = endpoint_probabilities(problem.initial_law, 0)
    terminal = endpoint_probabilities(problem.terminal_law, -1)
    probabilities = jnp.exp(log_weights)
    effective_sample_sizes = 1.0 / jnp.sum(probabilities**2, axis=-1)
    valid = (
        jnp.all(jnp.asarray(proposal_valid))
        & jnp.all(row_valid | ~masks[:-1])
        & jnp.all(jnp.isfinite(row_residuals))
        & jnp.all(jnp.isfinite(effective_sample_sizes))
    )
    if not bool(valid):
        raise ValueError("Diffusion bridge proposal/transition preparation failed.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-finite-diffusion-bridge-v1",
            "problem": problem.problem_id,
            "supports": plan.support_capacity,
            "time_grid": problem.time_grid.time_id,
            "reference": problem.reference.process_id,
        }
    )
    return PreparedDiffusionBridge(
        problem=problem,
        plan=plan,
        supports=supports,
        log_proposal_weights=log_weights,
        masks=masks,
        log_transitions=log_transitions,
        row_normalizer_residuals=row_residuals,
        endpoint_probabilities=jnp.stack((initial, terminal)),
        effective_sample_sizes=effective_sample_sizes,
        prepared_id=prepared_id,
    )


def solve_diffusion_bridge(prepared: PreparedDiffusionBridge, /) -> DiffusionBridgeResult:
    """Run the canonical finite IPF/Doob core on a prepared diffusion chain."""
    if not isinstance(prepared, PreparedDiffusionBridge):
        raise TypeError("prepared must be a PreparedDiffusionBridge.")
    initial, terminal = prepared.endpoint_probabilities
    log_reference = _endpoint_reference(prepared.log_transitions, initial)
    (
        log_a,
        log_b,
        coupling,
        endpoint_residual,
        initial_residual,
        terminal_residual,
        path_kl,
        _,
        _,
        ipf_status,
        num_iterations,
        _,
        feasible,
    ) = _ipf_case(
        initial,
        terminal,
        log_reference,
        max_iterations=prepared.plan.solver.max_iterations,
        tolerance=prepared.plan.solver.tolerance,
    )
    forward, backward = _messages(
        prepared.log_transitions,
        log_a,
        log_b,
        initial,
    )
    doob, marginals, row_valid = _doob_and_marginals(
        prepared.log_transitions,
        forward,
        backward,
    )
    uncovered = jnp.max(prepared.row_normalizer_residuals)
    support_valid = (
        jnp.all(prepared.effective_sample_sizes >= prepared.plan.minimum_ess)
        & (uncovered <= prepared.plan.maximum_tail_error)
        & jnp.all(row_valid | ~prepared.masks[:-1])
    )
    converged = ipf_status == 0
    valid = feasible & converged & support_valid & jnp.all(jnp.isfinite(marginals))
    diagnostics = DiffusionBridgeDiagnostics(
        effective_sample_sizes=prepared.effective_sample_sizes,
        row_normalizer_residuals=prepared.row_normalizer_residuals,
        uncovered_transition_mass=uncovered,
        endpoint_residual=endpoint_residual,
        initial_residual=initial_residual,
        terminal_residual=terminal_residual,
        path_kl=path_kl,
        num_iterations=num_iterations,
        ipf_status=ipf_status,
        support_valid=support_valid,
        valid=valid,
        evidence_kind="finite-support-quadrature-time-ipf",
    )
    return DiffusionBridgeResult(
        prepared=prepared,
        doob_transitions=doob,
        physical_marginals=marginals,
        forward_messages=forward,
        backward_messages=backward,
        endpoint_coupling=coupling,
        diagnostics=diagnostics,
        valid=valid,
        status=jnp.where(valid, 0, jnp.where(~support_valid, 2, 1)).astype(jnp.int32),
        approximation_kind="exact-prepared-chain-diffusion-approximation",
        bounded_non_claim=(
            "The Doob/IPF result is exact only on the prepared finite chain; support, "
            "quadrature, and time discretization remain continuum approximation errors."
        ),
    )


def sample_diffusion_bridge(
    result: DiffusionBridgeResult,
    key: Key[Array, ""],
    sample_shape: tuple[int, ...] = (),
    /,
) -> Array:
    """Sample exact paths from the represented finite Doob chain."""
    if not isinstance(result, DiffusionBridgeResult):
        raise TypeError("result must be a DiffusionBridgeResult.")
    if not bool(result.valid):
        raise ValueError("Cannot sample an invalid diffusion bridge.")
    shape = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in shape):
        raise ValueError("sample_shape dimensions must be positive.")
    count = prod(shape) if shape else 1
    keys = jr.split(key, count)
    initial_probabilities = result.physical_marginals[0]

    def one(path_key):
        first_key, path_key = jr.split(path_key)
        index = jr.categorical(first_key, jnp.log(initial_probabilities))
        indices = [index]
        for step in range(result.doob_transitions.shape[0]):
            step_key, path_key = jr.split(path_key)
            probabilities = result.doob_transitions[step, indices[-1]]
            indices.append(jr.categorical(step_key, jnp.log(probabilities)))
        indices_array = jnp.stack(indices)
        return result.prepared.supports[jnp.arange(indices_array.size), indices_array]

    paths = jax.vmap(one)(keys)
    return paths.reshape(shape + paths.shape[1:])


__all__ = [
    "DiffusionBridgeDiagnostics",
    "DiffusionBridgeResult",
    "prepare_diffusion_bridge",
    "sample_diffusion_bridge",
    "solve_diffusion_bridge",
]

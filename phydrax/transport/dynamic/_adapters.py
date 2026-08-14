#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ...stochastic._state_space import CategoricalStatePrior
from ._kernel import (
    _path_indices,
    bridge_path_log_prob,
    BridgePathSample,
    ControlledTransitionKernel,
    reference_path_log_prob,
    sample_bridge,
)
from ._solver import (
    require_converged_bridge,
    SchrodingerBridgeResult,
)


class BridgeInferenceAdapter(StrictModule):
    """State-space inference view of a converged finite-state bridge."""

    result: SchrodingerBridgeResult
    transition: ControlledTransitionKernel

    def __init__(self, result: SchrodingerBridgeResult, /):
        if not isinstance(result, SchrodingerBridgeResult):
            raise TypeError("result must be a SchrodingerBridgeResult.")
        result = require_converged_bridge(result)
        self.result = result
        self.transition = ControlledTransitionKernel(result)

    def initial_prior(self, case_index: int = 0, /) -> CategoricalStatePrior:
        """Compose the canonical categorical state-prior contract for one case."""
        index = int(case_index)
        if index < 0 or index >= self.result.problem.num_cases:
            raise ValueError("case_index is out of range.")
        problem = self.result.problem
        support = problem.state_support.reshape(
            (problem.num_cases, problem.num_states) + problem.state_shape
        )[index]
        probabilities = problem.initial_probabilities.reshape(
            (problem.num_cases, problem.num_states)
        )[index]
        return CategoricalStatePrior(
            support,
            probabilities,
            prior_id=f"{problem.provenance.initial}:bridge-initial",
        )

    def sample(
        self,
        key: Key[Array, ""],
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> BridgePathSample:
        return sample_bridge(key, self.result, sample_shape=sample_shape)


class TerminalDistributionControlAdapter(StrictModule):
    """Distributional-control view of a converged Doob-transformed kernel."""

    result: SchrodingerBridgeResult
    transition: ControlledTransitionKernel
    terminal_probabilities: Array
    terminal_weights: Array

    def __init__(self, result: SchrodingerBridgeResult, /):
        if not isinstance(result, SchrodingerBridgeResult):
            raise TypeError("result must be a SchrodingerBridgeResult.")
        result = require_converged_bridge(result)
        self.result = result
        self.transition = ControlledTransitionKernel(result)
        self.terminal_probabilities = result.problem.terminal_probabilities
        self.terminal_weights = result.problem.terminal_weights

    @property
    def path_kl_cost(self) -> Array:
        """Exact relative-entropy control cost for every physical case."""
        return self.result.diagnostics.path_kl

    @property
    def terminal_residual(self) -> Array:
        """Physical terminal-distribution control residual."""
        return self.result.problem.mass * self.result.diagnostics.terminal_residual


class BridgePathLawDiagnostics(StrictModule):
    """Fixed-structure exact-versus-empirical path-law diagnostics."""

    empirical_marginal_probabilities: Array
    empirical_marginal_residual: Array
    mean_log_likelihood_ratio: Array
    log_likelihood_ratio_standard_error: Array
    exact_path_kl: Array
    num_samples: Array
    valid: Array


def bridge_path_law_diagnostics(
    result: SchrodingerBridgeResult,
    paths: BridgePathSample | ArrayLike,
    /,
) -> BridgePathLawDiagnostics:
    """Compose bridge sampling and stochastic log-density contracts."""
    if not isinstance(result, SchrodingerBridgeResult):
        raise TypeError("result must be a SchrodingerBridgeResult.")
    result = require_converged_bridge(result)
    values = paths.values if isinstance(paths, BridgePathSample) else jnp.asarray(paths)
    indices, state_valid, sample_shape = _path_indices(result, values)
    sample_count = prod(sample_shape) if sample_shape else 1
    one_hot = jax.nn.one_hot(indices, result.problem.num_states)
    empirical = jnp.mean(one_hot, axis=1)
    exact = result.marginal_probabilities.reshape(
        (
            result.problem.num_cases,
            result.problem.num_steps + 1,
            result.problem.num_states,
        )
    )
    marginal_residual = jnp.max(jnp.sum(jnp.abs(empirical - exact), axis=-1), axis=-1)
    controlled_log_prob = bridge_path_log_prob(result, values).reshape(
        (result.problem.num_cases, sample_count)
    )
    reference_log_prob = reference_path_log_prob(result, values).reshape(
        (result.problem.num_cases, sample_count)
    )
    ratio = controlled_log_prob - reference_log_prob
    mean = jnp.mean(ratio, axis=-1)
    variance = jnp.var(ratio, axis=-1, ddof=0)
    standard_error = jnp.sqrt(variance / sample_count)
    valid = jnp.all(state_valid, axis=(-2, -1)) & jnp.all(jnp.isfinite(ratio), axis=-1)
    case_shape = result.problem.case_shape
    return BridgePathLawDiagnostics(
        empirical_marginal_probabilities=empirical.reshape(
            case_shape + (result.problem.num_steps + 1, result.problem.num_states)
        ),
        empirical_marginal_residual=marginal_residual.reshape(case_shape),
        mean_log_likelihood_ratio=mean.reshape(case_shape),
        log_likelihood_ratio_standard_error=standard_error.reshape(case_shape),
        exact_path_kl=result.diagnostics.path_kl,
        num_samples=jnp.full(case_shape, sample_count, dtype=jnp.int32),
        valid=valid.reshape(case_shape),
    )


__all__ = [
    "BridgeInferenceAdapter",
    "BridgePathLawDiagnostics",
    "TerminalDistributionControlAdapter",
    "bridge_path_law_diagnostics",
]

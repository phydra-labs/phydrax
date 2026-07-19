#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class MCMCDiagnostics(StrictModule):
    """Rank-normalized convergence diagnostics for chain-by-draw samples."""

    rhat: PyTree[Array]
    bulk_ess: PyTree[Array]
    tail_ess: PyTree[Array]
    mean_acceptance_rate: Array
    divergence_count: Array
    max_rhat: Array
    min_bulk_ess: Array
    min_tail_ess: Array

    def __init__(
        self,
        *,
        rhat: PyTree[Array],
        bulk_ess: PyTree[Array],
        tail_ess: PyTree[Array],
        acceptance_rate: Array,
        divergent: Array,
    ):
        self.rhat = rhat
        self.bulk_ess = bulk_ess
        self.tail_ess = tail_ess
        self.mean_acceptance_rate = jnp.mean(jnp.asarray(acceptance_rate))
        self.divergence_count = jnp.sum(jnp.asarray(divergent, dtype=jnp.int32))
        self.max_rhat = _tree_extreme(rhat, maximum=True)
        self.min_bulk_ess = _tree_extreme(bulk_ess, maximum=False)
        self.min_tail_ess = _tree_extreme(tail_ess, maximum=False)


class MCMCConvergenceThresholds(StrictModule):
    """Caller-controlled release gates for rank diagnostics and sampler failures."""

    max_rhat: float = eqx.field(static=True)
    min_bulk_ess: float = eqx.field(static=True)
    min_tail_ess: float = eqx.field(static=True)
    allow_divergences: bool = eqx.field(static=True)
    allow_trajectory_saturation: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_rhat: float = 1.01,
        min_bulk_ess: float = 400.0,
        min_tail_ess: float = 400.0,
        allow_divergences: bool = False,
        allow_trajectory_saturation: bool = False,
    ):
        maximum = float(max_rhat)
        minimum_bulk = float(min_bulk_ess)
        minimum_tail = float(min_tail_ess)
        if not jnp.isfinite(maximum) or maximum < 1.0:
            raise ValueError("max_rhat must be finite and at least one.")
        if not jnp.isfinite(minimum_bulk) or minimum_bulk <= 0.0:
            raise ValueError("min_bulk_ess must be finite and positive.")
        if not jnp.isfinite(minimum_tail) or minimum_tail <= 0.0:
            raise ValueError("min_tail_ess must be finite and positive.")
        self.max_rhat = maximum
        self.min_bulk_ess = minimum_bulk
        self.min_tail_ess = minimum_tail
        self.allow_divergences = bool(allow_divergences)
        self.allow_trajectory_saturation = bool(allow_trajectory_saturation)


class MCMCConvergenceReport(StrictModule):
    """Serializable convergence decision with exact failing PyTree locations."""

    thresholds: MCMCConvergenceThresholds
    passed: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    rhat_failures: tuple[str, ...] = eqx.field(static=True)
    bulk_ess_failures: tuple[str, ...] = eqx.field(static=True)
    tail_ess_failures: tuple[str, ...] = eqx.field(static=True)
    divergence_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    max_rhat: Array
    min_bulk_ess: Array
    min_tail_ess: Array
    mean_acceptance_rate: Array
    divergence_count: Array
    max_integration_steps: Array
    max_trajectory_expansions: Array
    trajectory_saturation_count: Array
    num_chains: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    adaptation_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    samples_per_second: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        diagnostics: MCMCDiagnostics,
        thresholds: MCMCConvergenceThresholds,
        divergent: Array,
        num_integration_steps: Array,
        num_trajectory_expansions: Array,
        max_num_doublings: int | None,
        num_chains: int,
        num_draws: int,
        sample_memory_bytes: int,
        duration_seconds: float,
        adaptation_duration_seconds: float,
        sampling_duration_seconds: float,
        samples_per_second: float,
    ):
        rhat_failures = _failing_locations(
            diagnostics.rhat,
            lambda value: ~jnp.isfinite(value) | (value > thresholds.max_rhat),
        )
        bulk_failures = _failing_locations(
            diagnostics.bulk_ess,
            lambda value: ~jnp.isfinite(value) | (value < thresholds.min_bulk_ess),
        )
        tail_failures = _failing_locations(
            diagnostics.tail_ess,
            lambda value: ~jnp.isfinite(value) | (value < thresholds.min_tail_ess),
        )
        divergent_array = jnp.asarray(divergent, dtype=bool)
        divergence_indices = tuple(
            (int(index[0]), int(index[1])) for index in jnp.argwhere(divergent_array)
        )
        integration_steps = jnp.asarray(num_integration_steps)
        trajectory_expansions = jnp.asarray(num_trajectory_expansions)
        if max_num_doublings is None:
            saturation_count = jnp.zeros((), dtype=jnp.int32)
        else:
            saturation_count = jnp.sum(
                trajectory_expansions >= int(max_num_doublings),
                dtype=jnp.int32,
            )
        failures: list[str] = []
        if rhat_failures:
            failures.append("rhat")
        if bulk_failures:
            failures.append("bulk_ess")
        if tail_failures:
            failures.append("tail_ess")
        if divergence_indices and not thresholds.allow_divergences:
            failures.append("divergences")
        if int(saturation_count) > 0 and not thresholds.allow_trajectory_saturation:
            failures.append("trajectory_saturation")
        self.thresholds = thresholds
        self.passed = not failures
        self.failures = tuple(failures)
        self.rhat_failures = rhat_failures
        self.bulk_ess_failures = bulk_failures
        self.tail_ess_failures = tail_failures
        self.divergence_indices = divergence_indices
        self.max_rhat = diagnostics.max_rhat
        self.min_bulk_ess = diagnostics.min_bulk_ess
        self.min_tail_ess = diagnostics.min_tail_ess
        self.mean_acceptance_rate = diagnostics.mean_acceptance_rate
        self.divergence_count = diagnostics.divergence_count
        self.max_integration_steps = jnp.max(integration_steps)
        self.max_trajectory_expansions = jnp.max(trajectory_expansions)
        self.trajectory_saturation_count = saturation_count
        self.num_chains = int(num_chains)
        self.num_draws = int(num_draws)
        self.sample_memory_bytes = int(sample_memory_bytes)
        self.duration_seconds = float(duration_seconds)
        self.adaptation_duration_seconds = float(adaptation_duration_seconds)
        self.sampling_duration_seconds = float(sampling_duration_seconds)
        self.samples_per_second = float(samples_per_second)

    def raise_for_failure(self) -> None:
        """Raise with this report attached unless every configured gate passed."""
        if not self.passed:
            raise MCMCConvergenceError(self)

    def as_dict(self) -> dict[str, Any]:
        """Return a machine-serializable summary containing no device arrays."""
        return {
            "passed": self.passed,
            "failures": self.failures,
            "rhat_failures": self.rhat_failures,
            "bulk_ess_failures": self.bulk_ess_failures,
            "tail_ess_failures": self.tail_ess_failures,
            "divergence_indices": self.divergence_indices,
            "max_rhat": float(self.max_rhat),
            "min_bulk_ess": float(self.min_bulk_ess),
            "min_tail_ess": float(self.min_tail_ess),
            "mean_acceptance_rate": float(self.mean_acceptance_rate),
            "divergence_count": int(self.divergence_count),
            "max_integration_steps": int(self.max_integration_steps),
            "max_trajectory_expansions": int(self.max_trajectory_expansions),
            "trajectory_saturation_count": int(self.trajectory_saturation_count),
            "num_chains": self.num_chains,
            "num_draws": self.num_draws,
            "sample_memory_bytes": self.sample_memory_bytes,
            "duration_seconds": self.duration_seconds,
            "adaptation_duration_seconds": self.adaptation_duration_seconds,
            "sampling_duration_seconds": self.sampling_duration_seconds,
            "samples_per_second": self.samples_per_second,
        }


class MCMCConvergenceError(RuntimeError):
    """Raised when an MCMC convergence report fails its configured gates."""

    report: MCMCConvergenceReport

    def __init__(self, report: MCMCConvergenceReport):
        self.report = report
        super().__init__(
            "MCMC convergence gates failed: " + ", ".join(report.failures) + "."
        )


def mcmc_diagnostics(
    samples: PyTree[Array],
    /,
    *,
    acceptance_rate: Array,
    divergent: Array,
) -> MCMCDiagnostics:
    """Compute split rank-normalized R-hat and bulk/tail effective sample sizes."""
    import blackjax.diagnostics as diagnostics

    leaves = jax.tree_util.tree_leaves(samples)
    if not leaves:
        raise ValueError("MCMC samples must contain array leaves.")
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if array.ndim < 2:
            raise ValueError("Every MCMC sample leaf needs leading chain and draw axes.")
        if int(array.shape[0]) < 2 or int(array.shape[1]) < 4:
            raise ValueError(
                "MCMC diagnostics require at least two chains and four draws."
            )

    rhat = jax.tree_util.tree_map(
        lambda value: diagnostics.rhat(value, chain_axis=0, sample_axis=1),
        samples,
    )
    bulk_ess = jax.tree_util.tree_map(
        lambda value: diagnostics.ess_bulk(value, chain_axis=0, sample_axis=1),
        samples,
    )
    tail_ess = jax.tree_util.tree_map(
        lambda value: diagnostics.ess_tail(value, chain_axis=0, sample_axis=1),
        samples,
    )
    return MCMCDiagnostics(
        rhat=rhat,
        bulk_ess=bulk_ess,
        tail_ess=tail_ess,
        acceptance_rate=acceptance_rate,
        divergent=divergent,
    )


def _failing_locations(
    tree: PyTree[Any],
    predicate: Any,
    /,
) -> tuple[str, ...]:
    locations: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = jnp.asarray(leaf, dtype=float)
        base = jax.tree_util.keystr(path) or "<root>"
        for index in jnp.argwhere(predicate(array)):
            suffix = ""
            if array.ndim:
                suffix = "[" + ",".join(str(int(value)) for value in index) + "]"
            locations.append(base + suffix)
    return tuple(locations)


def _tree_extreme(tree: PyTree[Any], /, *, maximum: bool) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    flattened = [jnp.ravel(jnp.asarray(leaf, dtype=float)) for leaf in leaves]
    values = jnp.concatenate(flattened)
    return jnp.max(values) if maximum else jnp.min(values)


__all__ = [
    "MCMCConvergenceError",
    "MCMCConvergenceReport",
    "MCMCConvergenceThresholds",
    "MCMCDiagnostics",
    "mcmc_diagnostics",
]

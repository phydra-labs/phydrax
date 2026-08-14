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


class SGMCMCDiagnostics(StrictModule):
    """Rank mixing and stochastic-gradient summaries for fixed-step chains."""

    rhat: PyTree[Array]
    bulk_ess: PyTree[Array]
    tail_ess: PyTree[Array]
    mean: PyTree[Array]
    standard_deviation: PyTree[Array]
    minimum: PyTree[Array]
    maximum: PyTree[Array]
    max_rhat: Array
    min_bulk_ess: Array
    min_tail_ess: Array
    mean_gradient_norm: Array
    standard_deviation_gradient_norm: Array
    max_gradient_norm: Array
    min_active_factors: int = eqx.field(static=True)
    max_active_factors: int = eqx.field(static=True)
    nonfinite_update_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        rhat: PyTree[Array],
        bulk_ess: PyTree[Array],
        tail_ess: PyTree[Array],
        mean: PyTree[Array],
        standard_deviation: PyTree[Array],
        minimum: PyTree[Array],
        maximum: PyTree[Array],
        gradient_norm: Array,
        min_active_factors: int,
        max_active_factors: int,
        nonfinite_update_count: int,
    ):
        gradient_values = jnp.asarray(gradient_norm, dtype=float)
        self.rhat = rhat
        self.bulk_ess = bulk_ess
        self.tail_ess = tail_ess
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.minimum = minimum
        self.maximum = maximum
        self.max_rhat = _tree_extreme(rhat, maximum=True)
        self.min_bulk_ess = _tree_extreme(bulk_ess, maximum=False)
        self.min_tail_ess = _tree_extreme(tail_ess, maximum=False)
        self.mean_gradient_norm = jnp.mean(gradient_values)
        self.standard_deviation_gradient_norm = jnp.std(gradient_values)
        self.max_gradient_norm = jnp.max(gradient_values)
        self.min_active_factors = int(min_active_factors)
        self.max_active_factors = int(max_active_factors)
        self.nonfinite_update_count = int(nonfinite_update_count)


class SGMCMCMixingThresholds(StrictModule):
    """Caller-controlled mixing gates for a fixed-step production window."""

    max_rhat: float = eqx.field(static=True)
    min_bulk_ess: float = eqx.field(static=True)
    min_tail_ess: float = eqx.field(static=True)
    allow_nonfinite_updates: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_rhat: float = 1.01,
        min_bulk_ess: float = 400.0,
        min_tail_ess: float = 400.0,
        allow_nonfinite_updates: bool = False,
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
        self.allow_nonfinite_updates = bool(allow_nonfinite_updates)


class SGMCMCMixingReport(StrictModule):
    """Serializable mixing decision for an explicitly approximate chain."""

    thresholds: SGMCMCMixingThresholds
    passed: bool = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    rhat_failures: tuple[str, ...] = eqx.field(static=True)
    bulk_ess_failures: tuple[str, ...] = eqx.field(static=True)
    tail_ess_failures: tuple[str, ...] = eqx.field(static=True)
    max_rhat: Array
    min_bulk_ess: Array
    min_tail_ess: Array
    max_gradient_norm: Array
    algorithm: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    num_chains: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    batch_fraction: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    samples_per_second: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        diagnostics: SGMCMCDiagnostics,
        thresholds: SGMCMCMixingThresholds,
        algorithm: str,
        approximation: str,
        num_chains: int,
        num_draws: int,
        step_size: float,
        batch_fraction: float,
        sample_memory_bytes: int,
        duration_seconds: float,
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
        failures: list[str] = []
        if rhat_failures:
            failures.append("rhat")
        if bulk_failures:
            failures.append("bulk_ess")
        if tail_failures:
            failures.append("tail_ess")
        if diagnostics.nonfinite_update_count and not thresholds.allow_nonfinite_updates:
            failures.append("nonfinite_updates")
        self.thresholds = thresholds
        self.passed = not failures
        self.failures = tuple(failures)
        self.rhat_failures = rhat_failures
        self.bulk_ess_failures = bulk_failures
        self.tail_ess_failures = tail_failures
        self.max_rhat = diagnostics.max_rhat
        self.min_bulk_ess = diagnostics.min_bulk_ess
        self.min_tail_ess = diagnostics.min_tail_ess
        self.max_gradient_norm = diagnostics.max_gradient_norm
        self.algorithm = str(algorithm)
        self.approximation = str(approximation)
        self.num_chains = int(num_chains)
        self.num_draws = int(num_draws)
        self.step_size = float(step_size)
        self.batch_fraction = float(batch_fraction)
        self.sample_memory_bytes = int(sample_memory_bytes)
        self.duration_seconds = float(duration_seconds)
        self.samples_per_second = float(samples_per_second)

    def raise_for_failure(self) -> None:
        if not self.passed:
            raise SGMCMCMixingError(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "rhat_failures": self.rhat_failures,
            "bulk_ess_failures": self.bulk_ess_failures,
            "tail_ess_failures": self.tail_ess_failures,
            "max_rhat": float(self.max_rhat),
            "min_bulk_ess": float(self.min_bulk_ess),
            "min_tail_ess": float(self.min_tail_ess),
            "max_gradient_norm": float(self.max_gradient_norm),
            "algorithm": self.algorithm,
            "approximation": self.approximation,
            "num_chains": self.num_chains,
            "num_draws": self.num_draws,
            "step_size": self.step_size,
            "batch_fraction": self.batch_fraction,
            "sample_memory_bytes": self.sample_memory_bytes,
            "duration_seconds": self.duration_seconds,
            "samples_per_second": self.samples_per_second,
        }


class SGMCMCMixingError(RuntimeError):
    """Raised when a fixed-step SG-MCMC mixing report fails its gates."""

    report: SGMCMCMixingReport

    def __init__(self, report: SGMCMCMixingReport):
        self.report = report
        super().__init__(
            "SG-MCMC mixing gates failed: " + ", ".join(report.failures) + "."
        )


def sgmcmc_diagnostics(
    samples: PyTree[Array],
    /,
    *,
    gradient_norm: Array,
    min_active_factors: int,
    max_active_factors: int,
    nonfinite_update_count: int = 0,
) -> SGMCMCDiagnostics:
    """Compute rank diagnostics without fabricating HMC transition statistics."""
    import blackjax.diagnostics as diagnostics

    leaves = jax.tree_util.tree_leaves(samples)
    if not leaves:
        raise ValueError("SG-MCMC samples must contain array leaves.")
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if array.ndim < 2 or int(array.shape[0]) < 2 or int(array.shape[1]) < 4:
            raise ValueError(
                "SG-MCMC diagnostics require at least two chains and four draws."
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
    return SGMCMCDiagnostics(
        rhat=rhat,
        bulk_ess=bulk_ess,
        tail_ess=tail_ess,
        mean=jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=(0, 1)), samples),
        standard_deviation=jax.tree_util.tree_map(
            lambda value: jnp.std(value, axis=(0, 1)), samples
        ),
        minimum=jax.tree_util.tree_map(
            lambda value: jnp.min(value, axis=(0, 1)), samples
        ),
        maximum=jax.tree_util.tree_map(
            lambda value: jnp.max(value, axis=(0, 1)), samples
        ),
        gradient_norm=gradient_norm,
        min_active_factors=min_active_factors,
        max_active_factors=max_active_factors,
        nonfinite_update_count=nonfinite_update_count,
    )


def _failing_locations(tree: PyTree[Any], predicate: Any, /) -> tuple[str, ...]:
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
    values = jnp.concatenate(
        [
            jnp.ravel(jnp.asarray(leaf, dtype=float))
            for leaf in jax.tree_util.tree_leaves(tree)
        ]
    )
    return jnp.max(values) if maximum else jnp.min(values)


__all__ = [
    "SGMCMCDiagnostics",
    "SGMCMCMixingError",
    "SGMCMCMixingReport",
    "SGMCMCMixingThresholds",
    "sgmcmc_diagnostics",
]

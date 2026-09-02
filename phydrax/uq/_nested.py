#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from ..integration import WeightedSampleTarget
from ._nested_diagnostics import NestedSamplingDiagnostics
from ._nested_extensions import NestedSamplingPlan
from ._particle import resample_indices
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


NestedSamplingStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]

NESTED_SAMPLING_SUCCESS: NestedSamplingStatus = 0
NESTED_SAMPLING_MAX_DEAD_POINTS: NestedSamplingStatus = 1
NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS: NestedSamplingStatus = 2
NESTED_SAMPLING_NO_FINITE_LIVE_POINT: NestedSamplingStatus = 3
NESTED_SAMPLING_LIKELIHOOD_PLATEAU: NestedSamplingStatus = 4
NESTED_SAMPLING_INVALID_LIKELIHOOD: NestedSamplingStatus = 5
NESTED_SAMPLING_INNER_KERNEL_FAILURE: NestedSamplingStatus = 6

_RESAMPLE_ADDRESS = SampleAddress("phydrax.uq", "nested-resample", role="posterior")


def nested_sampling_status_name(value: int, /) -> str:
    """Return the stable name of one nested-sampling status code."""
    names = (
        "success",
        "maximum_dead_points",
        "maximum_likelihood_evaluations",
        "no_finite_live_point",
        "likelihood_plateau",
        "invalid_likelihood",
        "inner_kernel_failure",
    )
    code = int(value)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown nested-sampling status code {code}.")
    return names[code]


class NestedSamplingResult(StrictModule):
    """Weighted nested quadrature, evidence, diagnostics, and prepared state."""

    problem: PosteriorProblem
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_prior: Array
    log_likelihood: Array
    birth_log_likelihood: Array
    posterior_log_weights: Array
    log_prior_volume: Array
    live_counts: Array
    sample_ids: Array
    batch_indices: Array
    log_evidence: Array
    log_evidence_replicates: Array
    log_evidence_shrinkage_std: Array
    information: Array
    posterior_effective_sample_size: Array
    remaining_log_evidence: Array
    remaining_evidence_fraction: Array
    final_state: Any
    diagnostics: NestedSamplingDiagnostics
    root_key: Array
    status: Array
    valid: Array
    num_live: int = eqx.field(static=True)
    num_dead: int = eqx.field(static=True)
    num_likelihood_evaluations: int = eqx.field(static=True)
    num_inner_steps: int = eqx.field(static=True)
    num_delete: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        log_prior: Array,
        log_likelihood: Array,
        birth_log_likelihood: Array,
        posterior_log_weights: Array,
        log_prior_volume: Array,
        live_counts: Array,
        sample_ids: Array,
        batch_indices: Array,
        log_evidence: Array,
        log_evidence_replicates: Array,
        log_evidence_shrinkage_std: Array,
        information: Array,
        posterior_effective_sample_size: Array,
        remaining_log_evidence: Array,
        remaining_evidence_fraction: Array,
        final_state: Any,
        diagnostics: NestedSamplingDiagnostics,
        root_key: Array,
        status: Array,
        valid: Array,
        num_live: int,
        num_dead: int,
        num_likelihood_evaluations: int,
        num_inner_steps: int,
        num_delete: int,
        method: str,
        duration_seconds: float,
    ):
        self.problem = problem
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.log_prior = jnp.asarray(log_prior)
        self.log_likelihood = jnp.asarray(log_likelihood)
        self.birth_log_likelihood = jnp.asarray(birth_log_likelihood)
        self.posterior_log_weights = jnp.asarray(posterior_log_weights)
        self.log_prior_volume = jnp.asarray(log_prior_volume)
        self.live_counts = jnp.asarray(live_counts, dtype=jnp.int32)
        self.sample_ids = jnp.asarray(sample_ids, dtype=jnp.int32)
        self.batch_indices = jnp.asarray(batch_indices, dtype=jnp.int32)
        self.log_evidence = jnp.asarray(log_evidence)
        self.log_evidence_replicates = jnp.asarray(log_evidence_replicates)
        self.log_evidence_shrinkage_std = jnp.asarray(log_evidence_shrinkage_std)
        self.information = jnp.asarray(information)
        self.posterior_effective_sample_size = jnp.asarray(
            posterior_effective_sample_size
        )
        self.remaining_log_evidence = jnp.asarray(remaining_log_evidence)
        self.remaining_evidence_fraction = jnp.asarray(remaining_evidence_fraction)
        self.final_state = final_state
        self.diagnostics = diagnostics
        self.root_key = jnp.asarray(root_key)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.num_live = int(num_live)
        self.num_dead = int(num_dead)
        self.num_likelihood_evaluations = int(num_likelihood_evaluations)
        self.num_inner_steps = int(num_inner_steps)
        self.num_delete = int(num_delete)
        self.method = str(method)
        self.duration_seconds = float(duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(samples) + _tree_nbytes(
            unconstrained_samples
        )

    @property
    def converged(self) -> bool:
        return bool(self.valid) and int(self.status) == NESTED_SAMPLING_SUCCESS

    @property
    def num_samples(self) -> int:
        return int(self.log_likelihood.shape[0])

    def resample_posterior(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
        constrained: bool = True,
    ) -> PyTree[Array]:
        """Draw equally weighted posterior samples from nested quadrature."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        indices = resample_indices(
            derive_key(key, _RESAMPLE_ADDRESS, count),
            self.posterior_log_weights,
            method="systematic",
        )
        if count != self.num_samples:
            indices = jr.choice(
                derive_key(key, _RESAMPLE_ADDRESS, count, self.num_samples),
                self.num_samples,
                shape=(count,),
                p=jnp.exp(self.posterior_log_weights),
                replace=True,
            )
        source = self.samples if constrained else self.unconstrained_samples
        return jax.tree.map(lambda value: value[indices], source)

    def posterior_measure(self) -> WeightedSampleTarget:
        """Expose the dependent weighted posterior as an empirical measure."""
        return WeightedSampleTarget(
            self.samples,
            self.posterior_log_weights,
            normalized=True,
            independent=False,
            ancestry=self.sample_ids,
            stratum_ids=self.batch_indices,
            sample_axes=0,
            provenance=f"nested-sampling:{self.method}",
        )

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate latent predictions at every nested quadrature sample."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_nested",),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_observation_samples: int = 1,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw conditional observations at every nested quadrature sample."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_nested",),
            sample_sources=("epistemic",),
            num_observation_samples=num_observation_samples,
            batch_size=batch_size,
            valid_policy=valid_policy,
            observation_dim=observation_dim,
            **kwargs,
        )

    def diagnostic_report(self) -> dict[str, Any]:
        """Return machine-readable termination and constrained-sampling evidence."""
        report = {
            "status": nested_sampling_status_name(int(self.status)),
            "valid": bool(self.valid),
            "converged": self.converged,
            "num_live": self.num_live,
            "num_dead": self.num_dead,
            "num_likelihood_evaluations": self.num_likelihood_evaluations,
            "log_evidence": float(self.log_evidence),
            "log_evidence_shrinkage_std": float(self.log_evidence_shrinkage_std),
            "posterior_effective_sample_size": float(
                self.posterior_effective_sample_size
            ),
            "diagnostics": self.diagnostics.as_dict(),
        }
        report["proposal_branches"] = self.final_state.adaptation.branch_evidence
        return report


def sample_nested(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    plan: NestedSamplingPlan,
    remaining_evidence_tolerance: float = 0.01,
    prior_position_sampler: Callable[[Array, int], PyTree[Array]] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    checkpoint_every: int = 100,
    resume_from: str | Path | None = None,
) -> NestedSamplingResult:
    """Run the fixed-capacity prepared nested-sampling lifecycle."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if not isinstance(plan, NestedSamplingPlan):
        raise TypeError("plan must be a NestedSamplingPlan.")
    tolerance = float(remaining_evidence_tolerance)
    if not jnp.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("remaining_evidence_tolerance must be finite and positive.")
    interval = int(checkpoint_every)
    if interval < 1:
        raise ValueError("checkpoint_every must be positive.")

    from ._nested_plan_execution import execute_prepared_nested

    return execute_prepared_nested(
        problem,
        plan,
        key=key,
        remaining_evidence_tolerance=tolerance,
        prior_position_sampler=prior_position_sampler,
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        checkpoint_every=interval,
        resume_from=resume_from,
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


__all__ = [
    "NESTED_SAMPLING_INNER_KERNEL_FAILURE",
    "NESTED_SAMPLING_INVALID_LIKELIHOOD",
    "NESTED_SAMPLING_LIKELIHOOD_PLATEAU",
    "NESTED_SAMPLING_MAX_DEAD_POINTS",
    "NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS",
    "NESTED_SAMPLING_NO_FINITE_LIVE_POINT",
    "NESTED_SAMPLING_SUCCESS",
    "NestedSamplingResult",
    "NestedSamplingStatus",
    "nested_sampling_status_name",
    "sample_nested",
]

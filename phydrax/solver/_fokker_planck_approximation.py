#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..domain._normalized_density import normalize_density_field, NormalizedDensityField
from ..integration import weighted
from ..integration._api import IntegrationRealization, reduce
from ..stochastic._path_ensemble import (
    prepare_stochastic_path_ensemble,
    solve_stochastic_path_ensemble,
    StochasticPathEnsemblePlan,
    StochasticPathEnsembleResult,
)


class WeakObservable(StrictModule):
    """One finite weak-law observable and optional generator action."""

    function: Any
    generator: Any
    observable_id: str = eqx.field(static=True)

    def __init__(self, function: Any, /, *, generator: Any = None, observable_id: str):
        if not callable(function) or (generator is not None and not callable(generator)):
            raise TypeError("function and optional generator must be callable.")
        if not isinstance(observable_id, str) or not observable_id:
            raise ValueError("observable_id must be non-empty.")
        self.function = function
        self.generator = generator
        self.observable_id = observable_id


class ParticleFokkerPlanckPlan(StrictModule):
    """Fixed-capacity empirical-law plan; it makes no pointwise density claim."""

    ensemble_plan: StochasticPathEnsemblePlan
    observables: tuple[WeakObservable, ...]
    validation_grid: Array
    confidence_level: float = eqx.field(static=True)

    def __init__(
        self,
        ensemble_plan: StochasticPathEnsemblePlan,
        observables: tuple[WeakObservable, ...],
        confidence_level: float,
        validation_grid: Array,
        /,
    ):
        selected = tuple(observables)
        if not isinstance(ensemble_plan, StochasticPathEnsemblePlan):
            raise TypeError("ensemble_plan must be a StochasticPathEnsemblePlan.")
        if not selected or any(not isinstance(item, WeakObservable) for item in selected):
            raise TypeError("observables must contain WeakObservable values.")
        confidence = float(confidence_level)
        if not isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        grid = jnp.asarray(validation_grid)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("validation_grid must be nonempty and rank one.")
        self.ensemble_plan = ensemble_plan
        self.observables = selected
        self.confidence_level = confidence
        self.validation_grid = grid


class ParticleFokkerPlanckResult(StrictModule):
    """Normalized empirical laws and declared-observable weak diagnostics."""

    ensemble: StochasticPathEnsembleResult
    laws: tuple[Any, ...]
    observable_means: Array
    sampling_errors: Array
    weak_residuals: Array
    valid: Array
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def solve_particle_fokker_planck(
    problem: Any,
    plan: ParticleFokkerPlanckPlan,
    /,
    *,
    key: Key[Array, ""],
) -> ParticleFokkerPlanckResult:
    """Propagate particles and audit a finite declared class of weak observables."""
    if not isinstance(plan, ParticleFokkerPlanckPlan):
        raise TypeError("plan must be a ParticleFokkerPlanckPlan.")
    prepared = prepare_stochastic_path_ensemble(problem, plan.ensemble_plan, key=key)
    ensemble = solve_stochastic_path_ensemble(prepared)
    states = ensemble.states
    path_count = plan.ensemble_plan.path_count
    time_count = int(states.shape[1])
    log_weights = -jnp.log(jnp.asarray(path_count, dtype=states.dtype)) * jnp.ones(
        (path_count,), dtype=states.dtype
    )
    laws = tuple(
        weighted(
            states[:, index],
            log_weights,
            normalized=True,
            independent=True,
            mask=ensemble.path_valid,
            sample_axes=0,
            provenance=f"particle-fokker-planck:{ensemble.prepared_id}:{index}",
        )
        for index in range(time_count)
    )
    means = []
    errors = []
    residuals = []
    times = jnp.asarray(plan.ensemble_plan.time_grid.times)
    for observable in plan.observables:
        values = jax.vmap(jax.vmap(observable.function))(states)
        observable_mean = jnp.mean(values, axis=0)
        centered = values - observable_mean[None, ...]
        standard_error = jnp.sqrt(
            jnp.sum(jnp.abs(centered) ** 2, axis=0)
            / jnp.asarray(max(path_count - 1, 1), dtype=states.dtype)
            / jnp.asarray(path_count, dtype=states.dtype)
        )
        if observable.generator is None:
            weak = jnp.diff(observable_mean, axis=0)
        else:
            generator_values = jax.vmap(jax.vmap(observable.generator))(states[:, :-1])
            generator_mean = jnp.mean(generator_values, axis=0)
            widths = jnp.diff(times).reshape((-1,) + (1,) * (observable_mean.ndim - 1))
            weak = jnp.diff(observable_mean, axis=0) / widths - generator_mean
        means.append(observable_mean)
        errors.append(standard_error)
        residuals.append(weak)
    observable_means = jnp.stack(means)
    sampling_errors = jnp.stack(errors)
    weak_residuals = jnp.stack(residuals)
    valid = (
        ensemble.valid
        & jnp.all(jnp.isfinite(observable_means))
        & jnp.all(jnp.isfinite(sampling_errors))
        & jnp.all(jnp.isfinite(weak_residuals))
    )
    return ParticleFokkerPlanckResult(
        ensemble=ensemble,
        laws=laws,
        observable_means=observable_means,
        sampling_errors=sampling_errors,
        weak_residuals=weak_residuals,
        valid=valid,
        approximation_kind="particle-weak-law",
        bounded_non_claim=(
            "The result is a normalized finite empirical measure and certifies only "
            "the declared observables; it is not a reconstructed pointwise density."
        ),
    )


class SparseGridFokkerPlanckPlan(StrictModule):
    """Frozen sparse-grid density-fit epoch with an independent audit realization."""

    realization: IntegrationRealization
    validation_realization: IntegrationRealization
    solver: Any
    validation_operator: Any
    reference: str = eqx.field(static=True)
    state_var: str = eqx.field(static=True)

    def __init__(
        self,
        realization: IntegrationRealization,
        validation_realization: IntegrationRealization,
        solver: Any,
        /,
        *,
        validation_operator: Any = None,
        reference: str = "coordinate",
        state_var: str = "x",
    ):
        if not isinstance(realization, IntegrationRealization) or not isinstance(
            validation_realization, IntegrationRealization
        ):
            raise TypeError("realization values must be IntegrationRealization objects.")
        if not callable(solver):
            raise TypeError(
                "solver must be callable and return a raw log DomainFunction."
            )
        if validation_operator is not None and not callable(validation_operator):
            raise TypeError("validation_operator must be callable or None.")
        self.realization = realization
        self.validation_realization = validation_realization
        self.solver = solver
        self.validation_operator = validation_operator
        self.reference = str(reference)
        self.state_var = str(state_var)


class DensityFokkerPlanckResult(StrictModule):
    normalized_density: NormalizedDensityField
    held_out_residual: Any
    normalization_error: Array | None
    valid: Array
    approximation_kind: str = eqx.field(static=True)
    bounded_non_claim: str = eqx.field(static=True)


def solve_sparse_grid_fokker_planck(
    plan: SparseGridFokkerPlanckPlan, /
) -> DensityFokkerPlanckResult:
    """Fit and represented-normalize one finite sparse-grid density epoch."""
    if not isinstance(plan, SparseGridFokkerPlanckPlan):
        raise TypeError("plan must be a SparseGridFokkerPlanckPlan.")
    log_field = plan.solver(plan.realization)
    normalized = normalize_density_field(
        log_field,
        plan.realization,
        reference=plan.reference,
        state_var=plan.state_var,
    )
    if plan.validation_operator is None:
        held_out = None
        residual_valid = jnp.asarray(True)
    else:
        residual_field = plan.validation_operator(normalized.field)
        held_out = reduce(residual_field * residual_field, plan.validation_realization)
        residual_valid = held_out.successful & jnp.isfinite(jnp.asarray(held_out.value))
    error = normalized.normalization.error_estimate
    valid = normalized.finite & normalized.normalization.successful & residual_valid
    return DensityFokkerPlanckResult(
        normalized_density=normalized,
        held_out_residual=held_out,
        normalization_error=error,
        valid=valid,
        approximation_kind="sparse-grid-normalized-density",
        bounded_non_claim=(
            "Normalization and residuals refer to frozen finite integration "
            "realizations; adaptive topology changes require a new epoch."
        ),
    )


__all__ = [
    "DensityFokkerPlanckResult",
    "ParticleFokkerPlanckPlan",
    "ParticleFokkerPlanckResult",
    "SparseGridFokkerPlanckPlan",
    "WeakObservable",
    "solve_particle_fokker_planck",
    "solve_sparse_grid_fokker_planck",
]

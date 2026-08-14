#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._doc import DOC_KEY0
from .._term import AbstractEvaluatedScalarTerm, TermEvaluation
from ..domain import DomainFunction
from ..integration import DiscreteMeasureTarget, weighted, WeightedSampleTarget
from ..transport import (
    AbstractBalancedTransportSolver,
    FixedSupportBarycenterProblem,
    PreparedSinkhornReference,
    sinkhorn_divergence_against,
    SinkhornBarycenter,
    sliced_wasserstein_distance,
    soft_quantile,
)


Provider = Callable[[Mapping[str, DomainFunction]], Any] | Any


class SpatialSinkhornDivergenceTerm(AbstractEvaluatedScalarTerm):
    """Sinkhorn divergence from a model-built physical measure to a fixed reference."""

    objective_vars: tuple[str, ...]
    measure_builder: Callable[
        [Mapping[str, DomainFunction]], DiscreteMeasureTarget | WeightedSampleTarget
    ]
    reference: PreparedSinkhornReference
    encoder: Callable[[Any], Any] | None
    weight: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        measure_builder: Callable[
            [Mapping[str, DomainFunction]], DiscreteMeasureTarget | WeightedSampleTarget
        ],
        reference: PreparedSinkhornReference,
        /,
        *,
        encoder: Callable[[Any], Any] | None = None,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not callable(measure_builder):
            raise TypeError("measure_builder must be callable.")
        if not isinstance(reference, PreparedSinkhornReference):
            raise TypeError("reference must be a PreparedSinkhornReference.")
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.measure_builder = measure_builder
        self.reference = reference
        self.encoder = encoder
        self.weight = _scalar_weight(weight)
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del key, kwargs
        source = self.measure_builder(functions)
        if not isinstance(source, (DiscreteMeasureTarget, WeightedSampleTarget)):
            raise TypeError(
                "measure_builder must return a DiscreteMeasureTarget or "
                "WeightedSampleTarget."
            )
        result = sinkhorn_divergence_against(
            source,
            self.reference,
            encoder=self.encoder,
        )
        value = eqx.error_if(
            result.value,
            ~result.converged,
            "SpatialSinkhornDivergenceTerm transport did not converge.",
        )
        return TermEvaluation(self.weight * value, diagnostics=result)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


class EmpiricalSinkhornDivergenceTerm(AbstractEvaluatedScalarTerm):
    """Sinkhorn divergence between a model-generated empirical law and a reference."""

    objective_vars: tuple[str, ...]
    samples: Provider
    log_weights: Provider | None
    reference: PreparedSinkhornReference
    encoder: Callable[[Any], Any] | None
    weight: Array
    sample_axis: int = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        samples: Provider,
        reference: PreparedSinkhornReference,
        /,
        *,
        log_weights: Provider | None = None,
        sample_axis: int = 0,
        encoder: Callable[[Any], Any] | None = None,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(reference, PreparedSinkhornReference):
            raise TypeError("reference must be a PreparedSinkhornReference.")
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.samples = samples
        self.log_weights = log_weights
        self.reference = reference
        self.encoder = encoder
        self.weight = _scalar_weight(weight)
        self.sample_axis = int(sample_axis)
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del key, kwargs
        samples = _resolve(self.samples, functions)
        log_weights = _resolve(self.log_weights, functions)
        if log_weights is None:
            leaf = _sample_leaf(samples)
            axis = (
                self.sample_axis + leaf.ndim if self.sample_axis < 0 else self.sample_axis
            )
            if axis < 0 or axis >= leaf.ndim:
                raise ValueError("sample_axis is out of range for empirical samples.")
            log_weights = jnp.zeros((leaf.shape[axis],), dtype=float)
        source = weighted(
            samples,
            log_weights,
            normalized=True,
            sample_axes=self.sample_axis,
            provenance="empirical-sinkhorn-term",
        )
        result = sinkhorn_divergence_against(
            source,
            self.reference,
            encoder=self.encoder,
        )
        value = eqx.error_if(
            result.value,
            ~result.converged,
            "EmpiricalSinkhornDivergenceTerm transport did not converge.",
        )
        return TermEvaluation(self.weight * value, diagnostics=result)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


class BarycenterObjectiveTerm(AbstractEvaluatedScalarTerm):
    """Composable scalar objective for a model-built finite barycenter problem."""

    objective_vars: tuple[str, ...]
    problem_builder: Callable[
        [Mapping[str, DomainFunction]], FixedSupportBarycenterProblem
    ]
    solver: SinkhornBarycenter
    weight: Array
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        problem_builder: Callable[
            [Mapping[str, DomainFunction]], FixedSupportBarycenterProblem
        ],
        solver: SinkhornBarycenter,
        /,
        *,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not callable(problem_builder):
            raise TypeError("problem_builder must be callable.")
        if not isinstance(solver, SinkhornBarycenter):
            raise TypeError("solver must be a SinkhornBarycenter.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.problem_builder = problem_builder
        self.solver = solver
        self.weight = _scalar_weight(weight)
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del key, kwargs
        problem = self.problem_builder(functions)
        if not isinstance(problem, FixedSupportBarycenterProblem):
            raise TypeError(
                "problem_builder must return a FixedSupportBarycenterProblem."
            )
        result = self.solver(problem)
        value = eqx.error_if(
            result.objective,
            ~result.converged,
            "BarycenterObjectiveTerm transport did not converge.",
        )
        return TermEvaluation(self.weight * value, diagnostics=result)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


class SlicedWassersteinTerm(AbstractEvaluatedScalarTerm):
    """Sliced Wasserstein discrepancy between model and reference event samples."""

    objective_vars: tuple[str, ...]
    samples: Provider
    target_samples: Provider
    source_weights: Provider | None
    target_weights: Provider | None
    projections: Array | None
    weight: Array
    p: float = eqx.field(static=True)
    num_projections: int = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        samples: Provider,
        target_samples: Provider,
        /,
        *,
        source_weights: Provider | None = None,
        target_weights: Provider | None = None,
        p: float = 2.0,
        num_projections: int = 128,
        projections: ArrayLike | None = None,
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.samples = samples
        self.target_samples = target_samples
        self.source_weights = source_weights
        self.target_weights = target_weights
        self.projections = (
            None if projections is None else jnp.asarray(projections, dtype=float)
        )
        self.weight = _scalar_weight(weight)
        self.p = float(p)
        self.num_projections = int(num_projections)
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del kwargs
        result = sliced_wasserstein_distance(
            _resolve(self.samples, functions),
            _resolve(self.target_samples, functions),
            source_weights=_resolve(self.source_weights, functions),
            target_weights=_resolve(self.target_weights, functions),
            p=self.p,
            num_projections=self.num_projections,
            key=None if self.projections is not None else key,
            projections=self.projections,
        )
        return TermEvaluation(self.weight * result.value, diagnostics=result)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


class SoftQuantileFunctional(AbstractEvaluatedScalarTerm):
    """Penalty on relaxed empirical quantiles with exact hard endpoints.

    Interior quantiles inherit the regularity of the soft-order solve. Exact
    endpoints are only almost-everywhere differentiable, and absolute discrepancy
    additionally has a kink at zero residual.
    """

    objective_vars: tuple[str, ...]
    values: Provider
    weights: Provider | None
    q: Array
    target_quantiles: Array
    solver: AbstractBalancedTransportSolver | None
    weight: Array
    axis: int | str = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    discrepancy: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        values: Provider,
        q: ArrayLike,
        target_quantiles: ArrayLike,
        /,
        *,
        weights: Provider | None = None,
        axis: int | str = -1,
        epsilon: float = 0.1,
        solver: AbstractBalancedTransportSolver | None = None,
        discrepancy: Literal["squared", "absolute"] = "squared",
        objective_vars: Sequence[str] | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        mode = str(discrepancy).lower()
        if mode not in ("squared", "absolute"):
            raise ValueError("discrepancy must be 'squared' or 'absolute'.")
        self.objective_vars = () if objective_vars is None else tuple(objective_vars)
        self.values = values
        self.weights = weights
        self.q = jnp.asarray(q, dtype=float)
        self.target_quantiles = jnp.asarray(target_quantiles, dtype=float)
        if self.target_quantiles.shape != self.q.shape:
            raise ValueError("target_quantiles must have the same shape as q.")
        self.solver = solver
        self.weight = _scalar_weight(weight)
        self.axis = axis
        epsilon_ = float(epsilon)
        if not math.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        self.epsilon = epsilon_
        self.discrepancy = mode
        self.label = None if label is None else str(label)

    def term_evaluation(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> TermEvaluation:
        del key, kwargs
        estimate = soft_quantile(
            _resolve(self.values, functions),
            self.q,
            weights=_resolve(self.weights, functions),
            axis=self.axis,
            epsilon=self.epsilon,
            solver=self.solver,
        )
        estimate_data = estimate.data if isinstance(estimate, cx.Field) else estimate
        residual = jnp.asarray(estimate_data) - self.target_quantiles
        if residual.shape != self.target_quantiles.shape:
            raise ValueError(
                "SoftQuantileFunctional values contain retained batch axes; "
                "target_quantiles must describe a scalar empirical law."
            )
        penalty = residual**2 if self.discrepancy == "squared" else jnp.abs(residual)
        effective_epsilon = (
            self.solver.epsilon
            if self.solver is not None
            else jnp.asarray(self.epsilon, dtype=residual.dtype)
        )
        endpoint_mask = (self.q == 0.0) | (self.q == 1.0)
        diagnostics = {
            "effective_epsilon": jnp.asarray(effective_epsilon),
            "endpoint_mask": endpoint_mask,
            "quantiles": jnp.asarray(estimate_data),
            "target_quantiles": self.target_quantiles,
            "residuals": residual,
        }
        return TermEvaluation(self.weight * jnp.mean(penalty), diagnostics=diagnostics)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Array = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        return self.term_evaluation(functions, key=key, **kwargs).value


def _resolve(value: Provider | None, functions: Mapping[str, DomainFunction], /):
    return value(functions) if callable(value) else value


def _sample_leaf(samples: Any, /) -> Array:
    if isinstance(samples, cx.Field):
        return jnp.asarray(samples.data)
    if eqx.is_array(samples):
        return jnp.asarray(samples)
    leaves = [leaf for leaf in jax.tree.leaves(samples) if eqx.is_array(leaf)]
    if not leaves:
        raise ValueError("Empirical samples must contain at least one array leaf.")
    return jnp.asarray(leaves[0])


def _scalar_weight(value: ArrayLike, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError("Term weight must be scalar.")
    return result


__all__ = [
    "BarycenterObjectiveTerm",
    "EmpiricalSinkhornDivergenceTerm",
    "SlicedWassersteinTerm",
    "SoftQuantileFunctional",
    "SpatialSinkhornDivergenceTerm",
]

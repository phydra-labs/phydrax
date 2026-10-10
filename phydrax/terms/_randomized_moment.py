#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._precision import PrecisionEvidenceEnvelope
from .._randomized_residual_modes import (
    RandomizedResidualLossMode,
    RealizationSamplingDesign,
)
from .._sampling._moments import (
    realization_moments,
    uncertainty_available,
    validate_sampling_design,
)
from .._sampling._types import IIDDesign
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..conditions._base import AbstractMomentCondition
from ..integration import (
    IntegrationPrecisionPolicy,
    IntegrationRealization,
    PerStepIntegration,
    reduce,
)
from ..integration._api import _requires_random_key
from ..integration._execution import resolve_integration
from ..integration._plans import (
    ImportanceSamplingPlan,
    MonteCarloPlan,
    MultilevelMonteCarloPlan,
    ProductIntegrationPlan,
    QuasiMonteCarloPlan,
    SampleMeanEstimator,
    StratifiedMonteCarloPlan,
)
from ..integration._targets import DensityTarget
from ..typing import checked, parse, PRNGKey
from ._integrated import checked_estimate_field, validate_condition_source
from ._randomized_quadratic import event_inner, randomized_squared_mean


class RandomizedMomentBatch(StrictModule):
    """Independent integration realizations frozen for one optimizer update."""

    left: tuple[IntegrationRealization, ...]
    right: tuple[IntegrationRealization, ...] | None
    sampling_design: RealizationSamplingDesign = eqx.field(static=True)
    population_size: int | None = eqx.field(static=True)
    right_sampling_design: RealizationSamplingDesign = eqx.field(static=True)
    right_population_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        left: tuple[IntegrationRealization, ...],
        right: tuple[IntegrationRealization, ...] | None = None,
        /,
        *,
        sampling_design: RealizationSamplingDesign = "unknown",
        population_size: int | None = None,
        right_sampling_design: RealizationSamplingDesign = "unknown",
        right_population_size: int | None = None,
    ) -> None:
        design = validate_sampling_design(sampling_design, population_size, len(left))
        if any(not isinstance(item, IntegrationRealization) for item in left):
            raise TypeError("left must contain only IntegrationRealization values.")
        if right is not None:
            right_design = validate_sampling_design(
                right_sampling_design, right_population_size, len(right)
            )
            if any(not isinstance(item, IntegrationRealization) for item in right):
                raise TypeError("right must contain only IntegrationRealization values.")
        else:
            right_design = parse(
                right_sampling_design, RealizationSamplingDesign, "right_sampling_design"
            )
        self.left = tuple(left)
        self.right = None if right is None else tuple(right)
        self.sampling_design = design
        self.population_size = population_size
        self.right_sampling_design = right_design
        self.right_population_size = right_population_size


class RandomizedMomentDiagnostics(StrictModule):
    """Estimator-aware moment objective and independent-integration evidence."""

    objective: Array
    plug_in_moment_norm: Array
    mean_standard_error: Array
    negative: Array
    finite: Array
    integration_diagnostics: tuple[Any, ...]
    precision_evidence: PrecisionEvidenceEnvelope
    num_realizations: int = eqx.field(static=True)
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)
    uncertainty_available: bool = eqx.field(static=True)
    sampling_design: RealizationSamplingDesign = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.finite)


def _integration_sampling_design(
    source: PerStepIntegration, /
) -> RealizationSamplingDesign:
    if isinstance(source.target, DensityTarget) and source.target.normalized:
        return "unknown"
    return _plan_sampling_design(source.plan)


def _plan_sampling_design(plan: Any, /) -> RealizationSamplingDesign:
    """Declare iid *replicate means* only for native known-unbiased integration."""
    if isinstance(plan, ProductIntegrationPlan):
        # Each realization keys its randomized factors independently and repeats
        # deterministic factor nodes, so replicate means are iid exactly when
        # every randomized factor is.
        designs = tuple(
            _plan_sampling_design(factor)
            for factor in plan.plans.values()
            if _requires_random_key(factor)
        )
        return "iid" if designs and all(item == "iid" for item in designs) else "unknown"
    if isinstance(plan, ImportanceSamplingPlan):
        return "iid" if isinstance(plan.estimator, SampleMeanEstimator) else "unknown"
    if isinstance(plan, (MonteCarloPlan, QuasiMonteCarloPlan)):
        if isinstance(plan, QuasiMonteCarloPlan) and not plan.design.scrambled:
            return "unknown"
        control = plan.control_variate
        if control is not None and control.coefficients is None:
            if control.same_sample_asymptotic or not isinstance(plan.design, IIDDesign):
                return "unknown"
        return "iid"
    if isinstance(plan, MultilevelMonteCarloPlan):
        # Adaptive allocation reuses the observations that decide when to stop.
        # A finite finest-level mean does not certify an infinite-limit mean.
        return (
            "iid"
            if plan.samples_per_level is not None and plan.estimand == "finest_level"
            else "unknown"
        )
    if isinstance(plan, StratifiedMonteCarloPlan):
        return "iid"
    return "unknown"


class RandomizedMomentPenalty(AbstractSamplingTerm):
    """Estimator-aware squared moment mismatch under resampled integration."""

    condition: AbstractMomentCondition
    source: PerStepIntegration
    fields: tuple[str, ...] = eqx.field(static=True)
    scale: Array
    weight: Array
    num_realizations: int = eqx.field(static=True)
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)
    precision: IntegrationPrecisionPolicy
    label: str | None = eqx.field(static=True)
    sampling_design: RealizationSamplingDesign = eqx.field(static=True)

    @checked
    def __init__(
        self,
        condition: AbstractMomentCondition,
        source: PerStepIntegration,
        /,
        *,
        num_realizations: int = 2,
        loss_mode: RandomizedResidualLossMode = "u_statistic",
        scale: ArrayLike = 1.0,
        label: str | None = None,
        precision: IntegrationPrecisionPolicy | None = None,
    ) -> None:
        if not _requires_random_key(source.plan):
            raise ValueError(
                "RandomizedMomentPenalty requires a randomized integration plan; "
                "use MomentPenalty for deterministic or fixed integration."
            )
        validate_condition_source(condition.on, source)
        count = int(num_realizations)
        loss_mode = parse(loss_mode, RandomizedResidualLossMode, "loss_mode")
        if count < 1 or (loss_mode == "u_statistic" and count < 2):
            raise ValueError(
                "num_realizations must be positive, and u_statistic requires at least two."
            )
        coefficient = jnp.asarray(scale, dtype=jnp.float64)
        if coefficient.shape != ():
            raise ValueError("Term scale must be a scalar.")
        if not bool(jnp.isfinite(coefficient)) or float(coefficient) < 0.0:
            raise ValueError("Term scale must be finite and nonnegative.")
        precision_ = IntegrationPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, IntegrationPrecisionPolicy):
            raise TypeError("precision must be an IntegrationPrecisionPolicy or None.")
        self.condition = condition
        self.source = source
        self.fields = condition.fields
        self.scale = coefficient.reshape(())
        self.weight = self.scale
        self.num_realizations = count
        self.loss_mode = loss_mode
        self.precision = precision_
        self.label = condition.label if label is None else str(label)
        self.sampling_design = _integration_sampling_design(source)

    def sample(self, *, key: PRNGKey = DOC_KEY0) -> RandomizedMomentBatch:
        group_count = 2 if self.loss_mode == "independent_product" else 1
        keys = tuple(jr.split(key, group_count * self.num_realizations))
        left = tuple(
            resolve_integration(self.source, key=sample_key)
            for sample_key in keys[: self.num_realizations]
        )
        right = (
            tuple(
                resolve_integration(self.source, key=sample_key)
                for sample_key in keys[self.num_realizations :]
            )
            if group_count == 2
            else None
        )
        return RandomizedMomentBatch(
            left,
            right,
            sampling_design=self.sampling_design,
            right_sampling_design=self.sampling_design,
        )

    def _mismatch(
        self,
        integrand: DomainFunction,
        realization: IntegrationRealization,
        /,
        *,
        runtime_kwargs: Mapping[str, Any],
    ) -> tuple[Array, Any, PrecisionEvidenceEnvelope | None]:
        estimate = reduce(integrand, realization, **runtime_kwargs)
        field = checked_estimate_field(estimate)
        named_dims = tuple(dim for dim in field.dims if dim is not None)
        if named_dims:
            raise ValueError(
                "RandomizedMomentPenalty integration left sampling dimensions "
                f"{named_dims!r}; the source must integrate them all."
            )
        integrated = jnp.asarray(field.data)
        target = jnp.asarray(self.condition.target)
        if jnp.broadcast_shapes(integrated.shape, target.shape) != integrated.shape:
            raise ValueError(
                f"Moment target shape {target.shape} cannot broadcast to integrated shape {integrated.shape}."
            )
        return integrated - target, estimate.diagnostics, estimate.precision_evidence

    @checked
    def _evaluate(
        self,
        functions: Mapping[str, DomainFunction],
        batch: RandomizedMomentBatch,
        /,
        *,
        iter_: int | Array | None,
        kwargs: Mapping[str, Any],
    ) -> tuple[
        Array,
        Array | None,
        tuple[Any, ...],
        tuple[PrecisionEvidenceEnvelope | None, ...],
    ]:
        runtime_kwargs = dict(kwargs)
        if iter_ is not None:
            runtime_kwargs["iter_"] = iter_
        integrand = self.condition.integrand(functions)
        left_results = tuple(
            self._mismatch(
                integrand,
                realization,
                runtime_kwargs=runtime_kwargs,
            )
            for realization in batch.left
        )
        left = self.precision.accumulation(
            jnp.stack(tuple(value for value, _, _ in left_results), axis=0)
        )
        diagnostics = tuple(item for _, item, _ in left_results)
        evidences = tuple(evidence for _, _, evidence in left_results)
        if batch.right is None:
            return left, None, diagnostics, evidences
        right_results = tuple(
            self._mismatch(
                integrand,
                realization,
                runtime_kwargs=runtime_kwargs,
            )
            for realization in batch.right
        )
        right = self.precision.accumulation(
            jnp.stack(tuple(value for value, _, _ in right_results), axis=0)
        )
        if (
            self.loss_mode == "independent_product"
            and batch.sampling_design != "exact"
            and batch.right_sampling_design != "exact"
            and batch.sampling_design != "unknown"
            and batch.right_sampling_design != "unknown"
        ):
            left_keys = tuple(
                parse(realization.key, PRNGKey, "left realization key")
                for realization in batch.left
            )
            right_keys = tuple(
                parse(realization.key, PRNGKey, "right realization key")
                for realization in batch.right
            )
            left_words = jr.key_data(jnp.stack(left_keys))
            right_words = jr.key_data(jnp.stack(right_keys))
            shared = jnp.any(
                jnp.all(left_words[:, None, :] == right_words[None, :, :], axis=-1)
            )
            right = eqx.error_if(
                right,
                shared,
                "independent_product requires distinct owner-generated integration keys.",
            )
        right_evidence = tuple(evidence for _, _, evidence in right_results)
        return (
            left,
            right,
            diagnostics + tuple(item for _, item, _ in right_results),
            evidences + right_evidence,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: PRNGKey = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: RandomizedMomentBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        materialized = self.sample(key=key) if batch is None else batch
        left, right, _, _ = self._evaluate(
            functions,
            materialized,
            iter_=iter_,
            kwargs=kwargs,
        )
        value = randomized_squared_mean(
            left,
            tuple(left.shape[1:]),
            self.loss_mode,
            right=right,
            precision=self.precision,
            sampling_design=materialized.sampling_design,
            right_sampling_design=materialized.right_sampling_design,
        )
        return self.precision.decision(
            self.precision.decision(self.scale) * value
        ).reshape(())

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: PRNGKey = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: RandomizedMomentBatch | None = None,
        **kwargs: Any,
    ) -> RandomizedMomentDiagnostics:
        materialized = self.sample(key=key) if batch is None else batch
        left, right, integration_diagnostics, integration_evidence = self._evaluate(
            functions,
            materialized,
            iter_=iter_,
            kwargs=kwargs,
        )
        event_shape = tuple(left.shape[1:])
        objective = self.precision.decision(self.scale) * randomized_squared_mean(
            left,
            event_shape,
            self.loss_mode,
            right=right,
            precision=self.precision,
            sampling_design=materialized.sampling_design,
            right_sampling_design=materialized.right_sampling_design,
        )
        mean, _, standard_error = realization_moments(
            self.precision.accumulation(left),
            materialized.sampling_design,
            materialized.population_size,
        )
        plug_in_norm = self.precision.decision(
            jnp.sqrt(
                event_inner(
                    mean,
                    event_shape,
                    precision=self.precision,
                )
            )
        )
        mean_standard_error = self.precision.decision(
            jnp.sqrt(
                event_inner(
                    standard_error,
                    event_shape,
                    precision=self.precision,
                )
            )
        )
        available = uncertainty_available(materialized.sampling_design, left.shape[0])
        finite = jnp.isfinite(objective) & jnp.isfinite(plug_in_norm)
        if available:
            finite = finite & jnp.isfinite(mean_standard_error)
        children = {
            f"integration-{index}": evidence
            for index, evidence in enumerate(integration_evidence)
            if evidence is not None
        }
        precision_evidence = self.precision.evidence_for(
            left,
            children=children,
        )
        return RandomizedMomentDiagnostics(
            objective=self.precision.decision(objective).reshape(()),
            plug_in_moment_norm=plug_in_norm.reshape(()),
            mean_standard_error=mean_standard_error.reshape(()),
            negative=objective < 0.0,
            finite=finite,
            integration_diagnostics=integration_diagnostics,
            num_realizations=left.shape[0],
            loss_mode=self.loss_mode,
            precision_evidence=precision_evidence,
            uncertainty_available=available,
            sampling_design=materialized.sampling_design,
        )


__all__ = [
    "RandomizedMomentBatch",
    "RandomizedMomentDiagnostics",
    "RandomizedMomentPenalty",
]

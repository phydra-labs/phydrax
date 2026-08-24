#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.conditions import AbstractMomentCondition
from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..integration import (
    IntegrationPrecisionPolicy,
    IntegrationRealization,
    PerStepIntegration,
    reduce,
)
from ..integration._api import _requires_random_key
from ..integration._execution import resolve_integration
from ._integrated import checked_estimate_field, validate_condition_source
from ._randomized_quadratic import event_inner, randomized_squared_mean
from ._randomized_residual import RandomizedResidualLossMode


class RandomizedMomentBatch(StrictModule):
    """Independent integration realizations frozen for one optimizer update."""

    left: tuple[IntegrationRealization, ...]
    right: tuple[IntegrationRealization, ...] | None

    def __init__(
        self,
        left: tuple[IntegrationRealization, ...],
        right: tuple[IntegrationRealization, ...] | None = None,
        /,
    ):
        if len(left) < 2:
            raise ValueError("Randomized moments require at least two realizations.")
        if any(not isinstance(item, IntegrationRealization) for item in left):
            raise TypeError("left must contain only IntegrationRealization values.")
        if right is not None:
            if len(right) != len(left):
                raise ValueError("Independent realization groups must have equal sizes.")
            if any(not isinstance(item, IntegrationRealization) for item in right):
                raise TypeError("right must contain only IntegrationRealization values.")
        self.left = tuple(left)
        self.right = None if right is None else tuple(right)


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

    @property
    def passed(self) -> bool:
        return bool(self.finite)


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
    ):
        if not isinstance(condition, AbstractMomentCondition):
            raise TypeError(
                "RandomizedMomentPenalty requires an AbstractMomentCondition."
            )
        if not isinstance(source, PerStepIntegration):
            raise TypeError(
                "RandomizedMomentPenalty requires a PerStepIntegration source."
            )
        if not _requires_random_key(source.plan):
            raise ValueError(
                "RandomizedMomentPenalty requires a randomized integration plan; "
                "use MomentPenalty for deterministic or fixed integration."
            )
        validate_condition_source(condition.on, source)
        count = int(num_realizations)
        if count < 2:
            raise ValueError("num_realizations must be at least two.")
        if loss_mode not in ("u_statistic", "independent_product", "plug_in"):
            raise ValueError("Unknown randomized moment loss_mode.")
        coefficient = jnp.asarray(scale, dtype=float)
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

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> RandomizedMomentBatch:
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
        return RandomizedMomentBatch(left, right)

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
                f"Moment target shape {target.shape} cannot broadcast to "
                f"integrated shape {integrated.shape}."
            )
        return integrated - target, estimate.diagnostics, estimate.precision_evidence

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
        if not isinstance(batch, RandomizedMomentBatch):
            raise TypeError("batch must be a RandomizedMomentBatch.")
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
        key: Key[Array, ""] = DOC_KEY0,
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
            tuple(int(size) for size in left.shape[1:]),
            self.loss_mode,
            right=right,
            precision=self.precision,
        )
        return self.precision.decision(
            self.precision.decision(self.scale) * value
        ).reshape(())

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
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
        event_shape = tuple(int(size) for size in left.shape[1:])
        objective = self.precision.decision(self.scale) * randomized_squared_mean(
            left,
            event_shape,
            self.loss_mode,
            right=right,
            precision=self.precision,
        )
        mean = jnp.mean(self.precision.accumulation(left), axis=0)
        centered = self.precision.accumulation(left - mean)
        variance = jnp.sum(
            self.precision.accumulation(jnp.abs(centered) ** 2),
            axis=0,
        ) / float(self.num_realizations - 1)
        standard_error = jnp.sqrt(
            self.precision.accumulation(variance / float(self.num_realizations))
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
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(plug_in_norm)
            & jnp.isfinite(mean_standard_error)
        )
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
            num_realizations=self.num_realizations,
            loss_mode=self.loss_mode,
            precision_evidence=precision_evidence,
        )


__all__ = [
    "RandomizedMomentBatch",
    "RandomizedMomentDiagnostics",
    "RandomizedMomentPenalty",
]

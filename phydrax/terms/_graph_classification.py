#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import BatchEvaluator, DomainComponent, DomainFunction, PointSampling
from phydrax.domain.graph import (
    graph_component_kind,
    GraphBatch,
    GraphComponentKind,
    GraphDatasetDomain,
    GraphTrajectoryDatasetDomain,
)

from .._classification import ClassificationKind, pointwise_classification_loss
from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain.graph._observation import (
    _graph_classification_target,
    _graph_trajectory_classification_signal,
    GraphClassificationTarget,
    GraphClassificationTargetEncoding,
    GraphTargetInterpolation,
    GraphTrajectoryClassificationSignal,
)
from ..integration import mean_over, over, per_step
from ..ml._classification import ClassificationObjective, ClassificationObjectiveKind
from ..ml._schema import TargetSchema
from ._integral_functional import IntegralFunctional


GraphClassificationReduction = Literal["mean", "integral"]


def _classification_configuration(
    target_schema: TargetSchema,
    objective: ClassificationObjective,
    /,
) -> tuple[ClassificationKind, int | None]:
    if not isinstance(target_schema, TargetSchema):
        raise TypeError("Graph classification requires a TargetSchema.")
    if not isinstance(objective, ClassificationObjective):
        raise TypeError("objective must be a ClassificationObjective.")
    kind = target_schema.kind
    if kind not in ("binary", "multiclass", "multilabel", "ordinal"):
        raise ValueError(
            "Graph classification TargetSchema kind must be binary, multiclass, "
            "multilabel, or ordinal."
        )

    class_count: int | None = None
    if kind in ("multiclass", "ordinal"):
        class_count = target_schema.num_classes
        if class_count < 2:
            raise ValueError(f"{kind} TargetSchema requires at least two classes.")
    elif kind == "multilabel":
        class_count = len(target_schema.names)
        if class_count < 1:
            raise ValueError("Multilabel TargetSchema requires at least one label name.")

    thresholds = objective.thresholds
    if kind == "ordinal":
        if objective.kind != "nll" or thresholds is None:
            raise ValueError(
                "Ordinal graph classification requires ClassificationObjective.nll "
                "with thresholds."
            )
        if class_count is None or len(thresholds) != class_count - 1:
            raise ValueError(
                "Ordinal objective threshold count must equal num_classes - 1."
            )
    elif thresholds is not None:
        raise ValueError(
            "Classification thresholds are defined only for ordinal targets."
        )

    if objective.kind == "focal" and objective.alpha is not None:
        alpha = objective.alpha
        if kind in ("binary", "multilabel"):
            if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
                raise ValueError(f"{kind} focal alpha must be scalar in (0, 1).")
        elif kind == "multiclass":
            if (
                not isinstance(alpha, tuple)
                or class_count is None
                or len(alpha) != class_count
                or any(value <= 0.0 for value in alpha)
            ):
                raise ValueError(
                    "Multiclass focal alpha must contain one positive weight per class."
                )
        else:
            raise ValueError("Ordinal focal objectives are unsupported.")
    return kind, class_count


def _component_kind(
    component: DomainComponent,
    graph_label: str,
    /,
) -> GraphComponentKind:
    return graph_component_kind(component.spec.selection_for(graph_label))


class _GraphClassificationScore(StrictModule, BatchEvaluator):
    logits: DomainFunction
    target: DomainFunction
    target_mask: DomainFunction | None
    classification_kind: ClassificationKind = eqx.field(static=True)
    objective_kind: ClassificationObjectiveKind = eqx.field(static=True)
    class_count: int | None = eqx.field(static=True)
    multilabel_count: int | None = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    alpha: float | tuple[float, ...] | None = eqx.field(static=True)
    thresholds: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        logits: DomainFunction,
        target: DomainFunction,
        target_mask: DomainFunction | None,
        classification_kind: ClassificationKind,
        objective: ClassificationObjective,
        class_count: int | None,
    ):
        self.logits = logits
        self.target = target
        self.target_mask = target_mask
        self.classification_kind = classification_kind
        self.objective_kind = objective.kind
        self.class_count = class_count if classification_kind != "multilabel" else None
        self.multilabel_count = (
            class_count if classification_kind == "multilabel" else None
        )
        self.gamma = objective.gamma
        self.alpha = objective.alpha
        self.thresholds = objective.thresholds

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("Graph classification scores require GraphBatch evaluation.")

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("Graph classification scores require GraphBatch evaluation.")
        logits = self.logits(batch, key=key, **kwargs)
        target = self.target(batch, key=key, **kwargs)
        mask = (
            None
            if self.target_mask is None
            else self.target_mask(batch, key=key, **kwargs)
        )
        if not isinstance(logits, cx.Field) or not isinstance(target, cx.Field):
            raise TypeError("Graph logits and classification targets must be Fields.")
        if mask is not None and not isinstance(mask, cx.Field):
            raise TypeError("Graph classification target masks must be Fields.")

        axis = batch.structure.axis_for(batch.graph_label)
        if axis is None:
            raise ValueError("GraphBatch is missing its graph sampling axis.")
        logits_data = jnp.asarray(logits.data)
        target_data = jnp.asarray(target.data)
        if logits_data.ndim == 0 or logits.dims[0] != axis:
            raise ValueError("Graph logits must begin with the graph sampling axis.")
        if target_data.ndim == 0 or target.dims[0] != axis:
            raise ValueError(
                "Graph classification targets must begin with the graph sampling axis."
            )
        if self.classification_kind in ("binary", "ordinal"):
            if logits_data.ndim != 1:
                raise ValueError(
                    f"{self.classification_kind} graph logits must be scalar per entity."
                )
        else:
            if logits_data.ndim != 2:
                raise ValueError(
                    f"{self.classification_kind} graph logits must have one terminal "
                    "statistical axis."
                )
        if (
            self.classification_kind == "multilabel"
            and int(logits_data.shape[-1]) != self.multilabel_count
        ):
            raise ValueError(
                "Multilabel graph logits terminal axis must match TargetSchema.names."
            )

        score = pointwise_classification_loss(
            logits_data,
            target_data,
            kind=self.classification_kind,
            objective=self.objective_kind,
            class_count=self.class_count,
            target_mask=None if mask is None else jnp.asarray(mask.data),
            gamma=self.gamma,
            alpha=self.alpha,
            thresholds=self.thresholds,
        )
        if mask is not None:
            mask_data = jnp.asarray(mask.data)
            observation_active = (
                jnp.any(mask_data, axis=-1)
                if self.classification_kind == "multilabel"
                and mask_data.ndim == score.ndim + 1
                else mask_data
            )
            score = jnp.where(observation_active, score, 0.0)
            active_count = jnp.sum(observation_active)
            score = jnp.where(
                active_count > 0,
                score * score.size / active_count,
                jnp.zeros_like(score),
            )
        if score.ndim != 1 or score.shape[0] != logits_data.shape[0]:
            raise ValueError(
                "Graph classification must produce one scalar score per sampled entity."
            )
        return cx.Field(score, dims=(axis,))


class _GraphClassificationIntegrand(StrictModule):
    field: str = eqx.field(static=True)
    domain: GraphDatasetDomain | GraphTrajectoryDatasetDomain
    target: DomainFunction
    target_mask: DomainFunction | None
    target_schema: TargetSchema
    objective: ClassificationObjective
    class_count: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        field: str,
        domain: GraphDatasetDomain | GraphTrajectoryDatasetDomain,
        target: DomainFunction,
        target_mask: DomainFunction | None,
        target_schema: TargetSchema,
        objective: ClassificationObjective,
        class_count: int | None,
    ):
        self.field = field
        self.domain = domain
        self.target = target
        self.target_mask = target_mask
        self.target_schema = target_schema
        self.objective = objective
        self.class_count = class_count

    def __call__(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> DomainFunction:
        if self.field not in functions:
            raise KeyError(f"Missing graph classification field {self.field!r}.")
        logits = functions[self.field]
        if not isinstance(logits, DomainFunction):
            raise TypeError("Graph classification logits must be a DomainFunction.")
        if not logits.domain.same_support(self.domain):
            raise ValueError(
                "Graph classification logits and target domains are incompatible."
            )
        dependencies = logits.deps + self.target.deps
        if self.target_mask is not None:
            dependencies += self.target_mask.deps
        return DomainFunction(
            domain=self.domain,
            deps=tuple(dict.fromkeys(dependencies)),
            func=_GraphClassificationScore(
                logits=logits,
                target=self.target,
                target_mask=self.target_mask,
                classification_kind=self.target_schema.kind,
                objective=self.objective,
                class_count=self.class_count,
            ),
            metadata=logits.metadata,
        )


def _integration_target(
    component: DomainComponent,
    reduction: GraphClassificationReduction,
    /,
):
    if reduction == "mean":
        return mean_over(component)
    if reduction == "integral":
        return over(component)
    raise ValueError("reduction must be 'mean' or 'integral'.")


def _validate_aligned_target_shapes(
    target: DomainFunction,
    target_mask: DomainFunction | None,
    /,
    *,
    kind: ClassificationKind,
    class_count: int | None,
    target_encoding: GraphClassificationTargetEncoding,
) -> None:
    values = jnp.asarray(getattr(target.func, "values"))
    if kind in ("binary", "ordinal"):
        expected_target: tuple[int, ...] = ()
    else:
        if class_count is None:
            raise RuntimeError(f"{kind} graph classification is missing its axis size.")
        if kind == "multiclass":
            expected_target = () if target_encoding == "hard" else (class_count,)
        else:
            expected_target = (class_count,)
    if values.shape[1:] != expected_target:
        raise ValueError(
            f"{kind} graph targets must have trailing shape "
            f"{expected_target}, got {values.shape[1:]}."
        )
    if target_mask is None:
        return
    mask = jnp.asarray(getattr(target_mask.func, "values"))
    expected_mask = (class_count,) if kind == "multilabel" else ()
    if mask.shape[1:] != expected_mask:
        raise ValueError(
            f"{kind} graph target_mask must have trailing shape "
            f"{expected_mask}, got {mask.shape[1:]}."
        )


def _checked_weight(weight: ArrayLike, /) -> Array:
    value = jnp.asarray(weight, dtype=float)
    if value.shape != ():
        raise ValueError("weight must be a scalar.")
    if not bool(jnp.isfinite(value)) or float(value) < 0.0:
        raise ValueError("weight must be finite and nonnegative.")
    return value.reshape(())


def GraphClassificationTerm(
    field: str,
    component: DomainComponent,
    values: ArrayLike | Sequence[ArrayLike],
    target_schema: TargetSchema,
    /,
    *,
    sampling: PointSampling,
    objective: ClassificationObjective = ClassificationObjective.nll(),
    target_mask: ArrayLike | Sequence[ArrayLike] | None = None,
    weight: ArrayLike = 1.0,
    reduction: GraphClassificationReduction = "mean",
    label: str | None = None,
) -> IntegralFunctional:
    """Build a pointwise graph classification risk over graph geometry.

    The selected node, edge, global, subset, or cochain-cell component owns the
    sampling axis, mask, measure, and mean/integral reduction. Classification is
    scored directly from logits; it is not represented as a squared Observation
    residual and composes as an independent scalar term with graph physics terms.
    """
    if not isinstance(component, DomainComponent):
        raise TypeError("component must be a DomainComponent.")
    if not isinstance(sampling, PointSampling):
        raise TypeError("sampling must be a PointSampling.")
    if not isinstance(component.domain, GraphDatasetDomain):
        raise TypeError("GraphClassificationTerm requires a GraphDatasetDomain.")
    kind, class_count = _classification_configuration(target_schema, objective)
    domain = component.domain
    component_kind = _component_kind(component, domain.label)
    target = _graph_classification_target(
        domain,
        values,
        component_kind=component_kind,
        target_encoding=objective.target_encoding,
        require_boolean=False,
    )
    mask = (
        None
        if target_mask is None
        else _graph_classification_target(
            domain,
            target_mask,
            component_kind=component_kind,
            target_encoding="hard",
            require_boolean=True,
        )
    )
    _validate_aligned_target_shapes(
        target,
        mask,
        kind=kind,
        class_count=class_count,
        target_encoding=objective.target_encoding,
    )
    return IntegralFunctional(
        source=per_step(_integration_target(component, reduction), sampling),
        integrand=_GraphClassificationIntegrand(
            field=str(field),
            domain=domain,
            target=target,
            target_mask=mask,
            target_schema=target_schema,
            objective=objective,
            class_count=class_count,
        ),
        objective_vars=(str(field),),
        weight=_checked_weight(weight),
        label=label,
    )


def GraphTrajectoryClassificationTerm(
    field: str,
    component: DomainComponent,
    values: ArrayLike | Sequence[ArrayLike],
    target_schema: TargetSchema,
    /,
    *,
    sampling: PointSampling,
    objective: ClassificationObjective = ClassificationObjective.nll(),
    interpolation: GraphTargetInterpolation = "nearest",
    target_mask: ArrayLike | Sequence[ArrayLike] | None = None,
    weight: ArrayLike = 1.0,
    reduction: GraphClassificationReduction = "mean",
    label: str | None = None,
) -> IntegralFunctional:
    """Build classification risk on ragged graph trajectories.

    Hard targets always use nearest stored frames. Soft targets use nearest lookup
    by default and permit linear interpolation only when explicitly requested.
    Time/entity axes and integration measures remain owned by graph trajectory
    geometry.
    """
    if not isinstance(component, DomainComponent):
        raise TypeError("component must be a DomainComponent.")
    if not isinstance(sampling, PointSampling):
        raise TypeError("sampling must be a PointSampling.")
    if not isinstance(component.domain, GraphTrajectoryDatasetDomain):
        raise TypeError(
            "GraphTrajectoryClassificationTerm requires a GraphTrajectoryDatasetDomain."
        )
    kind, class_count = _classification_configuration(target_schema, objective)
    domain = component.domain
    component_kind = _component_kind(component, domain.graph_label)
    target = _graph_trajectory_classification_signal(
        domain,
        values,
        component_kind=component_kind,
        interpolation=interpolation,
        target_encoding=objective.target_encoding,
        require_boolean=False,
    )
    mask = (
        None
        if target_mask is None
        else _graph_trajectory_classification_signal(
            domain,
            target_mask,
            component_kind=component_kind,
            interpolation=interpolation,
            target_encoding="hard",
            require_boolean=True,
        )
    )
    _validate_aligned_target_shapes(
        target,
        mask,
        kind=kind,
        class_count=class_count,
        target_encoding=objective.target_encoding,
    )
    return IntegralFunctional(
        source=per_step(_integration_target(component, reduction), sampling),
        integrand=_GraphClassificationIntegrand(
            field=str(field),
            domain=domain,
            target=target,
            target_mask=mask,
            target_schema=target_schema,
            objective=objective,
            class_count=class_count,
        ),
        objective_vars=(str(field),),
        weight=_checked_weight(weight),
        label=label,
    )


__all__ = [
    "GraphClassificationReduction",
    "GraphClassificationTarget",
    "GraphClassificationTargetEncoding",
    "GraphClassificationTerm",
    "GraphTrajectoryClassificationSignal",
    "GraphTrajectoryClassificationTerm",
]

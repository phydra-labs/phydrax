#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.discretization import AbstractAxisSpec, TensorGridPlan
from phydrax.domain import (
    DATASET_INDEX_KEY,
    DatasetDomain,
    DomainComponent,
    DomainFunction,
    GridBatch,
    GridSampling,
    PointSampling,
    SampleLayout,
)

from .._classification import (
    classification_probabilities,
    pointwise_classification_loss,
)
from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..integration._lowering import _coord_weights, component_factor_fields
from ..ml._classification import ClassificationObjective
from ..ml._overlap import (
    OverlapScoreConfig,
    reduce_overlap_score,
)
from ..ml._schema import TargetSchema
from ._data_metrics import (
    configured_case_indices,
    sample_case_indices,
    validate_case_weights,
    validate_supervised_targets,
)


SiteReduction = Literal["mean", "integral"]
SupportMeasure = Literal["statistical", "physical"]


class DenseSiteClassificationBatch(StrictModule):
    """One frozen case-by-site realization with aligned dense targets."""

    points: GridBatch
    target: Array
    target_mask: Array | None
    indices: Array
    sample_weight: Array | None
    case_axis: str = eqx.field(static=True)
    site_axes: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        points: GridBatch,
        target: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        target_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        case_axis: str,
        site_axes: tuple[str, ...],
    ):
        if not isinstance(points, GridBatch):
            raise TypeError("points must be a GridBatch.")
        index_array = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        target_array = jnp.asarray(target)
        if target_array.ndim == 0 or int(target_array.shape[0]) != int(
            index_array.shape[0]
        ):
            raise ValueError("Dense targets must retain the sampled leading case axis.")
        if target_mask is None:
            mask_array = None
        else:
            mask_array = jnp.asarray(target_mask)
            if mask_array.dtype != jnp.bool_:
                raise TypeError("target_mask must be Boolean.")
            if mask_array.ndim == 0 or int(mask_array.shape[0]) != int(
                index_array.shape[0]
            ):
                raise ValueError("target_mask must retain the sampled leading case axis.")
        if sample_weight is None:
            weight_array = None
        else:
            weight_array = jnp.asarray(sample_weight, dtype=float).reshape((-1,))
            if weight_array.shape != index_array.shape:
                raise ValueError("sample_weight must have one value per sampled case.")
        self.points = points
        self.target = target_array
        self.target_mask = mask_array
        self.indices = index_array
        self.sample_weight = weight_array
        self.case_axis = str(case_axis)
        self.site_axes = tuple(str(axis) for axis in site_axes)


def _dataset_factor(component: DomainComponent, /) -> DatasetDomain:
    factors = tuple(
        factor
        for factor in component.domain.joint_factors
        if isinstance(factor, DatasetDomain)
    )
    if len(factors) != 1:
        raise TypeError("Dense classification requires exactly one DatasetDomain factor.")
    return factors[0]


def _normalize_grid_sampling(
    component: DomainComponent,
    dataset: DatasetDomain,
    sampling: GridSampling,
    /,
) -> tuple[GridSampling, int, str, tuple[str, ...]]:
    if not isinstance(sampling, GridSampling):
        raise TypeError("Dense classification requires a GridSampling plan.")
    dense = sampling.dense
    if dense is None or not isinstance(dense.count, int) or dense.count <= 0:
        raise ValueError("GridSampling.dense must request a positive integer case count.")
    dense_layout = dense.layout or SampleLayout(((dataset.label,),))
    canonical = dense_layout.canonicalize((dataset.label,))
    case_axis = canonical.axis_for(dataset.label)
    if case_axis is None:
        raise ValueError("The dataset label must own a named dense case axis.")
    non_dataset_labels = tuple(
        label for label in component.domain.labels if label != dataset.label
    )
    if not non_dataset_labels:
        raise TypeError("Dense classification requires at least one site factor.")
    missing = tuple(label for label in non_dataset_labels if label not in sampling.axes)
    unknown = tuple(label for label in sampling.axes if label not in non_dataset_labels)
    if missing or unknown:
        raise ValueError(
            "GridSampling.axes must contain every non-dataset site label exactly; "
            f"missing={missing!r}, unknown={unknown!r}."
        )
    for label, request in sampling.axes.items():
        fixed = isinstance(request, (AbstractAxisSpec, TensorGridPlan)) or (
            isinstance(request, tuple)
            and bool(request)
            and all(isinstance(axis, AbstractAxisSpec) for axis in request)
        )
        if not fixed:
            raise ValueError(
                "Raw dense target arrays require fixed explicit axis specifications; "
                f"site label {label!r} uses a stochastic/count-based request."
            )
    site_template = GridSampling(
        sampling.axes,
        dense=PointSampling(0, layout=canonical, design=dense.design),
        design=sampling.design,
    )
    return site_template, int(dense.count), case_axis, non_dataset_labels


def _grid_with_cases(
    template: GridBatch,
    dataset: DatasetDomain,
    indices: Array,
    case_axis: str,
    /,
) -> GridBatch:
    rows = dataset.input_rows(indices)

    def to_field(value: ArrayLike) -> cx.Field:
        array = jnp.asarray(value)
        return cx.Field(array, dims=(case_axis,) + (None,) * (array.ndim - 1))

    points = dict(template.points)
    points[dataset.label] = jax.tree_util.tree_map(to_field, rows)
    points[DATASET_INDEX_KEY] = cx.Field(indices, dims=(case_axis,))
    return GridBatch(
        frozendict(points),
        dense_structure=template.dense_structure,
        coord_axes_by_label=template.coord_axes_by_label,
        coord_mask_by_label=template.coord_mask_by_label,
        coord_geometry_weight_by_label=template.coord_geometry_weight_by_label,
        coord_geometry_order_by_label=template.coord_geometry_order_by_label,
        axis_discretization_by_axis=template.axis_discretization_by_axis,
    )


def _class_count(schema: TargetSchema, /) -> int | None:
    if schema.kind in ("multiclass", "ordinal"):
        count = schema.num_classes
        if count < 2:
            raise ValueError(f"{schema.kind} schemas require explicit class labels.")
        return count
    if schema.kind == "multilabel":
        count = schema.num_labels
        if count < 1:
            raise ValueError("Multilabel schemas require named label coordinates.")
        return count
    return None


def _objective_alpha(objective: ClassificationObjective, /) -> ArrayLike | float | None:
    if objective.alpha is None or isinstance(objective.alpha, float):
        return objective.alpha
    return jnp.asarray(objective.alpha, dtype=float)


def _validate_focal_alpha(
    objective: ClassificationObjective,
    schema: TargetSchema,
    /,
) -> None:
    alpha = objective.alpha
    if objective.kind != "focal" or alpha is None:
        return
    if schema.kind in ("binary", "multilabel"):
        if not isinstance(alpha, float) or not 0.0 < alpha < 1.0:
            raise ValueError(f"{schema.kind} focal alpha must be scalar in (0, 1).")
    elif schema.kind == "multiclass":
        if (
            not isinstance(alpha, tuple)
            or len(alpha) != schema.num_classes
            or any(value <= 0.0 for value in alpha)
        ):
            raise ValueError(
                "Multiclass focal alpha must contain one positive weight per class."
            )
    else:
        raise ValueError("Ordinal focal objectives are unsupported.")


def _output_contract(
    value: cx.Field,
    schema: TargetSchema,
    objective: ClassificationObjective | None,
    /,
) -> tuple[Array, tuple[str | None, ...]]:
    logits = jnp.asarray(value.data)
    dims = tuple(value.dims)
    event_axis = schema.kind in ("multiclass", "multilabel")
    if event_axis:
        if logits.ndim == 0 or dims[-1] is not None:
            raise ValueError(
                f"{schema.kind} logits require one unnamed terminal statistical axis."
            )
        expected = _class_count(schema)
        if int(logits.shape[-1]) != expected:
            raise ValueError(
                f"{schema.kind} logits must end in {expected} coordinates; "
                f"got {logits.shape}."
            )
        observation_dims = dims[:-1]
    elif logits.ndim > 0 and dims[-1] is None:
        if int(logits.shape[-1]) != 1:
            raise ValueError(
                f"{schema.kind} scalar logits can only have a singleton statistical axis."
            )
        logits = logits[..., 0]
        observation_dims = dims[:-1]
    else:
        observation_dims = dims
    if schema.kind == "ordinal":
        if objective is None or objective.thresholds is None:
            raise ValueError("Ordinal classification requires objective thresholds.")
        if len(objective.thresholds) + 1 != schema.num_classes:
            raise ValueError(
                "Ordinal thresholds must contain exactly num_classes - 1 cutpoints."
            )
    return logits, observation_dims


def _target_observation_mask(
    target: Array,
    target_mask: Array | None,
    schema: TargetSchema,
    objective: ClassificationObjective | None,
    observation_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array]:
    soft_multiclass = schema.kind == "multiclass" and (
        (objective is not None and objective.kind == "soft_cross_entropy")
        or target.shape != observation_shape
    )
    if schema.kind == "multilabel" or soft_multiclass:
        if target.shape[:-1] != observation_shape:
            raise ValueError(
                f"{schema.kind} targets must match logits including the terminal "
                f"statistical axis; got target={target.shape}."
            )
    elif target.shape != observation_shape:
        raise ValueError(
            f"{schema.kind} targets must match the observation shape "
            f"{observation_shape}; got {target.shape}."
        )

    if target_mask is None:
        full_mask = jnp.ones(target.shape, dtype=bool)
    else:
        if target_mask.shape == target.shape:
            full_mask = target_mask
        elif target_mask.shape == observation_shape and target.shape != observation_shape:
            full_mask = jnp.broadcast_to(target_mask[..., None], target.shape)
        else:
            raise ValueError(
                "target_mask must match targets or their observation prefix; "
                f"got mask={target_mask.shape}, target={target.shape}."
            )
    if target.shape == observation_shape:
        observation_mask = full_mask
    else:
        observation_mask = jnp.any(full_mask, axis=-1)
    return full_mask, observation_mask


def _masked_logits_target(
    logits: Array,
    target: Array,
    full_mask: Array,
    observation_mask: Array,
    schema: TargetSchema,
    /,
) -> tuple[Array, Array]:
    if schema.kind in ("multiclass", "multilabel"):
        logit_mask = (
            full_mask if schema.kind == "multilabel" else observation_mask[..., None]
        )
        safe_logits = jnp.where(logit_mask, logits, 0.0)
    else:
        safe_logits = jnp.where(observation_mask, logits, 0.0)
    safe_target = jnp.where(full_mask, target, 0)
    return safe_logits, safe_target


class _AbstractDenseClassificationTerm(AbstractSamplingTerm):
    """Shared dense case/site sampling with geometry-authoritative support."""

    __strict_abstract__ = True

    fields: tuple[str, ...]
    field: str
    component: DomainComponent
    sampling: GridSampling
    _site_batch: GridBatch
    _case_count: int = eqx.field(static=True)
    _case_axis: str = eqx.field(static=True)
    _site_axes: tuple[str, ...] = eqx.field(static=True)
    values: Array
    target_mask: Array | None
    sample_weight: Array | None
    indices: Array | None
    target_schema: TargetSchema
    class_count: int | None = eqx.field(static=True)
    observation_operator: Callable[[DomainFunction], DomainFunction] | None
    case_reduction: Literal["mean", "sum"] = eqx.field(static=True)
    weight: Array
    label: str | None

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: GridSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        target_mask: ArrayLike | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        case_reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(component, DomainComponent):
            raise TypeError("component must be a DomainComponent.")
        if not isinstance(target_schema, TargetSchema) or target_schema.kind not in (
            "binary",
            "multiclass",
            "multilabel",
            "ordinal",
        ):
            raise TypeError("target_schema must declare a classification target kind.")
        dataset = _dataset_factor(component)
        template, case_count, case_axis, site_labels = _normalize_grid_sampling(
            component, dataset, sampling
        )
        site_batch = component.sample(template, key=DOC_KEY0)
        if not isinstance(site_batch, GridBatch):
            raise TypeError("Dense site sampling must materialize a GridBatch.")
        site_axes = tuple(
            axis
            for label in site_labels
            for axis in site_batch.coord_axes_by_label[label]
        )
        site_shape = tuple(
            int(site_batch.coord_mask_by_label[label].named_shape[axis])
            for label in site_labels
            for axis in site_batch.coord_axes_by_label[label]
        )
        values = validate_supervised_targets(
            targets,
            leading_size=dataset.size,
            name="dense classification target",
        )
        expected_observation_shape = (dataset.size,) + site_shape
        if target_schema.kind in ("binary", "ordinal"):
            valid_target_shapes = (expected_observation_shape,)
        elif target_schema.kind == "multilabel":
            valid_target_shapes = (
                expected_observation_shape + (target_schema.num_labels,),
            )
        else:
            valid_target_shapes = (
                expected_observation_shape,
                expected_observation_shape + (target_schema.num_classes,),
            )
        if values.shape not in valid_target_shapes:
            raise ValueError(
                "Dense targets must align exactly with the fixed site grid; "
                f"expected one of {valid_target_shapes!r}, got {values.shape}."
            )
        if target_mask is None:
            mask = None
        else:
            mask = validate_supervised_targets(
                target_mask,
                leading_size=dataset.size,
                name="dense classification target_mask",
            )
            if mask.dtype != jnp.bool_:
                raise TypeError("target_mask must be Boolean.")
            valid_mask_shapes = (
                (values.shape, expected_observation_shape)
                if target_schema.kind == "multilabel"
                else (expected_observation_shape,)
            )
            if mask.shape not in valid_mask_shapes:
                raise ValueError(
                    "target_mask must match the dense observation prefix"
                    + (
                        " or multilabel targets"
                        if target_schema.kind == "multilabel"
                        else ""
                    )
                    + f"; expected {valid_mask_shapes!r}, got {mask.shape}."
                )
        configured = configured_case_indices(
            indices,
            sample_mask,
            size=dataset.size,
        )
        statistical_weight = validate_case_weights(
            sample_weight,
            size=dataset.size,
            indices=configured,
        )
        if case_reduction not in ("mean", "sum"):
            raise ValueError("case_reduction must be 'mean' or 'sum'.")
        term_weight = jnp.asarray(weight, dtype=float)
        if (
            term_weight.shape != ()
            or not bool(jnp.isfinite(term_weight))
            or bool(term_weight < 0.0)
        ):
            raise ValueError("weight must be a finite nonnegative scalar.")
        self.fields = (str(field),)
        self.field = str(field)
        self.component = component
        self.sampling = sampling
        self._site_batch = site_batch
        self._case_count = case_count
        self._case_axis = case_axis
        self._site_axes = site_axes
        self.values = values
        self.target_mask = mask
        self.sample_weight = statistical_weight
        self.indices = configured
        self.target_schema = target_schema
        self.class_count = _class_count(target_schema)
        self.observation_operator = observation_operator
        self.case_reduction = case_reduction
        self.weight = term_weight
        self.label = None if label is None else str(label)

    @property
    def dataset(self) -> DatasetDomain:
        return _dataset_factor(self.component)

    def _batch_from_indices(
        self,
        indices: Array,
        /,
    ) -> DenseSiteClassificationBatch:
        points = _grid_with_cases(
            self._site_batch, self.dataset, indices, self._case_axis
        )
        return DenseSiteClassificationBatch(
            points,
            self.values[indices],
            indices,
            target_mask=(None if self.target_mask is None else self.target_mask[indices]),
            sample_weight=(
                None if self.sample_weight is None else self.sample_weight[indices]
            ),
            case_axis=self._case_axis,
            site_axes=self._site_axes,
        )

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> DenseSiteClassificationBatch:
        indices = sample_case_indices(
            size=self.dataset.size,
            num_samples=self._case_count,
            key=key,
            indices=self.indices,
        )
        return self._batch_from_indices(indices)

    def observed_batch(
        self, *, key: Key[Array, ""] = DOC_KEY0
    ) -> DenseSiteClassificationBatch:
        """Materialize every configured case once on the fixed site grid."""
        del key
        indices = (
            jnp.arange(self.dataset.size, dtype=jnp.int32)
            if self.indices is None
            else self.indices
        )
        return self._batch_from_indices(indices)

    def _logits(
        self,
        functions: Mapping[str, DomainFunction],
        batch: DenseSiteClassificationBatch,
        /,
        *,
        key: Key[Array, ""],
        **kwargs: Any,
    ) -> cx.Field:
        function = functions[self.field]
        if self.observation_operator is not None:
            function = self.observation_operator(function)
            if not isinstance(function, DomainFunction):
                raise TypeError("observation_operator must return a DomainFunction.")
        value = function(batch.points, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("Dense classification logits must be a coordax.Field.")
        return value

    def _support_weight(
        self,
        batch: DenseSiteClassificationBatch,
        reference: cx.Field,
        /,
        *,
        key: Key[Array, ""],
        physical: bool,
        target_observed: Array,
        **kwargs: Any,
    ) -> Array:
        geometry_mask, modifier = component_factor_fields(
            self.component,
            batch.points,
            key=key,
            kwargs=kwargs,
        )
        measure = cx.Field(jnp.asarray(1.0), dims=())
        if physical:
            for (
                label,
                geometry_weight,
            ) in batch.points.coord_geometry_weight_by_label.items():
                coordinate_axes = batch.points.coord_axes_by_label.get(label, ())
                if coordinate_axes and all(
                    axis in batch.site_axes for axis in coordinate_axes
                ):
                    measure = measure * geometry_weight
            weights_by_axis = _coord_weights(
                self.component, batch.points, batch.site_axes
            )
            for axis in batch.site_axes:
                measure = measure * weights_by_axis[axis]
        combined = geometry_mask * modifier * measure
        combined_data = jnp.asarray(combined.broadcast_like(reference).data, dtype=float)
        observed = jnp.broadcast_to(target_observed, reference.shape)
        active = observed & (combined_data != 0.0)
        return jnp.where(active, combined_data, 0.0)

    def _case_reduce(self, per_case: Array, batch: DenseSiteClassificationBatch) -> Array:
        values = jnp.asarray(per_case)
        active = jnp.isfinite(values)
        safe = jnp.where(active, values, 0.0)
        weights = (
            active.astype(values.dtype)
            if batch.sample_weight is None
            else jnp.where(active, batch.sample_weight, 0.0)
        )
        if self.case_reduction == "sum":
            return jnp.sum(weights * safe)
        mass = jnp.sum(weights)
        return jnp.where(mass > 0.0, jnp.sum(weights * safe) / mass, 0.0)


class DenseSiteClassificationTerm(_AbstractDenseClassificationTerm):
    """Pointwise dense classification with independent case and site reductions."""

    objective: ClassificationObjective
    site_reduction: SiteReduction = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: GridSampling,
        objective: ClassificationObjective | None = None,
        site_reduction: SiteReduction = "mean",
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        target_mask: ArrayLike | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        case_reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        objective_ = ClassificationObjective.nll() if objective is None else objective
        if not isinstance(objective_, ClassificationObjective):
            raise TypeError("objective must be a ClassificationObjective.")
        if site_reduction not in ("mean", "integral"):
            raise ValueError("site_reduction must be 'mean' or 'integral'.")
        super().__init__(
            field,
            component,
            targets,
            target_schema,
            sampling=sampling,
            observation_operator=observation_operator,
            target_mask=target_mask,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            case_reduction=case_reduction,
            indices=indices,
            weight=weight,
            label=label,
        )
        if target_schema.kind == "ordinal" and objective_.kind != "nll":
            raise ValueError("Ordinal dense classification currently requires NLL.")
        _validate_focal_alpha(objective_, target_schema)
        self.objective = objective_
        self.site_reduction = site_reduction

    def per_case_loss(
        self,
        functions: Mapping[str, DomainFunction],
        batch: DenseSiteClassificationBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        value = self._logits(functions, batch, key=key, **kwargs)
        logits, observation_dims = _output_contract(
            value, self.target_schema, self.objective
        )
        expected_dims = (batch.case_axis,) + batch.site_axes
        if observation_dims != expected_dims:
            raise ValueError(
                "Dense logits must preserve the named case/site axes in canonical "
                f"order {expected_dims!r}; got {observation_dims!r}."
            )
        target = jnp.asarray(batch.target)
        full_mask, observation_mask = _target_observation_mask(
            target,
            batch.target_mask,
            self.target_schema,
            self.objective,
            logits.shape[:-1]
            if self.target_schema.kind in ("multiclass", "multilabel")
            else logits.shape,
        )
        reference = cx.Field(
            jnp.zeros(observation_mask.shape, dtype=logits.dtype),
            dims=observation_dims,
        )
        support_weight = self._support_weight(
            batch,
            reference,
            key=key,
            physical=self.site_reduction == "integral",
            target_observed=observation_mask,
            **kwargs,
        )
        effective_observation = observation_mask & (support_weight != 0.0)
        effective_full_mask = (
            full_mask & effective_observation
            if full_mask.shape == effective_observation.shape
            else full_mask & effective_observation[..., None]
        )
        safe_logits, safe_target = _masked_logits_target(
            logits,
            target,
            effective_full_mask,
            effective_observation,
            self.target_schema,
        )
        pointwise = pointwise_classification_loss(
            safe_logits,
            safe_target,
            kind=self.target_schema.kind,
            objective=self.objective.kind,
            class_count=self.class_count,
            target_mask=(
                effective_full_mask if self.target_schema.kind == "multilabel" else None
            ),
            gamma=self.objective.gamma,
            alpha=_objective_alpha(self.objective),
            thresholds=self.objective.thresholds,
        )
        safe_pointwise = jnp.where(support_weight != 0.0, pointwise, 0.0)
        weighted_sum = jnp.sum(
            support_weight * safe_pointwise,
            axis=tuple(range(1, pointwise.ndim)),
        )
        mass = jnp.sum(support_weight, axis=tuple(range(1, pointwise.ndim)))
        reduced = (
            weighted_sum
            if self.site_reduction == "integral"
            else weighted_sum / jnp.where(mass > 0.0, mass, 1.0)
        )
        return jnp.where(mass > 0.0, reduced, jnp.nan)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: DenseSiteClassificationBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        batch_ = self.sample(key=key) if batch is None else batch

        def zero_loss() -> Array:
            return jnp.zeros((), dtype=jnp.result_type(self.weight, float))

        def active_loss() -> Array:
            reduced = self._case_reduce(
                self.per_case_loss(functions, batch_, key=key, **kwargs), batch_
            )
            return self.weight * jnp.asarray(reduced, dtype=float).reshape(())

        return jax.lax.cond(self.weight == 0.0, zero_loss, active_loss)


def _categorical_statistics(
    probability: Array,
    target: Array,
    support_weight: Array,
    /,
) -> tuple[Array, Array, Array]:
    cases = int(probability.shape[0])
    classes = int(probability.shape[-1])
    flattened_probability = probability.reshape((cases, -1, classes))
    flattened_target = target.reshape((cases, -1))
    flattened_weight = support_weight.reshape((cases, -1))
    target_value = flattened_target.astype(jnp.result_type(flattened_target, 0.0))
    valid = (
        jnp.isfinite(target_value)
        & (target_value >= 0.0)
        & (target_value < classes)
        & (target_value == jnp.floor(target_value))
    )
    active = flattened_weight != 0.0
    safe_target = jnp.where(valid, target_value, 0.0).astype(jnp.int32)
    safe_probability = jnp.where(active[..., None], flattened_probability, 0.0)
    safe_weight = jnp.where(active, flattened_weight, 0.0)
    selected = jnp.take_along_axis(safe_probability, safe_target[..., None], axis=-1)[
        ..., 0
    ]
    case_index = jnp.broadcast_to(
        jnp.arange(cases, dtype=jnp.int32)[:, None], safe_target.shape
    )
    truth = (
        jnp.zeros((cases, classes), dtype=probability.dtype)
        .at[case_index, safe_target]
        .add(safe_weight)
    )
    intersection = (
        jnp.zeros((cases, classes), dtype=probability.dtype)
        .at[case_index, safe_target]
        .add(safe_weight * selected)
    )
    prediction = jnp.sum(safe_weight[..., None] * safe_probability, axis=1)
    invalid_case = jnp.any(active & ~valid, axis=1)
    invalid = jnp.full_like(intersection, jnp.nan)
    return (
        jnp.where(invalid_case[:, None], invalid, intersection),
        jnp.where(invalid_case[:, None], invalid, prediction),
        jnp.where(invalid_case[:, None], invalid, truth),
    )


def _soft_statistics(
    probability: Array,
    target: Array,
    support_weight: Array,
    /,
) -> tuple[Array, Array, Array]:
    if support_weight.shape == probability.shape[:-1]:
        weight = support_weight[..., None]
    elif support_weight.shape == probability.shape:
        weight = support_weight
    else:
        raise ValueError("Overlap support weights do not align with class probabilities.")
    active = weight != 0.0
    safe_probability = jnp.where(active, probability, 0.0)
    safe_target = jnp.where(active, target, 0.0)
    safe_weight = jnp.where(active, weight, 0.0)
    axes = tuple(range(1, probability.ndim - 1))
    return (
        jnp.sum(safe_weight * safe_probability * safe_target, axis=axes),
        jnp.sum(safe_weight * safe_probability, axis=axes),
        jnp.sum(safe_weight * safe_target, axis=axes),
    )


class DenseOverlapClassificationTerm(_AbstractDenseClassificationTerm):
    """Dense Dice/Jaccard/Tversky loss formed after support aggregation."""

    objective: ClassificationObjective
    score: OverlapScoreConfig
    support_measure: SupportMeasure = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        score: OverlapScoreConfig,
        /,
        *,
        sampling: GridSampling,
        objective: ClassificationObjective | None = None,
        support_measure: SupportMeasure = "physical",
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        target_mask: ArrayLike | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        case_reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(score, OverlapScoreConfig):
            raise TypeError("score must be an OverlapScoreConfig.")
        if support_measure not in ("statistical", "physical"):
            raise ValueError("support_measure must be 'statistical' or 'physical'.")
        objective_ = ClassificationObjective.nll() if objective is None else objective
        if not isinstance(objective_, ClassificationObjective):
            raise TypeError("objective must be a ClassificationObjective.")
        if objective_.kind != "nll":
            raise ValueError("Overlap probabilities require an NLL objective.")
        super().__init__(
            field,
            component,
            targets,
            target_schema,
            sampling=sampling,
            observation_operator=observation_operator,
            target_mask=target_mask,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            case_reduction=case_reduction,
            indices=indices,
            weight=weight,
            label=label,
        )
        self.score = score
        self.objective = objective_
        self.support_measure = support_measure

    def per_case_score(
        self,
        functions: Mapping[str, DomainFunction],
        batch: DenseSiteClassificationBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        value = self._logits(functions, batch, key=key, **kwargs)
        logits, observation_dims = _output_contract(
            value, self.target_schema, self.objective
        )
        expected_dims = (batch.case_axis,) + batch.site_axes
        if observation_dims != expected_dims:
            raise ValueError(
                "Dense overlap logits must preserve the named case/site axes in "
                f"canonical order {expected_dims!r}; got {observation_dims!r}."
            )
        target = jnp.asarray(batch.target)
        observation_shape = (
            logits.shape
            if self.target_schema.kind == "binary"
            else logits.shape[:-1]
            if self.target_schema.kind in ("multiclass", "multilabel")
            else logits.shape
        )
        full_mask, observation_mask = _target_observation_mask(
            target,
            batch.target_mask,
            self.target_schema,
            self.objective,
            observation_shape,
        )
        reference = cx.Field(
            jnp.zeros(observation_shape, dtype=logits.dtype),
            dims=observation_dims,
        )
        support_weight = self._support_weight(
            batch,
            reference,
            key=key,
            physical=self.support_measure == "physical",
            target_observed=observation_mask,
            **kwargs,
        )
        effective_observation = observation_mask & (support_weight != 0.0)
        logit_mask = (
            full_mask & effective_observation[..., None]
            if self.target_schema.kind == "multilabel"
            else effective_observation[..., None]
            if self.target_schema.kind == "multiclass"
            else effective_observation
        )
        safe_logits = jnp.where(logit_mask, logits, 0.0)
        probability = classification_probabilities(
            safe_logits,
            kind=self.target_schema.kind,
            class_count=self.class_count,
            thresholds=self.objective.thresholds,
        )
        target_mask = (
            full_mask & effective_observation
            if full_mask.shape == effective_observation.shape
            else full_mask & effective_observation[..., None]
        )
        safe_target = jnp.where(target_mask, target, 0)

        if self.target_schema.kind == "binary":
            probability_ = probability[..., None]
            target_ = safe_target[..., None]
            statistics = _soft_statistics(probability_, target_, support_weight)
        elif (
            self.target_schema.kind in ("multiclass", "ordinal")
            and target.shape == observation_shape
        ):
            statistics = _categorical_statistics(probability, safe_target, support_weight)
        else:
            if target.shape != probability.shape:
                raise ValueError(
                    "Soft multiclass and multilabel overlap targets must match "
                    f"probabilities; got target={target.shape}, probability={probability.shape}."
                )
            support_weight_ = (
                support_weight[..., None] * full_mask
                if full_mask.shape == probability.shape
                else support_weight
            )
            statistics = _soft_statistics(probability, safe_target, support_weight_)
        return reduce_overlap_score(*statistics, self.score)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: DenseSiteClassificationBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_
        batch_ = self.sample(key=key) if batch is None else batch

        def zero_loss() -> Array:
            return jnp.zeros((), dtype=jnp.result_type(self.weight, float))

        def active_loss() -> Array:
            per_case = 1.0 - self.per_case_score(functions, batch_, key=key, **kwargs)
            reduced = self._case_reduce(per_case, batch_)
            return self.weight * jnp.asarray(reduced, dtype=float).reshape(())

        return jax.lax.cond(self.weight == 0.0, zero_loss, active_loss)


__all__ = [
    "DenseOverlapClassificationTerm",
    "DenseSiteClassificationBatch",
    "DenseSiteClassificationTerm",
    "SiteReduction",
    "SupportMeasure",
]

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    DomainComponent,
    DomainFunction,
    IrregularTrajectoryDatasetDomain,
    PointBatch,
    PointSampling,
    SampleLayout,
    TrajectoryDatasetDomain,
)

from .._classification import pointwise_classification_loss
from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..ml._classification import ClassificationObjective
from ..ml._schema import TargetSchema
from ._data_metrics import (
    case_sample_count,
    normalize_case_sampling,
    sample_case_indices as _sample_case_indices,
    validate_case_indices,
)
from ._ragged_time_series import (
    _flat_observation_indices,
    _gather_linear,
    _gather_linear_grid,
    _gather_linear_irregular,
    _gather_linear_irregular_grid,
    _gather_nearest,
    _gather_nearest_grid,
    _grid_points_from_case_time,
    _normalize_sampling,
    RaggedTimeSeriesInterpolation,
    RaggedTimeSeriesSampling,
)
from ._trajectory_data import TrajectoryCaseTime


TrajectoryClassificationMeasure = Literal["statistical", "physical"]
TrajectoryDomain = TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain


class TrajectoryCaseClassificationBatch(StrictModule):
    """A case-level trajectory classification mini-batch."""

    points: PointBatch
    target: Array
    target_mask: Array | None
    sample_weight: Array
    geometry_weight: Array
    case_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointBatch,
        target: ArrayLike,
        target_mask: ArrayLike | None,
        sample_weight: ArrayLike,
        geometry_weight: ArrayLike,
        case_indices: ArrayLike,
        times: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target)
        self.target_mask = (
            None if target_mask is None else jnp.asarray(target_mask, dtype=bool)
        )
        self.sample_weight = jnp.asarray(sample_weight, dtype=float)
        self.geometry_weight = jnp.asarray(geometry_weight, dtype=float)
        self.case_indices = jnp.asarray(case_indices, dtype=jnp.int32)
        self.times = jnp.asarray(times, dtype=float)


class RaggedTimeSeriesClassificationBatch(StrictModule):
    """A time-local trajectory classification mini-batch."""

    points: PointBatch
    target: Array
    target_mask: Array | None
    sample_weight: Array
    geometry_weight: Array
    case_indices: Array
    time_indices: Array
    times: Array

    def __init__(
        self,
        *,
        points: PointBatch,
        target: ArrayLike,
        target_mask: ArrayLike | None,
        sample_weight: ArrayLike,
        geometry_weight: ArrayLike,
        case_indices: ArrayLike,
        time_indices: ArrayLike,
        times: ArrayLike,
    ):
        self.points = points
        self.target = jnp.asarray(target)
        self.target_mask = (
            None if target_mask is None else jnp.asarray(target_mask, dtype=bool)
        )
        self.sample_weight = jnp.asarray(sample_weight, dtype=float)
        self.geometry_weight = jnp.asarray(geometry_weight, dtype=float)
        self.case_indices = jnp.asarray(case_indices, dtype=jnp.int32)
        self.time_indices = jnp.asarray(time_indices, dtype=jnp.int32)
        self.times = jnp.asarray(times, dtype=float)


def _normalize_objective(
    objective: ClassificationObjective | str, /
) -> ClassificationObjective:
    if isinstance(objective, ClassificationObjective):
        return objective
    objective_name = str(objective)
    if objective_name == "nll":
        return ClassificationObjective.nll()
    if objective_name == "soft_cross_entropy":
        return ClassificationObjective.soft_cross_entropy()
    if objective_name == "focal":
        return ClassificationObjective.focal()
    raise ValueError("objective must be 'nll', 'soft_cross_entropy', or 'focal'.")


def _classification_size(target_schema: TargetSchema, /) -> int:
    if not isinstance(target_schema, TargetSchema):
        raise TypeError("target_schema must be a TargetSchema.")
    kind = target_schema.kind
    if kind == "binary":
        return 2
    if kind == "multilabel":
        count = len(target_schema.names)
        if count <= 0:
            raise ValueError(
                "Multilabel classification requires at least one target name."
            )
        return count
    if kind == "multiclass":
        count = target_schema.num_classes
        if count < 2:
            raise ValueError(
                "Multiclass classification requires at least two class labels."
            )
        return count
    if kind == "ordinal":
        count = target_schema.num_classes
        if count < 3:
            raise ValueError("Ordinal classification requires at least three classes.")
        return count
    raise ValueError(
        "Trajectory classification requires a binary, multiclass, multilabel, "
        "or ordinal TargetSchema."
    )


def _validate_objective_schema(
    objective: ClassificationObjective,
    target_schema: TargetSchema,
    class_count: int,
    /,
) -> None:
    thresholds = objective.thresholds
    if target_schema.kind == "ordinal":
        if objective.kind != "nll":
            raise ValueError(
                "Ordinal classification currently requires the NLL objective."
            )
        if thresholds is None:
            raise ValueError("Ordinal classification requires objective thresholds.")
        threshold_array = jnp.asarray(thresholds, dtype=float)
        if threshold_array.shape != (class_count - 1,):
            raise ValueError(
                f"Ordinal objective thresholds must have shape ({class_count - 1},)."
            )
        if not bool(jnp.all(jnp.isfinite(threshold_array))):
            raise ValueError("Ordinal objective thresholds must be finite.")
        if not bool(jnp.all(jnp.diff(threshold_array) > 0.0)):
            raise ValueError("Ordinal objective thresholds must be strictly increasing.")
    elif thresholds is not None:
        raise ValueError("Objective thresholds are supported only for ordinal targets.")


def _normalize_reduction_measure(
    reduction: str,
    measure: str,
    /,
) -> tuple[Literal["mean", "sum"], TrajectoryClassificationMeasure]:
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be either 'mean' or 'sum'.")
    reduction_: Literal["mean", "sum"] = "mean" if reduction == "mean" else "sum"
    if measure not in ("statistical", "physical"):
        raise ValueError("measure must be either 'statistical' or 'physical'.")
    measure_: TrajectoryClassificationMeasure = (
        "statistical" if measure == "statistical" else "physical"
    )
    if measure_ == "physical" and reduction_ != "sum":
        raise ValueError("Physical trajectory measure requires reduction='sum'.")
    return reduction_, measure_


def _validate_term_weight(weight: ArrayLike, /) -> float:
    result = float(weight)
    if not math.isfinite(result):
        raise ValueError("weight must be a finite scalar.")
    if result < 0.0:
        raise ValueError("weight must be nonnegative.")
    return result


def _validate_case_weight(
    sample_weight: ArrayLike | None,
    /,
    *,
    size: int,
) -> Array:
    if sample_weight is None:
        return jnp.ones((size,), dtype=float)
    result = jnp.asarray(sample_weight, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"sample_weight must have shape ({size},), got {result.shape}.")
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError("sample_weight must be finite.")
    if bool(jnp.any(result <= 0.0)):
        raise ValueError("sample_weight must be strictly positive.")
    return result


def _validate_case_targets(values: ArrayLike, /, *, size: int) -> Array:
    result = jnp.asarray(values)
    if result.ndim == 0 or int(result.shape[0]) != size:
        raise ValueError(f"Case targets must have leading shape ({size}, ...).")
    return result


def _validate_ragged_targets(values: ArrayLike, /, *, domain: TrajectoryDomain) -> Array:
    result = jnp.asarray(values)
    if result.ndim < 2:
        raise ValueError("Ragged targets must have shape (N, T, ...).")
    if int(result.shape[0]) != domain.size:
        raise ValueError(
            f"Ragged targets must have leading size {domain.size}, got {result.shape[0]}."
        )
    if int(result.shape[1]) < domain.max_length:
        raise ValueError(
            f"Ragged target time axis must contain at least {domain.max_length} entries."
        )
    return result


def _canonicalize_scalar_target(
    values: Array,
    target_mask: ArrayLike | None,
    kind: str,
    /,
    *,
    prefix_shape: tuple[int, ...],
) -> tuple[Array, ArrayLike | None]:
    if kind not in ("binary", "multiclass", "ordinal"):
        return values, target_mask
    singleton_shape = prefix_shape + (1,)
    if values.shape != singleton_shape:
        return values, target_mask
    mask = target_mask
    if mask is not None and jnp.asarray(mask).shape == singleton_shape:
        mask = jnp.asarray(mask)[..., 0]
    return values[..., 0], mask


def _validate_target_mask(
    target_mask: ArrayLike | None,
    /,
    *,
    prefix_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    kind: str,
) -> Array | None:
    if target_mask is None:
        return None
    result = jnp.asarray(target_mask)
    if result.dtype != jnp.bool_:
        raise TypeError("target_mask must be Boolean.")
    if result.shape == prefix_shape:
        return result
    if kind == "multilabel":
        if result.shape == prefix_shape:
            result = result[..., None]
        try:
            return jnp.broadcast_to(result, target_shape)
        except ValueError as error:
            raise ValueError(
                f"Multilabel target_mask must broadcast to {target_shape}; "
                f"got {result.shape}."
            ) from error
    if len(target_shape) == len(prefix_shape) + 1:
        try:
            expanded = jnp.broadcast_to(result, target_shape)
        except ValueError as error:
            raise ValueError(
                "Categorical target_mask must have observation-prefix shape "
                f"{prefix_shape}; got {result.shape}."
            ) from error
        if not bool(jnp.all(expanded == expanded[..., :1])):
            raise ValueError(
                "Categorical target_mask cannot mask individual class coordinates."
            )
        return expanded[..., 0]
    raise ValueError(
        f"target_mask must have observation-prefix shape {prefix_shape}; "
        f"got {result.shape}."
    )


def _hard_targets(values: Array, /) -> bool:
    return values.dtype == jnp.bool_ or jnp.issubdtype(values.dtype, jnp.integer)


def _validate_target_shape(
    values: Array,
    target_schema: TargetSchema,
    objective: ClassificationObjective,
    class_count: int,
    /,
    *,
    prefix_shape: tuple[int, ...],
) -> None:
    if objective.target_encoding == "hard" and not _hard_targets(values):
        raise TypeError("Hard classification targets must be integer or Boolean labels.")
    shape = tuple(int(n) for n in values.shape)
    scalar_shapes = (prefix_shape, prefix_shape + (1,))
    kind = target_schema.kind
    if kind == "multilabel":
        expected = prefix_shape + (class_count,)
        if shape != expected:
            raise ValueError(
                f"Multilabel targets must have shape {expected}, got {shape}."
            )
        return
    if kind in ("binary", "ordinal") and shape in scalar_shapes:
        return
    if kind == "multiclass" and _hard_targets(values) and shape in scalar_shapes:
        return
    if objective.kind == "soft_cross_entropy" and kind in ("multiclass", "ordinal"):
        expected = prefix_shape + (class_count,)
        if shape == expected:
            return
    if kind == "multiclass":
        expected_text = f"{prefix_shape} for hard labels"
        if objective.kind == "soft_cross_entropy":
            expected_text += f" or {prefix_shape + (class_count,)} for soft targets"
        raise ValueError(
            f"Multiclass targets must have shape {expected_text}; got {shape}."
        )
    raise ValueError(f"{kind.capitalize()} targets have incompatible shape {shape}.")


def _observation_mask(
    target_mask: Array | None,
    /,
    *,
    prefix_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    kind: str,
) -> Array:
    if target_mask is None:
        return jnp.ones(prefix_shape, dtype=bool)
    mask = jnp.asarray(target_mask, dtype=bool)
    if mask.shape == prefix_shape:
        return mask
    expanded = jnp.broadcast_to(mask, target_shape)
    if len(target_shape) == len(prefix_shape):
        return expanded
    if kind != "multilabel":
        first = expanded[..., :1]
        if not bool(jnp.all(expanded == first)):
            raise ValueError(
                "Categorical target_mask cannot mask individual class coordinates."
            )
    return jnp.any(expanded, axis=-1)


def _validate_linear_targets(
    values: Array,
    target_mask: Array | None,
    target_schema: TargetSchema,
    objective: ClassificationObjective,
    class_count: int,
    domain: TrajectoryDomain,
    /,
    *,
    case_indices: Array | None,
) -> None:
    if _hard_targets(values):
        raise ValueError("Hard classification targets require nearest interpolation.")
    if objective.kind != "soft_cross_entropy":
        raise ValueError(
            "Linear target interpolation requires the soft_cross_entropy objective."
        )
    if jnp.iscomplexobj(values):
        raise TypeError("Soft classification targets must be real-valued.")

    prefix_shape = (domain.size, int(values.shape[1]))
    observation_active = _observation_mask(
        target_mask,
        prefix_shape=prefix_shape,
        target_shape=tuple(int(n) for n in values.shape),
        kind=target_schema.kind,
    )
    valid_time = jnp.arange(int(values.shape[1]))[None, :] < domain.lengths[:, None]
    observation_active = observation_active & valid_time
    if case_indices is not None:
        configured = jnp.zeros((domain.size,), dtype=bool).at[case_indices].set(True)
        observation_active = observation_active & configured[:, None]
    numeric = jnp.asarray(values, dtype=float)
    kind = target_schema.kind

    if kind == "binary":
        scalar = numeric[..., 0] if numeric.shape == prefix_shape + (1,) else numeric
        valid = jnp.isfinite(scalar) & (scalar >= 0.0) & (scalar <= 1.0)
        if bool(jnp.any(observation_active & ~valid)):
            raise ValueError("Active binary soft targets must lie in [0, 1].")
        return

    if kind == "multilabel":
        active = jnp.broadcast_to(observation_active[..., None], numeric.shape)
        if target_mask is not None:
            label_mask = jnp.asarray(target_mask, dtype=bool)
            if label_mask.shape == prefix_shape:
                label_mask = label_mask[..., None]
            active = active & jnp.broadcast_to(label_mask, numeric.shape)
        valid = jnp.isfinite(numeric) & (numeric >= 0.0) & (numeric <= 1.0)
        if bool(jnp.any(active & ~valid)):
            raise ValueError("Active multilabel soft targets must lie in [0, 1].")
        return

    expected = prefix_shape + (class_count,)
    if numeric.shape != expected:
        raise ValueError(
            f"Linear {kind} targets must be simplex arrays with shape {expected}."
        )
    finite_nonnegative = jnp.all(jnp.isfinite(numeric) & (numeric >= 0.0), axis=-1)
    simplex = jnp.abs(jnp.sum(numeric, axis=-1) - 1.0) <= 1e-6
    if bool(jnp.any(observation_active & ~(finite_nonnegative & simplex))):
        raise ValueError(
            "Active soft categorical targets must lie on the probability simplex."
        )


def _case_start_times(domain: TrajectoryDomain, case_indices: Array, /) -> Array:
    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        return domain.start_times[case_indices]
    return jnp.full(case_indices.shape, domain.start, dtype=float)


def _case_at_time(
    domain: TrajectoryDomain,
    case_indices: Array,
    case_time: TrajectoryCaseTime,
    /,
) -> tuple[Array, Array]:
    if case_time == "start":
        return _case_start_times(domain, case_indices), jnp.zeros_like(case_indices)
    if case_time == "end":
        return domain.end_times[case_indices], domain.lengths[case_indices] - 1
    if isinstance(case_time, str):
        raise ValueError("case_time must be 'start', 'end', or a floating time value.")
    value = jnp.asarray(float(case_time), dtype=float).reshape(())
    times = jnp.full(case_indices.shape, value, dtype=float)
    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        time_indices = domain.nearest_time_indices(case_indices, times)
    else:
        time_index = jnp.rint((value - domain.start) / domain.dt).astype(jnp.int32)
        time_indices = jnp.full(case_indices.shape, time_index, dtype=jnp.int32)
    return times, time_indices


def _configured_cases_at_time(
    domain: TrajectoryDomain,
    case_time: TrajectoryCaseTime,
    indices: Array | None,
    /,
) -> tuple[Array, Array]:
    if isinstance(case_time, str):
        raise ValueError("case_time must be 'start', 'end', or a floating time value.")
    value = jnp.asarray(float(case_time), dtype=float).reshape(())
    all_cases = jnp.arange(domain.size, dtype=jnp.int32)
    valid = (_case_start_times(domain, all_cases) <= value) & (value <= domain.end_times)
    allowed = all_cases if indices is None else indices
    active = valid[allowed]
    if not bool(jnp.any(active)):
        raise ValueError("No configured trajectory case is valid at case_time.")
    return allowed, active


def _sample_cases_at_time(
    domain: TrajectoryDomain,
    num_samples: int,
    key: Key[Array, ""],
    case_time: TrajectoryCaseTime,
    /,
    *,
    indices: Array | None,
) -> Array:
    if case_time == "start" or case_time == "end":
        return _sample_case_indices(
            size=domain.size,
            num_samples=num_samples,
            key=key,
            indices=indices,
        )
    allowed, active = _configured_cases_at_time(domain, case_time, indices)
    probabilities = active.astype(float) / jnp.sum(active.astype(float))
    positions = jr.choice(
        key,
        int(allowed.shape[0]),
        shape=(num_samples,),
        p=probabilities,
    )
    return allowed[positions].astype(jnp.int32)


def _case_geometry_weight(
    domain: TrajectoryDomain,
    case_indices: Array,
    measure: TrajectoryClassificationMeasure,
    /,
) -> Array:
    count = int(case_indices.shape[0])
    if measure == "statistical":
        return jnp.ones((count,), dtype=float)
    mass = 1.0
    if domain.measure_mode == "time_integral_sum":
        mass = float(domain.size)
    return jnp.full((count,), mass / float(count), dtype=float)


def _ragged_sample_weight(
    case_weight: Array, case_indices: Array, times: Array, /
) -> Array:
    selected = case_weight[case_indices]
    if times.ndim == 1:
        return selected
    return jnp.broadcast_to(selected[:, None], times.shape)


def _regular_node_widths(
    domain: TrajectoryDatasetDomain,
    case_indices: Array,
    time_indices: Array,
    /,
) -> Array:
    lengths = domain.lengths[case_indices]
    terminal = lengths - 1
    interior_width = jnp.asarray(domain.dt, dtype=float)
    endpoint_width = 0.5 * interior_width
    widths = jnp.where(
        (time_indices == 0) | (time_indices == terminal),
        endpoint_width,
        interior_width,
    )
    return jnp.where(lengths == 1, 0.0, widths)


def _ragged_node_widths(
    domain: TrajectoryDomain,
    case_indices: Array,
    time_indices: Array,
    /,
) -> Array:
    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        return domain.node_widths[case_indices, time_indices]
    return _regular_node_widths(domain, case_indices, time_indices)


def _ragged_geometry_weight(
    domain: TrajectoryDomain,
    batch_times: Array,
    case_indices: Array,
    time_indices: Array,
    selection: RaggedTimeSeriesSampling,
    measure: TrajectoryClassificationMeasure,
    /,
) -> Array:
    if measure == "statistical":
        return jnp.ones(batch_times.shape, dtype=float)

    count = int(batch_times.size)
    if domain.measure_mode == "case_time_probability":
        return jnp.full(batch_times.shape, 1.0 / float(count), dtype=float)

    case_grid = (
        case_indices
        if batch_times.ndim == 1
        else jnp.broadcast_to(case_indices[:, None], batch_times.shape)
    )
    if selection == "case_time_uniform":
        importance = domain.durations[case_grid]
    else:
        widths = _ragged_node_widths(domain, case_grid, time_indices)
        if selection == "observation_uniform" and batch_times.ndim == 1:
            importance = float(domain.total_observations) * widths / float(domain.size)
        else:
            importance = domain.lengths[case_grid].astype(float) * widths

    if domain.measure_mode == "time_integral_sum":
        importance = importance * float(domain.size)
    return importance / float(count)


def _gather_target_mask(
    domain: TrajectoryDomain,
    target_mask: Array | None,
    case_indices: Array,
    time_indices: Array,
    times: Array,
    interpolation: RaggedTimeSeriesInterpolation,
    /,
) -> Array | None:
    if target_mask is None:
        return None
    if interpolation == "nearest":
        if times.ndim == 1:
            return _gather_nearest(target_mask, case_indices, time_indices).astype(bool)
        return _gather_nearest_grid(target_mask, case_indices, time_indices).astype(bool)

    case_grid = (
        case_indices
        if times.ndim == 1
        else jnp.broadcast_to(case_indices[:, None], times.shape)
    )
    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        lower, upper, _ = domain.bracketing_time_indices(
            case_grid.reshape((-1,)), times.reshape((-1,))
        )
    else:
        tau = (times - domain.start) / domain.dt
        lower = jnp.floor(tau).astype(jnp.int32).reshape((-1,))
        lengths = domain.lengths[case_grid].reshape((-1,))
        upper = jnp.minimum(lower + 1, lengths - 1)
    flat_cases = case_grid.reshape((-1,))
    lower_mask = target_mask[flat_cases, lower]
    upper_mask = target_mask[flat_cases, upper]
    result_shape = times.shape + lower_mask.shape[1:]
    return (lower_mask & upper_mask).reshape(result_shape)


def _active_mass(
    score: Array,
    target: Array,
    target_mask: Array | None,
    kind: str,
    /,
) -> tuple[Array, Array]:
    score_array = jnp.asarray(score, dtype=float)
    target_array = jnp.asarray(target)

    if kind == "multilabel" and score_array.shape == target_array.shape:
        if target_mask is None:
            event_mask = jnp.ones(score_array.shape, dtype=bool)
        else:
            event_mask = jnp.asarray(target_mask, dtype=bool)
            if event_mask.shape == score_array.shape[:-1]:
                event_mask = event_mask[..., None]
            event_mask = jnp.broadcast_to(event_mask, score_array.shape)
        gated = jnp.where(event_mask, score_array, 0.0)
        return jnp.sum(gated, axis=-1), jnp.any(event_mask, axis=-1).astype(float)

    prefix_shape = score_array.shape
    if target_mask is None:
        if kind == "multilabel" and target_array.ndim == score_array.ndim + 1:
            mass = jnp.ones(prefix_shape, dtype=float)
        else:
            mass = jnp.ones(prefix_shape, dtype=float)
    else:
        mask = jnp.asarray(target_mask, dtype=bool)
        if mask.shape == prefix_shape:
            mass = mask.astype(float)
        elif kind == "multilabel" and mask.shape == target_array.shape:
            mass = jnp.any(mask, axis=-1).astype(float)
        else:
            expanded = jnp.broadcast_to(mask, target_array.shape)
            mass = jnp.any(expanded, axis=-1).astype(float)
    return jnp.where(mass > 0.0, score_array, 0.0), mass


def _reduce_scores(
    scores: Array,
    target: Array,
    target_mask: Array | None,
    sample_weight: Array,
    geometry_weight: Array,
    kind: str,
    reduction: Literal["mean", "sum"],
    measure: TrajectoryClassificationMeasure,
    /,
) -> Array:
    point_score, active_mass = _active_mass(scores, target, target_mask, kind)
    statistical_weight = jnp.broadcast_to(sample_weight, point_score.shape)
    geometry = jnp.broadcast_to(geometry_weight, point_score.shape)
    weighted_score = point_score * statistical_weight * geometry
    if measure == "physical" or reduction == "sum":
        return jnp.sum(weighted_score).reshape(())
    denominator = jnp.sum(active_mass * statistical_weight)
    return (jnp.sum(weighted_score) / denominator).reshape(())


def _classification_loss(
    prediction: cx.Field,
    batch: TrajectoryCaseClassificationBatch | RaggedTimeSeriesClassificationBatch,
    target_schema: TargetSchema,
    objective: ClassificationObjective,
    class_count: int,
    reduction: Literal["mean", "sum"],
    measure: TrajectoryClassificationMeasure,
    /,
) -> Array:
    logits = jnp.asarray(prediction.data)
    if target_schema.kind in (
        "binary",
        "ordinal",
    ) and logits.shape == batch.target.shape + (1,):
        logits = logits[..., 0]
    loss_mask = batch.target_mask
    if (
        target_schema.kind == "multilabel"
        and loss_mask is not None
        and loss_mask.shape == batch.target.shape[:-1]
    ):
        loss_mask = loss_mask[..., None]
    scores = pointwise_classification_loss(
        logits,
        batch.target,
        kind=target_schema.kind,
        objective=objective.kind,
        class_count=class_count,
        target_mask=loss_mask,
        gamma=objective.gamma,
        alpha=objective.alpha,
        thresholds=objective.thresholds,
    )
    return _reduce_scores(
        scores,
        batch.target,
        loss_mask,
        batch.sample_weight,
        batch.geometry_weight,
        target_schema.kind,
        reduction,
        measure,
    )


class TrajectoryCaseClassificationTerm(AbstractSamplingTerm):
    """Classify complete trajectory cases at an explicit representative time."""

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "sum"]
    measure: TrajectoryClassificationMeasure
    values: Array
    target_schema: TargetSchema
    objective: ClassificationObjective
    class_count: int
    target_mask: Array | None
    sample_weight: Array
    case_time: TrajectoryCaseTime
    case_indices: Array | None
    weight: float = eqx.field(static=True)
    label: str | None

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: PointSampling,
        objective: ClassificationObjective | str = "nll",
        target_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        case_time: TrajectoryCaseTime = "start",
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        measure: TrajectoryClassificationMeasure = "statistical",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        domain = component.domain
        if not isinstance(
            domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError(
                "TrajectoryCaseClassificationTerm requires a trajectory dataset domain."
            )
        objective_ = _normalize_objective(objective)
        class_count = _classification_size(target_schema)
        _validate_objective_schema(objective_, target_schema, class_count)
        values = _validate_case_targets(targets, size=domain.size)
        values, target_mask = _canonicalize_scalar_target(
            values,
            target_mask,
            target_schema.kind,
            prefix_shape=(domain.size,),
        )
        _validate_target_shape(
            values,
            target_schema,
            objective_,
            class_count,
            prefix_shape=(domain.size,),
        )
        mask = _validate_target_mask(
            target_mask,
            prefix_shape=(domain.size,),
            target_shape=tuple(int(n) for n in values.shape),
            kind=target_schema.kind,
        )
        reduction_, measure_ = _normalize_reduction_measure(reduction, measure)
        indices = validate_case_indices(
            case_indices, size=domain.size, name="case_indices"
        )
        if case_time != "start" and case_time != "end":
            _configured_cases_at_time(domain, case_time, indices)

        self.fields = (str(field),)
        self.component = component
        self.sampling = normalize_case_sampling(
            sampling,
            labels=domain.labels,
            owner="TrajectoryCaseClassificationTerm",
        )
        self.over = None
        self.reduction = reduction_
        self.measure = measure_
        self.values = values
        self.target_schema = target_schema
        self.objective = objective_
        self.class_count = class_count
        self.target_mask = mask
        self.sample_weight = _validate_case_weight(sample_weight, size=domain.size)
        self.case_time = case_time
        self.case_indices = indices
        self.weight = _validate_term_weight(weight)
        self.label = None if label is None else str(label)

    @property
    def domain(self) -> TrajectoryDomain:
        domain = self.component.domain
        if not isinstance(
            domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError("Trajectory case classification domain is not a trajectory.")
        return domain

    def sample(
        self, *, key: Key[Array, ""] = DOC_KEY0
    ) -> TrajectoryCaseClassificationBatch:
        """Draw cases and retain their integer or Boolean labels without encoding."""
        domain = self.domain
        case_indices = _sample_cases_at_time(
            domain,
            case_sample_count(self.sampling),
            key,
            self.case_time,
            indices=self.case_indices,
        )
        times, time_indices = _case_at_time(domain, case_indices, self.case_time)
        layout = self.sampling.layout
        assert layout is not None
        points = domain.points_from_case_time(
            case_indices,
            times,
            structure=layout,
            time_indices=time_indices,
        )
        return TrajectoryCaseClassificationBatch(
            points=points,
            target=self.values[case_indices],
            target_mask=(
                None if self.target_mask is None else self.target_mask[case_indices]
            ),
            sample_weight=self.sample_weight[case_indices],
            geometry_weight=_case_geometry_weight(domain, case_indices, self.measure),
            case_indices=case_indices,
            times=times,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: TrajectoryCaseClassificationBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the configured classification objective over sampled cases."""
        del iter_
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = functions[self.fields[0]](batch_.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError(
                "Expected classification prediction to return a coordax.Field."
            )
        return self.weight * _classification_loss(
            prediction,
            batch_,
            self.target_schema,
            self.objective,
            self.class_count,
            self.reduction,
            self.measure,
        )


class RaggedTimeSeriesClassificationTerm(AbstractSamplingTerm):
    """Classify valid time sites in regular or irregular ragged trajectories."""

    fields: tuple[str, ...]
    component: DomainComponent
    sampling: PointSampling
    over: str | tuple[str, ...] | None
    reduction: Literal["mean", "sum"]
    measure: TrajectoryClassificationMeasure
    values: Array
    target_schema: TargetSchema
    objective: ClassificationObjective
    class_count: int
    target_mask: Array | None
    sample_weight: Array
    case_indices: Array | None
    observation_case_indices: Array
    observation_time_indices: Array
    observation_count: int
    selection: RaggedTimeSeriesSampling
    interpolation: RaggedTimeSeriesInterpolation
    weight: float = eqx.field(static=True)
    label: str | None

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: PointSampling,
        objective: ClassificationObjective | str = "nll",
        target_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        selection: RaggedTimeSeriesSampling = "observation_uniform",
        interpolation: RaggedTimeSeriesInterpolation = "nearest",
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        measure: TrajectoryClassificationMeasure = "statistical",
        case_indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        domain = component.domain
        if not isinstance(
            domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError(
                "RaggedTimeSeriesClassificationTerm requires a trajectory dataset domain."
            )
        if selection not in (
            "observation_uniform",
            "case_uniform",
            "case_time_uniform",
        ):
            raise ValueError(
                "selection must be 'observation_uniform', 'case_uniform', "
                "or 'case_time_uniform'."
            )
        if interpolation not in ("nearest", "linear"):
            raise ValueError("interpolation must be either 'nearest' or 'linear'.")

        objective_ = _normalize_objective(objective)
        class_count = _classification_size(target_schema)
        _validate_objective_schema(objective_, target_schema, class_count)
        values = _validate_ragged_targets(targets, domain=domain)
        prefix_shape = (domain.size, int(values.shape[1]))
        values, target_mask = _canonicalize_scalar_target(
            values,
            target_mask,
            target_schema.kind,
            prefix_shape=prefix_shape,
        )
        _validate_target_shape(
            values,
            target_schema,
            objective_,
            class_count,
            prefix_shape=prefix_shape,
        )
        mask = _validate_target_mask(
            target_mask,
            prefix_shape=prefix_shape,
            target_shape=tuple(int(n) for n in values.shape),
            kind=target_schema.kind,
        )
        indices = validate_case_indices(
            case_indices, size=domain.size, name="case_indices"
        )
        if interpolation == "linear":
            _validate_linear_targets(
                values,
                mask,
                target_schema,
                objective_,
                class_count,
                domain,
                case_indices=indices,
            )
        reduction_, measure_ = _normalize_reduction_measure(reduction, measure)
        observation_cases, observation_times = _flat_observation_indices(domain, indices)

        self.fields = (str(field),)
        self.component = component
        self.sampling = _normalize_sampling(sampling, domain)
        self.over = None
        self.reduction = reduction_
        self.measure = measure_
        self.values = values
        self.target_schema = target_schema
        self.objective = objective_
        self.class_count = class_count
        self.target_mask = mask
        self.sample_weight = _validate_case_weight(sample_weight, size=domain.size)
        self.case_indices = indices
        self.observation_case_indices = observation_cases
        self.observation_time_indices = observation_times
        self.observation_count = int(observation_cases.shape[0])
        self.selection = selection
        self.interpolation = interpolation
        self.weight = _validate_term_weight(weight)
        self.label = None if label is None else str(label)

    @property
    def domain(self) -> TrajectoryDomain:
        domain = self.component.domain
        if not isinstance(
            domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
        ):
            raise TypeError("Ragged classification domain is not a trajectory domain.")
        return domain

    @property
    def _layout(self) -> SampleLayout:
        layout = self.sampling.layout
        assert layout is not None
        return layout

    def sample(
        self, *, key: Key[Array, ""] = DOC_KEY0
    ) -> RaggedTimeSeriesClassificationBatch:
        """Draw only valid ragged sites and gather their targets without one-hotting."""
        count = self.sampling.count
        if isinstance(count, tuple):
            return self._sample_case_time_grid(key=key)
        return self._sample_paired(count, key=key)

    def _sample_paired(
        self,
        count: int,
        /,
        *,
        key: Key[Array, ""],
    ) -> RaggedTimeSeriesClassificationBatch:
        domain = self.domain
        key_case, key_time = jr.split(key)
        if self.selection == "observation_uniform":
            observation_index = jr.randint(
                key,
                shape=(count,),
                minval=0,
                maxval=self.observation_count,
                dtype=jnp.int32,
            )
            case_indices = self.observation_case_indices[observation_index]
            time_indices = self.observation_time_indices[observation_index]
            times = domain.observation_times(case_indices, time_indices)
            target = _gather_nearest(self.values, case_indices, time_indices)
        else:
            case_indices = _sample_case_indices(
                size=domain.size,
                num_samples=count,
                key=key_case,
                indices=self.case_indices,
            )
            lengths = domain.lengths[case_indices]
            if self.selection == "case_time_uniform":
                if isinstance(domain, IrregularTrajectoryDatasetDomain):
                    starts = domain.start_times[case_indices]
                    ends = domain.end_times[case_indices]
                    times = starts + jr.uniform(key_time, shape=(count,)) * (
                        ends - starts
                    )
                    if self.interpolation == "linear":
                        target, time_indices = _gather_linear_irregular(
                            domain, self.values, case_indices, times
                        )
                    else:
                        time_indices = domain.nearest_time_indices(case_indices, times)
                        target = _gather_nearest(self.values, case_indices, time_indices)
                else:
                    tau = jr.uniform(key_time, shape=(count,)) * (
                        lengths.astype(float) - 1.0
                    )
                    times = domain.start + domain.dt * tau
                    if self.interpolation == "linear":
                        target, time_indices = _gather_linear(
                            self.values, case_indices, tau, lengths
                        )
                    else:
                        time_indices = jnp.rint(tau).astype(jnp.int32)
                        time_indices = jnp.clip(time_indices, 0, lengths - 1)
                        target = _gather_nearest(self.values, case_indices, time_indices)
            else:
                time_indices = jnp.floor(
                    jr.uniform(key_time, shape=(count,)) * lengths.astype(float)
                ).astype(jnp.int32)
                time_indices = jnp.clip(time_indices, 0, lengths - 1)
                times = domain.observation_times(case_indices, time_indices)
                target = _gather_nearest(self.values, case_indices, time_indices)

        return self._batch(case_indices, time_indices, times, target)

    def _sample_case_time_grid(
        self, *, key: Key[Array, ""]
    ) -> RaggedTimeSeriesClassificationBatch:
        domain = self.domain
        count = self.sampling.count
        if not isinstance(count, tuple):
            raise TypeError("Case-time grid sampling requires a tuple point count.")
        num_cases, num_times = count
        key_case, key_time = jr.split(key)
        case_indices = _sample_case_indices(
            size=domain.size,
            num_samples=num_cases,
            key=key_case,
            indices=self.case_indices,
        )
        lengths = domain.lengths[case_indices]
        case_grid = jnp.broadcast_to(case_indices[:, None], (num_cases, num_times))
        if self.selection == "case_time_uniform":
            if isinstance(domain, IrregularTrajectoryDatasetDomain):
                starts = domain.start_times[case_indices]
                ends = domain.end_times[case_indices]
                times = starts[:, None] + jr.uniform(
                    key_time, shape=(num_cases, num_times)
                ) * (ends[:, None] - starts[:, None])
                if self.interpolation == "linear":
                    target, time_indices = _gather_linear_irregular_grid(
                        domain, self.values, case_indices, times
                    )
                else:
                    time_indices = domain.nearest_time_indices(
                        case_grid.reshape((-1,)), times.reshape((-1,))
                    ).reshape((num_cases, num_times))
                    target = _gather_nearest_grid(self.values, case_indices, time_indices)
            else:
                tau = jr.uniform(key_time, shape=(num_cases, num_times)) * (
                    lengths[:, None].astype(float) - 1.0
                )
                times = domain.start + domain.dt * tau
                if self.interpolation == "linear":
                    target, time_indices = _gather_linear_grid(
                        self.values, case_indices, tau, lengths
                    )
                else:
                    time_indices = jnp.rint(tau).astype(jnp.int32)
                    time_indices = jnp.clip(time_indices, 0, lengths[:, None] - 1)
                    target = _gather_nearest_grid(self.values, case_indices, time_indices)
        else:
            time_indices = jnp.floor(
                jr.uniform(key_time, shape=(num_cases, num_times))
                * lengths[:, None].astype(float)
            ).astype(jnp.int32)
            time_indices = jnp.clip(time_indices, 0, lengths[:, None] - 1)
            times = domain.observation_times(case_grid, time_indices)
            target = _gather_nearest_grid(self.values, case_indices, time_indices)

        return self._batch(case_indices, time_indices, times, target)

    def _batch(
        self,
        case_indices: Array,
        time_indices: Array,
        times: Array,
        target: Array,
        /,
    ) -> RaggedTimeSeriesClassificationBatch:
        domain = self.domain
        if times.ndim == 1:
            points = domain.points_from_case_time(
                case_indices,
                times,
                structure=self._layout,
                time_indices=time_indices,
            )
        else:
            points = _grid_points_from_case_time(
                domain,
                case_indices,
                times,
                time_indices,
                structure=self._layout,
            )
        return RaggedTimeSeriesClassificationBatch(
            points=points,
            target=target,
            target_mask=_gather_target_mask(
                domain,
                self.target_mask,
                case_indices,
                time_indices,
                times,
                self.interpolation,
            ),
            sample_weight=_ragged_sample_weight(self.sample_weight, case_indices, times),
            geometry_weight=_ragged_geometry_weight(
                domain,
                times,
                case_indices,
                time_indices,
                self.selection,
                self.measure,
            ),
            case_indices=case_indices,
            time_indices=time_indices,
            times=times,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: RaggedTimeSeriesClassificationBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the configured objective without flattening case/time geometry."""
        del iter_
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        batch_ = self.sample(key=key) if batch is None else batch
        prediction = functions[self.fields[0]](batch_.points, key=key, **kwargs)
        if not isinstance(prediction, cx.Field):
            raise TypeError(
                "Expected classification prediction to return a coordax.Field."
            )
        return self.weight * _classification_loss(
            prediction,
            batch_,
            self.target_schema,
            self.objective,
            self.class_count,
            self.reduction,
            self.measure,
        )


__all__ = [
    "RaggedTimeSeriesClassificationBatch",
    "RaggedTimeSeriesClassificationTerm",
    "TrajectoryCaseClassificationBatch",
    "TrajectoryCaseClassificationTerm",
    "TrajectoryClassificationMeasure",
]

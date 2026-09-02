#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

import phydrax.ein as ein

from ...._doc import DOC_KEY0
from ..._keys import EvalKey, fold_in_eval_key, split_eval_key
from ..data import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
)
from ._losses import AbstractOperatorLossTerm, OperatorLossContext


OperatorReduction = Literal["none", "mean", "sum"]


def _sample_layout(
    values: Any,
    query: FunctionSamples,
    case_shape: Sequence[int],
    /,
) -> tuple[Array, tuple[int, ...]]:
    array = jnp.asarray(values)
    prefix = tuple(int(size) for size in case_shape) + query.sample_shape
    if tuple(int(size) for size in array.shape[: len(prefix)]) != prefix:
        raise ValueError(
            f"Operator values must start with case/query shape {prefix}; got {array.shape}."
        )
    trailing = tuple(int(size) for size in array.shape[len(prefix) :])
    if len(trailing) > 1:
        raise ValueError("Operator Hilbert utilities support at most one channel axis.")
    return array, trailing


def _expanded_measure(
    query: FunctionSamples,
    case_shape: Sequence[int],
    channel_shape: Sequence[int],
    /,
) -> Array:
    weights = query.weights(case_shape=case_shape)
    return weights.reshape(weights.shape + (1,) * len(tuple(channel_shape)))


def operator_integral(
    values: Any,
    query: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] = (),
) -> Array:
    """Integrate scalar or channel-last values with physical quadrature and masks."""
    array, channels = _sample_layout(values, query, case_shape)
    weights = _expanded_measure(query, case_shape, channels)
    start = len(tuple(case_shape))
    axes = tuple(range(start, start + len(query.sample_shape)))
    return jnp.sum(array * weights, axis=axes)


def operator_hilbert_inner_product(
    left: Any,
    right: Any,
    query: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] = (),
    channel_metric: Array | None = None,
    reduction: OperatorReduction = "none",
) -> Array:
    """Measure-aware complex Hilbert inner product, reduced independently by case."""
    left_array, left_channels = _sample_layout(left, query, case_shape)
    right_array, right_channels = _sample_layout(right, query, case_shape)
    if left_array.shape != right_array.shape or left_channels != right_channels:
        raise ValueError("Hilbert inner-product operands must have identical shapes.")
    if channel_metric is None:
        density = jnp.conj(left_array) * right_array
        if left_channels:
            density = jnp.sum(density, axis=-1)
    else:
        if not left_channels:
            raise ValueError("channel_metric requires channel-valued operands.")
        metric = jnp.asarray(channel_metric)
        channels = left_channels[0]
        if metric.shape != (channels, channels):
            raise ValueError(
                f"channel_metric must have shape {(channels, channels)}; got {metric.shape}."
            )
        density = ein.contract(
            "...i,ij,...j->...",
            jnp.conj(left_array),
            metric,
            right_array,
        )
    weights = query.weights(case_shape=case_shape)
    start = len(tuple(case_shape))
    axes = tuple(range(start, start + len(query.sample_shape)))
    values = jnp.sum(density * weights, axis=axes)
    if reduction == "none":
        return values
    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "sum":
        return jnp.sum(values)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def operator_hilbert_norm(
    values: Any,
    query: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] = (),
    channel_metric: Array | None = None,
    squared: bool = False,
    reduction: OperatorReduction = "none",
) -> Array:
    """Norm induced by ``operator_hilbert_inner_product``."""
    energy = jnp.real(
        operator_hilbert_inner_product(
            values,
            values,
            query,
            case_shape=case_shape,
            channel_metric=channel_metric,
            reduction="none",
        )
    )
    norms = jnp.maximum(energy, 0.0) if squared else jnp.sqrt(jnp.maximum(energy, 0.0))
    if reduction == "none":
        return norms
    if reduction == "mean":
        return jnp.mean(norms)
    if reduction == "sum":
        return jnp.sum(norms)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def operator_hilbert_relative_error(
    prediction: Any,
    target: Any,
    query: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] = (),
    channel_metric: Array | None = None,
    squared: bool = False,
    epsilon: float = 1e-12,
    reduction: OperatorReduction = "mean",
) -> Array:
    """Relative error in an explicitly measured scalar/channel Hilbert space."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    numerator = operator_hilbert_norm(
        jnp.asarray(prediction) - jnp.asarray(target),
        query,
        case_shape=case_shape,
        channel_metric=channel_metric,
        squared=True,
        reduction="none",
    )
    denominator = operator_hilbert_norm(
        target,
        query,
        case_shape=case_shape,
        channel_metric=channel_metric,
        squared=True,
        reduction="none",
    )
    ratio = numerator / jnp.maximum(denominator, float(epsilon))
    values = ratio if squared else jnp.sqrt(jnp.maximum(ratio, 0.0))
    if reduction == "none":
        return values
    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "sum":
        return jnp.sum(values)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def project_operator_conservation(
    values: Any,
    query: FunctionSamples,
    target_total: Any,
    /,
    *,
    case_shape: Sequence[int] = (),
    correction_basis: Any | None = None,
) -> Array:
    """Project onto an exact integral using a constant or supplied correction basis."""
    array, channels = _sample_layout(values, query, case_shape)
    current = operator_integral(array, query, case_shape=case_shape)
    target = jnp.asarray(target_total, dtype=array.dtype)
    if target.ndim + 1 == current.ndim:
        target = target[..., None]
    target = jnp.broadcast_to(target, current.shape)

    mask = query.mask_array(case_shape=case_shape)
    if channels:
        mask = mask[..., None]
    if correction_basis is None:
        basis = jnp.ones_like(array)
    else:
        basis = jnp.asarray(correction_basis, dtype=array.dtype)
        if basis.ndim + 1 == array.ndim and channels:
            basis = basis[..., None]
        basis = jnp.broadcast_to(basis, array.shape)
    basis = jnp.where(mask, basis, jnp.zeros((), dtype=array.dtype))
    basis_total = operator_integral(basis, query, case_shape=case_shape)
    defect = target - current
    array = eqx.error_if(
        array,
        jnp.any((jnp.abs(basis_total) == 0.0) & (jnp.abs(defect) > 0.0)),
        "The correction basis has zero integral for a nonzero conservation defect.",
    )
    scale = jnp.where(jnp.abs(basis_total) > 0.0, defect / basis_total, 0.0)
    scale_shape = (
        tuple(int(size) for size in case_shape)
        + ((1,) * len(query.sample_shape))
        + channels
    )
    return jnp.where(
        mask,
        array + basis * scale.reshape(scale_shape),
        jnp.zeros((), dtype=array.dtype),
    )


def _replace_prediction_field(
    prediction: OperatorPrediction,
    field_name: str,
    values: Array,
    /,
) -> OperatorPrediction:
    fields = dict(prediction.fields)
    field = fields[field_name]
    query = prediction.query_geometry(field.query_name)
    mask = query.mask_array(case_shape=prediction.case_shape)
    trailing = (1,) * (jnp.asarray(values).ndim - mask.ndim)
    values = jnp.where(
        mask.reshape(mask.shape + trailing),
        values,
        jnp.zeros((), dtype=jnp.asarray(values).dtype),
    )
    fields[field_name] = OperatorFieldBatch(
        values,
        query_name=field.query_name,
        spec=field.spec,
    )
    return OperatorPrediction(
        fields,
        prediction.queries,
        case_axes=prediction.case_axes,
        case_shape=prediction.case_shape,
    )


def _broadcast_field_value(value: Any, field: OperatorFieldBatch, /) -> Array:
    array = jnp.asarray(value, dtype=field.values.dtype)
    if array.ndim + 1 == field.values.ndim and field.spec.channels != "scalar":
        array = array[..., None]
    return jnp.broadcast_to(array, field.values.shape)


class AbstractOperatorOutputTransform(ABC):
    """Metadata-preserving differentiable transform of one physical prediction."""

    field_name: str

    @abstractmethod
    def __call__(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
    ) -> OperatorPrediction:
        raise NotImplementedError

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Stable identity used by exact-resume and artifact contracts."""
        raise NotImplementedError


@dataclass(frozen=True)
class HardConstraintTransform(AbstractOperatorOutputTransform):
    """Exact ansatz ``lift(x, sources) + envelope(x, sources) * raw(x)``."""

    field_name: str
    envelope_fn: Callable[..., Array]
    identity: str
    lift_fn: Callable[..., Array] | None = None

    def __post_init__(self):
        if not self.field_name:
            raise ValueError("field_name must be non-empty.")
        if not self.identity:
            raise ValueError("Hard constraint identity must be non-empty.")
        if not callable(self.envelope_fn):
            raise TypeError("envelope_fn must be callable.")
        if self.lift_fn is not None and not callable(self.lift_fn):
            raise TypeError("lift_fn must be callable when provided.")

    def __call__(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
    ) -> OperatorPrediction:
        field = prediction.field(self.field_name)
        query = prediction.query_geometry(field.query_name)
        coordinates = query.coordinates_array(case_shape=prediction.case_shape)
        envelope = _broadcast_field_value(
            self.envelope_fn(coordinates, batch, key=fold_in_eval_key(key, 0)),
            field,
        )
        lift = (
            jnp.zeros_like(field.values)
            if self.lift_fn is None
            else _broadcast_field_value(
                self.lift_fn(coordinates, batch, key=fold_in_eval_key(key, 1)),
                field,
            )
        )
        transformed = lift + envelope * field.values
        return _replace_prediction_field(
            prediction,
            self.field_name,
            transformed,
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "hard_constraint",
                "field_name": self.field_name,
                "identity": self.identity,
                "has_lift": self.lift_fn is not None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConservationProjection(AbstractOperatorOutputTransform):
    """Exact measure-aware conservation transform for one predicted field."""

    field_name: str
    source_name: str | None = None
    target_total_fn: Callable[..., Array] | None = None
    identity: str | None = None
    correction_fn: Callable[..., Array] | None = None

    def __post_init__(self):
        if not self.field_name:
            raise ValueError("field_name must be non-empty.")
        configured = int(self.source_name is not None) + int(
            self.target_total_fn is not None
        )
        if configured != 1:
            raise ValueError(
                "Configure exactly one of source_name or target_total_fn for conservation."
            )
        if self.target_total_fn is not None and not callable(self.target_total_fn):
            raise TypeError("target_total_fn must be callable.")
        if self.correction_fn is not None and not callable(self.correction_fn):
            raise TypeError("correction_fn must be callable when provided.")

        if (
            self.target_total_fn is not None or self.correction_fn is not None
        ) and not self.identity:
            raise ValueError(
                "Callable conservation transforms require a stable non-empty identity."
            )

    def __call__(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey,
    ) -> OperatorPrediction:
        field = prediction.field(self.field_name)
        query = prediction.query_geometry(field.query_name)
        if self.source_name is None:
            assert self.target_total_fn is not None
            target_total = self.target_total_fn(batch, key=fold_in_eval_key(key, 0))
        else:
            source = batch.input(self.source_name)
            if source.values is None:
                raise ValueError(
                    f"Conservation source {self.source_name!r} has no values."
                )
            target_total = operator_integral(
                source.values,
                source,
                case_shape=batch.case_shape,
            )
        correction_basis = None
        if self.correction_fn is not None:
            coordinates = query.coordinates_array(case_shape=prediction.case_shape)
            correction_basis = self.correction_fn(
                coordinates,
                batch,
                key=fold_in_eval_key(key, 1),
            )
        projected = project_operator_conservation(
            field.values,
            query,
            target_total,
            case_shape=prediction.case_shape,
            correction_basis=correction_basis,
        )
        return _replace_prediction_field(prediction, self.field_name, projected)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "conservation_projection",
                "field_name": self.field_name,
                "source_name": self.source_name,
                "identity": self.identity,
                "has_correction_basis": self.correction_fn is not None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OperatorOutputPipeline(eqx.Module):
    """Ordered physical-space output transforms with metadata preservation."""

    transforms: tuple[AbstractOperatorOutputTransform, ...] = eqx.field(static=True)

    def __init__(self, *transforms: AbstractOperatorOutputTransform):
        if any(
            not isinstance(item, AbstractOperatorOutputTransform) for item in transforms
        ):
            raise TypeError("Output pipeline entries must be operator output transforms.")
        leaves = jax.tree_util.tree_leaves(
            tuple(value for item in transforms for value in vars(item).values())
        )
        if any(eqx.is_array(leaf) for leaf in leaves):
            raise ValueError(
                "Output pipelines must not contain trainable or stateful array leaves."
            )
        self.transforms = tuple(transforms)

    def __call__(
        self,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        output = prediction
        keys = split_eval_key(key, len(self.transforms)) if self.transforms else ()
        for transform, transform_key in zip(self.transforms, keys, strict=True):
            output = transform(output, batch, key=transform_key)
        return output

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "operator_output_pipeline",
                "semantics": "physical-v1",
                "transforms": [item.fingerprint for item in self.transforms],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def operator_weak_form_loss(
    residual: Any,
    test_functions: Any,
    query: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] = (),
    normalize_tests: bool = True,
    reduction: OperatorReduction = "mean",
    epsilon: float = 1e-12,
) -> Array:
    """Squared weak residual moments against one or more test functions."""
    residual_array, channels = _sample_layout(residual, query, case_shape)
    tests = jnp.asarray(test_functions)
    case = tuple(int(size) for size in case_shape)
    sample = query.sample_shape
    shared_shape = sample + (int(tests.shape[-1]),)
    case_shape_with_tests = case + shared_shape
    if tuple(int(size) for size in tests.shape) == shared_shape:
        tests = jnp.broadcast_to(tests, case_shape_with_tests)
    elif tuple(int(size) for size in tests.shape) != case_shape_with_tests:
        raise ValueError(
            "test_functions must have shape sample_shape + (num_tests,) or "
            "case_shape + sample_shape + (num_tests,)."
        )
    case_count = prod(case) if case else 1
    sample_count = prod(sample)
    channel_count = channels[0] if channels else 1
    residual_flat = residual_array.reshape((case_count, sample_count, channel_count))
    tests_flat = tests.reshape((case_count, sample_count, tests.shape[-1]))
    weights = query.weights(case_shape=case).reshape((case_count, sample_count))
    moments = ein.contract(
        "cst,csk,cs->ctk",
        jnp.conj(tests_flat),
        residual_flat,
        weights,
    )
    energy = jnp.abs(moments) ** 2
    if normalize_tests:
        test_energy = ein.contract(
            "cst,cst,cs->ct",
            jnp.conj(tests_flat),
            tests_flat,
            weights,
        ).real
        energy = energy / jnp.maximum(test_energy[..., None], float(epsilon))
    case_values = jnp.sum(energy, axis=(-2, -1)).reshape(case)
    if reduction == "none":
        return case_values
    if reduction == "mean":
        return jnp.mean(case_values)
    if reduction == "sum":
        return jnp.sum(case_values)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


@dataclass(frozen=True)
class WeakOperatorLoss(AbstractOperatorLossTerm):
    """Dynamic weak-form term over arbitrary physical or execution residuals."""

    name: str
    residual_fn: Callable[..., Array]
    test_fn: Callable[..., Array]
    identity: str
    query_name: str | None = None
    weight: float = 1.0
    normalize_tests: bool = True
    space: Literal["execution", "physical"] = "physical"

    def __post_init__(self):
        if not self.name or not self.identity:
            raise ValueError("Weak loss name and identity must be non-empty.")
        if not callable(self.residual_fn) or not callable(self.test_fn):
            raise TypeError("Weak loss residual_fn and test_fn must be callable.")
        if not isfinite(float(self.weight)):
            raise ValueError("Weak loss weight must be finite.")
        if self.space not in ("execution", "physical"):
            raise ValueError("Weak loss space must be 'execution' or 'physical'.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: Any,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        residual_key, test_key = split_eval_key(key, 2)
        selected_prediction, selected_batch, selected_targets = context.view(self.space)
        residual = self.residual_fn(
            selected_prediction,
            selected_batch,
            selected_targets,
            model=model,
            key=residual_key,
            step=step,
            training=training,
            context=context,
        )
        query_name = self.query_name
        if query_name is None:
            query_name = selected_batch.single_query_name()
        tests = self.test_fn(
            selected_batch,
            context=context,
            key=test_key,
            step=step,
            training=training,
        )
        value = operator_weak_form_loss(
            residual,
            tests,
            selected_batch.query(query_name),
            case_shape=selected_batch.case_shape,
            normalize_tests=self.normalize_tests,
        )
        return jnp.asarray(self.weight, dtype=jnp.asarray(value).dtype) * value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "weak_operator_loss",
                "name": self.name,
                "identity": self.identity,
                "query_name": self.query_name,
                "weight": self.weight,
                "normalize_tests": self.normalize_tests,
                "space": self.space,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AbstractOperatorOutputTransform",
    "ConservationProjection",
    "HardConstraintTransform",
    "OperatorOutputPipeline",
    "WeakOperatorLoss",
    "operator_hilbert_inner_product",
    "operator_hilbert_norm",
    "operator_hilbert_relative_error",
    "operator_integral",
    "operator_weak_form_loss",
    "project_operator_conservation",
]

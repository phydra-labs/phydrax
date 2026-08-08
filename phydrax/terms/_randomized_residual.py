#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..operators.differential._dimension_estimators import DimensionOperatorSamples
from ..operators.differential._stochastic_estimators import StochasticOperatorSamples


RandomizedResidualLossMode: TypeAlias = Literal[
    "u_statistic",
    "independent_product",
    "plug_in",
]
RandomizedResidualSamplingMode: TypeAlias = Literal["fixed", "resample"]


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _batch_id(label: str, /) -> str:
    digest = sha256(b"phydrax-randomized-residual-batch\0")
    digest.update(label.encode("utf-8"))
    return digest.hexdigest()


class RandomizedResidualSamples(StrictModule):
    """Probe-first residual realizations with explicit sample and event axes."""

    values: Array
    mask: Array
    weights: Array
    dependence_ids: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int] = (),
        event_shape: Sequence[int] = (),
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        dependence_ids: ArrayLike | None = None,
        estimator_id: str = "randomized-residual",
    ):
        samples = jnp.asarray(values)
        sample_axes = _shape(sample_shape, owner="sample_shape")
        event_axes = _shape(event_shape, owner="event_shape")
        expected_ndim = 1 + len(sample_axes) + len(event_axes)
        if samples.ndim != expected_ndim:
            raise ValueError(
                "values must have shape (num_realizations,) + sample_shape + "
                "event_shape."
            )
        if samples.shape[1:] != sample_axes + event_axes:
            raise ValueError("values trailing dimensions do not match declared shapes.")
        count = int(samples.shape[0])
        if count < 2:
            raise ValueError("At least two residual realizations are required.")
        if mask is None:
            valid = jnp.ones(sample_axes, dtype=bool)
        else:
            valid = jnp.asarray(mask, dtype=bool)
            if valid.shape != sample_axes:
                raise ValueError("mask must have sample_shape.")
        if weights is None:
            sample_weights = jnp.ones(sample_axes, dtype=float)
        else:
            sample_weights = jnp.asarray(weights, dtype=float)
            if sample_weights.shape != sample_axes:
                raise ValueError("weights must have sample_shape.")
        sample_weights = eqx.error_if(
            sample_weights,
            jnp.any(~jnp.isfinite(sample_weights)) | jnp.any(sample_weights < 0.0),
            "weights must be finite and nonnegative.",
        )
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if dependence_ids is None
            else jnp.asarray(dependence_ids, dtype=jnp.int32)
        )
        if ids.shape != (count,):
            raise ValueError("dependence_ids must have shape (num_realizations,).")
        if not isinstance(estimator_id, str) or not estimator_id:
            raise ValueError("estimator_id must be a non-empty string.")
        self.values = samples
        self.mask = valid
        self.weights = sample_weights
        self.dependence_ids = ids
        self.sample_shape = sample_axes
        self.event_shape = event_axes
        self.estimator_id = estimator_id

    @property
    def num_realizations(self) -> int:
        return int(self.values.shape[0])

    @property
    def mean(self) -> Array:
        return jnp.mean(self.values, axis=0)

    @property
    def standard_error(self) -> Array:
        centered = self.values - self.mean
        variance = jnp.sum(jnp.abs(centered) ** 2, axis=0) / float(
            self.num_realizations - 1
        )
        return jnp.sqrt(variance / float(self.num_realizations))


class RandomizedResidualBatch(StrictModule):
    """One physical collocation batch and two independent derivative-probe keys."""

    collocation: Any
    left_key: Array
    right_key: Array
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        collocation: Any,
        left_key: Key[Array, ""],
        right_key: Key[Array, ""],
        /,
        *,
        batch_id: str = "randomized-residual",
    ):
        self.collocation = collocation
        self.left_key = left_key
        self.right_key = right_key
        self.batch_id = _batch_id(batch_id)


class RandomizedResidualDiagnostics(StrictModule):
    objective: Array
    plug_in_residual_norm: Array
    mean_probe_standard_error: Array
    valid_fraction: Array
    negative: Array
    finite: Array
    num_realizations: int = eqx.field(static=True)
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


ResidualEvaluator: TypeAlias = Callable[
    [Mapping[str, DomainFunction], Any, Key[Array, ""]],
    RandomizedResidualSamples | StochasticOperatorSamples | DimensionOperatorSamples,
]
BatchSampler: TypeAlias = Callable[[Key[Array, ""]], Any]


def _operator_samples(
    value: RandomizedResidualSamples
    | StochasticOperatorSamples
    | DimensionOperatorSamples,
    /,
) -> RandomizedResidualSamples:
    if isinstance(value, RandomizedResidualSamples):
        return value
    if isinstance(value, StochasticOperatorSamples):
        return RandomizedResidualSamples(
            value.values,
            event_shape=value.mean.shape,
            dependence_ids=value.dependence_ids,
            estimator_id=f"stochastic-{value.distribution}",
        )
    if isinstance(value, DimensionOperatorSamples):
        return RandomizedResidualSamples(
            value.values,
            event_shape=value.mean.shape,
            dependence_ids=value.dependence_ids,
            estimator_id=f"dimension-{value.policy_id}",
        )
    raise TypeError("residual_evaluator returned an unsupported sample object.")


def _event_inner(values: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.real(jnp.conj(values) * values)
    event_size = prod(event_shape)
    flattened = values.reshape(values.shape[: -len(event_shape)] + (event_size,))
    return jnp.sum(jnp.real(jnp.conj(flattened) * flattened), axis=-1)


def _cross_inner(left: Array, right: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.real(jnp.conj(left) * right)
    event_size = prod(event_shape)
    left_flat = left.reshape(left.shape[: -len(event_shape)] + (event_size,))
    right_flat = right.reshape(right.shape[: -len(event_shape)] + (event_size,))
    return jnp.sum(jnp.real(jnp.conj(left_flat) * right_flat), axis=-1)


def _per_sample_loss(
    left: RandomizedResidualSamples,
    mode: RandomizedResidualLossMode,
    /,
    *,
    right: RandomizedResidualSamples | None,
) -> Array:
    if mode == "plug_in":
        return _event_inner(left.mean, left.event_shape)
    if mode == "independent_product":
        if right is None:
            raise RuntimeError("Independent-product residual samples are unavailable.")
        if (
            right.sample_shape != left.sample_shape
            or right.event_shape != left.event_shape
            or right.values.shape != left.values.shape
        ):
            raise ValueError("Left and right residual sample shapes must match.")
        return _cross_inner(left.mean, right.mean, left.event_shape)
    count = left.num_realizations
    summed = jnp.sum(left.values, axis=0)
    total_cross = _event_inner(summed, left.event_shape) - jnp.sum(
        _event_inner(left.values, left.event_shape),
        axis=0,
    )
    return total_cross / float(count * (count - 1))


def _reduce(
    values: Array,
    mask: Array,
    weights: Array,
    /,
) -> Array:
    effective = jnp.where(mask, weights, 0.0)
    mass = jnp.sum(effective)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        "Randomized residual batch has zero valid sample mass.",
    )
    return jnp.sum(jnp.where(mask, effective * values, 0.0)) / mass


class RandomizedResidualTerm(AbstractSamplingTerm):
    """Estimator-aware squared residual term over raw probe realizations."""

    residual_evaluator: ResidualEvaluator
    fixed_collocation: Any
    batch_sampler: BatchSampler | None
    scalar_weight: Array
    loss_mode: RandomizedResidualLossMode = eqx.field(static=True)
    sampling_mode: RandomizedResidualSamplingMode = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        residual_evaluator: ResidualEvaluator,
        /,
        *,
        collocation: Any | BatchSampler,
        loss_mode: RandomizedResidualLossMode = "u_statistic",
        sampling_mode: RandomizedResidualSamplingMode = "resample",
        scalar_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not callable(residual_evaluator):
            raise TypeError("residual_evaluator must be callable.")
        if loss_mode not in ("u_statistic", "independent_product", "plug_in"):
            raise ValueError("Unknown randomized residual loss_mode.")
        if sampling_mode not in ("fixed", "resample"):
            raise ValueError("sampling_mode must be 'fixed' or 'resample'.")
        if sampling_mode == "resample":
            if not callable(collocation):
                raise TypeError("Resampled objectives require a collocation callable.")
            fixed = None
            sampler = collocation
        else:
            if callable(collocation):
                raise TypeError("Fixed objectives require a materialized collocation batch.")
            fixed = collocation
            sampler = None
        weight = jnp.asarray(scalar_weight, dtype=float).reshape(())
        if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
            raise ValueError("scalar_weight must be finite and nonnegative.")
        self.residual_evaluator = residual_evaluator
        self.fixed_collocation = fixed
        self.batch_sampler = sampler
        self.scalar_weight = weight
        self.loss_mode = loss_mode
        self.sampling_mode = sampling_mode
        self.label = label

    def sample(self, *, key: Key[Array, ""] = jr.key(0)) -> RandomizedResidualBatch:
        collocation_key, left_key, right_key = jr.split(key, 3)
        if self.sampling_mode == "fixed":
            collocation = self.fixed_collocation
        else:
            if self.batch_sampler is None:
                raise RuntimeError("Randomized residual batch sampler is unavailable.")
            collocation = self.batch_sampler(collocation_key)
        return RandomizedResidualBatch(
            collocation,
            left_key,
            right_key,
            batch_id=self.label or "randomized-residual",
        )

    def _evaluate(
        self,
        functions: Mapping[str, DomainFunction],
        batch: RandomizedResidualBatch,
        /,
    ) -> tuple[RandomizedResidualSamples, RandomizedResidualSamples | None]:
        if not isinstance(batch, RandomizedResidualBatch):
            raise TypeError("batch must be a RandomizedResidualBatch.")
        left = _operator_samples(
            self.residual_evaluator(functions, batch.collocation, batch.left_key)
        )
        right = (
            _operator_samples(
                self.residual_evaluator(functions, batch.collocation, batch.right_key)
            )
            if self.loss_mode == "independent_product"
            else None
        )
        if right is not None:
            if (
                right.sample_shape != left.sample_shape
                or right.event_shape != left.event_shape
                or right.values.shape != left.values.shape
            ):
                raise ValueError("Left and right residual sample shapes must match.")
            checked_values = eqx.error_if(
                right.values,
                ~(
                    jnp.array_equal(left.mask, right.mask)
                    & jnp.array_equal(left.weights, right.weights)
                ),
                "Independent residual ensembles must share masks and weights.",
            )
            right = eqx.tree_at(
                lambda samples: samples.values,
                right,
                checked_values,
            )
        return left, right

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        iter_: int | Array | None = None,
        batch: RandomizedResidualBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        materialized = self.sample(key=key) if batch is None else batch
        left, right = self._evaluate(functions, materialized)
        values = _per_sample_loss(left, self.loss_mode, right=right)
        return self.scalar_weight * _reduce(values, left.mask, left.weights)

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: RandomizedResidualBatch | None = None,
    ) -> RandomizedResidualDiagnostics:
        materialized = self.sample(key=key) if batch is None else batch
        left, right = self._evaluate(functions, materialized)
        objective = self.scalar_weight * _reduce(
            _per_sample_loss(left, self.loss_mode, right=right),
            left.mask,
            left.weights,
        )
        plug_in = jnp.sqrt(
            _reduce(
                _event_inner(left.mean, left.event_shape),
                left.mask,
                left.weights,
            )
        )
        standard_error = _reduce(
            _event_inner(left.standard_error, left.event_shape),
            left.mask,
            left.weights,
        )
        finite = jnp.isfinite(objective) & jnp.isfinite(plug_in)
        return RandomizedResidualDiagnostics(
            objective=objective,
            plug_in_residual_norm=plug_in,
            mean_probe_standard_error=jnp.sqrt(standard_error),
            valid_fraction=jnp.mean(left.mask),
            negative=objective < 0.0,
            finite=finite,
            num_realizations=left.num_realizations,
            loss_mode=self.loss_mode,
        )


__all__ = [
    "BatchSampler",
    "RandomizedResidualBatch",
    "RandomizedResidualDiagnostics",
    "RandomizedResidualLossMode",
    "RandomizedResidualTerm",
    "RandomizedResidualSamples",
    "RandomizedResidualSamplingMode",
    "ResidualEvaluator",
]

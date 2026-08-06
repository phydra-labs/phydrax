#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._objective import AbstractSamplingObjectiveTerm
from ..domain._function import DomainFunction
from ..stochastic._bsde import _pointwise_autodiff_control, _pointwise_values, BSDEProblem
from ..stochastic._feynman_kac import FeynmanKacLabelBatch, FeynmanKacSamplingPlan


LabelProvider = Callable[[Key[Array, ""]], FeynmanKacLabelBatch]


def _weight(value: ArrayLike, /, *, owner: str) -> Array:
    weight = jnp.asarray(value, dtype=float).reshape(())
    if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return weight


def _validate_batch(
    batch: FeynmanKacLabelBatch,
    problem: BSDEProblem,
    plan: FeynmanKacSamplingPlan,
    /,
) -> None:
    if not isinstance(batch, FeynmanKacLabelBatch):
        raise TypeError("Label providers must return FeynmanKacLabelBatch objects.")
    if (
        batch.problem_id != problem.problem_id
        or batch.process_id != problem.process_id
        or batch.plan_id != plan.plan_id
    ):
        raise ValueError("Feynman-Kac batch provenance does not match the objective.")
    if (
        batch.state_shape != problem.state_shape
        or batch.noise_shape != problem.noise_shape
        or batch.output_shape != problem.output_shape
    ):
        raise ValueError("Feynman-Kac batch event shapes do not match the problem.")


def _weighted_square(
    residual: Array,
    valid: Array,
    weights: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    squared = jnp.abs(residual) ** 2
    event_axes = tuple(range(squared.ndim - len(event_shape), squared.ndim))
    squared = jnp.sum(squared, axis=event_axes)
    effective = jnp.where(valid, weights, 0.0)
    mass = jnp.sum(effective)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        "Feynman-Kac regression batch has zero valid sample mass.",
    )
    return jnp.sum(jnp.where(valid, effective * squared, 0.0)) / mass


class FeynmanKacRegressionDiagnostics(eqx.Module):
    value_rmse: Array
    control_rmse: Array
    mean_value_standard_error: Array
    mean_control_standard_error: Array
    valid_fraction: Array
    control_valid_fraction: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(self.finite) and bool(self.valid_fraction > 0.0)


class FeynmanKacRegressionObjective(AbstractSamplingObjectiveTerm):
    """Weighted supervised regression on frozen stochastic Feynman--Kac labels."""

    problem: BSDEProblem
    plan: FeynmanKacSamplingPlan
    fixed_labels: FeynmanKacLabelBatch | None
    label_provider: LabelProvider | None
    value_weight: Array
    control_weight: Array
    interior_weight: Array
    terminal_weight: Array
    value_name: str = eqx.field(static=True)
    control_name: str | None = eqx.field(static=True)
    use_control_loss: bool = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        plan: FeynmanKacSamplingPlan,
        /,
        *,
        value_name: str,
        labels: FeynmanKacLabelBatch | LabelProvider,
        control_name: str | None = None,
        value_weight: ArrayLike = 1.0,
        control_weight: ArrayLike = 0.0,
        interior_weight: ArrayLike = 1.0,
        terminal_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        if not isinstance(plan, FeynmanKacSamplingPlan):
            raise TypeError("plan must be a FeynmanKacSamplingPlan.")
        if not isinstance(value_name, str) or not value_name:
            raise ValueError("value_name must be a non-empty string.")
        if control_name is not None and (
            not isinstance(control_name, str) or not control_name
        ):
            raise ValueError("control_name must be a non-empty string or None.")
        if isinstance(labels, FeynmanKacLabelBatch):
            fixed_labels = labels
            label_provider = None
            _validate_batch(labels, problem, plan)
        elif callable(labels):
            if plan.refresh_mode != "resample":
                raise ValueError("Callable label providers require refresh_mode='resample'.")
            fixed_labels = None
            label_provider = labels
        else:
            raise TypeError("labels must be a FeynmanKacLabelBatch or provider callable.")
        self.problem = problem
        self.plan = plan
        self.fixed_labels = fixed_labels
        self.label_provider = label_provider
        self.value_name = value_name
        self.control_name = control_name
        self.value_weight = _weight(value_weight, owner="value_weight")
        self.control_weight = _weight(control_weight, owner="control_weight")
        self.interior_weight = _weight(interior_weight, owner="interior_weight")
        self.terminal_weight = _weight(terminal_weight, owner="terminal_weight")
        self.use_control_loss = bool(float(self.control_weight) > 0.0)
        self.label = label

    def sample(self, *, key: Key[Array, ""] = jr.key(0)) -> FeynmanKacLabelBatch:
        if self.fixed_labels is not None:
            return self.fixed_labels
        if self.label_provider is None:
            raise RuntimeError("Feynman-Kac label provider is unavailable.")
        batch = self.label_provider(key)
        _validate_batch(batch, self.problem, self.plan)
        return batch

    def _predictions(
        self,
        functions: Mapping[str, DomainFunction],
        batch: FeynmanKacLabelBatch,
        /,
        *,
        key: Key[Array, ""],
    ) -> tuple[Array, Array | None]:
        if self.value_name not in functions:
            raise KeyError(f"Missing value function {self.value_name!r}.")
        value_model = functions[self.value_name]
        value_key, control_key = jr.split(key)
        values = _pointwise_values(
            value_model,
            batch.query_times,
            batch.query_states,
            self.problem,
            key=value_key,
            output_shape=self.problem.output_shape,
        )
        if batch.control_targets is None:
            return values, None
        if self.control_name is None:
            controls = _pointwise_autodiff_control(
                value_model,
                batch.query_times,
                batch.query_states,
                self.problem,
                key=control_key,
            )
        else:
            if self.control_name not in functions:
                raise KeyError(f"Missing control function {self.control_name!r}.")
            controls = _pointwise_values(
                functions[self.control_name],
                batch.query_times,
                batch.query_states,
                self.problem,
                key=control_key,
                output_shape=self.problem.output_shape + self.problem.noise_shape,
            )
        return values, controls

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        iter_: int | None = None,
        batch: FeynmanKacLabelBatch | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        label_key, prediction_key = jr.split(key)
        labels = self.sample(key=label_key) if batch is None else batch
        _validate_batch(labels, self.problem, self.plan)
        values, controls = self._predictions(
            functions,
            labels,
            key=prediction_key,
        )
        value_targets = jax.lax.stop_gradient(labels.value_targets)
        control_targets = (
            None
            if labels.control_targets is None
            else jax.lax.stop_gradient(labels.control_targets)
        )
        terminal = jnp.isclose(labels.query_times, self.plan.terminal_time)
        weights = labels.sample_weights * jnp.where(
            terminal,
            self.terminal_weight,
            self.interior_weight,
        )
        value_loss = _weighted_square(
            values - value_targets,
            labels.valid,
            weights,
            self.problem.output_shape,
        )
        total = self.value_weight * value_loss
        if self.use_control_loss:
            if controls is None or control_targets is None:
                raise ValueError(
                    "Positive control_weight requires control targets in every label batch."
                )
            control_loss = _weighted_square(
                controls - control_targets,
                labels.control_valid,
                weights,
                self.problem.output_shape + self.problem.noise_shape,
            )
            total = total + self.control_weight * control_loss
        return jnp.asarray(total, dtype=float).reshape(())

    def diagnostics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        batch: FeynmanKacLabelBatch | None = None,
    ) -> FeynmanKacRegressionDiagnostics:
        label_key, prediction_key = jr.split(key)
        labels = self.sample(key=label_key) if batch is None else batch
        _validate_batch(labels, self.problem, self.plan)
        values, controls = self._predictions(
            functions,
            labels,
            key=prediction_key,
        )
        value_residual = values - labels.value_targets
        value_rmse = jnp.sqrt(
            _weighted_square(
                value_residual,
                labels.valid,
                labels.sample_weights,
                self.problem.output_shape,
            )
        )
        value_error_axes = tuple(
            range(
                labels.value_standard_errors.ndim - len(self.problem.output_shape),
                labels.value_standard_errors.ndim,
            )
        )
        mean_value_error = jnp.nanmean(
            jnp.where(
                labels.valid,
                jnp.mean(labels.value_standard_errors, axis=value_error_axes),
                jnp.nan,
            )
        )
        if controls is None or labels.control_targets is None:
            control_rmse = jnp.asarray(jnp.nan)
            mean_control_error = jnp.asarray(jnp.nan)
        else:
            control_rmse = jnp.sqrt(
                _weighted_square(
                    controls - labels.control_targets,
                    labels.control_valid,
                    labels.sample_weights,
                    self.problem.output_shape + self.problem.noise_shape,
                )
            )
            if labels.control_standard_errors is None:
                mean_control_error = jnp.asarray(jnp.nan)
            else:
                control_event = self.problem.output_shape + self.problem.noise_shape
                control_error_axes = tuple(
                    range(
                        labels.control_standard_errors.ndim - len(control_event),
                        labels.control_standard_errors.ndim,
                    )
                )
                mean_control_error = jnp.nanmean(
                    jnp.where(
                        labels.control_valid,
                        jnp.mean(
                            labels.control_standard_errors,
                            axis=control_error_axes,
                        ),
                        jnp.nan,
                    )
                )
        finite = jnp.isfinite(value_rmse) & (
            jnp.isnan(control_rmse) | jnp.isfinite(control_rmse)
        )
        return FeynmanKacRegressionDiagnostics(
            value_rmse=value_rmse,
            control_rmse=control_rmse,
            mean_value_standard_error=mean_value_error,
            mean_control_standard_error=mean_control_error,
            valid_fraction=jnp.mean(labels.valid),
            control_valid_fraction=jnp.mean(labels.control_valid),
            finite=finite,
        )

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
        iter_: int | None = None,
        batch: FeynmanKacLabelBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        del iter_, kwargs
        diagnostics = self.diagnostics(functions, key=key, batch=batch)
        return {
            "value_rmse": diagnostics.value_rmse,
            "control_rmse": diagnostics.control_rmse,
            "valid_fraction": diagnostics.valid_fraction,
        }


__all__ = [
    "FeynmanKacRegressionDiagnostics",
    "FeynmanKacRegressionObjective",
    "LabelProvider",
]

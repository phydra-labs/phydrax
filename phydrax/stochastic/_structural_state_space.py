#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ._linear_gaussian import LinearGaussianParameterization
from ._state_space import (
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    StateSpaceModel,
    StateSpaceStepContext,
)


TransitionValue = Array | Callable[[Array, Array, StateSpaceStepContext], ArrayLike]
ObservationValue = Array | Callable[[Array, StateSpaceStepContext], ArrayLike]


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite_scalar(
    value: ArrayLike,
    /,
    *,
    owner: str,
    lower: float | None = None,
    upper: float | None = None,
    lower_open: bool = False,
    upper_open: bool = False,
) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{owner} must be scalar.")
    host = float(np.asarray(jax.device_get(array)))
    if not np.isfinite(host):
        raise ValueError(f"{owner} must be finite.")
    if lower is not None and (host <= lower if lower_open else host < lower):
        relation = "greater than" if lower_open else "at least"
        raise ValueError(f"{owner} must be {relation} {lower}.")
    if upper is not None and (host >= upper if upper_open else host > upper):
        relation = "less than" if upper_open else "at most"
        raise ValueError(f"{owner} must be {relation} {upper}.")
    return array


def _vector(value: ArrayLike, size: int, /, *, owner: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 0 and size == 1:
        array = array.reshape((1,))
    if array.shape != (size,):
        raise ValueError(f"{owner} must have shape {(size,)}.")
    if bool(jnp.any(~jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


def _covariance(value: ArrayLike, size: int, /, *, owner: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array * jnp.eye(size, dtype=array.dtype)
    if array.shape != (size, size):
        raise ValueError(f"{owner} must be scalar or have shape {(size, size)}.")
    host = np.asarray(jax.device_get(array))
    if not np.all(np.isfinite(host)) or not np.array_equal(host, host.T):
        raise ValueError(f"{owner} must be finite and symmetric.")
    if np.any(np.linalg.eigvalsh(host) < 0.0):
        raise ValueError(f"{owner} must be positive semidefinite.")
    return array


def _positive_interval(t0: Array, t1: Array) -> Array:
    interval = jnp.asarray(t1) - jnp.asarray(t0)
    return eqx.error_if(
        interval,
        ~jnp.isfinite(interval) | (interval < 0.0),
        "Structural transition intervals must be finite and nonnegative.",
    )


class StructuralComponentProvenance(StrictModule):
    """Static identity and state allocation for one compiled structural component."""

    name: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    state_slice: slice = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)
    process_noise_id: str = eqx.field(static=True)


class AbstractStructuralComponent(StrictModule):
    """Named scalar-observation block in an additive structural state-space model."""

    initial_mean: Array
    initial_covariance: Array
    name: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)
    process_noise_id: str = eqx.field(static=True)

    @abstractmethod
    def transition_matrix(
        self, t0: Array, t1: Array, context: StateSpaceStepContext, /
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def process_covariance(
        self, t0: Array, t1: Array, context: StateSpaceStepContext, /
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def observation_loading(
        self, time: Array, context: StateSpaceStepContext, /
    ) -> Array:
        raise NotImplementedError


class LocalLevelComponent(AbstractStructuralComponent):
    """Scalar random-walk level using elapsed physical time."""

    variance: Array

    def __init__(
        self,
        name: str,
        /,
        *,
        process_variance: ArrayLike,
        initial_mean: ArrayLike = 0.0,
        initial_variance: ArrayLike = 1.0,
    ):
        resolved = _name(name, owner="name")
        self.initial_mean = _vector(initial_mean, 1, owner="initial_mean")
        self.initial_covariance = _covariance(
            initial_variance, 1, owner="initial_variance"
        )
        self.variance = _finite_scalar(
            process_variance, owner="process_variance", lower=0.0
        )
        self.name = resolved
        self.kind = "local-level"
        self.state_size = 1
        self.component_id = f"structural:{resolved}:local-level"
        self.transition_id = "random-walk"
        self.process_noise_id = "elapsed-time-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del t0, t1, context
        return jnp.ones((1, 1), dtype=self.initial_mean.dtype)

    def process_covariance(self, t0, t1, context, /):
        del context
        return (self.variance * _positive_interval(t0, t1)).reshape((1, 1))

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.ones((1,), dtype=self.initial_mean.dtype)


class TrendComponent(AbstractStructuralComponent):
    """Local linear level-and-slope trend using elapsed physical time."""

    level_variance: Array
    slope_variance: Array

    def __init__(
        self,
        name: str,
        /,
        *,
        level_variance: ArrayLike,
        slope_variance: ArrayLike,
        initial_mean: ArrayLike = (0.0, 0.0),
        initial_covariance: ArrayLike = 1.0,
    ):
        resolved = _name(name, owner="name")
        self.initial_mean = _vector(initial_mean, 2, owner="initial_mean")
        self.initial_covariance = _covariance(
            initial_covariance, 2, owner="initial_covariance"
        )
        self.level_variance = _finite_scalar(
            level_variance, owner="level_variance", lower=0.0
        )
        self.slope_variance = _finite_scalar(
            slope_variance, owner="slope_variance", lower=0.0
        )
        self.name = resolved
        self.kind = "trend"
        self.state_size = 2
        self.component_id = f"structural:{resolved}:trend"
        self.transition_id = "local-linear-trend"
        self.process_noise_id = "elapsed-time-diagonal-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        one = jnp.ones_like(interval)
        zero = jnp.zeros_like(interval)
        return jnp.stack((jnp.stack((one, interval)), jnp.stack((zero, one))))

    def process_covariance(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        return jnp.diag(jnp.stack((self.level_variance, self.slope_variance)) * interval)

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.asarray((1.0, 0.0), dtype=self.initial_mean.dtype)


class DampedTrendComponent(AbstractStructuralComponent):
    """Damped local trend with one-unit slope retention in ``(0, 1)``."""

    damping: Array
    level_variance: Array
    slope_variance: Array

    def __init__(
        self,
        name: str,
        /,
        *,
        damping: ArrayLike,
        level_variance: ArrayLike,
        slope_variance: ArrayLike,
        initial_mean: ArrayLike = (0.0, 0.0),
        initial_covariance: ArrayLike = 1.0,
    ):
        resolved = _name(name, owner="name")
        self.initial_mean = _vector(initial_mean, 2, owner="initial_mean")
        self.initial_covariance = _covariance(
            initial_covariance, 2, owner="initial_covariance"
        )
        self.damping = _finite_scalar(
            damping,
            owner="damping",
            lower=0.0,
            upper=1.0,
            lower_open=True,
            upper_open=True,
        )
        self.level_variance = _finite_scalar(
            level_variance, owner="level_variance", lower=0.0
        )
        self.slope_variance = _finite_scalar(
            slope_variance, owner="slope_variance", lower=0.0
        )
        self.name = resolved
        self.kind = "damped-trend"
        self.state_size = 2
        self.component_id = f"structural:{resolved}:damped-trend"
        self.transition_id = "fractional-damped-trend"
        self.process_noise_id = "elapsed-time-diagonal-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        retention = self.damping**interval
        accumulation = self.damping * (1.0 - retention) / (1.0 - self.damping)
        one = jnp.ones_like(interval)
        zero = jnp.zeros_like(interval)
        return jnp.stack((jnp.stack((one, accumulation)), jnp.stack((zero, retention))))

    def process_covariance(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        return jnp.diag(jnp.stack((self.level_variance, self.slope_variance)) * interval)

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.asarray((1.0, 0.0), dtype=self.initial_mean.dtype)


class SeasonalComponent(AbstractStructuralComponent):
    """Trigonometric seasonal block with named physical period and harmonics."""

    process_variance: Array
    period: float = eqx.field(static=True)
    harmonics: int = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        period: float,
        /,
        *,
        harmonics: int | None = None,
        process_variance: ArrayLike = 0.0,
        initial_mean: ArrayLike | None = None,
        initial_covariance: ArrayLike = 1.0,
    ):
        resolved = _name(name, owner="name")
        period_value = float(period)
        if not np.isfinite(period_value) or period_value <= 1.0:
            raise ValueError("period must be finite and greater than one.")
        maximum = int(np.floor(period_value / 2.0))
        count = maximum if harmonics is None else int(harmonics)
        if count <= 0 or count > maximum:
            raise ValueError("harmonics must lie between one and floor(period / 2).")
        size = 2 * count
        mean = jnp.zeros((size,)) if initial_mean is None else initial_mean
        self.initial_mean = _vector(mean, size, owner="initial_mean")
        self.initial_covariance = _covariance(
            initial_covariance, size, owner="initial_covariance"
        )
        self.process_variance = _finite_scalar(
            process_variance, owner="process_variance", lower=0.0
        )
        self.name = resolved
        self.kind = "seasonal"
        self.state_size = size
        self.component_id = f"structural:{resolved}:seasonal"
        self.transition_id = "trigonometric-rotation"
        self.process_noise_id = "elapsed-time-isotropic-gaussian"
        self.period = period_value
        self.harmonics = count

    def transition_matrix(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        harmonic = jnp.arange(1, self.harmonics + 1, dtype=self.initial_mean.dtype)
        angles = 2.0 * jnp.pi * harmonic * interval / self.period
        cosine = jnp.cos(angles)
        sine = jnp.sin(angles)
        blocks = jnp.stack(
            (
                jnp.stack((cosine, sine), axis=-1),
                jnp.stack((-sine, cosine), axis=-1),
            ),
            axis=-2,
        )
        indices = jnp.arange(self.state_size).reshape((self.harmonics, 2))
        matrix = jnp.zeros((self.state_size, self.state_size), dtype=blocks.dtype)
        return matrix.at[indices[:, :, None], indices[:, None, :]].set(blocks)

    def process_covariance(self, t0, t1, context, /):
        del context
        interval = _positive_interval(t0, t1)
        return self.process_variance * interval * jnp.eye(self.state_size)

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.tile(jnp.asarray((1.0, 0.0)), self.harmonics)


class RegressionComponent(AbstractStructuralComponent):
    """Static or random-walk regression coefficients with a time-aware design row."""

    design: ObservationValue
    coefficient_covariance: Array

    def __init__(
        self,
        name: str,
        design: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
        /,
        *,
        initial_coefficients: ArrayLike,
        initial_covariance: ArrayLike = 1.0,
        process_covariance: ArrayLike = 0.0,
    ):
        resolved = _name(name, owner="name")
        coefficients = jnp.asarray(initial_coefficients, dtype=float)
        if coefficients.ndim != 1 or int(coefficients.size) <= 0:
            raise ValueError("initial_coefficients must be a non-empty vector.")
        if bool(jnp.any(~jnp.isfinite(coefficients))):
            raise ValueError("initial_coefficients must be finite.")
        size = int(coefficients.size)
        if callable(design):
            resolved_design = cast(
                Callable[[Array, StateSpaceStepContext], ArrayLike], design
            )
        else:
            resolved_design = _vector(design, size, owner="design")
            if bool(jnp.all(resolved_design == 0.0)):
                raise ValueError(
                    "An identically zero fixed regression design is structurally "
                    "unidentifiable."
                )
            if size > 1:
                raise ValueError(
                    "A fixed scalar-observation regression with multiple coefficients "
                    "is unidentifiable; provide a time-varying design callback."
                )
        self.initial_mean = coefficients
        self.initial_covariance = _covariance(
            initial_covariance, size, owner="initial_covariance"
        )
        self.design = resolved_design
        self.coefficient_covariance = _covariance(
            process_covariance, size, owner="process_covariance"
        )
        self.name = resolved
        self.kind = "regression"
        self.state_size = size
        self.component_id = f"structural:{resolved}:regression"
        self.transition_id = "coefficient-random-walk"
        self.process_noise_id = "elapsed-time-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del t0, t1, context
        return jnp.eye(self.state_size, dtype=self.initial_mean.dtype)

    def process_covariance(self, t0, t1, context, /):
        del context
        return self.coefficient_covariance * _positive_interval(t0, t1)

    def observation_loading(self, time, context, /):
        loading = self.design(time, context) if callable(self.design) else self.design
        array = jnp.asarray(loading, dtype=self.initial_mean.dtype)
        if array.shape != (self.state_size,):
            raise ValueError("Regression design callback returned an incompatible shape.")
        return array


class AutoregressiveComponent(AbstractStructuralComponent):
    """Discrete companion-form autoregression observed through its leading state."""

    coefficients: Array
    innovation_variance: Array

    def __init__(
        self,
        name: str,
        coefficients: ArrayLike,
        /,
        *,
        process_variance: ArrayLike,
        initial_mean: ArrayLike | None = None,
        initial_covariance: ArrayLike = 1.0,
    ):
        resolved = _name(name, owner="name")
        coefficient_array = jnp.asarray(coefficients, dtype=float)
        if coefficient_array.ndim != 1 or int(coefficient_array.size) <= 0:
            raise ValueError("coefficients must be a non-empty vector.")
        if bool(jnp.any(~jnp.isfinite(coefficient_array))):
            raise ValueError("coefficients must be finite.")
        size = int(coefficient_array.size)
        mean = jnp.zeros((size,)) if initial_mean is None else initial_mean
        self.initial_mean = _vector(mean, size, owner="initial_mean")
        self.initial_covariance = _covariance(
            initial_covariance, size, owner="initial_covariance"
        )
        self.coefficients = coefficient_array
        self.innovation_variance = _finite_scalar(
            process_variance, owner="process_variance", lower=0.0
        )
        self.name = resolved
        self.kind = "autoregressive"
        self.state_size = size
        self.component_id = f"structural:{resolved}:autoregressive"
        self.transition_id = "discrete-companion"
        self.process_noise_id = "leading-coordinate-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del t0, t1, context
        matrix = jnp.zeros(
            (self.state_size, self.state_size), dtype=self.coefficients.dtype
        )
        matrix = matrix.at[0, :].set(self.coefficients)
        if self.state_size > 1:
            matrix = matrix.at[1:, :-1].set(jnp.eye(self.state_size - 1))
        return matrix

    def process_covariance(self, t0, t1, context, /):
        del t0, t1, context
        covariance = jnp.zeros(
            (self.state_size, self.state_size), dtype=self.innovation_variance.dtype
        )
        return covariance.at[0, 0].set(self.innovation_variance)

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.eye(self.state_size, dtype=self.initial_mean.dtype)[0]


class DeterministicTransitionComponent(AbstractStructuralComponent):
    """Caller-defined transition block with exactly zero process covariance."""

    transition: TransitionValue
    observation: ObservationValue

    def __init__(
        self,
        name: str,
        transition: ArrayLike
        | Callable[[Array, Array, StateSpaceStepContext], ArrayLike],
        observation: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
        /,
        *,
        initial_mean: ArrayLike,
        initial_covariance: ArrayLike,
        transition_id: str = "provided-deterministic-transition",
    ):
        resolved = _name(name, owner="name")
        mean = jnp.asarray(initial_mean, dtype=float)
        if mean.ndim != 1 or int(mean.size) <= 0:
            raise ValueError("initial_mean must be a non-empty vector.")
        if bool(jnp.any(~jnp.isfinite(mean))):
            raise ValueError("initial_mean must be finite.")
        size = int(mean.size)
        resolved_transition: TransitionValue
        if callable(transition):
            resolved_transition = cast(
                Callable[[Array, Array, StateSpaceStepContext], ArrayLike], transition
            )
        else:
            resolved_transition = _covariance_like_matrix(
                transition, size, owner="transition"
            )
        resolved_observation: ObservationValue
        if callable(observation):
            resolved_observation = cast(
                Callable[[Array, StateSpaceStepContext], ArrayLike], observation
            )
        else:
            resolved_observation = _vector(observation, size, owner="observation")
            if bool(jnp.all(resolved_observation == 0.0)):
                raise ValueError("observation must expose at least one state coordinate.")
        self.initial_mean = mean
        self.initial_covariance = _covariance(
            initial_covariance, size, owner="initial_covariance"
        )
        self.transition = resolved_transition
        self.observation = resolved_observation
        self.name = resolved
        self.kind = "deterministic-transition"
        self.state_size = size
        self.component_id = f"structural:{resolved}:deterministic-transition"
        self.transition_id = _name(transition_id, owner="transition_id")
        self.process_noise_id = "none"

    def transition_matrix(self, t0, t1, context, /):
        matrix = (
            self.transition(t0, t1, context)
            if callable(self.transition)
            else self.transition
        )
        array = jnp.asarray(matrix, dtype=self.initial_mean.dtype)
        if array.shape != (self.state_size, self.state_size):
            raise ValueError("Transition callback returned an incompatible shape.")
        return array

    def process_covariance(self, t0, t1, context, /):
        del t0, t1, context
        return jnp.zeros(
            (self.state_size, self.state_size), dtype=self.initial_mean.dtype
        )

    def observation_loading(self, time, context, /):
        loading = (
            self.observation(time, context)
            if callable(self.observation)
            else self.observation
        )
        array = jnp.asarray(loading, dtype=self.initial_mean.dtype)
        if array.shape != (self.state_size,):
            raise ValueError("Observation callback returned an incompatible shape.")
        return array


class ProcessNoiseComponent(AbstractStructuralComponent):
    """Independent white process-noise state added directly to the observation."""

    variance: Array

    def __init__(
        self,
        name: str,
        /,
        *,
        variance: ArrayLike,
        initial_variance: ArrayLike | None = None,
    ):
        resolved = _name(name, owner="name")
        resolved_variance = _finite_scalar(variance, owner="variance", lower=0.0)
        prior_variance = variance if initial_variance is None else initial_variance
        self.initial_mean = jnp.zeros((1,), dtype=resolved_variance.dtype)
        self.initial_covariance = _covariance(prior_variance, 1, owner="initial_variance")
        self.variance = resolved_variance
        self.name = resolved
        self.kind = "process-noise"
        self.state_size = 1
        self.component_id = f"structural:{resolved}:process-noise"
        self.transition_id = "independent-white-noise"
        self.process_noise_id = "white-gaussian"

    def transition_matrix(self, t0, t1, context, /):
        del t0, t1, context
        return jnp.zeros((1, 1), dtype=self.initial_mean.dtype)

    def process_covariance(self, t0, t1, context, /):
        del t0, t1, context
        return self.variance.reshape((1, 1))

    def observation_loading(self, time, context, /):
        del time, context
        return jnp.ones((1,), dtype=self.initial_mean.dtype)


def _covariance_like_matrix(value: ArrayLike, size: int, /, *, owner: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != (size, size):
        raise ValueError(f"{owner} must have shape {(size, size)}.")
    if bool(jnp.any(~jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


class _StructuralTransitionMatrix(StrictModule):
    components: tuple[AbstractStructuralComponent, ...]
    slices: tuple[slice, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    def __call__(self, t0, t1, context, /):
        blocks = tuple(
            component.transition_matrix(t0, t1, context) for component in self.components
        )
        dtype = jnp.result_type(*(block.dtype for block in blocks))
        matrix = jnp.zeros((self.state_size, self.state_size), dtype=dtype)
        for state_slice, block in zip(self.slices, blocks):
            matrix = matrix.at[state_slice, state_slice].set(block)
        return matrix


class _StructuralProcessCovariance(StrictModule):
    components: tuple[AbstractStructuralComponent, ...]
    slices: tuple[slice, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    def __call__(self, t0, t1, context, /):
        blocks = tuple(
            component.process_covariance(t0, t1, context) for component in self.components
        )
        dtype = jnp.result_type(*(block.dtype for block in blocks))
        matrix = jnp.zeros((self.state_size, self.state_size), dtype=dtype)
        for state_slice, block in zip(self.slices, blocks):
            matrix = matrix.at[state_slice, state_slice].set(block)
        return matrix


class _StructuralObservationMatrix(StrictModule):
    components: tuple[AbstractStructuralComponent, ...]
    state_size: int = eqx.field(static=True)

    def __call__(self, time, context, /):
        loading = jnp.concatenate(
            tuple(
                component.observation_loading(time, context)
                for component in self.components
            )
        )
        return loading.reshape((1, self.state_size))


class _ScalarObservationParameter(StrictModule):
    value: ObservationValue
    parameter_name: str = eqx.field(static=True)

    def __call__(self, time, context, /):
        resolved = self.value(time, context) if callable(self.value) else self.value
        array = jnp.asarray(resolved, dtype=float)
        if array.ndim == 0:
            return (
                array.reshape((1, 1))
                if self.parameter_name == "covariance"
                else array.reshape((1,))
            )
        expected = (1, 1) if self.parameter_name == "covariance" else (1,)
        if array.shape != expected:
            raise ValueError(
                f"Observation {self.parameter_name} callback must return shape {expected}."
            )
        return array


def _validate_components(components: tuple[AbstractStructuralComponent, ...], /) -> None:
    if not components:
        raise ValueError("At least one structural component is required.")
    if any(
        not isinstance(component, AbstractStructuralComponent) for component in components
    ):
        raise TypeError("Every component must implement AbstractStructuralComponent.")
    names = tuple(component.name for component in components)
    if len(set(names)) != len(names):
        raise ValueError("Structural component names must be unique.")
    kinds = tuple(component.kind for component in components)
    if "local-level" in kinds and ("trend" in kinds or "damped-trend" in kinds):
        raise ValueError(
            "A local level is redundant with the level already present in a trend."
        )
    if "trend" in kinds and "damped-trend" in kinds:
        raise ValueError("Trend and damped trend components are redundant.")
    for kind in ("local-level", "trend", "damped-trend", "process-noise"):
        if sum(value == kind for value in kinds) > 1:
            raise ValueError(
                f"Multiple additive {kind} components are label-unidentifiable."
            )
    autoregressive = sum(kind == "autoregressive" for kind in kinds)
    if autoregressive > 1:
        raise ValueError(
            "Multiple additive autoregressive components are label-unidentifiable."
        )
    fixed_regressions = sum(
        isinstance(component, RegressionComponent) and not callable(component.design)
        for component in components
    )
    if fixed_regressions > 1:
        raise ValueError(
            "Multiple fixed scalar regressions are collinear and unidentifiable."
        )
    seasonal = [
        component for component in components if isinstance(component, SeasonalComponent)
    ]
    frequencies: set[float] = set()
    for component in seasonal:
        for harmonic in range(1, component.harmonics + 1):
            frequency = round(harmonic / component.period, 15)
            if frequency in frequencies:
                raise ValueError(
                    "Seasonal components contain a duplicated harmonic frequency."
                )
            frequencies.add(frequency)


def compile_structural_state_space(
    components: Sequence[AbstractStructuralComponent],
    observation_variance: ArrayLike | Callable[[Array, StateSpaceStepContext], ArrayLike],
    /,
    *,
    case_shape: Sequence[int] = (),
    observation_offset: ArrayLike
    | Callable[[Array, StateSpaceStepContext], ArrayLike] = 0.0,
    model_id: str = "structural-state-space",
    parameter_id: str | None = None,
    basis_id: str | None = None,
    discretization_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_state_size: int = 256,
) -> StateSpaceModel:
    """Compile named additive structural blocks into an existing ``StateSpaceModel``.

    Component states retain their declaration order. The returned model records exact
    ``slice`` objects and per-block method/noise identities in metadata; no state,
    physical-case, or time axis is collapsed by compilation.
    """
    resolved_components = tuple(components)
    _validate_components(resolved_components)
    cases = tuple(int(size) for size in case_shape)
    if any(size <= 0 for size in cases):
        raise ValueError("case_shape dimensions must be positive.")
    limit = int(max_state_size)
    if limit <= 0:
        raise ValueError("max_state_size must be positive.")
    state_size = sum(component.state_size for component in resolved_components)
    if state_size > limit:
        raise ValueError(
            f"Dense structural compilation supports at most {limit} states; got {state_size}."
        )

    starts: list[int] = []
    cursor = 0
    for component in resolved_components:
        starts.append(cursor)
        cursor += component.state_size
    slices = tuple(
        slice(start, start + component.state_size)
        for start, component in zip(starts, resolved_components)
    )
    component_slices = frozendict(
        {
            component.name: state_slice
            for component, state_slice in zip(resolved_components, slices)
        }
    )
    provenance = tuple(
        StructuralComponentProvenance(
            name=component.name,
            kind=component.kind,
            state_slice=state_slice,
            component_id=component.component_id,
            transition_id=component.transition_id,
            process_noise_id=component.process_noise_id,
        )
        for component, state_slice in zip(resolved_components, slices)
    )

    initial_mean = jnp.concatenate(
        tuple(component.initial_mean for component in resolved_components)
    )
    initial_covariance = jnp.zeros((state_size, state_size), dtype=initial_mean.dtype)
    for component, state_slice in zip(resolved_components, slices):
        initial_covariance = initial_covariance.at[state_slice, state_slice].set(
            component.initial_covariance
        )
    if cases:
        initial_mean = jnp.broadcast_to(initial_mean, cases + (state_size,))
        initial_covariance = jnp.broadcast_to(
            initial_covariance, cases + (state_size, state_size)
        )

    transition_parameters = LinearGaussianParameterization(
        _StructuralTransitionMatrix(resolved_components, slices, state_size),
        _StructuralProcessCovariance(resolved_components, slices, state_size),
        state_shape=(state_size,),
        parameterization_id=f"{model_id}:structural-blocks",
        resolved_method="named-block-structural",
    )
    transition = LinearGaussianTransitionKernel(
        transition_parameters,
        process_id=f"{model_id}:structural-process",
        approximation_id="exact-structural-linear-gaussian",
    )
    observation_covariance: ObservationValue
    if callable(observation_variance):
        observation_covariance = cast(
            Callable[[Array, StateSpaceStepContext], ArrayLike], observation_variance
        )
    else:
        variance_array = jnp.asarray(observation_variance, dtype=float)
        if variance_array.ndim == 0:
            _finite_scalar(variance_array, owner="observation_variance", lower=0.0)
        else:
            _covariance(variance_array, 1, owner="observation_variance")
        observation_covariance = variance_array
    observation_offset_value: ObservationValue
    if callable(observation_offset):
        observation_offset_value = cast(
            Callable[[Array, StateSpaceStepContext], ArrayLike], observation_offset
        )
    else:
        offset_array = jnp.asarray(observation_offset, dtype=float)
        if offset_array.ndim == 0:
            _finite_scalar(offset_array, owner="observation_offset")
        elif offset_array.shape != (1,) or bool(jnp.any(~jnp.isfinite(offset_array))):
            raise ValueError(
                "observation_offset must be finite and scalar or shape (1,)."
            )
        observation_offset_value = offset_array
    observation = LinearGaussianObservationModel(
        _StructuralObservationMatrix(resolved_components, state_size),
        _ScalarObservationParameter(observation_covariance, "covariance"),
        state_shape=(state_size,),
        observation_shape=(1,),
        offset=_ScalarObservationParameter(observation_offset_value, "offset"),
        observation_id=f"{model_id}:additive-observation",
    )
    prior = GaussianStatePrior(
        initial_mean,
        initial_covariance,
        state_shape=(state_size,),
        prior_id=f"{model_id}:structural-prior",
    )
    resolved_metadata = {} if metadata is None else dict(metadata)
    for reserved in (
        "structural_component_slices",
        "structural_component_provenance",
        "structural_component_order",
    ):
        if reserved in resolved_metadata:
            raise ValueError(f"metadata key {reserved!r} is reserved by the compiler.")
    resolved_metadata.update(
        {
            "structural_component_slices": component_slices,
            "structural_component_provenance": provenance,
            "structural_component_order": tuple(
                component.name for component in resolved_components
            ),
        }
    )
    return StateSpaceModel(
        prior,
        transition,
        observation,
        model_id=model_id,
        parameter_id=parameter_id,
        basis_id=basis_id,
        discretization_id=discretization_id,
        metadata=resolved_metadata,
    )


__all__ = [
    "AbstractStructuralComponent",
    "AutoregressiveComponent",
    "compile_structural_state_space",
    "DampedTrendComponent",
    "DeterministicTransitionComponent",
    "LocalLevelComponent",
    "ProcessNoiseComponent",
    "RegressionComponent",
    "SeasonalComponent",
    "StructuralComponentProvenance",
    "TrendComponent",
]

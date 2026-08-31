#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._linear_gaussian import degenerate_gaussian_log_prob
from ._state_space import (
    AbstractTransitionKernel,
    StateSpaceStepContext,
    TransitionSample,
)


if TYPE_CHECKING:
    from ..dynamics import ContinuousSystem, TrajectoryTransitions
    from ..solver import WienerTerm


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


def _identifier(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _scalar(value: ArrayLike, /, *, owner: str) -> Array:
    result = jnp.asarray(value)
    if result.shape != ():
        raise ValueError(f"{owner} must be scalar; got {result.shape}.")
    return result


class EulerMaruyamaParameters(StrictModule):
    """One local affine Gaussian Euler--Maruyama transition."""

    transition: Array
    offset: Array
    covariance: Array
    factor: Array
    interval: Array
    valid: Array


def _euler_maruyama_parameters(
    state: Array,
    drift: Array,
    coefficient: Array,
    interval: Array,
    /,
    *,
    state_size: int,
    valid: Array,
) -> EulerMaruyamaParameters:
    dtype = jnp.result_type(state, drift, coefficient, float)
    interval = interval.astype(dtype)
    state_flat = state.astype(dtype).reshape((state_size,))
    drift_flat = drift.astype(dtype).reshape((state_size,))
    coefficient = coefficient.astype(dtype)
    valid = (
        jnp.asarray(valid, dtype=bool)
        & jnp.isfinite(interval)
        & (interval > 0.0)
        & jnp.all(jnp.isfinite(state_flat))
        & jnp.all(jnp.isfinite(drift_flat))
        & jnp.all(jnp.isfinite(coefficient))
    )
    safe_interval = jnp.where(valid, interval, jnp.zeros_like(interval))
    transition = jnp.eye(state_size, dtype=dtype)
    offset = safe_interval * drift_flat
    factor = jnp.sqrt(safe_interval) * coefficient
    return EulerMaruyamaParameters(
        transition=transition,
        offset=offset,
        covariance=factor @ factor.T,
        factor=factor,
        interval=interval,
        valid=valid,
    )


def _euler_maruyama_mean(
    state: Array,
    parameters: EulerMaruyamaParameters,
    /,
    *,
    state_shape: tuple[int, ...],
    state_size: int,
) -> Array:
    flat = state.reshape((state_size,))
    mean = parameters.transition @ flat + parameters.offset
    return mean.reshape(state_shape)


def _euler_maruyama_sample(
    key: Array,
    state: Array,
    parameters: EulerMaruyamaParameters,
    /,
    *,
    state_shape: tuple[int, ...],
    state_size: int,
    noise_size: int,
) -> tuple[Array, Array]:
    mean = _euler_maruyama_mean(
        state,
        parameters,
        state_shape=state_shape,
        state_size=state_size,
    ).reshape((state_size,))
    noise = jr.normal(key, (noise_size,), dtype=mean.dtype)
    values = (mean + parameters.factor @ noise).reshape(state_shape)
    valid = parameters.valid & jnp.all(jnp.isfinite(values))
    return values, valid


def _euler_maruyama_log_prob(
    next_state: Array,
    state: Array,
    parameters: EulerMaruyamaParameters,
    /,
    *,
    state_shape: tuple[int, ...],
    state_size: int,
) -> Array:
    mean = _euler_maruyama_mean(
        state,
        parameters,
        state_shape=state_shape,
        state_size=state_size,
    )
    residual = next_state.reshape((state_size,)) - mean.reshape((state_size,))
    log_density = degenerate_gaussian_log_prob(residual, parameters.covariance)
    return jnp.where(parameters.valid, log_density, -jnp.inf)


class EulerMaruyamaTransitionKernel(AbstractTransitionKernel):
    """Euler--Maruyama Gaussian transition for a canonical continuous system."""

    system: ContinuousSystem
    wiener_terms: tuple[WienerTerm, ...]
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        system: ContinuousSystem,
        wiener_terms: Sequence[WienerTerm],
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        process_id: str,
        approximation_id: str = "euler-maruyama",
    ):
        from ..dynamics import ContinuousSystem
        from ..solver import WienerTerm

        if not isinstance(system, ContinuousSystem):
            raise TypeError("system must be a ContinuousSystem.")
        terms = tuple(wiener_terms)
        if not terms or any(not isinstance(term, WienerTerm) for term in terms):
            raise TypeError("wiener_terms must contain one or more WienerTerm objects.")
        if any(term.representation != "dense" for term in terms):
            raise ValueError(
                "EulerMaruyamaTransitionKernel currently requires dense Wiener terms."
            )
        names = tuple(term.name for term in terms)
        if len(set(names)) != len(names):
            raise ValueError(
                "WienerTerm names must be unique within a transition kernel."
            )
        resolved_state_shape = _shape(state_shape, owner="state_shape")
        resolved_noise_shape = _shape(noise_shape, owner="noise_shape")
        state_size = prod(resolved_state_shape) if resolved_state_shape else 1
        noise_size = prod(resolved_noise_shape) if resolved_noise_shape else 1
        term_noise_size = sum(term.noise_size for term in terms)
        if system.state_layout.shape != resolved_state_shape:
            raise ValueError(
                "state_shape must match the ContinuousSystem state layout; "
                f"got {resolved_state_shape} and {system.state_layout.shape}."
            )
        if noise_size != term_noise_size:
            raise ValueError(
                "noise_shape size must equal the combined WienerTerm noise size; "
                f"got {noise_size} and {term_noise_size}."
            )
        self.system = system
        self.wiener_terms = terms
        self.state_shape = resolved_state_shape
        self.noise_shape = resolved_noise_shape
        self.state_size = state_size
        self.noise_size = noise_size
        self.process_id = _identifier(process_id, owner="process_id")
        self.approximation_id = _identifier(approximation_id, owner="approximation_id")
        self.has_log_density = True

    def drift(
        self,
        time: ArrayLike,
        state: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        """Evaluate the declared continuous drift in physical state coordinates."""
        time_array = _scalar(time, owner="time")
        state_array = jnp.asarray(state)
        if tuple(state_array.shape) != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        inputs = None
        if self.system.input_layout is not None:
            inputs = context.transition_start_input
            if tuple(inputs.shape) != self.system.input_layout.shape:
                raise ValueError(
                    "Controlled Euler--Maruyama transitions require source-aligned "
                    f"inputs with shape {self.system.input_layout.shape}; got "
                    f"{inputs.shape}."
                )
        drift = jnp.asarray(
            self.system(time_array, state_array, context.args, inputs=inputs)
        )
        if tuple(drift.shape) != self.state_shape:
            raise ValueError(
                "ContinuousSystem drift must preserve state shape; "
                f"expected {self.state_shape}, got {drift.shape}."
            )
        return drift

    def dispersion(
        self,
        time: ArrayLike,
        state: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        """Evaluate the combined state-by-Wiener dispersion matrix."""
        time_array = _scalar(time, owner="time")
        state_array = jnp.asarray(state)
        if tuple(state_array.shape) != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        coefficients = tuple(
            term.coefficient_matrix(time_array, state_array, context.args)
            for term in self.wiener_terms
        )
        return jnp.concatenate(coefficients, axis=-1)

    def parameters(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> EulerMaruyamaParameters:
        state_array = jnp.asarray(state)
        if tuple(state_array.shape) != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        start = _scalar(t0, owner="t0")
        end = _scalar(t1, owner="t1")
        interval = end - start
        drift = self.drift(start, state_array, context)
        coefficient = self.dispersion(start, state_array, context)
        return _euler_maruyama_parameters(
            state_array,
            drift,
            coefficient,
            interval,
            state_size=self.state_size,
            valid=context.input_valid,
        )

    def mean(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        return _euler_maruyama_mean(
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
        )

    def covariance(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        return self.parameters(state, t0, t1, context).covariance

    def sample(
        self,
        key: Array,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> TransitionSample:
        state_array = jnp.asarray(state)
        values, valid = _euler_maruyama_sample(
            key,
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
            noise_size=self.noise_size,
        )
        return TransitionSample(
            values=values,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(
        self,
        next_state: ArrayLike,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> Array:
        next_array = jnp.asarray(next_state)
        if tuple(next_array.shape) != self.state_shape:
            raise ValueError(
                f"next_state must have shape {self.state_shape}; got {next_array.shape}."
            )
        state_array = jnp.asarray(state)
        return _euler_maruyama_log_prob(
            next_array,
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
        )


class EulerMaruyamaQuasiLikelihoodResult(StrictModule):
    """Masked trajectory quasi-likelihood with auditable provenance."""

    log_density: Array
    weighted_log_density: Array
    transition_valid: Array
    total_log_likelihood: Array
    mean_negative_log_likelihood: Array
    effective_weight: Array
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)


class EulerMaruyamaQuasiLikelihood(StrictModule):
    """Weighted Euler--Maruyama quasi-likelihood over trajectory transitions."""

    kernel: EulerMaruyamaTransitionKernel
    normalize_by_interval: bool = eqx.field(static=True)

    def __init__(
        self,
        kernel: EulerMaruyamaTransitionKernel,
        /,
        *,
        normalize_by_interval: bool = False,
    ):
        if not isinstance(kernel, EulerMaruyamaTransitionKernel):
            raise TypeError("kernel must be an EulerMaruyamaTransitionKernel.")
        self.kernel = kernel
        self.normalize_by_interval = bool(normalize_by_interval)

    @staticmethod
    def _context(
        args: Any,
        case_index: Array,
        step_index: Array,
        inputs: Array | None,
        /,
    ) -> StateSpaceStepContext:
        if inputs is None:
            return StateSpaceStepContext.empty(
                args=args,
                case_index=case_index,
                step_index=step_index,
            )
        absent = jnp.empty((0,), dtype=inputs.dtype)
        return StateSpaceStepContext(
            args=args,
            case_index=case_index,
            step_index=step_index,
            transition_start_input=inputs,
            transition_end_input=inputs,
            observation_input=inputs,
            input_breakpoints=absent,
            input_breakpoint_valid=jnp.empty((0,), dtype=bool),
            input_valid=jnp.asarray(True),
            input_signal=None,
        )

    def evaluate(
        self,
        transitions: TrajectoryTransitions,
        /,
        *,
        args: Any = None,
    ) -> EulerMaruyamaQuasiLikelihoodResult:
        from ..dynamics import TrajectoryTransitions

        if not isinstance(transitions, TrajectoryTransitions):
            raise TypeError("transitions must be a TrajectoryTransitions batch.")
        if transitions.state_shape != self.kernel.state_shape:
            raise ValueError(
                "Trajectory transition state shape does not match the kernel; "
                f"got {transitions.state_shape} and {self.kernel.state_shape}."
            )
        pair_shape = tuple(transitions.valid.shape)
        pair_count = pair_shape[-1]
        case_count = prod(pair_shape[:-1]) if pair_shape[:-1] else 1
        count = case_count * pair_count
        source = transitions.source_states.reshape((count,) + self.kernel.state_shape)
        target = transitions.target_states.reshape((count,) + self.kernel.state_shape)
        start = transitions.source_coordinates.reshape((count,))
        end = transitions.target_coordinates.reshape((count,))
        valid = transitions.valid.reshape((count,))
        weights = transitions.weights.reshape((count,))
        if transitions.inputs is None:
            inputs = None
        else:
            if transitions.input_shape is None:
                raise ValueError(
                    "Trajectory transitions with inputs require an input_shape."
                )
            inputs = transitions.inputs.reshape((count,) + transitions.input_shape)
        case_indices = jnp.repeat(jnp.arange(case_count, dtype=jnp.int32), pair_count)
        step_indices = jnp.tile(jnp.arange(pair_count, dtype=jnp.int32), case_count)

        def transition_log_density(source_, target_, start_, end_, case_, step_, input_):
            context = self._context(args, case_, step_, input_)
            return self.kernel.log_prob(target_, source_, start_, end_, context)

        if inputs is None:

            def score_without_input(source_, target_, start_, end_, case_, step_):
                context = self._context(args, case_, step_, None)
                return self.kernel.log_prob(target_, source_, start_, end_, context)

            log_density = jax.vmap(score_without_input)(
                source, target, start, end, case_indices, step_indices
            )
        else:
            log_density = jax.vmap(transition_log_density)(
                source,
                target,
                start,
                end,
                case_indices,
                step_indices,
                inputs,
            )
        intervals = end - start
        normalized_log_density = (
            log_density / jnp.where(intervals > 0.0, intervals, 1.0)
            if self.normalize_by_interval
            else log_density
        )
        active = valid & jnp.isfinite(weights) & (weights > 0.0)
        safe_log_density = jnp.where(active, normalized_log_density, 0.0)
        weighted = jnp.where(active, weights, 0.0) * safe_log_density
        effective_weight = jnp.sum(jnp.where(active, weights, 0.0))
        total = jnp.sum(weighted)
        mean_negative = jnp.where(
            effective_weight > 0.0,
            -total / effective_weight,
            jnp.asarray(jnp.inf, dtype=total.dtype),
        )
        transition_valid = active & jnp.isfinite(log_density)
        return EulerMaruyamaQuasiLikelihoodResult(
            log_density=jnp.where(active, log_density, 0.0).reshape(pair_shape),
            weighted_log_density=weighted.reshape(pair_shape),
            transition_valid=transition_valid.reshape(pair_shape),
            total_log_likelihood=total,
            mean_negative_log_likelihood=mean_negative,
            effective_weight=effective_weight,
            process_id=self.kernel.process_id,
            approximation_id=self.kernel.approximation_id,
            dataset_id=transitions.dataset_id,
        )

    def __call__(
        self,
        transitions: TrajectoryTransitions,
        /,
        *,
        args: Any = None,
    ) -> Array:
        return self.evaluate(
            transitions,
            args=args,
        ).mean_negative_log_likelihood


__all__ = [
    "EulerMaruyamaParameters",
    "EulerMaruyamaQuasiLikelihood",
    "EulerMaruyamaQuasiLikelihoodResult",
    "EulerMaruyamaTransitionKernel",
]

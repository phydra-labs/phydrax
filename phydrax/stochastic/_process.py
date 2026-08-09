#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from abc import abstractmethod
from collections.abc import Callable, Sequence
from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import AbstractAttribute, StrictModule
from ._trajectory import _TrajectoryRecord, StochasticTrajectory
from ._wiener import WienerRealization


ProcessReduction: TypeAlias = Literal["none", "mean", "sum"]


def _digest_array(digest: hashlib._Hash, value: ArrayLike, /) -> None:
    array = np.asarray(jax.device_get(jnp.asarray(value)))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _process_fingerprint(
    drift: Array,
    diffusion: Array,
    /,
    *,
    label: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"phydrax-gaussian-coefficient-process\0")
    _digest_array(digest, drift)
    _digest_array(digest, diffusion)
    digest.update(repr(label).encode("utf-8"))
    return digest.hexdigest()


def _realization_fingerprint(
    process_id: str,
    initial_state: Array,
    driver: WienerRealization,
    /,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"phydrax-process-realization\0")
    digest.update(process_id.encode("utf-8"))
    digest.update(driver.realization_id.encode("ascii"))
    _digest_array(digest, initial_state)
    return digest.hexdigest()


def _duration(t0: ArrayLike, t1: ArrayLike, /) -> Array:
    start = jnp.asarray(t0)
    end = jnp.asarray(t1)
    if start.shape != () or end.shape != ():
        raise ValueError("Process transition times must be scalar.")
    duration = end - start
    return eqx.error_if(
        duration,
        ~jnp.isfinite(duration) | (duration <= 0.0),
        "Process transitions require finite t1 > t0.",
    )


def _positive_shape(values: Sequence[int], /, *, name: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in values)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive.")
    return shape


def _trailing_shape(array: Array, shape: tuple[int, ...], /, *, name: str) -> None:
    if array.ndim < len(shape) or tuple(array.shape[-len(shape) :]) != shape:
        raise ValueError(f"{name} must end in shape {shape}; got {array.shape}.")


def _reduce(value: Array, reduction: ProcessReduction, /) -> Array:
    if reduction == "none":
        return value
    if reduction == "mean":
        return jnp.mean(value)
    if reduction == "sum":
        return jnp.sum(value)
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


class AbstractProcessDistribution(StrictModule):
    """One finite-dimensional process marginal with explicit process uncertainty."""

    event_shape: AbstractAttribute[tuple[int, ...]]
    batch_shape: AbstractAttribute[tuple[int, ...]]
    uncertainty_source: AbstractAttribute[Literal["process"]]

    @property
    @abstractmethod
    def location(self) -> Array:
        """A deterministic representative, not necessarily the distribution mean."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)

    def negative_log_likelihood(
        self,
        value: ArrayLike,
        /,
        *,
        reduction: ProcessReduction = "mean",
    ) -> Array:
        return _reduce(-self.log_prob(value), reduction)


class GaussianProcessDistribution(AbstractProcessDistribution):
    """A nonsingular Gaussian marginal over a latent process state."""

    mean: Array
    covariance: Array
    scale_tril: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    uncertainty_source: Literal["process"] = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        event_shape: Sequence[int],
    ):
        events = _positive_shape(event_shape, name="event_shape")
        size = prod(events)
        mean_array = jnp.asarray(mean)
        _trailing_shape(mean_array, events, name="Gaussian process mean")
        batches = tuple(int(value) for value in mean_array.shape[: -len(events)])
        covariance_array = jnp.asarray(covariance, dtype=mean_array.dtype)
        expected = batches + (size, size)
        if covariance_array.shape == (size, size):
            covariance_array = jnp.broadcast_to(covariance_array, expected)
        if covariance_array.shape != expected:
            raise ValueError(
                f"Gaussian process covariance must have shape {(size, size)} or "
                f"{expected}; got {covariance_array.shape}."
            )
        host_covariance = np.asarray(jax.device_get(covariance_array)).reshape(
            (-1, size, size)
        )
        if not np.all(np.isfinite(host_covariance)):
            raise ValueError("Gaussian process covariance must be finite.")
        if not np.allclose(
            host_covariance,
            np.swapaxes(host_covariance, -1, -2),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("Gaussian process covariance must be symmetric.")
        if np.any(np.linalg.eigvalsh(host_covariance) <= 0.0):
            raise ValueError("Gaussian process covariance must be positive definite.")
        self.mean = mean_array
        self.covariance = covariance_array
        self.scale_tril = jnp.linalg.cholesky(covariance_array)
        self.event_shape = events
        self.batch_shape = batches
        self.uncertainty_source = "process"

    @property
    def location(self) -> Array:
        return self.mean

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        samples = _positive_shape(sample_shape, name="sample_shape")
        noise = jax.random.normal(
            key,
            samples + self.batch_shape + (self.event_size,),
            dtype=self.mean.dtype,
        )
        centered = jnp.einsum("...ij,...j->...i", self.scale_tril, noise)
        mean = self.mean.reshape(self.batch_shape + (self.event_size,))
        return (centered + mean).reshape(samples + self.batch_shape + self.event_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        value_array = jnp.asarray(value, dtype=self.mean.dtype)
        if value_array.shape != self.mean.shape:
            raise ValueError(
                f"Gaussian process value must have shape {self.mean.shape}; "
                f"got {value_array.shape}."
            )
        residual = value_array.reshape(
            self.batch_shape + (self.event_size,)
        ) - self.mean.reshape(self.batch_shape + (self.event_size,))
        solved = jnp.linalg.solve(self.scale_tril, residual[..., None])[..., 0]
        quadratic = jnp.sum(solved**2, axis=-1)
        log_determinant = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)), axis=-1
        )
        return -0.5 * (
            quadratic + log_determinant + self.event_size * jnp.log(2.0 * jnp.pi)
        )


class AbstractPathwiseTransition(StrictModule):
    """A transition map conditioned on an explicit segment of one driver path."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    driver_shape: AbstractAttribute[tuple[int, ...]]
    process_id: AbstractAttribute[str]

    @abstractmethod
    def pathwise_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        driver_increment: ArrayLike,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def combine_driver_segments(
        self,
        first: ArrayLike,
        second: ArrayLike,
        /,
    ) -> Array:
        """Compose adjacent driver segments in chronological order."""
        raise NotImplementedError


class AbstractMarginalTransitionLaw(StrictModule):
    """A transition law after marginalizing the process driver."""

    state_shape: AbstractAttribute[tuple[int, ...]]
    process_id: AbstractAttribute[str]

    @abstractmethod
    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> AbstractProcessDistribution:
        raise NotImplementedError


class ProcessRealization(StrictModule):
    """Reusable initial state and global driver for one pathwise process draw."""

    initial_state: Array
    driver: WienerRealization
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    uncertainty_source: Literal["process"] = eqx.field(static=True)

    def __init__(
        self,
        initial_state: ArrayLike,
        driver: WienerRealization,
        /,
        *,
        state_shape: Sequence[int],
        process_id: str,
    ):
        states = _positive_shape(state_shape, name="state_shape")
        initial = jnp.asarray(initial_state)
        if initial.shape != states:
            raise ValueError(
                f"Process realization initial_state must have shape {states}; "
                f"got {initial.shape}. Random initial conditions must remain a separate "
                "input-uncertainty axis."
            )
        if not isinstance(driver, WienerRealization):
            raise TypeError("Process realizations require a WienerRealization driver.")
        if driver.levy_area != "brownian":
            raise ValueError(
                "Coefficient process realizations require Brownian increments."
            )
        if not isinstance(process_id, str) or not process_id:
            raise ValueError("process_id must be a non-empty string.")
        self.initial_state = initial
        self.driver = driver
        self.state_shape = states
        self.process_id = process_id
        self.realization_id = _realization_fingerprint(process_id, initial, driver)
        self.uncertainty_source = "process"

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.driver.sample_shape

    @property
    def support(self) -> tuple[float, float]:
        return self.driver.support

    def _driver_evaluations(
        self,
        starts: Array,
        ends: Array,
        /,
    ) -> Array:
        if starts.shape != ends.shape or starts.ndim != 1:
            raise ValueError(
                "Driver interval bounds must be matching one-dimensional arrays."
            )
        return self.driver.increments(
            starts,
            ends,
            dtype=self.initial_state.real.dtype,
        )

    def driver_values(self, times: ArrayLike, /) -> Array:
        """Evaluate the same global driver from the support start at every query."""
        query = jnp.asarray(times, dtype=self.initial_state.real.dtype)
        if query.ndim != 1 or int(query.shape[0]) <= 0:
            raise ValueError("Process query times must be a non-empty vector.")
        starts = jnp.full(query.shape, self.support[0], dtype=query.dtype)
        return self._driver_evaluations(starts, query)

    def driver_increment(self, t0: ArrayLike, t1: ArrayLike, /) -> Array:
        """Evaluate one segment of the reusable global driver."""
        start = jnp.asarray(t0, dtype=self.initial_state.real.dtype)
        end = jnp.asarray(t1, dtype=self.initial_state.real.dtype)
        if start.shape != () or end.shape != ():
            raise ValueError("Driver segment times must be scalar.")
        values = self._driver_evaluations(start[None], end[None])
        return jnp.take(values, 0, axis=len(self.sample_shape))


class LatentGaussianCoefficientProcess(
    AbstractPathwiseTransition,
    AbstractMarginalTransitionLaw,
):
    """Drifted Brownian dynamics in a finite latent coefficient space.

    The model is ``dC_t = drift dt + diffusion dW_t``. A
    :class:`ProcessRealization` owns the global Wiener path; querying the same
    realization on different time grids therefore preserves shared path values.
    """

    drift: Array
    diffusion: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        drift: ArrayLike,
        diffusion: ArrayLike,
        /,
        *,
        label: str | None = None,
        process_id: str | None = None,
    ):
        drift_array = jnp.asarray(drift)
        if drift_array.ndim < 1 or any(int(size) <= 0 for size in drift_array.shape):
            raise ValueError(
                "Gaussian coefficient drift must have non-empty state shape."
            )
        diffusion_array = jnp.asarray(diffusion, dtype=drift_array.dtype)
        states = tuple(int(size) for size in drift_array.shape)
        if diffusion_array.ndim != drift_array.ndim + 1:
            raise ValueError(
                "Gaussian coefficient diffusion needs one trailing driver dimension."
            )
        if tuple(diffusion_array.shape[:-1]) != states:
            raise ValueError(
                "Gaussian coefficient diffusion must begin with state_shape."
            )
        driver_size = int(diffusion_array.shape[-1])
        if driver_size <= 0:
            raise ValueError("Gaussian coefficient driver size must be positive.")
        flat = np.asarray(jax.device_get(diffusion_array)).reshape(
            (prod(states), driver_size)
        )
        if not np.all(np.isfinite(flat)) or not np.all(
            np.isfinite(np.asarray(jax.device_get(drift_array)))
        ):
            raise ValueError("Gaussian coefficient drift and diffusion must be finite.")
        if np.linalg.matrix_rank(flat) < prod(states):
            raise ValueError(
                "Gaussian coefficient diffusion must have full row rank so each "
                "marginal has a Lebesgue density."
            )
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be non-empty or None.")
        resolved_id = (
            _process_fingerprint(drift_array, diffusion_array, label=label)
            if process_id is None
            else process_id
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("process_id must be a non-empty string.")
        self.drift = drift_array
        self.diffusion = diffusion_array
        self.state_shape = states
        self.driver_shape = (driver_size,)
        self.process_id = resolved_id
        self.label = label

    def pathwise_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        driver_increment: ArrayLike,
    ) -> Array:
        duration = _duration(t0, t1)
        state_array = jnp.asarray(state, dtype=self.drift.dtype)
        increment = jnp.asarray(driver_increment, dtype=self.drift.dtype)
        _trailing_shape(state_array, self.state_shape, name="process state")
        _trailing_shape(increment, self.driver_shape, name="driver increment")
        state_batch = tuple(state_array.shape[: -len(self.state_shape)])
        driver_batch = tuple(increment.shape[: -len(self.driver_shape)])
        batch = jnp.broadcast_shapes(state_batch, driver_batch)
        state_flat = jnp.broadcast_to(
            state_array,
            batch + self.state_shape,
        ).reshape(batch + (prod(self.state_shape),))
        driver_flat = jnp.broadcast_to(
            increment,
            batch + self.driver_shape,
        ).reshape(batch + (prod(self.driver_shape),))
        diffusion = self.diffusion.reshape(
            (prod(self.state_shape), prod(self.driver_shape))
        )
        update = jnp.einsum("...j,ij->...i", driver_flat, diffusion)
        drift = self.drift.reshape((prod(self.state_shape),))
        return (state_flat + duration * drift + update).reshape(batch + self.state_shape)

    def combine_driver_segments(
        self,
        first: ArrayLike,
        second: ArrayLike,
        /,
    ) -> Array:
        first_array = jnp.asarray(first, dtype=self.drift.dtype)
        second_array = jnp.asarray(second, dtype=self.drift.dtype)
        _trailing_shape(first_array, self.driver_shape, name="first driver segment")
        _trailing_shape(second_array, self.driver_shape, name="second driver segment")
        return first_array + second_array

    def marginal_transition(
        self,
        state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
    ) -> GaussianProcessDistribution:
        duration = _duration(t0, t1)
        state_array = jnp.asarray(state, dtype=self.drift.dtype)
        _trailing_shape(state_array, self.state_shape, name="process state")
        mean = state_array + duration * self.drift
        diffusion = self.diffusion.reshape(
            (prod(self.state_shape), prod(self.driver_shape))
        )
        covariance = duration * (diffusion @ diffusion.T)
        return GaussianProcessDistribution(
            mean,
            covariance,
            event_shape=self.state_shape,
        )

    def realize(
        self,
        key: Key[Array, ""],
        initial_state: ArrayLike,
        /,
        *,
        support: tuple[float, float],
        sample_shape: Sequence[int] = (),
        tolerance: float = 1e-3,
        label: str | None = None,
    ) -> ProcessRealization:
        driver = WienerRealization.independent(
            key,
            self.driver_shape,
            support=support,
            sample_shape=sample_shape,
            tolerance=tolerance,
            noise_id=f"coefficient-process:{self.process_id}",
            label=label,
        )
        return ProcessRealization(
            initial_state,
            driver,
            state_shape=self.state_shape,
            process_id=self.process_id,
        )

    def evaluate(
        self,
        realization: ProcessRealization,
        times: ArrayLike,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        """Evaluate a reusable realization as a canonical stochastic trajectory."""
        if not isinstance(realization, ProcessRealization):
            raise TypeError("evaluate requires a ProcessRealization.")
        if realization.process_id != self.process_id:
            raise ValueError("Process realization was created by a different process.")
        if realization.state_shape != self.state_shape:
            raise ValueError(
                "Process realization state_shape does not match the process."
            )
        query = jnp.asarray(times, dtype=self.drift.real.dtype)
        if query.ndim != 1 or int(query.shape[0]) <= 0:
            raise ValueError("Process query times must be a non-empty vector.")
        if bool(jnp.any(jnp.diff(query) <= 0.0)):
            raise ValueError("Process query times must be strictly increasing.")
        driver_values = realization.driver_values(query)
        sample_shape = realization.sample_shape
        num_times = int(query.shape[0])
        diffusion = self.diffusion.reshape(
            (prod(self.state_shape), prod(self.driver_shape))
        )
        stochastic = jnp.einsum(
            "...tj,ij->...ti",
            driver_values.reshape(sample_shape + (num_times, prod(self.driver_shape))),
            diffusion,
        )
        elapsed = query - realization.support[0]
        deterministic = (
            realization.initial_state.reshape((prod(self.state_shape),))[None, :]
            + elapsed[:, None] * self.drift.reshape((prod(self.state_shape),))[None, :]
        )
        states = (stochastic + deterministic).reshape(
            sample_shape + (num_times,) + self.state_shape
        )
        realization_names = (
            tuple(f"process_{index}" for index in range(len(sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        state_names = (
            (
                ("coefficient",)
                if len(self.state_shape) == 1
                else tuple(
                    f"coefficient_{index}" for index in range(len(self.state_shape))
                )
            )
            if state_axes is None
            else tuple(state_axes)
        )
        record = _TrajectoryRecord(
            query,
            states,
            state_shape=self.state_shape,
            realization_shape=sample_shape,
            realizations=(realization.driver,),
            case_ids=(f"process:{self.process_id}",),
            parameter_ids=(self.process_id,),
            approximation_id=self.process_id,
            uncertainty_source="process",
            metadata={
                "process_id": self.process_id,
                "process_realization_id": realization.realization_id,
            },
        )
        return record.to_stochastic_trajectory(
            realization_axes=realization_names,
            state_axes=state_names,
        )


class ProcessQueryDiagnostics(StrictModule):
    """Consistency of shared path values across two query schedules."""

    max_absolute_error: Array
    root_mean_square_error: Array
    shared_times: int = eqx.field(static=True)
    consistent: bool = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)


class ProcessSampleStatistics(StrictModule):
    """Empirical finite-dimensional statistics for a process marginal."""

    mean: Array
    covariance: Array
    finite_fraction: Array
    average_log_prob: Array
    num_samples: int = eqx.field(static=True)
    uncertainty_source: Literal["process"] = eqx.field(static=True)


class GaussianProcessDiagnostics(StrictModule):
    """Replay, path consistency, cocycle, and marginal-moment diagnostics."""

    mean_relative_error: Array
    covariance_relative_error: Array
    query_max_absolute_error: Array
    cocycle_max_absolute_error: Array
    replay_exact: bool = eqx.field(static=True)
    uncertainty_source: Literal["process"] = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)


def process_query_consistency(
    process: LatentGaussianCoefficientProcess,
    realization: ProcessRealization,
    reference_times: ArrayLike,
    comparison_times: ArrayLike,
    /,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> ProcessQueryDiagnostics:
    """Compare every reference query against the same path on a second grid."""
    reference = np.asarray(reference_times, dtype=float)
    comparison = np.asarray(comparison_times, dtype=float)
    if reference.ndim != 1 or comparison.ndim != 1:
        raise ValueError("Process consistency query times must be vectors.")
    indices: list[int] = []
    for value in reference:
        matches = np.flatnonzero(np.isclose(comparison, value, rtol=rtol, atol=atol))
        if matches.size != 1:
            raise ValueError(
                "Every reference time must occur exactly once in comparison_times."
            )
        indices.append(int(matches[0]))
    left = process.evaluate(realization, reference).states
    right_full = process.evaluate(realization, comparison).states
    time_axis = len(realization.sample_shape)
    right = jnp.take(right_full, jnp.asarray(indices, dtype=jnp.int32), axis=time_axis)
    difference = left - right
    max_error = jnp.max(jnp.abs(difference))
    rms_error = jnp.sqrt(jnp.mean(jnp.abs(difference) ** 2))
    return ProcessQueryDiagnostics(
        max_absolute_error=max_error,
        root_mean_square_error=rms_error,
        shared_times=len(indices),
        consistent=bool(jnp.allclose(left, right, rtol=rtol, atol=atol)),
        realization_id=realization.realization_id,
    )


def process_sample_statistics(
    distribution: AbstractProcessDistribution,
    samples: ArrayLike,
    /,
) -> ProcessSampleStatistics:
    """Summarize leading-axis samples without relabeling process uncertainty."""
    if not isinstance(distribution, AbstractProcessDistribution):
        raise TypeError("distribution must implement AbstractProcessDistribution.")
    values = jnp.asarray(samples)
    expected_tail = distribution.batch_shape + distribution.event_shape
    if values.ndim != len(expected_tail) + 1 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(
            "Process samples must have shape (sample,) + batch_shape + event_shape; "
            f"expected {('sample',) + expected_tail}, got {values.shape}."
        )
    count = int(values.shape[0])
    if count < 2:
        raise ValueError("Process statistics require at least two samples.")
    flat = values.reshape(
        (count,) + distribution.batch_shape + (distribution.event_size,)
    )
    mean = jnp.mean(flat, axis=0)
    centered = flat - mean
    covariance = jnp.einsum("n...i,n...j->...ij", centered, centered) / float(count - 1)
    log_probabilities = jax.vmap(distribution.log_prob)(values)
    return ProcessSampleStatistics(
        mean=mean.reshape(distribution.batch_shape + distribution.event_shape),
        covariance=covariance,
        finite_fraction=jnp.mean(jnp.isfinite(values)),
        average_log_prob=jnp.mean(log_probabilities),
        num_samples=count,
        uncertainty_source="process",
    )


def cocycle_objective(
    transition: AbstractPathwiseTransition,
    state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    tmid: ArrayLike,
    t1: ArrayLike,
    first_driver_segment: ArrayLike,
    second_driver_segment: ArrayLike,
    reduction: ProcessReduction = "mean",
) -> Array:
    """Penalize failure to compose pathwise transitions on the same driver."""
    first_state = transition.pathwise_transition(
        state,
        t0=t0,
        t1=tmid,
        driver_increment=first_driver_segment,
    )
    composed = transition.pathwise_transition(
        first_state,
        t0=tmid,
        t1=t1,
        driver_increment=second_driver_segment,
    )
    combined_driver = transition.combine_driver_segments(
        first_driver_segment,
        second_driver_segment,
    )
    direct = transition.pathwise_transition(
        state,
        t0=t0,
        t1=t1,
        driver_increment=combined_driver,
    )
    return _reduce(jnp.abs(composed - direct) ** 2, reduction)


def semigroup_objective(
    law: AbstractMarginalTransitionLaw,
    state: ArrayLike,
    /,
    *,
    t0: ArrayLike,
    tmid: ArrayLike,
    t1: ArrayLike,
    key: Key[Array, ""],
    num_samples: int = 256,
    observable: Callable[[Array], Array] | None = None,
    reduction: ProcessReduction = "mean",
) -> Array:
    """Monte Carlo Chapman--Kolmogorov objective for a marginal transition law.

    ``observable`` must preserve its leading sample axis. The identity observable is
    used by default, yielding a first-moment semigroup objective.
    """
    count = int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be positive.")
    _duration(t0, tmid)
    _duration(tmid, t1)
    direct_key, middle_key, continuation_key = jax.random.split(key, 3)
    direct = law.marginal_transition(state, t0=t0, t1=t1).sample(
        direct_key,
        (count,),
    )
    middle = law.marginal_transition(state, t0=t0, t1=tmid).sample(
        middle_key,
        (count,),
    )
    composed = law.marginal_transition(middle, t0=tmid, t1=t1).sample(continuation_key)
    evaluate = (lambda value: value) if observable is None else observable
    direct_observable = jnp.asarray(evaluate(direct))
    composed_observable = jnp.asarray(evaluate(composed))
    if direct_observable.shape != composed_observable.shape:
        raise ValueError("Observable outputs differ between direct and composed samples.")
    residual = jnp.mean(direct_observable, axis=0) - jnp.mean(
        composed_observable,
        axis=0,
    )
    return _reduce(jnp.abs(residual) ** 2, reduction)


def gaussian_process_diagnostics(
    process: LatentGaussianCoefficientProcess,
    realization: ProcessRealization,
    times: ArrayLike,
    /,
) -> GaussianProcessDiagnostics:
    """Validate replay, query invariance, cocycle, and terminal Gaussian moments."""
    query = jnp.asarray(times, dtype=process.drift.real.dtype)
    if query.ndim != 1 or int(query.shape[0]) < 3:
        raise ValueError(
            "Gaussian process diagnostics require at least three query times."
        )
    trajectory = process.evaluate(realization, query)
    replay = process.evaluate(realization, query)
    sample_count = realization.driver.num_paths
    if sample_count < 2:
        raise ValueError("Gaussian process diagnostics require at least two paths.")
    terminal = jnp.take(
        trajectory.states,
        int(query.shape[0]) - 1,
        axis=len(realization.sample_shape),
    ).reshape((sample_count,) + process.state_shape)
    marginal = process.marginal_transition(
        realization.initial_state,
        t0=realization.support[0],
        t1=query[-1],
    )
    statistics = process_sample_statistics(marginal, terminal)
    expected_mean = marginal.mean
    mean_denominator = jnp.maximum(jnp.linalg.vector_norm(expected_mean), 1e-12)
    covariance_denominator = jnp.maximum(
        jnp.linalg.matrix_norm(marginal.covariance), 1e-12
    )
    midpoint = 0.5 * (query[:-1] + query[1:])
    comparison = jnp.sort(jnp.concatenate((query, midpoint)))
    consistency = process_query_consistency(
        process,
        realization,
        query,
        comparison,
    )
    t0 = query[0]
    tmid = query[int(query.shape[0]) // 2]
    t1 = query[-1]
    first_driver = realization.driver_increment(t0, tmid)
    second_driver = realization.driver_increment(tmid, t1)
    first_state = process.pathwise_transition(
        realization.initial_state,
        t0=t0,
        t1=tmid,
        driver_increment=first_driver,
    )
    composed = process.pathwise_transition(
        first_state,
        t0=tmid,
        t1=t1,
        driver_increment=second_driver,
    )
    direct = process.pathwise_transition(
        realization.initial_state,
        t0=t0,
        t1=t1,
        driver_increment=process.combine_driver_segments(first_driver, second_driver),
    )
    return GaussianProcessDiagnostics(
        mean_relative_error=jnp.linalg.vector_norm(statistics.mean - expected_mean)
        / mean_denominator,
        covariance_relative_error=jnp.linalg.matrix_norm(
            statistics.covariance - marginal.covariance
        )
        / covariance_denominator,
        query_max_absolute_error=consistency.max_absolute_error,
        cocycle_max_absolute_error=jnp.max(jnp.abs(composed - direct)),
        replay_exact=bool(jnp.array_equal(trajectory.states, replay.states)),
        uncertainty_source="process",
        num_samples=sample_count,
        realization_id=realization.realization_id,
    )


__all__ = [
    "AbstractMarginalTransitionLaw",
    "AbstractPathwiseTransition",
    "AbstractProcessDistribution",
    "GaussianProcessDiagnostics",
    "GaussianProcessDistribution",
    "LatentGaussianCoefficientProcess",
    "ProcessQueryDiagnostics",
    "ProcessRealization",
    "ProcessReduction",
    "ProcessSampleStatistics",
    "cocycle_objective",
    "gaussian_process_diagnostics",
    "process_query_consistency",
    "process_sample_statistics",
    "semigroup_objective",
]

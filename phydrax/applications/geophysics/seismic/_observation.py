#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-clock seismic observation sampling and native differentiable likelihood."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._likelihoods import GaussianLikelihood
from ...._numerics._checkpointed_scan import CheckpointedScanMode, PreparedReplaySchedule
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....series import SampledSeries
from ....units import conversion_factor, PASCAL, SECOND, UnitDefinition
from ....uq._posterior_terms import AbstractPosteriorTerm
from ._acquisition import SeismicAcquisition
from ._constant_density import AcousticSimulation, ConstantDensityAcousticPlan


class BoundedAcousticWavespeed(StrictModule):
    """Trainable unconstrained field mapped into a positive, CFL-bounded interval."""

    raw: Array
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)

    def __init__(self, raw: ArrayLike, /, *, minimum: float, maximum: float):
        if not isfinite(minimum) or not isfinite(maximum) or not 0 < minimum < maximum:
            raise ValueError(
                "Acoustic wavespeed bounds must be finite with 0 < minimum < maximum."
            )
        value = jnp.asarray(raw, dtype=float)
        self.raw = eqx.error_if(
            value, jnp.any(~jnp.isfinite(value)), "Raw acoustic wavespeed must be finite."
        )
        self.minimum, self.maximum = float(minimum), float(maximum)

    @property
    def values(self) -> Array:
        return self.minimum + (self.maximum - self.minimum) * jax.nn.sigmoid(self.raw)


class PreparedSeismicObservation(StrictModule, NonTrainableState):
    """Fixed linear sampling of simulated pressure at actual observation times.

    Observations are receiver-major SampledSeries. Invalid nodes/components are
    excluded, not imputed; disconnected observed edges never create interpolation
    bridges. Each valid datum is compared independently to the connected simulated
    clock. Unknown time references, extrapolation, and uncalibrated count units
    must be resolved before preparation. No data-dependent runtime clock changes.
    """

    left: Array
    fraction: Array
    active: Array
    target: Array
    predicted_times: Array
    plan_id: str = eqx.field(static=True)
    acquisition_id: str = eqx.field(static=True)
    predicted_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        plan: ConstantDensityAcousticPlan,
        acquisition: SeismicAcquisition,
        observed: SampledSeries,
        /,
        *,
        amplitude_unit: UnitDefinition = PASCAL,
        time_unit: UnitDefinition = SECOND,
        time_reference: str = "simulation",
    ):
        if time_reference != "simulation":
            raise ValueError(
                "Seismic observation times must explicitly reference the simulation clock."
            )
        if observed.alignment != "node" or observed.support.series_shape != (
            acquisition.receivers.count,
        ):
            raise ValueError(
                "Seismic observations require receiver-major node-aligned SampledSeries."
            )
        if observed.values.shape != observed.sample_shape:
            raise ValueError(
                "Seismic observations require one scalar pressure per receiver/sample."
            )
        if acquisition.receivers.grid_id != plan.grid.grid_id:
            raise ValueError("Seismic observations and acoustic plan must share a grid.")
        times = np.asarray(observed.support.broadcast_coordinates()) * float(
            conversion_factor(time_unit, SECOND)
        )
        active = np.asarray(observed.sample_valid, dtype=bool)
        if not np.any(active):
            raise ValueError("Seismic observations require at least one valid sample.")
        clock = (
            np.where(active, times, plan.time_origin) - plan.time_origin
        ) / plan.time_step
        coordinate_dtype = np.asarray(observed.support.coordinates).dtype
        epsilon = (
            np.finfo(coordinate_dtype).eps
            if np.issubdtype(coordinate_dtype, np.floating)
            else np.finfo(float).eps
        )
        tolerance = 32 * epsilon * max(1, plan.step_count)
        if np.any(
            active & ((clock < -tolerance) | (clock > plan.step_count + tolerance))
        ):
            raise ValueError(
                "Seismic observations cannot extrapolate beyond the simulated clock."
            )
        clock = np.clip(clock, 0, plan.step_count)
        left = np.minimum(np.floor(clock).astype(np.int32), plan.step_count - 1)
        target = np.asarray(observed.values) * float(
            conversion_factor(amplitude_unit, PASCAL)
        )
        self.left, self.fraction = jnp.asarray(left), jnp.asarray(clock - left)
        self.active = jnp.asarray(active)
        self.target = jnp.asarray(np.where(active, target, 0.0))
        self.predicted_times = plan.time_origin + plan.time_step * jnp.arange(
            plan.step_count + 1
        )
        self.plan_id, self.acquisition_id = plan.plan_id, acquisition.acquisition_id
        self.predicted_shape = (acquisition.receivers.count, plan.step_count + 1)

    def sample(self, traces: ArrayLike, /) -> Array:
        values = jnp.asarray(traces)
        if values.shape != self.predicted_shape:
            raise ValueError(
                "Simulated seismic trace shape does not match the prepared observation."
            )
        lower = jnp.take_along_axis(values, self.left, axis=1)
        upper = jnp.take_along_axis(values, self.left + 1, axis=1)
        return jnp.where(self.active, lower + self.fraction * (upper - lower), 0.0)

    def apply(self, simulation: AcousticSimulation, /) -> Array:
        if (
            simulation.plan_id != self.plan_id
            or simulation.acquisition_id != self.acquisition_id
        ):
            raise ValueError(
                "Seismic simulation does not match the prepared observation provenance."
            )
        clock = simulation.traces.support.broadcast_coordinates()
        expected = jnp.broadcast_to(self.predicted_times, self.predicted_shape)
        values = eqx.error_if(
            simulation.traces.values,
            jnp.any(clock != expected),
            "Seismic simulation clock differs from the prepared observation.",
        )
        return self.sample(values)

    def transpose(self, cotangent: ArrayLike, /) -> Array:
        values = jnp.asarray(cotangent, dtype=float)
        if values.shape != self.target.shape:
            raise ValueError(
                "Seismic observation cotangent must match the observed trace shape."
            )
        return jax.linear_transpose(
            self.sample, jnp.zeros(self.predicted_shape, dtype=values.dtype)
        )(values)[0]


class SeismicGaussianLikelihood(AbstractPosteriorTerm):
    """Native UQ posterior term with differentiable acoustic forward propagation.

    The parameter is a scalar/grid wavespeed in m/s or BoundedAcousticWavespeed.
    ``source_rates`` are fixed physical volume rates at interval midpoints.
    Noise is independent per active scalar observation; each receiver is a case.
    Full-storage and replay modes execute the same primal and discrete adjoint.
    This likelihood does not infer source signatures, density, or unknown units.
    """

    plan: ConstantDensityAcousticPlan
    acquisition: SeismicAcquisition
    observation: PreparedSeismicObservation
    source_rates: Array
    likelihood: GaussianLikelihood
    replay: CheckpointedScanMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    schedule: PreparedReplaySchedule | None

    def __init__(
        self,
        plan: ConstantDensityAcousticPlan,
        acquisition: SeismicAcquisition,
        observed: SampledSeries,
        source_rates: ArrayLike,
        noise_scale: ArrayLike,
        /,
        *,
        amplitude_unit: UnitDefinition = PASCAL,
        time_unit: UnitDefinition = SECOND,
        time_reference: str = "simulation",
        replay: CheckpointedScanMode = "full",
        block_size: int | None = None,
        schedule: PreparedReplaySchedule | None = None,
        label: str = "seismic-pressure",
    ):
        if not label.strip():
            raise ValueError("Seismic likelihood label must be nonempty.")
        observation = PreparedSeismicObservation(
            plan,
            acquisition,
            observed,
            amplitude_unit=amplitude_unit,
            time_unit=time_unit,
            time_reference=time_reference,
        )
        rates = jnp.asarray(source_rates, dtype=float)
        if rates.shape != (plan.step_count, acquisition.sources.count):
            raise ValueError(
                "Seismic likelihood source history must match the acoustic plan."
            )
        rates = eqx.error_if(
            rates, jnp.any(~jnp.isfinite(rates)), "Seismic source rates must be finite."
        )
        scale = jnp.asarray(noise_scale, dtype=float) * float(
            conversion_factor(amplitude_unit, PASCAL)
        )
        if (
            jnp.broadcast_shapes(scale.shape, observation.target.shape)
            != observation.target.shape
        ):
            raise ValueError(
                "Seismic noise scale must broadcast to the observed trace shape."
            )
        self.plan, self.acquisition, self.observation = plan, acquisition, observation
        self.source_rates, self.likelihood = rates, GaussianLikelihood(scale)
        self.replay, self.block_size, self.schedule = replay, block_size, schedule
        self.label = label.strip()

    def predict(self, parameters: ArrayLike | BoundedAcousticWavespeed, /) -> Array:
        speed = (
            parameters.values
            if isinstance(parameters, BoundedAcousticWavespeed)
            else parameters
        )
        simulation = self.plan.simulate(
            speed,
            self.acquisition,
            self.source_rates,
            replay=self.replay,
            block_size=self.block_size,
            schedule=self.schedule,
        )
        return self.observation.apply(simulation)

    def residual(self, parameters: ArrayLike | BoundedAcousticWavespeed, /) -> Array:
        return jnp.where(
            self.observation.active,
            self.predict(parameters) - self.observation.target,
            0.0,
        )

    def per_case_log_prob(
        self, parameters: ArrayLike | BoundedAcousticWavespeed, /
    ) -> Array:
        values = self.likelihood.log_prob(
            self.predict(parameters), self.observation.target
        )
        return jnp.sum(jnp.where(self.observation.active, values, 0.0), axis=-1)


__all__ = [
    "BoundedAcousticWavespeed",
    "PreparedSeismicObservation",
    "SeismicGaussianLikelihood",
]

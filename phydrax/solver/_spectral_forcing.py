#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..discretization.finite_volume import PreparedFiniteVolumeDynamics
from ..equations import CompressibleNavierStokesSystem, EulerSystem
from ..stochastic import OrnsteinUhlenbeckRealization
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._finite_volume_runtime import FiniteVolumeRuntimeState, PreparedFiniteVolumeRuntime


class SpectralOUForcingDiagnostics(eqx.Module):
    acceleration: Array
    rms_acceleration: Array
    mean_acceleration: Array
    solenoidal_power_fraction: Array
    band_power: Array
    successful: Array


class SpectralOUForcingPlan(AbstractBalanceLawProcessPlan):
    kmin: float = eqx.field(static=True)
    kmax: float = eqx.field(static=True)
    spectral_slope: float = eqx.field(static=True)
    solenoidal_fraction: float = eqx.field(static=True)
    correlation_time: float = eqx.field(static=True)
    rms_acceleration: float = eqx.field(static=True)
    correlation_argument: str | None = eqx.field(static=True)
    rms_argument: str | None = eqx.field(static=True)
    realization_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kmin: float = 1.0,
        kmax: float = 3.0,
        spectral_slope: float = 0.0,
        solenoidal_fraction: float = 1.0,
        correlation_time: float = 0.5,
        rms_acceleration: float = 1.0,
        correlation_argument: str | None = None,
        rms_argument: str | None = None,
        realization_name: str = "spectral_ou",
    ):
        values = tuple(
            float(value)
            for value in (
                kmin,
                kmax,
                spectral_slope,
                solenoidal_fraction,
                correlation_time,
                rms_acceleration,
            )
        )
        lower, upper, slope, fraction, correlation, rms = values
        correlation_name = (
            None if correlation_argument is None else str(correlation_argument)
        )
        rms_name = None if rms_argument is None else str(rms_argument)
        realization_name_ = str(realization_name)
        if (
            any(not np.isfinite(value) for value in values)
            or lower <= 0.0
            or upper < lower
            or not 0.0 <= fraction <= 1.0
            or correlation <= 0.0
            or rms <= 0.0
            or (correlation_name is not None and not correlation_name)
            or (rms_name is not None and not rms_name)
            or not realization_name_
        ):
            raise ValueError("Spectral OU forcing parameters are invalid.")
        self.kmin = lower
        self.kmax = upper
        self.spectral_slope = slope
        self.solenoidal_fraction = fraction
        self.correlation_time = correlation
        self.rms_acceleration = rms
        self.correlation_argument = correlation_name
        self.rms_argument = rms_name
        self.realization_name = realization_name_
        self.process_id = canonical_fingerprint(
            {
                "kind": "spectral-ou-forcing",
                "band": [lower, upper],
                "slope": slope,
                "solenoidal_fraction": fraction,
                "correlation_time": correlation,
                "rms_acceleration": rms,
                "correlation_argument": correlation_name,
                "rms_argument": rms_name,
                "realization_name": realization_name_,
            }
        )

    def prepare(
        self, runtime: PreparedFiniteVolumeRuntime, /
    ) -> PreparedSpectralOUForcing:
        return PreparedSpectralOUForcing(self, runtime)


class PreparedSpectralOUForcing(AbstractPreparedBalanceLawProcess):
    plan: SpectralOUForcingPlan
    runtime: PreparedFiniteVolumeRuntime
    dynamics: PreparedFiniteVolumeDynamics
    wavevectors: Array
    spectral_weight: Array
    nonzero: Array
    normalization: Array
    density_index: int = eqx.field(static=True)
    momentum_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralOUForcingPlan,
        runtime: PreparedFiniteVolumeRuntime,
        /,
    ):
        if not isinstance(plan, SpectralOUForcingPlan):
            raise TypeError("plan must be SpectralOUForcingPlan.")
        if not isinstance(runtime, PreparedFiniteVolumeRuntime) or not isinstance(
            runtime.dynamics, PreparedFiniteVolumeDynamics
        ):
            raise TypeError("Spectral OU forcing requires stationary structured FV.")
        dynamics = runtime.dynamics
        if not isinstance(dynamics.system, (EulerSystem, CompressibleNavierStokesSystem)):
            raise TypeError("Spectral OU forcing requires Euler/Navier-Stokes.")
        dimension = dynamics.system.dimension
        if dimension not in (2, 3):
            raise ValueError("Spectral OU forcing requires two or three dimensions.")
        grid = dynamics.discretization.grid
        if any(not axis.periodic for axis in grid.structured_axes):
            raise ValueError("Spectral OU forcing requires a fully periodic grid.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in grid.structured_axes)
        if any(
            not np.allclose(width, width[0], rtol=1e-12, atol=1e-12) for width in widths
        ):
            raise ValueError("Spectral OU forcing requires uniform grid spacing.")
        physical_modes = []
        integer_modes = []
        for count, width in zip(grid.shape, widths, strict=True):
            physical_modes.append(
                2.0 * jnp.pi * jnp.fft.fftfreq(count, d=float(width[0]))
            )
            integer_modes.append(jnp.fft.fftfreq(count) * count)
        physical_mesh = jnp.meshgrid(*physical_modes, indexing="ij")
        integer_mesh = jnp.meshgrid(*integer_modes, indexing="ij")
        wavevectors = jnp.stack(physical_mesh, axis=-1)
        integer_radius = jnp.sqrt(sum(component**2 for component in integer_mesh))
        squared = jnp.sum(wavevectors**2, axis=-1)
        nonzero = squared > 0.0
        band = (integer_radius >= plan.kmin) & (integer_radius <= plan.kmax) & nonzero
        safe_radius = jnp.where(nonzero, integer_radius, 1.0)
        weight = jnp.where(band, safe_radius ** (0.5 * plan.spectral_slope), 0.0)
        projector_trace = plan.solenoidal_fraction * (dimension - 1) + (
            1.0 - plan.solenoidal_fraction
        )
        vector_variance = jnp.mean(weight**2) * projector_trace
        normalization = 1.0 / jnp.sqrt(vector_variance)
        names = tuple(dynamics.system.component_names)
        self.plan = plan
        self.runtime = runtime
        self.dynamics = dynamics
        self.wavevectors = wavevectors
        self.spectral_weight = weight
        self.nonzero = nonzero
        self.normalization = normalization
        self.density_index = names.index("density")
        self.momentum_indices = tuple(
            names.index(f"momentum_{axis}") for axis in range(dimension)
        )
        self.energy_index = names.index("total_energy")
        self.cell_shape = tuple(grid.shape)
        self.dimension = dimension
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-ou-forcing",
                "plan": plan.process_id,
                "runtime": runtime.runtime_id,
                "cell_shape": list(grid.shape),
            }
        )
        self.requires_realization = True
        self.realization_name = plan.realization_name
        self.differentiability = "smooth_discrete"

    def initialize(
        self, transport_state: FiniteVolumeRuntimeState, args: Any = None, /
    ) -> BalanceLawProcessState:
        del transport_state, args
        real_dtype = jnp.dtype(self.runtime.precision.storage_dtype)
        complex_dtype = jnp.complex64 if real_dtype.itemsize == 4 else jnp.complex128
        coefficients = jnp.zeros(self.cell_shape + (self.dimension,), dtype=complex_dtype)
        return BalanceLawProcessState(
            self.process_id, ("spectral_acceleration",), (coefficients,)
        )

    def _parameter(self, fixed: float, argument: str | None, args: Any, dtype, name: str):
        raw = fixed if argument is None else args[argument]
        value = jnp.asarray(raw, dtype=dtype).reshape(())
        return eqx.error_if(
            value,
            ~jnp.isfinite(value) | (value <= 0.0),
            f"{name} must be positive and finite.",
        )

    def _field(self, cell_average: Array, /) -> Array:
        components = len(self.dynamics.system.component_names)
        expected = (int(np.prod(self.cell_shape)), components)
        value = jnp.asarray(cell_average)
        if value.shape != expected:
            raise ValueError(f"OU forcing cell_average must have shape {expected}.")
        return value.reshape(self.cell_shape + (components,))

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, cell_average, process_state, args
        return jnp.asarray(jnp.inf)

    def _project(self, coefficients: Array, /) -> tuple[Array, Array]:
        squared = jnp.sum(self.wavevectors**2, axis=-1)
        dot = jnp.sum(self.wavevectors * coefficients, axis=-1)
        parallel = (
            self.wavevectors * (dot / jnp.where(self.nonzero, squared, 1.0))[..., None]
        )
        transverse = coefficients - parallel
        fraction = self.plan.solenoidal_fraction
        transverse_part = (
            self.spectral_weight[..., None] * jnp.sqrt(fraction) * transverse
        )
        parallel_part = (
            self.spectral_weight[..., None] * jnp.sqrt(1.0 - fraction) * parallel
        )
        projected = transverse_part + parallel_part
        projected = jnp.where(self.nonzero[..., None], projected, 0.0)
        transverse_power = jnp.sum(jnp.abs(transverse_part) ** 2)
        total_power = transverse_power + jnp.sum(jnp.abs(parallel_part) ** 2)
        return projected, transverse_power / jnp.maximum(total_power, 1e-30)

    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        if process_state.process_id != self.process_id or process_state.field_names != (
            "spectral_acceleration",
        ):
            raise ValueError("Spectral OU process state changed.")
        if not isinstance(realization, OrnsteinUhlenbeckRealization):
            raise TypeError("Spectral OU forcing requires OrnsteinUhlenbeckRealization.")
        if realization.sample_shape or realization.noise_shape != self.cell_shape + (
            self.dimension,
        ):
            raise ValueError("OU realization shape does not match forcing grid.")
        step = jnp.asarray(end_time - start_time)
        correlation = self._parameter(
            self.plan.correlation_time,
            self.plan.correlation_argument,
            args,
            step.dtype,
            "OU correlation time",
        )
        target_rms = self._parameter(
            self.plan.rms_acceleration,
            self.plan.rms_argument,
            args,
            step.dtype,
            "OU RMS acceleration",
        )
        innovation = realization.innovations(
            start_time, end_time, correlation, dtype=step.dtype
        )
        axes = tuple(range(len(self.cell_shape)))
        innovation_hat = jnp.fft.fftn(innovation, axes=axes)
        projected_hat, solenoidal_fraction = self._project(innovation_hat)
        previous = process_state.field("spectral_acceleration")
        decay = jnp.exp(-step / correlation)
        coefficients = decay * previous + projected_hat
        acceleration = jnp.fft.ifftn(coefficients, axes=axes).real
        acceleration = self.normalization * target_rms * acceleration
        field = self._field(cell_average)
        density = field[..., self.density_index]
        momentum = field[..., self.momentum_indices]
        momentum_new = momentum + density[..., None] * acceleration * step
        kinetic_before = 0.5 * jnp.sum(momentum**2, axis=-1) / density
        kinetic_after = 0.5 * jnp.sum(momentum_new**2, axis=-1) / density
        candidate = field.at[..., self.momentum_indices].set(momentum_new)
        candidate = candidate.at[..., self.energy_index].add(
            kinetic_after - kinetic_before
        )
        successful = jnp.all(jnp.isfinite(candidate)) & jnp.all(
            self.dynamics.system.admissible(candidate)
        )
        accepted = jnp.where(successful, candidate, field)
        accepted_coefficients = jnp.where(successful, coefficients, previous)
        next_state = BalanceLawProcessState(
            self.process_id,
            ("spectral_acceleration",),
            (accepted_coefficients,),
        )
        spatial_axes = tuple(range(len(self.cell_shape)))
        diagnostics = SpectralOUForcingDiagnostics(
            acceleration=acceleration,
            rms_acceleration=jnp.sqrt(jnp.mean(jnp.sum(acceleration**2, axis=-1))),
            mean_acceleration=jnp.mean(acceleration, axis=spatial_axes),
            solenoidal_power_fraction=solenoidal_fraction,
            band_power=jnp.sum(jnp.abs(coefficients) ** 2),
            successful=successful,
        )
        incoming = field.reshape(cell_average.shape)
        accepted_flat = accepted.reshape(cell_average.shape)
        return BalanceLawProcessAdvance(
            cell_average=accepted_flat,
            process_state=next_state,
            successful=successful,
            source_change=accepted_flat - incoming,
            diagnostics=diagnostics,
        )


__all__ = [
    "PreparedSpectralOUForcing",
    "SpectralOUForcingDiagnostics",
    "SpectralOUForcingPlan",
]

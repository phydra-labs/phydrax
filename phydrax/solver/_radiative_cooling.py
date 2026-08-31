#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import PreparedFiniteVolumeDynamics
from ..equations import EulerSystem, IdealMHDSystem, TabulatedCoolingCurve
from ..nonlinear import (
    implicit_root_result,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class RadiativeCoolingDiagnostics(StrictModule):
    temperature_before: Array
    temperature_after: Array
    energy_change: Array
    maximum_residual: Array
    maximum_iterations: Array
    supported: Array
    successful: Array


class RadiativeCoolingProcessPlan(AbstractBalanceLawProcessPlan):
    curve: TabulatedCoolingCurve
    amplitude: float = eqx.field(static=True)
    amplitude_argument: str | None = eqx.field(static=True)
    heating_rate: float = eqx.field(static=True)
    heating_argument: str | None = eqx.field(static=True)
    integration: Literal["implicit", "exact"] = eqx.field(static=True)
    accuracy_fraction: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        curve: TabulatedCoolingCurve,
        /,
        *,
        amplitude: float = 1.0,
        amplitude_argument: str | None = None,
        heating_rate: float = 0.0,
        heating_argument: str | None = None,
        integration: Literal["implicit", "exact"] = "implicit",
        accuracy_fraction: float = 0.1,
        maximum_iterations: int = 20,
        tolerance: float = 1e-9,
    ):
        if not isinstance(curve, TabulatedCoolingCurve):
            raise TypeError("curve must be TabulatedCoolingCurve.")
        amplitude_ = float(amplitude)
        heating = float(heating_rate)
        heating_name = None if heating_argument is None else str(heating_argument)
        fraction = float(accuracy_fraction)
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        argument = None if amplitude_argument is None else str(amplitude_argument)
        if (
            not np.isfinite(amplitude_)
            or amplitude_ <= 0.0
            or not np.isfinite(heating)
            or heating < 0.0
            or not np.isfinite(fraction)
            or not 0.0 < fraction <= 1.0
            or iterations <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or (argument is not None and not argument)
            or (heating_name is not None and not heating_name)
            or integration not in ("implicit", "exact")
            or (integration == "exact" and (heating > 0.0 or heating_name is not None))
        ):
            raise ValueError("Radiative cooling process parameters are invalid.")
        self.curve = curve
        self.amplitude = amplitude_
        self.amplitude_argument = argument
        self.heating_rate = heating
        self.heating_argument = heating_name
        self.integration = integration
        self.accuracy_fraction = fraction
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.process_id = canonical_fingerprint(
            {
                "kind": "radiative-cooling-process",
                "curve": curve.curve_id,
                "amplitude": amplitude_,
                "amplitude_argument": argument,
                "heating_rate": heating,
                "heating_argument": heating_name,
                "integration": integration,
                "accuracy_fraction": fraction,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedRadiativeCoolingProcess:
        return PreparedRadiativeCoolingProcess(self, transport)


class _CoolingResidual(StrictModule, NonTrainableState):
    system: EulerSystem | IdealMHDSystem
    curve: TabulatedCoolingCurve

    def __call__(self, log_excess: Array, args: Any, /) -> Array:
        density, initial_internal, step, amplitude, heating, floor = args
        internal = floor + jnp.exp(log_excess)
        pressure = (self.system.gamma - 1.0) * internal
        temperature = self.system.material.temperature(density, pressure)
        evaluated = self.curve.evaluate(temperature)
        cooling = amplitude * density**2 * evaluated.rate
        residual = internal - initial_internal + step * (cooling - heating)
        return jnp.where(evaluated.supported, residual, jnp.asarray(jnp.nan))


class PreparedRadiativeCoolingProcess(AbstractPreparedBalanceLawProcess):
    plan: RadiativeCoolingProcessPlan
    transport: AbstractPreparedBalanceLawTransport
    dynamics: PreparedFiniteVolumeDynamics
    problem: NonlinearSystemProblem
    termination: NonlinearTermination
    density_index: int = eqx.field(static=True)
    momentum_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)
    magnetic_indices: tuple[int, ...] = eqx.field(static=True)
    cell_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        plan: RadiativeCoolingProcessPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        if not isinstance(plan, RadiativeCoolingProcessPlan):
            raise TypeError("plan must be RadiativeCoolingProcessPlan.")
        if not isinstance(
            transport, AbstractPreparedBalanceLawTransport
        ) or not isinstance(transport.dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError(
                "Radiative cooling requires stationary structured FV dynamics."
            )
        dynamics = transport.dynamics
        if not isinstance(dynamics.system, (EulerSystem, IdealMHDSystem)):
            raise TypeError("Radiative cooling requires ideal-gas Euler or MHD.")
        residual = _CoolingResidual(dynamics.system, plan.curve)
        problem_id = canonical_fingerprint(
            {
                "kind": "radiative-cooling-implicit-root",
                "system": dynamics.system.system_id,
                "curve": plan.curve.curve_id,
            }
        )
        problem = NonlinearSystemProblem(residual, problem_id=problem_id)
        termination = NonlinearTermination(
            absolute_residual=plan.tolerance,
            relative_residual=plan.tolerance,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=plan.maximum_iterations,
        )
        names = tuple(dynamics.system.component_names)
        self.plan = plan
        self.transport = transport
        self.dynamics = dynamics
        self.problem = problem
        self.termination = termination
        self.density_index = names.index("density")
        self.momentum_indices = tuple(
            index for index, name in enumerate(names) if name.startswith("momentum_")
        )
        if len(self.momentum_indices) != dynamics.system.dimension:
            raise RuntimeError("Cooling system momentum layout is inconsistent.")
        self.energy_index = names.index("total_energy")
        self.magnetic_indices = tuple(
            names.index(f"magnetic_{axis}")
            for axis in "xyz"
            if f"magnetic_{axis}" in names
        )
        self.cell_shape = tuple(dynamics.discretization.cell_shape)
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-radiative-cooling",
                "plan": plan.process_id,
                "transport": transport.transport_id,
                "root_problem": problem_id,
            }
        )
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = (
            "branchwise-exact" if plan.integration == "exact" else "branchwise-implicit"
        )
        self.modified_components = ("total_energy",)

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        return BalanceLawProcessState.empty(self.process_id)

    def _field(self, cell_average: Array, /) -> Array:
        components = len(self.dynamics.system.component_names)
        expected = (int(np.prod(self.cell_shape)), components)
        value = jnp.asarray(cell_average)
        if value.shape != expected:
            raise ValueError(f"Cooling cell_average must have shape {expected}.")
        return value.reshape(self.cell_shape + (components,))

    def _amplitude(self, args: Any, dtype, /) -> Array:
        raw = (
            self.plan.amplitude
            if self.plan.amplitude_argument is None
            else args[self.plan.amplitude_argument]
        )
        amplitude = jnp.asarray(raw, dtype=dtype).reshape(())
        return eqx.error_if(
            amplitude,
            ~jnp.isfinite(amplitude) | (amplitude <= 0.0),
            "Cooling amplitude must be positive and finite.",
        )

    def _heating(self, args: Any, dtype, /) -> Array:
        raw = (
            self.plan.heating_rate
            if self.plan.heating_argument is None
            else args[self.plan.heating_argument]
        )
        heating = jnp.asarray(raw, dtype=dtype).reshape(())
        return eqx.error_if(
            heating,
            ~jnp.isfinite(heating) | (heating < 0.0),
            "Heating rate must be nonnegative and finite.",
        )

    def _thermodynamics(self, field: Array, /) -> tuple[Array, Array, Array, Array]:
        density = field[..., self.density_index]
        momentum = field[..., self.momentum_indices]
        kinetic = 0.5 * jnp.sum(momentum**2, axis=-1) / density
        magnetic = (
            0.5 * jnp.sum(field[..., self.magnetic_indices] ** 2, axis=-1)
            if self.magnetic_indices
            else jnp.zeros_like(kinetic)
        )
        nonthermal = kinetic + magnetic
        internal = field[..., self.energy_index] - nonthermal
        pressure = (self.dynamics.system.gamma - 1.0) * internal
        temperature = self.dynamics.system.material.temperature(density, pressure)
        return density, nonthermal, internal, temperature

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, process_state
        field = self._field(cell_average)
        density, _, internal, temperature = self._thermodynamics(field)
        amplitude = self._amplitude(args, field.dtype)
        evaluated = self.plan.curve.evaluate(temperature)
        heating = self._heating(args, field.dtype)
        rate = jnp.abs(amplitude * density**2 * evaluated.rate - heating)
        local = jnp.where(rate > 0.0, internal / rate, jnp.inf)
        supported = jnp.all(evaluated.supported)
        return jnp.where(
            supported,
            self.plan.accuracy_fraction * jnp.min(local),
            jnp.asarray(jnp.nan, dtype=field.dtype),
        )

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
        del realization
        if process_state.process_id != self.process_id or process_state.values:
            raise ValueError("Radiative cooling process state changed.")
        step = jnp.asarray(end_time - start_time)
        field = self._field(cell_average)
        density, nonthermal, internal, temperature_before = self._thermodynamics(field)
        amplitude = self._amplitude(args, field.dtype)
        heating = self._heating(args, field.dtype)
        floor = jnp.asarray(
            self.dynamics.system.pressure_floor / (self.dynamics.system.gamma - 1.0),
            dtype=field.dtype,
        )
        if self.plan.integration == "exact":
            coordinate = self.plan.curve.cooling_coordinate(temperature_before)
            temperature_per_internal = temperature_before / internal
            target = coordinate - step * amplitude * density**2 * temperature_per_internal
            temperature_after = self.plan.curve.temperature_from_cooling_coordinate(
                target
            )
            internal_new = jnp.maximum(
                floor,
                internal * temperature_after / temperature_before,
            )
            cell_success = self.plan.curve.evaluate(temperature_before).supported
            residual = jnp.zeros_like(internal)
            iterations = jnp.zeros_like(internal, dtype=jnp.int32)
        else:
            excess = jnp.maximum(internal - floor, jnp.finfo(field.dtype).tiny)
            initial_log = jnp.log(excess)
            flat_log = initial_log.reshape((-1,))
            flat_density = density.reshape((-1,))
            flat_internal = internal.reshape((-1,))

            def solve_cell(log_value, density_value, internal_value):
                result = implicit_root_result(
                    self.problem,
                    log_value,
                    termination=self.termination,
                    args=(
                        density_value,
                        internal_value,
                        step,
                        amplitude,
                        heating,
                        floor,
                    ),
                )
                return (
                    result.state,
                    result.successful,
                    result.diagnostics.final_residual_norm,
                    result.diagnostics.iterations,
                )

            solved_log, cell_success, residual, iterations = jax.vmap(solve_cell)(
                flat_log, flat_density, flat_internal
            )
            internal_new = (floor + jnp.exp(solved_log)).reshape(self.cell_shape)
        energy_new = nonthermal + internal_new
        candidate = field.at[..., self.energy_index].set(energy_new)
        pressure_new = (self.dynamics.system.gamma - 1.0) * internal_new
        temperature_after = self.dynamics.system.material.temperature(
            density, pressure_new
        )
        supported = self.plan.curve.evaluate(temperature_after).supported
        successful = (
            jnp.all(cell_success)
            & jnp.all(supported)
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(self.dynamics.system.admissible(candidate))
        )
        accepted = jnp.where(successful, candidate, field)
        diagnostics = RadiativeCoolingDiagnostics(
            temperature_before=temperature_before,
            temperature_after=temperature_after,
            energy_change=accepted[..., self.energy_index]
            - field[..., self.energy_index],
            maximum_residual=jnp.max(residual),
            maximum_iterations=jnp.max(iterations),
            supported=jnp.all(supported),
            successful=successful,
        )
        incoming = field.reshape(cell_average.shape)
        accepted_flat = accepted.reshape(cell_average.shape)
        return BalanceLawProcessAdvance(
            cell_average=accepted_flat,
            process_state=process_state,
            successful=successful,
            source_change=accepted_flat - incoming,
            diagnostics=diagnostics,
        )


__all__ = [
    "PreparedRadiativeCoolingProcess",
    "RadiativeCoolingDiagnostics",
    "RadiativeCoolingProcessPlan",
]

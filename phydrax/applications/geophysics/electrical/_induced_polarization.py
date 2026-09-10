#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry.simplicial import AffineSimplexMap
from ._finite_patch import FinitePatchDCPlan


class ColeColeConductivity(StrictModule):
    """Passive Cole-Cole conductivity for the exp(-i omega t) convention.

    ``chargeability`` is (sigma_infinity-sigma_dc)/sigma_infinity. The model
    evaluates sigma_dc + (sigma_infinity-sigma_dc) x/(1+x), where
    x=(-i omega tau)^exponent. Positive frequencies therefore have nonnegative
    real conductivity and nonpositive imaginary conductivity.
    """

    dc_conductivity_S_m: Array
    chargeability: Array
    time_constant_s: Array
    exponent: Array

    def __init__(
        self,
        dc_conductivity_S_m: ArrayLike,
        chargeability: ArrayLike,
        time_constant_s: ArrayLike,
        exponent: ArrayLike,
        /,
    ):
        dc, chargeability_, time, exponent_ = jnp.broadcast_arrays(
            jnp.asarray(dc_conductivity_S_m),
            jnp.asarray(chargeability),
            jnp.asarray(time_constant_s),
            jnp.asarray(exponent),
        )
        invalid = (
            jnp.any(~jnp.isfinite(dc))
            | jnp.any(dc <= 0)
            | jnp.any(~jnp.isfinite(chargeability_))
            | jnp.any((chargeability_ < 0) | (chargeability_ >= 1))
            | jnp.any(~jnp.isfinite(time))
            | jnp.any(time <= 0)
            | jnp.any(~jnp.isfinite(exponent_))
            | jnp.any((exponent_ <= 0) | (exponent_ > 1))
        )
        self.dc_conductivity_S_m = eqx.error_if(
            dc, invalid, "Cole-Cole parameters are outside their passive physical domain."
        )
        self.chargeability = chargeability_
        self.time_constant_s = time
        self.exponent = exponent_

    def conductivity(self, angular_frequency: ArrayLike, /) -> Array:
        omega = jnp.asarray(angular_frequency)
        omega = eqx.error_if(
            omega,
            jnp.any(~jnp.isfinite(omega)) | jnp.any(omega < 0),
            "Cole-Cole angular frequency must be finite and nonnegative.",
        )
        x = (
            -1j * omega[..., None] * self.time_constant_s.reshape(-1)
        ) ** self.exponent.reshape(-1)
        dc = self.dc_conductivity_S_m.reshape(-1)
        chargeability = self.chargeability.reshape(-1)
        infinity = dc / (1.0 - chargeability)
        conductivity = dc + (infinity - dc) * x / (1.0 + x)
        return conductivity.reshape(omega.shape + self.dc_conductivity_S_m.shape)


class DebyeSpectrumConductivity(StrictModule):
    dc_conductivity_S_m: Array
    increments_S_m: Array
    time_constants_s: Array

    def __init__(
        self,
        dc_conductivity_S_m: ArrayLike,
        increments_S_m: ArrayLike,
        time_constants_s: ArrayLike,
        /,
    ):
        dc = jnp.asarray(dc_conductivity_S_m)
        increments = jnp.asarray(increments_S_m)
        times = jnp.asarray(time_constants_s)
        if increments.ndim < 1 or times.ndim != 1 or increments.shape[-1] != times.size:
            raise ValueError("Debye increments need one trailing relaxation axis.")
        dc = jnp.broadcast_to(dc, increments.shape[:-1])
        invalid = (
            jnp.any(~jnp.isfinite(dc))
            | jnp.any(dc <= 0)
            | jnp.any(~jnp.isfinite(increments))
            | jnp.any(increments < 0)
            | jnp.any(~jnp.isfinite(times))
            | jnp.any(times <= 0)
        )
        self.dc_conductivity_S_m = eqx.error_if(
            dc,
            invalid,
            "Debye conductivity parameters must be finite, passive, and positive.",
        )
        self.increments_S_m, self.time_constants_s = increments, times

    def conductivity(self, angular_frequency: ArrayLike, /) -> Array:
        omega = jnp.asarray(angular_frequency)
        flat = omega.reshape(-1)
        x = -1j * flat[:, None] * self.time_constants_s
        relaxation = x / (1.0 + x)
        increment = ein.contract("fr,...r->f...", relaxation, self.increments_S_m)
        result = self.dc_conductivity_S_m[None, ...] + increment
        return result.reshape(omega.shape + self.dc_conductivity_S_m.shape)

    def initial_memory(self, field_shape: tuple[int, ...]) -> Array:
        return jnp.zeros(field_shape + (self.time_constants_s.size,))

    def advance_memory(
        self,
        memory: ArrayLike,
        electric_field: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        field, state, dt = (
            jnp.asarray(electric_field),
            jnp.asarray(memory),
            jnp.asarray(dt_s),
        )
        expected = field.shape + (self.time_constants_s.size,)
        if state.shape != expected or dt.shape != ():
            raise ValueError("Debye memory shape or timestep is invalid.")
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0), "Debye timestep must be positive."
        )
        decay = jnp.exp(-dt / self.time_constants_s)
        next_state = decay * state + (1.0 - decay) * field[..., None]
        current = self.dc_conductivity_S_m * field + jnp.sum(
            self.increments_S_m * (field[..., None] - next_state), axis=-1
        )
        return next_state, current


class SpectralIPResult(StrictModule):
    angular_frequencies: Array
    voltages: Array
    residual_norms: Array
    dissipated_power_W: Array
    successful: Array


class SpectralIPPlan(StrictModule, NonTrainableState):
    finite_patch: FinitePatchDCPlan
    cells: Array
    gradients: Array
    volumes: Array
    gauge: Array
    space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, finite_patch: FinitePatchDCPlan, /):
        if not isinstance(finite_patch, FinitePatchDCPlan):
            raise TypeError("Spectral IP requires FinitePatchDCPlan geometry and survey.")
        cells = np.concatenate(
            [
                np.asarray(block.vertices, dtype=np.int32)
                for block in finite_patch.mesh.blocks
            ]
        )
        coordinates = np.asarray(finite_patch.mesh.coordinates, dtype=float)
        simplex = AffineSimplexMap(jnp.asarray(coordinates[cells]))
        if not bool(jnp.all(simplex.evidence.successful)):
            raise ValueError("Induced-polarization mesh contains degenerate tetrahedra.")
        gradients = simplex.barycentric_gradients
        gauge = np.ones(coordinates.shape[0])
        gauge /= np.sqrt(gauge.size)
        self.finite_patch = finite_patch
        self.cells, self.gradients = jnp.asarray(cells), gradients
        self.volumes = simplex.evidence.measure
        self.gauge = jnp.asarray(gauge)
        self.space = la.ArraySpace((gauge.size + 1,), dtype=jnp.complex128)
        self.policy = la.LinearSolvePolicy(
            la.GMRES(restart=40, stagnation_iterations=40),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-11, max_steps=1000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {"kind": "spectral-ip-plan", "finite_patch": finite_patch.plan_id}
        )

    def _operator(self, conductivity: Array):
        cell_count = self.cells.shape[0]
        values = jnp.broadcast_to(jnp.asarray(conductivity), (cell_count,))
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values))
            | jnp.any(jnp.real(values) <= 0)
            | jnp.any(jnp.imag(values) > 1e-12),
            "Spectral IP conductivity must be finite and passive for exp(-i omega t).",
        )
        local = (
            self.volumes[:, None, None]
            * values[:, None, None]
            * ein.contract("cvi,cwi->cvw", self.gradients, self.gradients)
        )
        nodes = self.gauge.size

        def action(unknown):
            potential, multiplier = unknown[:nodes], unknown[-1]
            cell_values = potential[self.cells]
            residual = ein.contract("cij,cj->ci", local, cell_values)
            assembled = jnp.zeros_like(potential).at[self.cells].add(residual)
            return jnp.concatenate(
                (
                    assembled + multiplier * self.gauge,
                    jnp.vdot(self.gauge, potential)[None],
                )
            )

        return la.FunctionLinearOperator(action, source=self.space, target=self.space)

    def solve(
        self,
        angular_frequencies: ArrayLike,
        model: ColeColeConductivity | DebyeSpectrumConductivity,
        /,
    ) -> SpectralIPResult:
        frequencies = jnp.asarray(angular_frequencies)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("Spectral IP frequencies must be a nonempty vector.")
        frequencies = eqx.error_if(
            frequencies,
            jnp.any(~jnp.isfinite(frequencies)) | jnp.any(frequencies <= 0),
            "Spectral IP frequencies must be finite and positive.",
        )
        if not isinstance(model, (ColeColeConductivity, DebyeSpectrumConductivity)):
            raise TypeError("Spectral IP model must be Cole-Cole or Debye spectrum.")
        conductivity = model.conductivity(frequencies)
        base = self.finite_patch.prepare()
        voltage_rows, residual_rows, power_rows, successful_rows = [], [], [], []
        for frequency, cell_conductivity in zip(frequencies, conductivity, strict=True):
            del frequency
            operator = self._operator(cell_conductivity)
            electrode_fields, residuals, powers, successes = [], [], [], []
            for current in self.finite_patch.survey.currents:
                load = base.current_load(current).astype(jnp.complex128)
                rhs = jnp.concatenate((load, jnp.zeros(1, dtype=load.dtype)))
                result = la.solve(la.LinearSystem(operator), rhs, policy=self.policy)
                field = result.value[:-1]
                residual = operator.mv(result.value) - rhs
                electrode_fields.append(base.electrode_potentials(field))
                residuals.append(jnp.sqrt(jnp.real(jnp.vdot(residual, residual))))
                powers.append(0.5 * jnp.real(jnp.vdot(load, field)))
                successes.append(result.successful)
            electrodes = jnp.stack(electrode_fields)
            voltage_rows.append(
                ein.contract(
                    "me,me->m",
                    self.finite_patch.survey.receiver_weights,
                    electrodes[self.finite_patch.survey.source_indices],
                )
            )
            residual_rows.append(jnp.stack(residuals))
            power_rows.append(jnp.stack(powers))
            successful_rows.append(jnp.all(jnp.stack(successes)))
        return SpectralIPResult(
            frequencies,
            jnp.stack(voltage_rows),
            jnp.stack(residual_rows),
            jnp.stack(power_rows),
            jnp.all(jnp.stack(successful_rows)),
        )


__all__ = [
    "ColeColeConductivity",
    "DebyeSpectrumConductivity",
    "SpectralIPPlan",
    "SpectralIPResult",
]

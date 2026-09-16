#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.lattice_boltzmann import (
    LatticeBoltzmannOperatingEnvelopePlan,
    LatticeBoltzmannOperatingPoint,
    LatticeBoltzmannRuntimeParameters,
    PreparedLatticeBoltzmannDynamics,
)


class DLDFlowResult(StrictModule):
    populations: Array
    density: Array
    velocity: Array
    pressure: Array
    steady_residual: Array
    mass_drift: Array
    iteration_count: Array
    header: AdmissibilityHeader
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class DLDLatticeBoltzmannFlowPlan(StrictModule, NonTrainableState):
    """Host-prepared LBM flow epoch bound to one DLD geometry identity."""

    dynamics: PreparedLatticeBoltzmannDynamics
    envelope: LatticeBoltzmannOperatingEnvelopePlan
    maximum_steps: int = eqx.field(static=True)
    steady_tolerance: float = eqx.field(static=True)
    knudsen_number: float = eqx.field(static=True)
    force_number: float = eqx.field(static=True)
    wall_resolution_cells: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedLatticeBoltzmannDynamics,
        envelope: LatticeBoltzmannOperatingEnvelopePlan,
        /,
        *,
        maximum_steps: int,
        steady_tolerance: float,
        knudsen_number: float,
        force_number: float,
        wall_resolution_cells: float,
        geometry_id: str,
    ) -> None:
        steps = int(maximum_steps)
        values = tuple(
            float(value)
            for value in (
                steady_tolerance,
                knudsen_number,
                force_number,
                wall_resolution_cells,
            )
        )
        identity = str(geometry_id)
        if (
            not isinstance(dynamics, PreparedLatticeBoltzmannDynamics)
            or not isinstance(envelope, LatticeBoltzmannOperatingEnvelopePlan)
            or envelope.lattice.lattice_id
            != dynamics.discretization.velocity_set.lattice_id
            or envelope.collision.family != dynamics.method.collision.plan.family
            or envelope.precision.policy_id != dynamics.discretization.precision.policy_id
            or (envelope.forcing is None) != (dynamics.method.forcing is None)
            or steps <= 0
            or any(not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[2] < 0.0
            or values[3] <= 0.0
            or not identity
        ):
            raise ValueError("DLD LBM flow plan inputs are invalid.")
        self.dynamics = dynamics
        self.envelope = envelope
        self.maximum_steps = steps
        (
            self.steady_tolerance,
            self.knudsen_number,
            self.force_number,
            self.wall_resolution_cells,
        ) = values
        self.geometry_id = identity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dld-lbm-flow",
                "dynamics": dynamics.prepared_id,
                "envelope": envelope.envelope_id,
                "maximum_steps": steps,
                "controls": values,
                "geometry": identity,
            }
        )

    def solve(
        self,
        initial_populations: ArrayLike,
        parameters: LatticeBoltzmannRuntimeParameters,
        /,
    ) -> DLDFlowResult:
        populations = self.dynamics.discretization.validate_populations(
            initial_populations
        )
        if not isinstance(parameters, LatticeBoltzmannRuntimeParameters):
            raise TypeError("parameters must be LatticeBoltzmannRuntimeParameters.")
        macro = self.dynamics.macroscopic_state(0.0, populations, parameters)
        velocity = macro.velocity
        residual = jnp.asarray(jnp.inf, dtype=populations.dtype)
        cumulative = jnp.asarray(True)
        mass_drift = jnp.asarray(0.0, dtype=populations.dtype)
        time_step = jnp.asarray(self.dynamics.scaling.time_step, dtype=populations.dtype)

        def body(index, carry):
            current, previous_velocity, _residual, cumulative_, drift_ = carry
            time = index.astype(populations.dtype) * time_step
            result = self.dynamics.step_detailed(
                index,
                time,
                current,
                time_step,
                parameters,
            )
            next_populations = jnp.where(
                cumulative_ & result.successful,
                result.accepted_state,
                current,
            )
            next_macro = self.dynamics.macroscopic_state(
                time + time_step, next_populations, parameters
            )
            next_residual = jnp.max(
                jnp.sqrt(jnp.sum((next_macro.velocity - previous_velocity) ** 2, axis=-1))
            )
            return (
                next_populations,
                next_macro.velocity,
                next_residual,
                cumulative_ & result.successful,
                jnp.maximum(drift_, result.diagnostics.mass_defect),
            )

        populations, velocity, residual, cumulative, mass_drift = jax.lax.fori_loop(
            0,
            self.maximum_steps,
            body,
            (populations, velocity, residual, cumulative, mass_drift),
        )
        macro = self.dynamics.macroscopic_state(
            self.maximum_steps * time_step, populations, parameters
        )
        relaxation = self.dynamics.scaling.relaxation_rate(parameters.kinematic_viscosity)
        point = LatticeBoltzmannOperatingPoint(
            mach_number=jnp.max(
                jnp.sqrt(jnp.sum(macro.velocity**2, axis=-1))
                / jnp.sqrt(self.dynamics.scaling.sound_speed_squared)
            ),
            knudsen_number=self.knudsen_number,
            relaxation_rate=relaxation,
            minimum_density=jnp.min(macro.density),
            maximum_density=jnp.max(macro.density),
            force_number=self.force_number,
            wall_resolution_cells=self.wall_resolution_cells,
            relative_mass_drift=mass_drift,
        )
        envelope = self.envelope.evaluate(point)
        converged = residual <= self.steady_tolerance
        finite = (
            jnp.all(jnp.isfinite(populations))
            & jnp.all(jnp.isfinite(macro.velocity))
            & jnp.isfinite(residual)
        )
        supported = cumulative & converged & finite & envelope.header.globally_eligible
        reasons = jnp.bitwise_or.reduce(jnp.ravel(envelope.header.reason_bits))
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            cumulative & converged,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, self.steady_tolerance - residual, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dld-lbm-flow-evidence", "plan": self.plan_id}
            ),
        )
        return DLDFlowResult(
            populations,
            macro.density,
            macro.velocity,
            macro.pressure,
            residual,
            mass_drift,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            header,
            self.geometry_id,
            self.plan_id,
        )


__all__ = ["DLDFlowResult", "DLDLatticeBoltzmannFlowPlan"]

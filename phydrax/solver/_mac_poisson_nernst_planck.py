#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    DerivativeAvailability,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._incompressible import FaceVelocity
from ..discretization.finite_volume._mac_electrochemical import (
    MACElectrochemicalFluxEvaluation,
    PreparedMACElectrochemicalFlux,
)
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ..equations._electrochemistry import (
    AbstractElectrochemicalClosure,
    ElectrochemicalLocalFields,
    ElectrolyteTransportParameters,
    FARADAY_CONSTANT,
)
from ._mac_electrostatic import MACElectrostaticPlan, MACElectrostaticResult


class MACPoissonNernstPlanckEvaluation(StrictModule):
    concentrations: Array
    concentration_rate: Array
    electrochemical: ElectrochemicalLocalFields
    electrostatic: MACElectrostaticResult
    flux: MACElectrochemicalFluxEvaluation
    total_free_energy: Array
    charge_rate_defect: Array
    explicit_step_restriction: Array
    header: AdmissibilityHeader
    derivative_availability: DerivativeAvailability = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MACPoissonNernstPlanckStepResult(StrictModule):
    candidate: Array
    accepted: Array
    evaluation: MACPoissonNernstPlanckEvaluation
    header: AdmissibilityHeader
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACPoissonNernstPlanckPlan(StrictModule, NonTrainableState):
    """Cell-centered PNP aligned exactly with MAC scalar and face layouts."""

    electrostatic: MACElectrostaticPlan
    closure: AbstractElectrochemicalClosure
    parameters: ElectrolyteTransportParameters
    flux: PreparedMACElectrochemicalFlux
    fixed_charge: Array
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrostatic: MACElectrostaticPlan,
        closure: AbstractElectrochemicalClosure,
        parameters: ElectrolyteTransportParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
        energy_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(electrostatic, MACElectrostaticPlan):
            raise TypeError("electrostatic must be MACElectrostaticPlan.")
        if not isinstance(closure, AbstractElectrochemicalClosure):
            raise TypeError("closure must implement AbstractElectrochemicalClosure.")
        if not isinstance(parameters, ElectrolyteTransportParameters):
            raise TypeError("parameters must be ElectrolyteTransportParameters.")
        if closure.schema.schema_id != parameters.schema.schema_id:
            raise ValueError("Electrochemical closure and parameters schemas differ.")
        fixed = jnp.broadcast_to(
            jnp.asarray(fixed_charge, dtype=electrostatic.operators.pressure_space.dtype),
            electrostatic.operators.discretization.cell_shape,
        )
        tolerance = float(energy_tolerance)
        if tolerance < 0.0:
            raise ValueError("energy_tolerance must be nonnegative.")
        flux = PreparedMACElectrochemicalFlux(
            electrostatic.operators, parameters.diffusivities
        )
        self.electrostatic = electrostatic
        self.closure = closure
        self.parameters = parameters
        self.flux = flux
        self.fixed_charge = fixed
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-poisson-nernst-planck",
                "electrostatic": electrostatic.plan_id,
                "closure": closure.closure_id,
                "parameters": parameters.parameters_id,
                "flux": flux.plan_id,
                "fixed_charge": array_tree_fingerprint(fixed),
                "energy_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        /,
        *,
        face_velocity: FaceVelocity | None = None,
        initial_potential: ArrayLike | None = None,
    ) -> MACPoissonNernstPlanckEvaluation:
        concentration = jnp.asarray(concentrations)
        expected = self.electrostatic.operators.discretization.cell_shape + (
            self.parameters.schema.species_count,
        )
        if concentration.shape != expected:
            raise ValueError("MAC concentrations must have cell/species shape.")
        charge = (
            FARADAY_CONSTANT
            * contract(
                "...s,s->...",
                concentration,
                self.parameters.schema.charges,
                backend="jax",
            )
            + self.fixed_charge
        )
        electrostatic = self.electrostatic.solve(
            charge, initial_potential=initial_potential
        )
        local = self.closure.evaluate(
            concentration,
            electrostatic.potential,
            self.parameters,
            fixed_charge=self.fixed_charge,
        )
        dimensionless_electrochemical = local.electrochemical_potential / (
            UNIVERSAL_GAS_CONSTANT * self.parameters.temperature
        )
        ideal = jnp.log(jnp.where(concentration > 0.0, concentration, 1.0))
        flux = self.flux.evaluate(
            concentration,
            dimensionless_electrochemical - ideal,
            face_velocity=face_velocity,
            dimensionless_electrochemical_potential=dimensionless_electrochemical,
        )
        volumes = self.electrostatic.operators.discretization.cell_volumes.astype(
            concentration.dtype
        )
        chemical_energy = jnp.sum(volumes * local.chemical_free_energy_density)
        total_energy = chemical_energy + electrostatic.field_energy
        charge_rate = FARADAY_CONSTANT * contract(
            "...s,s->...",
            flux.concentration_rate,
            self.parameters.schema.charges,
            backend="jax",
        )
        charge_defect = jnp.sum(volumes * charge_rate)
        charge_scale = jnp.maximum(jnp.sum(volumes * jnp.abs(charge_rate)), 1.0)
        charge_conserved = (
            jnp.abs(charge_defect)
            <= 512.0 * jnp.finfo(concentration.dtype).eps * charge_scale
        )
        finite = jnp.isfinite(total_energy) & jnp.isfinite(charge_defect)
        successful = (
            electrostatic.header.globally_eligible
            & local.successful
            & flux.header.globally_eligible
            & charge_conserved
            & finite
        )
        reasons = jnp.bitwise_or.reduce(
            jnp.ravel(electrostatic.header.reason_bits)
        ) | jnp.bitwise_or.reduce(jnp.ravel(flux.header.reason_bits))
        reasons = jnp.where(
            local.successful & charge_conserved,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(successful, jnp.min(concentration), -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "mac-pnp-evidence", "plan": self.plan_id}),
        )
        return MACPoissonNernstPlanckEvaluation(
            concentration,
            flux.concentration_rate,
            local,
            electrostatic,
            flux,
            total_energy,
            charge_defect,
            flux.explicit_step_restriction,
            header,
            DerivativeAvailability.IMPLICIT_FIXED_MODEL,
            self.plan_id,
        )

    def rate(
        self,
        time: Array,
        concentrations: Array,
        args: Any = None,
        /,
    ) -> Array:
        del time
        potential = None if args is None else args
        evaluation = self.evaluate(concentrations, initial_potential=potential)
        return jnp.where(
            evaluation.header.globally_eligible,
            evaluation.concentration_rate,
            jnp.full_like(concentrations, jnp.nan),
        )

    def step(
        self,
        concentrations: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> MACPoissonNernstPlanckStepResult:
        incoming = jnp.asarray(concentrations)
        step = jnp.asarray(time_step, dtype=incoming.dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        before = self.evaluate(incoming)
        candidate = incoming + step * before.concentration_rate
        after = self.evaluate(candidate, initial_potential=before.electrostatic.potential)
        energy_scale = jnp.maximum(jnp.abs(before.total_free_energy), 1.0)
        energy_admissible = (
            after.total_free_energy
            <= before.total_free_energy + self.energy_tolerance * energy_scale
        )
        successful = (
            before.header.globally_eligible
            & after.header.globally_eligible
            & jnp.isfinite(step)
            & (step > 0.0)
            & (step <= before.explicit_step_restriction)
            & jnp.all(candidate > 0.0)
            & energy_admissible
        )
        accepted = jnp.where(successful, candidate, incoming)
        evaluation = self.evaluate(
            accepted, initial_potential=before.electrostatic.potential
        )
        reasons = jnp.where(
            successful,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(successful, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "mac-pnp-step-evidence", "plan": self.plan_id}
            ),
        )
        return MACPoissonNernstPlanckStepResult(
            candidate,
            accepted,
            evaluation,
            header,
            successful,
            self.plan_id,
        )


__all__ = [
    "MACPoissonNernstPlanckEvaluation",
    "MACPoissonNernstPlanckPlan",
    "MACPoissonNernstPlanckStepResult",
]

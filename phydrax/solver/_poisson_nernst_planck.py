#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cochain_electrochemical import (
    CochainElectrochemicalFluxEvaluation,
    PreparedCochainElectrochemicalFlux,
)
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ..equations._electrochemistry import (
    AbstractElectrochemicalClosure,
    ElectrochemicalLocalFields,
    ElectrolyteTransportParameters,
    FARADAY_CONSTANT,
)
from ._cochain_electrostatic import (
    CochainElectrostaticPlan,
    CochainElectrostaticResult,
)


class PoissonNernstPlanckEvaluation(StrictModule):
    concentrations: Array
    concentration_rate: Array
    electrochemical: ElectrochemicalLocalFields
    electrostatic: CochainElectrostaticResult
    flux: CochainElectrochemicalFluxEvaluation
    total_free_energy: Array
    charge_rate_defect: Array
    explicit_step_restriction: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PoissonNernstPlanckStepResult(StrictModule):
    concentrations: Array
    evaluation: PoissonNernstPlanckEvaluation
    successful: Array
    plan_id: str = eqx.field(static=True)


class PoissonNernstPlanckPlan(StrictModule, NonTrainableState):
    electrostatic: CochainElectrostaticPlan
    closure: AbstractElectrochemicalClosure
    parameters: ElectrolyteTransportParameters
    flux: PreparedCochainElectrochemicalFlux
    fixed_charge: Array
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        electrostatic: CochainElectrostaticPlan,
        closure: AbstractElectrochemicalClosure,
        parameters: ElectrolyteTransportParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
        energy_tolerance: float = 1.0e-10,
    ):
        if not isinstance(electrostatic, CochainElectrostaticPlan):
            raise TypeError("electrostatic must be CochainElectrostaticPlan.")
        if not isinstance(closure, AbstractElectrochemicalClosure):
            raise TypeError("closure must implement AbstractElectrochemicalClosure.")
        if not isinstance(parameters, ElectrolyteTransportParameters):
            raise TypeError("parameters must be ElectrolyteTransportParameters.")
        if closure.schema.schema_id != parameters.schema.schema_id:
            raise ValueError("Electrochemical closure and parameters schemas differ.")
        node_count = electrostatic.bridge.cochain.cell_counts[0]
        fixed = jnp.broadcast_to(
            jnp.asarray(fixed_charge, dtype=electrostatic.permittivity.dtype),
            (node_count,),
        )
        tolerance = float(energy_tolerance)
        if tolerance < 0.0:
            raise ValueError("energy_tolerance must be nonnegative.")
        flux = PreparedCochainElectrochemicalFlux(
            electrostatic.bridge, parameters.diffusivities
        )
        self.electrostatic = electrostatic
        self.closure = closure
        self.parameters = parameters
        self.flux = flux
        self.fixed_charge = fixed
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "poisson-nernst-planck",
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
        initial_potential: ArrayLike | None = None,
    ) -> PoissonNernstPlanckEvaluation:
        concentration = jnp.asarray(concentrations)
        node_count = self.electrostatic.bridge.cochain.cell_counts[0]
        species_count = self.parameters.schema.species_count
        if concentration.shape != (node_count, species_count):
            raise ValueError("concentrations must have node/species shape.")
        charge = (
            FARADAY_CONSTANT
            * contract("ns,s->n", concentration, self.parameters.schema.charges)
            + self.fixed_charge
        )
        electrostatic = self.electrostatic.solve(
            charge,
            initial_potential=initial_potential,
        )
        local = self.closure.evaluate(
            concentration,
            electrostatic.potential,
            self.parameters,
            fixed_charge=self.fixed_charge,
        )
        dimensionless = local.electrochemical_potential / (
            UNIVERSAL_GAS_CONSTANT * self.parameters.temperature
        )
        flux = self.flux.evaluate(concentration, dimensionless)
        weights = self.electrostatic.bridge.cochain.hodge_stars[0].astype(
            concentration.dtype
        )
        chemical_energy = jnp.sum(weights * local.chemical_free_energy_density)
        total_energy = chemical_energy + electrostatic.field_energy
        charge_rate = FARADAY_CONSTANT * contract(
            "ns,s->n", flux.concentration_rate, self.parameters.schema.charges
        )
        charge_defect = jnp.sum(weights * charge_rate)
        scale = jnp.maximum(jnp.sum(weights * jnp.abs(charge_rate)), 1.0)
        successful = (
            electrostatic.successful
            & local.successful
            & flux.successful
            & jnp.isfinite(total_energy)
            & (
                jnp.abs(charge_defect)
                <= 256.0 * jnp.finfo(concentration.dtype).eps * scale
            )
        )
        return PoissonNernstPlanckEvaluation(
            concentration,
            flux.concentration_rate,
            local,
            electrostatic,
            flux,
            total_energy,
            charge_defect,
            flux.explicit_step_restriction,
            successful,
            self.plan_id,
        )

    def rate(self, time: Array, concentrations: Array, args: Any = None, /) -> Array:
        del time
        initial = None if args is None else args
        evaluation = self.evaluate(
            concentrations,
            initial_potential=initial,
        )
        return jnp.where(
            evaluation.successful,
            evaluation.concentration_rate,
            jnp.full_like(concentrations, jnp.nan),
        )

    def step(
        self,
        concentrations: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> PoissonNernstPlanckStepResult:
        incoming = jnp.asarray(concentrations)
        step = jnp.asarray(time_step, dtype=incoming.dtype)
        if step.shape != ():
            raise ValueError("time_step must be scalar.")
        before = self.evaluate(incoming)
        candidate = incoming + step * before.concentration_rate
        after = self.evaluate(
            candidate,
            initial_potential=before.electrostatic.potential,
        )
        energy_scale = jnp.maximum(jnp.abs(before.total_free_energy), 1.0)
        energy_admissible = (
            after.total_free_energy
            <= before.total_free_energy + self.energy_tolerance * energy_scale
        )
        successful = (
            before.successful
            & after.successful
            & jnp.isfinite(step)
            & (step > 0.0)
            & (step <= before.explicit_step_restriction)
            & jnp.all(candidate > 0.0)
            & energy_admissible
        )
        accepted = jnp.where(successful, candidate, incoming)
        evaluation = self.evaluate(
            accepted,
            initial_potential=before.electrostatic.potential,
        )
        return PoissonNernstPlanckStepResult(
            accepted,
            evaluation,
            successful,
            self.plan_id,
        )


__all__ = [
    "PoissonNernstPlanckEvaluation",
    "PoissonNernstPlanckPlan",
    "PoissonNernstPlanckStepResult",
]

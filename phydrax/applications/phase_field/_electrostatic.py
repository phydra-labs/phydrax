#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


ElectrostaticEnsemble = Literal["fixed-charge", "fixed-voltage"]


class PhasePermittivityLaw(StrictModule, NonTrainableState):
    phase_permittivities: Array
    law_id: str = eqx.field(static=True)

    def __init__(self, phase_permittivities: ArrayLike, /, *, law_id: str):
        values = np.asarray(phase_permittivities)
        identifier = str(law_id)
        if (
            values.ndim != 1
            or values.size < 2
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not identifier
        ):
            raise ValueError("Phase permittivity law is invalid.")
        self.phase_permittivities = jnp.asarray(values)
        self.law_id = canonical_fingerprint(
            {
                "kind": "phase-permittivity-law",
                "declared_id": identifier,
                "phase_permittivities": values.tolist(),
            }
        )

    def evaluate(self, phase_logits: ArrayLike, /) -> Array:
        logits = jnp.asarray(phase_logits)
        if logits.shape[-1] != self.phase_permittivities.size:
            raise ValueError("Phase permittivity logits have incompatible shape.")
        return ein.contract(
            "...p,p->...",
            jax.nn.softmax(logits, axis=-1),
            self.phase_permittivities.astype(logits.dtype),
        )


class ElectrostaticCouplingEvaluation(StrictModule):
    permittivity: Array
    electric_field: Array
    electric_displacement: Array
    field_energy: Array
    phase_force: Array
    maxwell_stress: Array
    gauss_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PhaseElectrostaticCouplingPlan(StrictModule, NonTrainableState):
    permittivity: PhasePermittivityLaw
    ensemble: ElectrostaticEnsemble = eqx.field(static=True)
    gauss_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        permittivity: PhasePermittivityLaw,
        /,
        *,
        ensemble: ElectrostaticEnsemble,
        gauss_tolerance: float = 1.0e-10,
    ):
        if not isinstance(permittivity, PhasePermittivityLaw):
            raise TypeError("permittivity must be PhasePermittivityLaw.")
        if ensemble not in ("fixed-charge", "fixed-voltage"):
            raise ValueError("Unknown electrostatic control ensemble.")
        tolerance = float(gauss_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Gauss-law tolerance must be nonnegative.")
        self.permittivity = permittivity
        self.ensemble = ensemble
        self.gauss_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "phase-electrostatic-coupling-plan",
                "permittivity": permittivity.law_id,
                "ensemble": ensemble,
                "gauss_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        phase_logits: ArrayLike,
        potential_gradient: ArrayLike,
        /,
        *,
        free_charge: ArrayLike,
        displacement_divergence: ArrayLike,
    ) -> ElectrostaticCouplingEvaluation:
        logits = jnp.asarray(phase_logits)
        gradient = jnp.asarray(potential_gradient, dtype=logits.dtype)
        charge = jnp.asarray(free_charge, dtype=logits.dtype)
        divergence = jnp.asarray(displacement_divergence, dtype=logits.dtype)
        if (
            gradient.shape[:-1] != logits.shape[:-1]
            or charge.shape != logits.shape[:-1]
            or divergence.shape != charge.shape
        ):
            raise ValueError("Electrostatic coupling fields have incompatible shapes.")
        electric = -gradient
        permittivity = self.permittivity.evaluate(logits)
        displacement = permittivity[..., None] * electric
        electric_norm = ein.contract("...d,...d->...", electric, electric)
        sign = 1.0 if self.ensemble == "fixed-charge" else -1.0
        field_energy = 0.5 * sign * permittivity * electric_norm

        def one_energy(local_logits, local_electric):
            epsilon = self.permittivity.evaluate(local_logits)
            return 0.5 * sign * epsilon * jnp.dot(local_electric, local_electric)

        flat_logits = logits.reshape((-1, logits.shape[-1]))
        flat_electric = electric.reshape((-1, electric.shape[-1]))
        phase_force = jax.vmap(jax.grad(one_energy, argnums=0))(
            flat_logits, flat_electric
        ).reshape(logits.shape)
        identity = jnp.eye(electric.shape[-1], dtype=electric.dtype)
        maxwell = permittivity[..., None, None] * (
            ein.contract("...i,...j->...ij", electric, electric)
            - 0.5 * electric_norm[..., None, None] * identity
        )
        gauss = divergence - charge
        scale = jnp.maximum(jnp.max(jnp.abs(charge)), 1.0)
        finite = (
            jnp.all(jnp.isfinite(field_energy))
            & jnp.all(jnp.isfinite(phase_force))
            & jnp.all(jnp.isfinite(maxwell))
            & jnp.all(jnp.isfinite(gauss))
        )
        successful = (
            finite
            & jnp.all(permittivity > 0.0)
            & (jnp.max(jnp.abs(gauss)) <= self.gauss_tolerance * scale)
        )
        return ElectrostaticCouplingEvaluation(
            permittivity,
            electric,
            displacement,
            field_energy,
            phase_force,
            maxwell,
            gauss,
            finite,
            successful,
            self.plan_id,
        )


class ElectrochemicalCouplingEvaluation(StrictModule):
    charge_density: Array
    electrochemical_potential: Array
    ionic_flux: Array
    electrical_dissipation: Array
    joule_heating: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ElectrochemicalCouplingPlan(StrictModule, NonTrainableState):
    valences: Array
    mobilities: Array
    faraday_constant: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        valences: ArrayLike,
        mobilities: ArrayLike,
        /,
        *,
        faraday_constant: ArrayLike,
    ):
        charges = np.asarray(valences)
        mobility = np.asarray(mobilities)
        faraday = np.asarray(faraday_constant)
        if (
            charges.ndim != 1
            or charges.size == 0
            or mobility.shape != charges.shape
            or np.any(~np.isfinite(charges))
            or np.any(~np.isfinite(mobility))
            or np.any(mobility < 0.0)
            or faraday.shape != ()
            or not np.isfinite(faraday)
            or faraday <= 0.0
        ):
            raise ValueError("Electrochemical coupling data are invalid.")
        self.valences = jnp.asarray(charges)
        self.mobilities = jnp.asarray(mobility)
        self.faraday_constant = jnp.asarray(faraday)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrochemical-coupling-plan",
                "valences": charges.tolist(),
                "mobilities": mobility.tolist(),
                "faraday_constant": float(faraday),
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        chemical_potentials: ArrayLike,
        chemical_gradients: ArrayLike,
        electric_potential: ArrayLike,
        electric_field: ArrayLike,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> ElectrochemicalCouplingEvaluation:
        concentration = jnp.asarray(concentrations)
        chemical = jnp.asarray(chemical_potentials, dtype=concentration.dtype)
        gradients = jnp.asarray(chemical_gradients, dtype=concentration.dtype)
        potential = jnp.asarray(electric_potential, dtype=concentration.dtype)
        electric = jnp.asarray(electric_field, dtype=concentration.dtype)
        fixed = jnp.asarray(fixed_charge, dtype=concentration.dtype)
        component_count = self.valences.size
        if (
            concentration.shape[-1] != component_count
            or chemical.shape != concentration.shape
            or gradients.shape[:-1] != concentration.shape
            or potential.shape != concentration.shape[:-1]
            or electric.shape != gradients.shape[:-2] + (gradients.shape[-1],)
            or fixed.shape != potential.shape
        ):
            raise ValueError("Electrochemical coupling fields have incompatible shapes.")
        charge_number = self.valences.astype(concentration.dtype)
        faraday = self.faraday_constant.astype(concentration.dtype)
        electrochemical = chemical + faraday * charge_number * potential[..., None]
        electrochemical_gradient = gradients - (
            faraday
            * charge_number.reshape((1,) * (gradients.ndim - 2) + (-1, 1))
            * electric[..., None, :]
        )
        mobility = self.mobilities.astype(concentration.dtype)
        flux = -(
            mobility.reshape((1,) * (gradients.ndim - 2) + (-1, 1))
            * concentration[..., :, None]
            * electrochemical_gradient
        )
        charge_density = fixed + faraday * ein.contract(
            "c,...c->...", charge_number, concentration
        )
        dissipation = -ein.contract("...cd,...cd->...", flux, electrochemical_gradient)
        current = faraday * ein.contract("c,...cd->...d", charge_number, flux)
        joule = ein.contract("...d,...d->...", current, electric)
        finite = (
            jnp.all(jnp.isfinite(charge_density))
            & jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(dissipation))
            & jnp.all(jnp.isfinite(joule))
        )
        successful = finite & jnp.all(concentration >= 0.0) & jnp.all(dissipation >= 0.0)
        return ElectrochemicalCouplingEvaluation(
            charge_density,
            electrochemical,
            flux,
            dissipation,
            joule,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "ElectrochemicalCouplingEvaluation",
    "ElectrochemicalCouplingPlan",
    "ElectrostaticCouplingEvaluation",
    "ElectrostaticEnsemble",
    "PhaseElectrostaticCouplingPlan",
    "PhasePermittivityLaw",
]

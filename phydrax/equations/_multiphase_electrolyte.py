#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._thermodynamics import (
    BinaryPhaseThermodynamicClosure,
    BinaryThermodynamicParameters,
)
from .._trainable import NonTrainableState
from ._electrochemistry import (
    AbstractElectrochemicalClosure,
    ElectrolyteTransportParameters,
)


class MultiphaseElectrolyteParameters(StrictModule, NonTrainableState):
    binary: BinaryThermodynamicParameters
    electrolyte: ElectrolyteTransportParameters
    solvation_coefficients: Array
    negative_phase_permittivity: Array
    positive_phase_permittivity: Array
    parameters_id: str = eqx.field(static=True)

    def __init__(
        self,
        binary: BinaryThermodynamicParameters,
        electrolyte: ElectrolyteTransportParameters,
        solvation_coefficients: ArrayLike,
        negative_phase_permittivity: ArrayLike,
        positive_phase_permittivity: ArrayLike,
        /,
    ):
        if not isinstance(binary, BinaryThermodynamicParameters):
            raise TypeError("binary must be BinaryThermodynamicParameters.")
        if not isinstance(electrolyte, ElectrolyteTransportParameters):
            raise TypeError("electrolyte must be ElectrolyteTransportParameters.")
        coefficients = jnp.asarray(solvation_coefficients)
        negative = jnp.asarray(negative_phase_permittivity, dtype=coefficients.dtype)
        positive = jnp.asarray(positive_phase_permittivity, dtype=coefficients.dtype)
        if coefficients.shape != (electrolyte.schema.species_count,) or (
            negative.shape != () or positive.shape != ()
        ):
            raise ValueError("Multiphase electrolyte parameter shapes are invalid.")
        if not bool(
            jnp.all(jnp.isfinite(coefficients))
            & jnp.isfinite(negative)
            & (negative > 0.0)
            & jnp.isfinite(positive)
            & (positive > 0.0)
        ):
            raise ValueError("Multiphase electrolyte parameters must be finite.")
        self.binary = binary
        self.electrolyte = electrolyte
        self.solvation_coefficients = coefficients
        self.negative_phase_permittivity = negative
        self.positive_phase_permittivity = positive
        self.parameters_id = canonical_fingerprint(
            {
                "kind": "multiphase-electrolyte-parameters",
                "binary": {
                    "bulk": array_tree_fingerprint(np.asarray(binary.bulk_scale)),
                    "gradient": array_tree_fingerprint(
                        np.asarray(binary.gradient_coefficient)
                    ),
                    "wetting": array_tree_fingerprint(
                        np.asarray(binary.wetting_strength)
                    ),
                },
                "electrolyte": electrolyte.parameters_id,
                "solvation": array_tree_fingerprint(np.asarray(coefficients)),
                "permittivity": [float(negative), float(positive)],
            }
        )


class MultiphaseElectrolyteFields(StrictModule):
    total_free_energy_density: Array
    phase_chemical_potential: Array
    ionic_electrochemical_potential: Array
    charge_density: Array
    permittivity: Array
    total_stress: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class MultiphaseElectrolyteClosure(StrictModule, NonTrainableState):
    binary: BinaryPhaseThermodynamicClosure
    electrochemical: AbstractElectrochemicalClosure
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        binary: BinaryPhaseThermodynamicClosure,
        electrochemical: AbstractElectrochemicalClosure,
        /,
    ):
        if not isinstance(binary, BinaryPhaseThermodynamicClosure):
            raise TypeError("binary must be BinaryPhaseThermodynamicClosure.")
        if not isinstance(electrochemical, AbstractElectrochemicalClosure):
            raise TypeError(
                "electrochemical must implement AbstractElectrochemicalClosure."
            )
        self.binary = binary
        self.electrochemical = electrochemical
        self.closure_id = canonical_fingerprint(
            {
                "kind": "multiphase-electrolyte-closure",
                "binary": binary.closure_id,
                "electrochemical": electrochemical.closure_id,
            }
        )

    def evaluate(
        self,
        phase: ArrayLike,
        phase_gradient: ArrayLike,
        phase_laplacian: ArrayLike,
        concentrations: ArrayLike,
        potential: ArrayLike,
        electric_field: ArrayLike,
        parameters: MultiphaseElectrolyteParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> MultiphaseElectrolyteFields:
        phase_ = jnp.asarray(phase)
        gradient = jnp.asarray(phase_gradient, dtype=phase_.dtype)
        laplacian = jnp.asarray(phase_laplacian, dtype=phase_.dtype)
        concentration = jnp.asarray(concentrations, dtype=phase_.dtype)
        electric = jnp.asarray(electric_field, dtype=phase_.dtype)
        if gradient.shape[:-1] != phase_.shape or electric.shape != gradient.shape:
            raise ValueError("Phase gradient and electric field must match phase shape.")
        if laplacian.shape != phase_.shape or concentration.shape != phase_.shape + (
            parameters.electrolyte.schema.species_count,
        ):
            raise ValueError("Multiphase electrolyte field shapes are incompatible.")
        binary = self.binary.evaluate_local(
            phase_, gradient, laplacian, parameters.binary
        )
        electrochemical = self.electrochemical.evaluate(
            concentration,
            potential,
            parameters.electrolyte,
            fixed_charge=fixed_charge,
        )
        solvation = contract(
            "...s,s->...", concentration, parameters.solvation_coefficients
        )
        phase_mu = binary.chemical_potential + solvation
        ionic_mu = electrochemical.electrochemical_potential + (
            phase_[..., None] * parameters.solvation_coefficients
        )
        positive_fraction = 0.5 * (phase_ + 1.0)
        permittivity = (
            (1.0 - positive_fraction) * parameters.negative_phase_permittivity
            + positive_fraction * parameters.positive_phase_permittivity
        )
        electric_squared = jnp.sum(electric * electric, axis=-1)
        electric_energy = 0.5 * permittivity * electric_squared
        total_energy = (
            binary.bulk_energy_density
            + binary.gradient_energy_density
            + electrochemical.chemical_free_energy_density
            + phase_ * solvation
            + electric_energy
        )
        dimension = electric.shape[-1]
        identity = jnp.eye(dimension, dtype=phase_.dtype)
        maxwell = permittivity[..., None, None] * (
            contract("...i,...j->...ij", electric, electric)
            - 0.5 * electric_squared[..., None, None] * identity
        )
        osmotic = -electrochemical.osmotic_pressure[..., None, None] * identity
        total_stress = binary.symmetric_stress + maxwell + osmotic
        binary_finite = (
            jnp.all(jnp.isfinite(binary.bulk_energy_density))
            & jnp.all(jnp.isfinite(binary.gradient_energy_density))
            & jnp.all(jnp.isfinite(binary.chemical_potential))
            & jnp.all(jnp.isfinite(binary.symmetric_stress))
        )
        successful = (
            binary_finite
            & electrochemical.successful
            & jnp.all(jnp.isfinite(total_energy))
            & jnp.all(jnp.isfinite(total_stress))
            & jnp.all(permittivity > 0.0)
        )
        return MultiphaseElectrolyteFields(
            total_energy,
            phase_mu,
            ionic_mu,
            electrochemical.charge_density,
            permittivity,
            total_stress,
            successful,
            self.closure_id,
        )


__all__ = [
    "MultiphaseElectrolyteClosure",
    "MultiphaseElectrolyteFields",
    "MultiphaseElectrolyteParameters",
]
